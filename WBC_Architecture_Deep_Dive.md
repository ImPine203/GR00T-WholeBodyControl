# GR00T-WholeBodyControl: Architecture & Workflow Deep Dive

Tài liệu này phân tích chi tiết cấu trúc kiến trúc, luồng hoạt động (workflow) và sự tương tác giữa các thành phần cốt lõi trong dự án **GR00T-WholeBodyControl**. Hệ thống này là một software stack hoàn chỉnh từ mô phỏng, huấn luyện (RL/BC) cho đến triển khai thời gian thực trên robot hình người (chủ yếu là Unitree G1).

---

## 1. Tổng Quan Kiến Trúc (System Overview)

Hệ thống được thiết kế dưới dạng modular, chia thành 4 mảng chính:
1. **Training Stack (`gear_sonic`, `trl`):** Hạ tầng huấn luyện Reinforcement Learning quy mô lớn dựa trên Isaac Lab và thuật toán PPO tối ưu hoá qua HuggingFace Accelerate.
2. **Simulation Environment (`mujoco_sim`):** Môi trường mô phỏng vật lý MuJoCo, đóng vai trò như một bản sao kỹ thuật số (digital twin) cung cấp giao tiếp giả lập mạng DDS y hệt robot thật.
3. **Deployment Stack (`gear_sonic_deploy`):** Hệ thống C++ đảm nhận việc chạy suy luận (inference) model AI với độ trễ thấp bằng ONNX Runtime / TensorRT.
4. **Teleoperation & Data Collection (`motionbricks`, VR scripts):** Luồng điều khiển từ xa bằng thiết bị VR (Pico) kết hợp mô hình tạo sinh (Generative framework) để chuyển đổi và thu thập chuyển động của con người sang robot.

---

## 2. Luồng Huấn Luyện (Training Pipeline: GEAR-SONIC)

Quá trình huấn luyện diễn ra hoàn toàn trong môi trường giả lập song song khối lượng lớn (Isaac Lab) để học chính sách (policy) điều khiển.

### Thành phần chính:
- **Entry point:** `gear_sonic/train_agent_trl.py`
- **Config Management:** Sử dụng `Hydra` (cấu hình cơ sở tại `gear_sonic/config/base.yaml`).
- **Môi trường (MDP - Markov Decision Process):**
  - Cấu hình môi trường nằm tại `gear_sonic/envs/manager_env/modular_tracking_env_cfg.py`. Nó định nghĩa robot, địa hình (terrain), các cảm biến ảo (camera, contact sensors) và các khối phần thưởng (rewards).
- **Thuật toán PPO (TRL):** 
  - `gear_sonic/trl/trainer/ppo_trainer.py`: Kế thừa từ thư viện TRL, hỗ trợ phân tán multi-GPU.
  - Model kiến trúc: `UniversalTokenModule` (`gear_sonic/trl/modules/universal_token_modules.py`). Kiến trúc này mã hóa (encode) các quan sát (proprioception, commands) qua một luồng FSQ (Finite Scalar Quantizer) để tạo token, sau đó giải mã (decode) ra góc/vận tốc các khớp.
- **Đầu ra (Export):** `gear_sonic/eval_agent_trl.py` được sử dụng để đánh giá và xuất chính sách đã huấn luyện ra định dạng **ONNX** để triển khai. (Lưu ý: yêu cầu thiết lập `num_envs=1`).

---

## 3. Hệ Thống Mô Phỏng (MuJoCo Simulation)

Khi muốn chạy thử nghiệm policy hoặc vận hành teleop, hệ thống sử dụng MuJoCo (chạy trên CPU/GPU cục bộ) thay vì Isaac Lab (nặng và dùng cho training).

### Thành phần chính:
- **Entry point:** `gear_sonic/scripts/run_sim_loop.py`
- **Core Simulator:** `gear_sonic/utils/mujoco_sim/base_sim.py`
  - Chịu trách nhiệm nạp mô hình MJCF/URDF (`DefaultEnv.init_scene`), thiết lập các tính toán động lực học, Elastic Band (để giữ thăng bằng giả lập), và vòng lặp vật lý.
- **Unitree SDK Bridge (`gear_sonic/utils/mujoco_sim/unitree_sdk2py_bridge.py`):**
  - Đây là **thành phần quan trọng nhất** kết nối thế giới ảo và luồng điều khiển C++. Nó đóng vai trò là một "DDS Broker".
  - **Publishers:** Đọc dữ liệu từ MuJoCo (Joint states, IMU) và gửi bản tin Unitree SDK (`LowState`, `IMUState`) qua mạng DDS.
  - **Subscribers:** Nhận lệnh `LowCmd` từ controller C++ qua mạng DDS và áp dụng lực/vị trí tương ứng vào các Actuators của MuJoCo.
- **Plugins:** Có thể tích hợp với các MuJoCo plugin như `camera_connector` để render và publish data hình ảnh sang ROS 2 trong luồng background.

---

## 4. Triển Khai và Suy Luận C++ (Deployment & Inference)

Thành phần này độc lập với Python, chuyên trách việc duy trì vòng lặp điều khiển thời gian thực (thường là 500Hz - 1000Hz).

### Luồng hoạt động tại `gear_sonic_deploy/src/g1/g1_deploy_onnx_ref`:
1. **Khởi tạo:** Script orchestration `gear_sonic_deploy/deploy.sh` xác định môi trường (`--interface sim` hoặc `real`), biên dịch mã C++ bằng CMake và khởi chạy Node.
2. **Giao tiếp (Interfaces):**
   - **Input:** `input_interface/` (nhận command từ Gamepad, Keyboard, ROS 2, ZMQ teleop).
   - **Output:** `output_interface/` (cơ chế gửi lệnh xuống phần cứng hoặc SDK Bridge).
3. **Chạy Suy luận (Inference):**
   - Source: `g1_deploy_onnx_ref.cpp` và các file `localmotion_kplanner_*.hpp`.
   - Node sẽ lấy Observation hiện tại (từ `robot_parameters.hpp`), kết hợp command từ input.
   - Truyền qua mạng Neural Network bằng thư viện **ONNX Runtime** hoặc **TensorRT** để lấy hành động (Actions).
   - Áp dụng bộ giải thuật chuyển đổi (PD Control / FK) để tính toán mô-men xoắn (torque) hoặc đích vị trí (position target) cho khớp.
4. **Log dữ liệu:** `state_logger.cpp` thu thập thông số real-time để phân tích sau triển khai.

---

## 5. Teleoperation và Thu Thập Dữ Liệu

Hệ thống hỗ trợ điều khiển robot "bắt chước" con người theo thời gian thực (Teleoperation).

- **Thiết bị:** Tích hợp kính VR (Pico, HTC Vive) hoặc LeapMotion.
- **Luồng dữ liệu:**
  1. `gear_sonic/scripts/pico_manager_thread_server.py`: Lấy tọa độ 3 điểm (Đầu, 2 Tay) từ API của Pico/XRoboToolkit. Sử dụng Inverse Kinematics (ví dụ `G1GripperInverseKinematicsSolver`) để tính toán vị trí khớp tay.
  2. Gửi dữ liệu qua hệ thống ZMQ (sử dụng ZeroMQ publisher/subscriber) đến máy chủ điều khiển.
  3. Dữ liệu này được kết hợp (`streamed_motion_merger.hpp` trong môi trường triển khai) hoặc chuyển tiếp tới mô hình tạo sinh.
- **MotionBricks (`motionbricks/`):**
  - Là framework chạy song song giúp tổng hợp chuyển động (Locomotion) mượt mà dựa trên mạng VQVAE và smart primitives. Khi người điều khiển (VR hoặc Keyboard) ra lệnh tiến/lùi, MotionBricks sinh ra tập hợp tọa độ khớp tự nhiên (freestyle, walk, crawl...) và đưa vào luồng điều khiển.
- **Lưu trữ:** Trong lúc vận hành teleop, script (ví dụ `decoupled_wbc/scripts/deploy_g1.py`) có thể lưu toàn bộ Trajectory vào format hdf5 để huấn luyện Behavior Cloning (BC) sau này.

---

## 6. Tổng Kết Luồng Thực Thi (End-to-End Workflow)

Để có một góc nhìn tổng thể, đây là chuỗi hành động khi chạy toàn bộ hệ thống (dựa theo `deploy.sh` / `deploy_g1.py`):

1. **Khởi động Simulator:** Python script (`run_sim_loop.py`) chạy MuJoCo. Bridge DDS bắt đầu phát Broadcast trạng thái giả lập của robot.
2. **Khởi động Teleop/Input:** Mở kết nối ZMQ từ Server lấy dữ liệu Gamepad, Keyboard, hoặc Pico VR.
3. **Khởi động Control Node (C++):** Chạy node C++. Node tải mô hình ONNX đã train từ `gear_sonic`, subscribe vào DDS để lấy trạng thái robot từ MuJoCo.
4. **Vòng lặp kín (Closed-loop):**
   - *Input nhận lệnh:* User bấm tiến tới.
   - *Control Node:* Lấy lệnh + trạng thái các khớp -> Chạy Inference ONNX -> Sinh ra chuỗi vị trí mục tiêu (Joint targets).
   - *Control Node:* Xuất chuỗi mục tiêu này thành `LowCmd` đẩy vào DDS.
   - *Simulator:* Unitree SDK Bridge nhận `LowCmd`, tính toán PID/lực đẩy vào vật lý MuJoCo. Robot giả lập bước đi, cập nhật trạng thái khớp mới, quay lại bước 3.

Tài liệu này cung cấp cái nhìn cốt lõi để các kỹ sư có thể nắm bắt cách code từ các phần như huấn luyện RL, C++ low-level, tới ROS 2/ZMQ streaming móc nối với nhau trong GR00T-WholeBodyControl.
