# MotionBricks Deep Dive

Tài liệu này mô tả MotionBricks theo source hiện tại trong `/home/tung/GR00T-WholeBodyControl/motionbricks/`.

MotionBricks là subproject tạo chuyển động thời gian thực cho robot/humanoid. Bản public trong repo này đang tập trung vào Unitree G1: có checkpoint, G1 MuJoCo assets, interactive demo, và script train synthetic để kiểm tra pipeline. Full dataset/training loader thật chưa nằm đầy đủ trong source public hiện tại.

---

## Bản Đồ Source Nhanh

| Cơ chế | File chính |
| --- | --- |
| Demo entrypoint | `motionbricks/scripts/interactive_demo_g1.py` |
| Demo bootstrap | `motionbricks/motionbricks/motion_backbone/demo/utils.py` |
| Full navigation agent | `motionbricks/motionbricks/motion_backbone/demo/full_agent.py` |
| Keyboard/random controllers | `motionbricks/motionbricks/motion_backbone/demo/controllers.py` |
| Reference clips/styles | `motionbricks/motionbricks/motion_backbone/demo/clips.py` |
| Inference orchestrator | `motionbricks/motionbricks/motion_backbone/inference/motion_inference.py` |
| VQVAE Lightning model | `motionbricks/motionbricks/vqvae/models/motion_vqvae.py` |
| Pose model | `motionbricks/motionbricks/motion_backbone/models/pose_model.py` |
| Root model | `motionbricks/motionbricks/motion_backbone/models/root_model.py` |
| Skeleton G1 | `motionbricks/motionbricks/motionlib/core/skeletons/g1.py` |
| Motion representation | `motionbricks/motionbricks/motionlib/core/motion_reps/` |
| MuJoCo conversion | `motionbricks/motionbricks/helper/mujoco_helper.py` |
| Synthetic data | `motionbricks/motionbricks/data/synthetic_dataset.py` |
| Checkpoint configs | `motionbricks/out/*/version_1/hparams.yaml` |

---

## 1. Kiến Trúc Tổng Quát

Runtime chia chuyển động thành hai phần:

- **Root**: vị trí pelvis và heading trong không gian.
- **Pose/body**: joint positions, rotations, velocities, foot contacts.

Luồng inference:

1. Controller đọc phím hoặc sinh lệnh random.
2. `full_navigation_agent` lấy context qpos hiện tại, canonicalize, và dựng target root bằng critical damping spring.
3. Reference clip tương ứng mode/style được lấy từ `clip_holder_G1`.
4. `motion_inference.predict(...)` chạy root model, pose model, rồi VQVAE decoder.
5. Feature motion được đổi về MuJoCo `qpos`.
6. `interactive_demo_g1.py` gán `mj_data.qpos[:] = qpos`, gọi `mujoco.mj_forward()`, rồi viewer render frame mới.

---

## 2. Motion Representation

Phần này trả lời câu hỏi: "Một chuyển động của robot được biến thành thứ gì để model học?"

Nếu chưa quen deep learning, hãy hiểu đơn giản thế này: neural network chỉ nhận **một dãy số**. Nó không tự hiểu MuJoCo, không tự hiểu file XML, cũng không tự biết "đây là đầu gối trái". Vì vậy trước khi đưa chuyển động vào model, MotionBricks phải đổi mỗi frame chuyển động thành một vector số có ý nghĩa rõ ràng. Vector đó gọi là **motion representation**.

Một clip chuyển động được nhìn như một chuỗi frame:

```text
motion clip = frame_0, frame_1, frame_2, ..., frame_T
```

Mỗi frame được mã hóa thành một vector:

```text
frame_i = [root numbers, body numbers]
```

Sau khi ghép nhiều frame lại, tensor model nhìn thấy có dạng:

```text
[T, D]
```

Trong đó:

- `T` là số frame trong clip.
- `D` là số feature trong mỗi frame.
- Với G1, `D` thường là 414 hoặc 413 tùy dùng global hay local representation.

Khi train nhiều clip cùng lúc, code thêm chiều batch:

```text
[B, T, D]
```

Trong đó `B` là số clip/sample trong một batch.

### 2.1. Vì Sao Không Dùng Thẳng MuJoCo `qpos`

MuJoCo `qpos` là format rất sát simulator:

```text
qpos = [root translation, root quaternion, 29 hinge joint angles]
```

Với G1 demo, `qpos` có 36 số. Format này tốt cho MuJoCo, nhưng không phải format tốt nhất cho model học, vì:

- Nó phụ thuộc vào XML MuJoCo và thứ tự joint trong XML.
- Rotation root dùng quaternion, còn các joint khác dùng hinge angle.
- Robot khác có số DOF khác thì `qpos` đổi shape.
- `qpos` không chứa sẵn thông tin hữu ích như foot contact, joint velocity, hay vị trí tương đối của các khớp.
- MuJoCo dùng Z-up/X-forward, còn motion data trong MotionBricks dùng Y-up/Z-forward.

MotionBricks vì vậy dùng một format trung gian:

```text
MuJoCo qpos <-> motion representation <-> neural network
```

Format trung gian này giàu thông tin hơn `qpos`, ổn định hơn cho training, và dễ tách phần "đường đi của robot" khỏi phần "tư thế cơ thể".

### 2.2. Lý Thuyết Tối Thiểu: Root Và Body

Khi mô tả chuyển động người/robot, ta thường tách thành hai lớp:

- **Root**: điểm gốc của cơ thể, ở đây là pelvis. Root trả lời: robot đang ở đâu, cao bao nhiêu, quay mặt hướng nào.
- **Body**: các khớp còn lại so với root. Body trả lời: chân tay đang co duỗi thế nào, vận tốc các khớp ra sao, chân có chạm đất không.

Ví dụ một người đi bộ:

- Root là đường đi của hông/pelvis tiến về phía trước.
- Body là nhịp chân, tay đánh, đầu gối gập, bàn chân tiếp xúc mặt đất.

Tách như vậy giúp model dễ học hơn. Root model có thể tập trung học "đi đâu và quay hướng nào". Pose/VQVAE tập trung học "cơ thể phải tạo dáng như thế nào để đi theo root đó".

### 2.3. Lý Thuyết Tối Thiểu: Position, Rotation, Velocity, Contact

Motion representation trong repo này dùng bốn nhóm thông tin body:

1. **Position**: vị trí khớp trong không gian 3D, gồm X/Y/Z.
2. **Rotation**: hướng xoay của khớp. MotionBricks dùng rotation 6D thay vì Euler angle hoặc quaternion cho body rotations.
3. **Velocity**: vận tốc, tức khớp thay đổi vị trí nhanh thế nào qua các frame.
4. **Foot contact**: chân có đang chạm đất không.

Vì sao cần velocity và foot contact nếu đã có position/rotation?

- Position/rotation nói "frame này trông như thế nào".
- Velocity nói "đang chuyển động nhanh/chậm thế nào".
- Foot contact giúp giảm hiện tượng chân trượt trên đất khi generate motion.

Deep learning thường học tốt hơn khi input chứa những tín hiệu này rõ ràng, thay vì bắt model tự suy ra mọi thứ từ `qpos`.

### 2.4. Lý Thuyết Tối Thiểu: Vì Sao Rotation Dùng 6D

Rotation 3D khó biểu diễn bằng số:

- Euler angle dễ bị gimbal lock và có điểm nhảy.
- Quaternion có 4 số nhưng `q` và `-q` có thể biểu diễn cùng một rotation, gây nhập nhằng cho model.
- Matrix rotation 3x3 có 9 số, hơi dư.

MotionBricks dùng **continuous 6D rotation representation**: mỗi rotation lưu bằng 6 số. Đây là cách phổ biến trong motion generation vì nó liên tục hơn và dễ học hơn cho neural network. Khi cần dùng thật, code convert 6D rotation trở lại rotation matrix.

Trong code, phần này nằm ở `global_rot_data`.

### 2.5. Global Root Và Local Root Là Gì

MotionBricks giữ hai cách mô tả root:

**Global root** mô tả trạng thái tuyệt đối:

```text
robot đang ở vị trí XYZ nào, quay mặt hướng nào
```

Nó phù hợp cho root model, vì root model cần dự đoán trajectory trong không gian.

**Local root** mô tả thay đổi theo thời gian:

```text
robot đang quay nhanh thế nào, đang chạy theo XZ nhanh thế nào, root cao bao nhiêu
```

Nó phù hợp cho pose/VQVAE, vì tư thế cơ thể thường phụ thuộc vào vận tốc và nhịp chuyển động hơn là tọa độ tuyệt đối trong phòng.

Ví dụ:

- Nếu robot đứng ở tọa độ `(10, 0, 5)` hay `(0, 0, 0)` nhưng đều đang đi thẳng 1 m/s, dáng đi cơ thể nên gần giống nhau.
- Vì vậy pose model không cần quá quan tâm robot đang ở đâu trong world, nó cần biết robot đang di chuyển thế nào.

Trong config hiện tại `removing_heading=False`, nên `local_root_vel` là velocity XZ dạng vi phân theo frame, không phải velocity đã xoay về heading-local frame. Chữ "local" ở đây chủ yếu nghĩa là root được biểu diễn bằng delta/velocity thay vì vị trí absolute.

### 2.6. Config Trong Repo

Config hiện tại instantiate object sau:

```yaml
motion_rep:
  _target_: motionbricks.motionlib.core.motion_reps.dual_root_global_joints.GlobalRootGlobalJoints
  name: ${skeleton.name}_dual_root_global_joints
```

Tên class là `GlobalRootGlobalJoints`, nhưng vì `name` chứa `dual_root_global_joints`, constructor sẽ tạo thêm `dual_rep` bên trong. Vì vậy object dùng trong code là một **global-rep view** có kèm một object dual để convert qua local-rep khi cần.

Nói đơn giản:

- Bên ngoài dataset/model setup thường cầm global view.
- Khi cần local view, code gọi `motion_rep.dual_rep.global_to_local(...)`.
- Khi cần quay lại global view, code gọi `motion_rep.dual_rep.local_to_global(...)`.

### 2.7. Ba Dạng Tensor Cần Phân Biệt

Với `G1Skeleton34`, có 34 joints. Motion representation liên quan tới ba dạng tensor:

| Subset | Dims | Dùng bởi |
| --- | ---: | --- |
| **Global subset** | 414 | dataset trả về, root model dùng chính |
| **Local subset** | 413 | VQVAE/pose tokenizer dùng chính |
| **Full dual stats** | 418 | `mean.npy`, `std.npy` trong checkpoint |

Không phải mọi nơi đều dùng full 418 dims. Trong training scripts hiện tại:

- Batch input `batch["motion"]` là normalized **global subset** `[B, T, 414]`.
- VQVAE và pose model sẽ convert global subset sang local subset `[B, T, 413]`.
- Stats trên disk lại là full dual `[418]`, để code có thể tách ra stats cho cả global và local subset.

### 2.8. Full Dual Layout

Full dual representation lưu cả hai kiểu root, nhưng body chỉ lưu một lần:

```text
full_dual = [global_root(5), local_root(4), body(409)]
          = 418 dims
```

Với G1, layout trong full dual stats là:

| Full-dual range | Feature | Dims |
| --- | --- | ---: |
| `0:3` | `global_root_pos` | 3 |
| `3:5` | `global_root_heading` | 2 |
| `5:6` | `local_root_rot_vel` | 1 |
| `6:8` | `local_root_vel` | 2 |
| `8:9` | `global_root_y` | 1 |
| `9:108` | `ric_data` | 99 |
| `108:312` | `global_rot_data` | 204 |
| `312:414` | `local_vel` | 102 |
| `414:418` | `foot_contacts` | 4 |

Compact global subset không giữ local root:

```text
global_subset = [global_root(5), body(409)] = 414 dims
```

Compact local subset không giữ global root:

```text
local_subset = [local_root(4), body(409)] = 413 dims
```

Body của global subset và local subset là cùng một body. Convert global <-> local chủ yếu là đổi cách biểu diễn root, còn body được giữ lại.

### 2.9. Root Features

Global root có 5 dims:

| Feature | Dims | Ý nghĩa |
| --- | ---: | --- |
| `global_root_pos` | 3 | vị trí root/pelvis XYZ trong motion space |
| `global_root_heading` | 2 | heading quanh trục Y, mã hóa bằng `(cos(theta), sin(theta))` |

Local root có 4 dims:

| Feature | Dims | Ý nghĩa |
| --- | ---: | --- |
| `local_root_rot_vel` | 1 | vận tốc góc yaw quanh trục Y |
| `local_root_vel` | 2 | vận tốc root trên mặt phẳng XZ |
| `global_root_y` | 1 | chiều cao root theo trục Y |

Root global và root local chứa cùng một chuyển động ở hai dạng khác nhau:

```text
global root: tôi đang ở đâu
local root : tôi đang thay đổi thế nào
```

### 2.10. Body Features

Body features chung là 409 dims:

| Feature | Dims | Ý nghĩa |
| --- | ---: | --- |
| `ric_data` | 99 | vị trí 33 non-root joints sau khi trừ root XZ, tức joint positions tương đối với root translation |
| `global_rot_data` | 204 | rotation 6D của 34 joints, mỗi joint 6 dims |
| `local_vel` | 102 | velocity XYZ của 34 joints |
| `foot_contacts` | 4 | contact flags cho left ankle/toe và right ankle/toe |

Công thức tổng quát với `J` joints:

```text
ric_data        = (J - 1) * 3
global_rot_data = J * 6
local_vel       = J * 3
foot_contacts   = 4
body            = 12J + 1
```

Với G1 `J=34`:

```text
body = 33*3 + 34*6 + 34*3 + 4
     = 99 + 204 + 102 + 4
     = 409
```

Tên `global_rot_data` nghĩa là rotation được lưu như global joint rotations, không phải local hinge DOF kiểu MuJoCo. Khi convert ra `qpos`, `mujoco_qpos_converter` mới đổi global rotations thành local joint rotations rồi project xuống hinge axis của MuJoCo.

### 2.11. Normalization Và Stats

Neural network học bằng số, nhưng nó rất nhạy với scale. Nếu một feature có giá trị khoảng `0.01`, feature khác khoảng `1000`, model sẽ khó học cân bằng. Vì vậy trước khi đưa vào model, MotionBricks chuẩn hóa feature về scale dễ học hơn.

Các model không ăn raw feature. Feature được normalize bằng:

```python
normalized = (feature - mean) / sqrt(std**2 + eps)
```

Trong đó:

- `mean` là giá trị trung bình của feature trên dataset training.
- `std` là độ lệch chuẩn, cho biết feature thường dao động mạnh hay yếu.
- `eps` là số nhỏ để tránh chia cho 0.

Ví dụ trực giác:

- Root height có thể quanh `0.75`.
- Joint velocity có thể dao động quanh `0`.
- Rotation 6D nằm trong khoảng khác nữa.

Normalization đưa chúng về thang đo tương đối đồng đều để model học ổn định hơn.

Stats nằm trong:

```text
out/motionbricks_vqvae/version_1/stats/motion/
out/motionbricks_pose/version_1/stats/motion/
out/motionbricks_root/version_1/stats/motion/
```

Với G1, `mean.npy` và `std.npy` đều có shape `(418,)`, tức full dual stats.

Khi load config, code làm như sau:

1. Load full stats 418 dims.
2. Tạo `dual_rep`.
3. Cắt stats global bằng indices `[global_root, body]`, thành 414 dims.
4. Cắt stats local bằng indices `[local_root, body]`, thành 413 dims.
5. Gán stats tương ứng cho `global_motion_rep` và `local_motion_rep`.

Vì vậy cùng một `stats/motion` folder phục vụ được cả root model và pose/VQVAE.

### 2.12. Convert Global Sang Local Trong Training

Dataset/training batch trả về:

```python
batch["motion"]  # normalized global subset, [B, T, 414]
```

Trong VQVAE/pose/root training step, code thường làm:

```python
global_motions = global_motion_rep.change_first_heading(...)
local_motions = motion_rep.dual_rep.global_to_local(
    global_motions,
    is_normalized=True,
    to_normalize=True,
    lengths=...
)
```

`global_to_local(...)` làm logic cốt lõi:

1. Unnormalize global root từ 414-dim global subset.
2. Extract `global_root_pos` và `global_root_heading`.
3. Tính local root:
   - yaw velocity từ heading theo thời gian.
   - XZ velocity từ root position theo thời gian.
   - root height Y.
4. Giữ nguyên body features. Nếu input đã normalized và output cũng normalized, body không cần normalize lại vì global/local dùng cùng body stats.
5. Normalize local root bằng local-root stats.
6. Trả về local subset `[B, T, 413]`.

Chiều ngược lại `local_to_global(...)` tích phân velocity/heading để phục hồi global root, rồi ghép lại với body.

### 2.13. Representation Trong Inference Demo

Trong `interactive_demo_g1.py`, có hai chiều chuyển đổi chính:

```text
MuJoCo qpos
  -> convert_mujoco_qpos_to_motion_transforms
  -> context joint positions/rotations trong motion space
  -> global/local root values + local pose conditions
  -> root model + pose model + VQVAE
  -> predicted global motion features
  -> convert_motion_features_to_mujoco_qpos
  -> MuJoCo qpos
```

Ở giữa neural inference, `motion_inference.predict(...)` dùng:

- `global_root_values`: root absolute position + heading, để root model dự đoán trajectory.
- `local_root_values`: root velocity + height, để pose decoder có điều kiện chuyển động mượt.
- `local_poses`: joint positions + 6D rotations, cộng thêm root height khi normalize pose condition.

Output cuối của inferencer là `pred_global_poses`, tức global motion subset. Converter MuJoCo dùng global root và `global_rot_data` để tạo lại qpos 36 chiều cho G1.

### 2.14. Tóm Tắt Bằng Một Ví Dụ

Giả sử ta có một frame robot đang đi bộ. MotionBricks không chỉ lưu:

```text
đầu gối trái = 0.3 rad, vai phải = -0.1 rad, ...
```

Nó lưu một bản mô tả giàu hơn:

```text
root đang ở đâu
root đang quay hướng nào
root đang di chuyển nhanh thế nào
các khớp nằm ở đâu so với root
các khớp xoay ra sao trong world
các khớp đang có vận tốc nào
chân trái/phải có đang chạm đất không
```

Từ một chuỗi các frame như vậy, model học được các quy luật:

- Nếu root đi nhanh hơn, chân phải bước dài/nhanh hơn.
- Nếu heading đổi, hông và chân phải xoay theo.
- Nếu foot contact đang bật, bàn chân không nên trượt nhiều.
- Nếu target pose là crawling, body phải hạ thấp và tay/chân có pattern khác walking.

Đó là lý do representation có vẻ nhiều chiều hơn `qpos`, nhưng lại dễ học hơn cho motion generation.

### 2.15. Điểm Dễ Nhầm

- `GlobalRootGlobalJoints` trong config không có nghĩa là toàn hệ thống chỉ dùng global. Nó là global view có `dual_rep` để sinh local view.
- `batch["motion"]` là 414 dims, không phải 418 dims.
- `mean.npy/std.npy` là 418 dims, vì stats lưu full dual layout.
- `local_root_vel` hiện không heading-aligned vì `removing_heading=False`.
- `local_vel` cũng không phải MuJoCo local joint velocity; nó là joint velocity XYZ trong motion representation.
- `global_rot_data` không phải 29 hinge angles. Nó là 34 global joint rotations dạng 6D.
- `foot_contacts` là feature học/đánh loss, không phải contact solver trực tiếp của MuJoCo.

---

## 3. Skeleton G1

Skeleton chính là `G1Skeleton34` trong `motionlib/core/skeletons/g1.py`.

Nó có 34 joints:

- 32 joints của G1 trong motion skeleton.
- 2 dummy toe joints: `left_toe_base`, `right_toe_base`.

MuJoCo model demo là 29-DOF:

| qpos range | Nội dung |
| --- | --- |
| `0:3` | root translation |
| `3:7` | root quaternion |
| `7:36` | 29 hinge joint angles |

`mujoco_qpos_converter` map giữa 34-joint motion skeleton và 29 hinge joints trong XML. Dead joints như toe/hand endpoints được populate bằng identity hoặc parent rotation tùy scheme.

---

## 4. Ba Model Chính

### VQVAE

File:

```text
motionbricks/motionbricks/vqvae/models/motion_vqvae.py
motionbricks/scripts/train_vqvae.py
```

Vai trò:

- Encode motion pose thành discrete code/token.
- Decode token về local/global motion feature.
- Root VQVAE chưa được dùng trong bản này; `root_net` đang là `None`.

Training step:

1. Nhận normalized global motion từ batch.
2. Randomize first heading bằng `change_first_heading`.
3. Convert global -> local qua `dual_rep.global_to_local`.
4. Encode/decode bằng pose VQVAE.
5. Loss chính gồm reconstruction, commit loss, foot skate contact, optional joint velocity.

### Pose Model

File:

```text
motionbricks/motionbricks/motion_backbone/models/pose_model.py
motionbricks/scripts/train_pose.py
```

Vai trò:

- Load checkpoint VQVAE.
- Freeze pose VQVAE.
- Encode ground-truth motion thành pose tokens.
- Train backbone để dự đoán masked/perturbed pose tokens từ root condition và sparse pose keyframes.

Trong config hiện tại, pose backbone condition root lấy từ global representation:

```yaml
cond_root_feature_is_from_motion_rep: global
cond_root_feature: root_without_hip_height
local_pose_feature: joint_positions_and_rotations_and_hip_height
```

### Root Model

File:

```text
motionbricks/motionbricks/motion_backbone/models/root_model.py
motionbricks/scripts/train_root.py
```

Vai trò:

- Không dùng VQVAE.
- Nhận sparse constraints về global root, local root, local pose.
- Predict continuous `pred_global_root_values`.
- Predict số token/frame interval qua `num_token_logits`.

Loss gồm global root reconstruction, local root reconstruction, và num-token classification.

---

## 5. Training Scripts Hiện Tại

Ba script trong `motionbricks/scripts/` không đọc `conf/` riêng. Chúng đọc `out/<model>/version_1/hparams.yaml`, patch path để chạy local/single GPU, rồi dùng synthetic random tensors.

```bash
cd /home/tung/GR00T-WholeBodyControl/motionbricks
python scripts/train_vqvae.py --max_steps 100
python scripts/train_pose.py --max_steps 100
python scripts/train_root.py --max_steps 100
```

`SyntheticMotionDataset` chỉ đảm bảo shape hợp lệ để exercise training code. Nó không sinh motion vật lý đúng. Vì vậy loss không phản ánh chất lượng robot.

Config checkpoint vẫn còn tham chiếu một số class full dataset như `motionbricks.motionlib.data.motion_dataset.MotionDataset`, nhưng package `motionlib/data` không có trong source public hiện tại. Runtime demo G1 tránh đường này bằng cách dùng `out/G1-clip.ckpt`.

---

## 6. Interactive Demo

Entrypoint:

```bash
cd /home/tung/GR00T-WholeBodyControl/motionbricks
python scripts/interactive_demo_g1.py
```

Bootstrap trong `navigation_demo`:

1. `_parse_args()` set default path:
   - `assets/skeletons/g1/scene_29dof.xml`
   - `assets/skeletons/g1/g1.xml`
   - `out/G1-clip.ckpt`
   - `out/`
2. `_initialize_inference_modles()` gọi `exp_setup.experiment.test(...)` để instantiate pose/root model từ hparams.
3. Load state dict từ:
   - `out/motionbricks_pose/version_1/checkpoints/model-step=2000000.ckpt`
   - `out/motionbricks_root/version_1/checkpoints/model-step=2000000.ckpt`
4. Tạo `motion_inference`.
5. Tạo `full_navigation_agent`.
6. Tạo `WASD_controller`.
7. Tạo MuJoCo model/data.

Lưu ý nhỏ trong CLI: `interactive_demo_g1.py` parse `--humanoid_xml`, nhưng `navigation_demo` dùng `args.humanoid_scene_xml`. Nếu cần override XML từ CLI, nên sửa script hoặc set `args.humanoid_scene_xml` trước khi tạo `navigation_demo`.

### 6.1. Sequence Chi Tiết Khi Chạy `interactive_demo_g1.py`

Sequence dưới đây mô tả nhánh mặc định khi chạy từ thư mục `motionbricks/`:

```bash
python scripts/interactive_demo_g1.py
```

#### Phase A: Python entrypoint và argparse

1. Python load `scripts/interactive_demo_g1.py`.
2. File import các dependency chính:
   - `torch as t`
   - `mujoco`, `mujoco.viewer`
   - `numpy`
   - `navigation_demo` từ `motionbricks.motion_backbone.demo.utils`
3. Block `if __name__ == "__main__":` tạo `ArgumentParser`.
4. Parser gán default runtime:
   - `--humanoid_xml="assets/skeletons/g1/scene_29dof.xml"`
   - `--result_dir="./out"`
   - `--data_root="./datasets"`
   - `--controller="wasd"`
   - `--has_viewer=1`
   - `--use_qpos=1`
   - `--planner="default"`
   - `--clips="G1"`
   - `--max_steps=10000`
   - `--num_runs=1`
5. Sau `parser.parse_args()`, script patch thêm các field runtime:
   - `args.return_model_configs = True`
   - `args.return_dataloader = True`
   - `args.recording_dir = None`
   - `args.EXP = args.planner`
   - `args.speed_scale = [0.8, 1.2]` sau khi parse string `"0.8,1.2"`
6. Script gọi `main(args)`.

Điểm cần để ý: parser có `args.humanoid_xml`, nhưng phần bootstrap bên dưới không dùng field này. `navigation_demo` dùng `args.humanoid_scene_xml`; vì field đó chưa tồn tại nên nó tự set default absolute path tới scene G1.

#### Phase B: Tạo `navigation_demo`

1. `main(args)` gọi:

```python
demo_agent = navigation_demo(args)
```

2. `navigation_demo.__init__` khởi tạo placeholder:
   - `self.full_agent = None`
   - `self.controller = None`
   - `self.mj_model = None`
   - `self.mj_data = None`
3. Sau đó gọi lần lượt:
   - `self._parse_args()`
   - `self._initialize_inference_modles()`
   - `self._initialize_controller()`
   - `self._initialize_mj_simulator()`

#### Phase C: `_parse_args()`

1. Ép:
   - `args.return_model_configs = True`
   - `args.return_dataloader = True`
2. Tính `project_base_path` bằng cách đi lên 4 cấp từ file `motion_backbone/demo/utils.py`, kết quả là thư mục subproject `motionbricks/`.
3. Nếu field chưa tồn tại, set default absolute path:
   - `args.humanoid_scene_xml = <motionbricks>/assets/skeletons/g1/scene_29dof.xml`
   - `args.skeleton_xml = <motionbricks>/assets/skeletons/g1/g1.xml`
   - `args.clips_ckpt = <cwd-or-result_dir>/out/G1-clip.ckpt`
4. Vì parser đã tạo sẵn `result_dir`, `data_root`, và `explicit_dataset_folder`, các field này không được `_parse_args()` đổi sang absolute path. Mặc định thực tế vẫn là:
   - `args.result_dir = "./out"`
   - `args.data_root = "./datasets"`
   - `args.explicit_dataset_folder = None`

#### Phase D: `_initialize_inference_modles()`

1. Lấy `reprocess_clips = args.reprocess_clips`, mặc định là `0`.
2. Kiểm tra clip cache:

```python
if args.clips_ckpt is None or not os.path.exists(args.clips_ckpt) or reprocess_clips:
    models, confs, train_dataloader, val_dataloader = test(args)
else:
    args.return_dataloader = False
    models, confs = test(args)
```

3. Nhánh mặc định của repo là `out/G1-clip.ckpt` tồn tại, nên code:
   - Set `args.return_dataloader = False`
   - Gọi `test(args)` chỉ để instantiate model/config
   - Set `args.train_dataloader = None`
   - Set `args.val_dataloader = None`

Nếu `G1-clip.ckpt` không tồn tại hoặc `--reprocess_clips=1`, code sẽ cố instantiate dataloader từ config checkpoint. Trong source public hiện tại, các class full dataset được config tham chiếu không có đủ trong package, nên nhánh mặc định ổn định là dùng clip cache.

#### Phase E: `exp_setup.experiment.test(args)`

`test(args)` là helper inference setup, không phải unit test.

1. Đọc `args.EXP`; mặc định là `"default"`.
2. `get_path_dir("default")` trả về mapping checkpoint:
   - VQVAE: `motionbricks_vqvae/version_1/checkpoints/model-step=2000000.ckpt`
   - Pose: `motionbricks_pose/version_1/checkpoints/model-step=2000000.ckpt`
   - Root: `motionbricks_root/version_1/checkpoints/model-step=2000000.ckpt`
3. Loop qua `model_name in ["pose", "root"]`.
4. Với mỗi model:
   - Tạo `ckpt_dir = f"{args.result_dir}/{model_path}/version_1"`
   - Tạo `ckpt_path = f"{ckpt_dir}/checkpoints/{model_ckpt}"`
   - Load `hparams.yaml` bằng `OmegaConf.load(config_path)`
   - Gán `conf.ckpt_path = ckpt_path`
5. Nếu config có `conf.model.args.vqvae_model_ckpt_path`, path VQVAE được sanitize:
   - Prefix `./out` hoặc `out/` được thay bằng `args.result_dir`
   - Sau đó chuyển sang absolute path
6. Gọi `load_motion_rep(conf)`:
   - Hydra instantiate `conf.skeleton`
   - Với G1, class là `motionbricks.motionlib.core.skeletons.g1.G1Skeleton34`
   - Skeleton load `joints.p`, `parents.p` từ `out/<model>/version_1/skeleton`
   - Hydra instantiate `conf.motion_rep`
   - Motion rep load full dual stats từ `out/<model>/version_1/stats/motion`
7. Nếu đang instantiate pose model:
   - Instantiate `pose_vqvae_network` với `motion_rep.dual_rep.local_motion_rep`
   - Đảm bảo checkpoint VQVAE đúng tên/path `motionbricks_vqvae/.../model-step=2000000.ckpt`
8. Tạo temp `run_dir` để logging.
9. Strip `optimizer` và `scheduler` khỏi `conf.model` vì đây là inference setup.
10. Instantiate backbone:

```python
backbone_network = instantiate(conf.model.backbone_network, motion_rep=motion_rep)
```

11. Instantiate Lightning model:
   - Pose: `motionbricks.motion_backbone.models.pose_model.MotionModel`
   - Root: `motionbricks.motion_backbone.models.root_model.MotionModel`
12. Trong constructor pose model:
   - `_load_vqvae_models()` load VQVAE checkpoint
   - Copy weights vào `pose_net`
   - Init embedding của pose backbone từ codebook VQVAE nếu backbone chưa init
   - Freeze `pose_net`
13. Trong constructor root model:
   - `_load_vqvae_models()` chỉ set `_vqvae_model_loaded=True`
   - Root model không dùng VQVAE
14. Vì `args.return_model_configs=True` và `args.return_dataloader=False`, `test(args)` return:

```python
return models, confs
```

#### Phase F: Load checkpoint weights pose/root

Quay lại `_initialize_inference_modles()`:

1. Loop qua `model_name in ["pose", "root"]`.
2. Load checkpoint:

```python
state_dict = t.load(confs[model_name].ckpt_path)["state_dict"]
models[model_name].load_state_dict(state_dict)
```

3. Tạo inferencer:

```python
self.inferencer = motion_inference(models, models["pose"].args)
```

4. `motion_inference.__init__` giữ references:
   - `self._pose_model = models["pose"].eval().to("cuda")`
   - `self._root_model = models["root"].eval().to("cuda")`
   - `self._vqvae_pose_model = pose_model.supporting_nets["pose_net"].eval().to("cuda")`
   - `self.global_motion_rep`
   - `self.local_motion_rep`
   - `self.motion_rep`
5. Inferencer assert:
   - pose backbone đã init
   - pose model đã load VQVAE
   - root model ở trạng thái loaded
   - root model không tokenized

#### Phase G: Tạo `full_navigation_agent`

`_initialize_inference_modles()` import và instantiate:

```python
self.full_agent = full_navigation_agent(
    self.inferencer,
    self.args.train_dataloader,
    device="cuda",
    speed_scale=speed_scale,
    target_root_realignment=target_root_realignment,
    source_root_realignment=source_root_realignment,
    force_canonicalization=force_canonicalization,
    skeleton_xml=self.args.skeleton_xml,
    skip_ending_target_cond=skip_ending_target_cond,
    filter_qpos=args.pre_filter_qpos,
    clips=args.clips,
    ckpt_path=args.clips_ckpt,
    reprocess_clips=reprocess_clips,
    val_dataloader=self.args.val_dataloader,
).to("cuda")
```

Trong `full_navigation_agent.__init__`:

1. Đưa inferencer sang eval/cuda.
2. Deepcopy `inferencer.motion_rep` sang `self._motion_rep`.
3. Tạo MuJoCo converter:

```python
self._converter = get_mujoco_converter(self._motion_rep, skeleton_xml).to(device)
```

4. `mujoco_qpos_converter` parse `assets/skeletons/g1/g1.xml`:
   - Lấy danh sách hinge joints trong XML
   - Map joint XML `*_joint` sang skeleton `*_skel`
   - Tạo mapping MuJoCo joint index -> motion skeleton index
   - Tạo mapping motion skeleton index -> MuJoCo joint index
   - Lưu axis joint, parent list, neutral MuJoCo joints, và coordinate transform
5. Tạo clip holder:

```python
self._clip_holder = clip_holder_G1(
    train_dataloader=None,
    ckpt_path=args.clips_ckpt,
    converter=self._converter,
    reprocess_clips=0,
)
```

6. Vì `G1-clip.ckpt` tồn tại, `clip_holder` load state dict từ cache:
   - `global_root_positions`
   - `global_joint_positions`
   - `global_joint_rotations`
   - `global_headings`
   - `motion_feature`
   - `mujoco_qpos`
   - `num_frames_per_clip`
7. `clip_holder_G1._apply_root_headings_correction()` chỉnh heading cho crawling clips.
8. Agent lưu các flag:
   - `PRED_OFFSETS = 4`
   - `NUM_FRAMES_PER_TOKEN = 4`
   - `FILTER_QPOS = args.pre_filter_qpos`
   - `FORCE_CANONICALIZATION = args.force_canonicalization`
9. Gọi `_initialize_frames()`:
   - Lấy idle clip đầu tiên từ `self._clip_holder.motion_feature[0]`
   - Convert idle motion feature -> MuJoCo qpos
   - Reorder root quaternion từ `[x, y, z, w]` sang `[w, x, y, z]`
   - Pad qpos buffer tối thiểu 64 frames
   - Set `_current_frame_idx = 0`

#### Phase H: Tạo controller

`navigation_demo._initialize_controller()`:

1. Lấy:
   - `min_tokens = self.inferencer._args["min_tokens"]`
   - `max_tokens = self.inferencer._args["max_tokens"]`
2. Vì `args.controller == "wasd"`, tạo:

```python
self.controller = WASD_controller(
    lookat_movement_direction=args.lookat_movement_direction,
    clips=args.clips,
    min_token=min_tokens,
    max_token=max_tokens,
)
```

3. `base_controller.__init__` set:
   - `_FPS = 30`
   - `_CONTROLLER_DT = 8 / 30`
   - `_clip_holder_class = clip_holder_G1`
4. Keyboard listener chưa nhất thiết được tạo ngay. Với Linux/macOS, `KeyboardHandler` được tạo lazy khi lần đầu đọc `snapshot_keyboard_control`.

#### Phase I: Tạo MuJoCo simulator

`navigation_demo._initialize_mj_simulator()` gọi:

```python
build_mj_simulator(args.humanoid_scene_xml, self.inferencer.motion_rep.fps)
```

1. `mujoco.MjModel.from_xml_path(<scene_29dof.xml>)` load MuJoCo model.
2. `mujoco.MjData(mj_model)` tạo runtime data.
3. Tắt hoặc giảm một số visual effects để tăng tốc.
4. Set:

```python
mj_model.opt.timestep = 1 / fps
```

Với G1 checkpoint hiện tại, `fps = 30`, nên timestep là `1/30`.

Đến đây `navigation_demo(args)` hoàn tất và `main(args)` đã có:

```python
demo_agent.full_agent
demo_agent.controller
demo_agent.mj_model
demo_agent.mj_data
```

#### Phase J: Main run loop

`main(args)` bắt đầu loop theo `args.num_runs`, mặc định 1 lần:

1. Tăng `num_runs`.
2. Tạo seed mới từ `args.random_seed`.
3. Gọi:

```python
demo_agent.full_agent.reset()
```

4. `reset()` set `_current_frame_idx = 0` và gọi lại `_initialize_frames()` để buffer bắt đầu từ idle clip.

#### Phase K: Viewer loop

Vì `args.has_viewer=1`, code mở MuJoCo viewer:

```python
with mujoco.viewer.launch_passive(demo_agent.mj_model, demo_agent.mj_data) as viewer:
```

1. Trên Linux, `_disable_mujoco_keyboard_shortcuts()` cố grab các phím `wasdrtfgeqzxcvb` ở X11 để MuJoCo viewer không ăn mất phím điều khiển.
2. Bắt đầu while loop:

```python
while viewer.is_running() and steps < args.max_steps:
```

Mỗi iteration là một frame simulation/render.

#### Phase L: Mỗi frame lấy qpos hiện tại

Trong mỗi iteration:

1. `force_idle = steps + 100 > args.max_steps`; 100 step cuối ép idle.
2. `steps += 1`.
3. Clear debug geoms:

```python
viewer.user_scn.ngeom = 0
```

4. Lấy qpos frame tiếp theo:

```python
qpos = demo_agent.full_agent.get_next_frame()
```

`get_next_frame()`:

- Lấy `self.frames["mujoco_qpos"][0, current_frame_idx]`.
- Tăng `_current_frame_idx` lên 1 nhưng không vượt cuối buffer.
- Trả về numpy qpos.

5. Lấy context cho inference kế tiếp:

```python
context_motion_features = full_agent.get_context_motion_features()
context_mujoco_qpos = full_agent.get_context_mujoco_qpos()
```

Hai hàm này lấy 4 frame context quanh current index, có offset `PRED_OFFSETS=4`.

6. Gán qpos hiện tại vào MuJoCo:

```python
demo_agent.mj_data.qpos[:] = qpos
```

Tại thời điểm này viewer vẫn chưa sync; qpos chỉ mới được ghi vào data.

#### Phase M: Controller đọc phím và tạo intent

Gọi:

```python
control_signals = demo_agent.controller.generate_control_signals(
    viewer,
    demo_agent.mj_model,
    demo_agent.mj_data,
    visualize=True,
    control_info={
        "force_idle": force_idle,
        "allowed_mode": args.allowed_mode,
    },
)
```

Trong `WASD_controller.generate_control_signals()`:

1. Nếu `_prev_qpos is None`, tạo history buffer `[5, mj_model.nq]` và fill bằng qpos hiện tại.
2. Đọc keyboard:
   - Linux/macOS: `snapshot_keyboard_control` tạo `KeyboardHandler` nếu chưa có.
   - `KeyboardHandler` start `pynput.keyboard.Listener` background thread.
   - Listener update set `_pressed_keys` trong `on_press`/`on_release`.
   - Snapshot trả về dict boolean cho `w`, `a`, `s`, `d`, arrows, style keys.
3. Xác định `mode`:
   - Không có WASD: `idle`
   - Có WASD: `walk`
   - Nếu style key được nhấn, override bằng mode tương ứng trong `clip_holder_G1.DEFAULT_KEYS`
4. Gọi `_generate_target_position_and_heading(...)`.
5. Trong `_generate_target_position_and_heading`:
   - Đọc `viewer.cam.lookat`, `distance`, `azimuth`, `elevation`.
   - Tính camera position.
   - Tính `camera_direction` trên mặt phẳng XY của MuJoCo.
   - Nếu mode không idle:
     - Gom WASD thành relative direction.
     - Rotate relative direction theo camera azimuth.
     - Sinh `movement_direction`.
     - `facing_direction` là movement direction nếu `lookat_movement_direction=True`, ngược lại là camera direction.
   - Nếu idle:
     - Ước lượng movement direction từ qpos velocity history.
     - Ước lượng facing direction từ root quaternion hiện tại.
6. Convert mode string sang mode id:

```python
mode = tensor([[index_in_CLIPS]])
```

7. Update `_prev_qpos` history.
8. Tạo output:

```python
{
    "movement_direction": Tensor[1, 3],
    "facing_direction": Tensor[1, 3],
    "mode": Tensor[1, 1],
    "allowed_pred_num_tokens": Tensor[1, 11],
}
```

#### Phase N: Gắn context vào control signal

Vì default `args.use_qpos=1`, code dùng qpos context:

```python
control_signals["context_mujoco_qpos"] = context_mujoco_qpos
```

Nếu `--use_qpos=0`, nó gắn:

```python
control_signals["context_motion_features"] = context_motion_features
```

Trong demo mặc định, đường `context_mujoco_qpos` là đường chính.

#### Phase O: `full_agent.generate_new_frames(...)`

Gọi:

```python
full_agent.generate_new_frames(
    control_signals,
    demo_agent.controller.get_controller_dt() * args.generate_dt,
)
```

Mặc định:

- `controller.get_controller_dt() = 8 / 30`
- `args.generate_dt = 2.0`
- `controller_dt` truyền vào agent là `16 / 30` giây
- Với `fps=30`, agent chỉ replan sau khoảng 16 frame, trừ khi buffer gần hết hoặc mode cần regenerate.

Đầu `generate_new_frames()` có early return:

1. Nếu `_current_frame_idx < controller_dt * fps` và chưa tới cuối buffer, return frames cũ.
2. Nếu `_should_regenerate(input)` trả false, cũng return frames cũ.
3. Với idle liên tục, `_should_regenerate` thường tránh regenerate không cần thiết.

Khi đủ điều kiện regenerate, agent chạy pipeline inference mới.

#### Phase P: Process context qpos thành motion transforms

`generate_new_frames()` gọi:

```python
_process_input_to_joint_transforms(input)
```

Với default `context_mujoco_qpos`:

1. Nếu `FORCE_CANONICALIZATION=True`, gọi `_canonicalize_mujoco_qpos(input)`.
2. `_canonicalize_mujoco_qpos`:
   - Lưu `raw_context_mujoco_qpos`.
   - Lấy vị trí root frame đầu.
   - Lấy heading root frame đầu từ quaternion.
   - Rotate toàn bộ context qpos về frame canonical.
   - Dời root X/Y MuJoCo frame đầu về origin, giữ Z/up theo transform.
   - Rotate `movement_direction` và `facing_direction` về canonical frame.
   - Lưu `first_frame_heading_angle` và `first_frame_position` để uncanonicalize sau.
3. Gọi converter:

```python
convert_mujoco_qpos_to_motion_transforms(context_mujoco_qpos)
```

4. Converter:
   - Đọc root translation/quaternion MuJoCo.
   - Convert 29 hinge DOF thành rotation matrices.
   - Chạy FK trên MuJoCo kinematic tree.
   - Convert MuJoCo coordinates sang motion coordinates.
   - Reorder joints theo motion skeleton.
   - Populate dead joints như toe/hand endpoints.
5. Output:
   - `context_global_joint_positions`
   - `context_global_joint_rotations`

#### Phase Q: Spring model tạo root target

Agent gọi:

```python
_generate_spring_model_position_and_heading(input)
```

Hàm này:

1. Lấy current root position XZ từ `context_global_joint_positions`.
2. Ước lượng current root velocity từ 2 frame đầu.
3. Clamp mode id không vượt số mode trong `clip_holder_G1.CLIPS`.
4. Tính tốc độ mục tiêu:
   - `idle`: dùng một phần velocity hiện tại.
   - mode khác: lấy `avg_root_vel` từ `clip_holder_G1.CLIPS[mode]`.
5. Nếu `random_speed_scale=True`, nhân speed theo random ratio trong range `args.speed_scale`.
6. Nếu input có `target_vel`, override speed.
7. Convert `movement_direction` từ MuJoCo convention sang motion XZ convention.
8. Nếu không có movement direction rõ ràng, dùng `facing_direction * 0.1` để hỗ trợ in-place turning.
9. Tính `target_root_pos = curr_root_pos + speed * direction`.
10. Dùng critical damping spring để tạo:
   - `start_root_positions`
   - `target_root_positions`
   - `target_root_position`
11. Tính current heading từ root rotation matrix.
12. Tính target heading từ `facing_direction`.
13. Dùng critical damping spring cho heading để tạo:
   - `start_root_headings`
   - `target_root_headings`
   - `target_root_heading`
14. Nếu `FORCE_CANONICALIZATION=True`, gọi:

```python
clip_holder_G1.blendspace_modes_remap_from_velocity(...)
```

Hàm này có thể remap `slow_walk`/`walk` sang `walk_left` hoặc `walk_right` nếu direction lệch ngang nhiều so với heading.

#### Phase R: Lấy target pose từ reference clip

Agent gọi:

```python
_generate_target_joint_transforms(input)
```

Hàm này:

1. Tạo one-hot mode theo `clip_holder_G1.CLIPS`.
2. Lấy `num_frames_per_clip` ứng với mode.
3. Chọn `frame_idx = random_seed % (num_frames_per_clip - 4)`.
4. Lấy 4 frame target từ buffers:
   - `global_root_positions`
   - `global_joint_positions`
   - `global_joint_rotations`
   - `global_headings`
   - `mujoco_qpos`
5. Dùng one-hot để select đúng clip theo mode.
6. Nếu `target_root_realignment=True`:
   - Tính heading diff giữa spring target heading và clip heading.
   - Rotate target joint rotations/positions theo heading diff.
   - Đặt target root XZ theo `target_root_positions`.
7. Nếu `source_root_realignment=True`:
   - Align context root heading với `start_root_headings`.
   - Align context root positions với `start_root_positions`.
8. Output:
   - `target_global_joint_positions`
   - `target_global_joint_rotations`
   - `target_global_root_positions`

#### Phase S: Assemble constraints và gọi model inference

Agent gọi:

```python
_generate_inbetween_frames(input)
```

Hàm này tạo input dày cho `motion_inference.predict(...)`:

1. Từ context transforms, tạo:
   - `context_global_root_values`: `[B, 4, 5]`
   - `context_local_root_values`: `[B, 4, 4]`
   - `context_local_poses`: joint positions + 6D rotations
2. Từ target transforms, tạo:
   - `target_global_root_values`: `[B, 4, 5]`
   - `target_local_root_values`: `[B, 4, 4]`
   - `target_local_poses`
3. Concat context + target thành 8 constraint frames:
   - `global_root_values`
   - `local_root_values`
   - `local_poses`
4. Tạo boolean masks:
   - `has_global_root_values`
   - `has_local_root_values`
   - `has_local_poses`
5. Tạo `num_tokens = MASKED_NUM_TOKENS`, để root model tự predict số token hợp lý.
6. Config inference:

```python
{
    "num_inference_step": 1,
    "smooth_root_traj": False,
    "allow_pred_out_of_reach_num_tokens": False,
    "pose_token_sampling_use_argmax": True,
    "skip_ending_target_cond": self.SKIP_ENDING_TARGET_COND,
}
```

7. Gọi:

```python
pred_global_motions, num_pred_tokens = self._inferencer.predict(...)
```

#### Phase T: Bên trong `motion_inference.predict(...)`

`motion_inference.predict(...)` nhận constraints ở dạng unnormalized.

1. Kiểm tra 4 frame đầu có đủ global root, local root, local pose.
2. Nếu `num_tokens` đang masked, giữ masked token count để root model predict.
3. `_extract_initial_root_info(global_root_values)`:
   - Lưu initial root XZ offset.
   - Lưu initial root heading.
   - Recenter global root XZ về origin.
4. Normalize:
   - `batch["global_root_values"]`
   - `batch["local_root_values"]`
   - `batch["local_poses"]`
5. `_predict_root_trajectories(batch, config)`:
   - Gọi root backbone.
   - Root backbone predict `pred_num_tokens`.
   - Root backbone predict normalized `pred_global_root_values`.
   - Convert global root -> local root bằng `dual_rep.global_to_local`.
   - Sửa final local root motion nếu target local root cuối được provide.
6. `_predict_pose_tokens(batch, config, info)`:
   - Init toàn bộ pose tokens bằng `POSE_MASK_ID`.
   - Tạo pose condition từ 4 frame đầu và 4 frame target cuối.
   - Lấy root condition từ predicted root.
   - Gọi pose backbone.
   - Vì `pose_token_sampling_use_argmax=True`, chọn token bằng `argmax`, không sample ngẫu nhiên.
7. `_decode_motions_from_predicted_root_and_pose_tokens(batch, config, info)`:
   - Lấy external root condition từ predicted local root.
   - Tạo token mask theo `pred_num_tokens`.
   - Gọi VQVAE decoder `forward_decoder(...)`.
   - Decoder trả `pred_poses` dạng local motion rep normalized.
   - Convert local -> global bằng `dual_rep.local_to_global`.
   - Mặc định final root lấy từ pose module (`final_root_pred_mode="from_pose_module"`).
8. `_reapply_initial_root_info(batch)`:
   - Add lại initial root XZ offset.
   - Add lại initial heading vào global heading.
9. Return:

```python
pred_global_poses, pred_num_tokens
```

#### Phase U: Convert predicted motion về MuJoCo qpos

Quay lại `_generate_inbetween_frames()`:

1. Lưu:

```python
self.frames["model_features"] = pred_global_motions
self.frames["num_pred_frames"] = 4 * num_pred_tokens
```

2. Convert predicted motion features sang qpos:

```python
self._converter.convert_motion_features_to_mujoco_qpos(...)
```

3. Converter:
   - Unnormalize nếu cần.
   - Extract root translation và root heading quaternion từ motion rep.
   - Extract `global_rot_data`.
   - Convert 6D rotations -> matrices.
   - Convert global joint rotations -> local joint rotations.
   - Apply rotation offsets từ MuJoCo XML.
   - Convert root translation từ motion space sang MuJoCo space.
   - Convert root rotation sang MuJoCo quaternion.
   - Project local joint rotation lên hinge axis để lấy 29 joint DOFs.
   - Output qpos `[B, T, 36]`.
4. Reorder root quaternion sang MuJoCo expected order trong demo.
5. Nếu `FORCE_CANONICALIZATION=True`, gọi `_uncanonicalize_mujoco_qpos(input)`:
   - Rotate root positions và rotations từ canonical frame về world frame ban đầu.
   - Add lại first frame position.
6. Set:

```python
self._current_frame_idx = self.NUM_FRAMES_PER_TOKEN - self.PRED_OFFSETS
```

Với default `4 - 4 = 0`, tức buffer mới sẽ phát từ frame đầu.

7. Nếu `FILTER_QPOS=True`, blend vài frame đầu với raw context qpos:
   - Blend root translation.
   - Blend joint DOFs.
   - Không blend root quaternion trong đoạn code này.
8. Return `model_features`, `mujoco_qpos`, `num_pred_frames`.

Quay lại `generate_new_frames()`, agent truncate buffer:

```python
self.frames["model_features"] = model_features[:, :num_pred_frames, :]
self.frames["mujoco_qpos"] = mujoco_qpos[:, :num_pred_frames, :]
```

#### Phase V: Forward physics/render

Quay lại loop trong `interactive_demo_g1.py`:

1. Sau `generate_new_frames(...)`, code gọi:

```python
mujoco.mj_forward(demo_agent.mj_model, demo_agent.mj_data)
```

2. Camera follow root history:

```python
viewer.cam.lookat[:] = demo_agent.controller.get_prev_qpos()[:, :3].mean(axis=0)
```

3. Sync viewer:

```python
viewer.sync()
```

4. Sleep phần thời gian còn lại để giữ nhịp theo `mj_model.opt.timestep`.
5. Frame kế tiếp quay lại Phase L:
   - `get_next_frame()` phát qpos tiếp theo trong buffer vừa generate.
   - Chỉ khi đi đủ replan interval hoặc cần regenerate, agent mới gọi model inference lần nữa.

#### Phase W: Nhánh no-viewer

Nếu chạy `--has_viewer=0`, code đi vào nhánh không mở MuJoCo viewer nhưng vẫn gọi `controller.generate_control_signals(None, ...)`.

Với default `--controller=wasd`, `WASD_controller` cần `viewer.cam`, nên nhánh no-viewer không phù hợp với WASD mặc định. Nếu muốn chạy không viewer để smoke test loop, nên dùng:

```bash
python scripts/interactive_demo_g1.py --has_viewer 0 --controller random
```

---

## 7. Từ Phím Bấm Đến Qpos

### 7.1. Keyboard Handler

File:

```text
motion_backbone/demo/controllers.py
```

Trên Linux/macOS, `pynput.keyboard.Listener` chạy background thread và duy trì set `_pressed_keys`. Trên Windows, code dùng package `keyboard`.

### 7.2. WASD Controller

`WASD_controller.generate_control_signals(...)` tạo:

```python
{
    "movement_direction": Tensor[1, 3],
    "facing_direction": Tensor[1, 3],
    "mode": Tensor[1, 1],
    "allowed_pred_num_tokens": Tensor[1, N],
}
```

Mode mặc định:

- Không có WASD: `idle`.
- Có WASD: `walk`.
- Style key như `z`, `x`, `b`, `r`, `t`, `c`, `e`, `f`, `g`, `q`, `v` override mode theo `clip_holder_G1.DEFAULT_KEYS`.

Hướng đi được tính theo camera azimuth của MuJoCo viewer, nên WASD là camera-relative.

### 7.3. Full Navigation Agent

`full_navigation_agent.generate_new_frames(...)` làm các bước:

1. Lấy context từ `context_mujoco_qpos` hoặc `context_motion_features`.
2. Canonicalize qpos về frame đầu nếu `FORCE_CANONICALIZATION=True`.
3. Convert MuJoCo qpos -> motion joint transforms.
4. Dùng spring model để sinh target root position/heading.
5. Lấy target pose từ reference clip theo `mode`.
6. Gọi `_inferencer.predict(...)`.
7. Convert predicted motion features -> MuJoCo qpos.
8. Uncanonicalize qpos về world frame.
9. Blend vài context frame đầu nếu `FILTER_QPOS=True`.

### 7.4. Motion Inference

`motion_inference.predict(...)` chạy theo thứ tự:

1. Recenter global root để motion bắt đầu ở origin.
2. Normalize global/local root và local pose conditions.
3. Root model predict `pred_num_tokens`, `pred_global_root_values`.
4. Convert root global -> local.
5. Pose model predict discrete pose tokens.
6. VQVAE decoder reconstruct local pose.
7. Convert local -> global motion features.
8. Add lại initial root offsets/headings.

---

## 8. Coordinate Conversion

Motion space:

- Y-up.
- Z-forward.

MuJoCo space:

- Z-up.
- X-forward.

Converter dùng ma trận:

```python
mujoco_to_motion = [[0, 1, 0],
                    [0, 0, 1],
                    [1, 0, 0]]
```

Tức là:

- Motion X = MuJoCo Y.
- Motion Y = MuJoCo Z.
- Motion Z = MuJoCo X.

Root quaternion cũng cần để ý order. Trong converter, có lúc output là `[x, y, z, w]`, sau đó demo đổi về `[w, x, y, z]` trước khi gán cho MuJoCo.

---

## 9. Các Điểm G1 Đang Hard-code

Nếu muốn đưa robot mới vào, đây là các điểm cần sửa trước:

- `full_agent.py` import và instantiate `clip_holder_G1` trực tiếp.
- `controllers.py` dùng `clip_holder_G1` trực tiếp.
- `clips.py` chỉ định nghĩa `clip_holder_G1`.
- `mujoco_helper.py` hard-code root `pelvis_skel` và qpos output 36.
- `demo/utils.py` default path đều trỏ G1.
- `build_dummy_mj_simulator()` dùng qpos 36.
- `interactive_demo_g1.py` parse `--humanoid_xml` nhưng runtime lại đọc `humanoid_scene_xml`.

Vì vậy, MotionBricks có kiến trúc motion-representation/model khá tách biệt, nhưng demo/runtime hiện tại vẫn là G1-first.

---

## 10. Tóm Tắt

Trong repo hiện tại, MotionBricks gồm ba lớp chính:

- Motion representation + skeleton system để biến motion thành feature.
- VQVAE, pose backbone, root backbone để sinh motion theo token/root constraints.
- Demo agent/controller/converter để biến phím bấm thành MuJoCo qpos thời gian thực.

Điểm cần nhớ khi đọc code: model abstraction tương đối tổng quát, nhưng data/demo/converter public hiện đang neo vào G1. Tích hợp robot mới cần cập nhật đồng thời skeleton, stats, checkpoints, MuJoCo converter, clip holder, và entrypoint demo.
