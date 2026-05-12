# Walkthrough: VR H3 Training Setup

## Mục tiêu

Mục tiêu của lượt làm việc này là chuẩn bị phần training cho robot `vr_h3` trước. Phần tạo demo script tương tự `interactive_demo_g1.py` chưa được xử lý trong bước này.

Giả định hiện tại:

- Asset robot đã có ở `motionbricks/assets/skeletons/vr_h3/`.
- Skeleton/config/stats placeholder cho `vr_h3` đã có ở các folder `out/motionbricks_vr_h3_*`.
- Mục tiêu trước mắt là chạy được smoke training bằng `SyntheticMotionDataset`, chưa phải full training với motion data retarget thật.

## Các file đã chỉnh

### `motionbricks/scripts/train_vqvae.py`

Đã thêm tham số:

```bash
--robot vr_h3
--num_workers <N>
```

Trước đó script hard-code config G1 ở:

```text
out/motionbricks_vqvae/version_1
```

Sau thay đổi, nếu chạy:

```bash
./.venv/bin/python scripts/train_vqvae.py --robot vr_h3
```

script sẽ dùng:

```text
out/motionbricks_vr_h3_vqvae/version_1
```

Ngoài ra VQVAE training đã được bật checkpointing để pose model có checkpoint VQVAE để load tiếp.

Checkpoint sẽ nằm ở:

```text
out/motionbricks_vr_h3_vqvae/version_1/checkpoints/
```

### `motionbricks/scripts/train_pose.py`

Đã thêm tham số:

```bash
--robot vr_h3
--vqvae_ckpt <path>
--num_workers <N>
```

Nếu không truyền `--vqvae_ckpt`, script sẽ tự tìm checkpoint mới nhất trong:

```text
out/motionbricks_vr_h3_vqvae/version_1/checkpoints/
```

Sau đó pose model sẽ dùng config:

```text
out/motionbricks_vr_h3_pose/version_1/hparams.yaml
```

### `motionbricks/scripts/train_root.py`

Đã thêm tham số:

```bash
--robot vr_h3
--num_workers <N>
```

Khi chạy với `--robot vr_h3`, script sẽ dùng:

```text
out/motionbricks_vr_h3_root/version_1
```

### `motionbricks/motionbricks/motionlib/core/skeletons/vr_h3.py`

Đã chỉnh foot joint mapping:

```python
left_foot_joint_names = ["left_ankle_roll_skel", "left_ankle_roll_skel"]
right_foot_joint_names = ["right_ankle_roll_skel", "right_ankle_roll_skel"]
```

Lý do: motion representation `DualRootGlobalJoints` đang hard-code `foot_contacts` là 4 channel:

```text
left foot, left toe, right foot, right toe
```

Trong khi skeleton `vr_h3` hiện chỉ có ankle roll joint, chưa có toe dummy joints. Vì vậy smoke training bị lỗi shape khi tính foot contact loss. Cách sửa tối thiểu là duplicate ankle-roll joint để đại diện tạm cho toe slot, giữ nguyên số joint/stats hiện có.

## Lệnh chạy training

Chạy từ folder MotionBricks:

```bash
cd /home/tung/GR00T-WholeBodyControl/motionbricks
```

Train VQVAE:

```bash
./.venv/bin/python scripts/train_vqvae.py --robot vr_h3 --max_steps 1000
```

Train pose:

```bash
./.venv/bin/python scripts/train_pose.py --robot vr_h3 --max_steps 1000
```

Train root:

```bash
./.venv/bin/python scripts/train_root.py --robot vr_h3 --max_steps 1000
```

Nếu môi trường cho phép multiprocessing, có thể tăng DataLoader workers:

```bash
./.venv/bin/python scripts/train_vqvae.py --robot vr_h3 --max_steps 1000 --num_workers 8
```

Trong sandbox hiện tại, `num_workers=2` gây lỗi permission từ multiprocessing socket, nên mặc định đã đổi về `0`.

## Verification đã chạy

Kiểm tra cú pháp:

```bash
python -m py_compile scripts/train_vqvae.py scripts/train_pose.py scripts/train_root.py
```

Kết quả: pass.

Smoke test VQVAE:

```bash
MPLCONFIGDIR=/tmp/mplconfig ./.venv/bin/python scripts/train_vqvae.py --robot vr_h3 --max_steps 1 --batch_size 1 --num_samples 2
```

Kết quả: pass.

Smoke test root:

```bash
MPLCONFIGDIR=/tmp/mplconfig ./.venv/bin/python scripts/train_root.py --robot vr_h3 --max_steps 1 --batch_size 1 --num_samples 2
```

Kết quả: pass.

Smoke test pose:

```bash
MPLCONFIGDIR=/tmp/mplconfig ./.venv/bin/python scripts/train_pose.py --robot vr_h3 --max_steps 1 --batch_size 1 --num_samples 2
```

Kết quả: pass, pose model load được VQVAE checkpoint.

Checkpoint smoke test 1-step đã được xóa lại vì chỉ là output tạm và chiếm khoảng 543 MB.

## Lưu ý quan trọng

Những thay đổi này mới đảm bảo pipeline training chạy được với config/skeleton/stats `vr_h3` ở mức smoke test.

Để train thật cho robot mới, vẫn cần motion data retarget đúng cho `vr_h3`, stats thật thay vì placeholder, và checkpoint được train đủ lâu. Synthetic data chỉ giúp kiểm tra code path, shape, config và model wiring.

## Next Step: Chuẩn bị data thật cho VR H3

Câu hỏi tiếp theo đã chốt: để train thật, việc cần làm tiếp theo là chuẩn bị motion data cho `vr_h3`.

Không đủ nếu chỉ có file animation raw. Data cần đạt đến dạng MotionBricks đọc được:

- Animation/motion mẫu đã được retarget sang kinematic tree của `vr_h3`.
- Joint order khớp với `H3Skeleton.bone_order_names_with_parents`.
- Motion có đủ thông tin để sinh feature: root translation, root rotation, joint rotations, posed joints, và foot contacts.
- Feature dimension khớp với representation hiện tại. Với `vr_h3` đang có `J=31`, `DualRootGlobalJoints` full stats là `12J + 10 = 382`, còn global subset dùng cho dataset là `12J + 6 = 378`.
- `mean.npy` và `std.npy` phải được tính lại từ data thật, không dùng placeholder toàn `0/1`.

Thứ tự công việc hợp lý:

1. Chọn format input motion hiện có: BVH, FBX, NPZ, hoặc format nội bộ.
2. Retarget motion sang skeleton `vr_h3`.
3. Convert motion đã retarget thành MotionBricks feature.
4. Tính lại stats thật cho `vr_h3`.
5. Viết hoặc chỉnh dataset để trả về:

```python
{"keyid": int, "motion": Tensor[T, 378]}
```

Sau khi có dataset thật, mới train VQVAE, rồi pose/root model bằng checkpoint và stats cùng skeleton `vr_h3`.

## Data Format: Có thể dùng PKL

Có thể dùng file `.pkl` làm input data, miễn là nội dung trong file đủ thông tin để convert sang MotionBricks feature cho `vr_h3`.

Điều quan trọng không phải extension `.pkl`, mà là schema bên trong. Một file `.pkl` phù hợp nên chứa tối thiểu:

- Root translation theo frame.
- Root rotation theo frame.
- Joint rotations cho đúng 31 joints của `H3Skeleton`.
- Joint order rõ ràng và khớp với `H3Skeleton.bone_order_names_with_parents`.
- FPS hoặc timestep.
- Optional: foot contacts. Nếu không có, có thể tính từ velocity/height của foot joints.

Pipeline mong muốn:

```text
retargeted_vr_h3.pkl
  -> loader đọc root/joint transforms
  -> converter sinh DualRootGlobalJoints feature
  -> dataset trả Tensor[T, 378]
  -> stats thật mean/std shape 382
  -> train VQVAE / pose / root
```

Nếu `.pkl` đang chứa sẵn feature `[T, 378]` đã normalize hoặc chưa normalize thì càng đơn giản, nhưng cần biết rõ nó là global subset hay full dual representation để không normalize/convert sai hai lần.

## Data Check: `motionbricks/datas`

Đã kiểm tra data mẫu trong:

```text
motionbricks/datas/230418/
```

Kết luận: các file `.pkl` hiện tại **chưa phải format train trực tiếp** `[T, 378]`. Chúng là zlib-compressed joblib pickle. Mỗi file chứa một clip với schema:

```text
root_trans_offset: [T, 3]
pose_aa:           [T, 29, 3]
dof:               [T, 28]
root_rot:          [T, 4]
smpl_joints:       [T, 24, 3]
fps:               30
```

Đã check toàn bộ 702 file:

- 702/702 file đọc được.
- Không thiếu key bắt buộc.
- Tất cả có `fps = 30`.
- Frame count giữa `root_trans_offset`, `pose_aa`, `dof`, `root_rot` khớp.

`pose_aa` có 29 joints: root + 28 body joints. `H3Skeleton` hiện có 31 joints, nên converter fill thêm identity rotation cho 2 head joints còn thiếu.

## Script Convert Data VR H3

Đã thêm script:

```text
motionbricks/scripts/prepare_vr_h3_data.py
```

Script này làm các việc:

1. Đọc `.pkl` dạng zlib + joblib.
2. Convert `pose_aa` axis-angle sang local rotation matrices.
3. Fill đủ 31 joints của `H3Skeleton`.
4. Sinh full dual feature `[T, 382]`.
5. Tách global subset `[T, 378]`.
6. Tính stats thật `mean.npy` / `std.npy` shape `(382,)`.
7. Normalize global subset và lưu dataset trainable.

Lệnh convert toàn bộ data:

```bash
cd /home/tung/GR00T-WholeBodyControl/motionbricks

./.venv/bin/python scripts/prepare_vr_h3_data.py \
  --input_dir datas \
  --output_dir datasets/vr_h3_motion_features
```

Output:

```text
datasets/vr_h3_motion_features/motions.pt
datasets/vr_h3_motion_features/stats/motion/mean.npy
datasets/vr_h3_motion_features/stats/motion/std.npy
```

Đã chạy thử với 5 clip:

```bash
./.venv/bin/python scripts/prepare_vr_h3_data.py \
  --input_dir datas \
  --output_dir datasets/vr_h3_motion_features_debug \
  --limit 5
```

Kết quả:

```text
Converted clips: 5
Failed files: 0
motions.pt clip đầu: [688, 378]
stats mean/std: [382]
```

## Training Với Dataset Đã Convert

Đã thêm `TorchMotionDataset` để train scripts đọc dataset `.pt` thay vì synthetic data.

Các train scripts hiện có thêm:

```bash
--dataset_pt datasets/vr_h3_motion_features/motions.pt
```

Ví dụ train VQVAE bằng data thật đã convert:

```bash
./.venv/bin/python scripts/train_vqvae.py \
  --robot vr_h3 \
  --dataset_pt datasets/vr_h3_motion_features/motions.pt \
  --max_steps 1000
```

Khi truyền `--dataset_pt`, train scripts sẽ tự dùng stats ở:

```text
datasets/vr_h3_motion_features/stats/motion/
```

Đã smoke test VQVAE bằng dataset debug 5 clip:

```bash
./.venv/bin/python scripts/train_vqvae.py \
  --robot vr_h3 \
  --dataset_pt datasets/vr_h3_motion_features_debug/motions.pt \
  --max_steps 1 \
  --batch_size 1
```

Kết quả: pass. Checkpoint smoke test đã được xóa lại vì chỉ là output tạm.

Đã smoke test root model bằng cùng dataset debug:

```bash
./.venv/bin/python scripts/train_root.py \
  --robot vr_h3 \
  --dataset_pt datasets/vr_h3_motion_features_debug/motions.pt \
  --max_steps 1 \
  --batch_size 1
```

Kết quả: pass.

Pose training với dataset thật cần chạy sau khi đã train VQVAE thật và có checkpoint trong `out/motionbricks_vr_h3_vqvae/version_1/checkpoints/`, hoặc truyền trực tiếp bằng `--vqvae_ckpt`.

## Demo VR H3: WASD Only

Đã cập nhật demo/runtime theo scope hiện tại: chỉ cần WASD với các mode cơ bản, chưa làm các style như zombie, happy, boxing, crawling.

Các thay đổi chính:

- `motion_backbone/demo/clips.py`
  - Thêm `get_clip_holder_class()`.
  - Thêm `clip_holder_VR_H3` với 3 mode:

```text
idle
slow_walk
walk
```

- `motion_backbone/demo/controllers.py`
  - Controller chọn clip holder theo `--clips`.

- `motion_backbone/demo/full_agent.py`
  - Agent chọn clip holder theo `--clips`, không còn hard-code `clip_holder_G1`.

- `motionbricks/helper/mujoco_helper.py`
  - Converter cache phân biệt theo XML path.
  - Bỏ qua free root joint trong MJCF.
  - Chỉ điều khiển các hinge joints có trong `H3Skeleton`.
  - Hand/finger joints trong MuJoCo được giữ trong qpos nhưng không map sang motion skeleton, nên chúng giữ giá trị 0.

- `motionbricks/exp_setup/experiment.py`
  - Thêm experiment `vr_h3`.
  - Tự tìm checkpoint mới nhất nếu có `last.ckpt` hoặc checkpoint khác trong folder.

- `motionbricks/scripts/build_vr_h3_clip_cache.py`
  - Tạo clip cache tối thiểu cho WASD từ `motions.pt`.

- `motionbricks/scripts/interactive_demo_vr_h3.py`
  - Script demo riêng cho VR H3.
  - Dùng:

```text
assets/skeletons/vr_h3/mjcf/origin/vr_h3_1_with_rh56e2_hand_scene.xml
assets/skeletons/vr_h3/mjcf/origin/vr_h3_1_with_rh56e2_hand.xml
out/VR_H3-clip.ckpt
--clips vr_h3
--planner vr_h3
```

Sau khi train xong VQVAE / pose / root thật, cần build clip cache:

```bash
cd /home/tung/GR00T-WholeBodyControl/motionbricks

./.venv/bin/python scripts/build_vr_h3_clip_cache.py \
  --dataset_pt datasets/vr_h3_motion_features/motions.pt \
  --output out/VR_H3-clip.ckpt
```

Sau đó chạy demo:

```bash
./.venv/bin/python scripts/interactive_demo_vr_h3.py
```

## VR H3 Feature Roundtrip Visualization

`scripts/check_vr_h3_feature_roundtrip.py` now supports MuJoCo playback for the
source qpos or the inverse-feature reconstructed qpos.

Use this to inspect whether:

```text
CSV/qpos -> motion feature -> inverse feature -> qpos
```

looks like the original retargeted clip.

Play the reconstructed qpos:

```bash
cd /home/tung/GR00T-WholeBodyControl/motionbricks

./.venv/bin/python scripts/check_vr_h3_feature_roundtrip.py \
  datas/vr_h3_data/vr_h3_1/211117/Neutral_walk_forward_003__A057.csv \
  --play reconstructed \
  --loop
```

Play the original source qpos for comparison:

```bash
./.venv/bin/python scripts/check_vr_h3_feature_roundtrip.py \
  datas/vr_h3_data/vr_h3_1/211117/Neutral_walk_forward_003__A057.csv \
  --play source \
  --loop
```

## VR H3 Training Input Audit

Before training, audit the selected CSV files so the input is explicit instead
of relying only on action names.

Small smoke test:

```bash
cd /home/tung/GR00T-WholeBodyControl/motionbricks

./.venv/bin/python scripts/audit_vr_h3_training_input.py \
  --allow_actions_file scripts/vr_h3_clean_forward_walk_actions.txt \
  --limit 10 \
  --output_dir out/vr_h3_training_input_audit_smoke
```

Full selected-set audit:

```bash
./.venv/bin/python scripts/audit_vr_h3_training_input.py \
  --allow_actions_file scripts/vr_h3_clean_forward_walk_actions.txt \
  --limit 0 \
  --output_dir out/vr_h3_training_input_audit_clean_forward_walk
```

Read:

```text
out/vr_h3_training_input_audit_clean_forward_walk/summary.md
out/vr_h3_training_input_audit_clean_forward_walk/clips.csv
```

Important columns:

```text
root_rot_err_mean_rad          root heading/orientation roundtrip error
joint_abs_err_max_rad          joint DOF roundtrip error
src_joint_limit_max_over_rad   source CSV joint-limit cleanliness
root_speed_xy_mps              clip translational speed
root_delta_y_m                 lateral drift in MuJoCo Y
```

## Lọc data train theo một style forward-walk sạch

Vấn đề phát hiện:

- Dùng filter rộng `idle,walk,jog,run,turn,start,stop,hurry` làm dataset bị trộn quá nhiều style.
- Chỉ riêng action có chữ `walk` đã có nhiều nhóm khác nhau: normal, injured, crutch, carry/lift, sideway, start/stop/turn, walk dog, talking, watering, hands on back.
- Với mục tiêu WASD baseline, train nhiều style cho cùng một động tác làm pose sinh ra dễ lẫn pha chân và dáng đi không tự nhiên.

Thay đổi đã làm:

- Thêm allowlist action sạch:

```text
motionbricks/scripts/vr_h3_clean_forward_walk_actions.txt
```

- Thêm option cho prepare script:

```text
--allow_actions_file
```

trong:

```text
motionbricks/scripts/prepare_vr_h3_data.py
```

- Update bash train:

```text
motionbricks/scripts/train_vr_h3_locomotion_toe.sh
```

để mặc định dùng allowlist thay vì include/exclude rộng.

Allowlist hiện tại:

```text
idle_loop
neutral_idle_loop
walk_ff_loop_180_R
walk_ff_loop_180_R_normal_pace
neutral_walk_180_R
walk_forward_loop
walk_forward_normal
Neutral_walk_forward
Relaxed_walk_forward
Loop_Forward_Walk
```

Kết quả filter:

```text
allow_actions: 10
csv_found: 142220
csv_selected: 1302
```

Output mặc định mới của bash:

```text
Dataset: datasets/vr_h3_motion_features_clean_forward_walk/motions.pt
Clip ckpt: out/VR_H3-clean-forward-walk-clip.ckpt
Plots: out/vr_h3_clean_forward_walk_training_plots
```

Verify đã chạy:

```bash
bash -n scripts/train_vr_h3_locomotion_toe.sh
./.venv/bin/python -m py_compile scripts/prepare_vr_h3_data.py scripts/inventory_vr_h3_csv_actions.py
./.venv/bin/python scripts/prepare_vr_h3_data.py \
  --input_dir datas/vr_h3_data/vr_h3_1 \
  --output_dir /tmp/vr_h3_clean_forward_walk_prepare_test \
  --allow_actions_file scripts/vr_h3_clean_forward_walk_actions.txt \
  --limit 3 \
  --progress_every 0
```

Prepare test:

```text
Converted clips: 3
Failed files: 0
CSV files selected: 3
Target FPS: 30.0
```

## Ignore CSV data local

Update `.gitignore` ở repo root để không đưa data/output train lớn vào git:

```text
motionbricks/.venv/
motionbricks/datas/
motionbricks/datasets/
motionbricks/out/
motionbricks/np
motionbricks/t
motionbricks/time
```

Lý do:

- `motionbricks/datas` là raw retarget CSV data, rất lớn và chỉ phục vụ train local.
- `motionbricks/datasets` là feature dataset được prepare từ raw CSV.
- `motionbricks/out` chứa checkpoint, metric, plot, debug rollout, clip cache sinh ra khi chạy train/demo.
- `motionbricks/.venv` là virtualenv local.
- Không ignore toàn bộ `*.csv` ở repo root để tránh vô tình che các CSV example/source đang được track ở module khác.

Verify:

```bash
git status --short
```

Kết quả không còn hiện `motionbricks/.venv`, `motionbricks/datas`, `motionbricks/datasets`, `motionbricks/out` trong status.

## Fix build clip cache cho dataset clean forward-walk

Lỗi gặp khi chạy bash train clean:

```text
ValueError: No clip matching 'hurry_idle_right_R_001__A349' found in dataset keyids
```

Nguyên nhân:

- `train_vr_h3_locomotion_toe.sh` đã chuyển sang dataset allowlist clean forward-walk.
- `build_vr_h3_clip_cache.py` vẫn hard-code các clip từ dataset cũ:
  - `hurry_idle_right_R_001__A349`
  - `walk_180_R_001__A349`
  - `walk_180_R_003__A349`
- Các keyid này không tồn tại trong `datasets/vr_h3_motion_features_clean_forward_walk/motions.pt`.

Fix ban đầu:

```text
idle_clip:      idle_loop_001__A021
slow_walk_clip: walk_forward_normal_001__A005
walk_clip:      walk_ff_loop_180_R_001__A046
```

Sau đó bỏ hard-code keyid cụ thể:

- `--idle_clip`, `--slow_walk_clip`, `--walk_clip` chuyển thành optional override.
- Nếu không truyền override, script tự chọn clip đầu tiên trong dataset theo pattern ưu tiên:
  - idle: `neutral_idle_loop`, `idle_loop`, `idle`
  - slow_walk: `walk_forward_normal`, `Relaxed_walk_forward`, `Neutral_walk_forward`, `walk_ff_loop_180_R_normal_pace`, `walk_forward_loop`, `walk`
  - walk: `walk_ff_loop_180_R`, `walk_forward_loop`, `neutral_walk_180_R`, `Loop_Forward_Walk`, `walk`
- Nếu dataset không có pattern phù hợp, script fail ngay với danh sách pattern đã thử và preview keyid đầu tiên, thay vì fail vì keyid hard-code cũ.

File sửa:

```text
motionbricks/scripts/build_vr_h3_clip_cache.py
```

Đồng thời thêm:

```python
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
```

để tránh warning matplotlib cache khi chạy script riêng.

Verify:

```bash
cd /home/tung/GR00T-WholeBodyControl/motionbricks

./.venv/bin/python scripts/build_vr_h3_clip_cache.py \
  --dataset_pt datasets/vr_h3_motion_features_clean_forward_walk/motions.pt \
  --output /tmp/VR_H3-clean-forward-walk-clip-test.ckpt
```

Kết quả sau khi bỏ hard-code:

```text
idle: neutral_idle_loop_001__A073 frames=64
slow_walk: walk_forward_normal_001__A005 frames=64
walk: walk_ff_loop_180_R_001__A046 frames=64
Saved VR H3 clip cache: /tmp/VR_H3-clean-forward-walk-auto-clip-test.ckpt
```

## Đánh giá train clean forward-walk

Log train clean forward-walk:

```text
Converted clips: 1302
Failed files: 0
CSV files found: 142220
CSV files selected: 1302
Allow actions:
  Loop_Forward_Walk
  Neutral_walk_forward
  Relaxed_walk_forward
  idle_loop
  neutral_idle_loop
  neutral_walk_180_R
  walk_ff_loop_180_R
  walk_ff_loop_180_R_normal_pace
  walk_forward_loop
  walk_forward_normal
Feature dim: 402
```

Train steps:

```text
VQVAE: 20000 steps, 123 epochs over filtered dataset
Pose:  50000 steps, 307 epochs
Root:  50000 steps, 307 epochs
```

Metric summary:

```text
VQVAE train_loss_epoch:             2.29893 -> 0.145832
VQVAE recons_pose_epoch:            1.79627 -> 0.0894741
VQVAE joint_vel_epoch:              0.249885 -> 0.0254812
VQVAE skate_contact_epoch:          0.239148 -> 0.050585
VQVAE perplexity_pose_epoch:        6.8371 -> 7.87079

Pose train_loss_epoch:              2.25532 -> 0.834075

Root train_loss_epoch:              8.66924 -> 2.00827
Root global_root_recons_epoch:      2.44874 -> 0.015092
Root local_root_recons_epoch:       1.22246 -> 0.069289
Root num_token_loss_epoch:          2.5493 -> 1.90879
Root top_1_accuracy_epoch:          0.0867896 -> 0.289447
Root top_3_accuracy_epoch:          0.274578 -> 0.554816
Root top_5_accuracy_epoch:          0.465054 -> 0.6875
```

Nhận định:

- Data prepare sạch: không có file fail, số clip giảm từ 60450/142220 xuống 1302 clip đúng allowlist.
- VQVAE học tốt: reconstruction giảm mạnh, contact/skate loss giảm, perplexity quanh 7-8 là không bị collapse codebook.
- Pose model tốt hơn đợt train rộng: pose loss cuối khoảng `0.834`, thấp hơn đợt locomotion rộng trước đó khoảng `1.17`.
- Root reconstruction rất tốt: global root recon xuống `0.015`, local root recon xuống `0.069`.
- Root token prediction vẫn là điểm cần chú ý: top-1 khoảng `29%`, top-5 khoảng `69%`. Tốt hơn đợt trước nhưng chưa đảm bảo demo WASD mượt nếu root mode/token bị chọn sai.
- Vì dataset chỉ 1302 clip nhưng train 307 epochs, có rủi ro overfit. Cần xem demo/rollout thực tế để kết luận.

Đã thử chạy:

```bash
./.venv/bin/python scripts/debug_vr_h3_rollout.py --steps 180 --warmup_steps 30 --key w
```

Trong môi trường tool hiện tại bị lỗi:

```text
RuntimeError: No CUDA GPUs are available
```

Nên phần gait rollout cần chạy trực tiếp trên máy có GPU của user để xác nhận:

```bash
cd /home/tung/GR00T-WholeBodyControl/motionbricks

./.venv/bin/python scripts/debug_vr_h3_rollout.py \
  --steps 180 \
  --warmup_steps 30 \
  --key w

./.venv/bin/python scripts/analyze_vr_h3_gait.py
```

## Fix default cache path cho clean forward-walk demo/debug

User chạy debug sau khi train clean nhưng output primitive vẫn giống cache cũ.

Nguyên nhân:

- `train_vr_h3_locomotion_toe.sh` tạo cache mới:

```text
out/VR_H3-clean-forward-walk-clip.ckpt
```

- Nhưng các script demo/debug/analyze vẫn default về cache cũ:

```text
out/VR_H3-locomotion-toe-clip.ckpt
```

File đã sửa default:

```text
motionbricks/scripts/interactive_demo_vr_h3.py
motionbricks/scripts/debug_vr_h3_rollout.py
motionbricks/scripts/analyze_vr_h3_gait.py
```

Default mới:

```text
out/VR_H3-clean-forward-walk-clip.ckpt
```

Verify primitive cache mới:

```bash
cd /home/tung/GR00T-WholeBodyControl/motionbricks
./.venv/bin/python scripts/analyze_vr_h3_gait.py --cache out/VR_H3-clean-forward-walk-clip.ckpt
```

Primitive mới:

```text
primitive idle:
  root_delta_xy: [0.0037, -0.0019]
  left_lift_p95: 0.0011
  right_lift_p95: 0.0005

primitive slow_walk:
  root_delta_xy: [0.7992, 0.0410]
  left_lift_p95: 0.2050
  right_lift_p95: 0.1118
  left_right_lift_corr: -0.3377

primitive walk:
  root_delta_xy: [0.5436, 0.0063]
  left_lift_p95: 0.2494
  right_lift_p95: 0.0860
  left_right_lift_corr: -0.2533
```

Lưu ý:

- Lần verify này vẫn dùng rollout NPZ cũ cho phần `generated rollout after warmup`.
- Cần chạy lại `debug_vr_h3_rollout.py` sau khi đổi default cache để có rollout mới đúng cache clean.

## Đánh giá rollout clean cache mới

User chạy lại sau khi default cache đã chuyển sang:

```text
out/VR_H3-clean-forward-walk-clip.ckpt
```

Kết quả rollout `w`:

```text
root_delta_after_warmup_xyz: [6.0242, -0.4294, -0.0176]
max_joint_limit_violation_rad: 0.0
joint_limit_violation_frames: 0
mode ids seen: [0, 2]
```

Gait generated sau warmup:

```text
frames: 150
root_delta_xy: [6.0242, -0.4294]
left_lift_p95:  0.0738
right_lift_p95: 0.1081
forward_step_asymmetry: 0.2745
left_right_lift_corr: 0.7880
```

Primitive clean cache:

```text
primitive idle:
  root_delta_xy: [0.0037, -0.0019]
  left_lift_p95: 0.0011
  right_lift_p95: 0.0005

primitive slow_walk:
  root_delta_xy: [0.7992, 0.0410]
  left_lift_p95: 0.2050
  right_lift_p95: 0.1118
  left_right_lift_corr: -0.3377

primitive walk:
  root_delta_xy: [0.5436, 0.0063]
  left_lift_p95: 0.2494
  right_lift_p95: 0.0860
  left_right_lift_corr: -0.2533
```

Nhận định:

- Joint limit sạch: không còn vặn khớp vượt giới hạn.
- Primitive cache sạch có pha chân đúng hơn: `left_right_lift_corr` âm cho slow_walk/walk.
- Generated rollout vẫn chưa đạt: `left_right_lift_corr = 0.7880` là hai chân nâng gần cùng pha, giống hopping/shuffling hơn walking tự nhiên.
- Root đi quá nhanh: `6.024m / 150 frames @ 30 FPS = ~1.20 m/s`, trong khi primitive walk chỉ khoảng `0.54m / 64 frames = ~0.25 m/s`.
- Vấn đề tiếp theo không còn nằm ở cache primitive hay joint limit, mà nằm ở root/pose generation khi bị điều khiển bằng WASD: target/root speed và token generation chưa giữ được gait phase của primitive.

Hướng xử lý tiếp:

- Thử giảm speed control trong demo/debug trước khi train lại.
- Nếu giảm speed giúp gait tự nhiên hơn, update default VR H3 speed/target velocity.
- Nếu vẫn bị hai chân cùng pha, cần kiểm tra conditioning của pose/root model hoặc tăng data clean walk loop cùng style.

## Fix speed_scale không có tác dụng trong VR H3 debug/demo

User chạy:

```bash
./.venv/bin/python scripts/debug_vr_h3_rollout.py \
  --steps 180 \
  --warmup_steps 30 \
  --key w \
  --speed_scale 0.25,0.25
```

nhưng rollout y hệt `speed_scale 1.0,1.0`.

Nguyên nhân:

- `motionbricks/motion_backbone/demo/utils.py` chỉ truyền `args.speed_scale` vào `full_navigation_agent` khi `args.random_speed_scale=True`.
- `debug_vr_h3_rollout.py` hard-code `random_speed_scale=0`, nên `--speed_scale` bị bỏ qua.
- `interactive_demo_vr_h3.py` cũng default `random_speed_scale=0`, nên user truyền `--speed_scale` cũng không tác dụng nếu không truyền thêm `--random_speed_scale 1`.

Fix:

- `scripts/debug_vr_h3_rollout.py`: set `random_speed_scale=1` trong `build_args`.
- `scripts/interactive_demo_vr_h3.py`: đổi default:

```text
random_speed_scale: 1
speed_scale:        1.0,1.0
```

Như vậy:

- Default demo vẫn deterministic vì min=max=1.0.
- Khi truyền `--speed_scale 0.25,0.25`, scale được áp dụng thật.
- Nếu muốn random speed, có thể truyền khoảng như `--speed_scale 0.8,1.2`.

Verify:

```bash
./.venv/bin/python -m py_compile scripts/debug_vr_h3_rollout.py scripts/interactive_demo_vr_h3.py
```

Đã pass. Chạy rollout trong tool hiện tại vẫn lỗi do môi trường không thấy CUDA:

```text
RuntimeError: No CUDA GPUs are available
```

Nên cần user chạy lại trên máy GPU để kiểm tra số gait mới.

## Thêm target_vel override cho VR H3 speed debug/demo

Sau khi bật `speed_scale`, user chạy `--speed_scale 0.25,0.25`:

```text
root_delta_after_warmup_xyz: [5.3833, -0.4090, -0.0128]
left_right_lift_corr: 0.6256
```

So với trước:

```text
root_delta_after_warmup_xyz: [6.0242, -0.4294, -0.0176]
left_right_lift_corr: 0.7880
```

Nhận định:

- `speed_scale` đã có tác dụng nhưng yếu, root speed chỉ giảm khoảng 10%.
- Gait vẫn sai phase: `left_right_lift_corr` vẫn dương cao.
- Cần một cơ chế ép target root speed trực tiếp hơn.

Thay đổi:

- Thêm helper vào `scripts/interactive_demo_g1.py`:

```text
_apply_optional_target_vel(control_signals, args)
```

- Thêm option:

```text
--target_vel
```

vào:

```text
scripts/debug_vr_h3_rollout.py
scripts/interactive_demo_vr_h3.py
```

Ý nghĩa:

- `--target_vel` là tốc độ root mong muốn theo m/s cho non-idle mode.
- Vì `full_agent` nhân `target_vel` với `2.0` nội bộ, script truyền vào `target_vel / 2.0` để CLI giữ đúng đơn vị m/s.

Verify:

```bash
./.venv/bin/python -m py_compile \
  scripts/debug_vr_h3_rollout.py \
  scripts/interactive_demo_vr_h3.py \
  scripts/interactive_demo_g1.py
```

Đã pass.

Lệnh test tiếp theo:

```bash
cd /home/tung/GR00T-WholeBodyControl/motionbricks

./.venv/bin/python scripts/debug_vr_h3_rollout.py \
  --steps 180 \
  --warmup_steps 30 \
  --key w \
  --target_vel 0.3

./.venv/bin/python scripts/analyze_vr_h3_gait.py
```

Nếu vẫn quá nhanh hoặc corr vẫn dương cao, thử tiếp:

```bash
./.venv/bin/python scripts/debug_vr_h3_rollout.py \
  --steps 180 \
  --warmup_steps 30 \
  --key w \
  --target_vel 0.2
```

## target_vel không tác động đủ, cần test slow_walk mode

User chạy:

```bash
./.venv/bin/python scripts/debug_vr_h3_rollout.py \
  --steps 180 \
  --warmup_steps 30 \
  --key w \
  --target_vel 0.3
```

Kết quả y hệt baseline không target_vel:

```text
root_delta_after_warmup_xyz: [6.0242, -0.4294, -0.0176]
left_right_lift_corr: 0.7880
```

Nhận định:

- `target_vel` chưa giúp điều khiển tốc độ/gait trong rollout này.
- Tuning speed không còn là hướng chính nếu output vẫn y hệt.
- VR H3 hiện có primitive `slow_walk` với phase tốt hơn `generated walk`.
- `clip_holder_VR_H3.DEFAULT_KEYS["slow_walk"] = "v"`, nên với WASD/debug có thể ép slow_walk bằng cách giữ `w` và `v` cùng lúc.

Lệnh test tiếp theo:

```bash
cd /home/tung/GR00T-WholeBodyControl/motionbricks

./.venv/bin/python scripts/debug_vr_h3_rollout.py \
  --steps 180 \
  --warmup_steps 30 \
  --key w,v

./.venv/bin/python scripts/analyze_vr_h3_gait.py
```

Kỳ vọng:

- `mode ids seen` nên có `[0, 1]` thay vì `[0, 2]`.
- Nếu `left_right_lift_corr` giảm xuống gần 0 hoặc âm, thì default WASD nên chuyển sang slow_walk hoặc giảm mapping `w -> walk`.
- Nếu vẫn dương cao, vấn đề nằm sâu hơn ở model generation/conditioning, không chỉ mode/speed.

## Đánh giá slow_walk mode bằng w,v

User chạy:

```bash
cd /home/tung/GR00T-WholeBodyControl/motionbricks

./.venv/bin/python scripts/debug_vr_h3_rollout.py \
  --steps 180 \
  --warmup_steps 30 \
  --key w,v

./.venv/bin/python scripts/analyze_vr_h3_gait.py
```

Kết quả:

```text
mode ids seen: [0, 1]
max_joint_limit_violation_rad: 0.0
joint_limit_violation_frames: 0
root_delta_after_warmup_xyz: [3.5166, 0.0917, 0.0167]
```

Generated slow_walk sau warmup:

```text
frames: 150
root_delta_xy: [3.5166, 0.0917]
left_lift_p95:  0.0469
right_lift_p95: 0.0849
forward_step_asymmetry: 0.0425
left_right_lift_corr: 0.4319
```

So sánh với generated walk trước đó:

```text
walk mode:
  root_delta_xy: [6.0242, -0.4294]
  forward_step_asymmetry: 0.2745
  left_right_lift_corr: 0.7880

slow_walk mode:
  root_delta_xy: [3.5166, 0.0917]
  forward_step_asymmetry: 0.0425
  left_right_lift_corr: 0.4319
```

Nhận định:

- `slow_walk` mode tốt hơn `walk` mode rõ rệt:
  - tốc độ root giảm đáng kể.
  - lateral drift giảm.
  - forward step asymmetry giảm rất mạnh.
  - lift correlation giảm từ `0.7880` xuống `0.4319`.
- Tuy nhiên `left_right_lift_corr = 0.4319` vẫn dương, tức hai chân vẫn còn nâng cùng pha một phần; chưa đạt gait tự nhiên.
- Vì primitive slow_walk có `left_right_lift_corr = -0.3377`, nhưng generated slow_walk vẫn dương, vấn đề còn nằm ở model generation/conditioning chứ không chỉ chọn primitive/mode.

Hướng tiếp theo:

- Tạm thời nên map phím `w` của VR H3 sang `slow_walk` để demo đỡ xấu hơn.
- Sau đó xử lý sâu hơn bằng cách:
  - lọc dataset walk loop đồng nhất hơn nữa,
  - tăng data clean cùng style,
  - hoặc kiểm tra root/pose conditioning vì generated không giữ gait phase của primitive.

## Fix phím WASD bị MuJoCo viewer ăn shortcut

Vấn đề:

- Khi chạy:

```bash
cd /home/tung/GR00T-WholeBodyControl/motionbricks
./.venv/bin/python scripts/interactive_demo_vr_h3.py
```

- Bấm `w` vừa điều khiển VR H3, vừa kích hoạt shortcut `w` của MuJoCo viewer.

Nguyên nhân:

- Controller đọc keyboard bằng `pynput`, nhưng MuJoCo viewer vẫn nhận keyboard event của cùng phím.
- `interactive_demo_g1.py` đã có helper `_disable_mujoco_keyboard_shortcuts`, nhưng phần X11 grab cũ dùng keycode không chắc chắn và chỉ tìm window MuJoCo một lần.

Fix:

- Sửa `scripts/interactive_demo_g1.py`:
  - dùng `XK.string_to_keysym` để grab đúng lowercase/uppercase keysyms.
  - retry tìm window MuJoCo trong khoảng 2 giây.
  - grab các phím controller để MuJoCo không xử lý shortcut trùng với WASD/controller.

Verify:

```bash
./.venv/bin/python -m py_compile scripts/interactive_demo_g1.py scripts/interactive_demo_vr_h3.py
```

Đã pass.

## Tạo pipeline train walk normal only

Mục tiêu:

- Test giả thuyết: train quá nhiều kiểu walk trong cùng một action làm generated gait bị lẫn phase.
- Tạo một experiment hẹp chỉ dùng một style walk: `walk_forward_normal`.
- Vẫn giữ idle để demo có trạng thái đứng yên.

File allowlist mới:

```text
motionbricks/scripts/vr_h3_walk_normal_only_actions.txt
```

Nội dung:

```text
idle_loop
neutral_idle_loop
walk_forward_normal
```

File bash mới:

```text
motionbricks/scripts/train_vr_h3_walk_normal_only.sh
```

Bash này chỉ wrap lại pipeline train hiện có, nhưng override output riêng:

```text
ALLOW_ACTIONS_FILE=scripts/vr_h3_walk_normal_only_actions.txt
OUTPUT_DIR=datasets/vr_h3_motion_features_walk_normal_only
CLIP_CKPT=out/VR_H3-walk-normal-only-clip.ckpt
PLOTS_DIR=out/vr_h3_walk_normal_only_training_plots
```

Kiểm tra số CSV được chọn:

```text
allow_actions: ['idle_loop', 'neutral_idle_loop', 'walk_forward_normal']
csv_found: 142220
csv_selected: 624
```

Prepare smoke test:

```bash
cd /home/tung/GR00T-WholeBodyControl/motionbricks

./.venv/bin/python scripts/prepare_vr_h3_data.py \
  --input_dir datas/vr_h3_data/vr_h3_1 \
  --output_dir /tmp/vr_h3_walk_normal_only_prepare_test \
  --allow_actions_file scripts/vr_h3_walk_normal_only_actions.txt \
  --limit 5 \
  --progress_every 0
```

Kết quả:

```text
Converted clips: 5
Failed files: 0
CSV files selected: 5
```

Lệnh train full:

```bash
cd /home/tung/GR00T-WholeBodyControl/motionbricks
bash scripts/train_vr_h3_walk_normal_only.sh
```

Sau train, chạy debug với cache riêng:

```bash
./.venv/bin/python scripts/debug_vr_h3_rollout.py \
  --clips_ckpt out/VR_H3-walk-normal-only-clip.ckpt \
  --steps 180 \
  --warmup_steps 30 \
  --key w

./.venv/bin/python scripts/analyze_vr_h3_gait.py \
  --cache out/VR_H3-walk-normal-only-clip.ckpt
```

## Đánh giá rollout walk-normal-only

User train xong pipeline:

```text
ALLOW_ACTIONS_FILE=scripts/vr_h3_walk_normal_only_actions.txt
Converted clips: 624
Failed files: 0
Dataset: datasets/vr_h3_motion_features_walk_normal_only/motions.pt
Clip ckpt: out/VR_H3-walk-normal-only-clip.ckpt
```

Debug:

```bash
cd /home/tung/GR00T-WholeBodyControl/motionbricks

./.venv/bin/python scripts/debug_vr_h3_rollout.py \
  --clips_ckpt out/VR_H3-walk-normal-only-clip.ckpt \
  --steps 180 \
  --warmup_steps 30 \
  --key w

./.venv/bin/python scripts/analyze_vr_h3_gait.py \
  --cache out/VR_H3-walk-normal-only-clip.ckpt
```

Kết quả:

```text
mode ids seen: [0, 2]
max_joint_limit_violation_rad: 5.245e-08
joint_limit_violation_frames: 435
root_delta_after_warmup_xyz: [3.5607, 1.3025, 0.0171]
```

Lưu ý:

- `max_joint_limit_violation_rad = 5e-08` là nhiễu số học rất nhỏ, không phải violation thực tế.
- `joint_limit_violation_frames = 435` xuất hiện vì nhiều joint chạm sát limit với sai số floating point.

Generated gait:

```text
left_lift_p95:  0.5887
right_lift_p95: 0.5629
left_speed_mean/max:  [3.3385, 23.6856]
right_speed_mean/max: [3.8317, 19.9810]
forward_step_asymmetry: 0.3755
left_right_lift_corr: 0.0666
```

Primitive walk-normal-only:

```text
left_lift_p95:  0.2042
right_lift_p95: 0.1128
left_right_lift_corr: -0.3403
```

Nhận định:

- Phase chân cải thiện lớn so với các lần trước:
  - trước đó `walk` corr khoảng `0.7880`.
  - `slow_walk` corr khoảng `0.4319`.
  - walk-normal-only corr còn `0.0666`, gần ngưỡng chấp nhận được.
- Nhưng generated motion bị phóng đại:
  - foot lift p95 khoảng `0.56-0.59m`, cao hơn primitive rất nhiều.
  - foot speed max lên `20-24 m/s`, quá lớn.
  - lateral drift `~1.3m`, không ổn.
  - forward asymmetry `0.3755`, vẫn cao.
- Kết luận: lọc về một walk style giúp phase, nhưng model đang overfit/unstable hoặc conditioning sinh pose quá mạnh. Dataset walk thật chỉ có 22 CSV, còn lại là idle, nên model không đủ walk variation.

Hướng tiếp theo:

- Không nên tiếp tục giảm dataset walk nữa.
- Cần scan joint-limit/pose cleanliness trên `walk_forward_normal`.
- Cần thêm root-relative round-trip metric để chắc pose relative-to-root không bị sai.
- Nếu data sạch, hướng tiếp là tăng walk data cùng style gần giống nhau, không phải chỉ 22 CSV.

## Test CSV -> feature -> qpos round-trip cho VR H3

Mục tiêu:

- Kiểm tra vòng chuyển đổi dữ liệu train có tự nhất quán không:

```text
CSV retarget DOF
  -> MuJoCo qpos
  -> MotionBricks motion features
  -> inverse features về MuJoCo qpos
```

Script mới:

```text
motionbricks/scripts/check_vr_h3_feature_roundtrip.py
```

Lệnh kiểm tra:

```bash
cd /home/tung/GR00T-WholeBodyControl/motionbricks

./.venv/bin/python scripts/check_vr_h3_feature_roundtrip.py \
  datas/vr_h3_data/vr_h3_1/210531/walk_forward_amateur_001__A002.csv
```

Kết quả:

```text
frames: 1179
global_motion_shape: (1179, 402)
qpos_shape: (1179, 61)
finite_src: True
finite_rec: True

root_pos_err_m:
  mean=3.76084733
  p95=10.30655289
  max=10.58492088

root_rot_err_rad:
  mean=1.56435907
  p95=1.56435907
  max=1.56435943

joint_abs_err_rad:
  mean=0.00000006
  p95=0.00000024
  max=0.00000066

foot_pos_err_m:
  mean=3.84795856
  p95=10.46239471
  max=10.82632923
```

Nhận định:

- Joint DOF reconstruct cực tốt: sai số max khoảng `6.6e-7 rad`, gần như đúng tuyệt đối.
- Absolute root position/heading sai lớn. Điều này có thể đến từ canonicalization/root-frame convention khi inverse feature về qpos, không nhất thiết nghĩa là joint mapping sai.
- Foot absolute position cũng sai lớn vì root absolute bị lệch; cần thêm metric root-relative foot/joint position để tách lỗi root transform khỏi lỗi pose.
- Reconstructed qpos có vài joint vượt limit nhẹ/vừa trên chính data source:

```text
left_hip_yaw_joint: over=0.27426132
right_shoulder_pitch_joint: over=0.21220367
right_hip_yaw_joint: over=0.14013686
```

Ý nghĩa:

- Data retarget gốc có thể chứa pose ngoài joint limit VR H3 thật.
- Pipeline feature không làm hỏng joint angle, nhưng cần cân nhắc clamp/filter data source để model không học pose ngoài limit.
- Round-trip hiện tại chưa đủ để kết luận root pipeline sai, nhưng đã xác nhận joint mapping CSV/DOF -> feature -> qpos là đúng.

## Test round-trip với VR H3 RL XML

User muốn thử XML:

```text
motionbricks/assets/skeletons/vr_h3/mjcf/origin/vr_h3_1_with_rh56e2_hand_rl.xml
```

Kiểm tra load MuJoCo:

```text
nq=61
nv=60
njnt=55
nbody=96
```

Joint limits chính giống XML đang dùng:

```text
left_hip_yaw_joint: [-0.785, 0.785]
right_hip_yaw_joint: [-0.785, 0.785]
right_shoulder_pitch_joint: [-3.14, 1.047]
```

Lệnh round-trip:

```bash
cd /home/tung/GR00T-WholeBodyControl/motionbricks

./.venv/bin/python scripts/check_vr_h3_feature_roundtrip.py \
  datas/vr_h3_data/vr_h3_1/210531/walk_forward_amateur_001__A002.csv \
  --scene_xml assets/skeletons/vr_h3/mjcf/origin/vr_h3_1_with_rh56e2_hand_rl.xml \
  --skeleton_xml assets/skeletons/vr_h3/mjcf/origin/vr_h3_1_with_rh56e2_hand_rl.xml
```

Kết quả:

```text
root_pos_err_m: mean=3.76084733, p95=10.30655289, max=10.58492088
root_rot_err_rad: mean=1.56435907, p95=1.56435907, max=1.56435943
joint_abs_err_rad: mean=0.00025205, p95=0.00000030, max=0.09117579
foot_pos_err_m: mean=3.84795809, p95=10.46239471, max=10.82632923
```

So với XML cũ:

```text
old joint_abs_err max: ~0.00000066 rad
rl  joint_abs_err max: ~0.09117579 rad
```

Worst joints với RL XML:

```text
right_shoulder_pitch_joint: max=0.09117579
left_shoulder_pitch_joint:  max=0.04418260
```

Nhận định:

- RL XML không cải thiện root round-trip; root pos/rot error gần như y hệt.
- RL XML làm joint round-trip tệ hơn ở shoulder pitch.
- Joint limits vẫn tương tự nên không giải quyết vấn đề cleanliness.
- Không nên chuyển training/converter sang `_rl.xml` nếu mục tiêu là giữ feature round-trip chính xác hơn. XML hiện tại cho joint reconstruction tốt hơn.

## Crawl danh sách action trong data VR H3

Tạo script:

```text
motionbricks/scripts/inventory_vr_h3_csv_actions.py
```

Mục đích:

- Quét toàn bộ CSV trong `motionbricks/datas/vr_h3_data/vr_h3_1`.
- Gom action name bằng cách bỏ hậu tố actor/take như `_001__A002`, `_001__A002_M`, `__A002`.
- Xuất thống kê để lọc data train locomotion sạch hơn.

Lệnh đã chạy:

```bash
cd /home/tung/GR00T-WholeBodyControl/motionbricks
./.venv/bin/python scripts/inventory_vr_h3_csv_actions.py
```

Kết quả:

```text
CSV files: 142220
Unique normalized actions: 4097
Output dir: out/vr_h3_data_inventory
```

Các report đã tạo:

```text
out/vr_h3_data_inventory/actions.tsv
out/vr_h3_data_inventory/families.tsv
out/vr_h3_data_inventory/walk_actions.tsv
out/vr_h3_data_inventory/candidate_forward_walk.tsv
```

Số dòng report:

```text
actions.tsv:                 4098 lines
families.tsv:                 332 lines
walk_actions.tsv:             877 lines
candidate_forward_walk.tsv:   121 lines
```

Top family theo số CSV:

```text
jog      12823
walk     10982
dance     9564
injured   8466
jump      7541
idle      4530
medium    4484
turn      4445
dancing   3583
itching   3508
sitting   3419
come      3160
```

Nhận định:

- Dataset rất lẫn hành động, không chỉ locomotion.
- Riêng các action chứa `walk` đã có 876 nhóm sau khi normalize.
- `walk` bao gồm nhiều kiểu không nên trộn thẳng vào baseline WASD: injured, crutch/crutches, carry/lift crate, sideway, start/stop/turn, walk dog, talking, watering plants, hands on back.
- `candidate_forward_walk.tsv` chỉ là danh sách ứng viên ban đầu, vẫn cần soi bằng viewer trước khi dùng train.

Hướng lọc tiếp theo:

- Ưu tiên các nhóm như `walk_ff_loop_180_R`, `walk_ff_loop_180_R_normal_pace`, `neutral_walk_180_R`, `Neutral_walk_forward`, `Relaxed_walk_forward`, `Loop_Forward_Walk`.
- Loại các nhóm có từ khóa: `inj`, `injured`, `crutch`, `crutches`, `lift`, `crate`, `carry`, `box`, `sideway`, `turn`, `start`, `stop`, `dog`, `talk`, `watering`, `hands_on_back`, `baby`, `heavy`, `one_hand`, `two_hands`.

## 2026-05-11 - Implement VR_H3 dummy toe pipeline va bash locomotion-only

Da trien khai plan dummy toe cho VR_H3:

- Update `motionbricks/motionbricks/motionlib/core/skeletons/vr_h3.py`:
  - them `left_toe_base`, parent `left_ankle_roll_skel`
  - them `right_toe_base`, parent `right_ankle_roll_skel`
  - foot contacts doi sang ankle + toe moi ben
- Update `motionbricks/scripts/prepare_vr_h3_data.py`:
  - `EXPECTED_GLOBAL_DIM = 402`
  - `EXPECTED_FULL_DIM = 406`
- Tao `motionbricks/scripts/regenerate_vr_h3_dummy_toe_skeleton.py`.
- Da regenerate skeleton files cho:
  - `out/motionbricks_vr_h3_vqvae/version_1/skeleton`
  - `out/motionbricks_vr_h3_pose/version_1/skeleton`
  - `out/motionbricks_vr_h3_root/version_1/skeleton`
- Da reset placeholder stats 406 dims trong cac VR_H3 checkpoint folders. Khi prepare
  dataset that va train, stats that tu dataset se duoc copy de len cac placeholder nay.
- Tao bash moi `motionbricks/scripts/train_vr_h3_locomotion_toe.sh`.

Bash moi chi prepare/train voi locomotion CSV:

```text
include = idle, walk, jog, run, turn, start, stop, hurry
exclude = jump, dance, combat, cartwheel, throw, body_check, box
output dataset = datasets/vr_h3_motion_features_loco_toe/motions.pt
clip ckpt = out/VR_H3-clip.ckpt
```

Script nay se chay:

```text
prepare_vr_h3_data.py
train_vqvae.py
train_pose.py
train_root.py
visualize_training_metrics.py
build_vr_h3_clip_cache.py
```

Verification da chay:

```text
bash -n scripts/train_vr_h3_locomotion_toe.sh: pass
motion_rep: joints=33, global=402, local=401, body=397, full_dual=406
prepare smoke 20 CSV: Converted clips=20, Failed files=0
smoke dataset: global_dim=402, full_dim=406, mean/std shape=(406,)
```

Chua chay train smoke 200 steps de tranh tao checkpoint it-step co the bi demo load nham.
Sau khi san sang train that, dung:

```bash
cd /home/tung/GR00T-WholeBodyControl/motionbricks
bash scripts/train_vr_h3_locomotion_toe.sh
```

## 2026-05-11 - Tach primitive clip cache cho VR_H3 toe pipeline

Nhan dinh:

- `interactive_demo_vr_h3.py` khong dung `clip_holder_G1` neu `--clips vr_h3`.
  No dung `clip_holder_VR_H3` voi 3 primitive clips: `idle`, `slow_walk`, `walk`.
- Tuy nhien cache path cu `out/VR_H3-clip.ckpt` de bi nham voi cache 31-joint/378-dim.

Da sua:

- `interactive_demo_vr_h3.py` default `--clips_ckpt` sang:

```text
out/VR_H3-locomotion-toe-clip.ckpt
```

- `train_vr_h3_locomotion_toe.sh` build clip cache cung vao path moi nay.
- `clip_holder._preprocess_clips_from_ckpt` co guard check `motion_feature` dim cua
  cache voi `motion_rep.motion_rep_dim`. Neu load nham cache 378-dim trong khi
  motion_rep moi can 402-dim, script se bao loi ro rang va yeu cau rebuild cache.

Verification:

```text
bash -n scripts/train_vr_h3_locomotion_toe.sh: pass
py_compile interactive_demo_vr_h3.py va clips.py: pass
```

## 2026-05-11 - Fix train_vr_h3_locomotion_toe.sh sau prepare

Sau khi prepare locomotion toe dataset xong:

```text
Converted clips: 60450
Failed files: 0
Dataset: datasets/vr_h3_motion_features_loco_toe/motions.pt
Stats: datasets/vr_h3_motion_features_loco_toe/stats/motion
```

Script bao loi:

```text
unexpected EOF while looking for matching `"`
```

Ban hien tai tren disk da pass `bash -n`. Them option `SKIP_PREPARE=1` de co the
chay tiep training tu dataset da prepare ma khong convert lai 60450 CSV.

Lenh chay tiep:

```bash
cd /home/tung/GR00T-WholeBodyControl/motionbricks
SKIP_PREPARE=1 bash scripts/train_vr_h3_locomotion_toe.sh
```

## 2026-05-11 - Danh gia train locomotion toe 33 joints

Dataset/cache:

```text
Dataset: datasets/vr_h3_motion_features_loco_toe/motions.pt
motions = 60450
global_dim = 402
full_dim = 406
first motion shape = (1202, 402)
stats mean/std shape = (406,)
Clip cache: out/VR_H3-locomotion-toe-clip.ckpt
motion_feature shape = (3, 64, 402)
mujoco_qpos shape = (3, 64, 61)
```

Checkpoint moi nhat:

```text
VQVAE: out/motionbricks_vr_h3_vqvae/version_1/checkpoints/last-v6.ckpt
Pose:  out/motionbricks_vr_h3_pose/version_1/checkpoints/last-v5.ckpt
Root:  out/motionbricks_vr_h3_root/version_1/checkpoints/last-v5.ckpt
```

Metrics:

```text
VQVAE train_loss_epoch:        0.317953 -> 0.121185
VQVAE recons_pose_epoch:       0.224255 -> 0.0728505
VQVAE joint_vel_epoch:         0.0446601 -> 0.0210494
VQVAE skate_contact_epoch:     0.131505 -> 0.0883775
VQVAE perplexity_epoch:        7.73814 -> 7.63108

Pose train_loss_epoch:         1.83797 -> 1.1711

Root train_loss_epoch:         3.19029 -> 2.08651
Root global_recons_epoch:      0.221852 -> 0.00715289
Root local_recons_epoch:       0.252677 -> 0.064996
Root num_token_loss_epoch:     2.49391 -> 2.00721
Root top1/top3/top5 epoch:     0.227753 / 0.553926 / 0.731953
```

Nhan dinh:

- Pipeline dummy toe 33 joints da train dung shape va build duoc cache 402-dim.
- VQVAE/pose/root deu giam loss ro rang.
- Root token accuracy tot hon full-data train truoc do va nhinh hon locomotion 31-joint
  cu o top-1/top-3/top-5.
- Buoc tiep theo nen chay `scripts/interactive_demo_vr_h3.py` voi cache moi de danh gia
  thuc te WASD.

## 2026-05-11 - Chinh VR_H3 demo target speed khi bam WASD

Quan sat demo: khi di chuyen robot loi nhieu.

Kiem tra primitive cache:

```text
out/VR_H3-locomotion-toe-clip.ckpt
idle root delta xy ~= 0.013 m / 64 frames
slow_walk root delta xy ~= 0.908 m / 64 frames
walk root delta xy ~= 0.993 m / 64 frames
```

Voi 64 frames @ 30 FPS, walk clip thuc te chi khoang 0.47 m/s. Trong
`clip_holder_VR_H3`, `walk.avg_root_vel` dang la `2.0`, qua nhanh so voi data, nen
spring target khi bam `w` co the day root qua xa va lam model sinh motion xau.

Da sua trong `motion_backbone/demo/clips.py`:

```text
slow_walk.avg_root_vel: 0.6 -> 0.4
walk.avg_root_vel:      2.0 -> 0.5
```

Verification:

```text
py_compile clips.py: pass
```

Thu lai:

```bash
cd /home/tung/GR00T-WholeBodyControl/motionbricks
./.venv/bin/python scripts/interactive_demo_vr_h3.py
```

## 2026-05-11 - Them debug rollout cho VR_H3 movement pose

Tinh trang sau khi giam target speed: movement do hon, nhung pose duoc sinh ra van
khong hop ly.

Da them script:

```text
motionbricks/scripts/debug_vr_h3_rollout.py
```

Muc dich:

- Chay demo headless, khong can viewer.
- Warmup idle roi giu mot phim, mac dinh `w`.
- Ghi qpos rollout ra `.npz`.
- Bao root delta, mode ids, NaN/Inf, joint-limit violation.

Lenh chay:

```bash
cd /home/tung/GR00T-WholeBodyControl/motionbricks
./.venv/bin/python scripts/debug_vr_h3_rollout.py \
  --steps 180 \
  --warmup_steps 30 \
  --key w
```

Neu output co joint-limit violation lon, uu tien fix converter/filter/qpos generation.
Neu joint limits on nhung pose van xau, uu tien debug model pose generation va data
distribution.

Verification:

```text
py_compile scripts/debug_vr_h3_rollout.py: pass
```

## 2026-05-11 - Fix debug_vr_h3_rollout headless issues

Fix loi:

```text
AttributeError: 'numpy.ndarray' object has no attribute 'detach'
```

Nguyen nhan: `full_agent.get_next_frame()` tra ve numpy array, khong phai torch tensor.
Da sua script de chap nhan ca numpy va torch tensor.

Dong thoi them:

```text
PYNPUT_BACKEND=dummy
MPLCONFIGDIR=/tmp/matplotlib
```

de script headless khong can X keyboard backend va khong can ghi vao
`~/.config/matplotlib`.

## 2026-05-11 - Danh gia debug rollout W cua VR_H3

Lenh da chay:

```bash
./.venv/bin/python scripts/debug_vr_h3_rollout.py \
  --steps 180 \
  --warmup_steps 30 \
  --key w
```

Ket qua chinh:

```text
qpos shape = (180, 61)
finite = True
mode ids seen = [0, 2]
root_delta_after_warmup_xyz = [3.7968, 0.7175, -0.0218]
max_joint_limit_violation_rad = 2.8227
joint_limit_violation_frames = 94
```

Worst violations:

```text
waist_roll_joint:        -3.0847 rad, limit [-0.262, 0.262], violation 2.8227
right_hip_roll_joint:     0.7464 rad, limit [-1.05, 0.17], violation 0.5764
left_knee_pitch_joint:   -0.7183 rad, limit [-0.17, 2.09], violation 0.5483
right_shoulder_pitch:     1.4996 rad, limit [-3.14, 1.047], violation 0.4526
left_hip_yaw_joint:       1.2352 rad, limit [-0.785, 0.785], violation 0.4502
```

Nhan dinh:

- Cache/model khong bi NaN va mode mapping dung: idle=0, walk=2.
- Root co di chuyen khi bam `w`, nen loi khong phai controller khong ra lenh.
- Loi chinh la generated qpos vuot joint limits, rat nang o waist roll.
- Violation bat dau sau khi da vao walk mot luc, khoang frame 78; khong phai idle
  ban dau bi sai.
- `FILTER_QPOS` hien chi blend context voi generated qpos, khong clamp joint limits.

Huong tiep theo:

- Them joint-limit clamp cho qpos generated sau
  `convert_motion_features_to_mujoco_qpos`.
- Sau khi clamp, chay lai `debug_vr_h3_rollout.py` de xem violation ve 0 va pose co
  bot vo ly khong.
- Neu clamp chi che dau trieu chung va pose van xau, tiep tuc debug model pose
  generation / data distribution.

## 2026-05-11 - Implement clamp joint limits trong full_agent

Da them clamp qpos theo MuJoCo joint limits trong:

```text
motionbricks/motionbricks/motion_backbone/demo/full_agent.py
```

Chi tiet:

- Load `MjModel` tu `skeleton_xml`.
- Tao `_qpos_lower/_qpos_upper` theo `model.jnt_range` cho cac hinge joints co limit.
- Root translation/quaternion giu `[-inf, inf]`, khong clamp.
- Clamp `self.frames["mujoco_qpos"]` o 3 diem:
  - sau khi convert motion features -> qpos trong `_initialize_frames`
  - sau khi generated qpos va uncanonicalize
  - sau khi blend context qpos voi generated qpos

Verification:

```text
py_compile motion_backbone/demo/full_agent.py: pass
```

Can chay lai:

```bash
cd /home/tung/GR00T-WholeBodyControl/motionbricks
./.venv/bin/python scripts/debug_vr_h3_rollout.py \
  --steps 180 \
  --warmup_steps 30 \
  --key w
```

Muc tieu:

```text
max_joint_limit_violation_rad = 0
joint_limit_violation_frames = 0
```

## 2026-05-11 - Danh gia sau clamp joint limits

Chay lai debug rollout sau khi clamp:

```text
qpos shape = (180, 61)
finite = True
mode ids seen = [0, 2]
max_joint_limit_violation_rad = 4.77e-09
joint_limit_violation_frames = 8
worst: right_ankle_pitch_joint chi cham limit voi sai so floating point
```

Nhan dinh:

- Joint-limit blocker da duoc giai quyet ve mat thuc te. Violation con lai la sai so
  so hoc ~`1e-9`, xem nhu 0.
- Root van di chuyen hoi cheo:

```text
active root delta xy = [3.9368, 1.2120]
active mean speed = 0.857 m/s
lateral/forward ~= 0.308
root z range = [0.7275, 0.8432]
```

- Neu demo van nhin xau sau clamp, loi khong con la joint limit nua. Ung vien tiep theo:
  - root target/facing direction dang tao drift ngang
  - model sinh pose hop le theo limit nhung khong tu nhien
  - clamp dang cat pose qua nhieu, lam chuyen dong giat/khong giong data

Buoc tiep theo nen test visual demo lai. Neu van xau, can log/theo doi raw unclamped
qpos so voi clamped qpos de xem clamp can thiep qua nhieu hay model da xau ngay tu dau.

## 2026-05-11 - Phan tich gait va chinh target cho VR_H3

User quan sat: robot van di xau, khong co dang tung chan mot nhac len di ve phia
truoc.

Da them/chạy gait analyzer:

```text
motionbricks/scripts/analyze_vr_h3_gait.py
```

Ket qua so sanh generated rollout vs primitive walk:

```text
generated left_right_lift_corr = +0.397
primitive walk left_right_lift_corr = -0.392
```

Nhan dinh:

- Primitive walk co foot lift anti-phase hop ly hon.
- Generated rollout co tuong quan duong giua left/right lift, tuc hai chan co xu
  huong nhac/cung pha hon, nen nhin khong giong gait tu nhien.
- Generated root speed sau warmup khoang `0.857 m/s`, cao hon primitive walk thuc te
  khoang `0.47 m/s`.
- Root drift ngang van con dang ke.

Da sua them:

```text
clip_holder_VR_H3.slow_walk.avg_root_vel: 0.4 -> 0.2
clip_holder_VR_H3.walk.avg_root_vel:      0.5 -> 0.3
interactive_demo_vr_h3.py lookat_movement_direction: 0 -> 1
debug_vr_h3_rollout.py lookat_movement_direction:    0 -> 1
```

Ly do:

- Dua target speed gan hon voi data locomotion de model khong bi ep sinh motion qua
  nhanh.
- Ep VR_H3 facing direction theo movement direction de uu tien gait tien thang tu nhien,
  tranh truong hop di theo phim nhung than/huong nhin theo camera tao dang side-walk.

Verification:

```text
py_compile clips.py, interactive_demo_vr_h3.py, debug_vr_h3_rollout.py,
analyze_vr_h3_gait.py: pass
```

Can chay lai:

```bash
./.venv/bin/python scripts/debug_vr_h3_rollout.py --steps 180 --warmup_steps 30 --key w
./.venv/bin/python scripts/analyze_vr_h3_gait.py
```

## 2026-05-11 - Danh gia gait sau giam speed va lookat_movement_direction

Ket qua debug moi:

```text
max_joint_limit_violation_rad = 0.0
joint_limit_violation_frames = 0
root_delta_after_warmup_xyz = [1.7781, 1.0145, -0.0101]
```

So sanh gait:

```text
generated left_lift_p95 = 0.0841
generated right_lift_p95 = 0.0852
generated left_right_lift_corr = +0.2076

primitive walk left_lift_p95 = 0.2131
primitive walk right_lift_p95 = 0.2287
primitive walk left_right_lift_corr = -0.3924
```

Nhan dinh:

- Joint limits da sach hoan toan.
- Root speed da giam, nhung root van drift ngang nhieu:
  `lateral/forward ~= 1.0145 / 1.7781 = 0.57`.
- Generated foot lift thap hon primitive walk khoang 2.5 lan.
- Left/right foot lift van tuong quan duong, trong khi primitive walk tuong quan am.
  Nghia la generated motion van chua co pha buoc trai/phai luan phien ro rang.

Ket luan hien tai:

- Loi khong con la qpos vuot limit.
- Primitive walk cache co gait dung hon, nhung model inference khong reproduce duoc
  gait primitive khi chuyen sang movement.
- Ung vien loi tiep theo:
  - root/facing target van tao drift ngang qua lon;
  - pose model sinh motion qua "conservative" / foot lift thap;
  - conditioning tu primitive clip sang generated pose chua du manh;
  - clamp co the dang cat raw output nhieu, can do raw-vs-clamped intervention.

## 2026-05-11 - Kiem tra docs va tinh dung cua pipeline train VR_H3

Da doi chieu `motionbricks/docs/motion_representation.md`,
`motionbricks/docs/adding_your_own_dataset.md`, skeleton VR_H3, script prepare data
va cac script train.

Ket luan:

- Pipeline train hien tai dung theo hop dong dataset cua MotionBricks: moi sample tra ve
  `{"keyid": ..., "motion": Tensor[T, feature_dim]}` va `motion` la normalized
  global representation.
- Voi VR_H3, dimension dung la:
  - skeleton: `h3skel`, 31 joints
  - global representation: 378 dims
  - local representation: 377 dims
  - body features: 373 dims
  - full dual representation/stats: 382 dims
- Cac con so trong `motion_representation.md` nhu 414/413/418 la cho G1Skeleton34,
  khong ap dung truc tiep cho VR_H3.
- `prepare_vr_h3_data.py` hien di theo duong CSV -> MuJoCo qpos -> motion transforms
  -> full dual features -> streaming stats -> normalized global features. Huong nay
  phu hop hon duong pkl `pose_aa` da tung gay van khop.
- Train order dung: VQVAE -> pose -> root. Pose script load VQVAE checkpoint moi nhat,
  va cac script train copy `mean.npy/std.npy` tu dataset vao thu muc checkpoint.

Rui ro con lai:

- `H3Skeleton` dang dung duplicate ankle roll joint cho 4 foot contact channels
  (`left_ankle_roll_skel` lap 2 lan, `right_ankle_roll_skel` lap 2 lan), vi VR_H3
  khong co dummy toe joints nhu G1. Viec nay dung format nhung foot contact kem giau
  thong tin hon, co the anh huong chat luong gait.
- Full-data train gom ca nhieu motion khong phai locomotion. Neu muc tieu truoc mat la
  WASD walking, locomotion-filtered dataset van la data mix hop ly hon de debug chat
  luong demo.
- Full-data prepare da duoc sua sang streaming de tranh OOM luc tinh stats, nhung
  `motions.pt` van duoc load vao RAM khi train. Neu full dataset qua lon va bi kill o
  buoc train, buoc tiep theo la lam lazy/sharded dataset.

## 2026-05-11 - Tao plan.md cho migration VR_H3 dummy toe

Da tao `plan.md` o repo root de ghi lai plan chuan hoa pipeline VR_H3 theo docs cua
MotionBricks. Plan nay chua implement code, chi luu cac buoc se lam sau khi pipeline
cu train xong:

- Them dummy toe joints cho `H3Skeleton`.
- Regenerate `joints.p` va `parents.p`.
- Doi dims VR_H3 sang `global=402`, `local=401`, `full=406`.
- Prepare data smoke, test converter, train smoke.
- Sau smoke pass moi prepare/train dataset that va rebuild demo cache.

## 2026-05-11 - Full-data train pipeline cu 31 joints da chay xong

Dataset:

```text
datasets/vr_h3_motion_features_full_all/motions.pt
num_motions = 142220
global_dim = 378
full_dim = 382
source_fps = 120
target_fps = 30.0
```

Train:

```text
VQVAE: 20000 steps
Pose: 50000 steps
Root: 50000 steps
```

Checkpoint moi nhat:

```text
VQVAE: out/motionbricks_vr_h3_vqvae/version_1/checkpoints/last-v5.ckpt
Pose:  out/motionbricks_vr_h3_pose/version_1/checkpoints/last-v4.ckpt
Root:  out/motionbricks_vr_h3_root/version_1/checkpoints/last-v4.ckpt
```

Metric summary:

```text
VQVAE train_loss_epoch: 0.273427 -> 0.15238
VQVAE recons_pose_epoch: 0.183996 -> 0.0897175
Pose train_loss_epoch: 1.6506 -> 1.10264
Root train_loss_epoch: 2.94908 -> 2.34873
Root global_root_recons_loss_epoch: 0.152614 -> 0.0138777
Root local_root_recons_loss_epoch: 0.178495 -> 0.0765139
Root top1/top3/top5 epoch: 0.152637 / 0.401019 / 0.588722
```

Nhan dinh:

- VQVAE va pose co hoc tot hon so voi dau train.
- Root reconstruction hoc tot, nhung root token accuracy con thap hon tap
  locomotion-filtered truoc do. Full data co the kho hon cho demo WASD vi gom nhieu
  action phi locomotion.
- Buoc tiep theo cua pipeline cu la rebuild `out/VR_H3-clip.ckpt` tu full dataset,
  sau do chay `interactive_demo_vr_h3.py` de danh gia demo.

## 2026-05-11 - Ghi chu ve dummy toe joints cho VR_H3

G1 co `G1Skeleton34` gom 32 joints that + 2 dummy toe joints. Hai toe joints nay
khong can ton tai trong MuJoCo XML, vi `mujoco_qpos_converter` co co che dead joint:
joint nao co trong motion skeleton nhung khong map duoc sang MuJoCo hinge se duoc
populate bang neutral offset tu `skeleton/joints.p` va rotation theo parent/dummy.

VR_H3 hien chua co dummy toe joints vi skeleton asset ban dau duoc tao theo cac joint
that trong robot/MJCF, chua them 2 neutral points ao o dau ban chan. Do do
`H3Skeleton` tam dung duplicate ankle roll cho 4 foot contact channels. Cach nay dung
format nhung foot contact kem chinh xac hon G1.

Neu muon them dummy toe cho VR_H3 can:

- Them `left_toe_base` va `right_toe_base` vao `H3Skeleton`, parent la ankle roll.
- Dat `left_foot_joint_names = [left_ankle_roll_skel, left_toe_base]` va tuong tu ben
  phai.
- Regenerate `skeleton/joints.p` va `parents.p` voi neutral toe offset hop ly theo
  hinh hoc ban chan.
- Update expected dims trong prepare data. Voi 33 joints, global/local/full dims se
  khac 378/377/382, nen phai prepare data va train lai tu dau.

## Fix OOM khi prepare full 142k CSV

User chạy:

```bash
bash scripts/train_vr_h3_full_data.sh
```

và bị:

```text
scripts/train_vr_h3_full_data.sh: line 35: ... Killed
```

Nguyên nhân:

- `prepare_vr_h3_data.py` cũ giữ toàn bộ `full_motions`, `global_motions`, rồi `torch.cat(full_motions)` trong RAM.
- Với 142220 CSV, process bị OOM và bị OS kill.

Đã sửa:

```text
scripts/prepare_vr_h3_data.py
```

Thay đổi:

- Prepare chuyển sang 2-pass streaming.
- Pass 1: convert từng CSV, chỉ cộng dồn `sum` / `sumsq` để tính `mean/std`, không giữ motion trong RAM.
- Pass 2: convert lại từng CSV, normalize bằng stats vừa tính, rồi append vào dataset list để save.
- Tốn thời gian prepare hơn, nhưng giảm RAM peak mạnh.
- Thêm progress log `--progress_every`.

Đã cập nhật:

```text
scripts/train_vr_h3_full_data.sh
```

Thêm:

```bash
PREPARE_PROGRESS_EVERY="${PREPARE_PROGRESS_EVERY:-1000}"
```

và truyền:

```bash
--progress_every "${PREPARE_PROGRESS_EVERY}"
```

Đã verify:

```bash
./.venv/bin/python -m py_compile scripts/prepare_vr_h3_data.py
bash -n scripts/train_vr_h3_full_data.sh
./.venv/bin/python scripts/prepare_vr_h3_data.py \
  --input_dir datas/vr_h3_data/vr_h3_1 \
  --output_dir datasets/vr_h3_motion_features_stream_debug \
  --limit 20 \
  --progress_every 5
```

Kết quả sample:

```text
Stats pass: 20/20 files, valid=20, failed=0
Save pass: 20/20 files, saved=20, failed=0
Converted clips: 20
Failed files: 0
```

Chạy lại full:

```bash
cd /home/tung/GR00T-WholeBodyControl/motionbricks
bash scripts/train_vr_h3_full_data.sh
```

Nếu muốn progress dày hơn:

```bash
PREPARE_PROGRESS_EVERY=200 bash scripts/train_vr_h3_full_data.sh
```

Lưu ý:

- Cách streaming giảm RAM khi prepare, nhưng dataset `.pt` full vẫn rất lớn.
- Nếu vẫn bị kill ở cuối lúc save hoặc lúc train load dataset, bước tiếp theo cần chuyển dataset sang shard/lazy loading.

## Tạo bash train toàn bộ VR H3 data

User muốn train với toàn bộ data trong:

```text
motionbricks/datas/vr_h3_data/vr_h3_1
```

Đã thêm script:

```text
motionbricks/scripts/train_vr_h3_full_data.sh
```

Script này chạy:

1. Prepare toàn bộ CSV, không filter include/exclude.
2. Train VQVAE.
3. Train pose.
4. Train root.

Default:

```text
INPUT_DIR=datas/vr_h3_data/vr_h3_1
OUTPUT_DIR=datasets/vr_h3_motion_features_full_all
VQVAE_STEPS=20000
POSE_STEPS=50000
ROOT_STEPS=50000
BATCH_SIZE=8
NUM_WORKERS=0
```

Cách chạy:

```bash
cd /home/tung/GR00T-WholeBodyControl/motionbricks
./scripts/train_vr_h3_full_data.sh
```

Override nếu muốn:

```bash
VQVAE_STEPS=30000 POSE_STEPS=80000 ROOT_STEPS=80000 BATCH_SIZE=8 \
  ./scripts/train_vr_h3_full_data.sh
```

Verify:

```bash
bash -n scripts/train_vr_h3_full_data.sh
```

Kết quả: pass.

Lưu ý:

- Full data là 142k CSV, có cả dance/jump/combat/etc.
- File dataset `.pt` có thể rất lớn và prepare có thể tốn RAM/disk.
- Demo cache vẫn cần rebuild sau khi train xong, nhưng script này chưa chạy demo/cache vì user chỉ yêu cầu train.

## Thêm visualize training metrics

User muốn visualize `metrics.csv` sau train để dễ đánh giá.

Đã thêm script:

```text
motionbricks/scripts/visualize_training_metrics.py
```

Script đọc:

```text
out/motionbricks_vr_h3_vqvae/version_1/training_logs/latest/metrics.csv
out/motionbricks_vr_h3_pose/version_1/training_logs/latest/metrics.csv
out/motionbricks_vr_h3_root/version_1/training_logs/latest/metrics.csv
```

Và tạo:

```text
out/vr_h3_training_plots/summary.md
out/vr_h3_training_plots/vqvae_vqvae_loss.png
out/vr_h3_training_plots/vqvae_vqvae_reconstruction.png
out/vr_h3_training_plots/vqvae_vqvae_codebook.png
out/vr_h3_training_plots/pose_pose_loss.png
out/vr_h3_training_plots/root_root_loss.png
out/vr_h3_training_plots/root_root_reconstruction.png
out/vr_h3_training_plots/root_root_accuracy.png
```

Đã cập nhật:

```text
motionbricks/scripts/train_vr_h3_full_data.sh
```

Sau khi train root xong, bash script tự gọi:

```bash
./.venv/bin/python scripts/visualize_training_metrics.py \
  --output_dir out/vr_h3_training_plots \
  --smooth_window 20
```

Có thể override:

```bash
PLOTS_DIR=out/my_plots SMOOTH_WINDOW=50 ./scripts/train_vr_h3_full_data.sh
```

Hoặc chạy visualize riêng:

```bash
cd /home/tung/GR00T-WholeBodyControl/motionbricks

MPLCONFIGDIR=/tmp/matplotlib ./.venv/bin/python scripts/visualize_training_metrics.py \
  --output_dir out/vr_h3_training_plots
```

Đã verify:

```bash
bash -n scripts/train_vr_h3_full_data.sh
./.venv/bin/python -m py_compile scripts/visualize_training_metrics.py
```

Kết quả: pass.

Đã verify:

```bash
python -m py_compile scripts/interactive_demo_vr_h3.py scripts/build_vr_h3_clip_cache.py motionbricks/motion_backbone/demo/clips.py motionbricks/motion_backbone/demo/controllers.py motionbricks/motion_backbone/demo/full_agent.py motionbricks/helper/mujoco_helper.py motionbricks/exp_setup/experiment.py
```

Kết quả: pass.

Đã test MuJoCo scene:

```text
nq = 61
njnt = 55
```

Đã build thử clip cache từ dataset debug 5 clip:

```text
mujoco_qpos: [3, 64, 61]
motion_feature: [3, 64, 378]
global_joint_positions: [3, 64, 31, 3]
```

Kết quả: pass.

Lưu ý: demo thật vẫn cần checkpoint pose/root đã train xong. Nếu chưa có checkpoint trong:

```text
out/motionbricks_vr_h3_pose/version_1/checkpoints/
out/motionbricks_vr_h3_root/version_1/checkpoints/
```

thì script demo chưa thể chạy inference.

## Fix lỗi clip holder VR H3 thiếu blendspace remap

User chạy:

```bash
./.venv/bin/python scripts/interactive_demo_vr_h3.py
```

và gặp lỗi:

```text
AttributeError: 'clip_holder_VR_H3' object has no attribute 'blendspace_modes_remap_from_velocity'
```

Nguyên nhân:

- `full_agent.py` luôn gọi `self._clip_holder.blendspace_modes_remap_from_velocity(...)` khi bật canonicalization.
- `clip_holder_G1` có hàm này vì G1 có thêm logic blendspace `walk_left` / `walk_right`.
- `clip_holder_VR_H3` hiện chỉ cần WASD cơ bản với `idle`, `slow_walk`, `walk`, nên chưa có hàm này.

Đã sửa:

- Thêm default implementation trong base `clip_holder` tại `motionbricks/motion_backbone/demo/clips.py`.
- Default này trả lại `mode` nguyên trạng.
- `clip_holder_G1` vẫn override và giữ logic cũ.
- `clip_holder_VR_H3` tự dùng default no-op, phù hợp với scope hiện tại là WASD trước.

Đã verify:

```bash
./.venv/bin/python -m py_compile motionbricks/motion_backbone/demo/clips.py scripts/interactive_demo_vr_h3.py
```

Kết quả: pass.

Test headless trong sandbox:

```bash
env PYNPUT_BACKEND=dummy MPLCONFIGDIR=/tmp/matplotlib ./.venv/bin/python scripts/interactive_demo_vr_h3.py --has_viewer 0 --max_steps 5
```

Kết quả:

- Không còn lỗi `blendspace_modes_remap_from_velocity`.
- Sandbox dừng ở `RuntimeError: No CUDA GPUs are available`, nên chưa verify được inference sâu hơn trong môi trường này.

## Thêm script kiểm tra trực tiếp file pkl retarget cho VR H3

Sau khi demo chạy được nhưng robot bị vặn vẹo, cần kiểm tra nguồn data trước khi debug model.

Đã thêm:

```text
motionbricks/scripts/play_vr_h3_pkl.py
```

Mục đích:

- Đọc một file `.pkl` retarget bất kỳ.
- In schema và thống kê cơ bản của `fps`, `root_trans_offset`, `root_rot`, `pose_aa`, `dof`, `smpl_joints`.
- Convert qua đúng pipeline đang dùng cho training: `pose_aa -> motion_feature -> mujoco_qpos`.
- Playback trực tiếp trên robot VR H3 trong MuJoCo.
- In các joint vượt MuJoCo joint limit để phát hiện joint order / axis mapping sai.
- Có option thử root rotation từ `root_rot` thay vì `pose_aa[:, 0]`.

Cách chạy mở viewer:

```bash
cd /home/tung/GR00T-WholeBodyControl/motionbricks

./.venv/bin/python scripts/play_vr_h3_pkl.py \
  datas/230418/walk_180_R_001__A335.pkl \
  --loop
```

Cách chạy không mở viewer, chỉ check convert và joint limit:

```bash
./.venv/bin/python scripts/play_vr_h3_pkl.py \
  datas/230418/walk_180_R_001__A335.pkl \
  --no_viewer
```

Thử dùng `root_rot` làm root rotation:

```bash
./.venv/bin/python scripts/play_vr_h3_pkl.py \
  datas/230418/walk_180_R_001__A335.pkl \
  --no_viewer \
  --root_rotation root_rot
```

Đã verify:

```bash
./.venv/bin/python -m py_compile scripts/play_vr_h3_pkl.py
```

Kết quả: pass.

Test headless với `datas/230418/walk_180_R_001__A335.pkl`:

```text
full_motion: shape=(301, 382)
global_motion: shape=(301, 378)
mujoco_qpos: shape=(301, 61)
mujoco_model: nq=61, nv=60, njnt=55
```

Phát hiện ban đầu:

- Convert không crash, qpos dim khớp MuJoCo.
- Nhưng có nhiều joint vượt limit lớn, ví dụ `head_pitch_joint`, `wrist_pitch`, `shoulder_pitch`, `hip_roll`, `ankle_roll`, `waist_roll`.
- `waist_roll_joint` bị gần `-1.570796` trong khi limit là `[-0.262, 0.262]`.
- Đây là dấu hiệu cần kiểm tra tiếp joint order / axis mapping / offset rotation trước khi kết luận model train sai.

## Sửa script kiểm tra pkl để đọc trực tiếp `dof`

Sau khi nghi ngờ cách đọc `.pkl` chưa chuẩn, đã đối chiếu lại:

- `dof` trong pkl có shape `[T, 28]`.
- 28 DOF này khớp với robot body không tính hand/head:
  - 6 left leg
  - 6 right leg
  - 2 waist
  - 7 left arm
  - 7 right arm
- MJCF lại chèn các finger joints giữa left arm và right arm, nên không thể fill `dof` tuần tự vào 28 qpos đầu tiên sau root.

Đã cập nhật:

```text
motionbricks/scripts/play_vr_h3_pkl.py
```

Thay đổi:

- Default source giờ là `--source dof`.
- Script fill `dof` theo tên joint cụ thể:
  - `left_hip_pitch_joint` ... `left_wrist_pitch_joint`
  - `right_shoulder_pitch_joint` ... `right_wrist_pitch_joint`
- Hand/head giữ 0.
- `root_rot` được đọc mặc định theo order `xyzw` rồi đổi sang MuJoCo `wxyz`.
- Vẫn giữ mode cũ `--source motion_feature` để debug pipeline training hiện tại.

Kết quả test với:

```bash
./.venv/bin/python scripts/play_vr_h3_pkl.py \
  datas/230418/walk_180_R_001__A335.pkl \
  --no_viewer
```

Output chính:

```text
Source: dof
mujoco_qpos: shape=(301, 61)
joint_limit_violations: 2 joints
  right_hip_roll_joint: max_over=0.049056
  left_ankle_roll_joint: max_over=0.001800
```

So với pipeline cũ:

```bash
./.venv/bin/python scripts/play_vr_h3_pkl.py \
  datas/230418/walk_180_R_001__A335.pkl \
  --no_viewer \
  --source motion_feature
```

Output chính:

```text
Source: motion_feature
joint_limit_violations: 21 joints
waist_roll_joint gần -1.570796 trong khi limit [-0.262, 0.262]
```

Kết luận:

- File `.pkl` thô nhiều khả năng không sai.
- Cách convert ban đầu từ `pose_aa` sang MotionBricks feature là sai/không phù hợp với data retarget này.
- Bước tiếp theo nên sửa `prepare_vr_h3_data.py` để tạo training feature từ `dof + root_trans_offset + root_rot`, tức là đi qua qpos robot thật, thay vì tin vào `pose_aa`.

## Kiểm tra dataset CSV `datas/230424`

User thêm:

```text
motionbricks/datas/230424
```

Kết quả khảo sát:

- Có 646 file `.csv`.
- Tất cả dùng cùng một header.
- Mỗi file có 35 cột:
  - `Frame`
  - `root_translateX/Y/Z`
  - `root_rotateX/Y/Z`
  - 28 cột joint dof có tên rõ ràng, ví dụ `left_hip_pitch_joint_dof`, ..., `right_wrist_pitch_joint_dof`.
- Số frame mỗi file:

```text
min = 103
max = 4207
mean ~= 695.84
```

Nhận định đơn vị:

- `root_translateZ` khoảng `80-85`, tương ứng chiều cao robot nếu tính bằng cm.
- `root_rotate*` và joint dof đang ở degree.
- Khi parse nên đổi:
  - root translation: `/ 100.0`
  - root rotation: Euler degree -> quaternion
  - joint dof: degree -> radian

Đã cập nhật:

```text
motionbricks/scripts/play_vr_h3_pkl.py
```

Script giờ đọc được cả `.pkl` và `.csv`.

Chạy CSV:

```bash
cd /home/tung/GR00T-WholeBodyControl/motionbricks

./.venv/bin/python scripts/play_vr_h3_pkl.py \
  datas/230424/walk_180_R_001__A349.csv \
  --loop
```

Headless:

```bash
./.venv/bin/python scripts/play_vr_h3_pkl.py \
  datas/230424/walk_180_R_001__A349.csv \
  --no_viewer
```

Kết quả với file walk:

```text
Source: dof
mujoco_qpos: shape=(1267, 61)
joint_limit_violations: 2 joints
  right_ankle_roll_joint: max_over=0.001800
  left_ankle_roll_joint: max_over=0.001800
```

Kết quả với file run:

```text
Source: dof
mujoco_qpos: shape=(403, 61)
joint_limit_violations: 6 joints
  right_knee_pitch_joint: max_over=0.200304
  right_elbow_pitch_joint: max_over=0.137328
  left_elbow_pitch_joint: max_over=0.095789
```

Kết luận:

- CSV dễ dùng hơn `.pkl` cho bước hiện tại vì header đã ghi rõ tên joint.
- Không cần đoán thứ tự `pose_aa`.
- Có thể build qpos trực tiếp từ CSV rất sạch: root translation + root Euler + named joint dof.
- Một số clip mạnh vẫn vượt joint limit, nhưng đây là vấn đề data/retarget hoặc cần clamp/filter, không phải lỗi parse nghiêm trọng như pipeline `pose_aa`.
- Nên ưu tiên CSV `datas/230424` để sửa pipeline prepare dataset tiếp theo.

## Chỉnh tốc độ playback CSV

User báo CSV playback nhìn như slow motion.

Nguyên nhân:

- Script đang gán CSV `fps = 30`.
- Nhưng các clip CSV có số frame rất lớn. Ví dụ `walk_180_R_003__A349.csv` có 1181 frame.
- Nếu chạy 30 FPS thì clip dài khoảng 39.4 giây.
- Với 120 FPS thì clip dài khoảng 9.8 giây, hợp lý hơn cho motion capture/retarget.

Đã sửa:

```text
motionbricks/scripts/play_vr_h3_pkl.py
```

Thay đổi:

- CSV default FPS từ `30` sang `120`.
- Vẫn có thể override bằng `--fps`.

Ví dụ chạy tốc độ mới:

```bash
./.venv/bin/python scripts/play_vr_h3_pkl.py \
  datas/230424/walk_180_R_003__A349.csv \
  --loop
```

Nếu muốn chỉnh thủ công:

```bash
./.venv/bin/python scripts/play_vr_h3_pkl.py \
  datas/230424/walk_180_R_003__A349.csv \
  --loop \
  --fps 120
```

hoặc:

```bash
./.venv/bin/python scripts/play_vr_h3_pkl.py \
  datas/230424/walk_180_R_003__A349.csv \
  --loop \
  --speed 1.5
```

Đã verify:

```bash
./.venv/bin/python -m py_compile scripts/play_vr_h3_pkl.py
./.venv/bin/python scripts/play_vr_h3_pkl.py datas/230424/walk_180_R_003__A349.csv --no_viewer
```

Kết quả: pass, CSV báo `fps = 120`.

## Viết lại prepare dataset để train từ CSV

User quyết định dùng CSV `datas/230424` làm nguồn train chính.

Đã viết lại:

```text
motionbricks/scripts/prepare_vr_h3_data.py
```

Pipeline mới:

1. Đọc CSV với header joint rõ ràng.
2. Convert đơn vị:
   - `root_translateX/Y/Z`: cm -> m
   - `root_rotateX/Y/Z`: Euler degree -> quaternion
   - `*_dof`: degree -> radian
3. Dựng MuJoCo `qpos` theo tên joint, không dựa vào thứ tự `pose_aa`.
4. Convert `qpos` sang MotionBricks transforms bằng `mujoco_qpos_converter`.
5. Tạo MotionBricks feature bằng:

```text
posed_joints + global_joint_rots -> dual_rep -> global subset [T, 378]
```

6. Tính stats full dim `[382]` và lưu:

```text
datasets/vr_h3_motion_features/motions.pt
datasets/vr_h3_motion_features/stats/motion/mean.npy
datasets/vr_h3_motion_features/stats/motion/std.npy
```

Quan trọng:

- CSV gốc đang là 120 FPS.
- Config VR H3 hiện tại là 30 FPS.
- Script mặc định downsample `120 -> 30` bằng stride 4 để train đúng tốc độ config.

Lệnh tạo dataset thật:

```bash
cd /home/tung/GR00T-WholeBodyControl/motionbricks

./.venv/bin/python scripts/prepare_vr_h3_data.py \
  --input_dir datas/230424 \
  --output_dir datasets/vr_h3_motion_features
```

Nếu muốn override FPS:

```bash
./.venv/bin/python scripts/prepare_vr_h3_data.py \
  --input_dir datas/230424 \
  --output_dir datasets/vr_h3_motion_features \
  --source_fps 120 \
  --target_fps 30
```

Đã verify compile:

```bash
./.venv/bin/python -m py_compile \
  scripts/prepare_vr_h3_data.py \
  scripts/play_vr_h3_pkl.py \
  scripts/train_vqvae.py \
  scripts/train_pose.py \
  scripts/train_root.py
```

Đã tạo thử dataset debug 5 file:

```bash
./.venv/bin/python scripts/prepare_vr_h3_data.py \
  --input_dir datas/230424 \
  --output_dir datasets/vr_h3_motion_features_csv_debug \
  --limit 5
```

Kết quả:

```text
Converted clips: 5
Failed files: 0
Source FPS: 120
Target FPS: 30.0
motions[0].shape = (957, 378)
mean.npy.shape = (382,)
std.npy.shape = (382,)
```

Đã smoke-test train VQVAE 1 step với dataset debug:

```bash
env MPLCONFIGDIR=/tmp/matplotlib ./.venv/bin/python scripts/train_vqvae.py \
  --robot vr_h3 \
  --dataset_pt datasets/vr_h3_motion_features_csv_debug/motions.pt \
  --max_steps 1 \
  --batch_size 2 \
  --num_workers 0
```

Kết quả: pass.

Lệnh train thật sau khi tạo dataset CSV:

```bash
./.venv/bin/python scripts/train_vqvae.py \
  --robot vr_h3 \
  --dataset_pt datasets/vr_h3_motion_features/motions.pt

./.venv/bin/python scripts/train_pose.py \
  --robot vr_h3 \
  --dataset_pt datasets/vr_h3_motion_features/motions.pt

./.venv/bin/python scripts/train_root.py \
  --robot vr_h3 \
  --dataset_pt datasets/vr_h3_motion_features/motions.pt
```

## Fix demo sau khi train CSV vẫn bị vặn khớp

User train xong 3 model với CSV dataset, nhưng chạy `interactive_demo_vr_h3.py` robot vẫn bị vặn khớp vô lý.

Nguyên nhân tìm thấy:

1. Demo vẫn ưu tiên checkpoint cũ.
   - `experiment.py` chọn `last.ckpt` nếu tồn tại.
   - Lần train CSV mới tạo `last-v1.ckpt` / `last-v2.ckpt`.
   - Vì `last.ckpt` cũ vẫn còn trong folder, demo load checkpoint cũ lúc 11:37 thay vì checkpoint mới lúc 14:00.

2. Stats trong `out/.../stats/motion` vẫn là zero/one.
   - Dataset mới có stats đúng tại:

```text
datasets/vr_h3_motion_features/stats/motion
```

   - Nhưng inference và clip cache đọc stats từ:

```text
out/motionbricks_vr_h3_vqvae/version_1/stats/motion
out/motionbricks_vr_h3_pose/version_1/stats/motion
out/motionbricks_vr_h3_root/version_1/stats/motion
```

   - Các stats này còn là zero mean / unit std, nên unnormalize sai và joint angle bị vặn.

3. `out/VR_H3-clip.ckpt` là clip cache cũ.
   - File cũ được build lúc 11:39 từ data/pipeline cũ.
   - Ngoài ra script build cache cũ lấy `motions[0:3]`, mà dataset CSV sort theo tên nên 3 clip đầu là `body_check`, không phải idle/walk.

Đã sửa code:

- `motionbricks/exp_setup/experiment.py`
  - `_latest_ckpt(...)` giờ chọn checkpoint mới nhất theo mtime, không ưu tiên cứng `last.ckpt`.

- `scripts/train_vqvae.py`
- `scripts/train_pose.py`
- `scripts/train_root.py`
  - Khi train bằng `--dataset_pt`, script copy `mean.npy/std.npy` từ dataset vào `version_1/stats/motion`.
  - Điều này giúp train và inference dùng cùng stats.

- `scripts/build_vr_h3_clip_cache.py`
  - Không lấy `motions[0:3]` nữa.
  - Chọn clip theo `keyids`:

```text
idle      -> hurry_idle_right_R_001__A349
slow_walk -> walk_180_R_001__A349
walk      -> walk_180_R_003__A349
```

Đã sửa artifact hiện tại:

- Copy stats mới từ dataset vào cả 3 thư mục model trong `out`.
- Rebuild lại:

```text
out/VR_H3-clip.ckpt
```

Kết quả verify checkpoint selector:

```text
pose_model_ckpt = last-v1.ckpt
root_model_ckpt = last-v1.ckpt
vqvae_ckpt = last-v2.ckpt
```

Kết quả verify clip cache mới:

```text
qpos shape = (3, 64, 61)
num_frames = [64, 64, 64]
idle violations = []
slow_walk violations ~= 0.0019 rad ở ankle roll
walk violations ~= 0.0019 rad ở ankle roll
```

Lệnh rebuild cache nếu cần chạy lại:

```bash
cd /home/tung/GR00T-WholeBodyControl/motionbricks

./.venv/bin/python scripts/build_vr_h3_clip_cache.py \
  --dataset_pt datasets/vr_h3_motion_features/motions.pt \
  --output out/VR_H3-clip.ckpt
```

Sau fix này chạy lại demo:

```bash
./.venv/bin/python scripts/interactive_demo_vr_h3.py
```

Lưu ý:

- `max_steps=200` vẫn là training rất ngắn, chất lượng motion có thể chưa tốt.
- Nhưng lỗi vặn khớp vô lý trước đó chủ yếu do artifact/stats/cache cũ, không phải do CSV data sai.

## Sau khi đứng đúng nhưng bấm W không tiến

User báo:

- Không bấm gì: robot về idle và đứng yên đúng.
- Bấm `W`: robot không tiến lên, chỉ rung nhẹ các bộ phận.

Kiểm tra:

- Clip cache mới có root motion đúng:

```text
idle root delta ~= 0
slow_walk root delta ~= 0.91m / 64 frames
walk root delta ~= 0.99m / 64 frames
```

- Do bấm `W` làm body rung, khả năng cao keyboard/controller đã vào `walk` mode.
- Vấn đề còn lại là chất lượng inference root/pose model.
- Lần train vừa rồi dùng default cũ `max_steps=200`, đây chỉ đủ smoke test. Với model:
  - VQVAE: 23.7M params
  - pose: 136M params
  - root: 34.1M params

200 steps là quá ít để học locomotion. Kết quả thường là pose rung nhẹ nhưng root không học tiến ổn định.

Đã cập nhật train scripts:

```text
scripts/train_vqvae.py
scripts/train_pose.py
scripts/train_root.py
```

Default mới:

- Nếu không dùng `--dataset_pt`: giữ smoke default `200`.
- Nếu dùng `--dataset_pt`:
  - VQVAE default `20000` steps.
  - pose default `50000` steps.
  - root default `50000` steps.

Compile check:

```bash
./.venv/bin/python -m py_compile \
  scripts/train_vqvae.py \
  scripts/train_pose.py \
  scripts/train_root.py
```

Kết quả: pass.

Lệnh train lại đề xuất:

```bash
cd /home/tung/GR00T-WholeBodyControl/motionbricks

./.venv/bin/python scripts/train_vqvae.py \
  --robot vr_h3 \
  --dataset_pt datasets/vr_h3_motion_features/motions.pt

./.venv/bin/python scripts/train_pose.py \
  --robot vr_h3 \
  --dataset_pt datasets/vr_h3_motion_features/motions.pt

./.venv/bin/python scripts/train_root.py \
  --robot vr_h3 \
  --dataset_pt datasets/vr_h3_motion_features/motions.pt
```

Hoặc chỉ định rõ:

```bash
./.venv/bin/python scripts/train_vqvae.py --robot vr_h3 --dataset_pt datasets/vr_h3_motion_features/motions.pt --max_steps 20000
./.venv/bin/python scripts/train_pose.py  --robot vr_h3 --dataset_pt datasets/vr_h3_motion_features/motions.pt --max_steps 50000
./.venv/bin/python scripts/train_root.py  --robot vr_h3 --dataset_pt datasets/vr_h3_motion_features/motions.pt --max_steps 50000
```

Sau khi train lại, rebuild clip cache:

```bash
./.venv/bin/python scripts/build_vr_h3_clip_cache.py \
  --dataset_pt datasets/vr_h3_motion_features/motions.pt \
  --output out/VR_H3-clip.ckpt
```

## Thêm logging để đánh giá train

User hỏi cách đánh giá training thay vì phải chạy demo kiểu blind box.

Đã bật CSV logger trong:

```text
scripts/train_vqvae.py
scripts/train_pose.py
scripts/train_root.py
```

Sau mỗi lần train, script sẽ in đường dẫn metrics:

```text
out/motionbricks_vr_h3_vqvae/version_1/training_logs/latest/metrics.csv
out/motionbricks_vr_h3_pose/version_1/training_logs/latest/metrics.csv
out/motionbricks_vr_h3_root/version_1/training_logs/latest/metrics.csv
```

Các metric chính:

VQVAE:

- `loss/train_loss`: tổng loss, cần giảm dần.
- `loss/train_l_recons_pose`: pose reconstruction loss, càng thấp càng tốt.
- `loss/train_l_recons_root_global`: root global reconstruction loss.
- `loss/train_l_recons_root_local`: root local velocity/heading reconstruction loss.
- `loss/train_perplexity_pose`: codebook usage. Nếu quá thấp kéo dài thì codebook có thể collapse.

Pose model:

- `loss/train_loss`
- `loss/train_pose_loss`

Root model:

- `loss/train_loss`
- `loss/train_global_root_recons_loss`
- `loss/train_local_root_recons_loss`
- `loss/train_num_token_loss`
- `loss/train_top_1_accuracy`, `top_3_accuracy`, `top_5_accuracy`: càng cao càng tốt.

Compile check:

```bash
./.venv/bin/python -m py_compile \
  scripts/train_vqvae.py \
  scripts/train_pose.py \
  scripts/train_root.py
```

Kết quả: pass.

Lưu ý:

- Metric train tốt chưa đảm bảo demo tốt 100%, nhưng giúp phát hiện sớm:
  - loss không giảm
  - root loss quá cao
  - VQVAE codebook collapse
  - root num-token accuracy không học
- Cần thêm eval script riêng để đo reconstruction/generation trên held-out clips nếu muốn đánh giá định lượng mạnh hơn.

## Sau khi train dài 20k/50k/50k

User đã train xong:

```text
VQVAE: 20000 steps
Pose: 50000 steps
Root: 50000 steps
```

Checkpoint selector hiện trỏ đúng checkpoint mới:

```text
vqvae_ckpt = last-v3.ckpt
pose_model_ckpt = last-v2.ckpt
root_model_ckpt = last-v2.ckpt
```

Đã rebuild clip cache sau train:

```bash
./.venv/bin/python scripts/build_vr_h3_clip_cache.py \
  --dataset_pt datasets/vr_h3_motion_features/motions.pt \
  --output out/VR_H3-clip.ckpt
```

Kết quả:

```text
idle: hurry_idle_right_R_001__A349 frames=64
slow_walk: walk_180_R_001__A349 frames=64
walk: walk_180_R_003__A349 frames=64
Saved VR H3 clip cache: out/VR_H3-clip.ckpt
```

Verify clip cache:

```text
qpos shape = (3, 64, 61)
num_frames = [64, 64, 64]
idle root_delta ~= [0.012, -0.004, 0.003]
slow_walk root_delta ~= [0.908, 0.021, -0.013]
walk root_delta ~= [0.993, -0.011, -0.0003]
```

Joint limit:

```text
idle: no violation
slow_walk: worst over ~= 0.00187 rad at right_ankle_roll_joint
walk: worst over ~= 0.00188 rad at left_ankle_roll_joint
```

Tiếp theo chạy demo:

```bash
cd /home/tung/GR00T-WholeBodyControl/motionbricks
./.venv/bin/python scripts/interactive_demo_vr_h3.py
```

Nếu cần khóa chỉ walk mode để test phím W trước:

```bash
./.venv/bin/python scripts/interactive_demo_vr_h3.py --allowed_mode walk
```

Metrics hiện có:

```text
out/motionbricks_vr_h3_root/version_1/training_logs/latest/metrics.csv
```

Lưu ý: VQVAE/Pose trong lần train này có thể chưa có metrics CSV nếu được launch trước lúc bật CSV logger; các lần train sau sẽ có đủ metrics.

## Thêm dataset lớn `vr_h3_data/vr_h3_1`

User thêm data mới:

```text
motionbricks/datas/vr_h3_data/vr_h3_1
```

Kết quả khảo sát:

```text
Tổng file: 142220
Đuôi file: csv
Số subfolder ngày/session: 124
```

Schema:

- Header giống CSV `datas/230424`.
- Sample 1000 file đầu có cùng header 35 cột:
  - `Frame`
  - `root_translateX/Y/Z`
  - `root_rotateX/Y/Z`
  - 28 cột `*_joint_dof`

Đã cập nhật:

```text
scripts/prepare_vr_h3_data.py
```

Thêm filter:

- `--include_patterns`
- `--exclude_patterns`

Mục đích:

- Không ingest toàn bộ 142k file ngay.
- Tạo locomotion-only subset trước để cải thiện WASD.

Đã test prepare sample:

```bash
./.venv/bin/python scripts/prepare_vr_h3_data.py \
  --input_dir datas/vr_h3_data/vr_h3_1 \
  --output_dir datasets/vr_h3_motion_features_big_debug \
  --include_patterns idle,walk,jog,run,turn,start,stop,hurry \
  --exclude_patterns jump,dance,combat,cartwheel,throw,body_check,box \
  --limit 20
```

Kết quả:

```text
Converted clips: 20
Failed files: 0
CSV files found: 142220
CSV files selected: 20
Source FPS: 120
Target FPS: 30.0
```

Nếu bỏ `--limit`, filter locomotion hiện chọn khoảng:

```text
60450 / 142220 clips
```

Khuyến nghị thực tế:

1. Đừng train full 142k ngay.
2. Bắt đầu bằng locomotion subset có limit 5000-10000 để iterate nhanh.
3. Nếu demo cải thiện, tăng lên 20k-60k.

Lệnh tạo dataset lớn mức 10k clip:

```bash
cd /home/tung/GR00T-WholeBodyControl/motionbricks

./.venv/bin/python scripts/prepare_vr_h3_data.py \
  --input_dir datas/vr_h3_data/vr_h3_1 \
  --output_dir datasets/vr_h3_motion_features_loco_10k \
  --include_patterns idle,walk,jog,run,turn,start,stop,hurry \
  --exclude_patterns jump,dance,combat,cartwheel,throw,body_check,box \
  --limit 10000
```

Sau đó train bằng dataset mới:

```bash
./.venv/bin/python scripts/train_vqvae.py \
  --robot vr_h3 \
  --dataset_pt datasets/vr_h3_motion_features_loco_10k/motions.pt

./.venv/bin/python scripts/train_pose.py \
  --robot vr_h3 \
  --dataset_pt datasets/vr_h3_motion_features_loco_10k/motions.pt

./.venv/bin/python scripts/train_root.py \
  --robot vr_h3 \
  --dataset_pt datasets/vr_h3_motion_features_loco_10k/motions.pt
```

Rồi rebuild cache:

```bash
./.venv/bin/python scripts/build_vr_h3_clip_cache.py \
  --dataset_pt datasets/vr_h3_motion_features_loco_10k/motions.pt \
  --output out/VR_H3-clip.ckpt
```

## Đánh giá train với dataset locomotion full

User đã tạo dataset:

```text
datasets/vr_h3_motion_features_loco_full/motions.pt
```

Từ:

```text
datas/vr_h3_data/vr_h3_1
```

Filter:

```text
include = idle, walk, jog, run, turn, start, stop, hurry
exclude = jump, dance, combat, cartwheel, throw, body_check, box
```

Kết quả prepare:

```text
Converted clips: 60450
Failed files: 0
CSV files found: 142220
CSV files selected: 60450
Source FPS: 120
Target FPS: 30.0
```

Train:

```text
VQVAE: 20000 steps
Pose: 50000 steps
Root: 50000 steps
```

Metrics đọc từ:

```text
out/motionbricks_vr_h3_vqvae/version_1/training_logs/latest/metrics.csv
out/motionbricks_vr_h3_pose/version_1/training_logs/latest/metrics.csv
out/motionbricks_vr_h3_root/version_1/training_logs/latest/metrics.csv
```

Summary:

```text
VQVAE loss epoch:            0.319869 -> 0.118066
VQVAE pose recon epoch:      0.227015 -> 0.0705965
VQVAE joint vel epoch:       0.0443291 -> 0.0211337
VQVAE perplexity epoch:      ~7.75 -> ~7.62

Pose loss epoch:             1.81175 -> 1.16493

Root loss epoch:             3.17334 -> 2.16897
Root global recon epoch:     0.208794 -> 0.00737247
Root local recon epoch:      0.259342 -> 0.0669245
Root num-token loss epoch:   2.49641 -> 2.0873
Root top-1 acc epoch:        0.0911 -> 0.2001
Root top-3 acc epoch:        0.2728 -> 0.5136
Root top-5 acc epoch:        0.4529 -> 0.7029
```

Nhận định:

- VQVAE học tốt hơn rõ rệt: reconstruction giảm mạnh.
- Pose model có học: pose loss giảm khoảng 36%.
- Root model có học root trajectory: global root recon giảm rất mạnh.
- Điểm còn yếu là root token classification:
  - top-1 khoảng 20% còn thấp.
  - top-5 khoảng 70% là tạm được nhưng chưa chắc demo WASD đã mượt.
- Kết quả này đủ đáng để rebuild cache và chạy demo thử, nhưng nếu demo vẫn kém thì ưu tiên cải thiện root model/token prediction hoặc train lâu hơn.

Việc cần làm sau train:

```bash
cd /home/tung/GR00T-WholeBodyControl/motionbricks

./.venv/bin/python scripts/build_vr_h3_clip_cache.py \
  --dataset_pt datasets/vr_h3_motion_features_loco_full/motions.pt \
  --output out/VR_H3-clip.ckpt
```

Sau đó chạy demo:

```bash
./.venv/bin/python scripts/interactive_demo_vr_h3.py
```
