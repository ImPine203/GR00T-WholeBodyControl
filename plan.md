# Plan: Chuan hoa pipeline VR_H3 theo MotionBricks guide

Trang thai hien tai: **da implement skeleton/dims va bash train locomotion**.

Da lam:

- Them dummy toe joints cho `H3Skeleton`.
- Regenerate VR_H3 `joints.p` va `parents.p` tu 31 joints sang 33 joints.
- Reset placeholder stats 406 dims trong cac VR_H3 checkpoint folders de motion_rep
  moi instantiate duoc truoc khi prepare data that.
- Doi expected dims trong `prepare_vr_h3_data.py` sang `global=402`, `full=406`.
- Tao script `scripts/regenerate_vr_h3_dummy_toe_skeleton.py`.
- Tao bash train locomotion-only `scripts/train_vr_h3_locomotion_toe.sh`.
- Prepare smoke 20 CSV thanh cong voi dataset:
  `datasets/vr_h3_motion_features_toe_smoke/motions.pt`.

Chua lam:

- Chua train smoke 200 steps de tranh ghi de checkpoint bang model it-step.
- Chua train locomotion dataset that; can chay bash moi khi san sang.

## Muc tieu

Chuyen VR_H3 sang pipeline giong huong dan goc cua MotionBricks:

- Van dung `DualRootGlobalJoints`.
- Van tach `global` representation cho root model va `local` representation cho
  pose/tokenizer.
- Van co 4 foot contact channels: left ankle, left toe, right ankle, right toe.
- Khac G1 o skeleton va dimensions phu hop voi VR_H3.

## Gia dinh

- Dummy toe joints khong can ton tai trong MuJoCo XML.
- Dummy toe joints chi can ton tai trong motion skeleton va neutral skeleton files.
- `mujoco_qpos_converter` se coi dummy toe la dead joints, roi populate position tu
  parent ankle + neutral offset.
- Checkpoint/dataset/cache cu voi VR_H3 31 joints se khong dung lai truc tiep duoc
  sau khi doi skeleton dims.

## Pipeline moi mong muon

VR_H3 hien co 31 joints. Ta se them 2 dummy toe joints:

- `left_toe_base`, parent: `left_ankle_roll_skel`
- `right_toe_base`, parent: `right_ankle_roll_skel`

Tong joints moi: 33.

Dimension moi:

```text
body = (33 - 1) * 3 + 33 * 6 + 33 * 3 + 4 = 397
global = 5 + 397 = 402
local = 4 + 397 = 401
full dual = 5 + 4 + 397 = 406
```

## Cac buoc implement da thuc hien / can kiem tra tiep

### 1. Update VR_H3 skeleton class

File:

```text
motionbricks/motionbricks/motionlib/core/skeletons/vr_h3.py
```

Viec can lam:

- Them `left_toe_base` sau `left_ankle_roll_skel`.
- Them `right_toe_base` sau `right_ankle_roll_skel`.
- Doi foot contact names:

```python
left_foot_joint_names = ["left_ankle_roll_skel", "left_toe_base"]
right_foot_joint_names = ["right_ankle_roll_skel", "right_toe_base"]
```

Verify:

- Instantiate `H3Skeleton` thanh cong.
- `nbjoints == 33`.
- Parent cua toe joints dung.

### 2. Regenerate neutral skeleton files

Files can update cho tung VR_H3 experiment folder:

```text
motionbricks/out/motionbricks_vr_h3_vqvae/version_1/skeleton/joints.p
motionbricks/out/motionbricks_vr_h3_vqvae/version_1/skeleton/parents.p
motionbricks/out/motionbricks_vr_h3_pose/version_1/skeleton/joints.p
motionbricks/out/motionbricks_vr_h3_pose/version_1/skeleton/parents.p
motionbricks/out/motionbricks_vr_h3_root/version_1/skeleton/joints.p
motionbricks/out/motionbricks_vr_h3_root/version_1/skeleton/parents.p
```

Viec can lam:

- Load neutral joints hien tai 31 joints.
- Insert neutral position cho 2 toe joints.
- Toe offset nen di ve phia truoc ban chan trong motion space.
- Motion space: Y-up, Z-forward.
- Parent index trong `parents.p` phai khop `H3Skeleton`.

Verify:

- `joints.p` shape moi la `[33, 3]`.
- `parents.p` shape moi la `[33]`.
- Toe positions khac ankle positions, nam phia truoc ankle hop ly.
- Root neutral joint van la `[0, 0, 0]`.

### 3. Update expected dims trong prepare script

File:

```text
motionbricks/scripts/prepare_vr_h3_data.py
```

Doi:

```python
EXPECTED_GLOBAL_DIM = 402
EXPECTED_FULL_DIM = 406
```

Verify:

- Prepare voi `--limit 20` pass.
- Output `motions.pt` co `global_dim == 402`.
- `stats/motion/mean.npy` va `std.npy` co shape `(406,)`.

### 4. Test converter voi dummy toe dead joints

Can test duong:

```text
CSV -> MuJoCo qpos -> motion transforms -> full dual features -> global features
```

Verify:

- Khong NaN/Inf.
- MuJoCo qpos output van theo so DOF that cua XML.
- Toe joints co trong motion transforms sau khi converter populate dead joints.
- Convert nguoc motion features -> qpos khong crash.

### 5. Prepare data nho

Lenh goi y:

```bash
cd /home/tung/GR00T-WholeBodyControl/motionbricks

./.venv/bin/python scripts/prepare_vr_h3_data.py \
  --input_dir datas/vr_h3_data/vr_h3_1 \
  --output_dir datasets/vr_h3_motion_features_toe_smoke \
  --limit 20
```

Verify:

- Converted clips > 0.
- Failed files = 0 hoac giai thich duoc.
- Dataset/stat dims dung.

### 6. Play/inspect data smoke

Dung mot CSV bat ky va duong motion feature de xem posture:

```bash
./.venv/bin/python scripts/play_vr_h3_pkl.py \
  datas/230424/walk_180_R_003__A349.csv \
  --source motion_feature
```

Verify:

- Robot khong van khop vo ly.
- Chan/ankle/toe contact khong lam hu posture.

### 7. Train smoke - optional

Train it step de chi kiem tra end-to-end:

```bash
./.venv/bin/python scripts/train_vqvae.py \
  --robot vr_h3 \
  --dataset_pt datasets/vr_h3_motion_features_toe_smoke/motions.pt \
  --max_steps 200

./.venv/bin/python scripts/train_pose.py \
  --robot vr_h3 \
  --dataset_pt datasets/vr_h3_motion_features_toe_smoke/motions.pt \
  --max_steps 200

./.venv/bin/python scripts/train_root.py \
  --robot vr_h3 \
  --dataset_pt datasets/vr_h3_motion_features_toe_smoke/motions.pt \
  --max_steps 200
```

Verify:

- 3 model deu train xong.
- Checkpoints duoc tao.
- Metrics CSV duoc tao.

### 8. Prepare dataset that va train lai

Sau khi smoke pass:

- Prepare locomotion-filtered hoac full dataset voi skeleton 33 joints.
- Train lai tu dau VQVAE -> pose -> root.
- Khong reuse checkpoint cu 31 joints.

Lenh moi:

```bash
cd /home/tung/GR00T-WholeBodyControl/motionbricks
bash scripts/train_vr_h3_locomotion_toe.sh
```

### 9. Rebuild clip cache va demo

Sau train:

```bash
./.venv/bin/python scripts/build_vr_h3_clip_cache.py \
  --dataset_pt <new_dataset>/motions.pt \
  --output out/VR_H3-clip.ckpt

./.venv/bin/python scripts/interactive_demo_vr_h3.py
```

Verify:

- Idle dung yen.
- Nhan `w` co root tien ve truoc.
- Gait it rung/truot hon pipeline duplicate ankle.

## Rui ro can theo doi

- Toe offset sai huong se lam contact features sai.
- Neu retarget data goc khong tot, dummy toe chi cai thien contact signal, khong sua
  het chat luong gait.
- Full dataset nhieu action phi locomotion co the lam WASD xau hon locomotion subset.
- Full dataset co the van ton RAM khi train vi `TorchMotionDataset` load `motions.pt`
  vao memory. Neu bi kill o train, can lam lazy/sharded dataset.

## Dau hieu thanh cong

- Feature dims moi khop:

```text
global_dim = 402
full_dim = 406
```

- Data smoke play khong van khop.
- Train smoke chay end-to-end.
- Full train metrics giam on dinh.
- Demo WASD tot hon pipeline 31 joints/duplicate ankle.
