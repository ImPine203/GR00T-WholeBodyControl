# Hướng Dẫn Tích Hợp Robot Tùy Chỉnh Vào MotionBricks

Tài liệu này đã được đối chiếu với source hiện tại trong `/home/tung/GR00T-WholeBodyControl/motionbricks/`.

Trạng thái quan trọng của repo hiện tại:

- MotionBricks đang được đóng gói như một subproject, không nằm trực tiếp ở repo root. Khi chạy lệnh, nên `cd motionbricks`.
- Runtime demo và checkpoint public hiện được chuẩn bị cho **Unitree G1**.
- Các script `train_vqvae.py`, `train_pose.py`, `train_root.py` hiện dùng `SyntheticMotionDataset` để smoke test pipeline; đây chưa phải full data pipeline cho robot mới.
- Custom robot không phải chỉ đổi YAML. Bạn cần chuẩn bị skeleton, metadata neutral pose, motion features/stats, converter MuJoCo, clip cache, và checkpoint tương ứng.

---

## 0. Bản Đồ Đường Dẫn

Từ repo root `/home/tung/GR00T-WholeBodyControl`:

| Thành phần | Đường dẫn hiện tại |
| --- | --- |
| Subproject MotionBricks | `motionbricks/` |
| Python package | `motionbricks/motionbricks/` |
| Script demo/train | `motionbricks/scripts/` |
| Skeleton class | `motionbricks/motionbricks/motionlib/core/skeletons/` |
| Motion representation | `motionbricks/motionbricks/motionlib/core/motion_reps/` |
| MuJoCo converter | `motionbricks/motionbricks/helper/mujoco_helper.py` |
| Demo controller/agent/clips | `motionbricks/motionbricks/motion_backbone/demo/` |
| G1 XML/mesh assets | `motionbricks/assets/skeletons/g1/` |
| Pretrained checkpoints | `motionbricks/out/` |

Các lệnh dưới đây mặc định chạy từ:

```bash
cd /home/tung/GR00T-WholeBodyControl/motionbricks
```

---

## 1. Hiểu Các Artifact Cần Có

Một robot mới cần hai nhóm artifact khác nhau:

| Artifact | Dùng bởi | Nội dung |
| --- | --- | --- |
| MuJoCo asset | demo/runtime converter | XML scene, XML robot, mesh STL, joint hinge names |
| Motion skeleton metadata | `SkeletonBase`, motion representation, stats | `joints.p`, `parents.p`, optional `skeleton.yaml`, neutral joint offsets trong motion space |

Với G1, MuJoCo asset nằm ở `assets/skeletons/g1/`, còn skeleton metadata nằm trong từng checkpoint version, ví dụ:

```text
out/motionbricks_vqvae/version_1/skeleton/
  joints.p
  parents.p
  skeleton.yaml
```

`SkeletonBase` sẽ load `joints.p` và `parents.p` nếu `folder` được truyền vào config.

---

## 2. Chuẩn Bị MuJoCo XML

Tạo thư mục mới:

```text
motionbricks/assets/skeletons/<robot_name>/
```

Cần có ít nhất:

- XML robot dùng cho converter, tương tự `assets/skeletons/g1/g1.xml`.
- XML scene dùng cho viewer, tương tự `assets/skeletons/g1/scene_29dof.xml`.
- Mesh files nếu XML tham chiếu đến STL/OBJ.

Lưu ý naming hiện tại của converter:

- `mujoco_qpos_converter` đọc các `<joint>` trong XML và map tên joint bằng quy ước `"<joint_name>_joint" -> "<joint_name>_skel"`.
- Root joint đang giả định là `pelvis_skel`.
- G1 qpos hiện hard-code là 36 chiều: root 7 + 29 hinge joints.

Vì vậy, với robot không phải G1, bạn gần như chắc chắn phải sửa `motionbricks/motionbricks/helper/mujoco_helper.py` để:

- Không hard-code `pelvis_skel` nếu root của bạn tên khác.
- Cấp phát qpos theo `7 + số_hinge_joint` thay vì 36.
- Map joint order giữa MuJoCo XML và skeleton class.
- Xử lý dead/end-effector joints nếu motion skeleton có joint không tồn tại trong MuJoCo.

---

## 3. Tạo Skeleton Class

Skeleton thật kế thừa từ `SkeletonBase`, không phải class `Skeleton` riêng.

File tham chiếu chính:

```text
motionbricks/motionbricks/motionlib/core/skeletons/base.py
motionbricks/motionbricks/motionlib/core/skeletons/g1.py
```

Ví dụ tối giản:

```python
from motionbricks.motionlib.core.skeletons.base import SkeletonBase


class H1Skeleton(SkeletonBase):
    name = "h1skel"

    bone_order_names_with_parents = [
        ("pelvis_skel", None),
        ("left_hip_pitch_skel", "pelvis_skel"),
        ("left_hip_roll_skel", "left_hip_pitch_skel"),
        ("left_hip_yaw_skel", "left_hip_roll_skel"),
        ("left_knee_skel", "left_hip_yaw_skel"),
        ("left_ankle_pitch_skel", "left_knee_skel"),
        ("left_ankle_roll_skel", "left_ankle_pitch_skel"),
        ("right_hip_pitch_skel", "pelvis_skel"),
        ("right_hip_roll_skel", "right_hip_pitch_skel"),
        ("right_hip_yaw_skel", "right_hip_roll_skel"),
        ("right_knee_skel", "right_hip_yaw_skel"),
        ("right_ankle_pitch_skel", "right_knee_skel"),
        ("right_ankle_roll_skel", "right_ankle_pitch_skel"),
    ]

    left_foot_joint_names = ["left_ankle_roll_skel"]
    right_foot_joint_names = ["right_ankle_roll_skel"]
    hip_joint_names = ["right_hip_pitch_skel", "left_hip_pitch_skel"]
```

Sau đó export class trong:

```text
motionbricks/motionbricks/motionlib/core/skeletons/__init__.py
```

Nếu bạn giữ biểu diễn mặc định `DualRootGlobalJoints`, số chiều phụ thuộc vào số joint `J`:

| Thành phần | Công thức |
| --- | --- |
| Body | `(J - 1) * 3 + J * 6 + J * 3 + 4 = 12J + 1` |
| Global subset | `5 + body = 12J + 6` |
| Local subset | `4 + body = 12J + 5` |
| Full dual stats | `5 + 4 + body = 12J + 10` |

G1 dùng `J=34`, nên global subset là 414 dims, local subset là 413 dims, full dual stats là 418 dims.

---

## 4. Chuẩn Bị Skeleton Metadata

`SkeletonBase` cần `joints.p` và `parents.p` khi config có `skeleton.folder`.

Yêu cầu:

- `joints.p`: tensor `[J, 3]`, neutral offsets/positions, root ở `[0, 0, 0]`.
- `parents.p`: tensor `[J]`, parent index theo đúng `bone_order_names_with_parents`, root là `-1`.
- Thứ tự trong hai file phải khớp tuyệt đối với `bone_order_names_with_parents`.

Bạn có thể đặt metadata ở một folder riêng hoặc trong checkpoint version mới:

```text
out/motionbricks_<robot>_vqvae/version_1/skeleton/
out/motionbricks_<robot>_pose/version_1/skeleton/
out/motionbricks_<robot>_root/version_1/skeleton/
```

Config mẫu:

```yaml
skeleton:
  name: h1skel
  folder: out/motionbricks_h1_vqvae/version_1/skeleton
  t_pose: capture
  _target_: motionbricks.motionlib.core.skeletons.h1.H1Skeleton
```

---

## 5. Chuẩn Bị Motion Data Và Stats

Dataset tối thiểu cần trả về batch theo interface hiện tại:

```python
{"keyid": int, "motion": Tensor[T, global_feature_dim]}
```

Trong pipeline hiện tại, `motion` là **normalized global subset**, không phải full 418-dim dual tensor. Với G1, `global_feature_dim = 414`.

Tuy vậy, `stats/motion/mean.npy` và `std.npy` trong checkpoint hiện có là full dual stats. Với G1 chúng có shape `(418,)`. Với robot mới, nếu dùng `DualRootGlobalJoints`, stats nên có chiều `12J + 10`.

Luồng feature chuẩn:

1. Retarget motion sang skeleton robot.
2. Có `posed_joints`, `local_joint_rots`, `global_joint_rots`, optional `foot_contacts`.
3. Gọi motion representation trong `motionlib/core/motion_reps/` để sinh feature.
4. Tính `mean.npy` và `std.npy` trên full dual representation.
5. Dataset trả về normalized **global** subset để model training step tự convert sang local khi cần.

Tham chiếu code:

- `motionbricks/motionbricks/data/synthetic_dataset.py`
- `motionbricks/motionbricks/motionlib/core/motion_reps/tools/motion_features.py`
- `motionbricks/motionbricks/motionlib/core/utils/stats.py`

Nếu robot khác G1 hoặc dữ liệu không khớp representation này, cách thực tế hơn là viết dataset/motion-rep riêng nhưng vẫn giữ contract đầu vào của ba model.

---

## 6. Cấu Hình Train

Repo hiện không có thư mục Hydra `conf/` độc lập cho MotionBricks. Ba script train load `hparams.yaml` từ các checkpoint version trong `out/`, rồi patch lại path skeleton/stats/data để chạy synthetic training.

Chạy smoke test hiện tại:

```bash
python scripts/train_vqvae.py --max_steps 100
python scripts/train_pose.py --max_steps 100
python scripts/train_root.py --max_steps 100
```

Muốn train robot mới, bạn cần tạo hoặc sửa config version tương ứng:

```text
out/motionbricks_<robot>_vqvae/version_1/hparams.yaml
out/motionbricks_<robot>_pose/version_1/hparams.yaml
out/motionbricks_<robot>_root/version_1/hparams.yaml
```

Các field bắt buộc phải khớp:

```yaml
skeleton:
  name: h1skel
  folder: out/motionbricks_h1_vqvae/version_1/skeleton
  t_pose: capture
  _target_: motionbricks.motionlib.core.skeletons.h1.H1Skeleton

motion_rep:
  name: ${skeleton.name}_dual_root_global_joints
  stats:
    _target_: motionbricks.motionlib.core.utils.stats.Stats
    folder: out/motionbricks_h1_vqvae/version_1/stats/motion
  _target_: motionbricks.motionlib.core.motion_reps.dual_root_global_joints.GlobalRootGlobalJoints

fps: 30
```

Với pose model, `model.args.vqvae_model_ckpt_path` phải trỏ tới checkpoint VQVAE của cùng skeleton/feature dimension.

Không thể reuse checkpoint G1 cho robot có số joint hoặc feature layout khác.

---

## 7. Cập Nhật Demo Tương Tác

Các file demo hiện còn hard-code G1 ở nhiều nơi:

- `motion_backbone/demo/full_agent.py` luôn dùng `clip_holder_G1`.
- `motion_backbone/demo/controllers.py` luôn dùng `clip_holder_G1`.
- `motion_backbone/demo/clips.py` chỉ có `clip_holder_G1`.
- `scripts/interactive_demo_g1.py` có parser `--humanoid_xml`, nhưng `navigation_demo` thực tế dùng `args.humanoid_scene_xml` và `args.skeleton_xml`.

Để tạo demo cho robot mới:

1. Tạo `clip_holder_<Robot>` trong `motion_backbone/demo/clips.py`.
2. Cho `CLIPS` chứa các mode như `idle`, `walk`, style motions, `avg_root_vel`, `allowed_pred_num_tokens`.
3. Sửa `full_agent.py` và `controllers.py` để chọn clip holder theo `args.clips` thay vì hard-code `clip_holder_G1`.
4. Tạo cache clip mới tương tự `out/G1-clip.ckpt`.
5. Copy `scripts/interactive_demo_g1.py` thành script riêng và set đúng:

```python
args.humanoid_scene_xml = "assets/skeletons/h1/scene.xml"
args.skeleton_xml = "assets/skeletons/h1/h1.xml"
args.clips_ckpt = "out/H1-clip.ckpt"
args.clips = "H1"
```

Nếu không có `clips_ckpt`, demo sẽ cố instantiate dataset từ config. Config hiện tham chiếu các class dataset full release chưa có trong source public, nên runtime G1 hiện dựa vào `out/G1-clip.ckpt` để tránh đường này.

---

## 8. Checklist Tích Hợp Robot Mới

- MuJoCo XML/scene/mesh chạy được độc lập.
- Joint names trong XML map được sang skeleton names.
- `SkeletonBase` subclass có đúng parent tree, foot joints, hip joints.
- `joints.p` và `parents.p` khớp class skeleton.
- Motion data đã retarget sang robot và có đủ joint transforms.
- `mean.npy`/`std.npy` full dual stats đúng chiều.
- Dataset trả normalized global subset đúng chiều.
- VQVAE train được với skeleton mới.
- Pose/root model dùng checkpoint và stats cùng skeleton.
- MuJoCo converter không còn hard-code qpos G1 nếu robot khác DOF.
- Clip holder/cache mới có reference clips cho demo.
- Demo script truyền đúng `humanoid_scene_xml`, `skeleton_xml`, `clips_ckpt`, `clips`.
