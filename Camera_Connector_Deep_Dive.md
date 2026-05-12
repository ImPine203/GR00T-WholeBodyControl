# Camera Connector Deep Dive

Tài liệu này giải thích cơ chế hoạt động của plugin `camera_connector` dựa trên source code trong thư mục `camera_connector/`. Plugin này là một MuJoCo passive plugin độc lập, có nhiệm vụ trích xuất dữ liệu depth/RGB từ camera trong simulation MuJoCo và publish ra ROS 2 topics dưới dạng `sensor_msgs/Image`, `sensor_msgs/PointCloud2`, và `sensor_msgs/CameraInfo`.

Các file chính đã dùng để đối chiếu:

- Config struct: `include/common.h`
- Plugin header: `include/camera_connector.h`
- ROS bridge header: `include/camera_ros_bridge.h`
- Plugin implementation: `src/camera_connector.cc`
- ROS bridge implementation: `src/camera_ros_bridge.cpp`
- Plugin registration: `src/register.cc`
- Build system: `CMakeLists.txt`
- Example MJCF: `examples/empty.xml`

## Bản đồ đọc source nhanh

| Cơ chế | File/vùng code nên đọc |
| --- | --- |
| Cấu hình plugin từ MJCF | `include/common.h:CameraConfig`, `src/camera_connector.cc:Create()` |
| Đăng ký plugin với MuJoCo | `src/register.cc`, `src/camera_connector.cc:RegisterPlugin()` |
| Vòng đời plugin (init, compute, destroy) | `src/camera_connector.cc:CameraConnector` |
| Khởi tạo offscreen rendering | `src/camera_ros_bridge.cpp:initRendering()` |
| Render trên main thread | `src/camera_ros_bridge.cpp:syncRender()` |
| Khởi tạo ROS 2 node và publishers | `src/camera_ros_bridge.cpp:initRos()` |
| Tính camera intrinsics | `src/camera_ros_bridge.cpp:computeIntrinsics()` |
| Xử lý depth/RGB/pointcloud và publish | `src/camera_ros_bridge.cpp:publishRosData()` |
| Background publish thread | `src/camera_connector.cc:PublishThread()` |
| Rate limiter | `include/camera_ros_bridge.h:Rate` |

## 1. Bức tranh tổng quát

Khác với `multiverse_connector` dùng ZeroMQ REQ/REP để đồng bộ dữ liệu qua Multiverse Server, `camera_connector` là một plugin **standalone**: nó tự tạo ROS 2 node ngay trong process MuJoCo, tự render offscreen, tự xử lý pixel, và tự publish ra ROS 2 topics. Không có server trung gian.

Luồng dữ liệu:

```
MuJoCo Simulation Step
        │
        ▼
  CameraConnector::Compute()     ← Main Thread, mỗi step
        │
        ▼ (throttled theo ros_pub_frequency)
  CameraRosBridge::syncRender()  ← Main Thread, render offscreen
        │
        ▼ (mutex + pixel_ready_ flag)
  CameraRosBridge::publishRosData()  ← Background Thread
        │
        ├──► CameraInfo topic
        ├──► Depth Image topic      (32FC1, metric meters)
        ├──► RGB Image topic        (rgb8)
        └──► PointCloud2 topic      (x,y,z float32)
```

Điểm quan trọng: rendering (OpenGL) **phải** chạy trên main thread vì OpenGL context không thread-safe. Publishing chạy trên background thread để không block simulation step.

## 2. Thành phần chính

### 2.1. `CameraConfig` (`common.h`)

Struct chứa toàn bộ cấu hình plugin:

```cpp
struct CameraConfig {
    bool enable_ros = true;
    std::string ros_node_name = "camera_ros_bridge";
    std::string camera_name = "depth_camera";
    double ros_pub_frequency = 30.0; // Hz
    bool point_cloud_enable = true;
    bool depth_enable = true;
    bool rgb_enable = true;
    std::string topic_prefix = "/camera_on_face_ob/zed_node";
    int width = 0;   // 0 = auto từ MJCF offwidth
    int height = 0;  // 0 = auto từ MJCF offheight
};
```

### 2.2. `CameraConnector` (`camera_connector.h/cc`)

Lớp plugin chính, implement giao diện MuJoCo plugin:

- `Create()`: factory, đọc config từ MJCF attributes.
- `Compute()`: được gọi mỗi simulation step, throttle và gọi `syncRender()`.
- `Reset()`: không có state để reset.
- `Advance()`: không dùng.
- `StateSize()`: trả 0, plugin không dùng `plugin_state`.
- `RegisterPlugin()`: đăng ký plugin vào global registry.
- `PublishThread()`: background thread chạy vòng publish.

Constructor khởi tạo `CameraRosBridge` và spawn `publish_thread_`. Destructor gọi `shutdown()` rồi join thread.

### 2.3. `CameraRosBridge` (`camera_ros_bridge.h/cpp`)

Lớp xử lý rendering và ROS 2. Chứa:

- MuJoCo rendering components: `mjvCamera`, `mjvOption`, `mjvScene`, `mjrContext`.
- ROS 2 components (static): `rclcpp::Node`, publishers cho CameraInfo, PointCloud2, Depth Image, RGB Image.
- Pixel buffers: `pixel_buffer_` (depth float), `rgb_buffer_` (unsigned char).
- Mutex `pixel_mtx_` và flag `pixel_ready_` để đồng bộ main thread ↔ background thread.

Lý do ROS 2 components là **static**: MuJoCo gọi `Create()` hai lần (một lần TryCompile, một lần thật). Static đảm bảo ROS context không bị tạo/huỷ khi TryCompile.

### 2.4. `Rate` (`camera_ros_bridge.h`)

Rate limiter đơn giản dùng `std::chrono::steady_clock`. Background thread dùng để sleep đúng tần số publish.

## 3. Plugin registration

`register.cc` dùng macro `mjPLUGIN_LIB_INIT` để gọi `CameraConnector::RegisterPlugin()` khi shared library được load.

`RegisterPlugin()` đăng ký plugin `"mujoco.camera_connector"` với:

- Capability: `mjPLUGIN_PASSIVE` — plugin chạy passive, không ảnh hưởng physics.
- 10 attributes (sorted alphabetical, bắt buộc bởi MuJoCo): `camera_name`, `depth_enable`, `enable_ros`, `height`, `point_cloud_enable`, `rgb_enable`, `ros_node_name`, `ros_pub_frequency`, `topic_prefix`, `width`.
- Lambdas cho `init`, `destroy`, `reset`, `compute`.

## 4. Cấu hình trong MJCF

```xml
<mujoco>
  <extension>
    <plugin plugin="mujoco.camera_connector">
      <instance name="camera_client">
        <config key="enable_ros" value="true" />
        <config key="ros_node_name" value="camera_ros_bridge" />
        <config key="camera_name" value="depth_camera" />
        <config key="ros_pub_frequency" value="30.0" />
        <config key="topic_prefix" value="/camera_on_face_ob/zed_node" />
        <config key="depth_enable" value="true" />
        <config key="rgb_enable" value="true" />
        <config key="point_cloud_enable" value="true" />
        <!-- width/height: bỏ trống hoặc 0 để auto từ MJCF offscreen buffer -->
      </instance>
    </plugin>
  </extension>
</mujoco>
```

| Key | Type | Default | Mô tả |
| --- | --- | --- | --- |
| `enable_ros` | bool | `true` | Bật/tắt toàn bộ ROS publishing |
| `ros_node_name` | string | `camera_ros_bridge` | Tên ROS 2 node |
| `camera_name` | string | `depth_camera` | Tên camera trong MJCF model |
| `ros_pub_frequency` | double | `30.0` | Tần số publish (Hz) |
| `topic_prefix` | string | `/camera_on_face_ob/zed_node` | Prefix cho topic names |
| `depth_enable` | bool | `true` | Bật depth image |
| `rgb_enable` | bool | `true` | Bật RGB image |
| `point_cloud_enable` | bool | `true` | Bật point cloud |
| `width` | int | `0` | Chiều rộng ảnh, 0 = auto |
| `height` | int | `0` | Chiều cao ảnh, 0 = auto |

## 5. Vòng đời plugin

### 5.1. Khởi tạo (`Create` → Constructor)

```
MuJoCo load MJCF
    │
    ▼
CameraConnector::Create(m, d, instance)
    │  đọc config từ mj_getPluginConfig()
    │  tạo CameraConfig
    ▼
CameraConnector constructor
    │  tạo CameraRosBridge(m, d)
    │  gán config vào bridge
    │  spawn publish_thread_
    ▼
PublishThread() bắt đầu chạy
    │  đợi d_ != nullptr
    ▼
Sẵn sàng, đợi pixel_ready_
```

Lưu ý: rendering và ROS **chưa** được khởi tạo trong constructor. Chúng được defer sang lần đầu `syncRender()` được gọi trên main thread (lazy init).

### 5.2. Mỗi simulation step (`Compute`)

```cpp
void CameraConnector::Compute(const mjModel* m, mjData* d, int instance) {
    double interval = 1.0 / max(1.0, config_.ros_pub_frequency);
    if (d->time - last_render_t_ >= interval) {
        last_render_t_ = d->time;
        camera_bridge_.syncRender();
    }
}
```

`Compute()` được gọi mỗi step nhưng chỉ render khi đủ thời gian theo `ros_pub_frequency`. Ví dụ nếu MuJoCo chạy 1000 Hz nhưng camera 30 Hz, chỉ render mỗi ~33 steps.

### 5.3. Shutdown (`Destructor`)

```
~CameraConnector()
    │  camera_bridge_.shutdown()  → set is_running = false, cleanup rendering
    │  publish_thread_.join()     → đợi background thread kết thúc
    ▼
~CameraRosBridge()
    │  cleanupRendering()  → mjr_freeContext, mjv_freeScene
    │  cleanupRos()        → reset publishers, node, shutdown context
    ▼
Done
```

## 6. Rendering pipeline

### 6.1. `initRendering()` — Lazy init trên main thread

Chỉ gọi một lần, khi `syncRender()` được gọi lần đầu:

1. Tìm camera theo tên: `mj_name2id(model, mjOBJ_CAMERA, camera_name)`.
2. Resolve resolution:
   - Nếu config `width > 0 && height > 0`: dùng config.
   - Ngược lại: dùng `model->vis.global.offwidth/offheight`.
3. Tính camera intrinsics từ `cam_fovy`.
4. Khởi tạo MuJoCo scene/camera/option/context.
5. Tạo hidden GLFW window cho OpenGL context:
   ```cpp
   glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
   GLFWwindow* window = glfwCreateWindow(w, h, "Hidden", NULL, NULL);
   glfwMakeContextCurrent(window);
   ```
6. Gọi `mjr_makeContext()` để tạo GPU rendering context.
7. Set `rendering_initialized_ = true`.

### 6.2. `computeIntrinsics()` — Camera intrinsics từ MuJoCo

```
fovy_rad = cam_fovy[cam_id] * π / 180
fy = height / (2 * tan(fovy_rad / 2))
fx = fy                               ← giả định square pixels
cx = width / 2
cy = height / 2
```

Intrinsics được dùng cho cả CameraInfo message và back-projection PointCloud.

### 6.3. `syncRender()` — Render trên main thread

Gọi từ `Compute()`, chạy trên MuJoCo main thread:

1. Lazy init: gọi `initRendering()` và `initRos()` nếu chưa init.
2. Update scene: `mjv_updateScene(model, data, opt, NULL, cam, mjCAT_ALL, scn)`.
3. Set offscreen buffer: `mjr_setBuffer(mjFB_OFFSCREEN, con)`.
4. Render: `mjr_render(viewport, scn, con)`.
5. Lock mutex, đọc pixels:
   ```cpp
   mjr_readPixels(rgb_ptr, depth_ptr, viewport, con);
   pixel_ready_ = true;
   ```

`mjr_readPixels()` đọc cả RGB (`unsigned char*`) và depth (`float*`) trong một lần gọi. Depth buffer của MuJoCo/OpenGL chứa giá trị normalized 0..1, chưa phải metric.

## 7. ROS 2 integration

### 7.1. `initRos()` — Khởi tạo ROS 2

1. Tạo `rclcpp::Context` riêng với `shutdown_on_signal = false` (để MuJoCo tự quản lý signal).
2. Tạo `rclcpp::Node` với tên từ config.
3. Tạo publishers dựa trên config enable flags:

| Điều kiện | Topic | Message type |
| --- | --- | --- |
| Luôn tạo | `{prefix}/depth/depth_registered/camera_info` | `sensor_msgs/CameraInfo` |
| `depth_enable` | `{prefix}/depth/depth_registered` | `sensor_msgs/Image` (32FC1) |
| `rgb_enable` | `{prefix}/left/image_rect_color` | `sensor_msgs/Image` (rgb8) |
| `point_cloud_enable` | `{prefix}/point_cloud/cloud_registered` | `sensor_msgs/PointCloud2` |

4. Pre-allocate messages: resize data vectors, set headers, set encoding.
5. Fill `CameraInfo.K` và `CameraInfo.P` matrices từ intrinsics.

### 7.2. ROS 2 components là static

```cpp
static rclcpp::Context::SharedPtr ros_context_;
static rclcpp::Node::SharedPtr ros_node_;
static rclcpp::Publisher<...>::SharedPtr info_pub_;
// ... các publisher khác
static bool ros_initialized_;
```

MuJoCo có thể gọi `Create()` rồi `destroy()` trong phase TryCompile trước khi tạo instance thật. Static đảm bảo ROS context sống sót qua quá trình này.

### 7.3. `cleanupRos()` — Shutdown ROS 2

Thứ tự cleanup:
1. Reset tất cả publishers.
2. Reset node.
3. Shutdown context nếu còn valid.
4. Reset context pointer.

## 8. Data processing và publishing

### 8.1. `publishRosData()` — Background thread

Chạy trong `PublishThread()` với tần số `ros_pub_frequency`. Mỗi vòng:

1. Kiểm tra `rendering_initialized_` và `pixel_ready_`.
2. Lock `pixel_mtx_`.
3. Gọi `rclcpp::spin_some(node)` để xử lý ROS callbacks.
4. Publish `CameraInfo` (luôn luôn).
5. Tính `znear` và `zfar` từ model:
   ```cpp
   float extent = model->stat.extent;
   float znear = model->vis.map.znear * extent;
   float zfar = model->vis.map.zfar * extent;
   ```
6. Vòng lặp xử lý pixel (unified loop qua toàn bộ resolution):

```
Với mỗi pixel (u, v):
    │
    ├── Flip vertical: src_idx dùng (height-1-v) vì OpenGL bottom-up, ROS top-down
    │
    ├── Tuyến tính hoá depth:
    │   d_raw = pixel_buffer[src_idx]           ← giá trị 0..1 từ OpenGL
    │   metric_z = (znear * zfar) / (zfar - d_raw * (zfar - znear))   ← mét
    │
    ├── Depth Image: ghi metric_z vào depth_msg (32FC1)
    │
    ├── RGB Image: copy 3 bytes từ rgb_buffer (đã flip)
    │
    └── PointCloud2:
        nếu znear < metric_z < 0.95*zfar:
            x = (u - cx) * metric_z / fx
            y = (v - cy) * metric_z / fy
            z = metric_z
        ngược lại:
            x = y = z = NaN
```

7. Publish các message đã bật.
8. Set `pixel_ready_ = false`.

### 8.2. Depth linearization

MuJoCo/OpenGL depth buffer chứa giá trị phi tuyến 0..1. Công thức chuyển sang metric:

```
metric_z = (znear × zfar) / (zfar - d_raw × (zfar - znear))
```

Trong đó `znear` và `zfar` được tính từ `model->vis.map.znear/zfar` nhân với `model->stat.extent` (kích thước tổng thể của scene).

### 8.3. PointCloud back-projection

Dùng pinhole camera model chuẩn:

```
X = (u - cx) × Z / fx
Y = (v - cy) × Z / fy
Z = metric_z
```

Các điểm ngoài khoảng hữu ích (`metric_z >= 0.95 * zfar` hoặc `<= znear`) được set NaN.

## 9. Threading model

```
┌─────────────────────────────────────────┐
│              MAIN THREAD                │
│                                         │
│  MuJoCo step loop:                      │
│    mj_step()                            │
│    CameraConnector::Compute()           │
│      └── syncRender() [throttled]       │
│            mjv_updateScene()            │
│            mjr_render()                 │
│            mjr_readPixels()             │
│            ───mutex lock───             │
│            pixel_buffer_ ← GPU data     │
│            rgb_buffer_ ← GPU data       │
│            pixel_ready_ = true          │
│            ───mutex unlock───           │
└─────────────────────────────────────────┘

┌─────────────────────────────────────────┐
│          BACKGROUND THREAD              │
│                                         │
│  PublishThread():                       │
│    while (is_running):                  │
│      ───mutex lock───                   │
│      if pixel_ready_:                   │
│        linearize depth                  │
│        flip + copy RGB                  │
│        compute pointcloud               │
│        publish ROS messages             │
│        pixel_ready_ = false             │
│      ───mutex unlock───                 │
│      rate.sleep()                       │
└─────────────────────────────────────────┘
```

Đồng bộ qua:
- `std::mutex pixel_mtx_`: bảo vệ `pixel_buffer_`, `rgb_buffer_`.
- `bool pixel_ready_`: flag báo main thread đã render xong.
- `std::atomic_bool is_running`: flag shutdown.

## 10. Build system

`CMakeLists.txt` build shared library `camera_connector.so`:

Dependencies:
- `rclcpp` — ROS 2 client library
- `sensor_msgs` — CameraInfo, Image, PointCloud2
- `cv_bridge` — (linked nhưng chưa dùng trực tiếp trong code hiện tại)
- `OpenCV` — (linked nhưng chưa dùng trực tiếp)
- `glfw3` — hidden window cho offscreen OpenGL
- `mujoco` — rendering và simulation API
- `spdlog` + `fmt` — logging
- `pthread` — background thread

ABI isolation:
```
-Wl,-Bsymbolic-functions
```
Đảm bảo symbols của plugin không xung đột với symbols của MuJoCo host hay ROS libraries khác.

RPATH:
```
$ORIGIN/../../plugin/camera_connector/lib;$ORIGIN/lib
```
Cho phép plugin tìm thư viện runtime trong thư mục `lib/` tương đối.

## 11. So sánh với multiverse_connector

| Đặc điểm | multiverse_connector | camera_connector |
| --- | --- | --- |
| Vai trò | Đồng bộ state giữa nhiều simulation | Stream camera data ra ROS 2 |
| Transport | ZeroMQ REQ/REP qua Multiverse Server | ROS 2 publishers trực tiếp |
| Dữ liệu | position, quaternion, joint, force... | Depth Image, RGB Image, PointCloud2 |
| Protocol | Binary streaming sau JSON handshake | ROS 2 message publish |
| Plugin type | `mjPLUGIN_PASSIVE` | `mjPLUGIN_PASSIVE` |
| Threading | Đồng bộ trong `Compute()` | Main thread render + background publish |
| Dependency | ZeroMQ, jsoncpp | ROS 2, GLFW, OpenGL |

## 12. Các điểm cần lưu ý trong source hiện tại

### 12.1. OpenGL context và GLFW window không được free

`initRendering()` tạo hidden GLFW window nhưng không lưu pointer. `cleanupRendering()` chỉ free MuJoCo context/scene, không gọi `glfwDestroyWindow()` hay `glfwTerminate()`. Trong thực tế, window sẽ bị leak cho đến khi process kết thúc.

### 12.2. `cv_bridge` và `OpenCV` linked nhưng không dùng

`CMakeLists.txt` link `cv_bridge` và `OpenCV` nhưng không có file nào `#include` chúng. Đây là dead dependencies có thể loại bỏ.

### 12.3. Tần số render bị ràng buộc bởi simulation timestep

Throttle trong `Compute()` dùng `d->time` (simulation time). Nếu simulation chạy realtime và timestep lớn (ví dụ 10ms), tần số render tối đa chỉ là 100 Hz. Nếu simulation chạy nhanh hơn realtime, render có thể gọi nhanh hơn `ros_pub_frequency` yêu cầu vì so sánh dựa trên sim time chứ không phải wall clock.

### 12.4. `spin_some()` gọi trong publish thread

`rclcpp::spin_some(ros_node_)` được gọi trong `publishRosData()` trên background thread. Nó xử lý pending ROS callbacks nhưng node hiện tại chỉ có publishers, không có subscribers hay timers, nên call này gần như no-op.

### 12.5. Static members và multiple instances

ROS node và publishers là static. Nếu MJCF khai báo nhiều instance `camera_connector` (ví dụ 2 camera), chỉ instance đầu tiên gọi `initRos()`. Instance thứ hai sẽ dùng chung node/publishers, dẫn đến conflict topic names hoặc data overwrite.

### 12.6. `init_attempted` static local trong `syncRender()`

```cpp
static bool init_attempted = false;
```

Biến static local này đảm bảo `initRendering()` chỉ được thử một lần. Nếu init fail, camera connector sẽ im lặng không hoạt động trong toàn bộ session, không có cơ chế retry.

## 13. Luồng end-to-end

1. MuJoCo load MJCF, thấy `<plugin plugin="mujoco.camera_connector">`.
2. MuJoCo load `camera_connector.so`, gọi `mjPLUGIN_LIB_INIT` → `RegisterPlugin()`.
3. MuJoCo gọi `init` lambda → `Create()` → constructor → spawn background thread.
4. Background thread chờ `d_` ready.
5. Mỗi simulation step, MuJoCo gọi `compute` lambda → `Compute()`.
6. `Compute()` throttle theo tần số, gọi `syncRender()`.
7. Lần đầu `syncRender()`: init GLFW hidden window, init MuJoCo rendering, init ROS 2.
8. Mỗi lần `syncRender()`: update scene → render offscreen → read pixels vào buffer.
9. Background thread thấy `pixel_ready_`, lock mutex, xử lý depth/RGB/pointcloud, publish ROS.
10. Khi MuJoCo shutdown: destructor gọi `shutdown()` → `is_running = false` → join thread → cleanup rendering → cleanup ROS.

## 14. Mental model ngắn gọn

Camera Connector là một MuJoCo passive plugin chạy offscreen rendering trên main thread theo tần số cấu hình, rồi đẩy pixel data sang background thread để tuyến tính hoá depth, back-project thành point cloud, và publish ra ROS 2 topics. Nó hoạt động hoàn toàn độc lập, không đi qua Multiverse Server.
