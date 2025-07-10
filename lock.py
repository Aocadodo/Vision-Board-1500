import sensor
import image
import time
import gc
import math
import lcd  # 添加LCD模块
from pyb import LED
from pyb import UART

red_led = LED(1)
green_led = LED(2)
blue_led = LED(3)

# 初始化UART通信
uart = UART(2, 9600, timeout_char=200)

# 强制清理内存
gc.collect()

# 初始化摄像头
sensor.reset()
sensor.set_pixformat(sensor.RGB565)  # 改为RGB565以支持LCD彩色显示
sensor.set_framesize(sensor.QVGA)
sensor.set_contrast(2)
sensor.set_brightness(0)
sensor.set_gainceiling(8)
sensor.skip_frames(time=1000)
sensor.set_auto_gain(True)
sensor.set_auto_whitebal(True)
sensor.set_hmirror(True)
#sensor.set_vflip(True)

# 初始化LCD
lcd.init()

# 加载人脸级联模型
face_cascade = image.HaarCascade("frontalface", stages=20)

# 存储用户人脸信息
user_faces = []

# 配置参数
num_users_to_enroll = 1
faces_per_user = 6  # 适中的样本数量
print("准备录入", num_users_to_enroll, "位用户的人脸，每人拍摄", faces_per_user, "张照片")

def preprocess_face(face_roi):
    """温和的人脸预处理"""
    try:
        # 转换为灰度进行处理
        if face_roi.format() == sensor.RGB565:
            face_roi = face_roi.to_grayscale()

        # 直方图均衡化增强对比度
        face_roi.histeq()

        # 轻微降噪
        face_roi.gaussian(1)

        return face_roi
    except Exception as e:
        print(f"预处理失败: {e}")
        return face_roi

def extract_stable_lbp_features(face_roi, radius=1, neighbors=8):
    """稳定的LBP特征提取"""
    try:
        # 确保是灰度图像
        if face_roi.format() == sensor.RGB565:
            face_roi = face_roi.to_grayscale()

        width = face_roi.width()
        height = face_roi.height()

        if width < 24 or height < 24:
            return None

        # LBP模式偏移量
        offsets = []
        for i in range(neighbors):
            angle = 2 * math.pi * i / neighbors
            dx = int(radius * math.cos(angle))
            dy = int(radius * math.sin(angle))
            offsets.append((dx, dy))

        # 更大的人脸区域划分，增加稳定性
        regions = [
            (0.1, 0.1, 0.5, 0.5, "左上"),     # 左上（包含左眼）
            (0.5, 0.1, 0.9, 0.5, "右上"),     # 右上（包含右眼）
            (0.2, 0.3, 0.8, 0.7, "中央"),     # 中央（鼻子区域）
            (0.2, 0.6, 0.8, 0.95, "下部"),    # 下部（嘴部区域）
        ]

        all_features = []

        for x_start_r, y_start_r, x_end_r, y_end_r, region_name in regions:
            x_start = int(width * x_start_r)
            y_start = int(height * y_start_r)
            x_end = int(width * x_end_r)
            y_end = int(height * y_end_r)

            # LBP直方图
            lbp_hist = [0] * 256
            total_pixels = 0

            # 适当的采样密度
            step = max(1, (x_end - x_start) // 12)

            for y in range(y_start + radius, y_end - radius, step):
                for x in range(x_start + radius, x_end - radius, step):
                    try:
                        center_pixel = face_roi.get_pixel(x, y)
                        if isinstance(center_pixel, tuple):
                            center_pixel = sum(center_pixel) // len(center_pixel)

                        lbp_value = 0
                        valid_neighbors = 0

                        for i, (dx, dy) in enumerate(offsets):
                            nx, ny = x + dx, y + dy
                            if 0 <= nx < width and 0 <= ny < height:
                                neighbor_pixel = face_roi.get_pixel(nx, ny)
                                if isinstance(neighbor_pixel, tuple):
                                    neighbor_pixel = sum(neighbor_pixel) // len(neighbor_pixel)

                                if neighbor_pixel >= center_pixel:
                                    lbp_value |= (1 << i)
                                valid_neighbors += 1

                        if valid_neighbors == neighbors:
                            lbp_hist[lbp_value] += 1
                            total_pixels += 1

                    except:
                        pass

            # 选择更有区分性的LBP模式
            if total_pixels > 0:
                # 均匀模式 + 一些重要的非均匀模式
                important_patterns = [
                    0, 1, 3, 7, 15, 31, 63, 127, 255,  # 均匀模式
                    2, 4, 6, 8, 12, 14, 16, 24, 28, 30,  # 重要非均匀模式
                    32, 48, 56, 60, 62, 64, 96, 112, 120, 124
                ]

                region_features = []
                for pattern in important_patterns[:16]:  # 取前16个模式
                    normalized_value = (lbp_hist[pattern] * 100) // total_pixels  # 降低归一化系数
                    region_features.append(min(255, normalized_value))

                all_features.extend(region_features)
            else:
                all_features.extend([0] * 16)

        return all_features

    except Exception as e:
        print(f"LBP特征提取失败: {e}")
        return None

def extract_simple_features(face_roi):
    """简化但稳定的特征提取"""
    try:
        # 预处理
        face_roi = preprocess_face(face_roi)

        width = face_roi.width()
        height = face_roi.height()

        if width < 24 or height < 24:
            return None

        all_features = []

        # 1. LBP特征（主要）
        lbp_features = extract_stable_lbp_features(face_roi)
        if lbp_features:
            all_features.extend(lbp_features)

        # 2. 简化的网格统计特征
        grid_size = 6  # 降低网格大小
        cell_w = width // grid_size
        cell_h = height // grid_size

        for i in range(grid_size):
            for j in range(grid_size):
                x_start = i * cell_w
                y_start = j * cell_h
                x_end = min(x_start + cell_w, width)
                y_end = min(y_start + cell_h, height)

                # 计算该网格的平均灰度
                pixel_sum = 0
                pixel_count = 0

                for y in range(y_start, y_end, 2):
                    for x in range(x_start, x_end, 2):
                        try:
                            pixel = face_roi.get_pixel(x, y)
                            if isinstance(pixel, tuple):
                                pixel = sum(pixel) // len(pixel)
                            pixel_sum += pixel
                            pixel_count += 1
                        except:
                            pass

                if pixel_count > 0:
                    avg_gray = pixel_sum // pixel_count
                    all_features.append(avg_gray)
                else:
                    all_features.append(128)  # 默认中等灰度

        # 只取部分网格特征避免过拟合
        all_features = all_features[:80]  # LBP(64) + 部分网格特征(16)

        print(f"特征维度: {len(all_features)}")
        return all_features

    except Exception as e:
        print(f"特征提取失败: {e}")
        return None

def calculate_balanced_similarity(features1, features2):
    """平衡的相似度计算"""
    try:
        if not features1 or not features2 or len(features1) != len(features2):
            return 0.0

        n_features = len(features1)

        # 1. 计算归一化的欧氏距离
        squared_diff_sum = 0
        for i in range(n_features):
            diff = features1[i] - features2[i]
            squared_diff_sum += diff * diff

        euclidean_distance = math.sqrt(squared_diff_sum) / math.sqrt(n_features)

        # 2. 计算曼哈顿距离
        manhattan_distance = sum(abs(features1[i] - features2[i]) for i in range(n_features)) / n_features

        # 3. 计算相关系数
        mean1 = sum(features1) / n_features
        mean2 = sum(features2) / n_features

        numerator = sum((features1[i] - mean1) * (features2[i] - mean2) for i in range(n_features))

        sum_sq1 = sum((features1[i] - mean1) ** 2 for i in range(n_features))
        sum_sq2 = sum((features2[i] - mean2) ** 2 for i in range(n_features))

        correlation = 0
        if sum_sq1 > 0 and sum_sq2 > 0:
            correlation = numerator / math.sqrt(sum_sq1 * sum_sq2)

        # 4. 距离到相似度的转换（更温和）
        # 基于实际测试调整这些参数
        max_euclidean = 40    # 降低阈值
        max_manhattan = 60    # 降低阈值

        euclidean_sim = max(0, 1 - euclidean_distance / max_euclidean)
        manhattan_sim = max(0, 1 - manhattan_distance / max_manhattan)
        correlation_sim = (correlation + 1) / 2  # 将-1到1映射到0到1

        # 综合相似度
        final_similarity = (
            euclidean_sim * 0.4 +
            manhattan_sim * 0.3 +
            correlation_sim * 0.3
        )

        # 温和的非线性变换
        final_similarity = math.sqrt(final_similarity)  # 开方而不是立方

        return max(0.0, min(1.0, final_similarity))

    except Exception as e:
        print(f"相似度计算失败: {e}")
        return 0.0

# 内存优化：LCD显示函数
def safe_lcd_display(img, text_lines, face_rect=None, rect_color=(255, 255, 255)):
    """安全的LCD显示函数，避免内存泄漏"""
    try:
        # 直接在原图上绘制，避免复制
        y_offset = 5
        for text, color in text_lines:
            img.draw_string(5, y_offset, text, color=color, scale=2)
            y_offset += 20

        # 绘制人脸框
        if face_rect:
            img.draw_rectangle(face_rect, color=rect_color, thickness=2)

        # 显示到LCD
        lcd.display(img)

        # 立即清理
        gc.collect()

    except Exception as e:
        print(f"LCD显示异常: {e}")
        gc.collect()

def lcd_turn_off():
    """LCD熄屏 - 显示黑屏"""
    try:
        # 创建一个全黑的图像
        black_img = image.Image(sensor.width(), sensor.height(), sensor.RGB565)
        black_img.clear()  # 清空为黑色
        lcd.display(black_img)
        del black_img
        gc.collect()
        print("LCD屏幕已熄灭")
    except Exception as e:
        print(f"LCD熄屏失败: {e}")
        gc.collect()

def lcd_wake_up():
    """LCD唤醒 - 重新初始化"""
    try:
        # LCD重新初始化已经在主循环中处理
        print("LCD屏幕已唤醒")
    except Exception as e:
        print(f"LCD唤醒失败: {e}")
        gc.collect()

def turn_off_all_leds():
    """关闭所有LED"""
    red_led.off()
    green_led.off()
    blue_led.off()  # 添加蓝色LED关闭

def set_led_status(status):
    """设置LED状态
    status: 'success' - 绿灯, 'fail' - 红灯, 'uncertain' - 黄灯, 'blue' - 蓝灯, 'off' - 全部关闭
    """
    turn_off_all_leds()
    if status == 'success':
        green_led.on()
    elif status == 'fail':
        red_led.on()
    elif status == 'uncertain':
        red_led.on()
        green_led.on()
    elif status == 'blue':
        blue_led.on()
    # 'off' 状态已经通过 turn_off_all_leds() 处理

# === 录入阶段 ===
print("开始录入阶段，共拍摄6张照片：\n1. 保持正脸朝向镜头\n2. 保持静止\n3. 保持光线充足且均匀")
for user_id in range(num_users_to_enroll):
    username = "用户" + str(user_id + 1)
    print(f"\n开始录入 {username}")

    user_face_data = {
        "name": username,
        "faces": []
    }

    face_count = 0
    while face_count < faces_per_user:
        print(f"拍摄第 {face_count + 1} 张照片...")
        time.sleep(2)

        attempt_count = 0
        face_captured = False

        while not face_captured and attempt_count < 30:
            try:
                # 强制内存清理
                gc.collect()

                img = sensor.snapshot()

                # 优化：减少文本信息，直接在原图上绘制
                text_lines = [
                    (f"录入: {username}", (255, 255, 255)),
                    (f"照片 {face_count + 1}/{faces_per_user}", (255, 255, 255))
                ]

                faces = img.find_features(face_cascade, threshold=0.5, scale_factor=1.25)
                attempt_count += 1

                if faces:
                    largest_face = max(faces, key=lambda f: f[2] * f[3])
                    x, y, w, h = largest_face

                    if w >= 24 and h >= 24:
                        print(f"检测到人脸: {w}x{h}")

                        face_roi = img.copy(roi=(x, y, w, h))
                        features = extract_simple_features(face_roi)

                        if features and len(features) > 40:
                            user_face_data["faces"].append(features)
                            print(f"第 {face_count + 1} 张照片保存成功! 特征数: {len(features)}")

                            # 成功显示
                            text_lines.append(("照片已保存!", (0, 255, 0)))
                            safe_lcd_display(img, text_lines, largest_face, (0, 255, 0))

                            face_captured = True
                            face_count += 1
                        else:
                            print("特征提取失败或特征不足")
                            text_lines.append(("特征提取失败", (255, 0, 0)))
                            safe_lcd_display(img, text_lines, largest_face, (255, 0, 0))

                        del face_roi
                        gc.collect()
                    else:
                        print(f"人脸过小: {w}x{h}")
                        text_lines.append(("人脸过小", (255, 0, 0)))
                        safe_lcd_display(img, text_lines, largest_face, (255, 0, 0))
                else:
                    if attempt_count % 5 == 0:
                        print(f"尝试 {attempt_count}: 未检测到人脸")
                    text_lines.append(("请面向摄像头", (255, 255, 0)))
                    safe_lcd_display(img, text_lines)

                # 立即删除图像对象
                del img
                gc.collect()

            except Exception as e:
                print(f"录入异常: {e}")
                gc.collect()

            time.sleep_ms(100)

    user_faces.append(user_face_data)
    print(f"{username} 录入完成! 共 {len(user_face_data['faces'])} 张照片")

    # 如果是第一个用户录入完成，点亮蓝LED并保持2秒
    if user_id == 0:  # 第一个用户 (索引为0)
        print("第一个用户录入完成，点亮蓝色LED")
        set_led_status('blue')
        time.sleep(2)  # 保持2秒
        print("蓝色LED熄灭，准备录入下一个用户")
        set_led_status('off')

    gc.collect()

print(f"\n录入阶段完成! 共录入 {len(user_faces)} 位用户")

# 全部用户录入完成，点亮蓝LED并保持2秒
print("全部用户录入完成，点亮蓝色LED")
set_led_status('blue')
time.sleep(2)  # 保持2秒
print("蓝色LED熄灭，开始计算识别基线")
set_led_status('off')

# 计算用户内部相似度基线
print("\n计算识别基线...")
intra_user_similarities = []

for user in user_faces:
    user_similarities = []
    faces = user["faces"]

    for i in range(len(faces)):
        for j in range(i + 1, len(faces)):
            sim = calculate_balanced_similarity(faces[i], faces[j])
            user_similarities.append(sim)

    if user_similarities:
        avg_sim = sum(user_similarities) / len(user_similarities)
        min_sim = min(user_similarities)
        max_sim = max(user_similarities)

        print(f"{user['name']} 内部相似度: 平均={avg_sim:.3f}, 范围=[{min_sim:.3f}, {max_sim:.3f}]")
        intra_user_similarities.extend(user_similarities)

if intra_user_similarities:
    baseline_similarity = sum(intra_user_similarities) / len(intra_user_similarities)
    min_baseline = min(intra_user_similarities)
    std_dev = 0
    if len(intra_user_similarities) > 1:
        variance = sum((s - baseline_similarity) ** 2 for s in intra_user_similarities) / len(intra_user_similarities)
        std_dev = math.sqrt(variance)

    print(f"系统基线: 平均={baseline_similarity:.3f}, 最低={min_baseline:.3f}, 标准差={std_dev:.3f}")

    # 保守的阈值设置
    recognition_threshold = 0.9  # 比最低内部相似度低一些
    reject_threshold = 0.89  # 拒绝阈值

    print(f"识别阈值: {recognition_threshold:.3f}")
    print(f"拒绝阈值: {reject_threshold:.3f}")
else:
    recognition_threshold = 0.9
    reject_threshold = 0.89
    print(f"使用默认阈值: 识别={recognition_threshold}, 拒绝={reject_threshold}")

time.sleep(2)

# === 识别阶段 ===
print("\n开始识别阶段：\n1. 保持正脸朝向镜头\n2. 保持静止\n3. 保持光线充足且均匀")
print("使用平衡的特征提取和相似度计算")
print(f"识别阈值: {recognition_threshold:.3f}, 拒绝阈值: {reject_threshold:.3f}")
print("LED控制: 10s后自动熄灭, LCD: 15s后关闭显示")
print("-" * 50)

recognition_count = 0
last_recognition_time = 0
last_face_detected_time = time.ticks_ms()  # 记录最后一次检测到人脸的时间
led_status_time = 0  # LED状态设置时间
current_led_status = 'off'  # 当前LED状态
lcd_active = True  # LCD是否激活
clock = time.clock()  # 添加FPS计算
lcd_update_counter = 0  # LCD更新计数器，降低更新频率

while True:
    try:
        clock.tick()
        current_time = time.ticks_ms()

        # 每隔几帧强制清理一次内存
        if lcd_update_counter % 10 == 0:
            gc.collect()

        # 检查LED状态 - 10秒后熄灭
        if current_led_status != 'off' and time.ticks_diff(current_time, led_status_time) > 10000:
            print("LED 10秒后自动熄灭")
            set_led_status('off')
            current_led_status = 'off'

        # 检查LCD状态 - 15秒后关闭
        if lcd_active and time.ticks_diff(current_time, last_face_detected_time) > 15000:
            print("LCD 15秒后关闭显示")
            lcd_turn_off()  # 熄灭LCD屏幕
            lcd_active = False

        img = sensor.snapshot()

        # 准备显示文本
        text_lines = [
            (f"FPS: {clock.fps():.1f}", (255, 255, 255)),
            ("人脸识别系统", (255, 255, 255))
        ]

        faces = img.find_features(face_cascade, threshold=0.5, scale_factor=1.25)

        if faces:
            # 检测到人脸，更新时间戳
            last_face_detected_time = current_time

            # 如果LCD关闭了，重新开启
            if not lcd_active:
                print("检测到人脸，重新激活LCD")
                lcd_wake_up()
                lcd_active = True

            largest_face = max(faces, key=lambda f: f[2] * f[3])
            x, y, w, h = largest_face

            if current_time - last_recognition_time > 3000:

                if w >= 24 and h >= 24:
                    print(f"\n[{recognition_count + 1}] 检测到人脸: {w}x{h}")
                    text_lines.append(("正在识别...", (255, 255, 0)))

                    # 立即显示识别状态
                    if lcd_active:
                        safe_lcd_display(img, text_lines, largest_face, (255, 0, 0))

                    current_face_roi = img.copy(roi=(x, y, w, h))
                    current_features = extract_simple_features(current_face_roi)

                    if current_features:
                        print("正在进行特征匹配...")

                        best_match = None
                        best_score = 0
                        all_results = []

                        for user in user_faces:
                            user_scores = []

                            for face_features in user["faces"]:
                                similarity = calculate_balanced_similarity(current_features, face_features)
                                user_scores.append(similarity)

                            if user_scores:
                                avg_score = sum(user_scores) / len(user_scores)
                                max_score = max(user_scores)

                                all_results.append({
                                    "name": user['name'],
                                    "avg_score": avg_score,
                                    "max_score": max_score,
                                    "scores": user_scores
                                })

                                print(f"  {user['name']}: 平均={avg_score:.3f}, 最高={max_score:.3f}")

                                if avg_score > best_score:
                                    best_score = avg_score
                                    best_match = user["name"]

                        print(f"\n=== 识别结果 {recognition_count + 1} ===")

                        # 更新显示文本（移除FPS显示以节省内存）
                        text_lines = [("人脸识别系统", (255, 255, 255))]

                        # 分层判断
                        if best_score >= recognition_threshold:
                            # 进一步验证：检查一致性
                            best_result = next(r for r in all_results if r["name"] == best_match)
                            high_scores = [s for s in best_result["scores"] if s > reject_threshold]
                            consistency = len(high_scores) / len(best_result["scores"])

                            if consistency >= 0.5:  # 至少50%的样本超过拒绝阈值
                                print(f"✅ 身份验证成功!")
                                print(f"识别用户: {best_match}")
                                print(f"置信度: {best_score:.3f}")
                                print(f"一致性: {consistency:.2f}")
                                print(f"🔓 访问授权!")

                                # 设置成功LED状态
                                set_led_status('success')
                                current_led_status = 'success'
                                led_status_time = current_time

                                # 成功识别显示
                                text_lines.extend([
                                    (f"欢迎 {best_match}!", (0, 255, 0)),
                                    ("访问已授权", (0, 255, 0))
                                ])
                                if lcd_active:
                                    safe_lcd_display(img, text_lines, largest_face, (0, 255, 0))

                                # 发送UART消息 - 仅在人脸识别成功时发送
                                try:
                                    str_buffer = "Hello World"
                                    uart.write(str_buffer)
                                    print("UART消息已发送: Hello World")
                                except Exception as uart_error:
                                    print(f"UART发送失败: {uart_error}")

                            else:
                                print(f"⚠️ 识别结果不稳定")
                                print(f"最相似: {best_match} (置信度: {best_score:.3f})")
                                print(f"一致性不足: {consistency:.2f} < 0.5")
                                print("🔒 拒绝访问 - 结果不稳定")

                                # 设置不确定LED状态
                                set_led_status('uncertain')
                                current_led_status = 'uncertain'
                                led_status_time = current_time

                                # 不稳定显示
                                text_lines.extend([
                                    ("识别不稳定", (255, 255, 0)),
                                    ("请重新尝试", (255, 255, 0))
                                ])
                                if lcd_active:
                                    safe_lcd_display(img, text_lines, largest_face, (255, 255, 0))

                        elif best_score >= reject_threshold:
                            print(f"⚠️ 可能是已知用户但置信度不足")
                            print(f"最相似: {best_match} (置信度: {best_score:.3f})")
                            print(f"需要重新尝试或改善拍摄条件")
                            print("🔒 临时拒绝访问")

                            # 设置不确定LED状态
                            set_led_status('uncertain')
                            current_led_status = 'uncertain'
                            led_status_time = current_time

                            # 置信度不足显示
                            text_lines.extend([
                                ("置信度不足", (255, 255, 0)),
                                ("请重新尝试", (255, 255, 0))
                            ])
                            if lcd_active:
                                safe_lcd_display(img, text_lines, largest_face, (255, 255, 0))

                        else:
                            print(f"❌ 未识别出已知用户")
                            if best_match:
                                print(f"最相似: {best_match} (置信度: {best_score:.3f})")
                            print("🔒 拒绝访问 - 未授权人员")

                            # 设置失败LED状态
                            set_led_status('fail')
                            current_led_status = 'fail'
                            led_status_time = current_time

                            # 拒绝访问显示
                            text_lines.extend([
                                ("访问被拒绝", (255, 0, 0)),
                                ("未授权人员", (255, 0, 0))
                            ])
                            if lcd_active:
                                safe_lcd_display(img, text_lines, largest_face, (255, 0, 0))

                        recognition_count += 1
                        last_recognition_time = current_time

                        print("-" * 50)
                    else:
                        print("⚠️ 特征提取失败")
                        text_lines.append(("特征提取失败", (255, 0, 0)))
                        if lcd_active:
                            safe_lcd_display(img, text_lines, largest_face, (255, 0, 0))

                    del current_face_roi
                    gc.collect()
                else:
                    print(f"⚠️ 人脸尺寸不足: {w}x{h}")
                    text_lines.append(("人脸过小", (255, 0, 0)))
                    if lcd_active:
                        safe_lcd_display(img, text_lines, largest_face, (255, 0, 0))
            else:
                # 等待状态 - 降低更新频率
                if lcd_active and lcd_update_counter % 5 == 0:  # 每5帧更新一次
                    text_lines.append(("请稍候...", (255, 255, 255)))
                    safe_lcd_display(img, text_lines, largest_face, (255, 0, 0))
        else:
            # 未检测到人脸
            # 检查是否需要关闭LED（超过10秒）
            if current_led_status != 'off' and time.ticks_diff(current_time, last_face_detected_time) > 10000:
                if current_led_status != 'off':  # 避免重复输出
                    print("10秒未检测到人脸，关闭LED")
                    set_led_status('off')
                    current_led_status = 'off'

            # 等待检测人脸 - 降低更新频率
            if lcd_active and lcd_update_counter % 3 == 0:  # 每3帧更新一次
                text_lines.append(("等待检测人脸", (255, 255, 255)))
                safe_lcd_display(img, text_lines)

        # 立即删除图像对象
        del img
        gc.collect()

        lcd_update_counter += 1

    except KeyboardInterrupt:
        print("\n程序终止")
        break
    except Exception as e:
        print(f"识别异常: {e}")
        gc.collect()

    time.sleep_ms(50)

print("\n人脸识别系统关闭")
# 清理资源
turn_off_all_leds()
if lcd_active:
    lcd_turn_off()
gc.collect()
