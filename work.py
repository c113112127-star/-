import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ============================================================
# 1️⃣ 影像載入與前處理
# ============================================================

image_path = "coin_img.jpg"  # ← 改成你的圖片名稱
image = cv2.imread(image_path)

if image is None:
    print(f"錯誤：無法讀取圖片 {image_path}")
    exit()

output = image.copy()
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (11, 11), 2)

# ============================================================
# 2️⃣ 使用霍夫圓變換偵測硬幣
# ============================================================

circles = cv2.HoughCircles(
    blurred,
    cv2.HOUGH_GRADIENT,
    dp=1,
    minDist=60,
    param1=100,
    param2=30,
    minRadius=25,
    maxRadius=90
)

# 初始化統計資料
coin_counts = {'1元': 0, '5元': 0, '10元': 0, '50元': 0}
total_value = 0

# ============================================================
# 3️⃣ 依直徑分類幣別 + 顏色區分
# ============================================================

if circles is not None:
    circles = np.round(circles[0, :]).astype("int")
    all_coin_data = []

    print("--- 偵測到的硬幣原始數據 ---")

    for (x, y, r) in circles:
        diameter = r * 2
        all_coin_data.append({'x': x, 'y': y, 'r': r, 'd': diameter, 'denomination': '未知'})

    # 排序方便觀察
    all_coin_data.sort(key=lambda c: c['d'])
    diameters = [coin['d'] for coin in all_coin_data]
    print(f"所有直徑: {diameters}")

    # 🎨 每種硬幣使用不同顏色（更柔和）
    color_map = {
        '1元': (255, 120, 120),   # 淡紅
        '5元': (255, 230, 120),   # 金黃
        '10元': (150, 150, 255),  # 淡紫
        '50元': (120, 255, 120)   # 淡綠
    }

    # 🎯 根據直徑分類幣別
    for coin in all_coin_data:
        d = coin['d']

        if d < 130:
            coin['denomination'] = '1元'
            coin_counts['1元'] += 1
            total_value += 1
        elif d < 140:
            coin['denomination'] = '5元'
            coin_counts['5元'] += 1
            total_value += 5
        elif d < 165:
            coin['denomination'] = '10元'
            coin_counts['10元'] += 1
            total_value += 10
        else:
            coin['denomination'] = '50元'
            coin_counts['50元'] += 1
            total_value += 50

    print("--- 分類結束 ---")

    # ============================================================
    # 4️⃣ 畫出不同顏色的圓框 + 中文文字
    # ============================================================

    font_path = "C:/Windows/Fonts/msjh.ttc"  # 微軟正黑體
    try:
        font_large = ImageFont.truetype(font_path, 42)
        font_medium = ImageFont.truetype(font_path, 35)
        font_small = ImageFont.truetype(font_path, 28)
    except IOError:
        print(f"⚠️ 找不到字型檔案 {font_path}，改用預設字型。")
        font_large = ImageFont.load_default()
        font_medium = ImageFont.load_default()
        font_small = ImageFont.load_default()

    # 先用 OpenCV 畫彩色圓框
    for coin in all_coin_data:
        x, y, r = coin['x'], coin['y'], coin['r']
        denom = coin['denomination']
        color = color_map[denom]
        cv2.circle(output, (x, y), r, color, 4)  # 不同幣別不同顏色
        cv2.circle(output, (x, y), 2, (255, 255, 255), 3)  # 中心點

    # ============================================================
    # 5️⃣ 用 PIL 標文字（有半透明背景）
    # ============================================================

    output_pil = Image.fromarray(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(output_pil, "RGBA")

    for coin in all_coin_data:
        x, y, r = coin['x'], coin['y'], coin['r']
        denom = coin['denomination']

        # 加底色框
        text = denom
        text_w = font_small.getlength(text)
        text_h = font_small.getbbox(text)[3]
        bg_color = (0, 0, 0, 120)  # 黑色半透明底
        draw.rounded_rectangle(
            [(x - text_w / 2 - 10, y - r - text_h - 10),
             (x + text_w / 2 + 10, y - r)],
            radius=8, fill=bg_color
        )
        draw.text((x - text_w / 2, y - r - text_h - 5),
                  text, fill=(255, 255, 255), font=font_small)

    # ============================================================
    # 6️⃣ 顯示右上角統計資訊框
    # ============================================================

    info_x, info_y = output_pil.size[0] - 330, 20
    draw.rounded_rectangle(
        [(info_x, info_y), (info_x + 310, info_y + 250)],
        radius=20, fill=(30, 30, 30, 150)
    )

    draw.text((info_x + 20, info_y + 20),
              f"總硬幣數: {len(all_coin_data)} 枚",
              fill=(255, 255, 255), font=font_medium)
    draw.text((info_x + 20, info_y + 70),
              f"總金額: ${total_value} 元",
              fill=(255, 255, 0), font=font_medium)

    y_offset = info_y + 120
    for denom, count in coin_counts.items():
        color = color_map[denom]
        draw.text((info_x + 25, y_offset),
                  f"{denom}: {count} 枚",
                  fill=color, font=font_small)
        y_offset += 35

    # 轉回 BGR 顯示
    output = cv2.cvtColor(np.array(output_pil), cv2.COLOR_RGB2BGR)

else:
    print("沒有偵測到任何硬幣！")

# ============================================================
# 7️⃣ 顯示與儲存結果
# ============================================================

output_filename = "detected_coins_pretty.jpg"
cv2.imwrite(output_filename, output)
print(f"辨識完成，結果已儲存為 '{output_filename}'")

cv2.imshow("Coin Detection (Enhanced)", output)
cv2.waitKey(0)
cv2.destroyAllWindows()
