import cv2
import os

# تنظیمات اولیه
output_dir = "dataset"  # پوشه‌ای که تصاویر در آن ذخیره می‌شوند
user_id = 1  # شناسه کاربر
image_number = 149  # شماره تصویر اولیه

# ایجاد پوشه خروجی در صورت عدم وجود
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# باز کردن دوربین
cap = cv2.VideoCapture(0)
print("[INFO] دوربین روشن شد. برای گرفتن عکس، کلید Space را فشار دهید. برای خروج، کلید q را فشار دهید.")

while True:
    ret, frame = cap.read()  # گرفتن تصویر از دوربین
    frame = cv2.flip(frame, 1)
    if not ret:
        print("[ERROR] خطا در دسترسی به دوربین.")
        break

    # نمایش تصویر زنده
    cv2.imshow("Camera", frame)

    # خواندن کلیدها
    key = cv2.waitKey(1) & 0xFF

    if key == ord(' '):  # اگر Space فشار داده شود
        # نام‌گذاری فایل
        file_name = f"user.{user_id}.{image_number}.jpg"
        file_path = os.path.join(output_dir, file_name)

        # ذخیره تصویر
        cv2.imwrite(file_path, frame)
        print(f"[INFO] تصویر ذخیره شد: {file_name}")
        image_number += 1  # شماره تصویر را افزایش می‌دهیم

    elif key == ord('q'):  # اگر q فشار داده شود
        print("[INFO] خروج از برنامه.")
        break

# آزاد کردن دوربین و بستن پنجره‌ها
cap.release()
cv2.destroyAllWindows()
