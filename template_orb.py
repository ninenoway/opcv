import cv2
import numpy as np

def find_object_with_orb_and_template_matching(original_image_path, template_path):
    # 读取原始图像和目标模板
    original_image = cv2.imread(original_image_path)
    template = cv2.imread(template_path)

    # 转换为灰度图像
    gray_original = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    gray_template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

    # 使用ORB特征点检测
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(gray_original, None)
    kp2, des2 = orb.detectAndCompute(gray_template, None)

    # 使用Brute-Force匹配器进行特征点匹配
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)

    # 根据匹配距离进行筛选
    matches = sorted(matches, key=lambda x: x.distance)
    good_matches = matches[:10]

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # 计算透视变换矩阵并执行透视变换
    M, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
    h, w = gray_template.shape
    transformed_corners = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
    transformed_corners = cv2.perspectiveTransform(transformed_corners, M)

    # 绘制矩形框标记目标位置
    original_image = cv2.polylines(original_image, [np.int32(transformed_corners)], True, (0, 255, 0), 2)

    # 显示结果图像
    cv2.imshow('Result', original_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    original_image_path = './test_photo/44.jpg'
    template_path = './template_photo/4.jpg'
    find_object_with_orb_and_template_matching(original_image_path, template_path)
