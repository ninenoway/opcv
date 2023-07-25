import cv2
import numpy as np

def find_object_with_sift_and_template_matching(original_image_path, template_path):
    # 读取原始图像和目标模板
    original_image = cv2.imread(original_image_path)
    template = cv2.imread(template_path)

    # 转换为灰度图像
    gray_original = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    gray_template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

    # 使用SIFT特征点检测
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(gray_original, None)
    kp2, des2 = sift.detectAndCompute(gray_template, None)

    # 使用FLANN匹配器进行特征点匹配
    index_params = dict(algorithm=0, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    # 根据Lowe's比率测试选择匹配的特征点
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

    if len(good_matches) > 10:
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
    find_object_with_sift_and_template_matching(original_image_path, template_path)
