import cv2
import numpy as np

def sift_feature_matching(img1, img2):
    # 创建SIFT对象
    sift = cv2.SIFT_create()

    # 在两张图片中分别检测特征点和计算特征描述子
    keypoints1, descriptors1 = sift.detectAndCompute(img1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(img2, None)

    # 使用FLANN匹配器
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # 使用KNN算法进行匹配
    matches = flann.knnMatch(descriptors1, descriptors2, k=2)

    # 仅保留好的匹配点
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

    # 绘制匹配结果
    img_matches = cv2.drawMatches(img1, keypoints1, img2, keypoints2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    return img_matches

def surf_feature_matching(img1, img2):
    # 创建SURF对象
    surf = cv2.xfeatures2d.SURF_create()

    # 在两张图片中分别检测特征点和计算特征描述子
    keypoints1, descriptors1 = surf.detectAndCompute(img1, None)
    keypoints2, descriptors2 = surf.detectAndCompute(img2, None)

    # 使用FLANN匹配器
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # 使用KNN算法进行匹配
    matches = flann.knnMatch(descriptors1, descriptors2, k=2)

    # 仅保留好的匹配点
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

    # 绘制匹配结果
    img_matches = cv2.drawMatches(img1, keypoints1, img2, keypoints2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    return img_matches

def orb_feature_matching(img1, img2):
    # 创建ORB对象
    orb = cv2.ORB_create()

    # 在两张图片中分别检测特征点和计算特征描述子
    keypoints1, descriptors1 = orb.detectAndCompute(img1, None)
    keypoints2, descriptors2 = orb.detectAndCompute(img2, None)

    # 将描述子转换为float32类型
    descriptors1 = descriptors1.astype(np.float32)
    descriptors2 = descriptors2.astype(np.float32)

    # 使用FLANN匹配器
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # 使用KNN算法进行匹配
    matches = flann.knnMatch(descriptors1, descriptors2, k=2)

    # 仅保留好的匹配点
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

    # 绘制匹配结果
    img_matches = cv2.drawMatches(img1, keypoints1, img2, keypoints2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    return img_matches
    # 创建ORB对象
    orb = cv2.ORB_create()

    # 在两张图片中分别检测特征点和计算特征描述子
    keypoints1, descriptors1 = orb.detectAndCompute(img1, None)
    keypoints2, descriptors2 = orb.detectAndCompute(img2, None)

    # 使用FLANN匹配器
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # 使用KNN算法进行匹配
    matches = flann.knnMatch(descriptors1, descriptors2, k=2)

    # 仅保留好的匹配点
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

    # 绘制匹配结果
    img_matches = cv2.drawMatches(img1, keypoints1, img2, keypoints2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    return img_matches

def main():
    img1 = cv2.imread('07.png', cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread('template.png', cv2.IMREAD_GRAYSCALE)

    sift_matches = sift_feature_matching(img1, img2)
    cv2.imshow('SIFT Matches', sift_matches)

    #surf_matches = surf_feature_matching(img1, img2)
    #cv2.imshow('SURF Matches', surf_matches)

    orb_matches = orb_feature_matching(img1, img2)
    cv2.imshow('ORB Matches', orb_matches)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
