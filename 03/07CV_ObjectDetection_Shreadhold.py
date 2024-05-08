import cv2
import numpy as np

def multi_scale_template_matching(img_gray, template_gray, scale_factors):
    matches = []
    for scale in scale_factors:
        scaled_template = cv2.resize(template_gray, None, fx=scale, fy=scale)
        result = cv2.matchTemplate(img_gray, scaled_template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)
        matches.append((max_val, max_loc, scaled_template.shape))
    return matches

def main():
    img_gray = cv2.imread("07.png", cv2.IMREAD_GRAYSCALE)
    template_gray = cv2.imread("template.png", cv2.IMREAD_GRAYSCALE)

    scale_factors = [0.5, 0.75, 1.0, 1.25, 1.5]  # 尝试不同的尺度因子

    matches = multi_scale_template_matching(img_gray, template_gray, scale_factors)

    # 设置阈值，只保留满足阈值条件的匹配结果
    threshold = 0.8
    matches_above_threshold = [match for match in matches if match[0] > threshold]

    if matches_above_threshold:
        best_match = max(matches_above_threshold, key=lambda x: x[0])  # 找到最好的匹配
        max_val, max_loc, (w, h) = best_match
        top_left = max_loc
        bottom_right = (top_left[0] + w, top_left[1] + h)

        # 画出匹配的矩形框
        img_matched = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
        cv2.rectangle(img_matched, top_left, bottom_right, (0, 255, 0), 2)

        cv2.imshow("Matched", img_matched)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("No match found above threshold.")

if __name__ == "__main__":
    main()
