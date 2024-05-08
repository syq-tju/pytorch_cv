import cv2
import numpy as np

def multi_template_matching(img_gray, templates):
    matches = []
    for template in templates:
        result = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)
        matches.append((max_val, max_loc, template.shape[::-1]))
    return matches

def main():
    img_gray = cv2.imread("07.png", cv2.IMREAD_GRAYSCALE)
    template1 = cv2.imread("template1.png", cv2.IMREAD_GRAYSCALE)
    template2 = cv2.imread("template2.png", cv2.IMREAD_GRAYSCALE)
    template3 = cv2.imread("template3.png", cv2.IMREAD_GRAYSCALE)
    
    templates = [template1, template2, template3]

    matches = multi_template_matching(img_gray, templates)

    best_match = max(matches, key=lambda x: x[0])  # 找到最好的匹配

    max_val, max_loc, (w, h) = best_match
    top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)

    # 画出匹配的矩形框
    img_matched = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
    cv2.rectangle(img_matched, top_left, bottom_right, (0, 255, 0), 2)

    cv2.imshow("Matched", img_matched)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
