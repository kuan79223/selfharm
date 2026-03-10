import cv2


class DrawHumans(object):

    # def draw_skeleton(image, humans, black=False):
    #     # 定義關節連線
    #     CocoPairs = [
    #         (1, 2), (1, 5), (2, 3), (3, 4), (5, 6), (6, 7), (1, 8), (8, 9), (9, 10), (1, 11),
    #         (11, 12), (12, 13), (1, 0), (0, 14), (14, 16), (0, 15), (15, 17), (2, 16), (5, 17)
    #     ]
    #     # 去除最後兩個連線，讓骨幹外支
    #     cocoPairsRender = CocoPairs[:-2]
    #     human_color = (0, 255, 0)

    #     if black:
    #         height = image.shape[0]
    #         width  = image.shape[1]
    #         cv2.rectangle(image, (0, 0), (width, height), (0,0,0), -1)

    #     for human in humans:
    #         centers = {}

    #         # Draw point
    #         for i in range(18):
    #             if i == 1:  # 排除繪製 neck
    #                 continue
    #             body_part = human[i]
    #             center = (int(body_part[0]), int(body_part[1]))
    #             if center != (0, 0):
    #                 centers[i] = center
    #                 cv2.circle(image, center, 1, human_color, 1)
    #                 #cv2.putText(image, f'{i}', center, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    #         # Draw line
    #         for pair_order, pair in enumerate(cocoPairsRender):

    #             if 1 in pair:  # 如果這條線包含 neck (1)，就不畫
    #                 continue
    #             if pair[0] not in centers.keys() or pair[1] not in centers.keys():
    #                 continue

    #             cv2.line(image, centers[pair[0]], centers[pair[1]], (0, 255, 0),1)

    #     return image

    def draw_skeleton(image, humans, black=False):

        YOLO_PAIRS = [
            (0, 1), (1, 3),  # Nose → Left Eye → Left Ear
            (0, 2), (2, 4),  # Nose → Right Eye → Right Ear
            (3, 5),  # Left Ear → Left Shoulder
            (4, 6),  # Right Ear → Right Shoulder
            (5, 7), (7, 9),  # Left Shoulder → Left Elbow → Left Wrist
            (6, 8), (8, 10),  # Right Shoulder → Right Elbow → Right Wrist
            (5, 6),  # Left Shoulder → Right Shoulder
            (5, 11), (6, 12),  # Left Shoulder → Left Hip, Right Shoulder → Right Hip
            (11, 13), (13, 15),  # Left Hip → Left Knee → Left Ankle
            (12, 14), (14, 16),  # Right Hip → Right Knee → Right Ankle
            (11, 12)  # Left Hip → Right Hip
        ]

        human_color = (0, 255, 0)  # 綠色骨架

        if black:
            height = image.shape[0]
            width = image.shape[1]
            cv2.rectangle(image, (0, 0), (width, height), (0, 0, 0), -1)

        for human in humans:
            centers = {}
            # 繪製關節點
            for i in range(17):  # YOLO-Pose 共有 17 個關鍵點
                body_part = human[i]
                center = (int(body_part[0]), int(body_part[1]))
                if center != (0, 0):  # 避免繪製無效點
                    centers[i] = center
                    cv2.circle(image, center, 1, human_color, 1)  # 繪製關節點

            # 繪製關節連線
            for pair in YOLO_PAIRS:
                if pair[0] in centers and pair[1] in centers:  # 只畫存在的關節
                    cv2.line(image, centers[pair[0]], centers[pair[1]], (0, 255, 0), 1)

        return image

    '''
    def draw_one_box(image, humans, black=False):
        xmin, ymin = (output[idx, 2]-output[idx, 4]/2), (output[idx, 3]-output[idx, 5]/2)
        xmax, ymax = (output[idx, 2]+output[idx, 4]/2), (output[idx, 3]+output[idx, 5]/2)

        Plotting key points on Image
        cv2.rectangle(image, (int(xmin), int(ymin)),(int(xmax), int(ymax)),color=(255, 0, 0), thickness=1)
    '''

# 計算手肘角度 (左手: 7, 右手: 8) ------------------------------------------
# if 5 in centers and 7 in centers and 9 in centers:
#     left_angle = getAngle([centers[5], centers[7], centers[9]])
#     cv2.putText(image, f'{int(left_angle)}°', centers[7],
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.2, (255, 0, 0), 1)

# if 6 in centers and 8 in centers and 10 in centers:
#     right_angle = getAngle([centers[6], centers[8], centers[10]])
#     cv2.putText(image, f'{int(right_angle)}°', centers[8],
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.2, (255, 0, 0), 1)