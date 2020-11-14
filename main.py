import cv2
import numpy as np
import argparse


def show_histogram(img):
    bgr_planes = cv2.split(img)

    hist_size = 256
    hist_range = (0, 256)

    b_hist = cv2.calcHist(bgr_planes, [0], None, [hist_size], hist_range, accumulate=False)
    g_hist = cv2.calcHist(bgr_planes, [1], None, [hist_size], hist_range, accumulate=False)
    r_hist = cv2.calcHist(bgr_planes, [2], None, [hist_size], hist_range, accumulate=False)

    hist_w = 512
    hist_h = 256

    b_hist = np.round(b_hist / max(b_hist) * hist_h)
    g_hist = np.round(g_hist / max(g_hist) * hist_h)
    r_hist = np.round(r_hist / max(r_hist) * hist_h)

    scale_w = round(hist_w / hist_size)
    hist_image = np.zeros((hist_h, hist_w, 3), dtype=np.uint8)
    thickness = 2

    for i in range(1, hist_size):
        cv2.line(hist_image, (scale_w * (i - 1), hist_h - int(np.round(b_hist[i - 1]))),
                 (scale_w * (i), hist_h - int(np.round(b_hist[i]))),
                 (255, 0, 0), thickness=thickness)
        cv2.line(hist_image, (scale_w * (i - 1), hist_h - int(np.round(g_hist[i - 1]))),
                 (scale_w * (i), hist_h - int(np.round(g_hist[i]))),
                 (0, 255, 0), thickness=thickness)
        cv2.line(hist_image, (scale_w * (i - 1), hist_h - int(np.round(r_hist[i - 1]))),
                 (scale_w * (i), hist_h - int(np.round(r_hist[i]))),
                 (0, 0, 255), thickness=thickness)

    cv2.imshow('calcHist Demo', hist_image)


def show_pixel_info(img, x, y):
    pixel_info_board = np.zeros(shape=(150, 200, 3), dtype=np.uint8)

    pixel = img[y][x].reshape(1, 1, 3)
    rgb = cv2.cvtColor(pixel, cv2.COLOR_BGR2RGB)[0][0]
    hsv = cv2.cvtColor(pixel, cv2.COLOR_BGR2HSV)[0][0]

    intensity = cv2.cvtColor(pixel, cv2.COLOR_BGR2GRAY)[0][0]

    text = f'pixel state:\n' \
           f'RGB: {rgb}\n' \
           f'HSV: {hsv}\n' \
           f'intensity: {intensity}'
    # print(text)

    color = tuple(int(i) for i in pixel.reshape(3))
    cv2.rectangle(pixel_info_board, (80, 0), (120, 40), color, -1)

    for y, line in enumerate(text.split('\n')):
        cv2.putText(pixel_info_board, line, (10, y * 20 + 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.imshow("pixel info", pixel_info_board)


def crop_square(img, x, y):
    x1 = max(x - 12, 0)
    y1 = max(y - 12, 0)
    x2 = max(x + 12, 0)
    y2 = max(y + 12, 0)
    square = img[y1:(y2+1), x1:(x2+1)]

    b_mean = np.mean(square[:, :, 0])
    g_mean = np.mean(square[:, :, 1])
    r_mean = np.mean(square[:, :, 2])

    b_std = np.std(square[:, :, 0])
    g_std = np.std(square[:, :, 1])
    r_std = np.std(square[:, :, 2])

    text = f'mean of red: {r_mean}\n' \
           f'mean of green: {g_mean}\n' \
           f'mean of blue: {b_mean}\n' \
           f'std of red: {r_std}\n' \
           f'std of green: {g_std}\n' \
           f'std of blue: {b_std}'
    # print(text)

    square = cv2.resize(square, (square.shape[0]*10, square.shape[1]*10), cv2.INTER_CUBIC)
    info = np.zeros(shape=(100, square.shape[1], 3), dtype = square.dtype)

    for i,txt in enumerate(text.split('\n')):
        cv2.putText(info, txt, (5, i*15+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    square = np.concatenate((square, info), axis=0)
    cv2.imshow("cropped square", square)


def mouse_event(event, x, y, flags, img):
    if event == cv2.EVENT_MOUSEMOVE:
        show_pixel_info(img, x, y)
    if event == cv2.EVENT_LBUTTONDOWN:
        crop_square(img, x, y)


def main():
    desc = 'this program take input image and show:\n' \
           'source image,\n' \
           'color histogram,\n' \
           'pixel info by hover mouse,\n' \
           'crop the square 25x25 by LBTN and show mean, std color info'

    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--src', help='Path to input image', default='test_img.jpg')
    args = parser.parse_args()

    img = cv2.imread(args.src)
    if img is None:
        print('Could not open or find the image:', args.input)
        exit(0)
    print('use ESC to exit')
    cv2.imshow("source", img)
    show_histogram(img)

    cx = int(img.shape[1] / 2)
    cy = int(img.shape[0] / 2)

    show_pixel_info(img, cx, cy)
    crop_square(img, cx, cy)
    cv2.setMouseCallback("source", mouse_event, img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
