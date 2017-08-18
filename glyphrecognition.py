import cv2
from glyphfunctions import *
from webcam import Webcam

webcam = Webcam()
webcam.start()

QUADRILATERAL_POINTS = 4
SHAPE_RESIZE = 100.0
BLACK_THRESHOLD = 100
WHITE_THRESHOLD = 155
BOY_PATTERN = [1, 1, 0, 1, 0, 1, 0, 1, 0]
GIRL_PATTERN = [1, 1, 0, 0, 1, 1, 1, 1, 0]
KOKOA_PATTERN = [0, 1, 0, 1, 0, 0, 0, 1, 1]
MEDAL_PATTERN = [1, 0, 0, 0, 1, 0, 1, 0, 1]
TROPHY_PATTERN = [0, 0, 1, 1, 1, 1, 1, 0, 0]


while True:
    image = webcam.get_current_frame()

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5,5), 0)
    edges = cv2.Canny(gray, 100, 200)

    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.01 * perimeter, True)

        if len(approx) == QUADRILATERAL_POINTS:
            topdown_quad = get_topdown_quad(gray, approx.reshape(4, 2))

            resized_shape = resize_image(topdown_quad, SHAPE_RESIZE)

            if resized_shape[3, 3] > BLACK_THRESHOLD: continue

            glyph_found = False

            for i in range(4):
                glyph_pattern = None

                try:
                    glyph_pattern = get_glyph_pattern(resized_shape, BLACK_THRESHOLD, WHITE_THRESHOLD)
                except:
                    continue

                if not glyph_pattern: continue

                if glyph_pattern == BOY_PATTERN:
                    substitute_image = cv2.imread('Personaje1.png')
                    image = add_substitute_quad(image, substitute_image, approx.reshape(4, 2))
                    glyph_found = True
                    break
                elif glyph_pattern == GIRL_PATTERN:
                    substitute_image = cv2.imread('personaje2.png')
                    image = add_substitute_quad(image, substitute_image, approx.reshape(4, 2))
                    glyph_found = True
                    break
                elif glyph_pattern == KOKOA_PATTERN:
                    substitute_image = cv2.imread('kokoa.jpg')
                    image = add_substitute_quad(image, substitute_image, approx.reshape(4, 2))
                    glyph_found = True
                    break
                elif glyph_pattern == MEDAL_PATTERN:
                    substitute_image = cv2.imread('medalla.jpg')
                    image = add_substitute_quad(image, substitute_image, approx.reshape(4, 2))
                    glyph_found = True
                    break
                elif glyph_pattern == TROPHY_PATTERN:
                    substitute_image = cv2.imread('trofeo.png')
                    image = add_substitute_quad(image, substitute_image, approx.reshape(4, 2))
                    glyph_found = True
                    break

                resized_shape = rotate_image(resized_shape, 90)

            if glyph_found:
                break

    cv2.imshow('Glyph Recognition', image)
    cv2.waitKey(10)