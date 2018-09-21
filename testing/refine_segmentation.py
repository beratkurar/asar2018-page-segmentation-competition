import cv2
import matplotlib.pyplot as plt
from xml.etree import ElementTree
from xml.dom import minidom
from xml.etree.ElementTree import Element, SubElement, Comment
import numpy as np
from pathlib import Path
import os


def segment_labeled_img(l_img):
    txt_l_img = np.ones(l_img.shape, dtype=np.uint8)*255
    txt_l_img[l_img == 0] = 0
    nontxt_l_img = np.ones(l_img.shape, dtype=np.uint8)*255
    nontxt_l_img[l_img == 128] = 0

    bin_txt_l_img = np.zeros(l_img.shape, dtype=np.uint8)
    bin_txt_l_img[l_img == 0] = 1

    bin_nontxt_l_img = np.zeros(l_img.shape, dtype=np.uint8)
    bin_nontxt_l_img[l_img == 128] = 1

    kernel = np.ones((6, 3), np.uint8)

    txt_l_img = cv2.dilate(txt_l_img, kernel, iterations=5)
    kernel = np.ones((3, 3), np.uint8)
    nontxt_l_img = cv2.dilate(nontxt_l_img, kernel, iterations=4)

    # plt.figure(1), plt.imshow(nontxt_l_img)

    kernel = np.ones((5, 7), np.uint8)
    txt_l_img = cv2.morphologyEx(txt_l_img, cv2.MORPH_OPEN, kernel, iterations=8)
    kernel = np.ones((5, 9), np.uint8)
    nontxt_l_img = cv2.morphologyEx(nontxt_l_img, cv2.MORPH_OPEN, kernel, iterations=10)

    # plt.figure(2), plt.imshow(nontxt_l_img)
    # plt.show()

    im2, txt_contours, hierarchy1 = cv2.findContours(txt_l_img, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    im2, nontxt_contours, hierarchy2 = cv2.findContours(nontxt_l_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    txt_l_img = -1 * txt_l_img + 255
    nontxt_l_img = -1 * nontxt_l_img + 255

    contours_img = np.zeros(txt_l_img.shape, dtype=np.uint8)

    txt_bbs = []
    nontxt_bbs = []

    for i in range(len(txt_contours)):
        cnt = txt_contours[i]

        area = cv2.contourArea(cnt)
        mask = np.zeros((txt_l_img.shape[0], txt_l_img.shape[1]))
        pts = np.asarray([cnt])
        mask = cv2.fillPoly(mask, pts, 1)
        t = np.multiply(mask, bin_txt_l_img)

        p = t.sum()/area
        intersection = np.logical_and(mask, contours_img)


        if p >= 0.5 and area > 100 and intersection.sum() == 0:
            cv2.fillPoly(contours_img, pts, 1)
            x, y, w, h = cv2.boundingRect(cnt)
            txt_bbs.append([[x, y], [x + w, y], [x + w, y + h], [x, y + h]])

            cv2.rectangle(img, (x, y), (x + w, y + h), (100, 0, 0), 10)



    contours_img = np.zeros(nontxt_l_img.shape, dtype=np.uint8)

    for i in range(len(nontxt_contours)):
        cnt = nontxt_contours[i]

        area = cv2.contourArea(cnt)
        mask = np.zeros((txt_l_img.shape[0], txt_l_img.shape[1]))
        pts = np.asarray([cnt])
        mask = cv2.fillPoly(mask, pts, 1)
        t = np.multiply(mask, bin_nontxt_l_img)

        p = t.sum() / area
        intersection = np.logical_and(mask, contours_img)

        if p >= 0.5 and area > 500 and intersection.sum() == 0:
            cv2.fillPoly(contours_img, pts, 1)
            x, y, w, h = cv2.boundingRect(cnt)
            nontxt_bbs.append([[x, y], [x + w, y], [x + w, y + h], [x, y + h]])

            cv2.rectangle(img, (x, y), (x + w, y + h), (200, 0, 0), 10)


    # plt.figure(1)
    # plt.imshow(img)
    # plt.show()

    return txt_bbs, nontxt_bbs


def segment_labeled_img_2(l_img):
    txt_l_img = np.ones(l_img.shape, dtype=np.uint8)*255
    txt_l_img[l_img == 0] = 0
    nontxt_l_img = np.ones(l_img.shape, dtype=np.uint8)*255
    nontxt_l_img[l_img == 128] = 0

    bin_txt_l_img = np.zeros(l_img.shape, dtype=np.uint8)
    bin_txt_l_img[l_img == 0] = 1

    bin_nontxt_l_img = np.zeros(l_img.shape, dtype=np.uint8)
    bin_nontxt_l_img[l_img == 128] = 1

    # plt.figure(5), plt.imshow(nontxt_l_img)
    # plt.figure(6), plt.imshow(txt_l_img)

    kernel = np.ones((6, 3), np.uint8)

    txt_l_img = cv2.dilate(txt_l_img, kernel, iterations=5)
    kernel = np.ones((4, 4), np.uint8)
    nontxt_l_img = cv2.dilate(nontxt_l_img, kernel, iterations=4)
    #
    # plt.figure(1), plt.imshow(nontxt_l_img)
    # plt.figure(0), plt.imshow(txt_l_img)

    kernel = np.ones((5, 7), np.uint8)
    txt_l_img = cv2.morphologyEx(txt_l_img, cv2.MORPH_OPEN, kernel, iterations=8)
    kernel = np.ones((5, 9), np.uint8)
    nontxt_l_img = cv2.morphologyEx(nontxt_l_img, cv2.MORPH_OPEN, kernel, iterations=10)

    # plt.figure(4), plt.imshow(nontxt_l_img)
    # plt.figure(3), plt.imshow(txt_l_img)


    im2, txt_contours, hierarchy1 = cv2.findContours(txt_l_img, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    im2, nontxt_contours, hierarchy2 = cv2.findContours(nontxt_l_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    txt_l_img = -1 * txt_l_img + 255
    nontxt_l_img = -1 * nontxt_l_img + 255

    contours_img = np.zeros(txt_l_img.shape, dtype=np.uint8)

    txt_bbs = []
    nontxt_bbs = []

    for i in range(len(txt_contours)):
        cnt = txt_contours[i]

        area = cv2.contourArea(cnt)
        mask = np.zeros((txt_l_img.shape[0], txt_l_img.shape[1]))
        pts = np.asarray([cnt])
        mask = cv2.fillPoly(mask, pts, 1)
        t = np.multiply(mask, bin_txt_l_img)

        p = t.sum()/area
        intersection = np.logical_and(mask, contours_img)


        if p >= 0.5 and area > 100 and intersection.sum() == 0:
            cv2.fillPoly(contours_img, pts, 1)
            epsilon = 0.01 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)

            t_cnt = []
            for p in approx:
                t_cnt.append([p[0][0], p[0][1]])
            txt_bbs.append(t_cnt)

            cv2.drawContours(img, [approx], 0, (100, 255, 0), 5)

            # cv2.rectangle(img, (x, y), (x + w, y + h), (100, 0, 0), 10)



    contours_img = np.zeros(nontxt_l_img.shape, dtype=np.uint8)

    for i in range(len(nontxt_contours)):
        cnt = nontxt_contours[i]

        area = cv2.contourArea(cnt)
        mask = np.zeros((txt_l_img.shape[0], txt_l_img.shape[1]))
        pts = np.asarray([cnt])
        mask = cv2.fillPoly(mask, pts, 1)
        t = np.multiply(mask, bin_nontxt_l_img)

        p = t.sum() / area
        intersection = np.logical_and(mask, contours_img)

        if p >= 0.5 and area > 500 and intersection.sum() == 0:
            cv2.fillPoly(contours_img, pts, 1)

            epsilon = 0.01 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)

            t_cnt = []
            for p in approx:
                t_cnt.append([p[0][0], p[0][1]])
            nontxt_bbs.append(t_cnt)

            cv2.drawContours(img, [approx], 0, (200, 0, 0), 5)
            # cv2.rectangle(img, (x, y), (x + w, y + h), (200, 0, 0), 10)


    plt.figure(1)
    plt.imshow(contours_img)
    # plt.show()

    return txt_bbs, nontxt_bbs

def prettify(elem):
    """Return a pretty-printed XML string for the Element.
    """
    rough_string = ElementTree.tostring(elem, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="  ")


def generate_xml_file(txt_bbs, nontxt_bbs, file_name='results.xml'):
    top = Element('Contours')

    for bb in txt_bbs:
        child = SubElement(top, 'contour')
        for i in range(len(bb)):
            sub_child = SubElement(child, 'x{}'.format(i))
            sub_child.text = '{}'.format(bb[i][1])
            sub_child = SubElement(child, 'y{}'.format(i))
            sub_child.text = '{}'.format(bb[i][0])
        label_child = SubElement(child, 'label')
        label_child.text = '1'

    for bb in nontxt_bbs:
        child = SubElement(top, 'contour')
        for i in range(len(bb)):
            sub_child = SubElement(child, 'x{}'.format(i))
            sub_child.text = '{}'.format(bb[i][1])
            sub_child = SubElement(child, 'y{}'.format(i))
            sub_child.text = '{}'.format(bb[i][0])
        label_child = SubElement(child, 'label')
        label_child.text = '0'

        'visualization'

    xml_file = open(file_name, 'w')
    xml_file.write(prettify(top))



images_path = Path('./').glob('./Results/seta/*.png')

for path in images_path:
    img = cv2.imread('{}'.format(path), 0)

    img[img < 64] = 0
    img[img > 192] = 255
    img[(img >= 64) & (img <= 192)] = 128

    txt_bbs, nontxt_bbs = segment_labeled_img_2(img)

    xml_path = os.path.splitext(os.path.basename(path))[0]

    generate_xml_file(txt_bbs, nontxt_bbs, file_name='./Results/xml_A/{}.xml'.format(xml_path))
    cv2.imwrite('./Results/Set A - Visualization/{}_v-img.png'.format(xml_path), img)

images_path = Path('./').glob('./Results/setb/*.png')

for path in images_path:
    img = cv2.imread('{}'.format(path), 0)

    img[img < 64] = 0
    img[img > 192] = 255
    img[(img >= 64) & (img <= 192)] = 128

    txt_bbs, nontxt_bbs = segment_labeled_img_2(img)

    xml_path = os.path.splitext(os.path.basename(path))[0]

    generate_xml_file(txt_bbs, nontxt_bbs, file_name='./Results/xml_B/{}.xml'.format(xml_path))
    cv2.imwrite('./Results/Set B - Visualization/{}_v-img.png'.format(xml_path), img)


images_path = Path('./').glob('./Results/setc/*.png')

for path in images_path:
    img = cv2.imread('{}'.format(path), 0)

    img[img < 64] = 0
    img[img > 192] = 255
    img[(img >= 64) & (img <= 192)] = 128

    txt_bbs, nontxt_bbs = segment_labeled_img_2(img)

    xml_path = os.path.splitext(os.path.basename(path))[0]

    generate_xml_file(txt_bbs, nontxt_bbs, file_name='./Results/xml_C/{}.xml'.format(xml_path))
    cv2.imwrite('./Results/Set C - Visualization/{}_v-img.png'.format(xml_path), img)