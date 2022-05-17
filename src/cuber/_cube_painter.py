from collections import namedtuple
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt
from functools import lru_cache
from PIL import Image
import io



RotTrans = namedtuple('rottrans', 'alpha beta gamma dx dy dz')

cos = np.cos
sin = np.sin


def rotate_image(img, rts, f, xrel=0.5, yrel=0.5):
    """
    apply multiple rotation/translation operations, where each rts is a RotTrans/tuple with:
      alpha, beta, gamma ==> degrees to rotate about x, y, z axis...
        clockwise looking at it from right (x-axis), from top (y-axis), from front (z-axis)
      dx, dy, dz ==> translation along x, y, z axis... 
        x-axis goes negative to positive left to right
        y-axis goes negative to positive bottom to top
        z-axis goes negative to positive front to back
        (z axis is distance to image)
    At the end, apply f ==> focal distance
    xrel ==> relative position of camera in x (0..1 from left side of image to right side of image)
    yrel ==> relative position of camera in y (0..1 from bottom side of top side of image)
    """
    h, w = img.shape[:2]
    A1 = np.array([[1, 0, -w/2],
                   [0, 1, -h/2],
                   [0, 0, 0],
                   [0, 0, 1]])
    M = A1

    for alpha, beta, gamma, dx, dy, dz in rts:
        alpha, beta, gamma = map(
            lambda angle: angle * np.pi/180, [alpha, beta, gamma])
        alpha, beta = -alpha, -beta
        dy = -dy
        RX = np.array([[1,          0,           0, 0],
                       [0, cos(alpha), -sin(alpha), 0],
                       [0, sin(alpha),  cos(alpha), 0],
                       [0,          0,           0, 1]])
        RY = np.array([[cos(beta), 0, -sin(beta), 0],
                       [0, 1,          0, 0],
                       [sin(beta), 0,  cos(beta), 0],
                       [0, 0,          0, 1]])
        RZ = np.array([[cos(gamma), -sin(gamma), 0, 0],
                       [sin(gamma),  cos(gamma), 0, 0],
                       [0,          0,           1, 0],
                       [0,          0,           0, 1]])

        # Composed rotation matrix with (RX, RY, RZ)
        # R = RX @ RY @ RZ
        R = RZ @ RY @ RX  # right to left

        # Translation matrix
        T = np.array([[1, 0, 0, dx],
                      [0, 1, 0, dy],
                      [0, 0, 1, dz],
                      [0, 0, 0, 1]])
        M = T @ R @ M

    # 3D -> 2D matrix
    A2 = np.array([[f, 0, w*xrel, 0],
                   [0, f, h - h*yrel, 0],
                   [0, 0,   1, 0]])

    trans = A2 @ M
    result = cv2.warpPerspective(
        img, trans, img.shape[:2], cv2.INTER_AREA, borderValue=(0, 0, 0, 0))
    return result


def alpha_add(*images):
    """Overlay images with alpha channel - might not work except for binary opaque/transparent alpha channel"""
    if isinstance(images[0], (list, tuple)):
        images = images[0]
    image = images[0].copy()
    for aimage in images[1:]:
        mask = 1 - aimage[:, :, 3:] / 255.
        image = (image * mask).astype(np.uint8) + aimage
    return image

def get_rt_faces_cube(size, overlap=1):
    rt_front = RotTrans(0, 0, 0, 0, 0, -size/2 + overlap)
    rt_upper = RotTrans(90, 0, 0, 0, size/2 - overlap, 0)
    rt_right =  RotTrans(0, -90, 0, size/2 - overlap, 0, 0)
    rt_back = RotTrans(0, 180, 0, 0, 0, size/2 - overlap)
    rt_down = RotTrans(-90, 0, 0, 0, -size/2 + overlap, 0)
    rt_left =  RotTrans(0, 90, 0, -size/2 + overlap, 0, 0)
    return {
        "F": [rt_front],
        "U": [rt_upper],
        "R": [rt_right],
        "B": [rt_back],
        "D": [rt_down],
        "L": [rt_left]
    }


def get_rt_faces_topview(size, overlap=1):
    # TODO: the size/1.2 value is just approximate... need to figure out exact translation after rotating, or translate edge to axis, rotate, then
    # translate again
    rot_angle = 52 # angle around upper face
    rt_upper = [RotTrans(0, 0, 0, 0, 0, -size/2 + overlap)]
    rt_front = [RotTrans(-rot_angle, 0, 0, 0, -size/1.2 + overlap, 0)]
    rt_right =  [RotTrans(0, 0, -90, 0, 0, 0), RotTrans(0, -rot_angle, 0, size/1.2 - overlap, 0, 0)]
    rt_back = [RotTrans(0, 0, 180, 0, 0, 0), RotTrans(rot_angle, 0, 0, 0, size/1.2 - overlap, 0)]
    # rt_down = RotTrans(-90, 0, 0, 0, -size/2 + overlap, 0)
    rt_left =  [RotTrans(0, 0, 90, 0, 0, 0), RotTrans(0, rot_angle, 0, -size/1.2 + overlap, 0, 0)]
    return {
        "F": rt_front,
        "U": rt_upper,
        "R": rt_right,
        "B": rt_back,
        # "D": rt_down,
        "L": rt_left
    }    

def get_rt_cube_dict(size, focal_distance, perspective):
    return {
        "front-right": {
            "rt_cube": [RotTrans(0, 30, 0, 0, 0, 0), RotTrans(-15, 0, 0, 0, -size, focal_distance*1.5 + size*1.5)],
            "visible": "FUR",
            "xrel": 0.5,
            "yrel": 0.9,
            "rt_faces": get_rt_faces_cube(size)
        },
        "top-view": {
            "rt_cube": [RotTrans(0, 0, 0, 0, 0, focal_distance*1.5 + size*1.5)],
            "visible": "FURBL",
            "xrel": 0.5,
            "yrel": 0.5,
            "rt_faces": get_rt_faces_topview(size)
        }        
    }[perspective]




def draw_cube_faces(image_dict, *, rt_faces, rt_cube, xrel=0.5, yrel=0.9, f=200, **kwargs):
    images = []
    for face, image in image_dict.items():
        img = rotate_image(image, [*rt_faces[face], *rt_cube], f=f, xrel=xrel, yrel=yrel)
        images.append(img)
    return alpha_add(*images)

@lru_cache
def get_plastic_image(N, size, color="#d0d0d0"):
    color = (np.array(matplotlib.colors.to_rgba(color))*256).clip(0, 255).astype(np.uint8)
    image = np.tile(color, (size, size, 1))  # rgba image of color
    tile_size = size/N
    sticker_size = tile_size * 0.85
    sticker_offset = (tile_size - sticker_size) / 2
    for i in range(N):
        y0 = round(i*tile_size + sticker_offset)
        y1 = round(i*tile_size + sticker_offset + sticker_size)
        for j in range(N):
            x0 = round(j*tile_size + sticker_offset)
            x1 = round(j*tile_size + sticker_offset + sticker_size)
            image[y0:y1, x0:x1] = (0, 0, 0, 0)   # set to transparent
    return image
            

def stickers_to_image(face_stickers, size, sticker_cmap=None):
    """3x3 stickers array to image of that array"""
    if sticker_cmap is None:
        sticker_colors = ["w", "#ffcf00", "#00008f", "#009f0f", "#ff6f00", "#cf0000"]
        sticker_colors = [matplotlib.colors.to_rgb(it) for it in sticker_colors]
        sticker_colors = (np.array(sticker_colors)*256).clip(0,255).astype(np.uint8)
    elif sticker_cmap == 'oll':
        sticker_colors = ["w", "#ffcf00", "#00008f", "#009f0f", "#ff6f00", "#cf0000"]
        sticker_colors = np.array([matplotlib.colors.to_rgb(it) for it in sticker_colors])
        yellow = sticker_colors[1].copy()
        sticker_colors /= 5
        sticker_colors[1] = yellow
        sticker_colors = (sticker_colors*256).clip(0,255).astype(np.uint8)
    elif sticker_cmap == 'pll':
        sticker_colors = ["w", "#ffcf00", "#00008f", "#009f0f", "#ff6f00", "#cf0000"]
        sticker_colors = np.array([matplotlib.colors.to_rgb(it) for it in sticker_colors])
        sticker_colors[1] /= 5
        sticker_colors = (sticker_colors*256).clip(0,255).astype(np.uint8)        
    face_rgb = np.zeros(face_stickers.shape + (3,), dtype=np.uint8)
    for i in range(face_rgb.shape[0]):
        for j in range(face_rgb.shape[1]):
            face_rgb[i, j] = sticker_colors[face_stickers[j, face_rgb.shape[0] - i - 1]]  # gotta swap things around here...
    # print(sticker_colors)
    # print(face_stickers)
    # print(face_rgb)
    face_rgba = cv2.cvtColor(face_rgb, cv2.COLOR_RGB2RGBA)
    face_resize = cv2.resize(face_rgba, (size, size), interpolation=cv2.INTER_AREA)
    if sticker_cmap == 'oll' or sticker_cmap == 'pll':
        plastic = get_plastic_image(face_rgb.shape[0], size, color="#404040")
    else:
        plastic = get_plastic_image(face_rgb.shape[0], size)
    # TODO: draw plastic
    result = alpha_add(face_resize, plastic)
    return result

def draw_cube(stickers, perspective="front-right", size=200, focal_distance=200, sticker_cmap=None):
    face_map = {"U":0, "D":1, "F":2, "B":3, "R":4, "L":5}
    rt_cube_dict = get_rt_cube_dict(size, focal_distance, perspective)
    image_dict = {}
    for face in rt_cube_dict['visible']:
        face_index = face_map[face]
        image_dict[face] = stickers_to_image(stickers[face_index], size, sticker_cmap=sticker_cmap)
    # return image_dict
    return draw_cube_faces(image_dict, **rt_cube_dict)


def render_horiz(*images, basefigsize=(2, 2)):
    """Render images horizontally, save to PIL Image since transparency isn't working right with matplotlib"""
    if basefigsize:
        fig, ax = plt.subplots(figsize=(basefigsize[0]*len(images), basefigsize[1]), nrows=1, ncols=len(images))
    else:
        fig, ax = plt.subplots(nrows=1, ncols=len(images))
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
    for i, image in enumerate(images):
        if isinstance(image, np.ndarray):
            # if issubclass(image.dtype.type, np.float64):
            #    image = (image*255).astype(np.uint8)
            # if len(image.shape)==3 and image.shape[-1]==3:
            #    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # if bgr2rgb:
            #    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)
        with plt.rc_context({'image.cmap': 'gray'}):
            try:
                ax[i].axis('off')
                ax[i].set_facecolor('#0F00FF')
                ax[i].imshow(image)
            except TypeError:
                ax.axis('off')
                ax.imshow(image)
    # plt.show()
    # convert to PIL Image object
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    pil_img = Image.open(buf)
    pil_img.load()
    buf.close()
    plt.close()
    return pil_img