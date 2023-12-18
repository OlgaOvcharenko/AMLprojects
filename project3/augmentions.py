import cv2
# import elasticdeform
import numpy as np
import skimage


def tr(image, seed=2):  # better less destructive
    return rotate(translate(image, seed=seed), seed=seed)


def dtr(image, seed):
    return rt(deformation_grid(image, seed=seed), seed=seed)


def rt(image, seed=2):  # more destructive
    return translate(rotate(image, seed=seed), seed=seed)


def trs(image, seed=2):  # better less destructive
    return rotate(translate(shear(image, seed=seed), seed=seed), seed=seed)


def dtrs(image, seed):
    return rt(deformation_grid(shear(image, seed=seed), seed=seed), seed=seed)


def rts(image, seed=2):  # more destructive
    return translate(rotate(shear(image, seed=seed), seed=seed), seed=seed)


def drt(image, seed):
    return rt(deformation_grid(image, seed=seed), seed=seed)


def rwt(image, seed=2):
    return translate(warp(rotate(image, seed=seed), seed=seed), seed=seed)


def drwt(image, seed):
    return rwt(deformation_grid(image, seed=seed), seed=seed)


def wtr(image, seed=2):
    return warp(translate(rotate(image, seed=seed), seed=seed), seed=seed)


def dwtr(image, seed):
    return wtr(deformation_grid(image, seed=seed), seed=seed)


def dr(image, seed=2):
    return rotate(deformation_grid(image, seed=seed), seed=seed)


def dz(image, seed=2):
    return zoom(deformation_grid(image, seed=seed), seed=seed)


def ez(image, seed=2):
    return zoom(elastic_deformation_grid(image, seed=seed), seed=seed)


def ezr(image, seed=2):
    return zoom(elastic_deformation_grid(image, seed=seed), seed=seed)


def er(image, seed=2):
    return rotate(elastic_deformation_grid(image, seed=seed), seed=seed)


def rotate(image, min_angle=10, max_angle=40, seed=42):
    np.random.seed(seed)
    positive = 1 if np.random.uniform(-1, 1) > 0 else -1
    angle = np.random.uniform(min_angle, max_angle) * positive
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(
        image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result


def warp(image, deform_max=.2, seed=42):
    np.random.seed(seed)

    def get_warp_transform(image, deform_max, seed=42):
        deform_max = image.shape[0] * deform_max
        x, y = image.shape[0], image.shape[1]
        source = np.float32([[0, 0], [x, 0], [0, y], [x, y]])
        ret = np.random.uniform(-1, 1, size=(4, 2)) * deform_max
        return cv2.getPerspectiveTransform(source, np.float32(source + ret))

    transform = get_warp_transform(image, deform_max=np.random.random() * deform_max + 0.04, seed=seed)
    dsize = (image.shape[0], image.shape[1])
    im2 = cv2.warpPerspective(image, transform, dsize)
    return im2


def translate(image, move=0.1, seed=42):
    move = image.shape[0] * move
    np.random.seed(seed)
    mv = (np.float32(np.random.uniform(size=(2, 1))) * 2 - 1) * move
    stas = np.float32([[1, 0], [0, 1]])
    m = np.hstack([stas, mv])
    return cv2.warpAffine(image, m, (image.shape[0], image.shape[1]))


def zoom(image, factor_min=0.5, factor_max=1.3, seed=42):
    np.random.seed(seed)
    v = np.random.uniform(factor_min, factor_max, size=(1, 1))
    sizeBefore = image.shape[0]
    sizeAfter = image.shape[0] * v
    diff = sizeAfter - sizeBefore
    m = np.float32(np.array([[v, 0, -diff / 2], [0, v, -diff / 2]], dtype=object))
    return cv2.warpAffine(image, m, (image.shape[0], image.shape[1]))


# !!! IMPORTANT: do not use for augmentations
def zoom_at(img, zoom=1, angle=0, coord=None):
    cy, cx = [i / 2 for i in img.shape] if coord is None else coord[::-1]

    rot_mat = cv2.getRotationMatrix2D((cx, cy), angle, zoom)
    result = cv2.warpAffine(img, rot_mat, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR)

    return result


def deformation_grid(img, density=None, deform_scale=None, seed=42):
    # https://www.programcreek.com/python/?code=DerWaldi%2Fyoutube-video-face-swap%2Fyoutube-video-face-swap-master%2Fimage_augmentation.py
    np.random.seed(seed)
    density = density if density != None else np.random.randint(3, 10)
    deform_scale = deform_scale if deform_scale != None else np.random.random() / 1.4 + 0.3
    mapx = np.broadcast_to(np.linspace(
        0, img.shape[0] * 2, density), (density, density))
    mapy = np.broadcast_to(np.linspace(
        0, img.shape[1] * 2, density), (density, density)).T

    correction = img.shape[0] / density
    mapx = mapx + (np.random.normal(size=(density, density),
                                    scale=deform_scale * correction))
    mapy = mapy + (np.random.normal(size=(density, density),
                                    scale=deform_scale * correction))
    interp_mx = cv2.resize(
        mapx, (img.shape[0] * 2, img.shape[1] * 2)).astype('float32')
    interp_my = cv2.resize(
        mapy, (img.shape[0] * 2, img.shape[1] * 2)).astype('float32')

    img2 = cv2.resize(
        img, (img.shape[0] * 2, img.shape[1] * 2), interpolation=cv2.INTER_AREA)

    ret1 = cv2.remap(img2, interp_mx, interp_my,
                     cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    scale = (img.shape[0] + correction * 2) / img2.shape[0]
    m = np.float32([[scale, 0, -correction], [0, scale, -correction]])
    return cv2.warpAffine(ret1, m, (img.shape[0], img.shape[1]))


def elastic_deformation_grid(img, density=None, deform_scale=0.5, seed=42):
    # https://www.programcreek.com/python/?code=DerWaldi%2Fyoutube-video-face-swap%2Fyoutube-video-face-swap-master%2Fimage_augmentation.py
    np.random.seed(seed)
    density = density if density != None else np.random.randint(10, 20)
    deform_scale = deform_scale if deform_scale != None else np.random.random() / 1.4 + 0.3
    mapx = np.broadcast_to(np.linspace(
        0, img.shape[0] * 2, density), (density, density))
    mapy = np.broadcast_to(np.linspace(
        0, img.shape[1] * 2, density), (density, density)).T

    correction = img.shape[0] / density
    mapx = mapx + (np.random.normal(size=(density, density),
                                    scale=deform_scale * correction))
    mapy = mapy + (np.random.normal(size=(density, density),
                                    scale=deform_scale * correction))
    interp_mx = cv2.resize(
        mapx, (img.shape[0] * 2, img.shape[1] * 2)).astype('float32')
    interp_my = cv2.resize(
        mapy, (img.shape[0] * 2, img.shape[1] * 2)).astype('float32')

    img2 = cv2.resize(
        img, (img.shape[0] * 2, img.shape[1] * 2), interpolation=cv2.INTER_AREA)

    ret1 = cv2.remap(img2, interp_mx, interp_my,
                     cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    scale = (img.shape[0] + correction * 2) / img2.shape[0]
    m = np.float32([[scale, 0, -correction], [0, scale, -correction]])
    return cv2.warpAffine(ret1, m, (img.shape[0], img.shape[1]))


def sheer(img, factor_max=0.2, seed=42):
    # TODO fix corrections diff
    np.random.seed(seed)
    v1 = np.random.uniform(size=(1, 1))[0, 0] * factor_max
    v2 = np.random.uniform(size=(1, 1))[0, 0] * factor_max

    diff1 = img.shape[0] * v1 - img.shape[0]
    diff2 = img.shape[1] * v2 - img.shape[1]

    m = np.float32([[1, v1, diff1 / 2], [v2, 1, diff2 / 2]])
    return cv2.warpAffine(img, m, (img.shape[0], img.shape[1]))


def shear(img, seed=42):
    # Create Afine transform
    np.random.seed(seed)
    v1 = np.random.randint(20, 30) / 100
    afine_tf = skimage.transform.AffineTransform(shear=v1)

    # Apply transform to image data
    return skimage.transform.warp(img, inverse_map=afine_tf)


# # img, sigma=15, min_points=15, max_points=20, seed=42
# def elastic_deformation_grid(img, sigma=20, min_points=10, max_points=15, seed=42):
#     np.random.seed(seed)
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     points = np.random.randint(min_points, max_points)
#     return elasticdeform.deform_random_grid(img, sigma=sigma, points=points)


def flip(img):
    return cv2.flip(img, 1)


def scale():
    # TODO
    pass


def brightness():
    # TODO
    pass


def contrast(image, size):
    # https://towardsdatascience.com/contrast-enhancement-of-grayscale-images-using-morphological-operators-de6d483545a1
    # Kushol R., Nishat R. M., Rahman A. B. M. A., Salekin M. M., “Contrast Enhancement of Medical X-Ray
    # Image Using Morphological Operators with Optimal Structuring Element,” arXiv:1905.08545v1 [cs.CV] 27 May 2019
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    # make values flot to make sure we do not overflow or underflow
    topHat = cv2.morphologyEx(image, cv2.MORPH_TOPHAT,
                              kernel).astype('float32')
    blackHat = cv2.morphologyEx(
        image, cv2.MORPH_BLACKHAT, kernel).astype('float32')
    image = image.astype('float32')
    ret = (image + topHat - blackHat)

    # correct overflows
    ret[ret > 255] = 255
    # correct underflows
    ret[ret < 0] = 0

    # go back to uint8
    return ret.astype('uint8')


def show(image, name):
    cv2.imshow(name, image)


if __name__ == "__main__":
    m = cv2.imread("ben.jpeg")
    dim = (int)(m.shape[0])
    # dim = (dim, dim)
    m2 = m
    # m2 = cv2.resize(m, dim, interpolation=cv2.INTER_AREA)
    # show(m2, "org")
    # m2 = cv2.cvtColor(m, cv2.COLOR_BGR2GRAY)

    # m2_r2 = rotate_image(m2, 10)
    # show(m2_r2, "r10")
    # show(m2_warp2, "warp")

    # m2_warp2 = warp(m2, seed=13)
    # show(m2_warp2, "warp2")

    # m2_warp2 = warp(m2, seed=44)
    # show(m2_warp2, "warp3")

    # m2_trans = translate(m2)
    # show(m2_trans, "translate")

    # m2_zoom = zoom(m2)
    # show(m2_zoom, "zoom")

    # show(mg, "gray")
    # show(contrast(mg, 2), "contras2")
    # show(contrast(mg, 4), "contras4")
    # show(contrast(mg, 10), "contrast10")
    # show(contrast(mg, 20), "contrast20")
    # show(contrast(mg, 40), "contrast40")

    # m2_flip = flip(m2)
    # show(m2_flip, "flipped")

    # m2_deform = deformation_grid(m2)
    # show(deformation_grid(m2), "deform")
    # deform_scale = 0.3
    # show(deformation_grid(m2, seed=64), "deform64")
    # show(deformation_grid(m2, seed=48), "deform48")
    # show(deformation_grid(m2, seed=32), "deform32")
    # show(deformation_grid(m2, seed=16), "deform16")
    # show(deformation_grid(m2, seed=8), "deform8")
    # show(deformation_grid(m2, seed=4), "deform4")
    # show(deformation_grid(m2, seed=2), "deform2")

    # show(deformation_grid(rotate_image(mg,42), density=9, deform_scale=0.3),"rotate_deform")

    # show(rotate(m, -30, 30), "sheer2")
    # show(rotate(m2, -30, 30), "sheer3")

    # m2_trans = sheer(m2)
    # print(m2_trans.shape)
    # print(m2.shape)

    # show(sheer(m2), "sheer")
    # show(rotate(m2, seed = 0),"rotate0")
    # show(rotate(m2, seed = 1),"rotate1")
    # show(rotate(m2, seed = 2),"rotate2")
    # show(rotate(m2, seed = 3),"rotate3")
    # show(rotate(m2, seed = 4),"rotate4")
    # show(rotate(m2, seed = 5),"rotate5")

    # show(warp(m2, seed= 3), "warp3")
    # show(warp(m2, seed= 4), "warp4")
    # show(warp(m2, seed= 5), "warp5")
    # show(warp(m2, seed= 6), "warp6")
    # show(warp(m2, seed= 7), "warp7")
    # show(warp(m2, seed= 8), "warp8")

    # show(translate(m2, seed= 2), "trans2")
    # show(translate(m2, seed= 3), "trans3")
    # show(translate(m2, seed= 4), "trans4")
    # show(translate(m2, seed= 5), "trans5")
    # show(translate(m2, seed= 6), "trans6")
    # show(translate(m2, seed= 7), "trans7")
    # show(translate(m2, seed= 8), "trans8")

    # show(rwt(m2, seed=2), "rwt2")
    # show(rwt(m2, seed=3), "rwt3")
    # show(rwt(m2, seed=4), "rwt4")
    # show(rwt(m2, seed=5), "rwt5")
    # show(rwt(m2, seed=6), "rwt6")
    # show(rwt(m2, seed=7), "rwt7")
    # show(rwt(m2, seed=8), "rwt8")

    # show(rt(m2, seed=2), "rt2")
    # show(rt(m2, seed=3), "rt3")
    # show(rt(m2, seed=4), "rt4")
    # show(rt(m2, seed=5), "rt5")
    # show(rt(m2, seed=6), "rt6")
    # show(rt(m2, seed=7), "rt7")
    # show(rt(m2, seed=8), "rt8")

    # show(tr(m2, seed=2), "tr2")
    # show(tr(m2, seed=3), "tr3")
    # show(tr(m2, seed=4), "tr4")
    # show(tr(m2, seed=5), "tr5")
    # show(tr(m2, seed=6), "tr6")
    # show(tr(m2, seed=7), "tr7")
    # show(tr(m2, seed=8), "tr8")

    # show(dr(m2, seed=2), "dr2")
    # show(dr(m2, seed=3), "dr3")
    # show(dr(m2, seed=4), "dr4")
    # show(dr(m2, seed=5), "dr5")
    # show(dr(m2, seed=6), "dr6")
    # show(dr(m2, seed=7), "dr7")
    # show(dr(m2, seed=8), "dr8")

    # show(dz(m2, seed=3), "dz3")
    # show(dz(m2, seed=4), "dz4")
    # show(dz(m2, seed=5), "dz5")
    # show(dz(m2, seed=6), "dz6")
    # show(dz(m2, seed=7), "dz7")
    # show(dz(m2, seed=8), "dz8")

    # show(drwt(m2, seed=3), "drwt3")
    # show(drwt(m2, seed=4), "drwt4")
    # show(drwt(m2, seed=5), "drwt5")
    # show(drwt(m2, seed=6), "drwt6")
    # show(drwt(m2, seed=7), "drwt7")
    # show(drwt(m2, seed=8), "drwt8")

    # show(dz(m2, seed=2), "dz2")
    # scaleFactor = 1.5
    # m3 = cv2.resize(m, (int(dim*scaleFactor),int(dim*scaleFactor)), interpolation=cv2.INTER_AREA)
    # show(dz(m3, seed=2), "dz2x2")

    show(m2, "m2")
    m_elastic = shear(m2, seed=10)
    m_elastic1 = shear(m2, seed=42)
    m_elastic2 = shear(m2, seed=100)
    m_elastic3 = shear(m2, seed=320)
    m_elastic4 = shear(m2, seed=18)
    show(m_elastic, "dz2x2")
    show(m_elastic1, "dz2x21")
    show(m_elastic2, "dz2x22")
    show(m_elastic3, "dz2x23")
    show(m_elastic4, "dz2x24")

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # time.sleep(10)
