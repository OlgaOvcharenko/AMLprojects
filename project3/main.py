import gzip
import pickle

import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

import model, model_box, augmentions
import skimage.measure


def validate_single_image(gt, pred):
    # a = pred.flatten().astype(bool)
    # b = gt.flatten().astype(bool)
    # jaccard = sum(a & b) / sum(a | b)

    return jaccard_coef(gt, pred)


def jaccard_coef(y_true, y_pred):
    SMOOTHING_FACTOR = 1.0
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    return (intersection + SMOOTHING_FACTOR) / (np.sum(y_true_f) + np.sum(y_pred_f) - intersection + SMOOTHING_FACTOR)


def load_zipped_pickle(filename):
    with gzip.open(filename, 'rb') as f:
        loaded_object = pickle.load(f)
        return loaded_object


def save_zipped_pickle(obj, filename):
    with gzip.open(filename, 'wb') as f:
        pickle.dump(obj, f, 2)


def make_predictions(test_data):
    # make prediction for test
    # predictions = []
    # for d in test_data:
    #     # TODO fix predictions
    #     prediction = np.array(np.zeros_like(d['video']), dtype=np.bool)
    #     height = prediction.shape[0]
    #     width = prediction.shape[1]
    #     prediction[int(height / 2) - 50:int(height / 2 + 50), int(width / 2) - 50:int(width / 2 + 50)] = True
    #
    #     # DATA Strucure
    #     predictions.append({
    #         'name': d['name'],
    #         'prediction': prediction
    #     }
    #     )

    predictions = []
    for d in test_data:
        # DATA Strucure
        predictions.append({
            'name': d['name'],
            'prediction': d['label']
        }
        )

    return predictions


def show(X, Y):
    X = cv2.resize(X, (Y.shape[0] * 4, Y.shape[1] * 4), interpolation=cv2.INTER_AREA)
    Y = cv2.resize(Y, (Y.shape[0] * 4, Y.shape[1] * 4), interpolation=cv2.INTER_AREA)
    img = np.zeros((Y.shape[0], Y.shape[1], 3))
    img[:, :, 2] = X / 256
    img[:, :, 1] = Y.astype(np.float32) * 0.5
    # img = cv2.resize(img, (img.shape[0] *4,img.shape[1] *4),interpolation=cv2.INTER_AREA)

    cv2.imshow("input", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def show_image(train_data_p, frame_id_p, video_id_p, name):
    image = train_data_p[video_id_p]['video'][:, :, frame_id_p]
    prediction = train_data_p[video_id_p]['label'][:, :, frame_id_p]
    # box = train_data_p[video_id_p]['box']

    # img1 = np.array([image, prediction, box]).transpose((2, 1, 0))

    # image2 = train_data_a[video_id_a]['video'][:, :, frame_id_a]
    # prediction2 = train_data_a[video_id_a]['label'][:, :, frame_id_a]
    # box2 = train_data_a[video_id_a]['box']
    #
    # img2 = np.array([image2, prediction2, box2]).transpose((2, 1, 0))

    # Hori = np.concatenate((img1, img2), axis=1)
    # cv2.imshow('HORIZONTAL', Hori)
    # cv2.waitKey(0)
    # exit()

    plt.imshow(image)
    plt.imshow(prediction, alpha=0.4)
    # plt.imshow(box, alpha=0.2)
    plt.show()

    plt.savefig('{}_{}_{}'.format(name, video_id_p, frame_id_p))


def split_to_get_validation(train_data, split):
    # TODO add cross-fold IMPORTANT!!!!
    train_data_amateur = []
    train_data_prof = []
    for data in train_data:
        if data['dataset'] == 'amateur':
            train_data_amateur.append(data)

        else:
            train_data_prof.append(data)

    train_data_prof, val_data_prof = train_test_split(train_data_prof, test_size=split, random_state=18)

    return train_data_amateur, train_data_prof, val_data_prof


def extract_labeled_frames(train_data):
    for data in train_data:
        data['valid_frames'] = data['video'][:, :, data['frames']]
        data['valid_labels'] = data['label'][:, :, data['frames']]


def resize_hist_equalization_and_sharpen(train_data, train, resize_label, resize_h_img, resize_w_img,
                                         resize_h_label, resize_w_label, hist_eq, sharpen,
                                         resize_method=cv2.INTER_LINEAR):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    kernel = 0.5 * np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])

    for data in train_data:
        data['augmented_frames'], data['augmented_labels'] = [], []
        video_f = 'valid_frames' if train or 'valid_frames' in data.keys() else 'video'

        frames_id = data[video_f].shape[2]

        for frame_id in range(frames_id):
            img = data[video_f][:, :, frame_id]

            # put into square
            img = resize_image_to_square(img)

            # resize
            img = cv2.resize(img, (resize_w_img, resize_h_img), interpolation=resize_method)

            if train:
                # put into square and resize labels
                label = data['valid_labels'][:, :, frame_id]
                label = resize_image_to_square(label)

                if resize_label:
                    old_type = img.dtype
                    label = label.astype(np.uint8)
                    label = cv2.resize(label, (resize_w_label, resize_h_label), interpolation=resize_method)
                    label = label.astype(old_type)

            #  adaptive hist equalization
            if hist_eq:
                img = clahe.apply(img)

            # sharpening
            if sharpen:
                # img = cv2.filter2D(img, -1, kernel)
                img = augmentions.contrast(img, size=4)

            # Hori = np.concatenate((img, img1, img2), axis=1)
            # cv2.imshow('HORIZONTAL', Hori)
            # cv2.waitKey(0)
            # exit()
            data['augmented_frames'].append(img)
            if train:
                data['augmented_labels'].append(label)


def preprocess_data_box(train_data, train, resize_box, resize_h_img, resize_w_img,
                        resize_h_label, resize_w_label, hist_eq,
                        resize_method=cv2.INTER_LINEAR):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    for data in train_data:
        data['augmented_frames'], data['augmented_box'] = [], []
        video_f = 'valid_frames' if train and 'valid_frames' in data.keys() else 'video'

        frames_id = data[video_f].shape[2]

        for frame_id in range(frames_id):
            img = data[video_f][:, :, frame_id]

            # put into square
            img = resize_image_to_square(img)

            # resize
            img = cv2.resize(img, (resize_w_img, resize_h_img), interpolation=resize_method)

            if train:
                # put into square and resize labels
                box = data['box']
                box = resize_image_to_square(box)

                if resize_box:
                    old_type = img.dtype
                    box = box.astype(np.uint8)
                    box = cv2.resize(box, (resize_w_label, resize_h_label), interpolation=resize_method)
                    box = box.astype(old_type)

            #  adaptive hist equalization
            if hist_eq:
                img = clahe.apply(img)

            # Hori = np.concatenate((img, img1, img2), axis=1)
            # cv2.imshow('HORIZONTAL', Hori)
            # cv2.waitKey(0)
            # exit()
            data['augmented_frames'].append(img)
            if train:
                data['augmented_box'].append(box)


def resize_image_to_square(img):
    width, height = img.shape
    if width == height:
        return img

    padded = max(width, height) - min(width, height)
    bottom, top, right, left = 0, 0, 0, 0

    if width < height:
        top, bottom = int(padded / 2) if padded % 2 == 0 else int(padded // 2), \
                      int(padded / 2) if padded % 2 == 0 else int((padded // 2) + 1)
    else:
        left, right = int(padded / 2) if padded % 2 == 0 else int(padded // 2), \
                      int(padded / 2) if padded % 2 == 0 else int((padded // 2) + 1)
    old_type = img.dtype
    img = img.astype(np.uint8)
    image = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT)

    image = image.astype(old_type)

    return image


def resize_image_to_unsquare(img, original_w, original_h):
    (s_width, s_height) = img.shape
    padded = max(original_w, original_h) - min(original_w, original_h)

    bottom, top, right, left = 0, 0, 0, 0
    if original_w < original_h:
        top, bottom = int(padded / 2) if padded % 2 == 0 else int(padded // 2), \
                      int(padded / 2) if padded % 2 == 0 else int((padded // 2) + 1)
    else:
        left, right = int(padded / 2) if padded % 2 == 0 else int(padded // 2), \
                      int(padded / 2) if padded % 2 == 0 else int((padded // 2) + 1)

    cropped = img[top:s_height - bottom, left:s_width - right]

    return cropped


def resize_unsquare(test_data, resize: bool, unsquare: bool, resize_method=cv2.INTER_LINEAR):
    for data in test_data:
        frames_id = len(data['predictions'])
        data['label'] = []
        for frame in range(frames_id):
            prediction = data['predictions'][frame]
            original = data['video'][:, :, frame]

            (resize_w_img, resize_h_img) = original.shape

            if resize:
                old_type = prediction.dtype
                prediction = prediction.astype(np.uint8)

                resize_square_size = max(resize_w_img, resize_h_img)
                prediction = cv2.resize(prediction, (resize_square_size, resize_square_size), interpolation=resize_method)
                prediction = prediction.astype(old_type)

            if unsquare:
                prediction = resize_image_to_unsquare(prediction, resize_w_img, resize_h_img)

            data['label'].append(prediction)

        data['label'] = np.dstack(data['label'])


def test(network, resize_w_img, resize_h_img, resize_h_label, resize_w_label):
    test_data = load_zipped_pickle('Data/test.pkl')

    # preprocess test without resize
    resize_hist_equalization_and_sharpen(test_data, train=False, resize_label=False,
                                         resize_h_img=resize_h_img, resize_w_img=resize_w_img,
                                         resize_h_label=resize_h_label, resize_w_label=resize_w_label,
                                         hist_eq=True, sharpen=False)

    # make predictions
    # test_data = test_data[0:2]
    test_data = model.predict(network, test_data)

    resize_unsquare(test_data, unsquare=True, resize=True)

    predictions = make_predictions(test_data=test_data)

    # for i in range(len(test_data)):
    #     for j in range(test_data[i]['video'].shape[2]):
    #         show_image(test_data[i], i, j, f'img_{i}_{j}')

    return predictions


def train(load_weights, path_weights, resize_w_img, resize_h_img, resize_h_label, resize_w_label):
    if load_weights:
        network = model.load_weights(path_weights)

    else:
        # load data
        train_data = load_zipped_pickle('Data/train.pkl')

        extract_labeled_frames(train_data)

        # split data
        train_data_amateur, train_data_prof, val_data_prof = split_to_get_validation(train_data, split=0.3)

        # create train of both prof and amateur
        train_data_all = train_data_prof

        # Expand with more of the validation data to get a bit more training data. aka now we have 90 % data
        moreTrainingData, val_data_prof = train_test_split(val_data_prof, test_size=0.66, random_state=18)

        train_data_all.extend(moreTrainingData)

        train_data_all.extend(train_data_all)
        train_data_all.extend(train_data_amateur)

        # preprocess train
        resize_hist_equalization_and_sharpen(train_data_all, train=True, resize_label=True,
                                             resize_h_img=resize_h_img, resize_w_img=resize_w_img,
                                             resize_h_label=resize_h_label, resize_w_label=resize_w_label,
                                             hist_eq=True, sharpen=False)

        # preprocess validation without resize (like test)
        resize_hist_equalization_and_sharpen(val_data_prof, train=True, resize_label=True,
                                             resize_h_img=resize_h_img, resize_w_img=resize_w_img,
                                             resize_h_label=resize_h_label, resize_w_label=resize_w_label,
                                             hist_eq=True, sharpen=False)
        
        network = model.train(prof_train=train_data_all, val_train=val_data_prof)

    return network


def train_box(load_weights, path_weights, resize_w_img, resize_h_img, resize_h_box, resize_w_box):
    if load_weights:
        network = model.load_weights(path_weights)

    else:
        # load data
        train_data = load_zipped_pickle('Data/train.pkl')

        # split data
        train_data_amateur, train_data_prof, val_data_prof = split_to_get_validation(train_data, split=0.1)

        # create train of both prof and amateur
        train_data_all = train_data_prof
        train_data_all.extend(train_data_amateur)

        # preprocess train
        preprocess_data_box(train_data_all, train=True, resize_box=True,
                            resize_h_img=resize_h_img, resize_w_img=resize_w_img,
                            resize_h_label=resize_h_box, resize_w_label=resize_w_box,
                            hist_eq=False)

        # preprocess validation without resize (like test)
        preprocess_data_box(val_data_prof, train=True, resize_box=True,
                            resize_h_img=resize_h_img, resize_w_img=resize_w_img,
                            resize_h_label=resize_h_box, resize_w_label=resize_w_box,
                            hist_eq=False)

        network = model_box.train(prof_train=train_data_all, val_train=val_data_prof)

    return network


def try_sample_and_predictions():
    samples = load_zipped_pickle('Data/sample.pkl')
    pred = load_zipped_pickle('Out/out_new.pkl')
    train_data = load_zipped_pickle('Data/train.pkl')
    test_data = load_zipped_pickle('Data/test.pkl')
    k = 10


def cut_box(path):
    my_pred = load_zipped_pickle('Out/my_predictions.pkl')
    for pred in my_pred:
        density = np.sum(pred['prediction'], axis=2).astype('float32')
        (w, h) = density.shape

        new_w, new_h = int(w / 50), int(h / 50)
        down = cv2.resize(density, (new_h, new_w), interpolation=cv2.INTER_AREA)

        up = cv2.resize(down, (h, w), interpolation=cv2.INTER_AREA)

        box = skimage.measure.block_reduce(density, (50, 50), np.max)
        box_up = cv2.resize(box, (h, w), interpolation=cv2.INTER_AREA)

        minus = box_up > np.quantile(np.unique(box_up.flatten()), 0.5)

        old_type = minus.dtype
        minus = minus.astype(np.uint8)
        minus = augmentions.zoom_at(minus, zoom=2)
        minus = minus.astype(old_type)

        for frame in range(pred['prediction'].shape[2]):
            # f, axarr = plt.subplots(1, 3)
            # axarr[0].imshow(pred['prediction'][:, :, frame], interpolation='nearest')

            pred['prediction'][:, :, frame] = pred['prediction'][:, :, frame] & minus

            # axarr[1].imshow(pred['prediction'][:, :, frame], interpolation='nearest')
            # axarr[2].imshow(minus, interpolation='nearest')
            #
            # plt.show()

        # f, axarr = plt.subplots(2, 3)
        # axarr[0, 0].imshow(density, interpolation='nearest')
        # axarr[0, 1].imshow(down, interpolation='nearest')
        # axarr[0, 2].imshow(up, interpolation='nearest')
        #
        # axarr[1, 0].imshow(minus, interpolation='nearest')
        # axarr[1, 1].imshow(box, interpolation='nearest')
        # axarr[1, 2].imshow(box_up, interpolation='nearest')
        #
        # plt.show()
    return my_pred


def train_box():
    resize_w_img, resize_h_img = 100, 100
    resize_w_box, resize_h_box = 100, 100

    network_box = train_box(load_weights=False, path_weights='Trained_boxes/signal_unet/ep-531.pth',
                            resize_w_img=resize_w_img, resize_h_img=resize_h_img,
                            resize_w_box=resize_w_box, resize_h_box=resize_h_box)


if __name__ == '__main__':
    # train_box()
    resize_w_img, resize_h_img = 400, 400
    resize_w_label, resize_h_label = 400, 400
    
    network = train(load_weights=False, path_weights='', #'Trained_small_model_512_euler/signal_unet/ep-107.pth', 
                    resize_w_img=resize_w_img, resize_h_img=resize_h_img,
                    resize_w_label=resize_w_label, resize_h_label=resize_h_label)
    
    predictions = test(network, resize_w_img=resize_w_img, resize_h_img=resize_h_img,
                       resize_w_label=resize_w_label, resize_h_label=resize_h_label)

    # save in correct format
    out_path = 'Out/out_new.pkl'
    # try_sample_and_predictions()
    save_zipped_pickle(predictions, out_path)

    # # # try_sample_and_predictions()
    # predictions = cut_box(out_path)
    # save_zipped_pickle(predictions, 'Out/my_predictions_400In_1024_size_Dice_it338_lowLoss_boxed.pkl')
