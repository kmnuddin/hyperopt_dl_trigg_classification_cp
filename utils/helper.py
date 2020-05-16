from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, load_model, save_model
from sklearn.metrics import classification_report
import os
import matplotlib.pyplot as plt
import cv2
import numpy as np
import pandas as pd
from bson import json_util
import json

class Helper:

    def __init__(self, train_directory, test_directory, results_directory, validation_directory = None):
        self.train_directory = train_directory
        self.validation_directory = validation_directory
        self.test_directory = test_directory
        self.model_directory = 'models/'
        self.results_directory = results_directory
        self.logs_directory = 'logs/cnn-topomap-log-{}'
        self.y_true = []

        if not os.path.exists(self.model_directory):
            os.mkdir(self.model_directory)

        if not os.path.exists(self.results_directory):
            os.makedirs(self.results_directory)

        if not os.path.exists('logs'):
            os.mkdir('logs')


    def construct_data_generator(self, batch_size=128, target_size=(224,224), shuffle=False):

        datagen = ImageDataGenerator(rescale=1./255)

        train_it = datagen.flow_from_directory(self.train_directory, batch_size=batch_size,
                                           target_size=target_size, shuffle=shuffle)

        test_it = datagen.flow_from_directory(self.test_directory, batch_size=batch_size,
                                          target_size=target_size, shuffle=False)

        self.y_true = test_it.classes

        return train_it, test_it

    def construct_data_generator_w_validation(self, batch_size=128, target_size=(224,224), shuffle=True):

        datagen = ImageDataGenerator(rescale=1./255)

        train_it = datagen.flow_from_directory(self.train_directory, batch_size=batch_size,
                                           target_size=target_size, shuffle=shuffle )

        validation_it = datagen.flow_from_directory(self.validation_directory, batch_size=batch_size,
                                                target_size=target_size, shuffle=shuffle)

        test_it = datagen.flow_from_directory(self.test_directory, batch_size=batch_size,
                                          target_size=target_size, shuffle=False)

        self.y_true = test_it.classes

        return train_it, validation_it, test_it

    def plot_examples(self, example_type = 'train', classes = [1, 2, 3, 4, 5]):
        path = ''
        if example_type is 'train':
            path = self.train_directory
        if example_type is 'test':
            path = self.test_directory
        if example_type is 'validation':
            path = self.validation_directory


        img_cls_0 = os.listdir(path + str(classes[0]))
        img_cls_1 = os.listdir(path + str(classes[1]))
        img_cls_2 = os.listdir(path + str(classes[2]))
        img_cls_3 = os.listdir(path + str(classes[3]))
        img_cls_4 = os.listdir(path + str(classes[4]))

        rand_index = np.random.randint(len(img_cls_0))

        img1 = cv2.imread(path + str(classes[0]) + '/' + img_cls_0[rand_index])
        img2 = cv2.imread(path + str(classes[1]) + '/' + img_cls_1[rand_index])
        img3 = cv2.imread(path + str(classes[2]) + '/' + img_cls_2[rand_index])
        img4 = cv2.imread(path + str(classes[3]) + '/' + img_cls_3[rand_index])
        img5 = cv2.imread(path + str(classes[4]) + '/' + img_cls_4[rand_index])

        print(path + str(classes[0]) + img_cls_0[rand_index])
        fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(ncols = 5, figsize = (20, 5))

        ax1.imshow(img1)
        ax2.imshow(img2)
        ax3.imshow(img3)
        ax4.imshow(img4)
        ax5.imshow(img5)



    def load(self, path):
        model = load_model(path)
        return model

    def save(self, model, history, evaluator, y_prob, name):
        path_model = self.model_directory + name + '.h5'
        path_results = self.results_directory + name + '.txt'

        save_model(model, path_model)
        print('model saved, path: {}'.format(path_model))

        y_pred = np.argmax(y_prob, axis=1)

        report = pd.DataFrame(classification_report(self.y_true, y_pred, output_dict=True)).transpose()

        with open(path_results, 'a') as file:
            model.summary(print_fn = lambda x: file.write(x + '\n'))
            result = '{}:{}\n{}:{}'.format(model.metrics_names[0], evaluator[0], model.metrics_names[1], evaluator[1])
            file.write(result)
            file.write('\n')
            file.write('trained using {} epochs'.format(len(history.epoch)))
            file.write('\n')
            file.write(report.to_string())


    def print_json(self, result):
        """Pretty-print a jsonable structure (e.g.: result)."""
        print(json.dumps(
            result,
            default=json_util.default, sort_keys=True,
            indent=4, separators=(',', ': '),
        ))


    def save_json_result(self, model_name, result):
        """Save json to a directory and a filename."""
        result_name = '{}.txt.json'.format(model_name)
        if not os.path.exists(self.results_directory):
            os.makedirs(self.results_directory)
        with open(os.path.join(self.results_directory, result_name), 'w') as f:
            json.dump(
                result, f,
                default=json_util.default, sort_keys=True,
                indent=4, separators=(',', ': ')
            )


    def load_json_result(self, best_result_name):
        """Load json from a path (directory + filename)."""
        result_path = os.path.join(self.results_directory, best_result_name)
        with open(result_path, 'r') as f:
            return json.JSONDecoder().decode(
                f.read()
                # default=json_util.default,
                # separators=(',', ': ')
            )


    def load_best_hyperspace(self):
        results = [
            f for f in list(sorted(os.listdir(self.results_directory))) if 'json' in f
        ]
        if len(results) == 0:
            return None

        best_result_name = results[-1]
        return self.load_json_result(best_result_name)["space"]
