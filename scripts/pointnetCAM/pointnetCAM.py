import os.path as osp
PATH_TO_ROOT = osp.join(osp.dirname(osp.realpath(__file__)), '..', '..')
import sys
sys.path.append(PATH_TO_ROOT)
import numpy as np
import copy
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad
import gen_contrib_heatmap as gch
from models.pointnet.src.utils import get_comment, get_data_path, data

from models.pointnet.src.models.pointnet2_regression_v2 import Net
# from models.pointnet.src.models.pointnet2_classification import Net

PATH_TO_POINTNET = osp.join(osp.dirname(osp.realpath(__file__)), '..', '..', 'models', 'pointnet') + '/'

desiredLabel = 1  # The index of the class label the object should be tested against. It matches with the line numbers of the shapes.txt files e.g. line 1 = airplane etc.
numTestRuns = 500  # Amount of tests for the current test label object.
maxNumPoints = 2048  # How many points should be considered? [256/512/1024/2048] [default: 1024]
storeResults = False  # Should the results of the algorithm be stored to files or not.

#Currently applied to regression task

class AdversarialPointCloud():

    def __init__(self, desired_class_label, num_classes, device):
        self.desired_class_label = desired_class_label
        self.num_classes = num_classes
        self.device = device

    def getGradient(self, poolingMode, class_activation_vector, feature_vec):
        # Compute gradient of the class prediction vector w.r.t. the feature vector. Use class_activation_vector[classIndex] to set which class shall be probed.

        maxgradients = grad(outputs=class_activation_vector, inputs=feature_vec,
                            grad_outputs=torch.ones_like(class_activation_vector), allow_unused=True)[0]
        # maxgradients = grad(outputs=class_activation_vector, inputs=feature_vec,
        #                     grad_outputs=torch.ones_like(class_activation_vector), allow_unused=True)[0]

        # print("maxgradients")
        # print(maxgradients)
        # print("feature_vec shape")
        # print(feature_vec.shape)
        # print("maxgradients")
        # print(maxgradients)
        # print("maxgradients shape")
        # print(maxgradients.shape)
        #maxgradients = maxgradients.squeeze(dim=0).squeeze(dim=2)

        # Average pooling of the weights over all batches
        if poolingMode == "avgpooling":
            maxgradients = torch.mean(maxgradients, dim=1)  # Average pooling
        elif poolingMode == "maxpooling":
            maxgradients = torch.max(maxgradients, dim=1).values  # Max pooling
        #             maxgradients = tf.squeeze(tf.map_fn(lambda x: x[100:101], maxgradients)) # Stride pooling
        # Multiply with original pre maxpool feature vector to get weights
        #feature_vector = feature_vec.squeeze(dim=0).squeeze(dim=2) # Remove empty dimensions of the feature vector so we get [batch_size,1024]

        #multiply = tf.constant(feature_vector[1].get_shape().as_list())  # Feature vector matrix
        multiply = list(feature_vec[1].size())
        # print("list(feature_vec[1].size())")
        # print(multiply)
        # multiply = list(feature_vec.size())
        # print("list(feature_vec.size())")
        # print(multiply)

        # multMatrix = tf.reshape(tf.tile(maxgradients, multiply), [multiply[0], maxgradients.get_shape().as_list()[
        #     0]])  # Reshape [batch_size,] to [1024, batch_size] by copying the row n times
        # print("maxgradients.dtype")
        # print(maxgradients.dtype)
        # print("maxgradients")
        # print(maxgradients)
        # print("multiply.dtype")
        # print(multiply.dtype)
        # print("multiply")
        # print(multiply)
        tiled_max_gradients = np.tile(maxgradients.cpu().numpy(), np.asarray(multiply))

        tiledMaxGradsTensor = torch.from_numpy(tiled_max_gradients)
        multMatrix = tiledMaxGradsTensor.reshape(multiply[0], list(maxgradients.size())[0])

        #maxgradients = torch.matmul(feature_vector, multMatrix)  # Multiply [batch_size, 1024] x [1024, batch_size]
        maxgradients = torch.matmul(feature_vec.to(self.device), multMatrix.to(self.device))

        # maxgradients = tf.diag_part(
        #     maxgradients)  # Due to Matmul the interesting values are on the diagonal part of the matrix.

        maxgradients = torch.diagonal(maxgradients)

        # ReLU out the negative values
        relu = nn.ReLU()
        maxgradients = relu(maxgradients)
        return maxgradients

    def drop_and_store_results(self, poolingMode, thresholdMode, numDeletePoints=None):
        # Some profiling
        # import time
        # start_time = time.time()
        # cpr.startProfiling()

        #pcTempResult = pointclouds_pl.copy()
        delCount = []
        vipPcPointsArr = []
        weightArray = []
        #i = 0

        # Multiply the class activation vector with a one hot vector to look only at the classes of interest.
        # class_activation_vector = torch.multiply(self.pred, F.one_hot(self.desired_class_label, self.num_classes))

        #while True:
        #i += 1
        #print("ITERATION: ", i)
        # Setup feed dict for current iteration
        # feed_dict = {self.pointclouds_pl: pcTempResult,
        #              self.labels_pl: labels_pl,
        #              self.is_training_pl: self.is_training}

        #maxgradients = self.getGradient(sess, poolingMode, class_activation_vector, feed_dict)

        # maxgradients = self.getGradient(poolingMode, class_activation_vector)

        # ops = {'pred': self.pred,
        #        'loss': self.classify_loss,
        #        'maxgradients': maxgradients}

        # ===================================================================
        # Evaluate over n batches now to get the accuracy for this iteration.
        # ===================================================================
        total_correct = 0
        total_seen = 0
        loss_sum = 0
        #pcEvalTest = copy.deepcopy(pcTempResult)

        #My code
        model.load_state_dict(torch.load(PATH, map_location=device))
        model.eval()

        # correct_nodes = total_nodes = 0
        #total_loss = []
        #with torch.no_grad():
        # print("len(test_loader)")
        # print(len(test_loader))
        # for batch_idx, data in enumerate(test_loader):
        for data in test_loader:
            # print("batch_idx")
            # print(batch_idx)

            one_hot = None
            class_activation_vector = None
            feature_vector = None
            out = None
            maxgradients = None
            pred = None

            # if batch_idx == 2:
            #     break

            torch.cuda.empty_cache()
            # 1. Get predictions and loss
            data = data.to(device)

            # with torch.no_grad():
            out, feature_vector = model(data)
            # with torch.set_grad_enabled(True):
            one_hot = F.one_hot(data.y[:, 0].float().clone().detach().requires_grad_(True).long(), -1)
            #one_hot = F.one_hot(torch.tensor(data.y[:, 0].long()).long(), -1)

            class_activation_vector = torch.mul(out, one_hot)


            # feature_vector.requires_grad = True
            # class_activation_vector.requires_grad = True
            # print("feature_vector.requires_grad")
            # print(feature_vector.requires_grad)
            # print("class_activation_vector.requires_grad")
            # print(class_activation_vector.requires_grad)
            # class_activation_vector.retain_grad()
            # feature_vector.retain_grad()
            maxgradients = self.getGradient(poolingMode, class_activation_vector, feature_vector)


            # #
            # print("feature_vector shape")
            # print(feature_vector.shape)
            # #
            # print("data.x shape")
            # print(data.x.shape)
            # print("data.y shape")
            # print(data.y.shape)

            pred = out.max(dim=0)[1]

            # one_hot = F.one_hot(torch.tensor(data.y[:, 0].long()).long(), -1)

            # print("data.x shape")
            # print(data.x.shape)
            # print("data.y shape")
            # print(data.y.shape)
            # print("pred")
            # print(pred)
            # print("pred shape")
            # print(pred.shape)
            # print("out.shape")
            # print(out.shape)
            # print("one_hot")
            # print(one_hot)
            # print("one_hot shape")
            # print(one_hot.shape)

            #class_activation_vector = torch.multiply(pred, one_hot)

            # class_activation_vector = torch.mul(out, one_hot)

            # print("class_activation_vector shape")
            # print(class_activation_vector.shape)
            # class_activation_vector.requires_grad = True
            # feature_vector.requires_grad = True
            # class_activation_vector.retain_grad()
            # feature_vector.retain_grad()

            # maxgradients = self.getGradient(poolingMode, class_activation_vector, feature_vector)

            # print("maxgradients shape")
            # print(maxgradients.shape)
            #loss = F.nll_loss(out, data.y[:, 0].long())
            #total_loss.append(loss)

            correct_nodes = pred.eq(data.y[:, 0].long()).sum().item()
            total_nodes = data.num_nodes

            accuracy = correct_nodes / total_nodes
            # Store data now if desired
            if storeResults:
                curRemainingPoints = maxNumPoints - sum(delCount)
                self.storeAmountOfPointsRemoved(curRemainingPoints)
                self.storeAccuracyPerPointsRemoved(accuracy)

            # Perform visual stuff here
            if thresholdMode == "+average" or thresholdMode == "+median" or thresholdMode == "+midrange":
                resultPCloudThresh, vipPointsArr, Count = gch.delete_above_threshold(maxgradients, data,
                                                                                     thresholdMode)
            if thresholdMode == "-average" or thresholdMode == "-median" or thresholdMode == "-midrange":
                resultPCloudThresh, vipPointsArr, Count = gch.delete_below_threshold(maxgradients, data,
                                                                                     thresholdMode)
            if thresholdMode == "nonzero":
                resultPCloudThresh, vipPointsArr, Count = gch.delete_all_nonzeros(maxgradients, data)
            if thresholdMode == "zero":
                resultPCloudThresh, vipPointsArr, Count = gch.delete_all_zeros(maxgradients, data)
            if thresholdMode == "+random" or thresholdMode == "-random":
                resultPCloudThresh, vipPointsArr = gch.delete_random_points(maxgradients, data,
                                                                            numDeletePoints[i])
                Count = numDeletePoints[i]
            print("REMOVING %s POINTS." % Count)

            # print("vipPointsArr")
            # print(vipPointsArr)

            delCount.append(Count)
            # print("vipPointsArr[0] shape")
            # print(vipPointsArr[0].shape)
            vipPcPointsArr.extend(vipPointsArr[0])
            # print("vipPcPointsArr shape after extending")
            # print(vipPcPointsArr.shape)
            weightArray.extend(vipPointsArr[1])
            # print("weightArray after extending")
            # print(weightArray)
            pcTempResult = copy.deepcopy(resultPCloudThresh)


        #Original code

        # for _ in range(numTestRuns):
        #     #pcEvalTest = provider.rotate_point_cloud_XYZ(pcEvalTest)
        #     # feed_dict2 = {self.pointclouds_pl: pcEvalTest,
        #     #               self.labels_pl: labels_pl,
        #     #               self.is_training_pl: self.is_training}
        #     # eval_prediction, eval_loss, heatGradient = sess.run([ops['pred'], ops['loss'], ops['maxgradients']],
        #     #                                                     feed_dict=feed_dict2)
        #     # eval_prediction = np.argmax(eval_prediction, 1)
        #     # correct = np.sum(eval_prediction == labels_pl)
        #     # total_correct += correct
        #     # total_seen += 1
        #     # loss_sum += eval_loss * BATCH_SIZE
        #     eval_prediction = np.argmax(self.pred, 1)
        #     correct = np.sum(eval_prediction == labels)
        #     total_correct += correct
        #     total_seen += 1
        #     loss_sum += self.loss * self.batch_size
        #
        # print("GROUND TRUTH: ", self.desired_class_label)
        # print("PREDICTION: ", eval_prediction)
        # print("LOSS: ", self.loss)
        # print("ACCURACY: ", (total_correct / total_seen))
        # accuracy = total_correct / float(total_seen)

        # # Stop iterating when the eval_prediction deviates from ground truth
        # if self.desired_class_label != eval_prediction and accuracy <= 0.5:
        #     print("GROUND TRUTH DEVIATED FROM PREDICTION AFTER %s ITERATIONS" % i)
        #     break

        # Stop profiling and show the results
        # endTime = time.time() - start_time
        # storeAmountOfUsedTime(endTime)
        # cpr.stopProfiling(numResults=20)
        # print("TIME NEEDED FOR ALGORITHM: ", endTime)

        totalRemoved = sum(delCount)
        print("TOTAL REMOVED POINTS: ", totalRemoved)
        print("TOTAL REMAINING POINTS: ", maxNumPoints - totalRemoved)
        #         gch.draw_pointcloud(pcTempResult) #-- Residual point cloud
        #         gch.draw_NewHeatcloud(vipPcPointsArr, weightArray) #-- Important points only

        # print("pcTempResult before calling draw_NewHeatcloud()")
        # print(pcTempResult)

        #vipPcPointsArr.extend(pcTempResult[0])

        # print("vipPcPointsArr shape")
        # print(vipPcPointsArr.shape)
        # Should be [[3], [3], ...]
        # print("weightArray")
        # print(weightArray)
        gch.draw_NewHeatcloud(vipPcPointsArr, weightArray)  # --All points combined
        return delCount

    def storeTestResults(self, mode, total_correct, total_seen, loss_sum, pred_val):
        '''
        This function stores the test data into seperate files for later retrieval.
        '''
        #curShape = getShapeName(desired_class_label)

        savePath = os.path.join(os.path.split(__file__)[0], "testdata", self.desired_class_label)
        if not os.path.exists(savePath):
            os.makedirs(savePath)
        mean_loss = loss_sum / float(total_seen)
        accuracy = total_correct / float(total_seen)
        filePath = os.path.join(savePath, self.desired_class_label + "_" + str(numTestRuns) + "_" + str(mode) + "_meanloss")
        print("STORING FILES TO: ", filePath)
        tdh.writeResult(filePath, mean_loss)
        filePath = os.path.join(savePath, self.desired_class_label + "_" + str(numTestRuns) + "_" + str(mode) + "_accuracy")
        print("STORING FILES TO: ", filePath)
        tdh.writeResult(filePath, accuracy)
        filePath = os.path.join(savePath, self.desired_class_label + "_" + str(numTestRuns) + "_" + str(mode) + "_prediction")
        print("STORING FILES TO: ", filePath)

        #tdh.writeResult(filePath, pred_val)

        f = open(filePath, 'ab')
        WriteFloat(f, pred_val)
        f.close()

        print('eval mean loss: %f' % mean_loss)
        print('eval accuracy: %f' % accuracy)

    def storeAmountOfPointsRemoved(self, numPointsRemoved):
        '''
        This function stores the amount of points removed per iteration.
        '''
        #curShape = getShapeName(desired_class_label)
        savePath = os.path.join(os.path.split(__file__)[0], "testdata", "p-grad-CAM_ppi")
        if not os.path.exists(savePath):
            os.makedirs(savePath)
        filePath = os.path.join(savePath, self.desired_class_label + "_points_removed")
        print("STORING FILES TO: ", filePath)

        #tdh.writeResult(filePath, numPointsRemoved)

        f = open(filePath, 'ab')
        WriteFloat(f, numPointsRemoved)
        f.close()

    def storeAccuracyPerPointsRemoved(self, accuracy):
        '''
        This function stores the amount of points removed per iteration.
        '''
        #curShape = getShapeName(desired_class_label)
        savePath = os.path.join(os.path.split(__file__)[0], "testdata", "p-grad-CAM_ppi")
        if not os.path.exists(savePath):
            os.makedirs(savePath)
        filePath = os.path.join(savePath, self.desired_class_label + "_accuracy")
        print("STORING FILES TO: ", filePath)
        #tdh.writeResult(filePath, accuracy)

        f = open(filePath, 'ab')
        WriteFloat(f, accuracy)
        f.close()

if __name__ == "__main__":

    local_features = ['corrected_thickness', 'curvature', 'sulcal_depth']
    global_features = []

    recording = True
    REPROCESS = True

    data_nativeness = 'native'
    data_compression = "10k"
    data_type = 'white'
    hemisphere = 'both'

    # data_nativeness = 'native'
    # data_compression = "20k"
    # data_type = 'white'
    # hemisphere = 'left'

    additional_comment = ''

    #experiment_name = f'{data_nativeness}_{data_type}_{data_compression}_{hemisphere}_{additional_comment}'

    #experiment_name = 'native_white_10k_both_'
    #################################################
    ############ EXPERIMENT DESCRIPTION #############
    #################################################

    # 1. Model Parameters
    ################################################
    lr = 0.001
    batch_size = 2
    gamma = 0.9875
    scheduler_step_size = 2
    target_class = 'scan_age'
    task = 'regression'
    numb_epochs = 200
    number_of_points = 10000

    # lr = 0.001
    # batch_size = 2
    # gamma = 0.9875
    # scheduler_step_size = 2
    # target_class = 'gender'
    # task = 'classification'
    # numb_epochs = 200
    # number_of_points = 10000
    ################################################



    ###### SPECIFY PATH TO YOUR DATA_SPLIT PICKLE #####
    # 2. Get the data splits indices
    with open(PATH_TO_POINTNET + 'src/names.pk', 'rb') as f:
        indices = pickle.load(f)



    # 4. Get experiment description
    comment = get_comment(data_nativeness, data_compression, data_type, hemisphere,
                          lr, batch_size, local_features, global_features, target_class)

    print('=' * 50 + '\n' + '=' * 50)
    print(comment)
    print('=' * 50 + '\n' + '=' * 50)

    ##### SPECIFY YOUR DATA_FOLDER AND FILES_ENDING #####
    # 5. Perform data processing.
    data_folder, files_ending = get_data_path(data_nativeness, data_compression, data_type, hemisphere=hemisphere)

    train_dataset, test_dataset, validation_dataset, train_loader, test_loader, val_loader, num_labels = data(  data_folder,
                                                                                                                files_ending,
                                                                                                                data_type,
                                                                                                                target_class,
                                                                                                                task,
                                                                                                                REPROCESS,
                                                                                                                local_features,
                                                                                                                global_features,
                                                                                                                indices,
                                                                                                                batch_size,
                                                                                                                num_workers=2,
                                                                                                                data_nativeness=data_nativeness,
                                                                                                                data_compression=data_compression,
                                                                                                                hemisphere=hemisphere
                                                                                                                )

    # 6. Getting the number of features to adapt the architecture
    try:
        num_local_features = train_dataset[0].x.size(1)
    except:
        num_local_features = 0
    print('Unique labels found: {}'.format(num_labels))

    num_global_features = len(global_features)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Net(num_local_features, num_global_features=num_global_features).to(device)

    #PATH = PATH_TO_POINTNET + 'experiment_data/new/{}-99/best_acc_model.pt'.format(experiment_name)

    # PATH = PATH_TO_ROOT + '/pointnetModels/classification/model_best.pt'

    # PATH = PATH_TO_ROOT + '/runs/classification/pointcloud_grad_cam/models/model_best.pt'
    PATH = PATH_TO_ROOT + '/runs/regression/Pointcloud_Grad_Cam/models/model_best.pt'
    adversarial_attack = AdversarialPointCloud(desired_class_label=desiredLabel, num_classes=num_labels, device=device)

    adversarial_attack.drop_and_store_results(poolingMode="maxpooling", thresholdMode="+midrange")
