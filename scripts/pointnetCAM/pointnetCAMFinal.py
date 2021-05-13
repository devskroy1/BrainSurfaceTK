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

# from models.pointnet.src.models.pointnet2_regression_v2 import Net
from models.pointnet.src.models.pointnet2_classification import Net

PATH_TO_POINTNET = osp.join(osp.dirname(osp.realpath(__file__)), '..', '..', 'models', 'pointnet') + '/'

desiredLabel = 1  # The index of the class label the object should be tested against. It matches with the line numbers of the shapes.txt files e.g. line 1 = airplane etc.
numTestRuns = 500  # Amount of tests for the current test label object.
maxNumPoints = 2048  # How many points should be considered? [256/512/1024/2048] [default: 1024]
storeResults = False  # Should the results of the algorithm be stored to files or not.

#Currently applied to classification task
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

    #def drop_and_store_results(self, pointclouds_pl, labels_pl, sess, poolingMode, thresholdMode, numDeletePoints=None):
    def drop_and_store_results(self, poolingMode, thresholdMode, numDeletePoints=None):
        # # Some profiling
        # import time
        # start_time = time.time()
        # cpr.startProfiling()

        # pcTempResult = pointclouds_pl.copy()
        delCount = []
        vipPcPointsArr = []
        weightArray = []
        i = 0

        # My code
        model.load_state_dict(torch.load(PATH, map_location=self.device))
        model.eval()

        for data in test_loader:

            torch.cuda.empty_cache()
            # 1. Get predictions and loss
            data = data.to(self.device)

            pcTempResult = data
            totNumPoints = data.pos.size(0)

            # Multiply the class activation vector with a one hot vector to look only at the classes of interest.
            #class_activation_vector = tf.multiply(self.pred, tf.one_hot(indices=desiredClassLabel, depth=40))

            while True:
                i += 1
                print("ITERATION: ", i)

                print("data.y[:, 0].long()")
                print(data.y[:, 0].long())
                # Setup feed dict for current iteration
                # feed_dict = {self.pointclouds_pl: pcTempResult,
                #              self.labels_pl: labels_pl,
                #              self.is_training_pl: self.is_training}

                out, feature_vector = model(pcTempResult)
                #pred = out.max(dim=0)[1]
                pred = out.max(dim=1)[1]
                # with torch.set_grad_enabled(True):
                one_hot = F.one_hot(data.y[:, 0].float().clone().detach().requires_grad_(True).long(), -1)

                class_activation_vector = torch.mul(out, one_hot)

                #maxgradients = self.getGradient(sess, poolingMode, class_activation_vector, feed_dict)
                maxgradients = self.getGradient(poolingMode, class_activation_vector, feature_vector)

                # ops = {'pred': self.pred,
                #        'loss': self.classify_loss,
                #        'maxgradients': maxgradients}

                # ===================================================================
                # Evaluate over n batches now to get the accuracy for this iteration.
                # ===================================================================
                total_correct = 0
                total_seen = 0
                loss_sum = 0
                pcEvalTest = copy.deepcopy(pcTempResult)
                accuracies = torch.empty(numTestRuns, device=self.device)
                for n in range(numTestRuns):

                    #TODO: Add this in later
                    #pcEvalTest = provider.rotate_point_cloud_XYZ(pcEvalTest)

                    # feed_dict2 = {self.pointclouds_pl: pcEvalTest,
                    #               self.labels_pl: labels_pl,
                    #               self.is_training_pl: self.is_training}

                    out, feature_vector = model(pcEvalTest)
                    # pred = out.max(dim=0)[1]
                    pred = out.max(dim=1)[1]

                    print("pred")
                    print(pred)
                    # eval_prediction, eval_loss, heatGradient = sess.run([ops['pred'], ops['loss'], ops['maxgradients']],
                    #                                                     feed_dict=feed_dict2)

                   # eval_prediction = np.argmax(eval_prediction, 1)
                    loss = F.nll_loss(out, data.y[:, 0].long())
                    correct_nodes = pred.eq(data.y[:, 0].long()).sum().item()
                    total_nodes = data.num_nodes
                    accuracy = correct_nodes / total_nodes
                    accuracies[n] = accuracy
                    #correct = np.sum(eval_prediction == labels_pl)
                    # total_correct += correct
                    # total_seen += 1
                    # loss_sum += loss * batch_size

                accuracy = torch.mean(accuracies)
                print("GROUND TRUTH: ", data.y[:, 0].long())
                print("PREDICTION: ", pred)
                print("LOSS: ", loss)
                print("ACCURACY: ", accuracy.item())

               # accuracy = total_correct / float(total_seen)

                # Store data now if desired
                # if storeResults:
                #     curRemainingPoints = NUM_POINT - sum(delCount)
                #     storeAmountOfPointsRemoved(curRemainingPoints)
                #     storeAccuracyPerPointsRemoved(accuracy)

                # Stop iterating when the eval_prediction deviates from ground truth
                if data.y[:, 0].long() != pred and accuracy.item() <= 0.5:
                    print("GROUND TRUTH DEVIATED FROM PREDICTION AFTER %s ITERATIONS" % i)
                    break

                # Perform visual stuff here
                if thresholdMode == "+average" or thresholdMode == "+median" or thresholdMode == "+midrange":
                    resultPCloudThresh, vipPointsArr, Count = gch.delete_above_threshold(maxgradients, pcTempResult,
                                                                                         thresholdMode)
                if thresholdMode == "-average" or thresholdMode == "-median" or thresholdMode == "-midrange":
                    resultPCloudThresh, vipPointsArr, Count = gch.delete_below_threshold(maxgradients, pcTempResult,
                                                                                         thresholdMode)
                if thresholdMode == "nonzero":
                    resultPCloudThresh, vipPointsArr, Count = gch.delete_all_nonzeros(maxgradients, pcTempResult)
                if thresholdMode == "zero":
                    resultPCloudThresh, vipPointsArr, Count = gch.delete_all_zeros(maxgradients, pcTempResult)
                if thresholdMode == "+random" or thresholdMode == "-random":
                    resultPCloudThresh, vipPointsArr = gch.delete_random_points(maxgradients, pcTempResult,
                                                                                numDeletePoints[i])
                    Count = numDeletePoints[i]
                print("REMOVING %s POINTS." % Count)

                delCount.append(Count)
                vipPcPointsArr.extend(vipPointsArr[0])
                weightArray.extend(vipPointsArr[1])
                pcTempResult = copy.deepcopy(resultPCloudThresh)

            # Stop profiling and show the results
            endTime = time.time() - start_time
            storeAmountOfUsedTime(endTime)
            cpr.stopProfiling(numResults=20)
            print("TIME NEEDED FOR ALGORITHM: ", endTime)

            totalRemoved = sum(delCount)
            print("TOTAL REMOVED POINTS: ", totalRemoved)
            print("TOTAL REMAINING POINTS: ", NUM_POINT - totalRemoved)
            #         gch.draw_pointcloud(pcTempResult) #-- Residual point cloud
            #         gch.draw_NewHeatcloud(vipPcPointsArr, weightArray) #-- Important points only
            vipPcPointsArr.extend(pcTempResult[0])
            gch.draw_NewHeatcloud(vipPcPointsArr, weightArray)  # --All points combined
            return delCount

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
    # lr = 0.001
    # batch_size = 2
    # gamma = 0.9875
    # scheduler_step_size = 2
    # target_class = 'scan_age'
    # task = 'regression'
    # numb_epochs = 200
    # number_of_points = 10000

    lr = 0.001
    batch_size = 1
    gamma = 0.9875
    scheduler_step_size = 2
    target_class = 'gender'
    task = 'classification'
    numb_epochs = 200
    number_of_points = 10000
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

    PATH = PATH_TO_ROOT + '/runs/classification/pointcloud_grad_cam/models/model_best.pt'
    # PATH = PATH_TO_ROOT + '/runs/regression/Pointcloud_Grad_Cam/models/model_best.pt'
    adversarial_attack = AdversarialPointCloud(desired_class_label=desiredLabel, num_classes=num_labels, device=device)

    # adversarial_attack.drop_and_store_results(poolingMode="maxpooling", thresholdMode="+midrange")
    adversarial_attack.drop_and_store_results(poolingMode="maxpooling", thresholdMode="+average")