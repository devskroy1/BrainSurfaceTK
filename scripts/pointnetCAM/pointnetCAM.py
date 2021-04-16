import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad

class AdversialPointCloud():

    def __init__(self, final_feature_vector, pred, desiredClassLabel):
        self.final_feature_vector = final_feature_vector
        self.pred = pred
        self.desiredClassLabel = desiredClassLabel

    def getGradient(self, poolingMode, class_activation_vector):
        # Compute gradient of the class prediction vector w.r.t. the feature vector. Use class_activation_vector[classIndex] to set which class shall be probed.
        maxgradients = grad(outputs=class_activation_vector, inputs=self.final_feature_vector)[0]

        maxgradients = maxgradients.squeeze(dim=0).squeeze(dim=2)

        # Average pooling of the weights over all batches
        if poolingMode == "avgpooling":
            maxgradients = torch.mean(maxgradients, dim=1)  # Average pooling
        elif poolingMode == "maxpooling":
            maxgradients = torch.max(maxgradients, dim=1)  # Max pooling
        #             maxgradients = tf.squeeze(tf.map_fn(lambda x: x[100:101], maxgradients)) # Stride pooling
        # Multiply with original pre maxpool feature vector to get weights
        feature_vector = self.feature_vec.squeeze(dim=0).squeeze(dim=2) # Remove empty dimensions of the feature vector so we get [batch_size,1024]

        #multiply = tf.constant(feature_vector[1].get_shape().as_list())  # Feature vector matrix
        multiply = list(feature_vector[1].size())

        # multMatrix = tf.reshape(tf.tile(maxgradients, multiply), [multiply[0], maxgradients.get_shape().as_list()[
        #     0]])  # Reshape [batch_size,] to [1024, batch_size] by copying the row n times
        tiledMaxGradsTensor = torch.from_numpy(np.tile(maxgradients, multiply))
        multMatrix = tiledMaxGradsTensor.reshape(multiply[0], list(maxgradients.size())[0])

        maxgradients = torch.matmul(feature_vector, multMatrix)  # Multiply [batch_size, 1024] x [1024, batch_size]

        # maxgradients = tf.diag_part(
        #     maxgradients)  # Due to Matmul the interesting values are on the diagonal part of the matrix.

        maxgradients = torch.diagonal(maxgradients)

        # ReLU out the negative values
        relu = nn.ReLU()
        maxgradients = relu(maxgradients)
        return maxgradients

    def drop_and_store_results(self, pointclouds_pl, labels_pl, sess, poolingMode, thresholdMode, numDeletePoints=None):
        # Some profiling
        # import time
        # start_time = time.time()
        # cpr.startProfiling()

        pcTempResult = pointclouds_pl.copy()
        delCount = []
        vipPcPointsArr = []
        weightArray = []
        i = 0

        # Multiply the class activation vector with a one hot vector to look only at the classes of interest.
        class_activation_vector = torch.multiply(self.pred, F.one_hot(self.desiredClassLabel, 40))

        while True:
            i += 1
            print("ITERATION: ", i)
            # Setup feed dict for current iteration
            # feed_dict = {self.pointclouds_pl: pcTempResult,
            #              self.labels_pl: labels_pl,
            #              self.is_training_pl: self.is_training}

            #maxgradients = self.getGradient(sess, poolingMode, class_activation_vector, feed_dict)

            maxgradients = self.getGradient(poolingMode, class_activation_vector)

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
            for _ in range(numTestRuns):
                pcEvalTest = provider.rotate_point_cloud_XYZ(pcEvalTest)
                feed_dict2 = {self.pointclouds_pl: pcEvalTest,
                              self.labels_pl: labels_pl,
                              self.is_training_pl: self.is_training}
                eval_prediction, eval_loss, heatGradient = sess.run([ops['pred'], ops['loss'], ops['maxgradients']],
                                                                    feed_dict=feed_dict2)
                eval_prediction = np.argmax(eval_prediction, 1)
                correct = np.sum(eval_prediction == labels_pl)
                total_correct += correct
                total_seen += 1
                loss_sum += eval_loss * BATCH_SIZE

            print("GROUND TRUTH: ", getShapeName(desiredClassLabel))
            print("PREDICTION: ", getPrediction(eval_prediction))
            print("LOSS: ", eval_loss)
            print("ACCURACY: ", (total_correct / total_seen))
            accuracy = total_correct / float(total_seen)

            # Store data now if desired
            if storeResults:
                curRemainingPoints = NUM_POINT - sum(delCount)
                storeAmountOfPointsRemoved(curRemainingPoints)
                storeAccuracyPerPointsRemoved(accuracy)

            # Stop iterating when the eval_prediction deviates from ground truth
            if desiredClassLabel != eval_prediction and accuracy <= 0.5:
                print("GROUND TRUTH DEVIATED FROM PREDICTION AFTER %s ITERATIONS" % i)
                break

            # Perform visual stuff here
            if thresholdMode == "+average" or thresholdMode == "+median" or thresholdMode == "+midrange":
                resultPCloudThresh, vipPointsArr, Count = gch.delete_above_threshold(heatGradient, pcTempResult,
                                                                                     thresholdMode)
            if thresholdMode == "-average" or thresholdMode == "-median" or thresholdMode == "-midrange":
                resultPCloudThresh, vipPointsArr, Count = gch.delete_below_threshold(heatGradient, pcTempResult,
                                                                                     thresholdMode)
            if thresholdMode == "nonzero":
                resultPCloudThresh, vipPointsArr, Count = gch.delete_all_nonzeros(heatGradient, pcTempResult)
            if thresholdMode == "zero":
                resultPCloudThresh, vipPointsArr, Count = gch.delete_all_zeros(heatGradient, pcTempResult)
            if thresholdMode == "+random" or thresholdMode == "-random":
                resultPCloudThresh, vipPointsArr = gch.delete_random_points(heatGradient, pcTempResult,
                                                                            numDeletePoints[i])
                Count = numDeletePoints[i]
            print("REMOVING %s POINTS." % Count)

            print("vipPointsArr")
            print(vipPointsArr)

            delCount.append(Count)
            vipPcPointsArr.extend(vipPointsArr[0])
            weightArray.extend(vipPointsArr[1])
            print("weightArray after extending")
            print(weightArray)
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
        print("vipPcPointsArr")
        print(vipPcPointsArr)
        print("weightArray")
        print(weightArray)
        gch.draw_NewHeatcloud(vipPcPointsArr, weightArray)  # --All points combined
        return delCount