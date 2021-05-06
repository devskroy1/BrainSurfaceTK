'''
Created on 19.05.2019
This file handles the processing of the maxpooled vector in order to see which
ones where the most contributing vector entries. This is then used tu generate
a heatmap similar to the Grad-CAM approach described here: https://arxiv.org/pdf/1610.02391.pdf
@author: Dennis Struhs
'''
import numpy as np
import torch
import torch.nn as nn
import random
import copy
import pyvista as pv
import open3d as o3d
from open3d import geometry, visualization, utility

def _return_workArr(inputArr):
    return inputArr[0][0]


def list_contrib_vectors(inputArr):
    workArr = _return_workArr(inputArr)
    testArr = set(workArr)
    print(testArr)


def count_occurance(inputArr):
    workArr = _return_workArr(inputArr)
    unique, counts = np.unique(workArr, return_counts=True)
    result = dict(zip(unique, counts))
    print(result)
    return result


def get_average(inputArr):
    valSum = 0
    count = 0
    for index in range(len(inputArr)):
        curVal = inputArr[index]
        if curVal > 0:
            valSum += curVal
            count += 1
    return (valSum / count)


def get_median(inputArr):
    locArr = copy.deepcopy(inputArr)
    locArr = locArr[locArr > 0]
    locArr.sort()
    median = locArr[int(len(locArr) / 2)]
    return median


def get_midrange(inputArr):
    locArr = copy.deepcopy(nn.Parameter(inputArr))
    #locArr = copy.deepcopy(inputArr)

    # locArr[locArr <= 0] = 0
    # locArr = locArr[locArr.nonzero()]
    print("Inside get_midrange()")
    print("locArr")
    print(locArr)
    locArr = locArr[locArr > 0]

    if len(locArr) == 0:
        minVal = 0
        maxVal = 0
    else:
        minVal = min(locArr)
        maxVal = max(locArr)
    result = (minVal + maxVal) / 2
    return result


def delete_max_point(inputheatMap, inputArr):
    locArr = copy.deepcopy(inputArr)
    maxWeight = max(inputheatMap)
    delIndex = np.where(inputheatMap == maxWeight)
    delPoint = locArr[0][delIndex]
    locArr = np.delete(locArr, delIndex, 1)
    return locArr, [delPoint, [maxWeight]], 1


def delete_all_nonzeros(inputheatMap, inputArr):
    locArr = copy.deepcopy(inputArr)
    pointArr = []
    weightArr = []
    candArr = []
    count = 0
    for index, eachItem in enumerate(inputheatMap):
        if eachItem > 0:
            candArr.append(index)
            pointArr.append(locArr[0][index])
            weightArr.append(eachItem)
            count += 1

    if len(candArr) > locArr.shape[1] or 10 > locArr.shape[1]:
        return locArr, [pointArr, weightArr], 0

    locArr = np.delete(locArr, candArr, 1)
    return locArr, [pointArr, weightArr], count


def delete_all_zeros(inputheatMap, inputArr):
    locArr = copy.deepcopy(inputArr)
    pointArr = []
    weightArr = []
    candArr = []
    count = 0
    for index, eachItem in enumerate(inputheatMap):
        if eachItem == 0:
            candArr.append(index)
            pointArr.append(locArr[0][index])
            weightArr.append(eachItem)
            count += 1
    locArr = np.delete(locArr, candArr, 1)
    return locArr, [pointArr, weightArr], count


def delete_random_points(inputheatMap, inputArr, numPoints):
    locArr = copy.deepcopy(inputArr)
    pointArr = []
    weightArr = []
    randomArr = random.sample(range(inputArr.shape[1]), numPoints)
    for curIndex in randomArr:
        pointArr.append(locArr[0][curIndex])
        weightArr.append(inputheatMap[curIndex])
    locArr = np.delete(locArr, randomArr, 1)
    return locArr, [pointArr, weightArr]


def delete_above_threshold(inputheatMap, inputArr, mode):
    # print("Inside delete_above_threshold()")
    # print("inputheatMap shape")
    # print(inputheatMap.shape)
    # print("inputArr")
    # print(inputArr)
    locArr = copy.deepcopy(inputArr)
    # print("locArr")
    # print(locArr)
    pointArr = []
    weightArr = []
    candArr = []
    allPointArr = []
    allWeightArr = []
    threshold = None
    count = 0

    # print("inputheatMap")
    # print(inputheatMap)

    if mode == "+average":
        threshold = get_average(inputheatMap)
    elif mode == "+median":
        threshold = get_median(inputheatMap)
    elif mode == "+midrange":
        threshold = get_midrange(inputheatMap)

    # print("theshold")
    # print(threshold)
    # print("locArr")
    # print(locArr)
    for index, eachItem in enumerate(inputheatMap):
        # print("index")
        # print(index)
        # print("eachItem")
        # print(eachItem)
        allPointArr.append(locArr.pos[index])
        allWeightArr.append(eachItem.item())
        if eachItem > threshold:
            # print("eachItem > thresh")
            candArr.append(index)
            # pointArr.append(locArr[0][index])
            # print("locArr.pos[index]")
            # print(locArr.pos[index])
            # print("locArr.pos[index] shape")
            # print(locArr.pos[index].shape)
            pointArr.append(locArr.pos[index])
            weightArr.append(eachItem.item())
            count += 1

    # print("locArr before np.delete")
    # print(locArr)
    locArr = np.delete(locArr.pos.detach().cpu().numpy(), candArr, axis=0)
    # print("locArr after np.delete")
    # print(locArr)
    # print("Just before returning from delete_above_threshold()")
    # print("pointArr")
    # print(pointArr)

    #locArr, pointArr, weightArr and count are for dropped points
    return locArr, [pointArr, weightArr], count, candArr, allPointArr, allWeightArr


def delete_below_threshold(inputheatMap, inputArr, mode):
    locArr = copy.deepcopy(inputArr)
    candArr = []
    pointArr = []
    weightArr = []
    threshold = None
    count = 0

    if mode == "-average":
        threshold = get_average(inputheatMap)
    elif mode == "-median":
        threshold = get_median(inputheatMap)
    elif mode == "-midrange":
        threshold = get_midrange(inputheatMap)

    for index, eachItem in enumerate(inputheatMap):
        if eachItem < threshold:
            candArr.append(index)
            pointArr.append(locArr[0][index])
            weightArr.append(eachItem)
            count += 1

    if len(candArr) > locArr.shape[1] or 10 > locArr.shape[1]:
        print("SIZE IS TOO SMALL!!! RETURNING UNCHANGED ARRAY!")
        return locArr, [pointArr, weightArr], 0

    locArr = np.delete(locArr, candArr, 1)

    return locArr, [pointArr, weightArr], count


def truncate_to_threshold(inputArr, mode):
    newArr = []
    counter = 0

    if mode == "+average" or mode == "-average":
        threshold = get_average(inputArr)
    elif mode == "+median" or mode == "-median":
        threshold = get_median(inputArr)
    elif mode == "+midrange" or mode == "-midrange":
        threshold = get_midrange(inputArr)

    for index in range(len(inputArr)):
        curVal = inputArr[index]
        if curVal > threshold:
            newArr.append(threshold)
            counter += 1
        else:
            newArr.append(inputArr[index])
    return newArr


def draw_heatcloud(inpCloud, hitCheckArr, mode):
    hitCheckArr = truncate_to_threshold(hitCheckArr, mode)
    pColors = np.zeros((len(hitCheckArr), 3), dtype=float)
    maxColVal = max(hitCheckArr)
    for index in range(len(hitCheckArr)):
        try:
            curVal = hitCheckArr[index]
            if curVal == 0:
                pColors[index] = [0, 0, 0]
            else:
                red = curVal / maxColVal
                green = 1 - (curVal / maxColVal)
                pColors[index] = [red, green, 0]
        except:
            print("INVALID VALUE FOR INDEX: ", index)
            pColors[index] = [0, 0, 0]

    pcd = geometry.PointCloud()
    pcd.points = utility.Vector3dVector(inpCloud[0])
    pcd.colors = utility.Vector3dVector(pColors)
    visualization.draw_geometries([pcd])


def draw_NewHeatcloud(inputPCArray, inputWeightArray, dropPointsArray, allPointsArr, allWeightArr, totNumPoints, device):
    #     inputWeightArray = truncate_to_threshold( np.array(inputWeightArray), "+midrange" )
    # pColors = np.zeros((len(inputPCArray), 3), dtype=float)
    pColors = np.zeros((len(allPointsArr), 3), dtype=float)
    # maxColVal = max(inputWeightArray)
    maxColVal = max(allWeightArr)
    # for index in range(len(inputPCArray)):
    for index in range(len(allPointsArr)):
        try:
            # curVal = inputWeightArray[index]
            curVal = allWeightArr[index]
            if curVal == 0:
                pColors[index] = [0, 0, 0]
            else:
                red = curVal / maxColVal
                green = 1 - (curVal / maxColVal)
                pColors[index] = [red, green, 0]
        except:
            #             print( "INVALID VALUE FOR INDEX: ", index )
            pColors[index] = [0, 0, 0]

    # pcd = geometry.PointCloud()
    # inputPCArray = torch.stack(inputPCArray)
    # # print("inputPCArray shape")
    # # print(inputPCArray.shape)
    # npInputPCArray = inputPCArray.cpu().detach().numpy()
    # # print("npInputPCArray shape")
    # # print(npInputPCArray.shape)
    # pcd.points = utility.Vector3dVector(npInputPCArray)
    # pcd.colors = utility.Vector3dVector(pColors)

    pcd = geometry.PointCloud()
    allPointsArr = torch.stack(allPointsArr)
    # print("inputPCArray shape")
    # print(inputPCArray.shape)
    npAllPointsArr = allPointsArr.cpu().detach().numpy()
    # print("npInputPCArray shape")
    # print(npInputPCArray.shape)
    pcd.points = utility.Vector3dVector(npAllPointsArr)
    pcd.colors = utility.Vector3dVector(pColors)

    heat_cloud_ply_filename = "/vol/bitbucket/sr4617/ForkedBrainSurfaceTK/pointnetHeatClouds/newPointnetClassifcnHeatCloud.ply"
    
    o3d.io.write_point_cloud(heat_cloud_ply_filename, pcd)
    mesh = pv.read(heat_cloud_ply_filename)

    dropped_points = torch.zeros(totNumPoints, device=device)
    for i in range(len(dropPointsArray)):
        drop_index = dropPointsArray[i]
        dropped_points[drop_index] = 1.0

    dropped_points_numpy = dropped_points.detach().cpu().numpy()
    mesh.point_arrays['dropped points'] = dropped_points_numpy
    heat_cloud_vtk_filename = "/vol/bitbucket/sr4617/ForkedBrainSurfaceTK/pointnetHeatClouds/newPointnetClassifcnHeatCloud.vtk"
    mesh.save(heat_cloud_vtk_filename)

    #visualization.draw_geometries([pcd])


def draw_pointcloud(inputPointCloudArr):
    pcd = geometry.PointCloud()
    pcd.points = utility.Vector3dVector(inputPointCloudArr[0])
    pcd.colors = utility.Vector3dVector(np.zeros((len(inputPointCloudArr[0]), 3), dtype=float))
    visualization.draw_geometries([pcd])

# arr = np.ndarray(shape=(1,1,6), dtype=int)
# pColors = np.zeros((1024,3),dtype=int)
# pColors[0] = [1, 255, 2]
# print(pColors)