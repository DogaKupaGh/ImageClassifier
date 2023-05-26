from app.src.main.service.GraphService import printGraph
from app.src.resources.Environments import pathSet, pathSetSplit, pathResultsMap, pathOutputs, pathModels, inputSizeW, \
    inputSizeH, epochsCount, trainPercentage, pathTestImage
from service.ModelService import trainModel, testModel

if __name__ == '__main__':

    # printGraph(trainModel(pathSet, pathSetSplit, pathResultsMap, pathOutputs, pathModels, inputSizeW, inputSizeH, epochsCount, trainPercentage), pathOutputs)
    # printGraph("eqgwgimytods.h5", pathOutputs)

    testModel("eqgwgimytods.h5", pathTestImage)
