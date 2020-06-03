import numpy

def getSame(output, target):
    pos1, pos2 = 0, 0
    id1, id2 = 0, 0
    right = 0
    len_sentence = sum([len(io) for io in output])
    while pos1 < len_sentence and pos2 < len_sentence:
        if pos1 < pos2:
            pos1 += len(output[id1])
            id1 += 1
        elif pos2 < pos1:
            pos2 += len(target[id2])
            id2 += 1
        else:
            if len(output[id1]) == len(target[id2]):
                right += 1
            pos1 += len(output[id1])
            id1 += 1
            pos2 += len(target[id2])
            id2 += 1
    return right

if __name__ == "__main__":
    print(getSame(['1', '23', '45', '6', '789'], ['12', '3', '456', '789']))

