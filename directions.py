def opp(direction):
    if direction == 'N':
        return "S"
    elif direction == 'S':
        return "N"
    elif direction == 'E':
        return "W"
    elif direction == 'W':
        return "E"
    else:
        assert False

def edge(elements, direction):
    if direction == 'N':
        return elements[-1][:]
    if direction == 'S':
        return elements[0][:]
    if direction == 'E':
        return elements[:][-1]
    if direction == 'W':
        return elements[:][0]
