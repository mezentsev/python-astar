import numpy
from heapq import *
import unittest


def heuristic(a, b):
    return (b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2

def astar(array, start, goal, reverse=False):
    '''
    Return shortest path in 2D list of points 
    or return False. First element in array
    is goal and last element is next after
    start point

    :array: list or numpy.array of points
    :start: (x,y) start point
    :goal: (x,y) end point
    '''

    if type(array) == numpy.ndarray:
        shape0 = array.shape[0]
        shape1 = array.shape[1]
    else:
        shape0 = len(array)
        shape1 = len(array[0])

    neighbors = [(0,1),(0,-1),(1,0),(-1,0),(1,1),(1,-1),(-1,1),(-1,-1)]

    close_set = set()
    came_from = {}
    gscore = {start:0}
    fscore = {start:heuristic(start, goal)}
    oheap = []

    heappush(oheap, (fscore[start], start))
    
    while oheap:

        current = heappop(oheap)[1]

        if current == goal:
            data = []
            while current in came_from:
                data.append(current)
                current = came_from[current]
            return data[::-1]  if reverse else data

        close_set.add(current)
        for i, j in neighbors:
            neighbor = current[0] + i, current[1] + j            
            tentative_g_score = gscore[current] + heuristic(current, neighbor)
            if 0 <= neighbor[0] < shape0:
                if 0 <= neighbor[1] < shape1:                
                    if array[neighbor[0]][neighbor[1]] == 1:
                        continue
                else:
                    # array bound y walls
                    continue
            else:
                # array bound x walls
                continue
                
            if neighbor in close_set and tentative_g_score >= gscore.get(neighbor, 0):
                continue
                
            if  tentative_g_score < gscore.get(neighbor, 0) or neighbor not in [i[1]for i in oheap]:
                came_from[neighbor] = current
                gscore[neighbor] = tentative_g_score
                fscore[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                heappush(oheap, (fscore[neighbor], neighbor))
                
    return False


class AstarTest(unittest.TestCase):
    def test_astar_list(self):
        nmap = list([
            [0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            [1,1,1,1,1,1,1,1,1,1,1,1,0,1],
            [0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            [1,0,1,1,1,1,1,1,1,1,1,1,1,1],
            [0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            [1,1,1,1,1,1,1,1,1,1,1,1,0,1],
            [0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            [1,0,1,1,1,1,1,1,1,1,1,1,1,1],
            [0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            [1,1,1,1,1,1,1,1,1,1,1,1,0,1],
            [0,0,0,0,0,0,0,0,0,0,0,0,0,0]])

        s = astar(nmap, (0,0), (0,5), reverse=True)
        self.assertEqual(s, [(0, 1), (0, 2), (0, 3), (0, 4), (0, 5)])

    def test_astar_numpy(self):
        nmap = numpy.array([
        #nmap = list([
            [0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            [1,1,1,1,1,1,1,1,1,1,1,1,0,1],
            [0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            [1,0,1,1,1,1,1,1,1,1,1,1,1,1],
            [0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            [1,1,1,1,1,1,1,1,1,1,1,1,0,1],
            [0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            [1,0,1,1,1,1,1,1,1,1,1,1,1,1],
            [0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            [1,1,1,1,1,1,1,1,1,1,1,1,0,1],
            [0,0,0,0,0,0,0,0,0,0,0,0,0,0]])
            
        s1 = astar(nmap, (0,0), (10,13))
        compare1 = [(10, 13), (9, 12), (8, 11), (8, 10), (8, 9), (8, 8), (8, 7), (8, 6), (8, 5), (8, 4), (8, 3), (8, 2), (7, 1), (6, 2), (6, 3), (6, 4), (6, 5), (6, 6), (6, 7), (6, 8), (6, 9), (6, 10), (6, 11), (5, 12), (4, 11), (4, 10), (4, 9), (4, 8), (4, 7), (4, 6), (4, 5), (4, 4), (4, 3), (4, 2), (3, 1), (2, 2), (2, 3), (2, 4), (2, 5), (2, 6), (2, 7), (2, 8), (2, 9), (2, 10), (2, 11), (1, 12), (0, 11), (0, 10), (0, 9), (0, 8), (0, 7), (0, 6), (0, 5), (0, 4), (0, 3), (0, 2), (0, 1)]
        self.assertEqual(s1, compare1)

        s2 = astar(nmap, (0,0), (0,1))
        self.assertEqual(s2, [(0,1)])

        s3 = astar(nmap, (0,0), (0,5), reverse=False)
        self.assertEqual(s3, [(0, 5), (0, 4), (0, 3), (0, 2), (0, 1)])

        s4 = astar(nmap, (0,0), (0,5), reverse=True)
        self.assertEqual(s4, [(0, 1), (0, 2), (0, 3), (0, 4), (0, 5)])

        s5 = astar(nmap, (0,0), (1,5), reverse=True)
        self.assertFalse(s5)

        s6 = astar(nmap, (-1,20), (100,500), reverse=True)
        self.assertFalse(s6)


if __name__ == '__main__':
    unittest.main()

