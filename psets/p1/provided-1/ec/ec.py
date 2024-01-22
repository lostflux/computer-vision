import numpy as np
import math

def myHoughLinesP(image, rho, theta, threshold, minLineLength, maxLineGap):
    # Step 1: Edge detection
    edges = np.where(image > 0)
    edge_points = list(zip(edges[0], edges[1]))

    # Step 2: Hough Transform
    accumulator = {}
    for y, x in edge_points:
        for angle in np.arange(0, np.pi, theta):
            r = x * np.cos(angle) + y * np.sin(angle)
            r = round(r / rho) * rho
            angle = round(angle / theta) * theta
            accumulator[(r, angle)] = accumulator.get((r, angle), []) + [(x, y)]

    # Step 3: Extract lines
    lines = []
    for (r, angle), points in accumulator.items():
        if len(points) < threshold:
            continue
        points.sort(key=lambda x: x[0])
        start = points[0]
        end = points[0]
        for current in points[1:]:
            if current[0] - end[0] > maxLineGap:
                if end[0] - start[0] >= minLineLength:
                    lines.append((start, end))
                start = current
            end = current
        if end[0] - start[0] >= minLineLength:
            lines.append((start, end))

    return np.array(lines)
