#!/usr/bin/env python2


from matplotlib import pyplot as plt
from random import randint, uniform
from pyceptron import Pyceptron
import sys




def line(w, x):
	w2=w[2]
	if w2 == 0:
		w2 = .000001
	return (w[1] * x + w[0]) / float(-1 * w2)


def classify(point, weights):
	point = [1] + list(point)

	def dot(u, v):
		result = 0.0
		for index in range(len(u)):
			result += u[index] * v[index]
		return result

	def sign(value):
		return -1 if value < 0 else 1

	return sign(dot(weights, point))


def render(points, weights, ideal):
	plt.axis([-10, 10, -10, 10])

	for point in points:
		if point[1] < 0:
			plt.plot(point[0][0], point[0][1], 'rv')
		else:
			plt.plot(point[0][0], point[0][1], 'g^')


	plt.plot([-10, 10], [line(ideal, -10), line(ideal, 10)], 'k-')
	plt.plot([-10, 10], [line(weights, -10), line(weights, 10)], 'b-')
	# print([0, 5], [line(weights, 0), line(weights, 5)])

	plt.show()




def main(argv=None):
	if argv == None:
		argv = sys.argv
	argc = len(argv)

	count = 500
	if argc >= 2:
		count = int(argv[1])

	# Our 'ideal' weights, used to create training data
	ideal = [34,77,81]

	# Create and classify some points
	points = []
	for _ in range(count):
		x = uniform(-10, 10)
		y = uniform(-10, 10)
		h = classify((x,y), ideal)
		points.append(((x, y), h))

	# Create and run perceptron
	pt = Pyceptron(2)
	pt.populate(points)
	pt.weights([randint(1,100), randint(1,100), randint(1,100)])
	if pt.train():
		print('Found a solution:')
		print(pt.weights())
		render(points, pt.weights(), ideal)
		return 0
	else:
		print('no soluion found...')
		return 1




if __name__ == '__main__':
	sys.exit(main())
