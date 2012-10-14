class Pyceptron:


	def __init__(self, dimension=2):
		self._dimension = dimension
		self._points = []
		self._weights = [0] * (dimension + 1)
		self._steps = 0


	def populate(self, points=None):
		if points != None:
			self._points += points
		return self._points



	def weights(self, weights=None):
		if weights != None:
			self._weights = weights

		def normalize(v):
			from math import sqrt
			sum = 0.0
			for dim in v:
				sum += dim * dim
			length = sqrt(sum)
			if length == 0:
				return [0.0] * len(v)
			return [dim/length for dim in v]

		return normalize(self._weights)


	def steps(self, steps=None):
		if steps != None:
			self._steps = steps
		return self._steps


	def _ein(self, weights):
		errors = 0
		for point in self._points:
			if point[1] != self._classify(point[0], weights):
				errors += 1
		return float(errors) / len(self._points)

	
	def _pick(self, weights=None):
		if weights == None:
			weights = self._weights
		for point in self._points:
			if point[1] != self._classify(point[0], weights):
				return point
		return None


	def _update(self, point, direction):
		point = [1.0] + list(point)
		for i in range(len(point)):
			self._weights[i] += direction * point[i]


	def _classify(self, point, weights):
		point = [1] + list(point)

		def dot(u, v):
			result = 0.0
			for index in range(len(u)):
				result += u[index] * v[index]
			return result

		def sign(value):
			if value > 0:
				return 1
			if value < 0:
				return -1
			return 0

		return sign(dot(weights, point))


	def train(self, steps=None, ein=None, pocket=False):

		wpocket = [x for x in self._weights]

		while True:

			if steps != None:
				if steps == 0:
					if pocket == True:
						self._weights = wpocket
					return False
				steps -= 1
			self._steps += 1

			target = self._pick()

			if target == None:
				if pocket == True:
					self._weights = wpocket
				return True

			self._update(target[0], target[1])

			ep = 1
			ew = 1

			if ein != None or pocket == True:
				ew = self._ein(self._weights)

			if pocket == True:
				ep = self._ein(wpocket)
				if ew < ep:
					wpocket = [x for x in self._weights]

			if ein != None:
				if ew <= ein:
					if pocket == True:
						self._weights = wpocket
					return True
