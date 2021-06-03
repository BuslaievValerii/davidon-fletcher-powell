import math
import numpy as np
import matplotlib.pyplot as plt

LAMBDA0 = 0
COUNTER = 0
SVEN_PAR = 0.25
EPSILON = 0.01
PRECISE = 0.01
H = 0.001
R = 0.0001
A0 = np.matrix([[1, 0], [0, 1]])
X0 = (-1.2, 0)

def penalty(x1, x2):
	try:
		log = math.log(g(x1, x2))
	except:
		log = -1000
	return -log


def f(x1, x2):
	global COUNTER
	COUNTER += 1

	foo = (1-x1)**2 + 100*((x2-x1**2)**2) + R*penalty(x1, x2)
	return foo


def g(x1, x2):
	return 1-(x1+1)**2-x2**2


def grad(x1, x2):
	deriv1 = (f(x1+H, x2) - f(x1-H, x2))/(2*H)
	deriv2 = (f(x1, x2+H) - f(x1, x2-H))/(2*H)
	return (deriv1, deriv2)


def norm(x1, x2):
	return math.sqrt(x1**2 + x2**2)


def delta_lambda(x1, x2, s1, s2):
	return SVEN_PAR*(norm(x1, x2)/norm(s1, s2))


def sven_approximate(lambda_in, lambda_out, x1, x2, s):
	lambdas = [(lambda_in, True), (lambda_out, False)]
	while not (lambdas[-1][1] == False and lambdas[-2][1] == False):

		true_lambda, false_lambda = 0, 0
		for el in lambdas:
			if el[1]:
				true_lambda = el[0]
			else:
				false_lambda = el[0]

		new_lambda = (true_lambda + false_lambda)/2
		if g(x1+s[0]*new_lambda, x2+s[1]*new_lambda) < 0:
			lambdas.append((new_lambda, False))
		else:
			lambdas.append((new_lambda, True))

		if len(lambdas) > 100:
			for el in lambdas[::-1]:
				if el[1]:
					return el[0]

	return lambdas[-3][0]


def alg_sven(x1, x2, delta, lambda0, s):
	lambdas = [lambda0]
	f_0 = f(x1+s[0]*lambda0, x2+s[1]*lambda0)

	lambda_left = lambda0 - delta
	f_left = f(x1+s[0]*lambda_left, x2+s[1]*lambda_left)

	lambda_right = lambda0 + delta
	f_right = f(x1+s[0]*lambda_right, x2+s[1]*lambda_right)

	if f_left < f_0:
		mode = 'left'
		lambdas.append(lambda_left)
	else:
		mode = 'right'
		lambdas.append(lambda_right)

	_continue = True
	out_lambda = 0
	while f(x1+s[0]*lambdas[-1], x2+s[1]*lambdas[-1]) < f(x1+s[0]*lambdas[-2], x2+s[1]*lambdas[-2]) and _continue:
		
		if mode == 'left':
			new_lambda = lambdas[-1] - delta*(2**(len(lambdas)-1))
		else:
			new_lambda = lambdas[-1] + delta*(2**(len(lambdas)-1))

		if g(x1+s[0]*new_lambda, x2+s[1]*new_lambda) >= 0:
			lambdas.append(new_lambda)
		else:
			_continue = False
			out_lambda = new_lambda

	if out_lambda == 0:
		new_lambda = (lambdas[-1]+lambdas[-2])/2
		if f(x1+s[0]*new_lambda, x2+s[1]*new_lambda) < f(x1+s[0]*lambdas[-2], x2+s[1]*lambdas[-2]):
			a, b = lambdas[-2], lambdas[-1]
		else:
			idx = -2 if len(lambdas)<3 else -3 
			a, b = lambdas[idx], new_lambda

	else:
		new_lambda = (lambdas[-1]+out_lambda)/2
		while g(x1+s[0]*new_lambda, x2+s[1]*new_lambda) < 0:
			new_lambda = (lambdas[-1]+new_lambda)/2

		if f(x1+s[0]*new_lambda, x2+s[1]*new_lambda) > f(x1+s[0]*lambdas[-1], x2+s[1]*lambdas[-1]):
			if f(x1+s[0]*new_lambda, x2+s[1]*new_lambda) < f(x1+s[0]*lambdas[-2], x2+s[1]*lambdas[-2]):
				a, b = lambdas[-1], new_lambda
			else:
				a, b = lambdas[-1], lambdas[-2]
		
		else:
			lambda_approx = sven_approximate(new_lambda, out_lambda, x1, x2, s)
			a, b = new_lambda, lambda_approx

	return min(a,b), max(a,b)


def golden(x1, x2, s, a, b, epsilon):
	L = abs(b-a)
	while L > epsilon:
		x1_gold = a + 0.382*L
		x2_gold = a + 0.618*L

		f_x1 = f(x1+s[0]*x1_gold, x2+s[1]*x1_gold)
		f_x2 = f(x1+s[0]*x2_gold, x2+s[1]*x2_gold)

		center = min(f_x1, f_x2)
		if center == f_x1:
			b = x2_gold
		else:
			a = x1_gold
		L = abs(b-a)

	return (a + b)/2


def dsk(x1, x2, s, a, b, epsilon):
	stop = False
	xm = (a + b)/2
	f_a = f(x1+s[0]*a, x2+s[1]*a)
	f_b = f(x1+s[0]*b, x2+s[1]*b)
	f_xm = f(x1+s[0]*xm, x2+s[1]*xm)
	x_star = xm + ((b-xm)*(f_a-f_b))/(2*(f_a-2*f_xm+f_b))
	assert a < x_star < b, f"{a}, {x_star}, {b}, {f_a}, {f_xm}, {f_b}"
	f_xstar = f(x1+s[0]*x_star, x2+s[1]*x_star)

	stop = abs(f_xm-f_xstar) <= epsilon and abs(xm-x_star) <= epsilon

	while not stop:
		dots = [(a, f_a), (xm, f_xm), (x_star, f_xstar), (b, f_b)]
		dots = sorted(dots, key=lambda p:p[0])

		if dots[1][1] <= dots[2][1]:
			a = dots[0][0]
			xm = dots[1][0]
			b = dots[2][0]
		else:
			a = dots[1][0]
			xm = dots[2][0]
			b = dots[3][0]

		assert a < xm < b, f"{a}, {xm}, {b}"

		f_a = f(x1+s[0]*a, x2+s[1]*a)
		f_b = f(x1+s[0]*b, x2+s[1]*b)
		f_xm = f(x1+s[0]*xm, x2+s[1]*xm)

		a1 = (f_xm-f_a)/(xm-a)
		a2 = ((f_b-f_a)/(b-a)-(f_xm-f_a)/(xm-a))/(b-xm)

		x_star = (a+xm)/2 - a1/(2*a2)
		f_xstar = f(x1+s[0]*x_star, x2+s[1]*x_star)
		stop = abs(f_xm-f_xstar) <= epsilon and abs(xm-x_star) <= epsilon

	return x_star


def count_s(x1, x2, A):
	gradient = np.matrix(grad(x1, x2))
	S = -A@gradient.T
	return S.item(0), S.item(1)


def x_delta(X1, X2):
	return X2[0]-X1[0], X2[1]-X1[1]


def g_delta(X1, X2):
	g1 = grad(X1[0], X1[1])
	g2 = grad(X2[0], X2[1])
	return g2[0]-g1[0], g2[1]-g1[1]


def count_A(A, X1, X2):
	g_del = np.matrix(g_delta(X1, X2)).T
	x_del = np.matrix(x_delta(X1, X2)).T
	first = (x_del @ x_del.T)/(x_del.T @ g_del)
	second = (A @ g_del @ g_del.T @ A)/(g_del.T @ A @ g_del)
	A_new = A + first - second
	return A_new


def iteration(X, A, func):
	S = count_s(X[0], X[1], A)
	delta = delta_lambda(X[0], X[1], S[0], S[1])
	a, b = alg_sven(X[0], X[1], delta, LAMBDA0, S)
	_lambda = func(X[0], X[1], S, a, b, EPSILON)
	return X[0]+_lambda*S[0], X[1]+_lambda*S[1]


def stop_grad(X):
	gradient = grad(X[0], X[1])
	norma = norm(gradient[0], gradient[1])
	if norma <= PRECISE:
		return True
	return False


def stop_func(X1, X2):
	norma = norm(X2[0]-X1[0], X2[1]-X1[1])/norm(X1[0], X1[1])
	absol = abs(f(X2[0], X2[1])-f(X1[0], X1[1]))
	if norma <= PRECISE and absol <= PRECISE:
		return True
	return False


def dfp():
	X1 = iteration(X0, A0, golden)
	print(f'X1 = {X1}')
	XS = [X0, X1]
	AS = [A0]

	while not stop_func(XS[-2], XS[-1]):
		A_new = count_A(AS[-1], XS[-2], XS[-1])
		X_new = iteration(XS[-1], A_new, golden)

		XS.append(X_new)
		AS.append(A_new)

		print(f'X{len(XS)-1} = {X_new}')

	print(f'\nFunction counted {COUNTER} times')

	dots_x = [X[0] for X in XS]
	dots_y = [X[1] for X in XS]

	plt.figure(0)
	plt.plot(dots_x, dots_y, marker='.')
	plt.show()

dfp()