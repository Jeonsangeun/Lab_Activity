# import cvxpy as cp
#
# x = cp.Variable() #() dimension
# y = cp.Variable()
#
# constraints = [x + y == 1, x - y >= 1]
#
# obj = cp.Minimize((x - y) ** 2)
#
# prob = cp.Problem(obj, constraints)
# prob.solve()
# print("Problem Statuses : ", prob.status) # optimal 상태
# print("optimal (x + y == 1) dual variable", constraints[0].dual_value) # v
# print("optimal (x - y >= 1) dual variable", constraints[1].dual_value) # lammda
# print("x - y value", (x - y).value)

import cvxpy as cp
import numpy as np
np.set_printoptions(precision=3)

n = 10
B = 100.0

data_rate = np.array([0.5, 2, 1, 1.5, 1.5, 2, 2.5, 1.5, 1, 0.5])
channel_gain = np.array([0.75, 0.55, 0.35, 0.75, 0.15, 0.01, 0.35, 0.1, 0.45, 0.65])
TS_slot = np.array([2, 1, 2, 1, 4, 3, 2, 2, 1, 3])

#나의 코드
x = cp.Variable(n)

constA = []
constB = []
for i in range(1, n+1):
    constA.append(TS_slot[:i].T @ (data_rate[:i] - x[:i]))

for k in range(1, n+1):
    constB.append(TS_slot[:k].T @ (x[:k] - data_rate[:k]) - B)


constA = cp.hstack(constA)
constB = cp.hstack(constB)
constC = -x

obj = cp.Minimize(TS_slot.T @ ((cp.exp(x) - np.ones(n)) / channel_gain))
prob = cp.Problem(obj, [constA <= 0, constB <= 0, constC <= 0])
prob.solve()

print("Sroblem status:\n", prob.status)
print("Object DCP:\n", obj.is_dcp())
print("Curvature x :\n", x.curvature)
print("optimal value:\n", prob.value)
print("The  solution x is:\n", x.value)
print("A dual solution corresponding to the inequality_1 is:\n", prob.constraints[0].dual_value) # lamda
print("A dual solution corresponding to the inequality_2 is:\n", prob.constraints[1].dual_value) # v
print("A dual solution corresponding to the inequality_3 is:\n", prob.constraints[2].dual_value) # v



# #공식 홈페이지의 예제 코드와의 비교
# def water_filling(n, a, sum_x=1):
#     '''
#     Boyd and Vandenberghe, Convex Optimization, example 5.2 page 145
#     Water-filling.
#
#     This problem arises in information theory, in allocating power to a set of
#     n communication channels in order to maximise the total channel capacity.
#     The variable x_i represents the transmitter power allocated to the ith channel,
#     and log(α_i+x_i) gives the capacity or maximum communication rate of the channel.
#     The objective is to minimise -∑log(α_i+x_i) subject to the constraint ∑x_i = 1
#     '''
#
#     # Declare variables and parameters
#     x = cp.Variable(shape=n)
#     alpha = cp.Parameter(n, nonneg=True)
#     alpha.value = a
#
#     # Choose objective function. Interpret as maximising the total communication rate of all the channels
#     obj = cp.Maximize(cp.sum(cp.log(alpha + x)))
#
#     # Declare constraints
#     constraints = [x >= 0, cp.sum(x) - sum_x == 0]
#
#     # Solve
#     prob2 = cp.Problem(obj, constraints)
#     prob2.solve()
#     if(prob2.status=='optimal'):
#         return prob2.status, prob2.value, x.value, prob2.constraints[0].dual_value, prob2.constraints[1].dual_value
#     else:
#         return prob2.status, np.nan, np.nan
#
#
# state, value, opt_x, lamda, v = water_filling(n, alpha)
#
# print("================================")
# print("Problem status:\n", state)
# print("optimal value:\n", value)
# print("The  solution x is:\n", opt_x)
# print("A dual solution corresponding to the inequality is:\n", lamda) # lamda
# print("A dual solution corresponding to the equality is:\n", v) # v
#
import matplotlib.pylab as plt
axis = [0]
for i in range(1, len(TS_slot)+1):
    axis.append(float(axis[i-1] + TS_slot[i-1]))

axis = np.array(axis)
index = axis[:-1] + 1/2*TS_slot

print("Data_rate accumulated energy : ", np.dot(TS_slot.T, ((np.exp(data_rate)-1)/channel_gain)), "J")
print("Optimal accumulated energy : ", prob.value, "J")

X = x.value.copy()

C = np.log(1/channel_gain)
C = np.concatenate((C, [C[-1]]))

D = np.concatenate((data_rate, [data_rate[-1]]))
X = np.concatenate((X, [X[-1]]))

lammda = prob.constraints[0].dual_value
mu = prob.constraints[1].dual_value
sigma = []

for i in range(n):
    sigma.append(np.log(np.sum(lammda[i:] - mu[i:])))

rr = sigma - np.log(1/channel_gain)
print(sigma)
print("공식으로 계산한 전송 비 : ", rr)
print("CVXPY로 계산한 전송 비 : ", X[:n])

plt.xticks(index, range(1, n+1))
plt.xlim(0, axis[-1])
#plt.yscale('log')
#plt.ylim(0, 3000)
plt.step(axis, D, where='post', label=r'd(t)', lw=2)
plt.step(axis, X, where='post', label=r'r(t)', lw=2)
plt.xlabel('Epoch Number')
plt.ylabel('Transmission Rate (Mnats/sec)')
plt.title('Energy efficiency Optimization')
plt.legend(loc='upper right')
plt.show()
