import cvxpy as cp
import numpy as np
np.set_printoptions(precision=3)

n = 5

np.random.seed(202055256)
alpha = np.round(np.random.rand(5), 2)
#나의 코드


x = cp.Variable(n)
obj = cp.Maximize(cp.sum(cp.log(x + alpha)))
prob = cp.Problem(obj, [x >= 0, cp.sum(x) == 1])
prob.solve()

print("alpha : ", alpha)
print("Sroblem status:\n", prob.status)
print("Curvature x :\n", x.curvature)
print("optimal value:\n", prob.value)
print("The  solution x is:\n", x.value)
print("A dual solution corresponding to the inequality is:\n", prob.constraints[0].dual_value) # lamda
print("A dual solution corresponding to the equality is:\n", prob.constraints[1].dual_value) # v

if prob.constraints[0].dual_value.all() >= 0:
    print("True")
else:
    print("False")

print("lammda * x = {0:.2f}".format(prob.constraints[0].dual_value.T @ x.value))

vv = prob.constraints[1].dual_value
x_list = []
lamda_list = []
for i in range(n):
    if vv >= 1/alpha[i]:
        lamda = vv - 1/alpha[i]
        xx = 0.0
        lamda_list.append(lamda)
        x_list.append(xx)
    else:
        lamda = 0.0
        xx = 1/(vv) - alpha[i]
        lamda_list.append(lamda)
        x_list.append(xx)

print("x = ", np.array(x_list))
print("lammda = ", np.array(lamda_list))


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

axis = np.arange(0.5, n+1.5,1)
index = axis+0.5
X = x.value.copy()
Y = alpha + X

A = np.concatenate((alpha,[alpha[-1]]))
X = np.concatenate((X,[X[-1]]))
Y = np.concatenate((Y,[Y[-1]]))

plt.xticks(index)
plt.xlim(0.5,n+0.5)
plt.ylim(0,1.5)
plt.step(axis,A,where='post', label =r'$\alpha$',lw=2)
plt.step(axis,Y,where='post', label=r'$\alpha + x$',lw=2)
plt.legend(loc='lower right')
plt.xlabel('N Number')
plt.ylabel('Value')
plt.title('Water Filling Solution')
plt.show()