import numpy as np
from scipy.stats import t
from scipy.optimize import minimize
from scipy.optimize import NonlinearConstraint

delta = 0.01


# Implicit: theta = np.ones(shape=(d, 1)) / np.sqrt(d)
def get_contexts(K, d):
    cur_contexts = []
    for _ in range(K):
        cur_contexts.append(np.random.uniform(size=d))
    cur_contexts = np.array(cur_contexts)  # K*d
    x_norm = np.linalg.norm(cur_contexts, ord=2, axis=1, keepdims=True)  # K*1
    cur_contexts = cur_contexts / x_norm
    best_arm = np.argmax(np.sum(cur_contexts, axis=1))
    return cur_contexts, best_arm


# Implicit: Student t's df
def get_payoff(selected_context, theta, r=1):
    if r == 1:
        cur_scale = np.random.uniform(0, scale, size=1)
        cur_payoff = np.dot(selected_context, theta) + np.power(10, cur_scale) * np.random.standard_t(df, 1)
        return cur_payoff, cur_scale
    else:
        cur_payoff = np.dot(selected_context, theta) + np.power(10, np.random.uniform(-scale, scale, size=(1, r))) * np.random.standard_t(df, (1, r))
        return cur_payoff


def get_estimator(hat_theta, a, r):
    dis = [[0] * r for _ in range(r)]
    for i in range(r):
        for j in range(i + 1, r):
            dis[i][j] = np.dot(np.dot(hat_theta[:, i].reshape(1, d) - hat_theta[:, j].reshape(1, d), a),
                               hat_theta[:, i].reshape(1, d).T - hat_theta[:, j].reshape(1, d).T)
            dis[j][i] = dis[i][j]
    dis = np.array(dis)
    median = np.median(dis, axis=1, keepdims=True)
    index = int(np.argmin(median))
    return hat_theta[:, index]


# Global: delta
# Implicit: theta = np.ones(shape=(d, 1)) / np.sqrt(d)
def menu(round, d, K, theta, epsilon, v):
    # r = int(24 + 24 * np.log(round / delta))
    r = 51
    m = int(round / r) + 1
    a, inverse_a = np.eye(d), np.eye(d)
    b = np.zeros((d, r))  # d*r
    hat_theta = np.zeros((d, 1))
    beta = 1
    t = 0

    cul_regret = 0
    with open("./data/menu_" + str(round) + "_noise_scale_" + str(scale)+ "_path_" + str(i_path) + ".txt", "w") as f:
        while t < m:
            # print("round", t)
            cur_contexts, best_arm = get_contexts(K, d)
            # print(np.linalg.norm(hat_theta - theta))
            estimated_payoff = np.dot(cur_contexts, hat_theta)  # K*1
            width = []
            for i in range(K):
                width.append(beta * np.sqrt(np.dot(np.dot(cur_contexts[i], inverse_a), cur_contexts[i].T)))
            width = np.array(width).reshape(K, 1)
            # print(estimated_payoff.shape)
            estimated_payoff += width
            selected_arm = int(np.argmax(estimated_payoff))
            for _ in range(r):
                cul_regret += np.dot(cur_contexts[best_arm] - cur_contexts[selected_arm], theta)
                f.write(str(cul_regret[0]) + '\t')
            t += 1
            cur_payoff = get_payoff(cur_contexts[selected_arm], theta, r)  # 1*r
            for i in range(r):
                b[:, i] += cur_contexts[selected_arm] * cur_payoff[0][i]
            a += np.dot(cur_contexts[selected_arm].reshape(d, 1), cur_contexts[selected_arm].reshape(1, d))
            inverse_a = np.linalg.inv(a)
            estimators = np.dot(inverse_a, b)  # d*r
            hat_theta = get_estimator(estimators, a, r).reshape(d, 1)
            beta = 3 * (9 * d * v) ** (1 / (1 + epsilon)) * t ** ((1 - epsilon) / (2 * (1 + epsilon))) + 3
            beta /= 50


# Global: delta
def tofu(round, d, K, theta, epsilon, v):
    a, inverse_a = np.eye(d), np.eye(d)
    b = np.zeros((d, 1))  # d*1
    hat_theta = np.zeros((d, 1))
    historical_contexts = np.empty((round, d))
    historical_payoff = np.empty((round, 1))
    hat_y = np.empty((d, round))

    beta = 1
    t = 0

    cul_regret = 0
    with open("./data/tofu_" + str(round) + "_noise_scale_" + str(scale)+ "_path_" + str(i_path) + ".txt", "w") as f:
        while t < round:
            # print("round", t)
            trun = (v / np.log(2 * round / delta)) ** (1 / 2) * t ** ((1 - epsilon) / (2 * (1 + epsilon)))
            cur_contexts, best_arm = get_contexts(K, d)
            # print(np.linalg.norm(hat_theta - theta))
            estimated_payoff = np.dot(cur_contexts, hat_theta)  # K*1
            width = []
            for i in range(K):
                width.append(beta * np.sqrt(np.dot(np.dot(cur_contexts[i], inverse_a), cur_contexts[i].T)))
            width = np.array(width).reshape(K, 1)
            estimated_payoff += width
            selected_arm = int(np.argmax(estimated_payoff))
            cur_payoff, _ = get_payoff(cur_contexts[selected_arm], theta)  # 1*1
            historical_contexts[t] = cur_contexts[selected_arm]
            historical_payoff[t] = cur_payoff[0]
            cul_regret += np.dot(cur_contexts[best_arm] - cur_contexts[selected_arm], theta)
            f.write(str(cul_regret[0]) + '\t')
            t += 1
            a += np.dot(cur_contexts[selected_arm].reshape(d, 1), cur_contexts[selected_arm].reshape(1, d))
            u, sigma, z = np.linalg.svd(a)
            half_a = np.dot(np.dot(u, np.diag(sigma**(-0.5))), z)  # d*d
            mid = np.dot(half_a, historical_contexts[:t, :].T)  # d*t
            inverse_a = np.linalg.inv(a)
            for i in range(d):
                for j in range(t):
                    hat_y[i][j] = historical_payoff[j][0] if abs(historical_payoff[j][0] * mid[i][j]) <= trun else 0
                b[i] = np.dot(mid[i], hat_y[i, :t].T)
            hat_theta = np.dot(half_a, b)  # d*1
            beta = 4 * np.sqrt(d) * v ** (1 / (1 + epsilon)) * (np.log(2 * d * round / delta)) ** (epsilon / (1 + epsilon)) * t ** ((1 - epsilon) / (2 * (1 + epsilon))) + 1
            beta /= 5

            print("round", t, np.linalg.norm(hat_theta[:,0] - theta[:,0]))

huber_loss = lambda x, tau: x**2 / 2 if np.abs(x) <= tau else tau * np.abs(x) - tau**2 / 2
huber_derivative = lambda x, tau: -tau if x < -tau else x if x <= tau else tau
huber_hessian = lambda x, tau: float(np.abs(x) <= tau)

# Global: delta
def heavy_oful(round, d, K, theta, epsilon, v):
    hat_theta = np.zeros((d, 1))
    historical_contexts = np.empty((round, d))
    historical_payoff = np.empty((round, 1))
    historical_sigma = np.empty(round)
    historical_tau = np.empty(round)

    lamda = d / 100
    sigmamin = 1 / np.sqrt(round)
    c0 = 1 / np.sqrt(23 * np.log(2*round**2 /delta))
    c1 = np.log(3*round)**((1-epsilon)/(1+epsilon)) / (48 * np.log(2*round**2/delta)**(2/(1+epsilon)))
    kappa = d * np.log(1 + round / (d * lamda * sigmamin**2))
    tau0 = np.sqrt(2*kappa) * np.log(3*round)**((1-epsilon)/(2*(1+epsilon))) / np.log(2*round**2/delta)**(1/(1+epsilon))
    H = lamda * np.eye(d)
    inverse_H = 1 / lamda * np.eye(d)
    # loss = lambda theta_t: lamda / 2 * theta_t @ theta_t
    # loss_derivative = lambda theta_t: lamda * theta_t
    # loss_hessian = lambda theta_t: lamda * np.eye(d)

    beta = np.sqrt(lamda)
    t = 0

    cul_regret = 0

    with open("./data/heavy_oful_" + str(round) + "_noise_scale_" + str(scale)+ "_path_" + str(i_path) + ".txt", "w") as f:
        while t < round:
            # print("round", t)
            cur_contexts, best_arm = get_contexts(K, d)
            estimated_payoff = np.dot(cur_contexts, hat_theta)  # K*1
            width = []
            for i in range(K):
                width.append(beta * np.sqrt(np.dot(np.dot(cur_contexts[i], inverse_H), cur_contexts[i].T)))
            width = np.array(width).reshape(K, 1)
            estimated_payoff += width
            selected_arm = int(np.argmax(estimated_payoff))
            cur_payoff, cur_scale = get_payoff(cur_contexts[selected_arm], theta)  # 1*1
            nu = v * 10 ** cur_scale
            historical_contexts[t] = cur_contexts[selected_arm]
            historical_payoff[t] = cur_payoff[0]
            cul_regret += np.dot(cur_contexts[best_arm] - cur_contexts[selected_arm], theta)
            f.write(str(cul_regret[0]) + '\t')

            phinorm = np.sqrt(cur_contexts[selected_arm] @ inverse_H @ cur_contexts[selected_arm])
            historical_sigma[t] = np.max([nu, sigmamin, phinorm / c0, np.sqrt(phinorm) / (2 * c1 * kappa)**0.25])
            w = phinorm / historical_sigma[t]
            historical_tau[t] = tau0 * np.sqrt(1 + w**2) / w * (t+1)**((1-epsilon)/(2*(1+epsilon)))
            # adaptive huber regression
            loss = lambda theta_t: lamda / 2 * theta_t @ theta_t + np.sum(np.array([huber_loss((historical_payoff[i,0] - historical_contexts[i] @ theta_t) / historical_sigma[i], historical_tau[i]) for i in range(t+1)]))
            loss_derivative = lambda theta_t: lamda * theta_t - np.sum(np.array([historical_contexts[i] / historical_sigma[i] * huber_derivative((historical_payoff[i,0] - historical_contexts[i] @ theta_t) / historical_sigma[i], historical_tau[i]) for i in range(t+1)]), axis = 0)
            # loss_hessian = lambda theta_t: lamda * np.eye(d) + np.sum([1 / historical_sigma[i] * huber_hessian((historical_payoff[i,0] - historical_contexts[i] @ theta_t) / historical_sigma[i], historical_tau[i]) for i in range(t+1)], axis = 0)
            nonlinear_constraint = NonlinearConstraint(lambda x: x@x, -np.inf, 1.1)
            # res = minimize(fun=loss, x0=theta[:,0], method = 'Newton-CG', jac=loss_derivative, hess=loss_hessian)
            res = minimize(fun=loss, x0=theta[:,0], jac=loss_derivative, constraints=nonlinear_constraint)
            # res = minimize(fun=loss, x0=theta[:,0], jac=loss_derivative)
            hat_theta = res.x.reshape(d, 1)
            print("round", t, str(res.success), np.linalg.norm(hat_theta[:,0] - theta[:,0]))
            H += 1 / historical_sigma[t]**2 * np.dot(cur_contexts[selected_arm].reshape(d, 1), cur_contexts[selected_arm].reshape(1, d))
            inverse_H = np.linalg.inv(H)
            beta = 3 * np.sqrt(lamda) +\
                24 * (t+1)**((1-epsilon)/(2*(1+epsilon))) * np.sqrt(2*kappa) * np.log(3*round)**((1-epsilon)/(2*(1+epsilon))) * np.log(2*round**2/delta)**(epsilon/(1+epsilon))
            # tune beta
            beta /= 10000
            t += 1



if __name__ == "__main__":
    num_path = 2
    algorithms = ['menu', 'tofu', 'heavy_oful']

    df = 2
    scales = [0, 2]

    for scale in scales:
        # heavy-tailed
        epsilon = 0.99
        v = t(df).expect(lambda x: abs(x) ** (1 + epsilon))
        v_scaled = v * 10 ** (scale * (1 + epsilon))

        round, K, d = 10000, 20, 10
        theta = np.ones(shape=(d, 1)) / np.sqrt(d)

        for i_path in range(num_path):
            menu(round, d, K, theta, epsilon, v_scaled)
            tofu(round, d, K, theta, epsilon, v_scaled)
            heavy_oful(round, d, K, theta, epsilon, v**(1/(1+epsilon)))

        for algorithm in algorithms:
            y = np.zeros((num_path, round))
            for i_path in range(num_path):
                with open("./data/" + algorithm + "_" + str(round) + "_noise_scale_" + str(scale)+ "_path_" + str(i_path) + ".txt", "r") as f:
                    y[i_path] = list(map(float, f.readline().strip().split()))[:round]
            y_many_paths = np.mean(y, axis=0)
            np.savetxt("./data/" + algorithm + "_" + str(round) + "_noise_scale_" + str(scale) + ".txt", y_many_paths, fmt='%.15f', newline='\t')
