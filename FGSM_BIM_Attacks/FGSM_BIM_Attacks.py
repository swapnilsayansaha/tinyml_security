
# Setting seed for reproducibility
np.random.seed(1234)
PYTHONHASHSEED = 0

class attacks():

# fast gradient sign method
    def fgsm(X, Y, model,epsilon,targeted= False):
        dlt = model.predict(X)- Y
        if targeted:
            dlt = model.predict(X) - Y
        else:
            dlt = -(model.predict(X) - Y)
        dir=np.sign(np.matmul(dlt, model.weight.T))
        return X + epsilon * dir, Y


#basic iterative method
    def bim(X, Y, model, epsilon, alpha, I):
        Xp= np.zeros_like(X)
        for t in range(I):
            dlt = model.predict(Xp) - Y
            dir = np.sign(np.matmul(dlt, model.weight.T))
            Xp = Xp + epsilon * dir
            Xp = np.where(Xp > X+epsilon, X+epsilon, Xp)
            Xp = np.where(Xp < X-epsilon, X-epsilon, Xp)
        return Xp, Y


