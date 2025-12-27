import numpy as np, matplotlib.pyplot as plt

def galton(M, N=1000):
    outcomes = np.random.choice([0,1], size=(N,M))
    positions = outcomes.sum(axis=1)
    values, counts = np.unique(positions, return_counts=True)
    return values, counts/N

if __name__ == "__main__":
    for M in [5,10,100]:
        vals, pmf = galton(M)
        plt.bar(vals, pmf)
        plt.title(f"Galton PMF (M={M})")
        plt.savefig(f"galton_M{M}.png")
        plt.close()
