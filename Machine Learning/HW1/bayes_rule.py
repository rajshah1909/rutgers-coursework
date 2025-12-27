def main():
    pA = pB = pC = 1/3
    pD_A, pD_B, pD_C = 0.001, 0.005, 0.01
    pD = pA*pD_A + pB*pD_B + pC*pD_C
    pA_given_D = (pD_A*pA)/pD
    pC_given_notD = ((1-pD_C)*pC) / (1-pD)
    print("P(defective) =", pD)
    print("P(A|defective) =", pA_given_D)
    print("P(C|non-defective) =", pC_given_notD)

if __name__ == "__main__":
    main()
