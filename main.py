from scipy.integrate import quad
import numpy as np
import matplotlib.pyplot as plt

n = 1
N = 100*n


def f(t):
    return t**(2*n)


def F(wk):
    def real_integrand(t):
        return f(t) * np.cos(-wk*np.pi*t)

    def imag_integrand(t):
        return f(t) * np.sin(-wk*np.pi*t)

    real, _ = quad(real_integrand, -N, N)
    imag, _ = quad(imag_integrand, -N, N)

    return real, imag


def specter(real, imag):
    return np.sqrt(real**2 + imag**2)


def display_results(T_values, k_values):
    print(f"{'T':<10}{'k':<10}{'wk':<10}{'Re(F(w_k))':<20}{'|F(w_k)|':<20}")
    print("-" * 60)
    for T in T_values:
        for k in k_values:
            wk = 2 * np.pi * k / T
            real, imag = F(wk)
            amplitude = specter(real, imag)
            print(f"{T:<10}{k:<10}{wk:<10.5f}{real:<20.5f}{amplitude:<20.5f}")
        print("-" * 60)  # Separate results for each T with a line


def main():
    t_values = np.linspace(0, 10, 500)  # 500 points from 0 to 10
    f_values = f(t_values)

    plt.figure(figsize=(10, 6))
    plt.plot(t_values, f_values, label=r"$f(t) = t^{2n}$", color="blue")
    plt.xlabel("t")
    plt.ylabel("f(t)")
    plt.title(r"Plot of $f(t) = t^{2n}$ for $t$ from 0 to 10")
    plt.grid(True)
    plt.legend()
    plt.show()
    T_values = [4, 8, 16, 32, 64, 128]
    k_values = np.arange(0, 21, 1)

    display_results(T_values, k_values)

    fig, axes = plt.subplots(3, 2, figsize=(14, 10))

    for i, T in enumerate(T_values):
        Re_values = []

        for k in k_values:
            wk = 2 * np.pi * k / T
            real, imag = F(wk)
            Re_values.append(real)

        row, col = divmod(i, 2)

        axes[row, col].stem(k_values, Re_values, basefmt=" ",
                            label=f"Re(F(w_k)), T = {T}")
        axes[row, col].set_xlabel("k")
        axes[row, col].set_ylabel("Re(F(w_k))")
        axes[row, col].grid(True)
        axes[row, col].legend()

    plt.show()

    fig, axes = plt.subplots(3, 2, figsize=(14, 10))

    for i, T in enumerate(T_values):
        Re_values = []
        Amplitude_values = []

        for k in k_values:
            wk = 2 * np.pi * k / T
            real, imag = F(wk)
            Re_values.append(real)
            Amplitude_values.append(specter(real, imag))

        row, col = divmod(i, 2)

        axes[row, col].stem(k_values, Amplitude_values, basefmt=" ", linefmt='orange',
                            markerfmt='o', label=f"|F(w_k)|, T = {T}")
        axes[row, col].set_xlabel("k")
        axes[row, col].set_ylabel("|F(w_k)|")
        axes[row, col].grid(True)
        axes[row, col].legend()

    plt.show()

    fig, axes = plt.subplots(3, 2, figsize=(14, 10))

    for i, T in enumerate(T_values):
        Re_values = []

        for k in k_values:
            wk = 2 * np.pi * k / T
            real, imag = F(wk)
            Re_values.append(real)

        row, col = divmod(i, 2)

        axes[row, col].plot(k_values, Re_values, marker='o',
                            linestyle='-', label=f"Re(F(w_k)), T = {T}")
        axes[row, col].set_xlabel("k")
        axes[row, col].set_ylabel("Re(F(w_k))")
        axes[row, col].grid(True)
        axes[row, col].legend()

    plt.show()

    fig, axes = plt.subplots(3, 2, figsize=(14, 10))

    for i, T in enumerate(T_values):
        Amplitude_values = []

        for k in k_values:
            wk = 2 * np.pi * k / T
            real, imag = F(wk)
            Amplitude_values.append(specter(real, imag))

        row, col = divmod(i, 2)

        axes[row, col].plot(k_values, Amplitude_values, marker='o',
                            linestyle='-', color='orange', label=f"|F(w_k)|, T = {T}")
        axes[row, col].set_xlabel("k")
        axes[row, col].set_ylabel("|F(w_k)|")
        axes[row, col].grid(True)
        axes[row, col].legend()

    plt.show()

    plt.figure(figsize=(10, 6))
    for T in T_values:
        Re_values = [F(2 * np.pi * k / T)[0] for k in k_values]
        plt.plot(k_values, Re_values, marker='o',
                 linestyle='-', label=f"T = {T}")
    plt.xlabel("k")
    plt.ylabel("Re(F(w_k))")
    plt.title("Real part of F(w_k) for different T values")
    plt.grid(True)
    plt.legend()
    plt.show()

    plt.figure(figsize=(10, 6))
    for T in T_values:
        Amplitude_values = [specter(*F(2 * np.pi * k / T)) for k in k_values]
        plt.plot(k_values, Amplitude_values, marker='o',
                 linestyle='-', label=f"T = {T}")
    plt.xlabel("k")
    plt.ylabel("|F(w_k)|")
    plt.title("Amplitude |F(w_k)| for different T values")
    plt.grid(True)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
