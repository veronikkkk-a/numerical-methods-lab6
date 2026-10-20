import numpy as np


def generate_and_save_data(n=100, filename_a="matrix_a.txt", filename_b="vector_b.txt"):
  """Генерує випадкову матрицю А, задає розв'язок і обчислює вектор b, записуючи у файли."""
  np.random.seed(42)
  A = np.random.rand(n, n) * 10 + 1  # Заповнюємо випадковими числами
  # Робимо матрицю з діагональним переважанням для кращої збіжності
  np.fill_diagonal(A, np.sum(np.abs(A), axis=1) + 10)

  # Задаємо точний розв'язок (усі xi = 2.5)
  x_exact = np.full(n, 2.5)

  # Обчислюємо вектор вільних членів B = A * X_exact
  b = np.dot(A, x_exact)

  # Зберігаємо у файли
  np.savetxt(filename_a, A, fmt="%.6f")
  np.savetxt(filename_b, b, fmt="%.6f")

  print(f"Матриця А та вектор B успішно згенеровані та записані у файли.")
  return A, b, x_exact


def read_matrix_from_file(filename):
  """Зчитування матриці з текстового файлу."""
  return np.loadtxt(filename)


def read_vector_from_file(filename):
  """Зчитування вектора з текстового файлу."""
  return np.loadtxt(filename)


def lu_decomposition(A):
  """Знаходить LU-розклад матриці А (A = L * U)."""
  n = A.shape[0]
  L = np.zeros((n, n))
  U = np.eye(n)

  for i in range(n):
    for k in range(i, n):
      s = sum(L[k][j] * U[j][i] for j in range(i))
      L[k][i] = A[k][i] - s

    for k in range(i + 1, n):
      s = sum(L[i][j] * U[j][k] for j in range(i))
      if L[i][i] == 0:
        raise ValueError("Діагональний елемент матриці L дорівнює нулю!")
      U[i][k] = (A[i][k] - s) / L[i][i]

  return L, U


def solve_lu(L, U, b):
  """Розв'язує систему AX = B за допомогою LU-розкладу (LZ = B, UX = Z)."""
  n = L.shape[0]

  # 1. Розв'язуємо LZ = B (прямий хід)
  z = np.zeros(n)
  for i in range(n):
    s = sum(L[i][j] * z[j] for j in range(i))
    z[i] = (b[i] - s) / L[i][i]

  # 2. Розв'язуємо UX = Z (зворотний хід)
  x = np.zeros(n)
  for i in range(n - 1, -1, -1):
    s = sum(U[i][j] * x[j] for j in range(i + 1, n))
    x[i] = z[i] - s

  return x


def vector_norm(v):
  """Обчислює норму вектора (максимальний модуль елемента)."""
  return np.max(np.abs(v))


def iterative_refinement(A, b, x0, L, U, eps_0=1e-14, max_iter=50):
  """Ітераційний метод уточнення розв'язку СЛАР з використанням існуючого LU-розкладу."""
  x = x0.copy()
  for iteration in range(1, max_iter + 1):
    r = b - np.dot(A, x)  # Вектор нев'язки R = B - A*X
    delta_x = solve_lu(L, U, r)  # Розв'язуємо A * delta_X = R
    x = x + delta_x  # Уточнюємо розв'язок

    # Перевірка умов закінчення
    norm_delta = vector_norm(delta_x)
    norm_residual = vector_norm(r)

    print(
        f"Ітерація {iteration}: норма похибки = {norm_delta:.2e}, норма"
        f" нев'язки = {norm_residual:.2e}"
    )

    if norm_delta <= eps_0 or norm_residual <= eps_0:
      print(f"Збіжність досягнута на ітерації {iteration}.")
      break

  return x, iteration


if __name__ == "__main__":
  # 1. Генеруємо та зберігаємо дані
  filename_a = "matrix_a.txt"
  filename_b = "vector_b.txt"
  A, b, x_exact = generate_and_save_data(n=100)

  # 2. Зчитуємо дані з файлів (згідно з ходом роботи)
  A_loaded = read_matrix_from_file(filename_a)
  b_loaded = read_vector_from_file(filename_b)

  # 3. Знаходимо LU-розклад та записуємо у файли
  L, U = lu_decomposition(A_loaded)
  np.savetxt("matrix_l.txt", L, fmt="%.6f")
  np.savetxt("matrix_u.txt", U, fmt="%.6f")
  print("LU-розклад знайдено і збережено у файли matrix_l.txt та matrix_u.txt.")

  # 4. Знаходимо початковий розв'язок системи
  x_initial = solve_lu(L, U, b_loaded)

  # 5. Оцінюємо початкову точність
  initial_error = vector_norm(np.dot(A_loaded, x_initial) - b_loaded)
  print(f"Початкова похибка (нев'язка): {initial_error:.2e}")

  # 6. Ітераційне уточнення розв'язку
  print("\n--- Початок ітераційного уточнення розв'язку ---")
  eps_target = 1e-14
  x_refined, iter_count = iterative_refinement(
      A_loaded, b_loaded, x_initial, L, U, eps_0=eps_target
  )

  print("\nРоботу програми завершено успішно!")