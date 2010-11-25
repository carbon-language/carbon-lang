// RUN: %llvmgcc -xc %s -S -o - | grep getelementptr

int *test(int *X, int Y) {
  return X + Y;
}
