// RUN: %llvmgcc -S %s -O1 -o - | grep {ret.*123}
// Check for bug compatibility with gcc.

const int x __attribute((weak)) = 123;

int* f(void) {
  return &x;
}

int g(void) {
  return *f();
}
