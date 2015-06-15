// RUN: %clang_safestack %s -o %t
// RUN: %run %t

// RUN: %clang_nosafestack -fno-stack-protector %s -o %t
// RUN: not %run %t

// Test that buffer overflows on the unsafe stack do not affect variables on the
// safe stack.

__attribute__((noinline))
void fct(volatile int *buffer)
{
  memset(buffer - 1, 0, 7 * sizeof(int));
}

int main(int argc, char **argv)
{
  int value1 = 42;
  int buffer[5];
  int value2 = 42;
  fct(buffer);
  return value1 != 42 || value2 != 42;
}
