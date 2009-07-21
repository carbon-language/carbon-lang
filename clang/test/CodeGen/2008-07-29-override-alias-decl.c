// RUN: clang-cc -emit-llvm -o - %s | grep -e "^@f" | count 1

int x() { return 1; }

int f() __attribute__((weak, alias("x")));

/* Test that we link to the alias correctly instead of making a new
   forward definition. */
int f();
int h() {
  return f();
}
