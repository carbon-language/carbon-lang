// RUN: %clang_cc1 -emit-llvm %s  -o /dev/null

static int unused_func(void) {
  return 1;
}

int foo(void) {
  (void)unused_func; /* avoid compiler warning */
  return 2;
}
