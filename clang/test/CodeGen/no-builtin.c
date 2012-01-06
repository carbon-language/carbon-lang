// RUN: %clang_cc1 -fno-builtin -emit-llvm %s -o - | FileCheck %s
//
// Check that -fno-builtin prevents us from constant-folding through builtins
// (PR11711)

double
cos(double x)
{
  printf("ok\n");
  exit(0);
}

int
main(int argc, char *argv[])
{
  cos(1); // CHECK: cos
  printf("not ok\n");
  abort();
}

