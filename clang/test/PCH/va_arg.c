// Test this without pch.
// RUN: clang-cc -triple=x86_64-unknown-freebsd7.0 -include %S/va_arg.h %s -emit-llvm -o -

// Test with pch.
// RUN: clang-cc -triple=x86_64-unknown-freebsd7.0 -emit-pch -o %t %S/va_arg.h
// RUN: clang-cc -triple=x86_64-unknown-freebsd7.0 -include-pch %t %s -emit-llvm -o -

char *g0(char** argv, int argc) { return argv[argc]; }

char *g(char **argv) {
  f(g0, argv, 1, 2, 3);
}
