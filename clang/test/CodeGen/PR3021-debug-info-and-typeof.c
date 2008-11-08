// RUN: clang -o %t --emit-llvm -g %s

void convert(void) {
  struct { typeof(0) f0; } v0;
}
