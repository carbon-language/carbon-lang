// RUN: clang -o %t --emit-llvm -g %s

// PR3023
void convert(void) {
  struct { typeof(0) f0; } v0;
}
