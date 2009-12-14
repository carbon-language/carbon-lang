// RUN: clang -cc1 -emit-llvm -o - %s -O2 | grep "ret i32 1"

int a() {
  return @encode(int) == @encode(int);
}
