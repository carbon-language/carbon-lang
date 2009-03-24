// RUN: clang-cc -emit-llvm -x c++ < %s

void f(int x) {
          if (x != 0) return;
}
