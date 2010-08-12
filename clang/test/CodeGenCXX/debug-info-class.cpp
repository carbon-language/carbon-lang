// RUN: %clang  -emit-llvm -g -S %s -o - | grep HdrSize
struct A {
  int one;
  static const int HdrSize = 52;
  int two;
  A() {
    int x = 1;
  }
};
int main() {
  A a;
}
