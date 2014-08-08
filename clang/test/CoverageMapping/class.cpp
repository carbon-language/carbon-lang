// RUN: %clang_cc1 -fprofile-instr-generate -fcoverage-mapping -dump-coverage-mapping -emit-llvm-only -main-file-name class.cpp %s | FileCheck %s

class Test {
  int x;
public:
  Test(int i)
    : x(i != 0 ? i : 11)
  {
  }
  ~Test() {
    x = 0;
  }
  int getX() const { return x; }
  Test(int i, int j):x(i + j){ }
  void setX(int i) {
    x = i;
  }
  inline int getXX() const {
    return x*x;
  }
  void setX2(int i);
};

void Test::setX2(int i) {
  x = i;
}

int main() {
  Test t(42);
  int i = t.getX();
  return 0;
}

// CHECK: File 0, 24:25 -> 26:2 = #0 (HasCodeBefore = 0)
// CHECK: File 0, 28:12 -> 32:2 = #0 (HasCodeBefore = 0)
// CHECK: File 0, 10:11 -> 12:4 = #0 (HasCodeBefore = 0)
// CHECK: File 0, 13:20 -> 13:33 = #0 (HasCodeBefore = 0)
// CHECK: File 0, 8:3 -> 9:4 = #0 (HasCodeBefore = 0)
// CHECK: File 0, 14:30 -> 14:33 = 0 (HasCodeBefore = 0)
// CHECK: File 0, 15:20 -> 17:4 = 0 (HasCodeBefore = 0)
// CHECK: File 0, 18:28 -> 20:4 = 0 (HasCodeBefore = 0)
