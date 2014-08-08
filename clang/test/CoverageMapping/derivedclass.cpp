// RUN: %clang_cc1 -fprofile-instr-generate -fcoverage-mapping -dump-coverage-mapping -emit-llvm-only -main-file-name derivedclass.cpp %s | FileCheck %s

class Base {
protected:
  int x;
public:
  Base(int i, int j)
    : x(i)
  {
  }
  virtual ~Base() {
    x = 0;
  }
  int getX() const { return x; }
  virtual void setX(int i) {
    x = i;
  }
};

class Derived: public Base {
  int y;
public:
  Derived(int i)
    : Base(i, i), y(0)
  { }
  virtual ~Derived() {
    y = 0;
  }
  virtual void setX(int i) {
    x = y = i;
  }
  int getY() const {
    return y;
  }
};

// CHECK: File 0, 14:20 -> 14:33 = #0 (HasCodeBefore = 0)
// CHECK: File 0, 25:3 -> 25:6 = #0 (HasCodeBefore = 0)
// CHECK: File 0, 29:28 -> 31:4 = #0 (HasCodeBefore = 0)
// CHECK: File 0, 26:22 -> 28:4 = #0 (HasCodeBefore = 0)
// CHECK: File 0, 11:19 -> 13:4 = #0 (HasCodeBefore = 0)
// CHECK: File 0, 15:28 -> 17:4 = #0 (HasCodeBefore = 0)
// CHECK: File 0, 9:3 -> 10:4 = #0 (HasCodeBefore = 0)
// CHECK: File 0, 32:20 -> 34:4 = 0 (HasCodeBefore = 0)

int main() {
  Base *B = new Derived(42);
  B->setX(B->getX());
  delete B;
  return 0;
}
