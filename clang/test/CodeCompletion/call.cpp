// Note: the run lines follow their respective tests, since line/column
// matter in this test.
void f(float x, float y);
void f(int i, int j, int k);
struct X { };
void f(X);
namespace N {
  struct Y { 
    Y(int = 0); 
    
    operator int() const;
  };
  void f(Y y, int ZZ);
}
typedef N::Y Y;
void f();

void test() {
  f(Y(), 0, 0);
  // RUN: clang-cc -fsyntax-only -code-completion-at=%s:19:9 %s -o - | FileCheck -check-prefix=CC1 %s &&
  // CHECK-CC1: int ZZ
  // CHECK-NEXT-CC1: int j
  // CHECK-NEXT-CC1: float y
  // RUN: clang-cc -fsyntax-only -code-completion-at=%s:19:13 %s -o - | FileCheck -check-prefix=CC2 %s &&
  // FIXME: two ellipses are showing up when they shouldn't
  // CHECK-CC2: int k
  // RUN: true
}
