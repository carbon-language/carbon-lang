// RUN: clang-tidy -checks=-*,google-readability-casting %s -- | FileCheck %s

// CHECK-NOT: warning:

bool g() { return false; }

void f(int a, double b) {
  int b1 = (int)b;
  // CHECK: :[[@LINE-1]]:12: warning: C-style casts are discouraged. Use static_cast{{.*}}

  // CHECK-NOT: warning:
  int b2 = int(b);
  int b3 = static_cast<double>(b);
  int b4 = b;
  double aa = a;
  (void)b2;
  return (void)g();
}

// CHECK-NOT: warning:
enum E { E1 = 1 };
template <E e>
struct A { static const E ee = e; };
struct B : public A<E1> {};
