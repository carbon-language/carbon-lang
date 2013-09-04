// RUN: %clangxx -O0 -g %s -o %t -c
// RUN: %clangxx %t -o %t.out
// RUN: %test_debuginfo %s %t.out
// XFAIL: *

// DEBUGGER: delete breakpoints
// DEBUGGER: break static-member.cpp:33
// DEBUGGER: r
// DEBUGGER: ptype C
// CHECK:      type = {{struct|class}} C {
// CHECK:      static const int a;
// CHECK-NEXT: static int b;
// CHECK-NEXT: static int c;
// CHECK-NEXT: int d;
// CHECK-NEXT: }
// DEBUGGER: p C::a
// CHECK: $1 = 4
// DEBUGGER: p C::c
// CHECK: $2 = 15

// PR14471, PR14734

class C {
public:
  const static int a = 4;
  static int b;
  static int c;
  int d;
};

int C::c = 15;
const int C::a;

int main() {
    C instance_C;
    return C::a;
}
