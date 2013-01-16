// RUN: %clangxx -O0 -g %s -o %t.out
// RUN: %test_debuginfo %s %t.out

// DEBUGGER: delete breakpoints
// DEBUGGER: break main
// DEBUGGER: r
// DEBUGGER: n
// DEBUGGER: ptype C
// CHECK:      type = class C {
// CHECK-NEXT: public:
// CHECK-NEXT: static const int a;
// CHECK-NEXT: static int b;
// CHECK-NEXT: static int c;
// CHECK-NEXT: int d;
// CHECK-NEXT: }
// DEBUGGER: p instance_C
// CHECK: $1 = {static a = 4, static b = {{.*}}, static c = 15, d = {{.*}}}

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
