// RUN: %clangxx -O0 -g %s -c -o %t.o
// RUN: %clangxx %t.o -o %t.out
// RUN: %test_debuginfo %s %t.out 
// Radar 8945514
// DEBUGGER: break 22
// DEBUGGER: r
// DEBUGGER: p v
// CHECK: $1 = {
// CHECK:  Data = 0x0, 
// CHECK:  Kind = 2142

class SVal {
public:
  ~SVal() {}
  const void* Data;
  unsigned Kind;
};

void bar(SVal &v) {}
class A {
public:
  void foo(SVal v) { bar(v); }
};

int main() {
  SVal v;
  v.Data = 0;
  v.Kind = 2142;
  A a;
  a.foo(v);
  return 0;
}
