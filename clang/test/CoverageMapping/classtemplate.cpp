// RUN: %clang_cc1 -fprofile-instr-generate -fcoverage-mapping -dump-coverage-mapping -emit-llvm-only -main-file-name classtemplate.cpp %s | FileCheck %s

template<class TT>
class Test {
public:
  enum BaseType {
    A, C, G, T, Invalid
  };
  const static int BaseCount = 4;
  double bases[BaseCount];

  Test() { }
  double get(TT position) const {
    return bases[position];
  }
  void set(TT position, double value) {
    bases[position] = value;
  }
};

// CHECK: set
// CHECK-NEXT: File 0, 16:39 -> 18:4 = #0 (HasCodeBefore = 0)

// CHECK-NEXT: Test
// CHECK-NEXT: File 0, 12:10 -> 12:13 = #0 (HasCodeBefore = 0)

// CHECK-NEXT: get
// CHECK-NEXT: File 0, 13:33 -> 15:4 = 0 (HasCodeBefore = 0)

int main() {
  Test<unsigned> t;
  t.set(Test<unsigned>::A, 5.5);
  t.set(Test<unsigned>::T, 5.6);
  t.set(Test<unsigned>::G, 5.7);
  t.set(Test<unsigned>::C, 5.8);
  return 0;
}
