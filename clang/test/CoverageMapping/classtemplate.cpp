// RUN: %clang_cc1 -triple %itanium_abi_triple -fprofile-instr-generate -fcoverage-mapping -dump-coverage-mapping -emit-llvm-only -main-file-name classtemplate.cpp %s > %tmapping
// RUN: FileCheck -input-file %tmapping %s --check-prefix=CHECK-CONSTRUCTOR
// RUN: FileCheck -input-file %tmapping %s --check-prefix=CHECK-GETTER
// RUN: FileCheck -input-file %tmapping %s --check-prefix=CHECK-SETTER

template<class TT>
class Test {
public:
  enum BaseType {
    A, C, G, T, Invalid
  };
  const static int BaseCount = 4;
  double bases[BaseCount];

                                        // CHECK-CONSTRUCTOR: Test
  Test() { }                            // CHECK-CONSTRUCTOR: File 0, [[@LINE]]:10 -> [[@LINE]]:13 = #0 (HasCodeBefore = 0)
                                        // CHECK-GETTER: get
  double get(TT position) const {       // CHECK-GETTER: File 0, [[@LINE]]:33 -> [[@LINE+2]]:4 = 0 (HasCodeBefore = 0)
    return bases[position];
  }
                                        // CHECK-SETTER: set
  void set(TT position, double value) { // CHECK-SETTER: File 0, [[@LINE]]:39 -> [[@LINE+2]]:4 = #0 (HasCodeBefore = 0)
    bases[position] = value;
  }
};

int main() {
  Test<unsigned> t;
  t.set(Test<unsigned>::A, 5.5);
  t.set(Test<unsigned>::T, 5.6);
  t.set(Test<unsigned>::G, 5.7);
  t.set(Test<unsigned>::C, 5.8);
  return 0;
}
