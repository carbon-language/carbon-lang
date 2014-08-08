// RUN: %clang_cc1 -fprofile-instr-generate -fcoverage-mapping -dump-coverage-mapping -emit-llvm-only -main-file-name nestedclass.cpp %s | FileCheck %s

struct Test {
  void emitTest() {
    int i = 0;
  }
  struct Test2 {
    void emitTest2() {
      int i = 0;
    }
    struct Test3 {
      static void emitTest3() {
        int i = 0;
      }
    };
  };
};

// CHECK: emitTest2
// CHECK-NEXT: File 0, 8:22 -> 10:6 = #0 (HasCodeBefore = 0)

// CHECK-NEXT: emitTest
// CHECK-NEXT: File 0, 4:19 -> 6:4 = #0 (HasCodeBefore = 0)

// CHECK-NEXT: emitTest3
// CHECK-NEXT: File 0, 12:31 -> 14:8 = 0 (HasCodeBefore = 0)

int main() {
  Test t;
  Test::Test2 t2;
  t.emitTest();
  t2.emitTest2();
  return 0;
}
