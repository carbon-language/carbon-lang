// RUN: rm -f %t
// RUN: clang-ast-dump "%s" -f test_namespace::TheClass::theMethod -- -c > %t 2>&1
// RUN: FileCheck --input-file=%t %s

// FIXME: Does this run regardless of +Asserts?
// REQUIRES: asserts

// CHECK: <CXXMethod ptr="0x{{[0-9a-f]+}}" name="theMethod" prototype="true">
// CHECK:  <ParmVar ptr="0x{{[0-9a-f]+}}" name="x" initstyle="c">
// CHECK: (CompoundStmt
// CHECK-NEXT:   (ReturnStmt
// CHECK-NEXT:     (BinaryOperator

namespace test_namespace {

class TheClass {
public:
  int theMethod(int x) {
    return x + x;
  }
};

}
