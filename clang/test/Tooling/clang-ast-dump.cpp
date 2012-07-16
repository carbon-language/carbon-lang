// RUN: clang-ast-dump "%t/test.cpp" -f test_namespace::TheClass::theMethod -- -c 2>&1|FileCheck %s
// FIXME: Make the above easier.

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

// FIXME: This is incompatible to -fms-compatibility.
// XFAIL: win32
