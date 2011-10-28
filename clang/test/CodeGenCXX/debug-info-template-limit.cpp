// RUN: %clang -flimit-debug-info -emit-llvm -g -S %s -o - | FileCheck %s

// Check that this pointer type is TC<int>
// CHECK: !10} ; [ DW_TAG_pointer_type
// CHECK-NEXT: !10 ={{.*}}"TC<int>"

template<typename T>
class TC {
public:
  TC(const TC &) {}
  TC() {}
};

TC<int> tci;

