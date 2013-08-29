// RUN: %clang -flimit-debug-info -emit-llvm -g -S %s -o - | FileCheck %s

// Check that this pointer type is TC<int>
// CHECK: ![[LINE:[0-9]+]] = {{.*}}"TC<int>"
// CHECK: ![[LINE]]} ; [ DW_TAG_pointer_type ]{{.*}}[from TC<int>]

template<typename T>
class TC {
public:
  TC(const TC &) {}
  TC() {}
};

TC<int> tci;
