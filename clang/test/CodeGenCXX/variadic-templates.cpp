// RUN: %clang_cc1 -std=c++0x -triple x86_64-apple-darwin10 -emit-llvm -o - %s | FileCheck %s

template<typename ...Types>
int get_num_types(Types...) {
  return sizeof...(Types);
}

// CHECK: define weak_odr i32 @_Z13get_num_typesIJifdEEispT_
// CHECK: ret i32 3
template int get_num_types(int, float, double);


