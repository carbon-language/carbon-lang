// RUN: %clang -emit-llvm -g -S %s -o - | FileCheck %s

//CHECK: TC<int>
//CHECK: DW_TAG_template_type_parameter

template<typename T>
class TC {
public:
  TC(const TC &) {}
  TC() {}
};

TC<int> tci;
