// RUN: %clang -emit-llvm -g -S %s -o - | FileCheck %s

template<typename T>
struct XF {
  T member;
};

template<typename T>
T fx(XF<T> xi) {
  return xi.member;
}

//CHECK: DW_TAG_template_type_parameter
//CHECK: XF<int>
template int fx(XF<int>);
