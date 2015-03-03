// RUN: %clang -emit-llvm -g -S %s -o - | FileCheck %s

template<typename T>
struct XF {
  T member;
};

template<typename T>
T fx(XF<T> xi) {
  return xi.member;
}

// CHECK: !MDCompositeType(tag: DW_TAG_structure_type, name: "XF<int>"
// CHECK: !MDTemplateTypeParameter(name: "T"
template int fx(XF<int>);
