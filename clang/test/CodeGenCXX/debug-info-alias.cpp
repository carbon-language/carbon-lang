// RUN: %clang -g -std=c++11 -S -emit-llvm %s -o - | FileCheck %s

template<typename T>
struct foo {
};
namespace x {
// splitting these over multiple lines to make sure the right token is used for
// the location
template<typename T>
using
# 42
bar
= foo<T*>;
}

// CHECK: !MDGlobalVariable(name: "bi",{{.*}} type: [[BINT:![0-9]+]]
// CHECK: [[BINT]] = !MDDerivedType(tag: DW_TAG_typedef, name: "bar<int>"
// CHECK-SAME:                      line: 42,
x::bar<int> bi;
// CHECK: !MDGlobalVariable(name: "bf",{{.*}} type: [[BFLOAT:![0-9]+]]
// CHECK: [[BFLOAT]] = !MDDerivedType(tag: DW_TAG_typedef, name: "bar<float>"
x::bar<float> bf;

using
// CHECK: !MDGlobalVariable(name: "n",{{.*}} type: [[NARF:![0-9]+]]
# 142
narf // CHECK: [[NARF]] = !MDDerivedType(tag: DW_TAG_typedef, name: "narf"
// CHECK-SAME:                           line: 142
= int;
narf n;

template <typename T>
using tv = void;
// CHECK: !MDDerivedType(tag: DW_TAG_typedef, name: "tv<int>"
tv<int> *tvp;

using v = void;
// CHECK: !MDDerivedType(tag: DW_TAG_typedef, name: "v"
v *vp;
