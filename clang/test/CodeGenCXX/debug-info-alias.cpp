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

// CHECK: metadata [[BINT:![0-9]*]], i32 0, i32 1, {{.*}} ; [ DW_TAG_variable ] [bi]
// CHECK: [[BINT]] = {{.*}} ; [ DW_TAG_typedef ] [bar<int>] [line 42
x::bar<int> bi;
// CHECK: metadata [[BFLOAT:![0-9]*]], i32 0, i32 1, {{.*}} ; [ DW_TAG_variable ] [bf]
// CHECK: [[BFLOAT]] = {{.*}} ; [ DW_TAG_typedef ] [bar<float>] [line 42
x::bar<float> bf;

using
// CHECK: metadata [[NARF:![0-9]*]], i32 0, i32 1, {{.*}} ; [ DW_TAG_variable ] [n]
# 142
narf // CHECK: [[NARF]] = {{.*}} ; [ DW_TAG_typedef ] [narf] [line 142
= int;
narf n;

template <typename T>
using tv = void;
// CHECK: null} ; [ DW_TAG_typedef ] [tv<int>] {{.*}} [from ]
tv<int> *tvp;

using v = void;
// CHECK: null} ; [ DW_TAG_typedef ] [v] {{.*}} [from ]
v *vp;
