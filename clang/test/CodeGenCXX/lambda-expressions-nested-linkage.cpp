// RUN: %clang_cc1 -triple x86_64-apple-darwin10.0.0 -fblocks -emit-llvm -o - %s -fexceptions -std=c++11 | FileCheck %s

// CHECK-LABEL: define void @_ZN19non_inline_function3fooEv()
// CHECK-LABEL: define internal void @"_ZZN19non_inline_function3fooEvENK3$_0clEi"
// CHECK-LABEL: define internal signext i8 @"_ZZZN19non_inline_function3fooEvENK3$_0clEiENKUlcE_clEc"
namespace non_inline_function {
void foo() {
  auto L = [](int a) {
    return [](char b) {
     return b;
    };
  };
  L(3)('a');
}
}
// CHECK-LABEL: define linkonce_odr i32 @_ZN15inline_function3fooEv
// CHECK-LABEL: define linkonce_odr void @_ZZN15inline_function3fooEvENKUliE_clEi
// CHECK-LABEL: define linkonce_odr signext i8 @_ZZZN15inline_function3fooEvENKUliE_clEiENKUlcE_clEc
namespace inline_function {
inline int foo() {
  auto L = [](int a) {
    return [](char b) {
     return b;
    };
  };
  L(3)('a');
}
int use = foo();
}

// CHECK-LABEL: define linkonce_odr void @_ZN12non_template1LC1Ev
// CHECK-LABEL: define linkonce_odr void @_ZNK12non_template1L1tMUliE_clEi(%class.anon
// CHECK-LABEL: define linkonce_odr i32 @_ZZNK12non_template1L1tMUliE_clEiENKUliE_clEi(%class.anon
namespace non_template {
  struct L {
    int t = ([](int a) { return [](int b) { return b; };})(2)(3);    
  };
  L l; 
}

// CHECK-LABEL: define linkonce_odr void @_ZN32lambdas_in_NSDMIs_template_class1LIiEC2Ev
// CHECK-LABEL: define linkonce_odr void @_ZNK32lambdas_in_NSDMIs_template_class1LIiEUliE_clEi(%class.anon
// CHECK-LABEL: linkonce_odr i32 @_ZZNK32lambdas_in_NSDMIs_template_class1LIiEUliE_clEiENKUliE_clEi(%class.anon
namespace lambdas_in_NSDMIs_template_class {
template<class T>
struct L {
    T t2 = ([](int a) { return [](int b) { return b; };})(T{})(T{});    
};
L<int> l;
}



