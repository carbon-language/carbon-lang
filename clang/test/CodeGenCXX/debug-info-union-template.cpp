// RUN: %clang_cc1 -emit-llvm -g -triple x86_64-linux-gnu  %s -o - | FileCheck %s

// Make sure that the union type has template parameters.

namespace PR15637 {
  template <typename T> union Value { int a; };
  void g(float value) {
    Value<float> tempValue;
  }
  Value<float> f;
}

// CHECK:  !"0x17\00Value<float>\00{{.*}}", {{.*}}, [[TTPARAM:![0-9]+]], !"_ZTSN7PR156375ValueIfEE"} ; [ DW_TAG_union_type ] [Value<float>]
// CHECK: [[TTPARAM]] = !{[[PARAMS:.*]]}
// CHECK: [[PARAMS]] = !{!"0x2f\00T\000\000", {{.*}} ; [ DW_TAG_template_type_parameter ]
