// RUN: %clang_cc1 -emit-llvm -g -triple x86_64-linux-gnu  %s -o - | FileCheck %s

// Make sure that the union type has template parameters.

namespace PR15637 {
  template <typename T> union Value { int a; };
  void g(float value) {
    Value<float> tempValue;
  }
  Value<float> f;
}

// CHECK: {{.*}}, metadata !"Value<float>", {{.*}}, null, metadata [[TTPARAM:.*]]} ; [ DW_TAG_union_type ] [Value<float>]
// CHECK: [[TTPARAM]] = metadata !{metadata [[PARAMS:.*]]}
// CHECK: [[PARAMS]] = metadata !{{{.*}}metadata !"T",{{.*}}} ; [ DW_TAG_template_type_parameter ]
