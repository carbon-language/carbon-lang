// RUN: %clang -emit-llvm -g -S %s -o - | FileCheck %s

namespace PR16214_1 {
// CHECK: [[PR16214_1:![0-9]*]] = {{.*}} [ DW_TAG_namespace ] [PR16214_1]
// CHECK: = metadata !{i32 {{[0-9]*}}, metadata !{{[0-9]*}}, metadata [[PR16214_1]], {{.*}} ; [ DW_TAG_structure_type ] [foo] {{.*}} [def]
struct foo {
  int i;
};

typedef foo bar;

bar *f;
bar g;
}
