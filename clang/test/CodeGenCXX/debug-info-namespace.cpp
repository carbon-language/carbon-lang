// RUN: %clang  -g -S -emit-llvm %s -o - | FileCheck %s

namespace A {
#line 1 "foo.cpp"
namespace B {
int i;
}
}

// CHECK: [[FILE:![0-9]*]] {{.*}}debug-info-namespace.cpp"
// CHECK: [[VAR:![0-9]*]] = {{.*}}, metadata [[NS:![0-9]*]], metadata !"i", {{.*}} ; [ DW_TAG_variable ] [i]
// CHECK: [[NS]] = {{.*}}, metadata [[FILE2:![0-9]*]], metadata [[CTXT:![0-9]*]], {{.*}} ; [ DW_TAG_namespace ] [B] [line 1]
// CHECK: [[CTXT]] = {{.*}}, metadata [[FILE]], null, {{.*}} ; [ DW_TAG_namespace ] [A] [line 3]
// CHECK: [[FILE2]]} ; [ DW_TAG_file_type ] [{{.*}}foo.cpp]
