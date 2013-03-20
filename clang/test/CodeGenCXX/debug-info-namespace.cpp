// RUN: %clang  -g -S -emit-llvm %s -o - | FileCheck %s

namespace A {
int i;
}

// CHECK: [[FILE:![0-9]*]] = {{.*}} ; [ DW_TAG_file_type ] [{{.*}}/test/CodeGenCXX/debug-info-namespace.cpp]
// CHECK: [[VAR:![0-9]*]] = {{.*}}, metadata [[NS:![0-9]*]], metadata !"i", {{.*}} ; [ DW_TAG_variable ] [i]
// CHECK: [[NS]] = {{.*}}, metadata [[FILE]], {{.*}} ; [ DW_TAG_namespace ] [A] [line 3]
