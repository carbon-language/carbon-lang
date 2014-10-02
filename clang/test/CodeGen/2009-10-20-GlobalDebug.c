// REQUIRES: x86-registered-target
// RUN: %clang -target i386-apple-darwin10 -flto -S -g %s -o - | FileCheck %s
int global;
int main() { 
  static int localstatic;
  return 0;
}

// CHECK: metadata !{i32 {{.*}}, i32 0, metadata !{{.*}}, metadata !"localstatic", metadata !"localstatic", metadata !"", metadata !{{.*}}, i32 5, metadata !{{.*}}, i32 1, i32 1, i32* @main.localstatic, null} ; [ DW_TAG_variable ]
// CHECK: metadata !{i32 {{.*}}, i32 0, null, metadata !"global", metadata !"global", metadata !"", metadata !{{.*}}, i32 3, metadata !{{.*}}, i32 0, i32 1, i32* @global, null} ; [ DW_TAG_variable ]
