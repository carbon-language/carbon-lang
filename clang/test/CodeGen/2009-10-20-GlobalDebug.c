// REQUIRES: x86-registered-target
// RUN: %clang -target i386-apple-darwin10 -flto -S -g %s -o - | FileCheck %s
int global;
int main() { 
  static int localstatic;
  return 0;
}

// CHECK: metadata !{metadata !"0x34\00localstatic\00localstatic\00\005\001\001", metadata !{{.*}}, metadata !{{.*}}, metadata !{{.*}}, i32* @main.localstatic, null} ; [ DW_TAG_variable ]
// CHECK: metadata !{metadata !"0x34\00global\00global\00\003\000\001", null, metadata !{{.*}}, metadata !{{.*}}, i32* @global, null} ; [ DW_TAG_variable ]
