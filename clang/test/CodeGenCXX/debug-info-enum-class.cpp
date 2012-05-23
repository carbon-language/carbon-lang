// RUN: %clang_cc1 -emit-llvm -g -triple x86_64-apple-darwin -std=c++11 %s -o - | FileCheck %s

enum class A { A1=1 };                 // underlying type is int by default
enum class B: unsigned long { B1=1 };  // underlying type is unsigned long
enum C { C1 = 1 };
A a;
B b;
C c;

// CHECK: metadata !{i32 {{.*}}, null, metadata !"A", metadata !4, i32 3, i64 32, i64 32, i32 0, i32 0, metadata !5, metadata !6, i32 0, i32 0} ; [ DW_TAG_enumeration_type ]
// CHECK: metadata !{i32 {{.*}}, null, metadata !"B", metadata !4, i32 4, i64 64, i64 64, i32 0, i32 0, metadata !9, metadata !10, i32 0, i32 0} ; [ DW_TAG_enumeration_type ]
// CHECK: metadata !{i32 {{.*}}, null, metadata !"C", metadata !4, i32 5, i64 32, i64 32, i32 0, i32 0, null, metadata !13, i32 0, i32 0} ; [ DW_TAG_enumeration_type ]
