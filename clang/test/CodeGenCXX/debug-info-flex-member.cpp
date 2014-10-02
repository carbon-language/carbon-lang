// RUN: %clang_cc1 -emit-llvm -g -triple x86_64-apple-darwin %s -o - | FileCheck %s

// CHECK: metadata !{metadata !"0x21\000\00-1"}        ; [ DW_TAG_subrange_type ]

struct StructName {
  int member[];
};

struct StructName SN;
