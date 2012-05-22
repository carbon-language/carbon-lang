// RUN: %clang_cc1 -emit-llvm -g -triple x86_64-apple-darwin %s -o - | FileCheck %s

// CHECK: DW_TAG_subrange_type

struct StructName {
  int member[];
};

struct StructName SN;
