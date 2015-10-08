// RUN: %clang_cc1 -emit-llvm -debug-info-kind=limited -triple x86_64-apple-darwin %s -o - | FileCheck %s

// CHECK: !DISubrange(count: -1)

struct StructName {
  int member[];
};

struct StructName SN;
