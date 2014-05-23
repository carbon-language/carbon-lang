// RUN: %clang_cc1 -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -emit-llvm -o - %s -main-file-name some.name | FileCheck -check-prefix NAMED %s

// CHECK: ; ModuleID = '{{.*}}main-file-name.c'
// NAMED: ; ModuleID = 'some.name'

