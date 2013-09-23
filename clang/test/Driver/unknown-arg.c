// RUN: not %clang_cc1 %s -cake-is-lie -%0 -%d 2> %t.log
// RUN: FileCheck %s -input-file=%t.log

// CHECK: unknown argument
// CHECK: unknown argument
// CHECK: unknown argument


// RUN: %clang -S %s -o %t.s -funknown-to-clang-option -Wunknown-to-clang-option -munknown-to-clang-optio

// IGNORED: warning: argument unused during compilation: '-funknown-to-clang-option'
// IGNORED: warning: argument unused during compilation: '-munknown-to-clang-option'
// IGNORED: warning: unknown warning option '-Wunknown-to-clang-option'
