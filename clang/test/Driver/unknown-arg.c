// RUN: not %clang %s -cake-is-lie -%0 -%d -HHHH -munknown-to-clang-option -print-stats 2>&1 | \
// RUN: FileCheck %s

// CHECK: unknown argument: '-cake-is-lie'
// CHECK: unknown argument: '-%0'
// CHECK: unknown argument: '-%d'
// CHECK: unknown argument: '-HHHH'
// CHECK: unknown argument: '-munknown-to-clang-option'
// CHECK: unknown argument: '-print-stats'


// RUN: %clang -S %s -o %t.s -funknown-to-clang-option -Wunknown-to-clang-option 2>&1 | FileCheck --check-prefix=IGNORED %s

// IGNORED: warning: argument unused during compilation: '-funknown-to-clang-option'
// IGNORED: warning: unknown warning option '-Wunknown-to-clang-option'
