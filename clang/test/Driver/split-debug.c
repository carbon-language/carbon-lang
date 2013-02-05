// Check that we split debug output properly
//
// REQUIRES: asserts
// RUN: %clang -target x86_64-unknown-linux-gnu -ccc-print-phases \
// RUN:   -gsplit-dwarf -arch x86_64 %s 2> %t
// RUN: FileCheck -check-prefix=CHECK-ACTIONS < %t %s
//
// CHECK-ACTIONS: 0: input, "{{.*}}split-debug.c", c
// CHECK-ACTIONS: 4: split-debug, {3}, object

// Check output name derivation.
//
// RUN: %clang -target x86_64-unknown-linux-gnu -ccc-print-bindings \
// RUN:   -gsplit-dwarf -arch x86_64 -c %s 2> %t
// RUN: FileCheck -check-prefix=CHECK-OUTPUT-NAME < %t %s
//
// CHECK-OUTPUT-NAME:# "x86_64-unknown-linux-gnu" - "clang", inputs: ["{{.*}}split-debug.c"], output: "{{.*}}split-debug{{.*}}.o"
// CHECK-OUTPUT-NAME:# "x86_64-unknown-linux-gnu" - "linuxtools::SplitDebug", inputs: ["{{.*}}split-debug{{.*}}.o"], output: "{{.*}}split-debug{{.*}}.o"

