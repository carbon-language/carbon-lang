// Check to make sure clang is somewhat picky about -g options.
// (Delived from debug-options.c)
// rdar://10383444
// RUN: %clang -### -c -save-temps -integrated-as -g %s 2>&1 \
// RUN:   | FileCheck -check-prefix=SAVE %s
//
// SAVE: "-cc1as"
// SAVE-NOT: "-g"

// Check to make sure clang with -g on a .s file gets passed.
// rdar://9275556
// RUN: touch %t.s
// RUN: %clang -### -c -integrated-as -g %t.s 2>&1 \
// RUN:   | FileCheck %s
//
// CHECK: "-cc1as"
// CHECK: "-g"

// Check to make sure clang with -g on a .s file gets passed -dwarf-debug-producer.
// rdar://12955296
// RUN: touch %t.s
// RUN: %clang -### -c -integrated-as -g %t.s 2>&1 \
// RUN:   | FileCheck -check-prefix=P %s
//
// P: "-cc1as"
// P: "-dwarf-debug-producer"
