// Check to make sure clang is somewhat picky about -g options.
// (Delived from debug-options.c)
// rdar://10383444
// RUN: %clang -### -c -save-temps -integrated-as -g %s 2>&1 \
// RUN:   | FileCheck -check-prefix=SAVE %s
//
// SAVE: "-cc1as"
// SAVE-NOT: "-g"

// Make sure that '-ggdb0' is not accidentally mistaken for '-g'
// RUN: %clang -### -ggdb0 -c -integrated-as -x assembler %s 2>&1 \
// RUN:   | FileCheck -check-prefix=GGDB0 %s
//
// GGDB0: "-cc1as"
// GGDB0-NOT: "-g"

// Check to make sure clang with -g on a .s file gets passed.
// rdar://9275556
// RUN: %clang -### -c -integrated-as -g -x assembler %s 2>&1 \
// RUN:   | FileCheck %s
//
// CHECK: "-cc1as"
// CHECK: "-g"

// Check to make sure clang with -g on a .s file gets passed -dwarf-debug-producer.
// rdar://12955296
// RUN: %clang -### -c -integrated-as -g -x assembler %s 2>&1 \
// RUN:   | FileCheck -check-prefix=P %s
//
// P: "-cc1as"
// P: "-dwarf-debug-producer"
