// REQUIRES: x86-registered-target

// Check that ps4-clang doesn't report a warning message when locating
// system header files (either by looking at the value of SCE_PS4_SDK_DIR
// or relative to the location of the compiler driver), if "-nostdinc",
// "--sysroot" or "-isysroot" option is specified on the command line.
// Otherwise, check that ps4-clang reports a warning.

// Check that clang doesn't report a warning message when locating
// system libraries (either by looking at the value of SCE_PS4_SDK_DIR
// or relative to the location of the compiler driver), if "-c", "-S", "-E",
// "--sysroot", "-nostdlib" or "-nodefaultlibs" option is specified on
// the command line.
// Otherwise, check that ps4-clang reports a warning.

// setting up SCE_PS4_SDK_DIR to existing location, which is not a PS4 SDK.
// RUN: env SCE_PS4_SDK_DIR=.. %clang -### -target x86_64-scei-ps4 %s 2>&1 | FileCheck -check-prefix=WARN-SYS-HEADERS -check-prefix=WARN-SYS-LIBS -check-prefix=NO-WARN %s

// RUN: env SCE_PS4_SDK_DIR=.. %clang -### -c -target x86_64-scei-ps4 %s 2>&1 | FileCheck -check-prefix=WARN-SYS-HEADERS -check-prefix=NO-WARN %s
// RUN: env SCE_PS4_SDK_DIR=.. %clang -### -S -target x86_64-scei-ps4 %s 2>&1 | FileCheck -check-prefix=WARN-SYS-HEADERS -check-prefix=NO-WARN %s
// RUN: env SCE_PS4_SDK_DIR=.. %clang -### -E -target x86_64-scei-ps4 %s 2>&1 | FileCheck -check-prefix=WARN-SYS-HEADERS -check-prefix=NO-WARN %s
// RUN: env SCE_PS4_SDK_DIR=.. %clang -### -emit-ast -target x86_64-scei-ps4 %s 2>&1 | FileCheck -check-prefix=WARN-SYS-HEADERS -check-prefix=NO-WARN %s
// RUN: env SCE_PS4_SDK_DIR=.. %clang -### -isysroot foo -target x86_64-scei-ps4 %s 2>&1 | FileCheck -check-prefix=WARN-ISYSROOT -check-prefix=WARN-SYS-LIBS -check-prefix=NO-WARN %s

// RUN: env SCE_PS4_SDK_DIR=.. %clang -### -c -nostdinc -target x86_64-scei-ps4 %s 2>&1 | FileCheck -check-prefix=NO-WARN %s
// RUN: env SCE_PS4_SDK_DIR=.. %clang -### -S -nostdinc -target x86_64-scei-ps4 %s 2>&1 | FileCheck -check-prefix=NO-WARN %s
// RUN: env SCE_PS4_SDK_DIR=.. %clang -### -E -nostdinc -target x86_64-scei-ps4 %s 2>&1 | FileCheck -check-prefix=NO-WARN %s
// RUN: env SCE_PS4_SDK_DIR=.. %clang -### -emit-ast -nostdinc -target x86_64-scei-ps4 %s 2>&1 | FileCheck -check-prefix=NO-WARN %s

// RUN: env SCE_PS4_SDK_DIR=.. %clang -### -c --sysroot=foo/ -target x86_64-scei-ps4 %s 2>&1 | FileCheck -check-prefix=NO-WARN %s
// RUN: env SCE_PS4_SDK_DIR=.. %clang -### -S --sysroot=foo/ -target x86_64-scei-ps4 %s 2>&1 | FileCheck -check-prefix=NO-WARN %s
// RUN: env SCE_PS4_SDK_DIR=.. %clang -### -E --sysroot=foo/ -target x86_64-scei-ps4 %s 2>&1 | FileCheck -check-prefix=NO-WARN %s
// RUN: env SCE_PS4_SDK_DIR=.. %clang -### -emit-ast --sysroot=foo/ -target x86_64-scei-ps4 %s 2>&1 | FileCheck -check-prefix=NO-WARN %s

// RUN: env SCE_PS4_SDK_DIR=.. %clang -### -c -isysroot foo -target x86_64-scei-ps4 %s 2>&1 | FileCheck -check-prefix=WARN-ISYSROOT -check-prefix=NO-WARN %s
// RUN: env SCE_PS4_SDK_DIR=.. %clang -### -S -isysroot foo -target x86_64-scei-ps4 %s 2>&1 | FileCheck -check-prefix=WARN-ISYSROOT -check-prefix=NO-WARN %s
// RUN: env SCE_PS4_SDK_DIR=.. %clang -### -E -isysroot foo -target x86_64-scei-ps4 %s 2>&1 | FileCheck -check-prefix=WARN-ISYSROOT -check-prefix=NO-WARN %s
// RUN: env SCE_PS4_SDK_DIR=.. %clang -### -emit-ast -isysroot foo -target x86_64-scei-ps4 %s 2>&1 | FileCheck -check-prefix=WARN-ISYSROOT -check-prefix=NO-WARN %s
// RUN: env SCE_PS4_SDK_DIR=.. %clang -### --sysroot=foo/ -isysroot foo -target x86_64-scei-ps4 %s 2>&1 | FileCheck -check-prefix=WARN-ISYSROOT -check-prefix=NO-WARN %s

// RUN: env SCE_PS4_SDK_DIR=.. %clang -### -nostdlib -target x86_64-scei-ps4 %s 2>&1 | FileCheck -check-prefix=WARN-SYS-HEADERS -check-prefix=NO-WARN %s
// RUN: env SCE_PS4_SDK_DIR=.. %clang -### -nodefaultlibs -target x86_64-scei-ps4 %s 2>&1 | FileCheck -check-prefix=WARN-SYS-HEADERS -check-prefix=NO-WARN %s

// NO-WARN-NOT: {{warning:|error:}}
// WARN-SYS-HEADERS: warning: unable to find PS4 system headers directory
// WARN-ISYSROOT: warning: no such sysroot directory: 'foo'
// WARN-SYS-LIBS: warning: unable to find PS4 system libraries directory
// NO-WARN-NOT: {{warning:|error:}}
