// RUN: %clang -target x86_64-pc-win32 -### -no-integrated-as %s -c 2>&1 | FileCheck %s

// CHECK: there is no external assembler we can use on windows
