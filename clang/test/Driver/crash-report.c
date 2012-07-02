// RUN: rm %T/crash-report-*.c %T/crash-report-*.sh
// RUN: TMP=%T %clang -fsyntax-only %s -DFOO=BAR 2>&1 | FileCheck %s
// RUN: FileCheck --check-prefix=CHECKSRC %s < %T/crash-report-*.c
// RUN: FileCheck --check-prefix=CHECKSH %s < %T/crash-report-*.sh
// REQUIRES: crash-recovery
// XFAIL: mingw32,win32

#pragma clang __debug parser_crash
// CHECK: Preprocessed source(s) and associated run script(s) are located at:
// CHECK-NEXT: note: diagnostic msg: {{.*}}.c
FOO
// CHECKSRC: FOO
// CHECKSH: -D FOO=BAR
