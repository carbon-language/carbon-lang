// RUN: rm -rf %t
// RUN: mkdir %t
// RUN: env TMPDIR=%t TEMP=%t TMP=%t %clang -fsyntax-only %s -DFOO=BAR 2>&1 | FileCheck %s
// RUN: cat %t/crash-report-*.c | FileCheck --check-prefix=CHECKSRC %s
// RUN: cat %t/crash-report-*.sh | FileCheck --check-prefix=CHECKSH %s
// REQUIRES: crash-recovery

#pragma clang __debug parser_crash
// CHECK: Preprocessed source(s) and associated run script(s) are located at:
// CHECK-NEXT: note: diagnostic msg: {{.*}}.c
FOO
// CHECKSRC: FOO
// CHECKSH: -D FOO=BAR
