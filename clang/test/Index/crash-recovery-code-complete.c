// RUN: env CINDEXTEST_EDITING=1 \
// RUN:   not c-index-test -code-completion-at=%s:20:1 \
// RUN:   "-remap-file=%s;%S/Inputs/crash-recovery-code-complete-remap.c" \
// RUN:   %s 2> %t.err
// RUN: FileCheck < %t.err -check-prefix=CHECK-CODE-COMPLETE-CRASH %s
// CHECK-CODE-COMPLETE-CRASH: Unable to perform code completion!
//
// XFAIL: win32

#warning parsing original file
