// RUN: env CINDEXTEST_EDITING=1 \
// RUN:   not c-index-test -test-load-source-reparse 1 local \
// RUN:   -remap-file="%s;%S/Inputs/crash-recovery-reparse-remap.c" \
// RUN:   %s 2> %t.err
// RUN: FileCheck < %t.err -check-prefix=CHECK-REPARSE-SOURCE-CRASH %s
// CHECK-REPARSE-SOURCE-CRASH: Unable to reparse translation unit
//
// REQUIRES: crash-recovery

#warning parsing original file
