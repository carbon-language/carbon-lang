// Disabled, pending investigation.
// RUN: false
// XFAIL: *
//
// RUNX: env CINDEXTEST_EDITING=1 \
// RUNX:   not c-index-test -test-load-source-reparse 1 local \
// RUNX:   -remap-file="%s;%S/Inputs/crash-recovery-reparse-remap.c" \
// RUNX:   %s 2> %t.err
// RUNX: FileCheck < %t.err -check-prefix=CHECK-REPARSE-SOURCE-CRASH %s
// CHECK-REPARSE-SOURCE-CRASH: Unable to reparse translation unit
//
// XFAIL: win32

#warning parsing original file
