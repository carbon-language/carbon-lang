// Note that the run lines are at the bottom of this file.

#define_other_target TEST1
TEST1   // diagnose

#define_other_target TEST2
#undef TEST2
TEST2   // no diagnose

#define_other_target TEST3
#define TEST3
TEST3   // no diagnose

#define TEST4
#define_other_target TEST4
TEST4   // diagnose


// check success:
// RUN: clang -Eonly %s &&

// Check proper # of notes is emitted.
// RUN: clang -Eonly %s 2>&1 | grep note | wc -l | grep 2 &&

// Check that the diagnostics are the right ones.
// RUN: clang %s -Eonly -fno-caret-diagnostics 2>&1 | grep ':4:1: note' &&
// RUN: clang %s -Eonly -fno-caret-diagnostics 2>&1 | grep ':16:1: note'
