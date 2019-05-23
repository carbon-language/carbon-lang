// RUN: %clang_cc1 -analyzer-checker-option-help 2>&1 | FileCheck %s

// RUN: %clang_cc1 -analyzer-checker-option-help-developer \
// RUN:   2>&1 | FileCheck %s -check-prefix=CHECK-HIDDEN

// CHECK: OVERVIEW: Clang Static Analyzer Checker and Package Option List
//
// CHECK: USAGE: -analyzer-config <OPTION1=VALUE,OPTION2=VALUE,...>
//
// CHECK:        -analyzer-config OPTION1=VALUE, -analyzer-config
// CHECK-SAME:   OPTION2=VALUE, ...
//
// CHECK: OPTIONS:
//
// CHECK:   alpha.clone.CloneChecker:MinimumCloneComplexity
// CHECK-SAME:   (int) Ensures that every clone has at least
// CHECK:        the given complexity. Complexity is here
// CHECK:        defined as the total amount of children
// CHECK:        of a statement. This constraint assumes
// CHECK:        the first statement in the group is representative
// CHECK:        for all other statements in the group in
// CHECK:        terms of complexity. (default: 50)

// CHECK-NOT:     optin.cplusplus.UninitializedObject:NotesAsWarnings
// CHECK-HIDDEN:  optin.cplusplus.UninitializedObject:NotesAsWarnings
