// RUN: mkdir -p %t/sub/dir
// RUN: echo > %t/sub/empty.h

// Test that system header paths are expanded
//
// RUN: %clang -fsyntax-only -MD -MF %t.d -MT foo %s -isystem %t/sub/dir/..
// RUN: FileCheck -check-prefix=TEST1 %s < %t.d
// TEST1: foo:
// TEST1: sub{{/|\\}}empty.h

// Test that system header paths are not expanded to a longer form
//
// RUN: cd %t && %clang -fsyntax-only -MD -MF %t.d -MT foo %s -isystem sub/dir/..
// RUN: FileCheck -check-prefix=TEST2 %s < %t.d
// TEST2: foo:
// TEST2: sub/dir/..{{/|\\}}empty.h

// Test that user header paths are not expanded
//
// RUN: %clang -fsyntax-only -MD -MF %t.d -MT foo %s -I %t/sub/dir/..
// RUN: FileCheck -check-prefix=TEST3 %s < %t.d
// TEST3: foo:
// TEST3: sub/dir/..{{/|\\}}empty.h

// Test that system header paths are not expanded with -fno-canonical-system-headers
// (and also that the -fsystem-system-headers option is accepted)
//
// RUN: %clang -fsyntax-only -MD -MF %t.d -MT foo %s -I %t/sub/dir/.. -fcanonical-system-headers -fno-canonical-system-headers
// RUN: FileCheck -check-prefix=TEST4 %s < %t.d
// TEST4: foo:
// TEST4: sub/dir/..{{/|\\}}empty.h

#include <empty.h>
