// RUN: not %clang_cc1 -fsyntax-only -triple i686-win32 %s 2>&1 \
// RUN:   | FileCheck %s -check-prefix CHECK-NO-COMPAT
// XFAIL: win32

// RUN: not %clang_cc1 -fsyntax-only -fms-compatibility -triple i686-win32 %s 2>&1 \
// RUN:   | FileCheck %s -check-prefix CHECK-COMPAT

#include "Inputs\success.h"

// CHECK-NO-COMPAT: error: 'Inputs\success.h' file not found
// CHECK-NO-COMPAT: #include "Inputs\success.h"
// CHECK-NO-COMPAT:          ^

// CHECK-COMPAT: error: success
