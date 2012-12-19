// RUN: not %clang_cc1 -fsyntax-only %s 2>&1 | FileCheck %s

#include "macro_arg_slocentry_merge.h"

// CHECK: macro_arg_slocentry_merge.h:7:19: error: unknown type name 'win'
// CHECK: macro_arg_slocentry_merge.h:5:16: note: expanded from macro 'WINDOW'
// CHECK: macro_arg_slocentry_merge.h:6:18: note: expanded from macro 'P_'
