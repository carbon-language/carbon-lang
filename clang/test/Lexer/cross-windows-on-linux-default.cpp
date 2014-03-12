// RUN: not %clang_cc1 -fsyntax-only -fms-compatibility -triple i686-win32 %s 2>&1 \
// RUN:   | FileCheck %s

#include "Inputs\success.h"

// CHECK: error: success
