// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

// --- non_aggregate_empty_struct.h

struct X {
  X() {}
};

// --- non_aggregate_empty_struct.carbon

import Cpp library "non_aggregate_empty_struct.h";

// This previously compiled fine or crashed later but should now be an error
// because X is empty but NOT an aggregate, so it shouldn't get the
// empty struct type representation.
fn F() {
  // CHECK:STDERR: non_aggregate_empty_struct.carbon:[[@LINE+1]]:19: ERROR: Cannot initialize Cpp type `X` from struct type `{}<{}>`.
  var x: Cpp.X = {};
}
