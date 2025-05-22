// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// EXTRA-ARGS: --dump-sem-ir-ranges=only
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/no_prelude/indirect_import_member.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/no_prelude/indirect_import_member.carbon

// ============================================================================
// Setup files
// ============================================================================

// --- a.carbon

library "[[@TEST_NAME]]";

class C {
  fn F() {}
}

// --- b.carbon

library "[[@TEST_NAME]]";

export import library "a";

// --- c.carbon

library "[[@TEST_NAME]]";

import library "b";

export C;

// --- d.carbon

library "[[@TEST_NAME]]";

export import library "c";

// --- e.carbon

library "[[@TEST_NAME]]";

import library "c";

class D {
  alias C = package.C;
}

// --- f.carbon

library "[[@TEST_NAME]]";

export import library "e";

// ============================================================================
// Test files
// ============================================================================

// --- use_b.carbon

library "[[@TEST_NAME]]";

import library "b";

var x: () = C.F();

// --- use_c.carbon

library "[[@TEST_NAME]]";

import library "c";

var x: () = C.F();

// --- use_d.carbon

library "[[@TEST_NAME]]";

import library "d";

var x: () = C.F();

// --- use_e.carbon

library "[[@TEST_NAME]]";

import library "e";

var x: () = D.C.F();

// --- use_f.carbon

library "[[@TEST_NAME]]";

import library "f";

var x: () = D.C.F();
