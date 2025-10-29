// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/none.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interface/require.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interface/require.carbon

// --- fail_todo_implicit_self_impls.carbon
library "[[@TEST_NAME]]";

interface Y {
  fn YY();
}

//@dump-sem-ir-begin
interface Z {
  // CHECK:STDERR: fail_todo_implicit_self_impls.carbon:[[@LINE+4]]:3: error: semantics TODO: `require` [SemanticsTodo]
  // CHECK:STDERR:   require impls Y;
  // CHECK:STDERR:   ^~~~~~~
  // CHECK:STDERR:
  require impls Y;
}
//@dump-sem-ir-end

fn F(T:! Z) {
  T.(Y.YY)();
}

// --- fail_todo_explicit_self_impls.carbon
library "[[@TEST_NAME]]";

interface Y {
  fn YY();
}

//@dump-sem-ir-begin
interface Z {
  // CHECK:STDERR: fail_todo_explicit_self_impls.carbon:[[@LINE+4]]:3: error: semantics TODO: `require` [SemanticsTodo]
  // CHECK:STDERR:   require Self impls Y;
  // CHECK:STDERR:   ^~~~~~~
  // CHECK:STDERR:
  require Self impls Y;
}
//@dump-sem-ir-end

fn F(T:! Z) {
  T.(Y.YY)();
}

// --- fail_todo_implicit_self_no_extend_name_lookup_fails.carbon
library "[[@TEST_NAME]]";

interface Y {
  fn YY();
}

interface Z {
  // CHECK:STDERR: fail_todo_implicit_self_no_extend_name_lookup_fails.carbon:[[@LINE+4]]:3: error: semantics TODO: `require` [SemanticsTodo]
  // CHECK:STDERR:   require impls Y;
  // CHECK:STDERR:   ^~~~~~~
  // CHECK:STDERR:
  require impls Y;
}

fn F(T:! Z) {
  // TODO: This should fail name lookup since Z does not extend Y.
  T.YY();
}

// --- fail_todo_explicit_self_no_extend_name_lookup_fails.carbon
library "[[@TEST_NAME]]";

interface Y {
  fn YY();
}

interface Z {
  // CHECK:STDERR: fail_todo_explicit_self_no_extend_name_lookup_fails.carbon:[[@LINE+4]]:3: error: semantics TODO: `require` [SemanticsTodo]
  // CHECK:STDERR:   require Self impls Y;
  // CHECK:STDERR:   ^~~~~~~
  // CHECK:STDERR:
  require Self impls Y;
}

fn F(T:! Z) {
  // TODO: This should fail name lookup since Z does not extend Y.
  T.YY();
}

// --- fail_todo_explicit_self_specific_impls.carbon
library "[[@TEST_NAME]]";

interface Y {}

class C(T:! type);

//@dump-sem-ir-begin
interface Z {
  // CHECK:STDERR: fail_todo_explicit_self_specific_impls.carbon:[[@LINE+4]]:3: error: semantics TODO: `require` [SemanticsTodo]
  // CHECK:STDERR:   require C(Self) impls Y;
  // CHECK:STDERR:   ^~~~~~~
  // CHECK:STDERR:
  require C(Self) impls Y;
}
//@dump-sem-ir-end

// --- fail_todo_impls_where.carbon
library "[[@TEST_NAME]]";

interface Y { let Y1:! type; }

//@dump-sem-ir-begin
interface Z {
  // CHECK:STDERR: fail_todo_impls_where.carbon:[[@LINE+4]]:3: error: semantics TODO: `require` [SemanticsTodo]
  // CHECK:STDERR:   require impls Y where .Y1 = ();
  // CHECK:STDERR:   ^~~~~~~
  // CHECK:STDERR:
  require impls Y where .Y1 = ();
}
//@dump-sem-ir-end

// --- fail_todo_other_impls_with_self.carbon
library "[[@TEST_NAME]]";

interface Y {}

//@dump-sem-ir-begin
interface Z(T:! type) {
  // CHECK:STDERR: fail_todo_other_impls_with_self.carbon:[[@LINE+4]]:3: error: semantics TODO: `require` [SemanticsTodo]
  // CHECK:STDERR:   require T impls Z(Self);
  // CHECK:STDERR:   ^~~~~~~
  // CHECK:STDERR:
  require T impls Z(Self);
}
//@dump-sem-ir-end

// --- fail_todo_other_impls_without_self.carbon
library "[[@TEST_NAME]]";

interface Y {}

//@dump-sem-ir-begin
interface Z(T:! type) {
  // TODO: Either the type `T` or the facet type `Y` must mention `Self`, but
  // they don't.
  // CHECK:STDERR: fail_todo_other_impls_without_self.carbon:[[@LINE+4]]:3: error: semantics TODO: `require` [SemanticsTodo]
  // CHECK:STDERR:   require T impls Y;
  // CHECK:STDERR:   ^~~~~~~
  // CHECK:STDERR:
  require T impls Y;
}
//@dump-sem-ir-end

// --- fail_todo_extend.carbon
library "[[@TEST_NAME]]";

interface Y {
  fn YY();
}

//@dump-sem-ir-begin
interface Z {
  // CHECK:STDERR: fail_todo_extend.carbon:[[@LINE+4]]:10: error: semantics TODO: `require` [SemanticsTodo]
  // CHECK:STDERR:   extend require impls Y;
  // CHECK:STDERR:          ^~~~~~~
  // CHECK:STDERR:
  extend require impls Y;
}
//@dump-sem-ir-end

fn F(T:! Z) {
  T.YY();
  T.(Y.YY)();
}

// CHECK:STDOUT: --- fail_todo_implicit_self_impls.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Z {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = <unexpected>.inst6000001E
// CHECK:STDOUT:   witness = invalid
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_todo_explicit_self_impls.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Z {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = <unexpected>.inst5000001E
// CHECK:STDOUT:   witness = invalid
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_todo_explicit_self_specific_impls.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Z {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = <unexpected>.inst68000024
// CHECK:STDOUT:   witness = invalid
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_todo_impls_where.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Z {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = <unexpected>.inst5800001B
// CHECK:STDOUT:   witness = invalid
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_todo_other_impls_with_self.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @Z(<unexpected>.inst78000018.loc6_13: type) {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = <unexpected>.inst78000022
// CHECK:STDOUT:     witness = invalid
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Z(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_todo_other_impls_without_self.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @Z(<unexpected>.inst44000018.loc6_13: type) {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = <unexpected>.inst44000022
// CHECK:STDOUT:     witness = invalid
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Z(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_todo_extend.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Z {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = <unexpected>.inst6400001E
// CHECK:STDOUT:   witness = invalid
// CHECK:STDOUT: }
// CHECK:STDOUT:
