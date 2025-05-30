// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// EXTRA-ARGS: --dump-sem-ir-ranges=only
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/builtins/float/negate.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/builtins/float/negate.carbon

// --- float_negate.carbon

library "[[@TEST_NAME]]";

fn Negate(a: f64) -> f64 = "float.negate";

fn RuntimeCallIsValid(a: f64, b: f64) -> f64 {
  //@dump-sem-ir-begin
  return Negate(a);
  //@dump-sem-ir-end
}

let a: f64 = Negate(1.5);

// --- fail_too_few.carbon

library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_too_few.carbon:[[@LINE+4]]:1: error: invalid signature for builtin function "float.negate" [InvalidBuiltinSignature]
// CHECK:STDERR: fn TooFew() -> f64 = "float.negate";
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
fn TooFew() -> f64 = "float.negate";

fn RuntimeCallIsValidTooFew() -> f64 {
  return TooFew();
}

// --- fail_too_many.carbon

library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_too_many.carbon:[[@LINE+4]]:1: error: invalid signature for builtin function "float.negate" [InvalidBuiltinSignature]
// CHECK:STDERR: fn TooMany(a: f64, b: f64) -> f64 = "float.negate";
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
fn TooMany(a: f64, b: f64) -> f64 = "float.negate";

fn RuntimeCallIsValidTooMany(a: f64, b: f64) -> f64 {
  return TooMany(a, b);
}

// --- fail_bad_return_type.carbon

library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_bad_return_type.carbon:[[@LINE+4]]:1: error: invalid signature for builtin function "float.negate" [InvalidBuiltinSignature]
// CHECK:STDERR: fn BadReturnType(a: f64) -> bool = "float.negate";
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
fn BadReturnType(a: f64) -> bool = "float.negate";

fn RuntimeCallIsValidBadReturnType(a: f64) -> bool {
  return BadReturnType(a);
}

// CHECK:STDOUT: --- float_negate.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Negate.type: type = fn_type @Negate [concrete]
// CHECK:STDOUT:   %Negate: %Negate.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @RuntimeCallIsValid(%a.param: f64, %b.param: f64) -> f64 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Negate.ref: %Negate.type = name_ref Negate, file.%Negate.decl [concrete = constants.%Negate]
// CHECK:STDOUT:   %a.ref: f64 = name_ref a, %a
// CHECK:STDOUT:   %float.negate: init f64 = call %Negate.ref(%a.ref)
// CHECK:STDOUT:   %.loc8_19.1: f64 = value_of_initializer %float.negate
// CHECK:STDOUT:   %.loc8_19.2: f64 = converted %float.negate, %.loc8_19.1
// CHECK:STDOUT:   return %.loc8_19.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
