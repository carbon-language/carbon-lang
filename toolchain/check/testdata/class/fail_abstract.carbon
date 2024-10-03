// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/fail_abstract.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/fail_abstract.carbon

abstract class Abstract {
  var a: i32;
}

class Derived {
  // CHECK:STDERR: fail_abstract.carbon:[[@LINE+7]]:16: error: base `Abstract` is an incomplete type
  // CHECK:STDERR:   extend base: Abstract;
  // CHECK:STDERR:                ^~~~~~~~
  // CHECK:STDERR: fail_abstract.carbon:[[@LINE-8]]:1: note: class was declared abstract here
  // CHECK:STDERR: abstract class Abstract {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  extend base: Abstract;

  var d: i32;
}

fn Make() -> Derived {
  // TODO: This should be valid, and should construct an instance of `partial Abstract` as the base.
  return {.base = {.a = 1}, .d = 7};
}

// CHECK:STDERR: fail_abstract.carbon:[[@LINE+7]]:13: error: function returns incomplete type `Abstract`
// CHECK:STDERR: fn Return() -> Abstract {
// CHECK:STDERR:             ^~~~~~~~~~~
// CHECK:STDERR: fail_abstract.carbon:[[@LINE-25]]:1: note: class was declared abstract here
// CHECK:STDERR: abstract class Abstract {
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
fn Return() -> Abstract {
  return {.a = 1};
}

// TODO: This should be valid.
// CHECK:STDERR: fail_abstract.carbon:[[@LINE+12]]:31: error: expected expression
// CHECK:STDERR: fn ReturnPartialAbstract() -> partial Abstract {
// CHECK:STDERR:                               ^~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_abstract.carbon:[[@LINE+8]]:31: error: `fn` declarations must either end with a `;` or have a `{ ... }` block for a definition
// CHECK:STDERR: fn ReturnPartialAbstract() -> partial Abstract {
// CHECK:STDERR:                               ^~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_abstract.carbon:[[@LINE+4]]:31: error: semantics TODO: `HandleInvalidParse`
// CHECK:STDERR: fn ReturnPartialAbstract() -> partial Abstract {
// CHECK:STDERR:                               ^~~~~~~
// CHECK:STDERR:
fn ReturnPartialAbstract() -> partial Abstract {
  return {.a = 1};
}

fn Access(d: Derived) -> (i32, i32) {
  return (d.d, d.base.a);
}

fn Var() {
  // TODO: This should be invalid.
  var v: Abstract;
  // TODO: This should be valid.
  // CHECK:STDERR: fail_abstract.carbon:[[@LINE+3]]:10: error: expected expression
  // CHECK:STDERR:   var v: partial Abstract;
  // CHECK:STDERR:          ^~~~~~~
  var v: partial Abstract;
}

// CHECK:STDOUT: --- fail_abstract.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Abstract: type = class_type @Abstract [template]
// CHECK:STDOUT:   %Int32.type: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %Int32: %Int32.type = struct_value () [template]
// CHECK:STDOUT:   %.2: type = unbound_element_type %Abstract, i32 [template]
// CHECK:STDOUT:   %.3: type = struct_type {.a: i32} [template]
// CHECK:STDOUT:   %.4: <witness> = complete_type_witness %.3 [template]
// CHECK:STDOUT:   %Derived: type = class_type @Derived [template]
// CHECK:STDOUT:   %.5: type = unbound_element_type %Derived, i32 [template]
// CHECK:STDOUT:   %Make.type: type = fn_type @Make [template]
// CHECK:STDOUT:   %Make: %Make.type = struct_value () [template]
// CHECK:STDOUT:   %.6: i32 = int_literal 1 [template]
// CHECK:STDOUT:   %.7: i32 = int_literal 7 [template]
// CHECK:STDOUT:   %.8: type = struct_type {.base: %.3, .d: i32} [template]
// CHECK:STDOUT:   %Return.type: type = fn_type @Return [template]
// CHECK:STDOUT:   %Return: %Return.type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {}
// CHECK:STDOUT:
// CHECK:STDOUT: class @Abstract {
// CHECK:STDOUT:   %int.make_type_32: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc12_10.1: type = value_of_initializer %int.make_type_32 [template = i32]
// CHECK:STDOUT:   %.loc12_10.2: type = converted %int.make_type_32, %.loc12_10.1 [template = i32]
// CHECK:STDOUT:   %.loc12_8: %.2 = field_decl a, element0 [template]
// CHECK:STDOUT:   %.loc13: <witness> = complete_type_witness %.3 [template = constants.%.4]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Abstract
// CHECK:STDOUT:   .a = %.loc12_8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Derived {
// CHECK:STDOUT:   %Abstract.ref: type = name_ref Abstract, <unexpected>.inst+3.loc11_25 [template = constants.%Abstract]
// CHECK:STDOUT:   %.loc23: <error> = base_decl <error>, element0 [template]
// CHECK:STDOUT:   %int.make_type_32: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc25_10.1: type = value_of_initializer %int.make_type_32 [template = i32]
// CHECK:STDOUT:   %.loc25_10.2: type = converted %int.make_type_32, %.loc25_10.1 [template = i32]
// CHECK:STDOUT:   %.loc25_8: %.5 = field_decl d, element1 [template]
// CHECK:STDOUT:   %.loc26: <witness> = complete_type_witness <error> [template = <error>]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Derived
// CHECK:STDOUT:   .base = %.loc23
// CHECK:STDOUT:   .d = %.loc25_8
// CHECK:STDOUT:   has_error
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Make() -> %return: %Derived {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc30_25: i32 = int_literal 1 [template = constants.%.6]
// CHECK:STDOUT:   %.loc30_26: %.3 = struct_literal (%.loc30_25)
// CHECK:STDOUT:   %.loc30_34: i32 = int_literal 7 [template = constants.%.7]
// CHECK:STDOUT:   %.loc30_35: %.8 = struct_literal (%.loc30_26, %.loc30_34)
// CHECK:STDOUT:   return <error> to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Return() -> %Abstract {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc41_16: i32 = int_literal 1 [template = constants.%.6]
// CHECK:STDOUT:   %.loc41_17: %.3 = struct_literal (%.loc41_16)
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
