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
  extend base: Abstract;

  var d: i32;
}

fn Make() -> Derived {
  // TODO: This should be valid, and should construct an instance of `partial Abstract` as the base.
  // CHECK:STDERR: fail_abstract.carbon:[[@LINE+4]]:19: error: cannot construct instance of abstract class; consider using `partial Abstract` instead
  // CHECK:STDERR:   return {.base = {.a = 1}, .d = 7};
  // CHECK:STDERR:                   ^~~~~~~~
  // CHECK:STDERR:
  return {.base = {.a = 1}, .d = 7};
}

fn Return() -> Abstract {
  // CHECK:STDERR: fail_abstract.carbon:[[@LINE+4]]:10: error: cannot construct instance of abstract class; consider using `partial Abstract` instead
  // CHECK:STDERR:   return {.a = 1};
  // CHECK:STDERR:          ^~~~~~~~
  // CHECK:STDERR:
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
// CHECK:STDOUT:   %.5: type = ptr_type %.3 [template]
// CHECK:STDOUT:   %.6: type = unbound_element_type %Derived, %Abstract [template]
// CHECK:STDOUT:   %.7: type = unbound_element_type %Derived, i32 [template]
// CHECK:STDOUT:   %.8: type = struct_type {.base: %Abstract, .d: i32} [template]
// CHECK:STDOUT:   %.9: <witness> = complete_type_witness %.8 [template]
// CHECK:STDOUT:   %Make.type: type = fn_type @Make [template]
// CHECK:STDOUT:   %Make: %Make.type = struct_value () [template]
// CHECK:STDOUT:   %.10: type = struct_type {.base: %.5, .d: i32} [template]
// CHECK:STDOUT:   %.11: type = ptr_type %.10 [template]
// CHECK:STDOUT:   %.12: type = ptr_type %.8 [template]
// CHECK:STDOUT:   %.13: i32 = int_literal 1 [template]
// CHECK:STDOUT:   %.14: i32 = int_literal 7 [template]
// CHECK:STDOUT:   %.15: type = struct_type {.base: %.3, .d: i32} [template]
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
// CHECK:STDOUT:   %.loc16: %.6 = base_decl %Abstract, element0 [template]
// CHECK:STDOUT:   %int.make_type_32: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc18_10.1: type = value_of_initializer %int.make_type_32 [template = i32]
// CHECK:STDOUT:   %.loc18_10.2: type = converted %int.make_type_32, %.loc18_10.1 [template = i32]
// CHECK:STDOUT:   %.loc18_8: %.7 = field_decl d, element1 [template]
// CHECK:STDOUT:   %.loc19: <witness> = complete_type_witness %.8 [template = constants.%.9]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Derived
// CHECK:STDOUT:   .base = %.loc16
// CHECK:STDOUT:   .d = %.loc18_8
// CHECK:STDOUT:   extend name_scope2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Make() -> %return: %Derived {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc27_25: i32 = int_literal 1 [template = constants.%.13]
// CHECK:STDOUT:   %.loc27_26: %.3 = struct_literal (%.loc27_25)
// CHECK:STDOUT:   %.loc27_34: i32 = int_literal 7 [template = constants.%.14]
// CHECK:STDOUT:   %.loc27_35: %.15 = struct_literal (%.loc27_26, %.loc27_34)
// CHECK:STDOUT:   return <error> to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Return() -> %return: %Abstract {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc35_16: i32 = int_literal 1 [template = constants.%.13]
// CHECK:STDOUT:   %.loc35_17: %.3 = struct_literal (%.loc35_16)
// CHECK:STDOUT:   return <error> to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
