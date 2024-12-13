// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/fail_scope.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/fail_scope.carbon

class Class {
  fn F() -> i32 {
    return 1;
  }
}

fn G() -> i32 {
  // CHECK:STDERR: fail_scope.carbon:[[@LINE+3]]:10: error: name `F` not found [NameNotFound]
  // CHECK:STDERR:   return F();
  // CHECK:STDERR:          ^
  return F();
}

// CHECK:STDOUT: --- fail_scope.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Class: type = class_type @Class [template]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %int_32 [template]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [template]
// CHECK:STDOUT:   %complete_type.1: <witness> = complete_type_witness %empty_struct_type [template]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic]
// CHECK:STDOUT:   %iN: type = int_type signed, %N [symbolic]
// CHECK:STDOUT:   %Convert.type.3: type = fn_type @Convert.2, @impl.1(%N) [symbolic]
// CHECK:STDOUT:   %Convert.3: %Convert.type.3 = struct_value () [symbolic]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (%Convert.3) [template]
// CHECK:STDOUT:   %Convert.bound: <bound method> = bound_method %int_1, %Convert.3 [symbolic]
// CHECK:STDOUT:   %Convert.specific_fn: <specific function> = specific_function %Convert.bound, @Convert.2(%N) [symbolic]
// CHECK:STDOUT:   %int.convert_checked: init %iN = call %Convert.specific_fn(%int_1) [symbolic]
// CHECK:STDOUT:   %G.type: type = fn_type @G [template]
// CHECK:STDOUT:   %G: %G.type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int = %import_ref.1
// CHECK:STDOUT:     .ImplicitAs = %import_ref.2
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Class = %Class.decl
// CHECK:STDOUT:     .G = %G.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Class.decl: type = class_decl @Class [template = constants.%Class] {} {}
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [template = constants.%G] {
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed: init type = call constants.%Int(%int_32) [template = constants.%i32]
// CHECK:STDOUT:     %.loc17_11.1: type = value_of_initializer %int.make_type_signed [template = constants.%i32]
// CHECK:STDOUT:     %.loc17_11.2: type = converted %int.make_type_signed, %.loc17_11.1 [template = constants.%i32]
// CHECK:STDOUT:     %return.param: ref %i32 = out_param runtime_param0
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Class {
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [template = constants.%F] {
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed: init type = call constants.%Int(%int_32) [template = constants.%i32]
// CHECK:STDOUT:     %.loc12_13.1: type = value_of_initializer %int.make_type_signed [template = constants.%i32]
// CHECK:STDOUT:     %.loc12_13.2: type = converted %int.make_type_signed, %.loc12_13.1 [template = constants.%i32]
// CHECK:STDOUT:     %return.param: ref %i32 = out_param runtime_param0
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template = constants.%complete_type.1]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Class
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [template = constants.%int_1]
// CHECK:STDOUT:   %impl.elem0: %Convert.type.2 = impl_witness_access constants.%impl_witness, element0 [symbolic = constants.%Convert.3]
// CHECK:STDOUT:   %Convert.bound: <bound method> = bound_method %int_1, %impl.elem0 [symbolic = constants.%Convert.bound]
// CHECK:STDOUT:   %Convert.specific_fn: <specific function> = specific_function %Convert.bound, @Convert.2(constants.%N) [symbolic = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %int.convert_checked: init %iN = call %Convert.specific_fn(%int_1) [symbolic = constants.%int.convert_checked]
// CHECK:STDOUT:   %.loc13_13.1: %i32 = value_of_initializer %int.convert_checked [symbolic = constants.%int.convert_checked]
// CHECK:STDOUT:   %.loc13_13.2: %i32 = converted %int_1, %.loc13_13.1 [symbolic = constants.%int.convert_checked]
// CHECK:STDOUT:   return %.loc13_13.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: <error> = name_ref F, <error> [template = <error>]
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
