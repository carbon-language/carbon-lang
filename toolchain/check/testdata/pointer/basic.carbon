// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/pointer/basic.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/pointer/basic.carbon

fn F() -> i32 {
  var n: i32 = 0;
  var p: i32* = &n;

  return *p;
}

// CHECK:STDOUT: --- basic.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int.32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %int.32 [template]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT:   %int.0.1: Core.IntLiteral = int_value 0 [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %Convert.type.14: type = fn_type @Convert.2, @impl.1(%int.32) [template]
// CHECK:STDOUT:   %Convert.14: %Convert.type.14 = struct_value () [template]
// CHECK:STDOUT:   %interface.9: <witness> = interface_witness (%Convert.14) [template]
// CHECK:STDOUT:   %Convert.bound: <bound method> = bound_method %int.0.1, %Convert.14 [template]
// CHECK:STDOUT:   %Convert.specific_fn: <specific function> = specific_function %Convert.bound, @Convert.2(%int.32) [template]
// CHECK:STDOUT:   %int.0.2: %i32 = int_value 0 [template]
// CHECK:STDOUT:   %.5: type = ptr_type %i32 [template]
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
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [template = constants.%F] {
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int.32.loc11: Core.IntLiteral = int_value 32 [template = constants.%int.32]
// CHECK:STDOUT:     %int.make_type_signed.loc11: init type = call constants.%Int(%int.32.loc11) [template = constants.%i32]
// CHECK:STDOUT:     %.loc11_11.1: type = value_of_initializer %int.make_type_signed.loc11 [template = constants.%i32]
// CHECK:STDOUT:     %.loc11_11.2: type = converted %int.make_type_signed.loc11, %.loc11_11.1 [template = constants.%i32]
// CHECK:STDOUT:     %return.param: ref %i32 = out_param runtime_param0
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int.32.loc12: Core.IntLiteral = int_value 32 [template = constants.%int.32]
// CHECK:STDOUT:   %int.make_type_signed.loc12: init type = call constants.%Int(%int.32.loc12) [template = constants.%i32]
// CHECK:STDOUT:   %.loc12_10.1: type = value_of_initializer %int.make_type_signed.loc12 [template = constants.%i32]
// CHECK:STDOUT:   %.loc12_10.2: type = converted %int.make_type_signed.loc12, %.loc12_10.1 [template = constants.%i32]
// CHECK:STDOUT:   %n.var: ref %i32 = var n
// CHECK:STDOUT:   %n: ref %i32 = bind_name n, %n.var
// CHECK:STDOUT:   %int.0: Core.IntLiteral = int_value 0 [template = constants.%int.0.1]
// CHECK:STDOUT:   %.loc12_17.1: %Convert.type.2 = interface_witness_access constants.%interface.9, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %Convert.bound: <bound method> = bound_method %int.0, %.loc12_17.1 [template = constants.%Convert.bound]
// CHECK:STDOUT:   %Convert.specific_fn: <specific function> = specific_function %Convert.bound, @Convert.2(constants.%int.32) [template = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %int.convert_checked: init %i32 = call %Convert.specific_fn(%int.0) [template = constants.%int.0.2]
// CHECK:STDOUT:   %.loc12_17.2: init %i32 = converted %int.0, %int.convert_checked [template = constants.%int.0.2]
// CHECK:STDOUT:   assign %n.var, %.loc12_17.2
// CHECK:STDOUT:   %int.32.loc13: Core.IntLiteral = int_value 32 [template = constants.%int.32]
// CHECK:STDOUT:   %int.make_type_signed.loc13: init type = call constants.%Int(%int.32.loc13) [template = constants.%i32]
// CHECK:STDOUT:   %.loc13_13.1: type = value_of_initializer %int.make_type_signed.loc13 [template = constants.%i32]
// CHECK:STDOUT:   %.loc13_13.2: type = converted %int.make_type_signed.loc13, %.loc13_13.1 [template = constants.%i32]
// CHECK:STDOUT:   %.loc13_13.3: type = ptr_type %i32 [template = constants.%.5]
// CHECK:STDOUT:   %p.var: ref %.5 = var p
// CHECK:STDOUT:   %p: ref %.5 = bind_name p, %p.var
// CHECK:STDOUT:   %n.ref: ref %i32 = name_ref n, %n
// CHECK:STDOUT:   %.loc13_17: %.5 = addr_of %n.ref
// CHECK:STDOUT:   assign %p.var, %.loc13_17
// CHECK:STDOUT:   %p.ref: ref %.5 = name_ref p, %p
// CHECK:STDOUT:   %.loc15_11: %.5 = bind_value %p.ref
// CHECK:STDOUT:   %.loc15_10.1: ref %i32 = deref %.loc15_11
// CHECK:STDOUT:   %.loc15_10.2: %i32 = bind_value %.loc15_10.1
// CHECK:STDOUT:   return %.loc15_10.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
