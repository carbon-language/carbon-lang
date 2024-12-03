// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/tuple/access/element_access.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/tuple/access/element_access.carbon

var a: (i32,) = (12,);
var b: (i32,) = a;
var c: i32 = b.0;

// CHECK:STDOUT: --- element_access.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %int_32 [template]
// CHECK:STDOUT:   %tuple.type.1: type = tuple_type (type) [template]
// CHECK:STDOUT:   %tuple.type.2: type = tuple_type (%i32) [template]
// CHECK:STDOUT:   %int_12.1: Core.IntLiteral = int_value 12 [template]
// CHECK:STDOUT:   %tuple.type.3: type = tuple_type (Core.IntLiteral) [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %Convert.type.14: type = fn_type @Convert.2, @impl.1(%int_32) [template]
// CHECK:STDOUT:   %Convert.14: %Convert.type.14 = struct_value () [template]
// CHECK:STDOUT:   %interface.9: <witness> = interface_witness (%Convert.14) [template]
// CHECK:STDOUT:   %Convert.bound: <bound method> = bound_method %int_12.1, %Convert.14 [template]
// CHECK:STDOUT:   %Convert.specific_fn: <specific function> = specific_function %Convert.bound, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_12.2: %i32 = int_value 12 [template]
// CHECK:STDOUT:   %tuple: %tuple.type.2 = tuple_value (%int_12.2) [template]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [template]
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
// CHECK:STDOUT:     .a = %a
// CHECK:STDOUT:     .b = %b
// CHECK:STDOUT:     .c = %c
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %int_32.loc11: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %int.make_type_signed.loc11: init type = call constants.%Int(%int_32.loc11) [template = constants.%i32]
// CHECK:STDOUT:   %.loc11_13.1: %tuple.type.1 = tuple_literal (%int.make_type_signed.loc11)
// CHECK:STDOUT:   %.loc11_13.2: type = value_of_initializer %int.make_type_signed.loc11 [template = constants.%i32]
// CHECK:STDOUT:   %.loc11_13.3: type = converted %int.make_type_signed.loc11, %.loc11_13.2 [template = constants.%i32]
// CHECK:STDOUT:   %.loc11_13.4: type = converted %.loc11_13.1, constants.%tuple.type.2 [template = constants.%tuple.type.2]
// CHECK:STDOUT:   %a.var: ref %tuple.type.2 = var a
// CHECK:STDOUT:   %a: ref %tuple.type.2 = bind_name a, %a.var
// CHECK:STDOUT:   %int_32.loc12: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %int.make_type_signed.loc12: init type = call constants.%Int(%int_32.loc12) [template = constants.%i32]
// CHECK:STDOUT:   %.loc12_13.1: %tuple.type.1 = tuple_literal (%int.make_type_signed.loc12)
// CHECK:STDOUT:   %.loc12_13.2: type = value_of_initializer %int.make_type_signed.loc12 [template = constants.%i32]
// CHECK:STDOUT:   %.loc12_13.3: type = converted %int.make_type_signed.loc12, %.loc12_13.2 [template = constants.%i32]
// CHECK:STDOUT:   %.loc12_13.4: type = converted %.loc12_13.1, constants.%tuple.type.2 [template = constants.%tuple.type.2]
// CHECK:STDOUT:   %b.var: ref %tuple.type.2 = var b
// CHECK:STDOUT:   %b: ref %tuple.type.2 = bind_name b, %b.var
// CHECK:STDOUT:   %int_32.loc13: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %int.make_type_signed.loc13: init type = call constants.%Int(%int_32.loc13) [template = constants.%i32]
// CHECK:STDOUT:   %.loc13_8.1: type = value_of_initializer %int.make_type_signed.loc13 [template = constants.%i32]
// CHECK:STDOUT:   %.loc13_8.2: type = converted %int.make_type_signed.loc13, %.loc13_8.1 [template = constants.%i32]
// CHECK:STDOUT:   %c.var: ref %i32 = var c
// CHECK:STDOUT:   %c: ref %i32 = bind_name c, %c.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_12: Core.IntLiteral = int_value 12 [template = constants.%int_12.1]
// CHECK:STDOUT:   %.loc11_21.1: %tuple.type.3 = tuple_literal (%int_12)
// CHECK:STDOUT:   %impl.elem0: %Convert.type.2 = interface_witness_access constants.%interface.9, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %Convert.bound: <bound method> = bound_method %int_12, %impl.elem0 [template = constants.%Convert.bound]
// CHECK:STDOUT:   %Convert.specific_fn: <specific function> = specific_function %Convert.bound, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %int.convert_checked: init %i32 = call %Convert.specific_fn(%int_12) [template = constants.%int_12.2]
// CHECK:STDOUT:   %.loc11_21.2: init %i32 = converted %int_12, %int.convert_checked [template = constants.%int_12.2]
// CHECK:STDOUT:   %.loc11_21.3: init %tuple.type.2 = tuple_init (%.loc11_21.2) to file.%a.var [template = constants.%tuple]
// CHECK:STDOUT:   %.loc11_22: init %tuple.type.2 = converted %.loc11_21.1, %.loc11_21.3 [template = constants.%tuple]
// CHECK:STDOUT:   assign file.%a.var, %.loc11_22
// CHECK:STDOUT:   %a.ref: ref %tuple.type.2 = name_ref a, file.%a
// CHECK:STDOUT:   %tuple.elem0.loc12: ref %i32 = tuple_access %a.ref, element0
// CHECK:STDOUT:   %.loc12_17.1: %i32 = bind_value %tuple.elem0.loc12
// CHECK:STDOUT:   %.loc12_17.2: init %tuple.type.2 = tuple_init (%.loc12_17.1) to file.%b.var
// CHECK:STDOUT:   %.loc12_18: init %tuple.type.2 = converted %a.ref, %.loc12_17.2
// CHECK:STDOUT:   assign file.%b.var, %.loc12_18
// CHECK:STDOUT:   %b.ref: ref %tuple.type.2 = name_ref b, file.%b
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [template = constants.%int_0]
// CHECK:STDOUT:   %tuple.elem0.loc13: ref %i32 = tuple_access %b.ref, element0
// CHECK:STDOUT:   %.loc13: %i32 = bind_value %tuple.elem0.loc13
// CHECK:STDOUT:   assign file.%c.var, %.loc13
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
