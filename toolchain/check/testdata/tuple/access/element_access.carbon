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
// CHECK:STDOUT:   %.1: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %.1 [template]
// CHECK:STDOUT:   %tuple.type.1: type = tuple_type (type) [template]
// CHECK:STDOUT:   %tuple.type.2: type = tuple_type (%i32) [template]
// CHECK:STDOUT:   %.2: Core.IntLiteral = int_value 12 [template]
// CHECK:STDOUT:   %tuple.type.3: type = tuple_type (Core.IntLiteral) [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %Convert.type.14: type = fn_type @Convert.2, @impl.1(%.1) [template]
// CHECK:STDOUT:   %Convert.14: %Convert.type.14 = struct_value () [template]
// CHECK:STDOUT:   %.26: <witness> = interface_witness (%Convert.14) [template]
// CHECK:STDOUT:   %.27: <bound method> = bound_method %.2, %Convert.14 [template]
// CHECK:STDOUT:   %.28: <specific function> = specific_function %.27, @Convert.2(%.1) [template]
// CHECK:STDOUT:   %.29: %i32 = int_value 12 [template]
// CHECK:STDOUT:   %tuple: %tuple.type.2 = tuple_value (%.29) [template]
// CHECK:STDOUT:   %.30: Core.IntLiteral = int_value 0 [template]
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
// CHECK:STDOUT:   %.loc11_9: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:   %int.make_type_signed.loc11: init type = call constants.%Int(%.loc11_9) [template = constants.%i32]
// CHECK:STDOUT:   %.loc11_13.1: %tuple.type.1 = tuple_literal (%int.make_type_signed.loc11)
// CHECK:STDOUT:   %.loc11_13.2: type = value_of_initializer %int.make_type_signed.loc11 [template = constants.%i32]
// CHECK:STDOUT:   %.loc11_13.3: type = converted %int.make_type_signed.loc11, %.loc11_13.2 [template = constants.%i32]
// CHECK:STDOUT:   %.loc11_13.4: type = converted %.loc11_13.1, constants.%tuple.type.2 [template = constants.%tuple.type.2]
// CHECK:STDOUT:   %a.var: ref %tuple.type.2 = var a
// CHECK:STDOUT:   %a: ref %tuple.type.2 = bind_name a, %a.var
// CHECK:STDOUT:   %.loc12_9: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:   %int.make_type_signed.loc12: init type = call constants.%Int(%.loc12_9) [template = constants.%i32]
// CHECK:STDOUT:   %.loc12_13.1: %tuple.type.1 = tuple_literal (%int.make_type_signed.loc12)
// CHECK:STDOUT:   %.loc12_13.2: type = value_of_initializer %int.make_type_signed.loc12 [template = constants.%i32]
// CHECK:STDOUT:   %.loc12_13.3: type = converted %int.make_type_signed.loc12, %.loc12_13.2 [template = constants.%i32]
// CHECK:STDOUT:   %.loc12_13.4: type = converted %.loc12_13.1, constants.%tuple.type.2 [template = constants.%tuple.type.2]
// CHECK:STDOUT:   %b.var: ref %tuple.type.2 = var b
// CHECK:STDOUT:   %b: ref %tuple.type.2 = bind_name b, %b.var
// CHECK:STDOUT:   %.loc13_8.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:   %int.make_type_signed.loc13: init type = call constants.%Int(%.loc13_8.1) [template = constants.%i32]
// CHECK:STDOUT:   %.loc13_8.2: type = value_of_initializer %int.make_type_signed.loc13 [template = constants.%i32]
// CHECK:STDOUT:   %.loc13_8.3: type = converted %int.make_type_signed.loc13, %.loc13_8.2 [template = constants.%i32]
// CHECK:STDOUT:   %c.var: ref %i32 = var c
// CHECK:STDOUT:   %c: ref %i32 = bind_name c, %c.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc11_18: Core.IntLiteral = int_value 12 [template = constants.%.2]
// CHECK:STDOUT:   %.loc11_21.1: %tuple.type.3 = tuple_literal (%.loc11_18)
// CHECK:STDOUT:   %.loc11_21.2: %Convert.type.2 = interface_witness_access constants.%.26, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc11_21.3: <bound method> = bound_method %.loc11_18, %.loc11_21.2 [template = constants.%.27]
// CHECK:STDOUT:   %.loc11_21.4: <specific function> = specific_function %.loc11_21.3, @Convert.2(constants.%.1) [template = constants.%.28]
// CHECK:STDOUT:   %int.convert_checked: init %i32 = call %.loc11_21.4(%.loc11_18) [template = constants.%.29]
// CHECK:STDOUT:   %.loc11_21.5: init %i32 = converted %.loc11_18, %int.convert_checked [template = constants.%.29]
// CHECK:STDOUT:   %.loc11_21.6: init %tuple.type.2 = tuple_init (%.loc11_21.5) to file.%a.var [template = constants.%tuple]
// CHECK:STDOUT:   %.loc11_22: init %tuple.type.2 = converted %.loc11_21.1, %.loc11_21.6 [template = constants.%tuple]
// CHECK:STDOUT:   assign file.%a.var, %.loc11_22
// CHECK:STDOUT:   %a.ref: ref %tuple.type.2 = name_ref a, file.%a
// CHECK:STDOUT:   %.loc12_17.1: ref %i32 = tuple_access %a.ref, element0
// CHECK:STDOUT:   %.loc12_17.2: %i32 = bind_value %.loc12_17.1
// CHECK:STDOUT:   %.loc12_17.3: init %tuple.type.2 = tuple_init (%.loc12_17.2) to file.%b.var
// CHECK:STDOUT:   %.loc12_18: init %tuple.type.2 = converted %a.ref, %.loc12_17.3
// CHECK:STDOUT:   assign file.%b.var, %.loc12_18
// CHECK:STDOUT:   %b.ref: ref %tuple.type.2 = name_ref b, file.%b
// CHECK:STDOUT:   %.loc13_16: Core.IntLiteral = int_value 0 [template = constants.%.30]
// CHECK:STDOUT:   %.loc13_15.1: ref %i32 = tuple_access %b.ref, element0
// CHECK:STDOUT:   %.loc13_15.2: %i32 = bind_value %.loc13_15.1
// CHECK:STDOUT:   assign file.%c.var, %.loc13_15.2
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
