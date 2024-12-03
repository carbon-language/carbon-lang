// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/tuple/access/index_not_literal.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/tuple/access/index_not_literal.carbon

var a: (i32, i32) = (12, 34);
var b: i32 = a.({.index = 1}.index);

// CHECK:STDOUT: --- index_not_literal.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int.32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %int.32 [template]
// CHECK:STDOUT:   %tuple.type.1: type = tuple_type (type, type) [template]
// CHECK:STDOUT:   %tuple.type.2: type = tuple_type (%i32, %i32) [template]
// CHECK:STDOUT:   %int.12.1: Core.IntLiteral = int_value 12 [template]
// CHECK:STDOUT:   %int.34.1: Core.IntLiteral = int_value 34 [template]
// CHECK:STDOUT:   %tuple.type.3: type = tuple_type (Core.IntLiteral, Core.IntLiteral) [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %Convert.type.14: type = fn_type @Convert.2, @impl.1(%int.32) [template]
// CHECK:STDOUT:   %Convert.14: %Convert.type.14 = struct_value () [template]
// CHECK:STDOUT:   %.25: <witness> = interface_witness (%Convert.14) [template]
// CHECK:STDOUT:   %.26: <bound method> = bound_method %int.12.1, %Convert.14 [template]
// CHECK:STDOUT:   %.27: <specific function> = specific_function %.26, @Convert.2(%int.32) [template]
// CHECK:STDOUT:   %int.12.2: %i32 = int_value 12 [template]
// CHECK:STDOUT:   %.28: <bound method> = bound_method %int.34.1, %Convert.14 [template]
// CHECK:STDOUT:   %.29: <specific function> = specific_function %.28, @Convert.2(%int.32) [template]
// CHECK:STDOUT:   %int.34.2: %i32 = int_value 34 [template]
// CHECK:STDOUT:   %tuple: %tuple.type.2 = tuple_value (%int.12.2, %int.34.2) [template]
// CHECK:STDOUT:   %int.1: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT:   %struct_type.index: type = struct_type {.index: Core.IntLiteral} [template]
// CHECK:STDOUT:   %struct: %struct_type.index = struct_value (%int.1) [template]
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
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %int.32.loc11_9: Core.IntLiteral = int_value 32 [template = constants.%int.32]
// CHECK:STDOUT:   %int.make_type_signed.loc11_9: init type = call constants.%Int(%int.32.loc11_9) [template = constants.%i32]
// CHECK:STDOUT:   %int.32.loc11_14: Core.IntLiteral = int_value 32 [template = constants.%int.32]
// CHECK:STDOUT:   %int.make_type_signed.loc11_14: init type = call constants.%Int(%int.32.loc11_14) [template = constants.%i32]
// CHECK:STDOUT:   %.loc11_17.1: %tuple.type.1 = tuple_literal (%int.make_type_signed.loc11_9, %int.make_type_signed.loc11_14)
// CHECK:STDOUT:   %.loc11_17.2: type = value_of_initializer %int.make_type_signed.loc11_9 [template = constants.%i32]
// CHECK:STDOUT:   %.loc11_17.3: type = converted %int.make_type_signed.loc11_9, %.loc11_17.2 [template = constants.%i32]
// CHECK:STDOUT:   %.loc11_17.4: type = value_of_initializer %int.make_type_signed.loc11_14 [template = constants.%i32]
// CHECK:STDOUT:   %.loc11_17.5: type = converted %int.make_type_signed.loc11_14, %.loc11_17.4 [template = constants.%i32]
// CHECK:STDOUT:   %.loc11_17.6: type = converted %.loc11_17.1, constants.%tuple.type.2 [template = constants.%tuple.type.2]
// CHECK:STDOUT:   %a.var: ref %tuple.type.2 = var a
// CHECK:STDOUT:   %a: ref %tuple.type.2 = bind_name a, %a.var
// CHECK:STDOUT:   %int.32.loc12: Core.IntLiteral = int_value 32 [template = constants.%int.32]
// CHECK:STDOUT:   %int.make_type_signed.loc12: init type = call constants.%Int(%int.32.loc12) [template = constants.%i32]
// CHECK:STDOUT:   %.loc12_8.1: type = value_of_initializer %int.make_type_signed.loc12 [template = constants.%i32]
// CHECK:STDOUT:   %.loc12_8.2: type = converted %int.make_type_signed.loc12, %.loc12_8.1 [template = constants.%i32]
// CHECK:STDOUT:   %b.var: ref %i32 = var b
// CHECK:STDOUT:   %b: ref %i32 = bind_name b, %b.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int.12: Core.IntLiteral = int_value 12 [template = constants.%int.12.1]
// CHECK:STDOUT:   %int.34: Core.IntLiteral = int_value 34 [template = constants.%int.34.1]
// CHECK:STDOUT:   %.loc11_28.1: %tuple.type.3 = tuple_literal (%int.12, %int.34)
// CHECK:STDOUT:   %.loc11_28.2: %Convert.type.2 = interface_witness_access constants.%.25, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc11_28.3: <bound method> = bound_method %int.12, %.loc11_28.2 [template = constants.%.26]
// CHECK:STDOUT:   %.loc11_28.4: <specific function> = specific_function %.loc11_28.3, @Convert.2(constants.%int.32) [template = constants.%.27]
// CHECK:STDOUT:   %int.convert_checked.loc11_28.1: init %i32 = call %.loc11_28.4(%int.12) [template = constants.%int.12.2]
// CHECK:STDOUT:   %.loc11_28.5: init %i32 = converted %int.12, %int.convert_checked.loc11_28.1 [template = constants.%int.12.2]
// CHECK:STDOUT:   %.loc11_28.6: ref %i32 = tuple_access file.%a.var, element0
// CHECK:STDOUT:   %.loc11_28.7: init %i32 = initialize_from %.loc11_28.5 to %.loc11_28.6 [template = constants.%int.12.2]
// CHECK:STDOUT:   %.loc11_28.8: %Convert.type.2 = interface_witness_access constants.%.25, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc11_28.9: <bound method> = bound_method %int.34, %.loc11_28.8 [template = constants.%.28]
// CHECK:STDOUT:   %.loc11_28.10: <specific function> = specific_function %.loc11_28.9, @Convert.2(constants.%int.32) [template = constants.%.29]
// CHECK:STDOUT:   %int.convert_checked.loc11_28.2: init %i32 = call %.loc11_28.10(%int.34) [template = constants.%int.34.2]
// CHECK:STDOUT:   %.loc11_28.11: init %i32 = converted %int.34, %int.convert_checked.loc11_28.2 [template = constants.%int.34.2]
// CHECK:STDOUT:   %.loc11_28.12: ref %i32 = tuple_access file.%a.var, element1
// CHECK:STDOUT:   %.loc11_28.13: init %i32 = initialize_from %.loc11_28.11 to %.loc11_28.12 [template = constants.%int.34.2]
// CHECK:STDOUT:   %.loc11_28.14: init %tuple.type.2 = tuple_init (%.loc11_28.7, %.loc11_28.13) to file.%a.var [template = constants.%tuple]
// CHECK:STDOUT:   %.loc11_29: init %tuple.type.2 = converted %.loc11_28.1, %.loc11_28.14 [template = constants.%tuple]
// CHECK:STDOUT:   assign file.%a.var, %.loc11_29
// CHECK:STDOUT:   %a.ref: ref %tuple.type.2 = name_ref a, file.%a
// CHECK:STDOUT:   %int.1: Core.IntLiteral = int_value 1 [template = constants.%int.1]
// CHECK:STDOUT:   %.loc12_28.1: %struct_type.index = struct_literal (%int.1)
// CHECK:STDOUT:   %struct: %struct_type.index = struct_value (%int.1) [template = constants.%struct]
// CHECK:STDOUT:   %.loc12_28.2: %struct_type.index = converted %.loc12_28.1, %struct [template = constants.%struct]
// CHECK:STDOUT:   %.loc12_29: Core.IntLiteral = struct_access %.loc12_28.2, element0 [template = constants.%int.1]
// CHECK:STDOUT:   %.loc12_15.1: ref %i32 = tuple_access %a.ref, element1
// CHECK:STDOUT:   %.loc12_15.2: %i32 = bind_value %.loc12_15.1
// CHECK:STDOUT:   assign file.%b.var, %.loc12_15.2
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
