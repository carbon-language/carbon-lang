// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/ir/duplicate_name_same_line.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/ir/duplicate_name_same_line.carbon

fn A() { if (true) { var n: i32 = 1; } if (true) { var n: i32 = 2; } }

// CHECK:STDOUT: --- duplicate_name_same_line.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %A.type: type = fn_type @A [template]
// CHECK:STDOUT:   %A: %A.type = struct_value () [template]
// CHECK:STDOUT:   %.1: bool = bool_literal true [template]
// CHECK:STDOUT:   %.2: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %.2 [template]
// CHECK:STDOUT:   %.3: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %Convert.type.14: type = fn_type @Convert.2, @impl.1(%.2) [template]
// CHECK:STDOUT:   %Convert.14: %Convert.type.14 = struct_value () [template]
// CHECK:STDOUT:   %.27: <witness> = interface_witness (%Convert.14) [template]
// CHECK:STDOUT:   %.28: <bound method> = bound_method %.3, %Convert.14 [template]
// CHECK:STDOUT:   %.29: <specific function> = specific_function %.28, @Convert.2(%.2) [template]
// CHECK:STDOUT:   %.30: %i32 = int_value 1 [template]
// CHECK:STDOUT:   %.31: Core.IntLiteral = int_value 2 [template]
// CHECK:STDOUT:   %.32: <bound method> = bound_method %.31, %Convert.14 [template]
// CHECK:STDOUT:   %.33: <specific function> = specific_function %.32, @Convert.2(%.2) [template]
// CHECK:STDOUT:   %.34: %i32 = int_value 2 [template]
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
// CHECK:STDOUT:     .A = %A.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %A.decl: %A.type = fn_decl @A [template = constants.%A] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @A() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc11_14: bool = bool_literal true [template = constants.%.1]
// CHECK:STDOUT:   if %.loc11_14 br !if.then.loc11_18 else br !if.else.loc11_18
// CHECK:STDOUT:
// CHECK:STDOUT: !if.then.loc11_18:
// CHECK:STDOUT:   %.loc11_29.1: Core.IntLiteral = int_value 32 [template = constants.%.2]
// CHECK:STDOUT:   %int.make_type_signed.loc11_29: init type = call constants.%Int(%.loc11_29.1) [template = constants.%i32]
// CHECK:STDOUT:   %.loc11_29.2: type = value_of_initializer %int.make_type_signed.loc11_29 [template = constants.%i32]
// CHECK:STDOUT:   %.loc11_29.3: type = converted %int.make_type_signed.loc11_29, %.loc11_29.2 [template = constants.%i32]
// CHECK:STDOUT:   %n.var.loc11_26: ref %i32 = var n
// CHECK:STDOUT:   %n.loc11_26: ref %i32 = bind_name n, %n.var.loc11_26
// CHECK:STDOUT:   %.loc11_35: Core.IntLiteral = int_value 1 [template = constants.%.3]
// CHECK:STDOUT:   %.loc11_36.1: %Convert.type.2 = interface_witness_access constants.%.27, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc11_36.2: <bound method> = bound_method %.loc11_35, %.loc11_36.1 [template = constants.%.28]
// CHECK:STDOUT:   %.loc11_36.3: <specific function> = specific_function %.loc11_36.2, @Convert.2(constants.%.2) [template = constants.%.29]
// CHECK:STDOUT:   %int.convert_checked.loc11_36: init %i32 = call %.loc11_36.3(%.loc11_35) [template = constants.%.30]
// CHECK:STDOUT:   %.loc11_36.4: init %i32 = converted %.loc11_35, %int.convert_checked.loc11_36 [template = constants.%.30]
// CHECK:STDOUT:   assign %n.var.loc11_26, %.loc11_36.4
// CHECK:STDOUT:   br !if.else.loc11_18
// CHECK:STDOUT:
// CHECK:STDOUT: !if.else.loc11_18:
// CHECK:STDOUT:   %.loc11_44: bool = bool_literal true [template = constants.%.1]
// CHECK:STDOUT:   if %.loc11_44 br !if.then.loc11_48 else br !if.else.loc11_48
// CHECK:STDOUT:
// CHECK:STDOUT: !if.then.loc11_48:
// CHECK:STDOUT:   %.loc11_59.1: Core.IntLiteral = int_value 32 [template = constants.%.2]
// CHECK:STDOUT:   %int.make_type_signed.loc11_59: init type = call constants.%Int(%.loc11_59.1) [template = constants.%i32]
// CHECK:STDOUT:   %.loc11_59.2: type = value_of_initializer %int.make_type_signed.loc11_59 [template = constants.%i32]
// CHECK:STDOUT:   %.loc11_59.3: type = converted %int.make_type_signed.loc11_59, %.loc11_59.2 [template = constants.%i32]
// CHECK:STDOUT:   %n.var.loc11_56: ref %i32 = var n
// CHECK:STDOUT:   %n.loc11_56: ref %i32 = bind_name n, %n.var.loc11_56
// CHECK:STDOUT:   %.loc11_65: Core.IntLiteral = int_value 2 [template = constants.%.31]
// CHECK:STDOUT:   %.loc11_66.1: %Convert.type.2 = interface_witness_access constants.%.27, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc11_66.2: <bound method> = bound_method %.loc11_65, %.loc11_66.1 [template = constants.%.32]
// CHECK:STDOUT:   %.loc11_66.3: <specific function> = specific_function %.loc11_66.2, @Convert.2(constants.%.2) [template = constants.%.33]
// CHECK:STDOUT:   %int.convert_checked.loc11_66: init %i32 = call %.loc11_66.3(%.loc11_65) [template = constants.%.34]
// CHECK:STDOUT:   %.loc11_66.4: init %i32 = converted %.loc11_65, %int.convert_checked.loc11_66 [template = constants.%.34]
// CHECK:STDOUT:   assign %n.var.loc11_56, %.loc11_66.4
// CHECK:STDOUT:   br !if.else.loc11_48
// CHECK:STDOUT:
// CHECK:STDOUT: !if.else.loc11_48:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
