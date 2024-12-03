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
// CHECK:STDOUT:   %int.32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %int.32 [template]
// CHECK:STDOUT:   %int.1.1: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %Convert.type.14: type = fn_type @Convert.2, @impl.1(%int.32) [template]
// CHECK:STDOUT:   %Convert.14: %Convert.type.14 = struct_value () [template]
// CHECK:STDOUT:   %interface.9: <witness> = interface_witness (%Convert.14) [template]
// CHECK:STDOUT:   %.17: <bound method> = bound_method %int.1.1, %Convert.14 [template]
// CHECK:STDOUT:   %.18: <specific function> = specific_function %.17, @Convert.2(%int.32) [template]
// CHECK:STDOUT:   %int.1.2: %i32 = int_value 1 [template]
// CHECK:STDOUT:   %int.2.1: Core.IntLiteral = int_value 2 [template]
// CHECK:STDOUT:   %.19: <bound method> = bound_method %int.2.1, %Convert.14 [template]
// CHECK:STDOUT:   %.20: <specific function> = specific_function %.19, @Convert.2(%int.32) [template]
// CHECK:STDOUT:   %int.2.2: %i32 = int_value 2 [template]
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
// CHECK:STDOUT:   %int.32.loc11_29: Core.IntLiteral = int_value 32 [template = constants.%int.32]
// CHECK:STDOUT:   %int.make_type_signed.loc11_29: init type = call constants.%Int(%int.32.loc11_29) [template = constants.%i32]
// CHECK:STDOUT:   %.loc11_29.1: type = value_of_initializer %int.make_type_signed.loc11_29 [template = constants.%i32]
// CHECK:STDOUT:   %.loc11_29.2: type = converted %int.make_type_signed.loc11_29, %.loc11_29.1 [template = constants.%i32]
// CHECK:STDOUT:   %n.var.loc11_26: ref %i32 = var n
// CHECK:STDOUT:   %n.loc11_26: ref %i32 = bind_name n, %n.var.loc11_26
// CHECK:STDOUT:   %int.1: Core.IntLiteral = int_value 1 [template = constants.%int.1.1]
// CHECK:STDOUT:   %.loc11_36.1: %Convert.type.2 = interface_witness_access constants.%interface.9, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc11_36.2: <bound method> = bound_method %int.1, %.loc11_36.1 [template = constants.%.17]
// CHECK:STDOUT:   %.loc11_36.3: <specific function> = specific_function %.loc11_36.2, @Convert.2(constants.%int.32) [template = constants.%.18]
// CHECK:STDOUT:   %int.convert_checked.loc11_36: init %i32 = call %.loc11_36.3(%int.1) [template = constants.%int.1.2]
// CHECK:STDOUT:   %.loc11_36.4: init %i32 = converted %int.1, %int.convert_checked.loc11_36 [template = constants.%int.1.2]
// CHECK:STDOUT:   assign %n.var.loc11_26, %.loc11_36.4
// CHECK:STDOUT:   br !if.else.loc11_18
// CHECK:STDOUT:
// CHECK:STDOUT: !if.else.loc11_18:
// CHECK:STDOUT:   %.loc11_44: bool = bool_literal true [template = constants.%.1]
// CHECK:STDOUT:   if %.loc11_44 br !if.then.loc11_48 else br !if.else.loc11_48
// CHECK:STDOUT:
// CHECK:STDOUT: !if.then.loc11_48:
// CHECK:STDOUT:   %int.32.loc11_59: Core.IntLiteral = int_value 32 [template = constants.%int.32]
// CHECK:STDOUT:   %int.make_type_signed.loc11_59: init type = call constants.%Int(%int.32.loc11_59) [template = constants.%i32]
// CHECK:STDOUT:   %.loc11_59.1: type = value_of_initializer %int.make_type_signed.loc11_59 [template = constants.%i32]
// CHECK:STDOUT:   %.loc11_59.2: type = converted %int.make_type_signed.loc11_59, %.loc11_59.1 [template = constants.%i32]
// CHECK:STDOUT:   %n.var.loc11_56: ref %i32 = var n
// CHECK:STDOUT:   %n.loc11_56: ref %i32 = bind_name n, %n.var.loc11_56
// CHECK:STDOUT:   %int.2: Core.IntLiteral = int_value 2 [template = constants.%int.2.1]
// CHECK:STDOUT:   %.loc11_66.1: %Convert.type.2 = interface_witness_access constants.%interface.9, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc11_66.2: <bound method> = bound_method %int.2, %.loc11_66.1 [template = constants.%.19]
// CHECK:STDOUT:   %.loc11_66.3: <specific function> = specific_function %.loc11_66.2, @Convert.2(constants.%int.32) [template = constants.%.20]
// CHECK:STDOUT:   %int.convert_checked.loc11_66: init %i32 = call %.loc11_66.3(%int.2) [template = constants.%int.2.2]
// CHECK:STDOUT:   %.loc11_66.4: init %i32 = converted %int.2, %int.convert_checked.loc11_66 [template = constants.%int.2.2]
// CHECK:STDOUT:   assign %n.var.loc11_56, %.loc11_66.4
// CHECK:STDOUT:   br !if.else.loc11_48
// CHECK:STDOUT:
// CHECK:STDOUT: !if.else.loc11_48:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
