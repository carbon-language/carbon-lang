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
// CHECK:STDOUT:   %true: bool = bool_literal true [template]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic]
// CHECK:STDOUT:   %Int: type = class_type @Int, @Int(%N) [symbolic]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [template]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %Convert.type.3: type = fn_type @Convert.2, @impl.1(%N) [symbolic]
// CHECK:STDOUT:   %Convert.3: %Convert.type.3 = struct_value () [symbolic]
// CHECK:STDOUT:   %impl_witness.5: <witness> = impl_witness (%Convert.3), @impl.1(%int_32) [template]
// CHECK:STDOUT:   %Convert.bound.1: <bound method> = bound_method %int_1, %Convert.3 [symbolic]
// CHECK:STDOUT:   %Convert.specific_fn.1: <specific function> = specific_function %Convert.bound.1, @Convert.2(%N) [symbolic]
// CHECK:STDOUT:   %int.convert_checked.1: init %Int = call %Convert.specific_fn.1(%int_1) [symbolic]
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [template]
// CHECK:STDOUT:   %Convert.bound.2: <bound method> = bound_method %int_2, %Convert.3 [symbolic]
// CHECK:STDOUT:   %Convert.specific_fn.2: <specific function> = specific_function %Convert.bound.2, @Convert.2(%N) [symbolic]
// CHECK:STDOUT:   %int.convert_checked.2: init %Int = call %Convert.specific_fn.2(%int_2) [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int = %import_ref.1
// CHECK:STDOUT:     .ImplicitAs = %import_ref.5
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
// CHECK:STDOUT:   %true.loc11_14: bool = bool_literal true [template = constants.%true]
// CHECK:STDOUT:   if %true.loc11_14 br !if.then.loc11_18 else br !if.else.loc11_18
// CHECK:STDOUT:
// CHECK:STDOUT: !if.then.loc11_18:
// CHECK:STDOUT:   %int_32.loc11_29: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %i32.loc11_29: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:   %n.var.loc11_26: ref %i32 = var n
// CHECK:STDOUT:   %n.loc11_26: ref %i32 = bind_name n, %n.var.loc11_26
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [template = constants.%int_1]
// CHECK:STDOUT:   %impl.elem0.loc11_36: %Convert.type.2 = impl_witness_access constants.%impl_witness.5, element0 [symbolic = constants.%Convert.3]
// CHECK:STDOUT:   %Convert.bound.loc11_36: <bound method> = bound_method %int_1, %impl.elem0.loc11_36 [symbolic = constants.%Convert.bound.1]
// CHECK:STDOUT:   %Convert.specific_fn.loc11_36: <specific function> = specific_function %Convert.bound.loc11_36, @Convert.2(constants.%N) [symbolic = constants.%Convert.specific_fn.1]
// CHECK:STDOUT:   %int.convert_checked.loc11_36: init %Int = call %Convert.specific_fn.loc11_36(%int_1) [symbolic = constants.%int.convert_checked.1]
// CHECK:STDOUT:   %.loc11_36: init %i32 = converted %int_1, %int.convert_checked.loc11_36 [symbolic = constants.%int.convert_checked.1]
// CHECK:STDOUT:   assign %n.var.loc11_26, %.loc11_36
// CHECK:STDOUT:   br !if.else.loc11_18
// CHECK:STDOUT:
// CHECK:STDOUT: !if.else.loc11_18:
// CHECK:STDOUT:   %true.loc11_44: bool = bool_literal true [template = constants.%true]
// CHECK:STDOUT:   if %true.loc11_44 br !if.then.loc11_48 else br !if.else.loc11_48
// CHECK:STDOUT:
// CHECK:STDOUT: !if.then.loc11_48:
// CHECK:STDOUT:   %int_32.loc11_59: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %i32.loc11_59: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:   %n.var.loc11_56: ref %i32 = var n
// CHECK:STDOUT:   %n.loc11_56: ref %i32 = bind_name n, %n.var.loc11_56
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [template = constants.%int_2]
// CHECK:STDOUT:   %impl.elem0.loc11_66: %Convert.type.2 = impl_witness_access constants.%impl_witness.5, element0 [symbolic = constants.%Convert.3]
// CHECK:STDOUT:   %Convert.bound.loc11_66: <bound method> = bound_method %int_2, %impl.elem0.loc11_66 [symbolic = constants.%Convert.bound.2]
// CHECK:STDOUT:   %Convert.specific_fn.loc11_66: <specific function> = specific_function %Convert.bound.loc11_66, @Convert.2(constants.%N) [symbolic = constants.%Convert.specific_fn.2]
// CHECK:STDOUT:   %int.convert_checked.loc11_66: init %Int = call %Convert.specific_fn.loc11_66(%int_2) [symbolic = constants.%int.convert_checked.2]
// CHECK:STDOUT:   %.loc11_66: init %i32 = converted %int_2, %int.convert_checked.loc11_66 [symbolic = constants.%int.convert_checked.2]
// CHECK:STDOUT:   assign %n.var.loc11_56, %.loc11_66
// CHECK:STDOUT:   br !if.else.loc11_48
// CHECK:STDOUT:
// CHECK:STDOUT: !if.else.loc11_48:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
