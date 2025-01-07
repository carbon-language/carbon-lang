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
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [template]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT:   %Convert.type.cd1: type = fn_type @Convert.1, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %impl_witness.5b0: <witness> = impl_witness (imports.%import_ref.723), @impl.1(%int_32) [template]
// CHECK:STDOUT:   %Convert.type.466: type = fn_type @Convert.2, @impl.1(%int_32) [template]
// CHECK:STDOUT:   %Convert.925: %Convert.type.466 = struct_value () [template]
// CHECK:STDOUT:   %Convert.bound.afd: <bound method> = bound_method %int_1.5b8, %Convert.925 [template]
// CHECK:STDOUT:   %Convert.specific_fn.b73: <specific function> = specific_function %Convert.bound.afd, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_1.c60: %i32 = int_value 1 [template]
// CHECK:STDOUT:   %int_2.ecc: Core.IntLiteral = int_value 2 [template]
// CHECK:STDOUT:   %Convert.bound.f0b: <bound method> = bound_method %int_2.ecc, %Convert.925 [template]
// CHECK:STDOUT:   %Convert.specific_fn.20e: <specific function> = specific_function %Convert.bound.f0b, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_2.166: %i32 = int_value 2 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int = %import_ref.187
// CHECK:STDOUT:     .ImplicitAs = %import_ref.a69
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
// CHECK:STDOUT:   %n.var.loc11_26: ref %i32 = var n
// CHECK:STDOUT:   %n.loc11_26: ref %i32 = bind_name n, %n.var.loc11_26
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [template = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem0.loc11_36: %Convert.type.cd1 = impl_witness_access constants.%impl_witness.5b0, element0 [template = constants.%Convert.925]
// CHECK:STDOUT:   %Convert.bound.loc11_36: <bound method> = bound_method %int_1, %impl.elem0.loc11_36 [template = constants.%Convert.bound.afd]
// CHECK:STDOUT:   %Convert.specific_fn.loc11_36: <specific function> = specific_function %Convert.bound.loc11_36, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.b73]
// CHECK:STDOUT:   %int.convert_checked.loc11_36: init %i32 = call %Convert.specific_fn.loc11_36(%int_1) [template = constants.%int_1.c60]
// CHECK:STDOUT:   %.loc11_36: init %i32 = converted %int_1, %int.convert_checked.loc11_36 [template = constants.%int_1.c60]
// CHECK:STDOUT:   assign %n.var.loc11_26, %.loc11_36
// CHECK:STDOUT:   br !if.else.loc11_18
// CHECK:STDOUT:
// CHECK:STDOUT: !if.else.loc11_18:
// CHECK:STDOUT:   %true.loc11_44: bool = bool_literal true [template = constants.%true]
// CHECK:STDOUT:   if %true.loc11_44 br !if.then.loc11_48 else br !if.else.loc11_48
// CHECK:STDOUT:
// CHECK:STDOUT: !if.then.loc11_48:
// CHECK:STDOUT:   %n.var.loc11_56: ref %i32 = var n
// CHECK:STDOUT:   %n.loc11_56: ref %i32 = bind_name n, %n.var.loc11_56
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [template = constants.%int_2.ecc]
// CHECK:STDOUT:   %impl.elem0.loc11_66: %Convert.type.cd1 = impl_witness_access constants.%impl_witness.5b0, element0 [template = constants.%Convert.925]
// CHECK:STDOUT:   %Convert.bound.loc11_66: <bound method> = bound_method %int_2, %impl.elem0.loc11_66 [template = constants.%Convert.bound.f0b]
// CHECK:STDOUT:   %Convert.specific_fn.loc11_66: <specific function> = specific_function %Convert.bound.loc11_66, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.20e]
// CHECK:STDOUT:   %int.convert_checked.loc11_66: init %i32 = call %Convert.specific_fn.loc11_66(%int_2) [template = constants.%int_2.166]
// CHECK:STDOUT:   %.loc11_66: init %i32 = converted %int_2, %int.convert_checked.loc11_66 [template = constants.%int_2.166]
// CHECK:STDOUT:   assign %n.var.loc11_56, %.loc11_66
// CHECK:STDOUT:   br !if.else.loc11_48
// CHECK:STDOUT:
// CHECK:STDOUT: !if.else.loc11_48:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
