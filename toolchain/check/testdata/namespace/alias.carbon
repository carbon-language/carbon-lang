// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/namespace/alias.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/namespace/alias.carbon

namespace NS;

alias ns = NS;

fn NS.A() -> i32 { return 0; }

fn B() -> i32 { return ns.A(); }

alias C = NS.A;

fn D() -> i32 { return C(); }

// CHECK:STDOUT: --- alias.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %.2: type = int_type signed, %.1 [template]
// CHECK:STDOUT:   %A.type: type = fn_type @A [template]
// CHECK:STDOUT:   %A: %A.type = struct_value () [template]
// CHECK:STDOUT:   %.3: Core.IntLiteral = int_value 0 [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(%.2) [template]
// CHECK:STDOUT:   %Convert.type.14: type = fn_type @Convert.2, @impl.1(%.1) [template]
// CHECK:STDOUT:   %Convert.14: %Convert.type.14 = struct_value () [template]
// CHECK:STDOUT:   %.25: <witness> = interface_witness (%Convert.14) [template]
// CHECK:STDOUT:   %.26: <bound method> = bound_method %.3, %Convert.14 [template]
// CHECK:STDOUT:   %.27: <specific function> = specific_function %.26, @Convert.2(%.1) [template]
// CHECK:STDOUT:   %.28: %.2 = int_value 0 [template]
// CHECK:STDOUT:   %B.type: type = fn_type @B [template]
// CHECK:STDOUT:   %B: %B.type = struct_value () [template]
// CHECK:STDOUT:   %D.type: type = fn_type @D [template]
// CHECK:STDOUT:   %D: %D.type = struct_value () [template]
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
// CHECK:STDOUT:     .NS = %NS
// CHECK:STDOUT:     .ns = %ns
// CHECK:STDOUT:     .B = %B.decl
// CHECK:STDOUT:     .C = %C
// CHECK:STDOUT:     .D = %D.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %NS: <namespace> = namespace [template] {
// CHECK:STDOUT:     .A = %A.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %NS.ref.loc13: <namespace> = name_ref NS, %NS [template = %NS]
// CHECK:STDOUT:   %ns: <namespace> = bind_alias ns, %NS [template = %NS]
// CHECK:STDOUT:   %A.decl: %A.type = fn_decl @A [template = constants.%A] {
// CHECK:STDOUT:     %return.patt: %.2 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %.2 = out_param_pattern %return.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc15_14.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:     %int.make_type_signed: init type = call constants.%Int(%.loc15_14.1) [template = constants.%.2]
// CHECK:STDOUT:     %.loc15_14.2: type = value_of_initializer %int.make_type_signed [template = constants.%.2]
// CHECK:STDOUT:     %.loc15_14.3: type = converted %int.make_type_signed, %.loc15_14.2 [template = constants.%.2]
// CHECK:STDOUT:     %return.param: ref %.2 = out_param runtime_param0
// CHECK:STDOUT:     %return: ref %.2 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %B.decl: %B.type = fn_decl @B [template = constants.%B] {
// CHECK:STDOUT:     %return.patt: %.2 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %.2 = out_param_pattern %return.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc17_11.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:     %int.make_type_signed: init type = call constants.%Int(%.loc17_11.1) [template = constants.%.2]
// CHECK:STDOUT:     %.loc17_11.2: type = value_of_initializer %int.make_type_signed [template = constants.%.2]
// CHECK:STDOUT:     %.loc17_11.3: type = converted %int.make_type_signed, %.loc17_11.2 [template = constants.%.2]
// CHECK:STDOUT:     %return.param: ref %.2 = out_param runtime_param0
// CHECK:STDOUT:     %return: ref %.2 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %NS.ref.loc19: <namespace> = name_ref NS, %NS [template = %NS]
// CHECK:STDOUT:   %A.ref: %A.type = name_ref A, %A.decl [template = constants.%A]
// CHECK:STDOUT:   %C: %A.type = bind_alias C, %A.decl [template = constants.%A]
// CHECK:STDOUT:   %D.decl: %D.type = fn_decl @D [template = constants.%D] {
// CHECK:STDOUT:     %return.patt: %.2 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %.2 = out_param_pattern %return.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc21_11.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:     %int.make_type_signed: init type = call constants.%Int(%.loc21_11.1) [template = constants.%.2]
// CHECK:STDOUT:     %.loc21_11.2: type = value_of_initializer %int.make_type_signed [template = constants.%.2]
// CHECK:STDOUT:     %.loc21_11.3: type = converted %int.make_type_signed, %.loc21_11.2 [template = constants.%.2]
// CHECK:STDOUT:     %return.param: ref %.2 = out_param runtime_param0
// CHECK:STDOUT:     %return: ref %.2 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @A() -> %.2 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc15_27: Core.IntLiteral = int_value 0 [template = constants.%.3]
// CHECK:STDOUT:   %.loc15_28.1: %Convert.type.2 = interface_witness_access constants.%.25, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc15_28.2: <bound method> = bound_method %.loc15_27, %.loc15_28.1 [template = constants.%.26]
// CHECK:STDOUT:   %.loc15_28.3: <specific function> = specific_function %.loc15_28.2, @Convert.2(constants.%.1) [template = constants.%.27]
// CHECK:STDOUT:   %int.convert_checked: init %.2 = call %.loc15_28.3(%.loc15_27) [template = constants.%.28]
// CHECK:STDOUT:   %.loc15_28.4: %.2 = value_of_initializer %int.convert_checked [template = constants.%.28]
// CHECK:STDOUT:   %.loc15_28.5: %.2 = converted %.loc15_27, %.loc15_28.4 [template = constants.%.28]
// CHECK:STDOUT:   return %.loc15_28.5
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @B() -> %.2 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %ns.ref: <namespace> = name_ref ns, file.%ns [template = file.%NS]
// CHECK:STDOUT:   %A.ref: %A.type = name_ref A, file.%A.decl [template = constants.%A]
// CHECK:STDOUT:   %A.call: init %.2 = call %A.ref()
// CHECK:STDOUT:   %.loc17_30.1: %.2 = value_of_initializer %A.call
// CHECK:STDOUT:   %.loc17_30.2: %.2 = converted %A.call, %.loc17_30.1
// CHECK:STDOUT:   return %.loc17_30.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @D() -> %.2 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %C.ref: %A.type = name_ref C, file.%C [template = constants.%A]
// CHECK:STDOUT:   %A.call: init %.2 = call %C.ref()
// CHECK:STDOUT:   %.loc21_27.1: %.2 = value_of_initializer %A.call
// CHECK:STDOUT:   %.loc21_27.2: %.2 = converted %A.call, %.loc21_27.1
// CHECK:STDOUT:   return %.loc21_27.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
