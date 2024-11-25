// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/syntactic_merge_literal.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/syntactic_merge_literal.carbon
// CHECK:STDERR: fail_int_mismatch.carbon: error: `Main//default` previously provided by `int_match.carbon` [DuplicateMainApi]
// CHECK:STDERR:

// --- int_match.carbon

class C(a:! i32) {}
class D(b:! C(1_000));
class D(b:! C(1_000)) {}

// --- fail_int_mismatch.carbon

class C(a:! i32) {}
class D(b:! C(1000));
// CHECK:STDERR: fail_int_mismatch.carbon:[[@LINE+6]]:15: error: redeclaration syntax differs here [RedeclParamSyntaxDiffers]
// CHECK:STDERR: class D(b:! C(1_000)) {}
// CHECK:STDERR:               ^~~~~
// CHECK:STDERR: fail_int_mismatch.carbon:[[@LINE-4]]:15: note: comparing with previous declaration here [RedeclParamSyntaxPrevious]
// CHECK:STDERR: class D(b:! C(1000));
// CHECK:STDERR:               ^~~~
class D(b:! C(1_000)) {}

// CHECK:STDOUT: --- int_match.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %.1 [template]
// CHECK:STDOUT:   %a: %i32 = bind_symbolic_name a, 0 [symbolic]
// CHECK:STDOUT:   %a.patt: %i32 = symbolic_binding_pattern a, 0 [symbolic]
// CHECK:STDOUT:   %C.type: type = generic_class_type @C [template]
// CHECK:STDOUT:   %C.1: %C.type = struct_value () [template]
// CHECK:STDOUT:   %C.2: type = class_type @C, @C(%a) [symbolic]
// CHECK:STDOUT:   %.2: type = struct_type {} [template]
// CHECK:STDOUT:   %.3: <witness> = complete_type_witness %.2 [template]
// CHECK:STDOUT:   %.4: Core.IntLiteral = int_value 1000 [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %Convert.type.14: type = fn_type @Convert.2, @impl.1(%.1) [template]
// CHECK:STDOUT:   %Convert.14: %Convert.type.14 = struct_value () [template]
// CHECK:STDOUT:   %.28: <witness> = interface_witness (%Convert.14) [template]
// CHECK:STDOUT:   %.29: <bound method> = bound_method %.4, %Convert.14 [template]
// CHECK:STDOUT:   %.30: <specific function> = specific_function %.29, @Convert.2(%.1) [template]
// CHECK:STDOUT:   %.31: %i32 = int_value 1000 [template]
// CHECK:STDOUT:   %C.3: type = class_type @C, @C(%.31) [template]
// CHECK:STDOUT:   %b: %C.3 = bind_symbolic_name b, 0 [symbolic]
// CHECK:STDOUT:   %b.patt: %C.3 = symbolic_binding_pattern b, 0 [symbolic]
// CHECK:STDOUT:   %D.type: type = generic_class_type @D [template]
// CHECK:STDOUT:   %D.1: %D.type = struct_value () [template]
// CHECK:STDOUT:   %D.2: type = class_type @D, @D(%b) [symbolic]
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
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .D = %D.decl.loc3
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %C.decl: %C.type = class_decl @C [template = constants.%C.1] {
// CHECK:STDOUT:     %a.patt.loc2_9.1: %i32 = symbolic_binding_pattern a, 0 [symbolic = %a.patt.loc2_9.2 (constants.%a.patt)]
// CHECK:STDOUT:     %a.param_patt: %i32 = value_param_pattern %a.patt.loc2_9.1, runtime_param<invalid> [symbolic = %a.patt.loc2_9.2 (constants.%a.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc2_13.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:     %int.make_type_signed: init type = call constants.%Int(%.loc2_13.1) [template = constants.%i32]
// CHECK:STDOUT:     %.loc2_13.2: type = value_of_initializer %int.make_type_signed [template = constants.%i32]
// CHECK:STDOUT:     %.loc2_13.3: type = converted %int.make_type_signed, %.loc2_13.2 [template = constants.%i32]
// CHECK:STDOUT:     %a.param: %i32 = value_param runtime_param<invalid>
// CHECK:STDOUT:     %a.loc2_9.1: %i32 = bind_symbolic_name a, 0, %a.param [symbolic = %a.loc2_9.2 (constants.%a)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %D.decl.loc3: %D.type = class_decl @D [template = constants.%D.1] {
// CHECK:STDOUT:     %b.patt.loc4: %C.3 = symbolic_binding_pattern b, 0 [symbolic = constants.%b.patt]
// CHECK:STDOUT:     %b.param_patt: %C.3 = value_param_pattern %b.patt.loc4, runtime_param<invalid> [symbolic = constants.%b.patt]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref.loc3: %C.type = name_ref C, file.%C.decl [template = constants.%C.1]
// CHECK:STDOUT:     %.loc3_15: Core.IntLiteral = int_value 1000 [template = constants.%.4]
// CHECK:STDOUT:     %.loc3_14.1: %Convert.type.2 = interface_witness_access constants.%.28, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:     %.loc3_14.2: <bound method> = bound_method %.loc3_15, %.loc3_14.1 [template = constants.%.29]
// CHECK:STDOUT:     %.loc3_14.3: <specific function> = specific_function %.loc3_14.2, @Convert.2(constants.%.1) [template = constants.%.30]
// CHECK:STDOUT:     %int.convert_checked.loc3: init %i32 = call %.loc3_14.3(%.loc3_15) [template = constants.%.31]
// CHECK:STDOUT:     %.loc3_14.4: %i32 = value_of_initializer %int.convert_checked.loc3 [template = constants.%.31]
// CHECK:STDOUT:     %.loc3_14.5: %i32 = converted %.loc3_15, %.loc3_14.4 [template = constants.%.31]
// CHECK:STDOUT:     %C.loc3: type = class_type @C, @C(constants.%.31) [template = constants.%C.3]
// CHECK:STDOUT:     %b.param.loc3: %C.3 = value_param runtime_param<invalid>
// CHECK:STDOUT:     %b.loc3_9.1: %C.3 = bind_symbolic_name b, 0, %b.param.loc3 [symbolic = %b.loc3_9.2 (constants.%b)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %D.decl.loc4: %D.type = class_decl @D [template = constants.%D.1] {
// CHECK:STDOUT:     %b.patt.loc4: %C.3 = symbolic_binding_pattern b, 0 [symbolic = constants.%b.patt]
// CHECK:STDOUT:     %b.param_patt: %C.3 = value_param_pattern %b.patt.loc4, runtime_param<invalid> [symbolic = constants.%b.patt]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref.loc4: %C.type = name_ref C, file.%C.decl [template = constants.%C.1]
// CHECK:STDOUT:     %.loc4_15: Core.IntLiteral = int_value 1000 [template = constants.%.4]
// CHECK:STDOUT:     %.loc4_14.1: %Convert.type.2 = interface_witness_access constants.%.28, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:     %.loc4_14.2: <bound method> = bound_method %.loc4_15, %.loc4_14.1 [template = constants.%.29]
// CHECK:STDOUT:     %.loc4_14.3: <specific function> = specific_function %.loc4_14.2, @Convert.2(constants.%.1) [template = constants.%.30]
// CHECK:STDOUT:     %int.convert_checked.loc4: init %i32 = call %.loc4_14.3(%.loc4_15) [template = constants.%.31]
// CHECK:STDOUT:     %.loc4_14.4: %i32 = value_of_initializer %int.convert_checked.loc4 [template = constants.%.31]
// CHECK:STDOUT:     %.loc4_14.5: %i32 = converted %.loc4_15, %.loc4_14.4 [template = constants.%.31]
// CHECK:STDOUT:     %C.loc4: type = class_type @C, @C(constants.%.31) [template = constants.%C.3]
// CHECK:STDOUT:     %b.param.loc4: %C.3 = value_param runtime_param<invalid>
// CHECK:STDOUT:     %b.loc4: %C.3 = bind_symbolic_name b, 0, %b.param.loc4 [symbolic = constants.%b]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @C(%a.loc2_9.1: %i32) {
// CHECK:STDOUT:   %a.loc2_9.2: %i32 = bind_symbolic_name a, 0 [symbolic = %a.loc2_9.2 (constants.%a)]
// CHECK:STDOUT:   %a.patt.loc2_9.2: %i32 = symbolic_binding_pattern a, 0 [symbolic = %a.patt.loc2_9.2 (constants.%a.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %.loc2_19: <witness> = complete_type_witness %.2 [template = constants.%.3]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%C.2
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @D(%b.loc3_9.1: %C.3) {
// CHECK:STDOUT:   %b.loc3_9.2: %C.3 = bind_symbolic_name b, 0 [symbolic = %b.loc3_9.2 (constants.%b)]
// CHECK:STDOUT:   %b.patt.loc3: %C.3 = symbolic_binding_pattern b, 0 [symbolic = %b.patt.loc3 (constants.%b.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %.loc4_24: <witness> = complete_type_witness %.2 [template = constants.%.3]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%D.2
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%a) {
// CHECK:STDOUT:   %a.loc2_9.2 => constants.%a
// CHECK:STDOUT:   %a.patt.loc2_9.2 => constants.%a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%.31) {
// CHECK:STDOUT:   %a.loc2_9.2 => constants.%.31
// CHECK:STDOUT:   %a.patt.loc2_9.2 => constants.%.31
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @D(constants.%b) {
// CHECK:STDOUT:   %b.loc3_9.2 => constants.%b
// CHECK:STDOUT:   %b.patt.loc3 => constants.%b
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_int_mismatch.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %.1 [template]
// CHECK:STDOUT:   %a: %i32 = bind_symbolic_name a, 0 [symbolic]
// CHECK:STDOUT:   %a.patt: %i32 = symbolic_binding_pattern a, 0 [symbolic]
// CHECK:STDOUT:   %C.type: type = generic_class_type @C [template]
// CHECK:STDOUT:   %C.1: %C.type = struct_value () [template]
// CHECK:STDOUT:   %C.2: type = class_type @C, @C(%a) [symbolic]
// CHECK:STDOUT:   %.2: type = struct_type {} [template]
// CHECK:STDOUT:   %.3: <witness> = complete_type_witness %.2 [template]
// CHECK:STDOUT:   %.4: Core.IntLiteral = int_value 1000 [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %Convert.type.14: type = fn_type @Convert.2, @impl.1(%.1) [template]
// CHECK:STDOUT:   %Convert.14: %Convert.type.14 = struct_value () [template]
// CHECK:STDOUT:   %.28: <witness> = interface_witness (%Convert.14) [template]
// CHECK:STDOUT:   %.29: <bound method> = bound_method %.4, %Convert.14 [template]
// CHECK:STDOUT:   %.30: <specific function> = specific_function %.29, @Convert.2(%.1) [template]
// CHECK:STDOUT:   %.31: %i32 = int_value 1000 [template]
// CHECK:STDOUT:   %C.3: type = class_type @C, @C(%.31) [template]
// CHECK:STDOUT:   %b: %C.3 = bind_symbolic_name b, 0 [symbolic]
// CHECK:STDOUT:   %b.patt: %C.3 = symbolic_binding_pattern b, 0 [symbolic]
// CHECK:STDOUT:   %D.type: type = generic_class_type @D [template]
// CHECK:STDOUT:   %D.1: %D.type = struct_value () [template]
// CHECK:STDOUT:   %.type: type = generic_class_type @.1 [template]
// CHECK:STDOUT:   %.32: %.type = struct_value () [template]
// CHECK:STDOUT:   %.33: type = class_type @.1, @.1(%b) [symbolic]
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
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .D = %D.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %C.decl: %C.type = class_decl @C [template = constants.%C.1] {
// CHECK:STDOUT:     %a.patt.loc2_9.1: %i32 = symbolic_binding_pattern a, 0 [symbolic = %a.patt.loc2_9.2 (constants.%a.patt)]
// CHECK:STDOUT:     %a.param_patt: %i32 = value_param_pattern %a.patt.loc2_9.1, runtime_param<invalid> [symbolic = %a.patt.loc2_9.2 (constants.%a.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc2_13.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:     %int.make_type_signed: init type = call constants.%Int(%.loc2_13.1) [template = constants.%i32]
// CHECK:STDOUT:     %.loc2_13.2: type = value_of_initializer %int.make_type_signed [template = constants.%i32]
// CHECK:STDOUT:     %.loc2_13.3: type = converted %int.make_type_signed, %.loc2_13.2 [template = constants.%i32]
// CHECK:STDOUT:     %a.param: %i32 = value_param runtime_param<invalid>
// CHECK:STDOUT:     %a.loc2_9.1: %i32 = bind_symbolic_name a, 0, %a.param [symbolic = %a.loc2_9.2 (constants.%a)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %D.decl: %D.type = class_decl @D [template = constants.%D.1] {
// CHECK:STDOUT:     %b.patt.loc3_9.1: %C.3 = symbolic_binding_pattern b, 0 [symbolic = %b.patt.loc3_9.2 (constants.%b.patt)]
// CHECK:STDOUT:     %b.param_patt: %C.3 = value_param_pattern %b.patt.loc3_9.1, runtime_param<invalid> [symbolic = %b.patt.loc3_9.2 (constants.%b.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref: %C.type = name_ref C, file.%C.decl [template = constants.%C.1]
// CHECK:STDOUT:     %.loc3_15: Core.IntLiteral = int_value 1000 [template = constants.%.4]
// CHECK:STDOUT:     %.loc3_14.1: %Convert.type.2 = interface_witness_access constants.%.28, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:     %.loc3_14.2: <bound method> = bound_method %.loc3_15, %.loc3_14.1 [template = constants.%.29]
// CHECK:STDOUT:     %.loc3_14.3: <specific function> = specific_function %.loc3_14.2, @Convert.2(constants.%.1) [template = constants.%.30]
// CHECK:STDOUT:     %int.convert_checked: init %i32 = call %.loc3_14.3(%.loc3_15) [template = constants.%.31]
// CHECK:STDOUT:     %.loc3_14.4: %i32 = value_of_initializer %int.convert_checked [template = constants.%.31]
// CHECK:STDOUT:     %.loc3_14.5: %i32 = converted %.loc3_15, %.loc3_14.4 [template = constants.%.31]
// CHECK:STDOUT:     %C: type = class_type @C, @C(constants.%.31) [template = constants.%C.3]
// CHECK:STDOUT:     %b.param: %C.3 = value_param runtime_param<invalid>
// CHECK:STDOUT:     %b.loc3_9.1: %C.3 = bind_symbolic_name b, 0, %b.param [symbolic = %b.loc3_9.2 (constants.%b)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.decl: %.type = class_decl @.1 [template = constants.%.32] {
// CHECK:STDOUT:     %b.patt.loc10_9.1: %C.3 = symbolic_binding_pattern b, 0 [symbolic = %b.patt.loc10_9.2 (constants.%b.patt)]
// CHECK:STDOUT:     %b.param_patt: %C.3 = value_param_pattern %b.patt.loc10_9.1, runtime_param<invalid> [symbolic = %b.patt.loc10_9.2 (constants.%b.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref: %C.type = name_ref C, file.%C.decl [template = constants.%C.1]
// CHECK:STDOUT:     %.loc10_15: Core.IntLiteral = int_value 1000 [template = constants.%.4]
// CHECK:STDOUT:     %.loc10_14.1: %Convert.type.2 = interface_witness_access constants.%.28, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:     %.loc10_14.2: <bound method> = bound_method %.loc10_15, %.loc10_14.1 [template = constants.%.29]
// CHECK:STDOUT:     %.loc10_14.3: <specific function> = specific_function %.loc10_14.2, @Convert.2(constants.%.1) [template = constants.%.30]
// CHECK:STDOUT:     %int.convert_checked: init %i32 = call %.loc10_14.3(%.loc10_15) [template = constants.%.31]
// CHECK:STDOUT:     %.loc10_14.4: %i32 = value_of_initializer %int.convert_checked [template = constants.%.31]
// CHECK:STDOUT:     %.loc10_14.5: %i32 = converted %.loc10_15, %.loc10_14.4 [template = constants.%.31]
// CHECK:STDOUT:     %C: type = class_type @C, @C(constants.%.31) [template = constants.%C.3]
// CHECK:STDOUT:     %b.param: %C.3 = value_param runtime_param<invalid>
// CHECK:STDOUT:     %b.loc10_9.1: %C.3 = bind_symbolic_name b, 0, %b.param [symbolic = %b.loc10_9.2 (constants.%b)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @C(%a.loc2_9.1: %i32) {
// CHECK:STDOUT:   %a.loc2_9.2: %i32 = bind_symbolic_name a, 0 [symbolic = %a.loc2_9.2 (constants.%a)]
// CHECK:STDOUT:   %a.patt.loc2_9.2: %i32 = symbolic_binding_pattern a, 0 [symbolic = %a.patt.loc2_9.2 (constants.%a.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %.loc2_19: <witness> = complete_type_witness %.2 [template = constants.%.3]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%C.2
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @D(%b.loc3_9.1: %C.3) {
// CHECK:STDOUT:   %b.loc3_9.2: %C.3 = bind_symbolic_name b, 0 [symbolic = %b.loc3_9.2 (constants.%b)]
// CHECK:STDOUT:   %b.patt.loc3_9.2: %C.3 = symbolic_binding_pattern b, 0 [symbolic = %b.patt.loc3_9.2 (constants.%b.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @.1(%b.loc10_9.1: %C.3) {
// CHECK:STDOUT:   %b.loc10_9.2: %C.3 = bind_symbolic_name b, 0 [symbolic = %b.loc10_9.2 (constants.%b)]
// CHECK:STDOUT:   %b.patt.loc10_9.2: %C.3 = symbolic_binding_pattern b, 0 [symbolic = %b.patt.loc10_9.2 (constants.%b.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %.loc10_24: <witness> = complete_type_witness %.2 [template = constants.%.3]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%.33
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%a) {
// CHECK:STDOUT:   %a.loc2_9.2 => constants.%a
// CHECK:STDOUT:   %a.patt.loc2_9.2 => constants.%a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%.31) {
// CHECK:STDOUT:   %a.loc2_9.2 => constants.%.31
// CHECK:STDOUT:   %a.patt.loc2_9.2 => constants.%.31
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @D(constants.%b) {
// CHECK:STDOUT:   %b.loc3_9.2 => constants.%b
// CHECK:STDOUT:   %b.patt.loc3_9.2 => constants.%b
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @.1(constants.%b) {
// CHECK:STDOUT:   %b.loc10_9.2 => constants.%b
// CHECK:STDOUT:   %b.patt.loc10_9.2 => constants.%b
// CHECK:STDOUT: }
// CHECK:STDOUT:
