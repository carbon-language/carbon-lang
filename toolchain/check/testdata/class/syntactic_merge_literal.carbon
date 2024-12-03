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
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %int_32 [template]
// CHECK:STDOUT:   %a: %i32 = bind_symbolic_name a, 0 [symbolic]
// CHECK:STDOUT:   %a.patt: %i32 = symbolic_binding_pattern a, 0 [symbolic]
// CHECK:STDOUT:   %C.type: type = generic_class_type @C [template]
// CHECK:STDOUT:   %C.generic: %C.type = struct_value () [template]
// CHECK:STDOUT:   %C.1: type = class_type @C, @C(%a) [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [template]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template]
// CHECK:STDOUT:   %int_1000.1: Core.IntLiteral = int_value 1000 [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %Convert.type.14: type = fn_type @Convert.2, @impl.1(%int_32) [template]
// CHECK:STDOUT:   %Convert.14: %Convert.type.14 = struct_value () [template]
// CHECK:STDOUT:   %interface.9: <witness> = interface_witness (%Convert.14) [template]
// CHECK:STDOUT:   %Convert.bound: <bound method> = bound_method %int_1000.1, %Convert.14 [template]
// CHECK:STDOUT:   %Convert.specific_fn: <specific function> = specific_function %Convert.bound, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_1000.2: %i32 = int_value 1000 [template]
// CHECK:STDOUT:   %C.2: type = class_type @C, @C(%int_1000.2) [template]
// CHECK:STDOUT:   %b: %C.2 = bind_symbolic_name b, 0 [symbolic]
// CHECK:STDOUT:   %b.patt: %C.2 = symbolic_binding_pattern b, 0 [symbolic]
// CHECK:STDOUT:   %D.type: type = generic_class_type @D [template]
// CHECK:STDOUT:   %D.generic: %D.type = struct_value () [template]
// CHECK:STDOUT:   %D: type = class_type @D, @D(%b) [symbolic]
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
// CHECK:STDOUT:   %C.decl: %C.type = class_decl @C [template = constants.%C.generic] {
// CHECK:STDOUT:     %a.patt.loc2_9.1: %i32 = symbolic_binding_pattern a, 0 [symbolic = %a.patt.loc2_9.2 (constants.%a.patt)]
// CHECK:STDOUT:     %a.param_patt: %i32 = value_param_pattern %a.patt.loc2_9.1, runtime_param<invalid> [symbolic = %a.patt.loc2_9.2 (constants.%a.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed: init type = call constants.%Int(%int_32) [template = constants.%i32]
// CHECK:STDOUT:     %.loc2_13.1: type = value_of_initializer %int.make_type_signed [template = constants.%i32]
// CHECK:STDOUT:     %.loc2_13.2: type = converted %int.make_type_signed, %.loc2_13.1 [template = constants.%i32]
// CHECK:STDOUT:     %a.param: %i32 = value_param runtime_param<invalid>
// CHECK:STDOUT:     %a.loc2_9.1: %i32 = bind_symbolic_name a, 0, %a.param [symbolic = %a.loc2_9.2 (constants.%a)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %D.decl.loc3: %D.type = class_decl @D [template = constants.%D.generic] {
// CHECK:STDOUT:     %b.patt.loc4: %C.2 = symbolic_binding_pattern b, 0 [symbolic = constants.%b.patt]
// CHECK:STDOUT:     %b.param_patt: %C.2 = value_param_pattern %b.patt.loc4, runtime_param<invalid> [symbolic = constants.%b.patt]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref.loc3: %C.type = name_ref C, file.%C.decl [template = constants.%C.generic]
// CHECK:STDOUT:     %int_1000.loc3: Core.IntLiteral = int_value 1000 [template = constants.%int_1000.1]
// CHECK:STDOUT:     %impl.elem0.loc3: %Convert.type.2 = interface_witness_access constants.%interface.9, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:     %Convert.bound.loc3: <bound method> = bound_method %int_1000.loc3, %impl.elem0.loc3 [template = constants.%Convert.bound]
// CHECK:STDOUT:     %Convert.specific_fn.loc3: <specific function> = specific_function %Convert.bound.loc3, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn]
// CHECK:STDOUT:     %int.convert_checked.loc3: init %i32 = call %Convert.specific_fn.loc3(%int_1000.loc3) [template = constants.%int_1000.2]
// CHECK:STDOUT:     %.loc3_20.1: %i32 = value_of_initializer %int.convert_checked.loc3 [template = constants.%int_1000.2]
// CHECK:STDOUT:     %.loc3_20.2: %i32 = converted %int_1000.loc3, %.loc3_20.1 [template = constants.%int_1000.2]
// CHECK:STDOUT:     %C.loc3: type = class_type @C, @C(constants.%int_1000.2) [template = constants.%C.2]
// CHECK:STDOUT:     %b.param.loc3: %C.2 = value_param runtime_param<invalid>
// CHECK:STDOUT:     %b.loc3_9.1: %C.2 = bind_symbolic_name b, 0, %b.param.loc3 [symbolic = %b.loc3_9.2 (constants.%b)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %D.decl.loc4: %D.type = class_decl @D [template = constants.%D.generic] {
// CHECK:STDOUT:     %b.patt.loc4: %C.2 = symbolic_binding_pattern b, 0 [symbolic = constants.%b.patt]
// CHECK:STDOUT:     %b.param_patt: %C.2 = value_param_pattern %b.patt.loc4, runtime_param<invalid> [symbolic = constants.%b.patt]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref.loc4: %C.type = name_ref C, file.%C.decl [template = constants.%C.generic]
// CHECK:STDOUT:     %int_1000.loc4: Core.IntLiteral = int_value 1000 [template = constants.%int_1000.1]
// CHECK:STDOUT:     %impl.elem0.loc4: %Convert.type.2 = interface_witness_access constants.%interface.9, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:     %Convert.bound.loc4: <bound method> = bound_method %int_1000.loc4, %impl.elem0.loc4 [template = constants.%Convert.bound]
// CHECK:STDOUT:     %Convert.specific_fn.loc4: <specific function> = specific_function %Convert.bound.loc4, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn]
// CHECK:STDOUT:     %int.convert_checked.loc4: init %i32 = call %Convert.specific_fn.loc4(%int_1000.loc4) [template = constants.%int_1000.2]
// CHECK:STDOUT:     %.loc4_20.1: %i32 = value_of_initializer %int.convert_checked.loc4 [template = constants.%int_1000.2]
// CHECK:STDOUT:     %.loc4_20.2: %i32 = converted %int_1000.loc4, %.loc4_20.1 [template = constants.%int_1000.2]
// CHECK:STDOUT:     %C.loc4: type = class_type @C, @C(constants.%int_1000.2) [template = constants.%C.2]
// CHECK:STDOUT:     %b.param.loc4: %C.2 = value_param runtime_param<invalid>
// CHECK:STDOUT:     %b.loc4: %C.2 = bind_symbolic_name b, 0, %b.param.loc4 [symbolic = constants.%b]
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
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness %empty_struct_type [template = constants.%complete_type]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%C.1
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @D(%b.loc3_9.1: %C.2) {
// CHECK:STDOUT:   %b.loc3_9.2: %C.2 = bind_symbolic_name b, 0 [symbolic = %b.loc3_9.2 (constants.%b)]
// CHECK:STDOUT:   %b.patt.loc3: %C.2 = symbolic_binding_pattern b, 0 [symbolic = %b.patt.loc3 (constants.%b.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness %empty_struct_type [template = constants.%complete_type]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%D
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%a) {
// CHECK:STDOUT:   %a.loc2_9.2 => constants.%a
// CHECK:STDOUT:   %a.patt.loc2_9.2 => constants.%a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%int_1000.2) {
// CHECK:STDOUT:   %a.loc2_9.2 => constants.%int_1000.2
// CHECK:STDOUT:   %a.patt.loc2_9.2 => constants.%int_1000.2
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
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %int_32 [template]
// CHECK:STDOUT:   %a: %i32 = bind_symbolic_name a, 0 [symbolic]
// CHECK:STDOUT:   %a.patt: %i32 = symbolic_binding_pattern a, 0 [symbolic]
// CHECK:STDOUT:   %C.type: type = generic_class_type @C [template]
// CHECK:STDOUT:   %C.generic: %C.type = struct_value () [template]
// CHECK:STDOUT:   %C.1: type = class_type @C, @C(%a) [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [template]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template]
// CHECK:STDOUT:   %int_1000.1: Core.IntLiteral = int_value 1000 [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %Convert.type.14: type = fn_type @Convert.2, @impl.1(%int_32) [template]
// CHECK:STDOUT:   %Convert.14: %Convert.type.14 = struct_value () [template]
// CHECK:STDOUT:   %interface.9: <witness> = interface_witness (%Convert.14) [template]
// CHECK:STDOUT:   %Convert.bound: <bound method> = bound_method %int_1000.1, %Convert.14 [template]
// CHECK:STDOUT:   %Convert.specific_fn: <specific function> = specific_function %Convert.bound, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_1000.2: %i32 = int_value 1000 [template]
// CHECK:STDOUT:   %C.2: type = class_type @C, @C(%int_1000.2) [template]
// CHECK:STDOUT:   %b: %C.2 = bind_symbolic_name b, 0 [symbolic]
// CHECK:STDOUT:   %b.patt: %C.2 = symbolic_binding_pattern b, 0 [symbolic]
// CHECK:STDOUT:   %D.type: type = generic_class_type @D [template]
// CHECK:STDOUT:   %D.generic: %D.type = struct_value () [template]
// CHECK:STDOUT:   %.type: type = generic_class_type @.1 [template]
// CHECK:STDOUT:   %.generic: %.type = struct_value () [template]
// CHECK:STDOUT:   %.1: type = class_type @.1, @.1(%b) [symbolic]
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
// CHECK:STDOUT:   %C.decl: %C.type = class_decl @C [template = constants.%C.generic] {
// CHECK:STDOUT:     %a.patt.loc2_9.1: %i32 = symbolic_binding_pattern a, 0 [symbolic = %a.patt.loc2_9.2 (constants.%a.patt)]
// CHECK:STDOUT:     %a.param_patt: %i32 = value_param_pattern %a.patt.loc2_9.1, runtime_param<invalid> [symbolic = %a.patt.loc2_9.2 (constants.%a.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed: init type = call constants.%Int(%int_32) [template = constants.%i32]
// CHECK:STDOUT:     %.loc2_13.1: type = value_of_initializer %int.make_type_signed [template = constants.%i32]
// CHECK:STDOUT:     %.loc2_13.2: type = converted %int.make_type_signed, %.loc2_13.1 [template = constants.%i32]
// CHECK:STDOUT:     %a.param: %i32 = value_param runtime_param<invalid>
// CHECK:STDOUT:     %a.loc2_9.1: %i32 = bind_symbolic_name a, 0, %a.param [symbolic = %a.loc2_9.2 (constants.%a)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %D.decl: %D.type = class_decl @D [template = constants.%D.generic] {
// CHECK:STDOUT:     %b.patt.loc3_9.1: %C.2 = symbolic_binding_pattern b, 0 [symbolic = %b.patt.loc3_9.2 (constants.%b.patt)]
// CHECK:STDOUT:     %b.param_patt: %C.2 = value_param_pattern %b.patt.loc3_9.1, runtime_param<invalid> [symbolic = %b.patt.loc3_9.2 (constants.%b.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref: %C.type = name_ref C, file.%C.decl [template = constants.%C.generic]
// CHECK:STDOUT:     %int_1000: Core.IntLiteral = int_value 1000 [template = constants.%int_1000.1]
// CHECK:STDOUT:     %impl.elem0: %Convert.type.2 = interface_witness_access constants.%interface.9, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:     %Convert.bound: <bound method> = bound_method %int_1000, %impl.elem0 [template = constants.%Convert.bound]
// CHECK:STDOUT:     %Convert.specific_fn: <specific function> = specific_function %Convert.bound, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn]
// CHECK:STDOUT:     %int.convert_checked: init %i32 = call %Convert.specific_fn(%int_1000) [template = constants.%int_1000.2]
// CHECK:STDOUT:     %.loc3_19.1: %i32 = value_of_initializer %int.convert_checked [template = constants.%int_1000.2]
// CHECK:STDOUT:     %.loc3_19.2: %i32 = converted %int_1000, %.loc3_19.1 [template = constants.%int_1000.2]
// CHECK:STDOUT:     %C: type = class_type @C, @C(constants.%int_1000.2) [template = constants.%C.2]
// CHECK:STDOUT:     %b.param: %C.2 = value_param runtime_param<invalid>
// CHECK:STDOUT:     %b.loc3_9.1: %C.2 = bind_symbolic_name b, 0, %b.param [symbolic = %b.loc3_9.2 (constants.%b)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.decl: %.type = class_decl @.1 [template = constants.%.generic] {
// CHECK:STDOUT:     %b.patt.loc10_9.1: %C.2 = symbolic_binding_pattern b, 0 [symbolic = %b.patt.loc10_9.2 (constants.%b.patt)]
// CHECK:STDOUT:     %b.param_patt: %C.2 = value_param_pattern %b.patt.loc10_9.1, runtime_param<invalid> [symbolic = %b.patt.loc10_9.2 (constants.%b.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref: %C.type = name_ref C, file.%C.decl [template = constants.%C.generic]
// CHECK:STDOUT:     %int_1000: Core.IntLiteral = int_value 1000 [template = constants.%int_1000.1]
// CHECK:STDOUT:     %impl.elem0: %Convert.type.2 = interface_witness_access constants.%interface.9, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:     %Convert.bound: <bound method> = bound_method %int_1000, %impl.elem0 [template = constants.%Convert.bound]
// CHECK:STDOUT:     %Convert.specific_fn: <specific function> = specific_function %Convert.bound, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn]
// CHECK:STDOUT:     %int.convert_checked: init %i32 = call %Convert.specific_fn(%int_1000) [template = constants.%int_1000.2]
// CHECK:STDOUT:     %.loc10_20.1: %i32 = value_of_initializer %int.convert_checked [template = constants.%int_1000.2]
// CHECK:STDOUT:     %.loc10_20.2: %i32 = converted %int_1000, %.loc10_20.1 [template = constants.%int_1000.2]
// CHECK:STDOUT:     %C: type = class_type @C, @C(constants.%int_1000.2) [template = constants.%C.2]
// CHECK:STDOUT:     %b.param: %C.2 = value_param runtime_param<invalid>
// CHECK:STDOUT:     %b.loc10_9.1: %C.2 = bind_symbolic_name b, 0, %b.param [symbolic = %b.loc10_9.2 (constants.%b)]
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
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness %empty_struct_type [template = constants.%complete_type]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%C.1
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @D(%b.loc3_9.1: %C.2) {
// CHECK:STDOUT:   %b.loc3_9.2: %C.2 = bind_symbolic_name b, 0 [symbolic = %b.loc3_9.2 (constants.%b)]
// CHECK:STDOUT:   %b.patt.loc3_9.2: %C.2 = symbolic_binding_pattern b, 0 [symbolic = %b.patt.loc3_9.2 (constants.%b.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @.1(%b.loc10_9.1: %C.2) {
// CHECK:STDOUT:   %b.loc10_9.2: %C.2 = bind_symbolic_name b, 0 [symbolic = %b.loc10_9.2 (constants.%b)]
// CHECK:STDOUT:   %b.patt.loc10_9.2: %C.2 = symbolic_binding_pattern b, 0 [symbolic = %b.patt.loc10_9.2 (constants.%b.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness %empty_struct_type [template = constants.%complete_type]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%.1
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%a) {
// CHECK:STDOUT:   %a.loc2_9.2 => constants.%a
// CHECK:STDOUT:   %a.patt.loc2_9.2 => constants.%a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%int_1000.2) {
// CHECK:STDOUT:   %a.loc2_9.2 => constants.%int_1000.2
// CHECK:STDOUT:   %a.patt.loc2_9.2 => constants.%int_1000.2
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
