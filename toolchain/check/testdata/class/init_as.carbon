// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/init_as.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/init_as.carbon

class Class {
  var a: i32;
  var b: i32;
}

fn F() -> i32 {
  return ({.a = 1, .b = 2} as Class).a;
}

// CHECK:STDOUT: --- init_as.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Class: type = class_type @Class [template]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic]
// CHECK:STDOUT:   %Int: type = class_type @Int, @Int(%N) [symbolic]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [template]
// CHECK:STDOUT:   %Class.elem: type = unbound_element_type %Class, %i32 [template]
// CHECK:STDOUT:   %struct_type.a.b.1: type = struct_type {.a: %i32, .b: %i32} [template]
// CHECK:STDOUT:   %complete_type.3: <witness> = complete_type_witness %struct_type.a.b.1 [template]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [template]
// CHECK:STDOUT:   %struct_type.a.b.2: type = struct_type {.a: Core.IntLiteral, .b: Core.IntLiteral} [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %Convert.type.3: type = fn_type @Convert.2, @impl.1(%N) [symbolic]
// CHECK:STDOUT:   %Convert.3: %Convert.type.3 = struct_value () [symbolic]
// CHECK:STDOUT:   %impl_witness.5: <witness> = impl_witness (%Convert.3), @impl.1(%int_32) [template]
// CHECK:STDOUT:   %Convert.bound.1: <bound method> = bound_method %int_1, %Convert.3 [symbolic]
// CHECK:STDOUT:   %Convert.specific_fn.1: <specific function> = specific_function %Convert.bound.1, @Convert.2(%N) [symbolic]
// CHECK:STDOUT:   %int.convert_checked.1: init %Int = call %Convert.specific_fn.1(%int_1) [symbolic]
// CHECK:STDOUT:   %Convert.bound.2: <bound method> = bound_method %int_2, %Convert.3 [symbolic]
// CHECK:STDOUT:   %Convert.specific_fn.2: <specific function> = specific_function %Convert.bound.2, @Convert.2(%N) [symbolic]
// CHECK:STDOUT:   %int.convert_checked.2: init %Int = call %Convert.specific_fn.2(%int_2) [symbolic]
// CHECK:STDOUT:   %Class.val: %Class = struct_value (%int.convert_checked.1, %int.convert_checked.2) [symbolic]
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
// CHECK:STDOUT:     .Class = %Class.decl
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Class.decl: type = class_decl @Class [template = constants.%Class] {} {}
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [template = constants.%F] {
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %i32: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:     %return.param: ref %i32 = out_param runtime_param0
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Class {
// CHECK:STDOUT:   %int_32.loc12: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %i32.loc12: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:   %.loc12: %Class.elem = field_decl a, element0 [template]
// CHECK:STDOUT:   %int_32.loc13: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %i32.loc13: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:   %.loc13: %Class.elem = field_decl b, element1 [template]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.a.b.1 [template = constants.%complete_type.3]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Class
// CHECK:STDOUT:   .a = %.loc12
// CHECK:STDOUT:   .b = %.loc13
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [template = constants.%int_1]
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [template = constants.%int_2]
// CHECK:STDOUT:   %.loc17_26.1: %struct_type.a.b.2 = struct_literal (%int_1, %int_2)
// CHECK:STDOUT:   %Class.ref: type = name_ref Class, file.%Class.decl [template = constants.%Class]
// CHECK:STDOUT:   %impl.elem0.loc17_26.1: %Convert.type.2 = impl_witness_access constants.%impl_witness.5, element0 [symbolic = constants.%Convert.3]
// CHECK:STDOUT:   %Convert.bound.loc17_26.1: <bound method> = bound_method %int_1, %impl.elem0.loc17_26.1 [symbolic = constants.%Convert.bound.1]
// CHECK:STDOUT:   %Convert.specific_fn.loc17_26.1: <specific function> = specific_function %Convert.bound.loc17_26.1, @Convert.2(constants.%N) [symbolic = constants.%Convert.specific_fn.1]
// CHECK:STDOUT:   %int.convert_checked.loc17_26.1: init %Int = call %Convert.specific_fn.loc17_26.1(%int_1) [symbolic = constants.%int.convert_checked.1]
// CHECK:STDOUT:   %.loc17_26.2: init %i32 = converted %int_1, %int.convert_checked.loc17_26.1 [symbolic = constants.%int.convert_checked.1]
// CHECK:STDOUT:   %.loc17_26.3: ref %Class = temporary_storage
// CHECK:STDOUT:   %.loc17_26.4: ref %i32 = class_element_access %.loc17_26.3, element0
// CHECK:STDOUT:   %.loc17_26.5: init %i32 = initialize_from %.loc17_26.2 to %.loc17_26.4 [symbolic = constants.%int.convert_checked.1]
// CHECK:STDOUT:   %impl.elem0.loc17_26.2: %Convert.type.2 = impl_witness_access constants.%impl_witness.5, element0 [symbolic = constants.%Convert.3]
// CHECK:STDOUT:   %Convert.bound.loc17_26.2: <bound method> = bound_method %int_2, %impl.elem0.loc17_26.2 [symbolic = constants.%Convert.bound.2]
// CHECK:STDOUT:   %Convert.specific_fn.loc17_26.2: <specific function> = specific_function %Convert.bound.loc17_26.2, @Convert.2(constants.%N) [symbolic = constants.%Convert.specific_fn.2]
// CHECK:STDOUT:   %int.convert_checked.loc17_26.2: init %Int = call %Convert.specific_fn.loc17_26.2(%int_2) [symbolic = constants.%int.convert_checked.2]
// CHECK:STDOUT:   %.loc17_26.6: init %i32 = converted %int_2, %int.convert_checked.loc17_26.2 [symbolic = constants.%int.convert_checked.2]
// CHECK:STDOUT:   %.loc17_26.7: ref %i32 = class_element_access %.loc17_26.3, element1
// CHECK:STDOUT:   %.loc17_26.8: init %i32 = initialize_from %.loc17_26.6 to %.loc17_26.7 [symbolic = constants.%int.convert_checked.2]
// CHECK:STDOUT:   %.loc17_26.9: init %Class = class_init (%.loc17_26.5, %.loc17_26.8), %.loc17_26.3 [symbolic = constants.%Class.val]
// CHECK:STDOUT:   %.loc17_26.10: ref %Class = temporary %.loc17_26.3, %.loc17_26.9
// CHECK:STDOUT:   %.loc17_28: ref %Class = converted %.loc17_26.1, %.loc17_26.10
// CHECK:STDOUT:   %a.ref: %Class.elem = name_ref a, @Class.%.loc12 [template = @Class.%.loc12]
// CHECK:STDOUT:   %.loc17_37.1: ref %i32 = class_element_access %.loc17_28, element0
// CHECK:STDOUT:   %.loc17_37.2: %i32 = bind_value %.loc17_37.1
// CHECK:STDOUT:   return %.loc17_37.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
