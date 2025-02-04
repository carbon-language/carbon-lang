// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/local.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/local.carbon

class A {
  fn F() -> i32 {
    class B {
      fn Make() -> Self {
        returned var b: Self = {.n = 1};
        return var;
      }

      var n: i32;
    }

    return B.Make().n;
  }
}

// CHECK:STDOUT: --- local.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %A: type = class_type @A [template]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [template]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [template]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [template]
// CHECK:STDOUT:   %B: type = class_type @B [template]
// CHECK:STDOUT:   %Make.type: type = fn_type @Make [template]
// CHECK:STDOUT:   %Make: %Make.type = struct_value () [template]
// CHECK:STDOUT:   %B.elem: type = unbound_element_type %B, %i32 [template]
// CHECK:STDOUT:   %struct_type.n.033: type = struct_type {.n: %i32} [template]
// CHECK:STDOUT:   %complete_type.54b: <witness> = complete_type_witness %struct_type.n.033 [template]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT:   %struct_type.n.44a: type = struct_type {.n: Core.IntLiteral} [template]
// CHECK:STDOUT:   %ImplicitAs.type.205: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [template]
// CHECK:STDOUT:   %Convert.type.1b6: type = fn_type @Convert.1, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %impl_witness.d39: <witness> = impl_witness (imports.%Core.import_ref.a5b), @impl.1(%int_32) [template]
// CHECK:STDOUT:   %Convert.type.035: type = fn_type @Convert.2, @impl.1(%int_32) [template]
// CHECK:STDOUT:   %Convert.956: %Convert.type.035 = struct_value () [template]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.205 = facet_value Core.IntLiteral, %impl_witness.d39 [template]
// CHECK:STDOUT:   %.a0b: type = fn_type_with_self_type %Convert.type.1b6, %ImplicitAs.facet [template]
// CHECK:STDOUT:   %Convert.bound: <bound method> = bound_method %int_1.5b8, %Convert.956 [template]
// CHECK:STDOUT:   %Convert.specific_fn: <specific function> = specific_function %Convert.bound, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_1.5d2: %i32 = int_value 1 [template]
// CHECK:STDOUT:   %B.val: %B = struct_value (%int_1.5d2) [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
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
// CHECK:STDOUT:   %A.decl: type = class_decl @A [template = constants.%A] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @A {
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [template = constants.%F] {
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %i32: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:     %return.param: ref %i32 = out_param runtime_param0
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template = constants.%complete_type.357]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%A
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @B {
// CHECK:STDOUT:   %Make.decl: %Make.type = fn_decl @Make [template = constants.%Make] {
// CHECK:STDOUT:     %return.patt: %B = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %B = out_param_pattern %return.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Self.ref.loc14: type = name_ref Self, constants.%B [template = constants.%B]
// CHECK:STDOUT:     %return.param: ref %B = out_param runtime_param0
// CHECK:STDOUT:     %return: ref %B = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc19_12: %B.elem = field_decl n, element0 [template]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %.loc19_7: %B.elem = var_pattern %.loc19_12
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.var: ref %B.elem = var <none>
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.n.033 [template = constants.%complete_type.54b]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%B
// CHECK:STDOUT:   .Make = %Make.decl
// CHECK:STDOUT:   .n = %.loc19_12
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %B.decl: type = class_decl @B [template = constants.%B] {} {}
// CHECK:STDOUT:   %B.ref: type = name_ref B, %B.decl [template = constants.%B]
// CHECK:STDOUT:   %Make.ref: %Make.type = name_ref Make, @B.%Make.decl [template = constants.%Make]
// CHECK:STDOUT:   %.loc22_19.1: ref %B = temporary_storage
// CHECK:STDOUT:   %Make.call: init %B = call %Make.ref() to %.loc22_19.1
// CHECK:STDOUT:   %.loc22_19.2: ref %B = temporary %.loc22_19.1, %Make.call
// CHECK:STDOUT:   %n.ref: %B.elem = name_ref n, @B.%.loc19_12 [template = @B.%.loc19_12]
// CHECK:STDOUT:   %.loc22_20.1: ref %i32 = class_element_access %.loc22_19.2, element0
// CHECK:STDOUT:   %.loc22_20.2: %i32 = bind_value %.loc22_20.1
// CHECK:STDOUT:   return %.loc22_20.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Make() -> %return.param_patt: %B {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %B = binding_pattern b
// CHECK:STDOUT:     %.loc15_18.1: %B = var_pattern %b.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [template = constants.%int_1.5b8]
// CHECK:STDOUT:   %.loc15_39.1: %struct_type.n.44a = struct_literal (%int_1)
// CHECK:STDOUT:   %impl.elem0: %.a0b = impl_witness_access constants.%impl_witness.d39, element0 [template = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %int_1, %impl.elem0 [template = constants.%Convert.bound]
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %bound_method, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %int.convert_checked: init %i32 = call %specific_fn(%int_1) [template = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc15_39.2: init %i32 = converted %int_1, %int.convert_checked [template = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc15_39.3: ref %i32 = class_element_access %return, element0
// CHECK:STDOUT:   %.loc15_39.4: init %i32 = initialize_from %.loc15_39.2 to %.loc15_39.3 [template = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc15_39.5: init %B = class_init (%.loc15_39.4), %return [template = constants.%B.val]
// CHECK:STDOUT:   %.loc15_18.2: init %B = converted %.loc15_39.1, %.loc15_39.5 [template = constants.%B.val]
// CHECK:STDOUT:   assign %return, %.loc15_18.2
// CHECK:STDOUT:   %Self.ref.loc15: type = name_ref Self, constants.%B [template = constants.%B]
// CHECK:STDOUT:   %b: ref %B = bind_name b, %return
// CHECK:STDOUT:   return %b to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
