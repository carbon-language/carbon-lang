// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/base.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/base.carbon

// --- base.carbon
package Base;

base class Base {
  var b: i32;
}

class Derived {
  extend base: Base;

  var d: i32;
}

fn Make() -> Derived {
  return {.base = {.b = 4}, .d = 7};
}

fn Access(d: Derived) -> (i32, i32) {
  return (d.d, d.base.b);
}

// --- fail_base_after_field.carbon
package BaseAfterField;

base class Base {
}

class Derived {
  var d: i32;

  // CHECK:STDERR: fail_base_after_field.carbon:[[@LINE+3]]:3: error: `base` declaration must appear before field declarations [BaseDeclAfterFieldDecl]
  // CHECK:STDERR:   extend base: Base;
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~
  extend base: Base;
}

// CHECK:STDOUT: --- base.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Base: type = class_type @Base [template]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [template]
// CHECK:STDOUT:   %Base.elem: type = unbound_element_type %Base, %i32 [template]
// CHECK:STDOUT:   %struct_type.b.fc0: type = struct_type {.b: %i32} [template]
// CHECK:STDOUT:   %complete_type.d62: <witness> = complete_type_witness %struct_type.b.fc0 [template]
// CHECK:STDOUT:   %Derived: type = class_type @Derived [template]
// CHECK:STDOUT:   %Derived.elem.69e: type = unbound_element_type %Derived, %Base [template]
// CHECK:STDOUT:   %Derived.elem.d12: type = unbound_element_type %Derived, %i32 [template]
// CHECK:STDOUT:   %struct_type.base.d.79e: type = struct_type {.base: %Base, .d: %i32} [template]
// CHECK:STDOUT:   %complete_type.71d: <witness> = complete_type_witness %struct_type.base.d.79e [template]
// CHECK:STDOUT:   %Make.type: type = fn_type @Make [template]
// CHECK:STDOUT:   %Make: %Make.type = struct_value () [template]
// CHECK:STDOUT:   %int_4.0c1: Core.IntLiteral = int_value 4 [template]
// CHECK:STDOUT:   %struct_type.b.a15: type = struct_type {.b: Core.IntLiteral} [template]
// CHECK:STDOUT:   %int_7.29f: Core.IntLiteral = int_value 7 [template]
// CHECK:STDOUT:   %struct_type.base.d.a20: type = struct_type {.base: %struct_type.b.a15, .d: Core.IntLiteral} [template]
// CHECK:STDOUT:   %Convert.type.cd1: type = fn_type @Convert.1, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %impl_witness.5b0: <witness> = impl_witness (imports.%import_ref.723), @impl.1(%int_32) [template]
// CHECK:STDOUT:   %Convert.type.466: type = fn_type @Convert.2, @impl.1(%int_32) [template]
// CHECK:STDOUT:   %Convert.925: %Convert.type.466 = struct_value () [template]
// CHECK:STDOUT:   %Convert.bound.fe5: <bound method> = bound_method %int_4.0c1, %Convert.925 [template]
// CHECK:STDOUT:   %Convert.specific_fn.c6f: <specific function> = specific_function %Convert.bound.fe5, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_4.2ec: %i32 = int_value 4 [template]
// CHECK:STDOUT:   %Base.val: %Base = struct_value (%int_4.2ec) [template]
// CHECK:STDOUT:   %Convert.bound.eba: <bound method> = bound_method %int_7.29f, %Convert.925 [template]
// CHECK:STDOUT:   %Convert.specific_fn.785: <specific function> = specific_function %Convert.bound.eba, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_7.2da: %i32 = int_value 7 [template]
// CHECK:STDOUT:   %Derived.val: %Derived = struct_value (%Base.val, %int_7.2da) [template]
// CHECK:STDOUT:   %tuple.type.24b: type = tuple_type (type, type) [template]
// CHECK:STDOUT:   %tuple.type.471: type = tuple_type (%i32, %i32) [template]
// CHECK:STDOUT:   %Access.type: type = fn_type @Access [template]
// CHECK:STDOUT:   %Access: %Access.type = struct_value () [template]
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
// CHECK:STDOUT:     .Base = %Base.decl
// CHECK:STDOUT:     .Derived = %Derived.decl
// CHECK:STDOUT:     .Make = %Make.decl
// CHECK:STDOUT:     .Access = %Access.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Base.decl: type = class_decl @Base [template = constants.%Base] {} {}
// CHECK:STDOUT:   %Derived.decl: type = class_decl @Derived [template = constants.%Derived] {} {}
// CHECK:STDOUT:   %Make.decl: %Make.type = fn_decl @Make [template = constants.%Make] {
// CHECK:STDOUT:     %return.patt: %Derived = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %Derived = out_param_pattern %return.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Derived.ref: type = name_ref Derived, file.%Derived.decl [template = constants.%Derived]
// CHECK:STDOUT:     %return.param: ref %Derived = out_param runtime_param0
// CHECK:STDOUT:     %return: ref %Derived = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Access.decl: %Access.type = fn_decl @Access [template = constants.%Access] {
// CHECK:STDOUT:     %d.patt: %Derived = binding_pattern d
// CHECK:STDOUT:     %d.param_patt: %Derived = value_param_pattern %d.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: %tuple.type.471 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %tuple.type.471 = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32.loc17_27: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %i32.loc17_27: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:     %int_32.loc17_32: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %i32.loc17_32: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:     %.loc17_35.1: %tuple.type.24b = tuple_literal (%i32.loc17_27, %i32.loc17_32)
// CHECK:STDOUT:     %.loc17_35.2: type = converted %.loc17_35.1, constants.%tuple.type.471 [template = constants.%tuple.type.471]
// CHECK:STDOUT:     %d.param: %Derived = value_param runtime_param0
// CHECK:STDOUT:     %Derived.ref: type = name_ref Derived, file.%Derived.decl [template = constants.%Derived]
// CHECK:STDOUT:     %d: %Derived = bind_name d, %d.param
// CHECK:STDOUT:     %return.param: ref %tuple.type.471 = out_param runtime_param1
// CHECK:STDOUT:     %return: ref %tuple.type.471 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Base {
// CHECK:STDOUT:   %.loc4: %Base.elem = field_decl b, element0 [template]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.b.fc0 [template = constants.%complete_type.d62]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Base
// CHECK:STDOUT:   .b = %.loc4
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Derived {
// CHECK:STDOUT:   %Base.ref: type = name_ref Base, file.%Base.decl [template = constants.%Base]
// CHECK:STDOUT:   %.loc8: %Derived.elem.69e = base_decl %Base.ref, element0 [template]
// CHECK:STDOUT:   %.loc10: %Derived.elem.d12 = field_decl d, element1 [template]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.base.d.79e [template = constants.%complete_type.71d]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Derived
// CHECK:STDOUT:   .base = %.loc8
// CHECK:STDOUT:   .d = %.loc10
// CHECK:STDOUT:   extend %Base.ref
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Make() -> %return.param_patt: %Derived {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_4: Core.IntLiteral = int_value 4 [template = constants.%int_4.0c1]
// CHECK:STDOUT:   %.loc14_26.1: %struct_type.b.a15 = struct_literal (%int_4)
// CHECK:STDOUT:   %int_7: Core.IntLiteral = int_value 7 [template = constants.%int_7.29f]
// CHECK:STDOUT:   %.loc14_35.1: %struct_type.base.d.a20 = struct_literal (%.loc14_26.1, %int_7)
// CHECK:STDOUT:   %impl.elem0.loc14_26: %Convert.type.cd1 = impl_witness_access constants.%impl_witness.5b0, element0 [template = constants.%Convert.925]
// CHECK:STDOUT:   %Convert.bound.loc14_26: <bound method> = bound_method %int_4, %impl.elem0.loc14_26 [template = constants.%Convert.bound.fe5]
// CHECK:STDOUT:   %Convert.specific_fn.loc14_26: <specific function> = specific_function %Convert.bound.loc14_26, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.c6f]
// CHECK:STDOUT:   %int.convert_checked.loc14_26: init %i32 = call %Convert.specific_fn.loc14_26(%int_4) [template = constants.%int_4.2ec]
// CHECK:STDOUT:   %.loc14_26.2: init %i32 = converted %int_4, %int.convert_checked.loc14_26 [template = constants.%int_4.2ec]
// CHECK:STDOUT:   %.loc14_35.2: ref %Base = class_element_access %return, element0
// CHECK:STDOUT:   %.loc14_26.3: ref %i32 = class_element_access %.loc14_35.2, element0
// CHECK:STDOUT:   %.loc14_26.4: init %i32 = initialize_from %.loc14_26.2 to %.loc14_26.3 [template = constants.%int_4.2ec]
// CHECK:STDOUT:   %.loc14_26.5: init %Base = class_init (%.loc14_26.4), %.loc14_35.2 [template = constants.%Base.val]
// CHECK:STDOUT:   %.loc14_35.3: init %Base = converted %.loc14_26.1, %.loc14_26.5 [template = constants.%Base.val]
// CHECK:STDOUT:   %impl.elem0.loc14_35: %Convert.type.cd1 = impl_witness_access constants.%impl_witness.5b0, element0 [template = constants.%Convert.925]
// CHECK:STDOUT:   %Convert.bound.loc14_35: <bound method> = bound_method %int_7, %impl.elem0.loc14_35 [template = constants.%Convert.bound.eba]
// CHECK:STDOUT:   %Convert.specific_fn.loc14_35: <specific function> = specific_function %Convert.bound.loc14_35, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.785]
// CHECK:STDOUT:   %int.convert_checked.loc14_35: init %i32 = call %Convert.specific_fn.loc14_35(%int_7) [template = constants.%int_7.2da]
// CHECK:STDOUT:   %.loc14_35.4: init %i32 = converted %int_7, %int.convert_checked.loc14_35 [template = constants.%int_7.2da]
// CHECK:STDOUT:   %.loc14_35.5: ref %i32 = class_element_access %return, element1
// CHECK:STDOUT:   %.loc14_35.6: init %i32 = initialize_from %.loc14_35.4 to %.loc14_35.5 [template = constants.%int_7.2da]
// CHECK:STDOUT:   %.loc14_35.7: init %Derived = class_init (%.loc14_35.3, %.loc14_35.6), %return [template = constants.%Derived.val]
// CHECK:STDOUT:   %.loc14_36: init %Derived = converted %.loc14_35.1, %.loc14_35.7 [template = constants.%Derived.val]
// CHECK:STDOUT:   return %.loc14_36 to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Access(%d.param_patt: %Derived) -> %return.param_patt: %tuple.type.471 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %d.ref.loc18_11: %Derived = name_ref d, %d
// CHECK:STDOUT:   %d.ref.loc18_12: %Derived.elem.d12 = name_ref d, @Derived.%.loc10 [template = @Derived.%.loc10]
// CHECK:STDOUT:   %.loc18_12.1: ref %i32 = class_element_access %d.ref.loc18_11, element1
// CHECK:STDOUT:   %.loc18_12.2: %i32 = bind_value %.loc18_12.1
// CHECK:STDOUT:   %d.ref.loc18_16: %Derived = name_ref d, %d
// CHECK:STDOUT:   %base.ref: %Derived.elem.69e = name_ref base, @Derived.%.loc8 [template = @Derived.%.loc8]
// CHECK:STDOUT:   %.loc18_17.1: ref %Base = class_element_access %d.ref.loc18_16, element0
// CHECK:STDOUT:   %.loc18_17.2: %Base = bind_value %.loc18_17.1
// CHECK:STDOUT:   %b.ref: %Base.elem = name_ref b, @Base.%.loc4 [template = @Base.%.loc4]
// CHECK:STDOUT:   %.loc18_22.1: ref %i32 = class_element_access %.loc18_17.2, element0
// CHECK:STDOUT:   %.loc18_22.2: %i32 = bind_value %.loc18_22.1
// CHECK:STDOUT:   %.loc18_24.1: %tuple.type.471 = tuple_literal (%.loc18_12.2, %.loc18_22.2)
// CHECK:STDOUT:   %tuple.elem0: ref %i32 = tuple_access %return, element0
// CHECK:STDOUT:   %.loc18_24.2: init %i32 = initialize_from %.loc18_12.2 to %tuple.elem0
// CHECK:STDOUT:   %tuple.elem1: ref %i32 = tuple_access %return, element1
// CHECK:STDOUT:   %.loc18_24.3: init %i32 = initialize_from %.loc18_22.2 to %tuple.elem1
// CHECK:STDOUT:   %.loc18_24.4: init %tuple.type.471 = tuple_init (%.loc18_24.2, %.loc18_24.3) to %return
// CHECK:STDOUT:   %.loc18_25: init %tuple.type.471 = converted %.loc18_24.1, %.loc18_24.4
// CHECK:STDOUT:   return %.loc18_25 to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_base_after_field.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Base: type = class_type @Base [template]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [template]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [template]
// CHECK:STDOUT:   %Derived: type = class_type @Derived [template]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [template]
// CHECK:STDOUT:   %Derived.elem: type = unbound_element_type %Derived, %i32 [template]
// CHECK:STDOUT:   %struct_type.d: type = struct_type {.d: %i32} [template]
// CHECK:STDOUT:   %complete_type.23b: <witness> = complete_type_witness %struct_type.d [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int = %import_ref.187
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Base = %Base.decl
// CHECK:STDOUT:     .Derived = %Derived.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Base.decl: type = class_decl @Base [template = constants.%Base] {} {}
// CHECK:STDOUT:   %Derived.decl: type = class_decl @Derived [template = constants.%Derived] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Base {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template = constants.%complete_type.357]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Base
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Derived {
// CHECK:STDOUT:   %.loc7: %Derived.elem = field_decl d, element0 [template]
// CHECK:STDOUT:   %Base.ref: type = name_ref Base, file.%Base.decl [template = constants.%Base]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.d [template = constants.%complete_type.23b]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Derived
// CHECK:STDOUT:   .d = %.loc7
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
