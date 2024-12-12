// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/generic/complete_in_conversion.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/generic/complete_in_conversion.carbon

// --- fail_derived_to_base.carbon

base class B {}

class A(N:! i32) {
  extend base: B;

  // CHECK:STDERR: fail_derived_to_base.carbon:[[@LINE+3]]:10: error: integer type width of 0 is not positive [IntWidthNotPositive]
  // CHECK:STDERR:   var n: Core.Int(N);
  // CHECK:STDERR:          ^~~~~~~~~~~
  var n: Core.Int(N);
}

fn F(a: A(0)*) {
  // CHECK:STDERR: fail_derived_to_base.carbon:[[@LINE+3]]:3: note: in `A(0)` used here [ResolvingSpecificHere]
  // CHECK:STDERR:   let b: B* = a;
  // CHECK:STDERR:   ^~~~~~~~~~~~~~
  let b: B* = a;
}

// CHECK:STDOUT: --- fail_derived_to_base.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %B: type = class_type @B [template]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [template]
// CHECK:STDOUT:   %complete_type.1: <witness> = complete_type_witness %empty_struct_type [template]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %int_32 [template]
// CHECK:STDOUT:   %N.1: %i32 = bind_symbolic_name N, 0 [symbolic]
// CHECK:STDOUT:   %N.patt.1: %i32 = symbolic_binding_pattern N, 0 [symbolic]
// CHECK:STDOUT:   %A.type: type = generic_class_type @A [template]
// CHECK:STDOUT:   %A.generic: %A.type = struct_value () [template]
// CHECK:STDOUT:   %A.1: type = class_type @A, @A(%N.1) [symbolic]
// CHECK:STDOUT:   %A.elem.1: type = unbound_element_type %A.1, %B [symbolic]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(Core.IntLiteral) [template]
// CHECK:STDOUT:   %Convert.type.13: type = fn_type @Convert.4, @impl.3(%int_32) [template]
// CHECK:STDOUT:   %Convert.13: %Convert.type.13 = struct_value () [template]
// CHECK:STDOUT:   %interface.9: <witness> = interface_witness (%Convert.13) [template]
// CHECK:STDOUT:   %Convert.bound.1: <bound method> = bound_method %N.1, %Convert.13 [symbolic]
// CHECK:STDOUT:   %Convert.specific_fn.1: <specific function> = specific_function %Convert.bound.1, @Convert.4(%int_32) [symbolic]
// CHECK:STDOUT:   %int.convert_checked: init Core.IntLiteral = call %Convert.specific_fn.1(%N.1) [symbolic]
// CHECK:STDOUT:   %iN.2: type = int_type signed, %int.convert_checked [symbolic]
// CHECK:STDOUT:   %require_complete.7: <witness> = require_complete_type %iN.2 [symbolic]
// CHECK:STDOUT:   %A.elem.2: type = unbound_element_type %A.1, %iN.2 [symbolic]
// CHECK:STDOUT:   %struct_type.base.n: type = struct_type {.base: %B, .n: %iN.2} [symbolic]
// CHECK:STDOUT:   %complete_type.3: <witness> = complete_type_witness %struct_type.base.n [symbolic]
// CHECK:STDOUT:   %int_0.1: Core.IntLiteral = int_value 0 [template]
// CHECK:STDOUT:   %Convert.type.14: type = fn_type @Convert.1, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %Convert.type.15: type = fn_type @Convert.2, @impl.1(%int_32) [template]
// CHECK:STDOUT:   %Convert.15: %Convert.type.15 = struct_value () [template]
// CHECK:STDOUT:   %interface.10: <witness> = interface_witness (%Convert.15) [template]
// CHECK:STDOUT:   %Convert.bound.2: <bound method> = bound_method %int_0.1, %Convert.15 [template]
// CHECK:STDOUT:   %Convert.specific_fn.2: <specific function> = specific_function %Convert.bound.2, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_0.2: %i32 = int_value 0 [template]
// CHECK:STDOUT:   %A.2: type = class_type @A, @A(%int_0.2) [template]
// CHECK:STDOUT:   %ptr.2: type = ptr_type %A.2 [template]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT:   %ptr.3: type = ptr_type %B [template]
// CHECK:STDOUT:   %A.elem.3: type = unbound_element_type %A.2, %B [template]
// CHECK:STDOUT:   %Convert.bound.3: <bound method> = bound_method %int_0.2, %Convert.13 [template]
// CHECK:STDOUT:   %Convert.specific_fn.3: <specific function> = specific_function %Convert.bound.3, @Convert.4(%int_32) [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int = %import_ref.1
// CHECK:STDOUT:     .ImplicitAs = %import_ref.2
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.1: %Int.type = import_ref Core//prelude/types, Int, loaded [template = constants.%Int]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .B = %B.decl
// CHECK:STDOUT:     .A = %A.decl
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %B.decl: type = class_decl @B [template = constants.%B] {} {}
// CHECK:STDOUT:   %A.decl: %A.type = class_decl @A [template = constants.%A.generic] {
// CHECK:STDOUT:     %N.patt.loc4_9.1: %i32 = symbolic_binding_pattern N, 0 [symbolic = %N.patt.loc4_9.2 (constants.%N.patt.1)]
// CHECK:STDOUT:     %N.param_patt: %i32 = value_param_pattern %N.patt.loc4_9.1, runtime_param<invalid> [symbolic = %N.patt.loc4_9.2 (constants.%N.patt.1)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed.loc4: init type = call constants.%Int(%int_32) [template = constants.%i32]
// CHECK:STDOUT:     %.loc4_13.1: type = value_of_initializer %int.make_type_signed.loc4 [template = constants.%i32]
// CHECK:STDOUT:     %.loc4_13.2: type = converted %int.make_type_signed.loc4, %.loc4_13.1 [template = constants.%i32]
// CHECK:STDOUT:     %N.param: %i32 = value_param runtime_param<invalid>
// CHECK:STDOUT:     %N.loc4_9.1: %i32 = bind_symbolic_name N, 0, %N.param [symbolic = %N.loc4_9.2 (constants.%N.1)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [template = constants.%F] {
// CHECK:STDOUT:     %a.patt: %ptr.2 = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: %ptr.2 = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %A.ref: %A.type = name_ref A, file.%A.decl [template = constants.%A.generic]
// CHECK:STDOUT:     %int_0: Core.IntLiteral = int_value 0 [template = constants.%int_0.1]
// CHECK:STDOUT:     %impl.elem0: %Convert.type.14 = interface_witness_access constants.%interface.10, element0 [template = constants.%Convert.15]
// CHECK:STDOUT:     %Convert.bound: <bound method> = bound_method %int_0, %impl.elem0 [template = constants.%Convert.bound.2]
// CHECK:STDOUT:     %Convert.specific_fn: <specific function> = specific_function %Convert.bound, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.2]
// CHECK:STDOUT:     %int.convert_checked: init %i32 = call %Convert.specific_fn(%int_0) [template = constants.%int_0.2]
// CHECK:STDOUT:     %.loc13_12.1: %i32 = value_of_initializer %int.convert_checked [template = constants.%int_0.2]
// CHECK:STDOUT:     %.loc13_12.2: %i32 = converted %int_0, %.loc13_12.1 [template = constants.%int_0.2]
// CHECK:STDOUT:     %A: type = class_type @A, @A(constants.%int_0.2) [template = constants.%A.2]
// CHECK:STDOUT:     %ptr.loc13: type = ptr_type %A.2 [template = constants.%ptr.2]
// CHECK:STDOUT:     %a.param: %ptr.2 = value_param runtime_param0
// CHECK:STDOUT:     %a: %ptr.2 = bind_name a, %a.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @B {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template = constants.%complete_type.1]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%B
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @A(%N.loc4_9.1: %i32) {
// CHECK:STDOUT:   %N.loc4_9.2: %i32 = bind_symbolic_name N, 0 [symbolic = %N.loc4_9.2 (constants.%N.1)]
// CHECK:STDOUT:   %N.patt.loc4_9.2: %i32 = symbolic_binding_pattern N, 0 [symbolic = %N.patt.loc4_9.2 (constants.%N.patt.1)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %A: type = class_type @A, @A(%N.loc4_9.2) [symbolic = %A (constants.%A.1)]
// CHECK:STDOUT:   %A.elem.loc5: type = unbound_element_type @A.%A (%A.1), %B [symbolic = %A.elem.loc5 (constants.%A.elem.1)]
// CHECK:STDOUT:   %Convert.bound.loc10_19.2: <bound method> = bound_method %N.loc4_9.2, constants.%Convert.13 [symbolic = %Convert.bound.loc10_19.2 (constants.%Convert.bound.1)]
// CHECK:STDOUT:   %Convert.specific_fn.loc10_19.2: <specific function> = specific_function %Convert.bound.loc10_19.2, @Convert.4(constants.%int_32) [symbolic = %Convert.specific_fn.loc10_19.2 (constants.%Convert.specific_fn.1)]
// CHECK:STDOUT:   %int.convert_checked.loc10_19.2: init Core.IntLiteral = call %Convert.specific_fn.loc10_19.2(%N.loc4_9.2) [symbolic = %int.convert_checked.loc10_19.2 (constants.%int.convert_checked)]
// CHECK:STDOUT:   %iN: type = int_type signed, %int.convert_checked.loc10_19.2 [symbolic = %iN (constants.%iN.2)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type @A.%iN (%iN.2) [symbolic = %require_complete (constants.%require_complete.7)]
// CHECK:STDOUT:   %A.elem.loc10: type = unbound_element_type @A.%A (%A.1), @A.%iN (%iN.2) [symbolic = %A.elem.loc10 (constants.%A.elem.2)]
// CHECK:STDOUT:   %struct_type.base.n: type = struct_type {.base: %B, .n: @A.%iN (%iN.2)} [symbolic = %struct_type.base.n (constants.%struct_type.base.n)]
// CHECK:STDOUT:   %complete_type.loc11_1.2: <witness> = complete_type_witness @A.%struct_type.base.n (%struct_type.base.n) [symbolic = %complete_type.loc11_1.2 (constants.%complete_type.3)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %B.ref: type = name_ref B, file.%B.decl [template = constants.%B]
// CHECK:STDOUT:     %.loc5: @A.%A.elem.loc5 (%A.elem.1) = base_decl %B.ref, element0 [template]
// CHECK:STDOUT:     %Core.ref: <namespace> = name_ref Core, imports.%Core [template = imports.%Core]
// CHECK:STDOUT:     %Int.ref: %Int.type = name_ref Int, imports.%import_ref.1 [template = constants.%Int]
// CHECK:STDOUT:     %N.ref: %i32 = name_ref N, %N.loc4_9.1 [symbolic = %N.loc4_9.2 (constants.%N.1)]
// CHECK:STDOUT:     %impl.elem0: %Convert.type.2 = interface_witness_access constants.%interface.9, element0 [template = constants.%Convert.13]
// CHECK:STDOUT:     %Convert.bound.loc10_19.1: <bound method> = bound_method %N.ref, %impl.elem0 [symbolic = %Convert.bound.loc10_19.2 (constants.%Convert.bound.1)]
// CHECK:STDOUT:     %Convert.specific_fn.loc10_19.1: <specific function> = specific_function %Convert.bound.loc10_19.1, @Convert.4(constants.%int_32) [symbolic = %Convert.specific_fn.loc10_19.2 (constants.%Convert.specific_fn.1)]
// CHECK:STDOUT:     %int.convert_checked.loc10_19.1: init Core.IntLiteral = call %Convert.specific_fn.loc10_19.1(%N.ref) [symbolic = %int.convert_checked.loc10_19.2 (constants.%int.convert_checked)]
// CHECK:STDOUT:     %.loc10_19.1: Core.IntLiteral = value_of_initializer %int.convert_checked.loc10_19.1 [symbolic = %int.convert_checked.loc10_19.2 (constants.%int.convert_checked)]
// CHECK:STDOUT:     %.loc10_19.2: Core.IntLiteral = converted %N.ref, %.loc10_19.1 [symbolic = %int.convert_checked.loc10_19.2 (constants.%int.convert_checked)]
// CHECK:STDOUT:     %int.make_type_signed.loc10: init type = call %Int.ref(%.loc10_19.2) [symbolic = %iN (constants.%iN.2)]
// CHECK:STDOUT:     %.loc10_20.1: type = value_of_initializer %int.make_type_signed.loc10 [symbolic = %iN (constants.%iN.2)]
// CHECK:STDOUT:     %.loc10_20.2: type = converted %int.make_type_signed.loc10, %.loc10_20.1 [symbolic = %iN (constants.%iN.2)]
// CHECK:STDOUT:     %.loc10_8: @A.%A.elem.loc10 (%A.elem.2) = field_decl n, element1 [template]
// CHECK:STDOUT:     %complete_type.loc11_1.1: <witness> = complete_type_witness %struct_type.base.n [symbolic = %complete_type.loc11_1.2 (constants.%complete_type.3)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%A.1
// CHECK:STDOUT:     .base = %.loc5
// CHECK:STDOUT:     .n = %.loc10_8
// CHECK:STDOUT:     extend %B.ref
// CHECK:STDOUT:     complete_type_witness = %complete_type.loc11_1.1
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%a.param_patt: %ptr.2) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %B.ref: type = name_ref B, file.%B.decl [template = constants.%B]
// CHECK:STDOUT:   %ptr.loc17: type = ptr_type %B [template = constants.%ptr.3]
// CHECK:STDOUT:   %a.ref: %ptr.2 = name_ref a, %a
// CHECK:STDOUT:   %.loc17_16.1: ref %A.2 = deref %a.ref
// CHECK:STDOUT:   %.loc17_16.2: ref %B = class_element_access %.loc17_16.1, element0
// CHECK:STDOUT:   %addr: %ptr.3 = addr_of %.loc17_16.2
// CHECK:STDOUT:   %.loc17_16.3: %ptr.3 = converted %a.ref, %addr
// CHECK:STDOUT:   %b: %ptr.3 = bind_name b, %.loc17_16.3
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A(constants.%N.1) {
// CHECK:STDOUT:   %N.loc4_9.2 => constants.%N.1
// CHECK:STDOUT:   %N.patt.loc4_9.2 => constants.%N.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A(%N.loc4_9.2) {
// CHECK:STDOUT:   %N.loc4_9.2 => constants.%N.1
// CHECK:STDOUT:   %N.patt.loc4_9.2 => constants.%N.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A(constants.%int_0.2) {
// CHECK:STDOUT:   %N.loc4_9.2 => constants.%int_0.2
// CHECK:STDOUT:   %N.patt.loc4_9.2 => constants.%int_0.2
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %A => constants.%A.2
// CHECK:STDOUT:   %A.elem.loc5 => constants.%A.elem.3
// CHECK:STDOUT:   %Convert.bound.loc10_19.2 => constants.%Convert.bound.3
// CHECK:STDOUT:   %Convert.specific_fn.loc10_19.2 => constants.%Convert.specific_fn.3
// CHECK:STDOUT:   %int.convert_checked.loc10_19.2 => constants.%int_0.1
// CHECK:STDOUT:   %iN => <error>
// CHECK:STDOUT:   %require_complete => <error>
// CHECK:STDOUT:   %A.elem.loc10 => <error>
// CHECK:STDOUT:   %struct_type.base.n => <error>
// CHECK:STDOUT:   %complete_type.loc11_1.2 => <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
