// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
// TODO: Add ranges and switch to "--dump-sem-ir-ranges=only".
// EXTRA-ARGS: --dump-sem-ir-ranges=if-present
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/generic/complete_in_conversion.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/generic/complete_in_conversion.carbon

// --- fail_derived_to_base.carbon

fn Int(N: Core.IntLiteral()) -> type = "int.make_type_signed";

base class B {}

class A(N:! i32) {
  extend base: B;

  var n: Int(N);
}

fn F(a: A(0)*) {
  // CHECK:STDERR: fail_derived_to_base.carbon:[[@LINE+7]]:22: error: unable to monomorphize specific `A(0)` [ResolvingSpecificHere]
  // CHECK:STDERR:   let unused b: B* = a;
  // CHECK:STDERR:                      ^
  // CHECK:STDERR: fail_derived_to_base.carbon:[[@LINE-7]]:10: note: integer type width of 0 is not positive [IntWidthNotPositive]
  // CHECK:STDERR:   var n: Int(N);
  // CHECK:STDERR:          ^~~~~~
  // CHECK:STDERR:
  let unused b: B* = a;
}

// CHECK:STDOUT: --- fail_derived_to_base.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %IntLiteral.type: type = fn_type @IntLiteral [concrete]
// CHECK:STDOUT:   %IntLiteral: %IntLiteral.type = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %.805: Core.Form = init_form type [concrete]
// CHECK:STDOUT:   %pattern_type.dc0: type = pattern_type Core.IntLiteral [concrete]
// CHECK:STDOUT:   %Int.type.d61: type = fn_type @Int.loc2 [concrete]
// CHECK:STDOUT:   %Int.69b: %Int.type.d61 = struct_value () [concrete]
// CHECK:STDOUT:   %B: type = class_type @B [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type.11a: type = generic_class_type @Int.1 [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type.11a = struct_value () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int.1, @Int.1(%int_32) [concrete]
// CHECK:STDOUT:   %pattern_type.c6b: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %N.afc: %i32 = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %A.type: type = generic_class_type @A [concrete]
// CHECK:STDOUT:   %A.generic: %A.type = struct_value () [concrete]
// CHECK:STDOUT:   %A.f6a: type = class_type @A, @A(%N.afc) [symbolic]
// CHECK:STDOUT:   %A.elem.4fe: type = unbound_element_type %A.f6a, %B [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.649: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.649 = struct_value () [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.766: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.ec3: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.766 = struct_value () [symbolic]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.type.73a: type = fn_type @Int.as.ImplicitAs.impl.Convert, @Int.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.735: %Int.as.ImplicitAs.impl.Convert.type.73a = struct_value () [concrete]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.bound.483: <bound method> = bound_method %N.afc, %Int.as.ImplicitAs.impl.Convert.735 [symbolic]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Int.as.ImplicitAs.impl.Convert.735, @Int.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.470: <bound method> = bound_method %N.afc, %Int.as.ImplicitAs.impl.Convert.specific_fn [symbolic]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.call: init Core.IntLiteral = call %bound_method.470(%N.afc) [symbolic]
// CHECK:STDOUT:   %iN.builtin.961: type = int_type signed, %Int.as.ImplicitAs.impl.Convert.call [symbolic]
// CHECK:STDOUT:   %require_complete.74e: <witness> = require_complete_type %iN.builtin.961 [symbolic]
// CHECK:STDOUT:   %A.elem.6b3: type = unbound_element_type %A.f6a, %iN.builtin.961 [symbolic]
// CHECK:STDOUT:   %struct_type.base.n: type = struct_type {.base: %B, .n: %iN.builtin.961} [symbolic]
// CHECK:STDOUT:   %complete_type.638: <witness> = complete_type_witness %struct_type.base.n [symbolic]
// CHECK:STDOUT:   %int_0.5c6: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.544: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.fb6: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.1a5, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.ef7: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.e6b: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.ef7 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.e78: %ImplicitAs.type.544 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.fb6) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.367: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%i32, %ImplicitAs.facet.e78) [concrete]
// CHECK:STDOUT:   %.205: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.367, %ImplicitAs.facet.e78 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_0.5c6, %Core.IntLiteral.as.ImplicitAs.impl.Convert.e6b [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.e6b, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.494: <bound method> = bound_method %int_0.5c6, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_0.155: %i32 = int_value 0 [concrete]
// CHECK:STDOUT:   %A.429: type = class_type @A, @A(%int_0.155) [concrete]
// CHECK:STDOUT:   %ptr.fa5: type = ptr_type %A.429 [concrete]
// CHECK:STDOUT:   %pattern_type.c61: type = pattern_type %ptr.fa5 [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.e4c: type = ptr_type %B [concrete]
// CHECK:STDOUT:   %pattern_type.383: type = pattern_type %ptr.e4c [concrete]
// CHECK:STDOUT:   %A.elem.02e: type = unbound_element_type %A.429, %B [concrete]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.bound.203: <bound method> = bound_method %int_0.155, %Int.as.ImplicitAs.impl.Convert.735 [concrete]
// CHECK:STDOUT:   %bound_method.8ab: <bound method> = bound_method %int_0.155, %Int.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .IntLiteral = %Core.IntLiteral
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.IntLiteral: %IntLiteral.type = import_ref Core//prelude/parts/int_literal, IntLiteral, loaded [concrete = constants.%IntLiteral]
// CHECK:STDOUT:   %Core.Int: %Int.type.11a = import_ref Core//prelude/parts/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.649 = import_ref Core//prelude/parts/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:   %Core.import_ref.dd3: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.766) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.ec3)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.1a5 = impl_witness_table (%Core.import_ref.dd3), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Int = %Int.decl
// CHECK:STDOUT:     .B = %B.decl
// CHECK:STDOUT:     .A = %A.decl
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Int.decl: %Int.type.d61 = fn_decl @Int.loc2 [concrete = constants.%Int.69b] {
// CHECK:STDOUT:     %N.param_patt: %pattern_type.dc0 = value_param_pattern [concrete]
// CHECK:STDOUT:     %N.patt: %pattern_type.dc0 = at_binding_pattern N, %N.param_patt [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.98f = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.98f = return_slot_pattern %return.param_patt, %.loc2_33.1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc2_33.1: type = type_literal type [concrete = type]
// CHECK:STDOUT:     %.loc2_33.2: Core.Form = init_form %.loc2_33.1 [concrete = constants.%.805]
// CHECK:STDOUT:     %N.param: Core.IntLiteral = value_param call_param0
// CHECK:STDOUT:     %.loc2_27.1: type = splice_block %.loc2_27.3 [concrete = Core.IntLiteral] {
// CHECK:STDOUT:       %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:       %IntLiteral.ref: %IntLiteral.type = name_ref IntLiteral, imports.%Core.IntLiteral [concrete = constants.%IntLiteral]
// CHECK:STDOUT:       %IntLiteral.call: init type = call %IntLiteral.ref() [concrete = Core.IntLiteral]
// CHECK:STDOUT:       %.loc2_27.2: type = value_of_initializer %IntLiteral.call [concrete = Core.IntLiteral]
// CHECK:STDOUT:       %.loc2_27.3: type = converted %IntLiteral.call, %.loc2_27.2 [concrete = Core.IntLiteral]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %N: Core.IntLiteral = value_binding N, %N.param
// CHECK:STDOUT:     %return.param: ref type = out_param call_param1
// CHECK:STDOUT:     %return: ref type = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %B.decl: type = class_decl @B [concrete = constants.%B] {} {}
// CHECK:STDOUT:   %A.decl: %A.type = class_decl @A [concrete = constants.%A.generic] {
// CHECK:STDOUT:     %N.patt: %pattern_type.c6b = symbolic_binding_pattern N, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc6: type = splice_block %i32 [concrete = constants.%i32] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %N.loc6_10.2: %i32 = symbolic_binding N, 0 [symbolic = %N.loc6_10.1 (constants.%N.afc)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %a.param_patt: %pattern_type.c61 = value_param_pattern [concrete]
// CHECK:STDOUT:     %a.patt: %pattern_type.c61 = at_binding_pattern a, %a.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %a.param: %ptr.fa5 = value_param call_param0
// CHECK:STDOUT:     %.loc12_13: type = splice_block %ptr.loc12 [concrete = constants.%ptr.fa5] {
// CHECK:STDOUT:       %A.ref: %A.type = name_ref A, file.%A.decl [concrete = constants.%A.generic]
// CHECK:STDOUT:       %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0.5c6]
// CHECK:STDOUT:       %impl.elem0: %.205 = impl_witness_access constants.%ImplicitAs.impl_witness.fb6, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.e6b]
// CHECK:STDOUT:       %bound_method.loc12_12.1: <bound method> = bound_method %int_0, %impl.elem0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:       %specific_fn: <specific function> = specific_function %impl.elem0, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:       %bound_method.loc12_12.2: <bound method> = bound_method %int_0, %specific_fn [concrete = constants.%bound_method.494]
// CHECK:STDOUT:       %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %i32 = call %bound_method.loc12_12.2(%int_0) [concrete = constants.%int_0.155]
// CHECK:STDOUT:       %.loc12_12.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_0.155]
// CHECK:STDOUT:       %.loc12_12.2: %i32 = converted %int_0, %.loc12_12.1 [concrete = constants.%int_0.155]
// CHECK:STDOUT:       %A: type = class_type @A, @A(constants.%int_0.155) [concrete = constants.%A.429]
// CHECK:STDOUT:       %ptr.loc12: type = ptr_type %A [concrete = constants.%ptr.fa5]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a: %ptr.fa5 = value_binding a, %a.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @B {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%B
// CHECK:STDOUT:   .Int = <poisoned>
// CHECK:STDOUT:   .N = <poisoned>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @A(%N.loc6_10.2: %i32) {
// CHECK:STDOUT:   %N.loc6_10.1: %i32 = symbolic_binding N, 0 [symbolic = %N.loc6_10.1 (constants.%N.afc)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %A: type = class_type @A, @A(%N.loc6_10.1) [symbolic = %A (constants.%A.f6a)]
// CHECK:STDOUT:   %A.elem.loc7: type = unbound_element_type %A, constants.%B [symbolic = %A.elem.loc7 (constants.%A.elem.4fe)]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %N.loc6_10.1, constants.%Int.as.ImplicitAs.impl.Convert.735 [symbolic = %Int.as.ImplicitAs.impl.Convert.bound (constants.%Int.as.ImplicitAs.impl.Convert.bound.483)]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %N.loc6_10.1, constants.%Int.as.ImplicitAs.impl.Convert.specific_fn [symbolic = %bound_method (constants.%bound_method.470)]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.call: init Core.IntLiteral = call %bound_method(%N.loc6_10.1) [symbolic = %Int.as.ImplicitAs.impl.Convert.call (constants.%Int.as.ImplicitAs.impl.Convert.call)]
// CHECK:STDOUT:   %iN.builtin: type = int_type signed, %Int.as.ImplicitAs.impl.Convert.call [symbolic = %iN.builtin (constants.%iN.builtin.961)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %iN.builtin [symbolic = %require_complete (constants.%require_complete.74e)]
// CHECK:STDOUT:   %A.elem.loc9: type = unbound_element_type %A, %iN.builtin [symbolic = %A.elem.loc9 (constants.%A.elem.6b3)]
// CHECK:STDOUT:   %struct_type.base.n: type = struct_type {.base: %B, .n: @A.%iN.builtin (%iN.builtin.961)} [symbolic = %struct_type.base.n (constants.%struct_type.base.n)]
// CHECK:STDOUT:   %complete_type.loc10_1.2: <witness> = complete_type_witness %struct_type.base.n [symbolic = %complete_type.loc10_1.2 (constants.%complete_type.638)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %B.ref: type = name_ref B, file.%B.decl [concrete = constants.%B]
// CHECK:STDOUT:     %.loc7: @A.%A.elem.loc7 (%A.elem.4fe) = base_decl %B.ref, element0 [concrete]
// CHECK:STDOUT:     %.loc9: @A.%A.elem.loc9 (%A.elem.6b3) = field_decl n, element1 [concrete]
// CHECK:STDOUT:     %complete_type.loc10_1.1: <witness> = complete_type_witness constants.%struct_type.base.n [symbolic = %complete_type.loc10_1.2 (constants.%complete_type.638)]
// CHECK:STDOUT:     complete_type_witness = %complete_type.loc10_1.1
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%A.f6a
// CHECK:STDOUT:     .B = <poisoned>
// CHECK:STDOUT:     .base = %.loc7
// CHECK:STDOUT:     .Int = <poisoned>
// CHECK:STDOUT:     .N = <poisoned>
// CHECK:STDOUT:     .n = %.loc9
// CHECK:STDOUT:     extend %B.ref
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int.loc2(%N.param: Core.IntLiteral) -> out %return.param: type = "int.make_type_signed";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%a.param: %ptr.fa5) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %pattern_type.383 = value_binding_pattern b [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.ref: %ptr.fa5 = name_ref a, %a
// CHECK:STDOUT:   %.loc20_18: type = splice_block %ptr.loc20 [concrete = constants.%ptr.e4c] {
// CHECK:STDOUT:     %B.ref: type = name_ref B, file.%B.decl [concrete = constants.%B]
// CHECK:STDOUT:     %ptr.loc20: type = ptr_type %B.ref [concrete = constants.%ptr.e4c]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc20_22.1: ref %A.429 = deref %a.ref
// CHECK:STDOUT:   %.loc20_22.2: ref %B = class_element_access %.loc20_22.1, element0
// CHECK:STDOUT:   %addr: %ptr.e4c = addr_of %.loc20_22.2
// CHECK:STDOUT:   %.loc20_22.3: %ptr.e4c = converted %a.ref, %addr
// CHECK:STDOUT:   %b: %ptr.e4c = value_binding b, %.loc20_22.3
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A(constants.%N.afc) {
// CHECK:STDOUT:   %N.loc6_10.1 => constants.%N.afc
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A(constants.%int_0.155) {
// CHECK:STDOUT:   %N.loc6_10.1 => constants.%int_0.155
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %A => constants.%A.429
// CHECK:STDOUT:   %A.elem.loc7 => constants.%A.elem.02e
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.bound => constants.%Int.as.ImplicitAs.impl.Convert.bound.203
// CHECK:STDOUT:   %bound_method => constants.%bound_method.8ab
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.call => constants.%int_0.5c6
// CHECK:STDOUT:   %iN.builtin => <error>
// CHECK:STDOUT:   %require_complete => <error>
// CHECK:STDOUT:   %A.elem.loc9 => <error>
// CHECK:STDOUT:   %struct_type.base.n => <error>
// CHECK:STDOUT:   %complete_type.loc10_1.2 => <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
