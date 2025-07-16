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
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/function/generic/resolve_used.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/function/generic/resolve_used.carbon
// CHECK:STDERR: min_prelude/parts/int.carbon:10:9: error: integer type width of 0 is not positive [IntWidthNotPositive]
// CHECK:STDERR:   adapt MakeInt(N);
// CHECK:STDERR:         ^~~~~~~~~~

// --- fail_todo_call_monomorphization_error.carbon

library "[[@TEST_NAME]]";

fn ErrorIfNIsZero(N:! Core.IntLiteral()) {
  // Check that we resolve the definition of a used specific function by
  // ensuring we produce an error when doing so. Notionally this error is
  // produced as a result of instantiating the `Core.Int` template, although
  // that's not how we currently model `Core.Int`.
  // CHECK:STDERR: fail_todo_call_monomorphization_error.carbon:[[@LINE+3]]:10: note: in `i0` used here [ResolvingSpecificHere]
  // CHECK:STDERR:   var v: Core.Int(N);
  // CHECK:STDERR:          ^~~~~~~~~~~
  var v: Core.Int(N);
}

fn CallNegative() {
  // CHECK:STDERR: fail_todo_call_monomorphization_error.carbon:[[@LINE+4]]:3: note: in `ErrorIfNIsZero(0)` used here [ResolvingSpecificHere]
  // CHECK:STDERR:   ErrorIfNIsZero(0);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~
  // CHECK:STDERR:
  ErrorIfNIsZero(0);
}

// CHECK:STDOUT: --- fail_todo_call_monomorphization_error.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %IntLiteral.type: type = fn_type @IntLiteral [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %IntLiteral: %IntLiteral.type = struct_value () [concrete]
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.dc0: type = pattern_type Core.IntLiteral [concrete]
// CHECK:STDOUT:   %ErrorIfNIsZero.type: type = fn_type @ErrorIfNIsZero [concrete]
// CHECK:STDOUT:   %ErrorIfNIsZero: %ErrorIfNIsZero.type = struct_value () [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %Int.49d0e6.1: type = class_type @Int, @Int(%N) [symbolic]
// CHECK:STDOUT:   %require_complete.b4f426.1: <witness> = require_complete_type %Int.49d0e6.1 [symbolic]
// CHECK:STDOUT:   %pattern_type.8963eb.1: type = pattern_type %Int.49d0e6.1 [symbolic]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %Op.type.bae: type = fn_type @Op.1 [concrete]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %Op.type.bc9: type = fn_type @Op.3, @Destroy.impl(%T) [symbolic]
// CHECK:STDOUT:   %Op.46f: %Op.type.bc9 = struct_value () [symbolic]
// CHECK:STDOUT:   %ptr.784: type = ptr_type %Int.49d0e6.1 [symbolic]
// CHECK:STDOUT:   %require_complete.0f5: <witness> = require_complete_type %ptr.784 [symbolic]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness: <witness> = lookup_impl_witness %Int.49d0e6.1, @Destroy [symbolic]
// CHECK:STDOUT:   %Destroy.facet.13d: %Destroy.type = facet_value %Int.49d0e6.1, (%Destroy.lookup_impl_witness) [symbolic]
// CHECK:STDOUT:   %.bbc: type = fn_type_with_self_type %Op.type.bae, %Destroy.facet.13d [symbolic]
// CHECK:STDOUT:   %impl.elem0.fe8: %.bbc = impl_witness_access %Destroy.lookup_impl_witness, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn: <specific function> = specific_impl_function %impl.elem0.fe8, @Op.1(%Destroy.facet.13d) [symbolic]
// CHECK:STDOUT:   %CallNegative.type: type = fn_type @CallNegative [concrete]
// CHECK:STDOUT:   %CallNegative: %CallNegative.type = struct_value () [concrete]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %ErrorIfNIsZero.specific_fn: <specific function> = specific_function %ErrorIfNIsZero, @ErrorIfNIsZero(%int_0) [concrete]
// CHECK:STDOUT:   %i0: type = class_type @Int, @Int(%int_0) [concrete]
// CHECK:STDOUT:   %complete_type.d94: <witness> = complete_type_witness <error> [concrete]
// CHECK:STDOUT:   %pattern_type.47b: type = pattern_type %i0 [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness.404: <witness> = impl_witness imports.%Destroy.impl_witness_table, @Destroy.impl(%i0) [concrete]
// CHECK:STDOUT:   %Op.type.1c2: type = fn_type @Op.3, @Destroy.impl(%i0) [concrete]
// CHECK:STDOUT:   %Op.912: %Op.type.1c2 = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.3f1: type = ptr_type %i0 [concrete]
// CHECK:STDOUT:   %complete_type.588: <witness> = complete_type_witness %ptr.3f1 [concrete]
// CHECK:STDOUT:   %Destroy.facet.973: %Destroy.type = facet_value %i0, (%Destroy.impl_witness.404) [concrete]
// CHECK:STDOUT:   %.9ab: type = fn_type_with_self_type %Op.type.bae, %Destroy.facet.973 [concrete]
// CHECK:STDOUT:   %Op.specific_fn: <specific function> = specific_function %Op.912, @Op.3(%i0) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .IntLiteral = %Core.IntLiteral
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.IntLiteral: %IntLiteral.type = import_ref Core//prelude/parts/int_literal, IntLiteral, loaded [concrete = constants.%IntLiteral]
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/parts/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Core.import_ref.0b9: @Destroy.impl.%Op.type (%Op.type.bc9) = import_ref Core//prelude/parts/destroy, loc8_29, loaded [symbolic = @Destroy.impl.%Op (constants.%Op.46f)]
// CHECK:STDOUT:   %Destroy.impl_witness_table = impl_witness_table (%Core.import_ref.0b9), @Destroy.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .ErrorIfNIsZero = %ErrorIfNIsZero.decl
// CHECK:STDOUT:     .CallNegative = %CallNegative.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %ErrorIfNIsZero.decl: %ErrorIfNIsZero.type = fn_decl @ErrorIfNIsZero [concrete = constants.%ErrorIfNIsZero] {
// CHECK:STDOUT:     %N.patt: %pattern_type.dc0 = symbolic_binding_pattern N, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc4_39.1: type = splice_block %.loc4_39.3 [concrete = Core.IntLiteral] {
// CHECK:STDOUT:       %Core.ref.loc4: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:       %IntLiteral.ref: %IntLiteral.type = name_ref IntLiteral, imports.%Core.IntLiteral [concrete = constants.%IntLiteral]
// CHECK:STDOUT:       %int_literal.make_type: init type = call %IntLiteral.ref() [concrete = Core.IntLiteral]
// CHECK:STDOUT:       %.loc4_39.2: type = value_of_initializer %int_literal.make_type [concrete = Core.IntLiteral]
// CHECK:STDOUT:       %.loc4_39.3: type = converted %int_literal.make_type, %.loc4_39.2 [concrete = Core.IntLiteral]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %N.loc4_19.1: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N.loc4_19.2 (constants.%N)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %CallNegative.decl: %CallNegative.type = fn_decl @CallNegative [concrete = constants.%CallNegative] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @ErrorIfNIsZero(%N.loc4_19.1: Core.IntLiteral) {
// CHECK:STDOUT:   %N.loc4_19.2: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N.loc4_19.2 (constants.%N)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Int.loc12_20.2: type = class_type @Int, @Int(%N.loc4_19.2) [symbolic = %Int.loc12_20.2 (constants.%Int.49d0e6.1)]
// CHECK:STDOUT:   %require_complete.loc12_20: <witness> = require_complete_type %Int.loc12_20.2 [symbolic = %require_complete.loc12_20 (constants.%require_complete.b4f426.1)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Int.loc12_20.2 [symbolic = %pattern_type (constants.%pattern_type.8963eb.1)]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness: <witness> = lookup_impl_witness %Int.loc12_20.2, @Destroy [symbolic = %Destroy.lookup_impl_witness (constants.%Destroy.lookup_impl_witness)]
// CHECK:STDOUT:   %Destroy.facet: %Destroy.type = facet_value %Int.loc12_20.2, (%Destroy.lookup_impl_witness) [symbolic = %Destroy.facet (constants.%Destroy.facet.13d)]
// CHECK:STDOUT:   %.loc12_3.2: type = fn_type_with_self_type constants.%Op.type.bae, %Destroy.facet [symbolic = %.loc12_3.2 (constants.%.bbc)]
// CHECK:STDOUT:   %impl.elem0.loc12_3.2: @ErrorIfNIsZero.%.loc12_3.2 (%.bbc) = impl_witness_access %Destroy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc12_3.2 (constants.%impl.elem0.fe8)]
// CHECK:STDOUT:   %specific_impl_fn.loc12_3.2: <specific function> = specific_impl_function %impl.elem0.loc12_3.2, @Op.1(%Destroy.facet) [symbolic = %specific_impl_fn.loc12_3.2 (constants.%specific_impl_fn)]
// CHECK:STDOUT:   %ptr: type = ptr_type %Int.loc12_20.2 [symbolic = %ptr (constants.%ptr.784)]
// CHECK:STDOUT:   %require_complete.loc12_3: <witness> = require_complete_type %ptr [symbolic = %require_complete.loc12_3 (constants.%require_complete.0f5)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %v.patt: @ErrorIfNIsZero.%pattern_type (%pattern_type.8963eb.1) = binding_pattern v [concrete]
// CHECK:STDOUT:       %v.var_patt: @ErrorIfNIsZero.%pattern_type (%pattern_type.8963eb.1) = var_pattern %v.patt [concrete]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %v.var: ref @ErrorIfNIsZero.%Int.loc12_20.2 (%Int.49d0e6.1) = var %v.var_patt
// CHECK:STDOUT:     %.loc12_20: type = splice_block %Int.loc12_20.1 [symbolic = %Int.loc12_20.2 (constants.%Int.49d0e6.1)] {
// CHECK:STDOUT:       %Core.ref.loc12: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:       %Int.ref: %Int.type = name_ref Int, imports.%Core.Int [concrete = constants.%Int.generic]
// CHECK:STDOUT:       %N.ref: Core.IntLiteral = name_ref N, %N.loc4_19.1 [symbolic = %N.loc4_19.2 (constants.%N)]
// CHECK:STDOUT:       %Int.loc12_20.1: type = class_type @Int, @Int(constants.%N) [symbolic = %Int.loc12_20.2 (constants.%Int.49d0e6.1)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %v: ref @ErrorIfNIsZero.%Int.loc12_20.2 (%Int.49d0e6.1) = bind_name v, %v.var
// CHECK:STDOUT:     %impl.elem0.loc12_3.1: @ErrorIfNIsZero.%.loc12_3.2 (%.bbc) = impl_witness_access constants.%Destroy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc12_3.2 (constants.%impl.elem0.fe8)]
// CHECK:STDOUT:     %bound_method.loc12_3.1: <bound method> = bound_method %v.var, %impl.elem0.loc12_3.1
// CHECK:STDOUT:     %specific_impl_fn.loc12_3.1: <specific function> = specific_impl_function %impl.elem0.loc12_3.1, @Op.1(constants.%Destroy.facet.13d) [symbolic = %specific_impl_fn.loc12_3.2 (constants.%specific_impl_fn)]
// CHECK:STDOUT:     %bound_method.loc12_3.2: <bound method> = bound_method %v.var, %specific_impl_fn.loc12_3.1
// CHECK:STDOUT:     %addr: @ErrorIfNIsZero.%ptr (%ptr.784) = addr_of %v.var
// CHECK:STDOUT:     %.loc12_3.1: init %empty_tuple.type = call %bound_method.loc12_3.2(%addr)
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CallNegative() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %ErrorIfNIsZero.ref: %ErrorIfNIsZero.type = name_ref ErrorIfNIsZero, file.%ErrorIfNIsZero.decl [concrete = constants.%ErrorIfNIsZero]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT:   %ErrorIfNIsZero.specific_fn: <specific function> = specific_function %ErrorIfNIsZero.ref, @ErrorIfNIsZero(constants.%int_0) [concrete = constants.%ErrorIfNIsZero.specific_fn]
// CHECK:STDOUT:   %ErrorIfNIsZero.call: init %empty_tuple.type = call %ErrorIfNIsZero.specific_fn()
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ErrorIfNIsZero(constants.%N) {
// CHECK:STDOUT:   %N.loc4_19.2 => constants.%N
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ErrorIfNIsZero(constants.%int_0) {
// CHECK:STDOUT:   %N.loc4_19.2 => constants.%int_0
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Int.loc12_20.2 => constants.%i0
// CHECK:STDOUT:   %require_complete.loc12_20 => constants.%complete_type.d94
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.47b
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%Destroy.impl_witness.404
// CHECK:STDOUT:   %Destroy.facet => constants.%Destroy.facet.973
// CHECK:STDOUT:   %.loc12_3.2 => constants.%.9ab
// CHECK:STDOUT:   %impl.elem0.loc12_3.2 => constants.%Op.912
// CHECK:STDOUT:   %specific_impl_fn.loc12_3.2 => constants.%Op.specific_fn
// CHECK:STDOUT:   %ptr => constants.%ptr.3f1
// CHECK:STDOUT:   %require_complete.loc12_3 => constants.%complete_type.588
// CHECK:STDOUT: }
// CHECK:STDOUT:
