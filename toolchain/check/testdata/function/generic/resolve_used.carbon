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

// --- fail_todo_call_monomorphization_error.carbon

library "[[@TEST_NAME]]";

fn ErrorIfNIsZero(N:! Core.IntLiteral()) {
  // Check that we resolve the definition of a used specific function by
  // ensuring we produce an error when doing so. Notionally this error is
  // produced as a result of instantiating the `Core.Int` template, although
  // that's not how we currently model `Core.Int`.
  // CHECK:STDERR: min_prelude/parts/int.carbon:11:9: error: integer type width of 0 is not positive [IntWidthNotPositive]
  // CHECK:STDERR:   adapt MakeInt(N);
  // CHECK:STDERR:         ^~~~~~~~~~
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
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = bind_symbolic_name .Self [symbolic_self]
// CHECK:STDOUT:   %IntLiteral.type: type = fn_type @IntLiteral [concrete]
// CHECK:STDOUT:   %IntLiteral: %IntLiteral.type = struct_value () [concrete]
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.dc0: type = pattern_type Core.IntLiteral [concrete]
// CHECK:STDOUT:   %ErrorIfNIsZero.type: type = fn_type @ErrorIfNIsZero [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %ErrorIfNIsZero: %ErrorIfNIsZero.type = struct_value () [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %Int: type = class_type @Int, @Int(%N) [symbolic]
// CHECK:STDOUT:   %require_complete.b4f: <witness> = require_complete_type %Int [symbolic]
// CHECK:STDOUT:   %pattern_type.896: type = pattern_type %Int [symbolic]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %Destroy.Op.type: type = fn_type @Destroy.Op [concrete]
// CHECK:STDOUT:   %type_where: type = facet_type <type where .Self impls <CanAggregateDestroy>> [concrete]
// CHECK:STDOUT:   %AggregateT: %type_where = bind_symbolic_name AggregateT, 0 [symbolic]
// CHECK:STDOUT:   %AggregateT.as_type.as.Destroy.impl.Op.type.190: type = fn_type @AggregateT.as_type.as.Destroy.impl.Op, @AggregateT.as_type.as.Destroy.impl(%AggregateT) [symbolic]
// CHECK:STDOUT:   %AggregateT.as_type.as.Destroy.impl.Op.8a0: %AggregateT.as_type.as.Destroy.impl.Op.type.190 = struct_value () [symbolic]
// CHECK:STDOUT:   %facet_value.818: %type_where = facet_value %Int, () [symbolic]
// CHECK:STDOUT:   %ptr.784: type = ptr_type %Int [symbolic]
// CHECK:STDOUT:   %require_complete.0f5: <witness> = require_complete_type %ptr.784 [symbolic]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.34a: <witness> = lookup_impl_witness %Int, @Destroy [symbolic]
// CHECK:STDOUT:   %Destroy.facet.267: %Destroy.type = facet_value %Int, (%Destroy.lookup_impl_witness.34a) [symbolic]
// CHECK:STDOUT:   %.358: type = fn_type_with_self_type %Destroy.Op.type, %Destroy.facet.267 [symbolic]
// CHECK:STDOUT:   %impl.elem0.999: %.358 = impl_witness_access %Destroy.lookup_impl_witness.34a, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.5f7: <specific function> = specific_impl_function %impl.elem0.999, @Destroy.Op(%Destroy.facet.267) [symbolic]
// CHECK:STDOUT:   %CallNegative.type: type = fn_type @CallNegative [concrete]
// CHECK:STDOUT:   %CallNegative: %CallNegative.type = struct_value () [concrete]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %ErrorIfNIsZero.specific_fn: <specific function> = specific_function %ErrorIfNIsZero, @ErrorIfNIsZero(%int_0) [concrete]
// CHECK:STDOUT:   %i0: type = class_type @Int, @Int(%int_0) [concrete]
// CHECK:STDOUT:   %complete_type.d94: <witness> = complete_type_witness <error> [concrete]
// CHECK:STDOUT:   %pattern_type.47b: type = pattern_type %i0 [concrete]
// CHECK:STDOUT:   %facet_value.5b4: %type_where = facet_value %i0, () [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness.20d: <witness> = impl_witness imports.%Destroy.impl_witness_table.2d3, @AggregateT.as_type.as.Destroy.impl(%facet_value.5b4) [concrete]
// CHECK:STDOUT:   %AggregateT.as_type.as.Destroy.impl.Op.type.59e: type = fn_type @AggregateT.as_type.as.Destroy.impl.Op, @AggregateT.as_type.as.Destroy.impl(%facet_value.5b4) [concrete]
// CHECK:STDOUT:   %AggregateT.as_type.as.Destroy.impl.Op.701: %AggregateT.as_type.as.Destroy.impl.Op.type.59e = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.3f1: type = ptr_type %i0 [concrete]
// CHECK:STDOUT:   %complete_type.588: <witness> = complete_type_witness %ptr.3f1 [concrete]
// CHECK:STDOUT:   %Destroy.facet.168: %Destroy.type = facet_value %i0, (%Destroy.impl_witness.20d) [concrete]
// CHECK:STDOUT:   %.6d5: type = fn_type_with_self_type %Destroy.Op.type, %Destroy.facet.168 [concrete]
// CHECK:STDOUT:   %AggregateT.as_type.as.Destroy.impl.Op.specific_fn: <specific function> = specific_function %AggregateT.as_type.as.Destroy.impl.Op.701, @AggregateT.as_type.as.Destroy.impl.Op(%facet_value.5b4) [concrete]
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
// CHECK:STDOUT:   %Core.import_ref.d51: @AggregateT.as_type.as.Destroy.impl.%AggregateT.as_type.as.Destroy.impl.Op.type (%AggregateT.as_type.as.Destroy.impl.Op.type.190) = import_ref Core//prelude/parts/destroy, loc{{\d+_\d+}}, loaded [symbolic = @AggregateT.as_type.as.Destroy.impl.%AggregateT.as_type.as.Destroy.impl.Op (constants.%AggregateT.as_type.as.Destroy.impl.Op.8a0)]
// CHECK:STDOUT:   %Destroy.impl_witness_table.2d3 = impl_witness_table (%Core.import_ref.d51), @AggregateT.as_type.as.Destroy.impl [concrete]
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
// CHECK:STDOUT:       %.Self: %type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %Core.ref.loc4: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:       %IntLiteral.ref: %IntLiteral.type = name_ref IntLiteral, imports.%Core.IntLiteral [concrete = constants.%IntLiteral]
// CHECK:STDOUT:       %IntLiteral.call: init type = call %IntLiteral.ref() [concrete = Core.IntLiteral]
// CHECK:STDOUT:       %.loc4_39.2: type = value_of_initializer %IntLiteral.call [concrete = Core.IntLiteral]
// CHECK:STDOUT:       %.loc4_39.3: type = converted %IntLiteral.call, %.loc4_39.2 [concrete = Core.IntLiteral]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %N.loc4_19.2: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N.loc4_19.1 (constants.%N)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %CallNegative.decl: %CallNegative.type = fn_decl @CallNegative [concrete = constants.%CallNegative] {} {}
// CHECK:STDOUT:   %facet_value: %type_where = facet_value constants.%i0, () [concrete = constants.%facet_value.5b4]
// CHECK:STDOUT:   %.loc15: %type_where = converted constants.%i0, %facet_value [concrete = constants.%facet_value.5b4]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @ErrorIfNIsZero(%N.loc4_19.2: Core.IntLiteral) {
// CHECK:STDOUT:   %N.loc4_19.1: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N.loc4_19.1 (constants.%N)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Int.loc15_20.2: type = class_type @Int, @Int(%N.loc4_19.1) [symbolic = %Int.loc15_20.2 (constants.%Int)]
// CHECK:STDOUT:   %require_complete.loc15_20: <witness> = require_complete_type %Int.loc15_20.2 [symbolic = %require_complete.loc15_20 (constants.%require_complete.b4f)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Int.loc15_20.2 [symbolic = %pattern_type (constants.%pattern_type.896)]
// CHECK:STDOUT:   %facet_value.loc15_3.2: %type_where = facet_value %Int.loc15_20.2, () [symbolic = %facet_value.loc15_3.2 (constants.%facet_value.818)]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness: <witness> = lookup_impl_witness %Int.loc15_20.2, @Destroy [symbolic = %Destroy.lookup_impl_witness (constants.%Destroy.lookup_impl_witness.34a)]
// CHECK:STDOUT:   %Destroy.facet: %Destroy.type = facet_value %Int.loc15_20.2, (%Destroy.lookup_impl_witness) [symbolic = %Destroy.facet (constants.%Destroy.facet.267)]
// CHECK:STDOUT:   %.loc15_3.3: type = fn_type_with_self_type constants.%Destroy.Op.type, %Destroy.facet [symbolic = %.loc15_3.3 (constants.%.358)]
// CHECK:STDOUT:   %impl.elem0.loc15_3.2: @ErrorIfNIsZero.%.loc15_3.3 (%.358) = impl_witness_access %Destroy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc15_3.2 (constants.%impl.elem0.999)]
// CHECK:STDOUT:   %specific_impl_fn.loc15_3.2: <specific function> = specific_impl_function %impl.elem0.loc15_3.2, @Destroy.Op(%Destroy.facet) [symbolic = %specific_impl_fn.loc15_3.2 (constants.%specific_impl_fn.5f7)]
// CHECK:STDOUT:   %ptr: type = ptr_type %Int.loc15_20.2 [symbolic = %ptr (constants.%ptr.784)]
// CHECK:STDOUT:   %require_complete.loc15_3: <witness> = require_complete_type %ptr [symbolic = %require_complete.loc15_3 (constants.%require_complete.0f5)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %v.patt: @ErrorIfNIsZero.%pattern_type (%pattern_type.896) = binding_pattern v [concrete]
// CHECK:STDOUT:       %v.var_patt: @ErrorIfNIsZero.%pattern_type (%pattern_type.896) = var_pattern %v.patt [concrete]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %v.var: ref @ErrorIfNIsZero.%Int.loc15_20.2 (%Int) = var %v.var_patt
// CHECK:STDOUT:     %.loc15_20: type = splice_block %Int.loc15_20.1 [symbolic = %Int.loc15_20.2 (constants.%Int)] {
// CHECK:STDOUT:       %Core.ref.loc15: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:       %Int.ref: %Int.type = name_ref Int, imports.%Core.Int [concrete = constants.%Int.generic]
// CHECK:STDOUT:       %N.ref: Core.IntLiteral = name_ref N, %N.loc4_19.2 [symbolic = %N.loc4_19.1 (constants.%N)]
// CHECK:STDOUT:       %Int.loc15_20.1: type = class_type @Int, @Int(constants.%N) [symbolic = %Int.loc15_20.2 (constants.%Int)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %v: ref @ErrorIfNIsZero.%Int.loc15_20.2 (%Int) = bind_name v, %v.var
// CHECK:STDOUT:     %facet_value.loc15_3.1: %type_where = facet_value constants.%Int, () [symbolic = %facet_value.loc15_3.2 (constants.%facet_value.818)]
// CHECK:STDOUT:     %.loc15_3.1: %type_where = converted constants.%Int, %facet_value.loc15_3.1 [symbolic = %facet_value.loc15_3.2 (constants.%facet_value.818)]
// CHECK:STDOUT:     %impl.elem0.loc15_3.1: @ErrorIfNIsZero.%.loc15_3.3 (%.358) = impl_witness_access constants.%Destroy.lookup_impl_witness.34a, element0 [symbolic = %impl.elem0.loc15_3.2 (constants.%impl.elem0.999)]
// CHECK:STDOUT:     %bound_method.loc15_3.1: <bound method> = bound_method %v.var, %impl.elem0.loc15_3.1
// CHECK:STDOUT:     %specific_impl_fn.loc15_3.1: <specific function> = specific_impl_function %impl.elem0.loc15_3.1, @Destroy.Op(constants.%Destroy.facet.267) [symbolic = %specific_impl_fn.loc15_3.2 (constants.%specific_impl_fn.5f7)]
// CHECK:STDOUT:     %bound_method.loc15_3.2: <bound method> = bound_method %v.var, %specific_impl_fn.loc15_3.1
// CHECK:STDOUT:     %addr: @ErrorIfNIsZero.%ptr (%ptr.784) = addr_of %v.var
// CHECK:STDOUT:     %.loc15_3.2: init %empty_tuple.type = call %bound_method.loc15_3.2(%addr)
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
// CHECK:STDOUT:   %N.loc4_19.1 => constants.%N
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ErrorIfNIsZero(constants.%int_0) {
// CHECK:STDOUT:   %N.loc4_19.1 => constants.%int_0
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Int.loc15_20.2 => constants.%i0
// CHECK:STDOUT:   %require_complete.loc15_20 => constants.%complete_type.d94
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.47b
// CHECK:STDOUT:   %facet_value.loc15_3.2 => constants.%facet_value.5b4
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%Destroy.impl_witness.20d
// CHECK:STDOUT:   %Destroy.facet => constants.%Destroy.facet.168
// CHECK:STDOUT:   %.loc15_3.3 => constants.%.6d5
// CHECK:STDOUT:   %impl.elem0.loc15_3.2 => constants.%AggregateT.as_type.as.Destroy.impl.Op.701
// CHECK:STDOUT:   %specific_impl_fn.loc15_3.2 => constants.%AggregateT.as_type.as.Destroy.impl.Op.specific_fn
// CHECK:STDOUT:   %ptr => constants.%ptr.3f1
// CHECK:STDOUT:   %require_complete.loc15_3 => constants.%complete_type.588
// CHECK:STDOUT: }
// CHECK:STDOUT:
