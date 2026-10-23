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

fn ErrorIfNIsZero(generic N: Core.IntLiteral) {
  // Check that we resolve the definition of a used specific function by
  // ensuring we produce an error when doing so. Notionally this error is
  // produced as a result of instantiating the `Core.Int` template, although
  // that's not how we currently model `Core.Int`.
  var unused v: Core.Int(N);
}

fn CallNegative() {
  // CHECK:STDERR: fail_todo_call_monomorphization_error.carbon:[[@LINE+4]]:3-16: error: unable to monomorphize specific `ErrorIfNIsZero(0)` [ResolvingSpecificHere]
  // CHECK:STDERR: fail_todo_call_monomorphization_error.carbon:[[@LINE-5]]:17-27: note: `Core.Int(N)` evaluates to incomplete type `i0` [IncompleteTypeInMonomorphization]
  // CHECK:STDERR: min_prelude/parts/int.carbon:11:9-18: note: integer type width of 0 is not positive [IntWidthNotPositive]
  // CHECK:STDERR:
  ErrorIfNIsZero(0);
}


// CHECK:STDOUT: --- fail_todo_call_monomorphization_error.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self.frozen: type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type.dc0: type = pattern_type Core.IntLiteral [concrete]
// CHECK:STDOUT:   %N.patt: %pattern_type.dc0 = symbolic_binding_pattern N, 0 [symbolic]
// CHECK:STDOUT:   %N: Core.IntLiteral = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %ErrorIfNIsZero.type: type = fn_type @ErrorIfNIsZero [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %ErrorIfNIsZero: %ErrorIfNIsZero.type = struct_value () [concrete]
// CHECK:STDOUT:   %Int.type: type = parameterized_class_name_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %Int: type = class_type @Int, @Int(%N) [symbolic]
// CHECK:STDOUT:   %require_complete.d0a: <witness> = require_complete_type %Int [symbolic]
// CHECK:STDOUT:   %pattern_type.551: type = pattern_type %Int [symbolic]
// CHECK:STDOUT:   %v.patt.fc3: %pattern_type.551 = ref_binding_pattern v [symbolic]
// CHECK:STDOUT:   %v.var_patt.c99: %pattern_type.551 = var_pattern %v.patt.fc3 [symbolic]
// CHECK:STDOUT:   %DefaultOrUnformed.type: type = facet_type <@DefaultOrUnformed> [concrete]
// CHECK:STDOUT:   %T.67d: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as.DefaultOrUnformed.impl.Op.type.5f6: type = fn_type @T.as.DefaultOrUnformed.impl.Op, @T.as.DefaultOrUnformed.impl(%T.67d) [symbolic]
// CHECK:STDOUT:   %T.as.DefaultOrUnformed.impl.Op.5c0: %T.as.DefaultOrUnformed.impl.Op.type.5f6 = struct_value () [symbolic]
// CHECK:STDOUT:   %DefaultOrUnformed.lookup_impl_witness: <witness> = lookup_impl_witness %Int, @DefaultOrUnformed [symbolic]
// CHECK:STDOUT:   %.517: require_specific_def_type = require_specific_def @T.as.DefaultOrUnformed.impl(%Int) [symbolic]
// CHECK:STDOUT:   %DefaultOrUnformed.facet.67a: %DefaultOrUnformed.type = facet_value %Int, (%DefaultOrUnformed.lookup_impl_witness) [symbolic]
// CHECK:STDOUT:   %DefaultOrUnformed.WithSelf.Op.type.cb5: type = fn_type @DefaultOrUnformed.WithSelf.Op, @DefaultOrUnformed.WithSelf(%DefaultOrUnformed.facet.67a) [symbolic]
// CHECK:STDOUT:   %.b59: type = fn_type_with_self_type %DefaultOrUnformed.WithSelf.Op.type.cb5, %DefaultOrUnformed.facet.67a [symbolic]
// CHECK:STDOUT:   %impl.elem0.f8f: %.b59 = impl_witness_access %DefaultOrUnformed.lookup_impl_witness, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.4db: <specific function> = specific_impl_function %impl.elem0.f8f, @DefaultOrUnformed.WithSelf.Op(%DefaultOrUnformed.facet.67a) [symbolic]
// CHECK:STDOUT:   %SelfDestruct.type: type = fn_type @SelfDestruct [concrete]
// CHECK:STDOUT:   %SelfDestruct: %SelfDestruct.type = struct_value () [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %facet_type: type = facet_type <@DestroyImpl & @SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %T.09f: %SubobjectDestroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.type.dea: type = fn_type @T.as_type.as.DestroyImpl.impl.Op, @T.as_type.as.DestroyImpl.impl(%T.09f) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.6e5: %T.as_type.as.DestroyImpl.impl.Op.type.dea = struct_value () [symbolic]
// CHECK:STDOUT:   %DestroyImpl.lookup_impl_witness.79f: <witness> = lookup_impl_witness %Int, @DestroyImpl [symbolic]
// CHECK:STDOUT:   %SubobjectDestroy.lookup_impl_witness.f5e: <witness> = lookup_impl_witness %Int, @SubobjectDestroy [symbolic]
// CHECK:STDOUT:   %SubobjectDestroy.facet.cbf: %SubobjectDestroy.type = facet_value %Int, (%SubobjectDestroy.lookup_impl_witness.f5e) [symbolic]
// CHECK:STDOUT:   %.811: require_specific_def_type = require_specific_def @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.cbf) [symbolic]
// CHECK:STDOUT:   %facet_value.a41: %facet_type = facet_value %Int, (%DestroyImpl.lookup_impl_witness.79f, %SubobjectDestroy.lookup_impl_witness.f5e) [symbolic]
// CHECK:STDOUT:   %SelfDestruct.specific_fn.603: <specific function> = specific_function %SelfDestruct, @SelfDestruct(%facet_value.a41) [symbolic]
// CHECK:STDOUT:   %CallNegative.type: type = fn_type @CallNegative [concrete]
// CHECK:STDOUT:   %CallNegative: %CallNegative.type = struct_value () [concrete]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %ErrorIfNIsZero.specific_fn: <specific function> = specific_function %ErrorIfNIsZero, @ErrorIfNIsZero(%int_0) [concrete]
// CHECK:STDOUT:   %i0: type = class_type @Int, @Int(%int_0) [concrete]
// CHECK:STDOUT:   %pattern_type.4ed: type = pattern_type %i0 [concrete]
// CHECK:STDOUT:   %v.patt.661: %pattern_type.4ed = ref_binding_pattern v [concrete]
// CHECK:STDOUT:   %v.var_patt.2b3: %pattern_type.4ed = var_pattern %v.patt.661 [concrete]
// CHECK:STDOUT:   %DefaultOrUnformed.impl_witness.a20: <witness> = impl_witness imports.%DefaultOrUnformed.impl_witness_table.856, @T.as.DefaultOrUnformed.impl(%i0) [concrete]
// CHECK:STDOUT:   %T.as.DefaultOrUnformed.impl.Op.type.260: type = fn_type @T.as.DefaultOrUnformed.impl.Op, @T.as.DefaultOrUnformed.impl(%i0) [concrete]
// CHECK:STDOUT:   %T.as.DefaultOrUnformed.impl.Op.353: %T.as.DefaultOrUnformed.impl.Op.type.260 = struct_value () [concrete]
// CHECK:STDOUT:   %.1e6: require_specific_def_type = require_specific_def @T.as.DefaultOrUnformed.impl(%i0) [concrete]
// CHECK:STDOUT:   %DefaultOrUnformed.facet.d1a: %DefaultOrUnformed.type = facet_value %i0, (%DefaultOrUnformed.impl_witness.a20) [concrete]
// CHECK:STDOUT:   %DefaultOrUnformed.WithSelf.Op.type.d5c: type = fn_type @DefaultOrUnformed.WithSelf.Op, @DefaultOrUnformed.WithSelf(%DefaultOrUnformed.facet.d1a) [concrete]
// CHECK:STDOUT:   %.3e6: type = fn_type_with_self_type %DefaultOrUnformed.WithSelf.Op.type.d5c, %DefaultOrUnformed.facet.d1a [concrete]
// CHECK:STDOUT:   %T.as.DefaultOrUnformed.impl.Op.specific_fn: <specific function> = specific_function %T.as.DefaultOrUnformed.impl.Op.353, @T.as.DefaultOrUnformed.impl.Op(%i0) [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.687: type = fn_type @SubobjectDestroy.WithSelf.Op.loc9 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f: %SubobjectDestroy.WithSelf.Op.type.687 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.26a: %SubobjectDestroy.type = facet_value %i0, (%custom_witness.b80) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.c8f: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.26a) [concrete]
// CHECK:STDOUT:   %.737: require_specific_def_type = require_specific_def @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.26a) [concrete]
// CHECK:STDOUT:   %facet_value.18e: %facet_type = facet_value %i0, (%DestroyImpl.impl_witness.c8f, %custom_witness.b80) [concrete]
// CHECK:STDOUT:   %SelfDestruct.specific_fn.adf: <specific function> = specific_function %SelfDestruct, @SelfDestruct(%facet_value.18e) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .IntLiteral = %Core.IntLiteral
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .DefaultOrUnformed = %Core.DefaultOrUnformed
// CHECK:STDOUT:     .SelfDestruct = %Core.SelfDestruct
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.IntLiteral: type = import_ref Core//prelude/parts/int_literal, IntLiteral, loaded [concrete = Core.IntLiteral]
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/parts/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT:   %Core.DefaultOrUnformed: type = import_ref Core//prelude/parts/default, DefaultOrUnformed, loaded [concrete = constants.%DefaultOrUnformed.type]
// CHECK:STDOUT:   %Core.import_ref.002: @T.as.DefaultOrUnformed.impl.%T.as.DefaultOrUnformed.impl.Op.type (%T.as.DefaultOrUnformed.impl.Op.type.5f6) = import_ref Core//prelude/parts/default, loc{{\d+_\d+}}, loaded [symbolic = @T.as.DefaultOrUnformed.impl.%T.as.DefaultOrUnformed.impl.Op (constants.%T.as.DefaultOrUnformed.impl.Op.5c0)]
// CHECK:STDOUT:   %DefaultOrUnformed.impl_witness_table.856 = impl_witness_table (%Core.import_ref.002), @T.as.DefaultOrUnformed.impl [concrete]
// CHECK:STDOUT:   %Core.SelfDestruct: %SelfDestruct.type = import_ref Core//prelude/parts/destroy, SelfDestruct, loaded [concrete = constants.%SelfDestruct]
// CHECK:STDOUT:   %Core.import_ref.74d: @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op.type (%T.as_type.as.DestroyImpl.impl.Op.type.dea) = import_ref Core//prelude/parts/destroy, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op (constants.%T.as_type.as.DestroyImpl.impl.Op.6e5)]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table = impl_witness_table (%Core.import_ref.74d), @T.as_type.as.DestroyImpl.impl [concrete]
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
// CHECK:STDOUT:     %N.patt.loc4_28.1: %pattern_type.dc0 = symbolic_binding_pattern N, 0 [symbolic = %N.patt.loc4_28.2 (constants.%N.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc4: type = splice_block %IntLiteral.ref [concrete = Core.IntLiteral] {
// CHECK:STDOUT:       %.Self.frozen: type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %Core.ref.loc4: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:       %IntLiteral.ref: type = name_ref IntLiteral, imports.%Core.IntLiteral [concrete = Core.IntLiteral]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %N.loc4_28.2: Core.IntLiteral = symbolic_binding N, 0 [symbolic = %N.loc4_28.1 (constants.%N)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %CallNegative.decl: %CallNegative.type = fn_decl @CallNegative [concrete = constants.%CallNegative] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @ErrorIfNIsZero(%N.loc4_28.2: Core.IntLiteral) {
// CHECK:STDOUT:   %N.patt.loc4_28.2: %pattern_type.dc0 = symbolic_binding_pattern N, 0 [symbolic = %N.patt.loc4_28.2 (constants.%N.patt)]
// CHECK:STDOUT:   %N.loc4_28.1: Core.IntLiteral = symbolic_binding N, 0 [symbolic = %N.loc4_28.1 (constants.%N)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Int.loc9_27.2: type = class_type @Int, @Int(%N.loc4_28.1) [symbolic = %Int.loc9_27.2 (constants.%Int)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %Int.loc9_27.2 [symbolic = %require_complete (constants.%require_complete.d0a)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Int.loc9_27.2 [symbolic = %pattern_type (constants.%pattern_type.551)]
// CHECK:STDOUT:   %v.patt.loc9_15.2: @ErrorIfNIsZero.%pattern_type (%pattern_type.551) = ref_binding_pattern v [symbolic = %v.patt.loc9_15.2 (constants.%v.patt.fc3)]
// CHECK:STDOUT:   %v.var_patt.loc9_3.2: @ErrorIfNIsZero.%pattern_type (%pattern_type.551) = var_pattern %v.patt.loc9_15.2 [symbolic = %v.var_patt.loc9_3.2 (constants.%v.var_patt.c99)]
// CHECK:STDOUT:   %.loc9_28.3: require_specific_def_type = require_specific_def @T.as.DefaultOrUnformed.impl(%Int.loc9_27.2) [symbolic = %.loc9_28.3 (constants.%.517)]
// CHECK:STDOUT:   %DefaultOrUnformed.lookup_impl_witness: <witness> = lookup_impl_witness %Int.loc9_27.2, @DefaultOrUnformed [symbolic = %DefaultOrUnformed.lookup_impl_witness (constants.%DefaultOrUnformed.lookup_impl_witness)]
// CHECK:STDOUT:   %DefaultOrUnformed.facet.loc9_28.2: %DefaultOrUnformed.type = facet_value %Int.loc9_27.2, (%DefaultOrUnformed.lookup_impl_witness) [symbolic = %DefaultOrUnformed.facet.loc9_28.2 (constants.%DefaultOrUnformed.facet.67a)]
// CHECK:STDOUT:   %DefaultOrUnformed.WithSelf.Op.type: type = fn_type @DefaultOrUnformed.WithSelf.Op, @DefaultOrUnformed.WithSelf(%DefaultOrUnformed.facet.loc9_28.2) [symbolic = %DefaultOrUnformed.WithSelf.Op.type (constants.%DefaultOrUnformed.WithSelf.Op.type.cb5)]
// CHECK:STDOUT:   %.loc9_28.4: type = fn_type_with_self_type %DefaultOrUnformed.WithSelf.Op.type, %DefaultOrUnformed.facet.loc9_28.2 [symbolic = %.loc9_28.4 (constants.%.b59)]
// CHECK:STDOUT:   %impl.elem0.loc9_28.2: @ErrorIfNIsZero.%.loc9_28.4 (%.b59) = impl_witness_access %DefaultOrUnformed.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc9_28.2 (constants.%impl.elem0.f8f)]
// CHECK:STDOUT:   %specific_impl_fn.loc9_28.2: <specific function> = specific_impl_function %impl.elem0.loc9_28.2, @DefaultOrUnformed.WithSelf.Op(%DefaultOrUnformed.facet.loc9_28.2) [symbolic = %specific_impl_fn.loc9_28.2 (constants.%specific_impl_fn.4db)]
// CHECK:STDOUT:   %SubobjectDestroy.lookup_impl_witness: <witness> = lookup_impl_witness %Int.loc9_27.2, @SubobjectDestroy [symbolic = %SubobjectDestroy.lookup_impl_witness (constants.%SubobjectDestroy.lookup_impl_witness.f5e)]
// CHECK:STDOUT:   %SubobjectDestroy.facet: %SubobjectDestroy.type = facet_value %Int.loc9_27.2, (%SubobjectDestroy.lookup_impl_witness) [symbolic = %SubobjectDestroy.facet (constants.%SubobjectDestroy.facet.cbf)]
// CHECK:STDOUT:   %.loc9_3.3: require_specific_def_type = require_specific_def @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet) [symbolic = %.loc9_3.3 (constants.%.811)]
// CHECK:STDOUT:   %DestroyImpl.lookup_impl_witness: <witness> = lookup_impl_witness %Int.loc9_27.2, @DestroyImpl [symbolic = %DestroyImpl.lookup_impl_witness (constants.%DestroyImpl.lookup_impl_witness.79f)]
// CHECK:STDOUT:   %facet_value.loc9_3.3: %facet_type = facet_value %Int.loc9_27.2, (%DestroyImpl.lookup_impl_witness, %SubobjectDestroy.lookup_impl_witness) [symbolic = %facet_value.loc9_3.3 (constants.%facet_value.a41)]
// CHECK:STDOUT:   %SelfDestruct.specific_fn.2: <specific function> = specific_function constants.%SelfDestruct, @SelfDestruct(%facet_value.loc9_3.3) [symbolic = %SelfDestruct.specific_fn.2 (constants.%SelfDestruct.specific_fn.603)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %v.var: ref @ErrorIfNIsZero.%Int.loc9_27.2 (%Int) = var_storage %v.var_patt.loc9_3.1
// CHECK:STDOUT:     %DefaultOrUnformed.facet.loc9_28.1: %DefaultOrUnformed.type = facet_value constants.%Int, (constants.%DefaultOrUnformed.lookup_impl_witness) [symbolic = %DefaultOrUnformed.facet.loc9_28.2 (constants.%DefaultOrUnformed.facet.67a)]
// CHECK:STDOUT:     %.loc9_28.1: %DefaultOrUnformed.type = converted constants.%Int, %DefaultOrUnformed.facet.loc9_28.1 [symbolic = %DefaultOrUnformed.facet.loc9_28.2 (constants.%DefaultOrUnformed.facet.67a)]
// CHECK:STDOUT:     %as_type: type = facet_access_type %.loc9_28.1 [symbolic = %Int.loc9_27.2 (constants.%Int)]
// CHECK:STDOUT:     %.loc9_28.2: type = converted %.loc9_28.1, %as_type [symbolic = %Int.loc9_27.2 (constants.%Int)]
// CHECK:STDOUT:     %impl.elem0.loc9_28.1: @ErrorIfNIsZero.%.loc9_28.4 (%.b59) = impl_witness_access constants.%DefaultOrUnformed.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc9_28.2 (constants.%impl.elem0.f8f)]
// CHECK:STDOUT:     %specific_impl_fn.loc9_28.1: <specific function> = specific_impl_function %impl.elem0.loc9_28.1, @DefaultOrUnformed.WithSelf.Op(constants.%DefaultOrUnformed.facet.67a) [symbolic = %specific_impl_fn.loc9_28.2 (constants.%specific_impl_fn.4db)]
// CHECK:STDOUT:     %DefaultOrUnformed.WithSelf.Op.call: init @ErrorIfNIsZero.%Int.loc9_27.2 (%Int) = call %specific_impl_fn.loc9_28.1()
// CHECK:STDOUT:     assign %v.var, %DefaultOrUnformed.WithSelf.Op.call
// CHECK:STDOUT:     %.loc9_27: type = splice_block %Int.loc9_27.1 [symbolic = %Int.loc9_27.2 (constants.%Int)] {
// CHECK:STDOUT:       %Core.ref.loc9: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:       %Int.ref: %Int.type = name_ref Int, imports.%Core.Int [concrete = constants.%Int.generic]
// CHECK:STDOUT:       %N.ref: Core.IntLiteral = name_ref N, %N.loc4_28.2 [symbolic = %N.loc4_28.1 (constants.%N)]
// CHECK:STDOUT:       %Int.loc9_27.1: type = class_type @Int, @Int(constants.%N) [symbolic = %Int.loc9_27.2 (constants.%Int)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %v: ref @ErrorIfNIsZero.%Int.loc9_27.2 (%Int) = wrapper_binding v, %v.var
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %v.patt.loc9_15.1: @ErrorIfNIsZero.%pattern_type (%pattern_type.551) = ref_binding_pattern v [symbolic = %v.patt.loc9_15.2 (constants.%v.patt.fc3)]
// CHECK:STDOUT:       %v.var_patt.loc9_3.1: @ErrorIfNIsZero.%pattern_type (%pattern_type.551) = var_pattern %v.patt.loc9_15.1 [symbolic = %v.var_patt.loc9_3.2 (constants.%v.var_patt.c99)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %facet_value.loc9_3.1: %facet_type = facet_value constants.%Int, (constants.%DestroyImpl.lookup_impl_witness.79f, constants.%SubobjectDestroy.lookup_impl_witness.f5e) [symbolic = %facet_value.loc9_3.3 (constants.%facet_value.a41)]
// CHECK:STDOUT:     %.loc9_3.1: %facet_type = converted constants.%Int, %facet_value.loc9_3.1 [symbolic = %facet_value.loc9_3.3 (constants.%facet_value.a41)]
// CHECK:STDOUT:     %facet_value.loc9_3.2: %facet_type = facet_value constants.%Int, (constants.%DestroyImpl.lookup_impl_witness.79f, constants.%SubobjectDestroy.lookup_impl_witness.f5e) [symbolic = %facet_value.loc9_3.3 (constants.%facet_value.a41)]
// CHECK:STDOUT:     %.loc9_3.2: %facet_type = converted constants.%Int, %facet_value.loc9_3.2 [symbolic = %facet_value.loc9_3.3 (constants.%facet_value.a41)]
// CHECK:STDOUT:     %SelfDestruct.specific_fn.1: <specific function> = specific_function constants.%SelfDestruct, @SelfDestruct(constants.%facet_value.a41) [symbolic = %SelfDestruct.specific_fn.2 (constants.%SelfDestruct.specific_fn.603)]
// CHECK:STDOUT:     %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%v.var)
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
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc9(%self.param: ref %i0) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ErrorIfNIsZero(constants.%N) {
// CHECK:STDOUT:   %N.patt.loc4_28.2 => constants.%N.patt
// CHECK:STDOUT:   %N.loc4_28.1 => constants.%N
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ErrorIfNIsZero(constants.%int_0) {
// CHECK:STDOUT:   %N.patt.loc4_28.2 => constants.%N.patt
// CHECK:STDOUT:   %N.loc4_28.1 => constants.%int_0
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Int.loc9_27.2 => constants.%i0
// CHECK:STDOUT:   %require_complete => <error>
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.4ed
// CHECK:STDOUT:   %v.patt.loc9_15.2 => constants.%v.patt.661
// CHECK:STDOUT:   %v.var_patt.loc9_3.2 => constants.%v.var_patt.2b3
// CHECK:STDOUT:   %.loc9_28.3 => constants.%.1e6
// CHECK:STDOUT:   %DefaultOrUnformed.lookup_impl_witness => constants.%DefaultOrUnformed.impl_witness.a20
// CHECK:STDOUT:   %DefaultOrUnformed.facet.loc9_28.2 => constants.%DefaultOrUnformed.facet.d1a
// CHECK:STDOUT:   %DefaultOrUnformed.WithSelf.Op.type => constants.%DefaultOrUnformed.WithSelf.Op.type.d5c
// CHECK:STDOUT:   %.loc9_28.4 => constants.%.3e6
// CHECK:STDOUT:   %impl.elem0.loc9_28.2 => constants.%T.as.DefaultOrUnformed.impl.Op.353
// CHECK:STDOUT:   %specific_impl_fn.loc9_28.2 => constants.%T.as.DefaultOrUnformed.impl.Op.specific_fn
// CHECK:STDOUT:   %SubobjectDestroy.lookup_impl_witness => constants.%custom_witness.b80
// CHECK:STDOUT:   %SubobjectDestroy.facet => constants.%SubobjectDestroy.facet.26a
// CHECK:STDOUT:   %.loc9_3.3 => constants.%.737
// CHECK:STDOUT:   %DestroyImpl.lookup_impl_witness => constants.%DestroyImpl.impl_witness.c8f
// CHECK:STDOUT:   %facet_value.loc9_3.3 => constants.%facet_value.18e
// CHECK:STDOUT:   %SelfDestruct.specific_fn.2 => constants.%SelfDestruct.specific_fn.adf
// CHECK:STDOUT: }
// CHECK:STDOUT:
