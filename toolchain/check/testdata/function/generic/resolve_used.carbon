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
  var unused v: Core.Int(N);
}

fn CallNegative() {
  // CHECK:STDERR: fail_todo_call_monomorphization_error.carbon:[[@LINE+10]]:3: error: unable to monomorphize specific `ErrorIfNIsZero(0)` [ResolvingSpecificHere]
  // CHECK:STDERR:   ErrorIfNIsZero(0);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~
  // CHECK:STDERR: fail_todo_call_monomorphization_error.carbon:[[@LINE-7]]:17: note: `Core.Int(N)` evaluates to incomplete type `i0` [IncompleteTypeInMonomorphization]
  // CHECK:STDERR:   var unused v: Core.Int(N);
  // CHECK:STDERR:                 ^~~~~~~~~~~
  // CHECK:STDERR: min_prelude/parts/int.carbon:11:9: note: integer type width of 0 is not positive [IntWidthNotPositive]
  // CHECK:STDERR:   adapt MakeInt(N);
  // CHECK:STDERR:         ^~~~~~~~~~
  // CHECK:STDERR:
  ErrorIfNIsZero(0);
}

// CHECK:STDOUT: --- fail_todo_call_monomorphization_error.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %IntLiteral.type: type = fn_type @IntLiteral [concrete]
// CHECK:STDOUT:   %IntLiteral: %IntLiteral.type = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.dc0: type = pattern_type Core.IntLiteral [concrete]
// CHECK:STDOUT:   %N: Core.IntLiteral = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %ErrorIfNIsZero.type: type = fn_type @ErrorIfNIsZero [concrete]
// CHECK:STDOUT:   %ErrorIfNIsZero: %ErrorIfNIsZero.type = struct_value () [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %Int: type = class_type @Int, @Int(%N) [symbolic]
// CHECK:STDOUT:   %require_complete.901: <witness> = require_complete_type %Int [symbolic]
// CHECK:STDOUT:   %pattern_type.764: type = pattern_type %Int [symbolic]
// CHECK:STDOUT:   %DefaultOrUnformed.type: type = facet_type <@DefaultOrUnformed> [concrete]
// CHECK:STDOUT:   %T.67d: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as.DefaultOrUnformed.impl.Op.type.6c6: type = fn_type @T.as.DefaultOrUnformed.impl.Op, @T.as.DefaultOrUnformed.impl(%T.67d) [symbolic]
// CHECK:STDOUT:   %T.as.DefaultOrUnformed.impl.Op.ae1: %T.as.DefaultOrUnformed.impl.Op.type.6c6 = struct_value () [symbolic]
// CHECK:STDOUT:   %DefaultOrUnformed.lookup_impl_witness: <witness> = lookup_impl_witness %Int, @DefaultOrUnformed [symbolic]
// CHECK:STDOUT:   %.4a9: require_specific_def_type = require_specific_def @T.as.DefaultOrUnformed.impl(%Int) [symbolic]
// CHECK:STDOUT:   %DefaultOrUnformed.facet.c82: %DefaultOrUnformed.type = facet_value %Int, (%DefaultOrUnformed.lookup_impl_witness) [symbolic]
// CHECK:STDOUT:   %DefaultOrUnformed.WithSelf.Op.type.814: type = fn_type @DefaultOrUnformed.WithSelf.Op, @DefaultOrUnformed.WithSelf(%DefaultOrUnformed.facet.c82) [symbolic]
// CHECK:STDOUT:   %.ca4: type = fn_type_with_self_type %DefaultOrUnformed.WithSelf.Op.type.814, %DefaultOrUnformed.facet.c82 [symbolic]
// CHECK:STDOUT:   %impl.elem0.12f: %.ca4 = impl_witness_access %DefaultOrUnformed.lookup_impl_witness, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.f0d: <specific function> = specific_impl_function %impl.elem0.12f, @DefaultOrUnformed.WithSelf.Op(%DefaultOrUnformed.facet.c82) [symbolic]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness: <witness> = lookup_impl_witness %Int, @Destroy [symbolic]
// CHECK:STDOUT:   %Destroy.facet.1c0: %Destroy.type = facet_value %Int, (%Destroy.lookup_impl_witness) [symbolic]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.297: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet.1c0) [symbolic]
// CHECK:STDOUT:   %.e63: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.297, %Destroy.facet.1c0 [symbolic]
// CHECK:STDOUT:   %impl.elem0.602: %.e63 = impl_witness_access %Destroy.lookup_impl_witness, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.d65: <specific function> = specific_impl_function %impl.elem0.602, @Destroy.WithSelf.Op(%Destroy.facet.1c0) [symbolic]
// CHECK:STDOUT:   %CallNegative.type: type = fn_type @CallNegative [concrete]
// CHECK:STDOUT:   %CallNegative: %CallNegative.type = struct_value () [concrete]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %ErrorIfNIsZero.specific_fn: <specific function> = specific_function %ErrorIfNIsZero, @ErrorIfNIsZero(%int_0) [concrete]
// CHECK:STDOUT:   %i0: type = class_type @Int, @Int(%int_0) [concrete]
// CHECK:STDOUT:   %pattern_type.47b: type = pattern_type %i0 [concrete]
// CHECK:STDOUT:   %DefaultOrUnformed.impl_witness.879: <witness> = impl_witness imports.%DefaultOrUnformed.impl_witness_table.349, @T.as.DefaultOrUnformed.impl(%i0) [concrete]
// CHECK:STDOUT:   %T.as.DefaultOrUnformed.impl.Op.type.7ec: type = fn_type @T.as.DefaultOrUnformed.impl.Op, @T.as.DefaultOrUnformed.impl(%i0) [concrete]
// CHECK:STDOUT:   %T.as.DefaultOrUnformed.impl.Op.5b7: %T.as.DefaultOrUnformed.impl.Op.type.7ec = struct_value () [concrete]
// CHECK:STDOUT:   %.d6a: require_specific_def_type = require_specific_def @T.as.DefaultOrUnformed.impl(%i0) [concrete]
// CHECK:STDOUT:   %DefaultOrUnformed.facet.255: %DefaultOrUnformed.type = facet_value %i0, (%DefaultOrUnformed.impl_witness.879) [concrete]
// CHECK:STDOUT:   %DefaultOrUnformed.WithSelf.Op.type.bd6: type = fn_type @DefaultOrUnformed.WithSelf.Op, @DefaultOrUnformed.WithSelf(%DefaultOrUnformed.facet.255) [concrete]
// CHECK:STDOUT:   %.639: type = fn_type_with_self_type %DefaultOrUnformed.WithSelf.Op.type.bd6, %DefaultOrUnformed.facet.255 [concrete]
// CHECK:STDOUT:   %T.as.DefaultOrUnformed.impl.Op.specific_fn: <specific function> = specific_function %T.as.DefaultOrUnformed.impl.Op.5b7, @T.as.DefaultOrUnformed.impl.Op(%i0) [concrete]
// CHECK:STDOUT:   %Destroy.Op.type: type = fn_type @Destroy.Op [concrete]
// CHECK:STDOUT:   %Destroy.Op: %Destroy.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.8d7: <witness> = custom_witness (%Destroy.Op), @Destroy [concrete]
// CHECK:STDOUT:   %Destroy.facet.eb2: %Destroy.type = facet_value %i0, (%custom_witness.8d7) [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.cef: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet.eb2) [concrete]
// CHECK:STDOUT:   %.98d: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.cef, %Destroy.facet.eb2 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .IntLiteral = %Core.IntLiteral
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .DefaultOrUnformed = %Core.DefaultOrUnformed
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.IntLiteral: %IntLiteral.type = import_ref Core//prelude/parts/int_literal, IntLiteral, loaded [concrete = constants.%IntLiteral]
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/parts/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT:   %Core.DefaultOrUnformed: type = import_ref Core//prelude/parts/default, DefaultOrUnformed, loaded [concrete = constants.%DefaultOrUnformed.type]
// CHECK:STDOUT:   %Core.import_ref.01d: @T.as.DefaultOrUnformed.impl.%T.as.DefaultOrUnformed.impl.Op.type (%T.as.DefaultOrUnformed.impl.Op.type.6c6) = import_ref Core//prelude/parts/default, loc{{\d+_\d+}}, loaded [symbolic = @T.as.DefaultOrUnformed.impl.%T.as.DefaultOrUnformed.impl.Op (constants.%T.as.DefaultOrUnformed.impl.Op.ae1)]
// CHECK:STDOUT:   %DefaultOrUnformed.impl_witness_table.349 = impl_witness_table (%Core.import_ref.01d), @T.as.DefaultOrUnformed.impl [concrete]
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
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
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %Core.ref.loc4: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:       %IntLiteral.ref: %IntLiteral.type = name_ref IntLiteral, imports.%Core.IntLiteral [concrete = constants.%IntLiteral]
// CHECK:STDOUT:       %IntLiteral.call: init type = call %IntLiteral.ref() [concrete = Core.IntLiteral]
// CHECK:STDOUT:       %.loc4_39.2: type = value_of_initializer %IntLiteral.call [concrete = Core.IntLiteral]
// CHECK:STDOUT:       %.loc4_39.3: type = converted %IntLiteral.call, %.loc4_39.2 [concrete = Core.IntLiteral]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %N.loc4_20.2: Core.IntLiteral = symbolic_binding N, 0 [symbolic = %N.loc4_20.1 (constants.%N)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %CallNegative.decl: %CallNegative.type = fn_decl @CallNegative [concrete = constants.%CallNegative] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @ErrorIfNIsZero(%N.loc4_20.2: Core.IntLiteral) {
// CHECK:STDOUT:   %N.loc4_20.1: Core.IntLiteral = symbolic_binding N, 0 [symbolic = %N.loc4_20.1 (constants.%N)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Int.loc9_27.2: type = class_type @Int, @Int(%N.loc4_20.1) [symbolic = %Int.loc9_27.2 (constants.%Int)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %Int.loc9_27.2 [symbolic = %require_complete (constants.%require_complete.901)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Int.loc9_27.2 [symbolic = %pattern_type (constants.%pattern_type.764)]
// CHECK:STDOUT:   %DefaultOrUnformed.lookup_impl_witness: <witness> = lookup_impl_witness %Int.loc9_27.2, @DefaultOrUnformed [symbolic = %DefaultOrUnformed.lookup_impl_witness (constants.%DefaultOrUnformed.lookup_impl_witness)]
// CHECK:STDOUT:   %.loc9_28.3: require_specific_def_type = require_specific_def @T.as.DefaultOrUnformed.impl(%Int.loc9_27.2) [symbolic = %.loc9_28.3 (constants.%.4a9)]
// CHECK:STDOUT:   %DefaultOrUnformed.facet.loc9_28.2: %DefaultOrUnformed.type = facet_value %Int.loc9_27.2, (%DefaultOrUnformed.lookup_impl_witness) [symbolic = %DefaultOrUnformed.facet.loc9_28.2 (constants.%DefaultOrUnformed.facet.c82)]
// CHECK:STDOUT:   %DefaultOrUnformed.WithSelf.Op.type: type = fn_type @DefaultOrUnformed.WithSelf.Op, @DefaultOrUnformed.WithSelf(%DefaultOrUnformed.facet.loc9_28.2) [symbolic = %DefaultOrUnformed.WithSelf.Op.type (constants.%DefaultOrUnformed.WithSelf.Op.type.814)]
// CHECK:STDOUT:   %.loc9_28.4: type = fn_type_with_self_type %DefaultOrUnformed.WithSelf.Op.type, %DefaultOrUnformed.facet.loc9_28.2 [symbolic = %.loc9_28.4 (constants.%.ca4)]
// CHECK:STDOUT:   %impl.elem0.loc9_28.2: @ErrorIfNIsZero.%.loc9_28.4 (%.ca4) = impl_witness_access %DefaultOrUnformed.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc9_28.2 (constants.%impl.elem0.12f)]
// CHECK:STDOUT:   %specific_impl_fn.loc9_28.2: <specific function> = specific_impl_function %impl.elem0.loc9_28.2, @DefaultOrUnformed.WithSelf.Op(%DefaultOrUnformed.facet.loc9_28.2) [symbolic = %specific_impl_fn.loc9_28.2 (constants.%specific_impl_fn.f0d)]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness: <witness> = lookup_impl_witness %Int.loc9_27.2, @Destroy [symbolic = %Destroy.lookup_impl_witness (constants.%Destroy.lookup_impl_witness)]
// CHECK:STDOUT:   %Destroy.facet: %Destroy.type = facet_value %Int.loc9_27.2, (%Destroy.lookup_impl_witness) [symbolic = %Destroy.facet (constants.%Destroy.facet.1c0)]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet) [symbolic = %Destroy.WithSelf.Op.type (constants.%Destroy.WithSelf.Op.type.297)]
// CHECK:STDOUT:   %.loc9_3: type = fn_type_with_self_type %Destroy.WithSelf.Op.type, %Destroy.facet [symbolic = %.loc9_3 (constants.%.e63)]
// CHECK:STDOUT:   %impl.elem0.loc9_3.2: @ErrorIfNIsZero.%.loc9_3 (%.e63) = impl_witness_access %Destroy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc9_3.2 (constants.%impl.elem0.602)]
// CHECK:STDOUT:   %specific_impl_fn.loc9_3.2: <specific function> = specific_impl_function %impl.elem0.loc9_3.2, @Destroy.WithSelf.Op(%Destroy.facet) [symbolic = %specific_impl_fn.loc9_3.2 (constants.%specific_impl_fn.d65)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %v.patt: @ErrorIfNIsZero.%pattern_type (%pattern_type.764) = ref_binding_pattern v [concrete]
// CHECK:STDOUT:       %v.var_patt: @ErrorIfNIsZero.%pattern_type (%pattern_type.764) = var_pattern %v.patt [concrete]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %v.var: ref @ErrorIfNIsZero.%Int.loc9_27.2 (%Int) = var %v.var_patt
// CHECK:STDOUT:     %DefaultOrUnformed.facet.loc9_28.1: %DefaultOrUnformed.type = facet_value constants.%Int, (constants.%DefaultOrUnformed.lookup_impl_witness) [symbolic = %DefaultOrUnformed.facet.loc9_28.2 (constants.%DefaultOrUnformed.facet.c82)]
// CHECK:STDOUT:     %.loc9_28.1: %DefaultOrUnformed.type = converted constants.%Int, %DefaultOrUnformed.facet.loc9_28.1 [symbolic = %DefaultOrUnformed.facet.loc9_28.2 (constants.%DefaultOrUnformed.facet.c82)]
// CHECK:STDOUT:     %as_type: type = facet_access_type %.loc9_28.1 [symbolic = %Int.loc9_27.2 (constants.%Int)]
// CHECK:STDOUT:     %.loc9_28.2: type = converted %.loc9_28.1, %as_type [symbolic = %Int.loc9_27.2 (constants.%Int)]
// CHECK:STDOUT:     %impl.elem0.loc9_28.1: @ErrorIfNIsZero.%.loc9_28.4 (%.ca4) = impl_witness_access constants.%DefaultOrUnformed.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc9_28.2 (constants.%impl.elem0.12f)]
// CHECK:STDOUT:     %specific_impl_fn.loc9_28.1: <specific function> = specific_impl_function %impl.elem0.loc9_28.1, @DefaultOrUnformed.WithSelf.Op(constants.%DefaultOrUnformed.facet.c82) [symbolic = %specific_impl_fn.loc9_28.2 (constants.%specific_impl_fn.f0d)]
// CHECK:STDOUT:     %DefaultOrUnformed.WithSelf.Op.call: init @ErrorIfNIsZero.%Int.loc9_27.2 (%Int) = call %specific_impl_fn.loc9_28.1()
// CHECK:STDOUT:     assign %v.var, %DefaultOrUnformed.WithSelf.Op.call
// CHECK:STDOUT:     %.loc9_27: type = splice_block %Int.loc9_27.1 [symbolic = %Int.loc9_27.2 (constants.%Int)] {
// CHECK:STDOUT:       %Core.ref.loc9: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:       %Int.ref: %Int.type = name_ref Int, imports.%Core.Int [concrete = constants.%Int.generic]
// CHECK:STDOUT:       %N.ref: Core.IntLiteral = name_ref N, %N.loc4_20.2 [symbolic = %N.loc4_20.1 (constants.%N)]
// CHECK:STDOUT:       %Int.loc9_27.1: type = class_type @Int, @Int(constants.%N) [symbolic = %Int.loc9_27.2 (constants.%Int)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %v: ref @ErrorIfNIsZero.%Int.loc9_27.2 (%Int) = ref_binding v, %v.var
// CHECK:STDOUT:     %impl.elem0.loc9_3.1: @ErrorIfNIsZero.%.loc9_3 (%.e63) = impl_witness_access constants.%Destroy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc9_3.2 (constants.%impl.elem0.602)]
// CHECK:STDOUT:     %bound_method.loc9_3.1: <bound method> = bound_method %v.var, %impl.elem0.loc9_3.1
// CHECK:STDOUT:     %specific_impl_fn.loc9_3.1: <specific function> = specific_impl_function %impl.elem0.loc9_3.1, @Destroy.WithSelf.Op(constants.%Destroy.facet.1c0) [symbolic = %specific_impl_fn.loc9_3.2 (constants.%specific_impl_fn.d65)]
// CHECK:STDOUT:     %bound_method.loc9_3.2: <bound method> = bound_method %v.var, %specific_impl_fn.loc9_3.1
// CHECK:STDOUT:     %Destroy.WithSelf.Op.call: init %empty_tuple.type = call %bound_method.loc9_3.2(%v.var)
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
// CHECK:STDOUT: fn @Destroy.Op(%self.param: ref %i0) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ErrorIfNIsZero(constants.%N) {
// CHECK:STDOUT:   %N.loc4_20.1 => constants.%N
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ErrorIfNIsZero(constants.%int_0) {
// CHECK:STDOUT:   %N.loc4_20.1 => constants.%int_0
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Int.loc9_27.2 => constants.%i0
// CHECK:STDOUT:   %require_complete => <error>
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.47b
// CHECK:STDOUT:   %DefaultOrUnformed.lookup_impl_witness => constants.%DefaultOrUnformed.impl_witness.879
// CHECK:STDOUT:   %.loc9_28.3 => constants.%.d6a
// CHECK:STDOUT:   %DefaultOrUnformed.facet.loc9_28.2 => constants.%DefaultOrUnformed.facet.255
// CHECK:STDOUT:   %DefaultOrUnformed.WithSelf.Op.type => constants.%DefaultOrUnformed.WithSelf.Op.type.bd6
// CHECK:STDOUT:   %.loc9_28.4 => constants.%.639
// CHECK:STDOUT:   %impl.elem0.loc9_28.2 => constants.%T.as.DefaultOrUnformed.impl.Op.5b7
// CHECK:STDOUT:   %specific_impl_fn.loc9_28.2 => constants.%T.as.DefaultOrUnformed.impl.Op.specific_fn
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.8d7
// CHECK:STDOUT:   %Destroy.facet => constants.%Destroy.facet.eb2
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type => constants.%Destroy.WithSelf.Op.type.cef
// CHECK:STDOUT:   %.loc9_3 => constants.%.98d
// CHECK:STDOUT:   %impl.elem0.loc9_3.2 => constants.%Destroy.Op
// CHECK:STDOUT:   %specific_impl_fn.loc9_3.2 => constants.%Destroy.Op
// CHECK:STDOUT: }
// CHECK:STDOUT:
