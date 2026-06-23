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

fn ErrorIfNIsZero(N:! Core.IntLiteral) {
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
// CHECK:STDOUT:   %pattern_type.dc0: type = pattern_type Core.IntLiteral [concrete]
// CHECK:STDOUT:   %N.patt: %pattern_type.dc0 = symbolic_binding_pattern N, 0 [symbolic]
// CHECK:STDOUT:   %N: Core.IntLiteral = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %ErrorIfNIsZero.type: type = fn_type @ErrorIfNIsZero [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %ErrorIfNIsZero: %ErrorIfNIsZero.type = struct_value () [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %Int: type = class_type @Int, @Int(%N) [symbolic]
// CHECK:STDOUT:   %require_complete.4df: <witness> = require_complete_type %Int [symbolic]
// CHECK:STDOUT:   %pattern_type.142: type = pattern_type %Int [symbolic]
// CHECK:STDOUT:   %v.patt.834: %pattern_type.142 = ref_binding_pattern v [symbolic]
// CHECK:STDOUT:   %v.var_patt.cd4: %pattern_type.142 = var_pattern %v.patt.834 [symbolic]
// CHECK:STDOUT:   %DefaultOrUnformed.type: type = facet_type <@DefaultOrUnformed> [concrete]
// CHECK:STDOUT:   %T.67d: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as.DefaultOrUnformed.impl.Op.type.150: type = fn_type @T.as.DefaultOrUnformed.impl.Op, @T.as.DefaultOrUnformed.impl(%T.67d) [symbolic]
// CHECK:STDOUT:   %T.as.DefaultOrUnformed.impl.Op.53a: %T.as.DefaultOrUnformed.impl.Op.type.150 = struct_value () [symbolic]
// CHECK:STDOUT:   %DefaultOrUnformed.lookup_impl_witness: <witness> = lookup_impl_witness %Int, @DefaultOrUnformed [symbolic]
// CHECK:STDOUT:   %.386: require_specific_def_type = require_specific_def @T.as.DefaultOrUnformed.impl(%Int) [symbolic]
// CHECK:STDOUT:   %DefaultOrUnformed.facet.e73: %DefaultOrUnformed.type = facet_value %Int, (%DefaultOrUnformed.lookup_impl_witness) [symbolic]
// CHECK:STDOUT:   %DefaultOrUnformed.WithSelf.Op.type.4ac: type = fn_type @DefaultOrUnformed.WithSelf.Op, @DefaultOrUnformed.WithSelf(%DefaultOrUnformed.facet.e73) [symbolic]
// CHECK:STDOUT:   %.b74: type = fn_type_with_self_type %DefaultOrUnformed.WithSelf.Op.type.4ac, %DefaultOrUnformed.facet.e73 [symbolic]
// CHECK:STDOUT:   %impl.elem0.b17: %.b74 = impl_witness_access %DefaultOrUnformed.lookup_impl_witness, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.cf5: <specific function> = specific_impl_function %impl.elem0.b17, @DefaultOrUnformed.WithSelf.Op(%DefaultOrUnformed.facet.e73) [symbolic]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.173: <witness> = lookup_impl_witness %Int, @Destroy [symbolic]
// CHECK:STDOUT:   %Destroy.facet.7bb: %Destroy.type = facet_value %Int, (%Destroy.lookup_impl_witness.173) [symbolic]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.8f1: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet.7bb) [symbolic]
// CHECK:STDOUT:   %.2c6: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.8f1, %Destroy.facet.7bb [symbolic]
// CHECK:STDOUT:   %impl.elem0.604: %.2c6 = impl_witness_access %Destroy.lookup_impl_witness.173, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.693: <specific function> = specific_impl_function %impl.elem0.604, @Destroy.WithSelf.Op(%Destroy.facet.7bb) [symbolic]
// CHECK:STDOUT:   %CallNegative.type: type = fn_type @CallNegative [concrete]
// CHECK:STDOUT:   %CallNegative: %CallNegative.type = struct_value () [concrete]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %ErrorIfNIsZero.specific_fn: <specific function> = specific_function %ErrorIfNIsZero, @ErrorIfNIsZero(%int_0) [concrete]
// CHECK:STDOUT:   %i0: type = class_type @Int, @Int(%int_0) [concrete]
// CHECK:STDOUT:   %pattern_type.d0e: type = pattern_type %i0 [concrete]
// CHECK:STDOUT:   %v.patt.53c: %pattern_type.d0e = ref_binding_pattern v [concrete]
// CHECK:STDOUT:   %v.var_patt.0d0: %pattern_type.d0e = var_pattern %v.patt.53c [concrete]
// CHECK:STDOUT:   %DefaultOrUnformed.impl_witness.8d2: <witness> = impl_witness imports.%DefaultOrUnformed.impl_witness_table.374, @T.as.DefaultOrUnformed.impl(%i0) [concrete]
// CHECK:STDOUT:   %T.as.DefaultOrUnformed.impl.Op.type.baf: type = fn_type @T.as.DefaultOrUnformed.impl.Op, @T.as.DefaultOrUnformed.impl(%i0) [concrete]
// CHECK:STDOUT:   %T.as.DefaultOrUnformed.impl.Op.38a: %T.as.DefaultOrUnformed.impl.Op.type.baf = struct_value () [concrete]
// CHECK:STDOUT:   %.060: require_specific_def_type = require_specific_def @T.as.DefaultOrUnformed.impl(%i0) [concrete]
// CHECK:STDOUT:   %DefaultOrUnformed.facet.4e3: %DefaultOrUnformed.type = facet_value %i0, (%DefaultOrUnformed.impl_witness.8d2) [concrete]
// CHECK:STDOUT:   %DefaultOrUnformed.WithSelf.Op.type.7c9: type = fn_type @DefaultOrUnformed.WithSelf.Op, @DefaultOrUnformed.WithSelf(%DefaultOrUnformed.facet.4e3) [concrete]
// CHECK:STDOUT:   %.21a: type = fn_type_with_self_type %DefaultOrUnformed.WithSelf.Op.type.7c9, %DefaultOrUnformed.facet.4e3 [concrete]
// CHECK:STDOUT:   %T.as.DefaultOrUnformed.impl.Op.specific_fn: <specific function> = specific_function %T.as.DefaultOrUnformed.impl.Op.38a, @T.as.DefaultOrUnformed.impl.Op(%i0) [concrete]
// CHECK:STDOUT:   %Destroy.Op.type: type = fn_type @Destroy.Op [concrete]
// CHECK:STDOUT:   %Destroy.Op: %Destroy.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.df9: <witness> = custom_witness (%Destroy.Op), @Destroy [concrete]
// CHECK:STDOUT:   %Destroy.facet.85b: %Destroy.type = facet_value %i0, (%custom_witness.df9) [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.b7e: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet.85b) [concrete]
// CHECK:STDOUT:   %.338: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.b7e, %Destroy.facet.85b [concrete]
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
// CHECK:STDOUT:   %Core.IntLiteral: type = import_ref Core//prelude/parts/int_literal, IntLiteral, loaded [concrete = Core.IntLiteral]
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/parts/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT:   %Core.DefaultOrUnformed: type = import_ref Core//prelude/parts/default, DefaultOrUnformed, loaded [concrete = constants.%DefaultOrUnformed.type]
// CHECK:STDOUT:   %Core.import_ref.cc8: @T.as.DefaultOrUnformed.impl.%T.as.DefaultOrUnformed.impl.Op.type (%T.as.DefaultOrUnformed.impl.Op.type.150) = import_ref Core//prelude/parts/default, loc{{\d+_\d+}}, loaded [symbolic = @T.as.DefaultOrUnformed.impl.%T.as.DefaultOrUnformed.impl.Op (constants.%T.as.DefaultOrUnformed.impl.Op.53a)]
// CHECK:STDOUT:   %DefaultOrUnformed.impl_witness_table.374 = impl_witness_table (%Core.import_ref.cc8), @T.as.DefaultOrUnformed.impl [concrete]
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
// CHECK:STDOUT:     %N.patt.loc4_20.1: %pattern_type.dc0 = symbolic_binding_pattern N, 0 [symbolic = %N.patt.loc4_20.2 (constants.%N.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc4: type = splice_block %IntLiteral.ref [concrete = Core.IntLiteral] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %Core.ref.loc4: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:       %IntLiteral.ref: type = name_ref IntLiteral, imports.%Core.IntLiteral [concrete = Core.IntLiteral]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %N.loc4_20.2: Core.IntLiteral = symbolic_binding N, 0 [symbolic = %N.loc4_20.1 (constants.%N)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %CallNegative.decl: %CallNegative.type = fn_decl @CallNegative [concrete = constants.%CallNegative] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @ErrorIfNIsZero(%N.loc4_20.2: Core.IntLiteral) {
// CHECK:STDOUT:   %N.patt.loc4_20.2: %pattern_type.dc0 = symbolic_binding_pattern N, 0 [symbolic = %N.patt.loc4_20.2 (constants.%N.patt)]
// CHECK:STDOUT:   %N.loc4_20.1: Core.IntLiteral = symbolic_binding N, 0 [symbolic = %N.loc4_20.1 (constants.%N)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Int.loc9_27.2: type = class_type @Int, @Int(%N.loc4_20.1) [symbolic = %Int.loc9_27.2 (constants.%Int)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %Int.loc9_27.2 [symbolic = %require_complete (constants.%require_complete.4df)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Int.loc9_27.2 [symbolic = %pattern_type (constants.%pattern_type.142)]
// CHECK:STDOUT:   %v.patt.loc9_15.2: @ErrorIfNIsZero.%pattern_type (%pattern_type.142) = ref_binding_pattern v [symbolic = %v.patt.loc9_15.2 (constants.%v.patt.834)]
// CHECK:STDOUT:   %v.var_patt.loc9_3.2: @ErrorIfNIsZero.%pattern_type (%pattern_type.142) = var_pattern %v.patt.loc9_15.2 [symbolic = %v.var_patt.loc9_3.2 (constants.%v.var_patt.cd4)]
// CHECK:STDOUT:   %.loc9_28.3: require_specific_def_type = require_specific_def @T.as.DefaultOrUnformed.impl(%Int.loc9_27.2) [symbolic = %.loc9_28.3 (constants.%.386)]
// CHECK:STDOUT:   %DefaultOrUnformed.lookup_impl_witness: <witness> = lookup_impl_witness %Int.loc9_27.2, @DefaultOrUnformed [symbolic = %DefaultOrUnformed.lookup_impl_witness (constants.%DefaultOrUnformed.lookup_impl_witness)]
// CHECK:STDOUT:   %DefaultOrUnformed.facet.loc9_28.2: %DefaultOrUnformed.type = facet_value %Int.loc9_27.2, (%DefaultOrUnformed.lookup_impl_witness) [symbolic = %DefaultOrUnformed.facet.loc9_28.2 (constants.%DefaultOrUnformed.facet.e73)]
// CHECK:STDOUT:   %DefaultOrUnformed.WithSelf.Op.type: type = fn_type @DefaultOrUnformed.WithSelf.Op, @DefaultOrUnformed.WithSelf(%DefaultOrUnformed.facet.loc9_28.2) [symbolic = %DefaultOrUnformed.WithSelf.Op.type (constants.%DefaultOrUnformed.WithSelf.Op.type.4ac)]
// CHECK:STDOUT:   %.loc9_28.4: type = fn_type_with_self_type %DefaultOrUnformed.WithSelf.Op.type, %DefaultOrUnformed.facet.loc9_28.2 [symbolic = %.loc9_28.4 (constants.%.b74)]
// CHECK:STDOUT:   %impl.elem0.loc9_28.2: @ErrorIfNIsZero.%.loc9_28.4 (%.b74) = impl_witness_access %DefaultOrUnformed.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc9_28.2 (constants.%impl.elem0.b17)]
// CHECK:STDOUT:   %specific_impl_fn.loc9_28.2: <specific function> = specific_impl_function %impl.elem0.loc9_28.2, @DefaultOrUnformed.WithSelf.Op(%DefaultOrUnformed.facet.loc9_28.2) [symbolic = %specific_impl_fn.loc9_28.2 (constants.%specific_impl_fn.cf5)]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness: <witness> = lookup_impl_witness %Int.loc9_27.2, @Destroy [symbolic = %Destroy.lookup_impl_witness (constants.%Destroy.lookup_impl_witness.173)]
// CHECK:STDOUT:   %Destroy.facet.loc9_3.3: %Destroy.type = facet_value %Int.loc9_27.2, (%Destroy.lookup_impl_witness) [symbolic = %Destroy.facet.loc9_3.3 (constants.%Destroy.facet.7bb)]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet.loc9_3.3) [symbolic = %Destroy.WithSelf.Op.type (constants.%Destroy.WithSelf.Op.type.8f1)]
// CHECK:STDOUT:   %.loc9_3.3: type = fn_type_with_self_type %Destroy.WithSelf.Op.type, %Destroy.facet.loc9_3.3 [symbolic = %.loc9_3.3 (constants.%.2c6)]
// CHECK:STDOUT:   %impl.elem0.loc9_3.2: @ErrorIfNIsZero.%.loc9_3.3 (%.2c6) = impl_witness_access %Destroy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc9_3.2 (constants.%impl.elem0.604)]
// CHECK:STDOUT:   %specific_impl_fn.loc9_3.2: <specific function> = specific_impl_function %impl.elem0.loc9_3.2, @Destroy.WithSelf.Op(%Destroy.facet.loc9_3.3) [symbolic = %specific_impl_fn.loc9_3.2 (constants.%specific_impl_fn.693)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %v.patt.loc9_15.1: @ErrorIfNIsZero.%pattern_type (%pattern_type.142) = ref_binding_pattern v [symbolic = %v.patt.loc9_15.2 (constants.%v.patt.834)]
// CHECK:STDOUT:       %v.var_patt.loc9_3.1: @ErrorIfNIsZero.%pattern_type (%pattern_type.142) = var_pattern %v.patt.loc9_15.1 [symbolic = %v.var_patt.loc9_3.2 (constants.%v.var_patt.cd4)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %v.var: ref @ErrorIfNIsZero.%Int.loc9_27.2 (%Int) = var %v.var_patt.loc9_3.1
// CHECK:STDOUT:     %DefaultOrUnformed.facet.loc9_28.1: %DefaultOrUnformed.type = facet_value constants.%Int, (constants.%DefaultOrUnformed.lookup_impl_witness) [symbolic = %DefaultOrUnformed.facet.loc9_28.2 (constants.%DefaultOrUnformed.facet.e73)]
// CHECK:STDOUT:     %.loc9_28.1: %DefaultOrUnformed.type = converted constants.%Int, %DefaultOrUnformed.facet.loc9_28.1 [symbolic = %DefaultOrUnformed.facet.loc9_28.2 (constants.%DefaultOrUnformed.facet.e73)]
// CHECK:STDOUT:     %as_type: type = facet_access_type %.loc9_28.1 [symbolic = %Int.loc9_27.2 (constants.%Int)]
// CHECK:STDOUT:     %.loc9_28.2: type = converted %.loc9_28.1, %as_type [symbolic = %Int.loc9_27.2 (constants.%Int)]
// CHECK:STDOUT:     %impl.elem0.loc9_28.1: @ErrorIfNIsZero.%.loc9_28.4 (%.b74) = impl_witness_access constants.%DefaultOrUnformed.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc9_28.2 (constants.%impl.elem0.b17)]
// CHECK:STDOUT:     %specific_impl_fn.loc9_28.1: <specific function> = specific_impl_function %impl.elem0.loc9_28.1, @DefaultOrUnformed.WithSelf.Op(constants.%DefaultOrUnformed.facet.e73) [symbolic = %specific_impl_fn.loc9_28.2 (constants.%specific_impl_fn.cf5)]
// CHECK:STDOUT:     %DefaultOrUnformed.WithSelf.Op.call: init @ErrorIfNIsZero.%Int.loc9_27.2 (%Int) = call %specific_impl_fn.loc9_28.1()
// CHECK:STDOUT:     assign %v.var, %DefaultOrUnformed.WithSelf.Op.call
// CHECK:STDOUT:     %.loc9_27: type = splice_block %Int.loc9_27.1 [symbolic = %Int.loc9_27.2 (constants.%Int)] {
// CHECK:STDOUT:       %Core.ref.loc9: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:       %Int.ref: %Int.type = name_ref Int, imports.%Core.Int [concrete = constants.%Int.generic]
// CHECK:STDOUT:       %N.ref: Core.IntLiteral = name_ref N, %N.loc4_20.2 [symbolic = %N.loc4_20.1 (constants.%N)]
// CHECK:STDOUT:       %Int.loc9_27.1: type = class_type @Int, @Int(constants.%N) [symbolic = %Int.loc9_27.2 (constants.%Int)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %v: ref @ErrorIfNIsZero.%Int.loc9_27.2 (%Int) = wrapper_binding v, %v.var
// CHECK:STDOUT:     %impl.elem0.loc9_3.1: @ErrorIfNIsZero.%.loc9_3.3 (%.2c6) = impl_witness_access constants.%Destroy.lookup_impl_witness.173, element0 [symbolic = %impl.elem0.loc9_3.2 (constants.%impl.elem0.604)]
// CHECK:STDOUT:     %bound_method.loc9_3.1: <bound method> = bound_method %v.var, %impl.elem0.loc9_3.1
// CHECK:STDOUT:     %Destroy.facet.loc9_3.1: %Destroy.type = facet_value constants.%Int, (constants.%Destroy.lookup_impl_witness.173) [symbolic = %Destroy.facet.loc9_3.3 (constants.%Destroy.facet.7bb)]
// CHECK:STDOUT:     %.loc9_3.1: %Destroy.type = converted constants.%Int, %Destroy.facet.loc9_3.1 [symbolic = %Destroy.facet.loc9_3.3 (constants.%Destroy.facet.7bb)]
// CHECK:STDOUT:     %Destroy.facet.loc9_3.2: %Destroy.type = facet_value constants.%Int, (constants.%Destroy.lookup_impl_witness.173) [symbolic = %Destroy.facet.loc9_3.3 (constants.%Destroy.facet.7bb)]
// CHECK:STDOUT:     %.loc9_3.2: %Destroy.type = converted constants.%Int, %Destroy.facet.loc9_3.2 [symbolic = %Destroy.facet.loc9_3.3 (constants.%Destroy.facet.7bb)]
// CHECK:STDOUT:     %specific_impl_fn.loc9_3.1: <specific function> = specific_impl_function %impl.elem0.loc9_3.1, @Destroy.WithSelf.Op(constants.%Destroy.facet.7bb) [symbolic = %specific_impl_fn.loc9_3.2 (constants.%specific_impl_fn.693)]
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
// CHECK:STDOUT: fn @Destroy.Op(%self.param: ref %i0) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ErrorIfNIsZero(constants.%N) {
// CHECK:STDOUT:   %N.patt.loc4_20.2 => constants.%N.patt
// CHECK:STDOUT:   %N.loc4_20.1 => constants.%N
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ErrorIfNIsZero(constants.%int_0) {
// CHECK:STDOUT:   %N.patt.loc4_20.2 => constants.%N.patt
// CHECK:STDOUT:   %N.loc4_20.1 => constants.%int_0
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Int.loc9_27.2 => constants.%i0
// CHECK:STDOUT:   %require_complete => <error>
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.d0e
// CHECK:STDOUT:   %v.patt.loc9_15.2 => constants.%v.patt.53c
// CHECK:STDOUT:   %v.var_patt.loc9_3.2 => constants.%v.var_patt.0d0
// CHECK:STDOUT:   %.loc9_28.3 => constants.%.060
// CHECK:STDOUT:   %DefaultOrUnformed.lookup_impl_witness => constants.%DefaultOrUnformed.impl_witness.8d2
// CHECK:STDOUT:   %DefaultOrUnformed.facet.loc9_28.2 => constants.%DefaultOrUnformed.facet.4e3
// CHECK:STDOUT:   %DefaultOrUnformed.WithSelf.Op.type => constants.%DefaultOrUnformed.WithSelf.Op.type.7c9
// CHECK:STDOUT:   %.loc9_28.4 => constants.%.21a
// CHECK:STDOUT:   %impl.elem0.loc9_28.2 => constants.%T.as.DefaultOrUnformed.impl.Op.38a
// CHECK:STDOUT:   %specific_impl_fn.loc9_28.2 => constants.%T.as.DefaultOrUnformed.impl.Op.specific_fn
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.df9
// CHECK:STDOUT:   %Destroy.facet.loc9_3.3 => constants.%Destroy.facet.85b
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type => constants.%Destroy.WithSelf.Op.type.b7e
// CHECK:STDOUT:   %.loc9_3.3 => constants.%.338
// CHECK:STDOUT:   %impl.elem0.loc9_3.2 => constants.%Destroy.Op
// CHECK:STDOUT:   %specific_impl_fn.loc9_3.2 => constants.%Destroy.Op
// CHECK:STDOUT: }
// CHECK:STDOUT:
