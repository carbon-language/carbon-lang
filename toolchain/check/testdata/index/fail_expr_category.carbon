// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/array.carbon
// TODO: Add ranges and switch to "--dump-sem-ir-ranges=only".
// EXTRA-ARGS: --dump-sem-ir-ranges=if-present
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/index/fail_expr_category.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/index/fail_expr_category.carbon

fn F() -> array(i32, 3);

fn G(b: array(i32, 3)) {
  // Indexing an array value gives a value.
  // CHECK:STDERR:        ╭─┤ fail_expr_category.carbon:[[@LINE+8]]:25
  // CHECK:STDERR: ->     │   var unused pb: i32* = &b[0];
  // CHECK:STDERR:        ·                         ┳
  // CHECK:STDERR:        ·                         ┗━━┫ error: cannot take the address of non-reference
  // CHECK:STDERR:        ·                            ┃        expression
  // CHECK:STDERR:        ·                            ┃        [AddrOfNonRef]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  var unused pb: i32* = &b[0];
  // CHECK:STDERR:        ╭─┤ fail_expr_category.carbon:[[@LINE+6]]:3
  // CHECK:STDERR: ->     │   b[0] = 4;
  // CHECK:STDERR:        ·   ━━┳━
  // CHECK:STDERR:        ·     ┗━━┫ error: expression is not assignable [AssignmentToNonAssignable]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  b[0] = 4;

  // Indexing an ephemeral reference (materialized from an initializing
  // expression) gives a value.
  // CHECK:STDERR:        ╭─┤ fail_expr_category.carbon:[[@LINE+8]]:25
  // CHECK:STDERR: ->     │   var unused pf: i32* = &F()[0];
  // CHECK:STDERR:        ·                         ┳
  // CHECK:STDERR:        ·                         ┗━━┫ error: cannot take the address of non-reference
  // CHECK:STDERR:        ·                            ┃        expression
  // CHECK:STDERR:        ·                            ┃        [AddrOfNonRef]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  var unused pf: i32* = &F()[0];
  // CHECK:STDERR:        ╭─┤ fail_expr_category.carbon:[[@LINE+6]]:3
  // CHECK:STDERR: ->     │   F()[0] = 4;
  // CHECK:STDERR:        ·   ━━━┳━━
  // CHECK:STDERR:        ·      ┗━━┫ error: expression is not assignable [AssignmentToNonAssignable]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  F()[0] = 4;
}

// CHECK:STDOUT: --- fail_expr_category.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = parameterized_class_name_type @Int [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %int_3: Core.IntLiteral = int_value 3 [concrete]
// CHECK:STDOUT:   %Array.type: type = parameterized_class_name_type @Array [concrete]
// CHECK:STDOUT:   %Array.generic: %Array.type = struct_value () [concrete]
// CHECK:STDOUT:   %T.67d: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %Array.95c: type = class_type @Array, @Array(%i32, %int_3) [concrete]
// CHECK:STDOUT:   %.d35: Core.Form = init_form %Array.95c [concrete]
// CHECK:STDOUT:   %pattern_type.d4e: type = pattern_type %Array.95c [concrete]
// CHECK:STDOUT:   %return.param_patt.9bd: %pattern_type.d4e = out_param_pattern [concrete]
// CHECK:STDOUT:   %return.patt.0ad: %pattern_type.d4e = return_slot_pattern %return.param_patt.9bd, %Array.95c [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %b.param_patt: %pattern_type.d4e = value_param_pattern [concrete]
// CHECK:STDOUT:   %b.patt: %pattern_type.d4e = wrapper_binding_pattern b, %b.param_patt [concrete]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.a17: type = ptr_type %i32 [concrete]
// CHECK:STDOUT:   %pattern_type.7dc: type = pattern_type %ptr.a17 [concrete]
// CHECK:STDOUT:   %pb.patt: %pattern_type.7dc = ref_binding_pattern pb [concrete]
// CHECK:STDOUT:   %pb.var_patt: %pattern_type.7dc = var_pattern %pb.patt [concrete]
// CHECK:STDOUT:   %int_0.5c6: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.0ff: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.0ff = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.6b0: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.3c6: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.6f2: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.3c6 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.d73: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.f53, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.766: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.f60: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.766 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.6b0 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.d73) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.58f: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%i32, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.33a: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.58f, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.762: <bound method> = bound_method %int_0.5c6, %Core.IntLiteral.as.ImplicitAs.impl.Convert.f60 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.f60, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.036: <bound method> = bound_method %int_0.5c6, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_0.753: %i32 = int_value 0 [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.type.20f: type = fn_type @ptr.as.Copy.impl.Op, @ptr.as.Copy.impl(%T.67d) [symbolic]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.f12: %ptr.as.Copy.impl.Op.type.20f = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.83a: <witness> = impl_witness imports.%Copy.impl_witness_table.fb0, @ptr.as.Copy.impl(%i32) [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.type.dfc: type = fn_type @ptr.as.Copy.impl.Op, @ptr.as.Copy.impl(%i32) [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.81f: %ptr.as.Copy.impl.Op.type.dfc = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %ptr.a17, (%Copy.impl_witness.83a) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.9a2: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet) [concrete]
// CHECK:STDOUT:   %.309: type = fn_type_with_self_type %Copy.WithSelf.Op.type.9a2, %Copy.facet [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %ptr.as.Copy.impl.Op.81f, @ptr.as.Copy.impl.Op(%i32) [concrete]
// CHECK:STDOUT:   %int_4.0c1: Core.IntLiteral = int_value 4 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.776: <bound method> = bound_method %int_4.0c1, %Core.IntLiteral.as.ImplicitAs.impl.Convert.f60 [concrete]
// CHECK:STDOUT:   %bound_method.7fa: <bound method> = bound_method %int_4.0c1, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_4.6bf: %i32 = int_value 4 [concrete]
// CHECK:STDOUT:   %pf.patt: %pattern_type.7dc = ref_binding_pattern pf [concrete]
// CHECK:STDOUT:   %pf.var_patt: %pattern_type.7dc = var_pattern %pf.patt [concrete]
// CHECK:STDOUT:   %SelfDestruct.type: type = fn_type @SelfDestruct [concrete]
// CHECK:STDOUT:   %SelfDestruct: %SelfDestruct.type = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %T.09f: %SubobjectDestroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.type.dea: type = fn_type @T.as_type.as.DestroyImpl.impl.Op, @T.as_type.as.DestroyImpl.impl(%T.09f) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.6e5: %T.as_type.as.DestroyImpl.impl.Op.type.dea = struct_value () [symbolic]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.1: type = fn_type @SubobjectDestroy.WithSelf.Op.loc53_5.1 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.1: %SubobjectDestroy.WithSelf.Op.type.6870fd.1 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.1: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.1), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.2: type = fn_type @SubobjectDestroy.WithSelf.Op.loc53_5.2 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.2: %SubobjectDestroy.WithSelf.Op.type.6870fd.2 = struct_value () [concrete]
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.49a: %SubobjectDestroy.type = facet_value %i32, (%custom_witness.b80ef6.1) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.077: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.49a) [concrete]
// CHECK:STDOUT:   %Destroy.facet.c2b: %Destroy.type = facet_value %i32, (%DestroyImpl.impl_witness.077, %custom_witness.b80ef6.1) [concrete]
// CHECK:STDOUT:   %SelfDestruct.specific_fn.845: <specific function> = specific_function %SelfDestruct, @SelfDestruct(%Destroy.facet.c2b) [concrete]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.2: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.2), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.79c: %SubobjectDestroy.type = facet_value %Array.95c, (%custom_witness.b80ef6.2) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.90e: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.79c) [concrete]
// CHECK:STDOUT:   %Destroy.facet.416: %Destroy.type = facet_value %Array.95c, (%DestroyImpl.impl_witness.90e, %custom_witness.b80ef6.2) [concrete]
// CHECK:STDOUT:   %SelfDestruct.specific_fn.cd8: <specific function> = specific_function %SelfDestruct, @SelfDestruct(%Destroy.facet.416) [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.3: type = fn_type @SubobjectDestroy.WithSelf.Op.loc46 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.3: %SubobjectDestroy.WithSelf.Op.type.6870fd.3 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.3: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.3), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.d4d: %SubobjectDestroy.type = facet_value %ptr.a17, (%custom_witness.b80ef6.3) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.b8f: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.d4d) [concrete]
// CHECK:STDOUT:   %Destroy.facet.abb: %Destroy.type = facet_value %ptr.a17, (%DestroyImpl.impl_witness.b8f, %custom_witness.b80ef6.3) [concrete]
// CHECK:STDOUT:   %SelfDestruct.specific_fn.690: <specific function> = specific_function %SelfDestruct, @SelfDestruct(%Destroy.facet.abb) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .Array = %Core.Array
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     .Copy = %Core.Copy
// CHECK:STDOUT:     .SelfDestruct = %Core.SelfDestruct
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/parts/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT:   %Core.Array: %Array.type = import_ref Core//prelude/parts/array, Array, loaded [concrete = constants.%Array.generic]
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.0ff = import_ref Core//prelude/parts/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:   %Core.import_ref.e474: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.3c6) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.6f2)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.f53 = impl_witness_table (%Core.import_ref.e474), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %Core.Copy: type = import_ref Core//prelude/parts/copy, Copy, loaded [concrete = constants.%Copy.type]
// CHECK:STDOUT:   %Core.import_ref.06a: @ptr.as.Copy.impl.%ptr.as.Copy.impl.Op.type (%ptr.as.Copy.impl.Op.type.20f) = import_ref Core//prelude/parts/copy, loc{{\d+_\d+}}, loaded [symbolic = @ptr.as.Copy.impl.%ptr.as.Copy.impl.Op (constants.%ptr.as.Copy.impl.Op.f12)]
// CHECK:STDOUT:   %Copy.impl_witness_table.fb0 = impl_witness_table (%Core.import_ref.06a), @ptr.as.Copy.impl [concrete]
// CHECK:STDOUT:   %Core.SelfDestruct: %SelfDestruct.type = import_ref Core//prelude/parts/destroy, SelfDestruct, loaded [concrete = constants.%SelfDestruct]
// CHECK:STDOUT:   %Core.import_ref.74d: @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op.type (%T.as_type.as.DestroyImpl.impl.Op.type.dea) = import_ref Core//prelude/parts/destroy, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op (constants.%T.as_type.as.DestroyImpl.impl.Op.6e5)]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table = impl_witness_table (%Core.import_ref.74d), @T.as_type.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:     .G = %G.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %return.param_patt: %pattern_type.d4e = out_param_pattern [concrete = constants.%return.param_patt.9bd]
// CHECK:STDOUT:     %return.patt: %pattern_type.d4e = return_slot_pattern %return.param_patt, %Array [concrete = constants.%return.patt.0ad]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %int_3: Core.IntLiteral = int_value 3 [concrete = constants.%int_3]
// CHECK:STDOUT:     %Array: type = class_type @Array, @Array(constants.%i32, constants.%int_3) [concrete = constants.%Array.95c]
// CHECK:STDOUT:     %.loc15: Core.Form = init_form %Array [concrete = constants.%.d35]
// CHECK:STDOUT:     %return.param: ref %Array.95c = out_param call_param0
// CHECK:STDOUT:     %return: ref %Array.95c = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {
// CHECK:STDOUT:     %b.param_patt: %pattern_type.d4e = value_param_pattern [concrete = constants.%b.param_patt]
// CHECK:STDOUT:     %b.patt: %pattern_type.d4e = wrapper_binding_pattern b, %b.param_patt [concrete = constants.%b.patt]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %b.param: %Array.95c = value_param call_param0
// CHECK:STDOUT:     %.loc17: type = splice_block %Array [concrete = constants.%Array.95c] {
// CHECK:STDOUT:       %i32.loc17: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:       %int_3: Core.IntLiteral = int_value 3 [concrete = constants.%int_3]
// CHECK:STDOUT:       %Array: type = class_type @Array, @Array(constants.%i32, constants.%int_3) [concrete = constants.%Array.95c]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %b: %Array.95c = wrapper_binding b, %b.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() -> out %return.param: %Array.95c;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G(%b.param: %Array.95c) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %pb.var: ref %ptr.a17 = var_storage %pb.var_patt
// CHECK:STDOUT:   %b.ref.loc27: %Array.95c = name_ref b, %b
// CHECK:STDOUT:   %int_0.loc27: Core.IntLiteral = int_value 0 [concrete = constants.%int_0.5c6]
// CHECK:STDOUT:   %impl.elem0.loc27_28: %.33a = impl_witness_access constants.%ImplicitAs.impl_witness.d73, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f60]
// CHECK:STDOUT:   %bound_method.loc27_28.1: <bound method> = bound_method %int_0.loc27, %impl.elem0.loc27_28 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.762]
// CHECK:STDOUT:   %specific_fn.loc27_28: <specific function> = specific_function %impl.elem0.loc27_28, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc27_28.2: <bound method> = bound_method %int_0.loc27, %specific_fn.loc27_28 [concrete = constants.%bound_method.036]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc27: init %i32 = call %bound_method.loc27_28.2(%int_0.loc27) [concrete = constants.%int_0.753]
// CHECK:STDOUT:   %.loc27_28.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc27 [concrete = constants.%int_0.753]
// CHECK:STDOUT:   %.loc27_28.2: %i32 = converted %int_0.loc27, %.loc27_28.1 [concrete = constants.%int_0.753]
// CHECK:STDOUT:   %.loc27_29.1: ref %Array.95c = value_as_ref %b.ref.loc27
// CHECK:STDOUT:   %.loc27_29.2: ref %i32 = array_index %.loc27_29.1, %.loc27_28.2
// CHECK:STDOUT:   %.loc27_29.3: %i32 = acquire_value %.loc27_29.2
// CHECK:STDOUT:   %addr.loc27: %ptr.a17 = addr_of <error> [concrete = <error>]
// CHECK:STDOUT:   %impl.elem0.loc27_25: %.309 = impl_witness_access constants.%Copy.impl_witness.83a, element0 [concrete = constants.%ptr.as.Copy.impl.Op.81f]
// CHECK:STDOUT:   %bound_method.loc27_25.1: <bound method> = bound_method %addr.loc27, %impl.elem0.loc27_25 [concrete = <error>]
// CHECK:STDOUT:   %specific_fn.loc27_25: <specific function> = specific_function %impl.elem0.loc27_25, @ptr.as.Copy.impl.Op(constants.%i32) [concrete = constants.%ptr.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc27_25.2: <bound method> = bound_method %addr.loc27, %specific_fn.loc27_25 [concrete = <error>]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.call.loc27: init %ptr.a17 = call %bound_method.loc27_25.2(%addr.loc27) [concrete = <error>]
// CHECK:STDOUT:   assign %pb.var, %ptr.as.Copy.impl.Op.call.loc27
// CHECK:STDOUT:   %.loc27_21: type = splice_block %ptr.loc27 [concrete = constants.%ptr.a17] {
// CHECK:STDOUT:     %i32.loc27: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %ptr.loc27: type = ptr_type %i32.loc27 [concrete = constants.%ptr.a17]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %pb: ref %ptr.a17 = wrapper_binding pb, %pb.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %pb.patt: %pattern_type.7dc = ref_binding_pattern pb [concrete = constants.%pb.patt]
// CHECK:STDOUT:     %pb.var_patt: %pattern_type.7dc = var_pattern %pb.patt [concrete = constants.%pb.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b.ref.loc34: %Array.95c = name_ref b, %b
// CHECK:STDOUT:   %int_0.loc34: Core.IntLiteral = int_value 0 [concrete = constants.%int_0.5c6]
// CHECK:STDOUT:   %impl.elem0.loc34_5: %.33a = impl_witness_access constants.%ImplicitAs.impl_witness.d73, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f60]
// CHECK:STDOUT:   %bound_method.loc34_5.1: <bound method> = bound_method %int_0.loc34, %impl.elem0.loc34_5 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.762]
// CHECK:STDOUT:   %specific_fn.loc34_5: <specific function> = specific_function %impl.elem0.loc34_5, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc34_5.2: <bound method> = bound_method %int_0.loc34, %specific_fn.loc34_5 [concrete = constants.%bound_method.036]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc34_5: init %i32 = call %bound_method.loc34_5.2(%int_0.loc34) [concrete = constants.%int_0.753]
// CHECK:STDOUT:   %.loc34_5.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc34_5 [concrete = constants.%int_0.753]
// CHECK:STDOUT:   %.loc34_5.2: %i32 = converted %int_0.loc34, %.loc34_5.1 [concrete = constants.%int_0.753]
// CHECK:STDOUT:   %.loc34_6.1: ref %Array.95c = value_as_ref %b.ref.loc34
// CHECK:STDOUT:   %.loc34_6.2: ref %i32 = array_index %.loc34_6.1, %.loc34_5.2
// CHECK:STDOUT:   %.loc34_6.3: %i32 = acquire_value %.loc34_6.2
// CHECK:STDOUT:   %int_4.loc34: Core.IntLiteral = int_value 4 [concrete = constants.%int_4.0c1]
// CHECK:STDOUT:   %impl.elem0.loc34_8: %.33a = impl_witness_access constants.%ImplicitAs.impl_witness.d73, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f60]
// CHECK:STDOUT:   %bound_method.loc34_8.1: <bound method> = bound_method %int_4.loc34, %impl.elem0.loc34_8 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.776]
// CHECK:STDOUT:   %specific_fn.loc34_8: <specific function> = specific_function %impl.elem0.loc34_8, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc34_8.2: <bound method> = bound_method %int_4.loc34, %specific_fn.loc34_8 [concrete = constants.%bound_method.7fa]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc34_8: init %i32 = call %bound_method.loc34_8.2(%int_4.loc34) [concrete = constants.%int_4.6bf]
// CHECK:STDOUT:   %.loc34_8: init %i32 = converted %int_4.loc34, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc34_8 [concrete = constants.%int_4.6bf]
// CHECK:STDOUT:   assign %.loc34_6.3, %.loc34_8
// CHECK:STDOUT:   %pf.var: ref %ptr.a17 = var_storage %pf.var_patt
// CHECK:STDOUT:   %F.ref.loc46: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:   %.loc46_28.1: ref %Array.95c = temporary_storage
// CHECK:STDOUT:   %F.call.loc46: init %Array.95c to %.loc46_28.1 = call %F.ref.loc46()
// CHECK:STDOUT:   %int_0.loc46: Core.IntLiteral = int_value 0 [concrete = constants.%int_0.5c6]
// CHECK:STDOUT:   %.loc46_28.2: ref %Array.95c = temporary %.loc46_28.1, %F.call.loc46
// CHECK:STDOUT:   %impl.elem0.loc46_30: %.33a = impl_witness_access constants.%ImplicitAs.impl_witness.d73, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f60]
// CHECK:STDOUT:   %bound_method.loc46_30.1: <bound method> = bound_method %int_0.loc46, %impl.elem0.loc46_30 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.762]
// CHECK:STDOUT:   %specific_fn.loc46_30: <specific function> = specific_function %impl.elem0.loc46_30, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc46_30.2: <bound method> = bound_method %int_0.loc46, %specific_fn.loc46_30 [concrete = constants.%bound_method.036]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc46: init %i32 = call %bound_method.loc46_30.2(%int_0.loc46) [concrete = constants.%int_0.753]
// CHECK:STDOUT:   %.loc46_30.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc46 [concrete = constants.%int_0.753]
// CHECK:STDOUT:   %.loc46_30.2: %i32 = converted %int_0.loc46, %.loc46_30.1 [concrete = constants.%int_0.753]
// CHECK:STDOUT:   %.loc46_31.1: ref %i32 = array_index %.loc46_28.2, %.loc46_30.2
// CHECK:STDOUT:   %.loc46_31.2: %i32 = acquire_value %.loc46_31.1
// CHECK:STDOUT:   %addr.loc46: %ptr.a17 = addr_of <error> [concrete = <error>]
// CHECK:STDOUT:   %impl.elem0.loc46_25: %.309 = impl_witness_access constants.%Copy.impl_witness.83a, element0 [concrete = constants.%ptr.as.Copy.impl.Op.81f]
// CHECK:STDOUT:   %bound_method.loc46_25.1: <bound method> = bound_method %addr.loc46, %impl.elem0.loc46_25 [concrete = <error>]
// CHECK:STDOUT:   %specific_fn.loc46_25: <specific function> = specific_function %impl.elem0.loc46_25, @ptr.as.Copy.impl.Op(constants.%i32) [concrete = constants.%ptr.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc46_25.2: <bound method> = bound_method %addr.loc46, %specific_fn.loc46_25 [concrete = <error>]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.call.loc46: init %ptr.a17 = call %bound_method.loc46_25.2(%addr.loc46) [concrete = <error>]
// CHECK:STDOUT:   assign %pf.var, %ptr.as.Copy.impl.Op.call.loc46
// CHECK:STDOUT:   %.loc46_21: type = splice_block %ptr.loc46 [concrete = constants.%ptr.a17] {
// CHECK:STDOUT:     %i32.loc46: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %ptr.loc46: type = ptr_type %i32.loc46 [concrete = constants.%ptr.a17]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %pf: ref %ptr.a17 = wrapper_binding pf, %pf.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %pf.patt: %pattern_type.7dc = ref_binding_pattern pf [concrete = constants.%pf.patt]
// CHECK:STDOUT:     %pf.var_patt: %pattern_type.7dc = var_pattern %pf.patt [concrete = constants.%pf.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.ref.loc53: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:   %.loc53_5.1: ref %Array.95c = temporary_storage
// CHECK:STDOUT:   %F.call.loc53: init %Array.95c to %.loc53_5.1 = call %F.ref.loc53()
// CHECK:STDOUT:   %int_0.loc53: Core.IntLiteral = int_value 0 [concrete = constants.%int_0.5c6]
// CHECK:STDOUT:   %.loc53_5.2: ref %Array.95c = temporary %.loc53_5.1, %F.call.loc53
// CHECK:STDOUT:   %impl.elem0.loc53_7: %.33a = impl_witness_access constants.%ImplicitAs.impl_witness.d73, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f60]
// CHECK:STDOUT:   %bound_method.loc53_7.1: <bound method> = bound_method %int_0.loc53, %impl.elem0.loc53_7 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.762]
// CHECK:STDOUT:   %specific_fn.loc53_7: <specific function> = specific_function %impl.elem0.loc53_7, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc53_7.2: <bound method> = bound_method %int_0.loc53, %specific_fn.loc53_7 [concrete = constants.%bound_method.036]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc53_7: init %i32 = call %bound_method.loc53_7.2(%int_0.loc53) [concrete = constants.%int_0.753]
// CHECK:STDOUT:   %.loc53_7.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc53_7 [concrete = constants.%int_0.753]
// CHECK:STDOUT:   %.loc53_7.2: %i32 = converted %int_0.loc53, %.loc53_7.1 [concrete = constants.%int_0.753]
// CHECK:STDOUT:   %.loc53_8.1: ref %i32 = array_index %.loc53_5.2, %.loc53_7.2
// CHECK:STDOUT:   %.loc53_8.2: %i32 = acquire_value %.loc53_8.1
// CHECK:STDOUT:   %int_4.loc53: Core.IntLiteral = int_value 4 [concrete = constants.%int_4.0c1]
// CHECK:STDOUT:   %impl.elem0.loc53_10: %.33a = impl_witness_access constants.%ImplicitAs.impl_witness.d73, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f60]
// CHECK:STDOUT:   %bound_method.loc53_10.1: <bound method> = bound_method %int_4.loc53, %impl.elem0.loc53_10 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.776]
// CHECK:STDOUT:   %specific_fn.loc53_10: <specific function> = specific_function %impl.elem0.loc53_10, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc53_10.2: <bound method> = bound_method %int_4.loc53, %specific_fn.loc53_10 [concrete = constants.%bound_method.7fa]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc53_10: init %i32 = call %bound_method.loc53_10.2(%int_4.loc53) [concrete = constants.%int_4.6bf]
// CHECK:STDOUT:   %.loc53_10: init %i32 = converted %int_4.loc53, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc53_10 [concrete = constants.%int_4.6bf]
// CHECK:STDOUT:   assign %.loc53_8.2, %.loc53_10
// CHECK:STDOUT:   %Destroy.facet.loc53_5.1: %Destroy.type = facet_value constants.%Array.95c, (constants.%DestroyImpl.impl_witness.90e, constants.%custom_witness.b80ef6.2) [concrete = constants.%Destroy.facet.416]
// CHECK:STDOUT:   %.loc53_5.3: %Destroy.type = converted constants.%Array.95c, %Destroy.facet.loc53_5.1 [concrete = constants.%Destroy.facet.416]
// CHECK:STDOUT:   %Destroy.facet.loc53_5.2: %Destroy.type = facet_value constants.%Array.95c, (constants.%DestroyImpl.impl_witness.90e, constants.%custom_witness.b80ef6.2) [concrete = constants.%Destroy.facet.416]
// CHECK:STDOUT:   %.loc53_5.4: %Destroy.type = converted constants.%Array.95c, %Destroy.facet.loc53_5.2 [concrete = constants.%Destroy.facet.416]
// CHECK:STDOUT:   %SelfDestruct.specific_fn.1: <specific function> = specific_function constants.%SelfDestruct, @SelfDestruct(constants.%Destroy.facet.416) [concrete = constants.%SelfDestruct.specific_fn.cd8]
// CHECK:STDOUT:   %SelfDestruct.call.loc53: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%.loc53_5.2)
// CHECK:STDOUT:   %Destroy.facet.loc46_28.1: %Destroy.type = facet_value constants.%Array.95c, (constants.%DestroyImpl.impl_witness.90e, constants.%custom_witness.b80ef6.2) [concrete = constants.%Destroy.facet.416]
// CHECK:STDOUT:   %.loc46_28.3: %Destroy.type = converted constants.%Array.95c, %Destroy.facet.loc46_28.1 [concrete = constants.%Destroy.facet.416]
// CHECK:STDOUT:   %Destroy.facet.loc46_28.2: %Destroy.type = facet_value constants.%Array.95c, (constants.%DestroyImpl.impl_witness.90e, constants.%custom_witness.b80ef6.2) [concrete = constants.%Destroy.facet.416]
// CHECK:STDOUT:   %.loc46_28.4: %Destroy.type = converted constants.%Array.95c, %Destroy.facet.loc46_28.2 [concrete = constants.%Destroy.facet.416]
// CHECK:STDOUT:   %SelfDestruct.specific_fn.2: <specific function> = specific_function constants.%SelfDestruct, @SelfDestruct(constants.%Destroy.facet.416) [concrete = constants.%SelfDestruct.specific_fn.cd8]
// CHECK:STDOUT:   %SelfDestruct.call.loc46_28: init %empty_tuple.type = call %SelfDestruct.specific_fn.2(%.loc46_28.2)
// CHECK:STDOUT:   %Destroy.facet.loc46_3.1: %Destroy.type = facet_value constants.%ptr.a17, (constants.%DestroyImpl.impl_witness.b8f, constants.%custom_witness.b80ef6.3) [concrete = constants.%Destroy.facet.abb]
// CHECK:STDOUT:   %.loc46_3.1: %Destroy.type = converted constants.%ptr.a17, %Destroy.facet.loc46_3.1 [concrete = constants.%Destroy.facet.abb]
// CHECK:STDOUT:   %Destroy.facet.loc46_3.2: %Destroy.type = facet_value constants.%ptr.a17, (constants.%DestroyImpl.impl_witness.b8f, constants.%custom_witness.b80ef6.3) [concrete = constants.%Destroy.facet.abb]
// CHECK:STDOUT:   %.loc46_3.2: %Destroy.type = converted constants.%ptr.a17, %Destroy.facet.loc46_3.2 [concrete = constants.%Destroy.facet.abb]
// CHECK:STDOUT:   %SelfDestruct.specific_fn.3: <specific function> = specific_function constants.%SelfDestruct, @SelfDestruct(constants.%Destroy.facet.abb) [concrete = constants.%SelfDestruct.specific_fn.690]
// CHECK:STDOUT:   %SelfDestruct.call.loc46_3: init %empty_tuple.type = call %SelfDestruct.specific_fn.3(%pf.var)
// CHECK:STDOUT:   %Destroy.facet.loc27_3.1: %Destroy.type = facet_value constants.%ptr.a17, (constants.%DestroyImpl.impl_witness.b8f, constants.%custom_witness.b80ef6.3) [concrete = constants.%Destroy.facet.abb]
// CHECK:STDOUT:   %.loc27_3.1: %Destroy.type = converted constants.%ptr.a17, %Destroy.facet.loc27_3.1 [concrete = constants.%Destroy.facet.abb]
// CHECK:STDOUT:   %Destroy.facet.loc27_3.2: %Destroy.type = facet_value constants.%ptr.a17, (constants.%DestroyImpl.impl_witness.b8f, constants.%custom_witness.b80ef6.3) [concrete = constants.%Destroy.facet.abb]
// CHECK:STDOUT:   %.loc27_3.2: %Destroy.type = converted constants.%ptr.a17, %Destroy.facet.loc27_3.2 [concrete = constants.%Destroy.facet.abb]
// CHECK:STDOUT:   %SelfDestruct.specific_fn.4: <specific function> = specific_function constants.%SelfDestruct, @SelfDestruct(constants.%Destroy.facet.abb) [concrete = constants.%SelfDestruct.specific_fn.690]
// CHECK:STDOUT:   %SelfDestruct.call.loc27: init %empty_tuple.type = call %SelfDestruct.specific_fn.4(%pb.var)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc53_5.1(%self.param: ref %i32) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc53_5.2(%self.param: ref %Array.95c) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [concrete = constants.%int_2]
// CHECK:STDOUT:   %.loc53_5.1: ref %i32 = array_index %self.param, %int_2
// CHECK:STDOUT:   %Destroy.facet.loc53_5.1: %Destroy.type = facet_value constants.%i32, (constants.%DestroyImpl.impl_witness.077, constants.%custom_witness.b80ef6.1) [concrete = constants.%Destroy.facet.c2b]
// CHECK:STDOUT:   %.loc53_5.2: %Destroy.type = converted constants.%i32, %Destroy.facet.loc53_5.1 [concrete = constants.%Destroy.facet.c2b]
// CHECK:STDOUT:   %Destroy.facet.loc53_5.2: %Destroy.type = facet_value constants.%i32, (constants.%DestroyImpl.impl_witness.077, constants.%custom_witness.b80ef6.1) [concrete = constants.%Destroy.facet.c2b]
// CHECK:STDOUT:   %.loc53_5.3: %Destroy.type = converted constants.%i32, %Destroy.facet.loc53_5.2 [concrete = constants.%Destroy.facet.c2b]
// CHECK:STDOUT:   %SelfDestruct.specific_fn.1: <specific function> = specific_function constants.%SelfDestruct, @SelfDestruct(constants.%Destroy.facet.c2b) [concrete = constants.%SelfDestruct.specific_fn.845]
// CHECK:STDOUT:   %SelfDestruct.call.loc53_5.1: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%.loc53_5.1)
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1]
// CHECK:STDOUT:   %.loc53_5.4: ref %i32 = array_index %self.param, %int_1
// CHECK:STDOUT:   %Destroy.facet.loc53_5.3: %Destroy.type = facet_value constants.%i32, (constants.%DestroyImpl.impl_witness.077, constants.%custom_witness.b80ef6.1) [concrete = constants.%Destroy.facet.c2b]
// CHECK:STDOUT:   %.loc53_5.5: %Destroy.type = converted constants.%i32, %Destroy.facet.loc53_5.3 [concrete = constants.%Destroy.facet.c2b]
// CHECK:STDOUT:   %Destroy.facet.loc53_5.4: %Destroy.type = facet_value constants.%i32, (constants.%DestroyImpl.impl_witness.077, constants.%custom_witness.b80ef6.1) [concrete = constants.%Destroy.facet.c2b]
// CHECK:STDOUT:   %.loc53_5.6: %Destroy.type = converted constants.%i32, %Destroy.facet.loc53_5.4 [concrete = constants.%Destroy.facet.c2b]
// CHECK:STDOUT:   %SelfDestruct.specific_fn.2: <specific function> = specific_function constants.%SelfDestruct, @SelfDestruct(constants.%Destroy.facet.c2b) [concrete = constants.%SelfDestruct.specific_fn.845]
// CHECK:STDOUT:   %SelfDestruct.call.loc53_5.2: init %empty_tuple.type = call %SelfDestruct.specific_fn.2(%.loc53_5.4)
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0.5c6]
// CHECK:STDOUT:   %.loc53_5.7: ref %i32 = array_index %self.param, %int_0
// CHECK:STDOUT:   %Destroy.facet.loc53_5.5: %Destroy.type = facet_value constants.%i32, (constants.%DestroyImpl.impl_witness.077, constants.%custom_witness.b80ef6.1) [concrete = constants.%Destroy.facet.c2b]
// CHECK:STDOUT:   %.loc53_5.8: %Destroy.type = converted constants.%i32, %Destroy.facet.loc53_5.5 [concrete = constants.%Destroy.facet.c2b]
// CHECK:STDOUT:   %Destroy.facet.loc53_5.6: %Destroy.type = facet_value constants.%i32, (constants.%DestroyImpl.impl_witness.077, constants.%custom_witness.b80ef6.1) [concrete = constants.%Destroy.facet.c2b]
// CHECK:STDOUT:   %.loc53_5.9: %Destroy.type = converted constants.%i32, %Destroy.facet.loc53_5.6 [concrete = constants.%Destroy.facet.c2b]
// CHECK:STDOUT:   %SelfDestruct.specific_fn.3: <specific function> = specific_function constants.%SelfDestruct, @SelfDestruct(constants.%Destroy.facet.c2b) [concrete = constants.%SelfDestruct.specific_fn.845]
// CHECK:STDOUT:   %SelfDestruct.call.loc53_5.3: init %empty_tuple.type = call %SelfDestruct.specific_fn.3(%.loc53_5.7)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc46(%self.param: ref %ptr.a17) = "no_op";
// CHECK:STDOUT:
