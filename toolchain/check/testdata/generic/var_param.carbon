// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/generic/var_param.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/generic/var_param.carbon

interface I {
  fn Op(self);
}

impl i32 as I {
  fn Op(unused self) {}
}

//@dump-sem-ir-begin
fn TestOp[T: I](unused var x: T) {}
//@dump-sem-ir-end

fn Test(x: i32) {
  //@dump-sem-ir-begin
  TestOp(x);
  //@dump-sem-ir-end
}

// CHECK:STDOUT: --- var_param.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %N: Core.IntLiteral = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness @i32.as.I.impl.%I.impl_witness_table [concrete]
// CHECK:STDOUT:   %pattern_type.d94: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %I.facet: %I.type = facet_value %i32, (%I.impl_witness) [concrete]
// CHECK:STDOUT:   %i32.builtin: type = int_type signed, %int_32 [concrete]
// CHECK:STDOUT:   %complete_type.f8a: <witness> = complete_type_witness %i32.builtin [concrete]
// CHECK:STDOUT:   %.Self.frozen: type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type.6cb: type = pattern_type %I.type [concrete]
// CHECK:STDOUT:   %T.patt.1df: %pattern_type.6cb = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %T.dda: %I.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.b38: type = facet_access_type %T.dda [symbolic]
// CHECK:STDOUT:   %pattern_type.b3a: type = pattern_type %T.as_type.b38 [symbolic]
// CHECK:STDOUT:   %x.patt.414: %pattern_type.b3a = ref_binding_pattern x [symbolic]
// CHECK:STDOUT:   %x.param_patt.1e1: %pattern_type.b3a = var_param_pattern %x.patt.414 [symbolic]
// CHECK:STDOUT:   %TestOp.type: type = fn_type @TestOp [concrete]
// CHECK:STDOUT:   %TestOp: %TestOp.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.27c: <witness> = require_complete_type %T.as_type.b38 [symbolic]
// CHECK:STDOUT:   %x.patt.b20: %pattern_type.d94 = ref_binding_pattern x [concrete]
// CHECK:STDOUT:   %x.param_patt.ed0: %pattern_type.d94 = var_param_pattern %x.patt.b20 [concrete]
// CHECK:STDOUT:   %TestOp.specific_fn: <specific function> = specific_function %TestOp, @TestOp(%I.facet) [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.a1d: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%N) [symbolic]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.3c1: %Int.as.Copy.impl.Op.type.a1d = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.8af: <witness> = impl_witness imports.%Copy.impl_witness_table.f44, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.648: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.cae: %Int.as.Copy.impl.Op.type.648 = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %i32, (%Copy.impl_witness.8af) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.f76: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet) [concrete]
// CHECK:STDOUT:   %.2b9: type = fn_type_with_self_type %Copy.WithSelf.Op.type.f76, %Copy.facet [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %Int.as.Copy.impl.Op.cae, @Int.as.Copy.impl.Op(%int_32) [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %T.09f: %SubobjectDestroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.type.dea: type = fn_type @T.as_type.as.DestroyImpl.impl.Op, @T.as_type.as.DestroyImpl.impl(%T.09f) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.6e5: %T.as_type.as.DestroyImpl.impl.Op.type.dea = struct_value () [symbolic]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.687: type = fn_type @SubobjectDestroy.WithSelf.Op.loc22 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f: %SubobjectDestroy.WithSelf.Op.type.687 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.49a: %SubobjectDestroy.type = facet_value %i32, (%custom_witness.b80) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.077: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.49a) [concrete]
// CHECK:STDOUT:   %Destroy.facet: %Destroy.type = facet_value %i32, (%DestroyImpl.impl_witness.077, %custom_witness.b80) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.72f: @Int.as.Copy.impl.%Int.as.Copy.impl.Op.type (%Int.as.Copy.impl.Op.type.a1d) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.Copy.impl.%Int.as.Copy.impl.Op (constants.%Int.as.Copy.impl.Op.3c1)]
// CHECK:STDOUT:   %Copy.impl_witness_table.f44 = impl_witness_table (%Core.import_ref.72f), @Int.as.Copy.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.74d: @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op.type (%T.as_type.as.DestroyImpl.impl.Op.type.dea) = import_ref Core//prelude/parts/destroy, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op (constants.%T.as_type.as.DestroyImpl.impl.Op.6e5)]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table = impl_witness_table (%Core.import_ref.74d), @T.as_type.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %TestOp.decl: %TestOp.type = fn_decl @TestOp [concrete = constants.%TestOp] {
// CHECK:STDOUT:     %T.patt.loc22_12.1: %pattern_type.6cb = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc22_12.2 (constants.%T.patt.1df)]
// CHECK:STDOUT:     %x.patt.loc22_29.1: @TestOp.%pattern_type (%pattern_type.b3a) = ref_binding_pattern x [symbolic = %x.patt.loc22_29.2 (constants.%x.patt.414)]
// CHECK:STDOUT:     %x.param_patt.loc22_24.1: @TestOp.%pattern_type (%pattern_type.b3a) = var_param_pattern %x.patt.loc22_29.1 [symbolic = %x.param_patt.loc22_24.2 (constants.%x.param_patt.1e1)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc22_14: type = splice_block %I.ref [concrete = constants.%I.type] {
// CHECK:STDOUT:       %.Self.frozen: type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %I.ref: type = name_ref I, file.%I.decl [concrete = constants.%I.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc22_12.2: %I.type = symbolic_binding T, 0 [symbolic = %T.loc22_12.1 (constants.%T.dda)]
// CHECK:STDOUT:     %x.param: ref @TestOp.%T.as_type.loc22_31.1 (%T.as_type.b38) = ref_param call_param0
// CHECK:STDOUT:     %.loc22_31.1: type = splice_block %.loc22_31.2 [symbolic = %T.as_type.loc22_31.1 (constants.%T.as_type.b38)] {
// CHECK:STDOUT:       %T.ref: %I.type = name_ref T, %T.loc22_12.2 [symbolic = %T.loc22_12.1 (constants.%T.dda)]
// CHECK:STDOUT:       %T.as_type.loc22_31.2: type = facet_access_type %T.ref [symbolic = %T.as_type.loc22_31.1 (constants.%T.as_type.b38)]
// CHECK:STDOUT:       %.loc22_31.2: type = converted %T.ref, %T.as_type.loc22_31.2 [symbolic = %T.as_type.loc22_31.1 (constants.%T.as_type.b38)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: ref @TestOp.%T.as_type.loc22_31.1 (%T.as_type.b38) = wrapper_binding x, %x.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @TestOp(%T.loc22_12.2: %I.type) {
// CHECK:STDOUT:   %T.patt.loc22_12.2: %pattern_type.6cb = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc22_12.2 (constants.%T.patt.1df)]
// CHECK:STDOUT:   %T.loc22_12.1: %I.type = symbolic_binding T, 0 [symbolic = %T.loc22_12.1 (constants.%T.dda)]
// CHECK:STDOUT:   %T.as_type.loc22_31.1: type = facet_access_type %T.loc22_12.1 [symbolic = %T.as_type.loc22_31.1 (constants.%T.as_type.b38)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %T.as_type.loc22_31.1 [symbolic = %pattern_type (constants.%pattern_type.b3a)]
// CHECK:STDOUT:   %x.patt.loc22_29.2: @TestOp.%pattern_type (%pattern_type.b3a) = ref_binding_pattern x [symbolic = %x.patt.loc22_29.2 (constants.%x.patt.414)]
// CHECK:STDOUT:   %x.param_patt.loc22_24.2: @TestOp.%pattern_type (%pattern_type.b3a) = var_param_pattern %x.patt.loc22_29.2 [symbolic = %x.param_patt.loc22_24.2 (constants.%x.param_patt.1e1)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T.as_type.loc22_31.1 [symbolic = %require_complete (constants.%require_complete.27c)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: ref @TestOp.%T.as_type.loc22_31.1 (%T.as_type.b38)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Test(%x.param: %i32) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %TestOp.ref: %TestOp.type = name_ref TestOp, file.%TestOp.decl [concrete = constants.%TestOp]
// CHECK:STDOUT:   %x.ref: %i32 = name_ref x, %x
// CHECK:STDOUT:   %I.facet.loc27_11.1: %I.type = facet_value constants.%i32, (constants.%I.impl_witness) [concrete = constants.%I.facet]
// CHECK:STDOUT:   %.loc27_11.1: %I.type = converted constants.%i32, %I.facet.loc27_11.1 [concrete = constants.%I.facet]
// CHECK:STDOUT:   %I.facet.loc27_11.2: %I.type = facet_value constants.%i32, (constants.%I.impl_witness) [concrete = constants.%I.facet]
// CHECK:STDOUT:   %.loc27_11.2: %I.type = converted constants.%i32, %I.facet.loc27_11.2 [concrete = constants.%I.facet]
// CHECK:STDOUT:   %TestOp.specific_fn: <specific function> = specific_function %TestOp.ref, @TestOp(constants.%I.facet) [concrete = constants.%TestOp.specific_fn]
// CHECK:STDOUT:   %impl.elem0: %.2b9 = impl_witness_access constants.%Copy.impl_witness.8af, element0 [concrete = constants.%Int.as.Copy.impl.Op.cae]
// CHECK:STDOUT:   %bound_method.loc27_10.1: <bound method> = bound_method %x.ref, %impl.elem0
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc27_10.2: <bound method> = bound_method %x.ref, %specific_fn
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call: init %i32 = call %bound_method.loc27_10.2(%x.ref)
// CHECK:STDOUT:   %x.var: ref %i32 = var_storage @TestOp.%x.param_patt.loc22_24.1
// CHECK:STDOUT:   assign %x.var, %Int.as.Copy.impl.Op.call
// CHECK:STDOUT:   %TestOp.call: init %empty_tuple.type = call %TestOp.specific_fn(%x.var)
// CHECK:STDOUT:   %Destroy.facet.loc22_24.1: %Destroy.type = facet_value constants.%i32, (constants.%DestroyImpl.impl_witness.077, constants.%custom_witness.b80) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc22_24.1: %Destroy.type = converted constants.%i32, %Destroy.facet.loc22_24.1 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %Destroy.facet.loc22_24.2: %Destroy.type = facet_value constants.%i32, (constants.%DestroyImpl.impl_witness.077, constants.%custom_witness.b80) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc22_24.2: %Destroy.type = converted constants.%i32, %Destroy.facet.loc22_24.2 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%x.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc22(%self.param: ref %i32) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TestOp(constants.%T.dda) {
// CHECK:STDOUT:   %T.patt.loc22_12.2 => constants.%T.patt.1df
// CHECK:STDOUT:   %T.loc22_12.1 => constants.%T.dda
// CHECK:STDOUT:   %T.as_type.loc22_31.1 => constants.%T.as_type.b38
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.b3a
// CHECK:STDOUT:   %x.patt.loc22_29.2 => constants.%x.patt.414
// CHECK:STDOUT:   %x.param_patt.loc22_24.2 => constants.%x.param_patt.1e1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TestOp(constants.%I.facet) {
// CHECK:STDOUT:   %T.patt.loc22_12.2 => constants.%T.patt.1df
// CHECK:STDOUT:   %T.loc22_12.1 => constants.%I.facet
// CHECK:STDOUT:   %T.as_type.loc22_31.1 => constants.%i32
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.d94
// CHECK:STDOUT:   %x.patt.loc22_29.2 => constants.%x.patt.b20
// CHECK:STDOUT:   %x.param_patt.loc22_24.2 => constants.%x.param_patt.ed0
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.f8a
// CHECK:STDOUT: }
// CHECK:STDOUT:
