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
  fn Op[self: Self]();
}

impl i32 as I {
  fn Op[unused self: Self]() {}
}

//@dump-sem-ir-begin
fn TestOp[T:! I](unused var x: T) {}
//@dump-sem-ir-end

fn Test(x: i32) {
  //@dump-sem-ir-begin
  TestOp(x);
  //@dump-sem-ir-end
}

// CHECK:STDOUT: --- var_param.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %N: Core.IntLiteral = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness @i32.as.I.impl.%I.impl_witness_table [concrete]
// CHECK:STDOUT:   %pattern_type.7ce: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %I.facet: %I.type = facet_value %i32, (%I.impl_witness) [concrete]
// CHECK:STDOUT:   %i32.builtin: type = int_type signed, %int_32 [concrete]
// CHECK:STDOUT:   %complete_type.f8a: <witness> = complete_type_witness %i32.builtin [concrete]
// CHECK:STDOUT:   %pattern_type.9d9: type = pattern_type %I.type [concrete]
// CHECK:STDOUT:   %T.651: %I.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.binding.as_type.3af: type = symbolic_binding_type T, 0, %T.651 [symbolic]
// CHECK:STDOUT:   %pattern_type.422: type = pattern_type %T.binding.as_type.3af [symbolic]
// CHECK:STDOUT:   %TestOp.type: type = fn_type @TestOp [concrete]
// CHECK:STDOUT:   %TestOp: %TestOp.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.2b8: <witness> = require_complete_type %T.binding.as_type.3af [symbolic]
// CHECK:STDOUT:   %TestOp.specific_fn: <specific function> = specific_function %TestOp, @TestOp(%I.facet) [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.824: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%N) [symbolic]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.9b9: %Int.as.Copy.impl.Op.type.824 = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.f17: <witness> = impl_witness imports.%Copy.impl_witness_table.e76, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.546: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.664: %Int.as.Copy.impl.Op.type.546 = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %i32, (%Copy.impl_witness.f17) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.081: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet) [concrete]
// CHECK:STDOUT:   %.8e2: type = fn_type_with_self_type %Copy.WithSelf.Op.type.081, %Copy.facet [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %Int.as.Copy.impl.Op.664, @Int.as.Copy.impl.Op(%int_32) [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.bae255.2: type = fn_type @Destroy.Op.loc22_25.2 [concrete]
// CHECK:STDOUT:   %Destroy.Op.651ba6.2: %Destroy.Op.type.bae255.2 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.18d: @Int.as.Copy.impl.%Int.as.Copy.impl.Op.type (%Int.as.Copy.impl.Op.type.824) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.Copy.impl.%Int.as.Copy.impl.Op (constants.%Int.as.Copy.impl.Op.9b9)]
// CHECK:STDOUT:   %Copy.impl_witness_table.e76 = impl_witness_table (%Core.import_ref.18d), @Int.as.Copy.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %TestOp.decl: %TestOp.type = fn_decl @TestOp [concrete = constants.%TestOp] {
// CHECK:STDOUT:     %T.patt: %pattern_type.9d9 = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:     %x.patt: @TestOp.%pattern_type (%pattern_type.422) = ref_binding_pattern x [concrete]
// CHECK:STDOUT:     %x.param_patt: @TestOp.%pattern_type (%pattern_type.422) = var_param_pattern %x.patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc22_15: type = splice_block %I.ref [concrete = constants.%I.type] {
// CHECK:STDOUT:       <elided>
// CHECK:STDOUT:       %I.ref: type = name_ref I, file.%I.decl [concrete = constants.%I.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc22_12.2: %I.type = symbolic_binding T, 0 [symbolic = %T.loc22_12.1 (constants.%T.651)]
// CHECK:STDOUT:     %x.param: ref @TestOp.%T.binding.as_type (%T.binding.as_type.3af) = ref_param call_param0
// CHECK:STDOUT:     %.loc22_32.1: type = splice_block %.loc22_32.2 [symbolic = %T.binding.as_type (constants.%T.binding.as_type.3af)] {
// CHECK:STDOUT:       %T.ref: %I.type = name_ref T, %T.loc22_12.2 [symbolic = %T.loc22_12.1 (constants.%T.651)]
// CHECK:STDOUT:       %T.as_type: type = facet_access_type %T.ref [symbolic = %T.binding.as_type (constants.%T.binding.as_type.3af)]
// CHECK:STDOUT:       %.loc22_32.2: type = converted %T.ref, %T.as_type [symbolic = %T.binding.as_type (constants.%T.binding.as_type.3af)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: ref @TestOp.%T.binding.as_type (%T.binding.as_type.3af) = ref_binding x, %x.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @TestOp(%T.loc22_12.2: %I.type) {
// CHECK:STDOUT:   %T.loc22_12.1: %I.type = symbolic_binding T, 0 [symbolic = %T.loc22_12.1 (constants.%T.651)]
// CHECK:STDOUT:   %T.binding.as_type: type = symbolic_binding_type T, 0, %T.loc22_12.1 [symbolic = %T.binding.as_type (constants.%T.binding.as_type.3af)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %T.binding.as_type [symbolic = %pattern_type (constants.%pattern_type.422)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T.binding.as_type [symbolic = %require_complete (constants.%require_complete.2b8)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: ref @TestOp.%T.binding.as_type (%T.binding.as_type.3af)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Test(%x.param: %i32) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %TestOp.ref: %TestOp.type = name_ref TestOp, file.%TestOp.decl [concrete = constants.%TestOp]
// CHECK:STDOUT:   %x.ref: %i32 = name_ref x, %x
// CHECK:STDOUT:   %I.facet: %I.type = facet_value constants.%i32, (constants.%I.impl_witness) [concrete = constants.%I.facet]
// CHECK:STDOUT:   %.loc27: %I.type = converted constants.%i32, %I.facet [concrete = constants.%I.facet]
// CHECK:STDOUT:   %TestOp.specific_fn: <specific function> = specific_function %TestOp.ref, @TestOp(constants.%I.facet) [concrete = constants.%TestOp.specific_fn]
// CHECK:STDOUT:   %.loc22_25.1: ref %i32 = temporary_storage
// CHECK:STDOUT:   %impl.elem0: %.8e2 = impl_witness_access constants.%Copy.impl_witness.f17, element0 [concrete = constants.%Int.as.Copy.impl.Op.664]
// CHECK:STDOUT:   %bound_method.loc27_10.1: <bound method> = bound_method %x.ref, %impl.elem0
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc27_10.2: <bound method> = bound_method %x.ref, %specific_fn
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call: init %i32 = call %bound_method.loc27_10.2(%x.ref)
// CHECK:STDOUT:   %.loc22_25.2: ref %i32 = temporary %.loc22_25.1, %Int.as.Copy.impl.Op.call
// CHECK:STDOUT:   %TestOp.call: init %empty_tuple.type = call %TestOp.specific_fn(%.loc22_25.2)
// CHECK:STDOUT:   %Destroy.Op.bound: <bound method> = bound_method %.loc22_25.2, constants.%Destroy.Op.651ba6.2
// CHECK:STDOUT:   %Destroy.Op.call: init %empty_tuple.type = call %Destroy.Op.bound(%.loc22_25.2)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc22_25.1(%self.param: ref %i32.builtin) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc22_25.2(%self.param: ref %i32) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TestOp(constants.%T.651) {
// CHECK:STDOUT:   %T.loc22_12.1 => constants.%T.651
// CHECK:STDOUT:   %T.binding.as_type => constants.%T.binding.as_type.3af
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.422
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TestOp(constants.%I.facet) {
// CHECK:STDOUT:   %T.loc22_12.1 => constants.%I.facet
// CHECK:STDOUT:   %T.binding.as_type => constants.%i32
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.7ce
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.f8a
// CHECK:STDOUT: }
// CHECK:STDOUT:
