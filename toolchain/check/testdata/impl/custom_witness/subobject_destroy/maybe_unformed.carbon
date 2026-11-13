// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/unformed.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/impl/custom_witness/subobject_destroy/maybe_unformed.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/impl/custom_witness/subobject_destroy/maybe_unformed.carbon

// --- todo_nontrivial_maybe_unformed.carbon
library "[[@TEST_NAME]]";

class ImplsDestroy {
  impl as Core.DestroyImpl {
    fn Op(unused self) {}
  }
}

alias MaybeUnformed = Core.MaybeUnformed(ImplsDestroy);
fn Make() -> MaybeUnformed;

fn Test() {
  MaybeUnformed as Core.SubobjectDestroy;
  MaybeUnformed as Core.Destroy;

  //@dump-sem-ir-begin
  var unused x: MaybeUnformed = Make();
  //@dump-sem-ir-end
}

// CHECK:STDOUT: --- todo_nontrivial_maybe_unformed.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %ImplsDestroy: type = class_type @ImplsDestroy [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %MaybeUnformed.439: type = class_type @MaybeUnformed, @MaybeUnformed(%ImplsDestroy) [concrete]
// CHECK:STDOUT:   %pattern_type.9ec: type = pattern_type %MaybeUnformed.439 [concrete]
// CHECK:STDOUT:   %Make.type: type = fn_type @Make [concrete]
// CHECK:STDOUT:   %Make: %Make.type = struct_value () [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.4: type = fn_type @SubobjectDestroy.WithSelf.Op.loc13_17.4 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.4: %SubobjectDestroy.WithSelf.Op.type.6870fd.4 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.4: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.4), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.fa1: %SubobjectDestroy.type = facet_value %MaybeUnformed.439, (%custom_witness.b80ef6.4) [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %T.09f: %SubobjectDestroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.type.dea: type = fn_type @T.as_type.as.DestroyImpl.impl.Op, @T.as_type.as.DestroyImpl.impl(%T.09f) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.6e5: %T.as_type.as.DestroyImpl.impl.Op.type.dea = struct_value () [symbolic]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.b7e: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.fa1) [concrete]
// CHECK:STDOUT:   %Destroy.facet: %Destroy.type = facet_value %MaybeUnformed.439, (%DestroyImpl.impl_witness.b7e, %custom_witness.b80ef6.4) [concrete]
// CHECK:STDOUT:   %x.patt: %pattern_type.9ec = ref_binding_pattern x [concrete]
// CHECK:STDOUT:   %x.var_patt: %pattern_type.9ec = var_pattern %x.patt [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.74d: @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op.type (%T.as_type.as.DestroyImpl.impl.Op.type.dea) = import_ref Core//prelude/parts/destroy, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op (constants.%T.as_type.as.DestroyImpl.impl.Op.6e5)]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table = impl_witness_table (%Core.import_ref.74d), @T.as_type.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Test() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %x.var: ref %MaybeUnformed.439 = var_storage %x.var_patt
// CHECK:STDOUT:   %Make.ref: %Make.type = name_ref Make, file.%Make.decl [concrete = constants.%Make]
// CHECK:STDOUT:   %.loc17_3.1: ref %MaybeUnformed.439 = splice_block %x.var {}
// CHECK:STDOUT:   %Make.call: init %MaybeUnformed.439 to %.loc17_3.1 = call %Make.ref()
// CHECK:STDOUT:   assign %x.var, %Make.call
// CHECK:STDOUT:   %MaybeUnformed.ref.loc17: type = name_ref MaybeUnformed, file.%MaybeUnformed.loc9_7 [concrete = constants.%MaybeUnformed.439]
// CHECK:STDOUT:   %x: ref %MaybeUnformed.439 = wrapper_binding x, %x.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %pattern_type.9ec = ref_binding_pattern x [concrete = constants.%x.patt]
// CHECK:STDOUT:     %x.var_patt: %pattern_type.9ec = var_pattern %x.patt [concrete = constants.%x.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.facet.loc17_3.1: %Destroy.type = facet_value constants.%MaybeUnformed.439, (constants.%DestroyImpl.impl_witness.b7e, constants.%custom_witness.b80ef6.4) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc17_3.2: %Destroy.type = converted constants.%MaybeUnformed.439, %Destroy.facet.loc17_3.1 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %Destroy.facet.loc17_3.2: %Destroy.type = facet_value constants.%MaybeUnformed.439, (constants.%DestroyImpl.impl_witness.b7e, constants.%custom_witness.b80ef6.4) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc17_3.3: %Destroy.type = converted constants.%MaybeUnformed.439, %Destroy.facet.loc17_3.2 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%x.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
