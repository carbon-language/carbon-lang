// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/unformed.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/impl/custom_witness/destruction/non_trivial/maybe_unformed.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/impl/custom_witness/destruction/non_trivial/maybe_unformed.carbon

// --- todo_nontrivial_maybe_unformed.carbon
library "[[@TEST_NAME]]";

// `HasNonTrivialDestruction.(SubobjectDestroy.Op)` is a no-op because it
// doesn't have any fields that require it to be non-trivial.
class HasNonTrivialDestruction {
  impl as Core.DestroyImpl {
    fn Op(unused self) {}
  }
}

alias MaybeUnformed = Core.MaybeUnformed(HasNonTrivialDestruction);
fn Make() -> MaybeUnformed;

fn Test() {
  //@dump-sem-ir-begin
  MaybeUnformed as Core.SubobjectDestroy;
  MaybeUnformed as Core.Destroy;

  var unused x: MaybeUnformed = Make();
  //@dump-sem-ir-end
}

// CHECK:STDOUT: --- todo_nontrivial_maybe_unformed.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %HasNonTrivialDestruction: type = class_type @HasNonTrivialDestruction [concrete]
// CHECK:STDOUT:   %DestroyImpl.type: type = facet_type <@DestroyImpl> [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %MaybeUnformed.type: type = parameterized_class_name_type @MaybeUnformed [concrete]
// CHECK:STDOUT:   %MaybeUnformed.generic: %MaybeUnformed.type = struct_value () [concrete]
// CHECK:STDOUT:   %MaybeUnformed.10a: type = class_type @MaybeUnformed, @MaybeUnformed(%HasNonTrivialDestruction) [concrete]
// CHECK:STDOUT:   %pattern_type.b12: type = pattern_type %MaybeUnformed.10a [concrete]
// CHECK:STDOUT:   %Make.type: type = fn_type @Make [concrete]
// CHECK:STDOUT:   %Make: %Make.type = struct_value () [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %.56d: type = maybe_unformed_type %HasNonTrivialDestruction [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.4: type = fn_type @SubobjectDestroy.WithSelf.Op.loc16_17.4 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.4: %SubobjectDestroy.WithSelf.Op.type.6870fd.4 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.4: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.4), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.81a: %SubobjectDestroy.type = facet_value %MaybeUnformed.10a, (%custom_witness.b80ef6.4) [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %T.09f: %SubobjectDestroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.type.dea: type = fn_type @T.as_type.as.DestroyImpl.impl.Op, @T.as_type.as.DestroyImpl.impl(%T.09f) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.6e5: %T.as_type.as.DestroyImpl.impl.Op.type.dea = struct_value () [symbolic]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.b76: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.81a) [concrete]
// CHECK:STDOUT:   %Destroy.facet: %Destroy.type = facet_value %MaybeUnformed.10a, (%DestroyImpl.impl_witness.b76, %custom_witness.b80ef6.4) [concrete]
// CHECK:STDOUT:   %x.patt: %pattern_type.b12 = ref_binding_pattern x [concrete]
// CHECK:STDOUT:   %x.var_patt: %pattern_type.b12 = var_pattern %x.patt [concrete]
// CHECK:STDOUT:   %SelfDestruct.type: type = fn_type @SelfDestruct [concrete]
// CHECK:STDOUT:   %SelfDestruct: %SelfDestruct.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .DestroyImpl = %Core.DestroyImpl
// CHECK:STDOUT:     .MaybeUnformed = %Core.MaybeUnformed
// CHECK:STDOUT:     .SubobjectDestroy = %Core.SubobjectDestroy
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     .SelfDestruct = %Core.SelfDestruct
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.DestroyImpl: type = import_ref Core//prelude/parts/destroy, DestroyImpl, loaded [concrete = constants.%DestroyImpl.type]
// CHECK:STDOUT:   %Core.MaybeUnformed: %MaybeUnformed.type = import_ref Core//prelude/parts/maybe_unformed, MaybeUnformed, loaded [concrete = constants.%MaybeUnformed.generic]
// CHECK:STDOUT:   %Core.SubobjectDestroy: type = import_ref Core//prelude/parts/destroy, SubobjectDestroy, loaded [concrete = constants.%SubobjectDestroy.type]
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Core.import_ref.74d: @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op.type (%T.as_type.as.DestroyImpl.impl.Op.type.dea) = import_ref Core//prelude/parts/destroy, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op (constants.%T.as_type.as.DestroyImpl.impl.Op.6e5)]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table = impl_witness_table (%Core.import_ref.74d), @T.as_type.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT:   %Core.SelfDestruct: %SelfDestruct.type = import_ref Core//prelude/parts/destroy, SelfDestruct, loaded [concrete = constants.%SelfDestruct]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Test() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %MaybeUnformed.ref.loc16: type = name_ref MaybeUnformed, file.%MaybeUnformed.loc11_7 [concrete = constants.%MaybeUnformed.10a]
// CHECK:STDOUT:   %Core.ref.loc16: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:   %SubobjectDestroy.ref: type = name_ref SubobjectDestroy, imports.%Core.SubobjectDestroy [concrete = constants.%SubobjectDestroy.type]
// CHECK:STDOUT:   %SubobjectDestroy.facet: %SubobjectDestroy.type = facet_value %MaybeUnformed.ref.loc16, (constants.%custom_witness.b80ef6.4) [concrete = constants.%SubobjectDestroy.facet.81a]
// CHECK:STDOUT:   %.loc16: %SubobjectDestroy.type = converted %MaybeUnformed.ref.loc16, %SubobjectDestroy.facet [concrete = constants.%SubobjectDestroy.facet.81a]
// CHECK:STDOUT:   %MaybeUnformed.ref.loc17: type = name_ref MaybeUnformed, file.%MaybeUnformed.loc11_7 [concrete = constants.%MaybeUnformed.10a]
// CHECK:STDOUT:   %Core.ref.loc17: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:   %Destroy.ref: type = name_ref Destroy, imports.%Core.Destroy [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Destroy.facet.loc17: %Destroy.type = facet_value %MaybeUnformed.ref.loc17, (constants.%DestroyImpl.impl_witness.b76, constants.%custom_witness.b80ef6.4) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc17: %Destroy.type = converted %MaybeUnformed.ref.loc17, %Destroy.facet.loc17 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %x.var: ref %MaybeUnformed.10a = var_storage %x.var_patt
// CHECK:STDOUT:   %Make.ref: %Make.type = name_ref Make, file.%Make.decl [concrete = constants.%Make]
// CHECK:STDOUT:   %.loc19_3.1: ref %MaybeUnformed.10a = splice_block %x.var {}
// CHECK:STDOUT:   %Make.call: init %MaybeUnformed.10a to %.loc19_3.1 = call %Make.ref()
// CHECK:STDOUT:   assign %x.var, %Make.call
// CHECK:STDOUT:   %MaybeUnformed.ref.loc19: type = name_ref MaybeUnformed, file.%MaybeUnformed.loc11_7 [concrete = constants.%MaybeUnformed.10a]
// CHECK:STDOUT:   %x: ref %MaybeUnformed.10a = wrapper_binding x, %x.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %pattern_type.b12 = ref_binding_pattern x [concrete = constants.%x.patt]
// CHECK:STDOUT:     %x.var_patt: %pattern_type.b12 = var_pattern %x.patt [concrete = constants.%x.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.facet.loc19_3.1: %Destroy.type = facet_value constants.%MaybeUnformed.10a, (constants.%DestroyImpl.impl_witness.b76, constants.%custom_witness.b80ef6.4) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc19_3.2: %Destroy.type = converted constants.%MaybeUnformed.10a, %Destroy.facet.loc19_3.1 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %Destroy.facet.loc19_3.2: %Destroy.type = facet_value constants.%MaybeUnformed.10a, (constants.%DestroyImpl.impl_witness.b76, constants.%custom_witness.b80ef6.4) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc19_3.3: %Destroy.type = converted constants.%MaybeUnformed.10a, %Destroy.facet.loc19_3.2 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%x.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc16_17.1(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc16_17.2(%self.param: ref %HasNonTrivialDestruction) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc16_17.3(%self.param: ref %.56d) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc16_17.4(%self.param: ref %MaybeUnformed.10a) = "no_op";
// CHECK:STDOUT:
