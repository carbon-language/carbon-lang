// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/destroy.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/impl/custom_witness/destruction/non_trivial/partial.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/impl/custom_witness/destruction/non_trivial/partial.carbon

// --- types.carbon
library "[[@TEST_NAME]]";

// `HasNonTrivialDestruction.(SubobjectDestroy.Op)` is a no-op because it
// doesn't have any fields that require it to be non-trivial.
// Similarly for other types like this.
class HasNonTrivialDestruction {
  impl as Core.DestroyImpl {
    fn Op(unused self) {}
  }
}

class AlsoHasNonTrivialDestruction {
  impl as Core.DestroyImpl {
    fn Op(unused self) {}
  }
}

abstract class Abstract {
  var x: HasNonTrivialDestruction;
  var y: AlsoHasNonTrivialDestruction;
}

// --- todo_nontrivial_member.carbon
library "[[@TEST_NAME]]";

import library "types";

base class Base {
  var x: HasNonTrivialDestruction;
  var y: AlsoHasNonTrivialDestruction;
}

alias Partial = partial Base;
fn Make() -> Partial;

fn Test() {
  //@dump-sem-ir-begin
  Partial as Core.SubobjectDestroy;
  Partial as Core.Destroy;

  var unused x: Partial = Make();
  //@dump-sem-ir-end
}

// --- todo_nontrivial_impls_destroy.carbon
library "[[@TEST_NAME]]";

base class Base {
  impl as Core.DestroyImpl {
    fn Op(unused self) {}
  }
}

alias Partial = partial Base;
fn Make() -> Base;

fn Test() {
  //@dump-sem-ir-begin
  Partial as Core.SubobjectDestroy;
  Partial as Core.Destroy;

  var unused x: partial Base = Make();
  //@dump-sem-ir-end
}


// --- todo_nontrivial_abstract.carbon
library "[[@TEST_NAME]]";

import library "types";

alias Partial = partial Abstract;
fn Make() -> Partial;

fn Test() {
  //@dump-sem-ir-begin
  Partial as Core.SubobjectDestroy;
  Partial as Core.Destroy;

  var unused x: Partial = Make();
  //@dump-sem-ir-end
}

// CHECK:STDOUT: --- todo_nontrivial_member.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %Base: type = class_type @Base [concrete]
// CHECK:STDOUT:   %HasNonTrivialDestruction: type = class_type @HasNonTrivialDestruction [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %AlsoHasNonTrivialDestruction: type = class_type @AlsoHasNonTrivialDestruction [concrete]
// CHECK:STDOUT:   %struct_type.x.y.ea4: type = struct_type {.x: %HasNonTrivialDestruction, .y: %AlsoHasNonTrivialDestruction} [concrete]
// CHECK:STDOUT:   %.edf: type = partial_type %Base [concrete]
// CHECK:STDOUT:   %pattern_type.234: type = pattern_type %.edf [concrete]
// CHECK:STDOUT:   %Make.type: type = fn_type @Make [concrete]
// CHECK:STDOUT:   %Make: %Make.type = struct_value () [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.2: type = fn_type @SubobjectDestroy.WithSelf.Op.loc15_11.2 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.2: %SubobjectDestroy.WithSelf.Op.type.6870fd.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.2: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.2), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.3: type = fn_type @SubobjectDestroy.WithSelf.Op.loc15_11.3 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.3: %SubobjectDestroy.WithSelf.Op.type.6870fd.3 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.3: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.3), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SelfDestruct.type: type = fn_type @SelfDestruct [concrete]
// CHECK:STDOUT:   %SelfDestruct: %SelfDestruct.type = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %T.09f: %SubobjectDestroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.type.dea: type = fn_type @T.as_type.as.DestroyImpl.impl.Op, @T.as_type.as.DestroyImpl.impl(%T.09f) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.6e5: %T.as_type.as.DestroyImpl.impl.Op.type.dea = struct_value () [symbolic]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.c4e: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table.86e [concrete]
// CHECK:STDOUT:   %Destroy.facet.7c1: %Destroy.type = facet_value %AlsoHasNonTrivialDestruction, (%custom_witness.b80ef6.3, %DestroyImpl.impl_witness.c4e) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.490: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table.467 [concrete]
// CHECK:STDOUT:   %Destroy.facet.4ef: %Destroy.type = facet_value %HasNonTrivialDestruction, (%custom_witness.b80ef6.2, %DestroyImpl.impl_witness.490) [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.5: type = fn_type @SubobjectDestroy.WithSelf.Op.loc15_11.5 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.5: %SubobjectDestroy.WithSelf.Op.type.6870fd.5 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.5: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.5), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.9a3: %SubobjectDestroy.type = facet_value %.edf, (%custom_witness.b80ef6.5) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.1d2: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table.a5e, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.9a3) [concrete]
// CHECK:STDOUT:   %Destroy.facet.96b: %Destroy.type = facet_value %.edf, (%custom_witness.b80ef6.5, %DestroyImpl.impl_witness.1d2) [concrete]
// CHECK:STDOUT:   %x.patt: %pattern_type.234 = ref_binding_pattern x [concrete]
// CHECK:STDOUT:   %x.var_patt: %pattern_type.234 = var_pattern %x.patt [concrete]
// CHECK:STDOUT:   %AlsoHasNonTrivialDestruction.as.DestroyImpl.impl.Op.type: type = fn_type @AlsoHasNonTrivialDestruction.as.DestroyImpl.impl.Op [concrete]
// CHECK:STDOUT:   %AlsoHasNonTrivialDestruction.as.DestroyImpl.impl.Op: %AlsoHasNonTrivialDestruction.as.DestroyImpl.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %HasNonTrivialDestruction.as.DestroyImpl.impl.Op.type: type = fn_type @HasNonTrivialDestruction.as.DestroyImpl.impl.Op [concrete]
// CHECK:STDOUT:   %HasNonTrivialDestruction.as.DestroyImpl.impl.Op: %HasNonTrivialDestruction.as.DestroyImpl.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .SubobjectDestroy = %Core.SubobjectDestroy
// CHECK:STDOUT:     .SelfDestruct = %Core.SelfDestruct
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.SubobjectDestroy: type = import_ref Core//prelude/parts/destroy, SubobjectDestroy, loaded [concrete = constants.%SubobjectDestroy.type]
// CHECK:STDOUT:   %Core.SelfDestruct: %SelfDestruct.type = import_ref Core//prelude/parts/destroy, SelfDestruct, loaded [concrete = constants.%SelfDestruct]
// CHECK:STDOUT:   %Core.import_ref.74d: @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op.type (%T.as_type.as.DestroyImpl.impl.Op.type.dea) = import_ref Core//prelude/parts/destroy, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op (constants.%T.as_type.as.DestroyImpl.impl.Op.6e5)]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table.a5e = impl_witness_table (%Core.import_ref.74d), @T.as_type.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT:   %Main.import_ref.307: %AlsoHasNonTrivialDestruction.as.DestroyImpl.impl.Op.type = import_ref Main//types, loc14_24, loaded [concrete = constants.%AlsoHasNonTrivialDestruction.as.DestroyImpl.impl.Op]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table.86e = impl_witness_table (%Main.import_ref.307), @AlsoHasNonTrivialDestruction.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT:   %Main.import_ref.546: %HasNonTrivialDestruction.as.DestroyImpl.impl.Op.type = import_ref Main//types, loc8_24, loaded [concrete = constants.%HasNonTrivialDestruction.as.DestroyImpl.impl.Op]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table.467 = impl_witness_table (%Main.import_ref.546), @HasNonTrivialDestruction.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Test() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Partial.ref.loc15: type = name_ref Partial, file.%Partial [concrete = constants.%.edf]
// CHECK:STDOUT:   %Core.ref.loc15: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:   %SubobjectDestroy.ref: type = name_ref SubobjectDestroy, imports.%Core.SubobjectDestroy [concrete = constants.%SubobjectDestroy.type]
// CHECK:STDOUT:   %SubobjectDestroy.facet: %SubobjectDestroy.type = facet_value %Partial.ref.loc15, (constants.%custom_witness.b80ef6.5) [concrete = constants.%SubobjectDestroy.facet.9a3]
// CHECK:STDOUT:   %.loc15: %SubobjectDestroy.type = converted %Partial.ref.loc15, %SubobjectDestroy.facet [concrete = constants.%SubobjectDestroy.facet.9a3]
// CHECK:STDOUT:   %Partial.ref.loc16: type = name_ref Partial, file.%Partial [concrete = constants.%.edf]
// CHECK:STDOUT:   %Core.ref.loc16: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:   %Destroy.ref: type = name_ref Destroy, imports.%Core.Destroy [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Destroy.facet.loc16: %Destroy.type = facet_value %Partial.ref.loc16, (constants.%custom_witness.b80ef6.5, constants.%DestroyImpl.impl_witness.1d2) [concrete = constants.%Destroy.facet.96b]
// CHECK:STDOUT:   %.loc16: %Destroy.type = converted %Partial.ref.loc16, %Destroy.facet.loc16 [concrete = constants.%Destroy.facet.96b]
// CHECK:STDOUT:   %x.var: ref %.edf = var_storage %x.var_patt
// CHECK:STDOUT:   %Make.ref: %Make.type = name_ref Make, file.%Make.decl [concrete = constants.%Make]
// CHECK:STDOUT:   %.loc18_3.1: ref %.edf = splice_block %x.var {}
// CHECK:STDOUT:   %Make.call: init %.edf to %.loc18_3.1 = call %Make.ref()
// CHECK:STDOUT:   assign %x.var, %Make.call
// CHECK:STDOUT:   %Partial.ref.loc18: type = name_ref Partial, file.%Partial [concrete = constants.%.edf]
// CHECK:STDOUT:   %x: ref %.edf = wrapper_binding x, %x.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %pattern_type.234 = ref_binding_pattern x [concrete = constants.%x.patt]
// CHECK:STDOUT:     %x.var_patt: %pattern_type.234 = var_pattern %x.patt [concrete = constants.%x.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.facet.loc18_3.1: %Destroy.type = facet_value constants.%.edf, (constants.%custom_witness.b80ef6.5, constants.%DestroyImpl.impl_witness.1d2) [concrete = constants.%Destroy.facet.96b]
// CHECK:STDOUT:   %.loc18_3.2: %Destroy.type = converted constants.%.edf, %Destroy.facet.loc18_3.1 [concrete = constants.%Destroy.facet.96b]
// CHECK:STDOUT:   %Destroy.facet.loc18_3.2: %Destroy.type = facet_value constants.%.edf, (constants.%custom_witness.b80ef6.5, constants.%DestroyImpl.impl_witness.1d2) [concrete = constants.%Destroy.facet.96b]
// CHECK:STDOUT:   %.loc18_3.3: %Destroy.type = converted constants.%.edf, %Destroy.facet.loc18_3.2 [concrete = constants.%Destroy.facet.96b]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%x.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc15_11.1(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc15_11.2(%self.param: ref %HasNonTrivialDestruction) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc15_11.3(%self.param: ref %AlsoHasNonTrivialDestruction) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc15_11.4(%self.param: ref %struct_type.x.y.ea4) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc15_11.1: ref %AlsoHasNonTrivialDestruction = struct_access %self.param, element1
// CHECK:STDOUT:   %Destroy.facet.loc15_11.1: %Destroy.type = facet_value constants.%AlsoHasNonTrivialDestruction, (constants.%custom_witness.b80ef6.3, constants.%DestroyImpl.impl_witness.c4e) [concrete = constants.%Destroy.facet.7c1]
// CHECK:STDOUT:   %.loc15_11.2: %Destroy.type = converted constants.%AlsoHasNonTrivialDestruction, %Destroy.facet.loc15_11.1 [concrete = constants.%Destroy.facet.7c1]
// CHECK:STDOUT:   %Destroy.facet.loc15_11.2: %Destroy.type = facet_value constants.%AlsoHasNonTrivialDestruction, (constants.%custom_witness.b80ef6.3, constants.%DestroyImpl.impl_witness.c4e) [concrete = constants.%Destroy.facet.7c1]
// CHECK:STDOUT:   %.loc15_11.3: %Destroy.type = converted constants.%AlsoHasNonTrivialDestruction, %Destroy.facet.loc15_11.2 [concrete = constants.%Destroy.facet.7c1]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc15_11.1: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%.loc15_11.1)
// CHECK:STDOUT:   %.loc15_11.4: ref %HasNonTrivialDestruction = struct_access %self.param, element0
// CHECK:STDOUT:   %Destroy.facet.loc15_11.3: %Destroy.type = facet_value constants.%HasNonTrivialDestruction, (constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.490) [concrete = constants.%Destroy.facet.4ef]
// CHECK:STDOUT:   %.loc15_11.5: %Destroy.type = converted constants.%HasNonTrivialDestruction, %Destroy.facet.loc15_11.3 [concrete = constants.%Destroy.facet.4ef]
// CHECK:STDOUT:   %Destroy.facet.loc15_11.4: %Destroy.type = facet_value constants.%HasNonTrivialDestruction, (constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.490) [concrete = constants.%Destroy.facet.4ef]
// CHECK:STDOUT:   %.loc15_11.6: %Destroy.type = converted constants.%HasNonTrivialDestruction, %Destroy.facet.loc15_11.4 [concrete = constants.%Destroy.facet.4ef]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc15_11.2: init %empty_tuple.type = call %SelfDestruct.specific_fn.2(%.loc15_11.4)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc15_11.5(%self.param: ref %.edf) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- todo_nontrivial_impls_destroy.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %Base: type = class_type @Base [concrete]
// CHECK:STDOUT:   %DestroyImpl.type: type = facet_type <@DestroyImpl> [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %.edf: type = partial_type %Base [concrete]
// CHECK:STDOUT:   %Make.type: type = fn_type @Make [concrete]
// CHECK:STDOUT:   %Make: %Make.type = struct_value () [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %pattern_type.234: type = pattern_type %.edf [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.2: type = fn_type @SubobjectDestroy.WithSelf.Op.loc14_11.2 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.2: %SubobjectDestroy.WithSelf.Op.type.6870fd.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.2: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.2), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.9a3: %SubobjectDestroy.type = facet_value %.edf, (%custom_witness.b80ef6.2) [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %T.09f: %SubobjectDestroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.type.dea: type = fn_type @T.as_type.as.DestroyImpl.impl.Op, @T.as_type.as.DestroyImpl.impl(%T.09f) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.6e5: %T.as_type.as.DestroyImpl.impl.Op.type.dea = struct_value () [symbolic]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.1d2: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.9a3) [concrete]
// CHECK:STDOUT:   %Destroy.facet: %Destroy.type = facet_value %.edf, (%DestroyImpl.impl_witness.1d2, %custom_witness.b80ef6.2) [concrete]
// CHECK:STDOUT:   %x.patt: %pattern_type.234 = ref_binding_pattern x [concrete]
// CHECK:STDOUT:   %x.var_patt: %pattern_type.234 = var_pattern %x.patt [concrete]
// CHECK:STDOUT:   %SelfDestruct.type: type = fn_type @SelfDestruct [concrete]
// CHECK:STDOUT:   %SelfDestruct: %SelfDestruct.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .DestroyImpl = %Core.DestroyImpl
// CHECK:STDOUT:     .SubobjectDestroy = %Core.SubobjectDestroy
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     .SelfDestruct = %Core.SelfDestruct
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.DestroyImpl: type = import_ref Core//prelude/parts/destroy, DestroyImpl, loaded [concrete = constants.%DestroyImpl.type]
// CHECK:STDOUT:   %Core.SubobjectDestroy: type = import_ref Core//prelude/parts/destroy, SubobjectDestroy, loaded [concrete = constants.%SubobjectDestroy.type]
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Core.import_ref.74d: @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op.type (%T.as_type.as.DestroyImpl.impl.Op.type.dea) = import_ref Core//prelude/parts/destroy, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op (constants.%T.as_type.as.DestroyImpl.impl.Op.6e5)]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table = impl_witness_table (%Core.import_ref.74d), @T.as_type.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT:   %Core.SelfDestruct: %SelfDestruct.type = import_ref Core//prelude/parts/destroy, SelfDestruct, loaded [concrete = constants.%SelfDestruct]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Test() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Partial.ref.loc14: type = name_ref Partial, file.%Partial [concrete = constants.%.edf]
// CHECK:STDOUT:   %Core.ref.loc14: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:   %SubobjectDestroy.ref: type = name_ref SubobjectDestroy, imports.%Core.SubobjectDestroy [concrete = constants.%SubobjectDestroy.type]
// CHECK:STDOUT:   %SubobjectDestroy.facet: %SubobjectDestroy.type = facet_value %Partial.ref.loc14, (constants.%custom_witness.b80ef6.2) [concrete = constants.%SubobjectDestroy.facet.9a3]
// CHECK:STDOUT:   %.loc14: %SubobjectDestroy.type = converted %Partial.ref.loc14, %SubobjectDestroy.facet [concrete = constants.%SubobjectDestroy.facet.9a3]
// CHECK:STDOUT:   %Partial.ref.loc15: type = name_ref Partial, file.%Partial [concrete = constants.%.edf]
// CHECK:STDOUT:   %Core.ref.loc15: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:   %Destroy.ref: type = name_ref Destroy, imports.%Core.Destroy [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Destroy.facet.loc15: %Destroy.type = facet_value %Partial.ref.loc15, (constants.%DestroyImpl.impl_witness.1d2, constants.%custom_witness.b80ef6.2) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc15: %Destroy.type = converted %Partial.ref.loc15, %Destroy.facet.loc15 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %x.var: ref %.edf = var_storage %x.var_patt
// CHECK:STDOUT:   %Make.ref: %Make.type = name_ref Make, file.%Make.decl [concrete = constants.%Make]
// CHECK:STDOUT:   %.loc17_3.1: ref %.edf = splice_block %x.var {}
// CHECK:STDOUT:   %Make.call: init %Base to %.loc17_3.1 = call %Make.ref()
// CHECK:STDOUT:   %.loc17_3.2: init %.edf = as_compatible %Make.call
// CHECK:STDOUT:   %.loc17_3.3: init %.edf = converted %Make.call, %.loc17_3.2
// CHECK:STDOUT:   assign %x.var, %.loc17_3.3
// CHECK:STDOUT:   %.loc17_17.1: type = splice_block %.loc17_17.2 [concrete = constants.%.edf] {
// CHECK:STDOUT:     %Base.ref: type = name_ref Base, file.%Base.decl [concrete = constants.%Base]
// CHECK:STDOUT:     %.loc17_17.2: type = partial_type %Base.ref [concrete = constants.%.edf]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %x: ref %.edf = wrapper_binding x, %x.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %pattern_type.234 = ref_binding_pattern x [concrete = constants.%x.patt]
// CHECK:STDOUT:     %x.var_patt: %pattern_type.234 = var_pattern %x.patt [concrete = constants.%x.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.facet.loc17_3.1: %Destroy.type = facet_value constants.%.edf, (constants.%DestroyImpl.impl_witness.1d2, constants.%custom_witness.b80ef6.2) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc17_3.4: %Destroy.type = converted constants.%.edf, %Destroy.facet.loc17_3.1 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %Destroy.facet.loc17_3.2: %Destroy.type = facet_value constants.%.edf, (constants.%DestroyImpl.impl_witness.1d2, constants.%custom_witness.b80ef6.2) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc17_3.5: %Destroy.type = converted constants.%.edf, %Destroy.facet.loc17_3.2 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%x.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc14_11.1(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc14_11.2(%self.param: ref %.edf) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: --- todo_nontrivial_abstract.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %Abstract: type = class_type @Abstract [concrete]
// CHECK:STDOUT:   %AlsoHasNonTrivialDestruction: type = class_type @AlsoHasNonTrivialDestruction [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %HasNonTrivialDestruction: type = class_type @HasNonTrivialDestruction [concrete]
// CHECK:STDOUT:   %struct_type.x.y.ea4: type = struct_type {.x: %HasNonTrivialDestruction, .y: %AlsoHasNonTrivialDestruction} [concrete]
// CHECK:STDOUT:   %.79f: type = partial_type %Abstract [concrete]
// CHECK:STDOUT:   %pattern_type.5f8: type = pattern_type %.79f [concrete]
// CHECK:STDOUT:   %Make.type: type = fn_type @Make [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Make: %Make.type = struct_value () [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.2: type = fn_type @SubobjectDestroy.WithSelf.Op.loc10_11.2 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.2: %SubobjectDestroy.WithSelf.Op.type.6870fd.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.2: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.2), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.3: type = fn_type @SubobjectDestroy.WithSelf.Op.loc10_11.3 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.3: %SubobjectDestroy.WithSelf.Op.type.6870fd.3 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.3: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.3), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SelfDestruct.type: type = fn_type @SelfDestruct [concrete]
// CHECK:STDOUT:   %SelfDestruct: %SelfDestruct.type = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %T.09f: %SubobjectDestroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.type.dea: type = fn_type @T.as_type.as.DestroyImpl.impl.Op, @T.as_type.as.DestroyImpl.impl(%T.09f) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.6e5: %T.as_type.as.DestroyImpl.impl.Op.type.dea = struct_value () [symbolic]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.c4e: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table.86e [concrete]
// CHECK:STDOUT:   %Destroy.facet.7c1: %Destroy.type = facet_value %AlsoHasNonTrivialDestruction, (%custom_witness.b80ef6.3, %DestroyImpl.impl_witness.c4e) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.490: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table.467 [concrete]
// CHECK:STDOUT:   %Destroy.facet.4ef: %Destroy.type = facet_value %HasNonTrivialDestruction, (%custom_witness.b80ef6.2, %DestroyImpl.impl_witness.490) [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.5: type = fn_type @SubobjectDestroy.WithSelf.Op.loc10_11.5 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.5: %SubobjectDestroy.WithSelf.Op.type.6870fd.5 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.5: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.5), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.c03: %SubobjectDestroy.type = facet_value %.79f, (%custom_witness.b80ef6.5) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.98a: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table.a5e, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.c03) [concrete]
// CHECK:STDOUT:   %Destroy.facet.280: %Destroy.type = facet_value %.79f, (%custom_witness.b80ef6.5, %DestroyImpl.impl_witness.98a) [concrete]
// CHECK:STDOUT:   %x.patt: %pattern_type.5f8 = ref_binding_pattern x [concrete]
// CHECK:STDOUT:   %x.var_patt: %pattern_type.5f8 = var_pattern %x.patt [concrete]
// CHECK:STDOUT:   %AlsoHasNonTrivialDestruction.as.DestroyImpl.impl.Op.type: type = fn_type @AlsoHasNonTrivialDestruction.as.DestroyImpl.impl.Op [concrete]
// CHECK:STDOUT:   %AlsoHasNonTrivialDestruction.as.DestroyImpl.impl.Op: %AlsoHasNonTrivialDestruction.as.DestroyImpl.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %HasNonTrivialDestruction.as.DestroyImpl.impl.Op.type: type = fn_type @HasNonTrivialDestruction.as.DestroyImpl.impl.Op [concrete]
// CHECK:STDOUT:   %HasNonTrivialDestruction.as.DestroyImpl.impl.Op: %HasNonTrivialDestruction.as.DestroyImpl.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .SubobjectDestroy = %Core.SubobjectDestroy
// CHECK:STDOUT:     .SelfDestruct = %Core.SelfDestruct
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.SubobjectDestroy: type = import_ref Core//prelude/parts/destroy, SubobjectDestroy, loaded [concrete = constants.%SubobjectDestroy.type]
// CHECK:STDOUT:   %Core.SelfDestruct: %SelfDestruct.type = import_ref Core//prelude/parts/destroy, SelfDestruct, loaded [concrete = constants.%SelfDestruct]
// CHECK:STDOUT:   %Core.import_ref.74d: @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op.type (%T.as_type.as.DestroyImpl.impl.Op.type.dea) = import_ref Core//prelude/parts/destroy, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op (constants.%T.as_type.as.DestroyImpl.impl.Op.6e5)]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table.a5e = impl_witness_table (%Core.import_ref.74d), @T.as_type.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT:   %Main.import_ref.307: %AlsoHasNonTrivialDestruction.as.DestroyImpl.impl.Op.type = import_ref Main//types, loc14_24, loaded [concrete = constants.%AlsoHasNonTrivialDestruction.as.DestroyImpl.impl.Op]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table.86e = impl_witness_table (%Main.import_ref.307), @AlsoHasNonTrivialDestruction.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT:   %Main.import_ref.546: %HasNonTrivialDestruction.as.DestroyImpl.impl.Op.type = import_ref Main//types, loc8_24, loaded [concrete = constants.%HasNonTrivialDestruction.as.DestroyImpl.impl.Op]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table.467 = impl_witness_table (%Main.import_ref.546), @HasNonTrivialDestruction.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Test() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Partial.ref.loc10: type = name_ref Partial, file.%Partial [concrete = constants.%.79f]
// CHECK:STDOUT:   %Core.ref.loc10: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:   %SubobjectDestroy.ref: type = name_ref SubobjectDestroy, imports.%Core.SubobjectDestroy [concrete = constants.%SubobjectDestroy.type]
// CHECK:STDOUT:   %SubobjectDestroy.facet: %SubobjectDestroy.type = facet_value %Partial.ref.loc10, (constants.%custom_witness.b80ef6.5) [concrete = constants.%SubobjectDestroy.facet.c03]
// CHECK:STDOUT:   %.loc10: %SubobjectDestroy.type = converted %Partial.ref.loc10, %SubobjectDestroy.facet [concrete = constants.%SubobjectDestroy.facet.c03]
// CHECK:STDOUT:   %Partial.ref.loc11: type = name_ref Partial, file.%Partial [concrete = constants.%.79f]
// CHECK:STDOUT:   %Core.ref.loc11: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:   %Destroy.ref: type = name_ref Destroy, imports.%Core.Destroy [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Destroy.facet.loc11: %Destroy.type = facet_value %Partial.ref.loc11, (constants.%custom_witness.b80ef6.5, constants.%DestroyImpl.impl_witness.98a) [concrete = constants.%Destroy.facet.280]
// CHECK:STDOUT:   %.loc11: %Destroy.type = converted %Partial.ref.loc11, %Destroy.facet.loc11 [concrete = constants.%Destroy.facet.280]
// CHECK:STDOUT:   %x.var: ref %.79f = var_storage %x.var_patt
// CHECK:STDOUT:   %Make.ref: %Make.type = name_ref Make, file.%Make.decl [concrete = constants.%Make]
// CHECK:STDOUT:   %.loc13_3.1: ref %.79f = splice_block %x.var {}
// CHECK:STDOUT:   %Make.call: init %.79f to %.loc13_3.1 = call %Make.ref()
// CHECK:STDOUT:   assign %x.var, %Make.call
// CHECK:STDOUT:   %Partial.ref.loc13: type = name_ref Partial, file.%Partial [concrete = constants.%.79f]
// CHECK:STDOUT:   %x: ref %.79f = wrapper_binding x, %x.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %pattern_type.5f8 = ref_binding_pattern x [concrete = constants.%x.patt]
// CHECK:STDOUT:     %x.var_patt: %pattern_type.5f8 = var_pattern %x.patt [concrete = constants.%x.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.facet.loc13_3.1: %Destroy.type = facet_value constants.%.79f, (constants.%custom_witness.b80ef6.5, constants.%DestroyImpl.impl_witness.98a) [concrete = constants.%Destroy.facet.280]
// CHECK:STDOUT:   %.loc13_3.2: %Destroy.type = converted constants.%.79f, %Destroy.facet.loc13_3.1 [concrete = constants.%Destroy.facet.280]
// CHECK:STDOUT:   %Destroy.facet.loc13_3.2: %Destroy.type = facet_value constants.%.79f, (constants.%custom_witness.b80ef6.5, constants.%DestroyImpl.impl_witness.98a) [concrete = constants.%Destroy.facet.280]
// CHECK:STDOUT:   %.loc13_3.3: %Destroy.type = converted constants.%.79f, %Destroy.facet.loc13_3.2 [concrete = constants.%Destroy.facet.280]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%x.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc10_11.1(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc10_11.2(%self.param: ref %HasNonTrivialDestruction) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc10_11.3(%self.param: ref %AlsoHasNonTrivialDestruction) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc10_11.4(%self.param: ref %struct_type.x.y.ea4) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc10_11.1: ref %AlsoHasNonTrivialDestruction = struct_access %self.param, element1
// CHECK:STDOUT:   %Destroy.facet.loc10_11.1: %Destroy.type = facet_value constants.%AlsoHasNonTrivialDestruction, (constants.%custom_witness.b80ef6.3, constants.%DestroyImpl.impl_witness.c4e) [concrete = constants.%Destroy.facet.7c1]
// CHECK:STDOUT:   %.loc10_11.2: %Destroy.type = converted constants.%AlsoHasNonTrivialDestruction, %Destroy.facet.loc10_11.1 [concrete = constants.%Destroy.facet.7c1]
// CHECK:STDOUT:   %Destroy.facet.loc10_11.2: %Destroy.type = facet_value constants.%AlsoHasNonTrivialDestruction, (constants.%custom_witness.b80ef6.3, constants.%DestroyImpl.impl_witness.c4e) [concrete = constants.%Destroy.facet.7c1]
// CHECK:STDOUT:   %.loc10_11.3: %Destroy.type = converted constants.%AlsoHasNonTrivialDestruction, %Destroy.facet.loc10_11.2 [concrete = constants.%Destroy.facet.7c1]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc10_11.1: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%.loc10_11.1)
// CHECK:STDOUT:   %.loc10_11.4: ref %HasNonTrivialDestruction = struct_access %self.param, element0
// CHECK:STDOUT:   %Destroy.facet.loc10_11.3: %Destroy.type = facet_value constants.%HasNonTrivialDestruction, (constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.490) [concrete = constants.%Destroy.facet.4ef]
// CHECK:STDOUT:   %.loc10_11.5: %Destroy.type = converted constants.%HasNonTrivialDestruction, %Destroy.facet.loc10_11.3 [concrete = constants.%Destroy.facet.4ef]
// CHECK:STDOUT:   %Destroy.facet.loc10_11.4: %Destroy.type = facet_value constants.%HasNonTrivialDestruction, (constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.490) [concrete = constants.%Destroy.facet.4ef]
// CHECK:STDOUT:   %.loc10_11.6: %Destroy.type = converted constants.%HasNonTrivialDestruction, %Destroy.facet.loc10_11.4 [concrete = constants.%Destroy.facet.4ef]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc10_11.2: init %empty_tuple.type = call %SelfDestruct.specific_fn.2(%.loc10_11.4)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc10_11.5(%self.param: ref %.79f) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
