// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/destroy.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/impl/custom_witness/destruction/non_trivial/tuples.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/impl/custom_witness/destruction/non_trivial/tuples.carbon

// --- types.carbon
library "[[@TEST_NAME]]";

// Used when mixing with `ImplsDestroy`.
class HasTrivialDestruction {}

class ImplsDestroy {
  impl as Core.DestroyImpl {
    fn Op(unused self) {}
  }
}

class AlsoImplsDestroy {
  impl as Core.DestroyImpl {
    fn Op(unused ref self) {}
  }
}

// --- todo_nontrivial_all_nontrivial_elements.carbon
library "[[@TEST_NAME]]";

import library "types";

alias Tuple = (ImplsDestroy, AlsoImplsDestroy, AlsoImplsDestroy, ImplsDestroy);
fn Make() -> Tuple;

fn Test() {
  Tuple as Core.SubobjectDestroy;
  Tuple as Core.Destroy;

  //@dump-sem-ir-begin
  var unused x: Tuple = Make();
  //@dump-sem-ir-end
}

// --- todo_nontrivial_one_trivial_element.carbon
library "[[@TEST_NAME]]";

import library "types";

alias Tuple = (ImplsDestroy, HasTrivialDestruction, AlsoImplsDestroy, ImplsDestroy);
fn Make() -> Tuple;

fn Test() {
  Tuple as Core.SubobjectDestroy;
  Tuple as Core.Destroy;

  //@dump-sem-ir-begin
  var unused x: Tuple = Make();
  //@dump-sem-ir-end
}

// --- todo_nontrivial_nested.carbon
library "[[@TEST_NAME]]";

import library "types";

alias Tuple = ((ImplsDestroy, HasTrivialDestruction), AlsoImplsDestroy);
fn Make() -> Tuple;

fn Test() {
  Tuple as Core.SubobjectDestroy;
  Tuple as Core.Destroy;

  //@dump-sem-ir-begin
  var unused x: Tuple = Make();
  //@dump-sem-ir-end
}

// CHECK:STDOUT: --- todo_nontrivial_all_nontrivial_elements.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %ImplsDestroy: type = class_type @ImplsDestroy [concrete]
// CHECK:STDOUT:   %AlsoImplsDestroy: type = class_type @AlsoImplsDestroy [concrete]
// CHECK:STDOUT:   %tuple.type.c31: type = tuple_type (type, type, type, type) [concrete]
// CHECK:STDOUT:   %tuple: %tuple.type.c31 = tuple_value (%ImplsDestroy, %AlsoImplsDestroy, %AlsoImplsDestroy, %ImplsDestroy) [concrete]
// CHECK:STDOUT:   %tuple.type.68b: type = tuple_type (%ImplsDestroy, %AlsoImplsDestroy, %AlsoImplsDestroy, %ImplsDestroy) [concrete]
// CHECK:STDOUT:   %pattern_type.d39: type = pattern_type %tuple.type.68b [concrete]
// CHECK:STDOUT:   %Make.type: type = fn_type @Make [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Make: %Make.type = struct_value () [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.4: type = fn_type @SubobjectDestroy.WithSelf.Op.loc9_9.4 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.4: %SubobjectDestroy.WithSelf.Op.type.6870fd.4 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %T.09f: %SubobjectDestroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.type.dea: type = fn_type @T.as_type.as.DestroyImpl.impl.Op, @T.as_type.as.DestroyImpl.impl(%T.09f) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.6e5: %T.as_type.as.DestroyImpl.impl.Op.type.dea = struct_value () [symbolic]
// CHECK:STDOUT:   %custom_witness.b80ef6.4: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.4), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.d00: %SubobjectDestroy.type = facet_value %tuple.type.68b, (%custom_witness.b80ef6.4) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.2fb: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table.a5e, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.d00) [concrete]
// CHECK:STDOUT:   %Destroy.facet.e30: %Destroy.type = facet_value %tuple.type.68b, (%custom_witness.b80ef6.4, %DestroyImpl.impl_witness.2fb) [concrete]
// CHECK:STDOUT:   %x.patt: %pattern_type.d39 = ref_binding_pattern x [concrete]
// CHECK:STDOUT:   %x.var_patt: %pattern_type.d39 = var_pattern %x.patt [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.74d: @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op.type (%T.as_type.as.DestroyImpl.impl.Op.type.dea) = import_ref Core//prelude/parts/destroy, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op (constants.%T.as_type.as.DestroyImpl.impl.Op.6e5)]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table.a5e = impl_witness_table (%Core.import_ref.74d), @T.as_type.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Test() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %x.var: ref %tuple.type.68b = var_storage %x.var_patt
// CHECK:STDOUT:   %Make.ref: %Make.type = name_ref Make, file.%Make.decl [concrete = constants.%Make]
// CHECK:STDOUT:   %.loc13_3.1: ref %tuple.type.68b = splice_block %x.var {}
// CHECK:STDOUT:   %Make.call: init %tuple.type.68b to %.loc13_3.1 = call %Make.ref()
// CHECK:STDOUT:   assign %x.var, %Make.call
// CHECK:STDOUT:   %.loc13_17.1: type = splice_block %.loc13_17.2 [concrete = constants.%tuple.type.68b] {
// CHECK:STDOUT:     %Tuple.ref.loc13: %tuple.type.c31 = name_ref Tuple, file.%Tuple [concrete = constants.%tuple]
// CHECK:STDOUT:     %.loc13_17.2: type = converted %Tuple.ref.loc13, constants.%tuple.type.68b [concrete = constants.%tuple.type.68b]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %x: ref %tuple.type.68b = wrapper_binding x, %x.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %pattern_type.d39 = ref_binding_pattern x [concrete = constants.%x.patt]
// CHECK:STDOUT:     %x.var_patt: %pattern_type.d39 = var_pattern %x.patt [concrete = constants.%x.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.facet.loc13_3.1: %Destroy.type = facet_value constants.%tuple.type.68b, (constants.%custom_witness.b80ef6.4, constants.%DestroyImpl.impl_witness.2fb) [concrete = constants.%Destroy.facet.e30]
// CHECK:STDOUT:   %.loc13_3.2: %Destroy.type = converted constants.%tuple.type.68b, %Destroy.facet.loc13_3.1 [concrete = constants.%Destroy.facet.e30]
// CHECK:STDOUT:   %Destroy.facet.loc13_3.2: %Destroy.type = facet_value constants.%tuple.type.68b, (constants.%custom_witness.b80ef6.4, constants.%DestroyImpl.impl_witness.2fb) [concrete = constants.%Destroy.facet.e30]
// CHECK:STDOUT:   %.loc13_3.3: %Destroy.type = converted constants.%tuple.type.68b, %Destroy.facet.loc13_3.2 [concrete = constants.%Destroy.facet.e30]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%x.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- todo_nontrivial_one_trivial_element.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %ImplsDestroy: type = class_type @ImplsDestroy [concrete]
// CHECK:STDOUT:   %HasTrivialDestruction: type = class_type @HasTrivialDestruction [concrete]
// CHECK:STDOUT:   %AlsoImplsDestroy: type = class_type @AlsoImplsDestroy [concrete]
// CHECK:STDOUT:   %tuple.type.c31: type = tuple_type (type, type, type, type) [concrete]
// CHECK:STDOUT:   %tuple: %tuple.type.c31 = tuple_value (%ImplsDestroy, %HasTrivialDestruction, %AlsoImplsDestroy, %ImplsDestroy) [concrete]
// CHECK:STDOUT:   %tuple.type.995: type = tuple_type (%ImplsDestroy, %HasTrivialDestruction, %AlsoImplsDestroy, %ImplsDestroy) [concrete]
// CHECK:STDOUT:   %pattern_type.982: type = pattern_type %tuple.type.995 [concrete]
// CHECK:STDOUT:   %Make.type: type = fn_type @Make [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Make: %Make.type = struct_value () [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.5: type = fn_type @SubobjectDestroy.WithSelf.Op.loc9_9.5 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.5: %SubobjectDestroy.WithSelf.Op.type.6870fd.5 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %T.09f: %SubobjectDestroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.type.dea: type = fn_type @T.as_type.as.DestroyImpl.impl.Op, @T.as_type.as.DestroyImpl.impl(%T.09f) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.6e5: %T.as_type.as.DestroyImpl.impl.Op.type.dea = struct_value () [symbolic]
// CHECK:STDOUT:   %custom_witness.b80ef6.5: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.5), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.fbb: %SubobjectDestroy.type = facet_value %tuple.type.995, (%custom_witness.b80ef6.5) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.1ca: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table.a5e, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.fbb) [concrete]
// CHECK:STDOUT:   %Destroy.facet.b25: %Destroy.type = facet_value %tuple.type.995, (%custom_witness.b80ef6.5, %DestroyImpl.impl_witness.1ca) [concrete]
// CHECK:STDOUT:   %x.patt: %pattern_type.982 = ref_binding_pattern x [concrete]
// CHECK:STDOUT:   %x.var_patt: %pattern_type.982 = var_pattern %x.patt [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.74d: @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op.type (%T.as_type.as.DestroyImpl.impl.Op.type.dea) = import_ref Core//prelude/parts/destroy, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op (constants.%T.as_type.as.DestroyImpl.impl.Op.6e5)]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table.a5e = impl_witness_table (%Core.import_ref.74d), @T.as_type.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Test() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %x.var: ref %tuple.type.995 = var_storage %x.var_patt
// CHECK:STDOUT:   %Make.ref: %Make.type = name_ref Make, file.%Make.decl [concrete = constants.%Make]
// CHECK:STDOUT:   %.loc13_3.1: ref %tuple.type.995 = splice_block %x.var {}
// CHECK:STDOUT:   %Make.call: init %tuple.type.995 to %.loc13_3.1 = call %Make.ref()
// CHECK:STDOUT:   assign %x.var, %Make.call
// CHECK:STDOUT:   %.loc13_17.1: type = splice_block %.loc13_17.2 [concrete = constants.%tuple.type.995] {
// CHECK:STDOUT:     %Tuple.ref.loc13: %tuple.type.c31 = name_ref Tuple, file.%Tuple [concrete = constants.%tuple]
// CHECK:STDOUT:     %.loc13_17.2: type = converted %Tuple.ref.loc13, constants.%tuple.type.995 [concrete = constants.%tuple.type.995]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %x: ref %tuple.type.995 = wrapper_binding x, %x.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %pattern_type.982 = ref_binding_pattern x [concrete = constants.%x.patt]
// CHECK:STDOUT:     %x.var_patt: %pattern_type.982 = var_pattern %x.patt [concrete = constants.%x.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.facet.loc13_3.1: %Destroy.type = facet_value constants.%tuple.type.995, (constants.%custom_witness.b80ef6.5, constants.%DestroyImpl.impl_witness.1ca) [concrete = constants.%Destroy.facet.b25]
// CHECK:STDOUT:   %.loc13_3.2: %Destroy.type = converted constants.%tuple.type.995, %Destroy.facet.loc13_3.1 [concrete = constants.%Destroy.facet.b25]
// CHECK:STDOUT:   %Destroy.facet.loc13_3.2: %Destroy.type = facet_value constants.%tuple.type.995, (constants.%custom_witness.b80ef6.5, constants.%DestroyImpl.impl_witness.1ca) [concrete = constants.%Destroy.facet.b25]
// CHECK:STDOUT:   %.loc13_3.3: %Destroy.type = converted constants.%tuple.type.995, %Destroy.facet.loc13_3.2 [concrete = constants.%Destroy.facet.b25]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%x.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- todo_nontrivial_nested.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %ImplsDestroy: type = class_type @ImplsDestroy [concrete]
// CHECK:STDOUT:   %HasTrivialDestruction: type = class_type @HasTrivialDestruction [concrete]
// CHECK:STDOUT:   %tuple.type.693: type = tuple_type (type, type) [concrete]
// CHECK:STDOUT:   %tuple.951: %tuple.type.693 = tuple_value (%ImplsDestroy, %HasTrivialDestruction) [concrete]
// CHECK:STDOUT:   %AlsoImplsDestroy: type = class_type @AlsoImplsDestroy [concrete]
// CHECK:STDOUT:   %tuple.type.55c: type = tuple_type (%tuple.type.693, type) [concrete]
// CHECK:STDOUT:   %tuple.9b5: %tuple.type.55c = tuple_value (%tuple.951, %AlsoImplsDestroy) [concrete]
// CHECK:STDOUT:   %tuple.type.cac: type = tuple_type (%ImplsDestroy, %HasTrivialDestruction) [concrete]
// CHECK:STDOUT:   %tuple.type.121: type = tuple_type (%tuple.type.cac, %AlsoImplsDestroy) [concrete]
// CHECK:STDOUT:   %pattern_type.76e: type = pattern_type %tuple.type.121 [concrete]
// CHECK:STDOUT:   %Make.type: type = fn_type @Make [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Make: %Make.type = struct_value () [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %T.09f: %SubobjectDestroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.type.dea: type = fn_type @T.as_type.as.DestroyImpl.impl.Op, @T.as_type.as.DestroyImpl.impl(%T.09f) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.6e5: %T.as_type.as.DestroyImpl.impl.Op.type.dea = struct_value () [symbolic]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.6: type = fn_type @SubobjectDestroy.WithSelf.Op.loc9_9.6 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.6: %SubobjectDestroy.WithSelf.Op.type.6870fd.6 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.6: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.6), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.773: %SubobjectDestroy.type = facet_value %tuple.type.121, (%custom_witness.b80ef6.6) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.0b5: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table.a5e, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.773) [concrete]
// CHECK:STDOUT:   %Destroy.facet.4aa: %Destroy.type = facet_value %tuple.type.121, (%custom_witness.b80ef6.6, %DestroyImpl.impl_witness.0b5) [concrete]
// CHECK:STDOUT:   %x.patt: %pattern_type.76e = ref_binding_pattern x [concrete]
// CHECK:STDOUT:   %x.var_patt: %pattern_type.76e = var_pattern %x.patt [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.74d: @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op.type (%T.as_type.as.DestroyImpl.impl.Op.type.dea) = import_ref Core//prelude/parts/destroy, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op (constants.%T.as_type.as.DestroyImpl.impl.Op.6e5)]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table.a5e = impl_witness_table (%Core.import_ref.74d), @T.as_type.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Test() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %x.var: ref %tuple.type.121 = var_storage %x.var_patt
// CHECK:STDOUT:   %Make.ref: %Make.type = name_ref Make, file.%Make.decl [concrete = constants.%Make]
// CHECK:STDOUT:   %.loc13_3.1: ref %tuple.type.121 = splice_block %x.var {}
// CHECK:STDOUT:   %Make.call: init %tuple.type.121 to %.loc13_3.1 = call %Make.ref()
// CHECK:STDOUT:   assign %x.var, %Make.call
// CHECK:STDOUT:   %.loc13_17.1: type = splice_block %.loc13_17.3 [concrete = constants.%tuple.type.121] {
// CHECK:STDOUT:     %Tuple.ref.loc13: %tuple.type.55c = name_ref Tuple, file.%Tuple [concrete = constants.%tuple.9b5]
// CHECK:STDOUT:     %.loc13_17.2: type = converted constants.%tuple.951, constants.%tuple.type.cac [concrete = constants.%tuple.type.cac]
// CHECK:STDOUT:     %.loc13_17.3: type = converted %Tuple.ref.loc13, constants.%tuple.type.121 [concrete = constants.%tuple.type.121]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %x: ref %tuple.type.121 = wrapper_binding x, %x.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %pattern_type.76e = ref_binding_pattern x [concrete = constants.%x.patt]
// CHECK:STDOUT:     %x.var_patt: %pattern_type.76e = var_pattern %x.patt [concrete = constants.%x.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.facet.loc13_3.1: %Destroy.type = facet_value constants.%tuple.type.121, (constants.%custom_witness.b80ef6.6, constants.%DestroyImpl.impl_witness.0b5) [concrete = constants.%Destroy.facet.4aa]
// CHECK:STDOUT:   %.loc13_3.2: %Destroy.type = converted constants.%tuple.type.121, %Destroy.facet.loc13_3.1 [concrete = constants.%Destroy.facet.4aa]
// CHECK:STDOUT:   %Destroy.facet.loc13_3.2: %Destroy.type = facet_value constants.%tuple.type.121, (constants.%custom_witness.b80ef6.6, constants.%DestroyImpl.impl_witness.0b5) [concrete = constants.%Destroy.facet.4aa]
// CHECK:STDOUT:   %.loc13_3.3: %Destroy.type = converted constants.%tuple.type.121, %Destroy.facet.loc13_3.2 [concrete = constants.%Destroy.facet.4aa]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%x.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
