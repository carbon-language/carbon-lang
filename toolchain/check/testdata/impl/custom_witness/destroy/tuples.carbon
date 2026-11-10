// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/destroy.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/impl/custom_witness/destroy/tuples.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/impl/custom_witness/destroy/tuples.carbon

// --- types.carbon

library "[[@TEST_NAME]]";

class A {}
class B {}

// Used when mixing with `ImplsDestroy`.
class HasTrivialDestruction {}

class ImplsDestroy {
  impl as Core.DestroyImpl {
    fn Op(unused self) {}
  }
}

// --- trivial_empty.carbon

library "[[@TEST_NAME]]";

fn Test() {
  var unused x: () = ();
}

// --- trivial_nonempty.carbon

library "[[@TEST_NAME]]";

import library "types";

fn Test() {
  var unused x: (A, B) = ({}, {});
}

// --- todo_nontrivial_element_first.carbon

library "[[@TEST_NAME]]";

import library "types";

fn Test() {
  //@dump-sem-ir-begin
  var unused x: (ImplsDestroy, HasTrivialDestruction) = ({}, {});
  //@dump-sem-ir-end
}

// --- todo_nontrivial_element_second.carbon

library "[[@TEST_NAME]]";

import library "types";

fn Test() {
  //@dump-sem-ir-begin
  var unused x: (HasTrivialDestruction, ImplsDestroy) = ({}, {});
  //@dump-sem-ir-end
}

// --- todo_nontrivial_all_nontrivial_elements.carbon

library "[[@TEST_NAME]]";

import library "types";

class AlsoImplsDestroy {
  impl as Core.DestroyImpl {
    fn Op(unused ref self) {}
  }
}

fn Test() {
  //@dump-sem-ir-begin
  var unused x: (ImplsDestroy, AlsoImplsDestroy, AlsoImplsDestroy, ImplsDestroy) = ({}, {}, {}, {});
  //@dump-sem-ir-end
}

// --- trivial_nested.carbon

library "[[@TEST_NAME]]";

import library "types";

fn Test() {
  var unused x: (A, (B, A)) = ({}, ({}, {}));
}

// --- todo_nontrivial_nested.carbon

library "[[@TEST_NAME]]";

import library "types";

fn Test() {
  //@dump-sem-ir-begin
  var unused x: ((ImplsDestroy,),) = (({},),);
  //@dump-sem-ir-end
}

// CHECK:STDOUT: --- todo_nontrivial_element_first.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %ImplsDestroy: type = class_type @ImplsDestroy [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %HasTrivialDestruction: type = class_type @HasTrivialDestruction [concrete]
// CHECK:STDOUT:   %tuple.type.693: type = tuple_type (type, type) [concrete]
// CHECK:STDOUT:   %tuple.951: %tuple.type.693 = tuple_value (%ImplsDestroy, %HasTrivialDestruction) [concrete]
// CHECK:STDOUT:   %tuple.type.cac: type = tuple_type (%ImplsDestroy, %HasTrivialDestruction) [concrete]
// CHECK:STDOUT:   %pattern_type.bfd: type = pattern_type %tuple.type.cac [concrete]
// CHECK:STDOUT:   %x.patt: %pattern_type.bfd = ref_binding_pattern x [concrete]
// CHECK:STDOUT:   %x.var_patt: %pattern_type.bfd = var_pattern %x.patt [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %tuple.type.b6b: type = tuple_type (%empty_struct_type, %empty_struct_type) [concrete]
// CHECK:STDOUT:   %tuple.9a3: %tuple.type.b6b = tuple_value (%empty_struct, %empty_struct) [concrete]
// CHECK:STDOUT:   %ImplsDestroy.val: %ImplsDestroy = struct_value () [concrete]
// CHECK:STDOUT:   %HasTrivialDestruction.val: %HasTrivialDestruction = struct_value () [concrete]
// CHECK:STDOUT:   %tuple.afe: %tuple.type.cac = tuple_value (%ImplsDestroy.val, %HasTrivialDestruction.val) [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %T.09f: %SubobjectDestroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.type.dea: type = fn_type @T.as_type.as.DestroyImpl.impl.Op, @T.as_type.as.DestroyImpl.impl(%T.09f) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.6e5: %T.as_type.as.DestroyImpl.impl.Op.type.dea = struct_value () [symbolic]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.2: type = fn_type @SubobjectDestroy.WithSelf.Op.loc8_3.2 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.2: %SubobjectDestroy.WithSelf.Op.type.6870fd.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.2: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.2), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.3: type = fn_type @SubobjectDestroy.WithSelf.Op.loc8_3.3 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.3: %SubobjectDestroy.WithSelf.Op.type.6870fd.3 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.3: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.3), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.4: type = fn_type @SubobjectDestroy.WithSelf.Op.loc8_3.4 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.4: %SubobjectDestroy.WithSelf.Op.type.6870fd.4 = struct_value () [concrete]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.5bc: %SubobjectDestroy.type = facet_value %HasTrivialDestruction, (%custom_witness.b80ef6.3) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.4ca: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table.a5e, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.5bc) [concrete]
// CHECK:STDOUT:   %Destroy.facet.dba: %Destroy.type = facet_value %HasTrivialDestruction, (%DestroyImpl.impl_witness.4ca, %custom_witness.b80ef6.3) [concrete]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.906: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table.9a5 [concrete]
// CHECK:STDOUT:   %Destroy.facet.140: %Destroy.type = facet_value %ImplsDestroy, (%DestroyImpl.impl_witness.906, %custom_witness.b80ef6.2) [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.4: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.4), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.d74: %SubobjectDestroy.type = facet_value %tuple.type.cac, (%custom_witness.b80ef6.4) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.ab0: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table.a5e, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.d74) [concrete]
// CHECK:STDOUT:   %Destroy.facet.d4d: %Destroy.type = facet_value %tuple.type.cac, (%DestroyImpl.impl_witness.ab0, %custom_witness.b80ef6.4) [concrete]
// CHECK:STDOUT:   %ImplsDestroy.as.DestroyImpl.impl.Op.type: type = fn_type @ImplsDestroy.as.DestroyImpl.impl.Op [concrete]
// CHECK:STDOUT:   %ImplsDestroy.as.DestroyImpl.impl.Op: %ImplsDestroy.as.DestroyImpl.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.HasTrivialDestruction: type = import_ref Main//types, HasTrivialDestruction, loaded [concrete = constants.%HasTrivialDestruction]
// CHECK:STDOUT:   %Main.ImplsDestroy: type = import_ref Main//types, ImplsDestroy, loaded [concrete = constants.%ImplsDestroy]
// CHECK:STDOUT:   %Core.import_ref.74d: @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op.type (%T.as_type.as.DestroyImpl.impl.Op.type.dea) = import_ref Core//prelude/parts/destroy, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op (constants.%T.as_type.as.DestroyImpl.impl.Op.6e5)]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table.a5e = impl_witness_table (%Core.import_ref.74d), @T.as_type.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT:   %Main.import_ref.a07: %ImplsDestroy.as.DestroyImpl.impl.Op.type = import_ref Main//types, loc12_24, loaded [concrete = constants.%ImplsDestroy.as.DestroyImpl.impl.Op]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table.9a5 = impl_witness_table (%Main.import_ref.a07), @ImplsDestroy.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Test() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %x.var: ref %tuple.type.cac = var_storage %x.var_patt
// CHECK:STDOUT:   %.loc8_59.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc8_63.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc8_64.1: %tuple.type.b6b = tuple_literal (%.loc8_59.1, %.loc8_63.1) [concrete = constants.%tuple.9a3]
// CHECK:STDOUT:   %tuple.elem0: ref %ImplsDestroy = tuple_access %x.var, element0
// CHECK:STDOUT:   %.loc8_59.2: init %ImplsDestroy to %tuple.elem0 = class_init () [concrete = constants.%ImplsDestroy.val]
// CHECK:STDOUT:   %.loc8_64.2: init %ImplsDestroy = converted %.loc8_59.1, %.loc8_59.2 [concrete = constants.%ImplsDestroy.val]
// CHECK:STDOUT:   %tuple.elem1: ref %HasTrivialDestruction = tuple_access %x.var, element1
// CHECK:STDOUT:   %.loc8_63.2: init %HasTrivialDestruction to %tuple.elem1 = class_init () [concrete = constants.%HasTrivialDestruction.val]
// CHECK:STDOUT:   %.loc8_64.3: init %HasTrivialDestruction = converted %.loc8_63.1, %.loc8_63.2 [concrete = constants.%HasTrivialDestruction.val]
// CHECK:STDOUT:   %.loc8_64.4: init %tuple.type.cac to %x.var = tuple_init (%.loc8_64.2, %.loc8_64.3) [concrete = constants.%tuple.afe]
// CHECK:STDOUT:   %.loc8_3.1: init %tuple.type.cac = converted %.loc8_64.1, %.loc8_64.4 [concrete = constants.%tuple.afe]
// CHECK:STDOUT:   assign %x.var, %.loc8_3.1
// CHECK:STDOUT:   %.loc8_53.1: type = splice_block %.loc8_53.3 [concrete = constants.%tuple.type.cac] {
// CHECK:STDOUT:     %ImplsDestroy.ref: type = name_ref ImplsDestroy, imports.%Main.ImplsDestroy [concrete = constants.%ImplsDestroy]
// CHECK:STDOUT:     %HasTrivialDestruction.ref: type = name_ref HasTrivialDestruction, imports.%Main.HasTrivialDestruction [concrete = constants.%HasTrivialDestruction]
// CHECK:STDOUT:     %.loc8_53.2: %tuple.type.693 = tuple_literal (%ImplsDestroy.ref, %HasTrivialDestruction.ref) [concrete = constants.%tuple.951]
// CHECK:STDOUT:     %.loc8_53.3: type = converted %.loc8_53.2, constants.%tuple.type.cac [concrete = constants.%tuple.type.cac]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %x: ref %tuple.type.cac = wrapper_binding x, %x.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %pattern_type.bfd = ref_binding_pattern x [concrete = constants.%x.patt]
// CHECK:STDOUT:     %x.var_patt: %pattern_type.bfd = var_pattern %x.patt [concrete = constants.%x.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.facet.loc8_3.1: %Destroy.type = facet_value constants.%tuple.type.cac, (constants.%DestroyImpl.impl_witness.ab0, constants.%custom_witness.b80ef6.4) [concrete = constants.%Destroy.facet.d4d]
// CHECK:STDOUT:   %.loc8_3.2: %Destroy.type = converted constants.%tuple.type.cac, %Destroy.facet.loc8_3.1 [concrete = constants.%Destroy.facet.d4d]
// CHECK:STDOUT:   %Destroy.facet.loc8_3.2: %Destroy.type = facet_value constants.%tuple.type.cac, (constants.%DestroyImpl.impl_witness.ab0, constants.%custom_witness.b80ef6.4) [concrete = constants.%Destroy.facet.d4d]
// CHECK:STDOUT:   %.loc8_3.3: %Destroy.type = converted constants.%tuple.type.cac, %Destroy.facet.loc8_3.2 [concrete = constants.%Destroy.facet.d4d]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%x.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc8_3.1(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc8_3.2(%self.param: ref %ImplsDestroy) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc8_3.3(%self.param: ref %HasTrivialDestruction) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc8_3.4(%self.param: ref %tuple.type.cac) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1]
// CHECK:STDOUT:   %tuple.elem1: ref %HasTrivialDestruction = tuple_access %self.param, element1
// CHECK:STDOUT:   %Destroy.facet.loc8_3.1: %Destroy.type = facet_value constants.%HasTrivialDestruction, (constants.%DestroyImpl.impl_witness.4ca, constants.%custom_witness.b80ef6.3) [concrete = constants.%Destroy.facet.dba]
// CHECK:STDOUT:   %.loc8_3.1: %Destroy.type = converted constants.%HasTrivialDestruction, %Destroy.facet.loc8_3.1 [concrete = constants.%Destroy.facet.dba]
// CHECK:STDOUT:   %Destroy.facet.loc8_3.2: %Destroy.type = facet_value constants.%HasTrivialDestruction, (constants.%DestroyImpl.impl_witness.4ca, constants.%custom_witness.b80ef6.3) [concrete = constants.%Destroy.facet.dba]
// CHECK:STDOUT:   %.loc8_3.2: %Destroy.type = converted constants.%HasTrivialDestruction, %Destroy.facet.loc8_3.2 [concrete = constants.%Destroy.facet.dba]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc8_3.1: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%tuple.elem1)
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT:   %tuple.elem0: ref %ImplsDestroy = tuple_access %self.param, element0
// CHECK:STDOUT:   %Destroy.facet.loc8_3.3: %Destroy.type = facet_value constants.%ImplsDestroy, (constants.%DestroyImpl.impl_witness.906, constants.%custom_witness.b80ef6.2) [concrete = constants.%Destroy.facet.140]
// CHECK:STDOUT:   %.loc8_3.3: %Destroy.type = converted constants.%ImplsDestroy, %Destroy.facet.loc8_3.3 [concrete = constants.%Destroy.facet.140]
// CHECK:STDOUT:   %Destroy.facet.loc8_3.4: %Destroy.type = facet_value constants.%ImplsDestroy, (constants.%DestroyImpl.impl_witness.906, constants.%custom_witness.b80ef6.2) [concrete = constants.%Destroy.facet.140]
// CHECK:STDOUT:   %.loc8_3.4: %Destroy.type = converted constants.%ImplsDestroy, %Destroy.facet.loc8_3.4 [concrete = constants.%Destroy.facet.140]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc8_3.2: init %empty_tuple.type = call %SelfDestruct.specific_fn.2(%tuple.elem0)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- todo_nontrivial_element_second.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %HasTrivialDestruction: type = class_type @HasTrivialDestruction [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %ImplsDestroy: type = class_type @ImplsDestroy [concrete]
// CHECK:STDOUT:   %tuple.type.693: type = tuple_type (type, type) [concrete]
// CHECK:STDOUT:   %tuple.8d7: %tuple.type.693 = tuple_value (%HasTrivialDestruction, %ImplsDestroy) [concrete]
// CHECK:STDOUT:   %tuple.type.871: type = tuple_type (%HasTrivialDestruction, %ImplsDestroy) [concrete]
// CHECK:STDOUT:   %pattern_type.dc0: type = pattern_type %tuple.type.871 [concrete]
// CHECK:STDOUT:   %x.patt: %pattern_type.dc0 = ref_binding_pattern x [concrete]
// CHECK:STDOUT:   %x.var_patt: %pattern_type.dc0 = var_pattern %x.patt [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %tuple.type.b6b: type = tuple_type (%empty_struct_type, %empty_struct_type) [concrete]
// CHECK:STDOUT:   %tuple.9a3: %tuple.type.b6b = tuple_value (%empty_struct, %empty_struct) [concrete]
// CHECK:STDOUT:   %HasTrivialDestruction.val: %HasTrivialDestruction = struct_value () [concrete]
// CHECK:STDOUT:   %ImplsDestroy.val: %ImplsDestroy = struct_value () [concrete]
// CHECK:STDOUT:   %tuple.e1d: %tuple.type.871 = tuple_value (%HasTrivialDestruction.val, %ImplsDestroy.val) [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %T.09f: %SubobjectDestroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.type.dea: type = fn_type @T.as_type.as.DestroyImpl.impl.Op, @T.as_type.as.DestroyImpl.impl(%T.09f) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.6e5: %T.as_type.as.DestroyImpl.impl.Op.type.dea = struct_value () [symbolic]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.2: type = fn_type @SubobjectDestroy.WithSelf.Op.loc8_3.2 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.2: %SubobjectDestroy.WithSelf.Op.type.6870fd.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.2: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.2), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.3: type = fn_type @SubobjectDestroy.WithSelf.Op.loc8_3.3 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.3: %SubobjectDestroy.WithSelf.Op.type.6870fd.3 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.3: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.3), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.4: type = fn_type @SubobjectDestroy.WithSelf.Op.loc8_3.4 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.4: %SubobjectDestroy.WithSelf.Op.type.6870fd.4 = struct_value () [concrete]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.906: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table.9a5 [concrete]
// CHECK:STDOUT:   %Destroy.facet.140: %Destroy.type = facet_value %ImplsDestroy, (%DestroyImpl.impl_witness.906, %custom_witness.b80ef6.3) [concrete]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.5bc: %SubobjectDestroy.type = facet_value %HasTrivialDestruction, (%custom_witness.b80ef6.2) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.4ca: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table.a5e, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.5bc) [concrete]
// CHECK:STDOUT:   %Destroy.facet.dba: %Destroy.type = facet_value %HasTrivialDestruction, (%DestroyImpl.impl_witness.4ca, %custom_witness.b80ef6.2) [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.4: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.4), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.b98: %SubobjectDestroy.type = facet_value %tuple.type.871, (%custom_witness.b80ef6.4) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.de7: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table.a5e, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.b98) [concrete]
// CHECK:STDOUT:   %Destroy.facet.ec3: %Destroy.type = facet_value %tuple.type.871, (%DestroyImpl.impl_witness.de7, %custom_witness.b80ef6.4) [concrete]
// CHECK:STDOUT:   %ImplsDestroy.as.DestroyImpl.impl.Op.type: type = fn_type @ImplsDestroy.as.DestroyImpl.impl.Op [concrete]
// CHECK:STDOUT:   %ImplsDestroy.as.DestroyImpl.impl.Op: %ImplsDestroy.as.DestroyImpl.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.HasTrivialDestruction: type = import_ref Main//types, HasTrivialDestruction, loaded [concrete = constants.%HasTrivialDestruction]
// CHECK:STDOUT:   %Main.ImplsDestroy: type = import_ref Main//types, ImplsDestroy, loaded [concrete = constants.%ImplsDestroy]
// CHECK:STDOUT:   %Core.import_ref.74d: @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op.type (%T.as_type.as.DestroyImpl.impl.Op.type.dea) = import_ref Core//prelude/parts/destroy, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op (constants.%T.as_type.as.DestroyImpl.impl.Op.6e5)]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table.a5e = impl_witness_table (%Core.import_ref.74d), @T.as_type.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT:   %Main.import_ref.a07: %ImplsDestroy.as.DestroyImpl.impl.Op.type = import_ref Main//types, loc12_24, loaded [concrete = constants.%ImplsDestroy.as.DestroyImpl.impl.Op]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table.9a5 = impl_witness_table (%Main.import_ref.a07), @ImplsDestroy.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Test() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %x.var: ref %tuple.type.871 = var_storage %x.var_patt
// CHECK:STDOUT:   %.loc8_59.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc8_63.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc8_64.1: %tuple.type.b6b = tuple_literal (%.loc8_59.1, %.loc8_63.1) [concrete = constants.%tuple.9a3]
// CHECK:STDOUT:   %tuple.elem0: ref %HasTrivialDestruction = tuple_access %x.var, element0
// CHECK:STDOUT:   %.loc8_59.2: init %HasTrivialDestruction to %tuple.elem0 = class_init () [concrete = constants.%HasTrivialDestruction.val]
// CHECK:STDOUT:   %.loc8_64.2: init %HasTrivialDestruction = converted %.loc8_59.1, %.loc8_59.2 [concrete = constants.%HasTrivialDestruction.val]
// CHECK:STDOUT:   %tuple.elem1: ref %ImplsDestroy = tuple_access %x.var, element1
// CHECK:STDOUT:   %.loc8_63.2: init %ImplsDestroy to %tuple.elem1 = class_init () [concrete = constants.%ImplsDestroy.val]
// CHECK:STDOUT:   %.loc8_64.3: init %ImplsDestroy = converted %.loc8_63.1, %.loc8_63.2 [concrete = constants.%ImplsDestroy.val]
// CHECK:STDOUT:   %.loc8_64.4: init %tuple.type.871 to %x.var = tuple_init (%.loc8_64.2, %.loc8_64.3) [concrete = constants.%tuple.e1d]
// CHECK:STDOUT:   %.loc8_3.1: init %tuple.type.871 = converted %.loc8_64.1, %.loc8_64.4 [concrete = constants.%tuple.e1d]
// CHECK:STDOUT:   assign %x.var, %.loc8_3.1
// CHECK:STDOUT:   %.loc8_53.1: type = splice_block %.loc8_53.3 [concrete = constants.%tuple.type.871] {
// CHECK:STDOUT:     %HasTrivialDestruction.ref: type = name_ref HasTrivialDestruction, imports.%Main.HasTrivialDestruction [concrete = constants.%HasTrivialDestruction]
// CHECK:STDOUT:     %ImplsDestroy.ref: type = name_ref ImplsDestroy, imports.%Main.ImplsDestroy [concrete = constants.%ImplsDestroy]
// CHECK:STDOUT:     %.loc8_53.2: %tuple.type.693 = tuple_literal (%HasTrivialDestruction.ref, %ImplsDestroy.ref) [concrete = constants.%tuple.8d7]
// CHECK:STDOUT:     %.loc8_53.3: type = converted %.loc8_53.2, constants.%tuple.type.871 [concrete = constants.%tuple.type.871]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %x: ref %tuple.type.871 = wrapper_binding x, %x.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %pattern_type.dc0 = ref_binding_pattern x [concrete = constants.%x.patt]
// CHECK:STDOUT:     %x.var_patt: %pattern_type.dc0 = var_pattern %x.patt [concrete = constants.%x.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.facet.loc8_3.1: %Destroy.type = facet_value constants.%tuple.type.871, (constants.%DestroyImpl.impl_witness.de7, constants.%custom_witness.b80ef6.4) [concrete = constants.%Destroy.facet.ec3]
// CHECK:STDOUT:   %.loc8_3.2: %Destroy.type = converted constants.%tuple.type.871, %Destroy.facet.loc8_3.1 [concrete = constants.%Destroy.facet.ec3]
// CHECK:STDOUT:   %Destroy.facet.loc8_3.2: %Destroy.type = facet_value constants.%tuple.type.871, (constants.%DestroyImpl.impl_witness.de7, constants.%custom_witness.b80ef6.4) [concrete = constants.%Destroy.facet.ec3]
// CHECK:STDOUT:   %.loc8_3.3: %Destroy.type = converted constants.%tuple.type.871, %Destroy.facet.loc8_3.2 [concrete = constants.%Destroy.facet.ec3]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%x.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc8_3.1(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc8_3.2(%self.param: ref %HasTrivialDestruction) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc8_3.3(%self.param: ref %ImplsDestroy) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc8_3.4(%self.param: ref %tuple.type.871) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1]
// CHECK:STDOUT:   %tuple.elem1: ref %ImplsDestroy = tuple_access %self.param, element1
// CHECK:STDOUT:   %Destroy.facet.loc8_3.1: %Destroy.type = facet_value constants.%ImplsDestroy, (constants.%DestroyImpl.impl_witness.906, constants.%custom_witness.b80ef6.3) [concrete = constants.%Destroy.facet.140]
// CHECK:STDOUT:   %.loc8_3.1: %Destroy.type = converted constants.%ImplsDestroy, %Destroy.facet.loc8_3.1 [concrete = constants.%Destroy.facet.140]
// CHECK:STDOUT:   %Destroy.facet.loc8_3.2: %Destroy.type = facet_value constants.%ImplsDestroy, (constants.%DestroyImpl.impl_witness.906, constants.%custom_witness.b80ef6.3) [concrete = constants.%Destroy.facet.140]
// CHECK:STDOUT:   %.loc8_3.2: %Destroy.type = converted constants.%ImplsDestroy, %Destroy.facet.loc8_3.2 [concrete = constants.%Destroy.facet.140]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc8_3.1: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%tuple.elem1)
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT:   %tuple.elem0: ref %HasTrivialDestruction = tuple_access %self.param, element0
// CHECK:STDOUT:   %Destroy.facet.loc8_3.3: %Destroy.type = facet_value constants.%HasTrivialDestruction, (constants.%DestroyImpl.impl_witness.4ca, constants.%custom_witness.b80ef6.2) [concrete = constants.%Destroy.facet.dba]
// CHECK:STDOUT:   %.loc8_3.3: %Destroy.type = converted constants.%HasTrivialDestruction, %Destroy.facet.loc8_3.3 [concrete = constants.%Destroy.facet.dba]
// CHECK:STDOUT:   %Destroy.facet.loc8_3.4: %Destroy.type = facet_value constants.%HasTrivialDestruction, (constants.%DestroyImpl.impl_witness.4ca, constants.%custom_witness.b80ef6.2) [concrete = constants.%Destroy.facet.dba]
// CHECK:STDOUT:   %.loc8_3.4: %Destroy.type = converted constants.%HasTrivialDestruction, %Destroy.facet.loc8_3.4 [concrete = constants.%Destroy.facet.dba]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc8_3.2: init %empty_tuple.type = call %SelfDestruct.specific_fn.2(%tuple.elem0)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- todo_nontrivial_all_nontrivial_elements.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %AlsoImplsDestroy: type = class_type @AlsoImplsDestroy [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.e7f: <witness> = impl_witness @AlsoImplsDestroy.as.DestroyImpl.impl.%DestroyImpl.impl_witness_table [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %ImplsDestroy: type = class_type @ImplsDestroy [concrete]
// CHECK:STDOUT:   %tuple.type.c31: type = tuple_type (type, type, type, type) [concrete]
// CHECK:STDOUT:   %tuple.53b: %tuple.type.c31 = tuple_value (%ImplsDestroy, %AlsoImplsDestroy, %AlsoImplsDestroy, %ImplsDestroy) [concrete]
// CHECK:STDOUT:   %tuple.type.68b: type = tuple_type (%ImplsDestroy, %AlsoImplsDestroy, %AlsoImplsDestroy, %ImplsDestroy) [concrete]
// CHECK:STDOUT:   %pattern_type.d39: type = pattern_type %tuple.type.68b [concrete]
// CHECK:STDOUT:   %x.patt: %pattern_type.d39 = ref_binding_pattern x [concrete]
// CHECK:STDOUT:   %x.var_patt: %pattern_type.d39 = var_pattern %x.patt [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %tuple.type.8ec: type = tuple_type (%empty_struct_type, %empty_struct_type, %empty_struct_type, %empty_struct_type) [concrete]
// CHECK:STDOUT:   %tuple.5a3: %tuple.type.8ec = tuple_value (%empty_struct, %empty_struct, %empty_struct, %empty_struct) [concrete]
// CHECK:STDOUT:   %ImplsDestroy.val: %ImplsDestroy = struct_value () [concrete]
// CHECK:STDOUT:   %AlsoImplsDestroy.val: %AlsoImplsDestroy = struct_value () [concrete]
// CHECK:STDOUT:   %tuple.c7d: %tuple.type.68b = tuple_value (%ImplsDestroy.val, %AlsoImplsDestroy.val, %AlsoImplsDestroy.val, %ImplsDestroy.val) [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %T.09f: %SubobjectDestroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.type.dea: type = fn_type @T.as_type.as.DestroyImpl.impl.Op, @T.as_type.as.DestroyImpl.impl(%T.09f) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.6e5: %T.as_type.as.DestroyImpl.impl.Op.type.dea = struct_value () [symbolic]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.2: type = fn_type @SubobjectDestroy.WithSelf.Op.loc14_3.2 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.2: %SubobjectDestroy.WithSelf.Op.type.6870fd.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.2: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.2), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.3: type = fn_type @SubobjectDestroy.WithSelf.Op.loc14_3.3 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.3: %SubobjectDestroy.WithSelf.Op.type.6870fd.3 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.3: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.3), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.4: type = fn_type @SubobjectDestroy.WithSelf.Op.loc14_3.4 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.4: %SubobjectDestroy.WithSelf.Op.type.6870fd.4 = struct_value () [concrete]
// CHECK:STDOUT:   %int_3: Core.IntLiteral = int_value 3 [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.906: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table.9a5 [concrete]
// CHECK:STDOUT:   %Destroy.facet.140: %Destroy.type = facet_value %ImplsDestroy, (%DestroyImpl.impl_witness.906, %custom_witness.b80ef6.2) [concrete]
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [concrete]
// CHECK:STDOUT:   %Destroy.facet.122: %Destroy.type = facet_value %AlsoImplsDestroy, (%DestroyImpl.impl_witness.e7f, %custom_witness.b80ef6.3) [concrete]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.4: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.4), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.d00: %SubobjectDestroy.type = facet_value %tuple.type.68b, (%custom_witness.b80ef6.4) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.2fb: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table.a5e, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.d00) [concrete]
// CHECK:STDOUT:   %Destroy.facet.0b4: %Destroy.type = facet_value %tuple.type.68b, (%DestroyImpl.impl_witness.2fb, %custom_witness.b80ef6.4) [concrete]
// CHECK:STDOUT:   %ImplsDestroy.as.DestroyImpl.impl.Op.type: type = fn_type @ImplsDestroy.as.DestroyImpl.impl.Op [concrete]
// CHECK:STDOUT:   %ImplsDestroy.as.DestroyImpl.impl.Op: %ImplsDestroy.as.DestroyImpl.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.ImplsDestroy: type = import_ref Main//types, ImplsDestroy, loaded [concrete = constants.%ImplsDestroy]
// CHECK:STDOUT:   %Core.import_ref.74d: @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op.type (%T.as_type.as.DestroyImpl.impl.Op.type.dea) = import_ref Core//prelude/parts/destroy, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op (constants.%T.as_type.as.DestroyImpl.impl.Op.6e5)]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table.a5e = impl_witness_table (%Core.import_ref.74d), @T.as_type.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT:   %Main.import_ref.a07: %ImplsDestroy.as.DestroyImpl.impl.Op.type = import_ref Main//types, loc12_24, loaded [concrete = constants.%ImplsDestroy.as.DestroyImpl.impl.Op]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table.9a5 = impl_witness_table (%Main.import_ref.a07), @ImplsDestroy.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Test() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %x.var: ref %tuple.type.68b = var_storage %x.var_patt
// CHECK:STDOUT:   %.loc14_86.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc14_90.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc14_94.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc14_98.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc14_99.1: %tuple.type.8ec = tuple_literal (%.loc14_86.1, %.loc14_90.1, %.loc14_94.1, %.loc14_98.1) [concrete = constants.%tuple.5a3]
// CHECK:STDOUT:   %tuple.elem0: ref %ImplsDestroy = tuple_access %x.var, element0
// CHECK:STDOUT:   %.loc14_86.2: init %ImplsDestroy to %tuple.elem0 = class_init () [concrete = constants.%ImplsDestroy.val]
// CHECK:STDOUT:   %.loc14_99.2: init %ImplsDestroy = converted %.loc14_86.1, %.loc14_86.2 [concrete = constants.%ImplsDestroy.val]
// CHECK:STDOUT:   %tuple.elem1: ref %AlsoImplsDestroy = tuple_access %x.var, element1
// CHECK:STDOUT:   %.loc14_90.2: init %AlsoImplsDestroy to %tuple.elem1 = class_init () [concrete = constants.%AlsoImplsDestroy.val]
// CHECK:STDOUT:   %.loc14_99.3: init %AlsoImplsDestroy = converted %.loc14_90.1, %.loc14_90.2 [concrete = constants.%AlsoImplsDestroy.val]
// CHECK:STDOUT:   %tuple.elem2: ref %AlsoImplsDestroy = tuple_access %x.var, element2
// CHECK:STDOUT:   %.loc14_94.2: init %AlsoImplsDestroy to %tuple.elem2 = class_init () [concrete = constants.%AlsoImplsDestroy.val]
// CHECK:STDOUT:   %.loc14_99.4: init %AlsoImplsDestroy = converted %.loc14_94.1, %.loc14_94.2 [concrete = constants.%AlsoImplsDestroy.val]
// CHECK:STDOUT:   %tuple.elem3: ref %ImplsDestroy = tuple_access %x.var, element3
// CHECK:STDOUT:   %.loc14_98.2: init %ImplsDestroy to %tuple.elem3 = class_init () [concrete = constants.%ImplsDestroy.val]
// CHECK:STDOUT:   %.loc14_99.5: init %ImplsDestroy = converted %.loc14_98.1, %.loc14_98.2 [concrete = constants.%ImplsDestroy.val]
// CHECK:STDOUT:   %.loc14_99.6: init %tuple.type.68b to %x.var = tuple_init (%.loc14_99.2, %.loc14_99.3, %.loc14_99.4, %.loc14_99.5) [concrete = constants.%tuple.c7d]
// CHECK:STDOUT:   %.loc14_3.1: init %tuple.type.68b = converted %.loc14_99.1, %.loc14_99.6 [concrete = constants.%tuple.c7d]
// CHECK:STDOUT:   assign %x.var, %.loc14_3.1
// CHECK:STDOUT:   %.loc14_80.1: type = splice_block %.loc14_80.3 [concrete = constants.%tuple.type.68b] {
// CHECK:STDOUT:     %ImplsDestroy.ref.loc14_18: type = name_ref ImplsDestroy, imports.%Main.ImplsDestroy [concrete = constants.%ImplsDestroy]
// CHECK:STDOUT:     %AlsoImplsDestroy.ref.loc14_32: type = name_ref AlsoImplsDestroy, file.%AlsoImplsDestroy.decl [concrete = constants.%AlsoImplsDestroy]
// CHECK:STDOUT:     %AlsoImplsDestroy.ref.loc14_50: type = name_ref AlsoImplsDestroy, file.%AlsoImplsDestroy.decl [concrete = constants.%AlsoImplsDestroy]
// CHECK:STDOUT:     %ImplsDestroy.ref.loc14_68: type = name_ref ImplsDestroy, imports.%Main.ImplsDestroy [concrete = constants.%ImplsDestroy]
// CHECK:STDOUT:     %.loc14_80.2: %tuple.type.c31 = tuple_literal (%ImplsDestroy.ref.loc14_18, %AlsoImplsDestroy.ref.loc14_32, %AlsoImplsDestroy.ref.loc14_50, %ImplsDestroy.ref.loc14_68) [concrete = constants.%tuple.53b]
// CHECK:STDOUT:     %.loc14_80.3: type = converted %.loc14_80.2, constants.%tuple.type.68b [concrete = constants.%tuple.type.68b]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %x: ref %tuple.type.68b = wrapper_binding x, %x.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %pattern_type.d39 = ref_binding_pattern x [concrete = constants.%x.patt]
// CHECK:STDOUT:     %x.var_patt: %pattern_type.d39 = var_pattern %x.patt [concrete = constants.%x.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.facet.loc14_3.1: %Destroy.type = facet_value constants.%tuple.type.68b, (constants.%DestroyImpl.impl_witness.2fb, constants.%custom_witness.b80ef6.4) [concrete = constants.%Destroy.facet.0b4]
// CHECK:STDOUT:   %.loc14_3.2: %Destroy.type = converted constants.%tuple.type.68b, %Destroy.facet.loc14_3.1 [concrete = constants.%Destroy.facet.0b4]
// CHECK:STDOUT:   %Destroy.facet.loc14_3.2: %Destroy.type = facet_value constants.%tuple.type.68b, (constants.%DestroyImpl.impl_witness.2fb, constants.%custom_witness.b80ef6.4) [concrete = constants.%Destroy.facet.0b4]
// CHECK:STDOUT:   %.loc14_3.3: %Destroy.type = converted constants.%tuple.type.68b, %Destroy.facet.loc14_3.2 [concrete = constants.%Destroy.facet.0b4]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%x.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc14_3.1(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc14_3.2(%self.param: ref %ImplsDestroy) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc14_3.3(%self.param: ref %AlsoImplsDestroy) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc14_3.4(%self.param: ref %tuple.type.68b) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_3: Core.IntLiteral = int_value 3 [concrete = constants.%int_3]
// CHECK:STDOUT:   %tuple.elem3: ref %ImplsDestroy = tuple_access %self.param, element3
// CHECK:STDOUT:   %Destroy.facet.loc14_3.1: %Destroy.type = facet_value constants.%ImplsDestroy, (constants.%DestroyImpl.impl_witness.906, constants.%custom_witness.b80ef6.2) [concrete = constants.%Destroy.facet.140]
// CHECK:STDOUT:   %.loc14_3.1: %Destroy.type = converted constants.%ImplsDestroy, %Destroy.facet.loc14_3.1 [concrete = constants.%Destroy.facet.140]
// CHECK:STDOUT:   %Destroy.facet.loc14_3.2: %Destroy.type = facet_value constants.%ImplsDestroy, (constants.%DestroyImpl.impl_witness.906, constants.%custom_witness.b80ef6.2) [concrete = constants.%Destroy.facet.140]
// CHECK:STDOUT:   %.loc14_3.2: %Destroy.type = converted constants.%ImplsDestroy, %Destroy.facet.loc14_3.2 [concrete = constants.%Destroy.facet.140]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc14_3.1: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%tuple.elem3)
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [concrete = constants.%int_2]
// CHECK:STDOUT:   %tuple.elem2: ref %AlsoImplsDestroy = tuple_access %self.param, element2
// CHECK:STDOUT:   %Destroy.facet.loc14_3.3: %Destroy.type = facet_value constants.%AlsoImplsDestroy, (constants.%DestroyImpl.impl_witness.e7f, constants.%custom_witness.b80ef6.3) [concrete = constants.%Destroy.facet.122]
// CHECK:STDOUT:   %.loc14_3.3: %Destroy.type = converted constants.%AlsoImplsDestroy, %Destroy.facet.loc14_3.3 [concrete = constants.%Destroy.facet.122]
// CHECK:STDOUT:   %Destroy.facet.loc14_3.4: %Destroy.type = facet_value constants.%AlsoImplsDestroy, (constants.%DestroyImpl.impl_witness.e7f, constants.%custom_witness.b80ef6.3) [concrete = constants.%Destroy.facet.122]
// CHECK:STDOUT:   %.loc14_3.4: %Destroy.type = converted constants.%AlsoImplsDestroy, %Destroy.facet.loc14_3.4 [concrete = constants.%Destroy.facet.122]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc14_3.2: init %empty_tuple.type = call %SelfDestruct.specific_fn.2(%tuple.elem2)
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1]
// CHECK:STDOUT:   %tuple.elem1: ref %AlsoImplsDestroy = tuple_access %self.param, element1
// CHECK:STDOUT:   %Destroy.facet.loc14_3.5: %Destroy.type = facet_value constants.%AlsoImplsDestroy, (constants.%DestroyImpl.impl_witness.e7f, constants.%custom_witness.b80ef6.3) [concrete = constants.%Destroy.facet.122]
// CHECK:STDOUT:   %.loc14_3.5: %Destroy.type = converted constants.%AlsoImplsDestroy, %Destroy.facet.loc14_3.5 [concrete = constants.%Destroy.facet.122]
// CHECK:STDOUT:   %Destroy.facet.loc14_3.6: %Destroy.type = facet_value constants.%AlsoImplsDestroy, (constants.%DestroyImpl.impl_witness.e7f, constants.%custom_witness.b80ef6.3) [concrete = constants.%Destroy.facet.122]
// CHECK:STDOUT:   %.loc14_3.6: %Destroy.type = converted constants.%AlsoImplsDestroy, %Destroy.facet.loc14_3.6 [concrete = constants.%Destroy.facet.122]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc14_3.3: init %empty_tuple.type = call %SelfDestruct.specific_fn.3(%tuple.elem1)
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT:   %tuple.elem0: ref %ImplsDestroy = tuple_access %self.param, element0
// CHECK:STDOUT:   %Destroy.facet.loc14_3.7: %Destroy.type = facet_value constants.%ImplsDestroy, (constants.%DestroyImpl.impl_witness.906, constants.%custom_witness.b80ef6.2) [concrete = constants.%Destroy.facet.140]
// CHECK:STDOUT:   %.loc14_3.7: %Destroy.type = converted constants.%ImplsDestroy, %Destroy.facet.loc14_3.7 [concrete = constants.%Destroy.facet.140]
// CHECK:STDOUT:   %Destroy.facet.loc14_3.8: %Destroy.type = facet_value constants.%ImplsDestroy, (constants.%DestroyImpl.impl_witness.906, constants.%custom_witness.b80ef6.2) [concrete = constants.%Destroy.facet.140]
// CHECK:STDOUT:   %.loc14_3.8: %Destroy.type = converted constants.%ImplsDestroy, %Destroy.facet.loc14_3.8 [concrete = constants.%Destroy.facet.140]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc14_3.4: init %empty_tuple.type = call %SelfDestruct.specific_fn.4(%tuple.elem0)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- todo_nontrivial_nested.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %ImplsDestroy: type = class_type @ImplsDestroy [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %tuple.type.135: type = tuple_type (type) [concrete]
// CHECK:STDOUT:   %tuple.1b6: %tuple.type.135 = tuple_value (%ImplsDestroy) [concrete]
// CHECK:STDOUT:   %tuple.type.712: type = tuple_type (%tuple.type.135) [concrete]
// CHECK:STDOUT:   %tuple.8a2: %tuple.type.712 = tuple_value (%tuple.1b6) [concrete]
// CHECK:STDOUT:   %tuple.type.159: type = tuple_type (%ImplsDestroy) [concrete]
// CHECK:STDOUT:   %tuple.type.efa: type = tuple_type (%tuple.type.159) [concrete]
// CHECK:STDOUT:   %pattern_type.87a: type = pattern_type %tuple.type.efa [concrete]
// CHECK:STDOUT:   %x.patt: %pattern_type.87a = ref_binding_pattern x [concrete]
// CHECK:STDOUT:   %x.var_patt: %pattern_type.87a = var_pattern %x.patt [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %tuple.type.572: type = tuple_type (%empty_struct_type) [concrete]
// CHECK:STDOUT:   %tuple.115: %tuple.type.572 = tuple_value (%empty_struct) [concrete]
// CHECK:STDOUT:   %tuple.type.6c7: type = tuple_type (%tuple.type.572) [concrete]
// CHECK:STDOUT:   %tuple.9f1: %tuple.type.6c7 = tuple_value (%tuple.115) [concrete]
// CHECK:STDOUT:   %ImplsDestroy.val: %ImplsDestroy = struct_value () [concrete]
// CHECK:STDOUT:   %tuple.a0b: %tuple.type.159 = tuple_value (%ImplsDestroy.val) [concrete]
// CHECK:STDOUT:   %tuple.3c4: %tuple.type.efa = tuple_value (%tuple.a0b) [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %T.09f: %SubobjectDestroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.type.dea: type = fn_type @T.as_type.as.DestroyImpl.impl.Op, @T.as_type.as.DestroyImpl.impl(%T.09f) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.6e5: %T.as_type.as.DestroyImpl.impl.Op.type.dea = struct_value () [symbolic]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.2: type = fn_type @SubobjectDestroy.WithSelf.Op.loc8_3.2 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.2: %SubobjectDestroy.WithSelf.Op.type.6870fd.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.2: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.2), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.3: type = fn_type @SubobjectDestroy.WithSelf.Op.loc8_3.3 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.3: %SubobjectDestroy.WithSelf.Op.type.6870fd.3 = struct_value () [concrete]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.906: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table.9a5 [concrete]
// CHECK:STDOUT:   %Destroy.facet.140: %Destroy.type = facet_value %ImplsDestroy, (%DestroyImpl.impl_witness.906, %custom_witness.b80ef6.2) [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.3: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.3), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.4: type = fn_type @SubobjectDestroy.WithSelf.Op.loc8_3.4 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.4: %SubobjectDestroy.WithSelf.Op.type.6870fd.4 = struct_value () [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.5c3: %SubobjectDestroy.type = facet_value %tuple.type.159, (%custom_witness.b80ef6.3) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.aa8: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table.a5e, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.5c3) [concrete]
// CHECK:STDOUT:   %Destroy.facet.c31: %Destroy.type = facet_value %tuple.type.159, (%DestroyImpl.impl_witness.aa8, %custom_witness.b80ef6.3) [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.4: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.4), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.778: %SubobjectDestroy.type = facet_value %tuple.type.efa, (%custom_witness.b80ef6.4) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.a9f: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table.a5e, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.778) [concrete]
// CHECK:STDOUT:   %Destroy.facet.210: %Destroy.type = facet_value %tuple.type.efa, (%DestroyImpl.impl_witness.a9f, %custom_witness.b80ef6.4) [concrete]
// CHECK:STDOUT:   %ImplsDestroy.as.DestroyImpl.impl.Op.type: type = fn_type @ImplsDestroy.as.DestroyImpl.impl.Op [concrete]
// CHECK:STDOUT:   %ImplsDestroy.as.DestroyImpl.impl.Op: %ImplsDestroy.as.DestroyImpl.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.ImplsDestroy: type = import_ref Main//types, ImplsDestroy, loaded [concrete = constants.%ImplsDestroy]
// CHECK:STDOUT:   %Core.import_ref.74d: @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op.type (%T.as_type.as.DestroyImpl.impl.Op.type.dea) = import_ref Core//prelude/parts/destroy, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op (constants.%T.as_type.as.DestroyImpl.impl.Op.6e5)]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table.a5e = impl_witness_table (%Core.import_ref.74d), @T.as_type.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT:   %Main.import_ref.a07: %ImplsDestroy.as.DestroyImpl.impl.Op.type = import_ref Main//types, loc12_24, loaded [concrete = constants.%ImplsDestroy.as.DestroyImpl.impl.Op]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table.9a5 = impl_witness_table (%Main.import_ref.a07), @ImplsDestroy.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Test() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %x.var: ref %tuple.type.efa = var_storage %x.var_patt
// CHECK:STDOUT:   %.loc8_41.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc8_43.1: %tuple.type.572 = tuple_literal (%.loc8_41.1) [concrete = constants.%tuple.115]
// CHECK:STDOUT:   %.loc8_45.1: %tuple.type.6c7 = tuple_literal (%.loc8_43.1) [concrete = constants.%tuple.9f1]
// CHECK:STDOUT:   %tuple.elem0.loc8_45: ref %tuple.type.159 = tuple_access %x.var, element0
// CHECK:STDOUT:   %tuple.elem0.loc8_43: ref %ImplsDestroy = tuple_access %tuple.elem0.loc8_45, element0
// CHECK:STDOUT:   %.loc8_41.2: init %ImplsDestroy to %tuple.elem0.loc8_43 = class_init () [concrete = constants.%ImplsDestroy.val]
// CHECK:STDOUT:   %.loc8_43.2: init %ImplsDestroy = converted %.loc8_41.1, %.loc8_41.2 [concrete = constants.%ImplsDestroy.val]
// CHECK:STDOUT:   %.loc8_43.3: init %tuple.type.159 to %tuple.elem0.loc8_45 = tuple_init (%.loc8_43.2) [concrete = constants.%tuple.a0b]
// CHECK:STDOUT:   %.loc8_45.2: init %tuple.type.159 = converted %.loc8_43.1, %.loc8_43.3 [concrete = constants.%tuple.a0b]
// CHECK:STDOUT:   %.loc8_45.3: init %tuple.type.efa to %x.var = tuple_init (%.loc8_45.2) [concrete = constants.%tuple.3c4]
// CHECK:STDOUT:   %.loc8_3.1: init %tuple.type.efa = converted %.loc8_45.1, %.loc8_45.3 [concrete = constants.%tuple.3c4]
// CHECK:STDOUT:   assign %x.var, %.loc8_3.1
// CHECK:STDOUT:   %.loc8_34.1: type = splice_block %.loc8_34.4 [concrete = constants.%tuple.type.efa] {
// CHECK:STDOUT:     %ImplsDestroy.ref: type = name_ref ImplsDestroy, imports.%Main.ImplsDestroy [concrete = constants.%ImplsDestroy]
// CHECK:STDOUT:     %.loc8_32: %tuple.type.135 = tuple_literal (%ImplsDestroy.ref) [concrete = constants.%tuple.1b6]
// CHECK:STDOUT:     %.loc8_34.2: %tuple.type.712 = tuple_literal (%.loc8_32) [concrete = constants.%tuple.8a2]
// CHECK:STDOUT:     %.loc8_34.3: type = converted constants.%tuple.1b6, constants.%tuple.type.159 [concrete = constants.%tuple.type.159]
// CHECK:STDOUT:     %.loc8_34.4: type = converted %.loc8_34.2, constants.%tuple.type.efa [concrete = constants.%tuple.type.efa]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %x: ref %tuple.type.efa = wrapper_binding x, %x.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %pattern_type.87a = ref_binding_pattern x [concrete = constants.%x.patt]
// CHECK:STDOUT:     %x.var_patt: %pattern_type.87a = var_pattern %x.patt [concrete = constants.%x.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.facet.loc8_3.1: %Destroy.type = facet_value constants.%tuple.type.efa, (constants.%DestroyImpl.impl_witness.a9f, constants.%custom_witness.b80ef6.4) [concrete = constants.%Destroy.facet.210]
// CHECK:STDOUT:   %.loc8_3.2: %Destroy.type = converted constants.%tuple.type.efa, %Destroy.facet.loc8_3.1 [concrete = constants.%Destroy.facet.210]
// CHECK:STDOUT:   %Destroy.facet.loc8_3.2: %Destroy.type = facet_value constants.%tuple.type.efa, (constants.%DestroyImpl.impl_witness.a9f, constants.%custom_witness.b80ef6.4) [concrete = constants.%Destroy.facet.210]
// CHECK:STDOUT:   %.loc8_3.3: %Destroy.type = converted constants.%tuple.type.efa, %Destroy.facet.loc8_3.2 [concrete = constants.%Destroy.facet.210]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%x.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc8_3.1(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc8_3.2(%self.param: ref %ImplsDestroy) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc8_3.3(%self.param: ref %tuple.type.159) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT:   %tuple.elem0: ref %ImplsDestroy = tuple_access %self.param, element0
// CHECK:STDOUT:   %Destroy.facet.loc8_3.1: %Destroy.type = facet_value constants.%ImplsDestroy, (constants.%DestroyImpl.impl_witness.906, constants.%custom_witness.b80ef6.2) [concrete = constants.%Destroy.facet.140]
// CHECK:STDOUT:   %.loc8_3.1: %Destroy.type = converted constants.%ImplsDestroy, %Destroy.facet.loc8_3.1 [concrete = constants.%Destroy.facet.140]
// CHECK:STDOUT:   %Destroy.facet.loc8_3.2: %Destroy.type = facet_value constants.%ImplsDestroy, (constants.%DestroyImpl.impl_witness.906, constants.%custom_witness.b80ef6.2) [concrete = constants.%Destroy.facet.140]
// CHECK:STDOUT:   %.loc8_3.2: %Destroy.type = converted constants.%ImplsDestroy, %Destroy.facet.loc8_3.2 [concrete = constants.%Destroy.facet.140]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%tuple.elem0)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc8_3.4(%self.param: ref %tuple.type.efa) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT:   %tuple.elem0: ref %tuple.type.159 = tuple_access %self.param, element0
// CHECK:STDOUT:   %Destroy.facet.loc8_3.1: %Destroy.type = facet_value constants.%tuple.type.159, (constants.%DestroyImpl.impl_witness.aa8, constants.%custom_witness.b80ef6.3) [concrete = constants.%Destroy.facet.c31]
// CHECK:STDOUT:   %.loc8_3.1: %Destroy.type = converted constants.%tuple.type.159, %Destroy.facet.loc8_3.1 [concrete = constants.%Destroy.facet.c31]
// CHECK:STDOUT:   %Destroy.facet.loc8_3.2: %Destroy.type = facet_value constants.%tuple.type.159, (constants.%DestroyImpl.impl_witness.aa8, constants.%custom_witness.b80ef6.3) [concrete = constants.%Destroy.facet.c31]
// CHECK:STDOUT:   %.loc8_3.2: %Destroy.type = converted constants.%tuple.type.159, %Destroy.facet.loc8_3.2 [concrete = constants.%Destroy.facet.c31]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%tuple.elem0)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
