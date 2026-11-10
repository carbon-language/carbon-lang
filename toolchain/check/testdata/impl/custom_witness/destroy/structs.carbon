// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/destroy.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/impl/custom_witness/destroy/structs.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/impl/custom_witness/destroy/structs.carbon

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
  var unused x: {} = {};
}

// --- trivial_nonempty.carbon

library "[[@TEST_NAME]]";

import library "types";

fn Test() {
  var unused x: {.a: A, .b: B} = {.a = {}, .b = {}};
}

// --- todo_nontrivial_nontrivial_field_first.carbon

library "[[@TEST_NAME]]";

import library "types";


fn Test() {
  //@dump-sem-ir-begin
  var unused x: {.a: ImplsDestroy, .b: HasTrivialDestruction} = {.a = {}, .b = {}};
  //@dump-sem-ir-end
}

// --- todo_nontrivial_nontrivial_field_second.carbon

library "[[@TEST_NAME]]";

import library "types";

fn Test() {
  //@dump-sem-ir-begin
  var unused x: {.a: HasTrivialDestruction, .b: ImplsDestroy} = {.a = {}, .b = {}};
  //@dump-sem-ir-end
}

// --- todo_nontrivial_all_nontrivial_fields.carbon

library "[[@TEST_NAME]]";

import library "types";

class AlsoImplsDestroy {
  impl as Core.DestroyImpl {
    fn Op(unused ref self) {}
  }
}

fn Test() {
  //@dump-sem-ir-begin
  var unused x: {.a: ImplsDestroy, .b: AlsoImplsDestroy, .c: AlsoImplsDestroy, .d: ImplsDestroy} = {.a = {}, .b = {}, .c = {}, .d = {}};
  //@dump-sem-ir-end
}

// --- trivial_nested.carbon

library "[[@TEST_NAME]]";

import library "types";


fn Test() {
  var unused x: {.a: {.b: B}} = {.a = {.b = {}}};
}

// --- todo_nontrivial_nested.carbon

library "[[@TEST_NAME]]";

import library "types";

fn Test() {
  //@dump-sem-ir-begin
  var unused x: {.a: {.b: ImplsDestroy}} = {.a = {.b = {}}};
  //@dump-sem-ir-end
}

// CHECK:STDOUT: --- todo_nontrivial_nontrivial_field_first.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %ImplsDestroy: type = class_type @ImplsDestroy [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %HasTrivialDestruction: type = class_type @HasTrivialDestruction [concrete]
// CHECK:STDOUT:   %struct_type.a.b.02c: type = struct_type {.a: %ImplsDestroy, .b: %HasTrivialDestruction} [concrete]
// CHECK:STDOUT:   %pattern_type.c60: type = pattern_type %struct_type.a.b.02c [concrete]
// CHECK:STDOUT:   %x.patt: %pattern_type.c60 = ref_binding_pattern x [concrete]
// CHECK:STDOUT:   %x.var_patt: %pattern_type.c60 = var_pattern %x.patt [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %struct_type.a.b.f95: type = struct_type {.a: %empty_struct_type, .b: %empty_struct_type} [concrete]
// CHECK:STDOUT:   %struct.767: %struct_type.a.b.f95 = struct_value (%empty_struct, %empty_struct) [concrete]
// CHECK:STDOUT:   %ImplsDestroy.val: %ImplsDestroy = struct_value () [concrete]
// CHECK:STDOUT:   %HasTrivialDestruction.val: %HasTrivialDestruction = struct_value () [concrete]
// CHECK:STDOUT:   %struct.027: %struct_type.a.b.02c = struct_value (%ImplsDestroy.val, %HasTrivialDestruction.val) [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %T.09f: %SubobjectDestroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.type.dea: type = fn_type @T.as_type.as.DestroyImpl.impl.Op, @T.as_type.as.DestroyImpl.impl(%T.09f) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.6e5: %T.as_type.as.DestroyImpl.impl.Op.type.dea = struct_value () [symbolic]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.2: type = fn_type @SubobjectDestroy.WithSelf.Op.loc9_3.2 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.2: %SubobjectDestroy.WithSelf.Op.type.6870fd.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.2: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.2), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.3: type = fn_type @SubobjectDestroy.WithSelf.Op.loc9_3.3 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.3: %SubobjectDestroy.WithSelf.Op.type.6870fd.3 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.3: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.3), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.4: type = fn_type @SubobjectDestroy.WithSelf.Op.loc9_3.4 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.4: %SubobjectDestroy.WithSelf.Op.type.6870fd.4 = struct_value () [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.5bc: %SubobjectDestroy.type = facet_value %HasTrivialDestruction, (%custom_witness.b80ef6.3) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.4ca: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table.a5e, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.5bc) [concrete]
// CHECK:STDOUT:   %Destroy.facet.dba: %Destroy.type = facet_value %HasTrivialDestruction, (%DestroyImpl.impl_witness.4ca, %custom_witness.b80ef6.3) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.906: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table.9a5 [concrete]
// CHECK:STDOUT:   %Destroy.facet.140: %Destroy.type = facet_value %ImplsDestroy, (%DestroyImpl.impl_witness.906, %custom_witness.b80ef6.2) [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.4: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.4), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.d56: %SubobjectDestroy.type = facet_value %struct_type.a.b.02c, (%custom_witness.b80ef6.4) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.65e: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table.a5e, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.d56) [concrete]
// CHECK:STDOUT:   %Destroy.facet.ff6: %Destroy.type = facet_value %struct_type.a.b.02c, (%DestroyImpl.impl_witness.65e, %custom_witness.b80ef6.4) [concrete]
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
// CHECK:STDOUT:   %x.var: ref %struct_type.a.b.02c = var_storage %x.var_patt
// CHECK:STDOUT:   %.loc9_72.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc9_81.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc9_82.1: %struct_type.a.b.f95 = struct_literal (%.loc9_72.1, %.loc9_81.1) [concrete = constants.%struct.767]
// CHECK:STDOUT:   %.loc9_82.2: ref %ImplsDestroy = struct_access %x.var, element0
// CHECK:STDOUT:   %.loc9_72.2: init %ImplsDestroy to %.loc9_82.2 = class_init () [concrete = constants.%ImplsDestroy.val]
// CHECK:STDOUT:   %.loc9_82.3: init %ImplsDestroy = converted %.loc9_72.1, %.loc9_72.2 [concrete = constants.%ImplsDestroy.val]
// CHECK:STDOUT:   %.loc9_82.4: ref %HasTrivialDestruction = struct_access %x.var, element1
// CHECK:STDOUT:   %.loc9_81.2: init %HasTrivialDestruction to %.loc9_82.4 = class_init () [concrete = constants.%HasTrivialDestruction.val]
// CHECK:STDOUT:   %.loc9_82.5: init %HasTrivialDestruction = converted %.loc9_81.1, %.loc9_81.2 [concrete = constants.%HasTrivialDestruction.val]
// CHECK:STDOUT:   %.loc9_82.6: init %struct_type.a.b.02c to %x.var = struct_init (%.loc9_82.3, %.loc9_82.5) [concrete = constants.%struct.027]
// CHECK:STDOUT:   %.loc9_3.1: init %struct_type.a.b.02c = converted %.loc9_82.1, %.loc9_82.6 [concrete = constants.%struct.027]
// CHECK:STDOUT:   assign %x.var, %.loc9_3.1
// CHECK:STDOUT:   %.loc9_61: type = splice_block %struct_type.a.b [concrete = constants.%struct_type.a.b.02c] {
// CHECK:STDOUT:     %ImplsDestroy.ref: type = name_ref ImplsDestroy, imports.%Main.ImplsDestroy [concrete = constants.%ImplsDestroy]
// CHECK:STDOUT:     %HasTrivialDestruction.ref: type = name_ref HasTrivialDestruction, imports.%Main.HasTrivialDestruction [concrete = constants.%HasTrivialDestruction]
// CHECK:STDOUT:     %struct_type.a.b: type = struct_type {.a: %ImplsDestroy, .b: %HasTrivialDestruction} [concrete = constants.%struct_type.a.b.02c]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %x: ref %struct_type.a.b.02c = wrapper_binding x, %x.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %pattern_type.c60 = ref_binding_pattern x [concrete = constants.%x.patt]
// CHECK:STDOUT:     %x.var_patt: %pattern_type.c60 = var_pattern %x.patt [concrete = constants.%x.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.facet.loc9_3.1: %Destroy.type = facet_value constants.%struct_type.a.b.02c, (constants.%DestroyImpl.impl_witness.65e, constants.%custom_witness.b80ef6.4) [concrete = constants.%Destroy.facet.ff6]
// CHECK:STDOUT:   %.loc9_3.2: %Destroy.type = converted constants.%struct_type.a.b.02c, %Destroy.facet.loc9_3.1 [concrete = constants.%Destroy.facet.ff6]
// CHECK:STDOUT:   %Destroy.facet.loc9_3.2: %Destroy.type = facet_value constants.%struct_type.a.b.02c, (constants.%DestroyImpl.impl_witness.65e, constants.%custom_witness.b80ef6.4) [concrete = constants.%Destroy.facet.ff6]
// CHECK:STDOUT:   %.loc9_3.3: %Destroy.type = converted constants.%struct_type.a.b.02c, %Destroy.facet.loc9_3.2 [concrete = constants.%Destroy.facet.ff6]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%x.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc9_3.1(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc9_3.2(%self.param: ref %ImplsDestroy) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc9_3.3(%self.param: ref %HasTrivialDestruction) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc9_3.4(%self.param: ref %struct_type.a.b.02c) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc9_3.1: ref %HasTrivialDestruction = struct_access %self.param, element1
// CHECK:STDOUT:   %Destroy.facet.loc9_3.1: %Destroy.type = facet_value constants.%HasTrivialDestruction, (constants.%DestroyImpl.impl_witness.4ca, constants.%custom_witness.b80ef6.3) [concrete = constants.%Destroy.facet.dba]
// CHECK:STDOUT:   %.loc9_3.2: %Destroy.type = converted constants.%HasTrivialDestruction, %Destroy.facet.loc9_3.1 [concrete = constants.%Destroy.facet.dba]
// CHECK:STDOUT:   %Destroy.facet.loc9_3.2: %Destroy.type = facet_value constants.%HasTrivialDestruction, (constants.%DestroyImpl.impl_witness.4ca, constants.%custom_witness.b80ef6.3) [concrete = constants.%Destroy.facet.dba]
// CHECK:STDOUT:   %.loc9_3.3: %Destroy.type = converted constants.%HasTrivialDestruction, %Destroy.facet.loc9_3.2 [concrete = constants.%Destroy.facet.dba]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc9_3.1: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%.loc9_3.1)
// CHECK:STDOUT:   %.loc9_3.4: ref %ImplsDestroy = struct_access %self.param, element0
// CHECK:STDOUT:   %Destroy.facet.loc9_3.3: %Destroy.type = facet_value constants.%ImplsDestroy, (constants.%DestroyImpl.impl_witness.906, constants.%custom_witness.b80ef6.2) [concrete = constants.%Destroy.facet.140]
// CHECK:STDOUT:   %.loc9_3.5: %Destroy.type = converted constants.%ImplsDestroy, %Destroy.facet.loc9_3.3 [concrete = constants.%Destroy.facet.140]
// CHECK:STDOUT:   %Destroy.facet.loc9_3.4: %Destroy.type = facet_value constants.%ImplsDestroy, (constants.%DestroyImpl.impl_witness.906, constants.%custom_witness.b80ef6.2) [concrete = constants.%Destroy.facet.140]
// CHECK:STDOUT:   %.loc9_3.6: %Destroy.type = converted constants.%ImplsDestroy, %Destroy.facet.loc9_3.4 [concrete = constants.%Destroy.facet.140]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc9_3.2: init %empty_tuple.type = call %SelfDestruct.specific_fn.2(%.loc9_3.4)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- todo_nontrivial_nontrivial_field_second.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %HasTrivialDestruction: type = class_type @HasTrivialDestruction [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %ImplsDestroy: type = class_type @ImplsDestroy [concrete]
// CHECK:STDOUT:   %struct_type.a.b.bb2: type = struct_type {.a: %HasTrivialDestruction, .b: %ImplsDestroy} [concrete]
// CHECK:STDOUT:   %pattern_type.fad: type = pattern_type %struct_type.a.b.bb2 [concrete]
// CHECK:STDOUT:   %x.patt: %pattern_type.fad = ref_binding_pattern x [concrete]
// CHECK:STDOUT:   %x.var_patt: %pattern_type.fad = var_pattern %x.patt [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %struct_type.a.b.f95: type = struct_type {.a: %empty_struct_type, .b: %empty_struct_type} [concrete]
// CHECK:STDOUT:   %struct.767: %struct_type.a.b.f95 = struct_value (%empty_struct, %empty_struct) [concrete]
// CHECK:STDOUT:   %HasTrivialDestruction.val: %HasTrivialDestruction = struct_value () [concrete]
// CHECK:STDOUT:   %ImplsDestroy.val: %ImplsDestroy = struct_value () [concrete]
// CHECK:STDOUT:   %struct.51b: %struct_type.a.b.bb2 = struct_value (%HasTrivialDestruction.val, %ImplsDestroy.val) [concrete]
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
// CHECK:STDOUT:   %DestroyImpl.impl_witness.906: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table.9a5 [concrete]
// CHECK:STDOUT:   %Destroy.facet.140: %Destroy.type = facet_value %ImplsDestroy, (%DestroyImpl.impl_witness.906, %custom_witness.b80ef6.3) [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.5bc: %SubobjectDestroy.type = facet_value %HasTrivialDestruction, (%custom_witness.b80ef6.2) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.4ca: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table.a5e, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.5bc) [concrete]
// CHECK:STDOUT:   %Destroy.facet.dba: %Destroy.type = facet_value %HasTrivialDestruction, (%DestroyImpl.impl_witness.4ca, %custom_witness.b80ef6.2) [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.4: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.4), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.bae: %SubobjectDestroy.type = facet_value %struct_type.a.b.bb2, (%custom_witness.b80ef6.4) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.de5: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table.a5e, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.bae) [concrete]
// CHECK:STDOUT:   %Destroy.facet.3e4: %Destroy.type = facet_value %struct_type.a.b.bb2, (%DestroyImpl.impl_witness.de5, %custom_witness.b80ef6.4) [concrete]
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
// CHECK:STDOUT:   %x.var: ref %struct_type.a.b.bb2 = var_storage %x.var_patt
// CHECK:STDOUT:   %.loc8_72.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc8_81.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc8_82.1: %struct_type.a.b.f95 = struct_literal (%.loc8_72.1, %.loc8_81.1) [concrete = constants.%struct.767]
// CHECK:STDOUT:   %.loc8_82.2: ref %HasTrivialDestruction = struct_access %x.var, element0
// CHECK:STDOUT:   %.loc8_72.2: init %HasTrivialDestruction to %.loc8_82.2 = class_init () [concrete = constants.%HasTrivialDestruction.val]
// CHECK:STDOUT:   %.loc8_82.3: init %HasTrivialDestruction = converted %.loc8_72.1, %.loc8_72.2 [concrete = constants.%HasTrivialDestruction.val]
// CHECK:STDOUT:   %.loc8_82.4: ref %ImplsDestroy = struct_access %x.var, element1
// CHECK:STDOUT:   %.loc8_81.2: init %ImplsDestroy to %.loc8_82.4 = class_init () [concrete = constants.%ImplsDestroy.val]
// CHECK:STDOUT:   %.loc8_82.5: init %ImplsDestroy = converted %.loc8_81.1, %.loc8_81.2 [concrete = constants.%ImplsDestroy.val]
// CHECK:STDOUT:   %.loc8_82.6: init %struct_type.a.b.bb2 to %x.var = struct_init (%.loc8_82.3, %.loc8_82.5) [concrete = constants.%struct.51b]
// CHECK:STDOUT:   %.loc8_3.1: init %struct_type.a.b.bb2 = converted %.loc8_82.1, %.loc8_82.6 [concrete = constants.%struct.51b]
// CHECK:STDOUT:   assign %x.var, %.loc8_3.1
// CHECK:STDOUT:   %.loc8_61: type = splice_block %struct_type.a.b [concrete = constants.%struct_type.a.b.bb2] {
// CHECK:STDOUT:     %HasTrivialDestruction.ref: type = name_ref HasTrivialDestruction, imports.%Main.HasTrivialDestruction [concrete = constants.%HasTrivialDestruction]
// CHECK:STDOUT:     %ImplsDestroy.ref: type = name_ref ImplsDestroy, imports.%Main.ImplsDestroy [concrete = constants.%ImplsDestroy]
// CHECK:STDOUT:     %struct_type.a.b: type = struct_type {.a: %HasTrivialDestruction, .b: %ImplsDestroy} [concrete = constants.%struct_type.a.b.bb2]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %x: ref %struct_type.a.b.bb2 = wrapper_binding x, %x.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %pattern_type.fad = ref_binding_pattern x [concrete = constants.%x.patt]
// CHECK:STDOUT:     %x.var_patt: %pattern_type.fad = var_pattern %x.patt [concrete = constants.%x.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.facet.loc8_3.1: %Destroy.type = facet_value constants.%struct_type.a.b.bb2, (constants.%DestroyImpl.impl_witness.de5, constants.%custom_witness.b80ef6.4) [concrete = constants.%Destroy.facet.3e4]
// CHECK:STDOUT:   %.loc8_3.2: %Destroy.type = converted constants.%struct_type.a.b.bb2, %Destroy.facet.loc8_3.1 [concrete = constants.%Destroy.facet.3e4]
// CHECK:STDOUT:   %Destroy.facet.loc8_3.2: %Destroy.type = facet_value constants.%struct_type.a.b.bb2, (constants.%DestroyImpl.impl_witness.de5, constants.%custom_witness.b80ef6.4) [concrete = constants.%Destroy.facet.3e4]
// CHECK:STDOUT:   %.loc8_3.3: %Destroy.type = converted constants.%struct_type.a.b.bb2, %Destroy.facet.loc8_3.2 [concrete = constants.%Destroy.facet.3e4]
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
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc8_3.4(%self.param: ref %struct_type.a.b.bb2) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc8_3.1: ref %ImplsDestroy = struct_access %self.param, element1
// CHECK:STDOUT:   %Destroy.facet.loc8_3.1: %Destroy.type = facet_value constants.%ImplsDestroy, (constants.%DestroyImpl.impl_witness.906, constants.%custom_witness.b80ef6.3) [concrete = constants.%Destroy.facet.140]
// CHECK:STDOUT:   %.loc8_3.2: %Destroy.type = converted constants.%ImplsDestroy, %Destroy.facet.loc8_3.1 [concrete = constants.%Destroy.facet.140]
// CHECK:STDOUT:   %Destroy.facet.loc8_3.2: %Destroy.type = facet_value constants.%ImplsDestroy, (constants.%DestroyImpl.impl_witness.906, constants.%custom_witness.b80ef6.3) [concrete = constants.%Destroy.facet.140]
// CHECK:STDOUT:   %.loc8_3.3: %Destroy.type = converted constants.%ImplsDestroy, %Destroy.facet.loc8_3.2 [concrete = constants.%Destroy.facet.140]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc8_3.1: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%.loc8_3.1)
// CHECK:STDOUT:   %.loc8_3.4: ref %HasTrivialDestruction = struct_access %self.param, element0
// CHECK:STDOUT:   %Destroy.facet.loc8_3.3: %Destroy.type = facet_value constants.%HasTrivialDestruction, (constants.%DestroyImpl.impl_witness.4ca, constants.%custom_witness.b80ef6.2) [concrete = constants.%Destroy.facet.dba]
// CHECK:STDOUT:   %.loc8_3.5: %Destroy.type = converted constants.%HasTrivialDestruction, %Destroy.facet.loc8_3.3 [concrete = constants.%Destroy.facet.dba]
// CHECK:STDOUT:   %Destroy.facet.loc8_3.4: %Destroy.type = facet_value constants.%HasTrivialDestruction, (constants.%DestroyImpl.impl_witness.4ca, constants.%custom_witness.b80ef6.2) [concrete = constants.%Destroy.facet.dba]
// CHECK:STDOUT:   %.loc8_3.6: %Destroy.type = converted constants.%HasTrivialDestruction, %Destroy.facet.loc8_3.4 [concrete = constants.%Destroy.facet.dba]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc8_3.2: init %empty_tuple.type = call %SelfDestruct.specific_fn.2(%.loc8_3.4)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- todo_nontrivial_all_nontrivial_fields.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %AlsoImplsDestroy: type = class_type @AlsoImplsDestroy [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.e7f: <witness> = impl_witness @AlsoImplsDestroy.as.DestroyImpl.impl.%DestroyImpl.impl_witness_table [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %ImplsDestroy: type = class_type @ImplsDestroy [concrete]
// CHECK:STDOUT:   %struct_type.a.b.c.d.f50: type = struct_type {.a: %ImplsDestroy, .b: %AlsoImplsDestroy, .c: %AlsoImplsDestroy, .d: %ImplsDestroy} [concrete]
// CHECK:STDOUT:   %pattern_type.ae4: type = pattern_type %struct_type.a.b.c.d.f50 [concrete]
// CHECK:STDOUT:   %x.patt: %pattern_type.ae4 = ref_binding_pattern x [concrete]
// CHECK:STDOUT:   %x.var_patt: %pattern_type.ae4 = var_pattern %x.patt [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %struct_type.a.b.c.d.b70: type = struct_type {.a: %empty_struct_type, .b: %empty_struct_type, .c: %empty_struct_type, .d: %empty_struct_type} [concrete]
// CHECK:STDOUT:   %struct.ee2: %struct_type.a.b.c.d.b70 = struct_value (%empty_struct, %empty_struct, %empty_struct, %empty_struct) [concrete]
// CHECK:STDOUT:   %ImplsDestroy.val: %ImplsDestroy = struct_value () [concrete]
// CHECK:STDOUT:   %AlsoImplsDestroy.val: %AlsoImplsDestroy = struct_value () [concrete]
// CHECK:STDOUT:   %struct.378: %struct_type.a.b.c.d.f50 = struct_value (%ImplsDestroy.val, %AlsoImplsDestroy.val, %AlsoImplsDestroy.val, %ImplsDestroy.val) [concrete]
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
// CHECK:STDOUT:   %DestroyImpl.impl_witness.906: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table.9a5 [concrete]
// CHECK:STDOUT:   %Destroy.facet.140: %Destroy.type = facet_value %ImplsDestroy, (%DestroyImpl.impl_witness.906, %custom_witness.b80ef6.2) [concrete]
// CHECK:STDOUT:   %Destroy.facet.122: %Destroy.type = facet_value %AlsoImplsDestroy, (%DestroyImpl.impl_witness.e7f, %custom_witness.b80ef6.3) [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.4: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.4), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.5bb: %SubobjectDestroy.type = facet_value %struct_type.a.b.c.d.f50, (%custom_witness.b80ef6.4) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.e15: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table.a5e, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.5bb) [concrete]
// CHECK:STDOUT:   %Destroy.facet.ecb: %Destroy.type = facet_value %struct_type.a.b.c.d.f50, (%DestroyImpl.impl_witness.e15, %custom_witness.b80ef6.4) [concrete]
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
// CHECK:STDOUT:   %x.var: ref %struct_type.a.b.c.d.f50 = var_storage %x.var_patt
// CHECK:STDOUT:   %.loc14_107.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc14_116.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc14_125.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc14_134.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc14_135.1: %struct_type.a.b.c.d.b70 = struct_literal (%.loc14_107.1, %.loc14_116.1, %.loc14_125.1, %.loc14_134.1) [concrete = constants.%struct.ee2]
// CHECK:STDOUT:   %.loc14_135.2: ref %ImplsDestroy = struct_access %x.var, element0
// CHECK:STDOUT:   %.loc14_107.2: init %ImplsDestroy to %.loc14_135.2 = class_init () [concrete = constants.%ImplsDestroy.val]
// CHECK:STDOUT:   %.loc14_135.3: init %ImplsDestroy = converted %.loc14_107.1, %.loc14_107.2 [concrete = constants.%ImplsDestroy.val]
// CHECK:STDOUT:   %.loc14_135.4: ref %AlsoImplsDestroy = struct_access %x.var, element1
// CHECK:STDOUT:   %.loc14_116.2: init %AlsoImplsDestroy to %.loc14_135.4 = class_init () [concrete = constants.%AlsoImplsDestroy.val]
// CHECK:STDOUT:   %.loc14_135.5: init %AlsoImplsDestroy = converted %.loc14_116.1, %.loc14_116.2 [concrete = constants.%AlsoImplsDestroy.val]
// CHECK:STDOUT:   %.loc14_135.6: ref %AlsoImplsDestroy = struct_access %x.var, element2
// CHECK:STDOUT:   %.loc14_125.2: init %AlsoImplsDestroy to %.loc14_135.6 = class_init () [concrete = constants.%AlsoImplsDestroy.val]
// CHECK:STDOUT:   %.loc14_135.7: init %AlsoImplsDestroy = converted %.loc14_125.1, %.loc14_125.2 [concrete = constants.%AlsoImplsDestroy.val]
// CHECK:STDOUT:   %.loc14_135.8: ref %ImplsDestroy = struct_access %x.var, element3
// CHECK:STDOUT:   %.loc14_134.2: init %ImplsDestroy to %.loc14_135.8 = class_init () [concrete = constants.%ImplsDestroy.val]
// CHECK:STDOUT:   %.loc14_135.9: init %ImplsDestroy = converted %.loc14_134.1, %.loc14_134.2 [concrete = constants.%ImplsDestroy.val]
// CHECK:STDOUT:   %.loc14_135.10: init %struct_type.a.b.c.d.f50 to %x.var = struct_init (%.loc14_135.3, %.loc14_135.5, %.loc14_135.7, %.loc14_135.9) [concrete = constants.%struct.378]
// CHECK:STDOUT:   %.loc14_3.1: init %struct_type.a.b.c.d.f50 = converted %.loc14_135.1, %.loc14_135.10 [concrete = constants.%struct.378]
// CHECK:STDOUT:   assign %x.var, %.loc14_3.1
// CHECK:STDOUT:   %.loc14_96: type = splice_block %struct_type.a.b.c.d [concrete = constants.%struct_type.a.b.c.d.f50] {
// CHECK:STDOUT:     %ImplsDestroy.ref.loc14_22: type = name_ref ImplsDestroy, imports.%Main.ImplsDestroy [concrete = constants.%ImplsDestroy]
// CHECK:STDOUT:     %AlsoImplsDestroy.ref.loc14_40: type = name_ref AlsoImplsDestroy, file.%AlsoImplsDestroy.decl [concrete = constants.%AlsoImplsDestroy]
// CHECK:STDOUT:     %AlsoImplsDestroy.ref.loc14_62: type = name_ref AlsoImplsDestroy, file.%AlsoImplsDestroy.decl [concrete = constants.%AlsoImplsDestroy]
// CHECK:STDOUT:     %ImplsDestroy.ref.loc14_84: type = name_ref ImplsDestroy, imports.%Main.ImplsDestroy [concrete = constants.%ImplsDestroy]
// CHECK:STDOUT:     %struct_type.a.b.c.d: type = struct_type {.a: %ImplsDestroy, .b: %AlsoImplsDestroy, .c: %AlsoImplsDestroy, .d: %ImplsDestroy} [concrete = constants.%struct_type.a.b.c.d.f50]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %x: ref %struct_type.a.b.c.d.f50 = wrapper_binding x, %x.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %pattern_type.ae4 = ref_binding_pattern x [concrete = constants.%x.patt]
// CHECK:STDOUT:     %x.var_patt: %pattern_type.ae4 = var_pattern %x.patt [concrete = constants.%x.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.facet.loc14_3.1: %Destroy.type = facet_value constants.%struct_type.a.b.c.d.f50, (constants.%DestroyImpl.impl_witness.e15, constants.%custom_witness.b80ef6.4) [concrete = constants.%Destroy.facet.ecb]
// CHECK:STDOUT:   %.loc14_3.2: %Destroy.type = converted constants.%struct_type.a.b.c.d.f50, %Destroy.facet.loc14_3.1 [concrete = constants.%Destroy.facet.ecb]
// CHECK:STDOUT:   %Destroy.facet.loc14_3.2: %Destroy.type = facet_value constants.%struct_type.a.b.c.d.f50, (constants.%DestroyImpl.impl_witness.e15, constants.%custom_witness.b80ef6.4) [concrete = constants.%Destroy.facet.ecb]
// CHECK:STDOUT:   %.loc14_3.3: %Destroy.type = converted constants.%struct_type.a.b.c.d.f50, %Destroy.facet.loc14_3.2 [concrete = constants.%Destroy.facet.ecb]
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
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc14_3.4(%self.param: ref %struct_type.a.b.c.d.f50) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc14_3.1: ref %ImplsDestroy = struct_access %self.param, element3
// CHECK:STDOUT:   %Destroy.facet.loc14_3.1: %Destroy.type = facet_value constants.%ImplsDestroy, (constants.%DestroyImpl.impl_witness.906, constants.%custom_witness.b80ef6.2) [concrete = constants.%Destroy.facet.140]
// CHECK:STDOUT:   %.loc14_3.2: %Destroy.type = converted constants.%ImplsDestroy, %Destroy.facet.loc14_3.1 [concrete = constants.%Destroy.facet.140]
// CHECK:STDOUT:   %Destroy.facet.loc14_3.2: %Destroy.type = facet_value constants.%ImplsDestroy, (constants.%DestroyImpl.impl_witness.906, constants.%custom_witness.b80ef6.2) [concrete = constants.%Destroy.facet.140]
// CHECK:STDOUT:   %.loc14_3.3: %Destroy.type = converted constants.%ImplsDestroy, %Destroy.facet.loc14_3.2 [concrete = constants.%Destroy.facet.140]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc14_3.1: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%.loc14_3.1)
// CHECK:STDOUT:   %.loc14_3.4: ref %AlsoImplsDestroy = struct_access %self.param, element2
// CHECK:STDOUT:   %Destroy.facet.loc14_3.3: %Destroy.type = facet_value constants.%AlsoImplsDestroy, (constants.%DestroyImpl.impl_witness.e7f, constants.%custom_witness.b80ef6.3) [concrete = constants.%Destroy.facet.122]
// CHECK:STDOUT:   %.loc14_3.5: %Destroy.type = converted constants.%AlsoImplsDestroy, %Destroy.facet.loc14_3.3 [concrete = constants.%Destroy.facet.122]
// CHECK:STDOUT:   %Destroy.facet.loc14_3.4: %Destroy.type = facet_value constants.%AlsoImplsDestroy, (constants.%DestroyImpl.impl_witness.e7f, constants.%custom_witness.b80ef6.3) [concrete = constants.%Destroy.facet.122]
// CHECK:STDOUT:   %.loc14_3.6: %Destroy.type = converted constants.%AlsoImplsDestroy, %Destroy.facet.loc14_3.4 [concrete = constants.%Destroy.facet.122]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc14_3.2: init %empty_tuple.type = call %SelfDestruct.specific_fn.2(%.loc14_3.4)
// CHECK:STDOUT:   %.loc14_3.7: ref %AlsoImplsDestroy = struct_access %self.param, element1
// CHECK:STDOUT:   %Destroy.facet.loc14_3.5: %Destroy.type = facet_value constants.%AlsoImplsDestroy, (constants.%DestroyImpl.impl_witness.e7f, constants.%custom_witness.b80ef6.3) [concrete = constants.%Destroy.facet.122]
// CHECK:STDOUT:   %.loc14_3.8: %Destroy.type = converted constants.%AlsoImplsDestroy, %Destroy.facet.loc14_3.5 [concrete = constants.%Destroy.facet.122]
// CHECK:STDOUT:   %Destroy.facet.loc14_3.6: %Destroy.type = facet_value constants.%AlsoImplsDestroy, (constants.%DestroyImpl.impl_witness.e7f, constants.%custom_witness.b80ef6.3) [concrete = constants.%Destroy.facet.122]
// CHECK:STDOUT:   %.loc14_3.9: %Destroy.type = converted constants.%AlsoImplsDestroy, %Destroy.facet.loc14_3.6 [concrete = constants.%Destroy.facet.122]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc14_3.3: init %empty_tuple.type = call %SelfDestruct.specific_fn.3(%.loc14_3.7)
// CHECK:STDOUT:   %.loc14_3.10: ref %ImplsDestroy = struct_access %self.param, element0
// CHECK:STDOUT:   %Destroy.facet.loc14_3.7: %Destroy.type = facet_value constants.%ImplsDestroy, (constants.%DestroyImpl.impl_witness.906, constants.%custom_witness.b80ef6.2) [concrete = constants.%Destroy.facet.140]
// CHECK:STDOUT:   %.loc14_3.11: %Destroy.type = converted constants.%ImplsDestroy, %Destroy.facet.loc14_3.7 [concrete = constants.%Destroy.facet.140]
// CHECK:STDOUT:   %Destroy.facet.loc14_3.8: %Destroy.type = facet_value constants.%ImplsDestroy, (constants.%DestroyImpl.impl_witness.906, constants.%custom_witness.b80ef6.2) [concrete = constants.%Destroy.facet.140]
// CHECK:STDOUT:   %.loc14_3.12: %Destroy.type = converted constants.%ImplsDestroy, %Destroy.facet.loc14_3.8 [concrete = constants.%Destroy.facet.140]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc14_3.4: init %empty_tuple.type = call %SelfDestruct.specific_fn.4(%.loc14_3.10)
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
// CHECK:STDOUT:   %struct_type.b.0cd: type = struct_type {.b: %ImplsDestroy} [concrete]
// CHECK:STDOUT:   %struct_type.a.13f: type = struct_type {.a: %struct_type.b.0cd} [concrete]
// CHECK:STDOUT:   %pattern_type.378: type = pattern_type %struct_type.a.13f [concrete]
// CHECK:STDOUT:   %x.patt: %pattern_type.378 = ref_binding_pattern x [concrete]
// CHECK:STDOUT:   %x.var_patt: %pattern_type.378 = var_pattern %x.patt [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %struct_type.b.347: type = struct_type {.b: %empty_struct_type} [concrete]
// CHECK:STDOUT:   %struct.692: %struct_type.b.347 = struct_value (%empty_struct) [concrete]
// CHECK:STDOUT:   %struct_type.a.287: type = struct_type {.a: %struct_type.b.347} [concrete]
// CHECK:STDOUT:   %struct.8a7: %struct_type.a.287 = struct_value (%struct.692) [concrete]
// CHECK:STDOUT:   %ImplsDestroy.val: %ImplsDestroy = struct_value () [concrete]
// CHECK:STDOUT:   %struct.2ea: %struct_type.b.0cd = struct_value (%ImplsDestroy.val) [concrete]
// CHECK:STDOUT:   %struct.4ac: %struct_type.a.13f = struct_value (%struct.2ea) [concrete]
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
// CHECK:STDOUT:   %DestroyImpl.impl_witness.906: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table.9a5 [concrete]
// CHECK:STDOUT:   %Destroy.facet.140: %Destroy.type = facet_value %ImplsDestroy, (%DestroyImpl.impl_witness.906, %custom_witness.b80ef6.2) [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.3: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.3), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.4: type = fn_type @SubobjectDestroy.WithSelf.Op.loc8_3.4 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.4: %SubobjectDestroy.WithSelf.Op.type.6870fd.4 = struct_value () [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.135: %SubobjectDestroy.type = facet_value %struct_type.b.0cd, (%custom_witness.b80ef6.3) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.436: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table.a5e, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.135) [concrete]
// CHECK:STDOUT:   %Destroy.facet.2df: %Destroy.type = facet_value %struct_type.b.0cd, (%DestroyImpl.impl_witness.436, %custom_witness.b80ef6.3) [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.4: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.4), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.985: %SubobjectDestroy.type = facet_value %struct_type.a.13f, (%custom_witness.b80ef6.4) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.973: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table.a5e, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.985) [concrete]
// CHECK:STDOUT:   %Destroy.facet.b26: %Destroy.type = facet_value %struct_type.a.13f, (%DestroyImpl.impl_witness.973, %custom_witness.b80ef6.4) [concrete]
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
// CHECK:STDOUT:   %x.var: ref %struct_type.a.13f = var_storage %x.var_patt
// CHECK:STDOUT:   %.loc8_57.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc8_58.1: %struct_type.b.347 = struct_literal (%.loc8_57.1) [concrete = constants.%struct.692]
// CHECK:STDOUT:   %.loc8_59.1: %struct_type.a.287 = struct_literal (%.loc8_58.1) [concrete = constants.%struct.8a7]
// CHECK:STDOUT:   %.loc8_59.2: ref %struct_type.b.0cd = struct_access %x.var, element0
// CHECK:STDOUT:   %.loc8_58.2: ref %ImplsDestroy = struct_access %.loc8_59.2, element0
// CHECK:STDOUT:   %.loc8_57.2: init %ImplsDestroy to %.loc8_58.2 = class_init () [concrete = constants.%ImplsDestroy.val]
// CHECK:STDOUT:   %.loc8_58.3: init %ImplsDestroy = converted %.loc8_57.1, %.loc8_57.2 [concrete = constants.%ImplsDestroy.val]
// CHECK:STDOUT:   %.loc8_58.4: init %struct_type.b.0cd to %.loc8_59.2 = struct_init (%.loc8_58.3) [concrete = constants.%struct.2ea]
// CHECK:STDOUT:   %.loc8_59.3: init %struct_type.b.0cd = converted %.loc8_58.1, %.loc8_58.4 [concrete = constants.%struct.2ea]
// CHECK:STDOUT:   %.loc8_59.4: init %struct_type.a.13f to %x.var = struct_init (%.loc8_59.3) [concrete = constants.%struct.4ac]
// CHECK:STDOUT:   %.loc8_3.1: init %struct_type.a.13f = converted %.loc8_59.1, %.loc8_59.4 [concrete = constants.%struct.4ac]
// CHECK:STDOUT:   assign %x.var, %.loc8_3.1
// CHECK:STDOUT:   %.loc8_40: type = splice_block %struct_type.a [concrete = constants.%struct_type.a.13f] {
// CHECK:STDOUT:     %ImplsDestroy.ref: type = name_ref ImplsDestroy, imports.%Main.ImplsDestroy [concrete = constants.%ImplsDestroy]
// CHECK:STDOUT:     %struct_type.b: type = struct_type {.b: %ImplsDestroy} [concrete = constants.%struct_type.b.0cd]
// CHECK:STDOUT:     %struct_type.a: type = struct_type {.a: %struct_type.b.0cd} [concrete = constants.%struct_type.a.13f]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %x: ref %struct_type.a.13f = wrapper_binding x, %x.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %pattern_type.378 = ref_binding_pattern x [concrete = constants.%x.patt]
// CHECK:STDOUT:     %x.var_patt: %pattern_type.378 = var_pattern %x.patt [concrete = constants.%x.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.facet.loc8_3.1: %Destroy.type = facet_value constants.%struct_type.a.13f, (constants.%DestroyImpl.impl_witness.973, constants.%custom_witness.b80ef6.4) [concrete = constants.%Destroy.facet.b26]
// CHECK:STDOUT:   %.loc8_3.2: %Destroy.type = converted constants.%struct_type.a.13f, %Destroy.facet.loc8_3.1 [concrete = constants.%Destroy.facet.b26]
// CHECK:STDOUT:   %Destroy.facet.loc8_3.2: %Destroy.type = facet_value constants.%struct_type.a.13f, (constants.%DestroyImpl.impl_witness.973, constants.%custom_witness.b80ef6.4) [concrete = constants.%Destroy.facet.b26]
// CHECK:STDOUT:   %.loc8_3.3: %Destroy.type = converted constants.%struct_type.a.13f, %Destroy.facet.loc8_3.2 [concrete = constants.%Destroy.facet.b26]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%x.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc8_3.1(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc8_3.2(%self.param: ref %ImplsDestroy) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc8_3.3(%self.param: ref %struct_type.b.0cd) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc8_3.1: ref %ImplsDestroy = struct_access %self.param, element0
// CHECK:STDOUT:   %Destroy.facet.loc8_3.1: %Destroy.type = facet_value constants.%ImplsDestroy, (constants.%DestroyImpl.impl_witness.906, constants.%custom_witness.b80ef6.2) [concrete = constants.%Destroy.facet.140]
// CHECK:STDOUT:   %.loc8_3.2: %Destroy.type = converted constants.%ImplsDestroy, %Destroy.facet.loc8_3.1 [concrete = constants.%Destroy.facet.140]
// CHECK:STDOUT:   %Destroy.facet.loc8_3.2: %Destroy.type = facet_value constants.%ImplsDestroy, (constants.%DestroyImpl.impl_witness.906, constants.%custom_witness.b80ef6.2) [concrete = constants.%Destroy.facet.140]
// CHECK:STDOUT:   %.loc8_3.3: %Destroy.type = converted constants.%ImplsDestroy, %Destroy.facet.loc8_3.2 [concrete = constants.%Destroy.facet.140]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%.loc8_3.1)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc8_3.4(%self.param: ref %struct_type.a.13f) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc8_3.1: ref %struct_type.b.0cd = struct_access %self.param, element0
// CHECK:STDOUT:   %Destroy.facet.loc8_3.1: %Destroy.type = facet_value constants.%struct_type.b.0cd, (constants.%DestroyImpl.impl_witness.436, constants.%custom_witness.b80ef6.3) [concrete = constants.%Destroy.facet.2df]
// CHECK:STDOUT:   %.loc8_3.2: %Destroy.type = converted constants.%struct_type.b.0cd, %Destroy.facet.loc8_3.1 [concrete = constants.%Destroy.facet.2df]
// CHECK:STDOUT:   %Destroy.facet.loc8_3.2: %Destroy.type = facet_value constants.%struct_type.b.0cd, (constants.%DestroyImpl.impl_witness.436, constants.%custom_witness.b80ef6.3) [concrete = constants.%Destroy.facet.2df]
// CHECK:STDOUT:   %.loc8_3.3: %Destroy.type = converted constants.%struct_type.b.0cd, %Destroy.facet.loc8_3.2 [concrete = constants.%Destroy.facet.2df]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%.loc8_3.1)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
