// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/destroy.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/impl/custom_witness/destruction/trivial/structs.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/impl/custom_witness/destruction/trivial/structs.carbon

// --- types.carbon
library "[[@TEST_NAME]]";

class A {}
class B {}

// Used when mixing with `HasNonTrivialDestruction`.
class HasTrivialDestruction {}

class HasNonTrivialDestruction {
  impl as Core.DestroyImpl {
    fn Op(unused self) {}
  }
}

// --- todo_trivial_empty.carbon
library "[[@TEST_NAME]]";

alias Struct = {};
fn Make() -> Struct;

fn Test() {
  //@dump-sem-ir-begin
  Struct as Core.SubobjectDestroy;
  Struct as Core.Destroy;
  Struct as Core.TrivialDestroy;

  var unused x: Struct = Make();
  //@dump-sem-ir-end
}

// --- todo_trivial_nonempty.carbon
library "[[@TEST_NAME]]";

import library "types";

alias Struct = {.a: A, .b: B};
fn Make() -> Struct;

fn Test() {
  //@dump-sem-ir-begin
  Struct as Core.SubobjectDestroy;
  Struct as Core.Destroy;
  Struct as Core.TrivialDestroy;

  var unused x: Struct = Make();
  //@dump-sem-ir-end
}

// --- todo_trivial_nested_struct_with_trivial_members.carbon
library "[[@TEST_NAME]]";

import library "types";

alias Struct = {.a: {.b: B}};
fn Make() -> Struct;

fn Test() {
  //@dump-sem-ir-begin
  Struct as Core.SubobjectDestroy;
  Struct as Core.Destroy;
  Struct as Core.TrivialDestroy;

  var unused x: Struct = Make();
  //@dump-sem-ir-end
}

// --- todo_fail_nontrivial_struct_nontrivial_field_first.carbon
library "[[@TEST_NAME]]";

import library "types";

alias Struct = {.a: HasNonTrivialDestruction, .b: HasTrivialDestruction};

fn Test() {
  Struct as Core.TrivialDestroy;
}

// --- todo_fail_nontrivial_struct_nontrivial_field_second.carbon
library "[[@TEST_NAME]]";

import library "types";

alias Struct = {.a: HasTrivialDestruction, .b: HasNonTrivialDestruction};

fn Test() {
  Struct as Core.TrivialDestroy;
}

// --- todo_fail_nontrivial_struct_nontrivial_field_nested.carbon
library "[[@TEST_NAME]]";

import library "types";

alias Struct = {.a: {.b: HasNonTrivialDestruction}};

fn Test() {
  Struct as Core.TrivialDestroy;
}

// CHECK:STDOUT: --- todo_trivial_empty.carbon = $todo_trivial_empty
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %pattern_type.a96: type = pattern_type %empty_struct_type [concrete]
// CHECK:STDOUT:   %Make.type: type = fn_type @Make [concrete]
// CHECK:STDOUT:   %Make: %Make.type = struct_value () [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.687: type = fn_type @SubobjectDestroy.WithSelf.Op.loc8 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f: %SubobjectDestroy.WithSelf.Op.type.687 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.6b7: %SubobjectDestroy.type = facet_value %empty_struct_type, (%custom_witness.b80) [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %T.09f: %SubobjectDestroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.type.dea: type = fn_type @T.as_type.as.DestroyImpl.impl.Op, @T.as_type.as.DestroyImpl.impl(%T.09f) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.6e5: %T.as_type.as.DestroyImpl.impl.Op.type.dea = struct_value () [symbolic]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.011: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.6b7) [concrete]
// CHECK:STDOUT:   %Destroy.facet: %Destroy.type = facet_value %empty_struct_type, (%custom_witness.b80, %DestroyImpl.impl_witness.011) [concrete]
// CHECK:STDOUT:   %x.patt: %pattern_type.a96 = ref_binding_pattern x [concrete]
// CHECK:STDOUT:   %x.var_patt: %pattern_type.a96 = var_pattern %x.patt [concrete]
// CHECK:STDOUT:   %SelfDestruct.type: type = fn_type @SelfDestruct [concrete]
// CHECK:STDOUT:   %SelfDestruct: %SelfDestruct.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .SubobjectDestroy = %Core.SubobjectDestroy
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     .TrivialDestroy = %Core.TrivialDestroy
// CHECK:STDOUT:     .SelfDestruct = %Core.SelfDestruct
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.SubobjectDestroy: type = import_ref Core//prelude/parts/destroy, SubobjectDestroy, loaded [concrete = constants.%SubobjectDestroy.type]
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Core.import_ref.74d: @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op.type (%T.as_type.as.DestroyImpl.impl.Op.type.dea) = import_ref Core//prelude/parts/destroy, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op (constants.%T.as_type.as.DestroyImpl.impl.Op.6e5)]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table = impl_witness_table (%Core.import_ref.74d), @T.as_type.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT:   %Core.TrivialDestroy: type = import_ref Core//prelude/parts/destroy, TrivialDestroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Core.SelfDestruct: %SelfDestruct.type = import_ref Core//prelude/parts/destroy, SelfDestruct, loaded [concrete = constants.%SelfDestruct]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Test() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Struct.ref.loc8: %empty_struct_type = name_ref Struct, file.%Struct [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %Core.ref.loc8: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:   %SubobjectDestroy.ref: type = name_ref SubobjectDestroy, imports.%Core.SubobjectDestroy [concrete = constants.%SubobjectDestroy.type]
// CHECK:STDOUT:   %SubobjectDestroy.facet: %SubobjectDestroy.type = facet_value constants.%empty_struct_type, (constants.%custom_witness.b80) [concrete = constants.%SubobjectDestroy.facet.6b7]
// CHECK:STDOUT:   %.loc8: %SubobjectDestroy.type = converted %Struct.ref.loc8, %SubobjectDestroy.facet [concrete = constants.%SubobjectDestroy.facet.6b7]
// CHECK:STDOUT:   %Struct.ref.loc9: %empty_struct_type = name_ref Struct, file.%Struct [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %Core.ref.loc9: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:   %Destroy.ref: type = name_ref Destroy, imports.%Core.Destroy [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Destroy.facet.loc9: %Destroy.type = facet_value constants.%empty_struct_type, (constants.%custom_witness.b80, constants.%DestroyImpl.impl_witness.011) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc9: %Destroy.type = converted %Struct.ref.loc9, %Destroy.facet.loc9 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %Struct.ref.loc10: %empty_struct_type = name_ref Struct, file.%Struct [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %Core.ref.loc10: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:   %TrivialDestroy.ref: type = name_ref TrivialDestroy, imports.%Core.TrivialDestroy [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Destroy.facet.loc10: %Destroy.type = facet_value constants.%empty_struct_type, (constants.%custom_witness.b80, constants.%DestroyImpl.impl_witness.011) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc10: %Destroy.type = converted %Struct.ref.loc10, %Destroy.facet.loc10 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %x.var: ref %empty_struct_type = var_storage %x.var_patt
// CHECK:STDOUT:   %Make.ref: %Make.type = name_ref Make, file.%Make.decl [concrete = constants.%Make]
// CHECK:STDOUT:   %Make.call: init %empty_struct_type = call %Make.ref()
// CHECK:STDOUT:   assign %x.var, %Make.call
// CHECK:STDOUT:   %.loc12_17.1: type = splice_block %.loc12_17.2 [concrete = constants.%empty_struct_type] {
// CHECK:STDOUT:     %Struct.ref.loc12: %empty_struct_type = name_ref Struct, file.%Struct [concrete = constants.%empty_struct]
// CHECK:STDOUT:     %.loc12_17.2: type = converted %Struct.ref.loc12, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %x: ref %empty_struct_type = wrapper_binding x, %x.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %pattern_type.a96 = ref_binding_pattern x [concrete = constants.%x.patt]
// CHECK:STDOUT:     %x.var_patt: %pattern_type.a96 = var_pattern %x.patt [concrete = constants.%x.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.facet.loc12_3.1: %Destroy.type = facet_value constants.%empty_struct_type, (constants.%custom_witness.b80, constants.%DestroyImpl.impl_witness.011) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc12_3.1: %Destroy.type = converted constants.%empty_struct_type, %Destroy.facet.loc12_3.1 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %Destroy.facet.loc12_3.2: %Destroy.type = facet_value constants.%empty_struct_type, (constants.%custom_witness.b80, constants.%DestroyImpl.impl_witness.011) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc12_3.2: %Destroy.type = converted constants.%empty_struct_type, %Destroy.facet.loc12_3.2 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%x.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc8(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: --- todo_trivial_nonempty.carbon = $todo_trivial_nonempty
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %A: type = class_type @A [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %B: type = class_type @B [concrete]
// CHECK:STDOUT:   %struct_type.a.b.06d: type = struct_type {.a: %A, .b: %B} [concrete]
// CHECK:STDOUT:   %pattern_type.657: type = pattern_type %struct_type.a.b.06d [concrete]
// CHECK:STDOUT:   %Make.type: type = fn_type @Make [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Make: %Make.type = struct_value () [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.2: type = fn_type @SubobjectDestroy.WithSelf.Op.loc10_10.2 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.2: %SubobjectDestroy.WithSelf.Op.type.6870fd.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.2: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.2), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.3: type = fn_type @SubobjectDestroy.WithSelf.Op.loc10_10.3 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.3: %SubobjectDestroy.WithSelf.Op.type.6870fd.3 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.3: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.3), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.4: type = fn_type @SubobjectDestroy.WithSelf.Op.loc10_10.4 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.4: %SubobjectDestroy.WithSelf.Op.type.6870fd.4 = struct_value () [concrete]
// CHECK:STDOUT:   %SelfDestruct.type: type = fn_type @SelfDestruct [concrete]
// CHECK:STDOUT:   %SelfDestruct: %SelfDestruct.type = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %T.09f: %SubobjectDestroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.type.dea: type = fn_type @T.as_type.as.DestroyImpl.impl.Op, @T.as_type.as.DestroyImpl.impl(%T.09f) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.6e5: %T.as_type.as.DestroyImpl.impl.Op.type.dea = struct_value () [symbolic]
// CHECK:STDOUT:   %SubobjectDestroy.facet.92e: %SubobjectDestroy.type = facet_value %B, (%custom_witness.b80ef6.3) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.131: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.92e) [concrete]
// CHECK:STDOUT:   %Destroy.facet.dc1: %Destroy.type = facet_value %B, (%custom_witness.b80ef6.3, %DestroyImpl.impl_witness.131) [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.9bf: %SubobjectDestroy.type = facet_value %A, (%custom_witness.b80ef6.2) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.6d1: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.9bf) [concrete]
// CHECK:STDOUT:   %Destroy.facet.3d0: %Destroy.type = facet_value %A, (%custom_witness.b80ef6.2, %DestroyImpl.impl_witness.6d1) [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.4: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.4), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.364: %SubobjectDestroy.type = facet_value %struct_type.a.b.06d, (%custom_witness.b80ef6.4) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.137: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.364) [concrete]
// CHECK:STDOUT:   %Destroy.facet.a88: %Destroy.type = facet_value %struct_type.a.b.06d, (%custom_witness.b80ef6.4, %DestroyImpl.impl_witness.137) [concrete]
// CHECK:STDOUT:   %x.patt: %pattern_type.657 = ref_binding_pattern x [concrete]
// CHECK:STDOUT:   %x.var_patt: %pattern_type.657 = var_pattern %x.patt [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .SubobjectDestroy = %Core.SubobjectDestroy
// CHECK:STDOUT:     .SelfDestruct = %Core.SelfDestruct
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     .TrivialDestroy = %Core.TrivialDestroy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.SubobjectDestroy: type = import_ref Core//prelude/parts/destroy, SubobjectDestroy, loaded [concrete = constants.%SubobjectDestroy.type]
// CHECK:STDOUT:   %Core.SelfDestruct: %SelfDestruct.type = import_ref Core//prelude/parts/destroy, SelfDestruct, loaded [concrete = constants.%SelfDestruct]
// CHECK:STDOUT:   %Core.import_ref.74d: @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op.type (%T.as_type.as.DestroyImpl.impl.Op.type.dea) = import_ref Core//prelude/parts/destroy, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op (constants.%T.as_type.as.DestroyImpl.impl.Op.6e5)]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table = impl_witness_table (%Core.import_ref.74d), @T.as_type.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Core.TrivialDestroy: type = import_ref Core//prelude/parts/destroy, TrivialDestroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Test() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Struct.ref.loc10: type = name_ref Struct, file.%Struct [concrete = constants.%struct_type.a.b.06d]
// CHECK:STDOUT:   %Core.ref.loc10: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:   %SubobjectDestroy.ref: type = name_ref SubobjectDestroy, imports.%Core.SubobjectDestroy [concrete = constants.%SubobjectDestroy.type]
// CHECK:STDOUT:   %SubobjectDestroy.facet: %SubobjectDestroy.type = facet_value %Struct.ref.loc10, (constants.%custom_witness.b80ef6.4) [concrete = constants.%SubobjectDestroy.facet.364]
// CHECK:STDOUT:   %.loc10: %SubobjectDestroy.type = converted %Struct.ref.loc10, %SubobjectDestroy.facet [concrete = constants.%SubobjectDestroy.facet.364]
// CHECK:STDOUT:   %Struct.ref.loc11: type = name_ref Struct, file.%Struct [concrete = constants.%struct_type.a.b.06d]
// CHECK:STDOUT:   %Core.ref.loc11: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:   %Destroy.ref: type = name_ref Destroy, imports.%Core.Destroy [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Destroy.facet.loc11: %Destroy.type = facet_value %Struct.ref.loc11, (constants.%custom_witness.b80ef6.4, constants.%DestroyImpl.impl_witness.137) [concrete = constants.%Destroy.facet.a88]
// CHECK:STDOUT:   %.loc11: %Destroy.type = converted %Struct.ref.loc11, %Destroy.facet.loc11 [concrete = constants.%Destroy.facet.a88]
// CHECK:STDOUT:   %Struct.ref.loc12: type = name_ref Struct, file.%Struct [concrete = constants.%struct_type.a.b.06d]
// CHECK:STDOUT:   %Core.ref.loc12: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:   %TrivialDestroy.ref: type = name_ref TrivialDestroy, imports.%Core.TrivialDestroy [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Destroy.facet.loc12: %Destroy.type = facet_value %Struct.ref.loc12, (constants.%custom_witness.b80ef6.4, constants.%DestroyImpl.impl_witness.137) [concrete = constants.%Destroy.facet.a88]
// CHECK:STDOUT:   %.loc12: %Destroy.type = converted %Struct.ref.loc12, %Destroy.facet.loc12 [concrete = constants.%Destroy.facet.a88]
// CHECK:STDOUT:   %x.var: ref %struct_type.a.b.06d = var_storage %x.var_patt
// CHECK:STDOUT:   %Make.ref: %Make.type = name_ref Make, file.%Make.decl [concrete = constants.%Make]
// CHECK:STDOUT:   %.loc14_3.1: ref %struct_type.a.b.06d = splice_block %x.var {}
// CHECK:STDOUT:   %Make.call: init %struct_type.a.b.06d to %.loc14_3.1 = call %Make.ref()
// CHECK:STDOUT:   assign %x.var, %Make.call
// CHECK:STDOUT:   %Struct.ref.loc14: type = name_ref Struct, file.%Struct [concrete = constants.%struct_type.a.b.06d]
// CHECK:STDOUT:   %x: ref %struct_type.a.b.06d = wrapper_binding x, %x.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %pattern_type.657 = ref_binding_pattern x [concrete = constants.%x.patt]
// CHECK:STDOUT:     %x.var_patt: %pattern_type.657 = var_pattern %x.patt [concrete = constants.%x.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.facet.loc14_3.1: %Destroy.type = facet_value constants.%struct_type.a.b.06d, (constants.%custom_witness.b80ef6.4, constants.%DestroyImpl.impl_witness.137) [concrete = constants.%Destroy.facet.a88]
// CHECK:STDOUT:   %.loc14_3.2: %Destroy.type = converted constants.%struct_type.a.b.06d, %Destroy.facet.loc14_3.1 [concrete = constants.%Destroy.facet.a88]
// CHECK:STDOUT:   %Destroy.facet.loc14_3.2: %Destroy.type = facet_value constants.%struct_type.a.b.06d, (constants.%custom_witness.b80ef6.4, constants.%DestroyImpl.impl_witness.137) [concrete = constants.%Destroy.facet.a88]
// CHECK:STDOUT:   %.loc14_3.3: %Destroy.type = converted constants.%struct_type.a.b.06d, %Destroy.facet.loc14_3.2 [concrete = constants.%Destroy.facet.a88]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%x.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc10_10.1(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc10_10.2(%self.param: ref %A) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc10_10.3(%self.param: ref %B) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc10_10.4(%self.param: ref %struct_type.a.b.06d) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc10_10.1: ref %B = struct_access %self.param, element1
// CHECK:STDOUT:   %Destroy.facet.loc10_10.1: %Destroy.type = facet_value constants.%B, (constants.%custom_witness.b80ef6.3, constants.%DestroyImpl.impl_witness.131) [concrete = constants.%Destroy.facet.dc1]
// CHECK:STDOUT:   %.loc10_10.2: %Destroy.type = converted constants.%B, %Destroy.facet.loc10_10.1 [concrete = constants.%Destroy.facet.dc1]
// CHECK:STDOUT:   %Destroy.facet.loc10_10.2: %Destroy.type = facet_value constants.%B, (constants.%custom_witness.b80ef6.3, constants.%DestroyImpl.impl_witness.131) [concrete = constants.%Destroy.facet.dc1]
// CHECK:STDOUT:   %.loc10_10.3: %Destroy.type = converted constants.%B, %Destroy.facet.loc10_10.2 [concrete = constants.%Destroy.facet.dc1]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc10_10.1: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%.loc10_10.1)
// CHECK:STDOUT:   %.loc10_10.4: ref %A = struct_access %self.param, element0
// CHECK:STDOUT:   %Destroy.facet.loc10_10.3: %Destroy.type = facet_value constants.%A, (constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.6d1) [concrete = constants.%Destroy.facet.3d0]
// CHECK:STDOUT:   %.loc10_10.5: %Destroy.type = converted constants.%A, %Destroy.facet.loc10_10.3 [concrete = constants.%Destroy.facet.3d0]
// CHECK:STDOUT:   %Destroy.facet.loc10_10.4: %Destroy.type = facet_value constants.%A, (constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.6d1) [concrete = constants.%Destroy.facet.3d0]
// CHECK:STDOUT:   %.loc10_10.6: %Destroy.type = converted constants.%A, %Destroy.facet.loc10_10.4 [concrete = constants.%Destroy.facet.3d0]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc10_10.2: init %empty_tuple.type = call %SelfDestruct.specific_fn.2(%.loc10_10.4)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- todo_trivial_nested_struct_with_trivial_members.carbon = $todo_trivial_nested_struct_with_trivial_members
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %B: type = class_type @B [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %struct_type.b.dce: type = struct_type {.b: %B} [concrete]
// CHECK:STDOUT:   %struct_type.a.657: type = struct_type {.a: %struct_type.b.dce} [concrete]
// CHECK:STDOUT:   %pattern_type.353: type = pattern_type %struct_type.a.657 [concrete]
// CHECK:STDOUT:   %Make.type: type = fn_type @Make [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Make: %Make.type = struct_value () [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.2: type = fn_type @SubobjectDestroy.WithSelf.Op.loc10_10.2 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.2: %SubobjectDestroy.WithSelf.Op.type.6870fd.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.2: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.2), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.3: type = fn_type @SubobjectDestroy.WithSelf.Op.loc10_10.3 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.3: %SubobjectDestroy.WithSelf.Op.type.6870fd.3 = struct_value () [concrete]
// CHECK:STDOUT:   %SelfDestruct.type: type = fn_type @SelfDestruct [concrete]
// CHECK:STDOUT:   %SelfDestruct: %SelfDestruct.type = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %T.09f: %SubobjectDestroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.type.dea: type = fn_type @T.as_type.as.DestroyImpl.impl.Op, @T.as_type.as.DestroyImpl.impl(%T.09f) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.6e5: %T.as_type.as.DestroyImpl.impl.Op.type.dea = struct_value () [symbolic]
// CHECK:STDOUT:   %SubobjectDestroy.facet.92e: %SubobjectDestroy.type = facet_value %B, (%custom_witness.b80ef6.2) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.131: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.92e) [concrete]
// CHECK:STDOUT:   %Destroy.facet.dc1: %Destroy.type = facet_value %B, (%custom_witness.b80ef6.2, %DestroyImpl.impl_witness.131) [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.3: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.3), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.4: type = fn_type @SubobjectDestroy.WithSelf.Op.loc10_10.4 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.4: %SubobjectDestroy.WithSelf.Op.type.6870fd.4 = struct_value () [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.1fc: %SubobjectDestroy.type = facet_value %struct_type.b.dce, (%custom_witness.b80ef6.3) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.c7f: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.1fc) [concrete]
// CHECK:STDOUT:   %Destroy.facet.4b3: %Destroy.type = facet_value %struct_type.b.dce, (%custom_witness.b80ef6.3, %DestroyImpl.impl_witness.c7f) [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.4: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.4), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.bd5: %SubobjectDestroy.type = facet_value %struct_type.a.657, (%custom_witness.b80ef6.4) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.e9e: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.bd5) [concrete]
// CHECK:STDOUT:   %Destroy.facet.685: %Destroy.type = facet_value %struct_type.a.657, (%custom_witness.b80ef6.4, %DestroyImpl.impl_witness.e9e) [concrete]
// CHECK:STDOUT:   %x.patt: %pattern_type.353 = ref_binding_pattern x [concrete]
// CHECK:STDOUT:   %x.var_patt: %pattern_type.353 = var_pattern %x.patt [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .SubobjectDestroy = %Core.SubobjectDestroy
// CHECK:STDOUT:     .SelfDestruct = %Core.SelfDestruct
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     .TrivialDestroy = %Core.TrivialDestroy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.SubobjectDestroy: type = import_ref Core//prelude/parts/destroy, SubobjectDestroy, loaded [concrete = constants.%SubobjectDestroy.type]
// CHECK:STDOUT:   %Core.SelfDestruct: %SelfDestruct.type = import_ref Core//prelude/parts/destroy, SelfDestruct, loaded [concrete = constants.%SelfDestruct]
// CHECK:STDOUT:   %Core.import_ref.74d: @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op.type (%T.as_type.as.DestroyImpl.impl.Op.type.dea) = import_ref Core//prelude/parts/destroy, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op (constants.%T.as_type.as.DestroyImpl.impl.Op.6e5)]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table = impl_witness_table (%Core.import_ref.74d), @T.as_type.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Core.TrivialDestroy: type = import_ref Core//prelude/parts/destroy, TrivialDestroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Test() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Struct.ref.loc10: type = name_ref Struct, file.%Struct [concrete = constants.%struct_type.a.657]
// CHECK:STDOUT:   %Core.ref.loc10: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:   %SubobjectDestroy.ref: type = name_ref SubobjectDestroy, imports.%Core.SubobjectDestroy [concrete = constants.%SubobjectDestroy.type]
// CHECK:STDOUT:   %SubobjectDestroy.facet: %SubobjectDestroy.type = facet_value %Struct.ref.loc10, (constants.%custom_witness.b80ef6.4) [concrete = constants.%SubobjectDestroy.facet.bd5]
// CHECK:STDOUT:   %.loc10: %SubobjectDestroy.type = converted %Struct.ref.loc10, %SubobjectDestroy.facet [concrete = constants.%SubobjectDestroy.facet.bd5]
// CHECK:STDOUT:   %Struct.ref.loc11: type = name_ref Struct, file.%Struct [concrete = constants.%struct_type.a.657]
// CHECK:STDOUT:   %Core.ref.loc11: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:   %Destroy.ref: type = name_ref Destroy, imports.%Core.Destroy [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Destroy.facet.loc11: %Destroy.type = facet_value %Struct.ref.loc11, (constants.%custom_witness.b80ef6.4, constants.%DestroyImpl.impl_witness.e9e) [concrete = constants.%Destroy.facet.685]
// CHECK:STDOUT:   %.loc11: %Destroy.type = converted %Struct.ref.loc11, %Destroy.facet.loc11 [concrete = constants.%Destroy.facet.685]
// CHECK:STDOUT:   %Struct.ref.loc12: type = name_ref Struct, file.%Struct [concrete = constants.%struct_type.a.657]
// CHECK:STDOUT:   %Core.ref.loc12: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:   %TrivialDestroy.ref: type = name_ref TrivialDestroy, imports.%Core.TrivialDestroy [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Destroy.facet.loc12: %Destroy.type = facet_value %Struct.ref.loc12, (constants.%custom_witness.b80ef6.4, constants.%DestroyImpl.impl_witness.e9e) [concrete = constants.%Destroy.facet.685]
// CHECK:STDOUT:   %.loc12: %Destroy.type = converted %Struct.ref.loc12, %Destroy.facet.loc12 [concrete = constants.%Destroy.facet.685]
// CHECK:STDOUT:   %x.var: ref %struct_type.a.657 = var_storage %x.var_patt
// CHECK:STDOUT:   %Make.ref: %Make.type = name_ref Make, file.%Make.decl [concrete = constants.%Make]
// CHECK:STDOUT:   %Make.call: init %struct_type.a.657 = call %Make.ref()
// CHECK:STDOUT:   assign %x.var, %Make.call
// CHECK:STDOUT:   %Struct.ref.loc14: type = name_ref Struct, file.%Struct [concrete = constants.%struct_type.a.657]
// CHECK:STDOUT:   %x: ref %struct_type.a.657 = wrapper_binding x, %x.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %pattern_type.353 = ref_binding_pattern x [concrete = constants.%x.patt]
// CHECK:STDOUT:     %x.var_patt: %pattern_type.353 = var_pattern %x.patt [concrete = constants.%x.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.facet.loc14_3.1: %Destroy.type = facet_value constants.%struct_type.a.657, (constants.%custom_witness.b80ef6.4, constants.%DestroyImpl.impl_witness.e9e) [concrete = constants.%Destroy.facet.685]
// CHECK:STDOUT:   %.loc14_3.1: %Destroy.type = converted constants.%struct_type.a.657, %Destroy.facet.loc14_3.1 [concrete = constants.%Destroy.facet.685]
// CHECK:STDOUT:   %Destroy.facet.loc14_3.2: %Destroy.type = facet_value constants.%struct_type.a.657, (constants.%custom_witness.b80ef6.4, constants.%DestroyImpl.impl_witness.e9e) [concrete = constants.%Destroy.facet.685]
// CHECK:STDOUT:   %.loc14_3.2: %Destroy.type = converted constants.%struct_type.a.657, %Destroy.facet.loc14_3.2 [concrete = constants.%Destroy.facet.685]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%x.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc10_10.1(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc10_10.2(%self.param: ref %B) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc10_10.3(%self.param: ref %struct_type.b.dce) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc10_10.1: ref %B = struct_access %self.param, element0
// CHECK:STDOUT:   %Destroy.facet.loc10_10.1: %Destroy.type = facet_value constants.%B, (constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.131) [concrete = constants.%Destroy.facet.dc1]
// CHECK:STDOUT:   %.loc10_10.2: %Destroy.type = converted constants.%B, %Destroy.facet.loc10_10.1 [concrete = constants.%Destroy.facet.dc1]
// CHECK:STDOUT:   %Destroy.facet.loc10_10.2: %Destroy.type = facet_value constants.%B, (constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.131) [concrete = constants.%Destroy.facet.dc1]
// CHECK:STDOUT:   %.loc10_10.3: %Destroy.type = converted constants.%B, %Destroy.facet.loc10_10.2 [concrete = constants.%Destroy.facet.dc1]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%.loc10_10.1)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc10_10.4(%self.param: ref %struct_type.a.657) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc10_10.1: ref %struct_type.b.dce = struct_access %self.param, element0
// CHECK:STDOUT:   %Destroy.facet.loc10_10.1: %Destroy.type = facet_value constants.%struct_type.b.dce, (constants.%custom_witness.b80ef6.3, constants.%DestroyImpl.impl_witness.c7f) [concrete = constants.%Destroy.facet.4b3]
// CHECK:STDOUT:   %.loc10_10.2: %Destroy.type = converted constants.%struct_type.b.dce, %Destroy.facet.loc10_10.1 [concrete = constants.%Destroy.facet.4b3]
// CHECK:STDOUT:   %Destroy.facet.loc10_10.2: %Destroy.type = facet_value constants.%struct_type.b.dce, (constants.%custom_witness.b80ef6.3, constants.%DestroyImpl.impl_witness.c7f) [concrete = constants.%Destroy.facet.4b3]
// CHECK:STDOUT:   %.loc10_10.3: %Destroy.type = converted constants.%struct_type.b.dce, %Destroy.facet.loc10_10.2 [concrete = constants.%Destroy.facet.4b3]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%.loc10_10.1)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
