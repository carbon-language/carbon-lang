// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/destroy.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/impl/custom_witness/destruction/non_trivial/adapters.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/impl/custom_witness/destruction/non_trivial/adapters.carbon

// --- types.carbon
library "[[@TEST_NAME]]";

class HasTrivialDestruction {}

// `HasNonTrivialDestruction.(SubobjectDestroy.Op)` is a no-op because it
// doesn't have any fields that require it to be non-trivial.
class HasNonTrivialDestruction {
  impl as Core.DestroyImpl {
    fn Op(unused self) {}
  }
}

// --- todo_nontrivial_class_adapter.carbon
library "[[@TEST_NAME]]";

import library "types";

class Adapter {
  adapt HasNonTrivialDestruction;
}

fn Make() -> Adapter;

fn Test() {
  //@dump-sem-ir-begin
  Adapter as Core.SubobjectDestroy;
  Adapter as Core.Destroy;

  var unused x: Adapter = Make();
  //@dump-sem-ir-end
}

// --- todo_nontrivial_destroy_adapter_with_destroy_impl.carbon
library "[[@TEST_NAME]]";

class HasTrivialDestruction {}

class Adapter {
  adapt HasTrivialDestruction;

  impl as Core.DestroyImpl {
    fn Op(unused self) {}
  }
}

fn Make() -> Adapter;

fn Test() {
  //@dump-sem-ir-begin
  Adapter as Core.SubobjectDestroy;
  Adapter as Core.Destroy;

  var unused x: Adapter = Make();
  //@dump-sem-ir-end
}

// --- todo_nontrivial_struct_adapter.carbon
library "[[@TEST_NAME]]";

import library "types";

class Adapter {
  adapt {.x: HasNonTrivialDestruction};
}

fn Make() -> Adapter;

fn Test() {
  //@dump-sem-ir-begin
  Adapter as Core.SubobjectDestroy;
  Adapter as Core.Destroy;

  var unused x: Adapter = Make();
  //@dump-sem-ir-end
}

// --- todo_nontrivial_nontrivial_tuple_adapter.carbon
library "[[@TEST_NAME]]";

import library "types";

class Adapter {
  adapt (HasNonTrivialDestruction, HasTrivialDestruction);
}

fn Make() -> Adapter;

fn Test() {
  //@dump-sem-ir-begin
  Adapter as Core.SubobjectDestroy;
  Adapter as Core.Destroy;

  var unused x: Adapter = Make();
  //@dump-sem-ir-end
}

// --- todo_nontrivial_const.carbon
library "[[@TEST_NAME]]";

import library "types";

alias Adapter = const HasNonTrivialDestruction;
fn Make() -> Adapter;

fn Test() {
  //@dump-sem-ir-begin
  Adapter as Core.SubobjectDestroy;
  Adapter as Core.Destroy;

  var unused x: Adapter = Make();
  //@dump-sem-ir-end
}

// CHECK:STDOUT: --- todo_nontrivial_class_adapter.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %Adapter: type = class_type @Adapter [concrete]
// CHECK:STDOUT:   %HasNonTrivialDestruction: type = class_type @HasNonTrivialDestruction [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %pattern_type.e90: type = pattern_type %Adapter [concrete]
// CHECK:STDOUT:   %Make.type: type = fn_type @Make [concrete]
// CHECK:STDOUT:   %Make: %Make.type = struct_value () [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.3: type = fn_type @SubobjectDestroy.WithSelf.Op.loc13_11.3 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.3: %SubobjectDestroy.WithSelf.Op.type.6870fd.3 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.3: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.3), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.e00: %SubobjectDestroy.type = facet_value %Adapter, (%custom_witness.b80ef6.3) [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %T.09f: %SubobjectDestroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.type.dea: type = fn_type @T.as_type.as.DestroyImpl.impl.Op, @T.as_type.as.DestroyImpl.impl(%T.09f) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.6e5: %T.as_type.as.DestroyImpl.impl.Op.type.dea = struct_value () [symbolic]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.7cf: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.e00) [concrete]
// CHECK:STDOUT:   %Destroy.facet: %Destroy.type = facet_value %Adapter, (%custom_witness.b80ef6.3, %DestroyImpl.impl_witness.7cf) [concrete]
// CHECK:STDOUT:   %x.patt: %pattern_type.e90 = ref_binding_pattern x [concrete]
// CHECK:STDOUT:   %x.var_patt: %pattern_type.e90 = var_pattern %x.patt [concrete]
// CHECK:STDOUT:   %SelfDestruct.type: type = fn_type @SelfDestruct [concrete]
// CHECK:STDOUT:   %SelfDestruct: %SelfDestruct.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .SubobjectDestroy = %Core.SubobjectDestroy
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     .SelfDestruct = %Core.SelfDestruct
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.SubobjectDestroy: type = import_ref Core//prelude/parts/destroy, SubobjectDestroy, loaded [concrete = constants.%SubobjectDestroy.type]
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Core.import_ref.74d: @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op.type (%T.as_type.as.DestroyImpl.impl.Op.type.dea) = import_ref Core//prelude/parts/destroy, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op (constants.%T.as_type.as.DestroyImpl.impl.Op.6e5)]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table = impl_witness_table (%Core.import_ref.74d), @T.as_type.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT:   %Core.SelfDestruct: %SelfDestruct.type = import_ref Core//prelude/parts/destroy, SelfDestruct, loaded [concrete = constants.%SelfDestruct]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Test() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Adapter.ref.loc13: type = name_ref Adapter, file.%Adapter.decl [concrete = constants.%Adapter]
// CHECK:STDOUT:   %Core.ref.loc13: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:   %SubobjectDestroy.ref: type = name_ref SubobjectDestroy, imports.%Core.SubobjectDestroy [concrete = constants.%SubobjectDestroy.type]
// CHECK:STDOUT:   %SubobjectDestroy.facet: %SubobjectDestroy.type = facet_value %Adapter.ref.loc13, (constants.%custom_witness.b80ef6.3) [concrete = constants.%SubobjectDestroy.facet.e00]
// CHECK:STDOUT:   %.loc13: %SubobjectDestroy.type = converted %Adapter.ref.loc13, %SubobjectDestroy.facet [concrete = constants.%SubobjectDestroy.facet.e00]
// CHECK:STDOUT:   %Adapter.ref.loc14: type = name_ref Adapter, file.%Adapter.decl [concrete = constants.%Adapter]
// CHECK:STDOUT:   %Core.ref.loc14: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:   %Destroy.ref: type = name_ref Destroy, imports.%Core.Destroy [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Destroy.facet.loc14: %Destroy.type = facet_value %Adapter.ref.loc14, (constants.%custom_witness.b80ef6.3, constants.%DestroyImpl.impl_witness.7cf) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc14: %Destroy.type = converted %Adapter.ref.loc14, %Destroy.facet.loc14 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %x.var: ref %Adapter = var_storage %x.var_patt
// CHECK:STDOUT:   %Make.ref: %Make.type = name_ref Make, file.%Make.decl [concrete = constants.%Make]
// CHECK:STDOUT:   %.loc16_3.1: ref %Adapter = splice_block %x.var {}
// CHECK:STDOUT:   %Make.call: init %Adapter to %.loc16_3.1 = call %Make.ref()
// CHECK:STDOUT:   assign %x.var, %Make.call
// CHECK:STDOUT:   %Adapter.ref.loc16: type = name_ref Adapter, file.%Adapter.decl [concrete = constants.%Adapter]
// CHECK:STDOUT:   %x: ref %Adapter = wrapper_binding x, %x.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %pattern_type.e90 = ref_binding_pattern x [concrete = constants.%x.patt]
// CHECK:STDOUT:     %x.var_patt: %pattern_type.e90 = var_pattern %x.patt [concrete = constants.%x.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.facet.loc16_3.1: %Destroy.type = facet_value constants.%Adapter, (constants.%custom_witness.b80ef6.3, constants.%DestroyImpl.impl_witness.7cf) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc16_3.2: %Destroy.type = converted constants.%Adapter, %Destroy.facet.loc16_3.1 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %Destroy.facet.loc16_3.2: %Destroy.type = facet_value constants.%Adapter, (constants.%custom_witness.b80ef6.3, constants.%DestroyImpl.impl_witness.7cf) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc16_3.3: %Destroy.type = converted constants.%Adapter, %Destroy.facet.loc16_3.2 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%x.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc13_11.1(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc13_11.2(%self.param: ref %HasNonTrivialDestruction) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc13_11.3(%self.param: ref %Adapter) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: --- todo_nontrivial_destroy_adapter_with_destroy_impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %HasTrivialDestruction: type = class_type @HasTrivialDestruction [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %Adapter: type = class_type @Adapter [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %DestroyImpl.type: type = facet_type <@DestroyImpl> [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.649: <witness> = impl_witness @Adapter.as.DestroyImpl.impl.%DestroyImpl.impl_witness_table [concrete]
// CHECK:STDOUT:   %pattern_type.e90: type = pattern_type %Adapter [concrete]
// CHECK:STDOUT:   %Make.type: type = fn_type @Make [concrete]
// CHECK:STDOUT:   %Make: %Make.type = struct_value () [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.3: type = fn_type @SubobjectDestroy.WithSelf.Op.loc17_11.3 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.3: %SubobjectDestroy.WithSelf.Op.type.6870fd.3 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.3: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.3), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.e00: %SubobjectDestroy.type = facet_value %Adapter, (%custom_witness.b80ef6.3) [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %Destroy.facet: %Destroy.type = facet_value %Adapter, (%DestroyImpl.impl_witness.649, %custom_witness.b80ef6.3) [concrete]
// CHECK:STDOUT:   %x.patt: %pattern_type.e90 = ref_binding_pattern x [concrete]
// CHECK:STDOUT:   %x.var_patt: %pattern_type.e90 = var_pattern %x.patt [concrete]
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
// CHECK:STDOUT:   %Core.SelfDestruct: %SelfDestruct.type = import_ref Core//prelude/parts/destroy, SelfDestruct, loaded [concrete = constants.%SelfDestruct]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Test() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Adapter.ref.loc17: type = name_ref Adapter, file.%Adapter.decl [concrete = constants.%Adapter]
// CHECK:STDOUT:   %Core.ref.loc17: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:   %SubobjectDestroy.ref: type = name_ref SubobjectDestroy, imports.%Core.SubobjectDestroy [concrete = constants.%SubobjectDestroy.type]
// CHECK:STDOUT:   %SubobjectDestroy.facet: %SubobjectDestroy.type = facet_value %Adapter.ref.loc17, (constants.%custom_witness.b80ef6.3) [concrete = constants.%SubobjectDestroy.facet.e00]
// CHECK:STDOUT:   %.loc17: %SubobjectDestroy.type = converted %Adapter.ref.loc17, %SubobjectDestroy.facet [concrete = constants.%SubobjectDestroy.facet.e00]
// CHECK:STDOUT:   %Adapter.ref.loc18: type = name_ref Adapter, file.%Adapter.decl [concrete = constants.%Adapter]
// CHECK:STDOUT:   %Core.ref.loc18: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:   %Destroy.ref: type = name_ref Destroy, imports.%Core.Destroy [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Destroy.facet.loc18: %Destroy.type = facet_value %Adapter.ref.loc18, (constants.%DestroyImpl.impl_witness.649, constants.%custom_witness.b80ef6.3) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc18: %Destroy.type = converted %Adapter.ref.loc18, %Destroy.facet.loc18 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %x.var: ref %Adapter = var_storage %x.var_patt
// CHECK:STDOUT:   %Make.ref: %Make.type = name_ref Make, file.%Make.decl [concrete = constants.%Make]
// CHECK:STDOUT:   %.loc20_3.1: ref %Adapter = splice_block %x.var {}
// CHECK:STDOUT:   %Make.call: init %Adapter to %.loc20_3.1 = call %Make.ref()
// CHECK:STDOUT:   assign %x.var, %Make.call
// CHECK:STDOUT:   %Adapter.ref.loc20: type = name_ref Adapter, file.%Adapter.decl [concrete = constants.%Adapter]
// CHECK:STDOUT:   %x: ref %Adapter = wrapper_binding x, %x.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %pattern_type.e90 = ref_binding_pattern x [concrete = constants.%x.patt]
// CHECK:STDOUT:     %x.var_patt: %pattern_type.e90 = var_pattern %x.patt [concrete = constants.%x.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.facet.loc20_3.1: %Destroy.type = facet_value constants.%Adapter, (constants.%DestroyImpl.impl_witness.649, constants.%custom_witness.b80ef6.3) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc20_3.2: %Destroy.type = converted constants.%Adapter, %Destroy.facet.loc20_3.1 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %Destroy.facet.loc20_3.2: %Destroy.type = facet_value constants.%Adapter, (constants.%DestroyImpl.impl_witness.649, constants.%custom_witness.b80ef6.3) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc20_3.3: %Destroy.type = converted constants.%Adapter, %Destroy.facet.loc20_3.2 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%x.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc17_11.1(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc17_11.2(%self.param: ref %HasTrivialDestruction) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc17_11.3(%self.param: ref %Adapter) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: --- todo_nontrivial_struct_adapter.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %Adapter: type = class_type @Adapter [concrete]
// CHECK:STDOUT:   %HasNonTrivialDestruction: type = class_type @HasNonTrivialDestruction [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %struct_type.x.11f: type = struct_type {.x: %HasNonTrivialDestruction} [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %pattern_type.e90: type = pattern_type %Adapter [concrete]
// CHECK:STDOUT:   %Make.type: type = fn_type @Make [concrete]
// CHECK:STDOUT:   %Make: %Make.type = struct_value () [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.2: type = fn_type @SubobjectDestroy.WithSelf.Op.loc13_11.2 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.2: %SubobjectDestroy.WithSelf.Op.type.6870fd.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.2: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.2), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SelfDestruct.type: type = fn_type @SelfDestruct [concrete]
// CHECK:STDOUT:   %SelfDestruct: %SelfDestruct.type = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %T.09f: %SubobjectDestroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.type.dea: type = fn_type @T.as_type.as.DestroyImpl.impl.Op, @T.as_type.as.DestroyImpl.impl(%T.09f) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.6e5: %T.as_type.as.DestroyImpl.impl.Op.type.dea = struct_value () [symbolic]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.490: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table.467 [concrete]
// CHECK:STDOUT:   %Destroy.facet.4ef: %Destroy.type = facet_value %HasNonTrivialDestruction, (%custom_witness.b80ef6.2, %DestroyImpl.impl_witness.490) [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.4: type = fn_type @SubobjectDestroy.WithSelf.Op.loc13_11.4 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.4: %SubobjectDestroy.WithSelf.Op.type.6870fd.4 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.4: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.4), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.e00: %SubobjectDestroy.type = facet_value %Adapter, (%custom_witness.b80ef6.4) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.7cf: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table.a5e, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.e00) [concrete]
// CHECK:STDOUT:   %Destroy.facet.534: %Destroy.type = facet_value %Adapter, (%custom_witness.b80ef6.4, %DestroyImpl.impl_witness.7cf) [concrete]
// CHECK:STDOUT:   %x.patt: %pattern_type.e90 = ref_binding_pattern x [concrete]
// CHECK:STDOUT:   %x.var_patt: %pattern_type.e90 = var_pattern %x.patt [concrete]
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
// CHECK:STDOUT:   %Main.import_ref.546: %HasNonTrivialDestruction.as.DestroyImpl.impl.Op.type = import_ref Main//types, loc9_24, loaded [concrete = constants.%HasNonTrivialDestruction.as.DestroyImpl.impl.Op]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table.467 = impl_witness_table (%Main.import_ref.546), @HasNonTrivialDestruction.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Test() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Adapter.ref.loc13: type = name_ref Adapter, file.%Adapter.decl [concrete = constants.%Adapter]
// CHECK:STDOUT:   %Core.ref.loc13: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:   %SubobjectDestroy.ref: type = name_ref SubobjectDestroy, imports.%Core.SubobjectDestroy [concrete = constants.%SubobjectDestroy.type]
// CHECK:STDOUT:   %SubobjectDestroy.facet: %SubobjectDestroy.type = facet_value %Adapter.ref.loc13, (constants.%custom_witness.b80ef6.4) [concrete = constants.%SubobjectDestroy.facet.e00]
// CHECK:STDOUT:   %.loc13: %SubobjectDestroy.type = converted %Adapter.ref.loc13, %SubobjectDestroy.facet [concrete = constants.%SubobjectDestroy.facet.e00]
// CHECK:STDOUT:   %Adapter.ref.loc14: type = name_ref Adapter, file.%Adapter.decl [concrete = constants.%Adapter]
// CHECK:STDOUT:   %Core.ref.loc14: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:   %Destroy.ref: type = name_ref Destroy, imports.%Core.Destroy [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Destroy.facet.loc14: %Destroy.type = facet_value %Adapter.ref.loc14, (constants.%custom_witness.b80ef6.4, constants.%DestroyImpl.impl_witness.7cf) [concrete = constants.%Destroy.facet.534]
// CHECK:STDOUT:   %.loc14: %Destroy.type = converted %Adapter.ref.loc14, %Destroy.facet.loc14 [concrete = constants.%Destroy.facet.534]
// CHECK:STDOUT:   %x.var: ref %Adapter = var_storage %x.var_patt
// CHECK:STDOUT:   %Make.ref: %Make.type = name_ref Make, file.%Make.decl [concrete = constants.%Make]
// CHECK:STDOUT:   %Make.call: init %Adapter = call %Make.ref()
// CHECK:STDOUT:   assign %x.var, %Make.call
// CHECK:STDOUT:   %Adapter.ref.loc16: type = name_ref Adapter, file.%Adapter.decl [concrete = constants.%Adapter]
// CHECK:STDOUT:   %x: ref %Adapter = wrapper_binding x, %x.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %pattern_type.e90 = ref_binding_pattern x [concrete = constants.%x.patt]
// CHECK:STDOUT:     %x.var_patt: %pattern_type.e90 = var_pattern %x.patt [concrete = constants.%x.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.facet.loc16_3.1: %Destroy.type = facet_value constants.%Adapter, (constants.%custom_witness.b80ef6.4, constants.%DestroyImpl.impl_witness.7cf) [concrete = constants.%Destroy.facet.534]
// CHECK:STDOUT:   %.loc16_3.1: %Destroy.type = converted constants.%Adapter, %Destroy.facet.loc16_3.1 [concrete = constants.%Destroy.facet.534]
// CHECK:STDOUT:   %Destroy.facet.loc16_3.2: %Destroy.type = facet_value constants.%Adapter, (constants.%custom_witness.b80ef6.4, constants.%DestroyImpl.impl_witness.7cf) [concrete = constants.%Destroy.facet.534]
// CHECK:STDOUT:   %.loc16_3.2: %Destroy.type = converted constants.%Adapter, %Destroy.facet.loc16_3.2 [concrete = constants.%Destroy.facet.534]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%x.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc13_11.1(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc13_11.2(%self.param: ref %HasNonTrivialDestruction) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc13_11.3(%self.param: ref %struct_type.x.11f) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc13_11.1: ref %HasNonTrivialDestruction = struct_access %self.param, element0
// CHECK:STDOUT:   %Destroy.facet.loc13_11.1: %Destroy.type = facet_value constants.%HasNonTrivialDestruction, (constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.490) [concrete = constants.%Destroy.facet.4ef]
// CHECK:STDOUT:   %.loc13_11.2: %Destroy.type = converted constants.%HasNonTrivialDestruction, %Destroy.facet.loc13_11.1 [concrete = constants.%Destroy.facet.4ef]
// CHECK:STDOUT:   %Destroy.facet.loc13_11.2: %Destroy.type = facet_value constants.%HasNonTrivialDestruction, (constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.490) [concrete = constants.%Destroy.facet.4ef]
// CHECK:STDOUT:   %.loc13_11.3: %Destroy.type = converted constants.%HasNonTrivialDestruction, %Destroy.facet.loc13_11.2 [concrete = constants.%Destroy.facet.4ef]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%.loc13_11.1)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc13_11.4(%self.param: ref %Adapter) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: --- todo_nontrivial_nontrivial_tuple_adapter.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %Adapter: type = class_type @Adapter [concrete]
// CHECK:STDOUT:   %HasNonTrivialDestruction: type = class_type @HasNonTrivialDestruction [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %HasTrivialDestruction: type = class_type @HasTrivialDestruction [concrete]
// CHECK:STDOUT:   %tuple.type.aad: type = tuple_type (%HasNonTrivialDestruction, %HasTrivialDestruction) [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %pattern_type.e90: type = pattern_type %Adapter [concrete]
// CHECK:STDOUT:   %Make.type: type = fn_type @Make [concrete]
// CHECK:STDOUT:   %Make: %Make.type = struct_value () [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.2: type = fn_type @SubobjectDestroy.WithSelf.Op.loc13_11.2 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.2: %SubobjectDestroy.WithSelf.Op.type.6870fd.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.2: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.2), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.3: type = fn_type @SubobjectDestroy.WithSelf.Op.loc13_11.3 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.3: %SubobjectDestroy.WithSelf.Op.type.6870fd.3 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.3: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.3), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %SelfDestruct.type: type = fn_type @SelfDestruct [concrete]
// CHECK:STDOUT:   %SelfDestruct: %SelfDestruct.type = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %T.09f: %SubobjectDestroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.type.dea: type = fn_type @T.as_type.as.DestroyImpl.impl.Op, @T.as_type.as.DestroyImpl.impl(%T.09f) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.6e5: %T.as_type.as.DestroyImpl.impl.Op.type.dea = struct_value () [symbolic]
// CHECK:STDOUT:   %SubobjectDestroy.facet.5bc: %SubobjectDestroy.type = facet_value %HasTrivialDestruction, (%custom_witness.b80ef6.3) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.4ca: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table.a5e, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.5bc) [concrete]
// CHECK:STDOUT:   %Destroy.facet.b89: %Destroy.type = facet_value %HasTrivialDestruction, (%custom_witness.b80ef6.3, %DestroyImpl.impl_witness.4ca) [concrete]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.490: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table.467 [concrete]
// CHECK:STDOUT:   %Destroy.facet.4ef: %Destroy.type = facet_value %HasNonTrivialDestruction, (%custom_witness.b80ef6.2, %DestroyImpl.impl_witness.490) [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.5: type = fn_type @SubobjectDestroy.WithSelf.Op.loc13_11.5 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.5: %SubobjectDestroy.WithSelf.Op.type.6870fd.5 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.5: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.5), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.e00: %SubobjectDestroy.type = facet_value %Adapter, (%custom_witness.b80ef6.5) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.7cf: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table.a5e, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.e00) [concrete]
// CHECK:STDOUT:   %Destroy.facet.534: %Destroy.type = facet_value %Adapter, (%custom_witness.b80ef6.5, %DestroyImpl.impl_witness.7cf) [concrete]
// CHECK:STDOUT:   %x.patt: %pattern_type.e90 = ref_binding_pattern x [concrete]
// CHECK:STDOUT:   %x.var_patt: %pattern_type.e90 = var_pattern %x.patt [concrete]
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
// CHECK:STDOUT:   %Main.import_ref.546: %HasNonTrivialDestruction.as.DestroyImpl.impl.Op.type = import_ref Main//types, loc9_24, loaded [concrete = constants.%HasNonTrivialDestruction.as.DestroyImpl.impl.Op]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table.467 = impl_witness_table (%Main.import_ref.546), @HasNonTrivialDestruction.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Test() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Adapter.ref.loc13: type = name_ref Adapter, file.%Adapter.decl [concrete = constants.%Adapter]
// CHECK:STDOUT:   %Core.ref.loc13: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:   %SubobjectDestroy.ref: type = name_ref SubobjectDestroy, imports.%Core.SubobjectDestroy [concrete = constants.%SubobjectDestroy.type]
// CHECK:STDOUT:   %SubobjectDestroy.facet: %SubobjectDestroy.type = facet_value %Adapter.ref.loc13, (constants.%custom_witness.b80ef6.5) [concrete = constants.%SubobjectDestroy.facet.e00]
// CHECK:STDOUT:   %.loc13: %SubobjectDestroy.type = converted %Adapter.ref.loc13, %SubobjectDestroy.facet [concrete = constants.%SubobjectDestroy.facet.e00]
// CHECK:STDOUT:   %Adapter.ref.loc14: type = name_ref Adapter, file.%Adapter.decl [concrete = constants.%Adapter]
// CHECK:STDOUT:   %Core.ref.loc14: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:   %Destroy.ref: type = name_ref Destroy, imports.%Core.Destroy [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Destroy.facet.loc14: %Destroy.type = facet_value %Adapter.ref.loc14, (constants.%custom_witness.b80ef6.5, constants.%DestroyImpl.impl_witness.7cf) [concrete = constants.%Destroy.facet.534]
// CHECK:STDOUT:   %.loc14: %Destroy.type = converted %Adapter.ref.loc14, %Destroy.facet.loc14 [concrete = constants.%Destroy.facet.534]
// CHECK:STDOUT:   %x.var: ref %Adapter = var_storage %x.var_patt
// CHECK:STDOUT:   %Make.ref: %Make.type = name_ref Make, file.%Make.decl [concrete = constants.%Make]
// CHECK:STDOUT:   %.loc16_3.1: ref %Adapter = splice_block %x.var {}
// CHECK:STDOUT:   %Make.call: init %Adapter to %.loc16_3.1 = call %Make.ref()
// CHECK:STDOUT:   assign %x.var, %Make.call
// CHECK:STDOUT:   %Adapter.ref.loc16: type = name_ref Adapter, file.%Adapter.decl [concrete = constants.%Adapter]
// CHECK:STDOUT:   %x: ref %Adapter = wrapper_binding x, %x.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %pattern_type.e90 = ref_binding_pattern x [concrete = constants.%x.patt]
// CHECK:STDOUT:     %x.var_patt: %pattern_type.e90 = var_pattern %x.patt [concrete = constants.%x.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.facet.loc16_3.1: %Destroy.type = facet_value constants.%Adapter, (constants.%custom_witness.b80ef6.5, constants.%DestroyImpl.impl_witness.7cf) [concrete = constants.%Destroy.facet.534]
// CHECK:STDOUT:   %.loc16_3.2: %Destroy.type = converted constants.%Adapter, %Destroy.facet.loc16_3.1 [concrete = constants.%Destroy.facet.534]
// CHECK:STDOUT:   %Destroy.facet.loc16_3.2: %Destroy.type = facet_value constants.%Adapter, (constants.%custom_witness.b80ef6.5, constants.%DestroyImpl.impl_witness.7cf) [concrete = constants.%Destroy.facet.534]
// CHECK:STDOUT:   %.loc16_3.3: %Destroy.type = converted constants.%Adapter, %Destroy.facet.loc16_3.2 [concrete = constants.%Destroy.facet.534]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%x.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc13_11.1(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc13_11.2(%self.param: ref %HasNonTrivialDestruction) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc13_11.3(%self.param: ref %HasTrivialDestruction) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc13_11.4(%self.param: ref %tuple.type.aad) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1]
// CHECK:STDOUT:   %tuple.elem1: ref %HasTrivialDestruction = tuple_access %self.param, element1
// CHECK:STDOUT:   %Destroy.facet.loc13_11.1: %Destroy.type = facet_value constants.%HasTrivialDestruction, (constants.%custom_witness.b80ef6.3, constants.%DestroyImpl.impl_witness.4ca) [concrete = constants.%Destroy.facet.b89]
// CHECK:STDOUT:   %.loc13_11.1: %Destroy.type = converted constants.%HasTrivialDestruction, %Destroy.facet.loc13_11.1 [concrete = constants.%Destroy.facet.b89]
// CHECK:STDOUT:   %Destroy.facet.loc13_11.2: %Destroy.type = facet_value constants.%HasTrivialDestruction, (constants.%custom_witness.b80ef6.3, constants.%DestroyImpl.impl_witness.4ca) [concrete = constants.%Destroy.facet.b89]
// CHECK:STDOUT:   %.loc13_11.2: %Destroy.type = converted constants.%HasTrivialDestruction, %Destroy.facet.loc13_11.2 [concrete = constants.%Destroy.facet.b89]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc13_11.1: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%tuple.elem1)
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT:   %tuple.elem0: ref %HasNonTrivialDestruction = tuple_access %self.param, element0
// CHECK:STDOUT:   %Destroy.facet.loc13_11.3: %Destroy.type = facet_value constants.%HasNonTrivialDestruction, (constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.490) [concrete = constants.%Destroy.facet.4ef]
// CHECK:STDOUT:   %.loc13_11.3: %Destroy.type = converted constants.%HasNonTrivialDestruction, %Destroy.facet.loc13_11.3 [concrete = constants.%Destroy.facet.4ef]
// CHECK:STDOUT:   %Destroy.facet.loc13_11.4: %Destroy.type = facet_value constants.%HasNonTrivialDestruction, (constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.490) [concrete = constants.%Destroy.facet.4ef]
// CHECK:STDOUT:   %.loc13_11.4: %Destroy.type = converted constants.%HasNonTrivialDestruction, %Destroy.facet.loc13_11.4 [concrete = constants.%Destroy.facet.4ef]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc13_11.2: init %empty_tuple.type = call %SelfDestruct.specific_fn.2(%tuple.elem0)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc13_11.5(%self.param: ref %Adapter) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: --- todo_nontrivial_const.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %HasNonTrivialDestruction: type = class_type @HasNonTrivialDestruction [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %const: type = const_type %HasNonTrivialDestruction [concrete]
// CHECK:STDOUT:   %pattern_type.18d: type = pattern_type %const [concrete]
// CHECK:STDOUT:   %Make.type: type = fn_type @Make [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Make: %Make.type = struct_value () [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.3: type = fn_type @SubobjectDestroy.WithSelf.Op.loc10_11.3 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.3: %SubobjectDestroy.WithSelf.Op.type.6870fd.3 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.3: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.3), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.f1b: %SubobjectDestroy.type = facet_value %const, (%custom_witness.b80ef6.3) [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %T.09f: %SubobjectDestroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.type.dea: type = fn_type @T.as_type.as.DestroyImpl.impl.Op, @T.as_type.as.DestroyImpl.impl(%T.09f) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.6e5: %T.as_type.as.DestroyImpl.impl.Op.type.dea = struct_value () [symbolic]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.da7: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.f1b) [concrete]
// CHECK:STDOUT:   %Destroy.facet: %Destroy.type = facet_value %const, (%custom_witness.b80ef6.3, %DestroyImpl.impl_witness.da7) [concrete]
// CHECK:STDOUT:   %x.patt: %pattern_type.18d = ref_binding_pattern x [concrete]
// CHECK:STDOUT:   %x.var_patt: %pattern_type.18d = var_pattern %x.patt [concrete]
// CHECK:STDOUT:   %SelfDestruct.type: type = fn_type @SelfDestruct [concrete]
// CHECK:STDOUT:   %SelfDestruct: %SelfDestruct.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .SubobjectDestroy = %Core.SubobjectDestroy
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     .SelfDestruct = %Core.SelfDestruct
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.SubobjectDestroy: type = import_ref Core//prelude/parts/destroy, SubobjectDestroy, loaded [concrete = constants.%SubobjectDestroy.type]
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Core.import_ref.74d: @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op.type (%T.as_type.as.DestroyImpl.impl.Op.type.dea) = import_ref Core//prelude/parts/destroy, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op (constants.%T.as_type.as.DestroyImpl.impl.Op.6e5)]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table = impl_witness_table (%Core.import_ref.74d), @T.as_type.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT:   %Core.SelfDestruct: %SelfDestruct.type = import_ref Core//prelude/parts/destroy, SelfDestruct, loaded [concrete = constants.%SelfDestruct]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Test() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Adapter.ref.loc10: type = name_ref Adapter, file.%Adapter [concrete = constants.%const]
// CHECK:STDOUT:   %Core.ref.loc10: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:   %SubobjectDestroy.ref: type = name_ref SubobjectDestroy, imports.%Core.SubobjectDestroy [concrete = constants.%SubobjectDestroy.type]
// CHECK:STDOUT:   %SubobjectDestroy.facet: %SubobjectDestroy.type = facet_value %Adapter.ref.loc10, (constants.%custom_witness.b80ef6.3) [concrete = constants.%SubobjectDestroy.facet.f1b]
// CHECK:STDOUT:   %.loc10: %SubobjectDestroy.type = converted %Adapter.ref.loc10, %SubobjectDestroy.facet [concrete = constants.%SubobjectDestroy.facet.f1b]
// CHECK:STDOUT:   %Adapter.ref.loc11: type = name_ref Adapter, file.%Adapter [concrete = constants.%const]
// CHECK:STDOUT:   %Core.ref.loc11: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:   %Destroy.ref: type = name_ref Destroy, imports.%Core.Destroy [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Destroy.facet.loc11: %Destroy.type = facet_value %Adapter.ref.loc11, (constants.%custom_witness.b80ef6.3, constants.%DestroyImpl.impl_witness.da7) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc11: %Destroy.type = converted %Adapter.ref.loc11, %Destroy.facet.loc11 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %x.var: ref %const = var_storage %x.var_patt
// CHECK:STDOUT:   %Make.ref: %Make.type = name_ref Make, file.%Make.decl [concrete = constants.%Make]
// CHECK:STDOUT:   %.loc13_3.1: ref %const = splice_block %x.var {}
// CHECK:STDOUT:   %Make.call: init %const to %.loc13_3.1 = call %Make.ref()
// CHECK:STDOUT:   assign %x.var, %Make.call
// CHECK:STDOUT:   %Adapter.ref.loc13: type = name_ref Adapter, file.%Adapter [concrete = constants.%const]
// CHECK:STDOUT:   %x: ref %const = wrapper_binding x, %x.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %pattern_type.18d = ref_binding_pattern x [concrete = constants.%x.patt]
// CHECK:STDOUT:     %x.var_patt: %pattern_type.18d = var_pattern %x.patt [concrete = constants.%x.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.facet.loc13_3.1: %Destroy.type = facet_value constants.%const, (constants.%custom_witness.b80ef6.3, constants.%DestroyImpl.impl_witness.da7) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc13_3.2: %Destroy.type = converted constants.%const, %Destroy.facet.loc13_3.1 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %Destroy.facet.loc13_3.2: %Destroy.type = facet_value constants.%const, (constants.%custom_witness.b80ef6.3, constants.%DestroyImpl.impl_witness.da7) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc13_3.3: %Destroy.type = converted constants.%const, %Destroy.facet.loc13_3.2 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%x.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc10_11.1(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc10_11.2(%self.param: ref %HasNonTrivialDestruction) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc10_11.3(%self.param: ref %const) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
