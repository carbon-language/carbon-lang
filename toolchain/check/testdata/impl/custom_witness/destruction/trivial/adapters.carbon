// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/destroy.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/impl/custom_witness/destruction/trivial/adapters.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/impl/custom_witness/destruction/trivial/adapters.carbon

// --- types.carbon
library "[[@TEST_NAME]]";

class HasTrivialDestruction {}

class HasNonTrivialDestruction {
  impl as Core.DestroyImpl {
    fn Op(unused self) {}
  }
}

// --- trivial_class_adapter.carbon
library "[[@TEST_NAME]]";

import library "types";

class Adapter {
  adapt HasTrivialDestruction;
}

fn MakeAdapter() -> Adapter;

fn Test() {
  //@dump-sem-ir-begin
  Adapter as Core.SubobjectDestroy;
  //@dump-sem-ir-end

  Adapter as Core.Destroy;
  Adapter as Core.TrivialDestroy;

  //@dump-sem-ir-begin
  MakeAdapter();
  //@dump-sem-ir-end
}

// --- todo_fail_nontrivial_class_adapter.carbon
library "[[@TEST_NAME]]";

import library "types";

class Adapter {
  adapt HasNonTrivialDestruction;
}

fn Test() {
  Adapter as Core.TrivialDestroy;
}

// --- todo_fail_trivial_destroy_adapter_with_destroy_impl.carbon
library "[[@TEST_NAME]]";

class HasTrivialDestruction {}

class Adapter {
  adapt HasTrivialDestruction;

  impl as Core.DestroyImpl {
    fn Op(unused self) {}
  }
}

fn Test() {
  Adapter as Core.TrivialDestroy;
}

// --- trivial_struct_adapter.carbon
library "[[@TEST_NAME]]";

import library "types";

class Adapter {
  adapt {.x: HasTrivialDestruction, .y: HasTrivialDestruction};
}

fn MakeAdapter() -> Adapter;

fn Test() {
  //@dump-sem-ir-begin
  Adapter as Core.SubobjectDestroy;
  //@dump-sem-ir-end

  Adapter as Core.Destroy;
  Adapter as Core.TrivialDestroy;

  //@dump-sem-ir-begin
  MakeAdapter();
  //@dump-sem-ir-end
}

// --- todo_fail_nontrivial_struct_adapter.carbon
library "[[@TEST_NAME]]";

import library "types";

class Adapter {
  adapt {.n: HasNonTrivialDestruction};
}

fn Test() {
  Adapter as Core.TrivialDestroy;
}

// --- trivial_tuple_adapter.carbon
library "[[@TEST_NAME]]";

import library "types";

class Adapter {
  adapt (HasTrivialDestruction, HasTrivialDestruction);
}

fn MakeAdapter() -> Adapter;

fn Test() {
  //@dump-sem-ir-begin
  Adapter as Core.SubobjectDestroy;
  //@dump-sem-ir-end

  Adapter as Core.Destroy;
  Adapter as Core.TrivialDestroy;

  //@dump-sem-ir-begin
  MakeAdapter();
  //@dump-sem-ir-end
}

// --- todo_fail_nontrivial_tuple_adapter.carbon
library "[[@TEST_NAME]]";

import library "types";

class Adapter {
  adapt (HasNonTrivialDestruction, HasTrivialDestruction);
}

fn Test() {
  Adapter as Core.TrivialDestroy;
}

// --- trivial_pointer.carbon
library "[[@TEST_NAME]]";

import library "types";

alias Pointer = HasNonTrivialDestruction*;
fn Make() -> Pointer;

fn Test() {
  //@dump-sem-ir-begin
  Pointer as Core.SubobjectDestroy;
  //@dump-sem-ir-end

  Pointer as Core.Destroy;
  Pointer as Core.TrivialDestroy;

  //@dump-sem-ir-begin
  var unused x: Pointer = Make();
  //@dump-sem-ir-end
}

// CHECK:STDOUT: --- trivial_class_adapter.carbon = $trivial_class_adapter
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %Adapter: type = class_type @Adapter [concrete]
// CHECK:STDOUT:   %HasTrivialDestruction: type = class_type @HasTrivialDestruction [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %MakeAdapter.type: type = fn_type @MakeAdapter [concrete]
// CHECK:STDOUT:   %MakeAdapter: %MakeAdapter.type = struct_value () [concrete]
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
// CHECK:STDOUT:   %Adapter.ref.loc13: type = name_ref Adapter, file.%Adapter.decl [concrete = constants.%Adapter]
// CHECK:STDOUT:   %Core.ref.loc13: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:   %SubobjectDestroy.ref: type = name_ref SubobjectDestroy, imports.%Core.SubobjectDestroy [concrete = constants.%SubobjectDestroy.type]
// CHECK:STDOUT:   %SubobjectDestroy.facet: %SubobjectDestroy.type = facet_value %Adapter.ref.loc13, (constants.%custom_witness.b80ef6.3) [concrete = constants.%SubobjectDestroy.facet.e00]
// CHECK:STDOUT:   %.loc13: %SubobjectDestroy.type = converted %Adapter.ref.loc13, %SubobjectDestroy.facet [concrete = constants.%SubobjectDestroy.facet.e00]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %MakeAdapter.ref: %MakeAdapter.type = name_ref MakeAdapter, file.%MakeAdapter.decl [concrete = constants.%MakeAdapter]
// CHECK:STDOUT:   %.loc20_15.1: ref %Adapter = temporary_storage
// CHECK:STDOUT:   %MakeAdapter.call: init %Adapter to %.loc20_15.1 = call %MakeAdapter.ref()
// CHECK:STDOUT:   %.loc20_15.2: ref %Adapter = temporary %.loc20_15.1, %MakeAdapter.call
// CHECK:STDOUT:   %Destroy.facet.loc20_15.1: %Destroy.type = facet_value constants.%Adapter, (constants.%custom_witness.b80ef6.3, constants.%DestroyImpl.impl_witness.7cf) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc20_15.3: %Destroy.type = converted constants.%Adapter, %Destroy.facet.loc20_15.1 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %Destroy.facet.loc20_15.2: %Destroy.type = facet_value constants.%Adapter, (constants.%custom_witness.b80ef6.3, constants.%DestroyImpl.impl_witness.7cf) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc20_15.4: %Destroy.type = converted constants.%Adapter, %Destroy.facet.loc20_15.2 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%.loc20_15.2)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc13_11.1(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc13_11.2(%self.param: ref %HasTrivialDestruction) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc13_11.3(%self.param: ref %Adapter) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: --- trivial_struct_adapter.carbon = $trivial_struct_adapter
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %Adapter: type = class_type @Adapter [concrete]
// CHECK:STDOUT:   %HasTrivialDestruction: type = class_type @HasTrivialDestruction [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %struct_type.x.y.a13: type = struct_type {.x: %HasTrivialDestruction, .y: %HasTrivialDestruction} [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %MakeAdapter.type: type = fn_type @MakeAdapter [concrete]
// CHECK:STDOUT:   %MakeAdapter: %MakeAdapter.type = struct_value () [concrete]
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
// CHECK:STDOUT:   %SubobjectDestroy.facet.5bc: %SubobjectDestroy.type = facet_value %HasTrivialDestruction, (%custom_witness.b80ef6.2) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.4ca: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.5bc) [concrete]
// CHECK:STDOUT:   %Destroy.facet.b89: %Destroy.type = facet_value %HasTrivialDestruction, (%custom_witness.b80ef6.2, %DestroyImpl.impl_witness.4ca) [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.4: type = fn_type @SubobjectDestroy.WithSelf.Op.loc13_11.4 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.4: %SubobjectDestroy.WithSelf.Op.type.6870fd.4 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.4: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.4), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.e00: %SubobjectDestroy.type = facet_value %Adapter, (%custom_witness.b80ef6.4) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.7cf: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.e00) [concrete]
// CHECK:STDOUT:   %Destroy.facet.534: %Destroy.type = facet_value %Adapter, (%custom_witness.b80ef6.4, %DestroyImpl.impl_witness.7cf) [concrete]
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
// CHECK:STDOUT:   %Adapter.ref.loc13: type = name_ref Adapter, file.%Adapter.decl [concrete = constants.%Adapter]
// CHECK:STDOUT:   %Core.ref.loc13: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:   %SubobjectDestroy.ref: type = name_ref SubobjectDestroy, imports.%Core.SubobjectDestroy [concrete = constants.%SubobjectDestroy.type]
// CHECK:STDOUT:   %SubobjectDestroy.facet: %SubobjectDestroy.type = facet_value %Adapter.ref.loc13, (constants.%custom_witness.b80ef6.4) [concrete = constants.%SubobjectDestroy.facet.e00]
// CHECK:STDOUT:   %.loc13: %SubobjectDestroy.type = converted %Adapter.ref.loc13, %SubobjectDestroy.facet [concrete = constants.%SubobjectDestroy.facet.e00]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %MakeAdapter.ref: %MakeAdapter.type = name_ref MakeAdapter, file.%MakeAdapter.decl [concrete = constants.%MakeAdapter]
// CHECK:STDOUT:   %.loc20_15.1: ref %Adapter = temporary_storage
// CHECK:STDOUT:   %MakeAdapter.call: init %Adapter to %.loc20_15.1 = call %MakeAdapter.ref()
// CHECK:STDOUT:   %.loc20_15.2: ref %Adapter = temporary %.loc20_15.1, %MakeAdapter.call
// CHECK:STDOUT:   %Destroy.facet.loc20_15.1: %Destroy.type = facet_value constants.%Adapter, (constants.%custom_witness.b80ef6.4, constants.%DestroyImpl.impl_witness.7cf) [concrete = constants.%Destroy.facet.534]
// CHECK:STDOUT:   %.loc20_15.3: %Destroy.type = converted constants.%Adapter, %Destroy.facet.loc20_15.1 [concrete = constants.%Destroy.facet.534]
// CHECK:STDOUT:   %Destroy.facet.loc20_15.2: %Destroy.type = facet_value constants.%Adapter, (constants.%custom_witness.b80ef6.4, constants.%DestroyImpl.impl_witness.7cf) [concrete = constants.%Destroy.facet.534]
// CHECK:STDOUT:   %.loc20_15.4: %Destroy.type = converted constants.%Adapter, %Destroy.facet.loc20_15.2 [concrete = constants.%Destroy.facet.534]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%.loc20_15.2)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc13_11.1(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc13_11.2(%self.param: ref %HasTrivialDestruction) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc13_11.3(%self.param: ref %struct_type.x.y.a13) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc13_11.1: ref %HasTrivialDestruction = struct_access %self.param, element1
// CHECK:STDOUT:   %Destroy.facet.loc13_11.1: %Destroy.type = facet_value constants.%HasTrivialDestruction, (constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.4ca) [concrete = constants.%Destroy.facet.b89]
// CHECK:STDOUT:   %.loc13_11.2: %Destroy.type = converted constants.%HasTrivialDestruction, %Destroy.facet.loc13_11.1 [concrete = constants.%Destroy.facet.b89]
// CHECK:STDOUT:   %Destroy.facet.loc13_11.2: %Destroy.type = facet_value constants.%HasTrivialDestruction, (constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.4ca) [concrete = constants.%Destroy.facet.b89]
// CHECK:STDOUT:   %.loc13_11.3: %Destroy.type = converted constants.%HasTrivialDestruction, %Destroy.facet.loc13_11.2 [concrete = constants.%Destroy.facet.b89]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc13_11.1: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%.loc13_11.1)
// CHECK:STDOUT:   %.loc13_11.4: ref %HasTrivialDestruction = struct_access %self.param, element0
// CHECK:STDOUT:   %Destroy.facet.loc13_11.3: %Destroy.type = facet_value constants.%HasTrivialDestruction, (constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.4ca) [concrete = constants.%Destroy.facet.b89]
// CHECK:STDOUT:   %.loc13_11.5: %Destroy.type = converted constants.%HasTrivialDestruction, %Destroy.facet.loc13_11.3 [concrete = constants.%Destroy.facet.b89]
// CHECK:STDOUT:   %Destroy.facet.loc13_11.4: %Destroy.type = facet_value constants.%HasTrivialDestruction, (constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.4ca) [concrete = constants.%Destroy.facet.b89]
// CHECK:STDOUT:   %.loc13_11.6: %Destroy.type = converted constants.%HasTrivialDestruction, %Destroy.facet.loc13_11.4 [concrete = constants.%Destroy.facet.b89]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc13_11.2: init %empty_tuple.type = call %SelfDestruct.specific_fn.2(%.loc13_11.4)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc13_11.4(%self.param: ref %Adapter) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: --- trivial_tuple_adapter.carbon = $trivial_tuple_adapter
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %Adapter: type = class_type @Adapter [concrete]
// CHECK:STDOUT:   %HasTrivialDestruction: type = class_type @HasTrivialDestruction [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %tuple.type.56a: type = tuple_type (%HasTrivialDestruction, %HasTrivialDestruction) [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %MakeAdapter.type: type = fn_type @MakeAdapter [concrete]
// CHECK:STDOUT:   %MakeAdapter: %MakeAdapter.type = struct_value () [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.2: type = fn_type @SubobjectDestroy.WithSelf.Op.loc13_11.2 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.2: %SubobjectDestroy.WithSelf.Op.type.6870fd.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.2: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.2), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %SelfDestruct.type: type = fn_type @SelfDestruct [concrete]
// CHECK:STDOUT:   %SelfDestruct: %SelfDestruct.type = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %T.09f: %SubobjectDestroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.type.dea: type = fn_type @T.as_type.as.DestroyImpl.impl.Op, @T.as_type.as.DestroyImpl.impl(%T.09f) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.6e5: %T.as_type.as.DestroyImpl.impl.Op.type.dea = struct_value () [symbolic]
// CHECK:STDOUT:   %SubobjectDestroy.facet.5bc: %SubobjectDestroy.type = facet_value %HasTrivialDestruction, (%custom_witness.b80ef6.2) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.4ca: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.5bc) [concrete]
// CHECK:STDOUT:   %Destroy.facet.b89: %Destroy.type = facet_value %HasTrivialDestruction, (%custom_witness.b80ef6.2, %DestroyImpl.impl_witness.4ca) [concrete]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.4: type = fn_type @SubobjectDestroy.WithSelf.Op.loc13_11.4 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.4: %SubobjectDestroy.WithSelf.Op.type.6870fd.4 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.4: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.4), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.e00: %SubobjectDestroy.type = facet_value %Adapter, (%custom_witness.b80ef6.4) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.7cf: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.e00) [concrete]
// CHECK:STDOUT:   %Destroy.facet.534: %Destroy.type = facet_value %Adapter, (%custom_witness.b80ef6.4, %DestroyImpl.impl_witness.7cf) [concrete]
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
// CHECK:STDOUT:   %Adapter.ref.loc13: type = name_ref Adapter, file.%Adapter.decl [concrete = constants.%Adapter]
// CHECK:STDOUT:   %Core.ref.loc13: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:   %SubobjectDestroy.ref: type = name_ref SubobjectDestroy, imports.%Core.SubobjectDestroy [concrete = constants.%SubobjectDestroy.type]
// CHECK:STDOUT:   %SubobjectDestroy.facet: %SubobjectDestroy.type = facet_value %Adapter.ref.loc13, (constants.%custom_witness.b80ef6.4) [concrete = constants.%SubobjectDestroy.facet.e00]
// CHECK:STDOUT:   %.loc13: %SubobjectDestroy.type = converted %Adapter.ref.loc13, %SubobjectDestroy.facet [concrete = constants.%SubobjectDestroy.facet.e00]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %MakeAdapter.ref: %MakeAdapter.type = name_ref MakeAdapter, file.%MakeAdapter.decl [concrete = constants.%MakeAdapter]
// CHECK:STDOUT:   %.loc20_15.1: ref %Adapter = temporary_storage
// CHECK:STDOUT:   %MakeAdapter.call: init %Adapter to %.loc20_15.1 = call %MakeAdapter.ref()
// CHECK:STDOUT:   %.loc20_15.2: ref %Adapter = temporary %.loc20_15.1, %MakeAdapter.call
// CHECK:STDOUT:   %Destroy.facet.loc20_15.1: %Destroy.type = facet_value constants.%Adapter, (constants.%custom_witness.b80ef6.4, constants.%DestroyImpl.impl_witness.7cf) [concrete = constants.%Destroy.facet.534]
// CHECK:STDOUT:   %.loc20_15.3: %Destroy.type = converted constants.%Adapter, %Destroy.facet.loc20_15.1 [concrete = constants.%Destroy.facet.534]
// CHECK:STDOUT:   %Destroy.facet.loc20_15.2: %Destroy.type = facet_value constants.%Adapter, (constants.%custom_witness.b80ef6.4, constants.%DestroyImpl.impl_witness.7cf) [concrete = constants.%Destroy.facet.534]
// CHECK:STDOUT:   %.loc20_15.4: %Destroy.type = converted constants.%Adapter, %Destroy.facet.loc20_15.2 [concrete = constants.%Destroy.facet.534]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%.loc20_15.2)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc13_11.1(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc13_11.2(%self.param: ref %HasTrivialDestruction) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc13_11.3(%self.param: ref %tuple.type.56a) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1]
// CHECK:STDOUT:   %tuple.elem1: ref %HasTrivialDestruction = tuple_access %self.param, element1
// CHECK:STDOUT:   %Destroy.facet.loc13_11.1: %Destroy.type = facet_value constants.%HasTrivialDestruction, (constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.4ca) [concrete = constants.%Destroy.facet.b89]
// CHECK:STDOUT:   %.loc13_11.1: %Destroy.type = converted constants.%HasTrivialDestruction, %Destroy.facet.loc13_11.1 [concrete = constants.%Destroy.facet.b89]
// CHECK:STDOUT:   %Destroy.facet.loc13_11.2: %Destroy.type = facet_value constants.%HasTrivialDestruction, (constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.4ca) [concrete = constants.%Destroy.facet.b89]
// CHECK:STDOUT:   %.loc13_11.2: %Destroy.type = converted constants.%HasTrivialDestruction, %Destroy.facet.loc13_11.2 [concrete = constants.%Destroy.facet.b89]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc13_11.1: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%tuple.elem1)
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT:   %tuple.elem0: ref %HasTrivialDestruction = tuple_access %self.param, element0
// CHECK:STDOUT:   %Destroy.facet.loc13_11.3: %Destroy.type = facet_value constants.%HasTrivialDestruction, (constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.4ca) [concrete = constants.%Destroy.facet.b89]
// CHECK:STDOUT:   %.loc13_11.3: %Destroy.type = converted constants.%HasTrivialDestruction, %Destroy.facet.loc13_11.3 [concrete = constants.%Destroy.facet.b89]
// CHECK:STDOUT:   %Destroy.facet.loc13_11.4: %Destroy.type = facet_value constants.%HasTrivialDestruction, (constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.4ca) [concrete = constants.%Destroy.facet.b89]
// CHECK:STDOUT:   %.loc13_11.4: %Destroy.type = converted constants.%HasTrivialDestruction, %Destroy.facet.loc13_11.4 [concrete = constants.%Destroy.facet.b89]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc13_11.2: init %empty_tuple.type = call %SelfDestruct.specific_fn.2(%tuple.elem0)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc13_11.4(%self.param: ref %Adapter) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: --- trivial_pointer.carbon = $trivial_pointer
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %HasNonTrivialDestruction: type = class_type @HasNonTrivialDestruction [concrete]
// CHECK:STDOUT:   %ptr: type = ptr_type %HasNonTrivialDestruction [concrete]
// CHECK:STDOUT:   %pattern_type.feb: type = pattern_type %ptr [concrete]
// CHECK:STDOUT:   %Make.type: type = fn_type @Make [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Make: %Make.type = struct_value () [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.687: type = fn_type @SubobjectDestroy.WithSelf.Op.loc10 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f: %SubobjectDestroy.WithSelf.Op.type.687 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.1d0: %SubobjectDestroy.type = facet_value %ptr, (%custom_witness.b80) [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %T.09f: %SubobjectDestroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.type.dea: type = fn_type @T.as_type.as.DestroyImpl.impl.Op, @T.as_type.as.DestroyImpl.impl(%T.09f) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.6e5: %T.as_type.as.DestroyImpl.impl.Op.type.dea = struct_value () [symbolic]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.ed5: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.1d0) [concrete]
// CHECK:STDOUT:   %Destroy.facet: %Destroy.type = facet_value %ptr, (%custom_witness.b80, %DestroyImpl.impl_witness.ed5) [concrete]
// CHECK:STDOUT:   %x.patt: %pattern_type.feb = ref_binding_pattern x [concrete]
// CHECK:STDOUT:   %x.var_patt: %pattern_type.feb = var_pattern %x.patt [concrete]
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
// CHECK:STDOUT:   %Pointer.ref.loc10: type = name_ref Pointer, file.%Pointer [concrete = constants.%ptr]
// CHECK:STDOUT:   %Core.ref.loc10: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:   %SubobjectDestroy.ref: type = name_ref SubobjectDestroy, imports.%Core.SubobjectDestroy [concrete = constants.%SubobjectDestroy.type]
// CHECK:STDOUT:   %SubobjectDestroy.facet: %SubobjectDestroy.type = facet_value %Pointer.ref.loc10, (constants.%custom_witness.b80) [concrete = constants.%SubobjectDestroy.facet.1d0]
// CHECK:STDOUT:   %.loc10: %SubobjectDestroy.type = converted %Pointer.ref.loc10, %SubobjectDestroy.facet [concrete = constants.%SubobjectDestroy.facet.1d0]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %x.var: ref %ptr = var_storage %x.var_patt
// CHECK:STDOUT:   %Make.ref: %Make.type = name_ref Make, file.%Make.decl [concrete = constants.%Make]
// CHECK:STDOUT:   %Make.call: init %ptr = call %Make.ref()
// CHECK:STDOUT:   assign %x.var, %Make.call
// CHECK:STDOUT:   %Pointer.ref.loc17: type = name_ref Pointer, file.%Pointer [concrete = constants.%ptr]
// CHECK:STDOUT:   %x: ref %ptr = wrapper_binding x, %x.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %pattern_type.feb = ref_binding_pattern x [concrete = constants.%x.patt]
// CHECK:STDOUT:     %x.var_patt: %pattern_type.feb = var_pattern %x.patt [concrete = constants.%x.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.facet.loc17_3.1: %Destroy.type = facet_value constants.%ptr, (constants.%custom_witness.b80, constants.%DestroyImpl.impl_witness.ed5) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc17_3.1: %Destroy.type = converted constants.%ptr, %Destroy.facet.loc17_3.1 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %Destroy.facet.loc17_3.2: %Destroy.type = facet_value constants.%ptr, (constants.%custom_witness.b80, constants.%DestroyImpl.impl_witness.ed5) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc17_3.2: %Destroy.type = converted constants.%ptr, %Destroy.facet.loc17_3.2 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%x.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc10(%self.param: ref %ptr) = "no_op";
// CHECK:STDOUT:
