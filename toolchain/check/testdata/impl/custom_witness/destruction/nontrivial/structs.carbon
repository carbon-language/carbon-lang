// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/destroy.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/impl/custom_witness/destruction/nontrivial/structs.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/impl/custom_witness/destruction/nontrivial/structs.carbon

// --- types.carbon
library "[[@TEST_NAME]]";

class HasTrivialDestruction {}

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
    fn Op(unused ref self) {}
  }
}

// --- nontrivial_all_nontrivial_fields.carbon
library "[[@TEST_NAME]]";

import library "types";

alias Struct = {.a: HasNonTrivialDestruction, .b: AlsoHasNonTrivialDestruction, .c: AlsoHasNonTrivialDestruction, .d: HasNonTrivialDestruction};
fn MakeStruct() -> Struct;

fn Test() {
  //@dump-sem-ir-begin
  Struct as Core.SubobjectDestroy;
  //@dump-sem-ir-end

  Struct as Core.Destroy;

  //@dump-sem-ir-begin
  MakeStruct();
  //@dump-sem-ir-end
}

// --- nontrivial_one_trivial_field.carbon
library "[[@TEST_NAME]]";

import library "types";

alias Struct = {.a: HasNonTrivialDestruction, .b: AlsoHasNonTrivialDestruction, .c: HasTrivialDestruction, .d: HasNonTrivialDestruction};
fn MakeStruct() -> Struct;

fn Test() {
  //@dump-sem-ir-begin
  Struct as Core.SubobjectDestroy;
  //@dump-sem-ir-end

  Struct as Core.Destroy;

  //@dump-sem-ir-begin
  MakeStruct();
  //@dump-sem-ir-end
}

// --- nontrivial_nested.carbon
library "[[@TEST_NAME]]";

import library "types";

alias Struct = {.a: {.b: HasNonTrivialDestruction}, .c: HasNonTrivialDestruction};
fn MakeStruct() -> Struct;

fn Test() {
  //@dump-sem-ir-begin
  Struct as Core.SubobjectDestroy;
  //@dump-sem-ir-end

  Struct as Core.Destroy;

  //@dump-sem-ir-begin
  MakeStruct();
  //@dump-sem-ir-end
}

// CHECK:STDOUT: --- nontrivial_all_nontrivial_fields.carbon = $nontrivial_all_nontrivial_fields
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %HasNonTrivialDestruction: type = class_type @HasNonTrivialDestruction [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %AlsoHasNonTrivialDestruction: type = class_type @AlsoHasNonTrivialDestruction [concrete]
// CHECK:STDOUT:   %struct_type.a.b.c.d.7ad: type = struct_type {.a: %HasNonTrivialDestruction, .b: %AlsoHasNonTrivialDestruction, .c: %AlsoHasNonTrivialDestruction, .d: %HasNonTrivialDestruction} [concrete]
// CHECK:STDOUT:   %MakeStruct.type: type = fn_type @MakeStruct [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %MakeStruct: %MakeStruct.type = struct_value () [concrete]
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
// CHECK:STDOUT:   %DestroyImpl.impl_witness.490: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table.467 [concrete]
// CHECK:STDOUT:   %Destroy.facet.4ef: %Destroy.type = facet_value %HasNonTrivialDestruction, (%custom_witness.b80ef6.2, %DestroyImpl.impl_witness.490) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.c4e: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table.86e [concrete]
// CHECK:STDOUT:   %Destroy.facet.7c1: %Destroy.type = facet_value %AlsoHasNonTrivialDestruction, (%custom_witness.b80ef6.3, %DestroyImpl.impl_witness.c4e) [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.4: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.4), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.9ff: %SubobjectDestroy.type = facet_value %struct_type.a.b.c.d.7ad, (%custom_witness.b80ef6.4) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.739: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table.a5e, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.9ff) [concrete]
// CHECK:STDOUT:   %Destroy.facet.76f: %Destroy.type = facet_value %struct_type.a.b.c.d.7ad, (%custom_witness.b80ef6.4, %DestroyImpl.impl_witness.739) [concrete]
// CHECK:STDOUT:   %HasNonTrivialDestruction.as.DestroyImpl.impl.Op.type: type = fn_type @HasNonTrivialDestruction.as.DestroyImpl.impl.Op [concrete]
// CHECK:STDOUT:   %HasNonTrivialDestruction.as.DestroyImpl.impl.Op: %HasNonTrivialDestruction.as.DestroyImpl.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %AlsoHasNonTrivialDestruction.as.DestroyImpl.impl.Op.type: type = fn_type @AlsoHasNonTrivialDestruction.as.DestroyImpl.impl.Op [concrete]
// CHECK:STDOUT:   %AlsoHasNonTrivialDestruction.as.DestroyImpl.impl.Op: %AlsoHasNonTrivialDestruction.as.DestroyImpl.impl.Op.type = struct_value () [concrete]
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
// CHECK:STDOUT:   %Main.import_ref.546: %HasNonTrivialDestruction.as.DestroyImpl.impl.Op.type = import_ref Main//types, loc10_24, loaded [concrete = constants.%HasNonTrivialDestruction.as.DestroyImpl.impl.Op]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table.467 = impl_witness_table (%Main.import_ref.546), @HasNonTrivialDestruction.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT:   %Main.import_ref.307: %AlsoHasNonTrivialDestruction.as.DestroyImpl.impl.Op.type = import_ref Main//types, loc16_28, loaded [concrete = constants.%AlsoHasNonTrivialDestruction.as.DestroyImpl.impl.Op]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table.86e = impl_witness_table (%Main.import_ref.307), @AlsoHasNonTrivialDestruction.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Test() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Struct.ref.loc10: type = name_ref Struct, file.%Struct [concrete = constants.%struct_type.a.b.c.d.7ad]
// CHECK:STDOUT:   %Core.ref.loc10: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:   %SubobjectDestroy.ref: type = name_ref SubobjectDestroy, imports.%Core.SubobjectDestroy [concrete = constants.%SubobjectDestroy.type]
// CHECK:STDOUT:   %SubobjectDestroy.facet: %SubobjectDestroy.type = facet_value %Struct.ref.loc10, (constants.%custom_witness.b80ef6.4) [concrete = constants.%SubobjectDestroy.facet.9ff]
// CHECK:STDOUT:   %.loc10: %SubobjectDestroy.type = converted %Struct.ref.loc10, %SubobjectDestroy.facet [concrete = constants.%SubobjectDestroy.facet.9ff]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %MakeStruct.ref: %MakeStruct.type = name_ref MakeStruct, file.%MakeStruct.decl [concrete = constants.%MakeStruct]
// CHECK:STDOUT:   %.loc16_14.1: ref %struct_type.a.b.c.d.7ad = temporary_storage
// CHECK:STDOUT:   %MakeStruct.call: init %struct_type.a.b.c.d.7ad to %.loc16_14.1 = call %MakeStruct.ref()
// CHECK:STDOUT:   %.loc16_14.2: ref %struct_type.a.b.c.d.7ad = temporary %.loc16_14.1, %MakeStruct.call
// CHECK:STDOUT:   %Destroy.facet.loc16_14.1: %Destroy.type = facet_value constants.%struct_type.a.b.c.d.7ad, (constants.%custom_witness.b80ef6.4, constants.%DestroyImpl.impl_witness.739) [concrete = constants.%Destroy.facet.76f]
// CHECK:STDOUT:   %.loc16_14.3: %Destroy.type = converted constants.%struct_type.a.b.c.d.7ad, %Destroy.facet.loc16_14.1 [concrete = constants.%Destroy.facet.76f]
// CHECK:STDOUT:   %Destroy.facet.loc16_14.2: %Destroy.type = facet_value constants.%struct_type.a.b.c.d.7ad, (constants.%custom_witness.b80ef6.4, constants.%DestroyImpl.impl_witness.739) [concrete = constants.%Destroy.facet.76f]
// CHECK:STDOUT:   %.loc16_14.4: %Destroy.type = converted constants.%struct_type.a.b.c.d.7ad, %Destroy.facet.loc16_14.2 [concrete = constants.%Destroy.facet.76f]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%.loc16_14.2)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc10_10.1(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc10_10.2(%self.param: ref %HasNonTrivialDestruction) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc10_10.3(%self.param: ref %AlsoHasNonTrivialDestruction) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc10_10.4(%self.param: ref %struct_type.a.b.c.d.7ad) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc10_10.1: ref %HasNonTrivialDestruction = struct_access %self.param, element3
// CHECK:STDOUT:   %Destroy.facet.loc10_10.1: %Destroy.type = facet_value constants.%HasNonTrivialDestruction, (constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.490) [concrete = constants.%Destroy.facet.4ef]
// CHECK:STDOUT:   %.loc10_10.2: %Destroy.type = converted constants.%HasNonTrivialDestruction, %Destroy.facet.loc10_10.1 [concrete = constants.%Destroy.facet.4ef]
// CHECK:STDOUT:   %Destroy.facet.loc10_10.2: %Destroy.type = facet_value constants.%HasNonTrivialDestruction, (constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.490) [concrete = constants.%Destroy.facet.4ef]
// CHECK:STDOUT:   %.loc10_10.3: %Destroy.type = converted constants.%HasNonTrivialDestruction, %Destroy.facet.loc10_10.2 [concrete = constants.%Destroy.facet.4ef]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc10_10.1: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%.loc10_10.1)
// CHECK:STDOUT:   %.loc10_10.4: ref %AlsoHasNonTrivialDestruction = struct_access %self.param, element2
// CHECK:STDOUT:   %Destroy.facet.loc10_10.3: %Destroy.type = facet_value constants.%AlsoHasNonTrivialDestruction, (constants.%custom_witness.b80ef6.3, constants.%DestroyImpl.impl_witness.c4e) [concrete = constants.%Destroy.facet.7c1]
// CHECK:STDOUT:   %.loc10_10.5: %Destroy.type = converted constants.%AlsoHasNonTrivialDestruction, %Destroy.facet.loc10_10.3 [concrete = constants.%Destroy.facet.7c1]
// CHECK:STDOUT:   %Destroy.facet.loc10_10.4: %Destroy.type = facet_value constants.%AlsoHasNonTrivialDestruction, (constants.%custom_witness.b80ef6.3, constants.%DestroyImpl.impl_witness.c4e) [concrete = constants.%Destroy.facet.7c1]
// CHECK:STDOUT:   %.loc10_10.6: %Destroy.type = converted constants.%AlsoHasNonTrivialDestruction, %Destroy.facet.loc10_10.4 [concrete = constants.%Destroy.facet.7c1]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc10_10.2: init %empty_tuple.type = call %SelfDestruct.specific_fn.2(%.loc10_10.4)
// CHECK:STDOUT:   %.loc10_10.7: ref %AlsoHasNonTrivialDestruction = struct_access %self.param, element1
// CHECK:STDOUT:   %Destroy.facet.loc10_10.5: %Destroy.type = facet_value constants.%AlsoHasNonTrivialDestruction, (constants.%custom_witness.b80ef6.3, constants.%DestroyImpl.impl_witness.c4e) [concrete = constants.%Destroy.facet.7c1]
// CHECK:STDOUT:   %.loc10_10.8: %Destroy.type = converted constants.%AlsoHasNonTrivialDestruction, %Destroy.facet.loc10_10.5 [concrete = constants.%Destroy.facet.7c1]
// CHECK:STDOUT:   %Destroy.facet.loc10_10.6: %Destroy.type = facet_value constants.%AlsoHasNonTrivialDestruction, (constants.%custom_witness.b80ef6.3, constants.%DestroyImpl.impl_witness.c4e) [concrete = constants.%Destroy.facet.7c1]
// CHECK:STDOUT:   %.loc10_10.9: %Destroy.type = converted constants.%AlsoHasNonTrivialDestruction, %Destroy.facet.loc10_10.6 [concrete = constants.%Destroy.facet.7c1]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc10_10.3: init %empty_tuple.type = call %SelfDestruct.specific_fn.3(%.loc10_10.7)
// CHECK:STDOUT:   %.loc10_10.10: ref %HasNonTrivialDestruction = struct_access %self.param, element0
// CHECK:STDOUT:   %Destroy.facet.loc10_10.7: %Destroy.type = facet_value constants.%HasNonTrivialDestruction, (constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.490) [concrete = constants.%Destroy.facet.4ef]
// CHECK:STDOUT:   %.loc10_10.11: %Destroy.type = converted constants.%HasNonTrivialDestruction, %Destroy.facet.loc10_10.7 [concrete = constants.%Destroy.facet.4ef]
// CHECK:STDOUT:   %Destroy.facet.loc10_10.8: %Destroy.type = facet_value constants.%HasNonTrivialDestruction, (constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.490) [concrete = constants.%Destroy.facet.4ef]
// CHECK:STDOUT:   %.loc10_10.12: %Destroy.type = converted constants.%HasNonTrivialDestruction, %Destroy.facet.loc10_10.8 [concrete = constants.%Destroy.facet.4ef]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc10_10.4: init %empty_tuple.type = call %SelfDestruct.specific_fn.4(%.loc10_10.10)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- nontrivial_one_trivial_field.carbon = $nontrivial_one_trivial_field
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %HasNonTrivialDestruction: type = class_type @HasNonTrivialDestruction [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %AlsoHasNonTrivialDestruction: type = class_type @AlsoHasNonTrivialDestruction [concrete]
// CHECK:STDOUT:   %HasTrivialDestruction: type = class_type @HasTrivialDestruction [concrete]
// CHECK:STDOUT:   %struct_type.a.b.c.d.23f: type = struct_type {.a: %HasNonTrivialDestruction, .b: %AlsoHasNonTrivialDestruction, .c: %HasTrivialDestruction, .d: %HasNonTrivialDestruction} [concrete]
// CHECK:STDOUT:   %MakeStruct.type: type = fn_type @MakeStruct [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %MakeStruct: %MakeStruct.type = struct_value () [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.2: type = fn_type @SubobjectDestroy.WithSelf.Op.loc10_10.2 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.2: %SubobjectDestroy.WithSelf.Op.type.6870fd.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.2: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.2), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.3: type = fn_type @SubobjectDestroy.WithSelf.Op.loc10_10.3 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.3: %SubobjectDestroy.WithSelf.Op.type.6870fd.3 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.3: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.3), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.4: type = fn_type @SubobjectDestroy.WithSelf.Op.loc10_10.4 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.4: %SubobjectDestroy.WithSelf.Op.type.6870fd.4 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.4: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.4), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.5: type = fn_type @SubobjectDestroy.WithSelf.Op.loc10_10.5 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.5: %SubobjectDestroy.WithSelf.Op.type.6870fd.5 = struct_value () [concrete]
// CHECK:STDOUT:   %SelfDestruct.type: type = fn_type @SelfDestruct [concrete]
// CHECK:STDOUT:   %SelfDestruct: %SelfDestruct.type = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %T.09f: %SubobjectDestroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.type.dea: type = fn_type @T.as_type.as.DestroyImpl.impl.Op, @T.as_type.as.DestroyImpl.impl(%T.09f) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.6e5: %T.as_type.as.DestroyImpl.impl.Op.type.dea = struct_value () [symbolic]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.490: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table.467 [concrete]
// CHECK:STDOUT:   %Destroy.facet.4ef: %Destroy.type = facet_value %HasNonTrivialDestruction, (%custom_witness.b80ef6.2, %DestroyImpl.impl_witness.490) [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.5bc: %SubobjectDestroy.type = facet_value %HasTrivialDestruction, (%custom_witness.b80ef6.4) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.4ca: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table.a5e, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.5bc) [concrete]
// CHECK:STDOUT:   %Destroy.facet.b89: %Destroy.type = facet_value %HasTrivialDestruction, (%custom_witness.b80ef6.4, %DestroyImpl.impl_witness.4ca) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.c4e: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table.86e [concrete]
// CHECK:STDOUT:   %Destroy.facet.7c1: %Destroy.type = facet_value %AlsoHasNonTrivialDestruction, (%custom_witness.b80ef6.3, %DestroyImpl.impl_witness.c4e) [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.5: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.5), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.baf: %SubobjectDestroy.type = facet_value %struct_type.a.b.c.d.23f, (%custom_witness.b80ef6.5) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.0c4: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table.a5e, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.baf) [concrete]
// CHECK:STDOUT:   %Destroy.facet.ee7: %Destroy.type = facet_value %struct_type.a.b.c.d.23f, (%custom_witness.b80ef6.5, %DestroyImpl.impl_witness.0c4) [concrete]
// CHECK:STDOUT:   %HasNonTrivialDestruction.as.DestroyImpl.impl.Op.type: type = fn_type @HasNonTrivialDestruction.as.DestroyImpl.impl.Op [concrete]
// CHECK:STDOUT:   %HasNonTrivialDestruction.as.DestroyImpl.impl.Op: %HasNonTrivialDestruction.as.DestroyImpl.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %AlsoHasNonTrivialDestruction.as.DestroyImpl.impl.Op.type: type = fn_type @AlsoHasNonTrivialDestruction.as.DestroyImpl.impl.Op [concrete]
// CHECK:STDOUT:   %AlsoHasNonTrivialDestruction.as.DestroyImpl.impl.Op: %AlsoHasNonTrivialDestruction.as.DestroyImpl.impl.Op.type = struct_value () [concrete]
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
// CHECK:STDOUT:   %Main.import_ref.546: %HasNonTrivialDestruction.as.DestroyImpl.impl.Op.type = import_ref Main//types, loc10_24, loaded [concrete = constants.%HasNonTrivialDestruction.as.DestroyImpl.impl.Op]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table.467 = impl_witness_table (%Main.import_ref.546), @HasNonTrivialDestruction.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT:   %Main.import_ref.307: %AlsoHasNonTrivialDestruction.as.DestroyImpl.impl.Op.type = import_ref Main//types, loc16_28, loaded [concrete = constants.%AlsoHasNonTrivialDestruction.as.DestroyImpl.impl.Op]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table.86e = impl_witness_table (%Main.import_ref.307), @AlsoHasNonTrivialDestruction.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Test() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Struct.ref.loc10: type = name_ref Struct, file.%Struct [concrete = constants.%struct_type.a.b.c.d.23f]
// CHECK:STDOUT:   %Core.ref.loc10: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:   %SubobjectDestroy.ref: type = name_ref SubobjectDestroy, imports.%Core.SubobjectDestroy [concrete = constants.%SubobjectDestroy.type]
// CHECK:STDOUT:   %SubobjectDestroy.facet: %SubobjectDestroy.type = facet_value %Struct.ref.loc10, (constants.%custom_witness.b80ef6.5) [concrete = constants.%SubobjectDestroy.facet.baf]
// CHECK:STDOUT:   %.loc10: %SubobjectDestroy.type = converted %Struct.ref.loc10, %SubobjectDestroy.facet [concrete = constants.%SubobjectDestroy.facet.baf]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %MakeStruct.ref: %MakeStruct.type = name_ref MakeStruct, file.%MakeStruct.decl [concrete = constants.%MakeStruct]
// CHECK:STDOUT:   %.loc16_14.1: ref %struct_type.a.b.c.d.23f = temporary_storage
// CHECK:STDOUT:   %MakeStruct.call: init %struct_type.a.b.c.d.23f to %.loc16_14.1 = call %MakeStruct.ref()
// CHECK:STDOUT:   %.loc16_14.2: ref %struct_type.a.b.c.d.23f = temporary %.loc16_14.1, %MakeStruct.call
// CHECK:STDOUT:   %Destroy.facet.loc16_14.1: %Destroy.type = facet_value constants.%struct_type.a.b.c.d.23f, (constants.%custom_witness.b80ef6.5, constants.%DestroyImpl.impl_witness.0c4) [concrete = constants.%Destroy.facet.ee7]
// CHECK:STDOUT:   %.loc16_14.3: %Destroy.type = converted constants.%struct_type.a.b.c.d.23f, %Destroy.facet.loc16_14.1 [concrete = constants.%Destroy.facet.ee7]
// CHECK:STDOUT:   %Destroy.facet.loc16_14.2: %Destroy.type = facet_value constants.%struct_type.a.b.c.d.23f, (constants.%custom_witness.b80ef6.5, constants.%DestroyImpl.impl_witness.0c4) [concrete = constants.%Destroy.facet.ee7]
// CHECK:STDOUT:   %.loc16_14.4: %Destroy.type = converted constants.%struct_type.a.b.c.d.23f, %Destroy.facet.loc16_14.2 [concrete = constants.%Destroy.facet.ee7]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%.loc16_14.2)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc10_10.1(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc10_10.2(%self.param: ref %HasNonTrivialDestruction) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc10_10.3(%self.param: ref %AlsoHasNonTrivialDestruction) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc10_10.4(%self.param: ref %HasTrivialDestruction) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc10_10.5(%self.param: ref %struct_type.a.b.c.d.23f) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc10_10.1: ref %HasNonTrivialDestruction = struct_access %self.param, element3
// CHECK:STDOUT:   %Destroy.facet.loc10_10.1: %Destroy.type = facet_value constants.%HasNonTrivialDestruction, (constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.490) [concrete = constants.%Destroy.facet.4ef]
// CHECK:STDOUT:   %.loc10_10.2: %Destroy.type = converted constants.%HasNonTrivialDestruction, %Destroy.facet.loc10_10.1 [concrete = constants.%Destroy.facet.4ef]
// CHECK:STDOUT:   %Destroy.facet.loc10_10.2: %Destroy.type = facet_value constants.%HasNonTrivialDestruction, (constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.490) [concrete = constants.%Destroy.facet.4ef]
// CHECK:STDOUT:   %.loc10_10.3: %Destroy.type = converted constants.%HasNonTrivialDestruction, %Destroy.facet.loc10_10.2 [concrete = constants.%Destroy.facet.4ef]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc10_10.1: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%.loc10_10.1)
// CHECK:STDOUT:   %.loc10_10.4: ref %HasTrivialDestruction = struct_access %self.param, element2
// CHECK:STDOUT:   %Destroy.facet.loc10_10.3: %Destroy.type = facet_value constants.%HasTrivialDestruction, (constants.%custom_witness.b80ef6.4, constants.%DestroyImpl.impl_witness.4ca) [concrete = constants.%Destroy.facet.b89]
// CHECK:STDOUT:   %.loc10_10.5: %Destroy.type = converted constants.%HasTrivialDestruction, %Destroy.facet.loc10_10.3 [concrete = constants.%Destroy.facet.b89]
// CHECK:STDOUT:   %Destroy.facet.loc10_10.4: %Destroy.type = facet_value constants.%HasTrivialDestruction, (constants.%custom_witness.b80ef6.4, constants.%DestroyImpl.impl_witness.4ca) [concrete = constants.%Destroy.facet.b89]
// CHECK:STDOUT:   %.loc10_10.6: %Destroy.type = converted constants.%HasTrivialDestruction, %Destroy.facet.loc10_10.4 [concrete = constants.%Destroy.facet.b89]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc10_10.2: init %empty_tuple.type = call %SelfDestruct.specific_fn.2(%.loc10_10.4)
// CHECK:STDOUT:   %.loc10_10.7: ref %AlsoHasNonTrivialDestruction = struct_access %self.param, element1
// CHECK:STDOUT:   %Destroy.facet.loc10_10.5: %Destroy.type = facet_value constants.%AlsoHasNonTrivialDestruction, (constants.%custom_witness.b80ef6.3, constants.%DestroyImpl.impl_witness.c4e) [concrete = constants.%Destroy.facet.7c1]
// CHECK:STDOUT:   %.loc10_10.8: %Destroy.type = converted constants.%AlsoHasNonTrivialDestruction, %Destroy.facet.loc10_10.5 [concrete = constants.%Destroy.facet.7c1]
// CHECK:STDOUT:   %Destroy.facet.loc10_10.6: %Destroy.type = facet_value constants.%AlsoHasNonTrivialDestruction, (constants.%custom_witness.b80ef6.3, constants.%DestroyImpl.impl_witness.c4e) [concrete = constants.%Destroy.facet.7c1]
// CHECK:STDOUT:   %.loc10_10.9: %Destroy.type = converted constants.%AlsoHasNonTrivialDestruction, %Destroy.facet.loc10_10.6 [concrete = constants.%Destroy.facet.7c1]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc10_10.3: init %empty_tuple.type = call %SelfDestruct.specific_fn.3(%.loc10_10.7)
// CHECK:STDOUT:   %.loc10_10.10: ref %HasNonTrivialDestruction = struct_access %self.param, element0
// CHECK:STDOUT:   %Destroy.facet.loc10_10.7: %Destroy.type = facet_value constants.%HasNonTrivialDestruction, (constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.490) [concrete = constants.%Destroy.facet.4ef]
// CHECK:STDOUT:   %.loc10_10.11: %Destroy.type = converted constants.%HasNonTrivialDestruction, %Destroy.facet.loc10_10.7 [concrete = constants.%Destroy.facet.4ef]
// CHECK:STDOUT:   %Destroy.facet.loc10_10.8: %Destroy.type = facet_value constants.%HasNonTrivialDestruction, (constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.490) [concrete = constants.%Destroy.facet.4ef]
// CHECK:STDOUT:   %.loc10_10.12: %Destroy.type = converted constants.%HasNonTrivialDestruction, %Destroy.facet.loc10_10.8 [concrete = constants.%Destroy.facet.4ef]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc10_10.4: init %empty_tuple.type = call %SelfDestruct.specific_fn.4(%.loc10_10.10)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- nontrivial_nested.carbon = $nontrivial_nested
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %HasNonTrivialDestruction: type = class_type @HasNonTrivialDestruction [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %struct_type.b.3f3: type = struct_type {.b: %HasNonTrivialDestruction} [concrete]
// CHECK:STDOUT:   %struct_type.a.c.df2: type = struct_type {.a: %struct_type.b.3f3, .c: %HasNonTrivialDestruction} [concrete]
// CHECK:STDOUT:   %MakeStruct.type: type = fn_type @MakeStruct [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %MakeStruct: %MakeStruct.type = struct_value () [concrete]
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
// CHECK:STDOUT:   %DestroyImpl.impl_witness.490: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table.467 [concrete]
// CHECK:STDOUT:   %Destroy.facet.4ef: %Destroy.type = facet_value %HasNonTrivialDestruction, (%custom_witness.b80ef6.2, %DestroyImpl.impl_witness.490) [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.3: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.3), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.4: type = fn_type @SubobjectDestroy.WithSelf.Op.loc10_10.4 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.4: %SubobjectDestroy.WithSelf.Op.type.6870fd.4 = struct_value () [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.ed5: %SubobjectDestroy.type = facet_value %struct_type.b.3f3, (%custom_witness.b80ef6.3) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.e90: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table.a5e, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.ed5) [concrete]
// CHECK:STDOUT:   %Destroy.facet.894: %Destroy.type = facet_value %struct_type.b.3f3, (%custom_witness.b80ef6.3, %DestroyImpl.impl_witness.e90) [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.4: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.4), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.3b5: %SubobjectDestroy.type = facet_value %struct_type.a.c.df2, (%custom_witness.b80ef6.4) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.99e: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table.a5e, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.3b5) [concrete]
// CHECK:STDOUT:   %Destroy.facet.05d: %Destroy.type = facet_value %struct_type.a.c.df2, (%custom_witness.b80ef6.4, %DestroyImpl.impl_witness.99e) [concrete]
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
// CHECK:STDOUT:   %Main.import_ref.546: %HasNonTrivialDestruction.as.DestroyImpl.impl.Op.type = import_ref Main//types, loc10_24, loaded [concrete = constants.%HasNonTrivialDestruction.as.DestroyImpl.impl.Op]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table.467 = impl_witness_table (%Main.import_ref.546), @HasNonTrivialDestruction.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Test() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Struct.ref.loc10: type = name_ref Struct, file.%Struct [concrete = constants.%struct_type.a.c.df2]
// CHECK:STDOUT:   %Core.ref.loc10: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:   %SubobjectDestroy.ref: type = name_ref SubobjectDestroy, imports.%Core.SubobjectDestroy [concrete = constants.%SubobjectDestroy.type]
// CHECK:STDOUT:   %SubobjectDestroy.facet: %SubobjectDestroy.type = facet_value %Struct.ref.loc10, (constants.%custom_witness.b80ef6.4) [concrete = constants.%SubobjectDestroy.facet.3b5]
// CHECK:STDOUT:   %.loc10: %SubobjectDestroy.type = converted %Struct.ref.loc10, %SubobjectDestroy.facet [concrete = constants.%SubobjectDestroy.facet.3b5]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %MakeStruct.ref: %MakeStruct.type = name_ref MakeStruct, file.%MakeStruct.decl [concrete = constants.%MakeStruct]
// CHECK:STDOUT:   %.loc16_14.1: ref %struct_type.a.c.df2 = temporary_storage
// CHECK:STDOUT:   %MakeStruct.call: init %struct_type.a.c.df2 to %.loc16_14.1 = call %MakeStruct.ref()
// CHECK:STDOUT:   %.loc16_14.2: ref %struct_type.a.c.df2 = temporary %.loc16_14.1, %MakeStruct.call
// CHECK:STDOUT:   %Destroy.facet.loc16_14.1: %Destroy.type = facet_value constants.%struct_type.a.c.df2, (constants.%custom_witness.b80ef6.4, constants.%DestroyImpl.impl_witness.99e) [concrete = constants.%Destroy.facet.05d]
// CHECK:STDOUT:   %.loc16_14.3: %Destroy.type = converted constants.%struct_type.a.c.df2, %Destroy.facet.loc16_14.1 [concrete = constants.%Destroy.facet.05d]
// CHECK:STDOUT:   %Destroy.facet.loc16_14.2: %Destroy.type = facet_value constants.%struct_type.a.c.df2, (constants.%custom_witness.b80ef6.4, constants.%DestroyImpl.impl_witness.99e) [concrete = constants.%Destroy.facet.05d]
// CHECK:STDOUT:   %.loc16_14.4: %Destroy.type = converted constants.%struct_type.a.c.df2, %Destroy.facet.loc16_14.2 [concrete = constants.%Destroy.facet.05d]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%.loc16_14.2)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc10_10.1(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc10_10.2(%self.param: ref %HasNonTrivialDestruction) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc10_10.3(%self.param: ref %struct_type.b.3f3) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc10_10.1: ref %HasNonTrivialDestruction = struct_access %self.param, element0
// CHECK:STDOUT:   %Destroy.facet.loc10_10.1: %Destroy.type = facet_value constants.%HasNonTrivialDestruction, (constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.490) [concrete = constants.%Destroy.facet.4ef]
// CHECK:STDOUT:   %.loc10_10.2: %Destroy.type = converted constants.%HasNonTrivialDestruction, %Destroy.facet.loc10_10.1 [concrete = constants.%Destroy.facet.4ef]
// CHECK:STDOUT:   %Destroy.facet.loc10_10.2: %Destroy.type = facet_value constants.%HasNonTrivialDestruction, (constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.490) [concrete = constants.%Destroy.facet.4ef]
// CHECK:STDOUT:   %.loc10_10.3: %Destroy.type = converted constants.%HasNonTrivialDestruction, %Destroy.facet.loc10_10.2 [concrete = constants.%Destroy.facet.4ef]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%.loc10_10.1)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc10_10.4(%self.param: ref %struct_type.a.c.df2) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc10_10.1: ref %HasNonTrivialDestruction = struct_access %self.param, element1
// CHECK:STDOUT:   %Destroy.facet.loc10_10.1: %Destroy.type = facet_value constants.%HasNonTrivialDestruction, (constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.490) [concrete = constants.%Destroy.facet.4ef]
// CHECK:STDOUT:   %.loc10_10.2: %Destroy.type = converted constants.%HasNonTrivialDestruction, %Destroy.facet.loc10_10.1 [concrete = constants.%Destroy.facet.4ef]
// CHECK:STDOUT:   %Destroy.facet.loc10_10.2: %Destroy.type = facet_value constants.%HasNonTrivialDestruction, (constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.490) [concrete = constants.%Destroy.facet.4ef]
// CHECK:STDOUT:   %.loc10_10.3: %Destroy.type = converted constants.%HasNonTrivialDestruction, %Destroy.facet.loc10_10.2 [concrete = constants.%Destroy.facet.4ef]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc10_10.1: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%.loc10_10.1)
// CHECK:STDOUT:   %.loc10_10.4: ref %struct_type.b.3f3 = struct_access %self.param, element0
// CHECK:STDOUT:   %Destroy.facet.loc10_10.3: %Destroy.type = facet_value constants.%struct_type.b.3f3, (constants.%custom_witness.b80ef6.3, constants.%DestroyImpl.impl_witness.e90) [concrete = constants.%Destroy.facet.894]
// CHECK:STDOUT:   %.loc10_10.5: %Destroy.type = converted constants.%struct_type.b.3f3, %Destroy.facet.loc10_10.3 [concrete = constants.%Destroy.facet.894]
// CHECK:STDOUT:   %Destroy.facet.loc10_10.4: %Destroy.type = facet_value constants.%struct_type.b.3f3, (constants.%custom_witness.b80ef6.3, constants.%DestroyImpl.impl_witness.e90) [concrete = constants.%Destroy.facet.894]
// CHECK:STDOUT:   %.loc10_10.6: %Destroy.type = converted constants.%struct_type.b.3f3, %Destroy.facet.loc10_10.4 [concrete = constants.%Destroy.facet.894]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc10_10.2: init %empty_tuple.type = call %SelfDestruct.specific_fn.2(%.loc10_10.4)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
