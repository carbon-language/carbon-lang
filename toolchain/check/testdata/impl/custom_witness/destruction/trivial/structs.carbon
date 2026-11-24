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

// --- trivial_empty.carbon
library "[[@TEST_NAME]]";

alias Struct = {};
fn Make() -> Struct;
fn Take(s: Struct);

fn Test() {
  //@dump-sem-ir-begin
  Struct as Core.SubobjectDestroy;
  //@dump-sem-ir-end

  Struct as Core.Destroy;
  Struct as Core.TrivialDestroy;

  Take(
  //@dump-sem-ir-begin
    Make()
  //@dump-sem-ir-end
  );
}

// --- trivial_nonempty.carbon
library "[[@TEST_NAME]]";

import library "types";

alias Struct = {.a: A, .b: B};
fn MakeStruct() -> Struct;

fn Test() {
  //@dump-sem-ir-begin
  Struct as Core.SubobjectDestroy;
  //@dump-sem-ir-end

  Struct as Core.Destroy;
  Struct as Core.TrivialDestroy;

  //@dump-sem-ir-begin
  MakeStruct();
  //@dump-sem-ir-end
}

// --- trivial_nested_struct_with_trivial_members.carbon
library "[[@TEST_NAME]]";

import library "types";

alias Struct = {.a: {.b: B}, .c: B};
fn MakeStruct() -> Struct;

fn Test() {
  //@dump-sem-ir-begin
  Struct as Core.SubobjectDestroy;
  //@dump-sem-ir-end

  Struct as Core.Destroy;
  Struct as Core.TrivialDestroy;

  //@dump-sem-ir-begin
  MakeStruct();
  //@dump-sem-ir-end
}

// --- todo_fail_nontrivial_struct_nontrivial_field_first.carbon
library "[[@TEST_NAME]]";

import library "types";

alias Struct = {.a: HasNonTrivialDestruction, .b: HasTrivialDestruction};

fn Test() {
  // CHECK:STDERR:        ╭─┤ todo_fail_nontrivial_struct_nontrivial_field_first.carbon:[[@LINE+8]]:3
  // CHECK:STDERR: ->     │   Struct as Core.TrivialDestroy;
  // CHECK:STDERR:        ·   ┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  // CHECK:STDERR:        ·   ┗━━┫ error: cannot convert type `Struct` into type implementing
  // CHECK:STDERR:        ·      ┃        `Core.TrivialDestroy`
  // CHECK:STDERR:        ·      ┃        [ConversionFailureTypeToFacet]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  Struct as Core.TrivialDestroy;
}

// --- todo_fail_nontrivial_struct_nontrivial_field_second.carbon
library "[[@TEST_NAME]]";

import library "types";

alias Struct = {.a: HasTrivialDestruction, .b: HasNonTrivialDestruction};

fn Test() {
  // CHECK:STDERR:        ╭─┤ todo_fail_nontrivial_struct_nontrivial_field_second.carbon:[[@LINE+8]]:3
  // CHECK:STDERR: ->     │   Struct as Core.TrivialDestroy;
  // CHECK:STDERR:        ·   ┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  // CHECK:STDERR:        ·   ┗━━┫ error: cannot convert type `Struct` into type implementing
  // CHECK:STDERR:        ·      ┃        `Core.TrivialDestroy`
  // CHECK:STDERR:        ·      ┃        [ConversionFailureTypeToFacet]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  Struct as Core.TrivialDestroy;
}

// --- todo_fail_nontrivial_struct_nontrivial_field_nested.carbon
library "[[@TEST_NAME]]";

import library "types";

alias Struct = {.a: {.b: HasNonTrivialDestruction}};

fn Test() {
  // CHECK:STDERR:        ╭─┤ todo_fail_nontrivial_struct_nontrivial_field_nested.carbon:[[@LINE+8]]:3
  // CHECK:STDERR: ->     │   Struct as Core.TrivialDestroy;
  // CHECK:STDERR:        ·   ┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  // CHECK:STDERR:        ·   ┗━━┫ error: cannot convert type `Struct` into type implementing
  // CHECK:STDERR:        ·      ┃        `Core.TrivialDestroy`
  // CHECK:STDERR:        ·      ┃        [ConversionFailureTypeToFacet]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  Struct as Core.TrivialDestroy;
}

// CHECK:STDOUT: --- trivial_empty.carbon = $trivial_empty
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Make.type: type = fn_type @Make [concrete]
// CHECK:STDOUT:   %Make: %Make.type = struct_value () [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %DestroyImpl.type: type = facet_type <@DestroyImpl> [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.687: type = fn_type @SubobjectDestroy.WithSelf.Op.loc9 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f: %SubobjectDestroy.WithSelf.Op.type.687 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.6b7: %SubobjectDestroy.type = facet_value %empty_struct_type, (%custom_witness.b80) [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %T.09f: %SubobjectDestroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.type.dea: type = fn_type @T.as_type.as.DestroyImpl.impl.Op, @T.as_type.as.DestroyImpl.impl(%T.09f) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.6e5: %T.as_type.as.DestroyImpl.impl.Op.type.dea = struct_value () [symbolic]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.011: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.6b7) [concrete]
// CHECK:STDOUT:   %Destroy.facet: %Destroy.type = facet_value %empty_struct_type, (%custom_witness.b80, %DestroyImpl.impl_witness.011) [concrete]
// CHECK:STDOUT:   %TrivialDestroy.type: type = facet_type <@TrivialDestroy> [concrete]
// CHECK:STDOUT:   %SelfDestruct.type: type = fn_type @SelfDestruct [concrete]
// CHECK:STDOUT:   %SelfDestruct: %SelfDestruct.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .SubobjectDestroy = %Core.SubobjectDestroy
// CHECK:STDOUT:     .DestroyImpl = %Core.DestroyImpl
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     .TrivialDestroy = %Core.TrivialDestroy
// CHECK:STDOUT:     .SelfDestruct = %Core.SelfDestruct
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.SubobjectDestroy: type = import_ref Core//prelude/parts/destroy, SubobjectDestroy, loaded [concrete = constants.%SubobjectDestroy.type]
// CHECK:STDOUT:   %Core.DestroyImpl: type = import_ref Core//prelude/parts/destroy, DestroyImpl, loaded [concrete = constants.%DestroyImpl.type]
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Core.import_ref.74d: @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op.type (%T.as_type.as.DestroyImpl.impl.Op.type.dea) = import_ref Core//prelude/parts/destroy, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op (constants.%T.as_type.as.DestroyImpl.impl.Op.6e5)]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table = impl_witness_table (%Core.import_ref.74d), @T.as_type.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT:   %Core.TrivialDestroy: type = import_ref Core//prelude/parts/destroy, TrivialDestroy, loaded [concrete = constants.%TrivialDestroy.type]
// CHECK:STDOUT:   %Core.SelfDestruct: %SelfDestruct.type = import_ref Core//prelude/parts/destroy, SelfDestruct, loaded [concrete = constants.%SelfDestruct]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Test() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Struct.ref.loc9: %empty_struct_type = name_ref Struct, file.%Struct [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %Core.ref.loc9: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:   %SubobjectDestroy.ref: type = name_ref SubobjectDestroy, imports.%Core.SubobjectDestroy [concrete = constants.%SubobjectDestroy.type]
// CHECK:STDOUT:   %SubobjectDestroy.facet: %SubobjectDestroy.type = facet_value constants.%empty_struct_type, (constants.%custom_witness.b80) [concrete = constants.%SubobjectDestroy.facet.6b7]
// CHECK:STDOUT:   %.loc9: %SubobjectDestroy.type = converted %Struct.ref.loc9, %SubobjectDestroy.facet [concrete = constants.%SubobjectDestroy.facet.6b7]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %Make.ref: %Make.type = name_ref Make, file.%Make.decl [concrete = constants.%Make]
// CHECK:STDOUT:   %Make.call: init %empty_struct_type = call %Make.ref()
// CHECK:STDOUT:   %.loc17_10.1: ref %empty_struct_type = temporary_storage
// CHECK:STDOUT:   %.loc17_10.2: ref %empty_struct_type = temporary %.loc17_10.1, %Make.call
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc17_10.3: %empty_struct_type = converted %Make.call, %empty_struct [concrete = constants.%empty_struct]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %Destroy.facet.loc17_10.1: %Destroy.type = facet_value constants.%empty_struct_type, (constants.%custom_witness.b80, constants.%DestroyImpl.impl_witness.011) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc17_10.4: %Destroy.type = converted constants.%empty_struct_type, %Destroy.facet.loc17_10.1 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %Destroy.facet.loc17_10.2: %Destroy.type = facet_value constants.%empty_struct_type, (constants.%custom_witness.b80, constants.%DestroyImpl.impl_witness.011) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc17_10.5: %Destroy.type = converted constants.%empty_struct_type, %Destroy.facet.loc17_10.2 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%.loc17_10.2)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc9(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: --- trivial_nonempty.carbon = $trivial_nonempty
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %A: type = class_type @A [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %B: type = class_type @B [concrete]
// CHECK:STDOUT:   %struct_type.a.b.06d: type = struct_type {.a: %A, .b: %B} [concrete]
// CHECK:STDOUT:   %MakeStruct.type: type = fn_type @MakeStruct [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %MakeStruct: %MakeStruct.type = struct_value () [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %DestroyImpl.type: type = facet_type <@DestroyImpl> [concrete]
// CHECK:STDOUT:   %T.09f: %SubobjectDestroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.type.dea: type = fn_type @T.as_type.as.DestroyImpl.impl.Op, @T.as_type.as.DestroyImpl.impl(%T.09f) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.6e5: %T.as_type.as.DestroyImpl.impl.Op.type.dea = struct_value () [symbolic]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.4: type = fn_type @SubobjectDestroy.WithSelf.Op.loc10_10.4 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.4: %SubobjectDestroy.WithSelf.Op.type.6870fd.4 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.4: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.4), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.364: %SubobjectDestroy.type = facet_value %struct_type.a.b.06d, (%custom_witness.b80ef6.4) [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.137: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.364) [concrete]
// CHECK:STDOUT:   %Destroy.facet: %Destroy.type = facet_value %struct_type.a.b.06d, (%custom_witness.b80ef6.4, %DestroyImpl.impl_witness.137) [concrete]
// CHECK:STDOUT:   %TrivialDestroy.type: type = facet_type <@TrivialDestroy> [concrete]
// CHECK:STDOUT:   %SelfDestruct.type: type = fn_type @SelfDestruct [concrete]
// CHECK:STDOUT:   %SelfDestruct: %SelfDestruct.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .SubobjectDestroy = %Core.SubobjectDestroy
// CHECK:STDOUT:     .DestroyImpl = %Core.DestroyImpl
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     .TrivialDestroy = %Core.TrivialDestroy
// CHECK:STDOUT:     .SelfDestruct = %Core.SelfDestruct
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.SubobjectDestroy: type = import_ref Core//prelude/parts/destroy, SubobjectDestroy, loaded [concrete = constants.%SubobjectDestroy.type]
// CHECK:STDOUT:   %Core.DestroyImpl: type = import_ref Core//prelude/parts/destroy, DestroyImpl, loaded [concrete = constants.%DestroyImpl.type]
// CHECK:STDOUT:   %Core.import_ref.74d: @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op.type (%T.as_type.as.DestroyImpl.impl.Op.type.dea) = import_ref Core//prelude/parts/destroy, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op (constants.%T.as_type.as.DestroyImpl.impl.Op.6e5)]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table = impl_witness_table (%Core.import_ref.74d), @T.as_type.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Core.TrivialDestroy: type = import_ref Core//prelude/parts/destroy, TrivialDestroy, loaded [concrete = constants.%TrivialDestroy.type]
// CHECK:STDOUT:   %Core.SelfDestruct: %SelfDestruct.type = import_ref Core//prelude/parts/destroy, SelfDestruct, loaded [concrete = constants.%SelfDestruct]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Test() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Struct.ref.loc10: type = name_ref Struct, file.%Struct [concrete = constants.%struct_type.a.b.06d]
// CHECK:STDOUT:   %Core.ref.loc10: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:   %SubobjectDestroy.ref: type = name_ref SubobjectDestroy, imports.%Core.SubobjectDestroy [concrete = constants.%SubobjectDestroy.type]
// CHECK:STDOUT:   %SubobjectDestroy.facet: %SubobjectDestroy.type = facet_value %Struct.ref.loc10, (constants.%custom_witness.b80ef6.4) [concrete = constants.%SubobjectDestroy.facet.364]
// CHECK:STDOUT:   %.loc10: %SubobjectDestroy.type = converted %Struct.ref.loc10, %SubobjectDestroy.facet [concrete = constants.%SubobjectDestroy.facet.364]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %MakeStruct.ref: %MakeStruct.type = name_ref MakeStruct, file.%MakeStruct.decl [concrete = constants.%MakeStruct]
// CHECK:STDOUT:   %.loc17_14.1: ref %struct_type.a.b.06d = temporary_storage
// CHECK:STDOUT:   %MakeStruct.call: init %struct_type.a.b.06d to %.loc17_14.1 = call %MakeStruct.ref()
// CHECK:STDOUT:   %.loc17_14.2: ref %struct_type.a.b.06d = temporary %.loc17_14.1, %MakeStruct.call
// CHECK:STDOUT:   %Destroy.facet.loc17_14.1: %Destroy.type = facet_value constants.%struct_type.a.b.06d, (constants.%custom_witness.b80ef6.4, constants.%DestroyImpl.impl_witness.137) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc17_14.3: %Destroy.type = converted constants.%struct_type.a.b.06d, %Destroy.facet.loc17_14.1 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %Destroy.facet.loc17_14.2: %Destroy.type = facet_value constants.%struct_type.a.b.06d, (constants.%custom_witness.b80ef6.4, constants.%DestroyImpl.impl_witness.137) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc17_14.4: %Destroy.type = converted constants.%struct_type.a.b.06d, %Destroy.facet.loc17_14.2 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%.loc17_14.2)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc10_10.1(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc10_10.2(%self.param: ref %A) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc10_10.3(%self.param: ref %B) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc10_10.4(%self.param: ref %struct_type.a.b.06d) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: --- trivial_nested_struct_with_trivial_members.carbon = $trivial_nested_struct_with_trivial_members
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %B: type = class_type @B [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %struct_type.b.dce: type = struct_type {.b: %B} [concrete]
// CHECK:STDOUT:   %struct_type.a.c.0dc: type = struct_type {.a: %struct_type.b.dce, .c: %B} [concrete]
// CHECK:STDOUT:   %MakeStruct.type: type = fn_type @MakeStruct [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %MakeStruct: %MakeStruct.type = struct_value () [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %DestroyImpl.type: type = facet_type <@DestroyImpl> [concrete]
// CHECK:STDOUT:   %T.09f: %SubobjectDestroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.type.dea: type = fn_type @T.as_type.as.DestroyImpl.impl.Op, @T.as_type.as.DestroyImpl.impl(%T.09f) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.6e5: %T.as_type.as.DestroyImpl.impl.Op.type.dea = struct_value () [symbolic]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.4: type = fn_type @SubobjectDestroy.WithSelf.Op.loc10_10.4 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.4: %SubobjectDestroy.WithSelf.Op.type.6870fd.4 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.4: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.4), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.570: %SubobjectDestroy.type = facet_value %struct_type.a.c.0dc, (%custom_witness.b80ef6.4) [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.f23: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.570) [concrete]
// CHECK:STDOUT:   %Destroy.facet: %Destroy.type = facet_value %struct_type.a.c.0dc, (%custom_witness.b80ef6.4, %DestroyImpl.impl_witness.f23) [concrete]
// CHECK:STDOUT:   %TrivialDestroy.type: type = facet_type <@TrivialDestroy> [concrete]
// CHECK:STDOUT:   %SelfDestruct.type: type = fn_type @SelfDestruct [concrete]
// CHECK:STDOUT:   %SelfDestruct: %SelfDestruct.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .SubobjectDestroy = %Core.SubobjectDestroy
// CHECK:STDOUT:     .DestroyImpl = %Core.DestroyImpl
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     .TrivialDestroy = %Core.TrivialDestroy
// CHECK:STDOUT:     .SelfDestruct = %Core.SelfDestruct
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.SubobjectDestroy: type = import_ref Core//prelude/parts/destroy, SubobjectDestroy, loaded [concrete = constants.%SubobjectDestroy.type]
// CHECK:STDOUT:   %Core.DestroyImpl: type = import_ref Core//prelude/parts/destroy, DestroyImpl, loaded [concrete = constants.%DestroyImpl.type]
// CHECK:STDOUT:   %Core.import_ref.74d: @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op.type (%T.as_type.as.DestroyImpl.impl.Op.type.dea) = import_ref Core//prelude/parts/destroy, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op (constants.%T.as_type.as.DestroyImpl.impl.Op.6e5)]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table = impl_witness_table (%Core.import_ref.74d), @T.as_type.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Core.TrivialDestroy: type = import_ref Core//prelude/parts/destroy, TrivialDestroy, loaded [concrete = constants.%TrivialDestroy.type]
// CHECK:STDOUT:   %Core.SelfDestruct: %SelfDestruct.type = import_ref Core//prelude/parts/destroy, SelfDestruct, loaded [concrete = constants.%SelfDestruct]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Test() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Struct.ref.loc10: type = name_ref Struct, file.%Struct [concrete = constants.%struct_type.a.c.0dc]
// CHECK:STDOUT:   %Core.ref.loc10: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:   %SubobjectDestroy.ref: type = name_ref SubobjectDestroy, imports.%Core.SubobjectDestroy [concrete = constants.%SubobjectDestroy.type]
// CHECK:STDOUT:   %SubobjectDestroy.facet: %SubobjectDestroy.type = facet_value %Struct.ref.loc10, (constants.%custom_witness.b80ef6.4) [concrete = constants.%SubobjectDestroy.facet.570]
// CHECK:STDOUT:   %.loc10: %SubobjectDestroy.type = converted %Struct.ref.loc10, %SubobjectDestroy.facet [concrete = constants.%SubobjectDestroy.facet.570]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %MakeStruct.ref: %MakeStruct.type = name_ref MakeStruct, file.%MakeStruct.decl [concrete = constants.%MakeStruct]
// CHECK:STDOUT:   %.loc17_14.1: ref %struct_type.a.c.0dc = temporary_storage
// CHECK:STDOUT:   %MakeStruct.call: init %struct_type.a.c.0dc to %.loc17_14.1 = call %MakeStruct.ref()
// CHECK:STDOUT:   %.loc17_14.2: ref %struct_type.a.c.0dc = temporary %.loc17_14.1, %MakeStruct.call
// CHECK:STDOUT:   %Destroy.facet.loc17_14.1: %Destroy.type = facet_value constants.%struct_type.a.c.0dc, (constants.%custom_witness.b80ef6.4, constants.%DestroyImpl.impl_witness.f23) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc17_14.3: %Destroy.type = converted constants.%struct_type.a.c.0dc, %Destroy.facet.loc17_14.1 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %Destroy.facet.loc17_14.2: %Destroy.type = facet_value constants.%struct_type.a.c.0dc, (constants.%custom_witness.b80ef6.4, constants.%DestroyImpl.impl_witness.f23) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc17_14.4: %Destroy.type = converted constants.%struct_type.a.c.0dc, %Destroy.facet.loc17_14.2 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%.loc17_14.2)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc10_10.1(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc10_10.2(%self.param: ref %B) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc10_10.3(%self.param: ref %struct_type.b.dce) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc10_10.4(%self.param: ref %struct_type.a.c.0dc) = "no_op";
// CHECK:STDOUT:
