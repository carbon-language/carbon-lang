// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/destroy.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/impl/custom_witness/destruction/nontrivial/partial.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/impl/custom_witness/destruction/nontrivial/partial.carbon

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

// --- nontrivial_member.carbon
library "[[@TEST_NAME]]";

import library "types";

base class Base {
  var x: HasNonTrivialDestruction;
  var y: AlsoHasNonTrivialDestruction;
}

alias Partial = partial Base;
fn MakePartial() -> Partial;

fn Test() {
  //@dump-sem-ir-begin
  Partial as Core.SubobjectDestroy;
  //@dump-sem-ir-end

  Partial as Core.Destroy;

  //@dump-sem-ir-begin
  MakePartial();
  //@dump-sem-ir-end
}

// --- nontrivial_impls_destroy.carbon
library "[[@TEST_NAME]]";

base class Base {
  impl as Core.DestroyImpl {
    fn Op(unused self) {}
  }
}

alias Partial = partial Base;
fn MakePartial() -> Partial;

fn Test() {
  //@dump-sem-ir-begin
  Partial as Core.SubobjectDestroy;
  //@dump-sem-ir-end

  Partial as Core.Destroy;

  //@dump-sem-ir-begin
  MakePartial();
  //@dump-sem-ir-end
}


// --- nontrivial_abstract.carbon
library "[[@TEST_NAME]]";

import library "types";

alias Partial = partial Abstract;
fn MakePartial() -> Partial;

fn Test() {
  //@dump-sem-ir-begin
  Partial as Core.SubobjectDestroy;
  //@dump-sem-ir-end

  Partial as Core.Destroy;

  //@dump-sem-ir-begin
  MakePartial();
  //@dump-sem-ir-end
}

// --- nontrivial_derived.carbon
library "[[@TEST_NAME]]";

import library "types";

base class Base {
  var x: HasNonTrivialDestruction;
}

base class Derived {
  extend base: Base;
  var y: AlsoHasNonTrivialDestruction;
}

alias Partial = partial Derived;
fn MakePartial() -> Partial;

fn Test() {
  //@dump-sem-ir-begin
  Partial as Core.SubobjectDestroy;
  //@dump-sem-ir-end

  Partial as Core.Destroy;

  //@dump-sem-ir-begin
  MakePartial();
  //@dump-sem-ir-end
}

// CHECK:STDOUT: --- nontrivial_member.carbon = $nontrivial_member
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
// CHECK:STDOUT:   %MakePartial.type: type = fn_type @MakePartial [concrete]
// CHECK:STDOUT:   %MakePartial: %MakePartial.type = struct_value () [concrete]
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
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %MakePartial.ref: %MakePartial.type = name_ref MakePartial, file.%MakePartial.decl [concrete = constants.%MakePartial]
// CHECK:STDOUT:   %.loc21_15.1: ref %.edf = temporary_storage
// CHECK:STDOUT:   %MakePartial.call: init %.edf to %.loc21_15.1 = call %MakePartial.ref()
// CHECK:STDOUT:   %.loc21_15.2: ref %.edf = temporary %.loc21_15.1, %MakePartial.call
// CHECK:STDOUT:   %Destroy.facet.loc21_15.1: %Destroy.type = facet_value constants.%.edf, (constants.%custom_witness.b80ef6.5, constants.%DestroyImpl.impl_witness.1d2) [concrete = constants.%Destroy.facet.96b]
// CHECK:STDOUT:   %.loc21_15.3: %Destroy.type = converted constants.%.edf, %Destroy.facet.loc21_15.1 [concrete = constants.%Destroy.facet.96b]
// CHECK:STDOUT:   %Destroy.facet.loc21_15.2: %Destroy.type = facet_value constants.%.edf, (constants.%custom_witness.b80ef6.5, constants.%DestroyImpl.impl_witness.1d2) [concrete = constants.%Destroy.facet.96b]
// CHECK:STDOUT:   %.loc21_15.4: %Destroy.type = converted constants.%.edf, %Destroy.facet.loc21_15.2 [concrete = constants.%Destroy.facet.96b]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%.loc21_15.2)
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
// CHECK:STDOUT: --- nontrivial_impls_destroy.carbon = $nontrivial_impls_destroy
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %Base: type = class_type @Base [concrete]
// CHECK:STDOUT:   %DestroyImpl.type: type = facet_type <@DestroyImpl> [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %.edf: type = partial_type %Base [concrete]
// CHECK:STDOUT:   %MakePartial.type: type = fn_type @MakePartial [concrete]
// CHECK:STDOUT:   %MakePartial: %MakePartial.type = struct_value () [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
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
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %MakePartial.ref: %MakePartial.type = name_ref MakePartial, file.%MakePartial.decl [concrete = constants.%MakePartial]
// CHECK:STDOUT:   %.loc20_15.1: ref %.edf = temporary_storage
// CHECK:STDOUT:   %MakePartial.call: init %.edf to %.loc20_15.1 = call %MakePartial.ref()
// CHECK:STDOUT:   %.loc20_15.2: ref %.edf = temporary %.loc20_15.1, %MakePartial.call
// CHECK:STDOUT:   %Destroy.facet.loc20_15.1: %Destroy.type = facet_value constants.%.edf, (constants.%DestroyImpl.impl_witness.1d2, constants.%custom_witness.b80ef6.2) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc20_15.3: %Destroy.type = converted constants.%.edf, %Destroy.facet.loc20_15.1 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %Destroy.facet.loc20_15.2: %Destroy.type = facet_value constants.%.edf, (constants.%DestroyImpl.impl_witness.1d2, constants.%custom_witness.b80ef6.2) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc20_15.4: %Destroy.type = converted constants.%.edf, %Destroy.facet.loc20_15.2 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%.loc20_15.2)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc14_11.1(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc14_11.2(%self.param: ref %.edf) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: --- nontrivial_abstract.carbon = $nontrivial_abstract
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %Abstract: type = class_type @Abstract [concrete]
// CHECK:STDOUT:   %AlsoHasNonTrivialDestruction: type = class_type @AlsoHasNonTrivialDestruction [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %HasNonTrivialDestruction: type = class_type @HasNonTrivialDestruction [concrete]
// CHECK:STDOUT:   %struct_type.x.y.ea4: type = struct_type {.x: %HasNonTrivialDestruction, .y: %AlsoHasNonTrivialDestruction} [concrete]
// CHECK:STDOUT:   %.79f: type = partial_type %Abstract [concrete]
// CHECK:STDOUT:   %MakePartial.type: type = fn_type @MakePartial [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %MakePartial: %MakePartial.type = struct_value () [concrete]
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
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %MakePartial.ref: %MakePartial.type = name_ref MakePartial, file.%MakePartial.decl [concrete = constants.%MakePartial]
// CHECK:STDOUT:   %.loc16_15.1: ref %.79f = temporary_storage
// CHECK:STDOUT:   %MakePartial.call: init %.79f to %.loc16_15.1 = call %MakePartial.ref()
// CHECK:STDOUT:   %.loc16_15.2: ref %.79f = temporary %.loc16_15.1, %MakePartial.call
// CHECK:STDOUT:   %Destroy.facet.loc16_15.1: %Destroy.type = facet_value constants.%.79f, (constants.%custom_witness.b80ef6.5, constants.%DestroyImpl.impl_witness.98a) [concrete = constants.%Destroy.facet.280]
// CHECK:STDOUT:   %.loc16_15.3: %Destroy.type = converted constants.%.79f, %Destroy.facet.loc16_15.1 [concrete = constants.%Destroy.facet.280]
// CHECK:STDOUT:   %Destroy.facet.loc16_15.2: %Destroy.type = facet_value constants.%.79f, (constants.%custom_witness.b80ef6.5, constants.%DestroyImpl.impl_witness.98a) [concrete = constants.%Destroy.facet.280]
// CHECK:STDOUT:   %.loc16_15.4: %Destroy.type = converted constants.%.79f, %Destroy.facet.loc16_15.2 [concrete = constants.%Destroy.facet.280]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%.loc16_15.2)
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
// CHECK:STDOUT: --- nontrivial_derived.carbon = $nontrivial_derived
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %Base: type = class_type @Base [concrete]
// CHECK:STDOUT:   %HasNonTrivialDestruction: type = class_type @HasNonTrivialDestruction [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Base.elem: type = unbound_element_type %Base, %HasNonTrivialDestruction [concrete]
// CHECK:STDOUT:   %struct_type.x.11f: type = struct_type {.x: %HasNonTrivialDestruction} [concrete]
// CHECK:STDOUT:   %Derived: type = class_type @Derived [concrete]
// CHECK:STDOUT:   %AlsoHasNonTrivialDestruction: type = class_type @AlsoHasNonTrivialDestruction [concrete]
// CHECK:STDOUT:   %struct_type.base.y.a60: type = struct_type {.base: %Base, .y: %AlsoHasNonTrivialDestruction} [concrete]
// CHECK:STDOUT:   %.546: type = partial_type %Derived [concrete]
// CHECK:STDOUT:   %MakePartial.type: type = fn_type @MakePartial [concrete]
// CHECK:STDOUT:   %MakePartial: %MakePartial.type = struct_value () [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.2: type = fn_type @SubobjectDestroy.WithSelf.Op.loc19_11.2 [concrete]
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
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.4: type = fn_type @SubobjectDestroy.WithSelf.Op.loc19_11.4 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.4: %SubobjectDestroy.WithSelf.Op.type.6870fd.4 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.4: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.4), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.5: type = fn_type @SubobjectDestroy.WithSelf.Op.loc19_11.5 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.5: %SubobjectDestroy.WithSelf.Op.type.6870fd.5 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.5: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.5), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.c4e: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table.86e [concrete]
// CHECK:STDOUT:   %Destroy.facet.7c1: %Destroy.type = facet_value %AlsoHasNonTrivialDestruction, (%custom_witness.b80ef6.5, %DestroyImpl.impl_witness.c4e) [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.0c7: %SubobjectDestroy.type = facet_value %Base, (%custom_witness.b80ef6.4) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.d83: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table.a5e, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.0c7) [concrete]
// CHECK:STDOUT:   %Destroy.facet.1d9: %Destroy.type = facet_value %Base, (%custom_witness.b80ef6.4, %DestroyImpl.impl_witness.d83) [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.7: type = fn_type @SubobjectDestroy.WithSelf.Op.loc19_11.7 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.7: %SubobjectDestroy.WithSelf.Op.type.6870fd.7 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.7: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.7), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.6d7: %SubobjectDestroy.type = facet_value %.546, (%custom_witness.b80ef6.7) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.65f: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table.a5e, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.6d7) [concrete]
// CHECK:STDOUT:   %Destroy.facet.f99: %Destroy.type = facet_value %.546, (%custom_witness.b80ef6.7, %DestroyImpl.impl_witness.65f) [concrete]
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
// CHECK:STDOUT:   %Main.import_ref.546: %HasNonTrivialDestruction.as.DestroyImpl.impl.Op.type = import_ref Main//types, loc8_24, loaded [concrete = constants.%HasNonTrivialDestruction.as.DestroyImpl.impl.Op]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table.467 = impl_witness_table (%Main.import_ref.546), @HasNonTrivialDestruction.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT:   %Main.import_ref.307: %AlsoHasNonTrivialDestruction.as.DestroyImpl.impl.Op.type = import_ref Main//types, loc14_24, loaded [concrete = constants.%AlsoHasNonTrivialDestruction.as.DestroyImpl.impl.Op]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table.86e = impl_witness_table (%Main.import_ref.307), @AlsoHasNonTrivialDestruction.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Test() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Partial.ref.loc19: type = name_ref Partial, file.%Partial [concrete = constants.%.546]
// CHECK:STDOUT:   %Core.ref.loc19: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:   %SubobjectDestroy.ref: type = name_ref SubobjectDestroy, imports.%Core.SubobjectDestroy [concrete = constants.%SubobjectDestroy.type]
// CHECK:STDOUT:   %SubobjectDestroy.facet: %SubobjectDestroy.type = facet_value %Partial.ref.loc19, (constants.%custom_witness.b80ef6.7) [concrete = constants.%SubobjectDestroy.facet.6d7]
// CHECK:STDOUT:   %.loc19: %SubobjectDestroy.type = converted %Partial.ref.loc19, %SubobjectDestroy.facet [concrete = constants.%SubobjectDestroy.facet.6d7]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %MakePartial.ref: %MakePartial.type = name_ref MakePartial, file.%MakePartial.decl [concrete = constants.%MakePartial]
// CHECK:STDOUT:   %.loc25_15.1: ref %.546 = temporary_storage
// CHECK:STDOUT:   %MakePartial.call: init %.546 to %.loc25_15.1 = call %MakePartial.ref()
// CHECK:STDOUT:   %.loc25_15.2: ref %.546 = temporary %.loc25_15.1, %MakePartial.call
// CHECK:STDOUT:   %Destroy.facet.loc25_15.1: %Destroy.type = facet_value constants.%.546, (constants.%custom_witness.b80ef6.7, constants.%DestroyImpl.impl_witness.65f) [concrete = constants.%Destroy.facet.f99]
// CHECK:STDOUT:   %.loc25_15.3: %Destroy.type = converted constants.%.546, %Destroy.facet.loc25_15.1 [concrete = constants.%Destroy.facet.f99]
// CHECK:STDOUT:   %Destroy.facet.loc25_15.2: %Destroy.type = facet_value constants.%.546, (constants.%custom_witness.b80ef6.7, constants.%DestroyImpl.impl_witness.65f) [concrete = constants.%Destroy.facet.f99]
// CHECK:STDOUT:   %.loc25_15.4: %Destroy.type = converted constants.%.546, %Destroy.facet.loc25_15.2 [concrete = constants.%Destroy.facet.f99]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%.loc25_15.2)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc19_11.1(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc19_11.2(%self.param: ref %HasNonTrivialDestruction) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc19_11.3(%self.param: ref %struct_type.x.11f) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc19_11.1: ref %HasNonTrivialDestruction = struct_access %self.param, element0
// CHECK:STDOUT:   %Destroy.facet.loc19_11.1: %Destroy.type = facet_value constants.%HasNonTrivialDestruction, (constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.490) [concrete = constants.%Destroy.facet.4ef]
// CHECK:STDOUT:   %.loc19_11.2: %Destroy.type = converted constants.%HasNonTrivialDestruction, %Destroy.facet.loc19_11.1 [concrete = constants.%Destroy.facet.4ef]
// CHECK:STDOUT:   %Destroy.facet.loc19_11.2: %Destroy.type = facet_value constants.%HasNonTrivialDestruction, (constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.490) [concrete = constants.%Destroy.facet.4ef]
// CHECK:STDOUT:   %.loc19_11.3: %Destroy.type = converted constants.%HasNonTrivialDestruction, %Destroy.facet.loc19_11.2 [concrete = constants.%Destroy.facet.4ef]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%.loc19_11.1)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc19_11.4(%self.param: ref %Base) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %x.ref: %Base.elem = name_ref x, @Base.%field_decl [concrete = @Base.%field_decl]
// CHECK:STDOUT:   %.loc19_11.1: ref %HasNonTrivialDestruction = class_element_access %self.param, element0
// CHECK:STDOUT:   %Destroy.facet.loc19_11.1: %Destroy.type = facet_value constants.%HasNonTrivialDestruction, (constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.490) [concrete = constants.%Destroy.facet.4ef]
// CHECK:STDOUT:   %.loc19_11.2: %Destroy.type = converted constants.%HasNonTrivialDestruction, %Destroy.facet.loc19_11.1 [concrete = constants.%Destroy.facet.4ef]
// CHECK:STDOUT:   %Destroy.facet.loc19_11.2: %Destroy.type = facet_value constants.%HasNonTrivialDestruction, (constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.490) [concrete = constants.%Destroy.facet.4ef]
// CHECK:STDOUT:   %.loc19_11.3: %Destroy.type = converted constants.%HasNonTrivialDestruction, %Destroy.facet.loc19_11.2 [concrete = constants.%Destroy.facet.4ef]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%.loc19_11.1)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc19_11.5(%self.param: ref %AlsoHasNonTrivialDestruction) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc19_11.6(%self.param: ref %struct_type.base.y.a60) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc19_11.1: ref %AlsoHasNonTrivialDestruction = struct_access %self.param, element1
// CHECK:STDOUT:   %Destroy.facet.loc19_11.1: %Destroy.type = facet_value constants.%AlsoHasNonTrivialDestruction, (constants.%custom_witness.b80ef6.5, constants.%DestroyImpl.impl_witness.c4e) [concrete = constants.%Destroy.facet.7c1]
// CHECK:STDOUT:   %.loc19_11.2: %Destroy.type = converted constants.%AlsoHasNonTrivialDestruction, %Destroy.facet.loc19_11.1 [concrete = constants.%Destroy.facet.7c1]
// CHECK:STDOUT:   %Destroy.facet.loc19_11.2: %Destroy.type = facet_value constants.%AlsoHasNonTrivialDestruction, (constants.%custom_witness.b80ef6.5, constants.%DestroyImpl.impl_witness.c4e) [concrete = constants.%Destroy.facet.7c1]
// CHECK:STDOUT:   %.loc19_11.3: %Destroy.type = converted constants.%AlsoHasNonTrivialDestruction, %Destroy.facet.loc19_11.2 [concrete = constants.%Destroy.facet.7c1]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc19_11.1: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%.loc19_11.1)
// CHECK:STDOUT:   %.loc19_11.4: ref %Base = struct_access %self.param, element0
// CHECK:STDOUT:   %Destroy.facet.loc19_11.3: %Destroy.type = facet_value constants.%Base, (constants.%custom_witness.b80ef6.4, constants.%DestroyImpl.impl_witness.d83) [concrete = constants.%Destroy.facet.1d9]
// CHECK:STDOUT:   %.loc19_11.5: %Destroy.type = converted constants.%Base, %Destroy.facet.loc19_11.3 [concrete = constants.%Destroy.facet.1d9]
// CHECK:STDOUT:   %Destroy.facet.loc19_11.4: %Destroy.type = facet_value constants.%Base, (constants.%custom_witness.b80ef6.4, constants.%DestroyImpl.impl_witness.d83) [concrete = constants.%Destroy.facet.1d9]
// CHECK:STDOUT:   %.loc19_11.6: %Destroy.type = converted constants.%Base, %Destroy.facet.loc19_11.4 [concrete = constants.%Destroy.facet.1d9]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc19_11.2: init %empty_tuple.type = call %SelfDestruct.specific_fn.2(%.loc19_11.4)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc19_11.7(%self.param: ref %.546) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
