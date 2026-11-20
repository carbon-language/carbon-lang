// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/destroy.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/impl/custom_witness/destruction/trivial/partial.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/impl/custom_witness/destruction/trivial/partial.carbon

// --- types.carbon
library "[[@TEST_NAME]]";

class HasNonTrivialDestruction {
  impl as Core.DestroyImpl {
    fn Op(unused self) {}
  }
}

// --- trivial_empty_partial_class.carbon
library "[[@TEST_NAME]]";

base class Base {}
alias PartialBase = partial Base;
fn MakePartialBase() -> PartialBase;

fn Test() {
  //@dump-sem-ir-begin
  PartialBase as Core.SubobjectDestroy;
  //@dump-sem-ir-end

  PartialBase as Core.Destroy;
  PartialBase as Core.TrivialDestroy;

  //@dump-sem-ir-begin
  MakePartialBase();
  //@dump-sem-ir-end
}

// --- trivial_nonempty_partial_class.carbon
library "[[@TEST_NAME]]";

class HasTrivalDestruction {}
class AlsoHasTrivalDestruction {}

base class Base {
  var a: HasTrivalDestruction;
  var b: AlsoHasTrivalDestruction;
}

alias PartialBase = partial Base;
fn MakePartialBase() -> PartialBase;

fn Test() {
  //@dump-sem-ir-begin
  PartialBase as Core.SubobjectDestroy;
  //@dump-sem-ir-end

  PartialBase as Core.Destroy;
  PartialBase as Core.TrivialDestroy;

  //@dump-sem-ir-begin
  MakePartialBase();
  //@dump-sem-ir-end
}

// --- todo_fail_nontrivial_member.carbon
library "[[@TEST_NAME]]";

import library "types";

base class Base {
  var x: HasNonTrivialDestruction;
}

fn Test() {
  partial Base as Core.TrivialDestroy;
}

// --- todo_fail_nontrivial_impls_destroy.carbon
library "[[@TEST_NAME]]";

base class Base {
  impl as Core.DestroyImpl {
    fn Op(unused self) {}
  }
}

fn Test() {
  partial Base as Core.TrivialDestroy;
}

// --- trivial_abstract.carbon
library "[[@TEST_NAME]]";

abstract class Abstract {}
alias PartialAbstract = partial Abstract;
fn MakePartialAbstract() -> PartialAbstract;

fn Test() {
  //@dump-sem-ir-begin
  PartialAbstract as Core.SubobjectDestroy;
  //@dump-sem-ir-end

  PartialAbstract as Core.Destroy;
  PartialAbstract as Core.TrivialDestroy;

  //@dump-sem-ir-begin
  MakePartialAbstract();
  //@dump-sem-ir-end
}

// --- todo_fail_nontrivial_abstract.carbon
library "[[@TEST_NAME]]";

import library "types";

abstract class Abstract {
  var x: HasNonTrivialDestruction;
}

fn Test() {
  partial Abstract as Core.TrivialDestroy;
}

// --- todo_fail_nontrivial_derived.carbon
library "[[@TEST_NAME]]";

import library "types";

base class Base {
  var x: HasNonTrivialDestruction;
}

base class Derived {
  extend base: Base;
}

fn Test() {
  partial Derived as Core.TrivialDestroy;
}

// CHECK:STDOUT: --- trivial_empty_partial_class.carbon = $trivial_empty_partial_class
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %Base: type = class_type @Base [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %.edf: type = partial_type %Base [concrete]
// CHECK:STDOUT:   %MakePartialBase.type: type = fn_type @MakePartialBase [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %MakePartialBase: %MakePartialBase.type = struct_value () [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.2: type = fn_type @SubobjectDestroy.WithSelf.Op.loc9_15.2 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.2: %SubobjectDestroy.WithSelf.Op.type.6870fd.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.2: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.2), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.9a3: %SubobjectDestroy.type = facet_value %.edf, (%custom_witness.b80ef6.2) [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %T.09f: %SubobjectDestroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.type.dea: type = fn_type @T.as_type.as.DestroyImpl.impl.Op, @T.as_type.as.DestroyImpl.impl(%T.09f) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.6e5: %T.as_type.as.DestroyImpl.impl.Op.type.dea = struct_value () [symbolic]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.1d2: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.9a3) [concrete]
// CHECK:STDOUT:   %Destroy.facet: %Destroy.type = facet_value %.edf, (%custom_witness.b80ef6.2, %DestroyImpl.impl_witness.1d2) [concrete]
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
// CHECK:STDOUT:   %PartialBase.ref.loc9: type = name_ref PartialBase, file.%PartialBase [concrete = constants.%.edf]
// CHECK:STDOUT:   %Core.ref.loc9: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:   %SubobjectDestroy.ref: type = name_ref SubobjectDestroy, imports.%Core.SubobjectDestroy [concrete = constants.%SubobjectDestroy.type]
// CHECK:STDOUT:   %SubobjectDestroy.facet: %SubobjectDestroy.type = facet_value %PartialBase.ref.loc9, (constants.%custom_witness.b80ef6.2) [concrete = constants.%SubobjectDestroy.facet.9a3]
// CHECK:STDOUT:   %.loc9: %SubobjectDestroy.type = converted %PartialBase.ref.loc9, %SubobjectDestroy.facet [concrete = constants.%SubobjectDestroy.facet.9a3]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %MakePartialBase.ref: %MakePartialBase.type = name_ref MakePartialBase, file.%MakePartialBase.decl [concrete = constants.%MakePartialBase]
// CHECK:STDOUT:   %.loc16_19.1: ref %.edf = temporary_storage
// CHECK:STDOUT:   %MakePartialBase.call: init %.edf to %.loc16_19.1 = call %MakePartialBase.ref()
// CHECK:STDOUT:   %.loc16_19.2: ref %.edf = temporary %.loc16_19.1, %MakePartialBase.call
// CHECK:STDOUT:   %Destroy.facet.loc16_19.1: %Destroy.type = facet_value constants.%.edf, (constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.1d2) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc16_19.3: %Destroy.type = converted constants.%.edf, %Destroy.facet.loc16_19.1 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %Destroy.facet.loc16_19.2: %Destroy.type = facet_value constants.%.edf, (constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.1d2) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc16_19.4: %Destroy.type = converted constants.%.edf, %Destroy.facet.loc16_19.2 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%.loc16_19.2)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc9_15.1(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc9_15.2(%self.param: ref %.edf) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: --- trivial_nonempty_partial_class.carbon = $trivial_nonempty_partial_class
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %HasTrivalDestruction: type = class_type @HasTrivalDestruction [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %AlsoHasTrivalDestruction: type = class_type @AlsoHasTrivalDestruction [concrete]
// CHECK:STDOUT:   %Base: type = class_type @Base [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %struct_type.a.b.f16: type = struct_type {.a: %HasTrivalDestruction, .b: %AlsoHasTrivalDestruction} [concrete]
// CHECK:STDOUT:   %.edf: type = partial_type %Base [concrete]
// CHECK:STDOUT:   %MakePartialBase.type: type = fn_type @MakePartialBase [concrete]
// CHECK:STDOUT:   %MakePartialBase: %MakePartialBase.type = struct_value () [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.2: type = fn_type @SubobjectDestroy.WithSelf.Op.loc16_15.2 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.2: %SubobjectDestroy.WithSelf.Op.type.6870fd.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.2: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.2), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.3: type = fn_type @SubobjectDestroy.WithSelf.Op.loc16_15.3 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.3: %SubobjectDestroy.WithSelf.Op.type.6870fd.3 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.3: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.3), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SelfDestruct.type: type = fn_type @SelfDestruct [concrete]
// CHECK:STDOUT:   %SelfDestruct: %SelfDestruct.type = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %T.09f: %SubobjectDestroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.type.dea: type = fn_type @T.as_type.as.DestroyImpl.impl.Op, @T.as_type.as.DestroyImpl.impl(%T.09f) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.6e5: %T.as_type.as.DestroyImpl.impl.Op.type.dea = struct_value () [symbolic]
// CHECK:STDOUT:   %SubobjectDestroy.facet.74d: %SubobjectDestroy.type = facet_value %AlsoHasTrivalDestruction, (%custom_witness.b80ef6.3) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.977: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.74d) [concrete]
// CHECK:STDOUT:   %Destroy.facet.a5a: %Destroy.type = facet_value %AlsoHasTrivalDestruction, (%custom_witness.b80ef6.3, %DestroyImpl.impl_witness.977) [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.547: %SubobjectDestroy.type = facet_value %HasTrivalDestruction, (%custom_witness.b80ef6.2) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.165: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.547) [concrete]
// CHECK:STDOUT:   %Destroy.facet.487: %Destroy.type = facet_value %HasTrivalDestruction, (%custom_witness.b80ef6.2, %DestroyImpl.impl_witness.165) [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.5: type = fn_type @SubobjectDestroy.WithSelf.Op.loc16_15.5 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.5: %SubobjectDestroy.WithSelf.Op.type.6870fd.5 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.5: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.5), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.9a3: %SubobjectDestroy.type = facet_value %.edf, (%custom_witness.b80ef6.5) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.1d2: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.9a3) [concrete]
// CHECK:STDOUT:   %Destroy.facet.96b: %Destroy.type = facet_value %.edf, (%custom_witness.b80ef6.5, %DestroyImpl.impl_witness.1d2) [concrete]
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
// CHECK:STDOUT:   %PartialBase.ref.loc16: type = name_ref PartialBase, file.%PartialBase [concrete = constants.%.edf]
// CHECK:STDOUT:   %Core.ref.loc16: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:   %SubobjectDestroy.ref: type = name_ref SubobjectDestroy, imports.%Core.SubobjectDestroy [concrete = constants.%SubobjectDestroy.type]
// CHECK:STDOUT:   %SubobjectDestroy.facet: %SubobjectDestroy.type = facet_value %PartialBase.ref.loc16, (constants.%custom_witness.b80ef6.5) [concrete = constants.%SubobjectDestroy.facet.9a3]
// CHECK:STDOUT:   %.loc16: %SubobjectDestroy.type = converted %PartialBase.ref.loc16, %SubobjectDestroy.facet [concrete = constants.%SubobjectDestroy.facet.9a3]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %MakePartialBase.ref: %MakePartialBase.type = name_ref MakePartialBase, file.%MakePartialBase.decl [concrete = constants.%MakePartialBase]
// CHECK:STDOUT:   %.loc23_19.1: ref %.edf = temporary_storage
// CHECK:STDOUT:   %MakePartialBase.call: init %.edf to %.loc23_19.1 = call %MakePartialBase.ref()
// CHECK:STDOUT:   %.loc23_19.2: ref %.edf = temporary %.loc23_19.1, %MakePartialBase.call
// CHECK:STDOUT:   %Destroy.facet.loc23_19.1: %Destroy.type = facet_value constants.%.edf, (constants.%custom_witness.b80ef6.5, constants.%DestroyImpl.impl_witness.1d2) [concrete = constants.%Destroy.facet.96b]
// CHECK:STDOUT:   %.loc23_19.3: %Destroy.type = converted constants.%.edf, %Destroy.facet.loc23_19.1 [concrete = constants.%Destroy.facet.96b]
// CHECK:STDOUT:   %Destroy.facet.loc23_19.2: %Destroy.type = facet_value constants.%.edf, (constants.%custom_witness.b80ef6.5, constants.%DestroyImpl.impl_witness.1d2) [concrete = constants.%Destroy.facet.96b]
// CHECK:STDOUT:   %.loc23_19.4: %Destroy.type = converted constants.%.edf, %Destroy.facet.loc23_19.2 [concrete = constants.%Destroy.facet.96b]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%.loc23_19.2)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc16_15.1(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc16_15.2(%self.param: ref %HasTrivalDestruction) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc16_15.3(%self.param: ref %AlsoHasTrivalDestruction) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc16_15.4(%self.param: ref %struct_type.a.b.f16) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc16_15.1: ref %AlsoHasTrivalDestruction = struct_access %self.param, element1
// CHECK:STDOUT:   %Destroy.facet.loc16_15.1: %Destroy.type = facet_value constants.%AlsoHasTrivalDestruction, (constants.%custom_witness.b80ef6.3, constants.%DestroyImpl.impl_witness.977) [concrete = constants.%Destroy.facet.a5a]
// CHECK:STDOUT:   %.loc16_15.2: %Destroy.type = converted constants.%AlsoHasTrivalDestruction, %Destroy.facet.loc16_15.1 [concrete = constants.%Destroy.facet.a5a]
// CHECK:STDOUT:   %Destroy.facet.loc16_15.2: %Destroy.type = facet_value constants.%AlsoHasTrivalDestruction, (constants.%custom_witness.b80ef6.3, constants.%DestroyImpl.impl_witness.977) [concrete = constants.%Destroy.facet.a5a]
// CHECK:STDOUT:   %.loc16_15.3: %Destroy.type = converted constants.%AlsoHasTrivalDestruction, %Destroy.facet.loc16_15.2 [concrete = constants.%Destroy.facet.a5a]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc16_15.1: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%.loc16_15.1)
// CHECK:STDOUT:   %.loc16_15.4: ref %HasTrivalDestruction = struct_access %self.param, element0
// CHECK:STDOUT:   %Destroy.facet.loc16_15.3: %Destroy.type = facet_value constants.%HasTrivalDestruction, (constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.165) [concrete = constants.%Destroy.facet.487]
// CHECK:STDOUT:   %.loc16_15.5: %Destroy.type = converted constants.%HasTrivalDestruction, %Destroy.facet.loc16_15.3 [concrete = constants.%Destroy.facet.487]
// CHECK:STDOUT:   %Destroy.facet.loc16_15.4: %Destroy.type = facet_value constants.%HasTrivalDestruction, (constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.165) [concrete = constants.%Destroy.facet.487]
// CHECK:STDOUT:   %.loc16_15.6: %Destroy.type = converted constants.%HasTrivalDestruction, %Destroy.facet.loc16_15.4 [concrete = constants.%Destroy.facet.487]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc16_15.2: init %empty_tuple.type = call %SelfDestruct.specific_fn.2(%.loc16_15.4)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc16_15.5(%self.param: ref %.edf) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- trivial_abstract.carbon = $trivial_abstract
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %Abstract: type = class_type @Abstract [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %.79f: type = partial_type %Abstract [concrete]
// CHECK:STDOUT:   %MakePartialAbstract.type: type = fn_type @MakePartialAbstract [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %MakePartialAbstract: %MakePartialAbstract.type = struct_value () [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.2: type = fn_type @SubobjectDestroy.WithSelf.Op.loc9_19.2 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.2: %SubobjectDestroy.WithSelf.Op.type.6870fd.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.2: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.2), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.c03: %SubobjectDestroy.type = facet_value %.79f, (%custom_witness.b80ef6.2) [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %T.09f: %SubobjectDestroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.type.dea: type = fn_type @T.as_type.as.DestroyImpl.impl.Op, @T.as_type.as.DestroyImpl.impl(%T.09f) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.6e5: %T.as_type.as.DestroyImpl.impl.Op.type.dea = struct_value () [symbolic]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.98a: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.c03) [concrete]
// CHECK:STDOUT:   %Destroy.facet: %Destroy.type = facet_value %.79f, (%custom_witness.b80ef6.2, %DestroyImpl.impl_witness.98a) [concrete]
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
// CHECK:STDOUT:   %PartialAbstract.ref.loc9: type = name_ref PartialAbstract, file.%PartialAbstract [concrete = constants.%.79f]
// CHECK:STDOUT:   %Core.ref.loc9: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:   %SubobjectDestroy.ref: type = name_ref SubobjectDestroy, imports.%Core.SubobjectDestroy [concrete = constants.%SubobjectDestroy.type]
// CHECK:STDOUT:   %SubobjectDestroy.facet: %SubobjectDestroy.type = facet_value %PartialAbstract.ref.loc9, (constants.%custom_witness.b80ef6.2) [concrete = constants.%SubobjectDestroy.facet.c03]
// CHECK:STDOUT:   %.loc9: %SubobjectDestroy.type = converted %PartialAbstract.ref.loc9, %SubobjectDestroy.facet [concrete = constants.%SubobjectDestroy.facet.c03]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %MakePartialAbstract.ref: %MakePartialAbstract.type = name_ref MakePartialAbstract, file.%MakePartialAbstract.decl [concrete = constants.%MakePartialAbstract]
// CHECK:STDOUT:   %.loc16_23.1: ref %.79f = temporary_storage
// CHECK:STDOUT:   %MakePartialAbstract.call: init %.79f to %.loc16_23.1 = call %MakePartialAbstract.ref()
// CHECK:STDOUT:   %.loc16_23.2: ref %.79f = temporary %.loc16_23.1, %MakePartialAbstract.call
// CHECK:STDOUT:   %Destroy.facet.loc16_23.1: %Destroy.type = facet_value constants.%.79f, (constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.98a) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc16_23.3: %Destroy.type = converted constants.%.79f, %Destroy.facet.loc16_23.1 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %Destroy.facet.loc16_23.2: %Destroy.type = facet_value constants.%.79f, (constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.98a) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc16_23.4: %Destroy.type = converted constants.%.79f, %Destroy.facet.loc16_23.2 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%.loc16_23.2)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc9_19.1(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc9_19.2(%self.param: ref %.79f) = "no_op";
// CHECK:STDOUT:
