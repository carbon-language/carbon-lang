// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/unformed.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/impl/custom_witness/destruction/nontrivial/maybe_unformed.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/impl/custom_witness/destruction/nontrivial/maybe_unformed.carbon

// --- nontrivial_maybe_unformed.carbon
library "[[@TEST_NAME]]";

// `HasNonTrivialDestruction.(SubobjectDestroy.Op)` is a no-op because it
// doesn't have any fields that require it to be non-trivial.
class HasNonTrivialDestruction {
  impl as Core.DestroyImpl {
    fn Op(unused self) {}
  }
}

alias MaybeUnformed = Core.MaybeUnformed(HasNonTrivialDestruction);
fn MakeMaybeUnformed() -> MaybeUnformed;

fn Test() {
  //@dump-sem-ir-begin
  MaybeUnformed as Core.SubobjectDestroy;
  //@dump-sem-ir-end

  MaybeUnformed as Core.Destroy;

  //@dump-sem-ir-begin
  MakeMaybeUnformed();
  //@dump-sem-ir-end
}

// --- nontrivial_maybe_unformed_member.carbon
library "[[@TEST_NAME]]";

class HasNonTrivialDestruction {
  impl as Core.DestroyImpl {
    fn Op(unused self) {}
  }
}

// `HasNonTrivialMember.(SubobjectDestroy.Op)` destroys `x`.
class HasNonTrivialMember {
  var x: HasNonTrivialDestruction;
}

alias MaybeUnformed = Core.MaybeUnformed(HasNonTrivialMember);
fn MakeMaybeUnformed() -> MaybeUnformed;

fn Test() {
  //@dump-sem-ir-begin
  MaybeUnformed as Core.SubobjectDestroy;
  //@dump-sem-ir-end

  MaybeUnformed as Core.Destroy;

  //@dump-sem-ir-begin
  MakeMaybeUnformed();
  //@dump-sem-ir-end
}

// --- nontrivial_maybe_unformed_of_const.carbon
library "[[@TEST_NAME]]";

// Not in `const.carbon`, since that file uses a smaller `min_prelude` than unformed.

class HasNonTrivialDestruction {
  impl as Core.DestroyImpl {
    fn Op(unused self) {}
  }
}

class HasNonTrivialMember {
  var x: HasNonTrivialDestruction;
}

alias MaybeUnformed = Core.MaybeUnformed(const HasNonTrivialMember);
fn MakeMaybeUnformed() -> MaybeUnformed;

fn Test() {
  //@dump-sem-ir-begin
  MaybeUnformed as Core.SubobjectDestroy;
  //@dump-sem-ir-end

  MaybeUnformed as Core.Destroy;

  //@dump-sem-ir-begin
  MakeMaybeUnformed();
  //@dump-sem-ir-end
}

// --- nontrivial_const_maybe_unformed.carbon
library "[[@TEST_NAME]]";

// Not in `const.carbon`, since that file uses a smaller `min_prelude` than unformed.

class HasNonTrivialDestruction {
  impl as Core.DestroyImpl {
    fn Op(unused self) {}
  }
}

class HasNonTrivialMember {
  var x: HasNonTrivialDestruction;
}

alias Const = const Core.MaybeUnformed(HasNonTrivialMember);
fn MakeConst() -> Const;

fn Test() {
  //@dump-sem-ir-begin
  Const as Core.SubobjectDestroy;
  //@dump-sem-ir-end

  Const as Core.Destroy;

  //@dump-sem-ir-begin
  MakeConst();
  //@dump-sem-ir-end
}

// CHECK:STDOUT: --- nontrivial_maybe_unformed.carbon = $nontrivial_maybe_unformed
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %HasNonTrivialDestruction: type = class_type @HasNonTrivialDestruction [concrete]
// CHECK:STDOUT:   %DestroyImpl.type: type = facet_type <@DestroyImpl> [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %MaybeUnformed.type: type = parameterized_class_name_type @MaybeUnformed [concrete]
// CHECK:STDOUT:   %MaybeUnformed.generic: %MaybeUnformed.type = struct_value () [concrete]
// CHECK:STDOUT:   %MaybeUnformed.10a: type = class_type @MaybeUnformed, @MaybeUnformed(%HasNonTrivialDestruction) [concrete]
// CHECK:STDOUT:   %MakeMaybeUnformed.type: type = fn_type @MakeMaybeUnformed [concrete]
// CHECK:STDOUT:   %MakeMaybeUnformed: %MakeMaybeUnformed.type = struct_value () [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %.56d: type = maybe_unformed_type %HasNonTrivialDestruction [concrete]
// CHECK:STDOUT:   %T.09f: %SubobjectDestroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.type.dea: type = fn_type @T.as_type.as.DestroyImpl.impl.Op, @T.as_type.as.DestroyImpl.impl(%T.09f) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.6e5: %T.as_type.as.DestroyImpl.impl.Op.type.dea = struct_value () [symbolic]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.4: type = fn_type @SubobjectDestroy.WithSelf.Op.loc16_17.4 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.4: %SubobjectDestroy.WithSelf.Op.type.6870fd.4 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.4: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.4), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.81a: %SubobjectDestroy.type = facet_value %MaybeUnformed.10a, (%custom_witness.b80ef6.4) [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.b76: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.81a) [concrete]
// CHECK:STDOUT:   %Destroy.facet: %Destroy.type = facet_value %MaybeUnformed.10a, (%DestroyImpl.impl_witness.b76, %custom_witness.b80ef6.4) [concrete]
// CHECK:STDOUT:   %SelfDestruct.type: type = fn_type @SelfDestruct [concrete]
// CHECK:STDOUT:   %SelfDestruct: %SelfDestruct.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .DestroyImpl = %Core.DestroyImpl
// CHECK:STDOUT:     .MaybeUnformed = %Core.MaybeUnformed
// CHECK:STDOUT:     .SubobjectDestroy = %Core.SubobjectDestroy
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     .SelfDestruct = %Core.SelfDestruct
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.DestroyImpl: type = import_ref Core//prelude/parts/destroy, DestroyImpl, loaded [concrete = constants.%DestroyImpl.type]
// CHECK:STDOUT:   %Core.MaybeUnformed: %MaybeUnformed.type = import_ref Core//prelude/parts/maybe_unformed, MaybeUnformed, loaded [concrete = constants.%MaybeUnformed.generic]
// CHECK:STDOUT:   %Core.SubobjectDestroy: type = import_ref Core//prelude/parts/destroy, SubobjectDestroy, loaded [concrete = constants.%SubobjectDestroy.type]
// CHECK:STDOUT:   %Core.import_ref.74d: @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op.type (%T.as_type.as.DestroyImpl.impl.Op.type.dea) = import_ref Core//prelude/parts/destroy, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op (constants.%T.as_type.as.DestroyImpl.impl.Op.6e5)]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table = impl_witness_table (%Core.import_ref.74d), @T.as_type.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Core.SelfDestruct: %SelfDestruct.type = import_ref Core//prelude/parts/destroy, SelfDestruct, loaded [concrete = constants.%SelfDestruct]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Test() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %MaybeUnformed.ref.loc16: type = name_ref MaybeUnformed, file.%MaybeUnformed.loc11_7 [concrete = constants.%MaybeUnformed.10a]
// CHECK:STDOUT:   %Core.ref.loc16: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:   %SubobjectDestroy.ref: type = name_ref SubobjectDestroy, imports.%Core.SubobjectDestroy [concrete = constants.%SubobjectDestroy.type]
// CHECK:STDOUT:   %SubobjectDestroy.facet: %SubobjectDestroy.type = facet_value %MaybeUnformed.ref.loc16, (constants.%custom_witness.b80ef6.4) [concrete = constants.%SubobjectDestroy.facet.81a]
// CHECK:STDOUT:   %.loc16: %SubobjectDestroy.type = converted %MaybeUnformed.ref.loc16, %SubobjectDestroy.facet [concrete = constants.%SubobjectDestroy.facet.81a]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %MakeMaybeUnformed.ref: %MakeMaybeUnformed.type = name_ref MakeMaybeUnformed, file.%MakeMaybeUnformed.decl [concrete = constants.%MakeMaybeUnformed]
// CHECK:STDOUT:   %.loc22_21.1: ref %MaybeUnformed.10a = temporary_storage
// CHECK:STDOUT:   %MakeMaybeUnformed.call: init %MaybeUnformed.10a to %.loc22_21.1 = call %MakeMaybeUnformed.ref()
// CHECK:STDOUT:   %.loc22_21.2: ref %MaybeUnformed.10a = temporary %.loc22_21.1, %MakeMaybeUnformed.call
// CHECK:STDOUT:   %Destroy.facet.loc22_21.1: %Destroy.type = facet_value constants.%MaybeUnformed.10a, (constants.%DestroyImpl.impl_witness.b76, constants.%custom_witness.b80ef6.4) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc22_21.3: %Destroy.type = converted constants.%MaybeUnformed.10a, %Destroy.facet.loc22_21.1 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %Destroy.facet.loc22_21.2: %Destroy.type = facet_value constants.%MaybeUnformed.10a, (constants.%DestroyImpl.impl_witness.b76, constants.%custom_witness.b80ef6.4) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc22_21.4: %Destroy.type = converted constants.%MaybeUnformed.10a, %Destroy.facet.loc22_21.2 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%.loc22_21.2)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc16_17.1(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc16_17.2(%self.param: ref %HasNonTrivialDestruction) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc16_17.3(%self.param: ref %.56d) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc16_17.4(%self.param: ref %MaybeUnformed.10a) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- nontrivial_maybe_unformed_member.carbon = $nontrivial_maybe_unformed_member
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %HasNonTrivialDestruction: type = class_type @HasNonTrivialDestruction [concrete]
// CHECK:STDOUT:   %DestroyImpl.type: type = facet_type <@DestroyImpl> [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.376: <witness> = impl_witness @HasNonTrivialDestruction.as.DestroyImpl.impl.%DestroyImpl.impl_witness_table [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %HasNonTrivialMember: type = class_type @HasNonTrivialMember [concrete]
// CHECK:STDOUT:   %HasNonTrivialMember.elem: type = unbound_element_type %HasNonTrivialMember, %HasNonTrivialDestruction [concrete]
// CHECK:STDOUT:   %struct_type.x.11f: type = struct_type {.x: %HasNonTrivialDestruction} [concrete]
// CHECK:STDOUT:   %MaybeUnformed.type: type = parameterized_class_name_type @MaybeUnformed [concrete]
// CHECK:STDOUT:   %MaybeUnformed.generic: %MaybeUnformed.type = struct_value () [concrete]
// CHECK:STDOUT:   %MaybeUnformed.fce: type = class_type @MaybeUnformed, @MaybeUnformed(%HasNonTrivialMember) [concrete]
// CHECK:STDOUT:   %MakeMaybeUnformed.type: type = fn_type @MakeMaybeUnformed [concrete]
// CHECK:STDOUT:   %MakeMaybeUnformed: %MakeMaybeUnformed.type = struct_value () [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %.a2e: type = maybe_unformed_type %HasNonTrivialMember [concrete]
// CHECK:STDOUT:   %T.09f: %SubobjectDestroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.type.dea: type = fn_type @T.as_type.as.DestroyImpl.impl.Op, @T.as_type.as.DestroyImpl.impl(%T.09f) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.6e5: %T.as_type.as.DestroyImpl.impl.Op.type.dea = struct_value () [symbolic]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.2: type = fn_type @SubobjectDestroy.WithSelf.Op.loc19_17.2 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.2: %SubobjectDestroy.WithSelf.Op.type.6870fd.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.2: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.2), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SelfDestruct.type: type = fn_type @SelfDestruct [concrete]
// CHECK:STDOUT:   %SelfDestruct: %SelfDestruct.type = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %Destroy.facet.f4c: %Destroy.type = facet_value %HasNonTrivialDestruction, (%DestroyImpl.impl_witness.376, %custom_witness.b80ef6.2) [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.6: type = fn_type @SubobjectDestroy.WithSelf.Op.loc19_17.6 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.6: %SubobjectDestroy.WithSelf.Op.type.6870fd.6 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.6: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.6), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.5df: %SubobjectDestroy.type = facet_value %MaybeUnformed.fce, (%custom_witness.b80ef6.6) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.6f0: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.5df) [concrete]
// CHECK:STDOUT:   %Destroy.facet.8ae: %Destroy.type = facet_value %MaybeUnformed.fce, (%DestroyImpl.impl_witness.6f0, %custom_witness.b80ef6.6) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .DestroyImpl = %Core.DestroyImpl
// CHECK:STDOUT:     .MaybeUnformed = %Core.MaybeUnformed
// CHECK:STDOUT:     .SubobjectDestroy = %Core.SubobjectDestroy
// CHECK:STDOUT:     .SelfDestruct = %Core.SelfDestruct
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.DestroyImpl: type = import_ref Core//prelude/parts/destroy, DestroyImpl, loaded [concrete = constants.%DestroyImpl.type]
// CHECK:STDOUT:   %Core.MaybeUnformed: %MaybeUnformed.type = import_ref Core//prelude/parts/maybe_unformed, MaybeUnformed, loaded [concrete = constants.%MaybeUnformed.generic]
// CHECK:STDOUT:   %Core.SubobjectDestroy: type = import_ref Core//prelude/parts/destroy, SubobjectDestroy, loaded [concrete = constants.%SubobjectDestroy.type]
// CHECK:STDOUT:   %Core.import_ref.74d: @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op.type (%T.as_type.as.DestroyImpl.impl.Op.type.dea) = import_ref Core//prelude/parts/destroy, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op (constants.%T.as_type.as.DestroyImpl.impl.Op.6e5)]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table = impl_witness_table (%Core.import_ref.74d), @T.as_type.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT:   %Core.SelfDestruct: %SelfDestruct.type = import_ref Core//prelude/parts/destroy, SelfDestruct, loaded [concrete = constants.%SelfDestruct]
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Test() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %MaybeUnformed.ref.loc19: type = name_ref MaybeUnformed, file.%MaybeUnformed.loc14_7 [concrete = constants.%MaybeUnformed.fce]
// CHECK:STDOUT:   %Core.ref.loc19: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:   %SubobjectDestroy.ref: type = name_ref SubobjectDestroy, imports.%Core.SubobjectDestroy [concrete = constants.%SubobjectDestroy.type]
// CHECK:STDOUT:   %SubobjectDestroy.facet: %SubobjectDestroy.type = facet_value %MaybeUnformed.ref.loc19, (constants.%custom_witness.b80ef6.6) [concrete = constants.%SubobjectDestroy.facet.5df]
// CHECK:STDOUT:   %.loc19: %SubobjectDestroy.type = converted %MaybeUnformed.ref.loc19, %SubobjectDestroy.facet [concrete = constants.%SubobjectDestroy.facet.5df]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %MakeMaybeUnformed.ref: %MakeMaybeUnformed.type = name_ref MakeMaybeUnformed, file.%MakeMaybeUnformed.decl [concrete = constants.%MakeMaybeUnformed]
// CHECK:STDOUT:   %.loc25_21.1: ref %MaybeUnformed.fce = temporary_storage
// CHECK:STDOUT:   %MakeMaybeUnformed.call: init %MaybeUnformed.fce to %.loc25_21.1 = call %MakeMaybeUnformed.ref()
// CHECK:STDOUT:   %.loc25_21.2: ref %MaybeUnformed.fce = temporary %.loc25_21.1, %MakeMaybeUnformed.call
// CHECK:STDOUT:   %Destroy.facet.loc25_21.1: %Destroy.type = facet_value constants.%MaybeUnformed.fce, (constants.%DestroyImpl.impl_witness.6f0, constants.%custom_witness.b80ef6.6) [concrete = constants.%Destroy.facet.8ae]
// CHECK:STDOUT:   %.loc25_21.3: %Destroy.type = converted constants.%MaybeUnformed.fce, %Destroy.facet.loc25_21.1 [concrete = constants.%Destroy.facet.8ae]
// CHECK:STDOUT:   %Destroy.facet.loc25_21.2: %Destroy.type = facet_value constants.%MaybeUnformed.fce, (constants.%DestroyImpl.impl_witness.6f0, constants.%custom_witness.b80ef6.6) [concrete = constants.%Destroy.facet.8ae]
// CHECK:STDOUT:   %.loc25_21.4: %Destroy.type = converted constants.%MaybeUnformed.fce, %Destroy.facet.loc25_21.2 [concrete = constants.%Destroy.facet.8ae]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%.loc25_21.2)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc19_17.1(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc19_17.2(%self.param: ref %HasNonTrivialDestruction) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc19_17.3(%self.param: ref %struct_type.x.11f) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc19_17.1: ref %HasNonTrivialDestruction = struct_access %self.param, element0
// CHECK:STDOUT:   %Destroy.facet.loc19_17.1: %Destroy.type = facet_value constants.%HasNonTrivialDestruction, (constants.%DestroyImpl.impl_witness.376, constants.%custom_witness.b80ef6.2) [concrete = constants.%Destroy.facet.f4c]
// CHECK:STDOUT:   %.loc19_17.2: %Destroy.type = converted constants.%HasNonTrivialDestruction, %Destroy.facet.loc19_17.1 [concrete = constants.%Destroy.facet.f4c]
// CHECK:STDOUT:   %Destroy.facet.loc19_17.2: %Destroy.type = facet_value constants.%HasNonTrivialDestruction, (constants.%DestroyImpl.impl_witness.376, constants.%custom_witness.b80ef6.2) [concrete = constants.%Destroy.facet.f4c]
// CHECK:STDOUT:   %.loc19_17.3: %Destroy.type = converted constants.%HasNonTrivialDestruction, %Destroy.facet.loc19_17.2 [concrete = constants.%Destroy.facet.f4c]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%.loc19_17.1)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc19_17.4(%self.param: ref %HasNonTrivialMember) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %x.ref: %HasNonTrivialMember.elem = name_ref x, @HasNonTrivialMember.%field_decl [concrete = @HasNonTrivialMember.%field_decl]
// CHECK:STDOUT:   %.loc19_17.1: ref %HasNonTrivialDestruction = class_element_access %self.param, element0
// CHECK:STDOUT:   %Destroy.facet.loc19_17.1: %Destroy.type = facet_value constants.%HasNonTrivialDestruction, (constants.%DestroyImpl.impl_witness.376, constants.%custom_witness.b80ef6.2) [concrete = constants.%Destroy.facet.f4c]
// CHECK:STDOUT:   %.loc19_17.2: %Destroy.type = converted constants.%HasNonTrivialDestruction, %Destroy.facet.loc19_17.1 [concrete = constants.%Destroy.facet.f4c]
// CHECK:STDOUT:   %Destroy.facet.loc19_17.2: %Destroy.type = facet_value constants.%HasNonTrivialDestruction, (constants.%DestroyImpl.impl_witness.376, constants.%custom_witness.b80ef6.2) [concrete = constants.%Destroy.facet.f4c]
// CHECK:STDOUT:   %.loc19_17.3: %Destroy.type = converted constants.%HasNonTrivialDestruction, %Destroy.facet.loc19_17.2 [concrete = constants.%Destroy.facet.f4c]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%.loc19_17.1)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc19_17.5(%self.param: ref %.a2e) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc19_17.6(%self.param: ref %MaybeUnformed.fce) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- nontrivial_maybe_unformed_of_const.carbon = $nontrivial_maybe_unformed_of_const
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %HasNonTrivialDestruction: type = class_type @HasNonTrivialDestruction [concrete]
// CHECK:STDOUT:   %DestroyImpl.type: type = facet_type <@DestroyImpl> [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.376: <witness> = impl_witness @HasNonTrivialDestruction.as.DestroyImpl.impl.%DestroyImpl.impl_witness_table [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %HasNonTrivialMember: type = class_type @HasNonTrivialMember [concrete]
// CHECK:STDOUT:   %HasNonTrivialMember.elem: type = unbound_element_type %HasNonTrivialMember, %HasNonTrivialDestruction [concrete]
// CHECK:STDOUT:   %struct_type.x.11f: type = struct_type {.x: %HasNonTrivialDestruction} [concrete]
// CHECK:STDOUT:   %MaybeUnformed.type: type = parameterized_class_name_type @MaybeUnformed [concrete]
// CHECK:STDOUT:   %MaybeUnformed.generic: %MaybeUnformed.type = struct_value () [concrete]
// CHECK:STDOUT:   %const.f0e: type = const_type %HasNonTrivialMember [concrete]
// CHECK:STDOUT:   %MaybeUnformed.941: type = class_type @MaybeUnformed, @MaybeUnformed(%const.f0e) [concrete]
// CHECK:STDOUT:   %MakeMaybeUnformed.type: type = fn_type @MakeMaybeUnformed [concrete]
// CHECK:STDOUT:   %MakeMaybeUnformed: %MakeMaybeUnformed.type = struct_value () [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %.be5: type = maybe_unformed_type %const.f0e [concrete]
// CHECK:STDOUT:   %T.09f: %SubobjectDestroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.type.dea: type = fn_type @T.as_type.as.DestroyImpl.impl.Op, @T.as_type.as.DestroyImpl.impl(%T.09f) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.6e5: %T.as_type.as.DestroyImpl.impl.Op.type.dea = struct_value () [symbolic]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.2: type = fn_type @SubobjectDestroy.WithSelf.Op.loc20_17.2 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.2: %SubobjectDestroy.WithSelf.Op.type.6870fd.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.2: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.2), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SelfDestruct.type: type = fn_type @SelfDestruct [concrete]
// CHECK:STDOUT:   %SelfDestruct: %SelfDestruct.type = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %Destroy.facet.f4c: %Destroy.type = facet_value %HasNonTrivialDestruction, (%DestroyImpl.impl_witness.376, %custom_witness.b80ef6.2) [concrete]
// CHECK:STDOUT:   %const.85d: type = const_type %HasNonTrivialDestruction [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.6: type = fn_type @SubobjectDestroy.WithSelf.Op.loc20_17.6 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.6: %SubobjectDestroy.WithSelf.Op.type.6870fd.6 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.5: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.6), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.f1b: %SubobjectDestroy.type = facet_value %const.85d, (%custom_witness.b80ef6.5) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.da7: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.f1b) [concrete]
// CHECK:STDOUT:   %Destroy.facet.85e: %Destroy.type = facet_value %const.85d, (%DestroyImpl.impl_witness.da7, %custom_witness.b80ef6.5) [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.8: type = fn_type @SubobjectDestroy.WithSelf.Op.loc20_17.8 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.8: %SubobjectDestroy.WithSelf.Op.type.6870fd.8 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.8: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.8), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.32f: %SubobjectDestroy.type = facet_value %MaybeUnformed.941, (%custom_witness.b80ef6.8) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.244: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.32f) [concrete]
// CHECK:STDOUT:   %Destroy.facet.7b3: %Destroy.type = facet_value %MaybeUnformed.941, (%DestroyImpl.impl_witness.244, %custom_witness.b80ef6.8) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .DestroyImpl = %Core.DestroyImpl
// CHECK:STDOUT:     .MaybeUnformed = %Core.MaybeUnformed
// CHECK:STDOUT:     .SubobjectDestroy = %Core.SubobjectDestroy
// CHECK:STDOUT:     .SelfDestruct = %Core.SelfDestruct
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.DestroyImpl: type = import_ref Core//prelude/parts/destroy, DestroyImpl, loaded [concrete = constants.%DestroyImpl.type]
// CHECK:STDOUT:   %Core.MaybeUnformed: %MaybeUnformed.type = import_ref Core//prelude/parts/maybe_unformed, MaybeUnformed, loaded [concrete = constants.%MaybeUnformed.generic]
// CHECK:STDOUT:   %Core.SubobjectDestroy: type = import_ref Core//prelude/parts/destroy, SubobjectDestroy, loaded [concrete = constants.%SubobjectDestroy.type]
// CHECK:STDOUT:   %Core.import_ref.74d: @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op.type (%T.as_type.as.DestroyImpl.impl.Op.type.dea) = import_ref Core//prelude/parts/destroy, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op (constants.%T.as_type.as.DestroyImpl.impl.Op.6e5)]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table = impl_witness_table (%Core.import_ref.74d), @T.as_type.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT:   %Core.SelfDestruct: %SelfDestruct.type = import_ref Core//prelude/parts/destroy, SelfDestruct, loaded [concrete = constants.%SelfDestruct]
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Test() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %MaybeUnformed.ref.loc20: type = name_ref MaybeUnformed, file.%MaybeUnformed.loc15_7 [concrete = constants.%MaybeUnformed.941]
// CHECK:STDOUT:   %Core.ref.loc20: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:   %SubobjectDestroy.ref: type = name_ref SubobjectDestroy, imports.%Core.SubobjectDestroy [concrete = constants.%SubobjectDestroy.type]
// CHECK:STDOUT:   %SubobjectDestroy.facet: %SubobjectDestroy.type = facet_value %MaybeUnformed.ref.loc20, (constants.%custom_witness.b80ef6.8) [concrete = constants.%SubobjectDestroy.facet.32f]
// CHECK:STDOUT:   %.loc20: %SubobjectDestroy.type = converted %MaybeUnformed.ref.loc20, %SubobjectDestroy.facet [concrete = constants.%SubobjectDestroy.facet.32f]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %MakeMaybeUnformed.ref: %MakeMaybeUnformed.type = name_ref MakeMaybeUnformed, file.%MakeMaybeUnformed.decl [concrete = constants.%MakeMaybeUnformed]
// CHECK:STDOUT:   %.loc26_21.1: ref %MaybeUnformed.941 = temporary_storage
// CHECK:STDOUT:   %MakeMaybeUnformed.call: init %MaybeUnformed.941 to %.loc26_21.1 = call %MakeMaybeUnformed.ref()
// CHECK:STDOUT:   %.loc26_21.2: ref %MaybeUnformed.941 = temporary %.loc26_21.1, %MakeMaybeUnformed.call
// CHECK:STDOUT:   %Destroy.facet.loc26_21.1: %Destroy.type = facet_value constants.%MaybeUnformed.941, (constants.%DestroyImpl.impl_witness.244, constants.%custom_witness.b80ef6.8) [concrete = constants.%Destroy.facet.7b3]
// CHECK:STDOUT:   %.loc26_21.3: %Destroy.type = converted constants.%MaybeUnformed.941, %Destroy.facet.loc26_21.1 [concrete = constants.%Destroy.facet.7b3]
// CHECK:STDOUT:   %Destroy.facet.loc26_21.2: %Destroy.type = facet_value constants.%MaybeUnformed.941, (constants.%DestroyImpl.impl_witness.244, constants.%custom_witness.b80ef6.8) [concrete = constants.%Destroy.facet.7b3]
// CHECK:STDOUT:   %.loc26_21.4: %Destroy.type = converted constants.%MaybeUnformed.941, %Destroy.facet.loc26_21.2 [concrete = constants.%Destroy.facet.7b3]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%.loc26_21.2)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc20_17.1(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc20_17.2(%self.param: ref %HasNonTrivialDestruction) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc20_17.3(%self.param: ref %struct_type.x.11f) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc20_17.1: ref %HasNonTrivialDestruction = struct_access %self.param, element0
// CHECK:STDOUT:   %Destroy.facet.loc20_17.1: %Destroy.type = facet_value constants.%HasNonTrivialDestruction, (constants.%DestroyImpl.impl_witness.376, constants.%custom_witness.b80ef6.2) [concrete = constants.%Destroy.facet.f4c]
// CHECK:STDOUT:   %.loc20_17.2: %Destroy.type = converted constants.%HasNonTrivialDestruction, %Destroy.facet.loc20_17.1 [concrete = constants.%Destroy.facet.f4c]
// CHECK:STDOUT:   %Destroy.facet.loc20_17.2: %Destroy.type = facet_value constants.%HasNonTrivialDestruction, (constants.%DestroyImpl.impl_witness.376, constants.%custom_witness.b80ef6.2) [concrete = constants.%Destroy.facet.f4c]
// CHECK:STDOUT:   %.loc20_17.3: %Destroy.type = converted constants.%HasNonTrivialDestruction, %Destroy.facet.loc20_17.2 [concrete = constants.%Destroy.facet.f4c]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%.loc20_17.1)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc20_17.4(%self.param: ref %HasNonTrivialMember) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %x.ref: %HasNonTrivialMember.elem = name_ref x, @HasNonTrivialMember.%field_decl [concrete = @HasNonTrivialMember.%field_decl]
// CHECK:STDOUT:   %.loc20_17.1: ref %HasNonTrivialDestruction = class_element_access %self.param, element0
// CHECK:STDOUT:   %Destroy.facet.loc20_17.1: %Destroy.type = facet_value constants.%HasNonTrivialDestruction, (constants.%DestroyImpl.impl_witness.376, constants.%custom_witness.b80ef6.2) [concrete = constants.%Destroy.facet.f4c]
// CHECK:STDOUT:   %.loc20_17.2: %Destroy.type = converted constants.%HasNonTrivialDestruction, %Destroy.facet.loc20_17.1 [concrete = constants.%Destroy.facet.f4c]
// CHECK:STDOUT:   %Destroy.facet.loc20_17.2: %Destroy.type = facet_value constants.%HasNonTrivialDestruction, (constants.%DestroyImpl.impl_witness.376, constants.%custom_witness.b80ef6.2) [concrete = constants.%Destroy.facet.f4c]
// CHECK:STDOUT:   %.loc20_17.3: %Destroy.type = converted constants.%HasNonTrivialDestruction, %Destroy.facet.loc20_17.2 [concrete = constants.%Destroy.facet.f4c]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%.loc20_17.1)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc20_17.5(%self.param: ref %const.f0e) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %x.ref: %HasNonTrivialMember.elem = name_ref x, @HasNonTrivialMember.%field_decl [concrete = @HasNonTrivialMember.%field_decl]
// CHECK:STDOUT:   %.loc20_17.1: ref %const.85d = class_element_access %self.param, element0
// CHECK:STDOUT:   %Destroy.facet.loc20_17.1: %Destroy.type = facet_value constants.%const.85d, (constants.%DestroyImpl.impl_witness.da7, constants.%custom_witness.b80ef6.5) [concrete = constants.%Destroy.facet.85e]
// CHECK:STDOUT:   %.loc20_17.2: %Destroy.type = converted constants.%const.85d, %Destroy.facet.loc20_17.1 [concrete = constants.%Destroy.facet.85e]
// CHECK:STDOUT:   %Destroy.facet.loc20_17.2: %Destroy.type = facet_value constants.%const.85d, (constants.%DestroyImpl.impl_witness.da7, constants.%custom_witness.b80ef6.5) [concrete = constants.%Destroy.facet.85e]
// CHECK:STDOUT:   %.loc20_17.3: %Destroy.type = converted constants.%const.85d, %Destroy.facet.loc20_17.2 [concrete = constants.%Destroy.facet.85e]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%.loc20_17.1)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc20_17.6(%self.param: ref %const.85d) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc20_17.7(%self.param: ref %.be5) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc20_17.8(%self.param: ref %MaybeUnformed.941) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- nontrivial_const_maybe_unformed.carbon = $nontrivial_const_maybe_unformed
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %HasNonTrivialDestruction: type = class_type @HasNonTrivialDestruction [concrete]
// CHECK:STDOUT:   %DestroyImpl.type: type = facet_type <@DestroyImpl> [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.376: <witness> = impl_witness @HasNonTrivialDestruction.as.DestroyImpl.impl.%DestroyImpl.impl_witness_table [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %HasNonTrivialMember: type = class_type @HasNonTrivialMember [concrete]
// CHECK:STDOUT:   %HasNonTrivialMember.elem: type = unbound_element_type %HasNonTrivialMember, %HasNonTrivialDestruction [concrete]
// CHECK:STDOUT:   %struct_type.x.11f: type = struct_type {.x: %HasNonTrivialDestruction} [concrete]
// CHECK:STDOUT:   %MaybeUnformed.type: type = parameterized_class_name_type @MaybeUnformed [concrete]
// CHECK:STDOUT:   %MaybeUnformed.generic: %MaybeUnformed.type = struct_value () [concrete]
// CHECK:STDOUT:   %MaybeUnformed.fce: type = class_type @MaybeUnformed, @MaybeUnformed(%HasNonTrivialMember) [concrete]
// CHECK:STDOUT:   %const: type = const_type %MaybeUnformed.fce [concrete]
// CHECK:STDOUT:   %MakeConst.type: type = fn_type @MakeConst [concrete]
// CHECK:STDOUT:   %MakeConst: %MakeConst.type = struct_value () [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %.a2e: type = maybe_unformed_type %HasNonTrivialMember [concrete]
// CHECK:STDOUT:   %T.09f: %SubobjectDestroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.type.dea: type = fn_type @T.as_type.as.DestroyImpl.impl.Op, @T.as_type.as.DestroyImpl.impl(%T.09f) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.6e5: %T.as_type.as.DestroyImpl.impl.Op.type.dea = struct_value () [symbolic]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.2: type = fn_type @SubobjectDestroy.WithSelf.Op.loc20_9.2 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.2: %SubobjectDestroy.WithSelf.Op.type.6870fd.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.2: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.2), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SelfDestruct.type: type = fn_type @SelfDestruct [concrete]
// CHECK:STDOUT:   %SelfDestruct: %SelfDestruct.type = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %Destroy.facet.f4c: %Destroy.type = facet_value %HasNonTrivialDestruction, (%DestroyImpl.impl_witness.376, %custom_witness.b80ef6.2) [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.7: type = fn_type @SubobjectDestroy.WithSelf.Op.loc20_9.7 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.7: %SubobjectDestroy.WithSelf.Op.type.6870fd.7 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.7: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.7), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.7c7: %SubobjectDestroy.type = facet_value %const, (%custom_witness.b80ef6.7) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.4b8: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.7c7) [concrete]
// CHECK:STDOUT:   %Destroy.facet.645: %Destroy.type = facet_value %const, (%DestroyImpl.impl_witness.4b8, %custom_witness.b80ef6.7) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .DestroyImpl = %Core.DestroyImpl
// CHECK:STDOUT:     .MaybeUnformed = %Core.MaybeUnformed
// CHECK:STDOUT:     .SubobjectDestroy = %Core.SubobjectDestroy
// CHECK:STDOUT:     .SelfDestruct = %Core.SelfDestruct
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.DestroyImpl: type = import_ref Core//prelude/parts/destroy, DestroyImpl, loaded [concrete = constants.%DestroyImpl.type]
// CHECK:STDOUT:   %Core.MaybeUnformed: %MaybeUnformed.type = import_ref Core//prelude/parts/maybe_unformed, MaybeUnformed, loaded [concrete = constants.%MaybeUnformed.generic]
// CHECK:STDOUT:   %Core.SubobjectDestroy: type = import_ref Core//prelude/parts/destroy, SubobjectDestroy, loaded [concrete = constants.%SubobjectDestroy.type]
// CHECK:STDOUT:   %Core.import_ref.74d: @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op.type (%T.as_type.as.DestroyImpl.impl.Op.type.dea) = import_ref Core//prelude/parts/destroy, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op (constants.%T.as_type.as.DestroyImpl.impl.Op.6e5)]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table = impl_witness_table (%Core.import_ref.74d), @T.as_type.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT:   %Core.SelfDestruct: %SelfDestruct.type = import_ref Core//prelude/parts/destroy, SelfDestruct, loaded [concrete = constants.%SelfDestruct]
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Test() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Const.ref.loc20: type = name_ref Const, file.%Const [concrete = constants.%const]
// CHECK:STDOUT:   %Core.ref.loc20: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:   %SubobjectDestroy.ref: type = name_ref SubobjectDestroy, imports.%Core.SubobjectDestroy [concrete = constants.%SubobjectDestroy.type]
// CHECK:STDOUT:   %SubobjectDestroy.facet: %SubobjectDestroy.type = facet_value %Const.ref.loc20, (constants.%custom_witness.b80ef6.7) [concrete = constants.%SubobjectDestroy.facet.7c7]
// CHECK:STDOUT:   %.loc20: %SubobjectDestroy.type = converted %Const.ref.loc20, %SubobjectDestroy.facet [concrete = constants.%SubobjectDestroy.facet.7c7]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %MakeConst.ref: %MakeConst.type = name_ref MakeConst, file.%MakeConst.decl [concrete = constants.%MakeConst]
// CHECK:STDOUT:   %.loc26_13.1: ref %const = temporary_storage
// CHECK:STDOUT:   %MakeConst.call: init %const to %.loc26_13.1 = call %MakeConst.ref()
// CHECK:STDOUT:   %.loc26_13.2: ref %const = temporary %.loc26_13.1, %MakeConst.call
// CHECK:STDOUT:   %Destroy.facet.loc26_13.1: %Destroy.type = facet_value constants.%const, (constants.%DestroyImpl.impl_witness.4b8, constants.%custom_witness.b80ef6.7) [concrete = constants.%Destroy.facet.645]
// CHECK:STDOUT:   %.loc26_13.3: %Destroy.type = converted constants.%const, %Destroy.facet.loc26_13.1 [concrete = constants.%Destroy.facet.645]
// CHECK:STDOUT:   %Destroy.facet.loc26_13.2: %Destroy.type = facet_value constants.%const, (constants.%DestroyImpl.impl_witness.4b8, constants.%custom_witness.b80ef6.7) [concrete = constants.%Destroy.facet.645]
// CHECK:STDOUT:   %.loc26_13.4: %Destroy.type = converted constants.%const, %Destroy.facet.loc26_13.2 [concrete = constants.%Destroy.facet.645]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%.loc26_13.2)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc20_9.1(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc20_9.2(%self.param: ref %HasNonTrivialDestruction) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc20_9.3(%self.param: ref %struct_type.x.11f) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc20_9.1: ref %HasNonTrivialDestruction = struct_access %self.param, element0
// CHECK:STDOUT:   %Destroy.facet.loc20_9.1: %Destroy.type = facet_value constants.%HasNonTrivialDestruction, (constants.%DestroyImpl.impl_witness.376, constants.%custom_witness.b80ef6.2) [concrete = constants.%Destroy.facet.f4c]
// CHECK:STDOUT:   %.loc20_9.2: %Destroy.type = converted constants.%HasNonTrivialDestruction, %Destroy.facet.loc20_9.1 [concrete = constants.%Destroy.facet.f4c]
// CHECK:STDOUT:   %Destroy.facet.loc20_9.2: %Destroy.type = facet_value constants.%HasNonTrivialDestruction, (constants.%DestroyImpl.impl_witness.376, constants.%custom_witness.b80ef6.2) [concrete = constants.%Destroy.facet.f4c]
// CHECK:STDOUT:   %.loc20_9.3: %Destroy.type = converted constants.%HasNonTrivialDestruction, %Destroy.facet.loc20_9.2 [concrete = constants.%Destroy.facet.f4c]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%.loc20_9.1)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc20_9.4(%self.param: ref %HasNonTrivialMember) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %x.ref: %HasNonTrivialMember.elem = name_ref x, @HasNonTrivialMember.%field_decl [concrete = @HasNonTrivialMember.%field_decl]
// CHECK:STDOUT:   %.loc20_9.1: ref %HasNonTrivialDestruction = class_element_access %self.param, element0
// CHECK:STDOUT:   %Destroy.facet.loc20_9.1: %Destroy.type = facet_value constants.%HasNonTrivialDestruction, (constants.%DestroyImpl.impl_witness.376, constants.%custom_witness.b80ef6.2) [concrete = constants.%Destroy.facet.f4c]
// CHECK:STDOUT:   %.loc20_9.2: %Destroy.type = converted constants.%HasNonTrivialDestruction, %Destroy.facet.loc20_9.1 [concrete = constants.%Destroy.facet.f4c]
// CHECK:STDOUT:   %Destroy.facet.loc20_9.2: %Destroy.type = facet_value constants.%HasNonTrivialDestruction, (constants.%DestroyImpl.impl_witness.376, constants.%custom_witness.b80ef6.2) [concrete = constants.%Destroy.facet.f4c]
// CHECK:STDOUT:   %.loc20_9.3: %Destroy.type = converted constants.%HasNonTrivialDestruction, %Destroy.facet.loc20_9.2 [concrete = constants.%Destroy.facet.f4c]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%.loc20_9.1)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc20_9.5(%self.param: ref %.a2e) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc20_9.6(%self.param: ref %MaybeUnformed.fce) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc20_9.7(%self.param: ref %const) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
