// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/unformed.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/impl/custom_witness/destruction/trivial/maybe_unformed.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/impl/custom_witness/destruction/trivial/maybe_unformed.carbon

// --- trivial_maybe_unformed.carbon
library "[[@TEST_NAME]]";

class HasTrivialDestruction {}

alias MaybeUnformed = Core.MaybeUnformed(HasTrivialDestruction);
fn MakeMaybeUnformed() -> MaybeUnformed;

fn Test() {
  //@dump-sem-ir-begin
  MaybeUnformed as Core.SubobjectDestroy;
  //@dump-sem-ir-end

  MaybeUnformed as Core.Destroy;
  MaybeUnformed as Core.TrivialDestroy;

  //@dump-sem-ir-begin
  MakeMaybeUnformed();
  //@dump-sem-ir-end
}

// --- fail_nontrivial_maybe_unformed.carbon
library "[[@TEST_NAME]]";

class HasNonTrivialDestruction {
  impl as Core.DestroyImpl {
    fn Op(unused self) {}
  }
}

alias MaybeUnformed = Core.MaybeUnformed(HasNonTrivialDestruction);

fn Test() {
  // CHECK:STDERR:        ╭─┤ fail_nontrivial_maybe_unformed.carbon:[[@LINE+8]]:3
  // CHECK:STDERR: ->     │   MaybeUnformed as Core.TrivialDestroy;
  // CHECK:STDERR:        ·   ┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  // CHECK:STDERR:        ·   ┗━━┫ error: cannot convert type `MaybeUnformed` into type implementing
  // CHECK:STDERR:        ·      ┃        `Core.TrivialDestroy`
  // CHECK:STDERR:        ·      ┃        [ConversionFailureTypeToFacet]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  MaybeUnformed as Core.TrivialDestroy;
}

// --- fail_nontrivial_maybe_unformed_member.carbon
library "[[@TEST_NAME]]";

class HasNonTrivialDestruction {
  impl as Core.DestroyImpl {
    fn Op(unused self) {}
  }
}

class HasNonTrivialMember {
  var x: HasNonTrivialDestruction;
}

alias MaybeUnformed = Core.MaybeUnformed(HasNonTrivialMember);

fn Test() {
  // CHECK:STDERR:        ╭─┤ fail_nontrivial_maybe_unformed_member.carbon:[[@LINE+8]]:3
  // CHECK:STDERR: ->     │   MaybeUnformed as Core.TrivialDestroy;
  // CHECK:STDERR:        ·   ┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  // CHECK:STDERR:        ·   ┗━━┫ error: cannot convert type `MaybeUnformed` into type implementing
  // CHECK:STDERR:        ·      ┃        `Core.TrivialDestroy`
  // CHECK:STDERR:        ·      ┃        [ConversionFailureTypeToFacet]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  MaybeUnformed as Core.TrivialDestroy;
}

// --- fail_nontrivial_maybe_unformed_of_const.carbon
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

fn Test() {
  // CHECK:STDERR:        ╭─┤ fail_nontrivial_maybe_unformed_of_const.carbon:[[@LINE+8]]:3
  // CHECK:STDERR: ->     │   MaybeUnformed as Core.TrivialDestroy;
  // CHECK:STDERR:        ·   ┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  // CHECK:STDERR:        ·   ┗━━┫ error: cannot convert type `MaybeUnformed` into type implementing
  // CHECK:STDERR:        ·      ┃        `Core.TrivialDestroy`
  // CHECK:STDERR:        ·      ┃        [ConversionFailureTypeToFacet]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  MaybeUnformed as Core.TrivialDestroy;
}

// --- fail_nontrivial_const_maybe_unformed.carbon
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

fn Test() {
  // CHECK:STDERR:        ╭─┤ fail_nontrivial_const_maybe_unformed.carbon:[[@LINE+8]]:3
  // CHECK:STDERR: ->     │   Const as Core.TrivialDestroy;
  // CHECK:STDERR:        ·   ┳━━━━━━━━━━━━━━━━━━━━━━━━━━━
  // CHECK:STDERR:        ·   ┗━━┫ error: cannot convert type `Const` into type implementing
  // CHECK:STDERR:        ·      ┃        `Core.TrivialDestroy`
  // CHECK:STDERR:        ·      ┃        [ConversionFailureTypeToFacet]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  Const as Core.TrivialDestroy;
}

// CHECK:STDOUT: --- trivial_maybe_unformed.carbon = $trivial_maybe_unformed
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %HasTrivialDestruction: type = class_type @HasTrivialDestruction [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %MaybeUnformed.type: type = parameterized_class_name_type @MaybeUnformed [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %MaybeUnformed.generic: %MaybeUnformed.type = struct_value () [concrete]
// CHECK:STDOUT:   %MaybeUnformed.f4e: type = class_type @MaybeUnformed, @MaybeUnformed(%HasTrivialDestruction) [concrete]
// CHECK:STDOUT:   %MakeMaybeUnformed.type: type = fn_type @MakeMaybeUnformed [concrete]
// CHECK:STDOUT:   %MakeMaybeUnformed: %MakeMaybeUnformed.type = struct_value () [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %.991: type = maybe_unformed_type %HasTrivialDestruction [concrete]
// CHECK:STDOUT:   %DestroyImpl.type: type = facet_type <@DestroyImpl> [concrete]
// CHECK:STDOUT:   %T.09f: %SubobjectDestroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.type.dea: type = fn_type @T.as_type.as.DestroyImpl.impl.Op, @T.as_type.as.DestroyImpl.impl(%T.09f) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.6e5: %T.as_type.as.DestroyImpl.impl.Op.type.dea = struct_value () [symbolic]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.4: type = fn_type @SubobjectDestroy.WithSelf.Op.loc10_17.4 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.4: %SubobjectDestroy.WithSelf.Op.type.6870fd.4 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.4: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.4), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.7d1: %SubobjectDestroy.type = facet_value %MaybeUnformed.f4e, (%custom_witness.b80ef6.4) [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.ce8: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.7d1) [concrete]
// CHECK:STDOUT:   %Destroy.facet: %Destroy.type = facet_value %MaybeUnformed.f4e, (%custom_witness.b80ef6.4, %DestroyImpl.impl_witness.ce8) [concrete]
// CHECK:STDOUT:   %TrivialDestroy.type: type = facet_type <@TrivialDestroy> [concrete]
// CHECK:STDOUT:   %SelfDestruct.type: type = fn_type @SelfDestruct [concrete]
// CHECK:STDOUT:   %SelfDestruct: %SelfDestruct.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .MaybeUnformed = %Core.MaybeUnformed
// CHECK:STDOUT:     .SubobjectDestroy = %Core.SubobjectDestroy
// CHECK:STDOUT:     .DestroyImpl = %Core.DestroyImpl
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     .TrivialDestroy = %Core.TrivialDestroy
// CHECK:STDOUT:     .SelfDestruct = %Core.SelfDestruct
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.MaybeUnformed: %MaybeUnformed.type = import_ref Core//prelude/parts/maybe_unformed, MaybeUnformed, loaded [concrete = constants.%MaybeUnformed.generic]
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
// CHECK:STDOUT:   %MaybeUnformed.ref.loc10: type = name_ref MaybeUnformed, file.%MaybeUnformed.loc5_7 [concrete = constants.%MaybeUnformed.f4e]
// CHECK:STDOUT:   %Core.ref.loc10: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:   %SubobjectDestroy.ref: type = name_ref SubobjectDestroy, imports.%Core.SubobjectDestroy [concrete = constants.%SubobjectDestroy.type]
// CHECK:STDOUT:   %SubobjectDestroy.facet: %SubobjectDestroy.type = facet_value %MaybeUnformed.ref.loc10, (constants.%custom_witness.b80ef6.4) [concrete = constants.%SubobjectDestroy.facet.7d1]
// CHECK:STDOUT:   %.loc10: %SubobjectDestroy.type = converted %MaybeUnformed.ref.loc10, %SubobjectDestroy.facet [concrete = constants.%SubobjectDestroy.facet.7d1]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %MakeMaybeUnformed.ref: %MakeMaybeUnformed.type = name_ref MakeMaybeUnformed, file.%MakeMaybeUnformed.decl [concrete = constants.%MakeMaybeUnformed]
// CHECK:STDOUT:   %.loc17_21.1: ref %MaybeUnformed.f4e = temporary_storage
// CHECK:STDOUT:   %MakeMaybeUnformed.call: init %MaybeUnformed.f4e to %.loc17_21.1 = call %MakeMaybeUnformed.ref()
// CHECK:STDOUT:   %.loc17_21.2: ref %MaybeUnformed.f4e = temporary %.loc17_21.1, %MakeMaybeUnformed.call
// CHECK:STDOUT:   %Destroy.facet.loc17_21.1: %Destroy.type = facet_value constants.%MaybeUnformed.f4e, (constants.%custom_witness.b80ef6.4, constants.%DestroyImpl.impl_witness.ce8) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc17_21.3: %Destroy.type = converted constants.%MaybeUnformed.f4e, %Destroy.facet.loc17_21.1 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %Destroy.facet.loc17_21.2: %Destroy.type = facet_value constants.%MaybeUnformed.f4e, (constants.%custom_witness.b80ef6.4, constants.%DestroyImpl.impl_witness.ce8) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc17_21.4: %Destroy.type = converted constants.%MaybeUnformed.f4e, %Destroy.facet.loc17_21.2 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%.loc17_21.2)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc10_17.1(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc10_17.2(%self.param: ref %HasTrivialDestruction) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc10_17.3(%self.param: ref %.991) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc10_17.4(%self.param: ref %MaybeUnformed.f4e) = "no_op";
// CHECK:STDOUT:
