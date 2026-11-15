// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/destroy.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/impl/custom_witness/destruction/trivial/tuples.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/impl/custom_witness/destruction/trivial/tuples.carbon

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

alias Tuple = ();
fn Make() -> Tuple;

fn Test() {
  //@dump-sem-ir-begin
  Tuple as Core.SubobjectDestroy;
  Tuple as Core.Destroy;
  Tuple as Core.TrivialDestroy;

  var unused x: Tuple = Make();
  //@dump-sem-ir-end
}

// --- todo_trivial_nonempty.carbon
library "[[@TEST_NAME]]";

import library "types";

alias Tuple = (A, B);
fn Make() -> Tuple;

fn Test() {
  //@dump-sem-ir-begin
  Tuple as Core.SubobjectDestroy;
  Tuple as Core.Destroy;
  Tuple as Core.TrivialDestroy;

  var unused x: Tuple = Make();
  //@dump-sem-ir-end
}

// --- fail_nontrivial_element_first.carbon
library "[[@TEST_NAME]]";

import library "types";

alias Tuple = (HasNonTrivialDestruction, HasTrivialDestruction);

fn Test() {
  // CHECK:STDERR:        ╭─┤ fail_nontrivial_element_first.carbon:[[@LINE+9]]:3
  // CHECK:STDERR: ->     │   Tuple as Core.TrivialDestroy;
  // CHECK:STDERR:        ·   ┳━━━━━━━━━━━━━━━━━━━━━━━━━━━
  // CHECK:STDERR:        ·   ┗━━┫ error: cannot convert type
  // CHECK:STDERR:        ·      ┃        `(HasNonTrivialDestruction, HasTrivialDestruction)` into type
  // CHECK:STDERR:        ·      ┃        implementing `Core.TrivialDestroy`
  // CHECK:STDERR:        ·      ┃        [ConversionFailureTypeToFacet]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  Tuple as Core.TrivialDestroy;
}

// --- fail_nontrivial_element_second.carbon
library "[[@TEST_NAME]]";

import library "types";

alias Tuple = (HasTrivialDestruction, HasNonTrivialDestruction);

fn Test() {
  // CHECK:STDERR:        ╭─┤ fail_nontrivial_element_second.carbon:[[@LINE+9]]:3
  // CHECK:STDERR: ->     │   Tuple as Core.TrivialDestroy;
  // CHECK:STDERR:        ·   ┳━━━━━━━━━━━━━━━━━━━━━━━━━━━
  // CHECK:STDERR:        ·   ┗━━┫ error: cannot convert type
  // CHECK:STDERR:        ·      ┃        `(HasTrivialDestruction, HasNonTrivialDestruction)` into type
  // CHECK:STDERR:        ·      ┃        implementing `Core.TrivialDestroy`
  // CHECK:STDERR:        ·      ┃        [ConversionFailureTypeToFacet]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  Tuple as Core.TrivialDestroy;
}

// --- todo_trivial_nested.carbon
library "[[@TEST_NAME]]";

import library "types";

alias Tuple = (A, (B, A));
fn Make() -> Tuple;

fn Test() {
  //@dump-sem-ir-begin
  Tuple as Core.SubobjectDestroy;
  Tuple as Core.Destroy;
  Tuple as Core.TrivialDestroy;

  var unused x: Tuple = Make();
  //@dump-sem-ir-end
}

// --- fail_nontrivial_nested.carbon
library "[[@TEST_NAME]]";

import library "types";

alias Tuple = ((HasNonTrivialDestruction,),);

fn Test() {
  // CHECK:STDERR:        ╭─┤ fail_nontrivial_nested.carbon:[[@LINE+8]]:3
  // CHECK:STDERR: ->     │   Tuple as Core.TrivialDestroy;
  // CHECK:STDERR:        ·   ┳━━━━━━━━━━━━━━━━━━━━━━━━━━━
  // CHECK:STDERR:        ·   ┗━━┫ error: cannot convert type `((HasNonTrivialDestruction,),)` into type
  // CHECK:STDERR:        ·      ┃        implementing `Core.TrivialDestroy`
  // CHECK:STDERR:        ·      ┃        [ConversionFailureTypeToFacet]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  Tuple as Core.TrivialDestroy;
}

// CHECK:STDOUT: --- todo_trivial_empty.carbon = $todo_trivial_empty
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %pattern_type.cb1: type = pattern_type %empty_tuple.type [concrete]
// CHECK:STDOUT:   %Make.type: type = fn_type @Make [concrete]
// CHECK:STDOUT:   %Make: %Make.type = struct_value () [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %DestroyImpl.type: type = facet_type <@DestroyImpl> [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.687: type = fn_type @SubobjectDestroy.WithSelf.Op.loc8 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f: %SubobjectDestroy.WithSelf.Op.type.687 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.f89: %SubobjectDestroy.type = facet_value %empty_tuple.type, (%custom_witness.b80) [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %T.09f: %SubobjectDestroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.type.dea: type = fn_type @T.as_type.as.DestroyImpl.impl.Op, @T.as_type.as.DestroyImpl.impl(%T.09f) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.6e5: %T.as_type.as.DestroyImpl.impl.Op.type.dea = struct_value () [symbolic]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.674: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.f89) [concrete]
// CHECK:STDOUT:   %Destroy.facet: %Destroy.type = facet_value %empty_tuple.type, (%custom_witness.b80, %DestroyImpl.impl_witness.674) [concrete]
// CHECK:STDOUT:   %TrivialDestroy.type: type = facet_type <@TrivialDestroy> [concrete]
// CHECK:STDOUT:   %custom_witness.447: <witness> = custom_witness (), @TrivialDestroyImpl [concrete]
// CHECK:STDOUT:   %TrivialDestroy.facet: %TrivialDestroy.type = facet_value %empty_tuple.type, (%custom_witness.b80, %DestroyImpl.impl_witness.674, %custom_witness.447) [concrete]
// CHECK:STDOUT:   %x.patt: %pattern_type.cb1 = ref_binding_pattern x [concrete]
// CHECK:STDOUT:   %x.var_patt: %pattern_type.cb1 = var_pattern %x.patt [concrete]
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
// CHECK:STDOUT:   %Tuple.ref.loc8: %empty_tuple.type = name_ref Tuple, file.%Tuple [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %Core.ref.loc8: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:   %SubobjectDestroy.ref: type = name_ref SubobjectDestroy, imports.%Core.SubobjectDestroy [concrete = constants.%SubobjectDestroy.type]
// CHECK:STDOUT:   %SubobjectDestroy.facet: %SubobjectDestroy.type = facet_value constants.%empty_tuple.type, (constants.%custom_witness.b80) [concrete = constants.%SubobjectDestroy.facet.f89]
// CHECK:STDOUT:   %.loc8: %SubobjectDestroy.type = converted %Tuple.ref.loc8, %SubobjectDestroy.facet [concrete = constants.%SubobjectDestroy.facet.f89]
// CHECK:STDOUT:   %Tuple.ref.loc9: %empty_tuple.type = name_ref Tuple, file.%Tuple [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %Core.ref.loc9: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:   %Destroy.ref: type = name_ref Destroy, imports.%Core.Destroy [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Destroy.facet.loc9: %Destroy.type = facet_value constants.%empty_tuple.type, (constants.%custom_witness.b80, constants.%DestroyImpl.impl_witness.674) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc9: %Destroy.type = converted %Tuple.ref.loc9, %Destroy.facet.loc9 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %Tuple.ref.loc10: %empty_tuple.type = name_ref Tuple, file.%Tuple [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %Core.ref.loc10: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:   %TrivialDestroy.ref: type = name_ref TrivialDestroy, imports.%Core.TrivialDestroy [concrete = constants.%TrivialDestroy.type]
// CHECK:STDOUT:   %TrivialDestroy.facet: %TrivialDestroy.type = facet_value constants.%empty_tuple.type, (constants.%custom_witness.b80, constants.%DestroyImpl.impl_witness.674, constants.%custom_witness.447) [concrete = constants.%TrivialDestroy.facet]
// CHECK:STDOUT:   %.loc10: %TrivialDestroy.type = converted %Tuple.ref.loc10, %TrivialDestroy.facet [concrete = constants.%TrivialDestroy.facet]
// CHECK:STDOUT:   %x.var: ref %empty_tuple.type = var_storage %x.var_patt
// CHECK:STDOUT:   %Make.ref: %Make.type = name_ref Make, file.%Make.decl [concrete = constants.%Make]
// CHECK:STDOUT:   %Make.call: init %empty_tuple.type = call %Make.ref()
// CHECK:STDOUT:   assign %x.var, %Make.call
// CHECK:STDOUT:   %.loc12_17.1: type = splice_block %.loc12_17.2 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:     %Tuple.ref.loc12: %empty_tuple.type = name_ref Tuple, file.%Tuple [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc12_17.2: type = converted %Tuple.ref.loc12, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %x: ref %empty_tuple.type = wrapper_binding x, %x.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %pattern_type.cb1 = ref_binding_pattern x [concrete = constants.%x.patt]
// CHECK:STDOUT:     %x.var_patt: %pattern_type.cb1 = var_pattern %x.patt [concrete = constants.%x.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.facet.loc12_3.1: %Destroy.type = facet_value constants.%empty_tuple.type, (constants.%custom_witness.b80, constants.%DestroyImpl.impl_witness.674) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc12_3.1: %Destroy.type = converted constants.%empty_tuple.type, %Destroy.facet.loc12_3.1 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %Destroy.facet.loc12_3.2: %Destroy.type = facet_value constants.%empty_tuple.type, (constants.%custom_witness.b80, constants.%DestroyImpl.impl_witness.674) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc12_3.2: %Destroy.type = converted constants.%empty_tuple.type, %Destroy.facet.loc12_3.2 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%x.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc8(%self.param: ref %empty_tuple.type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: --- todo_trivial_nonempty.carbon = $todo_trivial_nonempty
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %A: type = class_type @A [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %B: type = class_type @B [concrete]
// CHECK:STDOUT:   %tuple.type.693: type = tuple_type (type, type) [concrete]
// CHECK:STDOUT:   %tuple: %tuple.type.693 = tuple_value (%A, %B) [concrete]
// CHECK:STDOUT:   %tuple.type.1e7: type = tuple_type (%A, %B) [concrete]
// CHECK:STDOUT:   %pattern_type.7c6: type = pattern_type %tuple.type.1e7 [concrete]
// CHECK:STDOUT:   %Make.type: type = fn_type @Make [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Make: %Make.type = struct_value () [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %DestroyImpl.type: type = facet_type <@DestroyImpl> [concrete]
// CHECK:STDOUT:   %T.09f: %SubobjectDestroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.type.dea: type = fn_type @T.as_type.as.DestroyImpl.impl.Op, @T.as_type.as.DestroyImpl.impl(%T.09f) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.6e5: %T.as_type.as.DestroyImpl.impl.Op.type.dea = struct_value () [symbolic]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.4: type = fn_type @SubobjectDestroy.WithSelf.Op.loc10_9.4 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.4: %SubobjectDestroy.WithSelf.Op.type.6870fd.4 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.4: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.4), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.995: %SubobjectDestroy.type = facet_value %tuple.type.1e7, (%custom_witness.b80ef6.4) [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.f3a: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.995) [concrete]
// CHECK:STDOUT:   %Destroy.facet: %Destroy.type = facet_value %tuple.type.1e7, (%custom_witness.b80ef6.4, %DestroyImpl.impl_witness.f3a) [concrete]
// CHECK:STDOUT:   %TrivialDestroy.type: type = facet_type <@TrivialDestroy> [concrete]
// CHECK:STDOUT:   %custom_witness.447: <witness> = custom_witness (), @TrivialDestroyImpl [concrete]
// CHECK:STDOUT:   %TrivialDestroy.facet: %TrivialDestroy.type = facet_value %tuple.type.1e7, (%custom_witness.b80ef6.4, %DestroyImpl.impl_witness.f3a, %custom_witness.447) [concrete]
// CHECK:STDOUT:   %x.patt: %pattern_type.7c6 = ref_binding_pattern x [concrete]
// CHECK:STDOUT:   %x.var_patt: %pattern_type.7c6 = var_pattern %x.patt [concrete]
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
// CHECK:STDOUT:   %Tuple.ref.loc10: %tuple.type.693 = name_ref Tuple, file.%Tuple [concrete = constants.%tuple]
// CHECK:STDOUT:   %Core.ref.loc10: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:   %SubobjectDestroy.ref: type = name_ref SubobjectDestroy, imports.%Core.SubobjectDestroy [concrete = constants.%SubobjectDestroy.type]
// CHECK:STDOUT:   %SubobjectDestroy.facet: %SubobjectDestroy.type = facet_value constants.%tuple.type.1e7, (constants.%custom_witness.b80ef6.4) [concrete = constants.%SubobjectDestroy.facet.995]
// CHECK:STDOUT:   %.loc10: %SubobjectDestroy.type = converted %Tuple.ref.loc10, %SubobjectDestroy.facet [concrete = constants.%SubobjectDestroy.facet.995]
// CHECK:STDOUT:   %Tuple.ref.loc11: %tuple.type.693 = name_ref Tuple, file.%Tuple [concrete = constants.%tuple]
// CHECK:STDOUT:   %Core.ref.loc11: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:   %Destroy.ref: type = name_ref Destroy, imports.%Core.Destroy [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Destroy.facet.loc11: %Destroy.type = facet_value constants.%tuple.type.1e7, (constants.%custom_witness.b80ef6.4, constants.%DestroyImpl.impl_witness.f3a) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc11: %Destroy.type = converted %Tuple.ref.loc11, %Destroy.facet.loc11 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %Tuple.ref.loc12: %tuple.type.693 = name_ref Tuple, file.%Tuple [concrete = constants.%tuple]
// CHECK:STDOUT:   %Core.ref.loc12: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:   %TrivialDestroy.ref: type = name_ref TrivialDestroy, imports.%Core.TrivialDestroy [concrete = constants.%TrivialDestroy.type]
// CHECK:STDOUT:   %TrivialDestroy.facet: %TrivialDestroy.type = facet_value constants.%tuple.type.1e7, (constants.%custom_witness.b80ef6.4, constants.%DestroyImpl.impl_witness.f3a, constants.%custom_witness.447) [concrete = constants.%TrivialDestroy.facet]
// CHECK:STDOUT:   %.loc12: %TrivialDestroy.type = converted %Tuple.ref.loc12, %TrivialDestroy.facet [concrete = constants.%TrivialDestroy.facet]
// CHECK:STDOUT:   %x.var: ref %tuple.type.1e7 = var_storage %x.var_patt
// CHECK:STDOUT:   %Make.ref: %Make.type = name_ref Make, file.%Make.decl [concrete = constants.%Make]
// CHECK:STDOUT:   %.loc14_3.1: ref %tuple.type.1e7 = splice_block %x.var {}
// CHECK:STDOUT:   %Make.call: init %tuple.type.1e7 to %.loc14_3.1 = call %Make.ref()
// CHECK:STDOUT:   assign %x.var, %Make.call
// CHECK:STDOUT:   %.loc14_17.1: type = splice_block %.loc14_17.2 [concrete = constants.%tuple.type.1e7] {
// CHECK:STDOUT:     %Tuple.ref.loc14: %tuple.type.693 = name_ref Tuple, file.%Tuple [concrete = constants.%tuple]
// CHECK:STDOUT:     %.loc14_17.2: type = converted %Tuple.ref.loc14, constants.%tuple.type.1e7 [concrete = constants.%tuple.type.1e7]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %x: ref %tuple.type.1e7 = wrapper_binding x, %x.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %pattern_type.7c6 = ref_binding_pattern x [concrete = constants.%x.patt]
// CHECK:STDOUT:     %x.var_patt: %pattern_type.7c6 = var_pattern %x.patt [concrete = constants.%x.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.facet.loc14_3.1: %Destroy.type = facet_value constants.%tuple.type.1e7, (constants.%custom_witness.b80ef6.4, constants.%DestroyImpl.impl_witness.f3a) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc14_3.2: %Destroy.type = converted constants.%tuple.type.1e7, %Destroy.facet.loc14_3.1 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %Destroy.facet.loc14_3.2: %Destroy.type = facet_value constants.%tuple.type.1e7, (constants.%custom_witness.b80ef6.4, constants.%DestroyImpl.impl_witness.f3a) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc14_3.3: %Destroy.type = converted constants.%tuple.type.1e7, %Destroy.facet.loc14_3.2 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%x.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc10_9.1(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc10_9.2(%self.param: ref %A) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc10_9.3(%self.param: ref %B) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc10_9.4(%self.param: ref %tuple.type.1e7) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: --- todo_trivial_nested.carbon = $todo_trivial_nested
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %A: type = class_type @A [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %B: type = class_type @B [concrete]
// CHECK:STDOUT:   %tuple.type.693: type = tuple_type (type, type) [concrete]
// CHECK:STDOUT:   %tuple.0e1: %tuple.type.693 = tuple_value (%B, %A) [concrete]
// CHECK:STDOUT:   %tuple.type.bd0: type = tuple_type (type, %tuple.type.693) [concrete]
// CHECK:STDOUT:   %tuple.f99: %tuple.type.bd0 = tuple_value (%A, %tuple.0e1) [concrete]
// CHECK:STDOUT:   %tuple.type.df2: type = tuple_type (%B, %A) [concrete]
// CHECK:STDOUT:   %tuple.type.e0b: type = tuple_type (%A, %tuple.type.df2) [concrete]
// CHECK:STDOUT:   %pattern_type.a21: type = pattern_type %tuple.type.e0b [concrete]
// CHECK:STDOUT:   %Make.type: type = fn_type @Make [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Make: %Make.type = struct_value () [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %DestroyImpl.type: type = facet_type <@DestroyImpl> [concrete]
// CHECK:STDOUT:   %T.09f: %SubobjectDestroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.type.dea: type = fn_type @T.as_type.as.DestroyImpl.impl.Op, @T.as_type.as.DestroyImpl.impl(%T.09f) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.6e5: %T.as_type.as.DestroyImpl.impl.Op.type.dea = struct_value () [symbolic]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.5: type = fn_type @SubobjectDestroy.WithSelf.Op.loc10_9.5 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.5: %SubobjectDestroy.WithSelf.Op.type.6870fd.5 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.5: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.5), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.4ff: %SubobjectDestroy.type = facet_value %tuple.type.e0b, (%custom_witness.b80ef6.5) [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.ecd: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.4ff) [concrete]
// CHECK:STDOUT:   %Destroy.facet: %Destroy.type = facet_value %tuple.type.e0b, (%custom_witness.b80ef6.5, %DestroyImpl.impl_witness.ecd) [concrete]
// CHECK:STDOUT:   %TrivialDestroy.type: type = facet_type <@TrivialDestroy> [concrete]
// CHECK:STDOUT:   %custom_witness.447: <witness> = custom_witness (), @TrivialDestroyImpl [concrete]
// CHECK:STDOUT:   %TrivialDestroy.facet: %TrivialDestroy.type = facet_value %tuple.type.e0b, (%custom_witness.b80ef6.5, %DestroyImpl.impl_witness.ecd, %custom_witness.447) [concrete]
// CHECK:STDOUT:   %x.patt: %pattern_type.a21 = ref_binding_pattern x [concrete]
// CHECK:STDOUT:   %x.var_patt: %pattern_type.a21 = var_pattern %x.patt [concrete]
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
// CHECK:STDOUT:   %Tuple.ref.loc10: %tuple.type.bd0 = name_ref Tuple, file.%Tuple [concrete = constants.%tuple.f99]
// CHECK:STDOUT:   %Core.ref.loc10: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:   %SubobjectDestroy.ref: type = name_ref SubobjectDestroy, imports.%Core.SubobjectDestroy [concrete = constants.%SubobjectDestroy.type]
// CHECK:STDOUT:   %.loc10_9.1: type = converted constants.%tuple.0e1, constants.%tuple.type.df2 [concrete = constants.%tuple.type.df2]
// CHECK:STDOUT:   %SubobjectDestroy.facet: %SubobjectDestroy.type = facet_value constants.%tuple.type.e0b, (constants.%custom_witness.b80ef6.5) [concrete = constants.%SubobjectDestroy.facet.4ff]
// CHECK:STDOUT:   %.loc10_9.2: %SubobjectDestroy.type = converted %Tuple.ref.loc10, %SubobjectDestroy.facet [concrete = constants.%SubobjectDestroy.facet.4ff]
// CHECK:STDOUT:   %Tuple.ref.loc11: %tuple.type.bd0 = name_ref Tuple, file.%Tuple [concrete = constants.%tuple.f99]
// CHECK:STDOUT:   %Core.ref.loc11: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:   %Destroy.ref: type = name_ref Destroy, imports.%Core.Destroy [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %.loc11_9.1: type = converted constants.%tuple.0e1, constants.%tuple.type.df2 [concrete = constants.%tuple.type.df2]
// CHECK:STDOUT:   %Destroy.facet.loc11: %Destroy.type = facet_value constants.%tuple.type.e0b, (constants.%custom_witness.b80ef6.5, constants.%DestroyImpl.impl_witness.ecd) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc11_9.2: %Destroy.type = converted %Tuple.ref.loc11, %Destroy.facet.loc11 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %Tuple.ref.loc12: %tuple.type.bd0 = name_ref Tuple, file.%Tuple [concrete = constants.%tuple.f99]
// CHECK:STDOUT:   %Core.ref.loc12: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:   %TrivialDestroy.ref: type = name_ref TrivialDestroy, imports.%Core.TrivialDestroy [concrete = constants.%TrivialDestroy.type]
// CHECK:STDOUT:   %.loc12_9.1: type = converted constants.%tuple.0e1, constants.%tuple.type.df2 [concrete = constants.%tuple.type.df2]
// CHECK:STDOUT:   %TrivialDestroy.facet: %TrivialDestroy.type = facet_value constants.%tuple.type.e0b, (constants.%custom_witness.b80ef6.5, constants.%DestroyImpl.impl_witness.ecd, constants.%custom_witness.447) [concrete = constants.%TrivialDestroy.facet]
// CHECK:STDOUT:   %.loc12_9.2: %TrivialDestroy.type = converted %Tuple.ref.loc12, %TrivialDestroy.facet [concrete = constants.%TrivialDestroy.facet]
// CHECK:STDOUT:   %x.var: ref %tuple.type.e0b = var_storage %x.var_patt
// CHECK:STDOUT:   %Make.ref: %Make.type = name_ref Make, file.%Make.decl [concrete = constants.%Make]
// CHECK:STDOUT:   %.loc14_3.1: ref %tuple.type.e0b = splice_block %x.var {}
// CHECK:STDOUT:   %Make.call: init %tuple.type.e0b to %.loc14_3.1 = call %Make.ref()
// CHECK:STDOUT:   assign %x.var, %Make.call
// CHECK:STDOUT:   %.loc14_17.1: type = splice_block %.loc14_17.3 [concrete = constants.%tuple.type.e0b] {
// CHECK:STDOUT:     %Tuple.ref.loc14: %tuple.type.bd0 = name_ref Tuple, file.%Tuple [concrete = constants.%tuple.f99]
// CHECK:STDOUT:     %.loc14_17.2: type = converted constants.%tuple.0e1, constants.%tuple.type.df2 [concrete = constants.%tuple.type.df2]
// CHECK:STDOUT:     %.loc14_17.3: type = converted %Tuple.ref.loc14, constants.%tuple.type.e0b [concrete = constants.%tuple.type.e0b]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %x: ref %tuple.type.e0b = wrapper_binding x, %x.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %pattern_type.a21 = ref_binding_pattern x [concrete = constants.%x.patt]
// CHECK:STDOUT:     %x.var_patt: %pattern_type.a21 = var_pattern %x.patt [concrete = constants.%x.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.facet.loc14_3.1: %Destroy.type = facet_value constants.%tuple.type.e0b, (constants.%custom_witness.b80ef6.5, constants.%DestroyImpl.impl_witness.ecd) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc14_3.2: %Destroy.type = converted constants.%tuple.type.e0b, %Destroy.facet.loc14_3.1 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %Destroy.facet.loc14_3.2: %Destroy.type = facet_value constants.%tuple.type.e0b, (constants.%custom_witness.b80ef6.5, constants.%DestroyImpl.impl_witness.ecd) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc14_3.3: %Destroy.type = converted constants.%tuple.type.e0b, %Destroy.facet.loc14_3.2 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%x.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc10_9.1(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc10_9.2(%self.param: ref %A) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc10_9.3(%self.param: ref %B) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc10_9.4(%self.param: ref %tuple.type.df2) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc10_9.5(%self.param: ref %tuple.type.e0b) = "no_op";
// CHECK:STDOUT:
