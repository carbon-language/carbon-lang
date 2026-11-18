// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/destroy.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/impl/custom_witness/destruction/trivial/const.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/impl/custom_witness/destruction/trivial/const.carbon

// TODO: implement `Core.TrivialDestroy`.
// TODO: `Core.SelfDestruct` should either be a no-op or not generated for types with trivial destruction.

// --- types.carbon
library "[[@TEST_NAME]]";

class HasTrivialDestruction {}

class HasNonTrivialDestruction {
  impl as Core.DestroyImpl {
    fn Op(unused self) {}
  }
}

// --- todo_trivial_const_struct.carbon
library "[[@TEST_NAME]]";

alias Struct = const {};
fn Make() -> Struct;

fn Test() {
  //@dump-sem-ir-begin
  Struct as Core.SubobjectDestroy;
  Struct as Core.Destroy;
  Struct as Core.TrivialDestroy;

  var unused x: Struct = Make();
  //@dump-sem-ir-end
}

// --- todo_fail_nontrivial_const_element.carbon
library "[[@TEST_NAME]]";

import library "types";

alias Tuple = (const HasNonTrivialDestruction, HasNonTrivialDestruction);

fn Test() {
  Tuple as Core.TrivialDestroy;
}

// --- todo_fail_nontrivial_const_adapter.carbon
library "[[@TEST_NAME]]";

import library "types";

class Adapter {
  adapt HasNonTrivialDestruction;
}

alias Const = const Adapter;

fn Test() {
  Const as Core.TrivialDestroy;
}

// --- todo_trivial_const_class.carbon
library "[[@TEST_NAME]]";

class HasTrivialDestruction {}

alias Class = const HasTrivialDestruction;
fn Make() -> Class;

fn Test() {
  //@dump-sem-ir-begin
  Class as Core.SubobjectDestroy;
  Class as Core.Destroy;
  Class as Core.TrivialDestroy;

  var unused x: Class = Make();
  //@dump-sem-ir-end
}

// --- todo_fail_nontrivial_const_class.carbon
library "[[@TEST_NAME]]";

class HasNonTrivialDestruction {
  impl as Core.DestroyImpl {
    fn Op(unused self) {}
  }
}

alias Const = const HasNonTrivialDestruction;
fn Make() -> Const;

fn Test() {
  Const as Core.TrivialDestroy;
}

// --- todo_trivial_const_partial.carbon
library "[[@TEST_NAME]]";

// TODO: implement case for `PartialType` in `MakeSubobjectDestroyOpBody`.

base class Base {}
alias Const = const (partial Base);
fn Make() -> Const;

fn Test() {
  //@dump-sem-ir-begin
  Const as Core.SubobjectDestroy;
  Const as Core.Destroy;
  Const as Core.TrivialDestroy;

  var unused x: Const = Make();
  //@dump-sem-ir-end
}

// --- todo_fail_nontrivial_const_partial.carbon
library "[[@TEST_NAME]]";

// TODO: implement case for `PartialType` in `MakeSubobjectDestroyOpBody`.

import library "types";

base class Base {
  var x: HasNonTrivialDestruction;
}

alias Const = const (partial Base);

fn Test() {
  Const as Core.TrivialDestroy;
}

// --- todo_trivial_const_dynamic_class.carbon
library "[[@TEST_NAME]]";

base class TrivialDynamic {
  virtual fn F(self);
}

alias Class = const TrivialDynamic;

fn Test() {
  Class as Core.TrivialDestroy;
}

// --- todo_fail_nontrivial_const_dynamic_class.carbon
library "[[@TEST_NAME]]";

base class NonTrivialDynamic {
  virtual fn F(self);

  impl as Core.DestroyImpl {
    fn Op(unused self) {}
  }
}

alias Class = const NonTrivialDynamic;

fn Test() {
  Class as Core.TrivialDestroy;
}

// --- fail_todo_trivial_const_abstract_class.carbon
library "[[@TEST_NAME]]";

abstract class TrivialAbstract {}

alias Const = const TrivialAbstract;

fn Test() {
  // CHECK:STDERR:        ╭─┤ fail_todo_trivial_const_abstract_class.carbon:[[@LINE+7]]:3
  // CHECK:STDERR: ->     │   Const as Core.TrivialDestroy;
  // CHECK:STDERR:        ·   ━┳━━━━━━━━━━━━━━━━━━━━━━━━━━
  // CHECK:STDERR:        ·    ┗━━┫ error: cannot convert type `Const` into type implementing `Core.Destroy`
  // CHECK:STDERR:        ·       ┃        [ConversionFailureTypeToFacet]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  Const as Core.TrivialDestroy;
}

// --- fail_fail_nontrivial_const_abstract_class.carbon
library "[[@TEST_NAME]]";

abstract class NonTrivialAbstract {
  // CHECK:STDERR:        ╭─┤ fail_fail_nontrivial_const_abstract_class.carbon:[[@LINE+7]]:3
  // CHECK:STDERR: ->     │   impl as Core.Destroy {
  // CHECK:STDERR:        ·   ━━━━━━━━━━━┳━━━━━━━━━━
  // CHECK:STDERR:        ·              ┗━━┫ error: impl as 2 interfaces, expected 1
  // CHECK:STDERR:        ·                 ┃        [ImplOfNotOneInterface]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  impl as Core.Destroy {
    // CHECK:STDERR:        ╭─┤ fail_fail_nontrivial_const_abstract_class.carbon:[[@LINE+6]]:11
    // CHECK:STDERR: ->     │     fn Op(self) {}
    // CHECK:STDERR:        ·           ━━┳━
    // CHECK:STDERR:        ·             ┗━━┫ warning: binding `self` unused [UnusedBinding]
    // CHECK:STDERR:        ╰────
    // CHECK:STDERR:
    fn Op(self) {}
  }
}

alias Const = const NonTrivialAbstract;

fn Test() {
  // CHECK:STDERR:        ╭─┤ fail_fail_nontrivial_const_abstract_class.carbon:[[@LINE+7]]:3
  // CHECK:STDERR: ->     │   Const as Core.TrivialDestroy;
  // CHECK:STDERR:        ·   ━┳━━━━━━━━━━━━━━━━━━━━━━━━━━
  // CHECK:STDERR:        ·    ┗━━┫ error: cannot convert type `Const` into type implementing `Core.Destroy`
  // CHECK:STDERR:        ·       ┃        [ConversionFailureTypeToFacet]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  Const as Core.TrivialDestroy;
}

// CHECK:STDOUT: --- todo_trivial_const_struct.carbon = $todo_trivial_const_struct
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %const: type = const_type %empty_struct_type [concrete]
// CHECK:STDOUT:   %pattern_type.aee: type = pattern_type %const [concrete]
// CHECK:STDOUT:   %Make.type: type = fn_type @Make [concrete]
// CHECK:STDOUT:   %Make: %Make.type = struct_value () [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.2: type = fn_type @SubobjectDestroy.WithSelf.Op.loc8_10.2 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.2: %SubobjectDestroy.WithSelf.Op.type.6870fd.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.2: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.2), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.d45: %SubobjectDestroy.type = facet_value %const, (%custom_witness.b80ef6.2) [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %T.09f: %SubobjectDestroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.type.dea: type = fn_type @T.as_type.as.DestroyImpl.impl.Op, @T.as_type.as.DestroyImpl.impl(%T.09f) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.6e5: %T.as_type.as.DestroyImpl.impl.Op.type.dea = struct_value () [symbolic]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.fa3: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.d45) [concrete]
// CHECK:STDOUT:   %Destroy.facet: %Destroy.type = facet_value %const, (%custom_witness.b80ef6.2, %DestroyImpl.impl_witness.fa3) [concrete]
// CHECK:STDOUT:   %x.patt: %pattern_type.aee = ref_binding_pattern x [concrete]
// CHECK:STDOUT:   %x.var_patt: %pattern_type.aee = var_pattern %x.patt [concrete]
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
// CHECK:STDOUT:   %Struct.ref.loc8: type = name_ref Struct, file.%Struct [concrete = constants.%const]
// CHECK:STDOUT:   %Core.ref.loc8: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:   %SubobjectDestroy.ref: type = name_ref SubobjectDestroy, imports.%Core.SubobjectDestroy [concrete = constants.%SubobjectDestroy.type]
// CHECK:STDOUT:   %SubobjectDestroy.facet: %SubobjectDestroy.type = facet_value %Struct.ref.loc8, (constants.%custom_witness.b80ef6.2) [concrete = constants.%SubobjectDestroy.facet.d45]
// CHECK:STDOUT:   %.loc8: %SubobjectDestroy.type = converted %Struct.ref.loc8, %SubobjectDestroy.facet [concrete = constants.%SubobjectDestroy.facet.d45]
// CHECK:STDOUT:   %Struct.ref.loc9: type = name_ref Struct, file.%Struct [concrete = constants.%const]
// CHECK:STDOUT:   %Core.ref.loc9: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:   %Destroy.ref: type = name_ref Destroy, imports.%Core.Destroy [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Destroy.facet.loc9: %Destroy.type = facet_value %Struct.ref.loc9, (constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.fa3) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc9: %Destroy.type = converted %Struct.ref.loc9, %Destroy.facet.loc9 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %Struct.ref.loc10: type = name_ref Struct, file.%Struct [concrete = constants.%const]
// CHECK:STDOUT:   %Core.ref.loc10: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:   %TrivialDestroy.ref: type = name_ref TrivialDestroy, imports.%Core.TrivialDestroy [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Destroy.facet.loc10: %Destroy.type = facet_value %Struct.ref.loc10, (constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.fa3) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc10: %Destroy.type = converted %Struct.ref.loc10, %Destroy.facet.loc10 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %x.var: ref %const = var_storage %x.var_patt
// CHECK:STDOUT:   %Make.ref: %Make.type = name_ref Make, file.%Make.decl [concrete = constants.%Make]
// CHECK:STDOUT:   %Make.call: init %const = call %Make.ref()
// CHECK:STDOUT:   assign %x.var, %Make.call
// CHECK:STDOUT:   %Struct.ref.loc12: type = name_ref Struct, file.%Struct [concrete = constants.%const]
// CHECK:STDOUT:   %x: ref %const = wrapper_binding x, %x.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %pattern_type.aee = ref_binding_pattern x [concrete = constants.%x.patt]
// CHECK:STDOUT:     %x.var_patt: %pattern_type.aee = var_pattern %x.patt [concrete = constants.%x.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.facet.loc12_3.1: %Destroy.type = facet_value constants.%const, (constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.fa3) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc12_3.1: %Destroy.type = converted constants.%const, %Destroy.facet.loc12_3.1 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %Destroy.facet.loc12_3.2: %Destroy.type = facet_value constants.%const, (constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.fa3) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc12_3.2: %Destroy.type = converted constants.%const, %Destroy.facet.loc12_3.2 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%x.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc8_10.1(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc8_10.2(%self.param: ref %const) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- todo_trivial_const_class.carbon = $todo_trivial_const_class
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %HasTrivialDestruction: type = class_type @HasTrivialDestruction [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %const: type = const_type %HasTrivialDestruction [concrete]
// CHECK:STDOUT:   %pattern_type.872: type = pattern_type %const [concrete]
// CHECK:STDOUT:   %Make.type: type = fn_type @Make [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Make: %Make.type = struct_value () [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.3: type = fn_type @SubobjectDestroy.WithSelf.Op.loc10_9.3 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.3: %SubobjectDestroy.WithSelf.Op.type.6870fd.3 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.3: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.3), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.65d: %SubobjectDestroy.type = facet_value %const, (%custom_witness.b80ef6.3) [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %T.09f: %SubobjectDestroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.type.dea: type = fn_type @T.as_type.as.DestroyImpl.impl.Op, @T.as_type.as.DestroyImpl.impl(%T.09f) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.6e5: %T.as_type.as.DestroyImpl.impl.Op.type.dea = struct_value () [symbolic]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.f71: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.65d) [concrete]
// CHECK:STDOUT:   %Destroy.facet: %Destroy.type = facet_value %const, (%custom_witness.b80ef6.3, %DestroyImpl.impl_witness.f71) [concrete]
// CHECK:STDOUT:   %x.patt: %pattern_type.872 = ref_binding_pattern x [concrete]
// CHECK:STDOUT:   %x.var_patt: %pattern_type.872 = var_pattern %x.patt [concrete]
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
// CHECK:STDOUT:   %Class.ref.loc10: type = name_ref Class, file.%Class [concrete = constants.%const]
// CHECK:STDOUT:   %Core.ref.loc10: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:   %SubobjectDestroy.ref: type = name_ref SubobjectDestroy, imports.%Core.SubobjectDestroy [concrete = constants.%SubobjectDestroy.type]
// CHECK:STDOUT:   %SubobjectDestroy.facet: %SubobjectDestroy.type = facet_value %Class.ref.loc10, (constants.%custom_witness.b80ef6.3) [concrete = constants.%SubobjectDestroy.facet.65d]
// CHECK:STDOUT:   %.loc10: %SubobjectDestroy.type = converted %Class.ref.loc10, %SubobjectDestroy.facet [concrete = constants.%SubobjectDestroy.facet.65d]
// CHECK:STDOUT:   %Class.ref.loc11: type = name_ref Class, file.%Class [concrete = constants.%const]
// CHECK:STDOUT:   %Core.ref.loc11: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:   %Destroy.ref: type = name_ref Destroy, imports.%Core.Destroy [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Destroy.facet.loc11: %Destroy.type = facet_value %Class.ref.loc11, (constants.%custom_witness.b80ef6.3, constants.%DestroyImpl.impl_witness.f71) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc11: %Destroy.type = converted %Class.ref.loc11, %Destroy.facet.loc11 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %Class.ref.loc12: type = name_ref Class, file.%Class [concrete = constants.%const]
// CHECK:STDOUT:   %Core.ref.loc12: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:   %TrivialDestroy.ref: type = name_ref TrivialDestroy, imports.%Core.TrivialDestroy [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Destroy.facet.loc12: %Destroy.type = facet_value %Class.ref.loc12, (constants.%custom_witness.b80ef6.3, constants.%DestroyImpl.impl_witness.f71) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc12: %Destroy.type = converted %Class.ref.loc12, %Destroy.facet.loc12 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %x.var: ref %const = var_storage %x.var_patt
// CHECK:STDOUT:   %Make.ref: %Make.type = name_ref Make, file.%Make.decl [concrete = constants.%Make]
// CHECK:STDOUT:   %.loc14_3.1: ref %const = splice_block %x.var {}
// CHECK:STDOUT:   %Make.call: init %const to %.loc14_3.1 = call %Make.ref()
// CHECK:STDOUT:   assign %x.var, %Make.call
// CHECK:STDOUT:   %Class.ref.loc14: type = name_ref Class, file.%Class [concrete = constants.%const]
// CHECK:STDOUT:   %x: ref %const = wrapper_binding x, %x.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %pattern_type.872 = ref_binding_pattern x [concrete = constants.%x.patt]
// CHECK:STDOUT:     %x.var_patt: %pattern_type.872 = var_pattern %x.patt [concrete = constants.%x.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.facet.loc14_3.1: %Destroy.type = facet_value constants.%const, (constants.%custom_witness.b80ef6.3, constants.%DestroyImpl.impl_witness.f71) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc14_3.2: %Destroy.type = converted constants.%const, %Destroy.facet.loc14_3.1 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %Destroy.facet.loc14_3.2: %Destroy.type = facet_value constants.%const, (constants.%custom_witness.b80ef6.3, constants.%DestroyImpl.impl_witness.f71) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc14_3.3: %Destroy.type = converted constants.%const, %Destroy.facet.loc14_3.2 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%x.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc10_9.1(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc10_9.2(%self.param: ref %HasTrivialDestruction) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc10_9.3(%self.param: ref %const) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- todo_trivial_const_partial.carbon = $todo_trivial_const_partial
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %Base: type = class_type @Base [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %.edf: type = partial_type %Base [concrete]
// CHECK:STDOUT:   %const: type = const_type %.edf [concrete]
// CHECK:STDOUT:   %pattern_type.3b2: type = pattern_type %const [concrete]
// CHECK:STDOUT:   %Make.type: type = fn_type @Make [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Make: %Make.type = struct_value () [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.3: type = fn_type @SubobjectDestroy.WithSelf.Op.loc11_9.3 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.3: %SubobjectDestroy.WithSelf.Op.type.6870fd.3 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.3: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.3), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.235: %SubobjectDestroy.type = facet_value %const, (%custom_witness.b80ef6.3) [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %T.09f: %SubobjectDestroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.type.dea: type = fn_type @T.as_type.as.DestroyImpl.impl.Op, @T.as_type.as.DestroyImpl.impl(%T.09f) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.6e5: %T.as_type.as.DestroyImpl.impl.Op.type.dea = struct_value () [symbolic]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.4b7: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.235) [concrete]
// CHECK:STDOUT:   %Destroy.facet: %Destroy.type = facet_value %const, (%custom_witness.b80ef6.3, %DestroyImpl.impl_witness.4b7) [concrete]
// CHECK:STDOUT:   %x.patt: %pattern_type.3b2 = ref_binding_pattern x [concrete]
// CHECK:STDOUT:   %x.var_patt: %pattern_type.3b2 = var_pattern %x.patt [concrete]
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
// CHECK:STDOUT:   %Const.ref.loc11: type = name_ref Const, file.%Const [concrete = constants.%const]
// CHECK:STDOUT:   %Core.ref.loc11: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:   %SubobjectDestroy.ref: type = name_ref SubobjectDestroy, imports.%Core.SubobjectDestroy [concrete = constants.%SubobjectDestroy.type]
// CHECK:STDOUT:   %SubobjectDestroy.facet: %SubobjectDestroy.type = facet_value %Const.ref.loc11, (constants.%custom_witness.b80ef6.3) [concrete = constants.%SubobjectDestroy.facet.235]
// CHECK:STDOUT:   %.loc11: %SubobjectDestroy.type = converted %Const.ref.loc11, %SubobjectDestroy.facet [concrete = constants.%SubobjectDestroy.facet.235]
// CHECK:STDOUT:   %Const.ref.loc12: type = name_ref Const, file.%Const [concrete = constants.%const]
// CHECK:STDOUT:   %Core.ref.loc12: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:   %Destroy.ref: type = name_ref Destroy, imports.%Core.Destroy [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Destroy.facet.loc12: %Destroy.type = facet_value %Const.ref.loc12, (constants.%custom_witness.b80ef6.3, constants.%DestroyImpl.impl_witness.4b7) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc12: %Destroy.type = converted %Const.ref.loc12, %Destroy.facet.loc12 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %Const.ref.loc13: type = name_ref Const, file.%Const [concrete = constants.%const]
// CHECK:STDOUT:   %Core.ref.loc13: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:   %TrivialDestroy.ref: type = name_ref TrivialDestroy, imports.%Core.TrivialDestroy [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Destroy.facet.loc13: %Destroy.type = facet_value %Const.ref.loc13, (constants.%custom_witness.b80ef6.3, constants.%DestroyImpl.impl_witness.4b7) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc13: %Destroy.type = converted %Const.ref.loc13, %Destroy.facet.loc13 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %x.var: ref %const = var_storage %x.var_patt
// CHECK:STDOUT:   %Make.ref: %Make.type = name_ref Make, file.%Make.decl [concrete = constants.%Make]
// CHECK:STDOUT:   %.loc15_3.1: ref %const = splice_block %x.var {}
// CHECK:STDOUT:   %Make.call: init %const to %.loc15_3.1 = call %Make.ref()
// CHECK:STDOUT:   assign %x.var, %Make.call
// CHECK:STDOUT:   %Const.ref.loc15: type = name_ref Const, file.%Const [concrete = constants.%const]
// CHECK:STDOUT:   %x: ref %const = wrapper_binding x, %x.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %pattern_type.3b2 = ref_binding_pattern x [concrete = constants.%x.patt]
// CHECK:STDOUT:     %x.var_patt: %pattern_type.3b2 = var_pattern %x.patt [concrete = constants.%x.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.facet.loc15_3.1: %Destroy.type = facet_value constants.%const, (constants.%custom_witness.b80ef6.3, constants.%DestroyImpl.impl_witness.4b7) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc15_3.2: %Destroy.type = converted constants.%const, %Destroy.facet.loc15_3.1 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %Destroy.facet.loc15_3.2: %Destroy.type = facet_value constants.%const, (constants.%custom_witness.b80ef6.3, constants.%DestroyImpl.impl_witness.4b7) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc15_3.3: %Destroy.type = converted constants.%const, %Destroy.facet.loc15_3.2 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%x.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc11_9.1(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc11_9.2(%self.param: ref %.edf) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc11_9.3(%self.param: ref %const) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
