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

// --- types.carbon
library "[[@TEST_NAME]]";

class HasNonTrivialDestruction {
  impl as Core.DestroyImpl {
    fn Op(unused self) {}
  }
}

// --- trivial_const_struct.carbon
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

// --- trivial_const_element.carbon
library "[[@TEST_NAME]]";

class HasTrivialDestruction {}

alias Tuple = (const HasTrivialDestruction,);
fn Make() -> Tuple;

fn Test() {
  //@dump-sem-ir-begin
  Tuple as Core.SubobjectDestroy;
  Tuple as Core.Destroy;
  Tuple as Core.TrivialDestroy;

  var unused x: Tuple = Make();
  //@dump-sem-ir-end
}

// --- todo_fail_nontrivial_const_element.carbon
library "[[@TEST_NAME]]";

import library "types";

alias Tuple = (const HasNonTrivialDestruction,);

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

// --- trivial_const_class.carbon
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

// --- trivial_const_partial.carbon
library "[[@TEST_NAME]]";

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

import library "types";

base class Base {
  var x: HasNonTrivialDestruction;
}

alias Const = const (partial Base);

fn Test() {
  Const as Core.TrivialDestroy;
}

// --- trivial_const_dynamic_class.carbon
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

// TODO: Support destroying classes that derive from abstract types.

abstract class TrivialAbstract {}

alias Const = const TrivialAbstract;
fn Make() -> Const;

fn Test() {
  //@dump-sem-ir-begin
  // CHECK:STDERR:        ╭─┤ fail_todo_trivial_const_abstract_class.carbon:[[@LINE+8]]:3
  // CHECK:STDERR: ->     │   Const as Core.SubobjectDestroy;
  // CHECK:STDERR:        ·   ┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  // CHECK:STDERR:        ·   ┗━━┫ error: cannot convert type `Const` into type implementing
  // CHECK:STDERR:        ·      ┃        `Core.SubobjectDestroy`
  // CHECK:STDERR:        ·      ┃        [ConversionFailureTypeToFacet]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  Const as Core.SubobjectDestroy;
  // CHECK:STDERR:        ╭─┤ fail_todo_trivial_const_abstract_class.carbon:[[@LINE+7]]:3
  // CHECK:STDERR: ->     │   Const as Core.Destroy;
  // CHECK:STDERR:        ·   ━┳━━━━━━━━━━━━━━━━━━━
  // CHECK:STDERR:        ·    ┗━━┫ error: cannot convert type `Const` into type implementing `Core.Destroy`
  // CHECK:STDERR:        ·       ┃        [ConversionFailureTypeToFacet]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  Const as Core.Destroy;
  // CHECK:STDERR:        ╭─┤ fail_todo_trivial_const_abstract_class.carbon:[[@LINE+7]]:3
  // CHECK:STDERR: ->     │   Const as Core.TrivialDestroy;
  // CHECK:STDERR:        ·   ━┳━━━━━━━━━━━━━━━━━━━━━━━━━━
  // CHECK:STDERR:        ·    ┗━━┫ error: cannot convert type `Const` into type implementing `Core.Destroy`
  // CHECK:STDERR:        ·       ┃        [ConversionFailureTypeToFacet]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  Const as Core.TrivialDestroy;

  // CHECK:STDERR:        ╭─┤ fail_todo_trivial_const_abstract_class.carbon:[[@LINE+30]]:17
  // CHECK:STDERR:        │ abstract class TrivialAbstract {}
  // CHECK:STDERR:        · ────────────────┬───────────────
  // CHECK:STDERR:        ·                 ╰──┤ uses class that was declared abstract here
  // CHECK:STDERR:        ·                    │ [ClassAbstractHere]
  // CHECK:STDERR:        ┆
  // CHECK:STDERR: ->     │   var unused x: Const = Make();
  // CHECK:STDERR:        ·                 ┳━━━━
  // CHECK:STDERR:        ·                 ┗━━┫ error: binding pattern has abstract type
  // CHECK:STDERR:        ·                    ┃        `const TrivialAbstract` in `var` pattern
  // CHECK:STDERR:        ·                    ┃        [AbstractTypeInVarPattern]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  // CHECK:STDERR:        ╭─┤ fail_todo_trivial_const_abstract_class.carbon:[[@LINE+17]]:25
  // CHECK:STDERR:        │ abstract class TrivialAbstract {}
  // CHECK:STDERR:        · ────────────────┬───────────────
  // CHECK:STDERR:        ·                 ╰──┤ uses class that was declared abstract here
  // CHECK:STDERR:        ·                    │ [ClassAbstractHere]
  // CHECK:STDERR:        ┆
  // CHECK:STDERR:        │ fn Make() -> Const;
  // CHECK:STDERR:        ·           ────┬───
  // CHECK:STDERR:        ·               ╰──┤ return type declared here [IncompleteReturnTypeHere]
  // CHECK:STDERR:        ┆
  // CHECK:STDERR: ->     │   var unused x: Const = Make();
  // CHECK:STDERR:        ·                         ┳━━━━━
  // CHECK:STDERR:        ·                         ┗━━┫ error: function returns abstract type
  // CHECK:STDERR:        ·                            ┃        `const TrivialAbstract`
  // CHECK:STDERR:        ·                            ┃        [AbstractTypeInFunctionReturnType]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  var unused x: Const = Make();
  //@dump-sem-ir-end
}

// --- fail_nontrivial_const_abstract_class.carbon
library "[[@TEST_NAME]]";

abstract class NonTrivialAbstract {
  impl as Core.DestroyImpl {
    fn Op(unused self) {}
  }
}

alias Const = const NonTrivialAbstract;

fn Test() {
  // CHECK:STDERR:        ╭─┤ fail_nontrivial_const_abstract_class.carbon:[[@LINE+7]]:3
  // CHECK:STDERR: ->     │   Const as Core.TrivialDestroy;
  // CHECK:STDERR:        ·   ━┳━━━━━━━━━━━━━━━━━━━━━━━━━━
  // CHECK:STDERR:        ·    ┗━━┫ error: cannot convert type `Const` into type implementing `Core.Destroy`
  // CHECK:STDERR:        ·       ┃        [ConversionFailureTypeToFacet]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  Const as Core.TrivialDestroy;
}

// --- fail_nontrivial_const_class_derived_from_nontrivial_abstract.carbon
library "[[@TEST_NAME]]";

abstract class NonTrivialAbstract {
  impl as Core.DestroyImpl {
    fn Op(unused self) {}
  }
}

class Derived {
  extend base: NonTrivialAbstract;
}

alias Class = const Derived;

fn Test() {
  // CHECK:STDERR:        ╭─┤ fail_nontrivial_const_class_derived_from_nontrivial_abstract.carbon:[[@LINE+7]]:3
  // CHECK:STDERR: ->     │   Class as Core.TrivialDestroy;
  // CHECK:STDERR:        ·   ━┳━━━━━━━━━━━━━━━━━━━━━━━━━━
  // CHECK:STDERR:        ·    ┗━━┫ error: cannot convert type `Class` into type implementing `Core.Destroy`
  // CHECK:STDERR:        ·       ┃        [ConversionFailureTypeToFacet]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  Class as Core.TrivialDestroy;
}

// CHECK:STDOUT: --- trivial_const_struct.carbon = $trivial_const_struct
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
// CHECK:STDOUT: --- trivial_const_element.carbon = $trivial_const_element
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %HasTrivialDestruction: type = class_type @HasTrivialDestruction [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %const: type = const_type %HasTrivialDestruction [concrete]
// CHECK:STDOUT:   %tuple.type.135: type = tuple_type (type) [concrete]
// CHECK:STDOUT:   %tuple: %tuple.type.135 = tuple_value (%const) [concrete]
// CHECK:STDOUT:   %tuple.type.b44: type = tuple_type (%const) [concrete]
// CHECK:STDOUT:   %pattern_type.14a: type = pattern_type %tuple.type.b44 [concrete]
// CHECK:STDOUT:   %Make.type: type = fn_type @Make [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Make: %Make.type = struct_value () [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.3: type = fn_type @SubobjectDestroy.WithSelf.Op.loc10_9.3 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.3: %SubobjectDestroy.WithSelf.Op.type.6870fd.3 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.3: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.3), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.4: type = fn_type @SubobjectDestroy.WithSelf.Op.loc10_9.4 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.4: %SubobjectDestroy.WithSelf.Op.type.6870fd.4 = struct_value () [concrete]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %SelfDestruct.type: type = fn_type @SelfDestruct [concrete]
// CHECK:STDOUT:   %SelfDestruct: %SelfDestruct.type = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %T.09f: %SubobjectDestroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.type.dea: type = fn_type @T.as_type.as.DestroyImpl.impl.Op, @T.as_type.as.DestroyImpl.impl(%T.09f) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.6e5: %T.as_type.as.DestroyImpl.impl.Op.type.dea = struct_value () [symbolic]
// CHECK:STDOUT:   %SubobjectDestroy.facet.65d: %SubobjectDestroy.type = facet_value %const, (%custom_witness.b80ef6.3) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.f71: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.65d) [concrete]
// CHECK:STDOUT:   %Destroy.facet.2ca: %Destroy.type = facet_value %const, (%custom_witness.b80ef6.3, %DestroyImpl.impl_witness.f71) [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.4: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.4), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.450: %SubobjectDestroy.type = facet_value %tuple.type.b44, (%custom_witness.b80ef6.4) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.7ec: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.450) [concrete]
// CHECK:STDOUT:   %Destroy.facet.863: %Destroy.type = facet_value %tuple.type.b44, (%custom_witness.b80ef6.4, %DestroyImpl.impl_witness.7ec) [concrete]
// CHECK:STDOUT:   %x.patt: %pattern_type.14a = ref_binding_pattern x [concrete]
// CHECK:STDOUT:   %x.var_patt: %pattern_type.14a = var_pattern %x.patt [concrete]
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
// CHECK:STDOUT:   %Tuple.ref.loc10: %tuple.type.135 = name_ref Tuple, file.%Tuple [concrete = constants.%tuple]
// CHECK:STDOUT:   %Core.ref.loc10: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:   %SubobjectDestroy.ref: type = name_ref SubobjectDestroy, imports.%Core.SubobjectDestroy [concrete = constants.%SubobjectDestroy.type]
// CHECK:STDOUT:   %SubobjectDestroy.facet: %SubobjectDestroy.type = facet_value constants.%tuple.type.b44, (constants.%custom_witness.b80ef6.4) [concrete = constants.%SubobjectDestroy.facet.450]
// CHECK:STDOUT:   %.loc10: %SubobjectDestroy.type = converted %Tuple.ref.loc10, %SubobjectDestroy.facet [concrete = constants.%SubobjectDestroy.facet.450]
// CHECK:STDOUT:   %Tuple.ref.loc11: %tuple.type.135 = name_ref Tuple, file.%Tuple [concrete = constants.%tuple]
// CHECK:STDOUT:   %Core.ref.loc11: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:   %Destroy.ref: type = name_ref Destroy, imports.%Core.Destroy [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Destroy.facet.loc11: %Destroy.type = facet_value constants.%tuple.type.b44, (constants.%custom_witness.b80ef6.4, constants.%DestroyImpl.impl_witness.7ec) [concrete = constants.%Destroy.facet.863]
// CHECK:STDOUT:   %.loc11: %Destroy.type = converted %Tuple.ref.loc11, %Destroy.facet.loc11 [concrete = constants.%Destroy.facet.863]
// CHECK:STDOUT:   %Tuple.ref.loc12: %tuple.type.135 = name_ref Tuple, file.%Tuple [concrete = constants.%tuple]
// CHECK:STDOUT:   %Core.ref.loc12: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:   %TrivialDestroy.ref: type = name_ref TrivialDestroy, imports.%Core.TrivialDestroy [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Destroy.facet.loc12: %Destroy.type = facet_value constants.%tuple.type.b44, (constants.%custom_witness.b80ef6.4, constants.%DestroyImpl.impl_witness.7ec) [concrete = constants.%Destroy.facet.863]
// CHECK:STDOUT:   %.loc12: %Destroy.type = converted %Tuple.ref.loc12, %Destroy.facet.loc12 [concrete = constants.%Destroy.facet.863]
// CHECK:STDOUT:   %x.var: ref %tuple.type.b44 = var_storage %x.var_patt
// CHECK:STDOUT:   %Make.ref: %Make.type = name_ref Make, file.%Make.decl [concrete = constants.%Make]
// CHECK:STDOUT:   %Make.call: init %tuple.type.b44 = call %Make.ref()
// CHECK:STDOUT:   assign %x.var, %Make.call
// CHECK:STDOUT:   %.loc14_17.1: type = splice_block %.loc14_17.2 [concrete = constants.%tuple.type.b44] {
// CHECK:STDOUT:     %Tuple.ref.loc14: %tuple.type.135 = name_ref Tuple, file.%Tuple [concrete = constants.%tuple]
// CHECK:STDOUT:     %.loc14_17.2: type = converted %Tuple.ref.loc14, constants.%tuple.type.b44 [concrete = constants.%tuple.type.b44]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %x: ref %tuple.type.b44 = wrapper_binding x, %x.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %pattern_type.14a = ref_binding_pattern x [concrete = constants.%x.patt]
// CHECK:STDOUT:     %x.var_patt: %pattern_type.14a = var_pattern %x.patt [concrete = constants.%x.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.facet.loc14_3.1: %Destroy.type = facet_value constants.%tuple.type.b44, (constants.%custom_witness.b80ef6.4, constants.%DestroyImpl.impl_witness.7ec) [concrete = constants.%Destroy.facet.863]
// CHECK:STDOUT:   %.loc14_3.1: %Destroy.type = converted constants.%tuple.type.b44, %Destroy.facet.loc14_3.1 [concrete = constants.%Destroy.facet.863]
// CHECK:STDOUT:   %Destroy.facet.loc14_3.2: %Destroy.type = facet_value constants.%tuple.type.b44, (constants.%custom_witness.b80ef6.4, constants.%DestroyImpl.impl_witness.7ec) [concrete = constants.%Destroy.facet.863]
// CHECK:STDOUT:   %.loc14_3.2: %Destroy.type = converted constants.%tuple.type.b44, %Destroy.facet.loc14_3.2 [concrete = constants.%Destroy.facet.863]
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
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc10_9.4(%self.param: ref %tuple.type.b44) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT:   %tuple.elem0: ref %const = tuple_access %self.param, element0
// CHECK:STDOUT:   %Destroy.facet.loc10_9.1: %Destroy.type = facet_value constants.%const, (constants.%custom_witness.b80ef6.3, constants.%DestroyImpl.impl_witness.f71) [concrete = constants.%Destroy.facet.2ca]
// CHECK:STDOUT:   %.loc10_9.1: %Destroy.type = converted constants.%const, %Destroy.facet.loc10_9.1 [concrete = constants.%Destroy.facet.2ca]
// CHECK:STDOUT:   %Destroy.facet.loc10_9.2: %Destroy.type = facet_value constants.%const, (constants.%custom_witness.b80ef6.3, constants.%DestroyImpl.impl_witness.f71) [concrete = constants.%Destroy.facet.2ca]
// CHECK:STDOUT:   %.loc10_9.2: %Destroy.type = converted constants.%const, %Destroy.facet.loc10_9.2 [concrete = constants.%Destroy.facet.2ca]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%tuple.elem0)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- trivial_const_class.carbon = $trivial_const_class
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
// CHECK:STDOUT: --- trivial_const_partial.carbon = $trivial_const_partial
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
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.3: type = fn_type @SubobjectDestroy.WithSelf.Op.loc9_9.3 [concrete]
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
// CHECK:STDOUT:   %Const.ref.loc9: type = name_ref Const, file.%Const [concrete = constants.%const]
// CHECK:STDOUT:   %Core.ref.loc9: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:   %SubobjectDestroy.ref: type = name_ref SubobjectDestroy, imports.%Core.SubobjectDestroy [concrete = constants.%SubobjectDestroy.type]
// CHECK:STDOUT:   %SubobjectDestroy.facet: %SubobjectDestroy.type = facet_value %Const.ref.loc9, (constants.%custom_witness.b80ef6.3) [concrete = constants.%SubobjectDestroy.facet.235]
// CHECK:STDOUT:   %.loc9: %SubobjectDestroy.type = converted %Const.ref.loc9, %SubobjectDestroy.facet [concrete = constants.%SubobjectDestroy.facet.235]
// CHECK:STDOUT:   %Const.ref.loc10: type = name_ref Const, file.%Const [concrete = constants.%const]
// CHECK:STDOUT:   %Core.ref.loc10: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:   %Destroy.ref: type = name_ref Destroy, imports.%Core.Destroy [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Destroy.facet.loc10: %Destroy.type = facet_value %Const.ref.loc10, (constants.%custom_witness.b80ef6.3, constants.%DestroyImpl.impl_witness.4b7) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc10: %Destroy.type = converted %Const.ref.loc10, %Destroy.facet.loc10 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %Const.ref.loc11: type = name_ref Const, file.%Const [concrete = constants.%const]
// CHECK:STDOUT:   %Core.ref.loc11: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:   %TrivialDestroy.ref: type = name_ref TrivialDestroy, imports.%Core.TrivialDestroy [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Destroy.facet.loc11: %Destroy.type = facet_value %Const.ref.loc11, (constants.%custom_witness.b80ef6.3, constants.%DestroyImpl.impl_witness.4b7) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc11: %Destroy.type = converted %Const.ref.loc11, %Destroy.facet.loc11 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %x.var: ref %const = var_storage %x.var_patt
// CHECK:STDOUT:   %Make.ref: %Make.type = name_ref Make, file.%Make.decl [concrete = constants.%Make]
// CHECK:STDOUT:   %.loc13_3.1: ref %const = splice_block %x.var {}
// CHECK:STDOUT:   %Make.call: init %const to %.loc13_3.1 = call %Make.ref()
// CHECK:STDOUT:   assign %x.var, %Make.call
// CHECK:STDOUT:   %Const.ref.loc13: type = name_ref Const, file.%Const [concrete = constants.%const]
// CHECK:STDOUT:   %x: ref %const = wrapper_binding x, %x.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %pattern_type.3b2 = ref_binding_pattern x [concrete = constants.%x.patt]
// CHECK:STDOUT:     %x.var_patt: %pattern_type.3b2 = var_pattern %x.patt [concrete = constants.%x.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.facet.loc13_3.1: %Destroy.type = facet_value constants.%const, (constants.%custom_witness.b80ef6.3, constants.%DestroyImpl.impl_witness.4b7) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc13_3.2: %Destroy.type = converted constants.%const, %Destroy.facet.loc13_3.1 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %Destroy.facet.loc13_3.2: %Destroy.type = facet_value constants.%const, (constants.%custom_witness.b80ef6.3, constants.%DestroyImpl.impl_witness.4b7) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc13_3.3: %Destroy.type = converted constants.%const, %Destroy.facet.loc13_3.2 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%x.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc9_9.1(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc9_9.2(%self.param: ref %.edf) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc9_9.3(%self.param: ref %const) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_todo_trivial_const_abstract_class.carbon = $fail_todo_trivial_const_abstract_class
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %TrivialAbstract: type = class_type @TrivialAbstract [concrete]
// CHECK:STDOUT:   %const: type = const_type %TrivialAbstract [concrete]
// CHECK:STDOUT:   %Make.type: type = fn_type @Make [concrete]
// CHECK:STDOUT:   %Make: %Make.type = struct_value () [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
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
// CHECK:STDOUT:   %Core.TrivialDestroy: type = import_ref Core//prelude/parts/destroy, TrivialDestroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Core.SelfDestruct: %SelfDestruct.type = import_ref Core//prelude/parts/destroy, SelfDestruct, loaded [concrete = constants.%SelfDestruct]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Test() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Const.ref.loc20: type = name_ref Const, file.%Const [concrete = constants.%const]
// CHECK:STDOUT:   %Core.ref.loc20: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:   %SubobjectDestroy.ref: type = name_ref SubobjectDestroy, imports.%Core.SubobjectDestroy [concrete = constants.%SubobjectDestroy.type]
// CHECK:STDOUT:   %Const.ref.loc28: type = name_ref Const, file.%Const [concrete = constants.%const]
// CHECK:STDOUT:   %Core.ref.loc28: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:   %Destroy.ref: type = name_ref Destroy, imports.%Core.Destroy [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Const.ref.loc36: type = name_ref Const, file.%Const [concrete = constants.%const]
// CHECK:STDOUT:   %Core.ref.loc36: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:   %TrivialDestroy.ref: type = name_ref TrivialDestroy, imports.%Core.TrivialDestroy [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %x.var: ref <error> = var_storage %x.var_patt [concrete = <error>]
// CHECK:STDOUT:   %Make.ref: %Make.type = name_ref Make, file.%Make.decl [concrete = constants.%Make]
// CHECK:STDOUT:   %Make.call: init <error> = call %Make.ref()
// CHECK:STDOUT:   assign %x.var, <error>
// CHECK:STDOUT:   %Const.ref.loc68: type = name_ref Const, file.%Const [concrete = constants.%const]
// CHECK:STDOUT:   %x: <error> = wrapper_binding x, <error> [concrete = <error>]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: <error> = ref_binding_pattern x [concrete = <error>]
// CHECK:STDOUT:     %x.var_patt: <error> = var_pattern %x.patt [concrete = <error>]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
