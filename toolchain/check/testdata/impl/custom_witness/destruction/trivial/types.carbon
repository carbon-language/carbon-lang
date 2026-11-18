// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/impl/custom_witness/destruction/trivial/types.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/impl/custom_witness/destruction/trivial/types.carbon

// --- todo_trivial_associated_entity.carbon
library "[[@TEST_NAME]]";

// TODO: add `AssociatedEntityType` to `IsBuiltinWithTrivialDestruction`.

interface I {
  fn F();
}

alias AssociatedEntityType = typeof(I.F);

fn Make() -> AssociatedEntityType;

fn Test() {
  //@dump-sem-ir-begin
  // AssociatedEntityType as Core.SubobjectDestroy;
  // AssociatedEntityType as Core.Destroy;
  // AssociatedEntityType as Core.TrivialDestroy;
  //
  // var unused x: AssociatedEntityType = Make();
  //@dump-sem-ir-end
}

// --- todo_trivial_generic_interface.carbon
library "[[@TEST_NAME]]";

// TODO: add `GenericInterfaceType` to `IsBuiltinWithTrivialDestruction`.

interface GenericInterface(T: type) {}
alias GenericInterfaceType = typeof(GenericInterface);

fn Make() -> GenericInterfaceType;

fn Test() {
  //@dump-sem-ir-begin
  // GenericInterfaceType as Core.SubobjectDestroy;
  // GenericInterfaceType as Core.Destroy;
  // GenericInterfaceType as Core.TrivialDestroy;
  //
  // var unused x: GenericInterfaceType = Make();
  //@dump-sem-ir-end
}

// --- todo_trivial_generic_named_constraint.carbon
library "[[@TEST_NAME]]";

// TODO: add `GenericNamedConstraintType` to `IsBuiltinWithTrivialDestruction`.

constraint GenericClass(T: type) {}
alias GenericNamedConstraintType = typeof(GenericClass);

fn Test() {
  //@dump-sem-ir-begin
  // GenericNamedConstraintType as Core.SubobjectDestroy;
  // GenericNamedConstraintType as Core.Destroy;
  // GenericNamedConstraintType as Core.TrivialDestroy;
  //
  // var unused x: GenericNamedConstraintType
  //@dump-sem-ir-end
}

// --- todo_trivial_namespace.carbon
library "[[@TEST_NAME]]";

// TODO: add `NamespaceType` to `IsBuiltinWithTrivialDestruction`.

namespace N;
alias NamespaceType = typeof(N);

fn Make() -> NamespaceType;

fn Test() {
  //@dump-sem-ir-begin
  // NamespaceType as Core.SubobjectDestroy;
  // NamespaceType as Core.Destroy;
  // NamespaceType as Core.TrivialDestroy;
  //
  // var unused x: NamespaceType = Make();
  //@dump-sem-ir-end
}

// --- todo_trivial_parameterised_class.carbon
library "[[@TEST_NAME]]";

// TODO: add `ParameterizedClassType` to `IsBuiltinWithTrivialDestruction`.

class GenericClass(T: type) {}
alias ParameterizedClassType = GenericClass;

fn Test() {
  //@dump-sem-ir-begin
  // ParameterizedClassType as Core.SubobjectDestroy;
  // ParameterizedClassType as Core.Destroy;
  // ParameterizedClassType as Core.TrivialDestroy;
  //
  // var unused x: ParameterizedClassType = Make();
  //@dump-sem-ir-end
}

// --- todo_trivial_unbound_element.carbon
library "[[@TEST_NAME]]";

// TODO: add `UnboundElementType` to `IsBuiltinWithTrivialDestruction`.

class HasTrivialDestruction {}

class C {
  var x: HasTrivialDestruction;
}

alias UnboundElementType = typeof(C.x);

fn Test() {
  //@dump-sem-ir-begin
  // UnboundElementType as Core.SubobjectDestroy;
  // UnboundElementType as Core.Destroy;
  // UnboundElementType as Core.TrivialDestroy;
  //
  // var unused x: UnboundElementType = Make();
  //@dump-sem-ir-end
}

// --- trivial_trivial_tuple_access.carbon
library "[[@TEST_NAME]]";

alias Tuple = (type, type);
alias TupleAccessType = typeof(Tuple.0);

fn Make() -> TupleAccessType;

fn Test() {
  //@dump-sem-ir-begin
  TupleAccessType as Core.SubobjectDestroy;
  TupleAccessType as Core.Destroy;
  TupleAccessType as Core.TrivialDestroy;

  var unused x: TupleAccessType = Make();
  //@dump-sem-ir-end
}

// --- trivial_class_type.carbon
library "[[@TEST_NAME]]";

class C;
alias ClassType = typeof(C);

fn Make() -> ClassType;

fn Test() {
  //@dump-sem-ir-begin
  ClassType as Core.SubobjectDestroy;
  ClassType as Core.Destroy;
  ClassType as Core.TrivialDestroy;

  var unused x: ClassType = Make();
  //@dump-sem-ir-end
}

// CHECK:STDOUT: --- todo_trivial_associated_entity.carbon = $todo_trivial_associated_entity
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Test() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- todo_trivial_generic_interface.carbon = $todo_trivial_generic_interface
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Test() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- todo_trivial_generic_named_constraint.carbon = $todo_trivial_generic_named_constraint
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Test() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- todo_trivial_namespace.carbon = $todo_trivial_namespace
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Test() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- todo_trivial_parameterised_class.carbon = $todo_trivial_parameterised_class
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Test() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- todo_trivial_unbound_element.carbon = $todo_trivial_unbound_element
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Test() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- trivial_trivial_tuple_access.carbon = $trivial_trivial_tuple_access
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %pattern_type.9a587c.1: type = pattern_type type [concrete]
// CHECK:STDOUT:   %Make.type: type = fn_type @Make [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Make: %Make.type = struct_value () [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.687: type = fn_type @SubobjectDestroy.WithSelf.Op.loc10 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f: %SubobjectDestroy.WithSelf.Op.type.687 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.1f5: %SubobjectDestroy.type = facet_value type, (%custom_witness.b80) [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %T.09f: %SubobjectDestroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.type.dea: type = fn_type @T.as_type.as.DestroyImpl.impl.Op, @T.as_type.as.DestroyImpl.impl(%T.09f) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.6e5: %T.as_type.as.DestroyImpl.impl.Op.type.dea = struct_value () [symbolic]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.a54: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.1f5) [concrete]
// CHECK:STDOUT:   %Destroy.facet: %Destroy.type = facet_value type, (%custom_witness.b80, %DestroyImpl.impl_witness.a54) [concrete]
// CHECK:STDOUT:   %x.patt: %pattern_type.9a587c.1 = ref_binding_pattern x [concrete]
// CHECK:STDOUT:   %x.var_patt: %pattern_type.9a587c.1 = var_pattern %x.patt [concrete]
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
// CHECK:STDOUT:   %TupleAccessType.ref.loc10: type = name_ref TupleAccessType, file.%TupleAccessType [concrete = type]
// CHECK:STDOUT:   %Core.ref.loc10: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:   %SubobjectDestroy.ref: type = name_ref SubobjectDestroy, imports.%Core.SubobjectDestroy [concrete = constants.%SubobjectDestroy.type]
// CHECK:STDOUT:   %SubobjectDestroy.facet: %SubobjectDestroy.type = facet_value %TupleAccessType.ref.loc10, (constants.%custom_witness.b80) [concrete = constants.%SubobjectDestroy.facet.1f5]
// CHECK:STDOUT:   %.loc10: %SubobjectDestroy.type = converted %TupleAccessType.ref.loc10, %SubobjectDestroy.facet [concrete = constants.%SubobjectDestroy.facet.1f5]
// CHECK:STDOUT:   %TupleAccessType.ref.loc11: type = name_ref TupleAccessType, file.%TupleAccessType [concrete = type]
// CHECK:STDOUT:   %Core.ref.loc11: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:   %Destroy.ref: type = name_ref Destroy, imports.%Core.Destroy [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Destroy.facet.loc11: %Destroy.type = facet_value %TupleAccessType.ref.loc11, (constants.%custom_witness.b80, constants.%DestroyImpl.impl_witness.a54) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc11: %Destroy.type = converted %TupleAccessType.ref.loc11, %Destroy.facet.loc11 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %TupleAccessType.ref.loc12: type = name_ref TupleAccessType, file.%TupleAccessType [concrete = type]
// CHECK:STDOUT:   %Core.ref.loc12: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:   %TrivialDestroy.ref: type = name_ref TrivialDestroy, imports.%Core.TrivialDestroy [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Destroy.facet.loc12: %Destroy.type = facet_value %TupleAccessType.ref.loc12, (constants.%custom_witness.b80, constants.%DestroyImpl.impl_witness.a54) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc12: %Destroy.type = converted %TupleAccessType.ref.loc12, %Destroy.facet.loc12 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %x.var: ref type = var_storage %x.var_patt
// CHECK:STDOUT:   %Make.ref: %Make.type = name_ref Make, file.%Make.decl [concrete = constants.%Make]
// CHECK:STDOUT:   %Make.call: init type = call %Make.ref()
// CHECK:STDOUT:   assign %x.var, %Make.call
// CHECK:STDOUT:   %TupleAccessType.ref.loc14: type = name_ref TupleAccessType, file.%TupleAccessType [concrete = type]
// CHECK:STDOUT:   %x: ref type = wrapper_binding x, %x.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %pattern_type.9a587c.1 = ref_binding_pattern x [concrete = constants.%x.patt]
// CHECK:STDOUT:     %x.var_patt: %pattern_type.9a587c.1 = var_pattern %x.patt [concrete = constants.%x.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.facet.loc14_3.1: %Destroy.type = facet_value type, (constants.%custom_witness.b80, constants.%DestroyImpl.impl_witness.a54) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc14_3.1: %Destroy.type = converted type, %Destroy.facet.loc14_3.1 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %Destroy.facet.loc14_3.2: %Destroy.type = facet_value type, (constants.%custom_witness.b80, constants.%DestroyImpl.impl_witness.a54) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc14_3.2: %Destroy.type = converted type, %Destroy.facet.loc14_3.2 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%x.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc10(%self.param: ref type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: --- trivial_class_type.carbon = $trivial_class_type
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %pattern_type.9a587c.1: type = pattern_type type [concrete]
// CHECK:STDOUT:   %Make.type: type = fn_type @Make [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Make: %Make.type = struct_value () [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.687: type = fn_type @SubobjectDestroy.WithSelf.Op.loc10 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f: %SubobjectDestroy.WithSelf.Op.type.687 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.1f5: %SubobjectDestroy.type = facet_value type, (%custom_witness.b80) [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %T.09f: %SubobjectDestroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.type.dea: type = fn_type @T.as_type.as.DestroyImpl.impl.Op, @T.as_type.as.DestroyImpl.impl(%T.09f) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.6e5: %T.as_type.as.DestroyImpl.impl.Op.type.dea = struct_value () [symbolic]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.a54: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.1f5) [concrete]
// CHECK:STDOUT:   %Destroy.facet: %Destroy.type = facet_value type, (%custom_witness.b80, %DestroyImpl.impl_witness.a54) [concrete]
// CHECK:STDOUT:   %x.patt: %pattern_type.9a587c.1 = ref_binding_pattern x [concrete]
// CHECK:STDOUT:   %x.var_patt: %pattern_type.9a587c.1 = var_pattern %x.patt [concrete]
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
// CHECK:STDOUT:   %ClassType.ref.loc10: type = name_ref ClassType, file.%ClassType [concrete = type]
// CHECK:STDOUT:   %Core.ref.loc10: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:   %SubobjectDestroy.ref: type = name_ref SubobjectDestroy, imports.%Core.SubobjectDestroy [concrete = constants.%SubobjectDestroy.type]
// CHECK:STDOUT:   %SubobjectDestroy.facet: %SubobjectDestroy.type = facet_value %ClassType.ref.loc10, (constants.%custom_witness.b80) [concrete = constants.%SubobjectDestroy.facet.1f5]
// CHECK:STDOUT:   %.loc10: %SubobjectDestroy.type = converted %ClassType.ref.loc10, %SubobjectDestroy.facet [concrete = constants.%SubobjectDestroy.facet.1f5]
// CHECK:STDOUT:   %ClassType.ref.loc11: type = name_ref ClassType, file.%ClassType [concrete = type]
// CHECK:STDOUT:   %Core.ref.loc11: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:   %Destroy.ref: type = name_ref Destroy, imports.%Core.Destroy [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Destroy.facet.loc11: %Destroy.type = facet_value %ClassType.ref.loc11, (constants.%custom_witness.b80, constants.%DestroyImpl.impl_witness.a54) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc11: %Destroy.type = converted %ClassType.ref.loc11, %Destroy.facet.loc11 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %ClassType.ref.loc12: type = name_ref ClassType, file.%ClassType [concrete = type]
// CHECK:STDOUT:   %Core.ref.loc12: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:   %TrivialDestroy.ref: type = name_ref TrivialDestroy, imports.%Core.TrivialDestroy [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Destroy.facet.loc12: %Destroy.type = facet_value %ClassType.ref.loc12, (constants.%custom_witness.b80, constants.%DestroyImpl.impl_witness.a54) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc12: %Destroy.type = converted %ClassType.ref.loc12, %Destroy.facet.loc12 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %x.var: ref type = var_storage %x.var_patt
// CHECK:STDOUT:   %Make.ref: %Make.type = name_ref Make, file.%Make.decl [concrete = constants.%Make]
// CHECK:STDOUT:   %Make.call: init type = call %Make.ref()
// CHECK:STDOUT:   assign %x.var, %Make.call
// CHECK:STDOUT:   %ClassType.ref.loc14: type = name_ref ClassType, file.%ClassType [concrete = type]
// CHECK:STDOUT:   %x: ref type = wrapper_binding x, %x.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %pattern_type.9a587c.1 = ref_binding_pattern x [concrete = constants.%x.patt]
// CHECK:STDOUT:     %x.var_patt: %pattern_type.9a587c.1 = var_pattern %x.patt [concrete = constants.%x.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.facet.loc14_3.1: %Destroy.type = facet_value type, (constants.%custom_witness.b80, constants.%DestroyImpl.impl_witness.a54) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc14_3.1: %Destroy.type = converted type, %Destroy.facet.loc14_3.1 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %Destroy.facet.loc14_3.2: %Destroy.type = facet_value type, (constants.%custom_witness.b80, constants.%DestroyImpl.impl_witness.a54) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc14_3.2: %Destroy.type = converted type, %Destroy.facet.loc14_3.2 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%x.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc10(%self.param: ref type) = "no_op";
// CHECK:STDOUT:
