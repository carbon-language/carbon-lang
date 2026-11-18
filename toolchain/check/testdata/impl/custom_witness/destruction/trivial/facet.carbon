// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/destroy.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/impl/custom_witness/destruction/trivial/facet.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/impl/custom_witness/destruction/trivial/facet.carbon

// --- trivial_constrained_facet.carbon
library "[[@TEST_NAME]]";

interface Extra {}

alias Facet = Core.TrivialDestroy & Extra;
fn Make() -> Facet;

fn Test() {
  //@dump-sem-ir-begin
  Facet as Core.SubobjectDestroy;
  Facet as Core.Destroy;
  Facet as Core.TrivialDestroy;

  var unused x: Facet = Make();
  //@dump-sem-ir-end
}

// --- trivial_generic_class.carbon
library "[[@TEST_NAME]]";

class GenericClass(T: Core.TrivialDestroy) {
  var x: T;
}

alias Facet = GenericClass(Core.TrivialDestroy);
fn Make() -> Facet;

fn Test() {
  //@dump-sem-ir-begin
  Facet as Core.SubobjectDestroy;
  Facet as Core.Destroy;
  Facet as Core.TrivialDestroy;

  var unused x: Facet = Make();
  //@dump-sem-ir-end
}

// --- trivial_struct.carbon
library "[[@TEST_NAME]]";

interface Extra {}

alias Struct = {.a: Core.TrivialDestroy & Extra};
fn Make() -> Struct;

fn Test() {
  //@dump-sem-ir-begin
  Struct as Core.SubobjectDestroy;
  Struct as Core.Destroy;
  Struct as Core.TrivialDestroy;

  var unused x: Struct = Make();
  //@dump-sem-ir-end
}

// --- trivial_tuple.carbon
library "[[@TEST_NAME]]";

interface Extra {}
class HasTrivialDestruction {}

alias Tuple = (Core.TrivialDestroy & Extra, HasTrivialDestruction);
fn Make() -> Tuple;

fn Test() {
  //@dump-sem-ir-begin
  Tuple as Core.SubobjectDestroy;
  Tuple as Core.Destroy;
  Tuple as Core.TrivialDestroy;

  var unused x: Tuple = Make();
  //@dump-sem-ir-end
}

// CHECK:STDOUT: --- trivial_constrained_facet.carbon = $trivial_constrained_facet
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %BitAndWith.type.a06: type = generic_interface_type @BitAndWith [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %BitAndWith.generic: %BitAndWith.type.a06 = struct_value () [concrete]
// CHECK:STDOUT:   %facet_type.3be: type = facet_type <@Extra & @Destroy> [concrete]
// CHECK:STDOUT:   %pattern_type.59e: type = pattern_type %facet_type.3be [concrete]
// CHECK:STDOUT:   %Make.type: type = fn_type @Make [concrete]
// CHECK:STDOUT:   %Make: %Make.type = struct_value () [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.687: type = fn_type @SubobjectDestroy.WithSelf.Op.loc10 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f: %SubobjectDestroy.WithSelf.Op.type.687 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.ca6: %SubobjectDestroy.type = facet_value %facet_type.3be, (%custom_witness.b80) [concrete]
// CHECK:STDOUT:   %T.09f: %SubobjectDestroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.type.dea: type = fn_type @T.as_type.as.DestroyImpl.impl.Op, @T.as_type.as.DestroyImpl.impl(%T.09f) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.6e5: %T.as_type.as.DestroyImpl.impl.Op.type.dea = struct_value () [symbolic]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.99b: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.ca6) [concrete]
// CHECK:STDOUT:   %Destroy.facet: %Destroy.type = facet_value %facet_type.3be, (%DestroyImpl.impl_witness.99b, %custom_witness.b80) [concrete]
// CHECK:STDOUT:   %x.patt: %pattern_type.59e = ref_binding_pattern x [concrete]
// CHECK:STDOUT:   %x.var_patt: %pattern_type.59e = var_pattern %x.patt [concrete]
// CHECK:STDOUT:   %SelfDestruct.type: type = fn_type @SelfDestruct [concrete]
// CHECK:STDOUT:   %SelfDestruct: %SelfDestruct.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .TrivialDestroy = %Core.TrivialDestroy
// CHECK:STDOUT:     .BitAndWith = %Core.BitAndWith
// CHECK:STDOUT:     .SubobjectDestroy = %Core.SubobjectDestroy
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     .SelfDestruct = %Core.SelfDestruct
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.TrivialDestroy: type = import_ref Core//prelude/parts/destroy, TrivialDestroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Core.BitAndWith: %BitAndWith.type.a06 = import_ref Core//prelude/parts/as, BitAndWith, loaded [concrete = constants.%BitAndWith.generic]
// CHECK:STDOUT:   %Core.SubobjectDestroy: type = import_ref Core//prelude/parts/destroy, SubobjectDestroy, loaded [concrete = constants.%SubobjectDestroy.type]
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Core.import_ref.74d: @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op.type (%T.as_type.as.DestroyImpl.impl.Op.type.dea) = import_ref Core//prelude/parts/destroy, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op (constants.%T.as_type.as.DestroyImpl.impl.Op.6e5)]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table = impl_witness_table (%Core.import_ref.74d), @T.as_type.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT:   %Core.SelfDestruct: %SelfDestruct.type = import_ref Core//prelude/parts/destroy, SelfDestruct, loaded [concrete = constants.%SelfDestruct]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Test() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Facet.ref.loc10: type = name_ref Facet, file.%Facet [concrete = constants.%facet_type.3be]
// CHECK:STDOUT:   %Core.ref.loc10: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:   %SubobjectDestroy.ref: type = name_ref SubobjectDestroy, imports.%Core.SubobjectDestroy [concrete = constants.%SubobjectDestroy.type]
// CHECK:STDOUT:   %SubobjectDestroy.facet: %SubobjectDestroy.type = facet_value %Facet.ref.loc10, (constants.%custom_witness.b80) [concrete = constants.%SubobjectDestroy.facet.ca6]
// CHECK:STDOUT:   %.loc10: %SubobjectDestroy.type = converted %Facet.ref.loc10, %SubobjectDestroy.facet [concrete = constants.%SubobjectDestroy.facet.ca6]
// CHECK:STDOUT:   %Facet.ref.loc11: type = name_ref Facet, file.%Facet [concrete = constants.%facet_type.3be]
// CHECK:STDOUT:   %Core.ref.loc11: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:   %Destroy.ref: type = name_ref Destroy, imports.%Core.Destroy [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Destroy.facet.loc11: %Destroy.type = facet_value %Facet.ref.loc11, (constants.%DestroyImpl.impl_witness.99b, constants.%custom_witness.b80) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc11: %Destroy.type = converted %Facet.ref.loc11, %Destroy.facet.loc11 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %Facet.ref.loc12: type = name_ref Facet, file.%Facet [concrete = constants.%facet_type.3be]
// CHECK:STDOUT:   %Core.ref.loc12: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:   %TrivialDestroy.ref: type = name_ref TrivialDestroy, imports.%Core.TrivialDestroy [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Destroy.facet.loc12: %Destroy.type = facet_value %Facet.ref.loc12, (constants.%DestroyImpl.impl_witness.99b, constants.%custom_witness.b80) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc12: %Destroy.type = converted %Facet.ref.loc12, %Destroy.facet.loc12 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %x.var: ref %facet_type.3be = var_storage %x.var_patt
// CHECK:STDOUT:   %Make.ref: %Make.type = name_ref Make, file.%Make.decl [concrete = constants.%Make]
// CHECK:STDOUT:   %Make.call: init %facet_type.3be = call %Make.ref()
// CHECK:STDOUT:   assign %x.var, %Make.call
// CHECK:STDOUT:   %Facet.ref.loc14: type = name_ref Facet, file.%Facet [concrete = constants.%facet_type.3be]
// CHECK:STDOUT:   %x: ref %facet_type.3be = wrapper_binding x, %x.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %pattern_type.59e = ref_binding_pattern x [concrete = constants.%x.patt]
// CHECK:STDOUT:     %x.var_patt: %pattern_type.59e = var_pattern %x.patt [concrete = constants.%x.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.facet.loc14_3.1: %Destroy.type = facet_value constants.%facet_type.3be, (constants.%DestroyImpl.impl_witness.99b, constants.%custom_witness.b80) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc14_3.1: %Destroy.type = converted constants.%facet_type.3be, %Destroy.facet.loc14_3.1 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %Destroy.facet.loc14_3.2: %Destroy.type = facet_value constants.%facet_type.3be, (constants.%DestroyImpl.impl_witness.99b, constants.%custom_witness.b80) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc14_3.2: %Destroy.type = converted constants.%facet_type.3be, %Destroy.facet.loc14_3.2 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%x.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc10(%self.param: ref %facet_type.3be) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: --- trivial_generic_class.carbon = $trivial_generic_class
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %T.09f: %SubobjectDestroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.type.dea: type = fn_type @T.as_type.as.DestroyImpl.impl.Op, @T.as_type.as.DestroyImpl.impl(%T.09f) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.6e5: %T.as_type.as.DestroyImpl.impl.Op.type.dea = struct_value () [symbolic]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.1: type = fn_type @SubobjectDestroy.WithSelf.Op.loc7 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.1: %SubobjectDestroy.WithSelf.Op.type.6870fd.1 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.1: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.1), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.1f5: %SubobjectDestroy.type = facet_value %Destroy.type, (%custom_witness.b80ef6.1) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.a54: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.1f5) [concrete]
// CHECK:STDOUT:   %Destroy.facet.029: %Destroy.type = facet_value %Destroy.type, (%DestroyImpl.impl_witness.a54, %custom_witness.b80ef6.1) [concrete]
// CHECK:STDOUT:   %GenericClass.17a: type = class_type @GenericClass, @GenericClass(%Destroy.facet.029) [concrete]
// CHECK:STDOUT:   %pattern_type.07c: type = pattern_type %GenericClass.17a [concrete]
// CHECK:STDOUT:   %Make.type: type = fn_type @Make [concrete]
// CHECK:STDOUT:   %Make: %Make.type = struct_value () [concrete]
// CHECK:STDOUT:   %GenericClass.elem.9f1: type = unbound_element_type %GenericClass.17a, %Destroy.type [concrete]
// CHECK:STDOUT:   %struct_type.x.cab: type = struct_type {.x: %Destroy.type} [concrete]
// CHECK:STDOUT:   %SelfDestruct.type: type = fn_type @SelfDestruct [concrete]
// CHECK:STDOUT:   %SelfDestruct: %SelfDestruct.type = struct_value () [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.3: type = fn_type @SubobjectDestroy.WithSelf.Op.loc12_9.2 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.3: %SubobjectDestroy.WithSelf.Op.type.6870fd.3 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.3: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.3), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.e1f: %SubobjectDestroy.type = facet_value %GenericClass.17a, (%custom_witness.b80ef6.3) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.561: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.e1f) [concrete]
// CHECK:STDOUT:   %Destroy.facet.e70: %Destroy.type = facet_value %GenericClass.17a, (%DestroyImpl.impl_witness.561, %custom_witness.b80ef6.3) [concrete]
// CHECK:STDOUT:   %x.patt: %pattern_type.07c = ref_binding_pattern x [concrete]
// CHECK:STDOUT:   %x.var_patt: %pattern_type.07c = var_pattern %x.patt [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .TrivialDestroy = %Core.TrivialDestroy
// CHECK:STDOUT:     .SubobjectDestroy = %Core.SubobjectDestroy
// CHECK:STDOUT:     .SelfDestruct = %Core.SelfDestruct
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.TrivialDestroy: type = import_ref Core//prelude/parts/destroy, TrivialDestroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Core.import_ref.74d: @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op.type (%T.as_type.as.DestroyImpl.impl.Op.type.dea) = import_ref Core//prelude/parts/destroy, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op (constants.%T.as_type.as.DestroyImpl.impl.Op.6e5)]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table = impl_witness_table (%Core.import_ref.74d), @T.as_type.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT:   %Core.SubobjectDestroy: type = import_ref Core//prelude/parts/destroy, SubobjectDestroy, loaded [concrete = constants.%SubobjectDestroy.type]
// CHECK:STDOUT:   %Core.SelfDestruct: %SelfDestruct.type = import_ref Core//prelude/parts/destroy, SelfDestruct, loaded [concrete = constants.%SelfDestruct]
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Test() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Facet.ref.loc12: type = name_ref Facet, file.%Facet [concrete = constants.%GenericClass.17a]
// CHECK:STDOUT:   %Core.ref.loc12: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:   %SubobjectDestroy.ref: type = name_ref SubobjectDestroy, imports.%Core.SubobjectDestroy [concrete = constants.%SubobjectDestroy.type]
// CHECK:STDOUT:   %SubobjectDestroy.facet: %SubobjectDestroy.type = facet_value %Facet.ref.loc12, (constants.%custom_witness.b80ef6.3) [concrete = constants.%SubobjectDestroy.facet.e1f]
// CHECK:STDOUT:   %.loc12: %SubobjectDestroy.type = converted %Facet.ref.loc12, %SubobjectDestroy.facet [concrete = constants.%SubobjectDestroy.facet.e1f]
// CHECK:STDOUT:   %Facet.ref.loc13: type = name_ref Facet, file.%Facet [concrete = constants.%GenericClass.17a]
// CHECK:STDOUT:   %Core.ref.loc13: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:   %Destroy.ref: type = name_ref Destroy, imports.%Core.Destroy [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Destroy.facet.loc13: %Destroy.type = facet_value %Facet.ref.loc13, (constants.%DestroyImpl.impl_witness.561, constants.%custom_witness.b80ef6.3) [concrete = constants.%Destroy.facet.e70]
// CHECK:STDOUT:   %.loc13: %Destroy.type = converted %Facet.ref.loc13, %Destroy.facet.loc13 [concrete = constants.%Destroy.facet.e70]
// CHECK:STDOUT:   %Facet.ref.loc14: type = name_ref Facet, file.%Facet [concrete = constants.%GenericClass.17a]
// CHECK:STDOUT:   %Core.ref.loc14: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:   %TrivialDestroy.ref: type = name_ref TrivialDestroy, imports.%Core.TrivialDestroy [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Destroy.facet.loc14: %Destroy.type = facet_value %Facet.ref.loc14, (constants.%DestroyImpl.impl_witness.561, constants.%custom_witness.b80ef6.3) [concrete = constants.%Destroy.facet.e70]
// CHECK:STDOUT:   %.loc14: %Destroy.type = converted %Facet.ref.loc14, %Destroy.facet.loc14 [concrete = constants.%Destroy.facet.e70]
// CHECK:STDOUT:   %x.var: ref %GenericClass.17a = var_storage %x.var_patt
// CHECK:STDOUT:   %Make.ref: %Make.type = name_ref Make, file.%Make.decl [concrete = constants.%Make]
// CHECK:STDOUT:   %.loc16_3.1: ref %GenericClass.17a = splice_block %x.var {}
// CHECK:STDOUT:   %Make.call: init %GenericClass.17a to %.loc16_3.1 = call %Make.ref()
// CHECK:STDOUT:   assign %x.var, %Make.call
// CHECK:STDOUT:   %Facet.ref.loc16: type = name_ref Facet, file.%Facet [concrete = constants.%GenericClass.17a]
// CHECK:STDOUT:   %x: ref %GenericClass.17a = wrapper_binding x, %x.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %pattern_type.07c = ref_binding_pattern x [concrete = constants.%x.patt]
// CHECK:STDOUT:     %x.var_patt: %pattern_type.07c = var_pattern %x.patt [concrete = constants.%x.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.facet.loc16_3.1: %Destroy.type = facet_value constants.%GenericClass.17a, (constants.%DestroyImpl.impl_witness.561, constants.%custom_witness.b80ef6.3) [concrete = constants.%Destroy.facet.e70]
// CHECK:STDOUT:   %.loc16_3.2: %Destroy.type = converted constants.%GenericClass.17a, %Destroy.facet.loc16_3.1 [concrete = constants.%Destroy.facet.e70]
// CHECK:STDOUT:   %Destroy.facet.loc16_3.2: %Destroy.type = facet_value constants.%GenericClass.17a, (constants.%DestroyImpl.impl_witness.561, constants.%custom_witness.b80ef6.3) [concrete = constants.%Destroy.facet.e70]
// CHECK:STDOUT:   %.loc16_3.3: %Destroy.type = converted constants.%GenericClass.17a, %Destroy.facet.loc16_3.2 [concrete = constants.%Destroy.facet.e70]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%x.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc12_9.1(%self.param: ref %struct_type.x.cab) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc12_9.1: ref %Destroy.type = struct_access %self.param, element0
// CHECK:STDOUT:   %Destroy.facet.loc12_9.1: %Destroy.type = facet_value constants.%Destroy.type, (constants.%DestroyImpl.impl_witness.a54, constants.%custom_witness.b80ef6.1) [concrete = constants.%Destroy.facet.029]
// CHECK:STDOUT:   %.loc12_9.2: %Destroy.type = converted constants.%Destroy.type, %Destroy.facet.loc12_9.1 [concrete = constants.%Destroy.facet.029]
// CHECK:STDOUT:   %Destroy.facet.loc12_9.2: %Destroy.type = facet_value constants.%Destroy.type, (constants.%DestroyImpl.impl_witness.a54, constants.%custom_witness.b80ef6.1) [concrete = constants.%Destroy.facet.029]
// CHECK:STDOUT:   %.loc12_9.3: %Destroy.type = converted constants.%Destroy.type, %Destroy.facet.loc12_9.2 [concrete = constants.%Destroy.facet.029]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%.loc12_9.1)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc12_9.2(%self.param: ref %GenericClass.17a) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %x.ref: %GenericClass.elem.9f1 = name_ref x, @GenericClass.%field_decl [concrete = @GenericClass.%field_decl]
// CHECK:STDOUT:   %.loc12_9.1: ref %Destroy.type = class_element_access %self.param, element0
// CHECK:STDOUT:   %Destroy.facet.loc12_9.1: %Destroy.type = facet_value constants.%Destroy.type, (constants.%DestroyImpl.impl_witness.a54, constants.%custom_witness.b80ef6.1) [concrete = constants.%Destroy.facet.029]
// CHECK:STDOUT:   %.loc12_9.2: %Destroy.type = converted constants.%Destroy.type, %Destroy.facet.loc12_9.1 [concrete = constants.%Destroy.facet.029]
// CHECK:STDOUT:   %Destroy.facet.loc12_9.2: %Destroy.type = facet_value constants.%Destroy.type, (constants.%DestroyImpl.impl_witness.a54, constants.%custom_witness.b80ef6.1) [concrete = constants.%Destroy.facet.029]
// CHECK:STDOUT:   %.loc12_9.3: %Destroy.type = converted constants.%Destroy.type, %Destroy.facet.loc12_9.2 [concrete = constants.%Destroy.facet.029]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%.loc12_9.1)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- trivial_struct.carbon = $trivial_struct
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %BitAndWith.type.a06: type = generic_interface_type @BitAndWith [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %BitAndWith.generic: %BitAndWith.type.a06 = struct_value () [concrete]
// CHECK:STDOUT:   %facet_type.3be: type = facet_type <@Extra & @Destroy> [concrete]
// CHECK:STDOUT:   %struct_type.a: type = struct_type {.a: %facet_type.3be} [concrete]
// CHECK:STDOUT:   %pattern_type.c75: type = pattern_type %struct_type.a [concrete]
// CHECK:STDOUT:   %Make.type: type = fn_type @Make [concrete]
// CHECK:STDOUT:   %Make: %Make.type = struct_value () [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.1: type = fn_type @SubobjectDestroy.WithSelf.Op.loc10_10.1 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.1: %SubobjectDestroy.WithSelf.Op.type.6870fd.1 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.1: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.1), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.2: type = fn_type @SubobjectDestroy.WithSelf.Op.loc10_10.2 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.2: %SubobjectDestroy.WithSelf.Op.type.6870fd.2 = struct_value () [concrete]
// CHECK:STDOUT:   %SelfDestruct.type: type = fn_type @SelfDestruct [concrete]
// CHECK:STDOUT:   %SelfDestruct: %SelfDestruct.type = struct_value () [concrete]
// CHECK:STDOUT:   %T.09f: %SubobjectDestroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.type.dea: type = fn_type @T.as_type.as.DestroyImpl.impl.Op, @T.as_type.as.DestroyImpl.impl(%T.09f) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.6e5: %T.as_type.as.DestroyImpl.impl.Op.type.dea = struct_value () [symbolic]
// CHECK:STDOUT:   %SubobjectDestroy.facet.ca6: %SubobjectDestroy.type = facet_value %facet_type.3be, (%custom_witness.b80ef6.1) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.99b: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.ca6) [concrete]
// CHECK:STDOUT:   %Destroy.facet.295: %Destroy.type = facet_value %facet_type.3be, (%DestroyImpl.impl_witness.99b, %custom_witness.b80ef6.1) [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.2: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.2), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.503: %SubobjectDestroy.type = facet_value %struct_type.a, (%custom_witness.b80ef6.2) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.fbd: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.503) [concrete]
// CHECK:STDOUT:   %Destroy.facet.5b4: %Destroy.type = facet_value %struct_type.a, (%DestroyImpl.impl_witness.fbd, %custom_witness.b80ef6.2) [concrete]
// CHECK:STDOUT:   %x.patt: %pattern_type.c75 = ref_binding_pattern x [concrete]
// CHECK:STDOUT:   %x.var_patt: %pattern_type.c75 = var_pattern %x.patt [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .TrivialDestroy = %Core.TrivialDestroy
// CHECK:STDOUT:     .BitAndWith = %Core.BitAndWith
// CHECK:STDOUT:     .SubobjectDestroy = %Core.SubobjectDestroy
// CHECK:STDOUT:     .SelfDestruct = %Core.SelfDestruct
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.TrivialDestroy: type = import_ref Core//prelude/parts/destroy, TrivialDestroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Core.BitAndWith: %BitAndWith.type.a06 = import_ref Core//prelude/parts/as, BitAndWith, loaded [concrete = constants.%BitAndWith.generic]
// CHECK:STDOUT:   %Core.SubobjectDestroy: type = import_ref Core//prelude/parts/destroy, SubobjectDestroy, loaded [concrete = constants.%SubobjectDestroy.type]
// CHECK:STDOUT:   %Core.SelfDestruct: %SelfDestruct.type = import_ref Core//prelude/parts/destroy, SelfDestruct, loaded [concrete = constants.%SelfDestruct]
// CHECK:STDOUT:   %Core.import_ref.74d: @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op.type (%T.as_type.as.DestroyImpl.impl.Op.type.dea) = import_ref Core//prelude/parts/destroy, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op (constants.%T.as_type.as.DestroyImpl.impl.Op.6e5)]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table = impl_witness_table (%Core.import_ref.74d), @T.as_type.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Test() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Struct.ref.loc10: type = name_ref Struct, file.%Struct [concrete = constants.%struct_type.a]
// CHECK:STDOUT:   %Core.ref.loc10: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:   %SubobjectDestroy.ref: type = name_ref SubobjectDestroy, imports.%Core.SubobjectDestroy [concrete = constants.%SubobjectDestroy.type]
// CHECK:STDOUT:   %SubobjectDestroy.facet: %SubobjectDestroy.type = facet_value %Struct.ref.loc10, (constants.%custom_witness.b80ef6.2) [concrete = constants.%SubobjectDestroy.facet.503]
// CHECK:STDOUT:   %.loc10: %SubobjectDestroy.type = converted %Struct.ref.loc10, %SubobjectDestroy.facet [concrete = constants.%SubobjectDestroy.facet.503]
// CHECK:STDOUT:   %Struct.ref.loc11: type = name_ref Struct, file.%Struct [concrete = constants.%struct_type.a]
// CHECK:STDOUT:   %Core.ref.loc11: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:   %Destroy.ref: type = name_ref Destroy, imports.%Core.Destroy [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Destroy.facet.loc11: %Destroy.type = facet_value %Struct.ref.loc11, (constants.%DestroyImpl.impl_witness.fbd, constants.%custom_witness.b80ef6.2) [concrete = constants.%Destroy.facet.5b4]
// CHECK:STDOUT:   %.loc11: %Destroy.type = converted %Struct.ref.loc11, %Destroy.facet.loc11 [concrete = constants.%Destroy.facet.5b4]
// CHECK:STDOUT:   %Struct.ref.loc12: type = name_ref Struct, file.%Struct [concrete = constants.%struct_type.a]
// CHECK:STDOUT:   %Core.ref.loc12: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:   %TrivialDestroy.ref: type = name_ref TrivialDestroy, imports.%Core.TrivialDestroy [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Destroy.facet.loc12: %Destroy.type = facet_value %Struct.ref.loc12, (constants.%DestroyImpl.impl_witness.fbd, constants.%custom_witness.b80ef6.2) [concrete = constants.%Destroy.facet.5b4]
// CHECK:STDOUT:   %.loc12: %Destroy.type = converted %Struct.ref.loc12, %Destroy.facet.loc12 [concrete = constants.%Destroy.facet.5b4]
// CHECK:STDOUT:   %x.var: ref %struct_type.a = var_storage %x.var_patt
// CHECK:STDOUT:   %Make.ref: %Make.type = name_ref Make, file.%Make.decl [concrete = constants.%Make]
// CHECK:STDOUT:   %Make.call: init %struct_type.a = call %Make.ref()
// CHECK:STDOUT:   assign %x.var, %Make.call
// CHECK:STDOUT:   %Struct.ref.loc14: type = name_ref Struct, file.%Struct [concrete = constants.%struct_type.a]
// CHECK:STDOUT:   %x: ref %struct_type.a = wrapper_binding x, %x.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %pattern_type.c75 = ref_binding_pattern x [concrete = constants.%x.patt]
// CHECK:STDOUT:     %x.var_patt: %pattern_type.c75 = var_pattern %x.patt [concrete = constants.%x.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.facet.loc14_3.1: %Destroy.type = facet_value constants.%struct_type.a, (constants.%DestroyImpl.impl_witness.fbd, constants.%custom_witness.b80ef6.2) [concrete = constants.%Destroy.facet.5b4]
// CHECK:STDOUT:   %.loc14_3.1: %Destroy.type = converted constants.%struct_type.a, %Destroy.facet.loc14_3.1 [concrete = constants.%Destroy.facet.5b4]
// CHECK:STDOUT:   %Destroy.facet.loc14_3.2: %Destroy.type = facet_value constants.%struct_type.a, (constants.%DestroyImpl.impl_witness.fbd, constants.%custom_witness.b80ef6.2) [concrete = constants.%Destroy.facet.5b4]
// CHECK:STDOUT:   %.loc14_3.2: %Destroy.type = converted constants.%struct_type.a, %Destroy.facet.loc14_3.2 [concrete = constants.%Destroy.facet.5b4]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%x.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc10_10.1(%self.param: ref %facet_type.3be) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc10_10.2(%self.param: ref %struct_type.a) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc10_10.1: ref %facet_type.3be = struct_access %self.param, element0
// CHECK:STDOUT:   %Destroy.facet.loc10_10.1: %Destroy.type = facet_value constants.%facet_type.3be, (constants.%DestroyImpl.impl_witness.99b, constants.%custom_witness.b80ef6.1) [concrete = constants.%Destroy.facet.295]
// CHECK:STDOUT:   %.loc10_10.2: %Destroy.type = converted constants.%facet_type.3be, %Destroy.facet.loc10_10.1 [concrete = constants.%Destroy.facet.295]
// CHECK:STDOUT:   %Destroy.facet.loc10_10.2: %Destroy.type = facet_value constants.%facet_type.3be, (constants.%DestroyImpl.impl_witness.99b, constants.%custom_witness.b80ef6.1) [concrete = constants.%Destroy.facet.295]
// CHECK:STDOUT:   %.loc10_10.3: %Destroy.type = converted constants.%facet_type.3be, %Destroy.facet.loc10_10.2 [concrete = constants.%Destroy.facet.295]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%.loc10_10.1)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- trivial_tuple.carbon = $trivial_tuple
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %HasTrivialDestruction: type = class_type @HasTrivialDestruction [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %BitAndWith.type.a06: type = generic_interface_type @BitAndWith [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %BitAndWith.generic: %BitAndWith.type.a06 = struct_value () [concrete]
// CHECK:STDOUT:   %facet_type.3be: type = facet_type <@Extra & @Destroy> [concrete]
// CHECK:STDOUT:   %tuple.type.693: type = tuple_type (type, type) [concrete]
// CHECK:STDOUT:   %tuple: %tuple.type.693 = tuple_value (%facet_type.3be, %HasTrivialDestruction) [concrete]
// CHECK:STDOUT:   %tuple.type.494: type = tuple_type (%facet_type.3be, %HasTrivialDestruction) [concrete]
// CHECK:STDOUT:   %pattern_type.16f: type = pattern_type %tuple.type.494 [concrete]
// CHECK:STDOUT:   %Make.type: type = fn_type @Make [concrete]
// CHECK:STDOUT:   %Make: %Make.type = struct_value () [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.1: type = fn_type @SubobjectDestroy.WithSelf.Op.loc11_9.1 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.1: %SubobjectDestroy.WithSelf.Op.type.6870fd.1 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.1: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.1), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.3: type = fn_type @SubobjectDestroy.WithSelf.Op.loc11_9.3 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.3: %SubobjectDestroy.WithSelf.Op.type.6870fd.3 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.3: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.3), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.4: type = fn_type @SubobjectDestroy.WithSelf.Op.loc11_9.4 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.4: %SubobjectDestroy.WithSelf.Op.type.6870fd.4 = struct_value () [concrete]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %SelfDestruct.type: type = fn_type @SelfDestruct [concrete]
// CHECK:STDOUT:   %SelfDestruct: %SelfDestruct.type = struct_value () [concrete]
// CHECK:STDOUT:   %T.09f: %SubobjectDestroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.type.dea: type = fn_type @T.as_type.as.DestroyImpl.impl.Op, @T.as_type.as.DestroyImpl.impl(%T.09f) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.6e5: %T.as_type.as.DestroyImpl.impl.Op.type.dea = struct_value () [symbolic]
// CHECK:STDOUT:   %SubobjectDestroy.facet.5bc: %SubobjectDestroy.type = facet_value %HasTrivialDestruction, (%custom_witness.b80ef6.3) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.4ca: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.5bc) [concrete]
// CHECK:STDOUT:   %Destroy.facet.dba: %Destroy.type = facet_value %HasTrivialDestruction, (%DestroyImpl.impl_witness.4ca, %custom_witness.b80ef6.3) [concrete]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.ca6: %SubobjectDestroy.type = facet_value %facet_type.3be, (%custom_witness.b80ef6.1) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.99b: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.ca6) [concrete]
// CHECK:STDOUT:   %Destroy.facet.295: %Destroy.type = facet_value %facet_type.3be, (%DestroyImpl.impl_witness.99b, %custom_witness.b80ef6.1) [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.4: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.4), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.8a0: %SubobjectDestroy.type = facet_value %tuple.type.494, (%custom_witness.b80ef6.4) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.37c: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.8a0) [concrete]
// CHECK:STDOUT:   %Destroy.facet.646: %Destroy.type = facet_value %tuple.type.494, (%DestroyImpl.impl_witness.37c, %custom_witness.b80ef6.4) [concrete]
// CHECK:STDOUT:   %x.patt: %pattern_type.16f = ref_binding_pattern x [concrete]
// CHECK:STDOUT:   %x.var_patt: %pattern_type.16f = var_pattern %x.patt [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .TrivialDestroy = %Core.TrivialDestroy
// CHECK:STDOUT:     .BitAndWith = %Core.BitAndWith
// CHECK:STDOUT:     .SubobjectDestroy = %Core.SubobjectDestroy
// CHECK:STDOUT:     .SelfDestruct = %Core.SelfDestruct
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.TrivialDestroy: type = import_ref Core//prelude/parts/destroy, TrivialDestroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Core.BitAndWith: %BitAndWith.type.a06 = import_ref Core//prelude/parts/as, BitAndWith, loaded [concrete = constants.%BitAndWith.generic]
// CHECK:STDOUT:   %Core.SubobjectDestroy: type = import_ref Core//prelude/parts/destroy, SubobjectDestroy, loaded [concrete = constants.%SubobjectDestroy.type]
// CHECK:STDOUT:   %Core.SelfDestruct: %SelfDestruct.type = import_ref Core//prelude/parts/destroy, SelfDestruct, loaded [concrete = constants.%SelfDestruct]
// CHECK:STDOUT:   %Core.import_ref.74d: @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op.type (%T.as_type.as.DestroyImpl.impl.Op.type.dea) = import_ref Core//prelude/parts/destroy, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op (constants.%T.as_type.as.DestroyImpl.impl.Op.6e5)]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table = impl_witness_table (%Core.import_ref.74d), @T.as_type.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Test() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Tuple.ref.loc11: %tuple.type.693 = name_ref Tuple, file.%Tuple [concrete = constants.%tuple]
// CHECK:STDOUT:   %Core.ref.loc11: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:   %SubobjectDestroy.ref: type = name_ref SubobjectDestroy, imports.%Core.SubobjectDestroy [concrete = constants.%SubobjectDestroy.type]
// CHECK:STDOUT:   %SubobjectDestroy.facet: %SubobjectDestroy.type = facet_value constants.%tuple.type.494, (constants.%custom_witness.b80ef6.4) [concrete = constants.%SubobjectDestroy.facet.8a0]
// CHECK:STDOUT:   %.loc11: %SubobjectDestroy.type = converted %Tuple.ref.loc11, %SubobjectDestroy.facet [concrete = constants.%SubobjectDestroy.facet.8a0]
// CHECK:STDOUT:   %Tuple.ref.loc12: %tuple.type.693 = name_ref Tuple, file.%Tuple [concrete = constants.%tuple]
// CHECK:STDOUT:   %Core.ref.loc12: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:   %Destroy.ref: type = name_ref Destroy, imports.%Core.Destroy [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Destroy.facet.loc12: %Destroy.type = facet_value constants.%tuple.type.494, (constants.%DestroyImpl.impl_witness.37c, constants.%custom_witness.b80ef6.4) [concrete = constants.%Destroy.facet.646]
// CHECK:STDOUT:   %.loc12: %Destroy.type = converted %Tuple.ref.loc12, %Destroy.facet.loc12 [concrete = constants.%Destroy.facet.646]
// CHECK:STDOUT:   %Tuple.ref.loc13: %tuple.type.693 = name_ref Tuple, file.%Tuple [concrete = constants.%tuple]
// CHECK:STDOUT:   %Core.ref.loc13: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:   %TrivialDestroy.ref: type = name_ref TrivialDestroy, imports.%Core.TrivialDestroy [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Destroy.facet.loc13: %Destroy.type = facet_value constants.%tuple.type.494, (constants.%DestroyImpl.impl_witness.37c, constants.%custom_witness.b80ef6.4) [concrete = constants.%Destroy.facet.646]
// CHECK:STDOUT:   %.loc13: %Destroy.type = converted %Tuple.ref.loc13, %Destroy.facet.loc13 [concrete = constants.%Destroy.facet.646]
// CHECK:STDOUT:   %x.var: ref %tuple.type.494 = var_storage %x.var_patt
// CHECK:STDOUT:   %Make.ref: %Make.type = name_ref Make, file.%Make.decl [concrete = constants.%Make]
// CHECK:STDOUT:   %.loc15_3.1: ref %tuple.type.494 = splice_block %x.var {}
// CHECK:STDOUT:   %Make.call: init %tuple.type.494 to %.loc15_3.1 = call %Make.ref()
// CHECK:STDOUT:   assign %x.var, %Make.call
// CHECK:STDOUT:   %.loc15_17.1: type = splice_block %.loc15_17.2 [concrete = constants.%tuple.type.494] {
// CHECK:STDOUT:     %Tuple.ref.loc15: %tuple.type.693 = name_ref Tuple, file.%Tuple [concrete = constants.%tuple]
// CHECK:STDOUT:     %.loc15_17.2: type = converted %Tuple.ref.loc15, constants.%tuple.type.494 [concrete = constants.%tuple.type.494]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %x: ref %tuple.type.494 = wrapper_binding x, %x.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %pattern_type.16f = ref_binding_pattern x [concrete = constants.%x.patt]
// CHECK:STDOUT:     %x.var_patt: %pattern_type.16f = var_pattern %x.patt [concrete = constants.%x.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.facet.loc15_3.1: %Destroy.type = facet_value constants.%tuple.type.494, (constants.%DestroyImpl.impl_witness.37c, constants.%custom_witness.b80ef6.4) [concrete = constants.%Destroy.facet.646]
// CHECK:STDOUT:   %.loc15_3.2: %Destroy.type = converted constants.%tuple.type.494, %Destroy.facet.loc15_3.1 [concrete = constants.%Destroy.facet.646]
// CHECK:STDOUT:   %Destroy.facet.loc15_3.2: %Destroy.type = facet_value constants.%tuple.type.494, (constants.%DestroyImpl.impl_witness.37c, constants.%custom_witness.b80ef6.4) [concrete = constants.%Destroy.facet.646]
// CHECK:STDOUT:   %.loc15_3.3: %Destroy.type = converted constants.%tuple.type.494, %Destroy.facet.loc15_3.2 [concrete = constants.%Destroy.facet.646]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%x.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc11_9.1(%self.param: ref %facet_type.3be) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc11_9.2(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc11_9.3(%self.param: ref %HasTrivialDestruction) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc11_9.4(%self.param: ref %tuple.type.494) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1]
// CHECK:STDOUT:   %tuple.elem1: ref %HasTrivialDestruction = tuple_access %self.param, element1
// CHECK:STDOUT:   %Destroy.facet.loc11_9.1: %Destroy.type = facet_value constants.%HasTrivialDestruction, (constants.%DestroyImpl.impl_witness.4ca, constants.%custom_witness.b80ef6.3) [concrete = constants.%Destroy.facet.dba]
// CHECK:STDOUT:   %.loc11_9.1: %Destroy.type = converted constants.%HasTrivialDestruction, %Destroy.facet.loc11_9.1 [concrete = constants.%Destroy.facet.dba]
// CHECK:STDOUT:   %Destroy.facet.loc11_9.2: %Destroy.type = facet_value constants.%HasTrivialDestruction, (constants.%DestroyImpl.impl_witness.4ca, constants.%custom_witness.b80ef6.3) [concrete = constants.%Destroy.facet.dba]
// CHECK:STDOUT:   %.loc11_9.2: %Destroy.type = converted constants.%HasTrivialDestruction, %Destroy.facet.loc11_9.2 [concrete = constants.%Destroy.facet.dba]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc11_9.1: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%tuple.elem1)
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT:   %tuple.elem0: ref %facet_type.3be = tuple_access %self.param, element0
// CHECK:STDOUT:   %Destroy.facet.loc11_9.3: %Destroy.type = facet_value constants.%facet_type.3be, (constants.%DestroyImpl.impl_witness.99b, constants.%custom_witness.b80ef6.1) [concrete = constants.%Destroy.facet.295]
// CHECK:STDOUT:   %.loc11_9.3: %Destroy.type = converted constants.%facet_type.3be, %Destroy.facet.loc11_9.3 [concrete = constants.%Destroy.facet.295]
// CHECK:STDOUT:   %Destroy.facet.loc11_9.4: %Destroy.type = facet_value constants.%facet_type.3be, (constants.%DestroyImpl.impl_witness.99b, constants.%custom_witness.b80ef6.1) [concrete = constants.%Destroy.facet.295]
// CHECK:STDOUT:   %.loc11_9.4: %Destroy.type = converted constants.%facet_type.3be, %Destroy.facet.loc11_9.4 [concrete = constants.%Destroy.facet.295]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc11_9.2: init %empty_tuple.type = call %SelfDestruct.specific_fn.2(%tuple.elem0)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
