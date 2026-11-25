// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/impl/custom_witness/destruction/trivial/builtins.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/impl/custom_witness/destruction/trivial/builtins.carbon

// --- trivial_facet.carbon
library "[[@TEST_NAME]]";

// Tests we can destroy an object that has the built-in type `FacetType`.

fn Make() -> type;

fn Test() {
  //@dump-sem-ir-begin
  type as Core.SubobjectDestroy;
  //@dump-sem-ir-end

  type as Core.Destroy;
  type as Core.TrivialDestroy;

  //@dump-sem-ir-begin
  var unused x: type = Make();
  //@dump-sem-ir-end
}

// --- trivial_bound_method_via_class.carbon
library "[[@TEST_NAME]]";

// Tests we can destroy an object that has the built-in type `BoundMethodType`.
// Explores this using a class method.

class C {
  fn Get(self) -> C;
}

fn Test(c: C) {
  alias BoundMethodType = typeof(c.Get);
  fn Make(c: C) -> BoundMethodType;

  //@dump-sem-ir-begin
  BoundMethodType as Core.SubobjectDestroy;
  //@dump-sem-ir-end

  BoundMethodType as Core.Destroy;
  BoundMethodType as Core.TrivialDestroy;

  //@dump-sem-ir-begin
  var unused x: BoundMethodType = Make(c);
  //@dump-sem-ir-end
}

// --- trivial_bound_method_via_impl.carbon

// Tests we can destroy an object that has the built-in type `BoundMethodType`.
// Explores this using an interface method.

interface I { fn H(self); }
impl () as I { fn H(unused self) {} }
alias BoundMethodType = typeof(().(I.H));
fn Make() -> BoundMethodType;

fn Test() {

  //@dump-sem-ir-begin
  BoundMethodType as Core.SubobjectDestroy;
  //@dump-sem-ir-end

  BoundMethodType as Core.Destroy;
  BoundMethodType as Core.TrivialDestroy;

  //@dump-sem-ir-begin
  var unused x: BoundMethodType = Make();
  //@dump-sem-ir-end
}

// --- trivial_form.carbon
library "[[@TEST_NAME]]";

// Tests we can destroy an object that has the built-in type `FormType`.
// Uses an inner type that has non-trivial destruction to confirm that
// all form types have trivial destruction.

class HasNonTrivialDestruction {
  impl as Core.DestroyImpl {
    fn Op(unused ref self) {}
  }
}

alias FormType = typeof(form(var HasNonTrivialDestruction));
fn Make() -> FormType;

fn Test() {
  //@dump-sem-ir-begin
  FormType as Core.SubobjectDestroy;
  //@dump-sem-ir-end

  FormType as Core.Destroy;
  FormType as Core.TrivialDestroy;

  //@dump-sem-ir-begin
  var unused x: FormType = Make();
  //@dump-sem-ir-end
}

// --- trivial_function_via_function.carbon
library "[[@TEST_NAME]]";

// Tests we can destroy an object that has the built-in type `FunctionType`.
// Explores this using a function.

fn F() {}

alias FunctionType = typeof(F);
fn Make() -> FunctionType;

fn Test() {
  //@dump-sem-ir-begin
  FunctionType as Core.SubobjectDestroy;
  //@dump-sem-ir-end

  FunctionType as Core.Destroy;
  FunctionType as Core.TrivialDestroy;

  //@dump-sem-ir-begin
  var unused x: FunctionType = Make();
  //@dump-sem-ir-end
}

// --- trivial_function_via_method.carbon
library "[[@TEST_NAME]]";

// Tests we can destroy an object that has the built-in type `FunctionType`.
// Explores this using an instance method.

class C { fn M(self); }

alias FunctionType = typeof(C.M);
fn Make() -> FunctionType;

fn Test() {
  //@dump-sem-ir-begin
  FunctionType as Core.SubobjectDestroy;
  //@dump-sem-ir-end

  FunctionType as Core.Destroy;
  FunctionType as Core.TrivialDestroy;

  //@dump-sem-ir-begin
  var unused x: FunctionType = Make();
  //@dump-sem-ir-end
}

// CHECK:STDOUT: --- trivial_facet.carbon = $trivial_facet
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %pattern_type.9a587c.1: type = pattern_type type [concrete]
// CHECK:STDOUT:   %Make.type: type = fn_type @Make [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Make: %Make.type = struct_value () [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.687: type = fn_type @SubobjectDestroy.WithSelf.Op.loc9 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f: %SubobjectDestroy.WithSelf.Op.type.687 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.1f5: %SubobjectDestroy.type = facet_value type, (%custom_witness.b80) [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %DestroyImpl.type: type = facet_type <@DestroyImpl> [concrete]
// CHECK:STDOUT:   %T.09f: %SubobjectDestroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.type.dea: type = fn_type @T.as_type.as.DestroyImpl.impl.Op, @T.as_type.as.DestroyImpl.impl(%T.09f) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.6e5: %T.as_type.as.DestroyImpl.impl.Op.type.dea = struct_value () [symbolic]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.a54: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.1f5) [concrete]
// CHECK:STDOUT:   %Destroy.facet: %Destroy.type = facet_value type, (%custom_witness.b80, %DestroyImpl.impl_witness.a54) [concrete]
// CHECK:STDOUT:   %TrivialDestroy.type: type = facet_type <@TrivialDestroy> [concrete]
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
// CHECK:STDOUT:     .DestroyImpl = %Core.DestroyImpl
// CHECK:STDOUT:     .SelfDestruct = %Core.SelfDestruct
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.SubobjectDestroy: type = import_ref Core//prelude/parts/destroy, SubobjectDestroy, loaded [concrete = constants.%SubobjectDestroy.type]
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Core.import_ref.74d: @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op.type (%T.as_type.as.DestroyImpl.impl.Op.type.dea) = import_ref Core//prelude/parts/destroy, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op (constants.%T.as_type.as.DestroyImpl.impl.Op.6e5)]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table = impl_witness_table (%Core.import_ref.74d), @T.as_type.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT:   %Core.TrivialDestroy: type = import_ref Core//prelude/parts/destroy, TrivialDestroy, loaded [concrete = constants.%TrivialDestroy.type]
// CHECK:STDOUT:   %Core.DestroyImpl: type = import_ref Core//prelude/parts/destroy, DestroyImpl, loaded [concrete = constants.%DestroyImpl.type]
// CHECK:STDOUT:   %Core.SelfDestruct: %SelfDestruct.type = import_ref Core//prelude/parts/destroy, SelfDestruct, loaded [concrete = constants.%SelfDestruct]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Test() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc9_3: type = type_literal type [concrete = type]
// CHECK:STDOUT:   %Core.ref.loc9: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:   %SubobjectDestroy.ref: type = name_ref SubobjectDestroy, imports.%Core.SubobjectDestroy [concrete = constants.%SubobjectDestroy.type]
// CHECK:STDOUT:   %SubobjectDestroy.facet: %SubobjectDestroy.type = facet_value %.loc9_3, (constants.%custom_witness.b80) [concrete = constants.%SubobjectDestroy.facet.1f5]
// CHECK:STDOUT:   %.loc9_8: %SubobjectDestroy.type = converted %.loc9_3, %SubobjectDestroy.facet [concrete = constants.%SubobjectDestroy.facet.1f5]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %x.var: ref type = var_storage %x.var_patt
// CHECK:STDOUT:   %Make.ref: %Make.type = name_ref Make, file.%Make.decl [concrete = constants.%Make]
// CHECK:STDOUT:   %Make.call: init type = call %Make.ref()
// CHECK:STDOUT:   assign %x.var, %Make.call
// CHECK:STDOUT:   %.loc16_17: type = type_literal type [concrete = type]
// CHECK:STDOUT:   %x: ref type = wrapper_binding x, %x.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %pattern_type.9a587c.1 = ref_binding_pattern x [concrete = constants.%x.patt]
// CHECK:STDOUT:     %x.var_patt: %pattern_type.9a587c.1 = var_pattern %x.patt [concrete = constants.%x.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.facet.loc16_3.1: %Destroy.type = facet_value type, (constants.%custom_witness.b80, constants.%DestroyImpl.impl_witness.a54) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc16_3.1: %Destroy.type = converted type, %Destroy.facet.loc16_3.1 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %Destroy.facet.loc16_3.2: %Destroy.type = facet_value type, (constants.%custom_witness.b80, constants.%DestroyImpl.impl_witness.a54) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc16_3.2: %Destroy.type = converted type, %Destroy.facet.loc16_3.2 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%x.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc9(%self.param: ref type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: --- trivial_bound_method_via_class.carbon = $trivial_bound_method_via_class
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %pattern_type.061: type = pattern_type <bound method> [concrete]
// CHECK:STDOUT:   %Make.type: type = fn_type @Make [concrete]
// CHECK:STDOUT:   %Make: %Make.type = struct_value () [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.687: type = fn_type @SubobjectDestroy.WithSelf.Op.loc15 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f: %SubobjectDestroy.WithSelf.Op.type.687 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.8ad: %SubobjectDestroy.type = facet_value <bound method>, (%custom_witness.b80) [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %DestroyImpl.type: type = facet_type <@DestroyImpl> [concrete]
// CHECK:STDOUT:   %T.09f: %SubobjectDestroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.type.dea: type = fn_type @T.as_type.as.DestroyImpl.impl.Op, @T.as_type.as.DestroyImpl.impl(%T.09f) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.6e5: %T.as_type.as.DestroyImpl.impl.Op.type.dea = struct_value () [symbolic]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.691: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.8ad) [concrete]
// CHECK:STDOUT:   %Destroy.facet: %Destroy.type = facet_value <bound method>, (%custom_witness.b80, %DestroyImpl.impl_witness.691) [concrete]
// CHECK:STDOUT:   %TrivialDestroy.type: type = facet_type <@TrivialDestroy> [concrete]
// CHECK:STDOUT:   %x.patt: %pattern_type.061 = ref_binding_pattern x [concrete]
// CHECK:STDOUT:   %x.var_patt: %pattern_type.061 = var_pattern %x.patt [concrete]
// CHECK:STDOUT:   %SelfDestruct.type: type = fn_type @SelfDestruct [concrete]
// CHECK:STDOUT:   %SelfDestruct: %SelfDestruct.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .SubobjectDestroy = %Core.SubobjectDestroy
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     .TrivialDestroy = %Core.TrivialDestroy
// CHECK:STDOUT:     .DestroyImpl = %Core.DestroyImpl
// CHECK:STDOUT:     .SelfDestruct = %Core.SelfDestruct
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.SubobjectDestroy: type = import_ref Core//prelude/parts/destroy, SubobjectDestroy, loaded [concrete = constants.%SubobjectDestroy.type]
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Core.import_ref.74d: @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op.type (%T.as_type.as.DestroyImpl.impl.Op.type.dea) = import_ref Core//prelude/parts/destroy, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op (constants.%T.as_type.as.DestroyImpl.impl.Op.6e5)]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table = impl_witness_table (%Core.import_ref.74d), @T.as_type.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT:   %Core.TrivialDestroy: type = import_ref Core//prelude/parts/destroy, TrivialDestroy, loaded [concrete = constants.%TrivialDestroy.type]
// CHECK:STDOUT:   %Core.DestroyImpl: type = import_ref Core//prelude/parts/destroy, DestroyImpl, loaded [concrete = constants.%DestroyImpl.type]
// CHECK:STDOUT:   %Core.SelfDestruct: %SelfDestruct.type = import_ref Core//prelude/parts/destroy, SelfDestruct, loaded [concrete = constants.%SelfDestruct]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Test(%c.param: %C) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %BoundMethodType.ref.loc15: type = name_ref BoundMethodType, %BoundMethodType [concrete = <bound method>]
// CHECK:STDOUT:   %Core.ref.loc15: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:   %SubobjectDestroy.ref: type = name_ref SubobjectDestroy, imports.%Core.SubobjectDestroy [concrete = constants.%SubobjectDestroy.type]
// CHECK:STDOUT:   %SubobjectDestroy.facet: %SubobjectDestroy.type = facet_value %BoundMethodType.ref.loc15, (constants.%custom_witness.b80) [concrete = constants.%SubobjectDestroy.facet.8ad]
// CHECK:STDOUT:   %.loc15: %SubobjectDestroy.type = converted %BoundMethodType.ref.loc15, %SubobjectDestroy.facet [concrete = constants.%SubobjectDestroy.facet.8ad]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %x.var: ref <bound method> = var_storage %x.var_patt
// CHECK:STDOUT:   %Make.ref: %Make.type = name_ref Make, %Make.decl [concrete = constants.%Make]
// CHECK:STDOUT:   %c.ref.loc22: %C = name_ref c, %c
// CHECK:STDOUT:   %Make.call: init <bound method> = call %Make.ref(%c.ref.loc22)
// CHECK:STDOUT:   assign %x.var, %Make.call
// CHECK:STDOUT:   %BoundMethodType.ref.loc22: type = name_ref BoundMethodType, %BoundMethodType [concrete = <bound method>]
// CHECK:STDOUT:   %x: ref <bound method> = wrapper_binding x, %x.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %pattern_type.061 = ref_binding_pattern x [concrete = constants.%x.patt]
// CHECK:STDOUT:     %x.var_patt: %pattern_type.061 = var_pattern %x.patt [concrete = constants.%x.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.facet.loc22_3.1: %Destroy.type = facet_value <bound method>, (constants.%custom_witness.b80, constants.%DestroyImpl.impl_witness.691) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc22_3.1: %Destroy.type = converted <bound method>, %Destroy.facet.loc22_3.1 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %Destroy.facet.loc22_3.2: %Destroy.type = facet_value <bound method>, (constants.%custom_witness.b80, constants.%DestroyImpl.impl_witness.691) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc22_3.2: %Destroy.type = converted <bound method>, %Destroy.facet.loc22_3.2 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%x.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc15(%self.param: ref <bound method>) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: --- trivial_bound_method_via_impl.carbon = $trivial_bound_method_via_impl
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %pattern_type.061: type = pattern_type <bound method> [concrete]
// CHECK:STDOUT:   %Make.type: type = fn_type @Make [concrete]
// CHECK:STDOUT:   %Make: %Make.type = struct_value () [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.687: type = fn_type @SubobjectDestroy.WithSelf.Op.loc13 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f: %SubobjectDestroy.WithSelf.Op.type.687 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.8ad: %SubobjectDestroy.type = facet_value <bound method>, (%custom_witness.b80) [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %DestroyImpl.type: type = facet_type <@DestroyImpl> [concrete]
// CHECK:STDOUT:   %T.09f: %SubobjectDestroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.type.dea: type = fn_type @T.as_type.as.DestroyImpl.impl.Op, @T.as_type.as.DestroyImpl.impl(%T.09f) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.6e5: %T.as_type.as.DestroyImpl.impl.Op.type.dea = struct_value () [symbolic]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.691: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.8ad) [concrete]
// CHECK:STDOUT:   %Destroy.facet: %Destroy.type = facet_value <bound method>, (%custom_witness.b80, %DestroyImpl.impl_witness.691) [concrete]
// CHECK:STDOUT:   %TrivialDestroy.type: type = facet_type <@TrivialDestroy> [concrete]
// CHECK:STDOUT:   %x.patt: %pattern_type.061 = ref_binding_pattern x [concrete]
// CHECK:STDOUT:   %x.var_patt: %pattern_type.061 = var_pattern %x.patt [concrete]
// CHECK:STDOUT:   %SelfDestruct.type: type = fn_type @SelfDestruct [concrete]
// CHECK:STDOUT:   %SelfDestruct: %SelfDestruct.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .SubobjectDestroy = %Core.SubobjectDestroy
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     .TrivialDestroy = %Core.TrivialDestroy
// CHECK:STDOUT:     .DestroyImpl = %Core.DestroyImpl
// CHECK:STDOUT:     .SelfDestruct = %Core.SelfDestruct
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.SubobjectDestroy: type = import_ref Core//prelude/parts/destroy, SubobjectDestroy, loaded [concrete = constants.%SubobjectDestroy.type]
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Core.import_ref.74d: @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op.type (%T.as_type.as.DestroyImpl.impl.Op.type.dea) = import_ref Core//prelude/parts/destroy, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op (constants.%T.as_type.as.DestroyImpl.impl.Op.6e5)]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table = impl_witness_table (%Core.import_ref.74d), @T.as_type.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT:   %Core.TrivialDestroy: type = import_ref Core//prelude/parts/destroy, TrivialDestroy, loaded [concrete = constants.%TrivialDestroy.type]
// CHECK:STDOUT:   %Core.DestroyImpl: type = import_ref Core//prelude/parts/destroy, DestroyImpl, loaded [concrete = constants.%DestroyImpl.type]
// CHECK:STDOUT:   %Core.SelfDestruct: %SelfDestruct.type = import_ref Core//prelude/parts/destroy, SelfDestruct, loaded [concrete = constants.%SelfDestruct]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Test() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %BoundMethodType.ref.loc13: type = name_ref BoundMethodType, file.%BoundMethodType [concrete = <bound method>]
// CHECK:STDOUT:   %Core.ref.loc13: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:   %SubobjectDestroy.ref: type = name_ref SubobjectDestroy, imports.%Core.SubobjectDestroy [concrete = constants.%SubobjectDestroy.type]
// CHECK:STDOUT:   %SubobjectDestroy.facet: %SubobjectDestroy.type = facet_value %BoundMethodType.ref.loc13, (constants.%custom_witness.b80) [concrete = constants.%SubobjectDestroy.facet.8ad]
// CHECK:STDOUT:   %.loc13: %SubobjectDestroy.type = converted %BoundMethodType.ref.loc13, %SubobjectDestroy.facet [concrete = constants.%SubobjectDestroy.facet.8ad]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %x.var: ref <bound method> = var_storage %x.var_patt
// CHECK:STDOUT:   %Make.ref: %Make.type = name_ref Make, file.%Make.decl [concrete = constants.%Make]
// CHECK:STDOUT:   %Make.call: init <bound method> = call %Make.ref()
// CHECK:STDOUT:   assign %x.var, %Make.call
// CHECK:STDOUT:   %BoundMethodType.ref.loc20: type = name_ref BoundMethodType, file.%BoundMethodType [concrete = <bound method>]
// CHECK:STDOUT:   %x: ref <bound method> = wrapper_binding x, %x.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %pattern_type.061 = ref_binding_pattern x [concrete = constants.%x.patt]
// CHECK:STDOUT:     %x.var_patt: %pattern_type.061 = var_pattern %x.patt [concrete = constants.%x.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.facet.loc20_3.1: %Destroy.type = facet_value <bound method>, (constants.%custom_witness.b80, constants.%DestroyImpl.impl_witness.691) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc20_3.1: %Destroy.type = converted <bound method>, %Destroy.facet.loc20_3.1 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %Destroy.facet.loc20_3.2: %Destroy.type = facet_value <bound method>, (constants.%custom_witness.b80, constants.%DestroyImpl.impl_witness.691) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc20_3.2: %Destroy.type = converted <bound method>, %Destroy.facet.loc20_3.2 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%x.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc13(%self.param: ref <bound method>) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: --- trivial_form.carbon = $trivial_form
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %DestroyImpl.type: type = facet_type <@DestroyImpl> [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %pattern_type.13f: type = pattern_type Core.Form [concrete]
// CHECK:STDOUT:   %Make.type: type = fn_type @Make [concrete]
// CHECK:STDOUT:   %Make: %Make.type = struct_value () [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.687: type = fn_type @SubobjectDestroy.WithSelf.Op.loc18 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f: %SubobjectDestroy.WithSelf.Op.type.687 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.4de: %SubobjectDestroy.type = facet_value Core.Form, (%custom_witness.b80) [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %T.09f: %SubobjectDestroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.type.dea: type = fn_type @T.as_type.as.DestroyImpl.impl.Op, @T.as_type.as.DestroyImpl.impl(%T.09f) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.6e5: %T.as_type.as.DestroyImpl.impl.Op.type.dea = struct_value () [symbolic]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.fb5: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.4de) [concrete]
// CHECK:STDOUT:   %Destroy.facet: %Destroy.type = facet_value Core.Form, (%DestroyImpl.impl_witness.fb5, %custom_witness.b80) [concrete]
// CHECK:STDOUT:   %TrivialDestroy.type: type = facet_type <@TrivialDestroy> [concrete]
// CHECK:STDOUT:   %x.patt: %pattern_type.13f = ref_binding_pattern x [concrete]
// CHECK:STDOUT:   %x.var_patt: %pattern_type.13f = var_pattern %x.patt [concrete]
// CHECK:STDOUT:   %SelfDestruct.type: type = fn_type @SelfDestruct [concrete]
// CHECK:STDOUT:   %SelfDestruct: %SelfDestruct.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .DestroyImpl = %Core.DestroyImpl
// CHECK:STDOUT:     .SubobjectDestroy = %Core.SubobjectDestroy
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     .TrivialDestroy = %Core.TrivialDestroy
// CHECK:STDOUT:     .SelfDestruct = %Core.SelfDestruct
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.DestroyImpl: type = import_ref Core//prelude/parts/destroy, DestroyImpl, loaded [concrete = constants.%DestroyImpl.type]
// CHECK:STDOUT:   %Core.SubobjectDestroy: type = import_ref Core//prelude/parts/destroy, SubobjectDestroy, loaded [concrete = constants.%SubobjectDestroy.type]
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Core.import_ref.74d: @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op.type (%T.as_type.as.DestroyImpl.impl.Op.type.dea) = import_ref Core//prelude/parts/destroy, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op (constants.%T.as_type.as.DestroyImpl.impl.Op.6e5)]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table = impl_witness_table (%Core.import_ref.74d), @T.as_type.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT:   %Core.TrivialDestroy: type = import_ref Core//prelude/parts/destroy, TrivialDestroy, loaded [concrete = constants.%TrivialDestroy.type]
// CHECK:STDOUT:   %Core.SelfDestruct: %SelfDestruct.type = import_ref Core//prelude/parts/destroy, SelfDestruct, loaded [concrete = constants.%SelfDestruct]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Test() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %FormType.ref.loc18: type = name_ref FormType, file.%FormType [concrete = Core.Form]
// CHECK:STDOUT:   %Core.ref.loc18: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:   %SubobjectDestroy.ref: type = name_ref SubobjectDestroy, imports.%Core.SubobjectDestroy [concrete = constants.%SubobjectDestroy.type]
// CHECK:STDOUT:   %SubobjectDestroy.facet: %SubobjectDestroy.type = facet_value %FormType.ref.loc18, (constants.%custom_witness.b80) [concrete = constants.%SubobjectDestroy.facet.4de]
// CHECK:STDOUT:   %.loc18: %SubobjectDestroy.type = converted %FormType.ref.loc18, %SubobjectDestroy.facet [concrete = constants.%SubobjectDestroy.facet.4de]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %x.var: ref Core.Form = var_storage %x.var_patt
// CHECK:STDOUT:   %Make.ref: %Make.type = name_ref Make, file.%Make.decl [concrete = constants.%Make]
// CHECK:STDOUT:   %Make.call: init Core.Form = call %Make.ref()
// CHECK:STDOUT:   assign %x.var, %Make.call
// CHECK:STDOUT:   %FormType.ref.loc25: type = name_ref FormType, file.%FormType [concrete = Core.Form]
// CHECK:STDOUT:   %x: ref Core.Form = wrapper_binding x, %x.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %pattern_type.13f = ref_binding_pattern x [concrete = constants.%x.patt]
// CHECK:STDOUT:     %x.var_patt: %pattern_type.13f = var_pattern %x.patt [concrete = constants.%x.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.facet.loc25_3.1: %Destroy.type = facet_value Core.Form, (constants.%DestroyImpl.impl_witness.fb5, constants.%custom_witness.b80) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc25_3.1: %Destroy.type = converted Core.Form, %Destroy.facet.loc25_3.1 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %Destroy.facet.loc25_3.2: %Destroy.type = facet_value Core.Form, (constants.%DestroyImpl.impl_witness.fb5, constants.%custom_witness.b80) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc25_3.2: %Destroy.type = converted Core.Form, %Destroy.facet.loc25_3.2 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%x.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc18(%self.param: ref Core.Form) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: --- trivial_function_via_function.carbon = $trivial_function_via_function
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %pattern_type.738: type = pattern_type %F.type [concrete]
// CHECK:STDOUT:   %Make.type: type = fn_type @Make [concrete]
// CHECK:STDOUT:   %Make: %Make.type = struct_value () [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.687: type = fn_type @SubobjectDestroy.WithSelf.Op.loc13 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f: %SubobjectDestroy.WithSelf.Op.type.687 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.944: %SubobjectDestroy.type = facet_value %F.type, (%custom_witness.b80) [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %DestroyImpl.type: type = facet_type <@DestroyImpl> [concrete]
// CHECK:STDOUT:   %T.09f: %SubobjectDestroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.type.dea: type = fn_type @T.as_type.as.DestroyImpl.impl.Op, @T.as_type.as.DestroyImpl.impl(%T.09f) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.6e5: %T.as_type.as.DestroyImpl.impl.Op.type.dea = struct_value () [symbolic]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.9ef: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.944) [concrete]
// CHECK:STDOUT:   %Destroy.facet: %Destroy.type = facet_value %F.type, (%custom_witness.b80, %DestroyImpl.impl_witness.9ef) [concrete]
// CHECK:STDOUT:   %TrivialDestroy.type: type = facet_type <@TrivialDestroy> [concrete]
// CHECK:STDOUT:   %x.patt: %pattern_type.738 = ref_binding_pattern x [concrete]
// CHECK:STDOUT:   %x.var_patt: %pattern_type.738 = var_pattern %x.patt [concrete]
// CHECK:STDOUT:   %SelfDestruct.type: type = fn_type @SelfDestruct [concrete]
// CHECK:STDOUT:   %SelfDestruct: %SelfDestruct.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .SubobjectDestroy = %Core.SubobjectDestroy
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     .TrivialDestroy = %Core.TrivialDestroy
// CHECK:STDOUT:     .DestroyImpl = %Core.DestroyImpl
// CHECK:STDOUT:     .SelfDestruct = %Core.SelfDestruct
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.SubobjectDestroy: type = import_ref Core//prelude/parts/destroy, SubobjectDestroy, loaded [concrete = constants.%SubobjectDestroy.type]
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Core.import_ref.74d: @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op.type (%T.as_type.as.DestroyImpl.impl.Op.type.dea) = import_ref Core//prelude/parts/destroy, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op (constants.%T.as_type.as.DestroyImpl.impl.Op.6e5)]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table = impl_witness_table (%Core.import_ref.74d), @T.as_type.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT:   %Core.TrivialDestroy: type = import_ref Core//prelude/parts/destroy, TrivialDestroy, loaded [concrete = constants.%TrivialDestroy.type]
// CHECK:STDOUT:   %Core.DestroyImpl: type = import_ref Core//prelude/parts/destroy, DestroyImpl, loaded [concrete = constants.%DestroyImpl.type]
// CHECK:STDOUT:   %Core.SelfDestruct: %SelfDestruct.type = import_ref Core//prelude/parts/destroy, SelfDestruct, loaded [concrete = constants.%SelfDestruct]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Test() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %FunctionType.ref.loc13: type = name_ref FunctionType, file.%FunctionType [concrete = constants.%F.type]
// CHECK:STDOUT:   %Core.ref.loc13: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:   %SubobjectDestroy.ref: type = name_ref SubobjectDestroy, imports.%Core.SubobjectDestroy [concrete = constants.%SubobjectDestroy.type]
// CHECK:STDOUT:   %SubobjectDestroy.facet: %SubobjectDestroy.type = facet_value %FunctionType.ref.loc13, (constants.%custom_witness.b80) [concrete = constants.%SubobjectDestroy.facet.944]
// CHECK:STDOUT:   %.loc13: %SubobjectDestroy.type = converted %FunctionType.ref.loc13, %SubobjectDestroy.facet [concrete = constants.%SubobjectDestroy.facet.944]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %x.var: ref %F.type = var_storage %x.var_patt
// CHECK:STDOUT:   %Make.ref: %Make.type = name_ref Make, file.%Make.decl [concrete = constants.%Make]
// CHECK:STDOUT:   %Make.call: init %F.type = call %Make.ref()
// CHECK:STDOUT:   assign %x.var, %Make.call
// CHECK:STDOUT:   %FunctionType.ref.loc20: type = name_ref FunctionType, file.%FunctionType [concrete = constants.%F.type]
// CHECK:STDOUT:   %x: ref %F.type = wrapper_binding x, %x.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %pattern_type.738 = ref_binding_pattern x [concrete = constants.%x.patt]
// CHECK:STDOUT:     %x.var_patt: %pattern_type.738 = var_pattern %x.patt [concrete = constants.%x.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.facet.loc20_3.1: %Destroy.type = facet_value constants.%F.type, (constants.%custom_witness.b80, constants.%DestroyImpl.impl_witness.9ef) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc20_3.1: %Destroy.type = converted constants.%F.type, %Destroy.facet.loc20_3.1 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %Destroy.facet.loc20_3.2: %Destroy.type = facet_value constants.%F.type, (constants.%custom_witness.b80, constants.%DestroyImpl.impl_witness.9ef) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc20_3.2: %Destroy.type = converted constants.%F.type, %Destroy.facet.loc20_3.2 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%x.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc13(%self.param: ref %F.type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: --- trivial_function_via_method.carbon = $trivial_function_via_method
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %C.M.type: type = fn_type @C.M [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %pattern_type.bb7: type = pattern_type %C.M.type [concrete]
// CHECK:STDOUT:   %Make.type: type = fn_type @Make [concrete]
// CHECK:STDOUT:   %Make: %Make.type = struct_value () [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.687: type = fn_type @SubobjectDestroy.WithSelf.Op.loc13 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f: %SubobjectDestroy.WithSelf.Op.type.687 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.7cd: %SubobjectDestroy.type = facet_value %C.M.type, (%custom_witness.b80) [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %DestroyImpl.type: type = facet_type <@DestroyImpl> [concrete]
// CHECK:STDOUT:   %T.09f: %SubobjectDestroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.type.dea: type = fn_type @T.as_type.as.DestroyImpl.impl.Op, @T.as_type.as.DestroyImpl.impl(%T.09f) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.6e5: %T.as_type.as.DestroyImpl.impl.Op.type.dea = struct_value () [symbolic]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.c75: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.7cd) [concrete]
// CHECK:STDOUT:   %Destroy.facet: %Destroy.type = facet_value %C.M.type, (%custom_witness.b80, %DestroyImpl.impl_witness.c75) [concrete]
// CHECK:STDOUT:   %TrivialDestroy.type: type = facet_type <@TrivialDestroy> [concrete]
// CHECK:STDOUT:   %x.patt: %pattern_type.bb7 = ref_binding_pattern x [concrete]
// CHECK:STDOUT:   %x.var_patt: %pattern_type.bb7 = var_pattern %x.patt [concrete]
// CHECK:STDOUT:   %SelfDestruct.type: type = fn_type @SelfDestruct [concrete]
// CHECK:STDOUT:   %SelfDestruct: %SelfDestruct.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .SubobjectDestroy = %Core.SubobjectDestroy
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     .TrivialDestroy = %Core.TrivialDestroy
// CHECK:STDOUT:     .DestroyImpl = %Core.DestroyImpl
// CHECK:STDOUT:     .SelfDestruct = %Core.SelfDestruct
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.SubobjectDestroy: type = import_ref Core//prelude/parts/destroy, SubobjectDestroy, loaded [concrete = constants.%SubobjectDestroy.type]
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Core.import_ref.74d: @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op.type (%T.as_type.as.DestroyImpl.impl.Op.type.dea) = import_ref Core//prelude/parts/destroy, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op (constants.%T.as_type.as.DestroyImpl.impl.Op.6e5)]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table = impl_witness_table (%Core.import_ref.74d), @T.as_type.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT:   %Core.TrivialDestroy: type = import_ref Core//prelude/parts/destroy, TrivialDestroy, loaded [concrete = constants.%TrivialDestroy.type]
// CHECK:STDOUT:   %Core.DestroyImpl: type = import_ref Core//prelude/parts/destroy, DestroyImpl, loaded [concrete = constants.%DestroyImpl.type]
// CHECK:STDOUT:   %Core.SelfDestruct: %SelfDestruct.type = import_ref Core//prelude/parts/destroy, SelfDestruct, loaded [concrete = constants.%SelfDestruct]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Test() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %FunctionType.ref.loc13: type = name_ref FunctionType, file.%FunctionType [concrete = constants.%C.M.type]
// CHECK:STDOUT:   %Core.ref.loc13: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:   %SubobjectDestroy.ref: type = name_ref SubobjectDestroy, imports.%Core.SubobjectDestroy [concrete = constants.%SubobjectDestroy.type]
// CHECK:STDOUT:   %SubobjectDestroy.facet: %SubobjectDestroy.type = facet_value %FunctionType.ref.loc13, (constants.%custom_witness.b80) [concrete = constants.%SubobjectDestroy.facet.7cd]
// CHECK:STDOUT:   %.loc13: %SubobjectDestroy.type = converted %FunctionType.ref.loc13, %SubobjectDestroy.facet [concrete = constants.%SubobjectDestroy.facet.7cd]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %x.var: ref %C.M.type = var_storage %x.var_patt
// CHECK:STDOUT:   %Make.ref: %Make.type = name_ref Make, file.%Make.decl [concrete = constants.%Make]
// CHECK:STDOUT:   %Make.call: init %C.M.type = call %Make.ref()
// CHECK:STDOUT:   assign %x.var, %Make.call
// CHECK:STDOUT:   %FunctionType.ref.loc20: type = name_ref FunctionType, file.%FunctionType [concrete = constants.%C.M.type]
// CHECK:STDOUT:   %x: ref %C.M.type = wrapper_binding x, %x.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %pattern_type.bb7 = ref_binding_pattern x [concrete = constants.%x.patt]
// CHECK:STDOUT:     %x.var_patt: %pattern_type.bb7 = var_pattern %x.patt [concrete = constants.%x.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.facet.loc20_3.1: %Destroy.type = facet_value constants.%C.M.type, (constants.%custom_witness.b80, constants.%DestroyImpl.impl_witness.c75) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc20_3.1: %Destroy.type = converted constants.%C.M.type, %Destroy.facet.loc20_3.1 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %Destroy.facet.loc20_3.2: %Destroy.type = facet_value constants.%C.M.type, (constants.%custom_witness.b80, constants.%DestroyImpl.impl_witness.c75) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc20_3.2: %Destroy.type = converted constants.%C.M.type, %Destroy.facet.loc20_3.2 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%x.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc13(%self.param: ref %C.M.type) = "no_op";
// CHECK:STDOUT:
