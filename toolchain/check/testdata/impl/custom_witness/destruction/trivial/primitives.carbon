// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/primitives.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/impl/custom_witness/destruction/trivial/primitives.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/impl/custom_witness/destruction/trivial/primitives.carbon

// --- trivial_bool.carbon
library "[[@TEST_NAME]]";

fn MakeBool() -> bool;

fn Test() {
  //@dump-sem-ir-begin
  bool as Core.SubobjectDestroy;
  //@dump-sem-ir-end

  bool as Core.Destroy;
  bool as Core.TrivialDestroy;

  //@dump-sem-ir-begin
  var unused x: bool = MakeBool();
  //@dump-sem-ir-end
}

// --- trivial_bool_literal.carbon
library "[[@TEST_NAME]]";

alias BoolLiteral = typeof(true);

fn Test() {
  //@dump-sem-ir-begin
  BoolLiteral as Core.SubobjectDestroy;
  //@dump-sem-ir-end

  BoolLiteral as Core.Destroy;
  BoolLiteral as Core.TrivialDestroy;

  //@dump-sem-ir-begin
  var unused x: BoolLiteral = true;
  //@dump-sem-ir-end
}

// --- trivial_char.carbon
library "[[@TEST_NAME]]";

fn MakeChar() -> char;

fn Test() {
  //@dump-sem-ir-begin
  char as Core.SubobjectDestroy;
  //@dump-sem-ir-end

  char as Core.Destroy;
  char as Core.TrivialDestroy;

  //@dump-sem-ir-begin
  var unused x: char = MakeChar();
  //@dump-sem-ir-end
}

// --- todo_trivial_char_literal.carbon
library "[[@TEST_NAME]]";

// TODO: add `CharLiteralType` to `IsBuiltinWithTrivialDestruction`.

fn Test() {
  //@dump-sem-ir-begin
  // Core.CharLiteral as Core.SubobjectDestroy;
  //@dump-sem-ir-end

  // Core.CharLiteral as Core.Destroy;
  // Core.CharLiteral as Core.TrivialDestroy;

  //@dump-sem-ir-begin
  // var unused x: Core.CharLiteral = 'c';
  //@dump-sem-ir-end
}

// --- trivial_int.carbon
library "[[@TEST_NAME]]";

fn MakeInt() -> i32;

fn Test() {
  //@dump-sem-ir-begin
  i32 as Core.SubobjectDestroy;
  //@dump-sem-ir-end

  i32 as Core.Destroy;
  i32 as Core.TrivialDestroy;

  //@dump-sem-ir-begin
  var unused x: i32 = MakeInt();
  //@dump-sem-ir-end
}

// --- trivial_int_literal.carbon
library "[[@TEST_NAME]]";

fn Test() {
  //@dump-sem-ir-begin
  Core.IntLiteral as Core.SubobjectDestroy;
  //@dump-sem-ir-end

  Core.IntLiteral as Core.Destroy;
  Core.IntLiteral as Core.TrivialDestroy;

  //@dump-sem-ir-begin
  var unused x: Core.IntLiteral = 0;
  //@dump-sem-ir-end
}

// --- trivial_float.carbon
library "[[@TEST_NAME]]";

fn MakeFloat() -> f64;

fn Test() {
  //@dump-sem-ir-begin
  f64 as Core.SubobjectDestroy;
  //@dump-sem-ir-end

  f64 as Core.Destroy;
  f64 as Core.TrivialDestroy;

  //@dump-sem-ir-begin
  var unused x: f64 = MakeFloat();
  //@dump-sem-ir-end
}

// --- todo_trivial_float_literal.carbon
library "[[@TEST_NAME]]";

// TODO: add `FloatLiteralType` to `IsBuiltinWithTrivialDestruction`.

fn Test() {
  //@dump-sem-ir-begin
  // Core.FloatLiteral as Core.SubobjectDestroy;
  //@dump-sem-ir-end

  // Core.FloatLiteral as Core.Destroy;
  // Core.FloatLiteral as Core.TrivialDestroy;

  //@dump-sem-ir-begin
  // var unused x: Core.FloatLiteral = 0.0;
  //@dump-sem-ir-end
}

// --- trivial_uint.carbon
library "[[@TEST_NAME]]";

fn MakeUint() -> u32;

fn Test() {
  //@dump-sem-ir-begin
  u32 as Core.SubobjectDestroy;
  //@dump-sem-ir-end

  u32 as Core.Destroy;
  u32 as Core.TrivialDestroy;

  //@dump-sem-ir-begin
  var unused x: u32 = MakeUint();
  //@dump-sem-ir-end
}

// CHECK:STDOUT: --- trivial_bool.carbon = $trivial_bool
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %pattern_type.831: type = pattern_type bool [concrete]
// CHECK:STDOUT:   %MakeBool.type: type = fn_type @MakeBool [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %MakeBool: %MakeBool.type = struct_value () [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.687: type = fn_type @SubobjectDestroy.WithSelf.Op.loc7 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f: %SubobjectDestroy.WithSelf.Op.type.687 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.a87: %SubobjectDestroy.type = facet_value bool, (%custom_witness.b80) [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %DestroyImpl.type: type = facet_type <@DestroyImpl> [concrete]
// CHECK:STDOUT:   %T.09f: %SubobjectDestroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.type.dea: type = fn_type @T.as_type.as.DestroyImpl.impl.Op, @T.as_type.as.DestroyImpl.impl(%T.09f) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.6e5: %T.as_type.as.DestroyImpl.impl.Op.type.dea = struct_value () [symbolic]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.778: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.a87) [concrete]
// CHECK:STDOUT:   %Destroy.facet: %Destroy.type = facet_value bool, (%custom_witness.b80, %DestroyImpl.impl_witness.778) [concrete]
// CHECK:STDOUT:   %TrivialDestroy.type: type = facet_type <@TrivialDestroy> [concrete]
// CHECK:STDOUT:   %x.patt: %pattern_type.831 = ref_binding_pattern x [concrete]
// CHECK:STDOUT:   %x.var_patt: %pattern_type.831 = var_pattern %x.patt [concrete]
// CHECK:STDOUT:   %SelfDestruct.type: type = fn_type @SelfDestruct [concrete]
// CHECK:STDOUT:   %SelfDestruct: %SelfDestruct.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Bool = %Core.Bool
// CHECK:STDOUT:     .SubobjectDestroy = %Core.SubobjectDestroy
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     .TrivialDestroy = %Core.TrivialDestroy
// CHECK:STDOUT:     .DestroyImpl = %Core.DestroyImpl
// CHECK:STDOUT:     .SelfDestruct = %Core.SelfDestruct
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Bool: type = import_ref Core//prelude/parts/bool, Bool, loaded [concrete = bool]
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
// CHECK:STDOUT:   %.loc7_3: type = type_literal bool [concrete = bool]
// CHECK:STDOUT:   %Core.ref.loc7: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:   %SubobjectDestroy.ref: type = name_ref SubobjectDestroy, imports.%Core.SubobjectDestroy [concrete = constants.%SubobjectDestroy.type]
// CHECK:STDOUT:   %SubobjectDestroy.facet: %SubobjectDestroy.type = facet_value %.loc7_3, (constants.%custom_witness.b80) [concrete = constants.%SubobjectDestroy.facet.a87]
// CHECK:STDOUT:   %.loc7_8: %SubobjectDestroy.type = converted %.loc7_3, %SubobjectDestroy.facet [concrete = constants.%SubobjectDestroy.facet.a87]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %x.var: ref bool = var_storage %x.var_patt
// CHECK:STDOUT:   %MakeBool.ref: %MakeBool.type = name_ref MakeBool, file.%MakeBool.decl [concrete = constants.%MakeBool]
// CHECK:STDOUT:   %MakeBool.call: init bool = call %MakeBool.ref()
// CHECK:STDOUT:   assign %x.var, %MakeBool.call
// CHECK:STDOUT:   %.loc14_17: type = type_literal bool [concrete = bool]
// CHECK:STDOUT:   %x: ref bool = wrapper_binding x, %x.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %pattern_type.831 = ref_binding_pattern x [concrete = constants.%x.patt]
// CHECK:STDOUT:     %x.var_patt: %pattern_type.831 = var_pattern %x.patt [concrete = constants.%x.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.facet.loc14_3.1: %Destroy.type = facet_value bool, (constants.%custom_witness.b80, constants.%DestroyImpl.impl_witness.778) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc14_3.1: %Destroy.type = converted bool, %Destroy.facet.loc14_3.1 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %Destroy.facet.loc14_3.2: %Destroy.type = facet_value bool, (constants.%custom_witness.b80, constants.%DestroyImpl.impl_witness.778) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc14_3.2: %Destroy.type = converted bool, %Destroy.facet.loc14_3.2 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%x.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc7(%self.param: ref bool) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: --- trivial_bool_literal.carbon = $trivial_bool_literal
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %true: bool = bool_literal true [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %pattern_type.831: type = pattern_type bool [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.687: type = fn_type @SubobjectDestroy.WithSelf.Op.loc7 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f: %SubobjectDestroy.WithSelf.Op.type.687 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.a87: %SubobjectDestroy.type = facet_value bool, (%custom_witness.b80) [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %DestroyImpl.type: type = facet_type <@DestroyImpl> [concrete]
// CHECK:STDOUT:   %T.09f: %SubobjectDestroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.type.dea: type = fn_type @T.as_type.as.DestroyImpl.impl.Op, @T.as_type.as.DestroyImpl.impl(%T.09f) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.6e5: %T.as_type.as.DestroyImpl.impl.Op.type.dea = struct_value () [symbolic]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.778: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.a87) [concrete]
// CHECK:STDOUT:   %Destroy.facet: %Destroy.type = facet_value bool, (%custom_witness.b80, %DestroyImpl.impl_witness.778) [concrete]
// CHECK:STDOUT:   %TrivialDestroy.type: type = facet_type <@TrivialDestroy> [concrete]
// CHECK:STDOUT:   %x.patt: %pattern_type.831 = ref_binding_pattern x [concrete]
// CHECK:STDOUT:   %x.var_patt: %pattern_type.831 = var_pattern %x.patt [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Copy.impl_witness.5e1: <witness> = impl_witness imports.%Copy.impl_witness_table.bda [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value bool, (%Copy.impl_witness.5e1) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.c98: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet) [concrete]
// CHECK:STDOUT:   %.68a: type = fn_type_with_self_type %Copy.WithSelf.Op.type.c98, %Copy.facet [concrete]
// CHECK:STDOUT:   %bool.as.Copy.impl.Op.type: type = fn_type @bool.as.Copy.impl.Op [concrete]
// CHECK:STDOUT:   %bool.as.Copy.impl.Op: %bool.as.Copy.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %bool.as.Copy.impl.Op.bound: <bound method> = bound_method %true, %bool.as.Copy.impl.Op [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .SubobjectDestroy = %Core.SubobjectDestroy
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     .TrivialDestroy = %Core.TrivialDestroy
// CHECK:STDOUT:     .DestroyImpl = %Core.DestroyImpl
// CHECK:STDOUT:     .Copy = %Core.Copy
// CHECK:STDOUT:     .TrivialDestroyImpl = %Core.TrivialDestroyImpl
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.SubobjectDestroy: type = import_ref Core//prelude/parts/destroy, SubobjectDestroy, loaded [concrete = constants.%SubobjectDestroy.type]
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Core.import_ref.74d: @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op.type (%T.as_type.as.DestroyImpl.impl.Op.type.dea) = import_ref Core//prelude/parts/destroy, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op (constants.%T.as_type.as.DestroyImpl.impl.Op.6e5)]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table = impl_witness_table (%Core.import_ref.74d), @T.as_type.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT:   %Core.TrivialDestroy: type = import_ref Core//prelude/parts/destroy, TrivialDestroy, loaded [concrete = constants.%TrivialDestroy.type]
// CHECK:STDOUT:   %Core.DestroyImpl: type = import_ref Core//prelude/parts/destroy, DestroyImpl, loaded [concrete = constants.%DestroyImpl.type]
// CHECK:STDOUT:   %Core.Copy: type = import_ref Core//prelude/parts/copy, Copy, loaded [concrete = constants.%Copy.type]
// CHECK:STDOUT:   %Core.import_ref.9c3: %bool.as.Copy.impl.Op.type = import_ref Core//prelude/parts/copy, loc{{\d+_\d+}}, loaded [concrete = constants.%bool.as.Copy.impl.Op]
// CHECK:STDOUT:   %Copy.impl_witness_table.bda = impl_witness_table (%Core.import_ref.9c3), @bool.as.Copy.impl [concrete]
// CHECK:STDOUT:   %Core.TrivialDestroyImpl: type = import_ref Core//prelude/parts/destroy, TrivialDestroyImpl, loaded [concrete = constants.%TrivialDestroyImpl.type]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Test() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %BoolLiteral.ref.loc7: type = name_ref BoolLiteral, file.%BoolLiteral [concrete = bool]
// CHECK:STDOUT:   %Core.ref.loc7: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:   %SubobjectDestroy.ref: type = name_ref SubobjectDestroy, imports.%Core.SubobjectDestroy [concrete = constants.%SubobjectDestroy.type]
// CHECK:STDOUT:   %SubobjectDestroy.facet: %SubobjectDestroy.type = facet_value %BoolLiteral.ref.loc7, (constants.%custom_witness.b80) [concrete = constants.%SubobjectDestroy.facet.a87]
// CHECK:STDOUT:   %.loc7: %SubobjectDestroy.type = converted %BoolLiteral.ref.loc7, %SubobjectDestroy.facet [concrete = constants.%SubobjectDestroy.facet.a87]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %x.var: ref bool = var_storage %x.var_patt
// CHECK:STDOUT:   %true: bool = bool_literal true [concrete = constants.%true]
// CHECK:STDOUT:   %impl.elem0: %.68a = impl_witness_access constants.%Copy.impl_witness.5e1, element0 [concrete = constants.%bool.as.Copy.impl.Op]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %true, %impl.elem0 [concrete = constants.%bool.as.Copy.impl.Op.bound]
// CHECK:STDOUT:   %bool.as.Copy.impl.Op.call: init bool = call %bound_method(%true) [concrete = constants.%true]
// CHECK:STDOUT:   assign %x.var, %bool.as.Copy.impl.Op.call
// CHECK:STDOUT:   %BoolLiteral.ref.loc14: type = name_ref BoolLiteral, file.%BoolLiteral [concrete = bool]
// CHECK:STDOUT:   %x: ref bool = wrapper_binding x, %x.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %pattern_type.831 = ref_binding_pattern x [concrete = constants.%x.patt]
// CHECK:STDOUT:     %x.var_patt: %pattern_type.831 = var_pattern %x.patt [concrete = constants.%x.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.facet.loc14_3.1: %Destroy.type = facet_value bool, (constants.%custom_witness.b80, constants.%DestroyImpl.impl_witness.778) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc14_3.1: %Destroy.type = converted bool, %Destroy.facet.loc14_3.1 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %Destroy.facet.loc14_3.2: %Destroy.type = facet_value bool, (constants.%custom_witness.b80, constants.%DestroyImpl.impl_witness.778) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc14_3.2: %Destroy.type = converted bool, %Destroy.facet.loc14_3.2 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%x.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc7(%self.param: ref bool) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: --- trivial_char.carbon = $trivial_char
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %char: type = class_type @Char [concrete]
// CHECK:STDOUT:   %int_8: Core.IntLiteral = int_value 8 [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %u8: type = class_type @UInt, @UInt(%int_8) [concrete]
// CHECK:STDOUT:   %u8.builtin: type = int_type unsigned, %int_8 [concrete]
// CHECK:STDOUT:   %pattern_type.2ff: type = pattern_type %char [concrete]
// CHECK:STDOUT:   %MakeChar.type: type = fn_type @MakeChar [concrete]
// CHECK:STDOUT:   %MakeChar: %MakeChar.type = struct_value () [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %DestroyImpl.type: type = facet_type <@DestroyImpl> [concrete]
// CHECK:STDOUT:   %T.09f: %SubobjectDestroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.type.dea: type = fn_type @T.as_type.as.DestroyImpl.impl.Op, @T.as_type.as.DestroyImpl.impl(%T.09f) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.6e5: %T.as_type.as.DestroyImpl.impl.Op.type.dea = struct_value () [symbolic]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.3: type = fn_type @SubobjectDestroy.WithSelf.Op.loc7_8.3 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.3: %SubobjectDestroy.WithSelf.Op.type.6870fd.3 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.3: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.3), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.909: %SubobjectDestroy.type = facet_value %char, (%custom_witness.b80ef6.3) [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.114: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.909) [concrete]
// CHECK:STDOUT:   %Destroy.facet: %Destroy.type = facet_value %char, (%custom_witness.b80ef6.3, %DestroyImpl.impl_witness.114) [concrete]
// CHECK:STDOUT:   %TrivialDestroy.type: type = facet_type <@TrivialDestroy> [concrete]
// CHECK:STDOUT:   %x.patt: %pattern_type.2ff = ref_binding_pattern x [concrete]
// CHECK:STDOUT:   %x.var_patt: %pattern_type.2ff = var_pattern %x.patt [concrete]
// CHECK:STDOUT:   %SelfDestruct.type: type = fn_type @SelfDestruct [concrete]
// CHECK:STDOUT:   %SelfDestruct: %SelfDestruct.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Char = %Core.Char
// CHECK:STDOUT:     .SubobjectDestroy = %Core.SubobjectDestroy
// CHECK:STDOUT:     .DestroyImpl = %Core.DestroyImpl
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     .TrivialDestroy = %Core.TrivialDestroy
// CHECK:STDOUT:     .SelfDestruct = %Core.SelfDestruct
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Char: type = import_ref Core//prelude/parts/char, Char, loaded [concrete = constants.%char]
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
// CHECK:STDOUT:   %char.loc7: type = type_literal constants.%char [concrete = constants.%char]
// CHECK:STDOUT:   %Core.ref.loc7: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:   %SubobjectDestroy.ref: type = name_ref SubobjectDestroy, imports.%Core.SubobjectDestroy [concrete = constants.%SubobjectDestroy.type]
// CHECK:STDOUT:   %SubobjectDestroy.facet: %SubobjectDestroy.type = facet_value %char.loc7, (constants.%custom_witness.b80ef6.3) [concrete = constants.%SubobjectDestroy.facet.909]
// CHECK:STDOUT:   %.loc7: %SubobjectDestroy.type = converted %char.loc7, %SubobjectDestroy.facet [concrete = constants.%SubobjectDestroy.facet.909]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %x.var: ref %char = var_storage %x.var_patt
// CHECK:STDOUT:   %MakeChar.ref: %MakeChar.type = name_ref MakeChar, file.%MakeChar.decl [concrete = constants.%MakeChar]
// CHECK:STDOUT:   %MakeChar.call: init %char = call %MakeChar.ref()
// CHECK:STDOUT:   assign %x.var, %MakeChar.call
// CHECK:STDOUT:   %char.loc14: type = type_literal constants.%char [concrete = constants.%char]
// CHECK:STDOUT:   %x: ref %char = wrapper_binding x, %x.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %pattern_type.2ff = ref_binding_pattern x [concrete = constants.%x.patt]
// CHECK:STDOUT:     %x.var_patt: %pattern_type.2ff = var_pattern %x.patt [concrete = constants.%x.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.facet.loc14_3.1: %Destroy.type = facet_value constants.%char, (constants.%custom_witness.b80ef6.3, constants.%DestroyImpl.impl_witness.114) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc14_3.1: %Destroy.type = converted constants.%char, %Destroy.facet.loc14_3.1 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %Destroy.facet.loc14_3.2: %Destroy.type = facet_value constants.%char, (constants.%custom_witness.b80ef6.3, constants.%DestroyImpl.impl_witness.114) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc14_3.2: %Destroy.type = converted constants.%char, %Destroy.facet.loc14_3.2 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%x.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc7_8.1(%self.param: ref %u8.builtin) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc7_8.2(%self.param: ref %u8) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc7_8.3(%self.param: ref %char) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: --- todo_trivial_char_literal.carbon = $todo_trivial_char_literal
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Test() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- trivial_int.carbon = $trivial_int
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = parameterized_class_name_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %pattern_type.d94: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %MakeInt.type: type = fn_type @MakeInt [concrete]
// CHECK:STDOUT:   %MakeInt: %MakeInt.type = struct_value () [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %i32.builtin: type = int_type signed, %int_32 [concrete]
// CHECK:STDOUT:   %DestroyImpl.type: type = facet_type <@DestroyImpl> [concrete]
// CHECK:STDOUT:   %T.09f: %SubobjectDestroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.type.dea: type = fn_type @T.as_type.as.DestroyImpl.impl.Op, @T.as_type.as.DestroyImpl.impl(%T.09f) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.6e5: %T.as_type.as.DestroyImpl.impl.Op.type.dea = struct_value () [symbolic]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.2: type = fn_type @SubobjectDestroy.WithSelf.Op.loc7_7.2 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.2: %SubobjectDestroy.WithSelf.Op.type.6870fd.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.2: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.2), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.49a: %SubobjectDestroy.type = facet_value %i32, (%custom_witness.b80ef6.2) [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.077: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.49a) [concrete]
// CHECK:STDOUT:   %Destroy.facet: %Destroy.type = facet_value %i32, (%custom_witness.b80ef6.2, %DestroyImpl.impl_witness.077) [concrete]
// CHECK:STDOUT:   %TrivialDestroy.type: type = facet_type <@TrivialDestroy> [concrete]
// CHECK:STDOUT:   %x.patt: %pattern_type.d94 = ref_binding_pattern x [concrete]
// CHECK:STDOUT:   %x.var_patt: %pattern_type.d94 = var_pattern %x.patt [concrete]
// CHECK:STDOUT:   %SelfDestruct.type: type = fn_type @SelfDestruct [concrete]
// CHECK:STDOUT:   %SelfDestruct: %SelfDestruct.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .SubobjectDestroy = %Core.SubobjectDestroy
// CHECK:STDOUT:     .DestroyImpl = %Core.DestroyImpl
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     .TrivialDestroy = %Core.TrivialDestroy
// CHECK:STDOUT:     .SelfDestruct = %Core.SelfDestruct
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/parts/int, Int, loaded [concrete = constants.%Int.generic]
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
// CHECK:STDOUT:   %i32.loc7: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %Core.ref.loc7: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:   %SubobjectDestroy.ref: type = name_ref SubobjectDestroy, imports.%Core.SubobjectDestroy [concrete = constants.%SubobjectDestroy.type]
// CHECK:STDOUT:   %SubobjectDestroy.facet: %SubobjectDestroy.type = facet_value %i32.loc7, (constants.%custom_witness.b80ef6.2) [concrete = constants.%SubobjectDestroy.facet.49a]
// CHECK:STDOUT:   %.loc7: %SubobjectDestroy.type = converted %i32.loc7, %SubobjectDestroy.facet [concrete = constants.%SubobjectDestroy.facet.49a]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %x.var: ref %i32 = var_storage %x.var_patt
// CHECK:STDOUT:   %MakeInt.ref: %MakeInt.type = name_ref MakeInt, file.%MakeInt.decl [concrete = constants.%MakeInt]
// CHECK:STDOUT:   %MakeInt.call: init %i32 = call %MakeInt.ref()
// CHECK:STDOUT:   assign %x.var, %MakeInt.call
// CHECK:STDOUT:   %i32.loc14: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %x: ref %i32 = wrapper_binding x, %x.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %pattern_type.d94 = ref_binding_pattern x [concrete = constants.%x.patt]
// CHECK:STDOUT:     %x.var_patt: %pattern_type.d94 = var_pattern %x.patt [concrete = constants.%x.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.facet.loc14_3.1: %Destroy.type = facet_value constants.%i32, (constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.077) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc14_3.1: %Destroy.type = converted constants.%i32, %Destroy.facet.loc14_3.1 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %Destroy.facet.loc14_3.2: %Destroy.type = facet_value constants.%i32, (constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.077) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc14_3.2: %Destroy.type = converted constants.%i32, %Destroy.facet.loc14_3.2 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%x.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc7_7.1(%self.param: ref %i32.builtin) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc7_7.2(%self.param: ref %i32) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: --- trivial_int_literal.carbon = $trivial_int_literal
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %pattern_type.dc0: type = pattern_type Core.IntLiteral [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.687: type = fn_type @SubobjectDestroy.WithSelf.Op.loc5 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f: %SubobjectDestroy.WithSelf.Op.type.687 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.659: %SubobjectDestroy.type = facet_value Core.IntLiteral, (%custom_witness.b80) [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %DestroyImpl.type: type = facet_type <@DestroyImpl> [concrete]
// CHECK:STDOUT:   %T.09f: %SubobjectDestroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.type.dea: type = fn_type @T.as_type.as.DestroyImpl.impl.Op, @T.as_type.as.DestroyImpl.impl(%T.09f) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.6e5: %T.as_type.as.DestroyImpl.impl.Op.type.dea = struct_value () [symbolic]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.9b4: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.659) [concrete]
// CHECK:STDOUT:   %Destroy.facet: %Destroy.type = facet_value Core.IntLiteral, (%custom_witness.b80, %DestroyImpl.impl_witness.9b4) [concrete]
// CHECK:STDOUT:   %TrivialDestroy.type: type = facet_type <@TrivialDestroy> [concrete]
// CHECK:STDOUT:   %x.patt: %pattern_type.dc0 = ref_binding_pattern x [concrete]
// CHECK:STDOUT:   %x.var_patt: %pattern_type.dc0 = var_pattern %x.patt [concrete]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Copy.impl_witness.0b3: <witness> = impl_witness imports.%Copy.impl_witness_table.ee5 [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value Core.IntLiteral, (%Copy.impl_witness.0b3) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.d4d: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet) [concrete]
// CHECK:STDOUT:   %.1b2: type = fn_type_with_self_type %Copy.WithSelf.Op.type.d4d, %Copy.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.Copy.impl.Op.type: type = fn_type @Core.IntLiteral.as.Copy.impl.Op [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.Copy.impl.Op: %Core.IntLiteral.as.Copy.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.Copy.impl.Op.bound: <bound method> = bound_method %int_0, %Core.IntLiteral.as.Copy.impl.Op [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .IntLiteral = %Core.IntLiteral
// CHECK:STDOUT:     .SubobjectDestroy = %Core.SubobjectDestroy
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     .TrivialDestroy = %Core.TrivialDestroy
// CHECK:STDOUT:     .DestroyImpl = %Core.DestroyImpl
// CHECK:STDOUT:     .Copy = %Core.Copy
// CHECK:STDOUT:     .TrivialDestroyImpl = %Core.TrivialDestroyImpl
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.IntLiteral: type = import_ref Core//prelude/parts/int_literal, IntLiteral, loaded [concrete = Core.IntLiteral]
// CHECK:STDOUT:   %Core.SubobjectDestroy: type = import_ref Core//prelude/parts/destroy, SubobjectDestroy, loaded [concrete = constants.%SubobjectDestroy.type]
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Core.import_ref.74d: @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op.type (%T.as_type.as.DestroyImpl.impl.Op.type.dea) = import_ref Core//prelude/parts/destroy, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op (constants.%T.as_type.as.DestroyImpl.impl.Op.6e5)]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table = impl_witness_table (%Core.import_ref.74d), @T.as_type.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT:   %Core.TrivialDestroy: type = import_ref Core//prelude/parts/destroy, TrivialDestroy, loaded [concrete = constants.%TrivialDestroy.type]
// CHECK:STDOUT:   %Core.DestroyImpl: type = import_ref Core//prelude/parts/destroy, DestroyImpl, loaded [concrete = constants.%DestroyImpl.type]
// CHECK:STDOUT:   %Core.Copy: type = import_ref Core//prelude/parts/copy, Copy, loaded [concrete = constants.%Copy.type]
// CHECK:STDOUT:   %Core.import_ref.9c3: %Core.IntLiteral.as.Copy.impl.Op.type = import_ref Core//prelude/parts/copy, loc{{\d+_\d+}}, loaded [concrete = constants.%Core.IntLiteral.as.Copy.impl.Op]
// CHECK:STDOUT:   %Copy.impl_witness_table.ee5 = impl_witness_table (%Core.import_ref.9c3), @Core.IntLiteral.as.Copy.impl [concrete]
// CHECK:STDOUT:   %Core.TrivialDestroyImpl: type = import_ref Core//prelude/parts/destroy, TrivialDestroyImpl, loaded [concrete = constants.%TrivialDestroyImpl.type]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Test() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Core.ref.loc5_3: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:   %IntLiteral.ref.loc5: type = name_ref IntLiteral, imports.%Core.IntLiteral [concrete = Core.IntLiteral]
// CHECK:STDOUT:   %Core.ref.loc5_22: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:   %SubobjectDestroy.ref: type = name_ref SubobjectDestroy, imports.%Core.SubobjectDestroy [concrete = constants.%SubobjectDestroy.type]
// CHECK:STDOUT:   %SubobjectDestroy.facet: %SubobjectDestroy.type = facet_value %IntLiteral.ref.loc5, (constants.%custom_witness.b80) [concrete = constants.%SubobjectDestroy.facet.659]
// CHECK:STDOUT:   %.loc5: %SubobjectDestroy.type = converted %IntLiteral.ref.loc5, %SubobjectDestroy.facet [concrete = constants.%SubobjectDestroy.facet.659]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %x.var: ref Core.IntLiteral = var_storage %x.var_patt
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT:   %impl.elem0: %.1b2 = impl_witness_access constants.%Copy.impl_witness.0b3, element0 [concrete = constants.%Core.IntLiteral.as.Copy.impl.Op]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %int_0, %impl.elem0 [concrete = constants.%Core.IntLiteral.as.Copy.impl.Op.bound]
// CHECK:STDOUT:   %Core.IntLiteral.as.Copy.impl.Op.call: init Core.IntLiteral = call %bound_method(%int_0) [concrete = constants.%int_0]
// CHECK:STDOUT:   assign %x.var, %Core.IntLiteral.as.Copy.impl.Op.call
// CHECK:STDOUT:   %.loc12_21: type = splice_block %IntLiteral.ref.loc12 [concrete = Core.IntLiteral] {
// CHECK:STDOUT:     %Core.ref.loc12: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:     %IntLiteral.ref.loc12: type = name_ref IntLiteral, imports.%Core.IntLiteral [concrete = Core.IntLiteral]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %x: ref Core.IntLiteral = wrapper_binding x, %x.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %pattern_type.dc0 = ref_binding_pattern x [concrete = constants.%x.patt]
// CHECK:STDOUT:     %x.var_patt: %pattern_type.dc0 = var_pattern %x.patt [concrete = constants.%x.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.facet.loc12_3.1: %Destroy.type = facet_value Core.IntLiteral, (constants.%custom_witness.b80, constants.%DestroyImpl.impl_witness.9b4) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc12_3.1: %Destroy.type = converted Core.IntLiteral, %Destroy.facet.loc12_3.1 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %Destroy.facet.loc12_3.2: %Destroy.type = facet_value Core.IntLiteral, (constants.%custom_witness.b80, constants.%DestroyImpl.impl_witness.9b4) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc12_3.2: %Destroy.type = converted Core.IntLiteral, %Destroy.facet.loc12_3.2 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%x.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc5(%self.param: ref Core.IntLiteral) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: --- trivial_float.carbon = $trivial_float
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %int_64: Core.IntLiteral = int_value 64 [concrete]
// CHECK:STDOUT:   %Float.type: type = parameterized_class_name_type @Float [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Float.generic: %Float.type = struct_value () [concrete]
// CHECK:STDOUT:   %f64.761: type = class_type @Float, @Float(%int_64) [concrete]
// CHECK:STDOUT:   %pattern_type.667: type = pattern_type %f64.761 [concrete]
// CHECK:STDOUT:   %MakeFloat.type: type = fn_type @MakeFloat [concrete]
// CHECK:STDOUT:   %MakeFloat: %MakeFloat.type = struct_value () [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %f64.794: type = float_type %int_64, f64 [concrete]
// CHECK:STDOUT:   %DestroyImpl.type: type = facet_type <@DestroyImpl> [concrete]
// CHECK:STDOUT:   %T.09f: %SubobjectDestroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.type.dea: type = fn_type @T.as_type.as.DestroyImpl.impl.Op, @T.as_type.as.DestroyImpl.impl(%T.09f) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.6e5: %T.as_type.as.DestroyImpl.impl.Op.type.dea = struct_value () [symbolic]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.2: type = fn_type @SubobjectDestroy.WithSelf.Op.loc7_7.2 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.2: %SubobjectDestroy.WithSelf.Op.type.6870fd.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.2: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.2), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.47f: %SubobjectDestroy.type = facet_value %f64.761, (%custom_witness.b80ef6.2) [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.a23: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.47f) [concrete]
// CHECK:STDOUT:   %Destroy.facet: %Destroy.type = facet_value %f64.761, (%custom_witness.b80ef6.2, %DestroyImpl.impl_witness.a23) [concrete]
// CHECK:STDOUT:   %TrivialDestroy.type: type = facet_type <@TrivialDestroy> [concrete]
// CHECK:STDOUT:   %x.patt: %pattern_type.667 = ref_binding_pattern x [concrete]
// CHECK:STDOUT:   %x.var_patt: %pattern_type.667 = var_pattern %x.patt [concrete]
// CHECK:STDOUT:   %SelfDestruct.type: type = fn_type @SelfDestruct [concrete]
// CHECK:STDOUT:   %SelfDestruct: %SelfDestruct.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Float = %Core.Float
// CHECK:STDOUT:     .SubobjectDestroy = %Core.SubobjectDestroy
// CHECK:STDOUT:     .DestroyImpl = %Core.DestroyImpl
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     .TrivialDestroy = %Core.TrivialDestroy
// CHECK:STDOUT:     .SelfDestruct = %Core.SelfDestruct
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Float: %Float.type = import_ref Core//prelude/parts/float, Float, loaded [concrete = constants.%Float.generic]
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
// CHECK:STDOUT:   %f64.loc7: type = type_literal constants.%f64.761 [concrete = constants.%f64.761]
// CHECK:STDOUT:   %Core.ref.loc7: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:   %SubobjectDestroy.ref: type = name_ref SubobjectDestroy, imports.%Core.SubobjectDestroy [concrete = constants.%SubobjectDestroy.type]
// CHECK:STDOUT:   %SubobjectDestroy.facet: %SubobjectDestroy.type = facet_value %f64.loc7, (constants.%custom_witness.b80ef6.2) [concrete = constants.%SubobjectDestroy.facet.47f]
// CHECK:STDOUT:   %.loc7: %SubobjectDestroy.type = converted %f64.loc7, %SubobjectDestroy.facet [concrete = constants.%SubobjectDestroy.facet.47f]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %x.var: ref %f64.761 = var_storage %x.var_patt
// CHECK:STDOUT:   %MakeFloat.ref: %MakeFloat.type = name_ref MakeFloat, file.%MakeFloat.decl [concrete = constants.%MakeFloat]
// CHECK:STDOUT:   %MakeFloat.call: init %f64.761 = call %MakeFloat.ref()
// CHECK:STDOUT:   assign %x.var, %MakeFloat.call
// CHECK:STDOUT:   %f64.loc14: type = type_literal constants.%f64.761 [concrete = constants.%f64.761]
// CHECK:STDOUT:   %x: ref %f64.761 = wrapper_binding x, %x.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %pattern_type.667 = ref_binding_pattern x [concrete = constants.%x.patt]
// CHECK:STDOUT:     %x.var_patt: %pattern_type.667 = var_pattern %x.patt [concrete = constants.%x.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.facet.loc14_3.1: %Destroy.type = facet_value constants.%f64.761, (constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.a23) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc14_3.1: %Destroy.type = converted constants.%f64.761, %Destroy.facet.loc14_3.1 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %Destroy.facet.loc14_3.2: %Destroy.type = facet_value constants.%f64.761, (constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.a23) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc14_3.2: %Destroy.type = converted constants.%f64.761, %Destroy.facet.loc14_3.2 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%x.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc7_7.1(%self.param: ref %f64.794) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc7_7.2(%self.param: ref %f64.761) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: --- todo_trivial_float_literal.carbon = $todo_trivial_float_literal
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Test() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- trivial_uint.carbon = $trivial_uint
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %UInt.type: type = parameterized_class_name_type @UInt [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %UInt.generic: %UInt.type = struct_value () [concrete]
// CHECK:STDOUT:   %u32: type = class_type @UInt, @UInt(%int_32) [concrete]
// CHECK:STDOUT:   %pattern_type.129: type = pattern_type %u32 [concrete]
// CHECK:STDOUT:   %MakeUint.type: type = fn_type @MakeUint [concrete]
// CHECK:STDOUT:   %MakeUint: %MakeUint.type = struct_value () [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %u32.builtin: type = int_type unsigned, %int_32 [concrete]
// CHECK:STDOUT:   %DestroyImpl.type: type = facet_type <@DestroyImpl> [concrete]
// CHECK:STDOUT:   %T.09f: %SubobjectDestroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.type.dea: type = fn_type @T.as_type.as.DestroyImpl.impl.Op, @T.as_type.as.DestroyImpl.impl(%T.09f) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.6e5: %T.as_type.as.DestroyImpl.impl.Op.type.dea = struct_value () [symbolic]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.2: type = fn_type @SubobjectDestroy.WithSelf.Op.loc7_7.2 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.2: %SubobjectDestroy.WithSelf.Op.type.6870fd.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.2: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.2), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.439: %SubobjectDestroy.type = facet_value %u32, (%custom_witness.b80ef6.2) [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.ee7: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.439) [concrete]
// CHECK:STDOUT:   %Destroy.facet: %Destroy.type = facet_value %u32, (%custom_witness.b80ef6.2, %DestroyImpl.impl_witness.ee7) [concrete]
// CHECK:STDOUT:   %TrivialDestroy.type: type = facet_type <@TrivialDestroy> [concrete]
// CHECK:STDOUT:   %x.patt: %pattern_type.129 = ref_binding_pattern x [concrete]
// CHECK:STDOUT:   %x.var_patt: %pattern_type.129 = var_pattern %x.patt [concrete]
// CHECK:STDOUT:   %SelfDestruct.type: type = fn_type @SelfDestruct [concrete]
// CHECK:STDOUT:   %SelfDestruct: %SelfDestruct.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .UInt = %Core.UInt
// CHECK:STDOUT:     .SubobjectDestroy = %Core.SubobjectDestroy
// CHECK:STDOUT:     .DestroyImpl = %Core.DestroyImpl
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     .TrivialDestroy = %Core.TrivialDestroy
// CHECK:STDOUT:     .SelfDestruct = %Core.SelfDestruct
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.UInt: %UInt.type = import_ref Core//prelude/parts/uint, UInt, loaded [concrete = constants.%UInt.generic]
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
// CHECK:STDOUT:   %u32.loc7: type = type_literal constants.%u32 [concrete = constants.%u32]
// CHECK:STDOUT:   %Core.ref.loc7: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:   %SubobjectDestroy.ref: type = name_ref SubobjectDestroy, imports.%Core.SubobjectDestroy [concrete = constants.%SubobjectDestroy.type]
// CHECK:STDOUT:   %SubobjectDestroy.facet: %SubobjectDestroy.type = facet_value %u32.loc7, (constants.%custom_witness.b80ef6.2) [concrete = constants.%SubobjectDestroy.facet.439]
// CHECK:STDOUT:   %.loc7: %SubobjectDestroy.type = converted %u32.loc7, %SubobjectDestroy.facet [concrete = constants.%SubobjectDestroy.facet.439]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %x.var: ref %u32 = var_storage %x.var_patt
// CHECK:STDOUT:   %MakeUint.ref: %MakeUint.type = name_ref MakeUint, file.%MakeUint.decl [concrete = constants.%MakeUint]
// CHECK:STDOUT:   %MakeUint.call: init %u32 = call %MakeUint.ref()
// CHECK:STDOUT:   assign %x.var, %MakeUint.call
// CHECK:STDOUT:   %u32.loc14: type = type_literal constants.%u32 [concrete = constants.%u32]
// CHECK:STDOUT:   %x: ref %u32 = wrapper_binding x, %x.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %pattern_type.129 = ref_binding_pattern x [concrete = constants.%x.patt]
// CHECK:STDOUT:     %x.var_patt: %pattern_type.129 = var_pattern %x.patt [concrete = constants.%x.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.facet.loc14_3.1: %Destroy.type = facet_value constants.%u32, (constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.ee7) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc14_3.1: %Destroy.type = converted constants.%u32, %Destroy.facet.loc14_3.1 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %Destroy.facet.loc14_3.2: %Destroy.type = facet_value constants.%u32, (constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.ee7) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc14_3.2: %Destroy.type = converted constants.%u32, %Destroy.facet.loc14_3.2 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%x.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc7_7.1(%self.param: ref %u32.builtin) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc7_7.2(%self.param: ref %u32) = "no_op";
// CHECK:STDOUT:
