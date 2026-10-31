// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/auto/bindings.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/auto/bindings.carbon

// --- var.carbon
library "[[@TEST_NAME]]";

class A {}
impl A as Core.Copy {
    fn Op(self) -> Self { return self; }
}

fn F() {
    var a: A = {};
//@dump-sem-ir-begin
    var unused x: auto = a;
    var unused y: auto = {} as A;
    var unused z: auto = 20;
//@dump-sem-ir-end
}

// --- let.carbon
library "[[@TEST_NAME]]";

class A {}
fn F() {
    let a: A = {};
//@dump-sem-ir-begin
    let unused x: auto = a;
    let unused y: auto = {} as A;
    let unused z: auto = 20;
//@dump-sem-ir-end
}

// CHECK:STDOUT: --- var.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %A: type = class_type @A [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Copy.impl_witness.23e: <witness> = impl_witness @A.as.Copy.impl.%Copy.impl_witness_table [concrete]
// CHECK:STDOUT:   %A.as.Copy.impl.Op.type: type = fn_type @A.as.Copy.impl.Op [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %A.as.Copy.impl.Op: %A.as.Copy.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet.2d2: %Copy.type = facet_value %A, (%Copy.impl_witness.23e) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.527: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet.2d2) [concrete]
// CHECK:STDOUT:   %.e9e: type = fn_type_with_self_type %Copy.WithSelf.Op.type.527, %Copy.facet.2d2 [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %A.val: %A = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.f6d: type = pattern_type auto [concrete]
// CHECK:STDOUT:   %x.patt: %pattern_type.f6d = ref_binding_pattern x [concrete]
// CHECK:STDOUT:   %x.var_patt: %pattern_type.f6d = var_pattern %x.patt [concrete]
// CHECK:STDOUT:   %y.patt: %pattern_type.f6d = ref_binding_pattern y [concrete]
// CHECK:STDOUT:   %y.var_patt: %pattern_type.f6d = var_pattern %y.patt [concrete]
// CHECK:STDOUT:   %z.patt: %pattern_type.f6d = ref_binding_pattern z [concrete]
// CHECK:STDOUT:   %z.var_patt: %pattern_type.f6d = var_pattern %z.patt [concrete]
// CHECK:STDOUT:   %int_20: Core.IntLiteral = int_value 20 [concrete]
// CHECK:STDOUT:   %Copy.impl_witness.0b3: <witness> = impl_witness imports.%Copy.impl_witness_table.ee5 [concrete]
// CHECK:STDOUT:   %Copy.facet.d66: %Copy.type = facet_value Core.IntLiteral, (%Copy.impl_witness.0b3) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.d4d: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet.d66) [concrete]
// CHECK:STDOUT:   %.1b2: type = fn_type_with_self_type %Copy.WithSelf.Op.type.d4d, %Copy.facet.d66 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.Copy.impl.Op.type: type = fn_type @Core.IntLiteral.as.Copy.impl.Op [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.Copy.impl.Op: %Core.IntLiteral.as.Copy.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.Copy.impl.Op.bound: <bound method> = bound_method %int_20, %Core.IntLiteral.as.Copy.impl.Op [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %T.09f: %SubobjectDestroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.type.dea: type = fn_type @T.as_type.as.DestroyImpl.impl.Op, @T.as_type.as.DestroyImpl.impl(%T.09f) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.6e5: %T.as_type.as.DestroyImpl.impl.Op.type.dea = struct_value () [symbolic]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.1: type = fn_type @SubobjectDestroy.WithSelf.Op.loc13 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.1: %SubobjectDestroy.WithSelf.Op.type.6870fd.1 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.1: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.1), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.659: %SubobjectDestroy.type = facet_value Core.IntLiteral, (%custom_witness.b80ef6.1) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.9b4: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.659) [concrete]
// CHECK:STDOUT:   %Destroy.facet.4c6: %Destroy.type = facet_value Core.IntLiteral, (%DestroyImpl.impl_witness.9b4, %custom_witness.b80ef6.1) [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.3: type = fn_type @SubobjectDestroy.WithSelf.Op.loc12_5.2 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.3: %SubobjectDestroy.WithSelf.Op.type.6870fd.3 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.3: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.3), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.9bf: %SubobjectDestroy.type = facet_value %A, (%custom_witness.b80ef6.3) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.6d1: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.9bf) [concrete]
// CHECK:STDOUT:   %Destroy.facet.35e: %Destroy.type = facet_value %A, (%DestroyImpl.impl_witness.6d1, %custom_witness.b80ef6.3) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.9c3: %Core.IntLiteral.as.Copy.impl.Op.type = import_ref Core//prelude/parts/copy, loc{{\d+_\d+}}, loaded [concrete = constants.%Core.IntLiteral.as.Copy.impl.Op]
// CHECK:STDOUT:   %Copy.impl_witness_table.ee5 = impl_witness_table (%Core.import_ref.9c3), @Core.IntLiteral.as.Copy.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.74d: @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op.type (%T.as_type.as.DestroyImpl.impl.Op.type.dea) = import_ref Core//prelude/parts/destroy, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op (constants.%T.as_type.as.DestroyImpl.impl.Op.6e5)]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table = impl_witness_table (%Core.import_ref.74d), @T.as_type.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %x.var: ref %A = var_storage %x.var_patt
// CHECK:STDOUT:   %a.ref: ref %A = name_ref a, %a
// CHECK:STDOUT:   %.loc11_26: %A = acquire_value %a.ref
// CHECK:STDOUT:   %impl.elem0.loc11: %.e9e = impl_witness_access constants.%Copy.impl_witness.23e, element0 [concrete = constants.%A.as.Copy.impl.Op]
// CHECK:STDOUT:   %bound_method.loc11: <bound method> = bound_method %.loc11_26, %impl.elem0.loc11
// CHECK:STDOUT:   %.loc11_5.1: ref %A = splice_block %x.var {}
// CHECK:STDOUT:   %A.as.Copy.impl.Op.call: init %A to %.loc11_5.1 = call %bound_method.loc11(%.loc11_26)
// CHECK:STDOUT:   assign %x.var, %A.as.Copy.impl.Op.call
// CHECK:STDOUT:   %x: ref %A = wrapper_binding x, %x.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %pattern_type.f6d = ref_binding_pattern x [concrete = constants.%x.patt]
// CHECK:STDOUT:     %x.var_patt: %pattern_type.f6d = var_pattern %x.patt [concrete = constants.%x.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %y.var: ref %A = var_storage %y.var_patt
// CHECK:STDOUT:   %.loc12_27.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %A.ref.loc12: type = name_ref A, file.%A.decl [concrete = constants.%A]
// CHECK:STDOUT:   %.loc12_5.1: ref %A = splice_block %y.var {}
// CHECK:STDOUT:   %.loc12_27.2: init %A to %.loc12_5.1 = class_init () [concrete = constants.%A.val]
// CHECK:STDOUT:   %.loc12_29: init %A = converted %.loc12_27.1, %.loc12_27.2 [concrete = constants.%A.val]
// CHECK:STDOUT:   assign %y.var, %.loc12_29
// CHECK:STDOUT:   %y: ref %A = wrapper_binding y, %y.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %y.patt: %pattern_type.f6d = ref_binding_pattern y [concrete = constants.%y.patt]
// CHECK:STDOUT:     %y.var_patt: %pattern_type.f6d = var_pattern %y.patt [concrete = constants.%y.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %z.var: ref Core.IntLiteral = var_storage %z.var_patt
// CHECK:STDOUT:   %int_20: Core.IntLiteral = int_value 20 [concrete = constants.%int_20]
// CHECK:STDOUT:   %impl.elem0.loc13: %.1b2 = impl_witness_access constants.%Copy.impl_witness.0b3, element0 [concrete = constants.%Core.IntLiteral.as.Copy.impl.Op]
// CHECK:STDOUT:   %bound_method.loc13: <bound method> = bound_method %int_20, %impl.elem0.loc13 [concrete = constants.%Core.IntLiteral.as.Copy.impl.Op.bound]
// CHECK:STDOUT:   %Core.IntLiteral.as.Copy.impl.Op.call: init Core.IntLiteral = call %bound_method.loc13(%int_20) [concrete = constants.%int_20]
// CHECK:STDOUT:   assign %z.var, %Core.IntLiteral.as.Copy.impl.Op.call
// CHECK:STDOUT:   %z: ref Core.IntLiteral = wrapper_binding z, %z.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %z.patt: %pattern_type.f6d = ref_binding_pattern z [concrete = constants.%z.patt]
// CHECK:STDOUT:     %z.var_patt: %pattern_type.f6d = var_pattern %z.patt [concrete = constants.%z.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.facet.loc13_5.1: %Destroy.type = facet_value Core.IntLiteral, (constants.%DestroyImpl.impl_witness.9b4, constants.%custom_witness.b80ef6.1) [concrete = constants.%Destroy.facet.4c6]
// CHECK:STDOUT:   %.loc13_5.1: %Destroy.type = converted Core.IntLiteral, %Destroy.facet.loc13_5.1 [concrete = constants.%Destroy.facet.4c6]
// CHECK:STDOUT:   %Destroy.facet.loc13_5.2: %Destroy.type = facet_value Core.IntLiteral, (constants.%DestroyImpl.impl_witness.9b4, constants.%custom_witness.b80ef6.1) [concrete = constants.%Destroy.facet.4c6]
// CHECK:STDOUT:   %.loc13_5.2: %Destroy.type = converted Core.IntLiteral, %Destroy.facet.loc13_5.2 [concrete = constants.%Destroy.facet.4c6]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc13: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%z.var)
// CHECK:STDOUT:   %Destroy.facet.loc12_5.1: %Destroy.type = facet_value constants.%A, (constants.%DestroyImpl.impl_witness.6d1, constants.%custom_witness.b80ef6.3) [concrete = constants.%Destroy.facet.35e]
// CHECK:STDOUT:   %.loc12_5.2: %Destroy.type = converted constants.%A, %Destroy.facet.loc12_5.1 [concrete = constants.%Destroy.facet.35e]
// CHECK:STDOUT:   %Destroy.facet.loc12_5.2: %Destroy.type = facet_value constants.%A, (constants.%DestroyImpl.impl_witness.6d1, constants.%custom_witness.b80ef6.3) [concrete = constants.%Destroy.facet.35e]
// CHECK:STDOUT:   %.loc12_5.3: %Destroy.type = converted constants.%A, %Destroy.facet.loc12_5.2 [concrete = constants.%Destroy.facet.35e]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc12: init %empty_tuple.type = call %SelfDestruct.specific_fn.2(%y.var)
// CHECK:STDOUT:   %Destroy.facet.loc11_5.1: %Destroy.type = facet_value constants.%A, (constants.%DestroyImpl.impl_witness.6d1, constants.%custom_witness.b80ef6.3) [concrete = constants.%Destroy.facet.35e]
// CHECK:STDOUT:   %.loc11_5.2: %Destroy.type = converted constants.%A, %Destroy.facet.loc11_5.1 [concrete = constants.%Destroy.facet.35e]
// CHECK:STDOUT:   %Destroy.facet.loc11_5.2: %Destroy.type = facet_value constants.%A, (constants.%DestroyImpl.impl_witness.6d1, constants.%custom_witness.b80ef6.3) [concrete = constants.%Destroy.facet.35e]
// CHECK:STDOUT:   %.loc11_5.3: %Destroy.type = converted constants.%A, %Destroy.facet.loc11_5.2 [concrete = constants.%Destroy.facet.35e]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc11: init %empty_tuple.type = call %SelfDestruct.specific_fn.3(%x.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc13(%self.param: ref Core.IntLiteral) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc12_5.1(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc12_5.2(%self.param: ref %A) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: --- let.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %A: type = class_type @A [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %A.val: %A = struct_value () [concrete]
// CHECK:STDOUT:   %.367: ref %A = temporary invalid, %A.val [concrete]
// CHECK:STDOUT:   %pattern_type.f6d: type = pattern_type auto [concrete]
// CHECK:STDOUT:   %x.patt: %pattern_type.f6d = value_binding_pattern x [concrete]
// CHECK:STDOUT:   %y.patt: %pattern_type.f6d = value_binding_pattern y [concrete]
// CHECK:STDOUT:   %z.patt: %pattern_type.f6d = value_binding_pattern z [concrete]
// CHECK:STDOUT:   %int_20: Core.IntLiteral = int_value 20 [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %T.09f: %SubobjectDestroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.type.dea: type = fn_type @T.as_type.as.DestroyImpl.impl.Op, @T.as_type.as.DestroyImpl.impl(%T.09f) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.6e5: %T.as_type.as.DestroyImpl.impl.Op.type.dea = struct_value () [symbolic]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.2: type = fn_type @SubobjectDestroy.WithSelf.Op.loc8_29.2 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.2: %SubobjectDestroy.WithSelf.Op.type.6870fd.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.2: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.2), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.9bf: %SubobjectDestroy.type = facet_value %A, (%custom_witness.b80ef6.2) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.6d1: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.9bf) [concrete]
// CHECK:STDOUT:   %Destroy.facet: %Destroy.type = facet_value %A, (%DestroyImpl.impl_witness.6d1, %custom_witness.b80ef6.2) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.74d: @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op.type (%T.as_type.as.DestroyImpl.impl.Op.type.dea) = import_ref Core//prelude/parts/destroy, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op (constants.%T.as_type.as.DestroyImpl.impl.Op.6e5)]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table = impl_witness_table (%Core.import_ref.74d), @T.as_type.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %a.ref: %A = name_ref a, %a
// CHECK:STDOUT:   %x: %A = wrapper_binding x, %a.ref
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %pattern_type.f6d = value_binding_pattern x [concrete = constants.%x.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc8_27.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %A.ref.loc8: type = name_ref A, file.%A.decl [concrete = constants.%A]
// CHECK:STDOUT:   %.loc8_27.2: ref %A = temporary_storage
// CHECK:STDOUT:   %.loc8_27.3: init %A to %.loc8_27.2 = class_init () [concrete = constants.%A.val]
// CHECK:STDOUT:   %.loc8_29.1: init %A = converted %.loc8_27.1, %.loc8_27.3 [concrete = constants.%A.val]
// CHECK:STDOUT:   %.loc8_29.2: ref %A = temporary %.loc8_27.2, %.loc8_29.1 [concrete = constants.%.367]
// CHECK:STDOUT:   %.loc8_29.3: %A = acquire_value %.loc8_29.2 [concrete = constants.%A.val]
// CHECK:STDOUT:   %y: %A = wrapper_binding y, %.loc8_29.3
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %y.patt: %pattern_type.f6d = value_binding_pattern y [concrete = constants.%y.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int_20: Core.IntLiteral = int_value 20 [concrete = constants.%int_20]
// CHECK:STDOUT:   %z: Core.IntLiteral = wrapper_binding z, %int_20
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %z.patt: %pattern_type.f6d = value_binding_pattern z [concrete = constants.%z.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.facet.loc8_29.1: %Destroy.type = facet_value constants.%A, (constants.%DestroyImpl.impl_witness.6d1, constants.%custom_witness.b80ef6.2) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc8_29.4: %Destroy.type = converted constants.%A, %Destroy.facet.loc8_29.1 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %Destroy.facet.loc8_29.2: %Destroy.type = facet_value constants.%A, (constants.%DestroyImpl.impl_witness.6d1, constants.%custom_witness.b80ef6.2) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc8_29.5: %Destroy.type = converted constants.%A, %Destroy.facet.loc8_29.2 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc8: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%.loc8_29.2)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc8_29.1(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc8_29.2(%self.param: ref %A) = "no_op";
// CHECK:STDOUT:
