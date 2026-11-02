// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/primitives.carbon
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

fn F() -> bool { return true; }
fn G() {
//@dump-sem-ir-begin
    var x: auto = F();
    var y: auto = {} as A;
    var z: auto = 20;
    var unused w: auto = (x, y, z);
//@dump-sem-ir-end
}

// --- let.carbon
library "[[@TEST_NAME]]";

class A {}
fn F() -> bool { return true; }
fn G() {
//@dump-sem-ir-begin
    let x: auto = F();
    let y: auto = {} as A;
    let z: auto = 20;
    let unused w: auto = (x, y, z);
//@dump-sem-ir-end
}

// --- fail_todo_tuple.carbon
library "[[@TEST_NAME]]";

fn F() {
    // CHECK:STDERR: fail_todo_tuple.carbon:[[@LINE+2]]:20-23: error: semantics TODO: `HandleAutoTypeLiteral` [SemanticsTodo]
    // CHECK:STDERR:
    let unused x: (auto, auto) = (86, false);
    var unused y: (auto, auto) = ({}, "str");
}

// --- fail_uninitialized_var.carbon
library "[[@TEST_NAME]]";

fn F() {
    // CHECK:STDERR: fail_uninitialized_var.carbon:[[@LINE+2]]:5-23: error: expected `=`; `var` declaration with type `auto` must have an initializer [ExpectedInitializerAfterAutoTypedVar]
    // CHECK:STDERR:
    var unused x: auto;
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
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.impl_witness.5e1: <witness> = impl_witness imports.%Copy.impl_witness_table.bda [concrete]
// CHECK:STDOUT:   %Copy.facet.7af: %Copy.type = facet_value bool, (%Copy.impl_witness.5e1) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.c98: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet.7af) [concrete]
// CHECK:STDOUT:   %.68a: type = fn_type_with_self_type %Copy.WithSelf.Op.type.c98, %Copy.facet.7af [concrete]
// CHECK:STDOUT:   %bool.as.Copy.impl.Op.type: type = fn_type @bool.as.Copy.impl.Op [concrete]
// CHECK:STDOUT:   %bool.as.Copy.impl.Op: %bool.as.Copy.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.f6d: type = pattern_type auto [concrete]
// CHECK:STDOUT:   %x.patt: %pattern_type.f6d = ref_binding_pattern x [concrete]
// CHECK:STDOUT:   %x.var_patt: %pattern_type.f6d = var_pattern %x.patt [concrete]
// CHECK:STDOUT:   %y.patt: %pattern_type.f6d = ref_binding_pattern y [concrete]
// CHECK:STDOUT:   %y.var_patt: %pattern_type.f6d = var_pattern %y.patt [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %A.val: %A = struct_value () [concrete]
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
// CHECK:STDOUT:   %w.patt: %pattern_type.f6d = ref_binding_pattern w [concrete]
// CHECK:STDOUT:   %w.var_patt: %pattern_type.f6d = var_pattern %w.patt [concrete]
// CHECK:STDOUT:   %tuple.type.545: type = tuple_type (bool, %A, Core.IntLiteral) [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %T.09f: %SubobjectDestroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.type.dea: type = fn_type @T.as_type.as.DestroyImpl.impl.Op, @T.as_type.as.DestroyImpl.impl(%T.09f) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.6e5: %T.as_type.as.DestroyImpl.impl.Op.type.dea = struct_value () [symbolic]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.1: type = fn_type @SubobjectDestroy.WithSelf.Op.loc16_5.1 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.1: %SubobjectDestroy.WithSelf.Op.type.6870fd.1 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.1: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.1), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.3: type = fn_type @SubobjectDestroy.WithSelf.Op.loc16_5.3 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.3: %SubobjectDestroy.WithSelf.Op.type.6870fd.3 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.3: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.3), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.4: type = fn_type @SubobjectDestroy.WithSelf.Op.loc16_5.4 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.4: %SubobjectDestroy.WithSelf.Op.type.6870fd.4 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.4: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.4), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.5: type = fn_type @SubobjectDestroy.WithSelf.Op.loc16_5.5 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.5: %SubobjectDestroy.WithSelf.Op.type.6870fd.5 = struct_value () [concrete]
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.659: %SubobjectDestroy.type = facet_value Core.IntLiteral, (%custom_witness.b80ef6.4) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.9b4: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.659) [concrete]
// CHECK:STDOUT:   %Destroy.facet.4c6: %Destroy.type = facet_value Core.IntLiteral, (%DestroyImpl.impl_witness.9b4, %custom_witness.b80ef6.4) [concrete]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.9bf: %SubobjectDestroy.type = facet_value %A, (%custom_witness.b80ef6.3) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.6d1: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.9bf) [concrete]
// CHECK:STDOUT:   %Destroy.facet.35e: %Destroy.type = facet_value %A, (%DestroyImpl.impl_witness.6d1, %custom_witness.b80ef6.3) [concrete]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.a87: %SubobjectDestroy.type = facet_value bool, (%custom_witness.b80ef6.1) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.778: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.a87) [concrete]
// CHECK:STDOUT:   %Destroy.facet.e68: %Destroy.type = facet_value bool, (%DestroyImpl.impl_witness.778, %custom_witness.b80ef6.1) [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.5: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.5), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.418: %SubobjectDestroy.type = facet_value %tuple.type.545, (%custom_witness.b80ef6.5) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.7fe: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.418) [concrete]
// CHECK:STDOUT:   %Destroy.facet.919: %Destroy.type = facet_value %tuple.type.545, (%DestroyImpl.impl_witness.7fe, %custom_witness.b80ef6.5) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.9c3475.1: %bool.as.Copy.impl.Op.type = import_ref Core//prelude/parts/copy, loc{{\d+_\d+}}, loaded [concrete = constants.%bool.as.Copy.impl.Op]
// CHECK:STDOUT:   %Copy.impl_witness_table.bda = impl_witness_table (%Core.import_ref.9c3475.1), @bool.as.Copy.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.9c3475.2: %Core.IntLiteral.as.Copy.impl.Op.type = import_ref Core//prelude/parts/copy, loc{{\d+_\d+}}, loaded [concrete = constants.%Core.IntLiteral.as.Copy.impl.Op]
// CHECK:STDOUT:   %Copy.impl_witness_table.ee5 = impl_witness_table (%Core.import_ref.9c3475.2), @Core.IntLiteral.as.Copy.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.74d: @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op.type (%T.as_type.as.DestroyImpl.impl.Op.type.dea) = import_ref Core//prelude/parts/destroy, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op (constants.%T.as_type.as.DestroyImpl.impl.Op.6e5)]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table = impl_witness_table (%Core.import_ref.74d), @T.as_type.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %x.var: ref bool = var_storage %x.var_patt
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:   %F.call: init bool = call %F.ref()
// CHECK:STDOUT:   assign %x.var, %F.call
// CHECK:STDOUT:   %x: ref bool = wrapper_binding x, %x.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %pattern_type.f6d = ref_binding_pattern x [concrete = constants.%x.patt]
// CHECK:STDOUT:     %x.var_patt: %pattern_type.f6d = var_pattern %x.patt [concrete = constants.%x.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %y.var: ref %A = var_storage %y.var_patt
// CHECK:STDOUT:   %.loc12_20.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %A.ref: type = name_ref A, file.%A.decl [concrete = constants.%A]
// CHECK:STDOUT:   %.loc12_5.1: ref %A = splice_block %y.var {}
// CHECK:STDOUT:   %.loc12_20.2: init %A to %.loc12_5.1 = class_init () [concrete = constants.%A.val]
// CHECK:STDOUT:   %.loc12_22: init %A = converted %.loc12_20.1, %.loc12_20.2 [concrete = constants.%A.val]
// CHECK:STDOUT:   assign %y.var, %.loc12_22
// CHECK:STDOUT:   %y: ref %A = wrapper_binding y, %y.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %y.patt: %pattern_type.f6d = ref_binding_pattern y [concrete = constants.%y.patt]
// CHECK:STDOUT:     %y.var_patt: %pattern_type.f6d = var_pattern %y.patt [concrete = constants.%y.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %z.var: ref Core.IntLiteral = var_storage %z.var_patt
// CHECK:STDOUT:   %int_20: Core.IntLiteral = int_value 20 [concrete = constants.%int_20]
// CHECK:STDOUT:   %impl.elem0.loc13: %.1b2 = impl_witness_access constants.%Copy.impl_witness.0b3, element0 [concrete = constants.%Core.IntLiteral.as.Copy.impl.Op]
// CHECK:STDOUT:   %bound_method.loc13: <bound method> = bound_method %int_20, %impl.elem0.loc13 [concrete = constants.%Core.IntLiteral.as.Copy.impl.Op.bound]
// CHECK:STDOUT:   %Core.IntLiteral.as.Copy.impl.Op.call.loc13: init Core.IntLiteral = call %bound_method.loc13(%int_20) [concrete = constants.%int_20]
// CHECK:STDOUT:   assign %z.var, %Core.IntLiteral.as.Copy.impl.Op.call.loc13
// CHECK:STDOUT:   %z: ref Core.IntLiteral = wrapper_binding z, %z.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %z.patt: %pattern_type.f6d = ref_binding_pattern z [concrete = constants.%z.patt]
// CHECK:STDOUT:     %z.var_patt: %pattern_type.f6d = var_pattern %z.patt [concrete = constants.%z.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %w.var: ref %tuple.type.545 = var_storage %w.var_patt
// CHECK:STDOUT:   %x.ref: ref bool = name_ref x, %x
// CHECK:STDOUT:   %y.ref: ref %A = name_ref y, %y
// CHECK:STDOUT:   %z.ref: ref Core.IntLiteral = name_ref z, %z
// CHECK:STDOUT:   %.loc16_34.1: %tuple.type.545 = tuple_literal (%x.ref, %y.ref, %z.ref)
// CHECK:STDOUT:   %.loc16_27: bool = acquire_value %x.ref
// CHECK:STDOUT:   %impl.elem0.loc16_27: %.68a = impl_witness_access constants.%Copy.impl_witness.5e1, element0 [concrete = constants.%bool.as.Copy.impl.Op]
// CHECK:STDOUT:   %bound_method.loc16_27: <bound method> = bound_method %.loc16_27, %impl.elem0.loc16_27
// CHECK:STDOUT:   %bool.as.Copy.impl.Op.call: init bool = call %bound_method.loc16_27(%.loc16_27)
// CHECK:STDOUT:   %tuple.elem0: ref bool = tuple_access %w.var, element0
// CHECK:STDOUT:   %.loc16_34.2: init bool to %tuple.elem0 = in_place_init %bool.as.Copy.impl.Op.call
// CHECK:STDOUT:   %.loc16_30: %A = acquire_value %y.ref
// CHECK:STDOUT:   %impl.elem0.loc16_30: %.e9e = impl_witness_access constants.%Copy.impl_witness.23e, element0 [concrete = constants.%A.as.Copy.impl.Op]
// CHECK:STDOUT:   %bound_method.loc16_30: <bound method> = bound_method %.loc16_30, %impl.elem0.loc16_30
// CHECK:STDOUT:   %tuple.elem1: ref %A = tuple_access %w.var, element1
// CHECK:STDOUT:   %A.as.Copy.impl.Op.call: init %A to %tuple.elem1 = call %bound_method.loc16_30(%.loc16_30)
// CHECK:STDOUT:   %.loc16_33: Core.IntLiteral = acquire_value %z.ref
// CHECK:STDOUT:   %impl.elem0.loc16_33: %.1b2 = impl_witness_access constants.%Copy.impl_witness.0b3, element0 [concrete = constants.%Core.IntLiteral.as.Copy.impl.Op]
// CHECK:STDOUT:   %bound_method.loc16_33: <bound method> = bound_method %.loc16_33, %impl.elem0.loc16_33
// CHECK:STDOUT:   %Core.IntLiteral.as.Copy.impl.Op.call.loc16: init Core.IntLiteral = call %bound_method.loc16_33(%.loc16_33)
// CHECK:STDOUT:   %tuple.elem2: ref Core.IntLiteral = tuple_access %w.var, element2
// CHECK:STDOUT:   %.loc16_34.3: init Core.IntLiteral to %tuple.elem2 = in_place_init %Core.IntLiteral.as.Copy.impl.Op.call.loc16
// CHECK:STDOUT:   %.loc16_34.4: init %tuple.type.545 to %w.var = tuple_init (%.loc16_34.2, %A.as.Copy.impl.Op.call, %.loc16_34.3)
// CHECK:STDOUT:   %.loc16_5.1: init %tuple.type.545 = converted %.loc16_34.1, %.loc16_34.4
// CHECK:STDOUT:   assign %w.var, %.loc16_5.1
// CHECK:STDOUT:   %w: ref %tuple.type.545 = wrapper_binding w, %w.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %w.patt: %pattern_type.f6d = ref_binding_pattern w [concrete = constants.%w.patt]
// CHECK:STDOUT:     %w.var_patt: %pattern_type.f6d = var_pattern %w.patt [concrete = constants.%w.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.facet.loc16_5.1: %Destroy.type = facet_value constants.%tuple.type.545, (constants.%DestroyImpl.impl_witness.7fe, constants.%custom_witness.b80ef6.5) [concrete = constants.%Destroy.facet.919]
// CHECK:STDOUT:   %.loc16_5.2: %Destroy.type = converted constants.%tuple.type.545, %Destroy.facet.loc16_5.1 [concrete = constants.%Destroy.facet.919]
// CHECK:STDOUT:   %Destroy.facet.loc16_5.2: %Destroy.type = facet_value constants.%tuple.type.545, (constants.%DestroyImpl.impl_witness.7fe, constants.%custom_witness.b80ef6.5) [concrete = constants.%Destroy.facet.919]
// CHECK:STDOUT:   %.loc16_5.3: %Destroy.type = converted constants.%tuple.type.545, %Destroy.facet.loc16_5.2 [concrete = constants.%Destroy.facet.919]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc16: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%w.var)
// CHECK:STDOUT:   %Destroy.facet.loc13_5.1: %Destroy.type = facet_value Core.IntLiteral, (constants.%DestroyImpl.impl_witness.9b4, constants.%custom_witness.b80ef6.4) [concrete = constants.%Destroy.facet.4c6]
// CHECK:STDOUT:   %.loc13_5.1: %Destroy.type = converted Core.IntLiteral, %Destroy.facet.loc13_5.1 [concrete = constants.%Destroy.facet.4c6]
// CHECK:STDOUT:   %Destroy.facet.loc13_5.2: %Destroy.type = facet_value Core.IntLiteral, (constants.%DestroyImpl.impl_witness.9b4, constants.%custom_witness.b80ef6.4) [concrete = constants.%Destroy.facet.4c6]
// CHECK:STDOUT:   %.loc13_5.2: %Destroy.type = converted Core.IntLiteral, %Destroy.facet.loc13_5.2 [concrete = constants.%Destroy.facet.4c6]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc13: init %empty_tuple.type = call %SelfDestruct.specific_fn.2(%z.var)
// CHECK:STDOUT:   %Destroy.facet.loc12_5.1: %Destroy.type = facet_value constants.%A, (constants.%DestroyImpl.impl_witness.6d1, constants.%custom_witness.b80ef6.3) [concrete = constants.%Destroy.facet.35e]
// CHECK:STDOUT:   %.loc12_5.2: %Destroy.type = converted constants.%A, %Destroy.facet.loc12_5.1 [concrete = constants.%Destroy.facet.35e]
// CHECK:STDOUT:   %Destroy.facet.loc12_5.2: %Destroy.type = facet_value constants.%A, (constants.%DestroyImpl.impl_witness.6d1, constants.%custom_witness.b80ef6.3) [concrete = constants.%Destroy.facet.35e]
// CHECK:STDOUT:   %.loc12_5.3: %Destroy.type = converted constants.%A, %Destroy.facet.loc12_5.2 [concrete = constants.%Destroy.facet.35e]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc12: init %empty_tuple.type = call %SelfDestruct.specific_fn.3(%y.var)
// CHECK:STDOUT:   %Destroy.facet.loc11_5.1: %Destroy.type = facet_value bool, (constants.%DestroyImpl.impl_witness.778, constants.%custom_witness.b80ef6.1) [concrete = constants.%Destroy.facet.e68]
// CHECK:STDOUT:   %.loc11_5.1: %Destroy.type = converted bool, %Destroy.facet.loc11_5.1 [concrete = constants.%Destroy.facet.e68]
// CHECK:STDOUT:   %Destroy.facet.loc11_5.2: %Destroy.type = facet_value bool, (constants.%DestroyImpl.impl_witness.778, constants.%custom_witness.b80ef6.1) [concrete = constants.%Destroy.facet.e68]
// CHECK:STDOUT:   %.loc11_5.2: %Destroy.type = converted bool, %Destroy.facet.loc11_5.2 [concrete = constants.%Destroy.facet.e68]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc11: init %empty_tuple.type = call %SelfDestruct.specific_fn.4(%x.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc16_5.1(%self.param: ref bool) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc16_5.2(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc16_5.3(%self.param: ref %A) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc16_5.4(%self.param: ref Core.IntLiteral) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc16_5.5(%self.param: ref %tuple.type.545) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [concrete = constants.%int_2]
// CHECK:STDOUT:   %tuple.elem2: ref Core.IntLiteral = tuple_access %self.param, element2
// CHECK:STDOUT:   %Destroy.facet.loc16_5.1: %Destroy.type = facet_value Core.IntLiteral, (constants.%DestroyImpl.impl_witness.9b4, constants.%custom_witness.b80ef6.4) [concrete = constants.%Destroy.facet.4c6]
// CHECK:STDOUT:   %.loc16_5.1: %Destroy.type = converted Core.IntLiteral, %Destroy.facet.loc16_5.1 [concrete = constants.%Destroy.facet.4c6]
// CHECK:STDOUT:   %Destroy.facet.loc16_5.2: %Destroy.type = facet_value Core.IntLiteral, (constants.%DestroyImpl.impl_witness.9b4, constants.%custom_witness.b80ef6.4) [concrete = constants.%Destroy.facet.4c6]
// CHECK:STDOUT:   %.loc16_5.2: %Destroy.type = converted Core.IntLiteral, %Destroy.facet.loc16_5.2 [concrete = constants.%Destroy.facet.4c6]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc16_5.1: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%tuple.elem2)
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1]
// CHECK:STDOUT:   %tuple.elem1: ref %A = tuple_access %self.param, element1
// CHECK:STDOUT:   %Destroy.facet.loc16_5.3: %Destroy.type = facet_value constants.%A, (constants.%DestroyImpl.impl_witness.6d1, constants.%custom_witness.b80ef6.3) [concrete = constants.%Destroy.facet.35e]
// CHECK:STDOUT:   %.loc16_5.3: %Destroy.type = converted constants.%A, %Destroy.facet.loc16_5.3 [concrete = constants.%Destroy.facet.35e]
// CHECK:STDOUT:   %Destroy.facet.loc16_5.4: %Destroy.type = facet_value constants.%A, (constants.%DestroyImpl.impl_witness.6d1, constants.%custom_witness.b80ef6.3) [concrete = constants.%Destroy.facet.35e]
// CHECK:STDOUT:   %.loc16_5.4: %Destroy.type = converted constants.%A, %Destroy.facet.loc16_5.4 [concrete = constants.%Destroy.facet.35e]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc16_5.2: init %empty_tuple.type = call %SelfDestruct.specific_fn.2(%tuple.elem1)
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT:   %tuple.elem0: ref bool = tuple_access %self.param, element0
// CHECK:STDOUT:   %Destroy.facet.loc16_5.5: %Destroy.type = facet_value bool, (constants.%DestroyImpl.impl_witness.778, constants.%custom_witness.b80ef6.1) [concrete = constants.%Destroy.facet.e68]
// CHECK:STDOUT:   %.loc16_5.5: %Destroy.type = converted bool, %Destroy.facet.loc16_5.5 [concrete = constants.%Destroy.facet.e68]
// CHECK:STDOUT:   %Destroy.facet.loc16_5.6: %Destroy.type = facet_value bool, (constants.%DestroyImpl.impl_witness.778, constants.%custom_witness.b80ef6.1) [concrete = constants.%Destroy.facet.e68]
// CHECK:STDOUT:   %.loc16_5.6: %Destroy.type = converted bool, %Destroy.facet.loc16_5.6 [concrete = constants.%Destroy.facet.e68]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc16_5.3: init %empty_tuple.type = call %SelfDestruct.specific_fn.3(%tuple.elem0)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- let.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %A: type = class_type @A [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.f6d: type = pattern_type auto [concrete]
// CHECK:STDOUT:   %x.patt: %pattern_type.f6d = value_binding_pattern x [concrete]
// CHECK:STDOUT:   %y.patt: %pattern_type.f6d = value_binding_pattern y [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %A.val: %A = struct_value () [concrete]
// CHECK:STDOUT:   %.367: ref %A = temporary invalid, %A.val [concrete]
// CHECK:STDOUT:   %z.patt: %pattern_type.f6d = value_binding_pattern z [concrete]
// CHECK:STDOUT:   %int_20: Core.IntLiteral = int_value 20 [concrete]
// CHECK:STDOUT:   %w.patt: %pattern_type.f6d = value_binding_pattern w [concrete]
// CHECK:STDOUT:   %tuple.type.545: type = tuple_type (bool, %A, Core.IntLiteral) [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %T.09f: %SubobjectDestroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.type.dea: type = fn_type @T.as_type.as.DestroyImpl.impl.Op, @T.as_type.as.DestroyImpl.impl(%T.09f) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.6e5: %T.as_type.as.DestroyImpl.impl.Op.type.dea = struct_value () [symbolic]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.2: type = fn_type @SubobjectDestroy.WithSelf.Op.loc8_22.2 [concrete]
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
// CHECK:STDOUT: fn @G() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:   %F.call: init bool = call %F.ref()
// CHECK:STDOUT:   %.loc7_21.1: bool = value_of_initializer %F.call
// CHECK:STDOUT:   %.loc7_21.2: bool = converted %F.call, %.loc7_21.1
// CHECK:STDOUT:   %x: bool = wrapper_binding x, %.loc7_21.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %pattern_type.f6d = value_binding_pattern x [concrete = constants.%x.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc8_20.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %A.ref: type = name_ref A, file.%A.decl [concrete = constants.%A]
// CHECK:STDOUT:   %.loc8_20.2: ref %A = temporary_storage
// CHECK:STDOUT:   %.loc8_20.3: init %A to %.loc8_20.2 = class_init () [concrete = constants.%A.val]
// CHECK:STDOUT:   %.loc8_22.1: init %A = converted %.loc8_20.1, %.loc8_20.3 [concrete = constants.%A.val]
// CHECK:STDOUT:   %.loc8_22.2: ref %A = temporary %.loc8_20.2, %.loc8_22.1 [concrete = constants.%.367]
// CHECK:STDOUT:   %.loc8_22.3: %A = acquire_value %.loc8_22.2 [concrete = constants.%A.val]
// CHECK:STDOUT:   %y: %A = wrapper_binding y, %.loc8_22.3
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %y.patt: %pattern_type.f6d = value_binding_pattern y [concrete = constants.%y.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int_20: Core.IntLiteral = int_value 20 [concrete = constants.%int_20]
// CHECK:STDOUT:   %z: Core.IntLiteral = wrapper_binding z, %int_20
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %z.patt: %pattern_type.f6d = value_binding_pattern z [concrete = constants.%z.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %x.ref: bool = name_ref x, %x
// CHECK:STDOUT:   %y.ref: %A = name_ref y, %y
// CHECK:STDOUT:   %z.ref: Core.IntLiteral = name_ref z, %z
// CHECK:STDOUT:   %.loc12_34.1: %tuple.type.545 = tuple_literal (%x.ref, %y.ref, %z.ref)
// CHECK:STDOUT:   %tuple: %tuple.type.545 = tuple_value (%x.ref, %y.ref, %z.ref)
// CHECK:STDOUT:   %.loc12_34.2: %tuple.type.545 = converted %.loc12_34.1, %tuple
// CHECK:STDOUT:   %w: %tuple.type.545 = wrapper_binding w, %.loc12_34.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %w.patt: %pattern_type.f6d = value_binding_pattern w [concrete = constants.%w.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.facet.loc8_22.1: %Destroy.type = facet_value constants.%A, (constants.%DestroyImpl.impl_witness.6d1, constants.%custom_witness.b80ef6.2) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc8_22.4: %Destroy.type = converted constants.%A, %Destroy.facet.loc8_22.1 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %Destroy.facet.loc8_22.2: %Destroy.type = facet_value constants.%A, (constants.%DestroyImpl.impl_witness.6d1, constants.%custom_witness.b80ef6.2) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc8_22.5: %Destroy.type = converted constants.%A, %Destroy.facet.loc8_22.2 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%.loc8_22.2)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc8_22.1(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc8_22.2(%self.param: ref %A) = "no_op";
// CHECK:STDOUT:
