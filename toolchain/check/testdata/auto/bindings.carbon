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
    // CHECK:STDERR:        ╭─┤ fail_todo_tuple.carbon:[[@LINE+7]]:20
    // CHECK:STDERR: ->     │     let unused x: (auto, auto) = (86, false);
    // CHECK:STDERR:        ·                    ━━┳━
    // CHECK:STDERR:        ·                      ┗━━┫ error: semantics TODO: `HandleAutoTypeLiteral`
    // CHECK:STDERR:        ·                         ┃        [SemanticsTodo]
    // CHECK:STDERR:        ╰────
    // CHECK:STDERR:
    let unused x: (auto, auto) = (86, false);
    var unused y: (auto, auto) = ({}, "str");
}

// --- tuple_pattern.carbon
library "[[@TEST_NAME]]";

fn F() -> ((), {}, bool) { return ((), {}, true); }
fn G() {
//@dump-sem-ir-begin
    let (unused a: auto, unused b: auto) = ({}, ());
    let (unused c: auto, unused d: auto, unused e: auto) = F();
    var (unused p: auto, unused q: auto) = (({}, ()), {});
    var (unused r: auto, unused s: auto, unused t: auto) = F();
//@dump-sem-ir-end
}

// --- nested_tuple_pattern.carbon
library "[[@TEST_NAME]]";

fn F() {
//@dump-sem-ir-begin
    let (unused a: auto, (unused b: auto, unused c: auto)) = ({}, ((), {}));
    var ((unused d: auto, unused e: auto), unused f: auto) = (({}, ()), ());
//@dump-sem-ir-end
}

// --- fail_tuple_pattern_mismatch.carbon
library "[[@TEST_NAME]]";

fn F() {
    // CHECK:STDERR:        ╭─┤ fail_tuple_pattern_mismatch.carbon:[[@LINE+7]]:5
    // CHECK:STDERR: ->     │     var (unused x: auto, unused y: auto) = ((), (), ());
    // CHECK:STDERR:        ·     ┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // CHECK:STDERR:        ·     ┗━━┫ error: cannot initialize tuple of 2 elements from tuple with 3 elements
    // CHECK:STDERR:        ·        ┃        [TupleInitElementCountMismatch]
    // CHECK:STDERR:        ╰────
    // CHECK:STDERR:
    var (unused x: auto, unused y: auto) = ((), (), ());
}

// --- fail_nested_tuple_pattern_mismatch.carbon
library "[[@TEST_NAME]]";

fn F() {
    // CHECK:STDERR:        ╭─┤ fail_nested_tuple_pattern_mismatch.carbon:[[@LINE+7]]:5
    // CHECK:STDERR: ->     │     var (unused x: auto, (unused y: auto, unused z: auto, unused w: auto))...
    // CHECK:STDERR:        ·     ┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // CHECK:STDERR:        ·     ┗━━┫ error: cannot initialize tuple of 3 elements from tuple with 2 elements
    // CHECK:STDERR:        ·        ┃        [TupleInitElementCountMismatch]
    // CHECK:STDERR:        ╰────
    // CHECK:STDERR:
    var (unused x: auto, (unused y: auto, unused z: auto, unused w: auto)) = ((), ({}, ()));
}

// --- fail_uninitialized_var.carbon
library "[[@TEST_NAME]]";

fn F() {
    // CHECK:STDERR:        ╭─┤ fail_uninitialized_var.carbon:[[@LINE+8]]:5
    // CHECK:STDERR: ->     │     var unused x: auto;
    // CHECK:STDERR:        ·     ┳━━━━━━━━━━━━━━━━━━
    // CHECK:STDERR:        ·     ┗━━┫ error: expected `=`; `var` declaration with type `auto` must have an
    // CHECK:STDERR:        ·        ┃        initializer
    // CHECK:STDERR:        ·        ┃        [ExpectedInitializerAfterAutoTypedVar]
    // CHECK:STDERR:        ╰────
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
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.1: type = fn_type @SubobjectDestroy.WithSelf.Op.loc14_5.1 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.1: %SubobjectDestroy.WithSelf.Op.type.6870fd.1 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.1: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.1), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.3: type = fn_type @SubobjectDestroy.WithSelf.Op.loc14_5.3 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.3: %SubobjectDestroy.WithSelf.Op.type.6870fd.3 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.3: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.3), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.4: type = fn_type @SubobjectDestroy.WithSelf.Op.loc14_5.4 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.4: %SubobjectDestroy.WithSelf.Op.type.6870fd.4 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.4: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.4), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.5: type = fn_type @SubobjectDestroy.WithSelf.Op.loc14_5.5 [concrete]
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
// CHECK:STDOUT:   %.loc14_34.1: %tuple.type.545 = tuple_literal (%x.ref, %y.ref, %z.ref)
// CHECK:STDOUT:   %.loc14_27: bool = acquire_value %x.ref
// CHECK:STDOUT:   %impl.elem0.loc14_27: %.68a = impl_witness_access constants.%Copy.impl_witness.5e1, element0 [concrete = constants.%bool.as.Copy.impl.Op]
// CHECK:STDOUT:   %bound_method.loc14_27: <bound method> = bound_method %.loc14_27, %impl.elem0.loc14_27
// CHECK:STDOUT:   %bool.as.Copy.impl.Op.call: init bool = call %bound_method.loc14_27(%.loc14_27)
// CHECK:STDOUT:   %tuple.elem0: ref bool = tuple_access %w.var, element0
// CHECK:STDOUT:   %.loc14_34.2: init bool to %tuple.elem0 = in_place_init %bool.as.Copy.impl.Op.call
// CHECK:STDOUT:   %.loc14_30: %A = acquire_value %y.ref
// CHECK:STDOUT:   %impl.elem0.loc14_30: %.e9e = impl_witness_access constants.%Copy.impl_witness.23e, element0 [concrete = constants.%A.as.Copy.impl.Op]
// CHECK:STDOUT:   %bound_method.loc14_30: <bound method> = bound_method %.loc14_30, %impl.elem0.loc14_30
// CHECK:STDOUT:   %tuple.elem1: ref %A = tuple_access %w.var, element1
// CHECK:STDOUT:   %A.as.Copy.impl.Op.call: init %A to %tuple.elem1 = call %bound_method.loc14_30(%.loc14_30)
// CHECK:STDOUT:   %.loc14_33: Core.IntLiteral = acquire_value %z.ref
// CHECK:STDOUT:   %impl.elem0.loc14_33: %.1b2 = impl_witness_access constants.%Copy.impl_witness.0b3, element0 [concrete = constants.%Core.IntLiteral.as.Copy.impl.Op]
// CHECK:STDOUT:   %bound_method.loc14_33: <bound method> = bound_method %.loc14_33, %impl.elem0.loc14_33
// CHECK:STDOUT:   %Core.IntLiteral.as.Copy.impl.Op.call.loc14: init Core.IntLiteral = call %bound_method.loc14_33(%.loc14_33)
// CHECK:STDOUT:   %tuple.elem2: ref Core.IntLiteral = tuple_access %w.var, element2
// CHECK:STDOUT:   %.loc14_34.3: init Core.IntLiteral to %tuple.elem2 = in_place_init %Core.IntLiteral.as.Copy.impl.Op.call.loc14
// CHECK:STDOUT:   %.loc14_34.4: init %tuple.type.545 to %w.var = tuple_init (%.loc14_34.2, %A.as.Copy.impl.Op.call, %.loc14_34.3)
// CHECK:STDOUT:   %.loc14_5.1: init %tuple.type.545 = converted %.loc14_34.1, %.loc14_34.4
// CHECK:STDOUT:   assign %w.var, %.loc14_5.1
// CHECK:STDOUT:   %w: ref %tuple.type.545 = wrapper_binding w, %w.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %w.patt: %pattern_type.f6d = ref_binding_pattern w [concrete = constants.%w.patt]
// CHECK:STDOUT:     %w.var_patt: %pattern_type.f6d = var_pattern %w.patt [concrete = constants.%w.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.facet.loc14_5.1: %Destroy.type = facet_value constants.%tuple.type.545, (constants.%DestroyImpl.impl_witness.7fe, constants.%custom_witness.b80ef6.5) [concrete = constants.%Destroy.facet.919]
// CHECK:STDOUT:   %.loc14_5.2: %Destroy.type = converted constants.%tuple.type.545, %Destroy.facet.loc14_5.1 [concrete = constants.%Destroy.facet.919]
// CHECK:STDOUT:   %Destroy.facet.loc14_5.2: %Destroy.type = facet_value constants.%tuple.type.545, (constants.%DestroyImpl.impl_witness.7fe, constants.%custom_witness.b80ef6.5) [concrete = constants.%Destroy.facet.919]
// CHECK:STDOUT:   %.loc14_5.3: %Destroy.type = converted constants.%tuple.type.545, %Destroy.facet.loc14_5.2 [concrete = constants.%Destroy.facet.919]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc14: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%w.var)
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
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc14_5.1(%self.param: ref bool) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc14_5.2(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc14_5.3(%self.param: ref %A) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc14_5.4(%self.param: ref Core.IntLiteral) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc14_5.5(%self.param: ref %tuple.type.545) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [concrete = constants.%int_2]
// CHECK:STDOUT:   %tuple.elem2: ref Core.IntLiteral = tuple_access %self.param, element2
// CHECK:STDOUT:   %Destroy.facet.loc14_5.1: %Destroy.type = facet_value Core.IntLiteral, (constants.%DestroyImpl.impl_witness.9b4, constants.%custom_witness.b80ef6.4) [concrete = constants.%Destroy.facet.4c6]
// CHECK:STDOUT:   %.loc14_5.1: %Destroy.type = converted Core.IntLiteral, %Destroy.facet.loc14_5.1 [concrete = constants.%Destroy.facet.4c6]
// CHECK:STDOUT:   %Destroy.facet.loc14_5.2: %Destroy.type = facet_value Core.IntLiteral, (constants.%DestroyImpl.impl_witness.9b4, constants.%custom_witness.b80ef6.4) [concrete = constants.%Destroy.facet.4c6]
// CHECK:STDOUT:   %.loc14_5.2: %Destroy.type = converted Core.IntLiteral, %Destroy.facet.loc14_5.2 [concrete = constants.%Destroy.facet.4c6]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc14_5.1: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%tuple.elem2)
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1]
// CHECK:STDOUT:   %tuple.elem1: ref %A = tuple_access %self.param, element1
// CHECK:STDOUT:   %Destroy.facet.loc14_5.3: %Destroy.type = facet_value constants.%A, (constants.%DestroyImpl.impl_witness.6d1, constants.%custom_witness.b80ef6.3) [concrete = constants.%Destroy.facet.35e]
// CHECK:STDOUT:   %.loc14_5.3: %Destroy.type = converted constants.%A, %Destroy.facet.loc14_5.3 [concrete = constants.%Destroy.facet.35e]
// CHECK:STDOUT:   %Destroy.facet.loc14_5.4: %Destroy.type = facet_value constants.%A, (constants.%DestroyImpl.impl_witness.6d1, constants.%custom_witness.b80ef6.3) [concrete = constants.%Destroy.facet.35e]
// CHECK:STDOUT:   %.loc14_5.4: %Destroy.type = converted constants.%A, %Destroy.facet.loc14_5.4 [concrete = constants.%Destroy.facet.35e]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc14_5.2: init %empty_tuple.type = call %SelfDestruct.specific_fn.2(%tuple.elem1)
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT:   %tuple.elem0: ref bool = tuple_access %self.param, element0
// CHECK:STDOUT:   %Destroy.facet.loc14_5.5: %Destroy.type = facet_value bool, (constants.%DestroyImpl.impl_witness.778, constants.%custom_witness.b80ef6.1) [concrete = constants.%Destroy.facet.e68]
// CHECK:STDOUT:   %.loc14_5.5: %Destroy.type = converted bool, %Destroy.facet.loc14_5.5 [concrete = constants.%Destroy.facet.e68]
// CHECK:STDOUT:   %Destroy.facet.loc14_5.6: %Destroy.type = facet_value bool, (constants.%DestroyImpl.impl_witness.778, constants.%custom_witness.b80ef6.1) [concrete = constants.%Destroy.facet.e68]
// CHECK:STDOUT:   %.loc14_5.6: %Destroy.type = converted bool, %Destroy.facet.loc14_5.6 [concrete = constants.%Destroy.facet.e68]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc14_5.3: init %empty_tuple.type = call %SelfDestruct.specific_fn.3(%tuple.elem0)
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
// CHECK:STDOUT:   %.loc10_34.1: %tuple.type.545 = tuple_literal (%x.ref, %y.ref, %z.ref)
// CHECK:STDOUT:   %tuple: %tuple.type.545 = tuple_value (%x.ref, %y.ref, %z.ref)
// CHECK:STDOUT:   %.loc10_34.2: %tuple.type.545 = converted %.loc10_34.1, %tuple
// CHECK:STDOUT:   %w: %tuple.type.545 = wrapper_binding w, %.loc10_34.2
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
// CHECK:STDOUT: --- tuple_pattern.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %tuple.type.944: type = tuple_type (%empty_tuple.type, %empty_struct_type, bool) [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.f6d: type = pattern_type auto [concrete]
// CHECK:STDOUT:   %a.patt: %pattern_type.f6d = value_binding_pattern a [concrete]
// CHECK:STDOUT:   %b.patt: %pattern_type.f6d = value_binding_pattern b [concrete]
// CHECK:STDOUT:   %tuple.type.dd4: type = tuple_type (auto, auto) [concrete]
// CHECK:STDOUT:   %pattern_type.439: type = pattern_type %tuple.type.dd4 [concrete]
// CHECK:STDOUT:   %.81a: %pattern_type.439 = tuple_pattern (%a.patt, %b.patt) [concrete]
// CHECK:STDOUT:   %tuple.type.0e6: type = tuple_type (%empty_struct_type, %empty_tuple.type) [concrete]
// CHECK:STDOUT:   %tuple.7a9: %tuple.type.0e6 = tuple_value (%empty_struct, %empty_tuple) [concrete]
// CHECK:STDOUT:   %c.patt: %pattern_type.f6d = value_binding_pattern c [concrete]
// CHECK:STDOUT:   %d.patt: %pattern_type.f6d = value_binding_pattern d [concrete]
// CHECK:STDOUT:   %e.patt: %pattern_type.f6d = value_binding_pattern e [concrete]
// CHECK:STDOUT:   %tuple.type.89a: type = tuple_type (auto, auto, auto) [concrete]
// CHECK:STDOUT:   %pattern_type.6a2: type = pattern_type %tuple.type.89a [concrete]
// CHECK:STDOUT:   %.041: %pattern_type.6a2 = tuple_pattern (%c.patt, %d.patt, %e.patt) [concrete]
// CHECK:STDOUT:   %p.patt: %pattern_type.f6d = ref_binding_pattern p [concrete]
// CHECK:STDOUT:   %q.patt: %pattern_type.f6d = ref_binding_pattern q [concrete]
// CHECK:STDOUT:   %.e7a: %pattern_type.439 = tuple_pattern (%p.patt, %q.patt) [concrete]
// CHECK:STDOUT:   %.var_patt.a31: %pattern_type.439 = var_pattern %.e7a [concrete]
// CHECK:STDOUT:   %tuple.type.e8e: type = tuple_type (%tuple.type.0e6, %empty_struct_type) [concrete]
// CHECK:STDOUT:   %tuple.11d: %tuple.type.e8e = tuple_value (%tuple.7a9, %empty_struct) [concrete]
// CHECK:STDOUT:   %r.patt: %pattern_type.f6d = ref_binding_pattern r [concrete]
// CHECK:STDOUT:   %s.patt: %pattern_type.f6d = ref_binding_pattern s [concrete]
// CHECK:STDOUT:   %t.patt.444: %pattern_type.f6d = ref_binding_pattern t [concrete]
// CHECK:STDOUT:   %.de7: %pattern_type.6a2 = tuple_pattern (%r.patt, %s.patt, %t.patt.444) [concrete]
// CHECK:STDOUT:   %.var_patt.cc5: %pattern_type.6a2 = var_pattern %.de7 [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %T.09f: %SubobjectDestroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.type.dea: type = fn_type @T.as_type.as.DestroyImpl.impl.Op, @T.as_type.as.DestroyImpl.impl(%T.09f) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.6e5: %T.as_type.as.DestroyImpl.impl.Op.type.dea = struct_value () [symbolic]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.1: type = fn_type @SubobjectDestroy.WithSelf.Op.loc9_5.1 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.1: %SubobjectDestroy.WithSelf.Op.type.6870fd.1 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.1: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.1), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.2: type = fn_type @SubobjectDestroy.WithSelf.Op.loc9_5.2 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.2: %SubobjectDestroy.WithSelf.Op.type.6870fd.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.2: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.2), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.3: type = fn_type @SubobjectDestroy.WithSelf.Op.loc9_5.3 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.3: %SubobjectDestroy.WithSelf.Op.type.6870fd.3 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.3: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.3), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.4: type = fn_type @SubobjectDestroy.WithSelf.Op.loc9_5.4 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.4: %SubobjectDestroy.WithSelf.Op.type.6870fd.4 = struct_value () [concrete]
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.a87: %SubobjectDestroy.type = facet_value bool, (%custom_witness.b80ef6.3) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.778: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.a87) [concrete]
// CHECK:STDOUT:   %Destroy.facet.e68: %Destroy.type = facet_value bool, (%DestroyImpl.impl_witness.778, %custom_witness.b80ef6.3) [concrete]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.6b7: %SubobjectDestroy.type = facet_value %empty_struct_type, (%custom_witness.b80ef6.2) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.011: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.6b7) [concrete]
// CHECK:STDOUT:   %Destroy.facet.fed: %Destroy.type = facet_value %empty_struct_type, (%DestroyImpl.impl_witness.011, %custom_witness.b80ef6.2) [concrete]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.f89: %SubobjectDestroy.type = facet_value %empty_tuple.type, (%custom_witness.b80ef6.1) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.674: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.f89) [concrete]
// CHECK:STDOUT:   %Destroy.facet.939: %Destroy.type = facet_value %empty_tuple.type, (%DestroyImpl.impl_witness.674, %custom_witness.b80ef6.1) [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.4: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.4), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.42d: %SubobjectDestroy.type = facet_value %tuple.type.944, (%custom_witness.b80ef6.4) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.bb8: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.42d) [concrete]
// CHECK:STDOUT:   %Destroy.facet.25c: %Destroy.type = facet_value %tuple.type.944, (%DestroyImpl.impl_witness.bb8, %custom_witness.b80ef6.4) [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.5: type = fn_type @SubobjectDestroy.WithSelf.Op.loc8_5.1 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.5: %SubobjectDestroy.WithSelf.Op.type.6870fd.5 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.5: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.5), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.6: type = fn_type @SubobjectDestroy.WithSelf.Op.loc8_5.2 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.6: %SubobjectDestroy.WithSelf.Op.type.6870fd.6 = struct_value () [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.44a: %SubobjectDestroy.type = facet_value %tuple.type.0e6, (%custom_witness.b80ef6.5) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.462: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.44a) [concrete]
// CHECK:STDOUT:   %Destroy.facet.b0b: %Destroy.type = facet_value %tuple.type.0e6, (%DestroyImpl.impl_witness.462, %custom_witness.b80ef6.5) [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.6: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.6), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.d99: %SubobjectDestroy.type = facet_value %tuple.type.e8e, (%custom_witness.b80ef6.6) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.8d0: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.d99) [concrete]
// CHECK:STDOUT:   %Destroy.facet.8ed: %Destroy.type = facet_value %tuple.type.e8e, (%DestroyImpl.impl_witness.8d0, %custom_witness.b80ef6.6) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.74d: @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op.type (%T.as_type.as.DestroyImpl.impl.Op.type.dea) = import_ref Core//prelude/parts/destroy, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op (constants.%T.as_type.as.DestroyImpl.impl.Op.6e5)]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table = impl_witness_table (%Core.import_ref.74d), @T.as_type.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc6_46.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc6_50.1: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc6_51: %tuple.type.0e6 = tuple_literal (%.loc6_46.1, %.loc6_50.1) [concrete = constants.%tuple.7a9]
// CHECK:STDOUT:   %empty_struct.loc6: %empty_struct_type = struct_value () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc6_46.2: %empty_struct_type = converted %.loc6_46.1, %empty_struct.loc6 [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %a: %empty_struct_type = wrapper_binding a, %.loc6_46.2
// CHECK:STDOUT:   %empty_tuple.loc6: %empty_tuple.type = tuple_value () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc6_50.2: %empty_tuple.type = converted %.loc6_50.1, %empty_tuple.loc6 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %b: %empty_tuple.type = wrapper_binding b, %.loc6_50.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.f6d = value_binding_pattern a [concrete = constants.%a.patt]
// CHECK:STDOUT:     %b.patt: %pattern_type.f6d = value_binding_pattern b [concrete = constants.%b.patt]
// CHECK:STDOUT:     %.loc6_40: %pattern_type.439 = tuple_pattern (%a.patt, %b.patt) [concrete = constants.%.81a]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.ref.loc7: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:   %.loc7_62.1: ref %tuple.type.944 = temporary_storage
// CHECK:STDOUT:   %F.call.loc7: init %tuple.type.944 to %.loc7_62.1 = call %F.ref.loc7()
// CHECK:STDOUT:   %.loc7_62.2: ref %tuple.type.944 = temporary %.loc7_62.1, %F.call.loc7
// CHECK:STDOUT:   %tuple.elem0.loc7: ref %empty_tuple.type = tuple_access %.loc7_62.2, element0
// CHECK:STDOUT:   %tuple.elem1.loc7: ref %empty_struct_type = tuple_access %.loc7_62.2, element1
// CHECK:STDOUT:   %tuple.elem2.loc7: ref bool = tuple_access %.loc7_62.2, element2
// CHECK:STDOUT:   %empty_tuple.loc7: %empty_tuple.type = tuple_value () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc7_62.3: %empty_tuple.type = converted %tuple.elem0.loc7, %empty_tuple.loc7 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %c: %empty_tuple.type = wrapper_binding c, %.loc7_62.3
// CHECK:STDOUT:   %empty_struct.loc7: %empty_struct_type = struct_value () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc7_62.4: %empty_struct_type = converted %tuple.elem1.loc7, %empty_struct.loc7 [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %d: %empty_struct_type = wrapper_binding d, %.loc7_62.4
// CHECK:STDOUT:   %.loc7_62.5: bool = acquire_value %tuple.elem2.loc7
// CHECK:STDOUT:   %e: bool = wrapper_binding e, %.loc7_62.5
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c.patt: %pattern_type.f6d = value_binding_pattern c [concrete = constants.%c.patt]
// CHECK:STDOUT:     %d.patt: %pattern_type.f6d = value_binding_pattern d [concrete = constants.%d.patt]
// CHECK:STDOUT:     %e.patt: %pattern_type.f6d = value_binding_pattern e [concrete = constants.%e.patt]
// CHECK:STDOUT:     %.loc7_56: %pattern_type.6a2 = tuple_pattern (%c.patt, %d.patt, %e.patt) [concrete = constants.%.041]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.var.loc8: ref %tuple.type.e8e = var_storage %.var_patt.loc8
// CHECK:STDOUT:   %.loc8_47.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc8_51.1: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc8_52.1: %tuple.type.0e6 = tuple_literal (%.loc8_47.1, %.loc8_51.1) [concrete = constants.%tuple.7a9]
// CHECK:STDOUT:   %.loc8_56.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc8_57.1: %tuple.type.e8e = tuple_literal (%.loc8_52.1, %.loc8_56.1) [concrete = constants.%tuple.11d]
// CHECK:STDOUT:   %tuple.elem0.loc8_57: ref %tuple.type.0e6 = tuple_access %.var.loc8, element0
// CHECK:STDOUT:   %tuple.elem0.loc8_52: ref %empty_struct_type = tuple_access %tuple.elem0.loc8_57, element0
// CHECK:STDOUT:   %.loc8_47.2: init %empty_struct_type to %tuple.elem0.loc8_52 = struct_init () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc8_52.2: init %empty_struct_type = converted %.loc8_47.1, %.loc8_47.2 [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc8_52.3: init %empty_struct_type to %tuple.elem0.loc8_52 = in_place_init %.loc8_52.2 [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %tuple.elem1.loc8_52: ref %empty_tuple.type = tuple_access %tuple.elem0.loc8_57, element1
// CHECK:STDOUT:   %.loc8_51.2: init %empty_tuple.type to %tuple.elem1.loc8_52 = tuple_init () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc8_52.4: init %empty_tuple.type = converted %.loc8_51.1, %.loc8_51.2 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc8_52.5: init %empty_tuple.type to %tuple.elem1.loc8_52 = in_place_init %.loc8_52.4 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc8_52.6: init %tuple.type.0e6 to %tuple.elem0.loc8_57 = tuple_init (%.loc8_52.3, %.loc8_52.5) [concrete = constants.%tuple.7a9]
// CHECK:STDOUT:   %.loc8_57.2: init %tuple.type.0e6 = converted %.loc8_52.1, %.loc8_52.6 [concrete = constants.%tuple.7a9]
// CHECK:STDOUT:   %tuple.elem1.loc8_57: ref %empty_struct_type = tuple_access %.var.loc8, element1
// CHECK:STDOUT:   %.loc8_56.2: init %empty_struct_type to %tuple.elem1.loc8_57 = struct_init () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc8_57.3: init %empty_struct_type = converted %.loc8_56.1, %.loc8_56.2 [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc8_57.4: init %empty_struct_type to %tuple.elem1.loc8_57 = in_place_init %.loc8_57.3 [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc8_57.5: init %tuple.type.e8e to %.var.loc8 = tuple_init (%.loc8_57.2, %.loc8_57.4) [concrete = constants.%tuple.11d]
// CHECK:STDOUT:   %.loc8_5.1: init %tuple.type.e8e = converted %.loc8_57.1, %.loc8_57.5 [concrete = constants.%tuple.11d]
// CHECK:STDOUT:   assign %.var.loc8, %.loc8_5.1
// CHECK:STDOUT:   %tuple.elem0.loc8_5: ref %tuple.type.0e6 = tuple_access %.var.loc8, element0
// CHECK:STDOUT:   %tuple.elem1.loc8_5: ref %empty_struct_type = tuple_access %.var.loc8, element1
// CHECK:STDOUT:   %p: ref %tuple.type.0e6 = wrapper_binding p, %tuple.elem0.loc8_5
// CHECK:STDOUT:   %q: ref %empty_struct_type = wrapper_binding q, %tuple.elem1.loc8_5
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %p.patt: %pattern_type.f6d = ref_binding_pattern p [concrete = constants.%p.patt]
// CHECK:STDOUT:     %q.patt: %pattern_type.f6d = ref_binding_pattern q [concrete = constants.%q.patt]
// CHECK:STDOUT:     %.loc8_40: %pattern_type.439 = tuple_pattern (%p.patt, %q.patt) [concrete = constants.%.e7a]
// CHECK:STDOUT:     %.var_patt.loc8: %pattern_type.439 = var_pattern %.loc8_40 [concrete = constants.%.var_patt.a31]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.var.loc9: ref %tuple.type.944 = var_storage %.var_patt.loc9
// CHECK:STDOUT:   %F.ref.loc9: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:   %.loc9_5.1: ref %tuple.type.944 = splice_block %.var.loc9 {}
// CHECK:STDOUT:   %F.call.loc9: init %tuple.type.944 to %.loc9_5.1 = call %F.ref.loc9()
// CHECK:STDOUT:   assign %.var.loc9, %F.call.loc9
// CHECK:STDOUT:   %tuple.elem0.loc9: ref %empty_tuple.type = tuple_access %.var.loc9, element0
// CHECK:STDOUT:   %tuple.elem1.loc9: ref %empty_struct_type = tuple_access %.var.loc9, element1
// CHECK:STDOUT:   %tuple.elem2.loc9: ref bool = tuple_access %.var.loc9, element2
// CHECK:STDOUT:   %r: ref %empty_tuple.type = wrapper_binding r, %tuple.elem0.loc9
// CHECK:STDOUT:   %s: ref %empty_struct_type = wrapper_binding s, %tuple.elem1.loc9
// CHECK:STDOUT:   %t: ref bool = wrapper_binding t, %tuple.elem2.loc9
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %r.patt: %pattern_type.f6d = ref_binding_pattern r [concrete = constants.%r.patt]
// CHECK:STDOUT:     %s.patt: %pattern_type.f6d = ref_binding_pattern s [concrete = constants.%s.patt]
// CHECK:STDOUT:     %t.patt: %pattern_type.f6d = ref_binding_pattern t [concrete = constants.%t.patt.444]
// CHECK:STDOUT:     %.loc9_56: %pattern_type.6a2 = tuple_pattern (%r.patt, %s.patt, %t.patt) [concrete = constants.%.de7]
// CHECK:STDOUT:     %.var_patt.loc9: %pattern_type.6a2 = var_pattern %.loc9_56 [concrete = constants.%.var_patt.cc5]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.facet.loc9_5.1: %Destroy.type = facet_value constants.%tuple.type.944, (constants.%DestroyImpl.impl_witness.bb8, constants.%custom_witness.b80ef6.4) [concrete = constants.%Destroy.facet.25c]
// CHECK:STDOUT:   %.loc9_5.2: %Destroy.type = converted constants.%tuple.type.944, %Destroy.facet.loc9_5.1 [concrete = constants.%Destroy.facet.25c]
// CHECK:STDOUT:   %Destroy.facet.loc9_5.2: %Destroy.type = facet_value constants.%tuple.type.944, (constants.%DestroyImpl.impl_witness.bb8, constants.%custom_witness.b80ef6.4) [concrete = constants.%Destroy.facet.25c]
// CHECK:STDOUT:   %.loc9_5.3: %Destroy.type = converted constants.%tuple.type.944, %Destroy.facet.loc9_5.2 [concrete = constants.%Destroy.facet.25c]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc9: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%.var.loc9)
// CHECK:STDOUT:   %Destroy.facet.loc8_5.1: %Destroy.type = facet_value constants.%tuple.type.e8e, (constants.%DestroyImpl.impl_witness.8d0, constants.%custom_witness.b80ef6.6) [concrete = constants.%Destroy.facet.8ed]
// CHECK:STDOUT:   %.loc8_5.2: %Destroy.type = converted constants.%tuple.type.e8e, %Destroy.facet.loc8_5.1 [concrete = constants.%Destroy.facet.8ed]
// CHECK:STDOUT:   %Destroy.facet.loc8_5.2: %Destroy.type = facet_value constants.%tuple.type.e8e, (constants.%DestroyImpl.impl_witness.8d0, constants.%custom_witness.b80ef6.6) [concrete = constants.%Destroy.facet.8ed]
// CHECK:STDOUT:   %.loc8_5.3: %Destroy.type = converted constants.%tuple.type.e8e, %Destroy.facet.loc8_5.2 [concrete = constants.%Destroy.facet.8ed]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc8: init %empty_tuple.type = call %SelfDestruct.specific_fn.2(%.var.loc8)
// CHECK:STDOUT:   %Destroy.facet.loc7_62.1: %Destroy.type = facet_value constants.%tuple.type.944, (constants.%DestroyImpl.impl_witness.bb8, constants.%custom_witness.b80ef6.4) [concrete = constants.%Destroy.facet.25c]
// CHECK:STDOUT:   %.loc7_62.6: %Destroy.type = converted constants.%tuple.type.944, %Destroy.facet.loc7_62.1 [concrete = constants.%Destroy.facet.25c]
// CHECK:STDOUT:   %Destroy.facet.loc7_62.2: %Destroy.type = facet_value constants.%tuple.type.944, (constants.%DestroyImpl.impl_witness.bb8, constants.%custom_witness.b80ef6.4) [concrete = constants.%Destroy.facet.25c]
// CHECK:STDOUT:   %.loc7_62.7: %Destroy.type = converted constants.%tuple.type.944, %Destroy.facet.loc7_62.2 [concrete = constants.%Destroy.facet.25c]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc7: init %empty_tuple.type = call %SelfDestruct.specific_fn.3(%.loc7_62.2)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc9_5.1(%self.param: ref %empty_tuple.type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc9_5.2(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc9_5.3(%self.param: ref bool) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc9_5.4(%self.param: ref %tuple.type.944) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [concrete = constants.%int_2]
// CHECK:STDOUT:   %tuple.elem2: ref bool = tuple_access %self.param, element2
// CHECK:STDOUT:   %Destroy.facet.loc9_5.1: %Destroy.type = facet_value bool, (constants.%DestroyImpl.impl_witness.778, constants.%custom_witness.b80ef6.3) [concrete = constants.%Destroy.facet.e68]
// CHECK:STDOUT:   %.loc9_5.1: %Destroy.type = converted bool, %Destroy.facet.loc9_5.1 [concrete = constants.%Destroy.facet.e68]
// CHECK:STDOUT:   %Destroy.facet.loc9_5.2: %Destroy.type = facet_value bool, (constants.%DestroyImpl.impl_witness.778, constants.%custom_witness.b80ef6.3) [concrete = constants.%Destroy.facet.e68]
// CHECK:STDOUT:   %.loc9_5.2: %Destroy.type = converted bool, %Destroy.facet.loc9_5.2 [concrete = constants.%Destroy.facet.e68]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc9_5.1: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%tuple.elem2)
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1]
// CHECK:STDOUT:   %tuple.elem1: ref %empty_struct_type = tuple_access %self.param, element1
// CHECK:STDOUT:   %Destroy.facet.loc9_5.3: %Destroy.type = facet_value constants.%empty_struct_type, (constants.%DestroyImpl.impl_witness.011, constants.%custom_witness.b80ef6.2) [concrete = constants.%Destroy.facet.fed]
// CHECK:STDOUT:   %.loc9_5.3: %Destroy.type = converted constants.%empty_struct_type, %Destroy.facet.loc9_5.3 [concrete = constants.%Destroy.facet.fed]
// CHECK:STDOUT:   %Destroy.facet.loc9_5.4: %Destroy.type = facet_value constants.%empty_struct_type, (constants.%DestroyImpl.impl_witness.011, constants.%custom_witness.b80ef6.2) [concrete = constants.%Destroy.facet.fed]
// CHECK:STDOUT:   %.loc9_5.4: %Destroy.type = converted constants.%empty_struct_type, %Destroy.facet.loc9_5.4 [concrete = constants.%Destroy.facet.fed]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc9_5.2: init %empty_tuple.type = call %SelfDestruct.specific_fn.2(%tuple.elem1)
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT:   %tuple.elem0: ref %empty_tuple.type = tuple_access %self.param, element0
// CHECK:STDOUT:   %Destroy.facet.loc9_5.5: %Destroy.type = facet_value constants.%empty_tuple.type, (constants.%DestroyImpl.impl_witness.674, constants.%custom_witness.b80ef6.1) [concrete = constants.%Destroy.facet.939]
// CHECK:STDOUT:   %.loc9_5.5: %Destroy.type = converted constants.%empty_tuple.type, %Destroy.facet.loc9_5.5 [concrete = constants.%Destroy.facet.939]
// CHECK:STDOUT:   %Destroy.facet.loc9_5.6: %Destroy.type = facet_value constants.%empty_tuple.type, (constants.%DestroyImpl.impl_witness.674, constants.%custom_witness.b80ef6.1) [concrete = constants.%Destroy.facet.939]
// CHECK:STDOUT:   %.loc9_5.6: %Destroy.type = converted constants.%empty_tuple.type, %Destroy.facet.loc9_5.6 [concrete = constants.%Destroy.facet.939]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc9_5.3: init %empty_tuple.type = call %SelfDestruct.specific_fn.3(%tuple.elem0)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc8_5.1(%self.param: ref %tuple.type.0e6) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1]
// CHECK:STDOUT:   %tuple.elem1: ref %empty_tuple.type = tuple_access %self.param, element1
// CHECK:STDOUT:   %Destroy.facet.loc8_5.1: %Destroy.type = facet_value constants.%empty_tuple.type, (constants.%DestroyImpl.impl_witness.674, constants.%custom_witness.b80ef6.1) [concrete = constants.%Destroy.facet.939]
// CHECK:STDOUT:   %.loc8_5.1: %Destroy.type = converted constants.%empty_tuple.type, %Destroy.facet.loc8_5.1 [concrete = constants.%Destroy.facet.939]
// CHECK:STDOUT:   %Destroy.facet.loc8_5.2: %Destroy.type = facet_value constants.%empty_tuple.type, (constants.%DestroyImpl.impl_witness.674, constants.%custom_witness.b80ef6.1) [concrete = constants.%Destroy.facet.939]
// CHECK:STDOUT:   %.loc8_5.2: %Destroy.type = converted constants.%empty_tuple.type, %Destroy.facet.loc8_5.2 [concrete = constants.%Destroy.facet.939]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc8_5.1: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%tuple.elem1)
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT:   %tuple.elem0: ref %empty_struct_type = tuple_access %self.param, element0
// CHECK:STDOUT:   %Destroy.facet.loc8_5.3: %Destroy.type = facet_value constants.%empty_struct_type, (constants.%DestroyImpl.impl_witness.011, constants.%custom_witness.b80ef6.2) [concrete = constants.%Destroy.facet.fed]
// CHECK:STDOUT:   %.loc8_5.3: %Destroy.type = converted constants.%empty_struct_type, %Destroy.facet.loc8_5.3 [concrete = constants.%Destroy.facet.fed]
// CHECK:STDOUT:   %Destroy.facet.loc8_5.4: %Destroy.type = facet_value constants.%empty_struct_type, (constants.%DestroyImpl.impl_witness.011, constants.%custom_witness.b80ef6.2) [concrete = constants.%Destroy.facet.fed]
// CHECK:STDOUT:   %.loc8_5.4: %Destroy.type = converted constants.%empty_struct_type, %Destroy.facet.loc8_5.4 [concrete = constants.%Destroy.facet.fed]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc8_5.2: init %empty_tuple.type = call %SelfDestruct.specific_fn.2(%tuple.elem0)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc8_5.2(%self.param: ref %tuple.type.e8e) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1]
// CHECK:STDOUT:   %tuple.elem1: ref %empty_struct_type = tuple_access %self.param, element1
// CHECK:STDOUT:   %Destroy.facet.loc8_5.1: %Destroy.type = facet_value constants.%empty_struct_type, (constants.%DestroyImpl.impl_witness.011, constants.%custom_witness.b80ef6.2) [concrete = constants.%Destroy.facet.fed]
// CHECK:STDOUT:   %.loc8_5.1: %Destroy.type = converted constants.%empty_struct_type, %Destroy.facet.loc8_5.1 [concrete = constants.%Destroy.facet.fed]
// CHECK:STDOUT:   %Destroy.facet.loc8_5.2: %Destroy.type = facet_value constants.%empty_struct_type, (constants.%DestroyImpl.impl_witness.011, constants.%custom_witness.b80ef6.2) [concrete = constants.%Destroy.facet.fed]
// CHECK:STDOUT:   %.loc8_5.2: %Destroy.type = converted constants.%empty_struct_type, %Destroy.facet.loc8_5.2 [concrete = constants.%Destroy.facet.fed]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc8_5.1: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%tuple.elem1)
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT:   %tuple.elem0: ref %tuple.type.0e6 = tuple_access %self.param, element0
// CHECK:STDOUT:   %Destroy.facet.loc8_5.3: %Destroy.type = facet_value constants.%tuple.type.0e6, (constants.%DestroyImpl.impl_witness.462, constants.%custom_witness.b80ef6.5) [concrete = constants.%Destroy.facet.b0b]
// CHECK:STDOUT:   %.loc8_5.3: %Destroy.type = converted constants.%tuple.type.0e6, %Destroy.facet.loc8_5.3 [concrete = constants.%Destroy.facet.b0b]
// CHECK:STDOUT:   %Destroy.facet.loc8_5.4: %Destroy.type = facet_value constants.%tuple.type.0e6, (constants.%DestroyImpl.impl_witness.462, constants.%custom_witness.b80ef6.5) [concrete = constants.%Destroy.facet.b0b]
// CHECK:STDOUT:   %.loc8_5.4: %Destroy.type = converted constants.%tuple.type.0e6, %Destroy.facet.loc8_5.4 [concrete = constants.%Destroy.facet.b0b]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc8_5.2: init %empty_tuple.type = call %SelfDestruct.specific_fn.2(%tuple.elem0)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- nested_tuple_pattern.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %pattern_type.f6d: type = pattern_type auto [concrete]
// CHECK:STDOUT:   %a.patt: %pattern_type.f6d = value_binding_pattern a [concrete]
// CHECK:STDOUT:   %b.patt: %pattern_type.f6d = value_binding_pattern b [concrete]
// CHECK:STDOUT:   %c.patt: %pattern_type.f6d = value_binding_pattern c [concrete]
// CHECK:STDOUT:   %tuple.type.dd4: type = tuple_type (auto, auto) [concrete]
// CHECK:STDOUT:   %pattern_type.439: type = pattern_type %tuple.type.dd4 [concrete]
// CHECK:STDOUT:   %.4da: %pattern_type.439 = tuple_pattern (%b.patt, %c.patt) [concrete]
// CHECK:STDOUT:   %tuple.type.7e6: type = tuple_type (auto, %tuple.type.dd4) [concrete]
// CHECK:STDOUT:   %pattern_type.220: type = pattern_type %tuple.type.7e6 [concrete]
// CHECK:STDOUT:   %.582: %pattern_type.220 = tuple_pattern (%a.patt, %.4da) [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %tuple.type.980: type = tuple_type (%empty_tuple.type, %empty_struct_type) [concrete]
// CHECK:STDOUT:   %tuple.a1e: %tuple.type.980 = tuple_value (%empty_tuple, %empty_struct) [concrete]
// CHECK:STDOUT:   %tuple.type.ab5: type = tuple_type (%empty_struct_type, %tuple.type.980) [concrete]
// CHECK:STDOUT:   %tuple.a8b: %tuple.type.ab5 = tuple_value (%empty_struct, %tuple.a1e) [concrete]
// CHECK:STDOUT:   %d.patt: %pattern_type.f6d = ref_binding_pattern d [concrete]
// CHECK:STDOUT:   %e.patt: %pattern_type.f6d = ref_binding_pattern e [concrete]
// CHECK:STDOUT:   %.191: %pattern_type.439 = tuple_pattern (%d.patt, %e.patt) [concrete]
// CHECK:STDOUT:   %f.patt: %pattern_type.f6d = ref_binding_pattern f [concrete]
// CHECK:STDOUT:   %tuple.type.852: type = tuple_type (%tuple.type.dd4, auto) [concrete]
// CHECK:STDOUT:   %pattern_type.ad3: type = pattern_type %tuple.type.852 [concrete]
// CHECK:STDOUT:   %.56a: %pattern_type.ad3 = tuple_pattern (%.191, %f.patt) [concrete]
// CHECK:STDOUT:   %.var_patt: %pattern_type.ad3 = var_pattern %.56a [concrete]
// CHECK:STDOUT:   %tuple.type.0e6: type = tuple_type (%empty_struct_type, %empty_tuple.type) [concrete]
// CHECK:STDOUT:   %tuple.7a9: %tuple.type.0e6 = tuple_value (%empty_struct, %empty_tuple) [concrete]
// CHECK:STDOUT:   %tuple.type.102: type = tuple_type (%tuple.type.0e6, %empty_tuple.type) [concrete]
// CHECK:STDOUT:   %tuple.746: %tuple.type.102 = tuple_value (%tuple.7a9, %empty_tuple) [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %T.09f: %SubobjectDestroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.type.dea: type = fn_type @T.as_type.as.DestroyImpl.impl.Op, @T.as_type.as.DestroyImpl.impl(%T.09f) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.6e5: %T.as_type.as.DestroyImpl.impl.Op.type.dea = struct_value () [symbolic]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.1: type = fn_type @SubobjectDestroy.WithSelf.Op.loc6_5.1 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.1: %SubobjectDestroy.WithSelf.Op.type.6870fd.1 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.1: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.1), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.2: type = fn_type @SubobjectDestroy.WithSelf.Op.loc6_5.2 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.2: %SubobjectDestroy.WithSelf.Op.type.6870fd.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.2: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.2), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.3: type = fn_type @SubobjectDestroy.WithSelf.Op.loc6_5.3 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.3: %SubobjectDestroy.WithSelf.Op.type.6870fd.3 = struct_value () [concrete]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.f89: %SubobjectDestroy.type = facet_value %empty_tuple.type, (%custom_witness.b80ef6.2) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.674: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.f89) [concrete]
// CHECK:STDOUT:   %Destroy.facet.939: %Destroy.type = facet_value %empty_tuple.type, (%DestroyImpl.impl_witness.674, %custom_witness.b80ef6.2) [concrete]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.6b7: %SubobjectDestroy.type = facet_value %empty_struct_type, (%custom_witness.b80ef6.1) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.011: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.6b7) [concrete]
// CHECK:STDOUT:   %Destroy.facet.fed: %Destroy.type = facet_value %empty_struct_type, (%DestroyImpl.impl_witness.011, %custom_witness.b80ef6.1) [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.3: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.3), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.4: type = fn_type @SubobjectDestroy.WithSelf.Op.loc6_5.4 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.4: %SubobjectDestroy.WithSelf.Op.type.6870fd.4 = struct_value () [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.44a: %SubobjectDestroy.type = facet_value %tuple.type.0e6, (%custom_witness.b80ef6.3) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.462: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.44a) [concrete]
// CHECK:STDOUT:   %Destroy.facet.b0b: %Destroy.type = facet_value %tuple.type.0e6, (%DestroyImpl.impl_witness.462, %custom_witness.b80ef6.3) [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.4: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.4), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.18a: %SubobjectDestroy.type = facet_value %tuple.type.102, (%custom_witness.b80ef6.4) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.a82: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.18a) [concrete]
// CHECK:STDOUT:   %Destroy.facet.aeb: %Destroy.type = facet_value %tuple.type.102, (%DestroyImpl.impl_witness.a82, %custom_witness.b80ef6.4) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.74d: @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op.type (%T.as_type.as.DestroyImpl.impl.Op.type.dea) = import_ref Core//prelude/parts/destroy, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op (constants.%T.as_type.as.DestroyImpl.impl.Op.6e5)]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table = impl_witness_table (%Core.import_ref.74d), @T.as_type.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc5_64.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc5_69.1: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc5_73.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc5_74: %tuple.type.980 = tuple_literal (%.loc5_69.1, %.loc5_73.1) [concrete = constants.%tuple.a1e]
// CHECK:STDOUT:   %.loc5_75: %tuple.type.ab5 = tuple_literal (%.loc5_64.1, %.loc5_74) [concrete = constants.%tuple.a8b]
// CHECK:STDOUT:   %empty_struct.loc5_64: %empty_struct_type = struct_value () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc5_64.2: %empty_struct_type = converted %.loc5_64.1, %empty_struct.loc5_64 [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %a: %empty_struct_type = wrapper_binding a, %.loc5_64.2
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc5_69.2: %empty_tuple.type = converted %.loc5_69.1, %empty_tuple [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %b: %empty_tuple.type = wrapper_binding b, %.loc5_69.2
// CHECK:STDOUT:   %empty_struct.loc5_73: %empty_struct_type = struct_value () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc5_73.2: %empty_struct_type = converted %.loc5_73.1, %empty_struct.loc5_73 [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %c: %empty_struct_type = wrapper_binding c, %.loc5_73.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.f6d = value_binding_pattern a [concrete = constants.%a.patt]
// CHECK:STDOUT:     %b.patt: %pattern_type.f6d = value_binding_pattern b [concrete = constants.%b.patt]
// CHECK:STDOUT:     %c.patt: %pattern_type.f6d = value_binding_pattern c [concrete = constants.%c.patt]
// CHECK:STDOUT:     %.loc5_57: %pattern_type.439 = tuple_pattern (%b.patt, %c.patt) [concrete = constants.%.4da]
// CHECK:STDOUT:     %.loc5_58: %pattern_type.220 = tuple_pattern (%a.patt, %.loc5_57) [concrete = constants.%.582]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.var: ref %tuple.type.102 = var_storage %.var_patt
// CHECK:STDOUT:   %.loc6_65.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc6_69.1: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc6_70.1: %tuple.type.0e6 = tuple_literal (%.loc6_65.1, %.loc6_69.1) [concrete = constants.%tuple.7a9]
// CHECK:STDOUT:   %.loc6_74.1: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc6_75.1: %tuple.type.102 = tuple_literal (%.loc6_70.1, %.loc6_74.1) [concrete = constants.%tuple.746]
// CHECK:STDOUT:   %tuple.elem0.loc6_75: ref %tuple.type.0e6 = tuple_access %.var, element0
// CHECK:STDOUT:   %tuple.elem0.loc6_70: ref %empty_struct_type = tuple_access %tuple.elem0.loc6_75, element0
// CHECK:STDOUT:   %.loc6_65.2: init %empty_struct_type to %tuple.elem0.loc6_70 = struct_init () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc6_70.2: init %empty_struct_type = converted %.loc6_65.1, %.loc6_65.2 [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc6_70.3: init %empty_struct_type to %tuple.elem0.loc6_70 = in_place_init %.loc6_70.2 [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %tuple.elem1.loc6_70: ref %empty_tuple.type = tuple_access %tuple.elem0.loc6_75, element1
// CHECK:STDOUT:   %.loc6_69.2: init %empty_tuple.type to %tuple.elem1.loc6_70 = tuple_init () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc6_70.4: init %empty_tuple.type = converted %.loc6_69.1, %.loc6_69.2 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc6_70.5: init %empty_tuple.type to %tuple.elem1.loc6_70 = in_place_init %.loc6_70.4 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc6_70.6: init %tuple.type.0e6 to %tuple.elem0.loc6_75 = tuple_init (%.loc6_70.3, %.loc6_70.5) [concrete = constants.%tuple.7a9]
// CHECK:STDOUT:   %.loc6_75.2: init %tuple.type.0e6 = converted %.loc6_70.1, %.loc6_70.6 [concrete = constants.%tuple.7a9]
// CHECK:STDOUT:   %tuple.elem1.loc6_75: ref %empty_tuple.type = tuple_access %.var, element1
// CHECK:STDOUT:   %.loc6_74.2: init %empty_tuple.type to %tuple.elem1.loc6_75 = tuple_init () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc6_75.3: init %empty_tuple.type = converted %.loc6_74.1, %.loc6_74.2 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc6_75.4: init %empty_tuple.type to %tuple.elem1.loc6_75 = in_place_init %.loc6_75.3 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc6_75.5: init %tuple.type.102 to %.var = tuple_init (%.loc6_75.2, %.loc6_75.4) [concrete = constants.%tuple.746]
// CHECK:STDOUT:   %.loc6_5.1: init %tuple.type.102 = converted %.loc6_75.1, %.loc6_75.5 [concrete = constants.%tuple.746]
// CHECK:STDOUT:   assign %.var, %.loc6_5.1
// CHECK:STDOUT:   %tuple.elem0.loc6_5.1: ref %tuple.type.0e6 = tuple_access %.var, element0
// CHECK:STDOUT:   %tuple.elem1.loc6_5.1: ref %empty_tuple.type = tuple_access %.var, element1
// CHECK:STDOUT:   %tuple.elem0.loc6_5.2: ref %empty_struct_type = tuple_access %tuple.elem0.loc6_5.1, element0
// CHECK:STDOUT:   %tuple.elem1.loc6_5.2: ref %empty_tuple.type = tuple_access %tuple.elem0.loc6_5.1, element1
// CHECK:STDOUT:   %d: ref %empty_struct_type = wrapper_binding d, %tuple.elem0.loc6_5.2
// CHECK:STDOUT:   %e: ref %empty_tuple.type = wrapper_binding e, %tuple.elem1.loc6_5.2
// CHECK:STDOUT:   %f: ref %empty_tuple.type = wrapper_binding f, %tuple.elem1.loc6_5.1
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %d.patt: %pattern_type.f6d = ref_binding_pattern d [concrete = constants.%d.patt]
// CHECK:STDOUT:     %e.patt: %pattern_type.f6d = ref_binding_pattern e [concrete = constants.%e.patt]
// CHECK:STDOUT:     %.loc6_41: %pattern_type.439 = tuple_pattern (%d.patt, %e.patt) [concrete = constants.%.191]
// CHECK:STDOUT:     %f.patt: %pattern_type.f6d = ref_binding_pattern f [concrete = constants.%f.patt]
// CHECK:STDOUT:     %.loc6_58: %pattern_type.ad3 = tuple_pattern (%.loc6_41, %f.patt) [concrete = constants.%.56a]
// CHECK:STDOUT:     %.var_patt: %pattern_type.ad3 = var_pattern %.loc6_58 [concrete = constants.%.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.facet.loc6_5.1: %Destroy.type = facet_value constants.%tuple.type.102, (constants.%DestroyImpl.impl_witness.a82, constants.%custom_witness.b80ef6.4) [concrete = constants.%Destroy.facet.aeb]
// CHECK:STDOUT:   %.loc6_5.2: %Destroy.type = converted constants.%tuple.type.102, %Destroy.facet.loc6_5.1 [concrete = constants.%Destroy.facet.aeb]
// CHECK:STDOUT:   %Destroy.facet.loc6_5.2: %Destroy.type = facet_value constants.%tuple.type.102, (constants.%DestroyImpl.impl_witness.a82, constants.%custom_witness.b80ef6.4) [concrete = constants.%Destroy.facet.aeb]
// CHECK:STDOUT:   %.loc6_5.3: %Destroy.type = converted constants.%tuple.type.102, %Destroy.facet.loc6_5.2 [concrete = constants.%Destroy.facet.aeb]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc6_5.1(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc6_5.2(%self.param: ref %empty_tuple.type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc6_5.3(%self.param: ref %tuple.type.0e6) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1]
// CHECK:STDOUT:   %tuple.elem1: ref %empty_tuple.type = tuple_access %self.param, element1
// CHECK:STDOUT:   %Destroy.facet.loc6_5.1: %Destroy.type = facet_value constants.%empty_tuple.type, (constants.%DestroyImpl.impl_witness.674, constants.%custom_witness.b80ef6.2) [concrete = constants.%Destroy.facet.939]
// CHECK:STDOUT:   %.loc6_5.1: %Destroy.type = converted constants.%empty_tuple.type, %Destroy.facet.loc6_5.1 [concrete = constants.%Destroy.facet.939]
// CHECK:STDOUT:   %Destroy.facet.loc6_5.2: %Destroy.type = facet_value constants.%empty_tuple.type, (constants.%DestroyImpl.impl_witness.674, constants.%custom_witness.b80ef6.2) [concrete = constants.%Destroy.facet.939]
// CHECK:STDOUT:   %.loc6_5.2: %Destroy.type = converted constants.%empty_tuple.type, %Destroy.facet.loc6_5.2 [concrete = constants.%Destroy.facet.939]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc6_5.1: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%tuple.elem1)
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT:   %tuple.elem0: ref %empty_struct_type = tuple_access %self.param, element0
// CHECK:STDOUT:   %Destroy.facet.loc6_5.3: %Destroy.type = facet_value constants.%empty_struct_type, (constants.%DestroyImpl.impl_witness.011, constants.%custom_witness.b80ef6.1) [concrete = constants.%Destroy.facet.fed]
// CHECK:STDOUT:   %.loc6_5.3: %Destroy.type = converted constants.%empty_struct_type, %Destroy.facet.loc6_5.3 [concrete = constants.%Destroy.facet.fed]
// CHECK:STDOUT:   %Destroy.facet.loc6_5.4: %Destroy.type = facet_value constants.%empty_struct_type, (constants.%DestroyImpl.impl_witness.011, constants.%custom_witness.b80ef6.1) [concrete = constants.%Destroy.facet.fed]
// CHECK:STDOUT:   %.loc6_5.4: %Destroy.type = converted constants.%empty_struct_type, %Destroy.facet.loc6_5.4 [concrete = constants.%Destroy.facet.fed]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc6_5.2: init %empty_tuple.type = call %SelfDestruct.specific_fn.2(%tuple.elem0)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc6_5.4(%self.param: ref %tuple.type.102) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1]
// CHECK:STDOUT:   %tuple.elem1: ref %empty_tuple.type = tuple_access %self.param, element1
// CHECK:STDOUT:   %Destroy.facet.loc6_5.1: %Destroy.type = facet_value constants.%empty_tuple.type, (constants.%DestroyImpl.impl_witness.674, constants.%custom_witness.b80ef6.2) [concrete = constants.%Destroy.facet.939]
// CHECK:STDOUT:   %.loc6_5.1: %Destroy.type = converted constants.%empty_tuple.type, %Destroy.facet.loc6_5.1 [concrete = constants.%Destroy.facet.939]
// CHECK:STDOUT:   %Destroy.facet.loc6_5.2: %Destroy.type = facet_value constants.%empty_tuple.type, (constants.%DestroyImpl.impl_witness.674, constants.%custom_witness.b80ef6.2) [concrete = constants.%Destroy.facet.939]
// CHECK:STDOUT:   %.loc6_5.2: %Destroy.type = converted constants.%empty_tuple.type, %Destroy.facet.loc6_5.2 [concrete = constants.%Destroy.facet.939]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc6_5.1: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%tuple.elem1)
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT:   %tuple.elem0: ref %tuple.type.0e6 = tuple_access %self.param, element0
// CHECK:STDOUT:   %Destroy.facet.loc6_5.3: %Destroy.type = facet_value constants.%tuple.type.0e6, (constants.%DestroyImpl.impl_witness.462, constants.%custom_witness.b80ef6.3) [concrete = constants.%Destroy.facet.b0b]
// CHECK:STDOUT:   %.loc6_5.3: %Destroy.type = converted constants.%tuple.type.0e6, %Destroy.facet.loc6_5.3 [concrete = constants.%Destroy.facet.b0b]
// CHECK:STDOUT:   %Destroy.facet.loc6_5.4: %Destroy.type = facet_value constants.%tuple.type.0e6, (constants.%DestroyImpl.impl_witness.462, constants.%custom_witness.b80ef6.3) [concrete = constants.%Destroy.facet.b0b]
// CHECK:STDOUT:   %.loc6_5.4: %Destroy.type = converted constants.%tuple.type.0e6, %Destroy.facet.loc6_5.4 [concrete = constants.%Destroy.facet.b0b]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc6_5.2: init %empty_tuple.type = call %SelfDestruct.specific_fn.2(%tuple.elem0)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
