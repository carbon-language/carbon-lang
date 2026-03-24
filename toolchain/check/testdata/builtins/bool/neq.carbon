// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/full.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/builtins/bool/neq.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/builtins/bool/neq.carbon

// --- builtin_call.carbon

library "[[@TEST_NAME]]";

fn Neq(a: bool, b: bool) -> bool = "bool.neq";

class C(B:! bool) {}

fn True() -> C(true);
fn False() -> C(false);

//@dump-sem-ir-begin
var a: C(Neq(true, true)) = False();
//@dump-sem-ir-end
var b: C(Neq(true, false)) = True();
var c: C(Neq(false, true)) = True();
var d: C(Neq(false, false)) = False();

// --- prelude.carbon

library "[[@TEST_NAME]]";

class C(B:! bool) {}

fn True() -> C(true);
fn False() -> C(false);

//@dump-sem-ir-begin
var a: C(true != true) = False();
//@dump-sem-ir-end
var b: C(true != false) = True();
var c: C(false != true) = True();
var d: C(false != false) = False();

// CHECK:STDOUT: --- builtin_call.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Neq.type: type = fn_type @Neq [concrete]
// CHECK:STDOUT:   %Neq: %Neq.type = struct_value () [concrete]
// CHECK:STDOUT:   %C.type: type = generic_class_type @C [concrete]
// CHECK:STDOUT:   %C.generic: %C.type = struct_value () [concrete]
// CHECK:STDOUT:   %true: bool = bool_literal true [concrete]
// CHECK:STDOUT:   %false: bool = bool_literal false [concrete]
// CHECK:STDOUT:   %C.082: type = class_type @C, @C(%false) [concrete]
// CHECK:STDOUT:   %pattern_type.2a5: type = pattern_type %C.082 [concrete]
// CHECK:STDOUT:   %False.type: type = fn_type @False [concrete]
// CHECK:STDOUT:   %False: %False.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.2a5 = ref_binding_pattern a [concrete]
// CHECK:STDOUT:     %a.var_patt: %pattern_type.2a5 = var_pattern %a.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.var: ref %C.082 = var %a.var_patt [concrete]
// CHECK:STDOUT:   %.loc25_25.1: type = splice_block %C.loc25 [concrete = constants.%C.082] {
// CHECK:STDOUT:     %C.ref.loc25: %C.type = name_ref C, %C.decl [concrete = constants.%C.generic]
// CHECK:STDOUT:     %Neq.ref.loc25: %Neq.type = name_ref Neq, %Neq.decl [concrete = constants.%Neq]
// CHECK:STDOUT:     %true.loc25_14: bool = bool_literal true [concrete = constants.%true]
// CHECK:STDOUT:     %true.loc25_20: bool = bool_literal true [concrete = constants.%true]
// CHECK:STDOUT:     %Neq.call.loc25: init bool = call %Neq.ref.loc25(%true.loc25_14, %true.loc25_20) [concrete = constants.%false]
// CHECK:STDOUT:     %.loc25_25.2: bool = value_of_initializer %Neq.call.loc25 [concrete = constants.%false]
// CHECK:STDOUT:     %.loc25_25.3: bool = converted %Neq.call.loc25, %.loc25_25.2 [concrete = constants.%false]
// CHECK:STDOUT:     %C.loc25: type = class_type @C, @C(constants.%false) [concrete = constants.%C.082]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: ref %C.082 = ref_binding a, %a.var [concrete = %a.var]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %False.ref.loc25: %False.type = name_ref False, file.%False.decl [concrete = constants.%False]
// CHECK:STDOUT:   %.loc25: ref %C.082 = splice_block file.%a.var [concrete = file.%a.var] {}
// CHECK:STDOUT:   %False.call.loc25: init %C.082 to %.loc25 = call %False.ref.loc25()
// CHECK:STDOUT:   assign file.%a.var, %False.call.loc25
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- prelude.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C.type: type = generic_class_type @C [concrete]
// CHECK:STDOUT:   %C.generic: %C.type = struct_value () [concrete]
// CHECK:STDOUT:   %true: bool = bool_literal true [concrete]
// CHECK:STDOUT:   %false: bool = bool_literal false [concrete]
// CHECK:STDOUT:   %C.082: type = class_type @C, @C(%false) [concrete]
// CHECK:STDOUT:   %pattern_type.2a5: type = pattern_type %C.082 [concrete]
// CHECK:STDOUT:   %False.type: type = fn_type @False [concrete]
// CHECK:STDOUT:   %False: %False.type = struct_value () [concrete]
// CHECK:STDOUT:   %EqWith.type.863: type = facet_type <@EqWith, @EqWith(bool)> [concrete]
// CHECK:STDOUT:   %EqWith.impl_witness: <witness> = impl_witness imports.%EqWith.impl_witness_table [concrete]
// CHECK:STDOUT:   %EqWith.facet: %EqWith.type.863 = facet_value bool, (%EqWith.impl_witness) [concrete]
// CHECK:STDOUT:   %EqWith.WithSelf.NotEqual.type.14c: type = fn_type @EqWith.WithSelf.NotEqual, @EqWith.WithSelf(bool, %EqWith.facet) [concrete]
// CHECK:STDOUT:   %.d13: type = fn_type_with_self_type %EqWith.WithSelf.NotEqual.type.14c, %EqWith.facet [concrete]
// CHECK:STDOUT:   %bool.as.EqWith.impl.NotEqual.type: type = fn_type @bool.as.EqWith.impl.NotEqual [concrete]
// CHECK:STDOUT:   %bool.as.EqWith.impl.NotEqual: %bool.as.EqWith.impl.NotEqual.type = struct_value () [concrete]
// CHECK:STDOUT:   %bool.as.EqWith.impl.NotEqual.bound.a7d: <bound method> = bound_method %true, %bool.as.EqWith.impl.NotEqual [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.093 = import_ref Core//prelude/operators/comparison, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.fb0: %bool.as.EqWith.impl.NotEqual.type = import_ref Core//prelude/operators/comparison, loc{{\d+_\d+}}, loaded [concrete = constants.%bool.as.EqWith.impl.NotEqual]
// CHECK:STDOUT:   %EqWith.impl_witness_table = impl_witness_table (%Core.import_ref.093, %Core.import_ref.fb0), @bool.as.EqWith.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.2a5 = ref_binding_pattern a [concrete]
// CHECK:STDOUT:     %a.var_patt: %pattern_type.2a5 = var_pattern %a.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.var: ref %C.082 = var %a.var_patt [concrete]
// CHECK:STDOUT:   %.loc41_22.1: type = splice_block %C.loc41 [concrete = constants.%C.082] {
// CHECK:STDOUT:     %C.ref.loc41: %C.type = name_ref C, %C.decl [concrete = constants.%C.generic]
// CHECK:STDOUT:     %true.loc41_10: bool = bool_literal true [concrete = constants.%true]
// CHECK:STDOUT:     %true.loc41_18: bool = bool_literal true [concrete = constants.%true]
// CHECK:STDOUT:     %impl.elem1.loc41: %.d13 = impl_witness_access constants.%EqWith.impl_witness, element1 [concrete = constants.%bool.as.EqWith.impl.NotEqual]
// CHECK:STDOUT:     %bound_method.loc41: <bound method> = bound_method %true.loc41_10, %impl.elem1.loc41 [concrete = constants.%bool.as.EqWith.impl.NotEqual.bound.a7d]
// CHECK:STDOUT:     %bool.as.EqWith.impl.NotEqual.call.loc41: init bool = call %bound_method.loc41(%true.loc41_10, %true.loc41_18) [concrete = constants.%false]
// CHECK:STDOUT:     %.loc41_22.2: bool = value_of_initializer %bool.as.EqWith.impl.NotEqual.call.loc41 [concrete = constants.%false]
// CHECK:STDOUT:     %.loc41_22.3: bool = converted %bool.as.EqWith.impl.NotEqual.call.loc41, %.loc41_22.2 [concrete = constants.%false]
// CHECK:STDOUT:     %C.loc41: type = class_type @C, @C(constants.%false) [concrete = constants.%C.082]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: ref %C.082 = ref_binding a, %a.var [concrete = %a.var]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %False.ref.loc41: %False.type = name_ref False, file.%False.decl [concrete = constants.%False]
// CHECK:STDOUT:   %.loc41: ref %C.082 = splice_block file.%a.var [concrete = file.%a.var] {}
// CHECK:STDOUT:   %False.call.loc41: init %C.082 to %.loc41 = call %False.ref.loc41()
// CHECK:STDOUT:   assign file.%a.var, %False.call.loc41
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
