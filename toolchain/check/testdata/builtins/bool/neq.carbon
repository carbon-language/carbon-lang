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
// CHECK:STDOUT:   %C.2ba: type = class_type @C, @C(%false) [concrete]
// CHECK:STDOUT:   %pattern_type.d8f: type = pattern_type %C.2ba [concrete]
// CHECK:STDOUT:   %False.type: type = fn_type @False [concrete]
// CHECK:STDOUT:   %False: %False.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.d8f = binding_pattern a [concrete]
// CHECK:STDOUT:     %a.var_patt: %pattern_type.d8f = var_pattern %a.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.var: ref %C.2ba = var %a.var_patt [concrete]
// CHECK:STDOUT:   %.loc12_25.1: type = splice_block %C.loc12 [concrete = constants.%C.2ba] {
// CHECK:STDOUT:     %C.ref.loc12: %C.type = name_ref C, %C.decl [concrete = constants.%C.generic]
// CHECK:STDOUT:     %Neq.ref.loc12: %Neq.type = name_ref Neq, %Neq.decl [concrete = constants.%Neq]
// CHECK:STDOUT:     %true.loc12_14: bool = bool_literal true [concrete = constants.%true]
// CHECK:STDOUT:     %true.loc12_20: bool = bool_literal true [concrete = constants.%true]
// CHECK:STDOUT:     %Neq.call.loc12: init bool = call %Neq.ref.loc12(%true.loc12_14, %true.loc12_20) [concrete = constants.%false]
// CHECK:STDOUT:     %.loc12_25.2: bool = value_of_initializer %Neq.call.loc12 [concrete = constants.%false]
// CHECK:STDOUT:     %.loc12_25.3: bool = converted %Neq.call.loc12, %.loc12_25.2 [concrete = constants.%false]
// CHECK:STDOUT:     %C.loc12: type = class_type @C, @C(constants.%false) [concrete = constants.%C.2ba]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: ref %C.2ba = bind_name a, %a.var [concrete = %a.var]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %False.ref.loc12: %False.type = name_ref False, file.%False.decl [concrete = constants.%False]
// CHECK:STDOUT:   %.loc12: ref %C.2ba = splice_block file.%a.var [concrete = file.%a.var] {}
// CHECK:STDOUT:   %False.call.loc12: init %C.2ba = call %False.ref.loc12() to %.loc12
// CHECK:STDOUT:   assign file.%a.var, %False.call.loc12
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
// CHECK:STDOUT:   %C.2ba: type = class_type @C, @C(%false) [concrete]
// CHECK:STDOUT:   %pattern_type.d8f: type = pattern_type %C.2ba [concrete]
// CHECK:STDOUT:   %False.type: type = fn_type @False [concrete]
// CHECK:STDOUT:   %False: %False.type = struct_value () [concrete]
// CHECK:STDOUT:   %EqWith.type.a38: type = facet_type <@EqWith, @EqWith(bool)> [concrete]
// CHECK:STDOUT:   %EqWith.NotEqual.type.f03: type = fn_type @EqWith.NotEqual, @EqWith(bool) [concrete]
// CHECK:STDOUT:   %EqWith.impl_witness: <witness> = impl_witness imports.%EqWith.impl_witness_table [concrete]
// CHECK:STDOUT:   %EqWith.facet: %EqWith.type.a38 = facet_value bool, (%EqWith.impl_witness) [concrete]
// CHECK:STDOUT:   %.c2f: type = fn_type_with_self_type %EqWith.NotEqual.type.f03, %EqWith.facet [concrete]
// CHECK:STDOUT:   %bool.as.EqWith.impl.NotEqual.type: type = fn_type @bool.as.EqWith.impl.NotEqual [concrete]
// CHECK:STDOUT:   %bool.as.EqWith.impl.NotEqual: %bool.as.EqWith.impl.NotEqual.type = struct_value () [concrete]
// CHECK:STDOUT:   %bool.as.EqWith.impl.NotEqual.bound.0d4: <bound method> = bound_method %true, %bool.as.EqWith.impl.NotEqual [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.a67 = import_ref Core//prelude/operators/comparison, loc36_45, unloaded
// CHECK:STDOUT:   %Core.import_ref.76f: %bool.as.EqWith.impl.NotEqual.type = import_ref Core//prelude/operators/comparison, loc37_48, loaded [concrete = constants.%bool.as.EqWith.impl.NotEqual]
// CHECK:STDOUT:   %EqWith.impl_witness_table = impl_witness_table (%Core.import_ref.a67, %Core.import_ref.76f), @bool.as.EqWith.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.d8f = binding_pattern a [concrete]
// CHECK:STDOUT:     %a.var_patt: %pattern_type.d8f = var_pattern %a.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.var: ref %C.2ba = var %a.var_patt [concrete]
// CHECK:STDOUT:   %.loc10_22.1: type = splice_block %C.loc10 [concrete = constants.%C.2ba] {
// CHECK:STDOUT:     %C.ref.loc10: %C.type = name_ref C, %C.decl [concrete = constants.%C.generic]
// CHECK:STDOUT:     %true.loc10_10: bool = bool_literal true [concrete = constants.%true]
// CHECK:STDOUT:     %true.loc10_18: bool = bool_literal true [concrete = constants.%true]
// CHECK:STDOUT:     %impl.elem1.loc10: %.c2f = impl_witness_access constants.%EqWith.impl_witness, element1 [concrete = constants.%bool.as.EqWith.impl.NotEqual]
// CHECK:STDOUT:     %bound_method.loc10: <bound method> = bound_method %true.loc10_10, %impl.elem1.loc10 [concrete = constants.%bool.as.EqWith.impl.NotEqual.bound.0d4]
// CHECK:STDOUT:     %bool.as.EqWith.impl.NotEqual.call.loc10: init bool = call %bound_method.loc10(%true.loc10_10, %true.loc10_18) [concrete = constants.%false]
// CHECK:STDOUT:     %.loc10_22.2: bool = value_of_initializer %bool.as.EqWith.impl.NotEqual.call.loc10 [concrete = constants.%false]
// CHECK:STDOUT:     %.loc10_22.3: bool = converted %bool.as.EqWith.impl.NotEqual.call.loc10, %.loc10_22.2 [concrete = constants.%false]
// CHECK:STDOUT:     %C.loc10: type = class_type @C, @C(constants.%false) [concrete = constants.%C.2ba]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: ref %C.2ba = bind_name a, %a.var [concrete = %a.var]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %False.ref.loc10: %False.type = name_ref False, file.%False.decl [concrete = constants.%False]
// CHECK:STDOUT:   %.loc10: ref %C.2ba = splice_block file.%a.var [concrete = file.%a.var] {}
// CHECK:STDOUT:   %False.call.loc10: init %C.2ba = call %False.ref.loc10() to %.loc10
// CHECK:STDOUT:   assign file.%a.var, %False.call.loc10
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
