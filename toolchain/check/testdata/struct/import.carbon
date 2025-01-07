// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/struct/import.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/struct/import.carbon

// --- implicit.carbon

package Implicit;

var a_ref: {.a: i32} = {.a = 0};
var b_ref: {.a: {.b: i32, .c: (i32,)}, .d: i32} =
    {.a = {.b = 0, .c = (0,)}, .d = 0};

class C(S:! {.a: i32, .b: i32}) {}
fn F() -> C({.a = 1, .b = 2});

// --- implicit.impl.carbon

impl package Implicit;

var a: {.a: i32} = a_ref;
var b: {.a: {.b: i32, .c: (i32,)}, .d: i32} = b_ref;
var c: C({.a = 1, .b = 2}) = F();

// --- fail_bad_type.impl.carbon

impl package Implicit;
// CHECK:STDERR: fail_bad_type.impl.carbon:[[@LINE+8]]:14: error: missing value for field `a` in struct initialization [StructInitMissingFieldInLiteral]
// CHECK:STDERR: var c_bad: C({.c = 1, .d = 2}) = F();
// CHECK:STDERR:              ^~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_bad_type.impl.carbon:[[@LINE-4]]:6: in import [InImport]
// CHECK:STDERR: implicit.carbon:8:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
// CHECK:STDERR: class C(S:! {.a: i32, .b: i32}) {}
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
var c_bad: C({.c = 1, .d = 2}) = F();

// --- fail_bad_value.impl.carbon

impl package Implicit;
// CHECK:STDERR: fail_bad_value.impl.carbon:[[@LINE+6]]:1: error: cannot implicitly convert from `C({.a = 1, .b = 2})` to `C({.a = 3, .b = 4})` [ImplicitAsConversionFailure]
// CHECK:STDERR: var c_bad: C({.a = 3, .b = 4}) = F();
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_bad_value.impl.carbon:[[@LINE+3]]:1: note: type `C({.a = 1, .b = 2})` does not implement interface `Core.ImplicitAs(C({.a = 3, .b = 4}))` [MissingImplInMemberAccessNote]
// CHECK:STDERR: var c_bad: C({.a = 3, .b = 4}) = F();
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
var c_bad: C({.a = 3, .b = 4}) = F();

// CHECK:STDOUT: --- implicit.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [template]
// CHECK:STDOUT:   %struct_type.a.7cc1: type = struct_type {.a: %i32} [template]
// CHECK:STDOUT:   %int_0.5c61: Core.IntLiteral = int_value 0 [template]
// CHECK:STDOUT:   %struct_type.a.a6c3: type = struct_type {.a: Core.IntLiteral} [template]
// CHECK:STDOUT:   %Convert.type.cd1a: type = fn_type @Convert.1, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %impl_witness.5b05: <witness> = impl_witness (imports.%import_ref.7238), @impl.1(%int_32) [template]
// CHECK:STDOUT:   %Convert.type.4666: type = fn_type @Convert.2, @impl.1(%int_32) [template]
// CHECK:STDOUT:   %Convert.9251: %Convert.type.4666 = struct_value () [template]
// CHECK:STDOUT:   %Convert.bound.b412: <bound method> = bound_method %int_0.5c61, %Convert.9251 [template]
// CHECK:STDOUT:   %Convert.specific_fn.c350: <specific function> = specific_function %Convert.bound.b412, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_0.f61a: %i32 = int_value 0 [template]
// CHECK:STDOUT:   %struct.96f9: %struct_type.a.7cc1 = struct_value (%int_0.f61a) [template]
// CHECK:STDOUT:   %tuple.type.0e4e: type = tuple_type (%i32) [template]
// CHECK:STDOUT:   %struct_type.b.c.8794: type = struct_type {.b: %i32, .c: %tuple.type.0e4e} [template]
// CHECK:STDOUT:   %struct_type.a.d.1e9a: type = struct_type {.a: %struct_type.b.c.8794, .d: %i32} [template]
// CHECK:STDOUT:   %tuple.type.985f: type = tuple_type (Core.IntLiteral) [template]
// CHECK:STDOUT:   %struct_type.b.c.9af5: type = struct_type {.b: Core.IntLiteral, .c: %tuple.type.985f} [template]
// CHECK:STDOUT:   %struct_type.a.d.b82a: type = struct_type {.a: %struct_type.b.c.9af5, .d: Core.IntLiteral} [template]
// CHECK:STDOUT:   %tuple: %tuple.type.0e4e = tuple_value (%int_0.f61a) [template]
// CHECK:STDOUT:   %struct.d066: %struct_type.b.c.8794 = struct_value (%int_0.f61a, %tuple) [template]
// CHECK:STDOUT:   %struct.34f0: %struct_type.a.d.1e9a = struct_value (%struct.d066, %int_0.f61a) [template]
// CHECK:STDOUT:   %struct_type.a.b.3dec: type = struct_type {.a: %i32, .b: %i32} [template]
// CHECK:STDOUT:   %S: %struct_type.a.b.3dec = bind_symbolic_name S, 0 [symbolic]
// CHECK:STDOUT:   %S.patt: %struct_type.a.b.3dec = symbolic_binding_pattern S, 0 [symbolic]
// CHECK:STDOUT:   %C.type: type = generic_class_type @C [template]
// CHECK:STDOUT:   %C.generic: %C.type = struct_value () [template]
// CHECK:STDOUT:   %C.c616: type = class_type @C, @C(%S) [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [template]
// CHECK:STDOUT:   %complete_type.3571: <witness> = complete_type_witness %empty_struct_type [template]
// CHECK:STDOUT:   %int_1.5b81: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT:   %int_2.ecc8: Core.IntLiteral = int_value 2 [template]
// CHECK:STDOUT:   %struct_type.a.b.cfdb: type = struct_type {.a: Core.IntLiteral, .b: Core.IntLiteral} [template]
// CHECK:STDOUT:   %Convert.bound.afd1: <bound method> = bound_method %int_1.5b81, %Convert.9251 [template]
// CHECK:STDOUT:   %Convert.specific_fn.b730: <specific function> = specific_function %Convert.bound.afd1, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_1.c605: %i32 = int_value 1 [template]
// CHECK:STDOUT:   %Convert.bound.f0b0: <bound method> = bound_method %int_2.ecc8, %Convert.9251 [template]
// CHECK:STDOUT:   %Convert.specific_fn.20e1: <specific function> = specific_function %Convert.bound.f0b0, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_2.1667: %i32 = int_value 2 [template]
// CHECK:STDOUT:   %struct.8898: %struct_type.a.b.3dec = struct_value (%int_1.c605, %int_2.1667) [template]
// CHECK:STDOUT:   %C.61d5: type = class_type @C, @C(%struct.8898) [template]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int = %import_ref.187e
// CHECK:STDOUT:     .ImplicitAs = %import_ref.a691
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .a_ref = %a_ref
// CHECK:STDOUT:     .b_ref = %b_ref
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %a_ref.var: ref %struct_type.a.7cc1 = var a_ref
// CHECK:STDOUT:   %a_ref: ref %struct_type.a.7cc1 = bind_name a_ref, %a_ref.var
// CHECK:STDOUT:   %b_ref.var: ref %struct_type.a.d.1e9a = var b_ref
// CHECK:STDOUT:   %b_ref: ref %struct_type.a.d.1e9a = bind_name b_ref, %b_ref.var
// CHECK:STDOUT:   %C.decl: %C.type = class_decl @C [template = constants.%C.generic] {
// CHECK:STDOUT:     %S.patt.loc8_9.1: %struct_type.a.b.3dec = symbolic_binding_pattern S, 0 [symbolic = %S.patt.loc8_9.2 (constants.%S.patt)]
// CHECK:STDOUT:     %S.param_patt: %struct_type.a.b.3dec = value_param_pattern %S.patt.loc8_9.1, runtime_param<invalid> [symbolic = %S.patt.loc8_9.2 (constants.%S.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %S.param: %struct_type.a.b.3dec = value_param runtime_param<invalid>
// CHECK:STDOUT:     %.loc8: type = splice_block %struct_type.a.b [template = constants.%struct_type.a.b.3dec] {
// CHECK:STDOUT:       %int_32.loc8_18: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:       %i32.loc8_18: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:       %int_32.loc8_27: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:       %i32.loc8_27: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:       %struct_type.a.b: type = struct_type {.a: %i32, .b: %i32} [template = constants.%struct_type.a.b.3dec]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %S.loc8_9.1: %struct_type.a.b.3dec = bind_symbolic_name S, 0, %S.param [symbolic = %S.loc8_9.2 (constants.%S)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [template = constants.%F] {
// CHECK:STDOUT:     %return.patt: %C.61d5 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %C.61d5 = out_param_pattern %return.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref: %C.type = name_ref C, file.%C.decl [template = constants.%C.generic]
// CHECK:STDOUT:     %int_1: Core.IntLiteral = int_value 1 [template = constants.%int_1.5b81]
// CHECK:STDOUT:     %int_2: Core.IntLiteral = int_value 2 [template = constants.%int_2.ecc8]
// CHECK:STDOUT:     %.loc9_28.1: %struct_type.a.b.cfdb = struct_literal (%int_1, %int_2)
// CHECK:STDOUT:     %impl.elem0.loc9_28.1: %Convert.type.cd1a = impl_witness_access constants.%impl_witness.5b05, element0 [template = constants.%Convert.9251]
// CHECK:STDOUT:     %Convert.bound.loc9_28.1: <bound method> = bound_method %int_1, %impl.elem0.loc9_28.1 [template = constants.%Convert.bound.afd1]
// CHECK:STDOUT:     %Convert.specific_fn.loc9_28.1: <specific function> = specific_function %Convert.bound.loc9_28.1, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.b730]
// CHECK:STDOUT:     %int.convert_checked.loc9_28.1: init %i32 = call %Convert.specific_fn.loc9_28.1(%int_1) [template = constants.%int_1.c605]
// CHECK:STDOUT:     %.loc9_28.2: %i32 = value_of_initializer %int.convert_checked.loc9_28.1 [template = constants.%int_1.c605]
// CHECK:STDOUT:     %.loc9_28.3: %i32 = converted %int_1, %.loc9_28.2 [template = constants.%int_1.c605]
// CHECK:STDOUT:     %impl.elem0.loc9_28.2: %Convert.type.cd1a = impl_witness_access constants.%impl_witness.5b05, element0 [template = constants.%Convert.9251]
// CHECK:STDOUT:     %Convert.bound.loc9_28.2: <bound method> = bound_method %int_2, %impl.elem0.loc9_28.2 [template = constants.%Convert.bound.f0b0]
// CHECK:STDOUT:     %Convert.specific_fn.loc9_28.2: <specific function> = specific_function %Convert.bound.loc9_28.2, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.20e1]
// CHECK:STDOUT:     %int.convert_checked.loc9_28.2: init %i32 = call %Convert.specific_fn.loc9_28.2(%int_2) [template = constants.%int_2.1667]
// CHECK:STDOUT:     %.loc9_28.4: %i32 = value_of_initializer %int.convert_checked.loc9_28.2 [template = constants.%int_2.1667]
// CHECK:STDOUT:     %.loc9_28.5: %i32 = converted %int_2, %.loc9_28.4 [template = constants.%int_2.1667]
// CHECK:STDOUT:     %struct: %struct_type.a.b.3dec = struct_value (%.loc9_28.3, %.loc9_28.5) [template = constants.%struct.8898]
// CHECK:STDOUT:     %.loc9_29: %struct_type.a.b.3dec = converted %.loc9_28.1, %struct [template = constants.%struct.8898]
// CHECK:STDOUT:     %C: type = class_type @C, @C(constants.%struct.8898) [template = constants.%C.61d5]
// CHECK:STDOUT:     %return.param: ref %C.61d5 = out_param runtime_param0
// CHECK:STDOUT:     %return: ref %C.61d5 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @C(%S.loc8_9.1: %struct_type.a.b.3dec) {
// CHECK:STDOUT:   %S.loc8_9.2: %struct_type.a.b.3dec = bind_symbolic_name S, 0 [symbolic = %S.loc8_9.2 (constants.%S)]
// CHECK:STDOUT:   %S.patt.loc8_9.2: %struct_type.a.b.3dec = symbolic_binding_pattern S, 0 [symbolic = %S.patt.loc8_9.2 (constants.%S.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness %empty_struct_type [template = constants.%complete_type.3571]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%C.c616
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() -> %C.61d5;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_0.loc4: Core.IntLiteral = int_value 0 [template = constants.%int_0.5c61]
// CHECK:STDOUT:   %.loc4_31.1: %struct_type.a.a6c3 = struct_literal (%int_0.loc4)
// CHECK:STDOUT:   %impl.elem0.loc4: %Convert.type.cd1a = impl_witness_access constants.%impl_witness.5b05, element0 [template = constants.%Convert.9251]
// CHECK:STDOUT:   %Convert.bound.loc4: <bound method> = bound_method %int_0.loc4, %impl.elem0.loc4 [template = constants.%Convert.bound.b412]
// CHECK:STDOUT:   %Convert.specific_fn.loc4: <specific function> = specific_function %Convert.bound.loc4, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.c350]
// CHECK:STDOUT:   %int.convert_checked.loc4: init %i32 = call %Convert.specific_fn.loc4(%int_0.loc4) [template = constants.%int_0.f61a]
// CHECK:STDOUT:   %.loc4_31.2: init %i32 = converted %int_0.loc4, %int.convert_checked.loc4 [template = constants.%int_0.f61a]
// CHECK:STDOUT:   %.loc4_31.3: init %struct_type.a.7cc1 = struct_init (%.loc4_31.2) to file.%a_ref.var [template = constants.%struct.96f9]
// CHECK:STDOUT:   %.loc4_32: init %struct_type.a.7cc1 = converted %.loc4_31.1, %.loc4_31.3 [template = constants.%struct.96f9]
// CHECK:STDOUT:   assign file.%a_ref.var, %.loc4_32
// CHECK:STDOUT:   %int_0.loc6_17: Core.IntLiteral = int_value 0 [template = constants.%int_0.5c61]
// CHECK:STDOUT:   %int_0.loc6_26: Core.IntLiteral = int_value 0 [template = constants.%int_0.5c61]
// CHECK:STDOUT:   %.loc6_28.1: %tuple.type.985f = tuple_literal (%int_0.loc6_26)
// CHECK:STDOUT:   %.loc6_29.1: %struct_type.b.c.9af5 = struct_literal (%int_0.loc6_17, %.loc6_28.1)
// CHECK:STDOUT:   %int_0.loc6_37: Core.IntLiteral = int_value 0 [template = constants.%int_0.5c61]
// CHECK:STDOUT:   %.loc6_38.1: %struct_type.a.d.b82a = struct_literal (%.loc6_29.1, %int_0.loc6_37)
// CHECK:STDOUT:   %impl.elem0.loc6_29: %Convert.type.cd1a = impl_witness_access constants.%impl_witness.5b05, element0 [template = constants.%Convert.9251]
// CHECK:STDOUT:   %Convert.bound.loc6_29: <bound method> = bound_method %int_0.loc6_17, %impl.elem0.loc6_29 [template = constants.%Convert.bound.b412]
// CHECK:STDOUT:   %Convert.specific_fn.loc6_29: <specific function> = specific_function %Convert.bound.loc6_29, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.c350]
// CHECK:STDOUT:   %int.convert_checked.loc6_29: init %i32 = call %Convert.specific_fn.loc6_29(%int_0.loc6_17) [template = constants.%int_0.f61a]
// CHECK:STDOUT:   %.loc6_29.2: init %i32 = converted %int_0.loc6_17, %int.convert_checked.loc6_29 [template = constants.%int_0.f61a]
// CHECK:STDOUT:   %.loc6_38.2: ref %struct_type.b.c.8794 = struct_access file.%b_ref.var, element0
// CHECK:STDOUT:   %.loc6_29.3: ref %i32 = struct_access %.loc6_38.2, element0
// CHECK:STDOUT:   %.loc6_29.4: init %i32 = initialize_from %.loc6_29.2 to %.loc6_29.3 [template = constants.%int_0.f61a]
// CHECK:STDOUT:   %impl.elem0.loc6_28: %Convert.type.cd1a = impl_witness_access constants.%impl_witness.5b05, element0 [template = constants.%Convert.9251]
// CHECK:STDOUT:   %Convert.bound.loc6_28: <bound method> = bound_method %int_0.loc6_26, %impl.elem0.loc6_28 [template = constants.%Convert.bound.b412]
// CHECK:STDOUT:   %Convert.specific_fn.loc6_28: <specific function> = specific_function %Convert.bound.loc6_28, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.c350]
// CHECK:STDOUT:   %int.convert_checked.loc6_28: init %i32 = call %Convert.specific_fn.loc6_28(%int_0.loc6_26) [template = constants.%int_0.f61a]
// CHECK:STDOUT:   %.loc6_28.2: init %i32 = converted %int_0.loc6_26, %int.convert_checked.loc6_28 [template = constants.%int_0.f61a]
// CHECK:STDOUT:   %.loc6_29.5: ref %tuple.type.0e4e = struct_access %.loc6_38.2, element1
// CHECK:STDOUT:   %.loc6_28.3: init %tuple.type.0e4e = tuple_init (%.loc6_28.2) to %.loc6_29.5 [template = constants.%tuple]
// CHECK:STDOUT:   %.loc6_29.6: init %tuple.type.0e4e = converted %.loc6_28.1, %.loc6_28.3 [template = constants.%tuple]
// CHECK:STDOUT:   %.loc6_29.7: init %tuple.type.0e4e = initialize_from %.loc6_29.6 to %.loc6_29.5 [template = constants.%tuple]
// CHECK:STDOUT:   %.loc6_29.8: init %struct_type.b.c.8794 = struct_init (%.loc6_29.4, %.loc6_29.7) to %.loc6_38.2 [template = constants.%struct.d066]
// CHECK:STDOUT:   %.loc6_38.3: init %struct_type.b.c.8794 = converted %.loc6_29.1, %.loc6_29.8 [template = constants.%struct.d066]
// CHECK:STDOUT:   %impl.elem0.loc6_38: %Convert.type.cd1a = impl_witness_access constants.%impl_witness.5b05, element0 [template = constants.%Convert.9251]
// CHECK:STDOUT:   %Convert.bound.loc6_38: <bound method> = bound_method %int_0.loc6_37, %impl.elem0.loc6_38 [template = constants.%Convert.bound.b412]
// CHECK:STDOUT:   %Convert.specific_fn.loc6_38: <specific function> = specific_function %Convert.bound.loc6_38, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.c350]
// CHECK:STDOUT:   %int.convert_checked.loc6_38: init %i32 = call %Convert.specific_fn.loc6_38(%int_0.loc6_37) [template = constants.%int_0.f61a]
// CHECK:STDOUT:   %.loc6_38.4: init %i32 = converted %int_0.loc6_37, %int.convert_checked.loc6_38 [template = constants.%int_0.f61a]
// CHECK:STDOUT:   %.loc6_38.5: ref %i32 = struct_access file.%b_ref.var, element1
// CHECK:STDOUT:   %.loc6_38.6: init %i32 = initialize_from %.loc6_38.4 to %.loc6_38.5 [template = constants.%int_0.f61a]
// CHECK:STDOUT:   %.loc6_38.7: init %struct_type.a.d.1e9a = struct_init (%.loc6_38.3, %.loc6_38.6) to file.%b_ref.var [template = constants.%struct.34f0]
// CHECK:STDOUT:   %.loc6_39: init %struct_type.a.d.1e9a = converted %.loc6_38.1, %.loc6_38.7 [template = constants.%struct.34f0]
// CHECK:STDOUT:   assign file.%b_ref.var, %.loc6_39
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%S) {
// CHECK:STDOUT:   %S.loc8_9.2 => constants.%S
// CHECK:STDOUT:   %S.patt.loc8_9.2 => constants.%S
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%struct.8898) {
// CHECK:STDOUT:   %S.loc8_9.2 => constants.%struct.8898
// CHECK:STDOUT:   %S.patt.loc8_9.2 => constants.%struct.8898
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- implicit.impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [template]
// CHECK:STDOUT:   %struct_type.a: type = struct_type {.a: %i32} [template]
// CHECK:STDOUT:   %tuple.type.5907: type = tuple_type (%i32) [template]
// CHECK:STDOUT:   %struct_type.b.c: type = struct_type {.b: %i32, .c: %tuple.type.5907} [template]
// CHECK:STDOUT:   %struct_type.a.d.b93b: type = struct_type {.a: %struct_type.b.c, .d: %i32} [template]
// CHECK:STDOUT:   %C.type: type = generic_class_type @C [template]
// CHECK:STDOUT:   %C.generic: %C.type = struct_value () [template]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [template]
// CHECK:STDOUT:   %complete_type.3571: <witness> = complete_type_witness %empty_struct_type [template]
// CHECK:STDOUT:   %struct_type.a.b.29a5: type = struct_type {.a: %i32, .b: %i32} [template]
// CHECK:STDOUT:   %S: %struct_type.a.b.29a5 = bind_symbolic_name S, 0 [symbolic]
// CHECK:STDOUT:   %S.patt: %struct_type.a.b.29a5 = symbolic_binding_pattern S, 0 [symbolic]
// CHECK:STDOUT:   %int_1.0593: %i32 = int_value 1 [template]
// CHECK:STDOUT:   %int_2.ed97: %i32 = int_value 2 [template]
// CHECK:STDOUT:   %struct: %struct_type.a.b.29a5 = struct_value (%int_1.0593, %int_2.ed97) [template]
// CHECK:STDOUT:   %C.c372: type = class_type @C, @C(%struct) [template]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %import_ref.cd5b: ref %struct_type.a = import_ref Implicit//default, a_ref, loaded
// CHECK:STDOUT:   %import_ref.8492: ref %struct_type.a.d.b93b = import_ref Implicit//default, b_ref, loaded
// CHECK:STDOUT:   %import_ref.c8ab: %C.type = import_ref Implicit//default, C, loaded [template = constants.%C.generic]
// CHECK:STDOUT:   %import_ref.f0f6: %F.type = import_ref Implicit//default, F, loaded [template = constants.%F]
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int = %import_ref.d695
// CHECK:STDOUT:     .ImplicitAs = %import_ref.edd8
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.8f24: <witness> = import_ref Implicit//default, loc8_34, loaded [template = constants.%complete_type.3571]
// CHECK:STDOUT:   %import_ref.5c79 = import_ref Implicit//default, inst1017 [no loc], unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .a_ref = imports.%import_ref.cd5b
// CHECK:STDOUT:     .b_ref = imports.%import_ref.8492
// CHECK:STDOUT:     .C = imports.%import_ref.c8ab
// CHECK:STDOUT:     .F = imports.%import_ref.f0f6
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .a = %a
// CHECK:STDOUT:     .b = %b
// CHECK:STDOUT:     .c = %c
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Implicit.import = import Implicit
// CHECK:STDOUT:   %default.import = import <invalid>
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %a.var: ref %struct_type.a = var a
// CHECK:STDOUT:   %a: ref %struct_type.a = bind_name a, %a.var
// CHECK:STDOUT:   %b.var: ref %struct_type.a.d.b93b = var b
// CHECK:STDOUT:   %b: ref %struct_type.a.d.b93b = bind_name b, %b.var
// CHECK:STDOUT:   %c.var: ref %C.c372 = var c
// CHECK:STDOUT:   %c: ref %C.c372 = bind_name c, %c.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @C(constants.%S: %struct_type.a.b.29a5) [from "implicit.carbon"] {
// CHECK:STDOUT:   %S: %struct_type.a.b.29a5 = bind_symbolic_name S, 0 [symbolic = %S (constants.%S)]
// CHECK:STDOUT:   %S.patt: %struct_type.a.b.29a5 = symbolic_binding_pattern S, 0 [symbolic = %S.patt (constants.%S.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%import_ref.5c79
// CHECK:STDOUT:     complete_type_witness = imports.%import_ref.8f24
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() -> %C.c372 [from "implicit.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a_ref.ref: ref %struct_type.a = name_ref a_ref, imports.%import_ref.cd5b
// CHECK:STDOUT:   %.loc4_20.1: ref %i32 = struct_access %a_ref.ref, element0
// CHECK:STDOUT:   %.loc4_20.2: %i32 = bind_value %.loc4_20.1
// CHECK:STDOUT:   %.loc4_20.3: init %struct_type.a = struct_init (%.loc4_20.2) to file.%a.var
// CHECK:STDOUT:   %.loc4_25: init %struct_type.a = converted %a_ref.ref, %.loc4_20.3
// CHECK:STDOUT:   assign file.%a.var, %.loc4_25
// CHECK:STDOUT:   %b_ref.ref: ref %struct_type.a.d.b93b = name_ref b_ref, imports.%import_ref.8492
// CHECK:STDOUT:   %.loc5_47.1: ref %struct_type.b.c = struct_access %b_ref.ref, element0
// CHECK:STDOUT:   %.loc5_47.2: ref %i32 = struct_access %.loc5_47.1, element0
// CHECK:STDOUT:   %.loc5_47.3: %i32 = bind_value %.loc5_47.2
// CHECK:STDOUT:   %.loc5_47.4: ref %struct_type.b.c = struct_access file.%b.var, element0
// CHECK:STDOUT:   %.loc5_47.5: ref %i32 = struct_access %.loc5_47.4, element0
// CHECK:STDOUT:   %.loc5_47.6: init %i32 = initialize_from %.loc5_47.3 to %.loc5_47.5
// CHECK:STDOUT:   %.loc5_47.7: ref %tuple.type.5907 = struct_access %.loc5_47.1, element1
// CHECK:STDOUT:   %tuple.elem0: ref %i32 = tuple_access %.loc5_47.7, element0
// CHECK:STDOUT:   %.loc5_47.8: %i32 = bind_value %tuple.elem0
// CHECK:STDOUT:   %.loc5_47.9: ref %tuple.type.5907 = struct_access %.loc5_47.4, element1
// CHECK:STDOUT:   %.loc5_47.10: init %tuple.type.5907 = tuple_init (%.loc5_47.8) to %.loc5_47.9
// CHECK:STDOUT:   %.loc5_47.11: init %tuple.type.5907 = converted %.loc5_47.7, %.loc5_47.10
// CHECK:STDOUT:   %.loc5_47.12: init %tuple.type.5907 = initialize_from %.loc5_47.11 to %.loc5_47.9
// CHECK:STDOUT:   %.loc5_47.13: init %struct_type.b.c = struct_init (%.loc5_47.6, %.loc5_47.12) to %.loc5_47.4
// CHECK:STDOUT:   %.loc5_47.14: init %struct_type.b.c = converted %.loc5_47.1, %.loc5_47.13
// CHECK:STDOUT:   %.loc5_47.15: ref %i32 = struct_access %b_ref.ref, element1
// CHECK:STDOUT:   %.loc5_47.16: %i32 = bind_value %.loc5_47.15
// CHECK:STDOUT:   %.loc5_47.17: ref %i32 = struct_access file.%b.var, element1
// CHECK:STDOUT:   %.loc5_47.18: init %i32 = initialize_from %.loc5_47.16 to %.loc5_47.17
// CHECK:STDOUT:   %.loc5_47.19: init %struct_type.a.d.b93b = struct_init (%.loc5_47.14, %.loc5_47.18) to file.%b.var
// CHECK:STDOUT:   %.loc5_52: init %struct_type.a.d.b93b = converted %b_ref.ref, %.loc5_47.19
// CHECK:STDOUT:   assign file.%b.var, %.loc5_52
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, imports.%import_ref.f0f6 [template = constants.%F]
// CHECK:STDOUT:   %.loc6: ref %C.c372 = splice_block file.%c.var {}
// CHECK:STDOUT:   %F.call: init %C.c372 = call %F.ref() to %.loc6
// CHECK:STDOUT:   assign file.%c.var, %F.call
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%S) {
// CHECK:STDOUT:   %S => constants.%S
// CHECK:STDOUT:   %S.patt => constants.%S
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%struct) {
// CHECK:STDOUT:   %S => constants.%struct
// CHECK:STDOUT:   %S.patt => constants.%struct
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_bad_type.impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [template]
// CHECK:STDOUT:   %C.type: type = generic_class_type @C [template]
// CHECK:STDOUT:   %C.generic: %C.type = struct_value () [template]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [template]
// CHECK:STDOUT:   %complete_type.3571: <witness> = complete_type_witness %empty_struct_type [template]
// CHECK:STDOUT:   %struct_type.a.b: type = struct_type {.a: %i32, .b: %i32} [template]
// CHECK:STDOUT:   %S: %struct_type.a.b = bind_symbolic_name S, 0 [symbolic]
// CHECK:STDOUT:   %S.patt: %struct_type.a.b = symbolic_binding_pattern S, 0 [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT:   %int_2.ed97: %i32 = int_value 2 [template]
// CHECK:STDOUT:   %int_1.0593: %i32 = int_value 1 [template]
// CHECK:STDOUT:   %struct: %struct_type.a.b = struct_value (%int_1.0593, %int_2.ed97) [template]
// CHECK:STDOUT:   %C.c372: type = class_type @C, @C(%struct) [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %import_ref.9708 = import_ref Implicit//default, a_ref, unloaded
// CHECK:STDOUT:   %import_ref.dfde = import_ref Implicit//default, b_ref, unloaded
// CHECK:STDOUT:   %import_ref.c8ab: %C.type = import_ref Implicit//default, C, loaded [template = constants.%C.generic]
// CHECK:STDOUT:   %import_ref.f0f6: %F.type = import_ref Implicit//default, F, loaded [template = constants.%F]
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.8f24: <witness> = import_ref Implicit//default, loc8_34, loaded [template = constants.%complete_type.3571]
// CHECK:STDOUT:   %import_ref.5c79 = import_ref Implicit//default, inst1017 [no loc], unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .a_ref = imports.%import_ref.9708
// CHECK:STDOUT:     .b_ref = imports.%import_ref.dfde
// CHECK:STDOUT:     .C = imports.%import_ref.c8ab
// CHECK:STDOUT:     .F = imports.%import_ref.f0f6
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .c_bad = %c_bad
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Implicit.import = import Implicit
// CHECK:STDOUT:   %default.import = import <invalid>
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %c_bad.var: ref <error> = var c_bad
// CHECK:STDOUT:   %c_bad: ref <error> = bind_name c_bad, %c_bad.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @C(constants.%S: %struct_type.a.b) [from "implicit.carbon"] {
// CHECK:STDOUT:   %S: %struct_type.a.b = bind_symbolic_name S, 0 [symbolic = %S (constants.%S)]
// CHECK:STDOUT:   %S.patt: %struct_type.a.b = symbolic_binding_pattern S, 0 [symbolic = %S.patt (constants.%S.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%import_ref.5c79
// CHECK:STDOUT:     complete_type_witness = imports.%import_ref.8f24
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() -> %C.c372 [from "implicit.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, imports.%import_ref.f0f6 [template = constants.%F]
// CHECK:STDOUT:   %.loc11: ref %C.c372 = temporary_storage
// CHECK:STDOUT:   %F.call: init %C.c372 = call %F.ref() to %.loc11
// CHECK:STDOUT:   assign file.%c_bad.var, <error>
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%S) {
// CHECK:STDOUT:   %S => constants.%S
// CHECK:STDOUT:   %S.patt => constants.%S
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%struct) {
// CHECK:STDOUT:   %S => constants.%struct
// CHECK:STDOUT:   %S.patt => constants.%struct
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_bad_value.impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [template]
// CHECK:STDOUT:   %C.type: type = generic_class_type @C [template]
// CHECK:STDOUT:   %C.generic: %C.type = struct_value () [template]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [template]
// CHECK:STDOUT:   %complete_type.3571: <witness> = complete_type_witness %empty_struct_type [template]
// CHECK:STDOUT:   %struct_type.a.b.29a5: type = struct_type {.a: %i32, .b: %i32} [template]
// CHECK:STDOUT:   %S: %struct_type.a.b.29a5 = bind_symbolic_name S, 0 [symbolic]
// CHECK:STDOUT:   %S.patt: %struct_type.a.b.29a5 = symbolic_binding_pattern S, 0 [symbolic]
// CHECK:STDOUT:   %int_3.781a: %i32 = int_value 3 [template]
// CHECK:STDOUT:   %int_4.1ad1: %i32 = int_value 4 [template]
// CHECK:STDOUT:   %struct.c028: %struct_type.a.b.29a5 = struct_value (%int_3.781a, %int_4.1ad1) [template]
// CHECK:STDOUT:   %C.7293: type = class_type @C, @C(%struct.c028) [template]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT:   %int_2: %i32 = int_value 2 [template]
// CHECK:STDOUT:   %int_1: %i32 = int_value 1 [template]
// CHECK:STDOUT:   %struct.778c: %struct_type.a.b.29a5 = struct_value (%int_1, %int_2) [template]
// CHECK:STDOUT:   %C.c372: type = class_type @C, @C(%struct.778c) [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %import_ref.9708 = import_ref Implicit//default, a_ref, unloaded
// CHECK:STDOUT:   %import_ref.dfde = import_ref Implicit//default, b_ref, unloaded
// CHECK:STDOUT:   %import_ref.c8ab: %C.type = import_ref Implicit//default, C, loaded [template = constants.%C.generic]
// CHECK:STDOUT:   %import_ref.f0f6: %F.type = import_ref Implicit//default, F, loaded [template = constants.%F]
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .ImplicitAs = %import_ref.edd8
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.8f24: <witness> = import_ref Implicit//default, loc8_34, loaded [template = constants.%complete_type.3571]
// CHECK:STDOUT:   %import_ref.5c79 = import_ref Implicit//default, inst1017 [no loc], unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .a_ref = imports.%import_ref.9708
// CHECK:STDOUT:     .b_ref = imports.%import_ref.dfde
// CHECK:STDOUT:     .C = imports.%import_ref.c8ab
// CHECK:STDOUT:     .F = imports.%import_ref.f0f6
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .c_bad = %c_bad
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Implicit.import = import Implicit
// CHECK:STDOUT:   %default.import = import <invalid>
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %c_bad.var: ref %C.7293 = var c_bad
// CHECK:STDOUT:   %c_bad: ref %C.7293 = bind_name c_bad, %c_bad.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @C(constants.%S: %struct_type.a.b.29a5) [from "implicit.carbon"] {
// CHECK:STDOUT:   %S: %struct_type.a.b.29a5 = bind_symbolic_name S, 0 [symbolic = %S (constants.%S)]
// CHECK:STDOUT:   %S.patt: %struct_type.a.b.29a5 = symbolic_binding_pattern S, 0 [symbolic = %S.patt (constants.%S.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%import_ref.5c79
// CHECK:STDOUT:     complete_type_witness = imports.%import_ref.8f24
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() -> %C.c372 [from "implicit.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, imports.%import_ref.f0f6 [template = constants.%F]
// CHECK:STDOUT:   %.loc9_36: ref %C.c372 = temporary_storage
// CHECK:STDOUT:   %F.call: init %C.c372 = call %F.ref() to %.loc9_36
// CHECK:STDOUT:   %.loc9_37: %C.7293 = converted %F.call, <error> [template = <error>]
// CHECK:STDOUT:   assign file.%c_bad.var, <error>
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%S) {
// CHECK:STDOUT:   %S => constants.%S
// CHECK:STDOUT:   %S.patt => constants.%S
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%struct.c028) {
// CHECK:STDOUT:   %S => constants.%struct.c028
// CHECK:STDOUT:   %S.patt => constants.%struct.c028
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%struct.778c) {
// CHECK:STDOUT:   %S => constants.%struct.778c
// CHECK:STDOUT:   %S.patt => constants.%struct.778c
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
