// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
// TODO: Add ranges and switch to "--dump-sem-ir-ranges=only".
// EXTRA-ARGS: --dump-sem-ir-ranges=if-present
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/generic/adapt.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/generic/adapt.carbon

// --- adapt_specific_type.carbon

library "[[@TEST_NAME]]";

class C(T:! type) {
  var x: T;
}

class Adapter {
  adapt C(i32);
}

fn Access(a: Adapter) -> i32 {
  return (a as C(i32)).x;
}

// --- import_adapt_specific_type.carbon

library "[[@TEST_NAME]]";

import library "adapt_specific_type";

fn ImportedAccess(a: Adapter) -> i32 {
  return (a as C(i32)).x;
}

// --- fail_todo_extend_adapt_specific_type.carbon

library "[[@TEST_NAME]]";

class C(T:! type) {
  var x: T;
}

class Adapter {
  extend adapt C(i32);
}

fn Access(a: Adapter) -> i32 {
  // TODO: This should presumably work, but the design doesn't say how yet.
  // CHECK:STDERR: fail_todo_extend_adapt_specific_type.carbon:[[@LINE+7]]:10: error: cannot implicitly convert expression of type `Adapter` to `C(i32)` [ConversionFailure]
  // CHECK:STDERR:   return a.x;
  // CHECK:STDERR:          ^~~
  // CHECK:STDERR: fail_todo_extend_adapt_specific_type.carbon:[[@LINE+4]]:10: note: type `Adapter` does not implement interface `Core.ImplicitAs(C(i32))` [MissingImplInMemberAccessNote]
  // CHECK:STDERR:   return a.x;
  // CHECK:STDERR:          ^~~
  // CHECK:STDERR:
  return a.x;
}

// --- extend_adapt_specific_type_library.carbon

// TODO: Delete this file and change the next file to instead import the
// previous file once the previous file can be successfully type-checked.

library "[[@TEST_NAME]]";

class C(T:! type) {
  var x: T;
}

class Adapter {
  extend adapt C(i32);
}

// --- fail_todo_import_extend_adapt_specific_type.carbon

library "[[@TEST_NAME]]";

import library "extend_adapt_specific_type_library";

fn ImportedAccess(a: Adapter) -> i32 {
  // TODO: This should presumably work, but the design doesn't say how yet.
  // CHECK:STDERR: fail_todo_import_extend_adapt_specific_type.carbon:[[@LINE+7]]:10: error: cannot implicitly convert expression of type `Adapter` to `C(i32)` [ConversionFailure]
  // CHECK:STDERR:   return a.x;
  // CHECK:STDERR:          ^~~
  // CHECK:STDERR: fail_todo_import_extend_adapt_specific_type.carbon:[[@LINE+4]]:10: note: type `Adapter` does not implement interface `Core.ImplicitAs(C(i32))` [MissingImplInMemberAccessNote]
  // CHECK:STDERR:   return a.x;
  // CHECK:STDERR:          ^~~
  // CHECK:STDERR:
  return a.x;
}

// --- adapt_generic_type.carbon

library "[[@TEST_NAME]]";

class Adapter(T:! type) {
  adapt T;
}

fn Convert(a: Adapter(i32)) -> i32 {
  return a as i32;
}

// --- import_adapt_generic_type.carbon

library "[[@TEST_NAME]]";

import library "adapt_generic_type";

fn ImportedConvert(a: Adapter(i32)) -> i32 {
  return a as i32;
}

class C {
  var n: i32;
}

fn ImportedConvertLocal(a: Adapter(C)) -> i32 {
  return (a as C).n;
}

// CHECK:STDOUT: --- adapt_specific_type.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = bind_symbolic_name .Self [symbolic_self]
// CHECK:STDOUT:   %T.8b3: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %C.type: type = generic_class_type @C [concrete]
// CHECK:STDOUT:   %C.generic: %C.type = struct_value () [concrete]
// CHECK:STDOUT:   %C.f2e: type = class_type @C, @C(%T.8b3) [symbolic]
// CHECK:STDOUT:   %require_complete.4ae: <witness> = require_complete_type %T.8b3 [symbolic]
// CHECK:STDOUT:   %C.elem.66c: type = unbound_element_type %C.f2e, %T.8b3 [symbolic]
// CHECK:STDOUT:   %struct_type.x.2ac: type = struct_type {.x: %T.8b3} [symbolic]
// CHECK:STDOUT:   %complete_type.433: <witness> = complete_type_witness %struct_type.x.2ac [symbolic]
// CHECK:STDOUT:   %Adapter: type = class_type @Adapter [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %C.98a: type = class_type @C, @C(%i32) [concrete]
// CHECK:STDOUT:   %i32.builtin: type = int_type signed, %int_32 [concrete]
// CHECK:STDOUT:   %complete_type.f8a: <witness> = complete_type_witness %i32.builtin [concrete]
// CHECK:STDOUT:   %C.elem.476: type = unbound_element_type %C.98a, %i32 [concrete]
// CHECK:STDOUT:   %struct_type.x.ed6: type = struct_type {.x: %i32} [concrete]
// CHECK:STDOUT:   %complete_type.1ec: <witness> = complete_type_witness %struct_type.x.ed6 [concrete]
// CHECK:STDOUT:   %pattern_type.27e: type = pattern_type %Adapter [concrete]
// CHECK:STDOUT:   %pattern_type.7ce: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %Access.type: type = fn_type @Access [concrete]
// CHECK:STDOUT:   %Access: %Access.type = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Copy.Op.type: type = fn_type @Copy.Op [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.afd: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%N) [symbolic]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.6cd: %Int.as.Copy.impl.Op.type.afd = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.a32: <witness> = impl_witness imports.%Copy.impl_witness_table.1ed, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.276: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.f59: %Int.as.Copy.impl.Op.type.276 = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %i32, (%Copy.impl_witness.a32) [concrete]
// CHECK:STDOUT:   %.7fa: type = fn_type_with_self_type %Copy.Op.type, %Copy.facet [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %Int.as.Copy.impl.Op.f59, @Int.as.Copy.impl.Op(%int_32) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .Copy = %Core.Copy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/parts/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT:   %Core.Copy: type = import_ref Core//prelude/parts/copy, Copy, loaded [concrete = constants.%Copy.type]
// CHECK:STDOUT:   %Core.import_ref.d0f6: @Int.as.Copy.impl.%Int.as.Copy.impl.Op.type (%Int.as.Copy.impl.Op.type.afd) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.Copy.impl.%Int.as.Copy.impl.Op (constants.%Int.as.Copy.impl.Op.6cd)]
// CHECK:STDOUT:   %Copy.impl_witness_table.1ed = impl_witness_table (%Core.import_ref.d0f6), @Int.as.Copy.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .Adapter = %Adapter.decl
// CHECK:STDOUT:     .Access = %Access.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %C.decl: %C.type = class_decl @C [concrete = constants.%C.generic] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.Self: %type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %T.loc4_9.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc4_9.1 (constants.%T.8b3)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Adapter.decl: type = class_decl @Adapter [concrete = constants.%Adapter] {} {}
// CHECK:STDOUT:   %Access.decl: %Access.type = fn_decl @Access [concrete = constants.%Access] {
// CHECK:STDOUT:     %a.patt: %pattern_type.27e = value_binding_pattern a [concrete]
// CHECK:STDOUT:     %a.param_patt: %pattern_type.27e = value_param_pattern %a.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.7ce = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.7ce = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32.loc12: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc12: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %a.param: %Adapter = value_param call_param0
// CHECK:STDOUT:     %Adapter.ref: type = name_ref Adapter, file.%Adapter.decl [concrete = constants.%Adapter]
// CHECK:STDOUT:     %a: %Adapter = value_binding a, %a.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @C(%T.loc4_9.2: type) {
// CHECK:STDOUT:   %T.loc4_9.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc4_9.1 (constants.%T.8b3)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T.loc4_9.1 [symbolic = %require_complete (constants.%require_complete.4ae)]
// CHECK:STDOUT:   %C: type = class_type @C, @C(%T.loc4_9.1) [symbolic = %C (constants.%C.f2e)]
// CHECK:STDOUT:   %C.elem: type = unbound_element_type %C, %T.loc4_9.1 [symbolic = %C.elem (constants.%C.elem.66c)]
// CHECK:STDOUT:   %struct_type.x: type = struct_type {.x: @C.%T.loc4_9.1 (%T.8b3)} [symbolic = %struct_type.x (constants.%struct_type.x.2ac)]
// CHECK:STDOUT:   %complete_type.loc6_1.2: <witness> = complete_type_witness %struct_type.x [symbolic = %complete_type.loc6_1.2 (constants.%complete_type.433)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc4_9.2 [symbolic = %T.loc4_9.1 (constants.%T.8b3)]
// CHECK:STDOUT:     %.loc5: @C.%C.elem (%C.elem.66c) = field_decl x, element0 [concrete]
// CHECK:STDOUT:     %complete_type.loc6_1.1: <witness> = complete_type_witness constants.%struct_type.x.2ac [symbolic = %complete_type.loc6_1.2 (constants.%complete_type.433)]
// CHECK:STDOUT:     complete_type_witness = %complete_type.loc6_1.1
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%C.f2e
// CHECK:STDOUT:     .T = <poisoned>
// CHECK:STDOUT:     .x = %.loc5
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Adapter {
// CHECK:STDOUT:   %C.ref: %C.type = name_ref C, file.%C.decl [concrete = constants.%C.generic]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   %C: type = class_type @C, @C(constants.%i32) [concrete = constants.%C.98a]
// CHECK:STDOUT:   adapt_decl %C [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%struct_type.x.ed6 [concrete = constants.%complete_type.1ec]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Adapter
// CHECK:STDOUT:   .C = <poisoned>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Access(%a.param: %Adapter) -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: %Adapter = name_ref a, %a
// CHECK:STDOUT:   %C.ref: %C.type = name_ref C, file.%C.decl [concrete = constants.%C.generic]
// CHECK:STDOUT:   %int_32.loc13: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32.loc13: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   %C: type = class_type @C, @C(constants.%i32) [concrete = constants.%C.98a]
// CHECK:STDOUT:   %.loc13_13.1: %C.98a = as_compatible %a.ref
// CHECK:STDOUT:   %.loc13_13.2: %C.98a = converted %a.ref, %.loc13_13.1
// CHECK:STDOUT:   %x.ref: %C.elem.476 = name_ref x, @C.%.loc5 [concrete = @C.%.loc5]
// CHECK:STDOUT:   %.loc13_23.1: ref %i32 = class_element_access %.loc13_13.2, element0
// CHECK:STDOUT:   %.loc13_23.2: %i32 = bind_value %.loc13_23.1
// CHECK:STDOUT:   %impl.elem0: %.7fa = impl_witness_access constants.%Copy.impl_witness.a32, element0 [concrete = constants.%Int.as.Copy.impl.Op.f59]
// CHECK:STDOUT:   %bound_method.loc13_23.1: <bound method> = bound_method %.loc13_23.2, %impl.elem0
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc13_23.2: <bound method> = bound_method %.loc13_23.2, %specific_fn
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call: init %i32 = call %bound_method.loc13_23.2(%.loc13_23.2)
// CHECK:STDOUT:   return %Int.as.Copy.impl.Op.call to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%T.8b3) {
// CHECK:STDOUT:   %T.loc4_9.1 => constants.%T.8b3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%i32) {
// CHECK:STDOUT:   %T.loc4_9.1 => constants.%i32
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.f8a
// CHECK:STDOUT:   %C => constants.%C.98a
// CHECK:STDOUT:   %C.elem => constants.%C.elem.476
// CHECK:STDOUT:   %struct_type.x => constants.%struct_type.x.ed6
// CHECK:STDOUT:   %complete_type.loc6_1.2 => constants.%complete_type.1ec
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_adapt_specific_type.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Adapter: type = class_type @Adapter [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %C.type: type = generic_class_type @C [concrete]
// CHECK:STDOUT:   %C.generic: %C.type = struct_value () [concrete]
// CHECK:STDOUT:   %T.8b3: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %struct_type.x.2ac: type = struct_type {.x: %T.8b3} [symbolic]
// CHECK:STDOUT:   %complete_type.433: <witness> = complete_type_witness %struct_type.x.2ac [symbolic]
// CHECK:STDOUT:   %C.f2e: type = class_type @C, @C(%T.8b3) [symbolic]
// CHECK:STDOUT:   %C.239: type = class_type @C, @C(%i32) [concrete]
// CHECK:STDOUT:   %struct_type.x.767: type = struct_type {.x: %i32} [concrete]
// CHECK:STDOUT:   %complete_type.c07: <witness> = complete_type_witness %struct_type.x.767 [concrete]
// CHECK:STDOUT:   %require_complete.4ae: <witness> = require_complete_type %T.8b3 [symbolic]
// CHECK:STDOUT:   %C.elem.66c: type = unbound_element_type %C.f2e, %T.8b3 [symbolic]
// CHECK:STDOUT:   %i32.builtin: type = int_type signed, %int_32 [concrete]
// CHECK:STDOUT:   %complete_type.f8a: <witness> = complete_type_witness %i32.builtin [concrete]
// CHECK:STDOUT:   %C.elem.ed6: type = unbound_element_type %C.239, %i32 [concrete]
// CHECK:STDOUT:   %pattern_type.27e: type = pattern_type %Adapter [concrete]
// CHECK:STDOUT:   %pattern_type.501: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %ImportedAccess.type: type = fn_type @ImportedAccess [concrete]
// CHECK:STDOUT:   %ImportedAccess: %ImportedAccess.type = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Copy.Op.type: type = fn_type @Copy.Op [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.ea1: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%N) [symbolic]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.d6b: %Int.as.Copy.impl.Op.type.ea1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.d0b: <witness> = impl_witness imports.%Copy.impl_witness_table.608, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.edd: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.99d: %Int.as.Copy.impl.Op.type.edd = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %i32, (%Copy.impl_witness.d0b) [concrete]
// CHECK:STDOUT:   %.0ec: type = fn_type_with_self_type %Copy.Op.type, %Copy.facet [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %Int.as.Copy.impl.Op.99d, @Int.as.Copy.impl.Op(%int_32) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.C: %C.type = import_ref Main//adapt_specific_type, C, loaded [concrete = constants.%C.generic]
// CHECK:STDOUT:   %Main.Adapter: type = import_ref Main//adapt_specific_type, Adapter, loaded [concrete = constants.%Adapter]
// CHECK:STDOUT:   %Main.Access = import_ref Main//adapt_specific_type, Access, unloaded
// CHECK:STDOUT:   %Core.ece: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .Copy = %Core.Copy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Main.import_ref.5ab: type = import_ref Main//adapt_specific_type, loc4_9, loaded [symbolic = @C.%T (constants.%T.8b3)]
// CHECK:STDOUT:   %Main.import_ref.b5f: <witness> = import_ref Main//adapt_specific_type, loc6_1, loaded [symbolic = @C.%complete_type (constants.%complete_type.433)]
// CHECK:STDOUT:   %Main.import_ref.4c0 = import_ref Main//adapt_specific_type, inst{{[0-9A-F]+}} [no loc], unloaded
// CHECK:STDOUT:   %Main.import_ref.262: @C.%C.elem (%C.elem.66c) = import_ref Main//adapt_specific_type, loc5_8, loaded [concrete = %.22b]
// CHECK:STDOUT:   %Main.import_ref.709: <witness> = import_ref Main//adapt_specific_type, loc10_1, loaded [concrete = constants.%complete_type.c07]
// CHECK:STDOUT:   %Main.import_ref.feb = import_ref Main//adapt_specific_type, inst{{[0-9A-F]+}} [no loc], unloaded
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/parts/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT:   %.22b: @C.%C.elem (%C.elem.66c) = field_decl x, element0 [concrete]
// CHECK:STDOUT:   %Core.Copy: type = import_ref Core//prelude/parts/copy, Copy, loaded [concrete = constants.%Copy.type]
// CHECK:STDOUT:   %Core.import_ref.d8e: @Int.as.Copy.impl.%Int.as.Copy.impl.Op.type (%Int.as.Copy.impl.Op.type.ea1) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.Copy.impl.%Int.as.Copy.impl.Op (constants.%Int.as.Copy.impl.Op.d6b)]
// CHECK:STDOUT:   %Copy.impl_witness_table.608 = impl_witness_table (%Core.import_ref.d8e), @Int.as.Copy.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .C = imports.%Main.C
// CHECK:STDOUT:     .Adapter = imports.%Main.Adapter
// CHECK:STDOUT:     .Access = imports.%Main.Access
// CHECK:STDOUT:     .Core = imports.%Core.ece
// CHECK:STDOUT:     .ImportedAccess = %ImportedAccess.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %default.import = import <none>
// CHECK:STDOUT:   %ImportedAccess.decl: %ImportedAccess.type = fn_decl @ImportedAccess [concrete = constants.%ImportedAccess] {
// CHECK:STDOUT:     %a.patt: %pattern_type.27e = value_binding_pattern a [concrete]
// CHECK:STDOUT:     %a.param_patt: %pattern_type.27e = value_param_pattern %a.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.501 = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.501 = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32.loc6: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc6: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %a.param: %Adapter = value_param call_param0
// CHECK:STDOUT:     %Adapter.ref: type = name_ref Adapter, imports.%Main.Adapter [concrete = constants.%Adapter]
// CHECK:STDOUT:     %a: %Adapter = value_binding a, %a.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Adapter [from "adapt_specific_type.carbon"] {
// CHECK:STDOUT:   complete_type_witness = imports.%Main.import_ref.709
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.feb
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @C(imports.%Main.import_ref.5ab: type) [from "adapt_specific_type.carbon"] {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T.8b3)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T [symbolic = %require_complete (constants.%require_complete.4ae)]
// CHECK:STDOUT:   %C: type = class_type @C, @C(%T) [symbolic = %C (constants.%C.f2e)]
// CHECK:STDOUT:   %C.elem: type = unbound_element_type %C, %T [symbolic = %C.elem (constants.%C.elem.66c)]
// CHECK:STDOUT:   %struct_type.x: type = struct_type {.x: @C.%T (%T.8b3)} [symbolic = %struct_type.x (constants.%struct_type.x.2ac)]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.x [symbolic = %complete_type (constants.%complete_type.433)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     complete_type_witness = imports.%Main.import_ref.b5f
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%Main.import_ref.4c0
// CHECK:STDOUT:     .x = imports.%Main.import_ref.262
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ImportedAccess(%a.param: %Adapter) -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: %Adapter = name_ref a, %a
// CHECK:STDOUT:   %C.ref: %C.type = name_ref C, imports.%Main.C [concrete = constants.%C.generic]
// CHECK:STDOUT:   %int_32.loc7: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32.loc7: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   %C: type = class_type @C, @C(constants.%i32) [concrete = constants.%C.239]
// CHECK:STDOUT:   %.loc7_13.1: %C.239 = as_compatible %a.ref
// CHECK:STDOUT:   %.loc7_13.2: %C.239 = converted %a.ref, %.loc7_13.1
// CHECK:STDOUT:   %x.ref: %C.elem.ed6 = name_ref x, imports.%Main.import_ref.262 [concrete = imports.%.22b]
// CHECK:STDOUT:   %.loc7_23.1: ref %i32 = class_element_access %.loc7_13.2, element0
// CHECK:STDOUT:   %.loc7_23.2: %i32 = bind_value %.loc7_23.1
// CHECK:STDOUT:   %impl.elem0: %.0ec = impl_witness_access constants.%Copy.impl_witness.d0b, element0 [concrete = constants.%Int.as.Copy.impl.Op.99d]
// CHECK:STDOUT:   %bound_method.loc7_23.1: <bound method> = bound_method %.loc7_23.2, %impl.elem0
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc7_23.2: <bound method> = bound_method %.loc7_23.2, %specific_fn
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call: init %i32 = call %bound_method.loc7_23.2(%.loc7_23.2)
// CHECK:STDOUT:   return %Int.as.Copy.impl.Op.call to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%T.8b3) {
// CHECK:STDOUT:   %T => constants.%T.8b3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%i32) {
// CHECK:STDOUT:   %T => constants.%i32
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.f8a
// CHECK:STDOUT:   %C => constants.%C.239
// CHECK:STDOUT:   %C.elem => constants.%C.elem.ed6
// CHECK:STDOUT:   %struct_type.x => constants.%struct_type.x.767
// CHECK:STDOUT:   %complete_type => constants.%complete_type.c07
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_todo_extend_adapt_specific_type.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = bind_symbolic_name .Self [symbolic_self]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %C.type: type = generic_class_type @C [concrete]
// CHECK:STDOUT:   %C.generic: %C.type = struct_value () [concrete]
// CHECK:STDOUT:   %C.f2e: type = class_type @C, @C(%T) [symbolic]
// CHECK:STDOUT:   %require_complete.4ae: <witness> = require_complete_type %T [symbolic]
// CHECK:STDOUT:   %C.elem.66c: type = unbound_element_type %C.f2e, %T [symbolic]
// CHECK:STDOUT:   %struct_type.x.2ac: type = struct_type {.x: %T} [symbolic]
// CHECK:STDOUT:   %complete_type.433: <witness> = complete_type_witness %struct_type.x.2ac [symbolic]
// CHECK:STDOUT:   %Adapter: type = class_type @Adapter [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %C.98a: type = class_type @C, @C(%i32) [concrete]
// CHECK:STDOUT:   %i32.builtin: type = int_type signed, %int_32 [concrete]
// CHECK:STDOUT:   %complete_type.f8a: <witness> = complete_type_witness %i32.builtin [concrete]
// CHECK:STDOUT:   %C.elem.476: type = unbound_element_type %C.98a, %i32 [concrete]
// CHECK:STDOUT:   %struct_type.x.ed6: type = struct_type {.x: %i32} [concrete]
// CHECK:STDOUT:   %complete_type.1ec: <witness> = complete_type_witness %struct_type.x.ed6 [concrete]
// CHECK:STDOUT:   %pattern_type.27e: type = pattern_type %Adapter [concrete]
// CHECK:STDOUT:   %pattern_type.7ce: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %Access.type: type = fn_type @Access [concrete]
// CHECK:STDOUT:   %Access: %Access.type = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.cc7: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.cc7 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/parts/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.cc7 = import_ref Core//prelude/parts/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .Adapter = %Adapter.decl
// CHECK:STDOUT:     .Access = %Access.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %C.decl: %C.type = class_decl @C [concrete = constants.%C.generic] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.Self: %type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %T.loc4_9.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc4_9.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Adapter.decl: type = class_decl @Adapter [concrete = constants.%Adapter] {} {}
// CHECK:STDOUT:   %Access.decl: %Access.type = fn_decl @Access [concrete = constants.%Access] {
// CHECK:STDOUT:     %a.patt: %pattern_type.27e = value_binding_pattern a [concrete]
// CHECK:STDOUT:     %a.param_patt: %pattern_type.27e = value_param_pattern %a.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.7ce = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.7ce = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %a.param: %Adapter = value_param call_param0
// CHECK:STDOUT:     %Adapter.ref: type = name_ref Adapter, file.%Adapter.decl [concrete = constants.%Adapter]
// CHECK:STDOUT:     %a: %Adapter = value_binding a, %a.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @C(%T.loc4_9.2: type) {
// CHECK:STDOUT:   %T.loc4_9.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc4_9.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T.loc4_9.1 [symbolic = %require_complete (constants.%require_complete.4ae)]
// CHECK:STDOUT:   %C: type = class_type @C, @C(%T.loc4_9.1) [symbolic = %C (constants.%C.f2e)]
// CHECK:STDOUT:   %C.elem: type = unbound_element_type %C, %T.loc4_9.1 [symbolic = %C.elem (constants.%C.elem.66c)]
// CHECK:STDOUT:   %struct_type.x: type = struct_type {.x: @C.%T.loc4_9.1 (%T)} [symbolic = %struct_type.x (constants.%struct_type.x.2ac)]
// CHECK:STDOUT:   %complete_type.loc6_1.2: <witness> = complete_type_witness %struct_type.x [symbolic = %complete_type.loc6_1.2 (constants.%complete_type.433)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc4_9.2 [symbolic = %T.loc4_9.1 (constants.%T)]
// CHECK:STDOUT:     %.loc5: @C.%C.elem (%C.elem.66c) = field_decl x, element0 [concrete]
// CHECK:STDOUT:     %complete_type.loc6_1.1: <witness> = complete_type_witness constants.%struct_type.x.2ac [symbolic = %complete_type.loc6_1.2 (constants.%complete_type.433)]
// CHECK:STDOUT:     complete_type_witness = %complete_type.loc6_1.1
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%C.f2e
// CHECK:STDOUT:     .T = <poisoned>
// CHECK:STDOUT:     .x = %.loc5
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Adapter {
// CHECK:STDOUT:   %C.ref: %C.type = name_ref C, file.%C.decl [concrete = constants.%C.generic]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   %C: type = class_type @C, @C(constants.%i32) [concrete = constants.%C.98a]
// CHECK:STDOUT:   adapt_decl %C [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%struct_type.x.ed6 [concrete = constants.%complete_type.1ec]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Adapter
// CHECK:STDOUT:   .C = <poisoned>
// CHECK:STDOUT:   .x = <poisoned>
// CHECK:STDOUT:   extend %C
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Access(%a.param: %Adapter) -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: %Adapter = name_ref a, %a
// CHECK:STDOUT:   %x.ref: %C.elem.476 = name_ref x, @C.%.loc5 [concrete = @C.%.loc5]
// CHECK:STDOUT:   %.loc21_11.1: %C.98a = converted %a.ref, <error> [concrete = <error>]
// CHECK:STDOUT:   %.loc21_11.2: %i32 = class_element_access <error>, element0 [concrete = <error>]
// CHECK:STDOUT:   return <error> to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%T) {
// CHECK:STDOUT:   %T.loc4_9.1 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%i32) {
// CHECK:STDOUT:   %T.loc4_9.1 => constants.%i32
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.f8a
// CHECK:STDOUT:   %C => constants.%C.98a
// CHECK:STDOUT:   %C.elem => constants.%C.elem.476
// CHECK:STDOUT:   %struct_type.x => constants.%struct_type.x.ed6
// CHECK:STDOUT:   %complete_type.loc6_1.2 => constants.%complete_type.1ec
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- extend_adapt_specific_type_library.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = bind_symbolic_name .Self [symbolic_self]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %C.type: type = generic_class_type @C [concrete]
// CHECK:STDOUT:   %C.generic: %C.type = struct_value () [concrete]
// CHECK:STDOUT:   %C.f2e: type = class_type @C, @C(%T) [symbolic]
// CHECK:STDOUT:   %require_complete.4ae: <witness> = require_complete_type %T [symbolic]
// CHECK:STDOUT:   %C.elem.66c: type = unbound_element_type %C.f2e, %T [symbolic]
// CHECK:STDOUT:   %struct_type.x.2ac: type = struct_type {.x: %T} [symbolic]
// CHECK:STDOUT:   %complete_type.433: <witness> = complete_type_witness %struct_type.x.2ac [symbolic]
// CHECK:STDOUT:   %Adapter: type = class_type @Adapter [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %C.98a: type = class_type @C, @C(%i32) [concrete]
// CHECK:STDOUT:   %i32.builtin: type = int_type signed, %int_32 [concrete]
// CHECK:STDOUT:   %complete_type.f8a: <witness> = complete_type_witness %i32.builtin [concrete]
// CHECK:STDOUT:   %C.elem.476: type = unbound_element_type %C.98a, %i32 [concrete]
// CHECK:STDOUT:   %struct_type.x.ed6: type = struct_type {.x: %i32} [concrete]
// CHECK:STDOUT:   %complete_type.1ec: <witness> = complete_type_witness %struct_type.x.ed6 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/parts/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .Adapter = %Adapter.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %C.decl: %C.type = class_decl @C [concrete = constants.%C.generic] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.Self: %type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %T.loc7_9.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc7_9.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Adapter.decl: type = class_decl @Adapter [concrete = constants.%Adapter] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @C(%T.loc7_9.2: type) {
// CHECK:STDOUT:   %T.loc7_9.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc7_9.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T.loc7_9.1 [symbolic = %require_complete (constants.%require_complete.4ae)]
// CHECK:STDOUT:   %C: type = class_type @C, @C(%T.loc7_9.1) [symbolic = %C (constants.%C.f2e)]
// CHECK:STDOUT:   %C.elem: type = unbound_element_type %C, %T.loc7_9.1 [symbolic = %C.elem (constants.%C.elem.66c)]
// CHECK:STDOUT:   %struct_type.x: type = struct_type {.x: @C.%T.loc7_9.1 (%T)} [symbolic = %struct_type.x (constants.%struct_type.x.2ac)]
// CHECK:STDOUT:   %complete_type.loc9_1.2: <witness> = complete_type_witness %struct_type.x [symbolic = %complete_type.loc9_1.2 (constants.%complete_type.433)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc7_9.2 [symbolic = %T.loc7_9.1 (constants.%T)]
// CHECK:STDOUT:     %.loc8: @C.%C.elem (%C.elem.66c) = field_decl x, element0 [concrete]
// CHECK:STDOUT:     %complete_type.loc9_1.1: <witness> = complete_type_witness constants.%struct_type.x.2ac [symbolic = %complete_type.loc9_1.2 (constants.%complete_type.433)]
// CHECK:STDOUT:     complete_type_witness = %complete_type.loc9_1.1
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%C.f2e
// CHECK:STDOUT:     .T = <poisoned>
// CHECK:STDOUT:     .x = %.loc8
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Adapter {
// CHECK:STDOUT:   %C.ref: %C.type = name_ref C, file.%C.decl [concrete = constants.%C.generic]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   %C: type = class_type @C, @C(constants.%i32) [concrete = constants.%C.98a]
// CHECK:STDOUT:   adapt_decl %C [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%struct_type.x.ed6 [concrete = constants.%complete_type.1ec]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Adapter
// CHECK:STDOUT:   .C = <poisoned>
// CHECK:STDOUT:   extend %C
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%T) {
// CHECK:STDOUT:   %T.loc7_9.1 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%i32) {
// CHECK:STDOUT:   %T.loc7_9.1 => constants.%i32
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.f8a
// CHECK:STDOUT:   %C => constants.%C.98a
// CHECK:STDOUT:   %C.elem => constants.%C.elem.476
// CHECK:STDOUT:   %struct_type.x => constants.%struct_type.x.ed6
// CHECK:STDOUT:   %complete_type.loc9_1.2 => constants.%complete_type.1ec
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_todo_import_extend_adapt_specific_type.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Adapter: type = class_type @Adapter [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %struct_type.x.2ac: type = struct_type {.x: %T} [symbolic]
// CHECK:STDOUT:   %complete_type.433: <witness> = complete_type_witness %struct_type.x.2ac [symbolic]
// CHECK:STDOUT:   %C.f2e: type = class_type @C, @C(%T) [symbolic]
// CHECK:STDOUT:   %C.239: type = class_type @C, @C(%i32) [concrete]
// CHECK:STDOUT:   %struct_type.x.767: type = struct_type {.x: %i32} [concrete]
// CHECK:STDOUT:   %complete_type.c07: <witness> = complete_type_witness %struct_type.x.767 [concrete]
// CHECK:STDOUT:   %require_complete.4ae: <witness> = require_complete_type %T [symbolic]
// CHECK:STDOUT:   %C.elem.66c: type = unbound_element_type %C.f2e, %T [symbolic]
// CHECK:STDOUT:   %i32.builtin: type = int_type signed, %int_32 [concrete]
// CHECK:STDOUT:   %complete_type.f8a: <witness> = complete_type_witness %i32.builtin [concrete]
// CHECK:STDOUT:   %C.elem.ed6: type = unbound_element_type %C.239, %i32 [concrete]
// CHECK:STDOUT:   %pattern_type.27e: type = pattern_type %Adapter [concrete]
// CHECK:STDOUT:   %pattern_type.501: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %ImportedAccess.type: type = fn_type @ImportedAccess [concrete]
// CHECK:STDOUT:   %ImportedAccess: %ImportedAccess.type = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.cc7: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.cc7 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.C = import_ref Main//extend_adapt_specific_type_library, C, unloaded
// CHECK:STDOUT:   %Main.Adapter: type = import_ref Main//extend_adapt_specific_type_library, Adapter, loaded [concrete = constants.%Adapter]
// CHECK:STDOUT:   %Core.ece: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Main.import_ref.5ab: type = import_ref Main//extend_adapt_specific_type_library, loc7_9, loaded [symbolic = @C.%T (constants.%T)]
// CHECK:STDOUT:   %Main.import_ref.b5f: <witness> = import_ref Main//extend_adapt_specific_type_library, loc9_1, loaded [symbolic = @C.%complete_type (constants.%complete_type.433)]
// CHECK:STDOUT:   %Main.import_ref.4c0 = import_ref Main//extend_adapt_specific_type_library, inst{{[0-9A-F]+}} [no loc], unloaded
// CHECK:STDOUT:   %Main.import_ref.262: @C.%C.elem (%C.elem.66c) = import_ref Main//extend_adapt_specific_type_library, loc8_8, loaded [concrete = %.22b]
// CHECK:STDOUT:   %Main.import_ref.709: <witness> = import_ref Main//extend_adapt_specific_type_library, loc13_1, loaded [concrete = constants.%complete_type.c07]
// CHECK:STDOUT:   %Main.import_ref.feb = import_ref Main//extend_adapt_specific_type_library, inst{{[0-9A-F]+}} [no loc], unloaded
// CHECK:STDOUT:   %Main.import_ref.19d12e.2: type = import_ref Main//extend_adapt_specific_type_library, loc12_21, loaded [concrete = constants.%C.239]
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/parts/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT:   %.22b: @C.%C.elem (%C.elem.66c) = field_decl x, element0 [concrete]
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.cc7 = import_ref Core//prelude/parts/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .C = imports.%Main.C
// CHECK:STDOUT:     .Adapter = imports.%Main.Adapter
// CHECK:STDOUT:     .Core = imports.%Core.ece
// CHECK:STDOUT:     .ImportedAccess = %ImportedAccess.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %default.import = import <none>
// CHECK:STDOUT:   %ImportedAccess.decl: %ImportedAccess.type = fn_decl @ImportedAccess [concrete = constants.%ImportedAccess] {
// CHECK:STDOUT:     %a.patt: %pattern_type.27e = value_binding_pattern a [concrete]
// CHECK:STDOUT:     %a.param_patt: %pattern_type.27e = value_param_pattern %a.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.501 = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.501 = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %a.param: %Adapter = value_param call_param0
// CHECK:STDOUT:     %Adapter.ref: type = name_ref Adapter, imports.%Main.Adapter [concrete = constants.%Adapter]
// CHECK:STDOUT:     %a: %Adapter = value_binding a, %a.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Adapter [from "extend_adapt_specific_type_library.carbon"] {
// CHECK:STDOUT:   complete_type_witness = imports.%Main.import_ref.709
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.feb
// CHECK:STDOUT:   .x = <poisoned>
// CHECK:STDOUT:   extend imports.%Main.import_ref.19d12e.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @C(imports.%Main.import_ref.5ab: type) [from "extend_adapt_specific_type_library.carbon"] {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T [symbolic = %require_complete (constants.%require_complete.4ae)]
// CHECK:STDOUT:   %C: type = class_type @C, @C(%T) [symbolic = %C (constants.%C.f2e)]
// CHECK:STDOUT:   %C.elem: type = unbound_element_type %C, %T [symbolic = %C.elem (constants.%C.elem.66c)]
// CHECK:STDOUT:   %struct_type.x: type = struct_type {.x: @C.%T (%T)} [symbolic = %struct_type.x (constants.%struct_type.x.2ac)]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.x [symbolic = %complete_type (constants.%complete_type.433)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     complete_type_witness = imports.%Main.import_ref.b5f
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%Main.import_ref.4c0
// CHECK:STDOUT:     .x = imports.%Main.import_ref.262
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ImportedAccess(%a.param: %Adapter) -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: %Adapter = name_ref a, %a
// CHECK:STDOUT:   %x.ref: %C.elem.ed6 = name_ref x, imports.%Main.import_ref.262 [concrete = imports.%.22b]
// CHECK:STDOUT:   %.loc15_11.1: %C.239 = converted %a.ref, <error> [concrete = <error>]
// CHECK:STDOUT:   %.loc15_11.2: %i32 = class_element_access <error>, element0 [concrete = <error>]
// CHECK:STDOUT:   return <error> to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%i32) {
// CHECK:STDOUT:   %T => constants.%i32
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.f8a
// CHECK:STDOUT:   %C => constants.%C.239
// CHECK:STDOUT:   %C.elem => constants.%C.elem.ed6
// CHECK:STDOUT:   %struct_type.x => constants.%struct_type.x.767
// CHECK:STDOUT:   %complete_type => constants.%complete_type.c07
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- adapt_generic_type.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = bind_symbolic_name .Self [symbolic_self]
// CHECK:STDOUT:   %T.8b3: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %Adapter.type: type = generic_class_type @Adapter [concrete]
// CHECK:STDOUT:   %Adapter.generic: %Adapter.type = struct_value () [concrete]
// CHECK:STDOUT:   %Adapter.0e3: type = class_type @Adapter, @Adapter(%T.8b3) [symbolic]
// CHECK:STDOUT:   %require_complete.4ae: <witness> = require_complete_type %T.8b3 [symbolic]
// CHECK:STDOUT:   %complete_type.f87: <witness> = complete_type_witness %T.8b3 [symbolic]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %Adapter.e4c: type = class_type @Adapter, @Adapter(%i32) [concrete]
// CHECK:STDOUT:   %pattern_type.5a0: type = pattern_type %Adapter.e4c [concrete]
// CHECK:STDOUT:   %pattern_type.7ce: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %Convert.type: type = fn_type @Convert [concrete]
// CHECK:STDOUT:   %Convert: %Convert.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32.builtin: type = int_type signed, %int_32 [concrete]
// CHECK:STDOUT:   %complete_type.f8a: <witness> = complete_type_witness %i32.builtin [concrete]
// CHECK:STDOUT:   %complete_type.1eb: <witness> = complete_type_witness %i32 [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Copy.Op.type: type = fn_type @Copy.Op [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.afd: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%N) [symbolic]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.6cd: %Int.as.Copy.impl.Op.type.afd = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.a32: <witness> = impl_witness imports.%Copy.impl_witness_table.1ed, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.276: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.f59: %Int.as.Copy.impl.Op.type.276 = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %i32, (%Copy.impl_witness.a32) [concrete]
// CHECK:STDOUT:   %.7fa: type = fn_type_with_self_type %Copy.Op.type, %Copy.facet [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %Int.as.Copy.impl.Op.f59, @Int.as.Copy.impl.Op(%int_32) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .Copy = %Core.Copy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/parts/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT:   %Core.Copy: type = import_ref Core//prelude/parts/copy, Copy, loaded [concrete = constants.%Copy.type]
// CHECK:STDOUT:   %Core.import_ref.d0f6: @Int.as.Copy.impl.%Int.as.Copy.impl.Op.type (%Int.as.Copy.impl.Op.type.afd) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.Copy.impl.%Int.as.Copy.impl.Op (constants.%Int.as.Copy.impl.Op.6cd)]
// CHECK:STDOUT:   %Copy.impl_witness_table.1ed = impl_witness_table (%Core.import_ref.d0f6), @Int.as.Copy.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Adapter = %Adapter.decl
// CHECK:STDOUT:     .Convert = %Convert.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Adapter.decl: %Adapter.type = class_decl @Adapter [concrete = constants.%Adapter.generic] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.Self: %type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %T.loc4_15.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc4_15.1 (constants.%T.8b3)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Convert.decl: %Convert.type = fn_decl @Convert [concrete = constants.%Convert] {
// CHECK:STDOUT:     %a.patt: %pattern_type.5a0 = value_binding_pattern a [concrete]
// CHECK:STDOUT:     %a.param_patt: %pattern_type.5a0 = value_param_pattern %a.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.7ce = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.7ce = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32.loc8_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc8_32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %a.param: %Adapter.e4c = value_param call_param0
// CHECK:STDOUT:     %.loc8: type = splice_block %Adapter [concrete = constants.%Adapter.e4c] {
// CHECK:STDOUT:       %Adapter.ref: %Adapter.type = name_ref Adapter, file.%Adapter.decl [concrete = constants.%Adapter.generic]
// CHECK:STDOUT:       %int_32.loc8_23: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:       %i32.loc8_23: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:       %Adapter: type = class_type @Adapter, @Adapter(constants.%i32) [concrete = constants.%Adapter.e4c]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a: %Adapter.e4c = value_binding a, %a.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @Adapter(%T.loc4_15.2: type) {
// CHECK:STDOUT:   %T.loc4_15.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc4_15.1 (constants.%T.8b3)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T.loc4_15.1 [symbolic = %require_complete (constants.%require_complete.4ae)]
// CHECK:STDOUT:   %complete_type.loc6_1.2: <witness> = complete_type_witness %T.loc4_15.1 [symbolic = %complete_type.loc6_1.2 (constants.%complete_type.f87)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc4_15.2 [symbolic = %T.loc4_15.1 (constants.%T.8b3)]
// CHECK:STDOUT:     adapt_decl %T.ref [concrete]
// CHECK:STDOUT:     %complete_type.loc6_1.1: <witness> = complete_type_witness constants.%T.8b3 [symbolic = %complete_type.loc6_1.2 (constants.%complete_type.f87)]
// CHECK:STDOUT:     complete_type_witness = %complete_type.loc6_1.1
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%Adapter.0e3
// CHECK:STDOUT:     .T = <poisoned>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Convert(%a.param: %Adapter.e4c) -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: %Adapter.e4c = name_ref a, %a
// CHECK:STDOUT:   %int_32.loc9: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32.loc9: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   %.loc9_12.1: %i32 = as_compatible %a.ref
// CHECK:STDOUT:   %.loc9_12.2: %i32 = converted %a.ref, %.loc9_12.1
// CHECK:STDOUT:   %impl.elem0: %.7fa = impl_witness_access constants.%Copy.impl_witness.a32, element0 [concrete = constants.%Int.as.Copy.impl.Op.f59]
// CHECK:STDOUT:   %bound_method.loc9_12.1: <bound method> = bound_method %.loc9_12.2, %impl.elem0
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc9_12.2: <bound method> = bound_method %.loc9_12.2, %specific_fn
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call: init %i32 = call %bound_method.loc9_12.2(%.loc9_12.2)
// CHECK:STDOUT:   return %Int.as.Copy.impl.Op.call to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Adapter(constants.%T.8b3) {
// CHECK:STDOUT:   %T.loc4_15.1 => constants.%T.8b3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Adapter(constants.%i32) {
// CHECK:STDOUT:   %T.loc4_15.1 => constants.%i32
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.f8a
// CHECK:STDOUT:   %complete_type.loc6_1.2 => constants.%complete_type.1eb
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_adapt_generic_type.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Adapter.type: type = generic_class_type @Adapter [concrete]
// CHECK:STDOUT:   %Adapter.generic: %Adapter.type = struct_value () [concrete]
// CHECK:STDOUT:   %T.8b3: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %complete_type.f87: <witness> = complete_type_witness %T.8b3 [symbolic]
// CHECK:STDOUT:   %require_complete.4ae: <witness> = require_complete_type %T.8b3 [symbolic]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %Adapter.e4c: type = class_type @Adapter, @Adapter(%i32) [concrete]
// CHECK:STDOUT:   %pattern_type.5a0: type = pattern_type %Adapter.e4c [concrete]
// CHECK:STDOUT:   %pattern_type.7ce: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %ImportedConvert.type: type = fn_type @ImportedConvert [concrete]
// CHECK:STDOUT:   %ImportedConvert: %ImportedConvert.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32.builtin: type = int_type signed, %int_32 [concrete]
// CHECK:STDOUT:   %complete_type.f8a: <witness> = complete_type_witness %i32.builtin [concrete]
// CHECK:STDOUT:   %complete_type.1eb: <witness> = complete_type_witness %i32 [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Copy.Op.type: type = fn_type @Copy.Op [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.afd: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%N) [symbolic]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.6cd: %Int.as.Copy.impl.Op.type.afd = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.a32: <witness> = impl_witness imports.%Copy.impl_witness_table.1ed, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.276: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.f59: %Int.as.Copy.impl.Op.type.276 = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %i32, (%Copy.impl_witness.a32) [concrete]
// CHECK:STDOUT:   %.7fa: type = fn_type_with_self_type %Copy.Op.type, %Copy.facet [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %Int.as.Copy.impl.Op.f59, @Int.as.Copy.impl.Op(%int_32) [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %C.elem: type = unbound_element_type %C, %i32 [concrete]
// CHECK:STDOUT:   %struct_type.n: type = struct_type {.n: %i32} [concrete]
// CHECK:STDOUT:   %complete_type.54b: <witness> = complete_type_witness %struct_type.n [concrete]
// CHECK:STDOUT:   %Adapter.58f: type = class_type @Adapter, @Adapter(%C) [concrete]
// CHECK:STDOUT:   %pattern_type.5f8: type = pattern_type %Adapter.58f [concrete]
// CHECK:STDOUT:   %ImportedConvertLocal.type: type = fn_type @ImportedConvertLocal [concrete]
// CHECK:STDOUT:   %ImportedConvertLocal: %ImportedConvertLocal.type = struct_value () [concrete]
// CHECK:STDOUT:   %complete_type.af2: <witness> = complete_type_witness %C [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.Adapter: %Adapter.type = import_ref Main//adapt_generic_type, Adapter, loaded [concrete = constants.%Adapter.generic]
// CHECK:STDOUT:   %Main.Convert = import_ref Main//adapt_generic_type, Convert, unloaded
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .Copy = %Core.Copy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Main.import_ref.5ab: type = import_ref Main//adapt_generic_type, loc4_15, loaded [symbolic = @Adapter.%T (constants.%T.8b3)]
// CHECK:STDOUT:   %Main.import_ref.fb3: <witness> = import_ref Main//adapt_generic_type, loc6_1, loaded [symbolic = @Adapter.%complete_type (constants.%complete_type.f87)]
// CHECK:STDOUT:   %Main.import_ref.9a3 = import_ref Main//adapt_generic_type, inst{{[0-9A-F]+}} [no loc], unloaded
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/parts/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT:   %Core.Copy: type = import_ref Core//prelude/parts/copy, Copy, loaded [concrete = constants.%Copy.type]
// CHECK:STDOUT:   %Core.import_ref.d0f6: @Int.as.Copy.impl.%Int.as.Copy.impl.Op.type (%Int.as.Copy.impl.Op.type.afd) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.Copy.impl.%Int.as.Copy.impl.Op (constants.%Int.as.Copy.impl.Op.6cd)]
// CHECK:STDOUT:   %Copy.impl_witness_table.1ed = impl_witness_table (%Core.import_ref.d0f6), @Int.as.Copy.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Adapter = imports.%Main.Adapter
// CHECK:STDOUT:     .Convert = imports.%Main.Convert
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .ImportedConvert = %ImportedConvert.decl
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .ImportedConvertLocal = %ImportedConvertLocal.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %default.import = import <none>
// CHECK:STDOUT:   %ImportedConvert.decl: %ImportedConvert.type = fn_decl @ImportedConvert [concrete = constants.%ImportedConvert] {
// CHECK:STDOUT:     %a.patt: %pattern_type.5a0 = value_binding_pattern a [concrete]
// CHECK:STDOUT:     %a.param_patt: %pattern_type.5a0 = value_param_pattern %a.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.7ce = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.7ce = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32.loc6_40: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc6_40: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %a.param: %Adapter.e4c = value_param call_param0
// CHECK:STDOUT:     %.loc6: type = splice_block %Adapter [concrete = constants.%Adapter.e4c] {
// CHECK:STDOUT:       %Adapter.ref: %Adapter.type = name_ref Adapter, imports.%Main.Adapter [concrete = constants.%Adapter.generic]
// CHECK:STDOUT:       %int_32.loc6_31: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:       %i32.loc6_31: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:       %Adapter: type = class_type @Adapter, @Adapter(constants.%i32) [concrete = constants.%Adapter.e4c]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a: %Adapter.e4c = value_binding a, %a.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   %ImportedConvertLocal.decl: %ImportedConvertLocal.type = fn_decl @ImportedConvertLocal [concrete = constants.%ImportedConvertLocal] {
// CHECK:STDOUT:     %a.patt: %pattern_type.5f8 = value_binding_pattern a [concrete]
// CHECK:STDOUT:     %a.param_patt: %pattern_type.5f8 = value_param_pattern %a.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.7ce = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.7ce = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %a.param: %Adapter.58f = value_param call_param0
// CHECK:STDOUT:     %.loc14: type = splice_block %Adapter [concrete = constants.%Adapter.58f] {
// CHECK:STDOUT:       %Adapter.ref: %Adapter.type = name_ref Adapter, imports.%Main.Adapter [concrete = constants.%Adapter.generic]
// CHECK:STDOUT:       %C.ref.loc14: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:       %Adapter: type = class_type @Adapter, @Adapter(constants.%C) [concrete = constants.%Adapter.58f]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a: %Adapter.58f = value_binding a, %a.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @Adapter(imports.%Main.import_ref.5ab: type) [from "adapt_generic_type.carbon"] {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T.8b3)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T [symbolic = %require_complete (constants.%require_complete.4ae)]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %T [symbolic = %complete_type (constants.%complete_type.f87)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     complete_type_witness = imports.%Main.import_ref.fb3
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%Main.import_ref.9a3
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   %.loc11: %C.elem = field_decl n, element0 [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%struct_type.n [concrete = constants.%complete_type.54b]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT:   .n = %.loc11
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ImportedConvert(%a.param: %Adapter.e4c) -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: %Adapter.e4c = name_ref a, %a
// CHECK:STDOUT:   %int_32.loc7: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32.loc7: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   %.loc7_12.1: %i32 = as_compatible %a.ref
// CHECK:STDOUT:   %.loc7_12.2: %i32 = converted %a.ref, %.loc7_12.1
// CHECK:STDOUT:   %impl.elem0: %.7fa = impl_witness_access constants.%Copy.impl_witness.a32, element0 [concrete = constants.%Int.as.Copy.impl.Op.f59]
// CHECK:STDOUT:   %bound_method.loc7_12.1: <bound method> = bound_method %.loc7_12.2, %impl.elem0
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc7_12.2: <bound method> = bound_method %.loc7_12.2, %specific_fn
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call: init %i32 = call %bound_method.loc7_12.2(%.loc7_12.2)
// CHECK:STDOUT:   return %Int.as.Copy.impl.Op.call to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ImportedConvertLocal(%a.param: %Adapter.58f) -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: %Adapter.58f = name_ref a, %a
// CHECK:STDOUT:   %C.ref.loc15: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %.loc15_13.1: %C = as_compatible %a.ref
// CHECK:STDOUT:   %.loc15_13.2: %C = converted %a.ref, %.loc15_13.1
// CHECK:STDOUT:   %n.ref: %C.elem = name_ref n, @C.%.loc11 [concrete = @C.%.loc11]
// CHECK:STDOUT:   %.loc15_18.1: ref %i32 = class_element_access %.loc15_13.2, element0
// CHECK:STDOUT:   %.loc15_18.2: %i32 = bind_value %.loc15_18.1
// CHECK:STDOUT:   %impl.elem0: %.7fa = impl_witness_access constants.%Copy.impl_witness.a32, element0 [concrete = constants.%Int.as.Copy.impl.Op.f59]
// CHECK:STDOUT:   %bound_method.loc15_18.1: <bound method> = bound_method %.loc15_18.2, %impl.elem0
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc15_18.2: <bound method> = bound_method %.loc15_18.2, %specific_fn
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call: init %i32 = call %bound_method.loc15_18.2(%.loc15_18.2)
// CHECK:STDOUT:   return %Int.as.Copy.impl.Op.call to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Adapter(constants.%T.8b3) {
// CHECK:STDOUT:   %T => constants.%T.8b3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Adapter(constants.%i32) {
// CHECK:STDOUT:   %T => constants.%i32
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.f8a
// CHECK:STDOUT:   %complete_type => constants.%complete_type.1eb
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Adapter(constants.%C) {
// CHECK:STDOUT:   %T => constants.%C
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.54b
// CHECK:STDOUT:   %complete_type => constants.%complete_type.af2
// CHECK:STDOUT: }
// CHECK:STDOUT:
