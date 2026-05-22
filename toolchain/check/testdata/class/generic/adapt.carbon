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
  // CHECK:STDERR: fail_todo_extend_adapt_specific_type.carbon:[[@LINE+4]]:10: note: type `Adapter` does not implement interface `Core.ImplicitAs(C(i32))` [MissingImplInMemberAccessInContext]
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
  // CHECK:STDERR: fail_todo_import_extend_adapt_specific_type.carbon:[[@LINE+4]]:10: note: type `Adapter` does not implement interface `Core.ImplicitAs(C(i32))` [MissingImplInMemberAccessInContext]
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
// CHECK:STDOUT:   %.Self: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %T.67d: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %C.type: type = generic_class_type @C [concrete]
// CHECK:STDOUT:   %C.generic: %C.type = struct_value () [concrete]
// CHECK:STDOUT:   %C.e2f: type = class_type @C, @C(%T.67d) [symbolic]
// CHECK:STDOUT:   %require_complete.944: <witness> = require_complete_type %T.67d [symbolic]
// CHECK:STDOUT:   %C.elem.853: type = unbound_element_type %C.e2f, %T.67d [symbolic]
// CHECK:STDOUT:   %struct_type.x.0c5: type = struct_type {.x: %T.67d} [symbolic]
// CHECK:STDOUT:   %complete_type.735: <witness> = complete_type_witness %struct_type.x.0c5 [symbolic]
// CHECK:STDOUT:   %Adapter: type = class_type @Adapter [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %N: Core.IntLiteral = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %C.e7b: type = class_type @C, @C(%i32) [concrete]
// CHECK:STDOUT:   %i32.builtin: type = int_type signed, %int_32 [concrete]
// CHECK:STDOUT:   %complete_type.f8a: <witness> = complete_type_witness %i32.builtin [concrete]
// CHECK:STDOUT:   %C.elem.1d3: type = unbound_element_type %C.e7b, %i32 [concrete]
// CHECK:STDOUT:   %struct_type.x.2bb: type = struct_type {.x: %i32} [concrete]
// CHECK:STDOUT:   %complete_type.fa5: <witness> = complete_type_witness %struct_type.x.2bb [concrete]
// CHECK:STDOUT:   %pattern_type.9b0: type = pattern_type %Adapter [concrete]
// CHECK:STDOUT:   %.70c: Core.Form = init_form %i32 [concrete]
// CHECK:STDOUT:   %pattern_type.c6b: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %Access.type: type = fn_type @Access [concrete]
// CHECK:STDOUT:   %Access: %Access.type = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.b5d: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%N) [symbolic]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.c85: %Int.as.Copy.impl.Op.type.b5d = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.32d: <witness> = impl_witness imports.%Copy.impl_witness_table.07a, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.4a0: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.c4a: %Int.as.Copy.impl.Op.type.4a0 = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %i32, (%Copy.impl_witness.32d) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.afe: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet) [concrete]
// CHECK:STDOUT:   %.a5f: type = fn_type_with_self_type %Copy.WithSelf.Op.type.afe, %Copy.facet [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %Int.as.Copy.impl.Op.c4a, @Int.as.Copy.impl.Op(%int_32) [concrete]
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
// CHECK:STDOUT:   %Core.import_ref.809: @Int.as.Copy.impl.%Int.as.Copy.impl.Op.type (%Int.as.Copy.impl.Op.type.b5d) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.Copy.impl.%Int.as.Copy.impl.Op (constants.%Int.as.Copy.impl.Op.c85)]
// CHECK:STDOUT:   %Copy.impl_witness_table.07a = impl_witness_table (%Core.import_ref.809), @Int.as.Copy.impl [concrete]
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
// CHECK:STDOUT:     %.loc4_13.1: type = splice_block %.loc4_13.2 [concrete = type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.loc4_13.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc4_10.2: type = symbolic_binding T, 0 [symbolic = %T.loc4_10.1 (constants.%T.67d)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Adapter.decl: type = class_decl @Adapter [concrete = constants.%Adapter] {} {}
// CHECK:STDOUT:   %Access.decl: %Access.type = fn_decl @Access [concrete = constants.%Access] {
// CHECK:STDOUT:     %a.param_patt: %pattern_type.9b0 = value_param_pattern [concrete]
// CHECK:STDOUT:     %a.patt: %pattern_type.9b0 = at_binding_pattern a, %a.param_patt [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.c6b = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.c6b = return_slot_pattern %return.param_patt, %i32.loc12 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %i32.loc12: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc12: Core.Form = init_form %i32.loc12 [concrete = constants.%.70c]
// CHECK:STDOUT:     %a.param: %Adapter = value_param call_param0
// CHECK:STDOUT:     %Adapter.ref: type = name_ref Adapter, file.%Adapter.decl [concrete = constants.%Adapter]
// CHECK:STDOUT:     %a: %Adapter = value_binding a, %a.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @C(%T.loc4_10.2: type) {
// CHECK:STDOUT:   %T.loc4_10.1: type = symbolic_binding T, 0 [symbolic = %T.loc4_10.1 (constants.%T.67d)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T.loc4_10.1 [symbolic = %require_complete (constants.%require_complete.944)]
// CHECK:STDOUT:   %C: type = class_type @C, @C(%T.loc4_10.1) [symbolic = %C (constants.%C.e2f)]
// CHECK:STDOUT:   %C.elem: type = unbound_element_type %C, %T.loc4_10.1 [symbolic = %C.elem (constants.%C.elem.853)]
// CHECK:STDOUT:   %struct_type.x: type = struct_type {.x: @C.%T.loc4_10.1 (%T.67d)} [symbolic = %struct_type.x (constants.%struct_type.x.0c5)]
// CHECK:STDOUT:   %complete_type.loc6_1.2: <witness> = complete_type_witness %struct_type.x [symbolic = %complete_type.loc6_1.2 (constants.%complete_type.735)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %.loc5: @C.%C.elem (%C.elem.853) = field_decl x, element0 [concrete]
// CHECK:STDOUT:     %complete_type.loc6_1.1: <witness> = complete_type_witness constants.%struct_type.x.0c5 [symbolic = %complete_type.loc6_1.2 (constants.%complete_type.735)]
// CHECK:STDOUT:     complete_type_witness = %complete_type.loc6_1.1
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%C.e2f
// CHECK:STDOUT:     .T = <poisoned>
// CHECK:STDOUT:     .x = %.loc5
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Adapter {
// CHECK:STDOUT:   %C.ref: %C.type = name_ref C, file.%C.decl [concrete = constants.%C.generic]
// CHECK:STDOUT:   %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %C: type = class_type @C, @C(constants.%i32) [concrete = constants.%C.e7b]
// CHECK:STDOUT:   adapt_decl %C [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%struct_type.x.2bb [concrete = constants.%complete_type.fa5]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Adapter
// CHECK:STDOUT:   .C = <poisoned>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Access(%a.param: %Adapter) -> out %return.param: %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: %Adapter = name_ref a, %a
// CHECK:STDOUT:   %C.ref: %C.type = name_ref C, file.%C.decl [concrete = constants.%C.generic]
// CHECK:STDOUT:   %i32.loc13: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %C: type = class_type @C, @C(constants.%i32) [concrete = constants.%C.e7b]
// CHECK:STDOUT:   %.loc13_13.1: %C.e7b = as_compatible %a.ref
// CHECK:STDOUT:   %.loc13_13.2: %C.e7b = converted %a.ref, %.loc13_13.1
// CHECK:STDOUT:   %x.ref: %C.elem.1d3 = name_ref x, @C.%.loc5 [concrete = @C.%.loc5]
// CHECK:STDOUT:   %.loc13_23.1: ref %i32 = class_element_access %.loc13_13.2, element0
// CHECK:STDOUT:   %.loc13_23.2: %i32 = acquire_value %.loc13_23.1
// CHECK:STDOUT:   %impl.elem0: %.a5f = impl_witness_access constants.%Copy.impl_witness.32d, element0 [concrete = constants.%Int.as.Copy.impl.Op.c4a]
// CHECK:STDOUT:   %bound_method.loc13_23.1: <bound method> = bound_method %.loc13_23.2, %impl.elem0
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc13_23.2: <bound method> = bound_method %.loc13_23.2, %specific_fn
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call: init %i32 = call %bound_method.loc13_23.2(%.loc13_23.2)
// CHECK:STDOUT:   return %Int.as.Copy.impl.Op.call
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%T.67d) {
// CHECK:STDOUT:   %T.loc4_10.1 => constants.%T.67d
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%i32) {
// CHECK:STDOUT:   %T.loc4_10.1 => constants.%i32
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.f8a
// CHECK:STDOUT:   %C => constants.%C.e7b
// CHECK:STDOUT:   %C.elem => constants.%C.elem.1d3
// CHECK:STDOUT:   %struct_type.x => constants.%struct_type.x.2bb
// CHECK:STDOUT:   %complete_type.loc6_1.2 => constants.%complete_type.fa5
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_adapt_specific_type.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Adapter: type = class_type @Adapter [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %N: Core.IntLiteral = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %i32.builtin: type = int_type signed, %int_32 [concrete]
// CHECK:STDOUT:   %complete_type.f8a: <witness> = complete_type_witness %i32.builtin [concrete]
// CHECK:STDOUT:   %C.type: type = generic_class_type @C [concrete]
// CHECK:STDOUT:   %C.generic: %C.type = struct_value () [concrete]
// CHECK:STDOUT:   %T.67d: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %struct_type.x.0c5: type = struct_type {.x: %T.67d} [symbolic]
// CHECK:STDOUT:   %complete_type.735: <witness> = complete_type_witness %struct_type.x.0c5 [symbolic]
// CHECK:STDOUT:   %C.ebd: type = class_type @C, @C(%T.67d) [symbolic]
// CHECK:STDOUT:   %C.elem.992: type = unbound_element_type %C.ebd, %T.67d [symbolic]
// CHECK:STDOUT:   %require_complete.944: <witness> = require_complete_type %T.67d [symbolic]
// CHECK:STDOUT:   %C.6b0: type = class_type @C, @C(%i32) [concrete]
// CHECK:STDOUT:   %struct_type.x.740: type = struct_type {.x: %i32} [concrete]
// CHECK:STDOUT:   %complete_type.9ec: <witness> = complete_type_witness %struct_type.x.740 [concrete]
// CHECK:STDOUT:   %C.elem.bed: type = unbound_element_type %C.6b0, %i32 [concrete]
// CHECK:STDOUT:   %pattern_type.bff: type = pattern_type %Adapter [concrete]
// CHECK:STDOUT:   %.46a: Core.Form = init_form %i32 [concrete]
// CHECK:STDOUT:   %pattern_type.ec3: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %ImportedAccess.type: type = fn_type @ImportedAccess [concrete]
// CHECK:STDOUT:   %ImportedAccess: %ImportedAccess.type = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.c59: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%N) [symbolic]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.fb0: %Int.as.Copy.impl.Op.type.c59 = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.c36: <witness> = impl_witness imports.%Copy.impl_witness_table.03a, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.e20: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.123: %Int.as.Copy.impl.Op.type.e20 = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %i32, (%Copy.impl_witness.c36) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.56d: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet) [concrete]
// CHECK:STDOUT:   %.cf6: type = fn_type_with_self_type %Copy.WithSelf.Op.type.56d, %Copy.facet [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %Int.as.Copy.impl.Op.123, @Int.as.Copy.impl.Op(%int_32) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.C: %C.type = import_ref Main//adapt_specific_type, C, loaded [concrete = constants.%C.generic]
// CHECK:STDOUT:   %Main.Adapter: type = import_ref Main//adapt_specific_type, Adapter, loaded [concrete = constants.%Adapter]
// CHECK:STDOUT:   %Main.Access = import_ref Main//adapt_specific_type, Access, unloaded
// CHECK:STDOUT:   %Core.b38: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .Copy = %Core.Copy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Main.import_ref.7e3: <witness> = import_ref Main//adapt_specific_type, loc6_1, loaded [symbolic = @C.%complete_type (constants.%complete_type.735)]
// CHECK:STDOUT:   %Main.import_ref.668 = import_ref Main//adapt_specific_type, inst{{[0-9A-F]+}} [no loc], unloaded
// CHECK:STDOUT:   %Main.import_ref.17f: @C.%C.elem (%C.elem.992) = import_ref Main//adapt_specific_type, loc5_8, loaded [concrete = %.65d]
// CHECK:STDOUT:   %Main.import_ref.b3b: type = import_ref Main//adapt_specific_type, loc4_10, loaded [symbolic = @C.%T (constants.%T.67d)]
// CHECK:STDOUT:   %Main.import_ref.bc4: <witness> = import_ref Main//adapt_specific_type, loc10_1, loaded [concrete = constants.%complete_type.9ec]
// CHECK:STDOUT:   %Main.import_ref.bbb = import_ref Main//adapt_specific_type, inst{{[0-9A-F]+}} [no loc], unloaded
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/parts/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT:   %.65d: @C.%C.elem (%C.elem.992) = field_decl x, element0 [concrete]
// CHECK:STDOUT:   %Core.Copy: type = import_ref Core//prelude/parts/copy, Copy, loaded [concrete = constants.%Copy.type]
// CHECK:STDOUT:   %Core.import_ref.8e5: @Int.as.Copy.impl.%Int.as.Copy.impl.Op.type (%Int.as.Copy.impl.Op.type.c59) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.Copy.impl.%Int.as.Copy.impl.Op (constants.%Int.as.Copy.impl.Op.fb0)]
// CHECK:STDOUT:   %Copy.impl_witness_table.03a = impl_witness_table (%Core.import_ref.8e5), @Int.as.Copy.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .C = imports.%Main.C
// CHECK:STDOUT:     .Adapter = imports.%Main.Adapter
// CHECK:STDOUT:     .Access = imports.%Main.Access
// CHECK:STDOUT:     .Core = imports.%Core.b38
// CHECK:STDOUT:     .ImportedAccess = %ImportedAccess.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %default.import = import <none>
// CHECK:STDOUT:   %ImportedAccess.decl: %ImportedAccess.type = fn_decl @ImportedAccess [concrete = constants.%ImportedAccess] {
// CHECK:STDOUT:     %a.param_patt: %pattern_type.bff = value_param_pattern [concrete]
// CHECK:STDOUT:     %a.patt: %pattern_type.bff = at_binding_pattern a, %a.param_patt [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.ec3 = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.ec3 = return_slot_pattern %return.param_patt, %i32.loc6 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %i32.loc6: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc6: Core.Form = init_form %i32.loc6 [concrete = constants.%.46a]
// CHECK:STDOUT:     %a.param: %Adapter = value_param call_param0
// CHECK:STDOUT:     %Adapter.ref: type = name_ref Adapter, imports.%Main.Adapter [concrete = constants.%Adapter]
// CHECK:STDOUT:     %a: %Adapter = value_binding a, %a.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Adapter [from "adapt_specific_type.carbon"] {
// CHECK:STDOUT:   complete_type_witness = imports.%Main.import_ref.bc4
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.bbb
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @C(imports.%Main.import_ref.b3b: type) [from "adapt_specific_type.carbon"] {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T.67d)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T [symbolic = %require_complete (constants.%require_complete.944)]
// CHECK:STDOUT:   %C: type = class_type @C, @C(%T) [symbolic = %C (constants.%C.ebd)]
// CHECK:STDOUT:   %C.elem: type = unbound_element_type %C, %T [symbolic = %C.elem (constants.%C.elem.992)]
// CHECK:STDOUT:   %struct_type.x: type = struct_type {.x: @C.%T (%T.67d)} [symbolic = %struct_type.x (constants.%struct_type.x.0c5)]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.x [symbolic = %complete_type (constants.%complete_type.735)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     complete_type_witness = imports.%Main.import_ref.7e3
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%Main.import_ref.668
// CHECK:STDOUT:     .x = imports.%Main.import_ref.17f
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ImportedAccess(%a.param: %Adapter) -> out %return.param: %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: %Adapter = name_ref a, %a
// CHECK:STDOUT:   %C.ref: %C.type = name_ref C, imports.%Main.C [concrete = constants.%C.generic]
// CHECK:STDOUT:   %i32.loc7: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %C: type = class_type @C, @C(constants.%i32) [concrete = constants.%C.6b0]
// CHECK:STDOUT:   %.loc7_13.1: %C.6b0 = as_compatible %a.ref
// CHECK:STDOUT:   %.loc7_13.2: %C.6b0 = converted %a.ref, %.loc7_13.1
// CHECK:STDOUT:   %x.ref: %C.elem.bed = name_ref x, imports.%Main.import_ref.17f [concrete = imports.%.65d]
// CHECK:STDOUT:   %.loc7_23.1: ref %i32 = class_element_access %.loc7_13.2, element0
// CHECK:STDOUT:   %.loc7_23.2: %i32 = acquire_value %.loc7_23.1
// CHECK:STDOUT:   %impl.elem0: %.cf6 = impl_witness_access constants.%Copy.impl_witness.c36, element0 [concrete = constants.%Int.as.Copy.impl.Op.123]
// CHECK:STDOUT:   %bound_method.loc7_23.1: <bound method> = bound_method %.loc7_23.2, %impl.elem0
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc7_23.2: <bound method> = bound_method %.loc7_23.2, %specific_fn
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call: init %i32 = call %bound_method.loc7_23.2(%.loc7_23.2)
// CHECK:STDOUT:   return %Int.as.Copy.impl.Op.call
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%T.67d) {
// CHECK:STDOUT:   %T => constants.%T.67d
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%i32) {
// CHECK:STDOUT:   %T => constants.%i32
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.f8a
// CHECK:STDOUT:   %C => constants.%C.6b0
// CHECK:STDOUT:   %C.elem => constants.%C.elem.bed
// CHECK:STDOUT:   %struct_type.x => constants.%struct_type.x.740
// CHECK:STDOUT:   %complete_type => constants.%complete_type.9ec
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_todo_extend_adapt_specific_type.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %C.type: type = generic_class_type @C [concrete]
// CHECK:STDOUT:   %C.generic: %C.type = struct_value () [concrete]
// CHECK:STDOUT:   %C.aa9: type = class_type @C, @C(%T) [symbolic]
// CHECK:STDOUT:   %require_complete.944: <witness> = require_complete_type %T [symbolic]
// CHECK:STDOUT:   %C.elem.3ab: type = unbound_element_type %C.aa9, %T [symbolic]
// CHECK:STDOUT:   %struct_type.x.0c5: type = struct_type {.x: %T} [symbolic]
// CHECK:STDOUT:   %complete_type.735: <witness> = complete_type_witness %struct_type.x.0c5 [symbolic]
// CHECK:STDOUT:   %Adapter: type = class_type @Adapter [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %C.656: type = class_type @C, @C(%i32) [concrete]
// CHECK:STDOUT:   %i32.builtin: type = int_type signed, %int_32 [concrete]
// CHECK:STDOUT:   %complete_type.f8a: <witness> = complete_type_witness %i32.builtin [concrete]
// CHECK:STDOUT:   %C.elem.d1c: type = unbound_element_type %C.656, %i32 [concrete]
// CHECK:STDOUT:   %struct_type.x.2bb: type = struct_type {.x: %i32} [concrete]
// CHECK:STDOUT:   %complete_type.fa5: <witness> = complete_type_witness %struct_type.x.2bb [concrete]
// CHECK:STDOUT:   %pattern_type.6a6: type = pattern_type %Adapter [concrete]
// CHECK:STDOUT:   %.70c: Core.Form = init_form %i32 [concrete]
// CHECK:STDOUT:   %pattern_type.c6b: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %Access.type: type = fn_type @Access [concrete]
// CHECK:STDOUT:   %Access: %Access.type = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.649: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.649 = struct_value () [concrete]
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
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.649 = import_ref Core//prelude/parts/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
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
// CHECK:STDOUT:     %.loc4_13.1: type = splice_block %.loc4_13.2 [concrete = type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.loc4_13.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc4_10.2: type = symbolic_binding T, 0 [symbolic = %T.loc4_10.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Adapter.decl: type = class_decl @Adapter [concrete = constants.%Adapter] {} {}
// CHECK:STDOUT:   %Access.decl: %Access.type = fn_decl @Access [concrete = constants.%Access] {
// CHECK:STDOUT:     %a.param_patt: %pattern_type.6a6 = value_param_pattern [concrete]
// CHECK:STDOUT:     %a.patt: %pattern_type.6a6 = at_binding_pattern a, %a.param_patt [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.c6b = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.c6b = return_slot_pattern %return.param_patt, %i32 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc12: Core.Form = init_form %i32 [concrete = constants.%.70c]
// CHECK:STDOUT:     %a.param: %Adapter = value_param call_param0
// CHECK:STDOUT:     %Adapter.ref: type = name_ref Adapter, file.%Adapter.decl [concrete = constants.%Adapter]
// CHECK:STDOUT:     %a: %Adapter = value_binding a, %a.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @C(%T.loc4_10.2: type) {
// CHECK:STDOUT:   %T.loc4_10.1: type = symbolic_binding T, 0 [symbolic = %T.loc4_10.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T.loc4_10.1 [symbolic = %require_complete (constants.%require_complete.944)]
// CHECK:STDOUT:   %C: type = class_type @C, @C(%T.loc4_10.1) [symbolic = %C (constants.%C.aa9)]
// CHECK:STDOUT:   %C.elem: type = unbound_element_type %C, %T.loc4_10.1 [symbolic = %C.elem (constants.%C.elem.3ab)]
// CHECK:STDOUT:   %struct_type.x: type = struct_type {.x: @C.%T.loc4_10.1 (%T)} [symbolic = %struct_type.x (constants.%struct_type.x.0c5)]
// CHECK:STDOUT:   %complete_type.loc6_1.2: <witness> = complete_type_witness %struct_type.x [symbolic = %complete_type.loc6_1.2 (constants.%complete_type.735)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %.loc5: @C.%C.elem (%C.elem.3ab) = field_decl x, element0 [concrete]
// CHECK:STDOUT:     %complete_type.loc6_1.1: <witness> = complete_type_witness constants.%struct_type.x.0c5 [symbolic = %complete_type.loc6_1.2 (constants.%complete_type.735)]
// CHECK:STDOUT:     complete_type_witness = %complete_type.loc6_1.1
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%C.aa9
// CHECK:STDOUT:     .T = <poisoned>
// CHECK:STDOUT:     .x = %.loc5
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Adapter {
// CHECK:STDOUT:   %C.ref: %C.type = name_ref C, file.%C.decl [concrete = constants.%C.generic]
// CHECK:STDOUT:   %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %C: type = class_type @C, @C(constants.%i32) [concrete = constants.%C.656]
// CHECK:STDOUT:   adapt_decl %C [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%struct_type.x.2bb [concrete = constants.%complete_type.fa5]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Adapter
// CHECK:STDOUT:   .C = <poisoned>
// CHECK:STDOUT:   .x = <poisoned>
// CHECK:STDOUT:   extend %C
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Access(%a.param: %Adapter) -> out %return.param: %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: %Adapter = name_ref a, %a
// CHECK:STDOUT:   %x.ref: %C.elem.d1c = name_ref x, @C.%.loc5 [concrete = @C.%.loc5]
// CHECK:STDOUT:   %.loc21_11.1: %C.656 = converted %a.ref, <error> [concrete = <error>]
// CHECK:STDOUT:   %.loc21_11.2: %i32 = class_element_access <error>, element0 [concrete = <error>]
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%T) {
// CHECK:STDOUT:   %T.loc4_10.1 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%i32) {
// CHECK:STDOUT:   %T.loc4_10.1 => constants.%i32
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.f8a
// CHECK:STDOUT:   %C => constants.%C.656
// CHECK:STDOUT:   %C.elem => constants.%C.elem.d1c
// CHECK:STDOUT:   %struct_type.x => constants.%struct_type.x.2bb
// CHECK:STDOUT:   %complete_type.loc6_1.2 => constants.%complete_type.fa5
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- extend_adapt_specific_type_library.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %C.type: type = generic_class_type @C [concrete]
// CHECK:STDOUT:   %C.generic: %C.type = struct_value () [concrete]
// CHECK:STDOUT:   %C.342: type = class_type @C, @C(%T) [symbolic]
// CHECK:STDOUT:   %require_complete.944: <witness> = require_complete_type %T [symbolic]
// CHECK:STDOUT:   %C.elem.225: type = unbound_element_type %C.342, %T [symbolic]
// CHECK:STDOUT:   %struct_type.x.0c5: type = struct_type {.x: %T} [symbolic]
// CHECK:STDOUT:   %complete_type.735: <witness> = complete_type_witness %struct_type.x.0c5 [symbolic]
// CHECK:STDOUT:   %Adapter: type = class_type @Adapter [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %C.025: type = class_type @C, @C(%i32) [concrete]
// CHECK:STDOUT:   %i32.builtin: type = int_type signed, %int_32 [concrete]
// CHECK:STDOUT:   %complete_type.f8a: <witness> = complete_type_witness %i32.builtin [concrete]
// CHECK:STDOUT:   %C.elem.a68: type = unbound_element_type %C.025, %i32 [concrete]
// CHECK:STDOUT:   %struct_type.x.2bb: type = struct_type {.x: %i32} [concrete]
// CHECK:STDOUT:   %complete_type.fa5: <witness> = complete_type_witness %struct_type.x.2bb [concrete]
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
// CHECK:STDOUT:     %.loc7_13.1: type = splice_block %.loc7_13.2 [concrete = type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.loc7_13.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc7_10.2: type = symbolic_binding T, 0 [symbolic = %T.loc7_10.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Adapter.decl: type = class_decl @Adapter [concrete = constants.%Adapter] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @C(%T.loc7_10.2: type) {
// CHECK:STDOUT:   %T.loc7_10.1: type = symbolic_binding T, 0 [symbolic = %T.loc7_10.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T.loc7_10.1 [symbolic = %require_complete (constants.%require_complete.944)]
// CHECK:STDOUT:   %C: type = class_type @C, @C(%T.loc7_10.1) [symbolic = %C (constants.%C.342)]
// CHECK:STDOUT:   %C.elem: type = unbound_element_type %C, %T.loc7_10.1 [symbolic = %C.elem (constants.%C.elem.225)]
// CHECK:STDOUT:   %struct_type.x: type = struct_type {.x: @C.%T.loc7_10.1 (%T)} [symbolic = %struct_type.x (constants.%struct_type.x.0c5)]
// CHECK:STDOUT:   %complete_type.loc9_1.2: <witness> = complete_type_witness %struct_type.x [symbolic = %complete_type.loc9_1.2 (constants.%complete_type.735)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %.loc8: @C.%C.elem (%C.elem.225) = field_decl x, element0 [concrete]
// CHECK:STDOUT:     %complete_type.loc9_1.1: <witness> = complete_type_witness constants.%struct_type.x.0c5 [symbolic = %complete_type.loc9_1.2 (constants.%complete_type.735)]
// CHECK:STDOUT:     complete_type_witness = %complete_type.loc9_1.1
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%C.342
// CHECK:STDOUT:     .T = <poisoned>
// CHECK:STDOUT:     .x = %.loc8
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Adapter {
// CHECK:STDOUT:   %C.ref: %C.type = name_ref C, file.%C.decl [concrete = constants.%C.generic]
// CHECK:STDOUT:   %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %C: type = class_type @C, @C(constants.%i32) [concrete = constants.%C.025]
// CHECK:STDOUT:   adapt_decl %C [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%struct_type.x.2bb [concrete = constants.%complete_type.fa5]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Adapter
// CHECK:STDOUT:   .C = <poisoned>
// CHECK:STDOUT:   extend %C
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%T) {
// CHECK:STDOUT:   %T.loc7_10.1 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%i32) {
// CHECK:STDOUT:   %T.loc7_10.1 => constants.%i32
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.f8a
// CHECK:STDOUT:   %C => constants.%C.025
// CHECK:STDOUT:   %C.elem => constants.%C.elem.a68
// CHECK:STDOUT:   %struct_type.x => constants.%struct_type.x.2bb
// CHECK:STDOUT:   %complete_type.loc9_1.2 => constants.%complete_type.fa5
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
// CHECK:STDOUT:   %i32.builtin: type = int_type signed, %int_32 [concrete]
// CHECK:STDOUT:   %complete_type.f8a: <witness> = complete_type_witness %i32.builtin [concrete]
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %struct_type.x.0c5: type = struct_type {.x: %T} [symbolic]
// CHECK:STDOUT:   %complete_type.735: <witness> = complete_type_witness %struct_type.x.0c5 [symbolic]
// CHECK:STDOUT:   %C.f19: type = class_type @C, @C(%T) [symbolic]
// CHECK:STDOUT:   %C.elem.a43: type = unbound_element_type %C.f19, %T [symbolic]
// CHECK:STDOUT:   %require_complete.944: <witness> = require_complete_type %T [symbolic]
// CHECK:STDOUT:   %C.e72: type = class_type @C, @C(%i32) [concrete]
// CHECK:STDOUT:   %struct_type.x.740: type = struct_type {.x: %i32} [concrete]
// CHECK:STDOUT:   %complete_type.9ec: <witness> = complete_type_witness %struct_type.x.740 [concrete]
// CHECK:STDOUT:   %C.elem.5c5: type = unbound_element_type %C.e72, %i32 [concrete]
// CHECK:STDOUT:   %pattern_type.c59: type = pattern_type %Adapter [concrete]
// CHECK:STDOUT:   %.46a: Core.Form = init_form %i32 [concrete]
// CHECK:STDOUT:   %pattern_type.ec3: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %ImportedAccess.type: type = fn_type @ImportedAccess [concrete]
// CHECK:STDOUT:   %ImportedAccess: %ImportedAccess.type = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.649: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.649 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.C = import_ref Main//extend_adapt_specific_type_library, C, unloaded
// CHECK:STDOUT:   %Main.Adapter: type = import_ref Main//extend_adapt_specific_type_library, Adapter, loaded [concrete = constants.%Adapter]
// CHECK:STDOUT:   %Core.b38: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Main.import_ref.7e3: <witness> = import_ref Main//extend_adapt_specific_type_library, loc9_1, loaded [symbolic = @C.%complete_type (constants.%complete_type.735)]
// CHECK:STDOUT:   %Main.import_ref.38a = import_ref Main//extend_adapt_specific_type_library, inst{{[0-9A-F]+}} [no loc], unloaded
// CHECK:STDOUT:   %Main.import_ref.58c: @C.%C.elem (%C.elem.a43) = import_ref Main//extend_adapt_specific_type_library, loc8_8, loaded [concrete = %.c5a]
// CHECK:STDOUT:   %Main.import_ref.b3b: type = import_ref Main//extend_adapt_specific_type_library, loc7_10, loaded [symbolic = @C.%T (constants.%T)]
// CHECK:STDOUT:   %Main.import_ref.bc4: <witness> = import_ref Main//extend_adapt_specific_type_library, loc13_1, loaded [concrete = constants.%complete_type.9ec]
// CHECK:STDOUT:   %Main.import_ref.d90 = import_ref Main//extend_adapt_specific_type_library, inst{{[0-9A-F]+}} [no loc], unloaded
// CHECK:STDOUT:   %Main.import_ref.2e2d92.2: type = import_ref Main//extend_adapt_specific_type_library, loc12_21, loaded [concrete = constants.%C.e72]
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/parts/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT:   %.c5a: @C.%C.elem (%C.elem.a43) = field_decl x, element0 [concrete]
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.649 = import_ref Core//prelude/parts/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .C = imports.%Main.C
// CHECK:STDOUT:     .Adapter = imports.%Main.Adapter
// CHECK:STDOUT:     .Core = imports.%Core.b38
// CHECK:STDOUT:     .ImportedAccess = %ImportedAccess.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %default.import = import <none>
// CHECK:STDOUT:   %ImportedAccess.decl: %ImportedAccess.type = fn_decl @ImportedAccess [concrete = constants.%ImportedAccess] {
// CHECK:STDOUT:     %a.param_patt: %pattern_type.c59 = value_param_pattern [concrete]
// CHECK:STDOUT:     %a.patt: %pattern_type.c59 = at_binding_pattern a, %a.param_patt [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.ec3 = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.ec3 = return_slot_pattern %return.param_patt, %i32 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc6: Core.Form = init_form %i32 [concrete = constants.%.46a]
// CHECK:STDOUT:     %a.param: %Adapter = value_param call_param0
// CHECK:STDOUT:     %Adapter.ref: type = name_ref Adapter, imports.%Main.Adapter [concrete = constants.%Adapter]
// CHECK:STDOUT:     %a: %Adapter = value_binding a, %a.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Adapter [from "extend_adapt_specific_type_library.carbon"] {
// CHECK:STDOUT:   complete_type_witness = imports.%Main.import_ref.bc4
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.d90
// CHECK:STDOUT:   .x = <poisoned>
// CHECK:STDOUT:   extend imports.%Main.import_ref.2e2d92.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @C(imports.%Main.import_ref.b3b: type) [from "extend_adapt_specific_type_library.carbon"] {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T [symbolic = %require_complete (constants.%require_complete.944)]
// CHECK:STDOUT:   %C: type = class_type @C, @C(%T) [symbolic = %C (constants.%C.f19)]
// CHECK:STDOUT:   %C.elem: type = unbound_element_type %C, %T [symbolic = %C.elem (constants.%C.elem.a43)]
// CHECK:STDOUT:   %struct_type.x: type = struct_type {.x: @C.%T (%T)} [symbolic = %struct_type.x (constants.%struct_type.x.0c5)]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.x [symbolic = %complete_type (constants.%complete_type.735)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     complete_type_witness = imports.%Main.import_ref.7e3
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%Main.import_ref.38a
// CHECK:STDOUT:     .x = imports.%Main.import_ref.58c
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ImportedAccess(%a.param: %Adapter) -> out %return.param: %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: %Adapter = name_ref a, %a
// CHECK:STDOUT:   %x.ref: %C.elem.5c5 = name_ref x, imports.%Main.import_ref.58c [concrete = imports.%.c5a]
// CHECK:STDOUT:   %.loc15_11.1: %C.e72 = converted %a.ref, <error> [concrete = <error>]
// CHECK:STDOUT:   %.loc15_11.2: %i32 = class_element_access <error>, element0 [concrete = <error>]
// CHECK:STDOUT:   return <error>
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
// CHECK:STDOUT:   %C => constants.%C.e72
// CHECK:STDOUT:   %C.elem => constants.%C.elem.5c5
// CHECK:STDOUT:   %struct_type.x => constants.%struct_type.x.740
// CHECK:STDOUT:   %complete_type => constants.%complete_type.9ec
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- adapt_generic_type.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %T.67d: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %Adapter.type: type = generic_class_type @Adapter [concrete]
// CHECK:STDOUT:   %Adapter.generic: %Adapter.type = struct_value () [concrete]
// CHECK:STDOUT:   %Adapter.fc9: type = class_type @Adapter, @Adapter(%T.67d) [symbolic]
// CHECK:STDOUT:   %require_complete.944: <witness> = require_complete_type %T.67d [symbolic]
// CHECK:STDOUT:   %complete_type.1aa: <witness> = complete_type_witness %T.67d [symbolic]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %N: Core.IntLiteral = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %Adapter.316: type = class_type @Adapter, @Adapter(%i32) [concrete]
// CHECK:STDOUT:   %pattern_type.011: type = pattern_type %Adapter.316 [concrete]
// CHECK:STDOUT:   %.70c: Core.Form = init_form %i32 [concrete]
// CHECK:STDOUT:   %pattern_type.c6b: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %Convert.type: type = fn_type @Convert [concrete]
// CHECK:STDOUT:   %Convert: %Convert.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32.builtin: type = int_type signed, %int_32 [concrete]
// CHECK:STDOUT:   %complete_type.f8a: <witness> = complete_type_witness %i32.builtin [concrete]
// CHECK:STDOUT:   %complete_type.634: <witness> = complete_type_witness %i32 [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.b5d: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%N) [symbolic]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.c85: %Int.as.Copy.impl.Op.type.b5d = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.32d: <witness> = impl_witness imports.%Copy.impl_witness_table.07a, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.4a0: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.c4a: %Int.as.Copy.impl.Op.type.4a0 = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %i32, (%Copy.impl_witness.32d) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.afe: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet) [concrete]
// CHECK:STDOUT:   %.a5f: type = fn_type_with_self_type %Copy.WithSelf.Op.type.afe, %Copy.facet [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %Int.as.Copy.impl.Op.c4a, @Int.as.Copy.impl.Op(%int_32) [concrete]
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
// CHECK:STDOUT:   %Core.import_ref.809: @Int.as.Copy.impl.%Int.as.Copy.impl.Op.type (%Int.as.Copy.impl.Op.type.b5d) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.Copy.impl.%Int.as.Copy.impl.Op (constants.%Int.as.Copy.impl.Op.c85)]
// CHECK:STDOUT:   %Copy.impl_witness_table.07a = impl_witness_table (%Core.import_ref.809), @Int.as.Copy.impl [concrete]
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
// CHECK:STDOUT:     %.loc4_19.1: type = splice_block %.loc4_19.2 [concrete = type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.loc4_19.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc4_16.2: type = symbolic_binding T, 0 [symbolic = %T.loc4_16.1 (constants.%T.67d)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Convert.decl: %Convert.type = fn_decl @Convert [concrete = constants.%Convert] {
// CHECK:STDOUT:     %a.param_patt: %pattern_type.011 = value_param_pattern [concrete]
// CHECK:STDOUT:     %a.patt: %pattern_type.011 = at_binding_pattern a, %a.param_patt [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.c6b = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.c6b = return_slot_pattern %return.param_patt, %i32.loc8_32 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %i32.loc8_32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc8_32: Core.Form = init_form %i32.loc8_32 [concrete = constants.%.70c]
// CHECK:STDOUT:     %a.param: %Adapter.316 = value_param call_param0
// CHECK:STDOUT:     %.loc8_26: type = splice_block %Adapter [concrete = constants.%Adapter.316] {
// CHECK:STDOUT:       %Adapter.ref: %Adapter.type = name_ref Adapter, file.%Adapter.decl [concrete = constants.%Adapter.generic]
// CHECK:STDOUT:       %i32.loc8_23: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:       %Adapter: type = class_type @Adapter, @Adapter(constants.%i32) [concrete = constants.%Adapter.316]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a: %Adapter.316 = value_binding a, %a.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @Adapter(%T.loc4_16.2: type) {
// CHECK:STDOUT:   %T.loc4_16.1: type = symbolic_binding T, 0 [symbolic = %T.loc4_16.1 (constants.%T.67d)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T.loc4_16.1 [symbolic = %require_complete (constants.%require_complete.944)]
// CHECK:STDOUT:   %complete_type.loc6_1.2: <witness> = complete_type_witness %T.loc4_16.1 [symbolic = %complete_type.loc6_1.2 (constants.%complete_type.1aa)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc4_16.2 [symbolic = %T.loc4_16.1 (constants.%T.67d)]
// CHECK:STDOUT:     adapt_decl %T.ref [concrete]
// CHECK:STDOUT:     %complete_type.loc6_1.1: <witness> = complete_type_witness constants.%T.67d [symbolic = %complete_type.loc6_1.2 (constants.%complete_type.1aa)]
// CHECK:STDOUT:     complete_type_witness = %complete_type.loc6_1.1
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%Adapter.fc9
// CHECK:STDOUT:     .T = <poisoned>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Convert(%a.param: %Adapter.316) -> out %return.param: %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: %Adapter.316 = name_ref a, %a
// CHECK:STDOUT:   %i32.loc9: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %.loc9_12.1: %i32 = as_compatible %a.ref
// CHECK:STDOUT:   %.loc9_12.2: %i32 = converted %a.ref, %.loc9_12.1
// CHECK:STDOUT:   %impl.elem0: %.a5f = impl_witness_access constants.%Copy.impl_witness.32d, element0 [concrete = constants.%Int.as.Copy.impl.Op.c4a]
// CHECK:STDOUT:   %bound_method.loc9_12.1: <bound method> = bound_method %.loc9_12.2, %impl.elem0
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc9_12.2: <bound method> = bound_method %.loc9_12.2, %specific_fn
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call: init %i32 = call %bound_method.loc9_12.2(%.loc9_12.2)
// CHECK:STDOUT:   return %Int.as.Copy.impl.Op.call
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Adapter(constants.%T.67d) {
// CHECK:STDOUT:   %T.loc4_16.1 => constants.%T.67d
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Adapter(constants.%i32) {
// CHECK:STDOUT:   %T.loc4_16.1 => constants.%i32
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.f8a
// CHECK:STDOUT:   %complete_type.loc6_1.2 => constants.%complete_type.634
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_adapt_generic_type.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Adapter.type: type = generic_class_type @Adapter [concrete]
// CHECK:STDOUT:   %Adapter.generic: %Adapter.type = struct_value () [concrete]
// CHECK:STDOUT:   %T.67d: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %complete_type.1aa: <witness> = complete_type_witness %T.67d [symbolic]
// CHECK:STDOUT:   %require_complete.944: <witness> = require_complete_type %T.67d [symbolic]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %N: Core.IntLiteral = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %Adapter.d3c: type = class_type @Adapter, @Adapter(%i32) [concrete]
// CHECK:STDOUT:   %pattern_type.9f4: type = pattern_type %Adapter.d3c [concrete]
// CHECK:STDOUT:   %.70c: Core.Form = init_form %i32 [concrete]
// CHECK:STDOUT:   %pattern_type.c6b: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %ImportedConvert.type: type = fn_type @ImportedConvert [concrete]
// CHECK:STDOUT:   %ImportedConvert: %ImportedConvert.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32.builtin: type = int_type signed, %int_32 [concrete]
// CHECK:STDOUT:   %complete_type.f8a: <witness> = complete_type_witness %i32.builtin [concrete]
// CHECK:STDOUT:   %complete_type.634: <witness> = complete_type_witness %i32 [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.b5d: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%N) [symbolic]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.c85: %Int.as.Copy.impl.Op.type.b5d = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.32d: <witness> = impl_witness imports.%Copy.impl_witness_table.07a, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.4a0: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.c4a: %Int.as.Copy.impl.Op.type.4a0 = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %i32, (%Copy.impl_witness.32d) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.afe: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet) [concrete]
// CHECK:STDOUT:   %.a5f: type = fn_type_with_self_type %Copy.WithSelf.Op.type.afe, %Copy.facet [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %Int.as.Copy.impl.Op.c4a, @Int.as.Copy.impl.Op(%int_32) [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %C.elem: type = unbound_element_type %C, %i32 [concrete]
// CHECK:STDOUT:   %struct_type.n: type = struct_type {.n: %i32} [concrete]
// CHECK:STDOUT:   %complete_type.059: <witness> = complete_type_witness %struct_type.n [concrete]
// CHECK:STDOUT:   %Adapter.5e2: type = class_type @Adapter, @Adapter(%C) [concrete]
// CHECK:STDOUT:   %pattern_type.aca: type = pattern_type %Adapter.5e2 [concrete]
// CHECK:STDOUT:   %ImportedConvertLocal.type: type = fn_type @ImportedConvertLocal [concrete]
// CHECK:STDOUT:   %ImportedConvertLocal: %ImportedConvertLocal.type = struct_value () [concrete]
// CHECK:STDOUT:   %complete_type.0da: <witness> = complete_type_witness %C [concrete]
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
// CHECK:STDOUT:   %Main.import_ref.80d: <witness> = import_ref Main//adapt_generic_type, loc6_1, loaded [symbolic = @Adapter.%complete_type (constants.%complete_type.1aa)]
// CHECK:STDOUT:   %Main.import_ref.7c1 = import_ref Main//adapt_generic_type, inst{{[0-9A-F]+}} [no loc], unloaded
// CHECK:STDOUT:   %Main.import_ref.b3b: type = import_ref Main//adapt_generic_type, loc4_16, loaded [symbolic = @Adapter.%T (constants.%T.67d)]
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/parts/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT:   %Core.Copy: type = import_ref Core//prelude/parts/copy, Copy, loaded [concrete = constants.%Copy.type]
// CHECK:STDOUT:   %Core.import_ref.809: @Int.as.Copy.impl.%Int.as.Copy.impl.Op.type (%Int.as.Copy.impl.Op.type.b5d) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.Copy.impl.%Int.as.Copy.impl.Op (constants.%Int.as.Copy.impl.Op.c85)]
// CHECK:STDOUT:   %Copy.impl_witness_table.07a = impl_witness_table (%Core.import_ref.809), @Int.as.Copy.impl [concrete]
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
// CHECK:STDOUT:     %a.param_patt: %pattern_type.9f4 = value_param_pattern [concrete]
// CHECK:STDOUT:     %a.patt: %pattern_type.9f4 = at_binding_pattern a, %a.param_patt [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.c6b = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.c6b = return_slot_pattern %return.param_patt, %i32.loc6_40 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %i32.loc6_40: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc6_40: Core.Form = init_form %i32.loc6_40 [concrete = constants.%.70c]
// CHECK:STDOUT:     %a.param: %Adapter.d3c = value_param call_param0
// CHECK:STDOUT:     %.loc6_34: type = splice_block %Adapter [concrete = constants.%Adapter.d3c] {
// CHECK:STDOUT:       %Adapter.ref: %Adapter.type = name_ref Adapter, imports.%Main.Adapter [concrete = constants.%Adapter.generic]
// CHECK:STDOUT:       %i32.loc6_31: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:       %Adapter: type = class_type @Adapter, @Adapter(constants.%i32) [concrete = constants.%Adapter.d3c]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a: %Adapter.d3c = value_binding a, %a.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   %ImportedConvertLocal.decl: %ImportedConvertLocal.type = fn_decl @ImportedConvertLocal [concrete = constants.%ImportedConvertLocal] {
// CHECK:STDOUT:     %a.param_patt: %pattern_type.aca = value_param_pattern [concrete]
// CHECK:STDOUT:     %a.patt: %pattern_type.aca = at_binding_pattern a, %a.param_patt [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.c6b = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.c6b = return_slot_pattern %return.param_patt, %i32 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc14_43: Core.Form = init_form %i32 [concrete = constants.%.70c]
// CHECK:STDOUT:     %a.param: %Adapter.5e2 = value_param call_param0
// CHECK:STDOUT:     %.loc14_37: type = splice_block %Adapter [concrete = constants.%Adapter.5e2] {
// CHECK:STDOUT:       %Adapter.ref: %Adapter.type = name_ref Adapter, imports.%Main.Adapter [concrete = constants.%Adapter.generic]
// CHECK:STDOUT:       %C.ref.loc14: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:       %Adapter: type = class_type @Adapter, @Adapter(constants.%C) [concrete = constants.%Adapter.5e2]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a: %Adapter.5e2 = value_binding a, %a.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @Adapter(imports.%Main.import_ref.b3b: type) [from "adapt_generic_type.carbon"] {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T.67d)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T [symbolic = %require_complete (constants.%require_complete.944)]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %T [symbolic = %complete_type (constants.%complete_type.1aa)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     complete_type_witness = imports.%Main.import_ref.80d
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%Main.import_ref.7c1
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %.loc11: %C.elem = field_decl n, element0 [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%struct_type.n [concrete = constants.%complete_type.059]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT:   .n = %.loc11
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ImportedConvert(%a.param: %Adapter.d3c) -> out %return.param: %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: %Adapter.d3c = name_ref a, %a
// CHECK:STDOUT:   %i32.loc7: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %.loc7_12.1: %i32 = as_compatible %a.ref
// CHECK:STDOUT:   %.loc7_12.2: %i32 = converted %a.ref, %.loc7_12.1
// CHECK:STDOUT:   %impl.elem0: %.a5f = impl_witness_access constants.%Copy.impl_witness.32d, element0 [concrete = constants.%Int.as.Copy.impl.Op.c4a]
// CHECK:STDOUT:   %bound_method.loc7_12.1: <bound method> = bound_method %.loc7_12.2, %impl.elem0
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc7_12.2: <bound method> = bound_method %.loc7_12.2, %specific_fn
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call: init %i32 = call %bound_method.loc7_12.2(%.loc7_12.2)
// CHECK:STDOUT:   return %Int.as.Copy.impl.Op.call
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ImportedConvertLocal(%a.param: %Adapter.5e2) -> out %return.param: %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: %Adapter.5e2 = name_ref a, %a
// CHECK:STDOUT:   %C.ref.loc15: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %.loc15_13.1: %C = as_compatible %a.ref
// CHECK:STDOUT:   %.loc15_13.2: %C = converted %a.ref, %.loc15_13.1
// CHECK:STDOUT:   %n.ref: %C.elem = name_ref n, @C.%.loc11 [concrete = @C.%.loc11]
// CHECK:STDOUT:   %.loc15_18.1: ref %i32 = class_element_access %.loc15_13.2, element0
// CHECK:STDOUT:   %.loc15_18.2: %i32 = acquire_value %.loc15_18.1
// CHECK:STDOUT:   %impl.elem0: %.a5f = impl_witness_access constants.%Copy.impl_witness.32d, element0 [concrete = constants.%Int.as.Copy.impl.Op.c4a]
// CHECK:STDOUT:   %bound_method.loc15_18.1: <bound method> = bound_method %.loc15_18.2, %impl.elem0
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc15_18.2: <bound method> = bound_method %.loc15_18.2, %specific_fn
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call: init %i32 = call %bound_method.loc15_18.2(%.loc15_18.2)
// CHECK:STDOUT:   return %Int.as.Copy.impl.Op.call
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Adapter(constants.%T.67d) {
// CHECK:STDOUT:   %T => constants.%T.67d
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Adapter(constants.%i32) {
// CHECK:STDOUT:   %T => constants.%i32
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.f8a
// CHECK:STDOUT:   %complete_type => constants.%complete_type.634
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Adapter(constants.%C) {
// CHECK:STDOUT:   %T => constants.%C
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.059
// CHECK:STDOUT:   %complete_type => constants.%complete_type.0da
// CHECK:STDOUT: }
// CHECK:STDOUT:
