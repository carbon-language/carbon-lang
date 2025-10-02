// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/convert.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/generic/member_out_of_line.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/generic/member_out_of_line.carbon

// --- basic.carbon

library "[[@TEST_NAME]]";

//@dump-sem-ir-begin
class Class(T:! Core.Copy) {
  fn F(n: T) -> T;
  fn G[self: Self]() -> T;
  var n: T;
}

fn Class(T:! Core.Copy).F(n: T) -> T {
  return n;
}

fn Class(T:! Core.Copy).G[self: Self]() -> T {
  return self.n;
}
//@dump-sem-ir-end

// --- nested.carbon

library "[[@TEST_NAME]]";

//@dump-sem-ir-begin
class A(T:! type) {
  class B(N:! T) {
    fn F[self: Self](a: T);
  }
}

fn A(T:! type).B(N:! T).F[self: Self](a: T) {}
//@dump-sem-ir-end

// --- fail_mismatched_not_generic_vs_generic.carbon

library "[[@TEST_NAME]]";

class NotGeneric {
  fn F();
}

// CHECK:STDERR: fail_mismatched_not_generic_vs_generic.carbon:[[@LINE+7]]:4: error: redeclaration differs because of parameter list [RedeclParamListDiffers]
// CHECK:STDERR: fn NotGeneric(T:! type).F() {}
// CHECK:STDERR:    ^~~~~~~~~~
// CHECK:STDERR: fail_mismatched_not_generic_vs_generic.carbon:[[@LINE-7]]:1: note: previously declared without parameter list [RedeclParamListPrevious]
// CHECK:STDERR: class NotGeneric {
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
fn NotGeneric(T:! type).F() {}

// --- fail_mismatched_too_few_args.carbon

library "[[@TEST_NAME]]";

class Generic(T:! type) {
  fn TooFew();
}

// CHECK:STDERR: fail_mismatched_too_few_args.carbon:[[@LINE+7]]:4: error: redeclaration differs because of parameter count of 0 [RedeclParamCountDiffers]
// CHECK:STDERR: fn Generic().TooFew() {}
// CHECK:STDERR:    ^~~~~~~
// CHECK:STDERR: fail_mismatched_too_few_args.carbon:[[@LINE-7]]:1: note: previously declared with parameter count of 1 [RedeclParamCountPrevious]
// CHECK:STDERR: class Generic(T:! type) {
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
fn Generic().TooFew() {}

// --- fail_mismatched_too_many_args.carbon

library "[[@TEST_NAME]]";

class Generic(T:! type) {
  fn TooMany();
}

// CHECK:STDERR: fail_mismatched_too_many_args.carbon:[[@LINE+7]]:4: error: redeclaration differs because of parameter count of 2 [RedeclParamCountDiffers]
// CHECK:STDERR: fn Generic(T:! type, U:! type).TooMany() {}
// CHECK:STDERR:    ^~~~~~~
// CHECK:STDERR: fail_mismatched_too_many_args.carbon:[[@LINE-7]]:1: note: previously declared with parameter count of 1 [RedeclParamCountPrevious]
// CHECK:STDERR: class Generic(T:! type) {
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
fn Generic(T:! type, U:! type).TooMany() {}

// --- fail_mismatched_wrong_arg_type.carbon

library "[[@TEST_NAME]]";

class Generic(T:! type) {
  fn WrongType();
}

// CHECK:STDERR: fail_mismatched_wrong_arg_type.carbon:[[@LINE+7]]:12: error: type `<pattern for ()>` of parameter 1 in redeclaration differs from previous parameter type `<pattern for type>` [RedeclParamDiffersType]
// CHECK:STDERR: fn Generic(T:! ()).WrongType() {}
// CHECK:STDERR:            ^
// CHECK:STDERR: fail_mismatched_wrong_arg_type.carbon:[[@LINE-7]]:15: note: previous declaration's corresponding parameter here [RedeclParamPrevious]
// CHECK:STDERR: class Generic(T:! type) {
// CHECK:STDERR:               ^
// CHECK:STDERR:
fn Generic(T:! ()).WrongType() {}

// CHECK:STDOUT: --- basic.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %T.be8: %Copy.type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.322: type = pattern_type %Copy.type [concrete]
// CHECK:STDOUT:   %Class.type: type = generic_class_type @Class [concrete]
// CHECK:STDOUT:   %Class.generic: %Class.type = struct_value () [concrete]
// CHECK:STDOUT:   %Class: type = class_type @Class, @Class(%T.be8) [symbolic]
// CHECK:STDOUT:   %T.as_type: type = facet_access_type %T.be8 [symbolic]
// CHECK:STDOUT:   %pattern_type.965801.1: type = pattern_type %T.as_type [symbolic]
// CHECK:STDOUT:   %Class.F.type: type = fn_type @Class.F, @Class(%T.be8) [symbolic]
// CHECK:STDOUT:   %Class.F: %Class.F.type = struct_value () [symbolic]
// CHECK:STDOUT:   %pattern_type.38d: type = pattern_type %Class [symbolic]
// CHECK:STDOUT:   %Class.G.type: type = fn_type @Class.G, @Class(%T.be8) [symbolic]
// CHECK:STDOUT:   %Class.G: %Class.G.type = struct_value () [symbolic]
// CHECK:STDOUT:   %require_complete.07c: <witness> = require_complete_type %T.as_type [symbolic]
// CHECK:STDOUT:   %Class.elem: type = unbound_element_type %Class, %T.as_type [symbolic]
// CHECK:STDOUT:   %struct_type.n: type = struct_type {.n: %T.as_type} [symbolic]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.n [symbolic]
// CHECK:STDOUT:   %Copy.Op.type: type = fn_type @Copy.Op [concrete]
// CHECK:STDOUT:   %Copy.lookup_impl_witness.e15: <witness> = lookup_impl_witness %T.be8, @Copy [symbolic]
// CHECK:STDOUT:   %.427: type = fn_type_with_self_type %Copy.Op.type, %T.be8 [symbolic]
// CHECK:STDOUT:   %impl.elem0.168: %.427 = impl_witness_access %Copy.lookup_impl_witness.e15, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.2ce: <specific function> = specific_impl_function %impl.elem0.168, @Copy.Op(%T.be8) [symbolic]
// CHECK:STDOUT:   %require_complete.3e3: <witness> = require_complete_type %Class [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Copy = %Core.Copy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Copy: type = import_ref Core//prelude/parts/copy, Copy, loaded [concrete = constants.%Copy.type]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %Class.decl: %Class.type = class_decl @Class [concrete = constants.%Class.generic] {
// CHECK:STDOUT:     %T.patt: %pattern_type.322 = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc5: type = splice_block %Copy.ref [concrete = constants.%Copy.type] {
// CHECK:STDOUT:       <elided>
// CHECK:STDOUT:       %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:       %Copy.ref: type = name_ref Copy, imports.%Core.Copy [concrete = constants.%Copy.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc5_13.2: %Copy.type = bind_symbolic_name T, 0 [symbolic = %T.loc5_13.1 (constants.%T.be8)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Class.F.decl: %Class.F.type = fn_decl @Class.F [symbolic = constants.%Class.F] {
// CHECK:STDOUT:     %n.patt: @Class.F.%pattern_type (%pattern_type.965801.1) = binding_pattern n [concrete]
// CHECK:STDOUT:     %n.param_patt: @Class.F.%pattern_type (%pattern_type.965801.1) = value_param_pattern %n.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: @Class.F.%pattern_type (%pattern_type.965801.1) = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: @Class.F.%pattern_type (%pattern_type.965801.1) = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc11_18: type = splice_block %Copy.ref [concrete = constants.%Copy.type] {
// CHECK:STDOUT:       <elided>
// CHECK:STDOUT:       %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:       %Copy.ref: type = name_ref Copy, imports.%Core.Copy [concrete = constants.%Copy.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc11: %Copy.type = bind_symbolic_name T, 0 [symbolic = @Class.%T.loc5_13.1 (constants.%T.be8)]
// CHECK:STDOUT:     %T.ref.loc11_36: %Copy.type = name_ref T, %T.loc11 [symbolic = %T.loc6 (constants.%T.be8)]
// CHECK:STDOUT:     %T.as_type.loc11_36: type = facet_access_type %T.ref.loc11_36 [symbolic = %T.as_type.loc6_11.1 (constants.%T.as_type)]
// CHECK:STDOUT:     %.loc11_36: type = converted %T.ref.loc11_36, %T.as_type.loc11_36 [symbolic = %T.as_type.loc6_11.1 (constants.%T.as_type)]
// CHECK:STDOUT:     %n.param.loc11: @Class.F.%T.as_type.loc6_11.1 (%T.as_type) = value_param call_param0
// CHECK:STDOUT:     %.loc11_30.1: type = splice_block %.loc11_30.2 [symbolic = %T.as_type.loc6_11.1 (constants.%T.as_type)] {
// CHECK:STDOUT:       %T.ref.loc11_30: %Copy.type = name_ref T, %T.loc11 [symbolic = %T.loc6 (constants.%T.be8)]
// CHECK:STDOUT:       %T.as_type.loc11_30: type = facet_access_type %T.ref.loc11_30 [symbolic = %T.as_type.loc6_11.1 (constants.%T.as_type)]
// CHECK:STDOUT:       %.loc11_30.2: type = converted %T.ref.loc11_30, %T.as_type.loc11_30 [symbolic = %T.as_type.loc6_11.1 (constants.%T.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %n.loc11: @Class.F.%T.as_type.loc6_11.1 (%T.as_type) = bind_name n, %n.param.loc11
// CHECK:STDOUT:     %return.param.loc11: ref @Class.F.%T.as_type.loc6_11.1 (%T.as_type) = out_param call_param1
// CHECK:STDOUT:     %return.loc11: ref @Class.F.%T.as_type.loc6_11.1 (%T.as_type) = return_slot %return.param.loc11
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Class.G.decl: %Class.G.type = fn_decl @Class.G [symbolic = constants.%Class.G] {
// CHECK:STDOUT:     %self.patt: @Class.G.%pattern_type.loc7_8 (%pattern_type.38d) = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: @Class.G.%pattern_type.loc7_8 (%pattern_type.38d) = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: @Class.G.%pattern_type.loc7_22 (%pattern_type.965801.1) = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: @Class.G.%pattern_type.loc7_22 (%pattern_type.965801.1) = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc15_18: type = splice_block %Copy.ref [concrete = constants.%Copy.type] {
// CHECK:STDOUT:       <elided>
// CHECK:STDOUT:       %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:       %Copy.ref: type = name_ref Copy, imports.%Core.Copy [concrete = constants.%Copy.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc15: %Copy.type = bind_symbolic_name T, 0 [symbolic = @Class.%T.loc5_13.1 (constants.%T.be8)]
// CHECK:STDOUT:     %T.ref.loc15: %Copy.type = name_ref T, %T.loc15 [symbolic = %T.loc7 (constants.%T.be8)]
// CHECK:STDOUT:     %T.as_type.loc15: type = facet_access_type %T.ref.loc15 [symbolic = %T.as_type.loc7_25.1 (constants.%T.as_type)]
// CHECK:STDOUT:     %.loc15_44: type = converted %T.ref.loc15, %T.as_type.loc15 [symbolic = %T.as_type.loc7_25.1 (constants.%T.as_type)]
// CHECK:STDOUT:     %self.param.loc15: @Class.G.%Class (%Class) = value_param call_param0
// CHECK:STDOUT:     %.loc15_33.1: type = splice_block %Self.ref.loc15 [symbolic = %Class (constants.%Class)] {
// CHECK:STDOUT:       %.loc15_33.2: type = specific_constant constants.%Class, @Class(constants.%T.be8) [symbolic = %Class (constants.%Class)]
// CHECK:STDOUT:       %Self.ref.loc15: type = name_ref Self, %.loc15_33.2 [symbolic = %Class (constants.%Class)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %self.loc15: @Class.G.%Class (%Class) = bind_name self, %self.param.loc15
// CHECK:STDOUT:     %return.param.loc15: ref @Class.G.%T.as_type.loc7_25.1 (%T.as_type) = out_param call_param1
// CHECK:STDOUT:     %return.loc15: ref @Class.G.%T.as_type.loc7_25.1 (%T.as_type) = return_slot %return.param.loc15
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @Class(%T.loc5_13.2: %Copy.type) {
// CHECK:STDOUT:   %T.loc5_13.1: %Copy.type = bind_symbolic_name T, 0 [symbolic = %T.loc5_13.1 (constants.%T.be8)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Class.F.type: type = fn_type @Class.F, @Class(%T.loc5_13.1) [symbolic = %Class.F.type (constants.%Class.F.type)]
// CHECK:STDOUT:   %Class.F: @Class.%Class.F.type (%Class.F.type) = struct_value () [symbolic = %Class.F (constants.%Class.F)]
// CHECK:STDOUT:   %Class.G.type: type = fn_type @Class.G, @Class(%T.loc5_13.1) [symbolic = %Class.G.type (constants.%Class.G.type)]
// CHECK:STDOUT:   %Class.G: @Class.%Class.G.type (%Class.G.type) = struct_value () [symbolic = %Class.G (constants.%Class.G)]
// CHECK:STDOUT:   %T.as_type.loc8_10.2: type = facet_access_type %T.loc5_13.1 [symbolic = %T.as_type.loc8_10.2 (constants.%T.as_type)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T.as_type.loc8_10.2 [symbolic = %require_complete (constants.%require_complete.07c)]
// CHECK:STDOUT:   %Class: type = class_type @Class, @Class(%T.loc5_13.1) [symbolic = %Class (constants.%Class)]
// CHECK:STDOUT:   %Class.elem: type = unbound_element_type %Class, %T.as_type.loc8_10.2 [symbolic = %Class.elem (constants.%Class.elem)]
// CHECK:STDOUT:   %struct_type.n: type = struct_type {.n: @Class.%T.as_type.loc8_10.2 (%T.as_type)} [symbolic = %struct_type.n (constants.%struct_type.n)]
// CHECK:STDOUT:   %complete_type.loc9_1.2: <witness> = complete_type_witness %struct_type.n [symbolic = %complete_type.loc9_1.2 (constants.%complete_type)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %Class.F.decl: @Class.%Class.F.type (%Class.F.type) = fn_decl @Class.F [symbolic = @Class.%Class.F (constants.%Class.F)] {
// CHECK:STDOUT:       %n.patt: @Class.F.%pattern_type (%pattern_type.965801.1) = binding_pattern n [concrete]
// CHECK:STDOUT:       %n.param_patt: @Class.F.%pattern_type (%pattern_type.965801.1) = value_param_pattern %n.patt, call_param0 [concrete]
// CHECK:STDOUT:       %return.patt: @Class.F.%pattern_type (%pattern_type.965801.1) = return_slot_pattern [concrete]
// CHECK:STDOUT:       %return.param_patt: @Class.F.%pattern_type (%pattern_type.965801.1) = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %T.ref.loc6_17: %Copy.type = name_ref T, @Class.%T.loc5_13.2 [symbolic = %T.loc6 (constants.%T.be8)]
// CHECK:STDOUT:       %T.as_type.loc6_17: type = facet_access_type %T.ref.loc6_17 [symbolic = %T.as_type.loc6_11.1 (constants.%T.as_type)]
// CHECK:STDOUT:       %.loc6_17: type = converted %T.ref.loc6_17, %T.as_type.loc6_17 [symbolic = %T.as_type.loc6_11.1 (constants.%T.as_type)]
// CHECK:STDOUT:       %n.param.loc6: @Class.F.%T.as_type.loc6_11.1 (%T.as_type) = value_param call_param0
// CHECK:STDOUT:       %.loc6_11.1: type = splice_block %.loc6_11.2 [symbolic = %T.as_type.loc6_11.1 (constants.%T.as_type)] {
// CHECK:STDOUT:         %T.ref.loc6_11: %Copy.type = name_ref T, @Class.%T.loc5_13.2 [symbolic = %T.loc6 (constants.%T.be8)]
// CHECK:STDOUT:         %T.as_type.loc6_11.2: type = facet_access_type %T.ref.loc6_11 [symbolic = %T.as_type.loc6_11.1 (constants.%T.as_type)]
// CHECK:STDOUT:         %.loc6_11.2: type = converted %T.ref.loc6_11, %T.as_type.loc6_11.2 [symbolic = %T.as_type.loc6_11.1 (constants.%T.as_type)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %n.loc6: @Class.F.%T.as_type.loc6_11.1 (%T.as_type) = bind_name n, %n.param.loc6
// CHECK:STDOUT:       %return.param.loc6: ref @Class.F.%T.as_type.loc6_11.1 (%T.as_type) = out_param call_param1
// CHECK:STDOUT:       %return.loc6: ref @Class.F.%T.as_type.loc6_11.1 (%T.as_type) = return_slot %return.param.loc6
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %Class.G.decl: @Class.%Class.G.type (%Class.G.type) = fn_decl @Class.G [symbolic = @Class.%Class.G (constants.%Class.G)] {
// CHECK:STDOUT:       %self.patt: @Class.G.%pattern_type.loc7_8 (%pattern_type.38d) = binding_pattern self [concrete]
// CHECK:STDOUT:       %self.param_patt: @Class.G.%pattern_type.loc7_8 (%pattern_type.38d) = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:       %return.patt: @Class.G.%pattern_type.loc7_22 (%pattern_type.965801.1) = return_slot_pattern [concrete]
// CHECK:STDOUT:       %return.param_patt: @Class.G.%pattern_type.loc7_22 (%pattern_type.965801.1) = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %T.ref.loc7: %Copy.type = name_ref T, @Class.%T.loc5_13.2 [symbolic = %T.loc7 (constants.%T.be8)]
// CHECK:STDOUT:       %T.as_type.loc7_25.2: type = facet_access_type %T.ref.loc7 [symbolic = %T.as_type.loc7_25.1 (constants.%T.as_type)]
// CHECK:STDOUT:       %.loc7_25: type = converted %T.ref.loc7, %T.as_type.loc7_25.2 [symbolic = %T.as_type.loc7_25.1 (constants.%T.as_type)]
// CHECK:STDOUT:       %self.param.loc7: @Class.G.%Class (%Class) = value_param call_param0
// CHECK:STDOUT:       %.loc7_14.1: type = splice_block %Self.ref.loc7 [symbolic = %Class (constants.%Class)] {
// CHECK:STDOUT:         %.loc7_14.2: type = specific_constant constants.%Class, @Class(constants.%T.be8) [symbolic = %Class (constants.%Class)]
// CHECK:STDOUT:         %Self.ref.loc7: type = name_ref Self, %.loc7_14.2 [symbolic = %Class (constants.%Class)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %self.loc7: @Class.G.%Class (%Class) = bind_name self, %self.param.loc7
// CHECK:STDOUT:       %return.param.loc7: ref @Class.G.%T.as_type.loc7_25.1 (%T.as_type) = out_param call_param1
// CHECK:STDOUT:       %return.loc7: ref @Class.G.%T.as_type.loc7_25.1 (%T.as_type) = return_slot %return.param.loc7
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.ref: %Copy.type = name_ref T, %T.loc5_13.2 [symbolic = %T.loc5_13.1 (constants.%T.be8)]
// CHECK:STDOUT:     %T.as_type.loc8_10.1: type = facet_access_type %T.ref [symbolic = %T.as_type.loc8_10.2 (constants.%T.as_type)]
// CHECK:STDOUT:     %.loc8_10: type = converted %T.ref, %T.as_type.loc8_10.1 [symbolic = %T.as_type.loc8_10.2 (constants.%T.as_type)]
// CHECK:STDOUT:     %.loc8_8: @Class.%Class.elem (%Class.elem) = field_decl n, element0 [concrete]
// CHECK:STDOUT:     %complete_type.loc9_1.1: <witness> = complete_type_witness constants.%struct_type.n [symbolic = %complete_type.loc9_1.2 (constants.%complete_type)]
// CHECK:STDOUT:     complete_type_witness = %complete_type.loc9_1.1
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%Class
// CHECK:STDOUT:     .T = <poisoned>
// CHECK:STDOUT:     .F = %Class.F.decl
// CHECK:STDOUT:     .G = %Class.G.decl
// CHECK:STDOUT:     .n = %.loc8_8
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Class.F(@Class.%T.loc5_13.2: %Copy.type) {
// CHECK:STDOUT:   %T.loc6: %Copy.type = bind_symbolic_name T, 0 [symbolic = %T.loc6 (constants.%T.be8)]
// CHECK:STDOUT:   %T.as_type.loc6_11.1: type = facet_access_type %T.loc6 [symbolic = %T.as_type.loc6_11.1 (constants.%T.as_type)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %T.as_type.loc6_11.1 [symbolic = %pattern_type (constants.%pattern_type.965801.1)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T.as_type.loc6_11.1 [symbolic = %require_complete (constants.%require_complete.07c)]
// CHECK:STDOUT:   %Copy.lookup_impl_witness: <witness> = lookup_impl_witness %T.loc6, @Copy [symbolic = %Copy.lookup_impl_witness (constants.%Copy.lookup_impl_witness.e15)]
// CHECK:STDOUT:   %.loc12_10.2: type = fn_type_with_self_type constants.%Copy.Op.type, %T.loc6 [symbolic = %.loc12_10.2 (constants.%.427)]
// CHECK:STDOUT:   %impl.elem0.loc12_10.2: @Class.F.%.loc12_10.2 (%.427) = impl_witness_access %Copy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc12_10.2 (constants.%impl.elem0.168)]
// CHECK:STDOUT:   %specific_impl_fn.loc12_10.2: <specific function> = specific_impl_function %impl.elem0.loc12_10.2, @Copy.Op(%T.loc6) [symbolic = %specific_impl_fn.loc12_10.2 (constants.%specific_impl_fn.2ce)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%n.param.loc11: @Class.F.%T.as_type.loc6_11.1 (%T.as_type)) -> %return.param.loc11: @Class.F.%T.as_type.loc6_11.1 (%T.as_type) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %n.ref: @Class.F.%T.as_type.loc6_11.1 (%T.as_type) = name_ref n, %n.loc11
// CHECK:STDOUT:     %impl.elem0.loc12_10.1: @Class.F.%.loc12_10.2 (%.427) = impl_witness_access constants.%Copy.lookup_impl_witness.e15, element0 [symbolic = %impl.elem0.loc12_10.2 (constants.%impl.elem0.168)]
// CHECK:STDOUT:     %bound_method.loc12_10.1: <bound method> = bound_method %n.ref, %impl.elem0.loc12_10.1
// CHECK:STDOUT:     %specific_impl_fn.loc12_10.1: <specific function> = specific_impl_function %impl.elem0.loc12_10.1, @Copy.Op(constants.%T.be8) [symbolic = %specific_impl_fn.loc12_10.2 (constants.%specific_impl_fn.2ce)]
// CHECK:STDOUT:     %bound_method.loc12_10.2: <bound method> = bound_method %n.ref, %specific_impl_fn.loc12_10.1
// CHECK:STDOUT:     %.loc11_33: ref @Class.F.%T.as_type.loc6_11.1 (%T.as_type) = splice_block %return.loc11 {}
// CHECK:STDOUT:     %.loc12_10.1: init @Class.F.%T.as_type.loc6_11.1 (%T.as_type) = call %bound_method.loc12_10.2(%n.ref) to %.loc11_33
// CHECK:STDOUT:     return %.loc12_10.1 to %return.loc11
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Class.G(@Class.%T.loc5_13.2: %Copy.type) {
// CHECK:STDOUT:   %T.loc7: %Copy.type = bind_symbolic_name T, 0 [symbolic = %T.loc7 (constants.%T.be8)]
// CHECK:STDOUT:   %Class: type = class_type @Class, @Class(%T.loc7) [symbolic = %Class (constants.%Class)]
// CHECK:STDOUT:   %pattern_type.loc7_8: type = pattern_type %Class [symbolic = %pattern_type.loc7_8 (constants.%pattern_type.38d)]
// CHECK:STDOUT:   %T.as_type.loc7_25.1: type = facet_access_type %T.loc7 [symbolic = %T.as_type.loc7_25.1 (constants.%T.as_type)]
// CHECK:STDOUT:   %pattern_type.loc7_22: type = pattern_type %T.as_type.loc7_25.1 [symbolic = %pattern_type.loc7_22 (constants.%pattern_type.965801.1)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc15: <witness> = require_complete_type %Class [symbolic = %require_complete.loc15 (constants.%require_complete.3e3)]
// CHECK:STDOUT:   %Class.elem: type = unbound_element_type %Class, %T.as_type.loc7_25.1 [symbolic = %Class.elem (constants.%Class.elem)]
// CHECK:STDOUT:   %require_complete.loc16: <witness> = require_complete_type %T.as_type.loc7_25.1 [symbolic = %require_complete.loc16 (constants.%require_complete.07c)]
// CHECK:STDOUT:   %Copy.lookup_impl_witness: <witness> = lookup_impl_witness %T.loc7, @Copy [symbolic = %Copy.lookup_impl_witness (constants.%Copy.lookup_impl_witness.e15)]
// CHECK:STDOUT:   %.loc16_14.4: type = fn_type_with_self_type constants.%Copy.Op.type, %T.loc7 [symbolic = %.loc16_14.4 (constants.%.427)]
// CHECK:STDOUT:   %impl.elem0.loc16_14.2: @Class.G.%.loc16_14.4 (%.427) = impl_witness_access %Copy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc16_14.2 (constants.%impl.elem0.168)]
// CHECK:STDOUT:   %specific_impl_fn.loc16_14.2: <specific function> = specific_impl_function %impl.elem0.loc16_14.2, @Copy.Op(%T.loc7) [symbolic = %specific_impl_fn.loc16_14.2 (constants.%specific_impl_fn.2ce)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param.loc15: @Class.G.%Class (%Class)) -> %return.param.loc15: @Class.G.%T.as_type.loc7_25.1 (%T.as_type) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %self.ref: @Class.G.%Class (%Class) = name_ref self, %self.loc15
// CHECK:STDOUT:     %n.ref: @Class.G.%Class.elem (%Class.elem) = name_ref n, @Class.%.loc8_8 [concrete = @Class.%.loc8_8]
// CHECK:STDOUT:     %.loc16_14.1: ref @Class.G.%T.as_type.loc7_25.1 (%T.as_type) = class_element_access %self.ref, element0
// CHECK:STDOUT:     %.loc16_14.2: @Class.G.%T.as_type.loc7_25.1 (%T.as_type) = bind_value %.loc16_14.1
// CHECK:STDOUT:     %impl.elem0.loc16_14.1: @Class.G.%.loc16_14.4 (%.427) = impl_witness_access constants.%Copy.lookup_impl_witness.e15, element0 [symbolic = %impl.elem0.loc16_14.2 (constants.%impl.elem0.168)]
// CHECK:STDOUT:     %bound_method.loc16_14.1: <bound method> = bound_method %.loc16_14.2, %impl.elem0.loc16_14.1
// CHECK:STDOUT:     %specific_impl_fn.loc16_14.1: <specific function> = specific_impl_function %impl.elem0.loc16_14.1, @Copy.Op(constants.%T.be8) [symbolic = %specific_impl_fn.loc16_14.2 (constants.%specific_impl_fn.2ce)]
// CHECK:STDOUT:     %bound_method.loc16_14.2: <bound method> = bound_method %.loc16_14.2, %specific_impl_fn.loc16_14.1
// CHECK:STDOUT:     %.loc15_41: ref @Class.G.%T.as_type.loc7_25.1 (%T.as_type) = splice_block %return.loc15 {}
// CHECK:STDOUT:     %.loc16_14.3: init @Class.G.%T.as_type.loc7_25.1 (%T.as_type) = call %bound_method.loc16_14.2(%.loc16_14.2) to %.loc15_41
// CHECK:STDOUT:     return %.loc16_14.3 to %return.loc15
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class(constants.%T.be8) {
// CHECK:STDOUT:   %T.loc5_13.1 => constants.%T.be8
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Class.F.type => constants.%Class.F.type
// CHECK:STDOUT:   %Class.F => constants.%Class.F
// CHECK:STDOUT:   %Class.G.type => constants.%Class.G.type
// CHECK:STDOUT:   %Class.G => constants.%Class.G
// CHECK:STDOUT:   %T.as_type.loc8_10.2 => constants.%T.as_type
// CHECK:STDOUT:   %require_complete => constants.%require_complete.07c
// CHECK:STDOUT:   %Class => constants.%Class
// CHECK:STDOUT:   %Class.elem => constants.%Class.elem
// CHECK:STDOUT:   %struct_type.n => constants.%struct_type.n
// CHECK:STDOUT:   %complete_type.loc9_1.2 => constants.%complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class.F(constants.%T.be8) {
// CHECK:STDOUT:   %T.loc6 => constants.%T.be8
// CHECK:STDOUT:   %T.as_type.loc6_11.1 => constants.%T.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.965801.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class.G(constants.%T.be8) {
// CHECK:STDOUT:   %T.loc7 => constants.%T.be8
// CHECK:STDOUT:   %Class => constants.%Class
// CHECK:STDOUT:   %pattern_type.loc7_8 => constants.%pattern_type.38d
// CHECK:STDOUT:   %T.as_type.loc7_25.1 => constants.%T.as_type
// CHECK:STDOUT:   %pattern_type.loc7_22 => constants.%pattern_type.965801.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- nested.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %A.type: type = generic_class_type @A [concrete]
// CHECK:STDOUT:   %A.generic: %A.type = struct_value () [concrete]
// CHECK:STDOUT:   %A: type = class_type @A, @A(%T) [symbolic]
// CHECK:STDOUT:   %N: %T = bind_symbolic_name N, 1 [symbolic]
// CHECK:STDOUT:   %pattern_type.7dc: type = pattern_type %T [symbolic]
// CHECK:STDOUT:   %B.type: type = generic_class_type @B, @A(%T) [symbolic]
// CHECK:STDOUT:   %B.generic: %B.type = struct_value () [symbolic]
// CHECK:STDOUT:   %B: type = class_type @B, @B(%T, %N) [symbolic]
// CHECK:STDOUT:   %pattern_type.13e: type = pattern_type %B [symbolic]
// CHECK:STDOUT:   %B.F.type: type = fn_type @B.F, @B(%T, %N) [symbolic]
// CHECK:STDOUT:   %B.F: %B.F.type = struct_value () [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %require_complete.fca: <witness> = require_complete_type %B [symbolic]
// CHECK:STDOUT:   %require_complete.4ae: <witness> = require_complete_type %T [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %A.decl: %A.type = class_decl @A [concrete = constants.%A.generic] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %T.loc5_9.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc5_9.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %B.F.decl: %B.F.type = fn_decl @B.F [symbolic = constants.%B.F] {
// CHECK:STDOUT:     %self.patt: @B.F.%pattern_type.loc7_10 (%pattern_type.13e) = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: @B.F.%pattern_type.loc7_10 (%pattern_type.13e) = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %a.patt: @B.F.%pattern_type.loc7_22 (%pattern_type.7dc) = binding_pattern a [concrete]
// CHECK:STDOUT:     %a.param_patt: @B.F.%pattern_type.loc7_22 (%pattern_type.7dc) = value_param_pattern %a.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %T.loc11: type = bind_symbolic_name T, 0 [symbolic = @A.%T.loc5_9.1 (constants.%T)]
// CHECK:STDOUT:     %.loc11_22: type = splice_block %T.ref.loc11_22 [symbolic = @B.%T (constants.%T)] {
// CHECK:STDOUT:       <elided>
// CHECK:STDOUT:       %T.ref.loc11_22: type = name_ref T, %T.loc11 [symbolic = @B.%T (constants.%T)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %N.loc11: @B.%T (%T) = bind_symbolic_name N, 1 [symbolic = @B.%N.loc6_11.1 (constants.%N)]
// CHECK:STDOUT:     %self.param.loc11: @B.F.%B (%B) = value_param call_param0
// CHECK:STDOUT:     %.loc11_33.1: type = splice_block %Self.ref.loc11 [symbolic = %B (constants.%B)] {
// CHECK:STDOUT:       %.loc11_33.2: type = specific_constant constants.%B, @B(constants.%T, constants.%N) [symbolic = %B (constants.%B)]
// CHECK:STDOUT:       %Self.ref.loc11: type = name_ref Self, %.loc11_33.2 [symbolic = %B (constants.%B)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %self.loc11: @B.F.%B (%B) = bind_name self, %self.param.loc11
// CHECK:STDOUT:     %a.param.loc11: @B.F.%T.loc7 (%T) = value_param call_param1
// CHECK:STDOUT:     %T.ref.loc11_42: type = name_ref T, %T.loc11 [symbolic = %T.loc7 (constants.%T)]
// CHECK:STDOUT:     %a.loc11: @B.F.%T.loc7 (%T) = bind_name a, %a.param.loc11
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @A(%T.loc5_9.2: type) {
// CHECK:STDOUT:   %T.loc5_9.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc5_9.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %B.type: type = generic_class_type @B, @A(%T.loc5_9.1) [symbolic = %B.type (constants.%B.type)]
// CHECK:STDOUT:   %B.generic: @A.%B.type (%B.type) = struct_value () [symbolic = %B.generic (constants.%B.generic)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %B.decl: @A.%B.type (%B.type) = class_decl @B [symbolic = @A.%B.generic (constants.%B.generic)] {
// CHECK:STDOUT:       %N.patt: @B.%pattern_type (%pattern_type.7dc) = symbolic_binding_pattern N, 1 [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %.loc6: type = splice_block %T.ref [symbolic = %T (constants.%T)] {
// CHECK:STDOUT:         <elided>
// CHECK:STDOUT:         %T.ref: type = name_ref T, @A.%T.loc5_9.2 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %N.loc6_11.2: @B.%T (%T) = bind_symbolic_name N, 1 [symbolic = %N.loc6_11.1 (constants.%N)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%A
// CHECK:STDOUT:     .T = <poisoned>
// CHECK:STDOUT:     .B = %B.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @B(@A.%T.loc5_9.2: type, %N.loc6_11.2: @B.%T (%T)) {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %N.loc6_11.1: @B.%T (%T) = bind_symbolic_name N, 1 [symbolic = %N.loc6_11.1 (constants.%N)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %T [symbolic = %pattern_type (constants.%pattern_type.7dc)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %B.F.type: type = fn_type @B.F, @B(%T, %N.loc6_11.1) [symbolic = %B.F.type (constants.%B.F.type)]
// CHECK:STDOUT:   %B.F: @B.%B.F.type (%B.F.type) = struct_value () [symbolic = %B.F (constants.%B.F)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %B.F.decl: @B.%B.F.type (%B.F.type) = fn_decl @B.F [symbolic = @B.%B.F (constants.%B.F)] {
// CHECK:STDOUT:       %self.patt: @B.F.%pattern_type.loc7_10 (%pattern_type.13e) = binding_pattern self [concrete]
// CHECK:STDOUT:       %self.param_patt: @B.F.%pattern_type.loc7_10 (%pattern_type.13e) = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:       %a.patt: @B.F.%pattern_type.loc7_22 (%pattern_type.7dc) = binding_pattern a [concrete]
// CHECK:STDOUT:       %a.param_patt: @B.F.%pattern_type.loc7_22 (%pattern_type.7dc) = value_param_pattern %a.patt, call_param1 [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %self.param.loc7: @B.F.%B (%B) = value_param call_param0
// CHECK:STDOUT:       %.loc7_16.1: type = splice_block %Self.ref.loc7 [symbolic = %B (constants.%B)] {
// CHECK:STDOUT:         %.loc7_16.2: type = specific_constant constants.%B, @B(constants.%T, constants.%N) [symbolic = %B (constants.%B)]
// CHECK:STDOUT:         %Self.ref.loc7: type = name_ref Self, %.loc7_16.2 [symbolic = %B (constants.%B)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %self.loc7: @B.F.%B (%B) = bind_name self, %self.param.loc7
// CHECK:STDOUT:       %a.param.loc7: @B.F.%T.loc7 (%T) = value_param call_param1
// CHECK:STDOUT:       %T.ref.loc7: type = name_ref T, @A.%T.loc5_9.2 [symbolic = %T.loc7 (constants.%T)]
// CHECK:STDOUT:       %a.loc7: @B.F.%T.loc7 (%T) = bind_name a, %a.param.loc7
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%B
// CHECK:STDOUT:     .T = <poisoned>
// CHECK:STDOUT:     .F = %B.F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @B.F(@A.%T.loc5_9.2: type, @B.%N.loc6_11.2: @B.%T (%T)) {
// CHECK:STDOUT:   %T.loc7: type = bind_symbolic_name T, 0 [symbolic = %T.loc7 (constants.%T)]
// CHECK:STDOUT:   %N.loc7: @B.F.%T.loc7 (%T) = bind_symbolic_name N, 1 [symbolic = %N.loc7 (constants.%N)]
// CHECK:STDOUT:   %B: type = class_type @B, @B(%T.loc7, %N.loc7) [symbolic = %B (constants.%B)]
// CHECK:STDOUT:   %pattern_type.loc7_10: type = pattern_type %B [symbolic = %pattern_type.loc7_10 (constants.%pattern_type.13e)]
// CHECK:STDOUT:   %pattern_type.loc7_22: type = pattern_type %T.loc7 [symbolic = %pattern_type.loc7_22 (constants.%pattern_type.7dc)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc11_31: <witness> = require_complete_type %B [symbolic = %require_complete.loc11_31 (constants.%require_complete.fca)]
// CHECK:STDOUT:   %require_complete.loc11_40: <witness> = require_complete_type %T.loc7 [symbolic = %require_complete.loc11_40 (constants.%require_complete.4ae)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param.loc11: @B.F.%B (%B), %a.param.loc11: @B.F.%T.loc7 (%T)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A(constants.%T) {
// CHECK:STDOUT:   %T.loc5_9.1 => constants.%T
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %B.type => constants.%B.type
// CHECK:STDOUT:   %B.generic => constants.%B.generic
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @B(constants.%T, constants.%N) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %N.loc6_11.1 => constants.%N
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.7dc
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %B.F.type => constants.%B.F.type
// CHECK:STDOUT:   %B.F => constants.%B.F
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @B.F(constants.%T, constants.%N) {
// CHECK:STDOUT:   %T.loc7 => constants.%T
// CHECK:STDOUT:   %N.loc7 => constants.%N
// CHECK:STDOUT:   %B => constants.%B
// CHECK:STDOUT:   %pattern_type.loc7_10 => constants.%pattern_type.13e
// CHECK:STDOUT:   %pattern_type.loc7_22 => constants.%pattern_type.7dc
// CHECK:STDOUT: }
// CHECK:STDOUT:
