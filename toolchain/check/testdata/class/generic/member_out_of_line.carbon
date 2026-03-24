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
  class B(_:! T) {
    fn F[self: Self](a: T);
  }
}

fn A(T:! type).B(_:! T).F[unused self: Self](unused a: T) {}
//@dump-sem-ir-end

// --- fail_mismatched_not_generic_vs_generic.carbon

library "[[@TEST_NAME]]";

class NotGeneric {
  fn F();
}

// CHECK:STDERR: fail_mismatched_not_generic_vs_generic.carbon:[[@LINE+7]]:4: error: redeclaration differs because of parameter list [RedeclParamListDiffers]
// CHECK:STDERR: fn NotGeneric(unused T:! type).F() {}
// CHECK:STDERR:    ^~~~~~~~~~
// CHECK:STDERR: fail_mismatched_not_generic_vs_generic.carbon:[[@LINE-7]]:1: note: previously declared without parameter list [RedeclParamListPrevious]
// CHECK:STDERR: class NotGeneric {
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
fn NotGeneric(unused T:! type).F() {}

// --- fail_mismatched_too_few_args.carbon

library "[[@TEST_NAME]]";

class Generic(unused T:! type) {
  fn TooFew();
}

// CHECK:STDERR: fail_mismatched_too_few_args.carbon:[[@LINE+7]]:4: error: redeclaration differs because of parameter count of 0 [RedeclParamCountDiffers]
// CHECK:STDERR: fn Generic().TooFew() {}
// CHECK:STDERR:    ^~~~~~~
// CHECK:STDERR: fail_mismatched_too_few_args.carbon:[[@LINE-7]]:1: note: previously declared with parameter count of 1 [RedeclParamCountPrevious]
// CHECK:STDERR: class Generic(unused T:! type) {
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
fn Generic().TooFew() {}

// --- fail_mismatched_too_many_args.carbon

library "[[@TEST_NAME]]";

class Generic(T:! type) {
  fn TooMany();
}

// CHECK:STDERR: fail_mismatched_too_many_args.carbon:[[@LINE+7]]:4: error: redeclaration differs because of parameter count of 2 [RedeclParamCountDiffers]
// CHECK:STDERR: fn Generic(unused T:! type, unused U:! type).TooMany() {}
// CHECK:STDERR:    ^~~~~~~
// CHECK:STDERR: fail_mismatched_too_many_args.carbon:[[@LINE-7]]:1: note: previously declared with parameter count of 1 [RedeclParamCountPrevious]
// CHECK:STDERR: class Generic(T:! type) {
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
fn Generic(unused T:! type, unused U:! type).TooMany() {}

// --- fail_mismatched_wrong_arg_type.carbon

library "[[@TEST_NAME]]";

class Generic(unused T:! type) {
  fn WrongType();
}

// CHECK:STDERR: fail_mismatched_wrong_arg_type.carbon:[[@LINE+7]]:19: error: type `<pattern for ()>` of parameter 1 in redeclaration differs from previous parameter type `<pattern for type>` [RedeclParamDiffersType]
// CHECK:STDERR: fn Generic(unused T:! ()).WrongType() {}
// CHECK:STDERR:                   ^
// CHECK:STDERR: fail_mismatched_wrong_arg_type.carbon:[[@LINE-7]]:22: note: previous declaration's corresponding parameter here [RedeclParamPrevious]
// CHECK:STDERR: class Generic(unused T:! type) {
// CHECK:STDERR:                      ^
// CHECK:STDERR:
fn Generic(unused T:! ()).WrongType() {}

// CHECK:STDOUT: --- basic.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %pattern_type.ce2: type = pattern_type %Copy.type [concrete]
// CHECK:STDOUT:   %T.035: %Copy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %Class.type: type = generic_class_type @Class [concrete]
// CHECK:STDOUT:   %Class.generic: %Class.type = struct_value () [concrete]
// CHECK:STDOUT:   %Class: type = class_type @Class, @Class(%T.035) [symbolic]
// CHECK:STDOUT:   %T.binding.as_type: type = symbolic_binding_type T, 0, %T.035 [symbolic]
// CHECK:STDOUT:   %pattern_type.9b9f0c.2: type = pattern_type %T.binding.as_type [symbolic]
// CHECK:STDOUT:   %.076a48.2: Core.Form = init_form %T.binding.as_type [symbolic]
// CHECK:STDOUT:   %Class.F.type: type = fn_type @Class.F, @Class(%T.035) [symbolic]
// CHECK:STDOUT:   %Class.F: %Class.F.type = struct_value () [symbolic]
// CHECK:STDOUT:   %pattern_type.893: type = pattern_type %Class [symbolic]
// CHECK:STDOUT:   %Class.G.type: type = fn_type @Class.G, @Class(%T.035) [symbolic]
// CHECK:STDOUT:   %Class.G: %Class.G.type = struct_value () [symbolic]
// CHECK:STDOUT:   %require_complete.67c: <witness> = require_complete_type %T.binding.as_type [symbolic]
// CHECK:STDOUT:   %Class.elem: type = unbound_element_type %Class, %T.binding.as_type [symbolic]
// CHECK:STDOUT:   %struct_type.n: type = struct_type {.n: %T.binding.as_type} [symbolic]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.n [symbolic]
// CHECK:STDOUT:   %Copy.lookup_impl_witness.58d: <witness> = lookup_impl_witness %T.035, @Copy [symbolic]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.735e75.2: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%T.035) [symbolic]
// CHECK:STDOUT:   %.023: type = fn_type_with_self_type %Copy.WithSelf.Op.type.735e75.2, %T.035 [symbolic]
// CHECK:STDOUT:   %impl.elem0.594: %.023 = impl_witness_access %Copy.lookup_impl_witness.58d, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.bdc: <specific function> = specific_impl_function %impl.elem0.594, @Copy.WithSelf.Op(%T.035) [symbolic]
// CHECK:STDOUT:   %require_complete.904: <witness> = require_complete_type %Class [symbolic]
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
// CHECK:STDOUT:     %T.patt: %pattern_type.ce2 = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc18: type = splice_block %Copy.ref [concrete = constants.%Copy.type] {
// CHECK:STDOUT:       <elided>
// CHECK:STDOUT:       %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:       %Copy.ref: type = name_ref Copy, imports.%Core.Copy [concrete = constants.%Copy.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc18_13.2: %Copy.type = symbolic_binding T, 0 [symbolic = %T.loc18_13.1 (constants.%T.035)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Class.F.decl: %Class.F.type = fn_decl @Class.F [symbolic = constants.%Class.F] {
// CHECK:STDOUT:     %n.param_patt: @Class.F.%pattern_type (%pattern_type.9b9f0c.2) = value_param_pattern [concrete]
// CHECK:STDOUT:     %n.patt: @Class.F.%pattern_type (%pattern_type.9b9f0c.2) = at_binding_pattern n, %n.param_patt [concrete]
// CHECK:STDOUT:     %return.param_patt: @Class.F.%pattern_type (%pattern_type.9b9f0c.2) = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: @Class.F.%pattern_type (%pattern_type.9b9f0c.2) = return_slot_pattern %return.param_patt, %.loc24_36.2 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc24_18: type = splice_block %Copy.ref [concrete = constants.%Copy.type] {
// CHECK:STDOUT:       <elided>
// CHECK:STDOUT:       %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:       %Copy.ref: type = name_ref Copy, imports.%Core.Copy [concrete = constants.%Copy.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc24: %Copy.type = symbolic_binding T, 0 [symbolic = @Class.%T.loc18_13.1 (constants.%T.035)]
// CHECK:STDOUT:     %T.ref.loc24_36: %Copy.type = name_ref T, %T.loc24 [symbolic = %T.loc19 (constants.%T.035)]
// CHECK:STDOUT:     %T.as_type.loc24_36: type = facet_access_type %T.ref.loc24_36 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:     %.loc24_36.2: type = converted %T.ref.loc24_36, %T.as_type.loc24_36 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:     %.loc24_36.3: Core.Form = init_form %.loc24_36.2 [symbolic = %.loc19_17.1 (constants.%.076a48.2)]
// CHECK:STDOUT:     %n.param.loc24: @Class.F.%T.binding.as_type (%T.binding.as_type) = value_param call_param0
// CHECK:STDOUT:     %.loc24_30.1: type = splice_block %.loc24_30.2 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)] {
// CHECK:STDOUT:       %T.ref.loc24_30: %Copy.type = name_ref T, %T.loc24 [symbolic = %T.loc19 (constants.%T.035)]
// CHECK:STDOUT:       %T.as_type.loc24_30: type = facet_access_type %T.ref.loc24_30 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:       %.loc24_30.2: type = converted %T.ref.loc24_30, %T.as_type.loc24_30 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %n.loc24: @Class.F.%T.binding.as_type (%T.binding.as_type) = value_binding n, %n.param.loc24
// CHECK:STDOUT:     %return.param.loc24: ref @Class.F.%T.binding.as_type (%T.binding.as_type) = out_param call_param1
// CHECK:STDOUT:     %return.loc24: ref @Class.F.%T.binding.as_type (%T.binding.as_type) = return_slot %return.param.loc24
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Class.G.decl: %Class.G.type = fn_decl @Class.G [symbolic = constants.%Class.G] {
// CHECK:STDOUT:     %self.param_patt: @Class.G.%pattern_type.loc20_12 (%pattern_type.893) = value_param_pattern [concrete]
// CHECK:STDOUT:     %self.patt: @Class.G.%pattern_type.loc20_12 (%pattern_type.893) = at_binding_pattern self, %self.param_patt [concrete]
// CHECK:STDOUT:     %return.param_patt: @Class.G.%pattern_type.loc20_25 (%pattern_type.9b9f0c.2) = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: @Class.G.%pattern_type.loc20_25 (%pattern_type.9b9f0c.2) = return_slot_pattern %return.param_patt, %.loc28_44.2 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc28_18: type = splice_block %Copy.ref [concrete = constants.%Copy.type] {
// CHECK:STDOUT:       <elided>
// CHECK:STDOUT:       %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:       %Copy.ref: type = name_ref Copy, imports.%Core.Copy [concrete = constants.%Copy.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc28: %Copy.type = symbolic_binding T, 0 [symbolic = @Class.%T.loc18_13.1 (constants.%T.035)]
// CHECK:STDOUT:     %T.ref.loc28: %Copy.type = name_ref T, %T.loc28 [symbolic = %T.loc20 (constants.%T.035)]
// CHECK:STDOUT:     %T.as_type.loc28: type = facet_access_type %T.ref.loc28 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:     %.loc28_44.2: type = converted %T.ref.loc28, %T.as_type.loc28 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:     %.loc28_44.3: Core.Form = init_form %.loc28_44.2 [symbolic = %.loc20_25.1 (constants.%.076a48.2)]
// CHECK:STDOUT:     %self.param.loc28: @Class.G.%Class (%Class) = value_param call_param0
// CHECK:STDOUT:     %.loc28_33.1: type = splice_block %Self.ref.loc28 [symbolic = %Class (constants.%Class)] {
// CHECK:STDOUT:       %.loc28_33.2: type = specific_constant constants.%Class, @Class(constants.%T.035) [symbolic = %Class (constants.%Class)]
// CHECK:STDOUT:       %Self.ref.loc28: type = name_ref Self, %.loc28_33.2 [symbolic = %Class (constants.%Class)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %self.loc28: @Class.G.%Class (%Class) = value_binding self, %self.param.loc28
// CHECK:STDOUT:     %return.param.loc28: ref @Class.G.%T.binding.as_type (%T.binding.as_type) = out_param call_param1
// CHECK:STDOUT:     %return.loc28: ref @Class.G.%T.binding.as_type (%T.binding.as_type) = return_slot %return.param.loc28
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @Class(%T.loc18_13.2: %Copy.type) {
// CHECK:STDOUT:   %T.loc18_13.1: %Copy.type = symbolic_binding T, 0 [symbolic = %T.loc18_13.1 (constants.%T.035)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Class.F.type: type = fn_type @Class.F, @Class(%T.loc18_13.1) [symbolic = %Class.F.type (constants.%Class.F.type)]
// CHECK:STDOUT:   %Class.F: @Class.%Class.F.type (%Class.F.type) = struct_value () [symbolic = %Class.F (constants.%Class.F)]
// CHECK:STDOUT:   %Class.G.type: type = fn_type @Class.G, @Class(%T.loc18_13.1) [symbolic = %Class.G.type (constants.%Class.G.type)]
// CHECK:STDOUT:   %Class.G: @Class.%Class.G.type (%Class.G.type) = struct_value () [symbolic = %Class.G (constants.%Class.G)]
// CHECK:STDOUT:   %T.binding.as_type: type = symbolic_binding_type T, 0, %T.loc18_13.1 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T.binding.as_type [symbolic = %require_complete (constants.%require_complete.67c)]
// CHECK:STDOUT:   %Class: type = class_type @Class, @Class(%T.loc18_13.1) [symbolic = %Class (constants.%Class)]
// CHECK:STDOUT:   %Class.elem: type = unbound_element_type %Class, %T.binding.as_type [symbolic = %Class.elem (constants.%Class.elem)]
// CHECK:STDOUT:   %struct_type.n: type = struct_type {.n: @Class.%T.binding.as_type (%T.binding.as_type)} [symbolic = %struct_type.n (constants.%struct_type.n)]
// CHECK:STDOUT:   %complete_type.loc22_1.2: <witness> = complete_type_witness %struct_type.n [symbolic = %complete_type.loc22_1.2 (constants.%complete_type)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %Class.F.decl: @Class.%Class.F.type (%Class.F.type) = fn_decl @Class.F [symbolic = @Class.%Class.F (constants.%Class.F)] {
// CHECK:STDOUT:       %n.param_patt: @Class.F.%pattern_type (%pattern_type.9b9f0c.2) = value_param_pattern [concrete]
// CHECK:STDOUT:       %n.patt: @Class.F.%pattern_type (%pattern_type.9b9f0c.2) = at_binding_pattern n, %n.param_patt [concrete]
// CHECK:STDOUT:       %return.param_patt: @Class.F.%pattern_type (%pattern_type.9b9f0c.2) = out_param_pattern [concrete]
// CHECK:STDOUT:       %return.patt: @Class.F.%pattern_type (%pattern_type.9b9f0c.2) = return_slot_pattern %return.param_patt, %.loc24_36.2 [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %T.ref.loc19_17: %Copy.type = name_ref T, @Class.%T.loc18_13.2 [symbolic = %T.loc19 (constants.%T.035)]
// CHECK:STDOUT:       %T.as_type.loc19_17: type = facet_access_type %T.ref.loc19_17 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:       %.loc19_17.2: type = converted %T.ref.loc19_17, %T.as_type.loc19_17 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:       %.loc19_17.3: Core.Form = init_form %.loc19_17.2 [symbolic = %.loc19_17.1 (constants.%.076a48.2)]
// CHECK:STDOUT:       %n.param.loc19: @Class.F.%T.binding.as_type (%T.binding.as_type) = value_param call_param0
// CHECK:STDOUT:       %.loc19_11.1: type = splice_block %.loc19_11.2 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)] {
// CHECK:STDOUT:         %T.ref.loc19_11: %Copy.type = name_ref T, @Class.%T.loc18_13.2 [symbolic = %T.loc19 (constants.%T.035)]
// CHECK:STDOUT:         %T.as_type.loc19_11: type = facet_access_type %T.ref.loc19_11 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:         %.loc19_11.2: type = converted %T.ref.loc19_11, %T.as_type.loc19_11 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %n.loc19: @Class.F.%T.binding.as_type (%T.binding.as_type) = value_binding n, %n.param.loc19
// CHECK:STDOUT:       %return.param.loc19: ref @Class.F.%T.binding.as_type (%T.binding.as_type) = out_param call_param1
// CHECK:STDOUT:       %return.loc19: ref @Class.F.%T.binding.as_type (%T.binding.as_type) = return_slot %return.param.loc19
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %Class.G.decl: @Class.%Class.G.type (%Class.G.type) = fn_decl @Class.G [symbolic = @Class.%Class.G (constants.%Class.G)] {
// CHECK:STDOUT:       %self.param_patt: @Class.G.%pattern_type.loc20_12 (%pattern_type.893) = value_param_pattern [concrete]
// CHECK:STDOUT:       %self.patt: @Class.G.%pattern_type.loc20_12 (%pattern_type.893) = at_binding_pattern self, %self.param_patt [concrete]
// CHECK:STDOUT:       %return.param_patt: @Class.G.%pattern_type.loc20_25 (%pattern_type.9b9f0c.2) = out_param_pattern [concrete]
// CHECK:STDOUT:       %return.patt: @Class.G.%pattern_type.loc20_25 (%pattern_type.9b9f0c.2) = return_slot_pattern %return.param_patt, %.loc28_44.2 [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %T.ref.loc20: %Copy.type = name_ref T, @Class.%T.loc18_13.2 [symbolic = %T.loc20 (constants.%T.035)]
// CHECK:STDOUT:       %T.as_type.loc20: type = facet_access_type %T.ref.loc20 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:       %.loc20_25.2: type = converted %T.ref.loc20, %T.as_type.loc20 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:       %.loc20_25.3: Core.Form = init_form %.loc20_25.2 [symbolic = %.loc20_25.1 (constants.%.076a48.2)]
// CHECK:STDOUT:       %self.param.loc20: @Class.G.%Class (%Class) = value_param call_param0
// CHECK:STDOUT:       %.loc20_14.1: type = splice_block %Self.ref.loc20 [symbolic = %Class (constants.%Class)] {
// CHECK:STDOUT:         %.loc20_14.2: type = specific_constant constants.%Class, @Class(constants.%T.035) [symbolic = %Class (constants.%Class)]
// CHECK:STDOUT:         %Self.ref.loc20: type = name_ref Self, %.loc20_14.2 [symbolic = %Class (constants.%Class)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %self.loc20: @Class.G.%Class (%Class) = value_binding self, %self.param.loc20
// CHECK:STDOUT:       %return.param.loc20: ref @Class.G.%T.binding.as_type (%T.binding.as_type) = out_param call_param1
// CHECK:STDOUT:       %return.loc20: ref @Class.G.%T.binding.as_type (%T.binding.as_type) = return_slot %return.param.loc20
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.ref: %Copy.type = name_ref T, %T.loc18_13.2 [symbolic = %T.loc18_13.1 (constants.%T.035)]
// CHECK:STDOUT:     %T.as_type: type = facet_access_type %T.ref [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:     %.loc21_10: type = converted %T.ref, %T.as_type [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:     %.loc21_8: @Class.%Class.elem (%Class.elem) = field_decl n, element0 [concrete]
// CHECK:STDOUT:     %complete_type.loc22_1.1: <witness> = complete_type_witness constants.%struct_type.n [symbolic = %complete_type.loc22_1.2 (constants.%complete_type)]
// CHECK:STDOUT:     complete_type_witness = %complete_type.loc22_1.1
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%Class
// CHECK:STDOUT:     .T = <poisoned>
// CHECK:STDOUT:     .F = %Class.F.decl
// CHECK:STDOUT:     .G = %Class.G.decl
// CHECK:STDOUT:     .n = %.loc21_8
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Class.F(@Class.%T.loc18_13.2: %Copy.type) {
// CHECK:STDOUT:   %T.loc19: %Copy.type = symbolic_binding T, 0 [symbolic = %T.loc19 (constants.%T.035)]
// CHECK:STDOUT:   %T.binding.as_type: type = symbolic_binding_type T, 0, %T.loc19 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %T.binding.as_type [symbolic = %pattern_type (constants.%pattern_type.9b9f0c.2)]
// CHECK:STDOUT:   %.loc19_17.1: Core.Form = init_form %T.binding.as_type [symbolic = %.loc19_17.1 (constants.%.076a48.2)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T.binding.as_type [symbolic = %require_complete (constants.%require_complete.67c)]
// CHECK:STDOUT:   %Copy.lookup_impl_witness: <witness> = lookup_impl_witness %T.loc19, @Copy [symbolic = %Copy.lookup_impl_witness (constants.%Copy.lookup_impl_witness.58d)]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%T.loc19) [symbolic = %Copy.WithSelf.Op.type (constants.%Copy.WithSelf.Op.type.735e75.2)]
// CHECK:STDOUT:   %.loc25: type = fn_type_with_self_type %Copy.WithSelf.Op.type, %T.loc19 [symbolic = %.loc25 (constants.%.023)]
// CHECK:STDOUT:   %impl.elem0.loc25_10.2: @Class.F.%.loc25 (%.023) = impl_witness_access %Copy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc25_10.2 (constants.%impl.elem0.594)]
// CHECK:STDOUT:   %specific_impl_fn.loc25_10.2: <specific function> = specific_impl_function %impl.elem0.loc25_10.2, @Copy.WithSelf.Op(%T.loc19) [symbolic = %specific_impl_fn.loc25_10.2 (constants.%specific_impl_fn.bdc)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%n.param.loc24: @Class.F.%T.binding.as_type (%T.binding.as_type)) -> out %return.param.loc24: @Class.F.%T.binding.as_type (%T.binding.as_type) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %n.ref: @Class.F.%T.binding.as_type (%T.binding.as_type) = name_ref n, %n.loc24
// CHECK:STDOUT:     %impl.elem0.loc25_10.1: @Class.F.%.loc25 (%.023) = impl_witness_access constants.%Copy.lookup_impl_witness.58d, element0 [symbolic = %impl.elem0.loc25_10.2 (constants.%impl.elem0.594)]
// CHECK:STDOUT:     %bound_method.loc25_10.1: <bound method> = bound_method %n.ref, %impl.elem0.loc25_10.1
// CHECK:STDOUT:     %specific_impl_fn.loc25_10.1: <specific function> = specific_impl_function %impl.elem0.loc25_10.1, @Copy.WithSelf.Op(constants.%T.035) [symbolic = %specific_impl_fn.loc25_10.2 (constants.%specific_impl_fn.bdc)]
// CHECK:STDOUT:     %bound_method.loc25_10.2: <bound method> = bound_method %n.ref, %specific_impl_fn.loc25_10.1
// CHECK:STDOUT:     %.loc24_36.1: ref @Class.F.%T.binding.as_type (%T.binding.as_type) = splice_block %return.param.loc24 {}
// CHECK:STDOUT:     %Copy.WithSelf.Op.call: init @Class.F.%T.binding.as_type (%T.binding.as_type) to %.loc24_36.1 = call %bound_method.loc25_10.2(%n.ref)
// CHECK:STDOUT:     return %Copy.WithSelf.Op.call to %return.param.loc24
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Class.G(@Class.%T.loc18_13.2: %Copy.type) {
// CHECK:STDOUT:   %T.loc20: %Copy.type = symbolic_binding T, 0 [symbolic = %T.loc20 (constants.%T.035)]
// CHECK:STDOUT:   %Class: type = class_type @Class, @Class(%T.loc20) [symbolic = %Class (constants.%Class)]
// CHECK:STDOUT:   %pattern_type.loc20_12: type = pattern_type %Class [symbolic = %pattern_type.loc20_12 (constants.%pattern_type.893)]
// CHECK:STDOUT:   %T.binding.as_type: type = symbolic_binding_type T, 0, %T.loc20 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:   %.loc20_25.1: Core.Form = init_form %T.binding.as_type [symbolic = %.loc20_25.1 (constants.%.076a48.2)]
// CHECK:STDOUT:   %pattern_type.loc20_25: type = pattern_type %T.binding.as_type [symbolic = %pattern_type.loc20_25 (constants.%pattern_type.9b9f0c.2)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc28: <witness> = require_complete_type %Class [symbolic = %require_complete.loc28 (constants.%require_complete.904)]
// CHECK:STDOUT:   %Class.elem: type = unbound_element_type %Class, %T.binding.as_type [symbolic = %Class.elem (constants.%Class.elem)]
// CHECK:STDOUT:   %require_complete.loc29: <witness> = require_complete_type %T.binding.as_type [symbolic = %require_complete.loc29 (constants.%require_complete.67c)]
// CHECK:STDOUT:   %Copy.lookup_impl_witness: <witness> = lookup_impl_witness %T.loc20, @Copy [symbolic = %Copy.lookup_impl_witness (constants.%Copy.lookup_impl_witness.58d)]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%T.loc20) [symbolic = %Copy.WithSelf.Op.type (constants.%Copy.WithSelf.Op.type.735e75.2)]
// CHECK:STDOUT:   %.loc29_14.3: type = fn_type_with_self_type %Copy.WithSelf.Op.type, %T.loc20 [symbolic = %.loc29_14.3 (constants.%.023)]
// CHECK:STDOUT:   %impl.elem0.loc29_14.2: @Class.G.%.loc29_14.3 (%.023) = impl_witness_access %Copy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc29_14.2 (constants.%impl.elem0.594)]
// CHECK:STDOUT:   %specific_impl_fn.loc29_14.2: <specific function> = specific_impl_function %impl.elem0.loc29_14.2, @Copy.WithSelf.Op(%T.loc20) [symbolic = %specific_impl_fn.loc29_14.2 (constants.%specific_impl_fn.bdc)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param.loc28: @Class.G.%Class (%Class)) -> out %return.param.loc28: @Class.G.%T.binding.as_type (%T.binding.as_type) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %self.ref: @Class.G.%Class (%Class) = name_ref self, %self.loc28
// CHECK:STDOUT:     %n.ref: @Class.G.%Class.elem (%Class.elem) = name_ref n, @Class.%.loc21_8 [concrete = @Class.%.loc21_8]
// CHECK:STDOUT:     %.loc29_14.1: ref @Class.G.%T.binding.as_type (%T.binding.as_type) = class_element_access %self.ref, element0
// CHECK:STDOUT:     %.loc29_14.2: @Class.G.%T.binding.as_type (%T.binding.as_type) = acquire_value %.loc29_14.1
// CHECK:STDOUT:     %impl.elem0.loc29_14.1: @Class.G.%.loc29_14.3 (%.023) = impl_witness_access constants.%Copy.lookup_impl_witness.58d, element0 [symbolic = %impl.elem0.loc29_14.2 (constants.%impl.elem0.594)]
// CHECK:STDOUT:     %bound_method.loc29_14.1: <bound method> = bound_method %.loc29_14.2, %impl.elem0.loc29_14.1
// CHECK:STDOUT:     %specific_impl_fn.loc29_14.1: <specific function> = specific_impl_function %impl.elem0.loc29_14.1, @Copy.WithSelf.Op(constants.%T.035) [symbolic = %specific_impl_fn.loc29_14.2 (constants.%specific_impl_fn.bdc)]
// CHECK:STDOUT:     %bound_method.loc29_14.2: <bound method> = bound_method %.loc29_14.2, %specific_impl_fn.loc29_14.1
// CHECK:STDOUT:     %.loc28_44.1: ref @Class.G.%T.binding.as_type (%T.binding.as_type) = splice_block %return.param.loc28 {}
// CHECK:STDOUT:     %Copy.WithSelf.Op.call: init @Class.G.%T.binding.as_type (%T.binding.as_type) to %.loc28_44.1 = call %bound_method.loc29_14.2(%.loc29_14.2)
// CHECK:STDOUT:     return %Copy.WithSelf.Op.call to %return.param.loc28
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class(constants.%T.035) {
// CHECK:STDOUT:   %T.loc18_13.1 => constants.%T.035
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Class.F.type => constants.%Class.F.type
// CHECK:STDOUT:   %Class.F => constants.%Class.F
// CHECK:STDOUT:   %Class.G.type => constants.%Class.G.type
// CHECK:STDOUT:   %Class.G => constants.%Class.G
// CHECK:STDOUT:   %T.binding.as_type => constants.%T.binding.as_type
// CHECK:STDOUT:   %require_complete => constants.%require_complete.67c
// CHECK:STDOUT:   %Class => constants.%Class
// CHECK:STDOUT:   %Class.elem => constants.%Class.elem
// CHECK:STDOUT:   %struct_type.n => constants.%struct_type.n
// CHECK:STDOUT:   %complete_type.loc22_1.2 => constants.%complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class.F(constants.%T.035) {
// CHECK:STDOUT:   %T.loc19 => constants.%T.035
// CHECK:STDOUT:   %T.binding.as_type => constants.%T.binding.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.9b9f0c.2
// CHECK:STDOUT:   %.loc19_17.1 => constants.%.076a48.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class.G(constants.%T.035) {
// CHECK:STDOUT:   %T.loc20 => constants.%T.035
// CHECK:STDOUT:   %Class => constants.%Class
// CHECK:STDOUT:   %pattern_type.loc20_12 => constants.%pattern_type.893
// CHECK:STDOUT:   %T.binding.as_type => constants.%T.binding.as_type
// CHECK:STDOUT:   %.loc20_25.1 => constants.%.076a48.2
// CHECK:STDOUT:   %pattern_type.loc20_25 => constants.%pattern_type.9b9f0c.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- nested.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %A.type: type = generic_class_type @A [concrete]
// CHECK:STDOUT:   %A.generic: %A.type = struct_value () [concrete]
// CHECK:STDOUT:   %A: type = class_type @A, @A(%T) [symbolic]
// CHECK:STDOUT:   %pattern_type.51d: type = pattern_type %T [symbolic]
// CHECK:STDOUT:   %_: %T = symbolic_binding _, 1 [symbolic]
// CHECK:STDOUT:   %B.type: type = generic_class_type @B, @A(%T) [symbolic]
// CHECK:STDOUT:   %B.generic: %B.type = struct_value () [symbolic]
// CHECK:STDOUT:   %B: type = class_type @B, @B(%T, %_) [symbolic]
// CHECK:STDOUT:   %pattern_type.830: type = pattern_type %B [symbolic]
// CHECK:STDOUT:   %B.F.type: type = fn_type @B.F, @B(%T, %_) [symbolic]
// CHECK:STDOUT:   %B.F: %B.F.type = struct_value () [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %require_complete.7cc: <witness> = require_complete_type %B [symbolic]
// CHECK:STDOUT:   %require_complete.944: <witness> = require_complete_type %T [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %A.decl: %A.type = class_decl @A [concrete = constants.%A.generic] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc38_13.1: type = splice_block %.loc38_13.2 [concrete = type] {
// CHECK:STDOUT:       <elided>
// CHECK:STDOUT:       %.loc38_13.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc38_9.2: type = symbolic_binding T, 0 [symbolic = %T.loc38_9.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %B.F.decl: %B.F.type = fn_decl @B.F [symbolic = constants.%B.F] {
// CHECK:STDOUT:     %self.param_patt: @B.F.%pattern_type.loc40_14 (%pattern_type.830) = value_param_pattern [concrete]
// CHECK:STDOUT:     %self.patt: @B.F.%pattern_type.loc40_14 (%pattern_type.830) = at_binding_pattern self, %self.param_patt [concrete]
// CHECK:STDOUT:     %a.param_patt: @B.F.%pattern_type.loc40_23 (%pattern_type.51d) = value_param_pattern [concrete]
// CHECK:STDOUT:     %a.patt: @B.F.%pattern_type.loc40_23 (%pattern_type.51d) = at_binding_pattern a, %a.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc44_10.1: type = splice_block %.loc44_10.2 [concrete = type] {
// CHECK:STDOUT:       <elided>
// CHECK:STDOUT:       %.loc44_10.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc44: type = symbolic_binding T, 0 [symbolic = @A.%T.loc38_9.1 (constants.%T)]
// CHECK:STDOUT:     %.loc44_22: type = splice_block %T.ref.loc44_22 [symbolic = @B.%T (constants.%T)] {
// CHECK:STDOUT:       <elided>
// CHECK:STDOUT:       %T.ref.loc44_22: type = name_ref T, %T.loc44 [symbolic = @B.%T (constants.%T)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %_.loc44: @B.%T (%T) = symbolic_binding _, 1 [symbolic = @B.%_.loc39_11.1 (constants.%_)]
// CHECK:STDOUT:     %self.param.loc44: @B.F.%B (%B) = value_param call_param0
// CHECK:STDOUT:     %.loc44_40.1: type = splice_block %Self.ref.loc44 [symbolic = %B (constants.%B)] {
// CHECK:STDOUT:       %.loc44_40.2: type = specific_constant constants.%B, @B(constants.%T, constants.%_) [symbolic = %B (constants.%B)]
// CHECK:STDOUT:       %Self.ref.loc44: type = name_ref Self, %.loc44_40.2 [symbolic = %B (constants.%B)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %self.loc44: @B.F.%B (%B) = value_binding self, %self.param.loc44
// CHECK:STDOUT:     %a.param.loc44: @B.F.%T.loc40 (%T) = value_param call_param1
// CHECK:STDOUT:     %T.ref.loc44_56: type = name_ref T, %T.loc44 [symbolic = %T.loc40 (constants.%T)]
// CHECK:STDOUT:     %a.loc44: @B.F.%T.loc40 (%T) = value_binding a, %a.param.loc44
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @A(%T.loc38_9.2: type) {
// CHECK:STDOUT:   %T.loc38_9.1: type = symbolic_binding T, 0 [symbolic = %T.loc38_9.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %B.type: type = generic_class_type @B, @A(%T.loc38_9.1) [symbolic = %B.type (constants.%B.type)]
// CHECK:STDOUT:   %B.generic: @A.%B.type (%B.type) = struct_value () [symbolic = %B.generic (constants.%B.generic)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %B.decl: @A.%B.type (%B.type) = class_decl @B [symbolic = @A.%B.generic (constants.%B.generic)] {
// CHECK:STDOUT:       %_.patt: @B.%pattern_type (%pattern_type.51d) = symbolic_binding_pattern _, 1 [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %.loc39: type = splice_block %T.ref [symbolic = %T (constants.%T)] {
// CHECK:STDOUT:         <elided>
// CHECK:STDOUT:         %T.ref: type = name_ref T, @A.%T.loc38_9.2 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %_.loc39_11.2: @B.%T (%T) = symbolic_binding _, 1 [symbolic = %_.loc39_11.1 (constants.%_)]
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
// CHECK:STDOUT: generic class @B(@A.%T.loc38_9.2: type, %_.loc39_11.2: @B.%T (%T)) {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %_.loc39_11.1: @B.%T (%T) = symbolic_binding _, 1 [symbolic = %_.loc39_11.1 (constants.%_)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %T [symbolic = %pattern_type (constants.%pattern_type.51d)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %B.F.type: type = fn_type @B.F, @B(%T, %_.loc39_11.1) [symbolic = %B.F.type (constants.%B.F.type)]
// CHECK:STDOUT:   %B.F: @B.%B.F.type (%B.F.type) = struct_value () [symbolic = %B.F (constants.%B.F)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %B.F.decl: @B.%B.F.type (%B.F.type) = fn_decl @B.F [symbolic = @B.%B.F (constants.%B.F)] {
// CHECK:STDOUT:       %self.param_patt: @B.F.%pattern_type.loc40_14 (%pattern_type.830) = value_param_pattern [concrete]
// CHECK:STDOUT:       %self.patt: @B.F.%pattern_type.loc40_14 (%pattern_type.830) = at_binding_pattern self, %self.param_patt [concrete]
// CHECK:STDOUT:       %a.param_patt: @B.F.%pattern_type.loc40_23 (%pattern_type.51d) = value_param_pattern [concrete]
// CHECK:STDOUT:       %a.patt: @B.F.%pattern_type.loc40_23 (%pattern_type.51d) = at_binding_pattern a, %a.param_patt [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %self.param.loc40: @B.F.%B (%B) = value_param call_param0
// CHECK:STDOUT:       %.loc40_16.1: type = splice_block %Self.ref.loc40 [symbolic = %B (constants.%B)] {
// CHECK:STDOUT:         %.loc40_16.2: type = specific_constant constants.%B, @B(constants.%T, constants.%_) [symbolic = %B (constants.%B)]
// CHECK:STDOUT:         %Self.ref.loc40: type = name_ref Self, %.loc40_16.2 [symbolic = %B (constants.%B)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %self.loc40: @B.F.%B (%B) = value_binding self, %self.param.loc40
// CHECK:STDOUT:       %a.param.loc40: @B.F.%T.loc40 (%T) = value_param call_param1
// CHECK:STDOUT:       %T.ref.loc40: type = name_ref T, @A.%T.loc38_9.2 [symbolic = %T.loc40 (constants.%T)]
// CHECK:STDOUT:       %a.loc40: @B.F.%T.loc40 (%T) = value_binding a, %a.param.loc40
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
// CHECK:STDOUT: generic fn @B.F(@A.%T.loc38_9.2: type, @B.%_.loc39_11.2: @B.%T (%T)) {
// CHECK:STDOUT:   %T.loc40: type = symbolic_binding T, 0 [symbolic = %T.loc40 (constants.%T)]
// CHECK:STDOUT:   %_.loc40: @B.F.%T.loc40 (%T) = symbolic_binding _, 1 [symbolic = %_.loc40 (constants.%_)]
// CHECK:STDOUT:   %B: type = class_type @B, @B(%T.loc40, %_.loc40) [symbolic = %B (constants.%B)]
// CHECK:STDOUT:   %pattern_type.loc40_14: type = pattern_type %B [symbolic = %pattern_type.loc40_14 (constants.%pattern_type.830)]
// CHECK:STDOUT:   %pattern_type.loc40_23: type = pattern_type %T.loc40 [symbolic = %pattern_type.loc40_23 (constants.%pattern_type.51d)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc44_38: <witness> = require_complete_type %B [symbolic = %require_complete.loc44_38 (constants.%require_complete.7cc)]
// CHECK:STDOUT:   %require_complete.loc44_54: <witness> = require_complete_type %T.loc40 [symbolic = %require_complete.loc44_54 (constants.%require_complete.944)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param.loc44: @B.F.%B (%B), %a.param.loc44: @B.F.%T.loc40 (%T)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A(constants.%T) {
// CHECK:STDOUT:   %T.loc38_9.1 => constants.%T
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %B.type => constants.%B.type
// CHECK:STDOUT:   %B.generic => constants.%B.generic
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @B(constants.%T, constants.%_) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %_.loc39_11.1 => constants.%_
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.51d
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %B.F.type => constants.%B.F.type
// CHECK:STDOUT:   %B.F => constants.%B.F
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @B.F(constants.%T, constants.%_) {
// CHECK:STDOUT:   %T.loc40 => constants.%T
// CHECK:STDOUT:   %_.loc40 => constants.%_
// CHECK:STDOUT:   %B => constants.%B
// CHECK:STDOUT:   %pattern_type.loc40_14 => constants.%pattern_type.830
// CHECK:STDOUT:   %pattern_type.loc40_23 => constants.%pattern_type.51d
// CHECK:STDOUT: }
// CHECK:STDOUT:
