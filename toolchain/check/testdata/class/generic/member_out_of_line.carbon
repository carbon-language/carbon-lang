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
// CHECK:STDERR:                   ^~~~~~
// CHECK:STDERR: fail_mismatched_wrong_arg_type.carbon:[[@LINE-7]]:22: note: previous declaration's corresponding parameter here [RedeclParamPrevious]
// CHECK:STDERR: class Generic(unused T:! type) {
// CHECK:STDERR:                      ^~~~~~~~
// CHECK:STDERR:
fn Generic(unused T:! ()).WrongType() {}

// CHECK:STDOUT: --- basic.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %pattern_type.35b: type = pattern_type %Copy.type [concrete]
// CHECK:STDOUT:   %T.228: %Copy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %Class.type: type = generic_class_type @Class [concrete]
// CHECK:STDOUT:   %Class.generic: %Class.type = struct_value () [concrete]
// CHECK:STDOUT:   %Class: type = class_type @Class, @Class(%T.228) [symbolic]
// CHECK:STDOUT:   %T.as_type: type = facet_access_type %T.228 [symbolic]
// CHECK:STDOUT:   %pattern_type.4b69d0.2: type = pattern_type %T.as_type [symbolic]
// CHECK:STDOUT:   %.9ff809.2: Core.Form = init_form %T.as_type [symbolic]
// CHECK:STDOUT:   %Class.F.type: type = fn_type @Class.F, @Class(%T.228) [symbolic]
// CHECK:STDOUT:   %Class.F: %Class.F.type = struct_value () [symbolic]
// CHECK:STDOUT:   %pattern_type.b47: type = pattern_type %Class [symbolic]
// CHECK:STDOUT:   %Class.G.type: type = fn_type @Class.G, @Class(%T.228) [symbolic]
// CHECK:STDOUT:   %Class.G: %Class.G.type = struct_value () [symbolic]
// CHECK:STDOUT:   %require_complete.c28: <witness> = require_complete_type %T.as_type [symbolic]
// CHECK:STDOUT:   %Class.elem: type = unbound_element_type %Class, %T.as_type [symbolic]
// CHECK:STDOUT:   %struct_type.n: type = struct_type {.n: %T.as_type} [symbolic]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.n [symbolic]
// CHECK:STDOUT:   %Copy.lookup_impl_witness.c49: <witness> = lookup_impl_witness %T.228, @Copy [symbolic]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.d2ec0c.2: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%T.228) [symbolic]
// CHECK:STDOUT:   %.12f: type = fn_type_with_self_type %Copy.WithSelf.Op.type.d2ec0c.2, %T.228 [symbolic]
// CHECK:STDOUT:   %impl.elem0.cbd: %.12f = impl_witness_access %Copy.lookup_impl_witness.c49, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.c5f: <specific function> = specific_impl_function %impl.elem0.cbd, @Copy.WithSelf.Op(%T.228) [symbolic]
// CHECK:STDOUT:   %require_complete.478: <witness> = require_complete_type %Class [symbolic]
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
// CHECK:STDOUT:     %T.patt: %pattern_type.35b = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc5: type = splice_block %Copy.ref [concrete = constants.%Copy.type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:       %Copy.ref: type = name_ref Copy, imports.%Core.Copy [concrete = constants.%Copy.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc5_14.2: %Copy.type = symbolic_binding T, 0 [symbolic = %T.loc5_14.1 (constants.%T.228)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Class.F.decl: %Class.F.type = fn_decl @Class.F [symbolic = constants.%Class.F] {
// CHECK:STDOUT:     %n.param_patt: @Class.F.%pattern_type (%pattern_type.4b69d0.2) = value_param_pattern [concrete]
// CHECK:STDOUT:     %n.patt: @Class.F.%pattern_type (%pattern_type.4b69d0.2) = at_binding_pattern n, %n.param_patt [concrete]
// CHECK:STDOUT:     %return.param_patt: @Class.F.%pattern_type (%pattern_type.4b69d0.2) = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: @Class.F.%pattern_type (%pattern_type.4b69d0.2) = return_slot_pattern %return.param_patt, %.loc11_36.2 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc11_18: type = splice_block %Copy.ref [concrete = constants.%Copy.type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:       %Copy.ref: type = name_ref Copy, imports.%Core.Copy [concrete = constants.%Copy.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc11: %Copy.type = symbolic_binding T, 0 [symbolic = @Class.%T.loc5_14.1 (constants.%T.228)]
// CHECK:STDOUT:     %T.ref.loc11_36: %Copy.type = name_ref T, %T.loc11 [symbolic = %T.loc6 (constants.%T.228)]
// CHECK:STDOUT:     %T.as_type.loc11_36: type = facet_access_type %T.ref.loc11_36 [symbolic = %T.as_type.loc6_11.1 (constants.%T.as_type)]
// CHECK:STDOUT:     %.loc11_36.2: type = converted %T.ref.loc11_36, %T.as_type.loc11_36 [symbolic = %T.as_type.loc6_11.1 (constants.%T.as_type)]
// CHECK:STDOUT:     %.loc11_36.3: Core.Form = init_form %.loc11_36.2 [symbolic = %.loc6_17.1 (constants.%.9ff809.2)]
// CHECK:STDOUT:     %n.param.loc11: @Class.F.%T.as_type.loc6_11.1 (%T.as_type) = value_param call_param0
// CHECK:STDOUT:     %.loc11_30.1: type = splice_block %.loc11_30.2 [symbolic = %T.as_type.loc6_11.1 (constants.%T.as_type)] {
// CHECK:STDOUT:       %T.ref.loc11_30: %Copy.type = name_ref T, %T.loc11 [symbolic = %T.loc6 (constants.%T.228)]
// CHECK:STDOUT:       %T.as_type.loc11_30: type = facet_access_type %T.ref.loc11_30 [symbolic = %T.as_type.loc6_11.1 (constants.%T.as_type)]
// CHECK:STDOUT:       %.loc11_30.2: type = converted %T.ref.loc11_30, %T.as_type.loc11_30 [symbolic = %T.as_type.loc6_11.1 (constants.%T.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %n.loc11: @Class.F.%T.as_type.loc6_11.1 (%T.as_type) = value_binding n, %n.param.loc11
// CHECK:STDOUT:     %return.param.loc11: ref @Class.F.%T.as_type.loc6_11.1 (%T.as_type) = out_param call_param1
// CHECK:STDOUT:     %return.loc11: ref @Class.F.%T.as_type.loc6_11.1 (%T.as_type) = return_slot %return.param.loc11
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Class.G.decl: %Class.G.type = fn_decl @Class.G [symbolic = constants.%Class.G] {
// CHECK:STDOUT:     %self.param_patt: @Class.G.%pattern_type.loc7_12 (%pattern_type.b47) = value_param_pattern [concrete]
// CHECK:STDOUT:     %self.patt: @Class.G.%pattern_type.loc7_12 (%pattern_type.b47) = at_binding_pattern self, %self.param_patt [concrete]
// CHECK:STDOUT:     %return.param_patt: @Class.G.%pattern_type.loc7_25 (%pattern_type.4b69d0.2) = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: @Class.G.%pattern_type.loc7_25 (%pattern_type.4b69d0.2) = return_slot_pattern %return.param_patt, %.loc15_44.2 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc15_18: type = splice_block %Copy.ref [concrete = constants.%Copy.type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:       %Copy.ref: type = name_ref Copy, imports.%Core.Copy [concrete = constants.%Copy.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc15: %Copy.type = symbolic_binding T, 0 [symbolic = @Class.%T.loc5_14.1 (constants.%T.228)]
// CHECK:STDOUT:     %T.ref.loc15: %Copy.type = name_ref T, %T.loc15 [symbolic = %T.loc7 (constants.%T.228)]
// CHECK:STDOUT:     %T.as_type.loc15: type = facet_access_type %T.ref.loc15 [symbolic = %T.as_type.loc7_25.1 (constants.%T.as_type)]
// CHECK:STDOUT:     %.loc15_44.2: type = converted %T.ref.loc15, %T.as_type.loc15 [symbolic = %T.as_type.loc7_25.1 (constants.%T.as_type)]
// CHECK:STDOUT:     %.loc15_44.3: Core.Form = init_form %.loc15_44.2 [symbolic = %.loc7_25.1 (constants.%.9ff809.2)]
// CHECK:STDOUT:     %self.param.loc15: @Class.G.%Class (%Class) = value_param call_param0
// CHECK:STDOUT:     %.loc15_33.1: type = splice_block %Self.ref.loc15 [symbolic = %Class (constants.%Class)] {
// CHECK:STDOUT:       %.loc15_33.2: type = specific_constant constants.%Class, @Class(constants.%T.228) [symbolic = %Class (constants.%Class)]
// CHECK:STDOUT:       %Self.ref.loc15: type = name_ref Self, %.loc15_33.2 [symbolic = %Class (constants.%Class)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %self.loc15: @Class.G.%Class (%Class) = value_binding self, %self.param.loc15
// CHECK:STDOUT:     %return.param.loc15: ref @Class.G.%T.as_type.loc7_25.1 (%T.as_type) = out_param call_param1
// CHECK:STDOUT:     %return.loc15: ref @Class.G.%T.as_type.loc7_25.1 (%T.as_type) = return_slot %return.param.loc15
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @Class(%T.loc5_14.2: %Copy.type) {
// CHECK:STDOUT:   %T.loc5_14.1: %Copy.type = symbolic_binding T, 0 [symbolic = %T.loc5_14.1 (constants.%T.228)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Class.F.type: type = fn_type @Class.F, @Class(%T.loc5_14.1) [symbolic = %Class.F.type (constants.%Class.F.type)]
// CHECK:STDOUT:   %Class.F: @Class.%Class.F.type (%Class.F.type) = struct_value () [symbolic = %Class.F (constants.%Class.F)]
// CHECK:STDOUT:   %Class.G.type: type = fn_type @Class.G, @Class(%T.loc5_14.1) [symbolic = %Class.G.type (constants.%Class.G.type)]
// CHECK:STDOUT:   %Class.G: @Class.%Class.G.type (%Class.G.type) = struct_value () [symbolic = %Class.G (constants.%Class.G)]
// CHECK:STDOUT:   %T.as_type: type = facet_access_type %T.loc5_14.1 [symbolic = %T.as_type (constants.%T.as_type)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T.as_type [symbolic = %require_complete (constants.%require_complete.c28)]
// CHECK:STDOUT:   %Class: type = class_type @Class, @Class(%T.loc5_14.1) [symbolic = %Class (constants.%Class)]
// CHECK:STDOUT:   %Class.elem: type = unbound_element_type %Class, %T.as_type [symbolic = %Class.elem (constants.%Class.elem)]
// CHECK:STDOUT:   %struct_type.n: type = struct_type {.n: @Class.%T.as_type (%T.as_type)} [symbolic = %struct_type.n (constants.%struct_type.n)]
// CHECK:STDOUT:   %complete_type.loc9_1.2: <witness> = complete_type_witness %struct_type.n [symbolic = %complete_type.loc9_1.2 (constants.%complete_type)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %Class.F.decl: @Class.%Class.F.type (%Class.F.type) = fn_decl @Class.F [symbolic = @Class.%Class.F (constants.%Class.F)] {
// CHECK:STDOUT:       %n.param_patt: @Class.F.%pattern_type (%pattern_type.4b69d0.2) = value_param_pattern [concrete]
// CHECK:STDOUT:       %n.patt: @Class.F.%pattern_type (%pattern_type.4b69d0.2) = at_binding_pattern n, %n.param_patt [concrete]
// CHECK:STDOUT:       %return.param_patt: @Class.F.%pattern_type (%pattern_type.4b69d0.2) = out_param_pattern [concrete]
// CHECK:STDOUT:       %return.patt: @Class.F.%pattern_type (%pattern_type.4b69d0.2) = return_slot_pattern %return.param_patt, %.loc11_36.2 [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %T.ref.loc6_17: %Copy.type = name_ref T, @Class.%T.loc5_14.2 [symbolic = %T.loc6 (constants.%T.228)]
// CHECK:STDOUT:       %T.as_type.loc6_17: type = facet_access_type %T.ref.loc6_17 [symbolic = %T.as_type.loc6_11.1 (constants.%T.as_type)]
// CHECK:STDOUT:       %.loc6_17.2: type = converted %T.ref.loc6_17, %T.as_type.loc6_17 [symbolic = %T.as_type.loc6_11.1 (constants.%T.as_type)]
// CHECK:STDOUT:       %.loc6_17.3: Core.Form = init_form %.loc6_17.2 [symbolic = %.loc6_17.1 (constants.%.9ff809.2)]
// CHECK:STDOUT:       %n.param.loc6: @Class.F.%T.as_type.loc6_11.1 (%T.as_type) = value_param call_param0
// CHECK:STDOUT:       %.loc6_11.1: type = splice_block %.loc6_11.2 [symbolic = %T.as_type.loc6_11.1 (constants.%T.as_type)] {
// CHECK:STDOUT:         %T.ref.loc6_11: %Copy.type = name_ref T, @Class.%T.loc5_14.2 [symbolic = %T.loc6 (constants.%T.228)]
// CHECK:STDOUT:         %T.as_type.loc6_11.2: type = facet_access_type %T.ref.loc6_11 [symbolic = %T.as_type.loc6_11.1 (constants.%T.as_type)]
// CHECK:STDOUT:         %.loc6_11.2: type = converted %T.ref.loc6_11, %T.as_type.loc6_11.2 [symbolic = %T.as_type.loc6_11.1 (constants.%T.as_type)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %n.loc6: @Class.F.%T.as_type.loc6_11.1 (%T.as_type) = value_binding n, %n.param.loc6
// CHECK:STDOUT:       %return.param.loc6: ref @Class.F.%T.as_type.loc6_11.1 (%T.as_type) = out_param call_param1
// CHECK:STDOUT:       %return.loc6: ref @Class.F.%T.as_type.loc6_11.1 (%T.as_type) = return_slot %return.param.loc6
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %Class.G.decl: @Class.%Class.G.type (%Class.G.type) = fn_decl @Class.G [symbolic = @Class.%Class.G (constants.%Class.G)] {
// CHECK:STDOUT:       %self.param_patt: @Class.G.%pattern_type.loc7_12 (%pattern_type.b47) = value_param_pattern [concrete]
// CHECK:STDOUT:       %self.patt: @Class.G.%pattern_type.loc7_12 (%pattern_type.b47) = at_binding_pattern self, %self.param_patt [concrete]
// CHECK:STDOUT:       %return.param_patt: @Class.G.%pattern_type.loc7_25 (%pattern_type.4b69d0.2) = out_param_pattern [concrete]
// CHECK:STDOUT:       %return.patt: @Class.G.%pattern_type.loc7_25 (%pattern_type.4b69d0.2) = return_slot_pattern %return.param_patt, %.loc15_44.2 [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %T.ref.loc7: %Copy.type = name_ref T, @Class.%T.loc5_14.2 [symbolic = %T.loc7 (constants.%T.228)]
// CHECK:STDOUT:       %T.as_type.loc7_25.2: type = facet_access_type %T.ref.loc7 [symbolic = %T.as_type.loc7_25.1 (constants.%T.as_type)]
// CHECK:STDOUT:       %.loc7_25.2: type = converted %T.ref.loc7, %T.as_type.loc7_25.2 [symbolic = %T.as_type.loc7_25.1 (constants.%T.as_type)]
// CHECK:STDOUT:       %.loc7_25.3: Core.Form = init_form %.loc7_25.2 [symbolic = %.loc7_25.1 (constants.%.9ff809.2)]
// CHECK:STDOUT:       %self.param.loc7: @Class.G.%Class (%Class) = value_param call_param0
// CHECK:STDOUT:       %.loc7_14.1: type = splice_block %Self.ref.loc7 [symbolic = %Class (constants.%Class)] {
// CHECK:STDOUT:         %.loc7_14.2: type = specific_constant constants.%Class, @Class(constants.%T.228) [symbolic = %Class (constants.%Class)]
// CHECK:STDOUT:         %Self.ref.loc7: type = name_ref Self, %.loc7_14.2 [symbolic = %Class (constants.%Class)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %self.loc7: @Class.G.%Class (%Class) = value_binding self, %self.param.loc7
// CHECK:STDOUT:       %return.param.loc7: ref @Class.G.%T.as_type.loc7_25.1 (%T.as_type) = out_param call_param1
// CHECK:STDOUT:       %return.loc7: ref @Class.G.%T.as_type.loc7_25.1 (%T.as_type) = return_slot %return.param.loc7
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %.loc8: @Class.%Class.elem (%Class.elem) = field_decl n, element0 [concrete]
// CHECK:STDOUT:     %complete_type.loc9_1.1: <witness> = complete_type_witness constants.%struct_type.n [symbolic = %complete_type.loc9_1.2 (constants.%complete_type)]
// CHECK:STDOUT:     complete_type_witness = %complete_type.loc9_1.1
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%Class
// CHECK:STDOUT:     .T = <poisoned>
// CHECK:STDOUT:     .F = %Class.F.decl
// CHECK:STDOUT:     .G = %Class.G.decl
// CHECK:STDOUT:     .n = %.loc8
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Class.F(@Class.%T.loc5_14.2: %Copy.type) {
// CHECK:STDOUT:   %T.loc6: %Copy.type = symbolic_binding T, 0 [symbolic = %T.loc6 (constants.%T.228)]
// CHECK:STDOUT:   %T.as_type.loc6_11.1: type = facet_access_type %T.loc6 [symbolic = %T.as_type.loc6_11.1 (constants.%T.as_type)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %T.as_type.loc6_11.1 [symbolic = %pattern_type (constants.%pattern_type.4b69d0.2)]
// CHECK:STDOUT:   %.loc6_17.1: Core.Form = init_form %T.as_type.loc6_11.1 [symbolic = %.loc6_17.1 (constants.%.9ff809.2)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T.as_type.loc6_11.1 [symbolic = %require_complete (constants.%require_complete.c28)]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%T.loc6) [symbolic = %Copy.WithSelf.Op.type (constants.%Copy.WithSelf.Op.type.d2ec0c.2)]
// CHECK:STDOUT:   %.loc12: type = fn_type_with_self_type %Copy.WithSelf.Op.type, %T.loc6 [symbolic = %.loc12 (constants.%.12f)]
// CHECK:STDOUT:   %Copy.lookup_impl_witness: <witness> = lookup_impl_witness %T.loc6, @Copy [symbolic = %Copy.lookup_impl_witness (constants.%Copy.lookup_impl_witness.c49)]
// CHECK:STDOUT:   %impl.elem0.loc12_10.2: @Class.F.%.loc12 (%.12f) = impl_witness_access %Copy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc12_10.2 (constants.%impl.elem0.cbd)]
// CHECK:STDOUT:   %specific_impl_fn.loc12_10.2: <specific function> = specific_impl_function %impl.elem0.loc12_10.2, @Copy.WithSelf.Op(%T.loc6) [symbolic = %specific_impl_fn.loc12_10.2 (constants.%specific_impl_fn.c5f)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%n.param.loc11: @Class.F.%T.as_type.loc6_11.1 (%T.as_type)) -> out %return.param.loc11: @Class.F.%T.as_type.loc6_11.1 (%T.as_type) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %n.ref: @Class.F.%T.as_type.loc6_11.1 (%T.as_type) = name_ref n, %n.loc11
// CHECK:STDOUT:     %impl.elem0.loc12_10.1: @Class.F.%.loc12 (%.12f) = impl_witness_access constants.%Copy.lookup_impl_witness.c49, element0 [symbolic = %impl.elem0.loc12_10.2 (constants.%impl.elem0.cbd)]
// CHECK:STDOUT:     %bound_method.loc12_10.1: <bound method> = bound_method %n.ref, %impl.elem0.loc12_10.1
// CHECK:STDOUT:     %specific_impl_fn.loc12_10.1: <specific function> = specific_impl_function %impl.elem0.loc12_10.1, @Copy.WithSelf.Op(constants.%T.228) [symbolic = %specific_impl_fn.loc12_10.2 (constants.%specific_impl_fn.c5f)]
// CHECK:STDOUT:     %bound_method.loc12_10.2: <bound method> = bound_method %n.ref, %specific_impl_fn.loc12_10.1
// CHECK:STDOUT:     %.loc11_36.1: ref @Class.F.%T.as_type.loc6_11.1 (%T.as_type) = splice_block %return.param.loc11 {}
// CHECK:STDOUT:     %Copy.WithSelf.Op.call: init @Class.F.%T.as_type.loc6_11.1 (%T.as_type) to %.loc11_36.1 = call %bound_method.loc12_10.2(%n.ref)
// CHECK:STDOUT:     return %Copy.WithSelf.Op.call to %return.param.loc11
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Class.G(@Class.%T.loc5_14.2: %Copy.type) {
// CHECK:STDOUT:   %T.loc7: %Copy.type = symbolic_binding T, 0 [symbolic = %T.loc7 (constants.%T.228)]
// CHECK:STDOUT:   %Class: type = class_type @Class, @Class(%T.loc7) [symbolic = %Class (constants.%Class)]
// CHECK:STDOUT:   %pattern_type.loc7_12: type = pattern_type %Class [symbolic = %pattern_type.loc7_12 (constants.%pattern_type.b47)]
// CHECK:STDOUT:   %T.as_type.loc7_25.1: type = facet_access_type %T.loc7 [symbolic = %T.as_type.loc7_25.1 (constants.%T.as_type)]
// CHECK:STDOUT:   %.loc7_25.1: Core.Form = init_form %T.as_type.loc7_25.1 [symbolic = %.loc7_25.1 (constants.%.9ff809.2)]
// CHECK:STDOUT:   %pattern_type.loc7_25: type = pattern_type %T.as_type.loc7_25.1 [symbolic = %pattern_type.loc7_25 (constants.%pattern_type.4b69d0.2)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc15: <witness> = require_complete_type %Class [symbolic = %require_complete.loc15 (constants.%require_complete.478)]
// CHECK:STDOUT:   %Class.elem: type = unbound_element_type %Class, %T.as_type.loc7_25.1 [symbolic = %Class.elem (constants.%Class.elem)]
// CHECK:STDOUT:   %require_complete.loc16: <witness> = require_complete_type %T.as_type.loc7_25.1 [symbolic = %require_complete.loc16 (constants.%require_complete.c28)]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%T.loc7) [symbolic = %Copy.WithSelf.Op.type (constants.%Copy.WithSelf.Op.type.d2ec0c.2)]
// CHECK:STDOUT:   %.loc16_14.3: type = fn_type_with_self_type %Copy.WithSelf.Op.type, %T.loc7 [symbolic = %.loc16_14.3 (constants.%.12f)]
// CHECK:STDOUT:   %Copy.lookup_impl_witness: <witness> = lookup_impl_witness %T.loc7, @Copy [symbolic = %Copy.lookup_impl_witness (constants.%Copy.lookup_impl_witness.c49)]
// CHECK:STDOUT:   %impl.elem0.loc16_14.2: @Class.G.%.loc16_14.3 (%.12f) = impl_witness_access %Copy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc16_14.2 (constants.%impl.elem0.cbd)]
// CHECK:STDOUT:   %specific_impl_fn.loc16_14.2: <specific function> = specific_impl_function %impl.elem0.loc16_14.2, @Copy.WithSelf.Op(%T.loc7) [symbolic = %specific_impl_fn.loc16_14.2 (constants.%specific_impl_fn.c5f)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param.loc15: @Class.G.%Class (%Class)) -> out %return.param.loc15: @Class.G.%T.as_type.loc7_25.1 (%T.as_type) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %self.ref: @Class.G.%Class (%Class) = name_ref self, %self.loc15
// CHECK:STDOUT:     %n.ref: @Class.G.%Class.elem (%Class.elem) = name_ref n, @Class.%.loc8 [concrete = @Class.%.loc8]
// CHECK:STDOUT:     %.loc16_14.1: ref @Class.G.%T.as_type.loc7_25.1 (%T.as_type) = class_element_access %self.ref, element0
// CHECK:STDOUT:     %.loc16_14.2: @Class.G.%T.as_type.loc7_25.1 (%T.as_type) = acquire_value %.loc16_14.1
// CHECK:STDOUT:     %impl.elem0.loc16_14.1: @Class.G.%.loc16_14.3 (%.12f) = impl_witness_access constants.%Copy.lookup_impl_witness.c49, element0 [symbolic = %impl.elem0.loc16_14.2 (constants.%impl.elem0.cbd)]
// CHECK:STDOUT:     %bound_method.loc16_14.1: <bound method> = bound_method %.loc16_14.2, %impl.elem0.loc16_14.1
// CHECK:STDOUT:     %specific_impl_fn.loc16_14.1: <specific function> = specific_impl_function %impl.elem0.loc16_14.1, @Copy.WithSelf.Op(constants.%T.228) [symbolic = %specific_impl_fn.loc16_14.2 (constants.%specific_impl_fn.c5f)]
// CHECK:STDOUT:     %bound_method.loc16_14.2: <bound method> = bound_method %.loc16_14.2, %specific_impl_fn.loc16_14.1
// CHECK:STDOUT:     %.loc15_44.1: ref @Class.G.%T.as_type.loc7_25.1 (%T.as_type) = splice_block %return.param.loc15 {}
// CHECK:STDOUT:     %Copy.WithSelf.Op.call: init @Class.G.%T.as_type.loc7_25.1 (%T.as_type) to %.loc15_44.1 = call %bound_method.loc16_14.2(%.loc16_14.2)
// CHECK:STDOUT:     return %Copy.WithSelf.Op.call to %return.param.loc15
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class(constants.%T.228) {
// CHECK:STDOUT:   %T.loc5_14.1 => constants.%T.228
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Class.F.type => constants.%Class.F.type
// CHECK:STDOUT:   %Class.F => constants.%Class.F
// CHECK:STDOUT:   %Class.G.type => constants.%Class.G.type
// CHECK:STDOUT:   %Class.G => constants.%Class.G
// CHECK:STDOUT:   %T.as_type => constants.%T.as_type
// CHECK:STDOUT:   %require_complete => constants.%require_complete.c28
// CHECK:STDOUT:   %Class => constants.%Class
// CHECK:STDOUT:   %Class.elem => constants.%Class.elem
// CHECK:STDOUT:   %struct_type.n => constants.%struct_type.n
// CHECK:STDOUT:   %complete_type.loc9_1.2 => constants.%complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class.F(constants.%T.228) {
// CHECK:STDOUT:   %T.loc6 => constants.%T.228
// CHECK:STDOUT:   %T.as_type.loc6_11.1 => constants.%T.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.4b69d0.2
// CHECK:STDOUT:   %.loc6_17.1 => constants.%.9ff809.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class.G(constants.%T.228) {
// CHECK:STDOUT:   %T.loc7 => constants.%T.228
// CHECK:STDOUT:   %Class => constants.%Class
// CHECK:STDOUT:   %pattern_type.loc7_12 => constants.%pattern_type.b47
// CHECK:STDOUT:   %T.as_type.loc7_25.1 => constants.%T.as_type
// CHECK:STDOUT:   %.loc7_25.1 => constants.%.9ff809.2
// CHECK:STDOUT:   %pattern_type.loc7_25 => constants.%pattern_type.4b69d0.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- nested.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = symbolic_binding .Self [symbolic_self]
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
// CHECK:STDOUT:   %pattern_type.f4b: type = pattern_type %B [symbolic]
// CHECK:STDOUT:   %B.F.type: type = fn_type @B.F, @B(%T, %_) [symbolic]
// CHECK:STDOUT:   %B.F: %B.F.type = struct_value () [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %require_complete.6d6: <witness> = require_complete_type %B [symbolic]
// CHECK:STDOUT:   %require_complete.944: <witness> = require_complete_type %T [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %A.decl: %A.type = class_decl @A [concrete = constants.%A.generic] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc5_13.1: type = splice_block %.loc5_13.2 [concrete = type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.loc5_13.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc5_10.2: type = symbolic_binding T, 0 [symbolic = %T.loc5_10.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %B.F.decl: %B.F.type = fn_decl @B.F [symbolic = constants.%B.F] {
// CHECK:STDOUT:     %self.param_patt: @B.F.%pattern_type.loc7_14 (%pattern_type.f4b) = value_param_pattern [concrete]
// CHECK:STDOUT:     %self.patt: @B.F.%pattern_type.loc7_14 (%pattern_type.f4b) = at_binding_pattern self, %self.param_patt [concrete]
// CHECK:STDOUT:     %a.param_patt: @B.F.%pattern_type.loc7_23 (%pattern_type.51d) = value_param_pattern [concrete]
// CHECK:STDOUT:     %a.patt: @B.F.%pattern_type.loc7_23 (%pattern_type.51d) = at_binding_pattern a, %a.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc11_10.1: type = splice_block %.loc11_10.2 [concrete = type] {
// CHECK:STDOUT:       %.Self.loc11_7: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.loc11_10.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc11: type = symbolic_binding T, 0 [symbolic = @A.%T.loc5_10.1 (constants.%T)]
// CHECK:STDOUT:     %.loc11_22: type = splice_block %T.ref.loc11_22 [symbolic = @B.%T (constants.%T)] {
// CHECK:STDOUT:       %.Self.loc11_19: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %T.ref.loc11_22: type = name_ref T, %T.loc11 [symbolic = @B.%T (constants.%T)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %_.loc11: @B.%T (%T) = symbolic_binding _, 1 [symbolic = @B.%_.loc6_12.1 (constants.%_)]
// CHECK:STDOUT:     %self.param.loc11: @B.F.%B (%B) = value_param call_param0
// CHECK:STDOUT:     %.loc11_40.1: type = splice_block %Self.ref.loc11 [symbolic = %B (constants.%B)] {
// CHECK:STDOUT:       %.loc11_40.2: type = specific_constant constants.%B, @B(constants.%T, constants.%_) [symbolic = %B (constants.%B)]
// CHECK:STDOUT:       %Self.ref.loc11: type = name_ref Self, %.loc11_40.2 [symbolic = %B (constants.%B)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %self.loc11: @B.F.%B (%B) = value_binding self, %self.param.loc11
// CHECK:STDOUT:     %a.param.loc11: @B.F.%T.loc7 (%T) = value_param call_param1
// CHECK:STDOUT:     %T.ref.loc11_56: type = name_ref T, %T.loc11 [symbolic = %T.loc7 (constants.%T)]
// CHECK:STDOUT:     %a.loc11: @B.F.%T.loc7 (%T) = value_binding a, %a.param.loc11
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @A(%T.loc5_10.2: type) {
// CHECK:STDOUT:   %T.loc5_10.1: type = symbolic_binding T, 0 [symbolic = %T.loc5_10.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %B.type: type = generic_class_type @B, @A(%T.loc5_10.1) [symbolic = %B.type (constants.%B.type)]
// CHECK:STDOUT:   %B.generic: @A.%B.type (%B.type) = struct_value () [symbolic = %B.generic (constants.%B.generic)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %B.decl: @A.%B.type (%B.type) = class_decl @B [symbolic = @A.%B.generic (constants.%B.generic)] {
// CHECK:STDOUT:       %_.patt: @B.%pattern_type (%pattern_type.51d) = symbolic_binding_pattern _, 1 [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %.loc6: type = splice_block %T.ref [symbolic = %T (constants.%T)] {
// CHECK:STDOUT:         %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:         %T.ref: type = name_ref T, @A.%T.loc5_10.2 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %_.loc6_12.2: @B.%T (%T) = symbolic_binding _, 1 [symbolic = %_.loc6_12.1 (constants.%_)]
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
// CHECK:STDOUT: generic class @B(@A.%T.loc5_10.2: type, %_.loc6_12.2: @B.%T (%T)) {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %_.loc6_12.1: @B.%T (%T) = symbolic_binding _, 1 [symbolic = %_.loc6_12.1 (constants.%_)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %T [symbolic = %pattern_type (constants.%pattern_type.51d)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %B.F.type: type = fn_type @B.F, @B(%T, %_.loc6_12.1) [symbolic = %B.F.type (constants.%B.F.type)]
// CHECK:STDOUT:   %B.F: @B.%B.F.type (%B.F.type) = struct_value () [symbolic = %B.F (constants.%B.F)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %B.F.decl: @B.%B.F.type (%B.F.type) = fn_decl @B.F [symbolic = @B.%B.F (constants.%B.F)] {
// CHECK:STDOUT:       %self.param_patt: @B.F.%pattern_type.loc7_14 (%pattern_type.f4b) = value_param_pattern [concrete]
// CHECK:STDOUT:       %self.patt: @B.F.%pattern_type.loc7_14 (%pattern_type.f4b) = at_binding_pattern self, %self.param_patt [concrete]
// CHECK:STDOUT:       %a.param_patt: @B.F.%pattern_type.loc7_23 (%pattern_type.51d) = value_param_pattern [concrete]
// CHECK:STDOUT:       %a.patt: @B.F.%pattern_type.loc7_23 (%pattern_type.51d) = at_binding_pattern a, %a.param_patt [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %self.param.loc7: @B.F.%B (%B) = value_param call_param0
// CHECK:STDOUT:       %.loc7_16.1: type = splice_block %Self.ref.loc7 [symbolic = %B (constants.%B)] {
// CHECK:STDOUT:         %.loc7_16.2: type = specific_constant constants.%B, @B(constants.%T, constants.%_) [symbolic = %B (constants.%B)]
// CHECK:STDOUT:         %Self.ref.loc7: type = name_ref Self, %.loc7_16.2 [symbolic = %B (constants.%B)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %self.loc7: @B.F.%B (%B) = value_binding self, %self.param.loc7
// CHECK:STDOUT:       %a.param.loc7: @B.F.%T.loc7 (%T) = value_param call_param1
// CHECK:STDOUT:       %T.ref.loc7: type = name_ref T, @A.%T.loc5_10.2 [symbolic = %T.loc7 (constants.%T)]
// CHECK:STDOUT:       %a.loc7: @B.F.%T.loc7 (%T) = value_binding a, %a.param.loc7
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
// CHECK:STDOUT: generic fn @B.F(@A.%T.loc5_10.2: type, @B.%_.loc6_12.2: @B.%T (%T)) {
// CHECK:STDOUT:   %T.loc7: type = symbolic_binding T, 0 [symbolic = %T.loc7 (constants.%T)]
// CHECK:STDOUT:   %_.loc7: @B.F.%T.loc7 (%T) = symbolic_binding _, 1 [symbolic = %_.loc7 (constants.%_)]
// CHECK:STDOUT:   %B: type = class_type @B, @B(%T.loc7, %_.loc7) [symbolic = %B (constants.%B)]
// CHECK:STDOUT:   %pattern_type.loc7_14: type = pattern_type %B [symbolic = %pattern_type.loc7_14 (constants.%pattern_type.f4b)]
// CHECK:STDOUT:   %pattern_type.loc7_23: type = pattern_type %T.loc7 [symbolic = %pattern_type.loc7_23 (constants.%pattern_type.51d)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc11_38: <witness> = require_complete_type %B [symbolic = %require_complete.loc11_38 (constants.%require_complete.6d6)]
// CHECK:STDOUT:   %require_complete.loc11_54: <witness> = require_complete_type %T.loc7 [symbolic = %require_complete.loc11_54 (constants.%require_complete.944)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param.loc11: @B.F.%B (%B), %a.param.loc11: @B.F.%T.loc7 (%T)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A(constants.%T) {
// CHECK:STDOUT:   %T.loc5_10.1 => constants.%T
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %B.type => constants.%B.type
// CHECK:STDOUT:   %B.generic => constants.%B.generic
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @B(constants.%T, constants.%_) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %_.loc6_12.1 => constants.%_
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.51d
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %B.F.type => constants.%B.F.type
// CHECK:STDOUT:   %B.F => constants.%B.F
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @B.F(constants.%T, constants.%_) {
// CHECK:STDOUT:   %T.loc7 => constants.%T
// CHECK:STDOUT:   %_.loc7 => constants.%_
// CHECK:STDOUT:   %B => constants.%B
// CHECK:STDOUT:   %pattern_type.loc7_14 => constants.%pattern_type.f4b
// CHECK:STDOUT:   %pattern_type.loc7_23 => constants.%pattern_type.51d
// CHECK:STDOUT: }
// CHECK:STDOUT:
