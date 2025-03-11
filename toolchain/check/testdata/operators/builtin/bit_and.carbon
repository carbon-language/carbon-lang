// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/operators/builtin/bit_and.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/operators/builtin/bit_and.carbon

// --- bit_and_facet_types.carbon
library "[[@TEST_NAME]]";

interface I {}
interface J {}

fn F[T:! I & J](t: T) {}

// --- fail_bit_and_non_facet_types.carbon
library "[[@TEST_NAME]]";

interface I {}
class J {}

// CHECK:STDERR: fail_bit_and_non_facet_types.carbon:[[@LINE+4]]:10: error: non-facet type `J` combined with `&` operator [FacetTypeRequiredForTypeAndOperator]
// CHECK:STDERR: fn F[T:! I & J](t: T) {}
// CHECK:STDERR:          ^~~~~
// CHECK:STDERR:
fn F[T:! I & J](t: T) {}

// --- bit_and_values_with_impl.carbon
library "[[@TEST_NAME]]";

class J { adapt {}; }
impl J as Core.BitAnd {
  fn Op[self: Self](other: Self) -> Self { return {} as J; }
}

fn F() {
  let a: J = {} as J;
  let b: J = {} as J;
  a & b;
}

// --- fail_bit_and_values_no_impl.carbon
library "[[@TEST_NAME]]";

class J { adapt {}; }

fn F() {
  let a: J = {} as J;
  let b: J = {} as J;
  // CHECK:STDERR: fail_bit_and_values_no_impl.carbon:[[@LINE+4]]:3: error: cannot access member of interface `Core.BitAnd` in type `J` that does not implement that interface [MissingImplInMemberAccess]
  // CHECK:STDERR:   a & b;
  // CHECK:STDERR:   ^~~~~
  // CHECK:STDERR:
  a & b;
}

// CHECK:STDOUT: --- bit_and_facet_types.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [concrete]
// CHECK:STDOUT:   %Self.826: %I.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %J.type: type = facet_type <@J> [concrete]
// CHECK:STDOUT:   %Self.ccd: %J.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %BitAnd.type: type = facet_type <@BitAnd> [concrete]
// CHECK:STDOUT:   %Op.type.27a: type = fn_type @Op.1 [concrete]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (imports.%Core.import_ref.d90) [concrete]
// CHECK:STDOUT:   %BitAnd.facet: %BitAnd.type = facet_value type, (%impl_witness) [concrete]
// CHECK:STDOUT:   %.3a2: type = fn_type_with_self_type %Op.type.27a, %BitAnd.facet [concrete]
// CHECK:STDOUT:   %Op.type.c1b: type = fn_type @Op.2 [concrete]
// CHECK:STDOUT:   %Op.0a6: %Op.type.c1b = struct_value () [concrete]
// CHECK:STDOUT:   %Op.bound: <bound method> = bound_method %I.type, %Op.0a6 [concrete]
// CHECK:STDOUT:   %facet_type: type = facet_type <@I & @J> [concrete]
// CHECK:STDOUT:   %T: %facet_type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt: %facet_type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type: type = facet_access_type %T [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T.as_type [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .BitAnd = %Core.BitAnd
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .I = %I.decl
// CHECK:STDOUT:     .J = %J.decl
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %I.decl: type = interface_decl @I [concrete = constants.%I.type] {} {}
// CHECK:STDOUT:   %J.decl: type = interface_decl @J [concrete = constants.%J.type] {} {}
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %T.patt.loc6_6.1: %facet_type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc6_6.2 (constants.%T.patt)]
// CHECK:STDOUT:     %t.patt: @F.%T.as_type.loc6_20.2 (%T.as_type) = binding_pattern t
// CHECK:STDOUT:     %t.param_patt: @F.%T.as_type.loc6_20.2 (%T.as_type) = value_param_pattern %t.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc6_12.1: type = splice_block %.loc6_12.3 [concrete = constants.%facet_type] {
// CHECK:STDOUT:       %I.ref: type = name_ref I, file.%I.decl [concrete = constants.%I.type]
// CHECK:STDOUT:       %J.ref: type = name_ref J, file.%J.decl [concrete = constants.%J.type]
// CHECK:STDOUT:       %impl.elem0: %.3a2 = impl_witness_access constants.%impl_witness, element0 [concrete = constants.%Op.0a6]
// CHECK:STDOUT:       %bound_method: <bound method> = bound_method %I.ref, %impl.elem0 [concrete = constants.%Op.bound]
// CHECK:STDOUT:       %type.and: init type = call %bound_method(%I.ref, %J.ref) [concrete = constants.%facet_type]
// CHECK:STDOUT:       %.loc6_12.2: type = value_of_initializer %type.and [concrete = constants.%facet_type]
// CHECK:STDOUT:       %.loc6_12.3: type = converted %type.and, %.loc6_12.2 [concrete = constants.%facet_type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc6_6.1: %facet_type = bind_symbolic_name T, 0 [symbolic = %T.loc6_6.2 (constants.%T)]
// CHECK:STDOUT:     %t.param: @F.%T.as_type.loc6_20.2 (%T.as_type) = value_param call_param0
// CHECK:STDOUT:     %.loc6_20.1: type = splice_block %.loc6_20.2 [symbolic = %T.as_type.loc6_20.2 (constants.%T.as_type)] {
// CHECK:STDOUT:       %T.ref: %facet_type = name_ref T, %T.loc6_6.1 [symbolic = %T.loc6_6.2 (constants.%T)]
// CHECK:STDOUT:       %T.as_type.loc6_20.1: type = facet_access_type %T.ref [symbolic = %T.as_type.loc6_20.2 (constants.%T.as_type)]
// CHECK:STDOUT:       %.loc6_20.2: type = converted %T.ref, %T.as_type.loc6_20.1 [symbolic = %T.as_type.loc6_20.2 (constants.%T.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %t: @F.%T.as_type.loc6_20.2 (%T.as_type) = bind_name t, %t.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I {
// CHECK:STDOUT:   %Self: %I.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.826]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @J {
// CHECK:STDOUT:   %Self: %J.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.ccd]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%T.loc6_6.1: %facet_type) {
// CHECK:STDOUT:   %T.loc6_6.2: %facet_type = bind_symbolic_name T, 0 [symbolic = %T.loc6_6.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc6_6.2: %facet_type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc6_6.2 (constants.%T.patt)]
// CHECK:STDOUT:   %T.as_type.loc6_20.2: type = facet_access_type %T.loc6_6.2 [symbolic = %T.as_type.loc6_20.2 (constants.%T.as_type)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type @F.%T.as_type.loc6_20.2 (%T.as_type) [symbolic = %require_complete (constants.%require_complete)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%T.patt.loc6_6.1: %facet_type](%t.param_patt: @F.%T.as_type.loc6_20.2 (%T.as_type)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T) {
// CHECK:STDOUT:   %T.loc6_6.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc6_6.2 => constants.%T
// CHECK:STDOUT:   %T.as_type.loc6_20.2 => constants.%T.as_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_bit_and_non_facet_types.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [concrete]
// CHECK:STDOUT:   %Self.826: %I.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %J: type = class_type @J [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %BitAnd.type: type = facet_type <@BitAnd> [concrete]
// CHECK:STDOUT:   %Op.type.27a: type = fn_type @Op.1 [concrete]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (imports.%Core.import_ref.d90) [concrete]
// CHECK:STDOUT:   %BitAnd.facet: %BitAnd.type = facet_value type, (%impl_witness) [concrete]
// CHECK:STDOUT:   %.3a2: type = fn_type_with_self_type %Op.type.27a, %BitAnd.facet [concrete]
// CHECK:STDOUT:   %Op.type.c1b: type = fn_type @Op.2 [concrete]
// CHECK:STDOUT:   %Op.0a6: %Op.type.c1b = struct_value () [concrete]
// CHECK:STDOUT:   %Op.bound: <bound method> = bound_method %I.type, %Op.0a6 [concrete]
// CHECK:STDOUT:   %T.patt: <error> = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .BitAnd = %Core.BitAnd
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .I = %I.decl
// CHECK:STDOUT:     .J = %J.decl
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %I.decl: type = interface_decl @I [concrete = constants.%I.type] {} {}
// CHECK:STDOUT:   %J.decl: type = class_decl @J [concrete = constants.%J] {} {}
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %T.patt.loc10_6.1: <error> = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc10_6.2 (constants.%T.patt)]
// CHECK:STDOUT:     %t.patt: <error> = binding_pattern t
// CHECK:STDOUT:     %t.param_patt: <error> = value_param_pattern %t.patt, call_param0 [concrete = <error>]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc10_12.1: type = splice_block %.loc10_12.3 [concrete = <error>] {
// CHECK:STDOUT:       %I.ref: type = name_ref I, file.%I.decl [concrete = constants.%I.type]
// CHECK:STDOUT:       %J.ref: type = name_ref J, file.%J.decl [concrete = constants.%J]
// CHECK:STDOUT:       %impl.elem0: %.3a2 = impl_witness_access constants.%impl_witness, element0 [concrete = constants.%Op.0a6]
// CHECK:STDOUT:       %bound_method: <bound method> = bound_method %I.ref, %impl.elem0 [concrete = constants.%Op.bound]
// CHECK:STDOUT:       %type.and: init type = call %bound_method(%I.ref, %J.ref) [concrete = <error>]
// CHECK:STDOUT:       %.loc10_12.2: type = value_of_initializer %type.and [concrete = <error>]
// CHECK:STDOUT:       %.loc10_12.3: type = converted %type.and, %.loc10_12.2 [concrete = <error>]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T: <error> = bind_symbolic_name T, 0 [concrete = <error>]
// CHECK:STDOUT:     %t.param: <error> = value_param call_param0
// CHECK:STDOUT:     %T.ref: <error> = name_ref T, %T [concrete = <error>]
// CHECK:STDOUT:     %t: <error> = bind_name t, %t.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I {
// CHECK:STDOUT:   %Self: %I.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.826]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @J {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%J
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%T: <error>) {
// CHECK:STDOUT:   %T.patt.loc10_6.2: <error> = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc10_6.2 (constants.%T.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%T.patt.loc10_6.1: <error>](%t.param_patt: <error>) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(<error>) {
// CHECK:STDOUT:   %T.patt.loc10_6.2 => <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- bit_and_values_with_impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %J: type = class_type @J [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %BitAnd.type: type = facet_type <@BitAnd> [concrete]
// CHECK:STDOUT:   %Op.type.27a: type = fn_type @Op.1 [concrete]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (@impl.d19.%Op.decl) [concrete]
// CHECK:STDOUT:   %Op.type.18f: type = fn_type @Op.2 [concrete]
// CHECK:STDOUT:   %Op.cc7: %Op.type.18f = struct_value () [concrete]
// CHECK:STDOUT:   %BitAnd.facet: %BitAnd.type = facet_value %J, (%impl_witness) [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %J.val: %J = struct_value () [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %.d77: type = fn_type_with_self_type %Op.type.27a, %BitAnd.facet [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .BitAnd = %Core.BitAnd
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.BitAnd: type = import_ref Core//prelude/operators/bitwise, BitAnd, loaded [concrete = constants.%BitAnd.type]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .J = %J.decl
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %J.decl: type = class_decl @J [concrete = constants.%J] {} {}
// CHECK:STDOUT:   impl_decl @impl.d19 [concrete] {} {
// CHECK:STDOUT:     %J.ref: type = name_ref J, file.%J.decl [concrete = constants.%J]
// CHECK:STDOUT:     %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:     %BitAnd.ref: type = name_ref BitAnd, imports.%Core.BitAnd [concrete = constants.%BitAnd.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (@impl.d19.%Op.decl) [concrete = constants.%impl_witness]
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.d19: %J.ref as %BitAnd.ref {
// CHECK:STDOUT:   %Op.decl: %Op.type.18f = fn_decl @Op.2 [concrete = constants.%Op.cc7] {
// CHECK:STDOUT:     %self.patt: %J = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: %J = value_param_pattern %self.patt, call_param0
// CHECK:STDOUT:     %other.patt: %J = binding_pattern other
// CHECK:STDOUT:     %other.param_patt: %J = value_param_pattern %other.patt, call_param1
// CHECK:STDOUT:     %return.patt: %J = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %J = out_param_pattern %return.patt, call_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Self.ref.loc5_37: type = name_ref Self, @impl.d19.%J.ref [concrete = constants.%J]
// CHECK:STDOUT:     %self.param: %J = value_param call_param0
// CHECK:STDOUT:     %Self.ref.loc5_15: type = name_ref Self, @impl.d19.%J.ref [concrete = constants.%J]
// CHECK:STDOUT:     %self: %J = bind_name self, %self.param
// CHECK:STDOUT:     %other.param: %J = value_param call_param1
// CHECK:STDOUT:     %Self.ref.loc5_28: type = name_ref Self, @impl.d19.%J.ref [concrete = constants.%J]
// CHECK:STDOUT:     %other: %J = bind_name other, %other.param
// CHECK:STDOUT:     %return.param: ref %J = out_param call_param2
// CHECK:STDOUT:     %return: ref %J = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Op = %Op.decl
// CHECK:STDOUT:   .J = <poisoned>
// CHECK:STDOUT:   witness = file.%impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @J {
// CHECK:STDOUT:   %.loc3_18: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %.loc3_19: type = converted %.loc3_18, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:   adapt_decl %.loc3_19 [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%J
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Op.2[%self.param_patt: %J](%other.param_patt: %J) -> %J {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc5_52: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %J.ref: type = name_ref J, file.%J.decl [concrete = constants.%J]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc5_54.1: %J = as_compatible %empty_struct [concrete = constants.%J.val]
// CHECK:STDOUT:   %.loc5_54.2: %J = converted %.loc5_52, %.loc5_54.1 [concrete = constants.%J.val]
// CHECK:STDOUT:   return %.loc5_54.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %J = binding_pattern a
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc9_15: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %J.ref.loc9_20: type = name_ref J, file.%J.decl [concrete = constants.%J]
// CHECK:STDOUT:   %empty_struct.loc9: %empty_struct_type = struct_value () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc9_17.1: %J = as_compatible %empty_struct.loc9 [concrete = constants.%J.val]
// CHECK:STDOUT:   %.loc9_17.2: %J = converted %.loc9_15, %.loc9_17.1 [concrete = constants.%J.val]
// CHECK:STDOUT:   %J.ref.loc9_10: type = name_ref J, file.%J.decl [concrete = constants.%J]
// CHECK:STDOUT:   %a: %J = bind_name a, %.loc9_17.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %J = binding_pattern b
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc10_15: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %J.ref.loc10_20: type = name_ref J, file.%J.decl [concrete = constants.%J]
// CHECK:STDOUT:   %empty_struct.loc10: %empty_struct_type = struct_value () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc10_17.1: %J = as_compatible %empty_struct.loc10 [concrete = constants.%J.val]
// CHECK:STDOUT:   %.loc10_17.2: %J = converted %.loc10_15, %.loc10_17.1 [concrete = constants.%J.val]
// CHECK:STDOUT:   %J.ref.loc10_10: type = name_ref J, file.%J.decl [concrete = constants.%J]
// CHECK:STDOUT:   %b: %J = bind_name b, %.loc10_17.2
// CHECK:STDOUT:   %a.ref: %J = name_ref a, %a
// CHECK:STDOUT:   %b.ref: %J = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0: %.d77 = impl_witness_access constants.%impl_witness, element0 [concrete = constants.%Op.cc7]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %a.ref, %impl.elem0
// CHECK:STDOUT:   %Op.call: init %J = call %bound_method(%a.ref, %b.ref)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_bit_and_values_no_impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %J: type = class_type @J [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %J.val: %J = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .BitAnd = %Core.BitAnd
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .J = %J.decl
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %J.decl: type = class_decl @J [concrete = constants.%J] {} {}
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @J {
// CHECK:STDOUT:   %.loc3_18: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %.loc3_19: type = converted %.loc3_18, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:   adapt_decl %.loc3_19 [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%J
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %J = binding_pattern a
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc6_15: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %J.ref.loc6_20: type = name_ref J, file.%J.decl [concrete = constants.%J]
// CHECK:STDOUT:   %empty_struct.loc6: %empty_struct_type = struct_value () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc6_17.1: %J = as_compatible %empty_struct.loc6 [concrete = constants.%J.val]
// CHECK:STDOUT:   %.loc6_17.2: %J = converted %.loc6_15, %.loc6_17.1 [concrete = constants.%J.val]
// CHECK:STDOUT:   %J.ref.loc6_10: type = name_ref J, file.%J.decl [concrete = constants.%J]
// CHECK:STDOUT:   %a: %J = bind_name a, %.loc6_17.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %J = binding_pattern b
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc7_15: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %J.ref.loc7_20: type = name_ref J, file.%J.decl [concrete = constants.%J]
// CHECK:STDOUT:   %empty_struct.loc7: %empty_struct_type = struct_value () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc7_17.1: %J = as_compatible %empty_struct.loc7 [concrete = constants.%J.val]
// CHECK:STDOUT:   %.loc7_17.2: %J = converted %.loc7_15, %.loc7_17.1 [concrete = constants.%J.val]
// CHECK:STDOUT:   %J.ref.loc7_10: type = name_ref J, file.%J.decl [concrete = constants.%J]
// CHECK:STDOUT:   %b: %J = bind_name b, %.loc7_17.2
// CHECK:STDOUT:   %a.ref: %J = name_ref a, %a
// CHECK:STDOUT:   %b.ref: %J = name_ref b, %b
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
