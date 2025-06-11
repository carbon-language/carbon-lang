// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// TODO: Add ranges and switch to "--dump-sem-ir-ranges=only".
// EXTRA-ARGS: --dump-sem-ir-ranges=if-present
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/operators/overloaded/fail_no_impl_for_arg.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/operators/overloaded/fail_no_impl_for_arg.carbon

package User;

class C {};
class D {};

impl C as Core.AddWith(C) where .Result = C {
  fn Op[self: C](other: C) -> C;
}
impl C as Core.AddAssignWith(C) {
  fn Op[addr self: C*](other: C);
}

fn Test(a: C, b: D) -> C {
  // CHECK:STDERR: fail_no_impl_for_arg.carbon:[[@LINE+4]]:10: error: cannot access member of interface `Core.AddWith(D)` in type `C` that does not implement that interface [MissingImplInMemberAccess]
  // CHECK:STDERR:   return a + b;
  // CHECK:STDERR:          ^~~~~
  // CHECK:STDERR:
  return a + b;
}

fn TestAssign(b: D) {
  var a: C = {};
  // CHECK:STDERR: fail_no_impl_for_arg.carbon:[[@LINE+4]]:3: error: cannot access member of interface `Core.AddAssignWith(D)` in type `C` that does not implement that interface [MissingImplInMemberAccess]
  // CHECK:STDERR:   a += b;
  // CHECK:STDERR:   ^~~~~~
  // CHECK:STDERR:
  a += b;
}

// CHECK:STDOUT: --- fail_no_impl_for_arg.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %D: type = class_type @D [concrete]
// CHECK:STDOUT:   %AddWith.type.e05: type = generic_interface_type @AddWith [concrete]
// CHECK:STDOUT:   %AddWith.generic: %AddWith.type.e05 = struct_value () [concrete]
// CHECK:STDOUT:   %Other: type = bind_symbolic_name Other, 0 [symbolic]
// CHECK:STDOUT:   %AddWith.type.6d3: type = facet_type <@AddWith, @AddWith(%Other)> [symbolic]
// CHECK:STDOUT:   %Self.6c1: %AddWith.type.6d3 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %AddWith.assoc_type.c10: type = assoc_entity_type @AddWith, @AddWith(%Other) [symbolic]
// CHECK:STDOUT:   %AddWith.type.ffb: type = facet_type <@AddWith, @AddWith(%C)> [concrete]
// CHECK:STDOUT:   %.Self.652: %AddWith.type.ffb = bind_symbolic_name .Self [symbolic_self]
// CHECK:STDOUT:   %AddWith.assoc_type.ee0: type = assoc_entity_type @AddWith, @AddWith(%C) [concrete]
// CHECK:STDOUT:   %assoc0.6c8: %AddWith.assoc_type.ee0 = assoc_entity element0, imports.%Core.import_ref.830 [concrete]
// CHECK:STDOUT:   %assoc0.0a5: %AddWith.assoc_type.c10 = assoc_entity element0, imports.%Core.import_ref.f3db8d.1 [symbolic]
// CHECK:STDOUT:   %.Self.as_type: type = facet_access_type %.Self.652 [symbolic_self]
// CHECK:STDOUT:   %AddWith.lookup_impl_witness.4a9: <witness> = lookup_impl_witness %.Self.652, @AddWith, @AddWith(%C) [symbolic_self]
// CHECK:STDOUT:   %AddWith.facet.37f: %AddWith.type.ffb = facet_value %.Self.as_type, (%AddWith.lookup_impl_witness.4a9) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.4ba: type = impl_witness_access %AddWith.lookup_impl_witness.4a9, element0 [symbolic_self]
// CHECK:STDOUT:   %AddWith_where.type.57e: type = facet_type <@AddWith, @AddWith(%C) where %impl.elem0.4ba = %C> [concrete]
// CHECK:STDOUT:   %AddWith.impl_witness: <witness> = impl_witness file.%AddWith.impl_witness_table [concrete]
// CHECK:STDOUT:   %pattern_type.c48: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %Op.type.638: type = fn_type @Op.6 [concrete]
// CHECK:STDOUT:   %Op.276: %Op.type.638 = struct_value () [concrete]
// CHECK:STDOUT:   %AddAssignWith.type.fc6: type = generic_interface_type @AddAssignWith [concrete]
// CHECK:STDOUT:   %AddAssignWith.generic: %AddAssignWith.type.fc6 = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.f6d: type = pattern_type auto [concrete]
// CHECK:STDOUT:   %AddAssignWith.type.393: type = facet_type <@AddAssignWith, @AddAssignWith(%C)> [concrete]
// CHECK:STDOUT:   %AddAssignWith.impl_witness: <witness> = impl_witness file.%AddAssignWith.impl_witness_table [concrete]
// CHECK:STDOUT:   %ptr.019: type = ptr_type %C [concrete]
// CHECK:STDOUT:   %pattern_type.44a: type = pattern_type %ptr.019 [concrete]
// CHECK:STDOUT:   %Op.type.0d7: type = fn_type @Op.8 [concrete]
// CHECK:STDOUT:   %Op.572: %Op.type.0d7 = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.510: type = pattern_type %D [concrete]
// CHECK:STDOUT:   %Test.type: type = fn_type @Test [concrete]
// CHECK:STDOUT:   %Test: %Test.type = struct_value () [concrete]
// CHECK:STDOUT:   %TestAssign.type: type = fn_type @TestAssign [concrete]
// CHECK:STDOUT:   %TestAssign: %TestAssign.type = struct_value () [concrete]
// CHECK:STDOUT:   %C.val: %C = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .AddWith = %Core.AddWith
// CHECK:STDOUT:     .AddAssignWith = %Core.AddAssignWith
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.AddWith: %AddWith.type.e05 = import_ref Core//prelude/operators/arithmetic, AddWith, loaded [concrete = constants.%AddWith.generic]
// CHECK:STDOUT:   %Core.import_ref.7ab: @AddWith.%AddWith.assoc_type (%AddWith.assoc_type.c10) = import_ref Core//prelude/operators/arithmetic, loc16_13, loaded [symbolic = @AddWith.%assoc0 (constants.%assoc0.0a5)]
// CHECK:STDOUT:   %Core.import_ref.830: type = import_ref Core//prelude/operators/arithmetic, loc16_13, loaded [concrete = %Result]
// CHECK:STDOUT:   %Core.import_ref.f3db8d.1 = import_ref Core//prelude/operators/arithmetic, loc16_13, unloaded
// CHECK:STDOUT:   %Result: type = assoc_const_decl @Result [concrete] {}
// CHECK:STDOUT:   %Core.AddAssignWith: %AddAssignWith.type.fc6 = import_ref Core//prelude/operators/arithmetic, AddAssignWith, loaded [concrete = constants.%AddAssignWith.generic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .D = %D.decl
// CHECK:STDOUT:     .Test = %Test.decl
// CHECK:STDOUT:     .TestAssign = %TestAssign.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   %D.decl: type = class_decl @D [concrete = constants.%D] {} {}
// CHECK:STDOUT:   impl_decl @impl.be8 [concrete] {} {
// CHECK:STDOUT:     %C.ref.loc19_6: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:     %AddWith.ref: %AddWith.type.e05 = name_ref AddWith, imports.%Core.AddWith [concrete = constants.%AddWith.generic]
// CHECK:STDOUT:     %C.ref.loc19_24: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %AddWith.type: type = facet_type <@AddWith, @AddWith(constants.%C)> [concrete = constants.%AddWith.type.ffb]
// CHECK:STDOUT:     %.Self: %AddWith.type.ffb = bind_symbolic_name .Self [symbolic_self = constants.%.Self.652]
// CHECK:STDOUT:     %.Self.ref: %AddWith.type.ffb = name_ref .Self, %.Self [symbolic_self = constants.%.Self.652]
// CHECK:STDOUT:     %.loc19_33.1: %AddWith.assoc_type.ee0 = specific_constant imports.%Core.import_ref.7ab, @AddWith(constants.%C) [concrete = constants.%assoc0.6c8]
// CHECK:STDOUT:     %Result.ref: %AddWith.assoc_type.ee0 = name_ref Result, %.loc19_33.1 [concrete = constants.%assoc0.6c8]
// CHECK:STDOUT:     %.Self.as_type: type = facet_access_type %.Self.ref [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:     %.loc19_33.2: type = converted %.Self.ref, %.Self.as_type [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:     %impl.elem0: type = impl_witness_access constants.%AddWith.lookup_impl_witness.4a9, element0 [symbolic_self = constants.%impl.elem0.4ba]
// CHECK:STDOUT:     %C.ref.loc19_43: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %.loc19_27: type = where_expr %.Self [concrete = constants.%AddWith_where.type.57e] {
// CHECK:STDOUT:       requirement_rewrite %impl.elem0, %C.ref.loc19_43
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %AddWith.impl_witness_table = impl_witness_table (%impl_witness_assoc_constant, @impl.be8.%Op.decl), @impl.be8 [concrete]
// CHECK:STDOUT:   %AddWith.impl_witness: <witness> = impl_witness %AddWith.impl_witness_table [concrete = constants.%AddWith.impl_witness]
// CHECK:STDOUT:   %impl_witness_assoc_constant: type = impl_witness_assoc_constant constants.%C [concrete = constants.%C]
// CHECK:STDOUT:   impl_decl @impl.a95 [concrete] {} {
// CHECK:STDOUT:     %C.ref.loc22_6: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:     %AddAssignWith.ref: %AddAssignWith.type.fc6 = name_ref AddAssignWith, imports.%Core.AddAssignWith [concrete = constants.%AddAssignWith.generic]
// CHECK:STDOUT:     %C.ref.loc22_30: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %AddAssignWith.type: type = facet_type <@AddAssignWith, @AddAssignWith(constants.%C)> [concrete = constants.%AddAssignWith.type.393]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %AddAssignWith.impl_witness_table = impl_witness_table (@impl.a95.%Op.decl), @impl.a95 [concrete]
// CHECK:STDOUT:   %AddAssignWith.impl_witness: <witness> = impl_witness %AddAssignWith.impl_witness_table [concrete = constants.%AddAssignWith.impl_witness]
// CHECK:STDOUT:   %Test.decl: %Test.type = fn_decl @Test [concrete = constants.%Test] {
// CHECK:STDOUT:     %a.patt: %pattern_type.c48 = binding_pattern a [concrete]
// CHECK:STDOUT:     %a.param_patt: %pattern_type.c48 = value_param_pattern %a.patt, call_param0 [concrete]
// CHECK:STDOUT:     %b.patt: %pattern_type.510 = binding_pattern b [concrete]
// CHECK:STDOUT:     %b.param_patt: %pattern_type.510 = value_param_pattern %b.patt, call_param1 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.c48 = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.c48 = out_param_pattern %return.patt, call_param2 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref.loc26_24: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %a.param: %C = value_param call_param0
// CHECK:STDOUT:     %C.ref.loc26_12: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %a: %C = bind_name a, %a.param
// CHECK:STDOUT:     %b.param: %D = value_param call_param1
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %b: %D = bind_name b, %b.param
// CHECK:STDOUT:     %return.param: ref %C = out_param call_param2
// CHECK:STDOUT:     %return: ref %C = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %TestAssign.decl: %TestAssign.type = fn_decl @TestAssign [concrete = constants.%TestAssign] {
// CHECK:STDOUT:     %b.patt: %pattern_type.510 = binding_pattern b [concrete]
// CHECK:STDOUT:     %b.param_patt: %pattern_type.510 = value_param_pattern %b.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %b.param: %D = value_param call_param0
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %b: %D = bind_name b, %b.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.be8: %C.ref.loc19_6 as %.loc19_27 {
// CHECK:STDOUT:   %Op.decl: %Op.type.638 = fn_decl @Op.6 [concrete = constants.%Op.276] {
// CHECK:STDOUT:     %self.patt: %pattern_type.c48 = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.c48 = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %other.patt: %pattern_type.c48 = binding_pattern other [concrete]
// CHECK:STDOUT:     %other.param_patt: %pattern_type.c48 = value_param_pattern %other.patt, call_param1 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.c48 = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.c48 = out_param_pattern %return.patt, call_param2 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref.loc20_31: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %self.param: %C = value_param call_param0
// CHECK:STDOUT:     %C.ref.loc20_15: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %self: %C = bind_name self, %self.param
// CHECK:STDOUT:     %other.param: %C = value_param call_param1
// CHECK:STDOUT:     %C.ref.loc20_25: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %other: %C = bind_name other, %other.param
// CHECK:STDOUT:     %return.param: ref %C = out_param call_param2
// CHECK:STDOUT:     %return: ref %C = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .C = <poisoned>
// CHECK:STDOUT:   .Op = %Op.decl
// CHECK:STDOUT:   witness = file.%AddWith.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.a95: %C.ref.loc22_6 as %AddAssignWith.type {
// CHECK:STDOUT:   %Op.decl: %Op.type.0d7 = fn_decl @Op.8 [concrete = constants.%Op.572] {
// CHECK:STDOUT:     %self.patt: %pattern_type.44a = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.44a = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %.loc23_9: %pattern_type.f6d = addr_pattern %self.param_patt [concrete]
// CHECK:STDOUT:     %other.patt: %pattern_type.c48 = binding_pattern other [concrete]
// CHECK:STDOUT:     %other.param_patt: %pattern_type.c48 = value_param_pattern %other.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %ptr.019 = value_param call_param0
// CHECK:STDOUT:     %.loc23_21: type = splice_block %ptr [concrete = constants.%ptr.019] {
// CHECK:STDOUT:       %C.ref.loc23_20: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:       %ptr: type = ptr_type %C.ref.loc23_20 [concrete = constants.%ptr.019]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %self: %ptr.019 = bind_name self, %self.param
// CHECK:STDOUT:     %other.param: %C = value_param call_param1
// CHECK:STDOUT:     %C.ref.loc23_31: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %other: %C = bind_name other, %other.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .C = <poisoned>
// CHECK:STDOUT:   .Op = %Op.decl
// CHECK:STDOUT:   witness = file.%AddAssignWith.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @D {
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%D
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Op.6(%self.param: %C, %other.param: %C) -> %return.param: %C;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Op.8(%self.param: %ptr.019, %other.param: %C);
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Test(%a.param: %C, %b.param: %D) -> %return.param: %C {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: %C = name_ref a, %a
// CHECK:STDOUT:   %b.ref: %D = name_ref b, %b
// CHECK:STDOUT:   return <error> to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestAssign(%b.param: %D) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.c48 = binding_pattern a [concrete]
// CHECK:STDOUT:     %a.var_patt: %pattern_type.c48 = var_pattern %a.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.var: ref %C = var %a.var_patt
// CHECK:STDOUT:   %.loc35_15.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %.loc35_15.2: init %C = class_init (), %a.var [concrete = constants.%C.val]
// CHECK:STDOUT:   %.loc35_3: init %C = converted %.loc35_15.1, %.loc35_15.2 [concrete = constants.%C.val]
// CHECK:STDOUT:   assign %a.var, %.loc35_3
// CHECK:STDOUT:   %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %a: ref %C = bind_name a, %a.var
// CHECK:STDOUT:   %a.ref: ref %C = name_ref a, %a
// CHECK:STDOUT:   %b.ref: %D = name_ref b, %b
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Result(constants.%Other, constants.%Self.6c1) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Result(constants.%C, constants.%AddWith.facet.37f) {}
// CHECK:STDOUT:
