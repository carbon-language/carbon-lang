// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/bool.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/typeof/basics.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/typeof/basics.carbon

// --- basic.carbon

library "[[@TEST_NAME]]";

class C {}

var c: C = {};

//@dump-sem-ir-begin
var d: typeof(c) = {};
//@dump-sem-ir-end

// --- type_of_type.carbon

library "[[@TEST_NAME]]";

class C {}

//@dump-sem-ir-begin
let T: type = typeof(C);
//@dump-sem-ir-end

// --- not_evaluated.carbon

library "[[@TEST_NAME]]";

class C {}

fn Make() -> C;

fn F() {
  //@dump-sem-ir-begin
  var unused c: typeof(Make()) = {};
  //@dump-sem-ir-end
}

// --- control_flow.carbon

library "[[@TEST_NAME]]";

fn F(a: bool, b: bool) {
  //@dump-sem-ir-begin
  let unused c: typeof(a and b) = true;
  //@dump-sem-ir-end
}

// --- fail_todo_control_flow_file_scope.carbon

library "[[@TEST_NAME]]";

let a: bool = true;
// CHECK:STDERR: fail_todo_control_flow_file_scope.carbon:[[@LINE+2]]:15-20: error: semantics TODO: `Control flow expressions are currently only supported inside functions.` [SemanticsTodo]
// CHECK:STDERR:
let b: typeof(a or a) = false;

// --- member_access.carbon

library "[[@TEST_NAME]]";

class C {
  fn G();
}

fn F(c: C) {
  //@dump-sem-ir-begin
  typeof(c).G();
  //@dump-sem-ir-end
}

// --- generic.carbon

library "[[@TEST_NAME]]";

//@dump-sem-ir-begin
fn G[T: type](x: T, unused y: typeof(x)) {}
//@dump-sem-ir-end

fn Call() {
  G(true, false);
}

// --- fail_generic_mismatch.carbon

library "[[@TEST_NAME]]";

class C {}

fn G[T: type](x: T, unused y: typeof(x)) {}

fn Call() {
  // CHECK:STDERR: fail_generic_mismatch.carbon:[[@LINE+3]]:3-18: error: inconsistent deductions for value of generic parameter `T` [DeductionInconsistent]
  // CHECK:STDERR: fail_generic_mismatch.carbon:[[@LINE-4]]:1-42: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR:
  G(true, {} as C);
}

// --- fail_undeclared.carbon

library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_undeclared.carbon:[[@LINE+2]]:15-24: error: name `undeclared` not found [NameNotFound]
// CHECK:STDERR:
var x: typeof(undeclared);

// CHECK:STDOUT: --- basic.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %C.val: %C = struct_value () [concrete]
// CHECK:STDOUT:   %d.patt: %pattern_type = ref_binding_pattern d [concrete]
// CHECK:STDOUT:   %d.var_patt: %pattern_type = var_pattern %d.patt [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %d.var: ref %C = var_storage %d.var_patt [concrete]
// CHECK:STDOUT:   %.loc9: type = type_of %c.ref in [concrete = constants.%C] {
// CHECK:STDOUT:     %c.ref: ref %C = name_ref c, %c [concrete = %c.var]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %d: ref %C = wrapper_binding d, %d.var [concrete = %d.var]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %d.patt: %pattern_type = ref_binding_pattern d [concrete = constants.%d.patt]
// CHECK:STDOUT:     %d.var_patt: %pattern_type = var_pattern %d.patt [concrete = constants.%d.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %.loc9_21.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc9_21.2: init %C to file.%d.var = class_init () [concrete = constants.%C.val]
// CHECK:STDOUT:   %.loc9_1: init %C = converted %.loc9_21.1, %.loc9_21.2 [concrete = constants.%C.val]
// CHECK:STDOUT:   assign file.%d.var, %.loc9_1
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- type_of_type.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %pattern_type: type = pattern_type type [concrete]
// CHECK:STDOUT:   %T.patt: %pattern_type = value_binding_pattern T [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %.loc7: type = type_literal type [concrete = type]
// CHECK:STDOUT:   %T: type = wrapper_binding T, @__global_init.%.loc7
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %T.patt: %pattern_type = value_binding_pattern T [concrete = constants.%T.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc7: type = type_of %C.ref in [concrete = type] {
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- not_evaluated.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %pattern_type.98b: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %Make.type: type = fn_type @Make [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Make: %Make.type = struct_value () [concrete]
// CHECK:STDOUT:   %c.patt: %pattern_type.98b = ref_binding_pattern c [concrete]
// CHECK:STDOUT:   %c.var_patt: %pattern_type.98b = var_pattern %c.patt [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %C.val: %C = struct_value () [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %facet_type: type = facet_type <@DestroyImpl & @SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %T.09f: %SubobjectDestroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.type.dea: type = fn_type @T.as_type.as.DestroyImpl.impl.Op, @T.as_type.as.DestroyImpl.impl(%T.09f) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.6e5: %T.as_type.as.DestroyImpl.impl.Op.type.dea = struct_value () [symbolic]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.2: type = fn_type @SubobjectDestroy.WithSelf.Op.loc10_3.2 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.2: %SubobjectDestroy.WithSelf.Op.type.6870fd.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.2: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.2), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.978: %SubobjectDestroy.type = facet_value %C, (%custom_witness.b80ef6.2) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.1e5: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.978) [concrete]
// CHECK:STDOUT:   %facet_value: %facet_type = facet_value %C, (%DestroyImpl.impl_witness.1e5, %custom_witness.b80ef6.2) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.74d: @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op.type (%T.as_type.as.DestroyImpl.impl.Op.type.dea) = import_ref Core//prelude/parts/destroy, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op (constants.%T.as_type.as.DestroyImpl.impl.Op.6e5)]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table = impl_witness_table (%Core.import_ref.74d), @T.as_type.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %c.var: ref %C = var_storage %c.var_patt
// CHECK:STDOUT:   %.loc10_35.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc10_35.2: init %C to %c.var = class_init () [concrete = constants.%C.val]
// CHECK:STDOUT:   %.loc10_3.1: init %C = converted %.loc10_35.1, %.loc10_35.2 [concrete = constants.%C.val]
// CHECK:STDOUT:   assign %c.var, %.loc10_3.1
// CHECK:STDOUT:   %.loc10_30: type = type_of %Make.call in [concrete = constants.%C] {
// CHECK:STDOUT:     %Make.ref: %Make.type = name_ref Make, file.%Make.decl [concrete = constants.%Make]
// CHECK:STDOUT:     %.loc10_29: ref %C = temporary_storage
// CHECK:STDOUT:     %Make.call: init %C to %.loc10_29 = call %Make.ref()
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c: ref %C = wrapper_binding c, %c.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c.patt: %pattern_type.98b = ref_binding_pattern c [concrete = constants.%c.patt]
// CHECK:STDOUT:     %c.var_patt: %pattern_type.98b = var_pattern %c.patt [concrete = constants.%c.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %facet_value.loc10_3.1: %facet_type = facet_value constants.%C, (constants.%DestroyImpl.impl_witness.1e5, constants.%custom_witness.b80ef6.2) [concrete = constants.%facet_value]
// CHECK:STDOUT:   %.loc10_3.2: %facet_type = converted constants.%C, %facet_value.loc10_3.1 [concrete = constants.%facet_value]
// CHECK:STDOUT:   %facet_value.loc10_3.2: %facet_type = facet_value constants.%C, (constants.%DestroyImpl.impl_witness.1e5, constants.%custom_witness.b80ef6.2) [concrete = constants.%facet_value]
// CHECK:STDOUT:   %.loc10_3.3: %facet_type = converted constants.%C, %facet_value.loc10_3.2 [concrete = constants.%facet_value]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%c.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc10_3.1(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc10_3.2(%self.param: ref %C) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: --- control_flow.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %pattern_type: type = pattern_type bool [concrete]
// CHECK:STDOUT:   %false: bool = bool_literal false [concrete]
// CHECK:STDOUT:   %c.patt: %pattern_type = value_binding_pattern c [concrete]
// CHECK:STDOUT:   %true: bool = bool_literal true [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%a.param: bool, %b.param: bool) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %true: bool = bool_literal true [concrete = constants.%true]
// CHECK:STDOUT:   %.loc6_31: type = type_of %.loc6_26 in [concrete = bool] {
// CHECK:STDOUT:     %a.ref: bool = name_ref a, %a
// CHECK:STDOUT:     %false: bool = bool_literal false [concrete = constants.%false]
// CHECK:STDOUT:     if %a.ref br !and.rhs else br !.loc6(%false)
// CHECK:STDOUT:
// CHECK:STDOUT:   !and.rhs:
// CHECK:STDOUT:     %b.ref: bool = name_ref b, %b
// CHECK:STDOUT:     br !.loc6(%b.ref)
// CHECK:STDOUT:
// CHECK:STDOUT:   !.loc6:
// CHECK:STDOUT:     %.loc6_26: bool = block_arg !.loc6
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c: bool = wrapper_binding c, %true
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c.patt: %pattern_type = value_binding_pattern c [concrete = constants.%c.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- member_access.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %C.G.type: type = fn_type @C.G [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %C.G: %C.G.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%c.param: %C) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc10: type = type_of %c.ref in [concrete = constants.%C] {
// CHECK:STDOUT:     %c.ref: %C = name_ref c, %c
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.ref: %C.G.type = name_ref G, @C.%C.G.decl [concrete = constants.%C.G]
// CHECK:STDOUT:   %C.G.call: init %empty_tuple.type = call %G.ref()
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- generic.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self.frozen: type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type.9a5: type = pattern_type type [concrete]
// CHECK:STDOUT:   %T.patt: %pattern_type.9a5 = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.51d: type = pattern_type %T [symbolic]
// CHECK:STDOUT:   %x.param_patt.91d: %pattern_type.51d = value_param_pattern [symbolic]
// CHECK:STDOUT:   %x.patt.260: %pattern_type.51d = wrapper_binding_pattern x, %x.param_patt.91d [symbolic]
// CHECK:STDOUT:   %y.param_patt.be3: %pattern_type.51d = value_param_pattern [symbolic]
// CHECK:STDOUT:   %y.patt.678: %pattern_type.51d = wrapper_binding_pattern y, %y.param_patt.be3 [symbolic]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T [symbolic]
// CHECK:STDOUT:   %pattern_type.831: type = pattern_type bool [concrete]
// CHECK:STDOUT:   %x.param_patt.37d: %pattern_type.831 = value_param_pattern [concrete]
// CHECK:STDOUT:   %x.patt.e47: %pattern_type.831 = wrapper_binding_pattern x, %x.param_patt.37d [concrete]
// CHECK:STDOUT:   %y.param_patt.077: %pattern_type.831 = value_param_pattern [concrete]
// CHECK:STDOUT:   %y.patt.252: %pattern_type.831 = wrapper_binding_pattern y, %y.param_patt.077 [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness bool [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {
// CHECK:STDOUT:     %T.patt.loc5_7.1: %pattern_type.9a5 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc5_7.2 (constants.%T.patt)]
// CHECK:STDOUT:     %x.param_patt.loc5_16.1: @G.%pattern_type (%pattern_type.51d) = value_param_pattern [symbolic = %x.param_patt.loc5_16.2 (constants.%x.param_patt.91d)]
// CHECK:STDOUT:     %x.patt.loc5_16.1: @G.%pattern_type (%pattern_type.51d) = wrapper_binding_pattern x, %x.param_patt.loc5_16.1 [symbolic = %x.patt.loc5_16.2 (constants.%x.patt.260)]
// CHECK:STDOUT:     %y.param_patt.loc5_29.1: @G.%pattern_type (%pattern_type.51d) = value_param_pattern [symbolic = %y.param_patt.loc5_29.2 (constants.%y.param_patt.be3)]
// CHECK:STDOUT:     %y.patt.loc5_29.1: @G.%pattern_type (%pattern_type.51d) = wrapper_binding_pattern y, %y.param_patt.loc5_29.1 [symbolic = %y.patt.loc5_29.2 (constants.%y.patt.678)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc5_9.1: type = splice_block %.loc5_9.2 [concrete = type] {
// CHECK:STDOUT:       %.Self.frozen: type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %.loc5_9.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc5_7.2: type = symbolic_binding T, 0 [symbolic = %T.loc5_7.1 (constants.%T)]
// CHECK:STDOUT:     %x.param: @G.%T.loc5_7.1 (%T) = value_param call_param0
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc5_7.2 [symbolic = %T.loc5_7.1 (constants.%T)]
// CHECK:STDOUT:     %x: @G.%T.loc5_7.1 (%T) = wrapper_binding x, %x.param
// CHECK:STDOUT:     %y.param: @G.%T.loc5_7.1 (%T) = value_param call_param1
// CHECK:STDOUT:     %.loc5_39: type = type_of %x.ref in [symbolic = %T.loc5_7.1 (constants.%T)] {
// CHECK:STDOUT:       %x.ref: @G.%T.loc5_7.1 (%T) = name_ref x, %x
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %y: @G.%T.loc5_7.1 (%T) = wrapper_binding y, %y.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @G(%T.loc5_7.2: type) {
// CHECK:STDOUT:   %T.patt.loc5_7.2: %pattern_type.9a5 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc5_7.2 (constants.%T.patt)]
// CHECK:STDOUT:   %T.loc5_7.1: type = symbolic_binding T, 0 [symbolic = %T.loc5_7.1 (constants.%T)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %T.loc5_7.1 [symbolic = %pattern_type (constants.%pattern_type.51d)]
// CHECK:STDOUT:   %x.param_patt.loc5_16.2: @G.%pattern_type (%pattern_type.51d) = value_param_pattern [symbolic = %x.param_patt.loc5_16.2 (constants.%x.param_patt.91d)]
// CHECK:STDOUT:   %x.patt.loc5_16.2: @G.%pattern_type (%pattern_type.51d) = wrapper_binding_pattern x, %x.param_patt.loc5_16.2 [symbolic = %x.patt.loc5_16.2 (constants.%x.patt.260)]
// CHECK:STDOUT:   %y.param_patt.loc5_29.2: @G.%pattern_type (%pattern_type.51d) = value_param_pattern [symbolic = %y.param_patt.loc5_29.2 (constants.%y.param_patt.be3)]
// CHECK:STDOUT:   %y.patt.loc5_29.2: @G.%pattern_type (%pattern_type.51d) = wrapper_binding_pattern y, %y.param_patt.loc5_29.2 [symbolic = %y.patt.loc5_29.2 (constants.%y.patt.678)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T.loc5_7.1 [symbolic = %require_complete (constants.%require_complete)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: @G.%T.loc5_7.1 (%T), %y.param: @G.%T.loc5_7.1 (%T)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @G(constants.%T) {
// CHECK:STDOUT:   %T.patt.loc5_7.2 => constants.%T.patt
// CHECK:STDOUT:   %T.loc5_7.1 => constants.%T
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.51d
// CHECK:STDOUT:   %x.param_patt.loc5_16.2 => constants.%x.param_patt.91d
// CHECK:STDOUT:   %x.patt.loc5_16.2 => constants.%x.patt.260
// CHECK:STDOUT:   %y.param_patt.loc5_29.2 => constants.%y.param_patt.be3
// CHECK:STDOUT:   %y.patt.loc5_29.2 => constants.%y.patt.678
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @G(bool) {
// CHECK:STDOUT:   %T.patt.loc5_7.2 => constants.%T.patt
// CHECK:STDOUT:   %T.loc5_7.1 => bool
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.831
// CHECK:STDOUT:   %x.param_patt.loc5_16.2 => constants.%x.param_patt.37d
// CHECK:STDOUT:   %x.patt.loc5_16.2 => constants.%x.patt.e47
// CHECK:STDOUT:   %y.param_patt.loc5_29.2 => constants.%y.param_patt.077
// CHECK:STDOUT:   %y.patt.loc5_29.2 => constants.%y.patt.252
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
