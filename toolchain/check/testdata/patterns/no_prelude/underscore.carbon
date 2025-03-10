// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// This is mostly checking against crashes for compile time bindings in
// difficult contexts.
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/patterns/no_prelude/underscore.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/patterns/no_prelude/underscore.carbon

// --- basic.carbon

library "[[@TEST_NAME]]";

let _: {} = {};
var _: {} = {};

fn F() {
  let _: {} = {};
  var _: {} = {};
}

// --- function.carbon

library "[[@TEST_NAME]]";

fn F(_: {});

fn G(_: {}) {}

fn H() {
  F({});
}

// --- fail_class.carbon

library "[[@TEST_NAME]]";

class C {
  // CHECK:STDERR: fail_class.carbon:[[@LINE+4]]:7: error: semantics TODO: `_ used as field name` [SemanticsTodo]
  // CHECK:STDERR:   var _: ();
  // CHECK:STDERR:       ^~~~~
  // CHECK:STDERR:
  var _: ();
}

// --- fail_interface.carbon

library "[[@TEST_NAME]]";

interface I {
  // CHECK:STDERR: fail_interface.carbon:[[@LINE+4]]:7: error: semantics TODO: `_ used as associated constant name` [SemanticsTodo]
  // CHECK:STDERR:   let _:! {};
  // CHECK:STDERR:       ^~~~~~
  // CHECK:STDERR:
  let _:! {};
}

// --- fail_use.carbon

fn F() -> {} {
  let _: {} = {};
  // CHECK:STDERR: fail_use.carbon:[[@LINE+12]]:10: error: expected expression [ExpectedExpr]
  // CHECK:STDERR:   return _;
  // CHECK:STDERR:          ^
  // CHECK:STDERR:
  // CHECK:STDERR: fail_use.carbon:[[@LINE+8]]:10: error: `return` statements must end with a `;` [ExpectedStatementSemi]
  // CHECK:STDERR:   return _;
  // CHECK:STDERR:          ^
  // CHECK:STDERR:
  // CHECK:STDERR: fail_use.carbon:[[@LINE+4]]:10: error: semantics TODO: `handle invalid parse trees in `check`` [SemanticsTodo]
  // CHECK:STDERR:   return _;
  // CHECK:STDERR:          ^
  // CHECK:STDERR:
  return _;
}

// CHECK:STDOUT: --- basic.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %_.patt.loc4: %empty_struct_type = binding_pattern _
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc4_9.1: type = splice_block %.loc4_9.3 [concrete = constants.%empty_struct_type] {
// CHECK:STDOUT:     %.loc4_9.2: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:     %.loc4_9.3: type = converted %.loc4_9.2, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc4_14: %empty_struct_type = converted @__global_init.%.loc4, %empty_struct [concrete = constants.%empty_struct]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %_.patt.loc5: %empty_struct_type = binding_pattern _
// CHECK:STDOUT:     %.loc5_1: %empty_struct_type = var_pattern %_.patt.loc5
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %_.var: ref %empty_struct_type = var _
// CHECK:STDOUT:   %.loc5_9.1: type = splice_block %.loc5_9.3 [concrete = constants.%empty_struct_type] {
// CHECK:STDOUT:     %.loc5_9.2: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:     %.loc5_9.3: type = converted %.loc5_9.2, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %_.patt.loc8: %empty_struct_type = binding_pattern _
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc8_16.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %.loc8_11.1: type = splice_block %.loc8_11.3 [concrete = constants.%empty_struct_type] {
// CHECK:STDOUT:     %.loc8_11.2: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:     %.loc8_11.3: type = converted %.loc8_11.2, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc8_16.2: %empty_struct_type = converted %.loc8_16.1, %empty_struct [concrete = constants.%empty_struct]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %_.patt.loc9: %empty_struct_type = binding_pattern _
// CHECK:STDOUT:     %.loc9_3.1: %empty_struct_type = var_pattern %_.patt.loc9
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %_.var: ref %empty_struct_type = var _
// CHECK:STDOUT:   %.loc9_16.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %.loc9_16.2: init %empty_struct_type = struct_init () to %_.var [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc9_3.2: init %empty_struct_type = converted %.loc9_16.1, %.loc9_16.2 [concrete = constants.%empty_struct]
// CHECK:STDOUT:   assign %_.var, %.loc9_3.2
// CHECK:STDOUT:   %.loc9_11.1: type = splice_block %.loc9_11.3 [concrete = constants.%empty_struct_type] {
// CHECK:STDOUT:     %.loc9_11.2: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:     %.loc9_11.3: type = converted %.loc9_11.2, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc4: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %.loc5_14.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %.loc5_14.2: init %empty_struct_type = struct_init () to file.%_.var [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc5_1: init %empty_struct_type = converted %.loc5_14.1, %.loc5_14.2 [concrete = constants.%empty_struct]
// CHECK:STDOUT:   assign file.%_.var, %.loc5_1
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- function.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %H.type: type = fn_type @H [concrete]
// CHECK:STDOUT:   %H: %H.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:     .G = %G.decl
// CHECK:STDOUT:     .H = %H.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %_.patt: %empty_struct_type = binding_pattern _
// CHECK:STDOUT:     %_.param_patt: %empty_struct_type = value_param_pattern %_.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %_.param: %empty_struct_type = value_param call_param0
// CHECK:STDOUT:     %.loc4_10.1: type = splice_block %.loc4_10.3 [concrete = constants.%empty_struct_type] {
// CHECK:STDOUT:       %.loc4_10.2: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:       %.loc4_10.3: type = converted %.loc4_10.2, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {
// CHECK:STDOUT:     %_.patt: %empty_struct_type = binding_pattern _
// CHECK:STDOUT:     %_.param_patt: %empty_struct_type = value_param_pattern %_.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %_.param: %empty_struct_type = value_param call_param0
// CHECK:STDOUT:     %.loc6_10.1: type = splice_block %.loc6_10.3 [concrete = constants.%empty_struct_type] {
// CHECK:STDOUT:       %.loc6_10.2: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:       %.loc6_10.3: type = converted %.loc6_10.2, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %H.decl: %H.type = fn_decl @H [concrete = constants.%H] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%_.param_patt: %empty_struct_type);
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G(%_.param_patt: %empty_struct_type) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @H() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:   %.loc9_6.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc9_6.2: %empty_struct_type = converted %.loc9_6.1, %empty_struct [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %F.call: init %empty_tuple.type = call %F.ref(%.loc9_6.2)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_class.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %C.elem: type = unbound_element_type %C, %empty_tuple.type [concrete]
// CHECK:STDOUT:   %struct_type._: type = struct_type {._: %empty_tuple.type} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type._ [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %.loc9_8: %C.elem = field_decl _, element0 [concrete]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %.loc9_3: %C.elem = var_pattern %.loc9_8
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.var: ref %C.elem = var <none>
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type._ [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT:   ._ = %.loc9_8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_interface.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [concrete]
// CHECK:STDOUT:   %Self: %I.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %I.assoc_type: type = assoc_entity_type %I.type [concrete]
// CHECK:STDOUT:   %assoc0: %I.assoc_type = assoc_entity element0, @I.%_ [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .I = %I.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %I.decl: type = interface_decl @I [concrete = constants.%I.type] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I {
// CHECK:STDOUT:   %Self: %I.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   %_: %empty_struct_type = assoc_const_decl @_ [concrete] {
// CHECK:STDOUT:     %assoc0: %I.assoc_type = assoc_entity element0, @I.%_ [concrete = constants.%assoc0]
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   ._ = @_.%assoc0
// CHECK:STDOUT:   witness = (%_)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic assoc_const @_(@I.%Self: %I.type) {
// CHECK:STDOUT:   assoc_const _:! %empty_struct_type;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @_(constants.%Self) {}
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_use.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {}
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() -> %empty_struct_type;
// CHECK:STDOUT:
