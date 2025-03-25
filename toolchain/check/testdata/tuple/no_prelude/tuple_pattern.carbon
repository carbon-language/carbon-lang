// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/tuple/no_prelude/tuple_pattern.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/tuple/no_prelude/tuple_pattern.carbon


// --- basic.carbon

library "[[@TEST_NAME]]";

fn F() {
  let (x: {}, y: {}) = ({}, {});
  var (a: {}, b: {});
  var (c: {}, d: {}) = ({}, {});
}

// --- variable.carbon

library "[[@TEST_NAME]]";

fn F() {
  var tuple: ({}, {}) = ({}, {});
  var (x: {}, y: {}) = tuple;
}

fn G() {
  let tuple: ({}, {}) = ({}, {});
  var (x: {}, y: {}) = tuple;
}

fn MakeTuple() -> ({}, {});

fn H() {
  var (x: {}, y: {}) = MakeTuple();
}

// --- nested.carbon

library "[[@TEST_NAME]]";

fn F() {
  let (x: {}, (y: {}, z: {})) = ({}, ({}, {}));
}

// --- package_scope.carbon

library "[[@TEST_NAME]]";

let (x: {}, y: {}) = ({}, {});

// --- fail_in_interface.carbon

library "[[@TEST_NAME]]";

interface I {
  // CHECK:STDERR: fail_in_interface.carbon:[[@LINE+4]]:7: error: pattern in associated constant declaration must be a single `:!` binding [ExpectedSymbolicBindingInAssociatedConstant]
  // CHECK:STDERR:   let (x: {}, y: {});
  // CHECK:STDERR:       ^~~~~~~~~~~~~~
  // CHECK:STDERR:
  let (x: {}, y: {});
}

// --- fail_in_class.carbon

library "[[@TEST_NAME]]";

class C {
  // CHECK:STDERR: fail_in_class.carbon:[[@LINE+4]]:7: error: pattern in field declaration is not a single `:` binding [ExpectedSymbolicBindingInFieldDecl]
  // CHECK:STDERR:   var (x: {}, y: {});
  // CHECK:STDERR:       ^~~~~~~~~~~~~~
  // CHECK:STDERR:
  var (x: {}, y: {});
}

// --- fail_initializer_mismatch.carbon

library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_initializer_mismatch.carbon:[[@LINE+4]]:5: error: tuple pattern expects 2 elements, but tuple literal has 1 [TuplePatternSizeDoesntMatchLiteral]
// CHECK:STDERR: let (x: {}, y: {}) = ({},);
// CHECK:STDERR:     ^~~~~~~~~~~~~~
// CHECK:STDERR:
let (x: {}, y: {}) = ({},);

// CHECK:STDERR: fail_initializer_mismatch.carbon:[[@LINE+4]]:5: error: tuple pattern expects 2 elements, but tuple literal has 3 [TuplePatternSizeDoesntMatchLiteral]
// CHECK:STDERR: let (a: {}, b: {}) = ({}, {}, {});
// CHECK:STDERR:     ^~~~~~~~~~~~~~
// CHECK:STDERR:
let (a: {}, b: {}) = ({}, {}, {});

// CHECK:STDOUT: --- basic.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %tuple.type.b6b: type = tuple_type (%empty_struct_type, %empty_struct_type) [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %tuple: %tuple.type.b6b = tuple_value (%empty_struct, %empty_struct) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %empty_struct_type = binding_pattern x
// CHECK:STDOUT:     %y.patt: %empty_struct_type = binding_pattern y
// CHECK:STDOUT:     %.loc5_20: %tuple.type.b6b = tuple_pattern (%x.patt, %y.patt)
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc5_26.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %.loc5_30.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %.loc5_31: %tuple.type.b6b = tuple_literal (%.loc5_26.1, %.loc5_30.1)
// CHECK:STDOUT:   %.loc5_12.1: type = splice_block %.loc5_12.3 [concrete = constants.%empty_struct_type] {
// CHECK:STDOUT:     %.loc5_12.2: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:     %.loc5_12.3: type = converted %.loc5_12.2, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %empty_struct.loc5_26: %empty_struct_type = struct_value () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc5_26.2: %empty_struct_type = converted %.loc5_26.1, %empty_struct.loc5_26 [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %x: %empty_struct_type = bind_name x, %.loc5_26.2
// CHECK:STDOUT:   %.loc5_19.1: type = splice_block %.loc5_19.3 [concrete = constants.%empty_struct_type] {
// CHECK:STDOUT:     %.loc5_19.2: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:     %.loc5_19.3: type = converted %.loc5_19.2, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %empty_struct.loc5_30: %empty_struct_type = struct_value () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc5_30.2: %empty_struct_type = converted %.loc5_30.1, %empty_struct.loc5_30 [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %y: %empty_struct_type = bind_name y, %.loc5_30.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %empty_struct_type = binding_pattern a
// CHECK:STDOUT:     %b.patt: %empty_struct_type = binding_pattern b
// CHECK:STDOUT:     %.loc6_20: %tuple.type.b6b = tuple_pattern (%a.patt, %b.patt)
// CHECK:STDOUT:     %.loc6_3: %tuple.type.b6b = var_pattern %.loc6_20
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.var.loc6: ref %tuple.type.b6b = var <none>
// CHECK:STDOUT:   %tuple.elem0.loc6: ref %empty_struct_type = tuple_access %.var.loc6, element0
// CHECK:STDOUT:   %tuple.elem1.loc6: ref %empty_struct_type = tuple_access %.var.loc6, element1
// CHECK:STDOUT:   %.loc6_12.1: type = splice_block %.loc6_12.3 [concrete = constants.%empty_struct_type] {
// CHECK:STDOUT:     %.loc6_12.2: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:     %.loc6_12.3: type = converted %.loc6_12.2, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: ref %empty_struct_type = bind_name a, %tuple.elem0.loc6
// CHECK:STDOUT:   %.loc6_19.1: type = splice_block %.loc6_19.3 [concrete = constants.%empty_struct_type] {
// CHECK:STDOUT:     %.loc6_19.2: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:     %.loc6_19.3: type = converted %.loc6_19.2, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b: ref %empty_struct_type = bind_name b, %tuple.elem1.loc6
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c.patt: %empty_struct_type = binding_pattern c
// CHECK:STDOUT:     %d.patt: %empty_struct_type = binding_pattern d
// CHECK:STDOUT:     %.loc7_20: %tuple.type.b6b = tuple_pattern (%c.patt, %d.patt)
// CHECK:STDOUT:     %.loc7_3.1: %tuple.type.b6b = var_pattern %.loc7_20
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.var.loc7: ref %tuple.type.b6b = var <none>
// CHECK:STDOUT:   %.loc7_26.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %.loc7_30.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %.loc7_31.1: %tuple.type.b6b = tuple_literal (%.loc7_26.1, %.loc7_30.1)
// CHECK:STDOUT:   %tuple.elem0.loc7_31: ref %empty_struct_type = tuple_access %.var.loc7, element0
// CHECK:STDOUT:   %.loc7_26.2: init %empty_struct_type = struct_init () to %tuple.elem0.loc7_31 [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc7_31.2: init %empty_struct_type = converted %.loc7_26.1, %.loc7_26.2 [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %tuple.elem1.loc7_31: ref %empty_struct_type = tuple_access %.var.loc7, element1
// CHECK:STDOUT:   %.loc7_30.2: init %empty_struct_type = struct_init () to %tuple.elem1.loc7_31 [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc7_31.3: init %empty_struct_type = converted %.loc7_30.1, %.loc7_30.2 [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc7_31.4: init %tuple.type.b6b = tuple_init (%.loc7_31.2, %.loc7_31.3) to %.var.loc7 [concrete = constants.%tuple]
// CHECK:STDOUT:   %.loc7_3.2: init %tuple.type.b6b = converted %.loc7_31.1, %.loc7_31.4 [concrete = constants.%tuple]
// CHECK:STDOUT:   assign %.var.loc7, %.loc7_3.2
// CHECK:STDOUT:   %tuple.elem0.loc7_3: ref %empty_struct_type = tuple_access %.var.loc7, element0
// CHECK:STDOUT:   %tuple.elem1.loc7_3: ref %empty_struct_type = tuple_access %.var.loc7, element1
// CHECK:STDOUT:   %.loc7_12.1: type = splice_block %.loc7_12.3 [concrete = constants.%empty_struct_type] {
// CHECK:STDOUT:     %.loc7_12.2: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:     %.loc7_12.3: type = converted %.loc7_12.2, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c: ref %empty_struct_type = bind_name c, %tuple.elem0.loc7_3
// CHECK:STDOUT:   %.loc7_19.1: type = splice_block %.loc7_19.3 [concrete = constants.%empty_struct_type] {
// CHECK:STDOUT:     %.loc7_19.2: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:     %.loc7_19.3: type = converted %.loc7_19.2, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %d: ref %empty_struct_type = bind_name d, %tuple.elem1.loc7_3
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- variable.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %tuple.type.b6b: type = tuple_type (%empty_struct_type, %empty_struct_type) [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %tuple: %tuple.type.b6b = tuple_value (%empty_struct, %empty_struct) [concrete]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %MakeTuple.type: type = fn_type @MakeTuple [concrete]
// CHECK:STDOUT:   %MakeTuple: %MakeTuple.type = struct_value () [concrete]
// CHECK:STDOUT:   %H.type: type = fn_type @H [concrete]
// CHECK:STDOUT:   %H: %H.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:     .G = %G.decl
// CHECK:STDOUT:     .MakeTuple = %MakeTuple.decl
// CHECK:STDOUT:     .H = %H.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {} {}
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {} {}
// CHECK:STDOUT:   %MakeTuple.decl: %MakeTuple.type = fn_decl @MakeTuple [concrete = constants.%MakeTuple] {
// CHECK:STDOUT:     %return.patt: %tuple.type.b6b = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %tuple.type.b6b = out_param_pattern %return.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc14_21: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:     %.loc14_25: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:     %.loc14_26.1: %tuple.type.b6b = tuple_literal (%.loc14_21, %.loc14_25)
// CHECK:STDOUT:     %.loc14_26.2: type = converted %.loc14_21, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     %.loc14_26.3: type = converted %.loc14_25, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     %.loc14_26.4: type = converted %.loc14_26.1, constants.%tuple.type.b6b [concrete = constants.%tuple.type.b6b]
// CHECK:STDOUT:     %return.param: ref %tuple.type.b6b = out_param call_param0
// CHECK:STDOUT:     %return: ref %tuple.type.b6b = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %H.decl: %H.type = fn_decl @H [concrete = constants.%H] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %tuple.patt: %tuple.type.b6b = binding_pattern tuple
// CHECK:STDOUT:     %.loc5_3.1: %tuple.type.b6b = var_pattern %tuple.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %tuple.var: ref %tuple.type.b6b = var tuple
// CHECK:STDOUT:   %.loc5_27.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %.loc5_31.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %.loc5_32.1: %tuple.type.b6b = tuple_literal (%.loc5_27.1, %.loc5_31.1)
// CHECK:STDOUT:   %tuple.elem0.loc5: ref %empty_struct_type = tuple_access %tuple.var, element0
// CHECK:STDOUT:   %.loc5_27.2: init %empty_struct_type = struct_init () to %tuple.elem0.loc5 [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc5_32.2: init %empty_struct_type = converted %.loc5_27.1, %.loc5_27.2 [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %tuple.elem1.loc5: ref %empty_struct_type = tuple_access %tuple.var, element1
// CHECK:STDOUT:   %.loc5_31.2: init %empty_struct_type = struct_init () to %tuple.elem1.loc5 [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc5_32.3: init %empty_struct_type = converted %.loc5_31.1, %.loc5_31.2 [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc5_32.4: init %tuple.type.b6b = tuple_init (%.loc5_32.2, %.loc5_32.3) to %tuple.var [concrete = constants.%tuple]
// CHECK:STDOUT:   %.loc5_3.2: init %tuple.type.b6b = converted %.loc5_32.1, %.loc5_32.4 [concrete = constants.%tuple]
// CHECK:STDOUT:   assign %tuple.var, %.loc5_3.2
// CHECK:STDOUT:   %.loc5_21.1: type = splice_block %.loc5_21.5 [concrete = constants.%tuple.type.b6b] {
// CHECK:STDOUT:     %.loc5_16: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:     %.loc5_20: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:     %.loc5_21.2: %tuple.type.b6b = tuple_literal (%.loc5_16, %.loc5_20)
// CHECK:STDOUT:     %.loc5_21.3: type = converted %.loc5_16, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     %.loc5_21.4: type = converted %.loc5_20, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     %.loc5_21.5: type = converted %.loc5_21.2, constants.%tuple.type.b6b [concrete = constants.%tuple.type.b6b]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %tuple: ref %tuple.type.b6b = bind_name tuple, %tuple.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %empty_struct_type = binding_pattern x
// CHECK:STDOUT:     %y.patt: %empty_struct_type = binding_pattern y
// CHECK:STDOUT:     %.loc6_20: %tuple.type.b6b = tuple_pattern (%x.patt, %y.patt)
// CHECK:STDOUT:     %.loc6_3.1: %tuple.type.b6b = var_pattern %.loc6_20
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.var: ref %tuple.type.b6b = var <none>
// CHECK:STDOUT:   %tuple.ref: ref %tuple.type.b6b = name_ref tuple, %tuple
// CHECK:STDOUT:   %tuple.elem0.loc6_24.1: ref %empty_struct_type = tuple_access %tuple.ref, element0
// CHECK:STDOUT:   %tuple.elem0.loc6_24.2: ref %empty_struct_type = tuple_access %.var, element0
// CHECK:STDOUT:   %.loc6_24.1: init %empty_struct_type = struct_init () to %tuple.elem0.loc6_24.2 [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc6_24.2: init %empty_struct_type = converted %tuple.elem0.loc6_24.1, %.loc6_24.1 [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %tuple.elem1.loc6_24.1: ref %empty_struct_type = tuple_access %tuple.ref, element1
// CHECK:STDOUT:   %tuple.elem1.loc6_24.2: ref %empty_struct_type = tuple_access %.var, element1
// CHECK:STDOUT:   %.loc6_24.3: init %empty_struct_type = struct_init () to %tuple.elem1.loc6_24.2 [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc6_24.4: init %empty_struct_type = converted %tuple.elem1.loc6_24.1, %.loc6_24.3 [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc6_24.5: init %tuple.type.b6b = tuple_init (%.loc6_24.2, %.loc6_24.4) to %.var [concrete = constants.%tuple]
// CHECK:STDOUT:   %.loc6_3.2: init %tuple.type.b6b = converted %tuple.ref, %.loc6_24.5 [concrete = constants.%tuple]
// CHECK:STDOUT:   assign %.var, %.loc6_3.2
// CHECK:STDOUT:   %tuple.elem0.loc6_3: ref %empty_struct_type = tuple_access %.var, element0
// CHECK:STDOUT:   %tuple.elem1.loc6_3: ref %empty_struct_type = tuple_access %.var, element1
// CHECK:STDOUT:   %.loc6_12.1: type = splice_block %.loc6_12.3 [concrete = constants.%empty_struct_type] {
// CHECK:STDOUT:     %.loc6_12.2: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:     %.loc6_12.3: type = converted %.loc6_12.2, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %x: ref %empty_struct_type = bind_name x, %tuple.elem0.loc6_3
// CHECK:STDOUT:   %.loc6_19.1: type = splice_block %.loc6_19.3 [concrete = constants.%empty_struct_type] {
// CHECK:STDOUT:     %.loc6_19.2: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:     %.loc6_19.3: type = converted %.loc6_19.2, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %y: ref %empty_struct_type = bind_name y, %tuple.elem1.loc6_3
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %tuple.patt: %tuple.type.b6b = binding_pattern tuple
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc10_27: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %.loc10_31: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %.loc10_32.1: %tuple.type.b6b = tuple_literal (%.loc10_27, %.loc10_31)
// CHECK:STDOUT:   %.loc10_21.1: type = splice_block %.loc10_21.5 [concrete = constants.%tuple.type.b6b] {
// CHECK:STDOUT:     %.loc10_16: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:     %.loc10_20: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:     %.loc10_21.2: %tuple.type.b6b = tuple_literal (%.loc10_16, %.loc10_20)
// CHECK:STDOUT:     %.loc10_21.3: type = converted %.loc10_16, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     %.loc10_21.4: type = converted %.loc10_20, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     %.loc10_21.5: type = converted %.loc10_21.2, constants.%tuple.type.b6b [concrete = constants.%tuple.type.b6b]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %empty_struct.loc10_27: %empty_struct_type = struct_value () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc10_32.2: %empty_struct_type = converted %.loc10_27, %empty_struct.loc10_27 [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %empty_struct.loc10_31: %empty_struct_type = struct_value () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc10_32.3: %empty_struct_type = converted %.loc10_31, %empty_struct.loc10_31 [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %tuple.loc10_32: %tuple.type.b6b = tuple_value (%.loc10_32.2, %.loc10_32.3) [concrete = constants.%tuple]
// CHECK:STDOUT:   %.loc10_32.4: %tuple.type.b6b = converted %.loc10_32.1, %tuple.loc10_32 [concrete = constants.%tuple]
// CHECK:STDOUT:   %tuple.loc10_7: %tuple.type.b6b = bind_name tuple, %.loc10_32.4
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %empty_struct_type = binding_pattern x
// CHECK:STDOUT:     %y.patt: %empty_struct_type = binding_pattern y
// CHECK:STDOUT:     %.loc11_20: %tuple.type.b6b = tuple_pattern (%x.patt, %y.patt)
// CHECK:STDOUT:     %.loc11_3.1: %tuple.type.b6b = var_pattern %.loc11_20
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.var: ref %tuple.type.b6b = var <none>
// CHECK:STDOUT:   %tuple.ref: %tuple.type.b6b = name_ref tuple, %tuple.loc10_7
// CHECK:STDOUT:   %tuple.elem0.loc11_24.1: %empty_struct_type = tuple_access %tuple.ref, element0
// CHECK:STDOUT:   %tuple.elem0.loc11_24.2: ref %empty_struct_type = tuple_access %.var, element0
// CHECK:STDOUT:   %.loc11_24.1: init %empty_struct_type = struct_init () to %tuple.elem0.loc11_24.2 [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc11_24.2: init %empty_struct_type = converted %tuple.elem0.loc11_24.1, %.loc11_24.1 [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %tuple.elem1.loc11_24.1: %empty_struct_type = tuple_access %tuple.ref, element1
// CHECK:STDOUT:   %tuple.elem1.loc11_24.2: ref %empty_struct_type = tuple_access %.var, element1
// CHECK:STDOUT:   %.loc11_24.3: init %empty_struct_type = struct_init () to %tuple.elem1.loc11_24.2 [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc11_24.4: init %empty_struct_type = converted %tuple.elem1.loc11_24.1, %.loc11_24.3 [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc11_24.5: init %tuple.type.b6b = tuple_init (%.loc11_24.2, %.loc11_24.4) to %.var [concrete = constants.%tuple]
// CHECK:STDOUT:   %.loc11_3.2: init %tuple.type.b6b = converted %tuple.ref, %.loc11_24.5 [concrete = constants.%tuple]
// CHECK:STDOUT:   assign %.var, %.loc11_3.2
// CHECK:STDOUT:   %tuple.elem0.loc11_3: ref %empty_struct_type = tuple_access %.var, element0
// CHECK:STDOUT:   %tuple.elem1.loc11_3: ref %empty_struct_type = tuple_access %.var, element1
// CHECK:STDOUT:   %.loc11_12.1: type = splice_block %.loc11_12.3 [concrete = constants.%empty_struct_type] {
// CHECK:STDOUT:     %.loc11_12.2: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:     %.loc11_12.3: type = converted %.loc11_12.2, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %x: ref %empty_struct_type = bind_name x, %tuple.elem0.loc11_3
// CHECK:STDOUT:   %.loc11_19.1: type = splice_block %.loc11_19.3 [concrete = constants.%empty_struct_type] {
// CHECK:STDOUT:     %.loc11_19.2: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:     %.loc11_19.3: type = converted %.loc11_19.2, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %y: ref %empty_struct_type = bind_name y, %tuple.elem1.loc11_3
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @MakeTuple() -> %tuple.type.b6b;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @H() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %empty_struct_type = binding_pattern x
// CHECK:STDOUT:     %y.patt: %empty_struct_type = binding_pattern y
// CHECK:STDOUT:     %.loc17_20: %tuple.type.b6b = tuple_pattern (%x.patt, %y.patt)
// CHECK:STDOUT:     %.loc17_3.1: %tuple.type.b6b = var_pattern %.loc17_20
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.var: ref %tuple.type.b6b = var <none>
// CHECK:STDOUT:   %MakeTuple.ref: %MakeTuple.type = name_ref MakeTuple, file.%MakeTuple.decl [concrete = constants.%MakeTuple]
// CHECK:STDOUT:   %.loc17_3.2: ref %tuple.type.b6b = splice_block %.var {}
// CHECK:STDOUT:   %MakeTuple.call: init %tuple.type.b6b = call %MakeTuple.ref() to %.loc17_3.2
// CHECK:STDOUT:   assign %.var, %MakeTuple.call
// CHECK:STDOUT:   %tuple.elem0: ref %empty_struct_type = tuple_access %.var, element0
// CHECK:STDOUT:   %tuple.elem1: ref %empty_struct_type = tuple_access %.var, element1
// CHECK:STDOUT:   %.loc17_12.1: type = splice_block %.loc17_12.3 [concrete = constants.%empty_struct_type] {
// CHECK:STDOUT:     %.loc17_12.2: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:     %.loc17_12.3: type = converted %.loc17_12.2, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %x: ref %empty_struct_type = bind_name x, %tuple.elem0
// CHECK:STDOUT:   %.loc17_19.1: type = splice_block %.loc17_19.3 [concrete = constants.%empty_struct_type] {
// CHECK:STDOUT:     %.loc17_19.2: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:     %.loc17_19.3: type = converted %.loc17_19.2, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %y: ref %empty_struct_type = bind_name y, %tuple.elem1
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- nested.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %tuple.type.b6b: type = tuple_type (%empty_struct_type, %empty_struct_type) [concrete]
// CHECK:STDOUT:   %tuple.type.6ca: type = tuple_type (%empty_struct_type, %tuple.type.b6b) [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %empty_struct_type = binding_pattern x
// CHECK:STDOUT:     %y.patt: %empty_struct_type = binding_pattern y
// CHECK:STDOUT:     %z.patt: %empty_struct_type = binding_pattern z
// CHECK:STDOUT:     %.loc5_28: %tuple.type.b6b = tuple_pattern (%y.patt, %z.patt)
// CHECK:STDOUT:     %.loc5_29: %tuple.type.6ca = tuple_pattern (%x.patt, %.loc5_28)
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc5_35.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %.loc5_40.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %.loc5_44.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %.loc5_45: %tuple.type.b6b = tuple_literal (%.loc5_40.1, %.loc5_44.1)
// CHECK:STDOUT:   %.loc5_46: %tuple.type.6ca = tuple_literal (%.loc5_35.1, %.loc5_45)
// CHECK:STDOUT:   %.loc5_12.1: type = splice_block %.loc5_12.3 [concrete = constants.%empty_struct_type] {
// CHECK:STDOUT:     %.loc5_12.2: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:     %.loc5_12.3: type = converted %.loc5_12.2, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %empty_struct.loc5_35: %empty_struct_type = struct_value () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc5_35.2: %empty_struct_type = converted %.loc5_35.1, %empty_struct.loc5_35 [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %x: %empty_struct_type = bind_name x, %.loc5_35.2
// CHECK:STDOUT:   %.loc5_20.1: type = splice_block %.loc5_20.3 [concrete = constants.%empty_struct_type] {
// CHECK:STDOUT:     %.loc5_20.2: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:     %.loc5_20.3: type = converted %.loc5_20.2, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %empty_struct.loc5_40: %empty_struct_type = struct_value () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc5_40.2: %empty_struct_type = converted %.loc5_40.1, %empty_struct.loc5_40 [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %y: %empty_struct_type = bind_name y, %.loc5_40.2
// CHECK:STDOUT:   %.loc5_27.1: type = splice_block %.loc5_27.3 [concrete = constants.%empty_struct_type] {
// CHECK:STDOUT:     %.loc5_27.2: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:     %.loc5_27.3: type = converted %.loc5_27.2, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %empty_struct.loc5_44: %empty_struct_type = struct_value () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc5_44.2: %empty_struct_type = converted %.loc5_44.1, %empty_struct.loc5_44 [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %z: %empty_struct_type = bind_name z, %.loc5_44.2
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- package_scope.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (%empty_struct_type, %empty_struct_type) [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .x = %x
// CHECK:STDOUT:     .y = %y
// CHECK:STDOUT:   }
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %empty_struct_type = binding_pattern x
// CHECK:STDOUT:     %y.patt: %empty_struct_type = binding_pattern y
// CHECK:STDOUT:     %.loc4_18: %tuple.type = tuple_pattern (%x.patt, %y.patt)
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc4_10.1: type = splice_block %.loc4_10.3 [concrete = constants.%empty_struct_type] {
// CHECK:STDOUT:     %.loc4_10.2: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:     %.loc4_10.3: type = converted %.loc4_10.2, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %empty_struct.loc4_24: %empty_struct_type = struct_value () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc4_24: %empty_struct_type = converted @__global_init.%.loc4_24, %empty_struct.loc4_24 [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %x: %empty_struct_type = bind_name x, %.loc4_24
// CHECK:STDOUT:   %.loc4_17.1: type = splice_block %.loc4_17.3 [concrete = constants.%empty_struct_type] {
// CHECK:STDOUT:     %.loc4_17.2: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:     %.loc4_17.3: type = converted %.loc4_17.2, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %empty_struct.loc4_28: %empty_struct_type = struct_value () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc4_28: %empty_struct_type = converted @__global_init.%.loc4_28, %empty_struct.loc4_28 [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %y: %empty_struct_type = bind_name y, %.loc4_28
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc4_24: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %.loc4_28: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %.loc4_29: %tuple.type = tuple_literal (%.loc4_24, %.loc4_28)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_in_interface.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [concrete]
// CHECK:STDOUT:   %Self: %I.type = bind_symbolic_name Self, 0 [symbolic]
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
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .x = <unexpected>.inst22.loc9_8
// CHECK:STDOUT:   .y = <unexpected>.inst26.loc9_15
// CHECK:STDOUT:   has_error
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_in_class.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %C.elem: type = unbound_element_type %C, %empty_struct_type [concrete]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (%C.elem, %C.elem) [concrete]
// CHECK:STDOUT:   %struct_type.x.y: type = struct_type {.x: %empty_struct_type, .y: %empty_struct_type} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.x.y [concrete]
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
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %.loc9_20: %tuple.type = tuple_pattern (<unexpected>.inst21.loc9_9, <unexpected>.inst24.loc9_16)
// CHECK:STDOUT:     %.loc9_3: %tuple.type = var_pattern %.loc9_20
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.var: ref %tuple.type = var <none>
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.x.y [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT:   .x = <unexpected>.inst21.loc9_9
// CHECK:STDOUT:   .y = <unexpected>.inst24.loc9_16
// CHECK:STDOUT:   has_error
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_initializer_mismatch.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %tuple.type.b6b: type = tuple_type (%empty_struct_type, %empty_struct_type) [concrete]
// CHECK:STDOUT:   %tuple.type.572: type = tuple_type (%empty_struct_type) [concrete]
// CHECK:STDOUT:   %tuple.type.8d4: type = tuple_type (%empty_struct_type, %empty_struct_type, %empty_struct_type) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .x = <unexpected>.inst18.loc8_6
// CHECK:STDOUT:     .y = <unexpected>.inst22.loc8_13
// CHECK:STDOUT:     .a = <unexpected>.inst32.loc14_6
// CHECK:STDOUT:     .b = <unexpected>.inst36.loc14_13
// CHECK:STDOUT:   }
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %empty_struct_type = binding_pattern x
// CHECK:STDOUT:     %y.patt: %empty_struct_type = binding_pattern y
// CHECK:STDOUT:     %.loc8: %tuple.type.b6b = tuple_pattern (%x.patt, %y.patt)
// CHECK:STDOUT:   }
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %empty_struct_type = binding_pattern a
// CHECK:STDOUT:     %b.patt: %empty_struct_type = binding_pattern b
// CHECK:STDOUT:     %.loc14: %tuple.type.b6b = tuple_pattern (%a.patt, %b.patt)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc8_24: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %.loc8_26: %tuple.type.572 = tuple_literal (%.loc8_24)
// CHECK:STDOUT:   %.loc14_24: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %.loc14_28: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %.loc14_32: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %.loc14_33: %tuple.type.8d4 = tuple_literal (%.loc14_24, %.loc14_28, %.loc14_32)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
