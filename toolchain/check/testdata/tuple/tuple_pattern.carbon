// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/destroy.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/tuple/tuple_pattern.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/tuple/tuple_pattern.carbon


// --- basic.carbon

library "[[@TEST_NAME]]";

fn F(v: ({}, {})) {
  //@dump-sem-ir-begin
  let (unused x: {}, unused y: {}) = ({}, {});
  var (unused a: {}, unused b: {}) = v;
  var (unused c: {}, unused d: {}) = ({}, {});
  //@dump-sem-ir-end
}

// --- variable.carbon

library "[[@TEST_NAME]]";

fn F() {
  //@dump-sem-ir-begin
  var tuple: ({}, {}) = ({}, {});
  var (unused x: {}, unused y: {}) = tuple;
  //@dump-sem-ir-end
}

fn G() {
  //@dump-sem-ir-begin
  let tuple: ({}, {}) = ({}, {});
  var (unused x: {}, unused y: {}) = tuple;
  //@dump-sem-ir-end
}

fn MakeTuple() -> ({}, {});

fn H() {
  //@dump-sem-ir-begin
  var (unused x: {}, unused y: {}) = MakeTuple();
  //@dump-sem-ir-end
}

// --- nested.carbon

library "[[@TEST_NAME]]";

fn F() {
  //@dump-sem-ir-begin
  let (unused x: {}, (unused y: {}, unused z: {})) = ({}, ({}, {}));
  //@dump-sem-ir-end
}

// --- package_scope.carbon

library "[[@TEST_NAME]]";

//@dump-sem-ir-begin
let (x: {}, y: {}) = ({}, {});
//@dump-sem-ir-end

// --- fail_in_class.carbon

library "[[@TEST_NAME]]";

class C {
  // CHECK:STDERR: fail_in_class.carbon:[[@LINE+8]]:7: error: expected identifier in field declaration [ExpectedFieldIdentifier]
  // CHECK:STDERR:   var (x: {}, y: {});
  // CHECK:STDERR:       ^
  // CHECK:STDERR:
  // CHECK:STDERR: fail_in_class.carbon:[[@LINE+4]]:3: error: semantics TODO: `handle invalid parse trees in `check`` [SemanticsTodo]
  // CHECK:STDERR:   var (x: {}, y: {});
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~
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
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %tuple.type.b6b: type = tuple_type (%empty_struct_type, %empty_struct_type) [concrete]
// CHECK:STDOUT:   %tuple: %tuple.type.b6b = tuple_value (%empty_struct, %empty_struct) [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %pattern_type.de4: type = pattern_type %tuple.type.b6b [concrete]
// CHECK:STDOUT:   %pattern_type.a96: type = pattern_type %empty_struct_type [concrete]
// CHECK:STDOUT:   %Destroy.Op.type: type = fn_type @Destroy.Op [concrete]
// CHECK:STDOUT:   %Destroy.Op: %Destroy.Op.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%v.param: %tuple.type.b6b) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %pattern_type.a96 = value_binding_pattern x [concrete]
// CHECK:STDOUT:     %y.patt: %pattern_type.a96 = value_binding_pattern y [concrete]
// CHECK:STDOUT:     %.loc20_34: %pattern_type.de4 = tuple_pattern (%x.patt, %y.patt) [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc20_40.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc20_44.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc20_45: %tuple.type.b6b = tuple_literal (%.loc20_40.1, %.loc20_44.1) [concrete = constants.%tuple]
// CHECK:STDOUT:   %.loc20_19.1: type = splice_block %.loc20_19.3 [concrete = constants.%empty_struct_type] {
// CHECK:STDOUT:     %.loc20_19.2: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:     %.loc20_19.3: type = converted %.loc20_19.2, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %empty_struct.loc20_40: %empty_struct_type = struct_value () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc20_40.2: %empty_struct_type = converted %.loc20_40.1, %empty_struct.loc20_40 [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %x: %empty_struct_type = value_binding x, %.loc20_40.2
// CHECK:STDOUT:   %.loc20_33.1: type = splice_block %.loc20_33.3 [concrete = constants.%empty_struct_type] {
// CHECK:STDOUT:     %.loc20_33.2: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:     %.loc20_33.3: type = converted %.loc20_33.2, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %empty_struct.loc20_44: %empty_struct_type = struct_value () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc20_44.2: %empty_struct_type = converted %.loc20_44.1, %empty_struct.loc20_44 [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %y: %empty_struct_type = value_binding y, %.loc20_44.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.a96 = ref_binding_pattern a [concrete]
// CHECK:STDOUT:     %b.patt: %pattern_type.a96 = ref_binding_pattern b [concrete]
// CHECK:STDOUT:     %.loc21_34: %pattern_type.de4 = tuple_pattern (%a.patt, %b.patt) [concrete]
// CHECK:STDOUT:     %.var_patt.loc21: %pattern_type.de4 = var_pattern %.loc21_34 [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.var.loc21: ref %tuple.type.b6b = var %.var_patt.loc21
// CHECK:STDOUT:   %v.ref: %tuple.type.b6b = name_ref v, %v
// CHECK:STDOUT:   %tuple.elem0.loc21_38.1: %empty_struct_type = tuple_access %v.ref, element0
// CHECK:STDOUT:   %tuple.elem0.loc21_38.2: ref %empty_struct_type = tuple_access %.var.loc21, element0
// CHECK:STDOUT:   %.loc21_38.1: init %empty_struct_type = struct_init () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc21_38.2: init %empty_struct_type = converted %tuple.elem0.loc21_38.1, %.loc21_38.1 [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %tuple.elem1.loc21_38.1: %empty_struct_type = tuple_access %v.ref, element1
// CHECK:STDOUT:   %tuple.elem1.loc21_38.2: ref %empty_struct_type = tuple_access %.var.loc21, element1
// CHECK:STDOUT:   %.loc21_38.3: init %empty_struct_type = struct_init () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc21_38.4: init %empty_struct_type = converted %tuple.elem1.loc21_38.1, %.loc21_38.3 [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc21_38.5: init %tuple.type.b6b to %.var.loc21 = tuple_init (%.loc21_38.2, %.loc21_38.4) [concrete = constants.%tuple]
// CHECK:STDOUT:   %.loc21_3: init %tuple.type.b6b = converted %v.ref, %.loc21_38.5 [concrete = constants.%tuple]
// CHECK:STDOUT:   assign %.var.loc21, %.loc21_3
// CHECK:STDOUT:   %tuple.elem0.loc21_3: ref %empty_struct_type = tuple_access %.var.loc21, element0
// CHECK:STDOUT:   %tuple.elem1.loc21_3: ref %empty_struct_type = tuple_access %.var.loc21, element1
// CHECK:STDOUT:   %.loc21_19.1: type = splice_block %.loc21_19.3 [concrete = constants.%empty_struct_type] {
// CHECK:STDOUT:     %.loc21_19.2: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:     %.loc21_19.3: type = converted %.loc21_19.2, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: ref %empty_struct_type = ref_binding a, %tuple.elem0.loc21_3
// CHECK:STDOUT:   %.loc21_33.1: type = splice_block %.loc21_33.3 [concrete = constants.%empty_struct_type] {
// CHECK:STDOUT:     %.loc21_33.2: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:     %.loc21_33.3: type = converted %.loc21_33.2, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b: ref %empty_struct_type = ref_binding b, %tuple.elem1.loc21_3
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c.patt: %pattern_type.a96 = ref_binding_pattern c [concrete]
// CHECK:STDOUT:     %d.patt: %pattern_type.a96 = ref_binding_pattern d [concrete]
// CHECK:STDOUT:     %.loc22_34: %pattern_type.de4 = tuple_pattern (%c.patt, %d.patt) [concrete]
// CHECK:STDOUT:     %.var_patt.loc22: %pattern_type.de4 = var_pattern %.loc22_34 [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.var.loc22: ref %tuple.type.b6b = var %.var_patt.loc22
// CHECK:STDOUT:   %.loc22_40.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc22_44.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc22_45.1: %tuple.type.b6b = tuple_literal (%.loc22_40.1, %.loc22_44.1) [concrete = constants.%tuple]
// CHECK:STDOUT:   %tuple.elem0.loc22_45: ref %empty_struct_type = tuple_access %.var.loc22, element0
// CHECK:STDOUT:   %.loc22_40.2: init %empty_struct_type = struct_init () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc22_45.2: init %empty_struct_type = converted %.loc22_40.1, %.loc22_40.2 [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %tuple.elem1.loc22_45: ref %empty_struct_type = tuple_access %.var.loc22, element1
// CHECK:STDOUT:   %.loc22_44.2: init %empty_struct_type = struct_init () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc22_45.3: init %empty_struct_type = converted %.loc22_44.1, %.loc22_44.2 [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc22_45.4: init %tuple.type.b6b to %.var.loc22 = tuple_init (%.loc22_45.2, %.loc22_45.3) [concrete = constants.%tuple]
// CHECK:STDOUT:   %.loc22_3: init %tuple.type.b6b = converted %.loc22_45.1, %.loc22_45.4 [concrete = constants.%tuple]
// CHECK:STDOUT:   assign %.var.loc22, %.loc22_3
// CHECK:STDOUT:   %tuple.elem0.loc22_3: ref %empty_struct_type = tuple_access %.var.loc22, element0
// CHECK:STDOUT:   %tuple.elem1.loc22_3: ref %empty_struct_type = tuple_access %.var.loc22, element1
// CHECK:STDOUT:   %.loc22_19.1: type = splice_block %.loc22_19.3 [concrete = constants.%empty_struct_type] {
// CHECK:STDOUT:     %.loc22_19.2: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:     %.loc22_19.3: type = converted %.loc22_19.2, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c: ref %empty_struct_type = ref_binding c, %tuple.elem0.loc22_3
// CHECK:STDOUT:   %.loc22_33.1: type = splice_block %.loc22_33.3 [concrete = constants.%empty_struct_type] {
// CHECK:STDOUT:     %.loc22_33.2: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:     %.loc22_33.3: type = converted %.loc22_33.2, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %d: ref %empty_struct_type = ref_binding d, %tuple.elem1.loc22_3
// CHECK:STDOUT:   %Destroy.Op.bound.loc22: <bound method> = bound_method %.var.loc22, constants.%Destroy.Op
// CHECK:STDOUT:   %Destroy.Op.call.loc22: init %empty_tuple.type = call %Destroy.Op.bound.loc22(%.var.loc22)
// CHECK:STDOUT:   %Destroy.Op.bound.loc21: <bound method> = bound_method %.var.loc21, constants.%Destroy.Op
// CHECK:STDOUT:   %Destroy.Op.call.loc21: init %empty_tuple.type = call %Destroy.Op.bound.loc21(%.var.loc21)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op(%self.param: ref %tuple.type.b6b) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: --- variable.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %tuple.type.b6b: type = tuple_type (%empty_struct_type, %empty_struct_type) [concrete]
// CHECK:STDOUT:   %tuple: %tuple.type.b6b = tuple_value (%empty_struct, %empty_struct) [concrete]
// CHECK:STDOUT:   %pattern_type.de4: type = pattern_type %tuple.type.b6b [concrete]
// CHECK:STDOUT:   %pattern_type.a96: type = pattern_type %empty_struct_type [concrete]
// CHECK:STDOUT:   %Destroy.Op.type: type = fn_type @Destroy.Op [concrete]
// CHECK:STDOUT:   %Destroy.Op: %Destroy.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %MakeTuple.type: type = fn_type @MakeTuple [concrete]
// CHECK:STDOUT:   %MakeTuple: %MakeTuple.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %tuple.patt: %pattern_type.de4 = ref_binding_pattern tuple [concrete]
// CHECK:STDOUT:     %tuple.var_patt: %pattern_type.de4 = var_pattern %tuple.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %tuple.var: ref %tuple.type.b6b = var %tuple.var_patt
// CHECK:STDOUT:   %.loc32_27.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc32_31.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc32_32.1: %tuple.type.b6b = tuple_literal (%.loc32_27.1, %.loc32_31.1) [concrete = constants.%tuple]
// CHECK:STDOUT:   %tuple.elem0.loc32: ref %empty_struct_type = tuple_access %tuple.var, element0
// CHECK:STDOUT:   %.loc32_27.2: init %empty_struct_type = struct_init () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc32_32.2: init %empty_struct_type = converted %.loc32_27.1, %.loc32_27.2 [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %tuple.elem1.loc32: ref %empty_struct_type = tuple_access %tuple.var, element1
// CHECK:STDOUT:   %.loc32_31.2: init %empty_struct_type = struct_init () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc32_32.3: init %empty_struct_type = converted %.loc32_31.1, %.loc32_31.2 [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc32_32.4: init %tuple.type.b6b to %tuple.var = tuple_init (%.loc32_32.2, %.loc32_32.3) [concrete = constants.%tuple]
// CHECK:STDOUT:   %.loc32_3: init %tuple.type.b6b = converted %.loc32_32.1, %.loc32_32.4 [concrete = constants.%tuple]
// CHECK:STDOUT:   assign %tuple.var, %.loc32_3
// CHECK:STDOUT:   %.loc32_21.1: type = splice_block %.loc32_21.5 [concrete = constants.%tuple.type.b6b] {
// CHECK:STDOUT:     %.loc32_16: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:     %.loc32_20: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:     %.loc32_21.2: %tuple.type.b6b = tuple_literal (%.loc32_16, %.loc32_20) [concrete = constants.%tuple]
// CHECK:STDOUT:     %.loc32_21.3: type = converted constants.%empty_struct, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     %.loc32_21.4: type = converted constants.%empty_struct, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     %.loc32_21.5: type = converted %.loc32_21.2, constants.%tuple.type.b6b [concrete = constants.%tuple.type.b6b]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %tuple: ref %tuple.type.b6b = ref_binding tuple, %tuple.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %pattern_type.a96 = ref_binding_pattern x [concrete]
// CHECK:STDOUT:     %y.patt: %pattern_type.a96 = ref_binding_pattern y [concrete]
// CHECK:STDOUT:     %.loc33_34: %pattern_type.de4 = tuple_pattern (%x.patt, %y.patt) [concrete]
// CHECK:STDOUT:     %.var_patt: %pattern_type.de4 = var_pattern %.loc33_34 [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.var: ref %tuple.type.b6b = var %.var_patt
// CHECK:STDOUT:   %tuple.ref: ref %tuple.type.b6b = name_ref tuple, %tuple
// CHECK:STDOUT:   %tuple.elem0.loc33_38.1: ref %empty_struct_type = tuple_access %tuple.ref, element0
// CHECK:STDOUT:   %tuple.elem0.loc33_38.2: ref %empty_struct_type = tuple_access %.var, element0
// CHECK:STDOUT:   %.loc33_38.1: init %empty_struct_type = struct_init () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc33_38.2: init %empty_struct_type = converted %tuple.elem0.loc33_38.1, %.loc33_38.1 [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %tuple.elem1.loc33_38.1: ref %empty_struct_type = tuple_access %tuple.ref, element1
// CHECK:STDOUT:   %tuple.elem1.loc33_38.2: ref %empty_struct_type = tuple_access %.var, element1
// CHECK:STDOUT:   %.loc33_38.3: init %empty_struct_type = struct_init () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc33_38.4: init %empty_struct_type = converted %tuple.elem1.loc33_38.1, %.loc33_38.3 [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc33_38.5: init %tuple.type.b6b to %.var = tuple_init (%.loc33_38.2, %.loc33_38.4) [concrete = constants.%tuple]
// CHECK:STDOUT:   %.loc33_3: init %tuple.type.b6b = converted %tuple.ref, %.loc33_38.5 [concrete = constants.%tuple]
// CHECK:STDOUT:   assign %.var, %.loc33_3
// CHECK:STDOUT:   %tuple.elem0.loc33_3: ref %empty_struct_type = tuple_access %.var, element0
// CHECK:STDOUT:   %tuple.elem1.loc33_3: ref %empty_struct_type = tuple_access %.var, element1
// CHECK:STDOUT:   %.loc33_19.1: type = splice_block %.loc33_19.3 [concrete = constants.%empty_struct_type] {
// CHECK:STDOUT:     %.loc33_19.2: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:     %.loc33_19.3: type = converted %.loc33_19.2, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %x: ref %empty_struct_type = ref_binding x, %tuple.elem0.loc33_3
// CHECK:STDOUT:   %.loc33_33.1: type = splice_block %.loc33_33.3 [concrete = constants.%empty_struct_type] {
// CHECK:STDOUT:     %.loc33_33.2: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:     %.loc33_33.3: type = converted %.loc33_33.2, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %y: ref %empty_struct_type = ref_binding y, %tuple.elem1.loc33_3
// CHECK:STDOUT:   %Destroy.Op.bound.loc33: <bound method> = bound_method %.var, constants.%Destroy.Op
// CHECK:STDOUT:   %Destroy.Op.call.loc33: init %empty_tuple.type = call %Destroy.Op.bound.loc33(%.var)
// CHECK:STDOUT:   %Destroy.Op.bound.loc32: <bound method> = bound_method %tuple.var, constants.%Destroy.Op
// CHECK:STDOUT:   %Destroy.Op.call.loc32: init %empty_tuple.type = call %Destroy.Op.bound.loc32(%tuple.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op(%self.param: ref %tuple.type.b6b) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %tuple.patt: %pattern_type.de4 = value_binding_pattern tuple [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc39_27: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc39_31: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc39_32.1: %tuple.type.b6b = tuple_literal (%.loc39_27, %.loc39_31) [concrete = constants.%tuple]
// CHECK:STDOUT:   %.loc39_21.1: type = splice_block %.loc39_21.5 [concrete = constants.%tuple.type.b6b] {
// CHECK:STDOUT:     %.loc39_16: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:     %.loc39_20: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:     %.loc39_21.2: %tuple.type.b6b = tuple_literal (%.loc39_16, %.loc39_20) [concrete = constants.%tuple]
// CHECK:STDOUT:     %.loc39_21.3: type = converted constants.%empty_struct, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     %.loc39_21.4: type = converted constants.%empty_struct, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     %.loc39_21.5: type = converted %.loc39_21.2, constants.%tuple.type.b6b [concrete = constants.%tuple.type.b6b]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %empty_struct.loc39_27: %empty_struct_type = struct_value () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc39_32.2: %empty_struct_type = converted %.loc39_27, %empty_struct.loc39_27 [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %empty_struct.loc39_31: %empty_struct_type = struct_value () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc39_32.3: %empty_struct_type = converted %.loc39_31, %empty_struct.loc39_31 [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %tuple.loc39_32: %tuple.type.b6b = tuple_value (%.loc39_32.2, %.loc39_32.3) [concrete = constants.%tuple]
// CHECK:STDOUT:   %.loc39_32.4: %tuple.type.b6b = converted %.loc39_32.1, %tuple.loc39_32 [concrete = constants.%tuple]
// CHECK:STDOUT:   %tuple.loc39_7: %tuple.type.b6b = value_binding tuple, %.loc39_32.4
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %pattern_type.a96 = ref_binding_pattern x [concrete]
// CHECK:STDOUT:     %y.patt: %pattern_type.a96 = ref_binding_pattern y [concrete]
// CHECK:STDOUT:     %.loc40_34: %pattern_type.de4 = tuple_pattern (%x.patt, %y.patt) [concrete]
// CHECK:STDOUT:     %.var_patt: %pattern_type.de4 = var_pattern %.loc40_34 [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.var: ref %tuple.type.b6b = var %.var_patt
// CHECK:STDOUT:   %tuple.ref: %tuple.type.b6b = name_ref tuple, %tuple.loc39_7
// CHECK:STDOUT:   %tuple.elem0.loc40_38.1: %empty_struct_type = tuple_access %tuple.ref, element0
// CHECK:STDOUT:   %tuple.elem0.loc40_38.2: ref %empty_struct_type = tuple_access %.var, element0
// CHECK:STDOUT:   %.loc40_38.1: init %empty_struct_type = struct_init () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc40_38.2: init %empty_struct_type = converted %tuple.elem0.loc40_38.1, %.loc40_38.1 [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %tuple.elem1.loc40_38.1: %empty_struct_type = tuple_access %tuple.ref, element1
// CHECK:STDOUT:   %tuple.elem1.loc40_38.2: ref %empty_struct_type = tuple_access %.var, element1
// CHECK:STDOUT:   %.loc40_38.3: init %empty_struct_type = struct_init () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc40_38.4: init %empty_struct_type = converted %tuple.elem1.loc40_38.1, %.loc40_38.3 [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc40_38.5: init %tuple.type.b6b to %.var = tuple_init (%.loc40_38.2, %.loc40_38.4) [concrete = constants.%tuple]
// CHECK:STDOUT:   %.loc40_3: init %tuple.type.b6b = converted %tuple.ref, %.loc40_38.5 [concrete = constants.%tuple]
// CHECK:STDOUT:   assign %.var, %.loc40_3
// CHECK:STDOUT:   %tuple.elem0.loc40_3: ref %empty_struct_type = tuple_access %.var, element0
// CHECK:STDOUT:   %tuple.elem1.loc40_3: ref %empty_struct_type = tuple_access %.var, element1
// CHECK:STDOUT:   %.loc40_19.1: type = splice_block %.loc40_19.3 [concrete = constants.%empty_struct_type] {
// CHECK:STDOUT:     %.loc40_19.2: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:     %.loc40_19.3: type = converted %.loc40_19.2, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %x: ref %empty_struct_type = ref_binding x, %tuple.elem0.loc40_3
// CHECK:STDOUT:   %.loc40_33.1: type = splice_block %.loc40_33.3 [concrete = constants.%empty_struct_type] {
// CHECK:STDOUT:     %.loc40_33.2: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:     %.loc40_33.3: type = converted %.loc40_33.2, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %y: ref %empty_struct_type = ref_binding y, %tuple.elem1.loc40_3
// CHECK:STDOUT:   %Destroy.Op.bound: <bound method> = bound_method %.var, constants.%Destroy.Op
// CHECK:STDOUT:   %Destroy.Op.call: init %empty_tuple.type = call %Destroy.Op.bound(%.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @H() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %pattern_type.a96 = ref_binding_pattern x [concrete]
// CHECK:STDOUT:     %y.patt: %pattern_type.a96 = ref_binding_pattern y [concrete]
// CHECK:STDOUT:     %.loc48_34: %pattern_type.de4 = tuple_pattern (%x.patt, %y.patt) [concrete]
// CHECK:STDOUT:     %.var_patt: %pattern_type.de4 = var_pattern %.loc48_34 [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.var: ref %tuple.type.b6b = var %.var_patt
// CHECK:STDOUT:   %MakeTuple.ref: %MakeTuple.type = name_ref MakeTuple, file.%MakeTuple.decl [concrete = constants.%MakeTuple]
// CHECK:STDOUT:   %.loc48_3: ref %tuple.type.b6b = splice_block %.var {}
// CHECK:STDOUT:   %MakeTuple.call: init %tuple.type.b6b to %.loc48_3 = call %MakeTuple.ref()
// CHECK:STDOUT:   assign %.var, %MakeTuple.call
// CHECK:STDOUT:   %tuple.elem0: ref %empty_struct_type = tuple_access %.var, element0
// CHECK:STDOUT:   %tuple.elem1: ref %empty_struct_type = tuple_access %.var, element1
// CHECK:STDOUT:   %.loc48_19.1: type = splice_block %.loc48_19.3 [concrete = constants.%empty_struct_type] {
// CHECK:STDOUT:     %.loc48_19.2: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:     %.loc48_19.3: type = converted %.loc48_19.2, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %x: ref %empty_struct_type = ref_binding x, %tuple.elem0
// CHECK:STDOUT:   %.loc48_33.1: type = splice_block %.loc48_33.3 [concrete = constants.%empty_struct_type] {
// CHECK:STDOUT:     %.loc48_33.2: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:     %.loc48_33.3: type = converted %.loc48_33.2, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %y: ref %empty_struct_type = ref_binding y, %tuple.elem1
// CHECK:STDOUT:   %Destroy.Op.bound: <bound method> = bound_method %.var, constants.%Destroy.Op
// CHECK:STDOUT:   %Destroy.Op.call: init %empty_tuple.type = call %Destroy.Op.bound(%.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- nested.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.a96: type = pattern_type %empty_struct_type [concrete]
// CHECK:STDOUT:   %tuple.type.b6b: type = tuple_type (%empty_struct_type, %empty_struct_type) [concrete]
// CHECK:STDOUT:   %pattern_type.de4: type = pattern_type %tuple.type.b6b [concrete]
// CHECK:STDOUT:   %tuple.type.6ca: type = tuple_type (%empty_struct_type, %tuple.type.b6b) [concrete]
// CHECK:STDOUT:   %pattern_type.361: type = pattern_type %tuple.type.6ca [concrete]
// CHECK:STDOUT:   %tuple.9a3: %tuple.type.b6b = tuple_value (%empty_struct, %empty_struct) [concrete]
// CHECK:STDOUT:   %tuple.56c: %tuple.type.6ca = tuple_value (%empty_struct, %tuple.9a3) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %pattern_type.a96 = value_binding_pattern x [concrete]
// CHECK:STDOUT:     %y.patt: %pattern_type.a96 = value_binding_pattern y [concrete]
// CHECK:STDOUT:     %z.patt: %pattern_type.a96 = value_binding_pattern z [concrete]
// CHECK:STDOUT:     %.loc58_49: %pattern_type.de4 = tuple_pattern (%y.patt, %z.patt) [concrete]
// CHECK:STDOUT:     %.loc58_50: %pattern_type.361 = tuple_pattern (%x.patt, %.loc58_49) [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc58_56.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc58_61.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc58_65.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc58_66: %tuple.type.b6b = tuple_literal (%.loc58_61.1, %.loc58_65.1) [concrete = constants.%tuple.9a3]
// CHECK:STDOUT:   %.loc58_67: %tuple.type.6ca = tuple_literal (%.loc58_56.1, %.loc58_66) [concrete = constants.%tuple.56c]
// CHECK:STDOUT:   %.loc58_19.1: type = splice_block %.loc58_19.3 [concrete = constants.%empty_struct_type] {
// CHECK:STDOUT:     %.loc58_19.2: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:     %.loc58_19.3: type = converted %.loc58_19.2, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %empty_struct.loc58_56: %empty_struct_type = struct_value () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc58_56.2: %empty_struct_type = converted %.loc58_56.1, %empty_struct.loc58_56 [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %x: %empty_struct_type = value_binding x, %.loc58_56.2
// CHECK:STDOUT:   %.loc58_34.1: type = splice_block %.loc58_34.3 [concrete = constants.%empty_struct_type] {
// CHECK:STDOUT:     %.loc58_34.2: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:     %.loc58_34.3: type = converted %.loc58_34.2, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %empty_struct.loc58_61: %empty_struct_type = struct_value () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc58_61.2: %empty_struct_type = converted %.loc58_61.1, %empty_struct.loc58_61 [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %y: %empty_struct_type = value_binding y, %.loc58_61.2
// CHECK:STDOUT:   %.loc58_48.1: type = splice_block %.loc58_48.3 [concrete = constants.%empty_struct_type] {
// CHECK:STDOUT:     %.loc58_48.2: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:     %.loc58_48.3: type = converted %.loc58_48.2, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %empty_struct.loc58_65: %empty_struct_type = struct_value () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc58_65.2: %empty_struct_type = converted %.loc58_65.1, %empty_struct.loc58_65 [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %z: %empty_struct_type = value_binding z, %.loc58_65.2
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- package_scope.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.a96: type = pattern_type %empty_struct_type [concrete]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (%empty_struct_type, %empty_struct_type) [concrete]
// CHECK:STDOUT:   %pattern_type.de4: type = pattern_type %tuple.type [concrete]
// CHECK:STDOUT:   %tuple: %tuple.type = tuple_value (%empty_struct, %empty_struct) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %pattern_type.a96 = value_binding_pattern x [concrete]
// CHECK:STDOUT:     %y.patt: %pattern_type.a96 = value_binding_pattern y [concrete]
// CHECK:STDOUT:     %.loc67_18: %pattern_type.de4 = tuple_pattern (%x.patt, %y.patt) [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc67_10.1: type = splice_block %.loc67_10.3 [concrete = constants.%empty_struct_type] {
// CHECK:STDOUT:     %.loc67_10.2: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:     %.loc67_10.3: type = converted %.loc67_10.2, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %empty_struct.loc67_24: %empty_struct_type = struct_value () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc67_24: %empty_struct_type = converted @__global_init.%.loc67_24, %empty_struct.loc67_24 [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %x: %empty_struct_type = value_binding x, %.loc67_24
// CHECK:STDOUT:   %.loc67_17.1: type = splice_block %.loc67_17.3 [concrete = constants.%empty_struct_type] {
// CHECK:STDOUT:     %.loc67_17.2: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:     %.loc67_17.3: type = converted %.loc67_17.2, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %empty_struct.loc67_28: %empty_struct_type = struct_value () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc67_28: %empty_struct_type = converted @__global_init.%.loc67_28, %empty_struct.loc67_28 [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %y: %empty_struct_type = value_binding y, %.loc67_28
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc67_24: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc67_28: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc67_29: %tuple.type = tuple_literal (%.loc67_24, %.loc67_28) [concrete = constants.%tuple]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
