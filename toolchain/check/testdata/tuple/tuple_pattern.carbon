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
// CHECK:STDOUT:     %.loc31: %pattern_type.de4 = tuple_pattern (%x.patt, %y.patt) [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc35.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc38.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc39: %tuple.type.b6b = tuple_literal (%.loc35.1, %.loc38.1) [concrete = constants.%tuple]
// CHECK:STDOUT:   %.loc24.1: type = splice_block %.loc24.3 [concrete = constants.%empty_struct_type] {
// CHECK:STDOUT:     %.loc24.2: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:     %.loc24.3: type = converted %.loc24.2, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %empty_struct.loc35: %empty_struct_type = struct_value () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc35.2: %empty_struct_type = converted %.loc35.1, %empty_struct.loc35 [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %x: %empty_struct_type = value_binding x, %.loc35.2
// CHECK:STDOUT:   %.loc30.1: type = splice_block %.loc30.3 [concrete = constants.%empty_struct_type] {
// CHECK:STDOUT:     %.loc30.2: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:     %.loc30.3: type = converted %.loc30.2, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %empty_struct.loc38: %empty_struct_type = struct_value () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc38.2: %empty_struct_type = converted %.loc38.1, %empty_struct.loc38 [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %y: %empty_struct_type = value_binding y, %.loc38.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.a96 = ref_binding_pattern a [concrete]
// CHECK:STDOUT:     %b.patt: %pattern_type.a96 = ref_binding_pattern b [concrete]
// CHECK:STDOUT:     %.loc54: %pattern_type.de4 = tuple_pattern (%a.patt, %b.patt) [concrete]
// CHECK:STDOUT:     %.var_patt.loc41: %pattern_type.de4 = var_pattern %.loc54 [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.var.loc41: ref %tuple.type.b6b = var %.var_patt.loc41
// CHECK:STDOUT:   %v.ref: %tuple.type.b6b = name_ref v, %v
// CHECK:STDOUT:   %tuple.elem0.loc56.1: %empty_struct_type = tuple_access %v.ref, element0
// CHECK:STDOUT:   %tuple.elem0.loc56.2: ref %empty_struct_type = tuple_access %.var.loc41, element0
// CHECK:STDOUT:   %.loc56.1: init %empty_struct_type = struct_init () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc56.2: init %empty_struct_type = converted %tuple.elem0.loc56.1, %.loc56.1 [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %tuple.elem1.loc56.1: %empty_struct_type = tuple_access %v.ref, element1
// CHECK:STDOUT:   %tuple.elem1.loc56.2: ref %empty_struct_type = tuple_access %.var.loc41, element1
// CHECK:STDOUT:   %.loc56.3: init %empty_struct_type = struct_init () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc56.4: init %empty_struct_type = converted %tuple.elem1.loc56.1, %.loc56.3 [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc56.5: init %tuple.type.b6b to %.var.loc41 = tuple_init (%.loc56.2, %.loc56.4) [concrete = constants.%tuple]
// CHECK:STDOUT:   %.loc41: init %tuple.type.b6b = converted %v.ref, %.loc56.5 [concrete = constants.%tuple]
// CHECK:STDOUT:   assign %.var.loc41, %.loc41
// CHECK:STDOUT:   %tuple.elem0.loc41: ref %empty_struct_type = tuple_access %.var.loc41, element0
// CHECK:STDOUT:   %tuple.elem1.loc41: ref %empty_struct_type = tuple_access %.var.loc41, element1
// CHECK:STDOUT:   %.loc47.1: type = splice_block %.loc47.3 [concrete = constants.%empty_struct_type] {
// CHECK:STDOUT:     %.loc47.2: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:     %.loc47.3: type = converted %.loc47.2, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: ref %empty_struct_type = ref_binding a, %tuple.elem0.loc41
// CHECK:STDOUT:   %.loc53.1: type = splice_block %.loc53.3 [concrete = constants.%empty_struct_type] {
// CHECK:STDOUT:     %.loc53.2: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:     %.loc53.3: type = converted %.loc53.2, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b: ref %empty_struct_type = ref_binding b, %tuple.elem1.loc41
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c.patt: %pattern_type.a96 = ref_binding_pattern c [concrete]
// CHECK:STDOUT:     %d.patt: %pattern_type.a96 = ref_binding_pattern d [concrete]
// CHECK:STDOUT:     %.loc71: %pattern_type.de4 = tuple_pattern (%c.patt, %d.patt) [concrete]
// CHECK:STDOUT:     %.var_patt.loc58: %pattern_type.de4 = var_pattern %.loc71 [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.var.loc58: ref %tuple.type.b6b = var %.var_patt.loc58
// CHECK:STDOUT:   %.loc75.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc78.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc79.1: %tuple.type.b6b = tuple_literal (%.loc75.1, %.loc78.1) [concrete = constants.%tuple]
// CHECK:STDOUT:   %tuple.elem0.loc79: ref %empty_struct_type = tuple_access %.var.loc58, element0
// CHECK:STDOUT:   %.loc75.2: init %empty_struct_type = struct_init () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc79.2: init %empty_struct_type = converted %.loc75.1, %.loc75.2 [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %tuple.elem1.loc79: ref %empty_struct_type = tuple_access %.var.loc58, element1
// CHECK:STDOUT:   %.loc78.2: init %empty_struct_type = struct_init () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc79.3: init %empty_struct_type = converted %.loc78.1, %.loc78.2 [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc79.4: init %tuple.type.b6b to %.var.loc58 = tuple_init (%.loc79.2, %.loc79.3) [concrete = constants.%tuple]
// CHECK:STDOUT:   %.loc58: init %tuple.type.b6b = converted %.loc79.1, %.loc79.4 [concrete = constants.%tuple]
// CHECK:STDOUT:   assign %.var.loc58, %.loc58
// CHECK:STDOUT:   %tuple.elem0.loc58: ref %empty_struct_type = tuple_access %.var.loc58, element0
// CHECK:STDOUT:   %tuple.elem1.loc58: ref %empty_struct_type = tuple_access %.var.loc58, element1
// CHECK:STDOUT:   %.loc64.1: type = splice_block %.loc64.3 [concrete = constants.%empty_struct_type] {
// CHECK:STDOUT:     %.loc64.2: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:     %.loc64.3: type = converted %.loc64.2, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c: ref %empty_struct_type = ref_binding c, %tuple.elem0.loc58
// CHECK:STDOUT:   %.loc70.1: type = splice_block %.loc70.3 [concrete = constants.%empty_struct_type] {
// CHECK:STDOUT:     %.loc70.2: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:     %.loc70.3: type = converted %.loc70.2, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %d: ref %empty_struct_type = ref_binding d, %tuple.elem1.loc58
// CHECK:STDOUT:   %Destroy.Op.bound.loc58: <bound method> = bound_method %.var.loc58, constants.%Destroy.Op
// CHECK:STDOUT:   %Destroy.Op.call.loc58: init %empty_tuple.type = call %Destroy.Op.bound.loc58(%.var.loc58)
// CHECK:STDOUT:   %Destroy.Op.bound.loc41: <bound method> = bound_method %.var.loc41, constants.%Destroy.Op
// CHECK:STDOUT:   %Destroy.Op.call.loc41: init %empty_tuple.type = call %Destroy.Op.bound.loc41(%.var.loc41)
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
// CHECK:STDOUT:   %.loc22.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc25.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc26.1: %tuple.type.b6b = tuple_literal (%.loc22.1, %.loc25.1) [concrete = constants.%tuple]
// CHECK:STDOUT:   %tuple.elem0.loc26: ref %empty_struct_type = tuple_access %tuple.var, element0
// CHECK:STDOUT:   %.loc22.2: init %empty_struct_type = struct_init () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc26.2: init %empty_struct_type = converted %.loc22.1, %.loc22.2 [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %tuple.elem1.loc26: ref %empty_struct_type = tuple_access %tuple.var, element1
// CHECK:STDOUT:   %.loc25.2: init %empty_struct_type = struct_init () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc26.3: init %empty_struct_type = converted %.loc25.1, %.loc25.2 [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc26.4: init %tuple.type.b6b to %tuple.var = tuple_init (%.loc26.2, %.loc26.3) [concrete = constants.%tuple]
// CHECK:STDOUT:   %.loc9: init %tuple.type.b6b = converted %.loc26.1, %.loc26.4 [concrete = constants.%tuple]
// CHECK:STDOUT:   assign %tuple.var, %.loc9
// CHECK:STDOUT:   %.loc18.1: type = splice_block %.loc18.5 [concrete = constants.%tuple.type.b6b] {
// CHECK:STDOUT:     %.loc14: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:     %.loc17: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:     %.loc18.2: %tuple.type.b6b = tuple_literal (%.loc14, %.loc17) [concrete = constants.%tuple]
// CHECK:STDOUT:     %.loc18.3: type = converted constants.%empty_struct, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     %.loc18.4: type = converted constants.%empty_struct, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     %.loc18.5: type = converted %.loc18.2, constants.%tuple.type.b6b [concrete = constants.%tuple.type.b6b]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %tuple: ref %tuple.type.b6b = ref_binding tuple, %tuple.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %pattern_type.a96 = ref_binding_pattern x [concrete]
// CHECK:STDOUT:     %y.patt: %pattern_type.a96 = ref_binding_pattern y [concrete]
// CHECK:STDOUT:     %.loc41: %pattern_type.de4 = tuple_pattern (%x.patt, %y.patt) [concrete]
// CHECK:STDOUT:     %.var_patt: %pattern_type.de4 = var_pattern %.loc41 [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.var: ref %tuple.type.b6b = var %.var_patt
// CHECK:STDOUT:   %tuple.ref: ref %tuple.type.b6b = name_ref tuple, %tuple
// CHECK:STDOUT:   %tuple.elem0.loc43.1: ref %empty_struct_type = tuple_access %tuple.ref, element0
// CHECK:STDOUT:   %tuple.elem0.loc43.2: ref %empty_struct_type = tuple_access %.var, element0
// CHECK:STDOUT:   %.loc43.1: init %empty_struct_type = struct_init () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc43.2: init %empty_struct_type = converted %tuple.elem0.loc43.1, %.loc43.1 [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %tuple.elem1.loc43.1: ref %empty_struct_type = tuple_access %tuple.ref, element1
// CHECK:STDOUT:   %tuple.elem1.loc43.2: ref %empty_struct_type = tuple_access %.var, element1
// CHECK:STDOUT:   %.loc43.3: init %empty_struct_type = struct_init () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc43.4: init %empty_struct_type = converted %tuple.elem1.loc43.1, %.loc43.3 [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc43.5: init %tuple.type.b6b to %.var = tuple_init (%.loc43.2, %.loc43.4) [concrete = constants.%tuple]
// CHECK:STDOUT:   %.loc28: init %tuple.type.b6b = converted %tuple.ref, %.loc43.5 [concrete = constants.%tuple]
// CHECK:STDOUT:   assign %.var, %.loc28
// CHECK:STDOUT:   %tuple.elem0.loc28: ref %empty_struct_type = tuple_access %.var, element0
// CHECK:STDOUT:   %tuple.elem1.loc28: ref %empty_struct_type = tuple_access %.var, element1
// CHECK:STDOUT:   %.loc34.1: type = splice_block %.loc34.3 [concrete = constants.%empty_struct_type] {
// CHECK:STDOUT:     %.loc34.2: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:     %.loc34.3: type = converted %.loc34.2, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %x: ref %empty_struct_type = ref_binding x, %tuple.elem0.loc28
// CHECK:STDOUT:   %.loc40.1: type = splice_block %.loc40.3 [concrete = constants.%empty_struct_type] {
// CHECK:STDOUT:     %.loc40.2: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:     %.loc40.3: type = converted %.loc40.2, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %y: ref %empty_struct_type = ref_binding y, %tuple.elem1.loc28
// CHECK:STDOUT:   %Destroy.Op.bound.loc28: <bound method> = bound_method %.var, constants.%Destroy.Op
// CHECK:STDOUT:   %Destroy.Op.call.loc28: init %empty_tuple.type = call %Destroy.Op.bound.loc28(%.var)
// CHECK:STDOUT:   %Destroy.Op.bound.loc9: <bound method> = bound_method %tuple.var, constants.%Destroy.Op
// CHECK:STDOUT:   %Destroy.Op.call.loc9: init %empty_tuple.type = call %Destroy.Op.bound.loc9(%tuple.var)
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
// CHECK:STDOUT:   %.loc64: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc67: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc68.1: %tuple.type.b6b = tuple_literal (%.loc64, %.loc67) [concrete = constants.%tuple]
// CHECK:STDOUT:   %.loc60.1: type = splice_block %.loc60.5 [concrete = constants.%tuple.type.b6b] {
// CHECK:STDOUT:     %.loc56: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:     %.loc59: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:     %.loc60.2: %tuple.type.b6b = tuple_literal (%.loc56, %.loc59) [concrete = constants.%tuple]
// CHECK:STDOUT:     %.loc60.3: type = converted constants.%empty_struct, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     %.loc60.4: type = converted constants.%empty_struct, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     %.loc60.5: type = converted %.loc60.2, constants.%tuple.type.b6b [concrete = constants.%tuple.type.b6b]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %empty_struct.loc64: %empty_struct_type = struct_value () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc68.2: %empty_struct_type = converted %.loc64, %empty_struct.loc64 [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %empty_struct.loc67: %empty_struct_type = struct_value () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc68.3: %empty_struct_type = converted %.loc67, %empty_struct.loc67 [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %tuple.loc68: %tuple.type.b6b = tuple_value (%.loc68.2, %.loc68.3) [concrete = constants.%tuple]
// CHECK:STDOUT:   %.loc68.4: %tuple.type.b6b = converted %.loc68.1, %tuple.loc68 [concrete = constants.%tuple]
// CHECK:STDOUT:   %tuple.loc52: %tuple.type.b6b = value_binding tuple, %.loc68.4
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %pattern_type.a96 = ref_binding_pattern x [concrete]
// CHECK:STDOUT:     %y.patt: %pattern_type.a96 = ref_binding_pattern y [concrete]
// CHECK:STDOUT:     %.loc83: %pattern_type.de4 = tuple_pattern (%x.patt, %y.patt) [concrete]
// CHECK:STDOUT:     %.var_patt: %pattern_type.de4 = var_pattern %.loc83 [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.var: ref %tuple.type.b6b = var %.var_patt
// CHECK:STDOUT:   %tuple.ref: %tuple.type.b6b = name_ref tuple, %tuple.loc52
// CHECK:STDOUT:   %tuple.elem0.loc85.1: %empty_struct_type = tuple_access %tuple.ref, element0
// CHECK:STDOUT:   %tuple.elem0.loc85.2: ref %empty_struct_type = tuple_access %.var, element0
// CHECK:STDOUT:   %.loc85.1: init %empty_struct_type = struct_init () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc85.2: init %empty_struct_type = converted %tuple.elem0.loc85.1, %.loc85.1 [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %tuple.elem1.loc85.1: %empty_struct_type = tuple_access %tuple.ref, element1
// CHECK:STDOUT:   %tuple.elem1.loc85.2: ref %empty_struct_type = tuple_access %.var, element1
// CHECK:STDOUT:   %.loc85.3: init %empty_struct_type = struct_init () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc85.4: init %empty_struct_type = converted %tuple.elem1.loc85.1, %.loc85.3 [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc85.5: init %tuple.type.b6b to %.var = tuple_init (%.loc85.2, %.loc85.4) [concrete = constants.%tuple]
// CHECK:STDOUT:   %.loc70: init %tuple.type.b6b = converted %tuple.ref, %.loc85.5 [concrete = constants.%tuple]
// CHECK:STDOUT:   assign %.var, %.loc70
// CHECK:STDOUT:   %tuple.elem0.loc70: ref %empty_struct_type = tuple_access %.var, element0
// CHECK:STDOUT:   %tuple.elem1.loc70: ref %empty_struct_type = tuple_access %.var, element1
// CHECK:STDOUT:   %.loc76.1: type = splice_block %.loc76.3 [concrete = constants.%empty_struct_type] {
// CHECK:STDOUT:     %.loc76.2: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:     %.loc76.3: type = converted %.loc76.2, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %x: ref %empty_struct_type = ref_binding x, %tuple.elem0.loc70
// CHECK:STDOUT:   %.loc82.1: type = splice_block %.loc82.3 [concrete = constants.%empty_struct_type] {
// CHECK:STDOUT:     %.loc82.2: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:     %.loc82.3: type = converted %.loc82.2, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %y: ref %empty_struct_type = ref_binding y, %tuple.elem1.loc70
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
// CHECK:STDOUT:     %.loc119: %pattern_type.de4 = tuple_pattern (%x.patt, %y.patt) [concrete]
// CHECK:STDOUT:     %.var_patt: %pattern_type.de4 = var_pattern %.loc119 [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.var: ref %tuple.type.b6b = var %.var_patt
// CHECK:STDOUT:   %MakeTuple.ref: %MakeTuple.type = name_ref MakeTuple, file.%MakeTuple.decl [concrete = constants.%MakeTuple]
// CHECK:STDOUT:   %.loc106: ref %tuple.type.b6b = splice_block %.var {}
// CHECK:STDOUT:   %MakeTuple.call: init %tuple.type.b6b to %.loc106 = call %MakeTuple.ref()
// CHECK:STDOUT:   assign %.var, %MakeTuple.call
// CHECK:STDOUT:   %tuple.elem0: ref %empty_struct_type = tuple_access %.var, element0
// CHECK:STDOUT:   %tuple.elem1: ref %empty_struct_type = tuple_access %.var, element1
// CHECK:STDOUT:   %.loc112.1: type = splice_block %.loc112.3 [concrete = constants.%empty_struct_type] {
// CHECK:STDOUT:     %.loc112.2: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:     %.loc112.3: type = converted %.loc112.2, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %x: ref %empty_struct_type = ref_binding x, %tuple.elem0
// CHECK:STDOUT:   %.loc118.1: type = splice_block %.loc118.3 [concrete = constants.%empty_struct_type] {
// CHECK:STDOUT:     %.loc118.2: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:     %.loc118.3: type = converted %.loc118.2, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
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
// CHECK:STDOUT:     %.loc29: %pattern_type.de4 = tuple_pattern (%y.patt, %z.patt) [concrete]
// CHECK:STDOUT:     %.loc30: %pattern_type.361 = tuple_pattern (%x.patt, %.loc29) [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc34.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc38.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc41.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc42: %tuple.type.b6b = tuple_literal (%.loc38.1, %.loc41.1) [concrete = constants.%tuple.9a3]
// CHECK:STDOUT:   %.loc43: %tuple.type.6ca = tuple_literal (%.loc34.1, %.loc42) [concrete = constants.%tuple.56c]
// CHECK:STDOUT:   %.loc15.1: type = splice_block %.loc15.3 [concrete = constants.%empty_struct_type] {
// CHECK:STDOUT:     %.loc15.2: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:     %.loc15.3: type = converted %.loc15.2, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %empty_struct.loc34: %empty_struct_type = struct_value () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc34.2: %empty_struct_type = converted %.loc34.1, %empty_struct.loc34 [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %x: %empty_struct_type = value_binding x, %.loc34.2
// CHECK:STDOUT:   %.loc22.1: type = splice_block %.loc22.3 [concrete = constants.%empty_struct_type] {
// CHECK:STDOUT:     %.loc22.2: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:     %.loc22.3: type = converted %.loc22.2, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %empty_struct.loc38: %empty_struct_type = struct_value () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc38.2: %empty_struct_type = converted %.loc38.1, %empty_struct.loc38 [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %y: %empty_struct_type = value_binding y, %.loc38.2
// CHECK:STDOUT:   %.loc28.1: type = splice_block %.loc28.3 [concrete = constants.%empty_struct_type] {
// CHECK:STDOUT:     %.loc28.2: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:     %.loc28.3: type = converted %.loc28.2, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %empty_struct.loc41: %empty_struct_type = struct_value () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc41.2: %empty_struct_type = converted %.loc41.1, %empty_struct.loc41 [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %z: %empty_struct_type = value_binding z, %.loc41.2
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
// CHECK:STDOUT:     %.loc15: %pattern_type.de4 = tuple_pattern (%x.patt, %y.patt) [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc9.1: type = splice_block %.loc9.3 [concrete = constants.%empty_struct_type] {
// CHECK:STDOUT:     %.loc9.2: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:     %.loc9.3: type = converted %.loc9.2, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %empty_struct.loc19: %empty_struct_type = struct_value () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc19: %empty_struct_type = converted @__global_init.%.loc19, %empty_struct.loc19 [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %x: %empty_struct_type = value_binding x, %.loc19
// CHECK:STDOUT:   %.loc14.1: type = splice_block %.loc14.3 [concrete = constants.%empty_struct_type] {
// CHECK:STDOUT:     %.loc14.2: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:     %.loc14.3: type = converted %.loc14.2, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %empty_struct.loc22: %empty_struct_type = struct_value () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc22: %empty_struct_type = converted @__global_init.%.loc22, %empty_struct.loc22 [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %y: %empty_struct_type = value_binding y, %.loc22
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc19: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc22: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc23: %tuple.type = tuple_literal (%.loc19, %.loc22) [concrete = constants.%tuple]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
