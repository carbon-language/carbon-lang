// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// This is mostly checking against crashes for compile time bindings in
// difficult contexts.
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/none.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/patterns/tuple.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/patterns/tuple.carbon

// --- literal.carbon

library "[[@TEST_NAME]]";

//@dump-sem-ir-begin
let (a: (), b: ()) = ((), ());
//@dump-sem-ir-end

// --- init_expr.carbon

library "[[@TEST_NAME]]";

fn F() -> ((), ());

//@dump-sem-ir-begin
let (a: (), b: ()) = F();
//@dump-sem-ir-end

// --- ref_expr.carbon

library "[[@TEST_NAME]]";

var t: ((), ()) = ((), ());
//@dump-sem-ir-begin
let (a: (), b: ()) = t;
//@dump-sem-ir-end

// --- grouping_parens.carbon

library "[[@TEST_NAME]]";

//@dump-sem-ir-begin
let (a: ()) = ();
let (((b: ()))) = ();
//@dump-sem-ir-end

// --- one_tuple.carbon

library "[[@TEST_NAME]]";

//@dump-sem-ir-begin
let (a: (),) = ((),);
let (((b: ()),)) = ((),);
//@dump-sem-ir-end

// --- fail_in_class_var.carbon

library "[[@TEST_NAME]]";

class C {
  //@dump-sem-ir-begin
  // CHECK:STDERR:        ╭─┤ fail_in_class_var.carbon:[[@LINE+7]]:7
  // CHECK:STDERR: ->     │   var (a: (), b: ());
  // CHECK:STDERR:        ·       ━━━━━━━┳━━━━━━
  // CHECK:STDERR:        ·              ┗━━┫ error: tuple pattern in field declaration
  // CHECK:STDERR:        ·                 ┃        [FieldWithTuplePattern]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  var (a: (), b: ());
  //@dump-sem-ir-end
}

// --- fail_empty_in_class_var.carbon

library "[[@TEST_NAME]]";

class C {
  //@dump-sem-ir-begin
  // CHECK:STDERR:        ╭─┤ fail_empty_in_class_var.carbon:[[@LINE+6]]:7
  // CHECK:STDERR: ->     │   var ();
  // CHECK:STDERR:        ·       ━┳
  // CHECK:STDERR:        ·        ┗━━┫ error: tuple pattern in field declaration [FieldWithTuplePattern]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  var ();
  //@dump-sem-ir-end
}

// --- in_static_class_var.carbon

library "[[@TEST_NAME]]";

class C {
  //@dump-sem-ir-begin
  static var (a: (), b: ()) = ((), ());
  //@dump-sem-ir-end
}

// --- in_class_let.carbon

library "[[@TEST_NAME]]";

class C {
  //@dump-sem-ir-begin
  let (a: (), b: ()) = ((), ());
  //@dump-sem-ir-end
}

// --- empty_in_class_let.carbon

library "[[@TEST_NAME]]";

class C {
  //@dump-sem-ir-begin
  let () = ();
  //@dump-sem-ir-end
}

// CHECK:STDOUT: --- literal.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %pattern_type.cb1: type = pattern_type %empty_tuple.type [concrete]
// CHECK:STDOUT:   %a.patt: %pattern_type.cb1 = value_binding_pattern a [concrete]
// CHECK:STDOUT:   %b.patt: %pattern_type.cb1 = value_binding_pattern b [concrete]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (%empty_tuple.type, %empty_tuple.type) [concrete]
// CHECK:STDOUT:   %pattern_type.5b8: type = pattern_type %tuple.type [concrete]
// CHECK:STDOUT:   %.045: %pattern_type.5b8 = tuple_pattern (%a.patt, %b.patt) [concrete]
// CHECK:STDOUT:   %tuple: %tuple.type = tuple_value (%empty_tuple, %empty_tuple) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %.loc5_10.1: type = splice_block %.loc5_10.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:     %.loc5_10.2: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc5_10.3: type = converted %.loc5_10.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: %empty_tuple.type = wrapper_binding a, @__global_init.%.loc5_24.2
// CHECK:STDOUT:   %.loc5_17.1: type = splice_block %.loc5_17.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:     %.loc5_17.2: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc5_17.3: type = converted %.loc5_17.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b: %empty_tuple.type = wrapper_binding b, @__global_init.%.loc5_28.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.cb1 = value_binding_pattern a [concrete = constants.%a.patt]
// CHECK:STDOUT:     %b.patt: %pattern_type.cb1 = value_binding_pattern b [concrete = constants.%b.patt]
// CHECK:STDOUT:     %.loc5_18: %pattern_type.5b8 = tuple_pattern (%a.patt, %b.patt) [concrete = constants.%.045]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc5_24.1: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc5_28.1: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc5_29: %tuple.type = tuple_literal (%.loc5_24.1, %.loc5_28.1) [concrete = constants.%tuple]
// CHECK:STDOUT:   %empty_tuple.loc5_24: %empty_tuple.type = tuple_value () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc5_24.2: %empty_tuple.type = converted %.loc5_24.1, %empty_tuple.loc5_24 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %empty_tuple.loc5_28: %empty_tuple.type = tuple_value () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc5_28.2: %empty_tuple.type = converted %.loc5_28.1, %empty_tuple.loc5_28 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- init_expr.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (%empty_tuple.type, %empty_tuple.type) [concrete]
// CHECK:STDOUT:   %pattern_type.5b8: type = pattern_type %tuple.type [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.cb1: type = pattern_type %empty_tuple.type [concrete]
// CHECK:STDOUT:   %a.patt: %pattern_type.cb1 = value_binding_pattern a [concrete]
// CHECK:STDOUT:   %b.patt: %pattern_type.cb1 = value_binding_pattern b [concrete]
// CHECK:STDOUT:   %.045: %pattern_type.5b8 = tuple_pattern (%a.patt, %b.patt) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %.loc7_10.1: type = splice_block %.loc7_10.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:     %.loc7_10.2: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc7_10.3: type = converted %.loc7_10.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: %empty_tuple.type = wrapper_binding a, @__global_init.%.loc7_24.3
// CHECK:STDOUT:   %.loc7_17.1: type = splice_block %.loc7_17.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:     %.loc7_17.2: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc7_17.3: type = converted %.loc7_17.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b: %empty_tuple.type = wrapper_binding b, @__global_init.%.loc7_24.4
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.cb1 = value_binding_pattern a [concrete = constants.%a.patt]
// CHECK:STDOUT:     %b.patt: %pattern_type.cb1 = value_binding_pattern b [concrete = constants.%b.patt]
// CHECK:STDOUT:     %.loc7_18: %pattern_type.5b8 = tuple_pattern (%a.patt, %b.patt) [concrete = constants.%.045]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:   %.loc7_24.1: ref %tuple.type = temporary_storage
// CHECK:STDOUT:   %F.call: init %tuple.type to %.loc7_24.1 = call %F.ref()
// CHECK:STDOUT:   %.loc7_24.2: ref %tuple.type = temporary %.loc7_24.1, %F.call
// CHECK:STDOUT:   %tuple.elem0: ref %empty_tuple.type = tuple_access %.loc7_24.2, element0
// CHECK:STDOUT:   %tuple.elem1: ref %empty_tuple.type = tuple_access %.loc7_24.2, element1
// CHECK:STDOUT:   %empty_tuple.loc7_24.1: %empty_tuple.type = tuple_value () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc7_24.3: %empty_tuple.type = converted %tuple.elem0, %empty_tuple.loc7_24.1 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %empty_tuple.loc7_24.2: %empty_tuple.type = tuple_value () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc7_24.4: %empty_tuple.type = converted %tuple.elem1, %empty_tuple.loc7_24.2 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- ref_expr.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (%empty_tuple.type, %empty_tuple.type) [concrete]
// CHECK:STDOUT:   %pattern_type.5b8: type = pattern_type %tuple.type [concrete]
// CHECK:STDOUT:   %tuple.elem0: ref %empty_tuple.type = tuple_access file.%t.var, element0 [concrete]
// CHECK:STDOUT:   %tuple.elem1: ref %empty_tuple.type = tuple_access file.%t.var, element1 [concrete]
// CHECK:STDOUT:   %pattern_type.cb1: type = pattern_type %empty_tuple.type [concrete]
// CHECK:STDOUT:   %a.patt: %pattern_type.cb1 = value_binding_pattern a [concrete]
// CHECK:STDOUT:   %b.patt: %pattern_type.cb1 = value_binding_pattern b [concrete]
// CHECK:STDOUT:   %.045: %pattern_type.5b8 = tuple_pattern (%a.patt, %b.patt) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %.loc6_10.1: type = splice_block %.loc6_10.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:     %.loc6_10.2: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc6_10.3: type = converted %.loc6_10.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: %empty_tuple.type = wrapper_binding a, @__global_init.%.loc6_22.1
// CHECK:STDOUT:   %.loc6_17.1: type = splice_block %.loc6_17.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:     %.loc6_17.2: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc6_17.3: type = converted %.loc6_17.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b: %empty_tuple.type = wrapper_binding b, @__global_init.%.loc6_22.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.cb1 = value_binding_pattern a [concrete = constants.%a.patt]
// CHECK:STDOUT:     %b.patt: %pattern_type.cb1 = value_binding_pattern b [concrete = constants.%b.patt]
// CHECK:STDOUT:     %.loc6_18: %pattern_type.5b8 = tuple_pattern (%a.patt, %b.patt) [concrete = constants.%.045]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %t.ref: ref %tuple.type = name_ref t, file.%t [concrete = file.%t.var]
// CHECK:STDOUT:   %tuple.elem0.loc6: ref %empty_tuple.type = tuple_access %t.ref, element0 [concrete = constants.%tuple.elem0]
// CHECK:STDOUT:   %tuple.elem1.loc6: ref %empty_tuple.type = tuple_access %t.ref, element1 [concrete = constants.%tuple.elem1]
// CHECK:STDOUT:   %empty_tuple.loc6_22.1: %empty_tuple.type = tuple_value () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc6_22.1: %empty_tuple.type = converted %tuple.elem0.loc6, %empty_tuple.loc6_22.1 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %empty_tuple.loc6_22.2: %empty_tuple.type = tuple_value () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc6_22.2: %empty_tuple.type = converted %tuple.elem1.loc6, %empty_tuple.loc6_22.2 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- grouping_parens.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %empty_tuple.type [concrete]
// CHECK:STDOUT:   %a.patt: %pattern_type = value_binding_pattern a [concrete]
// CHECK:STDOUT:   %b.patt: %pattern_type = value_binding_pattern b [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %.loc5_10.1: type = splice_block %.loc5_10.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:     %.loc5_10.2: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc5_10.3: type = converted %.loc5_10.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: %empty_tuple.type = wrapper_binding a, @__global_init.%.loc5_16.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type = value_binding_pattern a [concrete = constants.%a.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc6_12.1: type = splice_block %.loc6_12.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:     %.loc6_12.2: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc6_12.3: type = converted %.loc6_12.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b: %empty_tuple.type = wrapper_binding b, @__global_init.%.loc6_20.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %pattern_type = value_binding_pattern b [concrete = constants.%b.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc5_16.1: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %empty_tuple.loc5: %empty_tuple.type = tuple_value () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc5_16.2: %empty_tuple.type = converted %.loc5_16.1, %empty_tuple.loc5 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc6_20.1: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %empty_tuple.loc6: %empty_tuple.type = tuple_value () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc6_20.2: %empty_tuple.type = converted %.loc6_20.1, %empty_tuple.loc6 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- one_tuple.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %pattern_type.cb1: type = pattern_type %empty_tuple.type [concrete]
// CHECK:STDOUT:   %a.patt: %pattern_type.cb1 = value_binding_pattern a [concrete]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (%empty_tuple.type) [concrete]
// CHECK:STDOUT:   %pattern_type.559: type = pattern_type %tuple.type [concrete]
// CHECK:STDOUT:   %.62f: %pattern_type.559 = tuple_pattern (%a.patt) [concrete]
// CHECK:STDOUT:   %tuple: %tuple.type = tuple_value (%empty_tuple) [concrete]
// CHECK:STDOUT:   %b.patt: %pattern_type.cb1 = value_binding_pattern b [concrete]
// CHECK:STDOUT:   %.00b: %pattern_type.559 = tuple_pattern (%b.patt) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %.loc5_10.1: type = splice_block %.loc5_10.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:     %.loc5_10.2: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc5_10.3: type = converted %.loc5_10.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: %empty_tuple.type = wrapper_binding a, @__global_init.%.loc5_18.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.cb1 = value_binding_pattern a [concrete = constants.%a.patt]
// CHECK:STDOUT:     %.loc5_12: %pattern_type.559 = tuple_pattern (%a.patt) [concrete = constants.%.62f]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc6_12.1: type = splice_block %.loc6_12.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:     %.loc6_12.2: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc6_12.3: type = converted %.loc6_12.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b: %empty_tuple.type = wrapper_binding b, @__global_init.%.loc6_22.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %pattern_type.cb1 = value_binding_pattern b [concrete = constants.%b.patt]
// CHECK:STDOUT:     %.loc6_15: %pattern_type.559 = tuple_pattern (%b.patt) [concrete = constants.%.00b]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc5_18.1: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc5_20: %tuple.type = tuple_literal (%.loc5_18.1) [concrete = constants.%tuple]
// CHECK:STDOUT:   %empty_tuple.loc5: %empty_tuple.type = tuple_value () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc5_18.2: %empty_tuple.type = converted %.loc5_18.1, %empty_tuple.loc5 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc6_22.1: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc6_24: %tuple.type = tuple_literal (%.loc6_22.1) [concrete = constants.%tuple]
// CHECK:STDOUT:   %empty_tuple.loc6: %empty_tuple.type = tuple_value () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc6_22.2: %empty_tuple.type = converted %.loc6_22.1, %empty_tuple.loc6 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_in_class_var.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %C.elem: type = unbound_element_type %C, %empty_tuple.type [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %field_decl.loc13_9: %C.elem = field_decl a, element0, %.loc13_12.2 in [concrete] {
// CHECK:STDOUT:     %.loc13_12.1: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc13_12.2: type = converted %.loc13_12.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %field_decl.loc13_16: %C.elem = field_decl b, element1, %.loc13_19.2 in [concrete] {
// CHECK:STDOUT:     %.loc13_19.1: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc13_19.2: type = converted %.loc13_19.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT:   .a = %field_decl.loc13_9
// CHECK:STDOUT:   .b = %field_decl.loc13_16
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_empty_in_class_var.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- in_static_class_var.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %pattern_type.cb1: type = pattern_type %empty_tuple.type [concrete]
// CHECK:STDOUT:   %a.patt: %pattern_type.cb1 = ref_binding_pattern a [concrete]
// CHECK:STDOUT:   %b.patt: %pattern_type.cb1 = ref_binding_pattern b [concrete]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (%empty_tuple.type, %empty_tuple.type) [concrete]
// CHECK:STDOUT:   %pattern_type.5b8: type = pattern_type %tuple.type [concrete]
// CHECK:STDOUT:   %.009: %pattern_type.5b8 = tuple_pattern (%a.patt, %b.patt) [concrete]
// CHECK:STDOUT:   %.var_patt: %pattern_type.5b8 = var_pattern %.009 [concrete]
// CHECK:STDOUT:   %tuple: %tuple.type = tuple_value (%empty_tuple, %empty_tuple) [concrete]
// CHECK:STDOUT:   %tuple.elem0: ref %empty_tuple.type = tuple_access @C.%.var, element0 [concrete]
// CHECK:STDOUT:   %tuple.elem1: ref %empty_tuple.type = tuple_access @C.%.var, element1 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %.var: ref %tuple.type = var_storage %.var_patt [concrete]
// CHECK:STDOUT:   %.loc6_19.1: type = splice_block %.loc6_19.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:     %.loc6_19.2: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc6_19.3: type = converted %.loc6_19.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: ref %empty_tuple.type = wrapper_binding a, @__global_init.%tuple.elem0.loc6_10 [concrete = constants.%tuple.elem0]
// CHECK:STDOUT:   %.loc6_26.1: type = splice_block %.loc6_26.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:     %.loc6_26.2: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc6_26.3: type = converted %.loc6_26.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b: ref %empty_tuple.type = wrapper_binding b, @__global_init.%tuple.elem1.loc6_10 [concrete = constants.%tuple.elem1]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.cb1 = ref_binding_pattern a [concrete = constants.%a.patt]
// CHECK:STDOUT:     %b.patt: %pattern_type.cb1 = ref_binding_pattern b [concrete = constants.%b.patt]
// CHECK:STDOUT:     %.loc6_27: %pattern_type.5b8 = tuple_pattern (%a.patt, %b.patt) [concrete = constants.%.009]
// CHECK:STDOUT:     %.var_patt: %pattern_type.5b8 = var_pattern %.loc6_27 [concrete = constants.%.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT:   .a = %a
// CHECK:STDOUT:   .b = %b
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc6_33.1: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc6_37.1: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc6_38.1: %tuple.type = tuple_literal (%.loc6_33.1, %.loc6_37.1) [concrete = constants.%tuple]
// CHECK:STDOUT:   %tuple.elem0.loc6_38: ref %empty_tuple.type = tuple_access @C.%.var, element0 [concrete = constants.%tuple.elem0]
// CHECK:STDOUT:   %.loc6_33.2: init %empty_tuple.type to %tuple.elem0.loc6_38 = tuple_init () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc6_38.2: init %empty_tuple.type = converted %.loc6_33.1, %.loc6_33.2 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc6_38.3: init %empty_tuple.type to %tuple.elem0.loc6_38 = in_place_init %.loc6_38.2 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %tuple.elem1.loc6_38: ref %empty_tuple.type = tuple_access @C.%.var, element1 [concrete = constants.%tuple.elem1]
// CHECK:STDOUT:   %.loc6_37.2: init %empty_tuple.type to %tuple.elem1.loc6_38 = tuple_init () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc6_38.4: init %empty_tuple.type = converted %.loc6_37.1, %.loc6_37.2 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc6_38.5: init %empty_tuple.type to %tuple.elem1.loc6_38 = in_place_init %.loc6_38.4 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc6_38.6: init %tuple.type to @C.%.var = tuple_init (%.loc6_38.3, %.loc6_38.5) [concrete = constants.%tuple]
// CHECK:STDOUT:   %.loc6_10: init %tuple.type = converted %.loc6_38.1, %.loc6_38.6 [concrete = constants.%tuple]
// CHECK:STDOUT:   assign @C.%.var, %.loc6_10
// CHECK:STDOUT:   %tuple.elem0.loc6_10: ref %empty_tuple.type = tuple_access @C.%.var, element0 [concrete = constants.%tuple.elem0]
// CHECK:STDOUT:   %tuple.elem1.loc6_10: ref %empty_tuple.type = tuple_access @C.%.var, element1 [concrete = constants.%tuple.elem1]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- in_class_let.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %pattern_type.cb1: type = pattern_type %empty_tuple.type [concrete]
// CHECK:STDOUT:   %a.patt: %pattern_type.cb1 = value_binding_pattern a [concrete]
// CHECK:STDOUT:   %b.patt: %pattern_type.cb1 = value_binding_pattern b [concrete]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (%empty_tuple.type, %empty_tuple.type) [concrete]
// CHECK:STDOUT:   %pattern_type.5b8: type = pattern_type %tuple.type [concrete]
// CHECK:STDOUT:   %.737: %pattern_type.5b8 = tuple_pattern (%a.patt, %b.patt) [concrete]
// CHECK:STDOUT:   %tuple: %tuple.type = tuple_value (%empty_tuple, %empty_tuple) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %.loc6_12.1: type = splice_block %.loc6_12.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:     %.loc6_12.2: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc6_12.3: type = converted %.loc6_12.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: %empty_tuple.type = wrapper_binding a, @__global_init.%.loc6_26.2
// CHECK:STDOUT:   %.loc6_19.1: type = splice_block %.loc6_19.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:     %.loc6_19.2: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc6_19.3: type = converted %.loc6_19.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b: %empty_tuple.type = wrapper_binding b, @__global_init.%.loc6_30.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.cb1 = value_binding_pattern a [concrete = constants.%a.patt]
// CHECK:STDOUT:     %b.patt: %pattern_type.cb1 = value_binding_pattern b [concrete = constants.%b.patt]
// CHECK:STDOUT:     %.loc6_20: %pattern_type.5b8 = tuple_pattern (%a.patt, %b.patt) [concrete = constants.%.737]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT:   .a = %a
// CHECK:STDOUT:   .b = %b
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc6_26.1: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc6_30.1: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc6_31: %tuple.type = tuple_literal (%.loc6_26.1, %.loc6_30.1) [concrete = constants.%tuple]
// CHECK:STDOUT:   %empty_tuple.loc6_26: %empty_tuple.type = tuple_value () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc6_26.2: %empty_tuple.type = converted %.loc6_26.1, %empty_tuple.loc6_26 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %empty_tuple.loc6_30: %empty_tuple.type = tuple_value () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc6_30.2: %empty_tuple.type = converted %.loc6_30.1, %empty_tuple.loc6_30 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- empty_in_class_let.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %empty_tuple.type [concrete]
// CHECK:STDOUT:   %.e84: %pattern_type = tuple_pattern () [concrete]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %.loc6: %pattern_type = tuple_pattern () [concrete = constants.%.e84]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc6: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
