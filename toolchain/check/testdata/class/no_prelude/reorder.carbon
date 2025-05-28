// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// EXTRA-ARGS: --dump-sem-ir-ranges=only
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/no_prelude/reorder.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/no_prelude/reorder.carbon

// --- reorder.carbon
library "[[@TEST_NAME]]";

class Class {
  fn G() -> () {
    //@dump-sem-ir-begin
    return Class.F();
    //@dump-sem-ir-end
  }

  fn F() -> () {
    return ();
  }
}

// --- reorder_qualified.carbon
library "[[@TEST_NAME]]";

class A {
  class B {
    class C;

    fn BF();
    var b: ();
  }

  class B.C {
    class D {
      fn F();

      fn DF();
      var d: ();
    }

    fn D.DF() {
      //@dump-sem-ir-begin
      // A, B, C, and D are complete here.
      var a: A = {.a = ()};
      var b: B = {.b = ()};
      var c: C = {.c = ()};
      var d: D = {.d = ()};

      // Unqualified lookup looks in all of them.
      AF();
      BF();
      CF();
      DF();
      //@dump-sem-ir-end
    }

    fn CF();
    var c: ();
  }

  fn AF();
  var a: ();
}

// CHECK:STDOUT: --- reorder.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Class: type = class_type @Class [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Class {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Class
// CHECK:STDOUT:   .G = %G.decl
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   .Class = <poisoned>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() -> %empty_tuple.type {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Class.ref: type = name_ref Class, file.%Class.decl [concrete = constants.%Class]
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, @Class.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:   %F.call: init %empty_tuple.type = call %F.ref()
// CHECK:STDOUT:   %.loc6_20.1: ref %empty_tuple.type = temporary_storage
// CHECK:STDOUT:   %.loc6_20.2: ref %empty_tuple.type = temporary %.loc6_20.1, %F.call
// CHECK:STDOUT:   %tuple: %empty_tuple.type = tuple_value () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc6_21: %empty_tuple.type = converted %F.call, %tuple [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   return %.loc6_21
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- reorder_qualified.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %A: type = class_type @A [concrete]
// CHECK:STDOUT:   %B: type = class_type @B [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %BF.type: type = fn_type @BF [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %BF: %BF.type = struct_value () [concrete]
// CHECK:STDOUT:   %struct_type.b: type = struct_type {.b: %empty_tuple.type} [concrete]
// CHECK:STDOUT:   %D: type = class_type @D [concrete]
// CHECK:STDOUT:   %DF.type: type = fn_type @DF [concrete]
// CHECK:STDOUT:   %DF: %DF.type = struct_value () [concrete]
// CHECK:STDOUT:   %struct_type.d: type = struct_type {.d: %empty_tuple.type} [concrete]
// CHECK:STDOUT:   %CF.type: type = fn_type @CF [concrete]
// CHECK:STDOUT:   %CF: %CF.type = struct_value () [concrete]
// CHECK:STDOUT:   %struct_type.c: type = struct_type {.c: %empty_tuple.type} [concrete]
// CHECK:STDOUT:   %AF.type: type = fn_type @AF [concrete]
// CHECK:STDOUT:   %AF: %AF.type = struct_value () [concrete]
// CHECK:STDOUT:   %struct_type.a: type = struct_type {.a: %empty_tuple.type} [concrete]
// CHECK:STDOUT:   %pattern_type.c10: type = pattern_type %A [concrete]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %A.val: %A = struct_value (%empty_tuple) [concrete]
// CHECK:STDOUT:   %pattern_type.98e: type = pattern_type %B [concrete]
// CHECK:STDOUT:   %B.val: %B = struct_value (%empty_tuple) [concrete]
// CHECK:STDOUT:   %pattern_type.6d8: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %C.val: %C = struct_value (%empty_tuple) [concrete]
// CHECK:STDOUT:   %pattern_type.d58: type = pattern_type %D [concrete]
// CHECK:STDOUT:   %D.val: %D = struct_value (%empty_tuple) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @A {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%A
// CHECK:STDOUT:   .B = %B.decl
// CHECK:STDOUT:   .AF = %AF.decl
// CHECK:STDOUT:   .a = %.loc40_8
// CHECK:STDOUT:   .A = <poisoned>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT:   .D = %D.decl
// CHECK:STDOUT:   .CF = %CF.decl
// CHECK:STDOUT:   .c = %.loc36_10
// CHECK:STDOUT:   .A = <poisoned>
// CHECK:STDOUT:   .B = <poisoned>
// CHECK:STDOUT:   .C = <poisoned>
// CHECK:STDOUT:   .AF = <poisoned>
// CHECK:STDOUT:   .BF = <poisoned>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @DF() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.c10 = binding_pattern a [concrete]
// CHECK:STDOUT:     %a.var_patt: %pattern_type.c10 = var_pattern %a.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.var: ref %A = var %a.var_patt
// CHECK:STDOUT:   %.loc22_25.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %.loc22_26.1: %struct_type.a = struct_literal (%.loc22_25.1)
// CHECK:STDOUT:   %.loc22_26.2: ref %empty_tuple.type = class_element_access %a.var, element0
// CHECK:STDOUT:   %.loc22_25.2: init %empty_tuple.type = tuple_init () to %.loc22_26.2 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc22_26.3: init %empty_tuple.type = converted %.loc22_25.1, %.loc22_25.2 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc22_26.4: init %A = class_init (%.loc22_26.3), %a.var [concrete = constants.%A.val]
// CHECK:STDOUT:   %.loc22_7: init %A = converted %.loc22_26.1, %.loc22_26.4 [concrete = constants.%A.val]
// CHECK:STDOUT:   assign %a.var, %.loc22_7
// CHECK:STDOUT:   %A.ref: type = name_ref A, file.%A.decl [concrete = constants.%A]
// CHECK:STDOUT:   %a: ref %A = bind_name a, %a.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %pattern_type.98e = binding_pattern b [concrete]
// CHECK:STDOUT:     %b.var_patt: %pattern_type.98e = var_pattern %b.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b.var: ref %B = var %b.var_patt
// CHECK:STDOUT:   %.loc23_25.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %.loc23_26.1: %struct_type.b = struct_literal (%.loc23_25.1)
// CHECK:STDOUT:   %.loc23_26.2: ref %empty_tuple.type = class_element_access %b.var, element0
// CHECK:STDOUT:   %.loc23_25.2: init %empty_tuple.type = tuple_init () to %.loc23_26.2 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc23_26.3: init %empty_tuple.type = converted %.loc23_25.1, %.loc23_25.2 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc23_26.4: init %B = class_init (%.loc23_26.3), %b.var [concrete = constants.%B.val]
// CHECK:STDOUT:   %.loc23_7: init %B = converted %.loc23_26.1, %.loc23_26.4 [concrete = constants.%B.val]
// CHECK:STDOUT:   assign %b.var, %.loc23_7
// CHECK:STDOUT:   %B.ref: type = name_ref B, @A.%B.decl [concrete = constants.%B]
// CHECK:STDOUT:   %b: ref %B = bind_name b, %b.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c.patt: %pattern_type.6d8 = binding_pattern c [concrete]
// CHECK:STDOUT:     %c.var_patt: %pattern_type.6d8 = var_pattern %c.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c.var: ref %C = var %c.var_patt
// CHECK:STDOUT:   %.loc24_25.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %.loc24_26.1: %struct_type.c = struct_literal (%.loc24_25.1)
// CHECK:STDOUT:   %.loc24_26.2: ref %empty_tuple.type = class_element_access %c.var, element0
// CHECK:STDOUT:   %.loc24_25.2: init %empty_tuple.type = tuple_init () to %.loc24_26.2 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc24_26.3: init %empty_tuple.type = converted %.loc24_25.1, %.loc24_25.2 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc24_26.4: init %C = class_init (%.loc24_26.3), %c.var [concrete = constants.%C.val]
// CHECK:STDOUT:   %.loc24_7: init %C = converted %.loc24_26.1, %.loc24_26.4 [concrete = constants.%C.val]
// CHECK:STDOUT:   assign %c.var, %.loc24_7
// CHECK:STDOUT:   %C.ref: type = name_ref C, @B.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %c: ref %C = bind_name c, %c.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %d.patt: %pattern_type.d58 = binding_pattern d [concrete]
// CHECK:STDOUT:     %d.var_patt: %pattern_type.d58 = var_pattern %d.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %d.var: ref %D = var %d.var_patt
// CHECK:STDOUT:   %.loc25_25.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %.loc25_26.1: %struct_type.d = struct_literal (%.loc25_25.1)
// CHECK:STDOUT:   %.loc25_26.2: ref %empty_tuple.type = class_element_access %d.var, element0
// CHECK:STDOUT:   %.loc25_25.2: init %empty_tuple.type = tuple_init () to %.loc25_26.2 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc25_26.3: init %empty_tuple.type = converted %.loc25_25.1, %.loc25_25.2 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc25_26.4: init %D = class_init (%.loc25_26.3), %d.var [concrete = constants.%D.val]
// CHECK:STDOUT:   %.loc25_7: init %D = converted %.loc25_26.1, %.loc25_26.4 [concrete = constants.%D.val]
// CHECK:STDOUT:   assign %d.var, %.loc25_7
// CHECK:STDOUT:   %D.ref: type = name_ref D, @C.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:   %d: ref %D = bind_name d, %d.var
// CHECK:STDOUT:   %AF.ref: %AF.type = name_ref AF, @A.%AF.decl [concrete = constants.%AF]
// CHECK:STDOUT:   %AF.call: init %empty_tuple.type = call %AF.ref()
// CHECK:STDOUT:   %BF.ref: %BF.type = name_ref BF, @B.%BF.decl [concrete = constants.%BF]
// CHECK:STDOUT:   %BF.call: init %empty_tuple.type = call %BF.ref()
// CHECK:STDOUT:   %CF.ref: %CF.type = name_ref CF, @C.%CF.decl [concrete = constants.%CF]
// CHECK:STDOUT:   %CF.call: init %empty_tuple.type = call %CF.ref()
// CHECK:STDOUT:   %DF.ref: %DF.type = name_ref DF, @D.%DF.decl [concrete = constants.%DF]
// CHECK:STDOUT:   %DF.call: init %empty_tuple.type = call %DF.ref()
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
