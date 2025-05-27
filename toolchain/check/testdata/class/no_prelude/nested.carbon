// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// EXTRA-ARGS: --dump-sem-ir-ranges=only
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/no_prelude/nested.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/no_prelude/nested.carbon

// --- nested.carbon
library "[[@TEST_NAME]]";

class Outer {
  fn F() {
    // Outer and Inner are both complete here.
    var o: Outer;
    var i: Inner;
  }

  class Inner {
    var pi: Self*;
    var po: Outer*;
    var qi: Inner*;

    fn G() {
      // Outer and Inner are both complete here.
      var o: Outer;
      var i: Inner;
    }
  }

  fn H() {
    // Outer and Inner are both complete here.
    var o: Outer;
    var i: Inner;
  }

  var po: Self*;
  var qo: Outer*;
  var pi: Inner*;
}

fn F(a: Outer*) {
  //@dump-sem-ir-begin
  let b: Outer.Inner* = (*a).pi;

  a->po = a;
  a->qo = a;
  a->pi = a->pi;
  b->po = a;
  b->pi = a->pi;
  b->qi = a->pi;
  //@dump-sem-ir-end
}

// --- nested_name.carbon
library "[[@TEST_NAME]]";

class Outer {
  //@dump-sem-ir-begin
  class Inner {
    var n: ();
  }
  //@dump-sem-ir-end
}

fn F(oi: Outer.Inner) -> () {
  //@dump-sem-ir-begin
  return oi.n;
  //@dump-sem-ir-end
}

fn G(o: Outer) {
  var i: o.Inner;
}

// CHECK:STDOUT: --- nested.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Outer: type = class_type @Outer [concrete]
// CHECK:STDOUT:   %Inner: type = class_type @Inner [concrete]
// CHECK:STDOUT:   %ptr.36a: type = ptr_type %Inner [concrete]
// CHECK:STDOUT:   %Inner.elem.640: type = unbound_element_type %Inner, %ptr.36a [concrete]
// CHECK:STDOUT:   %ptr.5df: type = ptr_type %Outer [concrete]
// CHECK:STDOUT:   %Inner.elem.c30: type = unbound_element_type %Inner, %ptr.5df [concrete]
// CHECK:STDOUT:   %Outer.elem.a16: type = unbound_element_type %Outer, %ptr.5df [concrete]
// CHECK:STDOUT:   %Outer.elem.fe9: type = unbound_element_type %Outer, %ptr.36a [concrete]
// CHECK:STDOUT:   %pattern_type.27f: type = pattern_type %ptr.36a [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.2(%a.param: %ptr.5df) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %pattern_type.27f = binding_pattern b [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.ref.loc35: %ptr.5df = name_ref a, %a
// CHECK:STDOUT:   %.loc35_26: ref %Outer = deref %a.ref.loc35
// CHECK:STDOUT:   %pi.ref.loc35: %Outer.elem.fe9 = name_ref pi, @Outer.%.loc30 [concrete = @Outer.%.loc30]
// CHECK:STDOUT:   %.loc35_29: ref %ptr.36a = class_element_access %.loc35_26, element2
// CHECK:STDOUT:   %.loc35_21: type = splice_block %ptr.loc35 [concrete = constants.%ptr.36a] {
// CHECK:STDOUT:     %Outer.ref.loc35: type = name_ref Outer, file.%Outer.decl [concrete = constants.%Outer]
// CHECK:STDOUT:     %Inner.ref: type = name_ref Inner, @Outer.%Inner.decl [concrete = constants.%Inner]
// CHECK:STDOUT:     %ptr.loc35: type = ptr_type %Inner.ref [concrete = constants.%ptr.36a]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b: ref %ptr.36a = bind_name b, %.loc35_29
// CHECK:STDOUT:   %a.ref.loc37_3: %ptr.5df = name_ref a, %a
// CHECK:STDOUT:   %.loc37_4.1: ref %Outer = deref %a.ref.loc37_3
// CHECK:STDOUT:   %po.ref.loc37: %Outer.elem.a16 = name_ref po, @Outer.%.loc28 [concrete = @Outer.%.loc28]
// CHECK:STDOUT:   %.loc37_4.2: ref %ptr.5df = class_element_access %.loc37_4.1, element0
// CHECK:STDOUT:   %a.ref.loc37_11: %ptr.5df = name_ref a, %a
// CHECK:STDOUT:   assign %.loc37_4.2, %a.ref.loc37_11
// CHECK:STDOUT:   %a.ref.loc38_3: %ptr.5df = name_ref a, %a
// CHECK:STDOUT:   %.loc38_4.1: ref %Outer = deref %a.ref.loc38_3
// CHECK:STDOUT:   %qo.ref: %Outer.elem.a16 = name_ref qo, @Outer.%.loc29 [concrete = @Outer.%.loc29]
// CHECK:STDOUT:   %.loc38_4.2: ref %ptr.5df = class_element_access %.loc38_4.1, element1
// CHECK:STDOUT:   %a.ref.loc38_11: %ptr.5df = name_ref a, %a
// CHECK:STDOUT:   assign %.loc38_4.2, %a.ref.loc38_11
// CHECK:STDOUT:   %a.ref.loc39_3: %ptr.5df = name_ref a, %a
// CHECK:STDOUT:   %.loc39_4.1: ref %Outer = deref %a.ref.loc39_3
// CHECK:STDOUT:   %pi.ref.loc39_4: %Outer.elem.fe9 = name_ref pi, @Outer.%.loc30 [concrete = @Outer.%.loc30]
// CHECK:STDOUT:   %.loc39_4.2: ref %ptr.36a = class_element_access %.loc39_4.1, element2
// CHECK:STDOUT:   %a.ref.loc39_11: %ptr.5df = name_ref a, %a
// CHECK:STDOUT:   %.loc39_12.1: ref %Outer = deref %a.ref.loc39_11
// CHECK:STDOUT:   %pi.ref.loc39_12: %Outer.elem.fe9 = name_ref pi, @Outer.%.loc30 [concrete = @Outer.%.loc30]
// CHECK:STDOUT:   %.loc39_12.2: ref %ptr.36a = class_element_access %.loc39_12.1, element2
// CHECK:STDOUT:   %.loc39_12.3: %ptr.36a = bind_value %.loc39_12.2
// CHECK:STDOUT:   assign %.loc39_4.2, %.loc39_12.3
// CHECK:STDOUT:   %b.ref.loc40: ref %ptr.36a = name_ref b, %b
// CHECK:STDOUT:   %.loc40_3: %ptr.36a = bind_value %b.ref.loc40
// CHECK:STDOUT:   %.loc40_4.1: ref %Inner = deref %.loc40_3
// CHECK:STDOUT:   %po.ref.loc40: %Inner.elem.c30 = name_ref po, @Inner.%.loc12 [concrete = @Inner.%.loc12]
// CHECK:STDOUT:   %.loc40_4.2: ref %ptr.5df = class_element_access %.loc40_4.1, element1
// CHECK:STDOUT:   %a.ref.loc40: %ptr.5df = name_ref a, %a
// CHECK:STDOUT:   assign %.loc40_4.2, %a.ref.loc40
// CHECK:STDOUT:   %b.ref.loc41: ref %ptr.36a = name_ref b, %b
// CHECK:STDOUT:   %.loc41_3: %ptr.36a = bind_value %b.ref.loc41
// CHECK:STDOUT:   %.loc41_4.1: ref %Inner = deref %.loc41_3
// CHECK:STDOUT:   %pi.ref.loc41_4: %Inner.elem.640 = name_ref pi, @Inner.%.loc11 [concrete = @Inner.%.loc11]
// CHECK:STDOUT:   %.loc41_4.2: ref %ptr.36a = class_element_access %.loc41_4.1, element0
// CHECK:STDOUT:   %a.ref.loc41: %ptr.5df = name_ref a, %a
// CHECK:STDOUT:   %.loc41_12.1: ref %Outer = deref %a.ref.loc41
// CHECK:STDOUT:   %pi.ref.loc41_12: %Outer.elem.fe9 = name_ref pi, @Outer.%.loc30 [concrete = @Outer.%.loc30]
// CHECK:STDOUT:   %.loc41_12.2: ref %ptr.36a = class_element_access %.loc41_12.1, element2
// CHECK:STDOUT:   %.loc41_12.3: %ptr.36a = bind_value %.loc41_12.2
// CHECK:STDOUT:   assign %.loc41_4.2, %.loc41_12.3
// CHECK:STDOUT:   %b.ref.loc42: ref %ptr.36a = name_ref b, %b
// CHECK:STDOUT:   %.loc42_3: %ptr.36a = bind_value %b.ref.loc42
// CHECK:STDOUT:   %.loc42_4.1: ref %Inner = deref %.loc42_3
// CHECK:STDOUT:   %qi.ref: %Inner.elem.640 = name_ref qi, @Inner.%.loc13 [concrete = @Inner.%.loc13]
// CHECK:STDOUT:   %.loc42_4.2: ref %ptr.36a = class_element_access %.loc42_4.1, element2
// CHECK:STDOUT:   %a.ref.loc42: %ptr.5df = name_ref a, %a
// CHECK:STDOUT:   %.loc42_12.1: ref %Outer = deref %a.ref.loc42
// CHECK:STDOUT:   %pi.ref.loc42: %Outer.elem.fe9 = name_ref pi, @Outer.%.loc30 [concrete = @Outer.%.loc30]
// CHECK:STDOUT:   %.loc42_12.2: ref %ptr.36a = class_element_access %.loc42_12.1, element2
// CHECK:STDOUT:   %.loc42_12.3: %ptr.36a = bind_value %.loc42_12.2
// CHECK:STDOUT:   assign %.loc42_4.2, %.loc42_12.3
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- nested_name.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Outer: type = class_type @Outer [concrete]
// CHECK:STDOUT:   %Inner: type = class_type @Inner [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Inner.elem: type = unbound_element_type %Inner, %empty_tuple.type [concrete]
// CHECK:STDOUT:   %struct_type.n: type = struct_type {.n: %empty_tuple.type} [concrete]
// CHECK:STDOUT:   %complete_type.880: <witness> = complete_type_witness %struct_type.n [concrete]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Outer {
// CHECK:STDOUT:   %Inner.decl: type = class_decl @Inner [concrete = constants.%Inner] {} {}
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Outer
// CHECK:STDOUT:   .Inner = %Inner.decl
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Inner {
// CHECK:STDOUT:   %.loc6_13.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %.loc6_13.2: type = converted %.loc6_13.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   %.loc6_10: %Inner.elem = field_decl n, element0 [concrete]
// CHECK:STDOUT:   %struct_type.n: type = struct_type {.n: %empty_tuple.type} [concrete = constants.%struct_type.n]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.n [concrete = constants.%complete_type.880]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Inner
// CHECK:STDOUT:   .n = %.loc6_10
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%oi.param: %Inner) -> %empty_tuple.type {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %oi.ref: %Inner = name_ref oi, %oi
// CHECK:STDOUT:   %n.ref: %Inner.elem = name_ref n, @Inner.%.loc6_10 [concrete = @Inner.%.loc6_10]
// CHECK:STDOUT:   %.loc13_12.1: ref %empty_tuple.type = class_element_access %oi.ref, element0
// CHECK:STDOUT:   %tuple: %empty_tuple.type = tuple_value () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc13_12.2: %empty_tuple.type = converted %.loc13_12.1, %tuple [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   return %.loc13_12.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
