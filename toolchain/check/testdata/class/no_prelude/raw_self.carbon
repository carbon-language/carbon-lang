// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// EXTRA-ARGS: --dump-sem-ir-ranges=only
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/no_prelude/raw_self.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/no_prelude/raw_self.carbon

// --- raw_self.carbon
library "[[@TEST_NAME]]";

class Class {
  fn F[addr self: Self*](r#self: ());
  fn G[self: Self](r#self: ()) -> ((), ());
  var n: ();
}

fn Class.F[addr self: Self*](r#self: ()) {
  //@dump-sem-ir-begin
  (*self).n = r#self;
  //@dump-sem-ir-end
}

fn Class.G[self: Self](r#self: ()) -> ((), ()) {
  //@dump-sem-ir-begin
  return (self.n, r#self);
  //@dump-sem-ir-end
}

// --- raw_self_type.carbon
library "[[@TEST_NAME]]";

class Class {
  fn F() {
    //@dump-sem-ir-begin
    var r#Self: Self*;
    //@dump-sem-ir-end
    var p: Self* = r#Self;
  }
}

class MemberNamedSelf {
  class r#Self {}

  fn F(x: Self, y: r#Self);
}

fn MemberNamedSelf.F(x: Self, y: r#Self) {}

// CHECK:STDOUT: --- raw_self.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Class: type = class_type @Class [concrete]
// CHECK:STDOUT:   %ptr.e71: type = ptr_type %Class [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (%empty_tuple.type, %empty_tuple.type) [concrete]
// CHECK:STDOUT:   %Class.elem: type = unbound_element_type %Class, %empty_tuple.type [concrete]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %tuple: %tuple.type = tuple_value (%empty_tuple, %empty_tuple) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%self.param.loc9_21: %ptr.e71, %self.param.loc9_36: %empty_tuple.type) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %self.ref.loc11_5: %ptr.e71 = name_ref self, %self.loc9_17
// CHECK:STDOUT:   %.loc11_4: ref %Class = deref %self.ref.loc11_5
// CHECK:STDOUT:   %n.ref: %Class.elem = name_ref n, @Class.%.loc6_8 [concrete = @Class.%.loc6_8]
// CHECK:STDOUT:   %.loc11_10: ref %empty_tuple.type = class_element_access %.loc11_4, element0
// CHECK:STDOUT:   %self.ref.loc11_15: %empty_tuple.type = name_ref r#self, %self.loc9_30
// CHECK:STDOUT:   %.loc11_15: init %empty_tuple.type = tuple_init () to %.loc11_10 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc11_13: init %empty_tuple.type = converted %self.ref.loc11_15, %.loc11_15 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   assign %.loc11_10, %.loc11_13
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G(%self.param.loc15_16: %Class, %self.param.loc15_30: %empty_tuple.type) -> %return.param.loc15: %tuple.type {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %self.ref.loc17_11: %Class = name_ref self, %self.loc15_12
// CHECK:STDOUT:   %n.ref: %Class.elem = name_ref n, @Class.%.loc6_8 [concrete = @Class.%.loc6_8]
// CHECK:STDOUT:   %.loc17_15.1: ref %empty_tuple.type = class_element_access %self.ref.loc17_11, element0
// CHECK:STDOUT:   %tuple: %empty_tuple.type = tuple_value () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc17_15.2: %empty_tuple.type = converted %.loc17_15.1, %tuple [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %self.ref.loc17_19: %empty_tuple.type = name_ref r#self, %self.loc15_24
// CHECK:STDOUT:   %.loc17_25.1: %tuple.type = tuple_literal (%.loc17_15.2, %self.ref.loc17_19)
// CHECK:STDOUT:   %tuple.elem0: ref %empty_tuple.type = tuple_access %return.loc15, element0
// CHECK:STDOUT:   %.loc17_15.3: init %empty_tuple.type = tuple_init () to %tuple.elem0 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc17_25.2: init %empty_tuple.type = converted %.loc17_15.2, %.loc17_15.3 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %tuple.elem1: ref %empty_tuple.type = tuple_access %return.loc15, element1
// CHECK:STDOUT:   %.loc17_19: init %empty_tuple.type = tuple_init () to %tuple.elem1 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc17_25.3: init %empty_tuple.type = converted %self.ref.loc17_19, %.loc17_19 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc17_25.4: init %tuple.type = tuple_init (%.loc17_25.2, %.loc17_25.3) to %return.loc15 [concrete = constants.%tuple]
// CHECK:STDOUT:   %.loc17_26: init %tuple.type = converted %.loc17_25.1, %.loc17_25.4 [concrete = constants.%tuple]
// CHECK:STDOUT:   return %.loc17_26 to %return.loc15
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- raw_self_type.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Class: type = class_type @Class [concrete]
// CHECK:STDOUT:   %ptr.e71: type = ptr_type %Class [concrete]
// CHECK:STDOUT:   %pattern_type.796: type = pattern_type %ptr.e71 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Class {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Class
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.1() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %Self.patt: %pattern_type.796 = binding_pattern r#Self [concrete]
// CHECK:STDOUT:     %Self.var_patt: %pattern_type.796 = var_pattern %Self.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Self.var: ref %ptr.e71 = var %Self.var_patt
// CHECK:STDOUT:   %.loc6: type = splice_block %ptr.loc6 [concrete = constants.%ptr.e71] {
// CHECK:STDOUT:     %Self.ref.loc6: type = name_ref Self, constants.%Class [concrete = constants.%Class]
// CHECK:STDOUT:     %ptr.loc6: type = ptr_type %Self.ref.loc6 [concrete = constants.%ptr.e71]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Self: ref %ptr.e71 = bind_name r#Self, %Self.var
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
