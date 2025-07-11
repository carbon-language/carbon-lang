// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/destroy.carbon
// TODO: Add ranges and switch to "--dump-sem-ir-ranges=only".
// EXTRA-ARGS: --dump-sem-ir-ranges=if-present
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/impl/fail_impl_as_scope.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/impl/fail_impl_as_scope.carbon

// --- fail_impl_as_file_scope.carbon
library "[[@TEST_NAME]]";

interface I {}

fn G() {}

// CHECK:STDERR: fail_impl_as_file_scope.carbon:[[@LINE+4]]:6: error: `impl as` can only be used in a class [ImplAsOutsideClass]
// CHECK:STDERR: impl as I {}
// CHECK:STDERR:      ^~
// CHECK:STDERR:
impl as I {}

fn F() { G(); }

// --- fail_impl_as_function_scope.carbon
library "[[@TEST_NAME]]";

interface J {}

fn G() {}

fn F() {
  // CHECK:STDERR: fail_impl_as_function_scope.carbon:[[@LINE+4]]:8: error: `impl as` can only be used in a class [ImplAsOutsideClass]
  // CHECK:STDERR:   impl as J {}
  // CHECK:STDERR:        ^~
  // CHECK:STDERR:
  impl as J {}
  G();
}

// --- fail_impl_as_self_interface.carbon
library "[[@TEST_NAME]]";

interface Z {
  fn Zero();
  fn Method[self: Self]();

   // CHECK:STDERR: fail_impl_as_self_interface.carbon:[[@LINE+4]]:9: error: `impl as` can only be used in a class [ImplAsOutsideClass]
   // CHECK:STDERR:    impl as Z {
   // CHECK:STDERR:         ^~
   // CHECK:STDERR:
   impl as Z {
    fn Zero() {}
    fn Method[self: Self]() {}
  }
}

class Point {
   impl as Z {
    fn Zero() {}
    fn Method[self: Self]() {}
  }
}

fn F() {
  // Even if the `impl` is diagnosed above, we must not add the impl of the
  // interface to itself in a way that allows it to be used during impl lookup,
  // or we end up with infinite impl lookup recursion here.
  Point.(Z.Zero)();
  ({} as Point).(Z.Method)();
}

// --- fail_impl_as_other_interface.carbon

library "[[@TEST_NAME]]";

interface A {
  fn B();
}
interface C {}

fn G() {}

class X {
  impl as A {
    // CHECK:STDERR: fail_impl_as_other_interface.carbon:[[@LINE+4]]:10: error: `impl as` can only be used in a class [ImplAsOutsideClass]
    // CHECK:STDERR:     impl as C {}
    // CHECK:STDERR:          ^~
    // CHECK:STDERR:
    impl as C {}
    fn B() {
      G();
    }
  }
}

// CHECK:STDOUT: --- fail_impl_as_file_scope.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [concrete]
// CHECK:STDOUT:   %Self: %I.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .I = %I.decl
// CHECK:STDOUT:     .G = %G.decl
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %I.decl: type = interface_decl @I [concrete = constants.%I.type] {} {}
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {} {}
// CHECK:STDOUT:   impl_decl @I.impl [concrete] {} {
// CHECK:STDOUT:     %I.ref: type = name_ref I, file.%I.decl [concrete = constants.%I.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I {
// CHECK:STDOUT:   %Self: %I.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @I.impl: <error> as %I.ref {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %G.ref: %G.type = name_ref G, file.%G.decl [concrete = constants.%G]
// CHECK:STDOUT:   %G.call: init %empty_tuple.type = call %G.ref()
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_impl_as_function_scope.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %J.type: type = facet_type <@J> [concrete]
// CHECK:STDOUT:   %Self: %J.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .J = %J.decl
// CHECK:STDOUT:     .G = %G.decl
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %J.decl: type = interface_decl @J [concrete = constants.%J.type] {} {}
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {} {}
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @J {
// CHECK:STDOUT:   %Self: %J.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @J.impl: <error> as %J.ref {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   impl_decl @J.impl [concrete] {} {
// CHECK:STDOUT:     %J.ref: type = name_ref J, file.%J.decl [concrete = constants.%J.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.ref: %G.type = name_ref G, file.%G.decl [concrete = constants.%G]
// CHECK:STDOUT:   %G.call: init %empty_tuple.type = call %G.ref()
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_impl_as_self_interface.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Z.type: type = facet_type <@Z> [concrete]
// CHECK:STDOUT:   %Self.6e6: %Z.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %Zero.type.822: type = fn_type @Zero.1 [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Zero.d14: %Zero.type.822 = struct_value () [concrete]
// CHECK:STDOUT:   %Z.assoc_type: type = assoc_entity_type @Z [concrete]
// CHECK:STDOUT:   %assoc0.534: %Z.assoc_type = assoc_entity element0, @Z.%Zero.decl [concrete]
// CHECK:STDOUT:   %Self.as_type.cd1: type = facet_access_type %Self.6e6 [symbolic]
// CHECK:STDOUT:   %pattern_type.a40: type = pattern_type %Self.as_type.cd1 [symbolic]
// CHECK:STDOUT:   %Method.type.f12: type = fn_type @Method.1 [concrete]
// CHECK:STDOUT:   %Method.724: %Method.type.f12 = struct_value () [concrete]
// CHECK:STDOUT:   %assoc1: %Z.assoc_type = assoc_entity element1, @Z.%Method.decl [concrete]
// CHECK:STDOUT:   %Zero.type.db4: type = fn_type @Zero.2, @Z.impl.6b5(%Self.6e6) [symbolic]
// CHECK:STDOUT:   %Zero.8fb: %Zero.type.db4 = struct_value () [symbolic]
// CHECK:STDOUT:   %Method.type.163: type = fn_type @Method.2, @Z.impl.6b5(%Self.6e6) [symbolic]
// CHECK:STDOUT:   %Method.84d: %Method.type.163 = struct_value () [symbolic]
// CHECK:STDOUT:   %require_complete.dbd: <witness> = require_complete_type %Self.as_type.cd1 [symbolic]
// CHECK:STDOUT:   %Point: type = class_type @Point [concrete]
// CHECK:STDOUT:   %Z.impl_witness: <witness> = impl_witness @Point.%Z.impl_witness_table [concrete]
// CHECK:STDOUT:   %Zero.type.e33: type = fn_type @Zero.3 [concrete]
// CHECK:STDOUT:   %Zero.dec: %Zero.type.e33 = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.35b: type = pattern_type %Point [concrete]
// CHECK:STDOUT:   %Method.type.2ed: type = fn_type @Method.3 [concrete]
// CHECK:STDOUT:   %Method.2c2: %Method.type.2ed = struct_value () [concrete]
// CHECK:STDOUT:   %Z.facet: %Z.type = facet_value %Point, (%Z.impl_witness) [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %.d52: type = fn_type_with_self_type %Zero.type.822, %Z.facet [concrete]
// CHECK:STDOUT:   %Point.val: %Point = struct_value () [concrete]
// CHECK:STDOUT:   %.462: type = fn_type_with_self_type %Method.type.f12, %Z.facet [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %Op.type.003: type = fn_type @Op.2, @Destroy.impl(%Point) [concrete]
// CHECK:STDOUT:   %Op.d2d: %Op.type.003 = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.536: type = ptr_type %Point [concrete]
// CHECK:STDOUT:   %Op.specific_fn: <specific function> = specific_function %Op.d2d, @Op.2(%Point) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Z = %Z.decl
// CHECK:STDOUT:     .Point = %Point.decl
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Z.decl: type = interface_decl @Z [concrete = constants.%Z.type] {} {}
// CHECK:STDOUT:   %Point.decl: type = class_decl @Point [concrete = constants.%Point] {} {}
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Z {
// CHECK:STDOUT:   %Self: %Z.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.6e6]
// CHECK:STDOUT:   %Zero.decl: %Zero.type.822 = fn_decl @Zero.1 [concrete = constants.%Zero.d14] {} {}
// CHECK:STDOUT:   %assoc0: %Z.assoc_type = assoc_entity element0, %Zero.decl [concrete = constants.%assoc0.534]
// CHECK:STDOUT:   %Method.decl: %Method.type.f12 = fn_decl @Method.1 [concrete = constants.%Method.724] {
// CHECK:STDOUT:     %self.patt: @Method.1.%pattern_type (%pattern_type.a40) = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: @Method.1.%pattern_type (%pattern_type.a40) = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: @Method.1.%Self.as_type.loc5_19.1 (%Self.as_type.cd1) = value_param call_param0
// CHECK:STDOUT:     %.loc5_19.1: type = splice_block %.loc5_19.2 [symbolic = %Self.as_type.loc5_19.1 (constants.%Self.as_type.cd1)] {
// CHECK:STDOUT:       %Self.ref: %Z.type = name_ref Self, @Z.%Self [symbolic = %Self (constants.%Self.6e6)]
// CHECK:STDOUT:       %Self.as_type.loc5_19.2: type = facet_access_type %Self.ref [symbolic = %Self.as_type.loc5_19.1 (constants.%Self.as_type.cd1)]
// CHECK:STDOUT:       %.loc5_19.2: type = converted %Self.ref, %Self.as_type.loc5_19.2 [symbolic = %Self.as_type.loc5_19.1 (constants.%Self.as_type.cd1)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %self: @Method.1.%Self.as_type.loc5_19.1 (%Self.as_type.cd1) = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %assoc1: %Z.assoc_type = assoc_entity element1, %Method.decl [concrete = constants.%assoc1]
// CHECK:STDOUT:   impl_decl @Z.impl.6b5 [concrete] {} {
// CHECK:STDOUT:     %Z.ref: type = name_ref Z, file.%Z.decl [concrete = constants.%Z.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .Zero = %assoc0
// CHECK:STDOUT:   .Method = %assoc1
// CHECK:STDOUT:   .Z = <poisoned>
// CHECK:STDOUT:   witness = (%Zero.decl, %Method.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @Z.impl.6b5(@Z.%Self: %Z.type) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self: %Z.type = bind_symbolic_name Self, 0 [symbolic = %Self (constants.%Self.6e6)]
// CHECK:STDOUT:   %Zero.type: type = fn_type @Zero.2, @Z.impl.6b5(%Self) [symbolic = %Zero.type (constants.%Zero.type.db4)]
// CHECK:STDOUT:   %Zero: @Z.impl.6b5.%Zero.type (%Zero.type.db4) = struct_value () [symbolic = %Zero (constants.%Zero.8fb)]
// CHECK:STDOUT:   %Method.type: type = fn_type @Method.2, @Z.impl.6b5(%Self) [symbolic = %Method.type (constants.%Method.type.163)]
// CHECK:STDOUT:   %Method: @Z.impl.6b5.%Method.type (%Method.type.163) = struct_value () [symbolic = %Method (constants.%Method.84d)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: <error> as %Z.ref {
// CHECK:STDOUT:     %Zero.decl: @Z.impl.6b5.%Zero.type (%Zero.type.db4) = fn_decl @Zero.2 [symbolic = @Z.impl.6b5.%Zero (constants.%Zero.8fb)] {} {}
// CHECK:STDOUT:     %Method.decl: @Z.impl.6b5.%Method.type (%Method.type.163) = fn_decl @Method.2 [symbolic = @Z.impl.6b5.%Method (constants.%Method.84d)] {
// CHECK:STDOUT:       %self.patt: @Method.2.%pattern_type (%pattern_type.a40) = binding_pattern self [concrete]
// CHECK:STDOUT:       %self.param_patt: @Method.2.%pattern_type (%pattern_type.a40) = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %self.param: @Method.2.%Self.as_type.loc13_21.1 (%Self.as_type.cd1) = value_param call_param0
// CHECK:STDOUT:       %.loc13_21.1: type = splice_block %.loc13_21.2 [symbolic = %Self.as_type.loc13_21.1 (constants.%Self.as_type.cd1)] {
// CHECK:STDOUT:         %Self.ref: %Z.type = name_ref Self, @Z.%Self [symbolic = %Self (constants.%Self.6e6)]
// CHECK:STDOUT:         %Self.as_type.loc13_21.2: type = facet_access_type %Self.ref [symbolic = %Self.as_type.loc13_21.1 (constants.%Self.as_type.cd1)]
// CHECK:STDOUT:         %.loc13_21.2: type = converted %Self.ref, %Self.as_type.loc13_21.2 [symbolic = %Self.as_type.loc13_21.1 (constants.%Self.as_type.cd1)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %self: @Method.2.%Self.as_type.loc13_21.1 (%Self.as_type.cd1) = bind_name self, %self.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Zero = %Zero.decl
// CHECK:STDOUT:     .Method = %Method.decl
// CHECK:STDOUT:     witness = <error>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @Z.impl.c94: %Self.ref as %Z.ref {
// CHECK:STDOUT:   %Zero.decl: %Zero.type.e33 = fn_decl @Zero.3 [concrete = constants.%Zero.dec] {} {}
// CHECK:STDOUT:   %Method.decl: %Method.type.2ed = fn_decl @Method.3 [concrete = constants.%Method.2c2] {
// CHECK:STDOUT:     %self.patt: %pattern_type.35b = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.35b = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %Point = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%Point [concrete = constants.%Point]
// CHECK:STDOUT:     %self: %Point = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Zero = %Zero.decl
// CHECK:STDOUT:   .Method = %Method.decl
// CHECK:STDOUT:   witness = @Point.%Z.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Point {
// CHECK:STDOUT:   impl_decl @Z.impl.c94 [concrete] {} {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%Point [concrete = constants.%Point]
// CHECK:STDOUT:     %Z.ref: type = name_ref Z, file.%Z.decl [concrete = constants.%Z.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Z.impl_witness_table = impl_witness_table (@Z.impl.c94.%Zero.decl, @Z.impl.c94.%Method.decl), @Z.impl.c94 [concrete]
// CHECK:STDOUT:   %Z.impl_witness: <witness> = impl_witness %Z.impl_witness_table [concrete = constants.%Z.impl_witness]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Point
// CHECK:STDOUT:   .Z = <poisoned>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Zero.1(@Z.%Self: %Z.type) {
// CHECK:STDOUT:   fn();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Method.1(@Z.%Self: %Z.type) {
// CHECK:STDOUT:   %Self: %Z.type = bind_symbolic_name Self, 0 [symbolic = %Self (constants.%Self.6e6)]
// CHECK:STDOUT:   %Self.as_type.loc5_19.1: type = facet_access_type %Self [symbolic = %Self.as_type.loc5_19.1 (constants.%Self.as_type.cd1)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Self.as_type.loc5_19.1 [symbolic = %pattern_type (constants.%pattern_type.a40)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @Method.1.%Self.as_type.loc5_19.1 (%Self.as_type.cd1));
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Zero.2(@Z.%Self: %Z.type) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Method.2(@Z.%Self: %Z.type) {
// CHECK:STDOUT:   %Self: %Z.type = bind_symbolic_name Self, 0 [symbolic = %Self (constants.%Self.6e6)]
// CHECK:STDOUT:   %Self.as_type.loc13_21.1: type = facet_access_type %Self [symbolic = %Self.as_type.loc13_21.1 (constants.%Self.as_type.cd1)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Self.as_type.loc13_21.1 [symbolic = %pattern_type (constants.%pattern_type.a40)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %Self.as_type.loc13_21.1 [symbolic = %require_complete (constants.%require_complete.dbd)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @Method.2.%Self.as_type.loc13_21.1 (%Self.as_type.cd1)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Zero.3() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Method.3(%self.param: %Point) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Point.ref.loc28: type = name_ref Point, file.%Point.decl [concrete = constants.%Point]
// CHECK:STDOUT:   %Z.ref.loc28: type = name_ref Z, file.%Z.decl [concrete = constants.%Z.type]
// CHECK:STDOUT:   %Zero.ref: %Z.assoc_type = name_ref Zero, @Z.%assoc0 [concrete = constants.%assoc0.534]
// CHECK:STDOUT:   %Z.facet: %Z.type = facet_value constants.%Point, (constants.%Z.impl_witness) [concrete = constants.%Z.facet]
// CHECK:STDOUT:   %.loc28: %Z.type = converted %Point.ref.loc28, %Z.facet [concrete = constants.%Z.facet]
// CHECK:STDOUT:   %impl.elem0: %.d52 = impl_witness_access constants.%Z.impl_witness, element0 [concrete = constants.%Zero.dec]
// CHECK:STDOUT:   %Zero.call: init %empty_tuple.type = call %impl.elem0()
// CHECK:STDOUT:   %.loc29_5.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %Point.ref.loc29: type = name_ref Point, file.%Point.decl [concrete = constants.%Point]
// CHECK:STDOUT:   %.loc29_5.2: ref %Point = temporary_storage
// CHECK:STDOUT:   %.loc29_5.3: init %Point = class_init (), %.loc29_5.2 [concrete = constants.%Point.val]
// CHECK:STDOUT:   %.loc29_5.4: ref %Point = temporary %.loc29_5.2, %.loc29_5.3
// CHECK:STDOUT:   %.loc29_7.1: ref %Point = converted %.loc29_5.1, %.loc29_5.4
// CHECK:STDOUT:   %Z.ref.loc29: type = name_ref Z, file.%Z.decl [concrete = constants.%Z.type]
// CHECK:STDOUT:   %Method.ref: %Z.assoc_type = name_ref Method, @Z.%assoc1 [concrete = constants.%assoc1]
// CHECK:STDOUT:   %impl.elem1: %.462 = impl_witness_access constants.%Z.impl_witness, element1 [concrete = constants.%Method.2c2]
// CHECK:STDOUT:   %bound_method.loc29_16: <bound method> = bound_method %.loc29_7.1, %impl.elem1
// CHECK:STDOUT:   %.loc29_7.2: %Point = bind_value %.loc29_7.1
// CHECK:STDOUT:   %Method.call: init %empty_tuple.type = call %bound_method.loc29_16(%.loc29_7.2)
// CHECK:STDOUT:   %Op.bound: <bound method> = bound_method %.loc29_5.2, constants.%Op.d2d
// CHECK:STDOUT:   %Op.specific_fn: <specific function> = specific_function constants.%Op.d2d, @Op.2(constants.%Point) [concrete = constants.%Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc29_5: <bound method> = bound_method %.loc29_5.2, %Op.specific_fn
// CHECK:STDOUT:   %addr: %ptr.536 = addr_of %.loc29_5.2
// CHECK:STDOUT:   %no_op: init %empty_tuple.type = call %bound_method.loc29_5(%addr)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Zero.1(constants.%Self.6e6) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Method.1(constants.%Self.6e6) {
// CHECK:STDOUT:   %Self => constants.%Self.6e6
// CHECK:STDOUT:   %Self.as_type.loc5_19.1 => constants.%Self.as_type.cd1
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.a40
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Z.impl.6b5(constants.%Self.6e6) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%Self.6e6
// CHECK:STDOUT:   %Zero.type => constants.%Zero.type.db4
// CHECK:STDOUT:   %Zero => constants.%Zero.8fb
// CHECK:STDOUT:   %Method.type => constants.%Method.type.163
// CHECK:STDOUT:   %Method => constants.%Method.84d
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Zero.2(constants.%Self.6e6) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Method.2(constants.%Self.6e6) {
// CHECK:STDOUT:   %Self => constants.%Self.6e6
// CHECK:STDOUT:   %Self.as_type.loc13_21.1 => constants.%Self.as_type.cd1
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.a40
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Zero.1(constants.%Z.facet) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Method.1(constants.%Z.facet) {
// CHECK:STDOUT:   %Self => constants.%Z.facet
// CHECK:STDOUT:   %Self.as_type.loc5_19.1 => constants.%Point
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.35b
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_impl_as_other_interface.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %A.type: type = facet_type <@A> [concrete]
// CHECK:STDOUT:   %Self.31d: %A.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %B.type.1c3: type = fn_type @B.1 [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %B.b08: %B.type.1c3 = struct_value () [concrete]
// CHECK:STDOUT:   %A.assoc_type: type = assoc_entity_type @A [concrete]
// CHECK:STDOUT:   %assoc0: %A.assoc_type = assoc_entity element0, @A.%B.decl [concrete]
// CHECK:STDOUT:   %C.type: type = facet_type <@C> [concrete]
// CHECK:STDOUT:   %Self.02e: %C.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %X: type = class_type @X [concrete]
// CHECK:STDOUT:   %A.impl_witness: <witness> = impl_witness @X.%A.impl_witness_table [concrete]
// CHECK:STDOUT:   %B.type.d47: type = fn_type @B.2 [concrete]
// CHECK:STDOUT:   %B.4af: %B.type.d47 = struct_value () [concrete]
// CHECK:STDOUT:   %A.facet: %A.type = facet_value %X, (%A.impl_witness) [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .A = %A.decl
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .G = %G.decl
// CHECK:STDOUT:     .X = %X.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %A.decl: type = interface_decl @A [concrete = constants.%A.type] {} {}
// CHECK:STDOUT:   %C.decl: type = interface_decl @C [concrete = constants.%C.type] {} {}
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {} {}
// CHECK:STDOUT:   %X.decl: type = class_decl @X [concrete = constants.%X] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @A {
// CHECK:STDOUT:   %Self: %A.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.31d]
// CHECK:STDOUT:   %B.decl: %B.type.1c3 = fn_decl @B.1 [concrete = constants.%B.b08] {} {}
// CHECK:STDOUT:   %assoc0: %A.assoc_type = assoc_entity element0, %B.decl [concrete = constants.%assoc0]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .B = %assoc0
// CHECK:STDOUT:   witness = (%B.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @C {
// CHECK:STDOUT:   %Self: %C.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.02e]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @A.impl: %Self.ref as %A.ref {
// CHECK:STDOUT:   impl_decl @C.impl [concrete] {} {
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %B.decl: %B.type.d47 = fn_decl @B.2 [concrete = constants.%B.4af] {} {}
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .C = <poisoned>
// CHECK:STDOUT:   .B = %B.decl
// CHECK:STDOUT:   .G = <poisoned>
// CHECK:STDOUT:   witness = @X.%A.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @C.impl: <error> as %C.ref {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @X {
// CHECK:STDOUT:   impl_decl @A.impl [concrete] {} {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%X [concrete = constants.%X]
// CHECK:STDOUT:     %A.ref: type = name_ref A, file.%A.decl [concrete = constants.%A.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %A.impl_witness_table = impl_witness_table (@A.impl.%B.decl), @A.impl [concrete]
// CHECK:STDOUT:   %A.impl_witness: <witness> = impl_witness %A.impl_witness_table [concrete = constants.%A.impl_witness]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%X
// CHECK:STDOUT:   .A = <poisoned>
// CHECK:STDOUT:   .C = <poisoned>
// CHECK:STDOUT:   .G = <poisoned>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @B.1(@A.%Self: %A.type) {
// CHECK:STDOUT:   fn();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @B.2() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %G.ref: %G.type = name_ref G, file.%G.decl [concrete = constants.%G]
// CHECK:STDOUT:   %G.call: init %empty_tuple.type = call %G.ref()
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @B.1(constants.%Self.31d) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @B.1(constants.%A.facet) {}
// CHECK:STDOUT:
