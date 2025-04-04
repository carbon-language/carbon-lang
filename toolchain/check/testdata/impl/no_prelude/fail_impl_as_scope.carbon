// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/impl/no_prelude/fail_impl_as_scope.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/impl/no_prelude/fail_impl_as_scope.carbon

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
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .I = %I.decl
// CHECK:STDOUT:     .G = %G.decl
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %I.decl: type = interface_decl @I [concrete = constants.%I.type] {} {}
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {} {}
// CHECK:STDOUT:   impl_decl @impl [concrete] {} {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, <error> [concrete = <error>]
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
// CHECK:STDOUT: impl @impl: %Self.ref as %I.ref {
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
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .J = %J.decl
// CHECK:STDOUT:     .G = %G.decl
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
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
// CHECK:STDOUT: impl @impl: %Self.ref as %J.ref {
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
// CHECK:STDOUT:   impl_decl @impl [concrete] {} {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, <error> [concrete = <error>]
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
// CHECK:STDOUT:   %Self: %Z.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %Zero.type.822: type = fn_type @Zero.1 [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Zero.d14: %Zero.type.822 = struct_value () [concrete]
// CHECK:STDOUT:   %Z.assoc_type: type = assoc_entity_type @Z [concrete]
// CHECK:STDOUT:   %assoc0: %Z.assoc_type = assoc_entity element0, @Z.%Zero.decl [concrete]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic]
// CHECK:STDOUT:   %Method.type.f12: type = fn_type @Method.1 [concrete]
// CHECK:STDOUT:   %Method.724: %Method.type.f12 = struct_value () [concrete]
// CHECK:STDOUT:   %assoc1: %Z.assoc_type = assoc_entity element1, @Z.%Method.decl [concrete]
// CHECK:STDOUT:   %Zero.type.db4: type = fn_type @Zero.2, @impl.6b5(%Self) [symbolic]
// CHECK:STDOUT:   %Zero.8fb: %Zero.type.db4 = struct_value () [symbolic]
// CHECK:STDOUT:   %Method.type.163: type = fn_type @Method.2, @impl.6b5(%Self) [symbolic]
// CHECK:STDOUT:   %Method.84d: %Method.type.163 = struct_value () [symbolic]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %Self.as_type [symbolic]
// CHECK:STDOUT:   %Point: type = class_type @Point [concrete]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (@impl.c94.%Zero.decl, @impl.c94.%Method.decl) [concrete]
// CHECK:STDOUT:   %Zero.type.e33: type = fn_type @Zero.3 [concrete]
// CHECK:STDOUT:   %Zero.dec: %Zero.type.e33 = struct_value () [concrete]
// CHECK:STDOUT:   %Method.type.2ed: type = fn_type @Method.3 [concrete]
// CHECK:STDOUT:   %Method.2c2: %Method.type.2ed = struct_value () [concrete]
// CHECK:STDOUT:   %Z.facet: %Z.type = facet_value %Point, (%impl_witness) [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %.071: type = fn_type_with_self_type %Zero.type.822, %Z.facet [concrete]
// CHECK:STDOUT:   %Point.val: %Point = struct_value () [concrete]
// CHECK:STDOUT:   %.2c0: type = fn_type_with_self_type %Method.type.f12, %Z.facet [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Z = %Z.decl
// CHECK:STDOUT:     .Point = %Point.decl
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Z.decl: type = interface_decl @Z [concrete = constants.%Z.type] {} {}
// CHECK:STDOUT:   %Point.decl: type = class_decl @Point [concrete = constants.%Point] {} {}
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Z {
// CHECK:STDOUT:   %Self: %Z.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   %Zero.decl: %Zero.type.822 = fn_decl @Zero.1 [concrete = constants.%Zero.d14] {} {}
// CHECK:STDOUT:   %assoc0: %Z.assoc_type = assoc_entity element0, %Zero.decl [concrete = constants.%assoc0]
// CHECK:STDOUT:   %Method.decl: %Method.type.f12 = fn_decl @Method.1 [concrete = constants.%Method.724] {
// CHECK:STDOUT:     %self.patt: @Method.1.%Self.as_type.loc5_19.1 (%Self.as_type) = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: @Method.1.%Self.as_type.loc5_19.1 (%Self.as_type) = value_param_pattern %self.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: @Method.1.%Self.as_type.loc5_19.1 (%Self.as_type) = value_param call_param0
// CHECK:STDOUT:     %.loc5_19.1: type = splice_block %.loc5_19.2 [symbolic = %Self.as_type.loc5_19.1 (constants.%Self.as_type)] {
// CHECK:STDOUT:       %Self.ref: %Z.type = name_ref Self, @Z.%Self [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:       %Self.as_type.loc5_19.2: type = facet_access_type %Self.ref [symbolic = %Self.as_type.loc5_19.1 (constants.%Self.as_type)]
// CHECK:STDOUT:       %.loc5_19.2: type = converted %Self.ref, %Self.as_type.loc5_19.2 [symbolic = %Self.as_type.loc5_19.1 (constants.%Self.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %self: @Method.1.%Self.as_type.loc5_19.1 (%Self.as_type) = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %assoc1: %Z.assoc_type = assoc_entity element1, %Method.decl [concrete = constants.%assoc1]
// CHECK:STDOUT:   impl_decl @impl.6b5 [concrete] {} {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, <error> [concrete = <error>]
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
// CHECK:STDOUT: generic impl @impl.6b5(@Z.%Self: %Z.type) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self: %Z.type = bind_symbolic_name Self, 0 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:   %Zero.type: type = fn_type @Zero.2, @impl.6b5(%Self) [symbolic = %Zero.type (constants.%Zero.type.db4)]
// CHECK:STDOUT:   %Zero: @impl.6b5.%Zero.type (%Zero.type.db4) = struct_value () [symbolic = %Zero (constants.%Zero.8fb)]
// CHECK:STDOUT:   %Method.type: type = fn_type @Method.2, @impl.6b5(%Self) [symbolic = %Method.type (constants.%Method.type.163)]
// CHECK:STDOUT:   %Method: @impl.6b5.%Method.type (%Method.type.163) = struct_value () [symbolic = %Method (constants.%Method.84d)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %Self.ref as %Z.ref {
// CHECK:STDOUT:     %Zero.decl: @impl.6b5.%Zero.type (%Zero.type.db4) = fn_decl @Zero.2 [symbolic = @impl.6b5.%Zero (constants.%Zero.8fb)] {} {}
// CHECK:STDOUT:     %Method.decl: @impl.6b5.%Method.type (%Method.type.163) = fn_decl @Method.2 [symbolic = @impl.6b5.%Method (constants.%Method.84d)] {
// CHECK:STDOUT:       %self.patt: @Method.2.%Self.as_type.loc13_21.1 (%Self.as_type) = binding_pattern self
// CHECK:STDOUT:       %self.param_patt: @Method.2.%Self.as_type.loc13_21.1 (%Self.as_type) = value_param_pattern %self.patt, call_param0
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %self.param: @Method.2.%Self.as_type.loc13_21.1 (%Self.as_type) = value_param call_param0
// CHECK:STDOUT:       %.loc13_21.1: type = splice_block %.loc13_21.2 [symbolic = %Self.as_type.loc13_21.1 (constants.%Self.as_type)] {
// CHECK:STDOUT:         %Self.ref: %Z.type = name_ref Self, @Z.%Self [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:         %Self.as_type.loc13_21.2: type = facet_access_type %Self.ref [symbolic = %Self.as_type.loc13_21.1 (constants.%Self.as_type)]
// CHECK:STDOUT:         %.loc13_21.2: type = converted %Self.ref, %Self.as_type.loc13_21.2 [symbolic = %Self.as_type.loc13_21.1 (constants.%Self.as_type)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %self: @Method.2.%Self.as_type.loc13_21.1 (%Self.as_type) = bind_name self, %self.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Zero = %Zero.decl
// CHECK:STDOUT:     .Method = %Method.decl
// CHECK:STDOUT:     witness = <error>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.c94: %Self.ref as %Z.ref {
// CHECK:STDOUT:   %Zero.decl: %Zero.type.e33 = fn_decl @Zero.3 [concrete = constants.%Zero.dec] {} {}
// CHECK:STDOUT:   %Method.decl: %Method.type.2ed = fn_decl @Method.3 [concrete = constants.%Method.2c2] {
// CHECK:STDOUT:     %self.patt: %Point = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: %Point = value_param_pattern %self.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %Point = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%Point [concrete = constants.%Point]
// CHECK:STDOUT:     %self: %Point = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Zero = %Zero.decl
// CHECK:STDOUT:   .Method = %Method.decl
// CHECK:STDOUT:   witness = @Point.%impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Point {
// CHECK:STDOUT:   impl_decl @impl.c94 [concrete] {} {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%Point [concrete = constants.%Point]
// CHECK:STDOUT:     %Z.ref: type = name_ref Z, file.%Z.decl [concrete = constants.%Z.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (@impl.c94.%Zero.decl, @impl.c94.%Method.decl) [concrete = constants.%impl_witness]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type]
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
// CHECK:STDOUT:   %Self: %Z.type = bind_symbolic_name Self, 0 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:   %Self.as_type.loc5_19.1: type = facet_access_type %Self [symbolic = %Self.as_type.loc5_19.1 (constants.%Self.as_type)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @Method.1.%Self.as_type.loc5_19.1 (%Self.as_type)]();
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
// CHECK:STDOUT:   %Self: %Z.type = bind_symbolic_name Self, 0 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:   %Self.as_type.loc13_21.1: type = facet_access_type %Self [symbolic = %Self.as_type.loc13_21.1 (constants.%Self.as_type)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type @Method.2.%Self.as_type.loc13_21.1 (%Self.as_type) [symbolic = %require_complete (constants.%require_complete)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @Method.2.%Self.as_type.loc13_21.1 (%Self.as_type)]() {
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
// CHECK:STDOUT: fn @Method.3[%self.param_patt: %Point]() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Point.ref.loc28: type = name_ref Point, file.%Point.decl [concrete = constants.%Point]
// CHECK:STDOUT:   %Z.ref.loc28: type = name_ref Z, file.%Z.decl [concrete = constants.%Z.type]
// CHECK:STDOUT:   %Zero.ref: %Z.assoc_type = name_ref Zero, @Z.%assoc0 [concrete = constants.%assoc0]
// CHECK:STDOUT:   %Z.facet: %Z.type = facet_value constants.%Point, (constants.%impl_witness) [concrete = constants.%Z.facet]
// CHECK:STDOUT:   %.loc28: %Z.type = converted %Point.ref.loc28, %Z.facet [concrete = constants.%Z.facet]
// CHECK:STDOUT:   %as_wit.iface0: <witness> = facet_access_witness %.loc28, element0 [concrete = constants.%impl_witness]
// CHECK:STDOUT:   %impl.elem0: %.071 = impl_witness_access %as_wit.iface0, element0 [concrete = constants.%Zero.dec]
// CHECK:STDOUT:   %Zero.call: init %empty_tuple.type = call %impl.elem0()
// CHECK:STDOUT:   %.loc29_5.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %Point.ref.loc29: type = name_ref Point, file.%Point.decl [concrete = constants.%Point]
// CHECK:STDOUT:   %.loc29_5.2: ref %Point = temporary_storage
// CHECK:STDOUT:   %.loc29_5.3: init %Point = class_init (), %.loc29_5.2 [concrete = constants.%Point.val]
// CHECK:STDOUT:   %.loc29_5.4: ref %Point = temporary %.loc29_5.2, %.loc29_5.3
// CHECK:STDOUT:   %.loc29_7.1: ref %Point = converted %.loc29_5.1, %.loc29_5.4
// CHECK:STDOUT:   %Z.ref.loc29: type = name_ref Z, file.%Z.decl [concrete = constants.%Z.type]
// CHECK:STDOUT:   %Method.ref: %Z.assoc_type = name_ref Method, @Z.%assoc1 [concrete = constants.%assoc1]
// CHECK:STDOUT:   %impl.elem1: %.2c0 = impl_witness_access constants.%impl_witness, element1 [concrete = constants.%Method.2c2]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %.loc29_7.1, %impl.elem1
// CHECK:STDOUT:   %.loc29_7.2: %Point = bind_value %.loc29_7.1
// CHECK:STDOUT:   %Method.call: init %empty_tuple.type = call %bound_method(%.loc29_7.2)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Zero.1(constants.%Self) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Method.1(constants.%Self) {
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT:   %Self.as_type.loc5_19.1 => constants.%Self.as_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.6b5(constants.%Self) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT:   %Zero.type => constants.%Zero.type.db4
// CHECK:STDOUT:   %Zero => constants.%Zero.8fb
// CHECK:STDOUT:   %Method.type => constants.%Method.type.163
// CHECK:STDOUT:   %Method => constants.%Method.84d
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Zero.2(constants.%Self) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Method.2(constants.%Self) {
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT:   %Self.as_type.loc13_21.1 => constants.%Self.as_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.6b5(%Self) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Zero.1(constants.%Z.facet) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Method.1(constants.%Z.facet) {
// CHECK:STDOUT:   %Self => constants.%Z.facet
// CHECK:STDOUT:   %Self.as_type.loc5_19.1 => constants.%Point
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
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (@impl.3b4.%B.decl) [concrete]
// CHECK:STDOUT:   %B.type.d47: type = fn_type @B.2 [concrete]
// CHECK:STDOUT:   %B.4af: %B.type.d47 = struct_value () [concrete]
// CHECK:STDOUT:   %A.facet: %A.type = facet_value %X, (%impl_witness) [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .A = %A.decl
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .G = %G.decl
// CHECK:STDOUT:     .X = %X.decl
// CHECK:STDOUT:   }
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
// CHECK:STDOUT: impl @impl.3b4: %Self.ref as %A.ref {
// CHECK:STDOUT:   impl_decl @impl.4cc [concrete] {} {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, <error> [concrete = <error>]
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %B.decl: %B.type.d47 = fn_decl @B.2 [concrete = constants.%B.4af] {} {}
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .C = <poisoned>
// CHECK:STDOUT:   .B = %B.decl
// CHECK:STDOUT:   .G = <poisoned>
// CHECK:STDOUT:   witness = @X.%impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.4cc: %Self.ref as %C.ref {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @X {
// CHECK:STDOUT:   impl_decl @impl.3b4 [concrete] {} {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%X [concrete = constants.%X]
// CHECK:STDOUT:     %A.ref: type = name_ref A, file.%A.decl [concrete = constants.%A.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (@impl.3b4.%B.decl) [concrete = constants.%impl_witness]
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
