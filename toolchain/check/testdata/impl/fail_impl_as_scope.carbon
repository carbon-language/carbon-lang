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
// CHECK:STDOUT:   impl_decl @<error>.as.I.impl [concrete] {} {
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
// CHECK:STDOUT: impl @<error>.as.I.impl: <error> as %I.ref {
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
// CHECK:STDOUT: impl @<error>.as.J.impl: <error> as %J.ref {
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
// CHECK:STDOUT:   impl_decl @<error>.as.J.impl [concrete] {} {
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
// CHECK:STDOUT:   %Z.Zero.type: type = fn_type @Z.Zero [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Z.Zero: %Z.Zero.type = struct_value () [concrete]
// CHECK:STDOUT:   %Z.assoc_type: type = assoc_entity_type @Z [concrete]
// CHECK:STDOUT:   %assoc0.534: %Z.assoc_type = assoc_entity element0, @Z.%Z.Zero.decl [concrete]
// CHECK:STDOUT:   %Self.as_type.cd1: type = facet_access_type %Self.6e6 [symbolic]
// CHECK:STDOUT:   %pattern_type.a40: type = pattern_type %Self.as_type.cd1 [symbolic]
// CHECK:STDOUT:   %Z.Method.type: type = fn_type @Z.Method [concrete]
// CHECK:STDOUT:   %Z.Method: %Z.Method.type = struct_value () [concrete]
// CHECK:STDOUT:   %assoc1: %Z.assoc_type = assoc_entity element1, @Z.%Z.Method.decl [concrete]
// CHECK:STDOUT:   %<error>.as.Z.impl.Zero.type: type = fn_type @<error>.as.Z.impl.Zero, @<error>.as.Z.impl(%Self.6e6) [symbolic]
// CHECK:STDOUT:   %<error>.as.Z.impl.Zero: %<error>.as.Z.impl.Zero.type = struct_value () [symbolic]
// CHECK:STDOUT:   %<error>.as.Z.impl.Method.type: type = fn_type @<error>.as.Z.impl.Method, @<error>.as.Z.impl(%Self.6e6) [symbolic]
// CHECK:STDOUT:   %<error>.as.Z.impl.Method: %<error>.as.Z.impl.Method.type = struct_value () [symbolic]
// CHECK:STDOUT:   %require_complete.dbd: <witness> = require_complete_type %Self.as_type.cd1 [symbolic]
// CHECK:STDOUT:   %Point: type = class_type @Point [concrete]
// CHECK:STDOUT:   %Z.impl_witness: <witness> = impl_witness @Point.%Z.impl_witness_table [concrete]
// CHECK:STDOUT:   %Point.as.Z.impl.Zero.type: type = fn_type @Point.as.Z.impl.Zero [concrete]
// CHECK:STDOUT:   %Point.as.Z.impl.Zero: %Point.as.Z.impl.Zero.type = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.35b: type = pattern_type %Point [concrete]
// CHECK:STDOUT:   %Point.as.Z.impl.Method.type: type = fn_type @Point.as.Z.impl.Method [concrete]
// CHECK:STDOUT:   %Point.as.Z.impl.Method: %Point.as.Z.impl.Method.type = struct_value () [concrete]
// CHECK:STDOUT:   %Z.facet: %Z.type = facet_value %Point, (%Z.impl_witness) [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %pattern_type.f6d: type = pattern_type auto [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness.e19: <witness> = impl_witness @Point.%Destroy.impl_witness_table [concrete]
// CHECK:STDOUT:   %ptr.536: type = ptr_type %Point [concrete]
// CHECK:STDOUT:   %pattern_type.bbc: type = pattern_type %ptr.536 [concrete]
// CHECK:STDOUT:   %Point.as.Destroy.impl.Op.type: type = fn_type @Point.as.Destroy.impl.Op [concrete]
// CHECK:STDOUT:   %Point.as.Destroy.impl.Op: %Point.as.Destroy.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %.d52: type = fn_type_with_self_type %Z.Zero.type, %Z.facet [concrete]
// CHECK:STDOUT:   %Point.val: %Point = struct_value () [concrete]
// CHECK:STDOUT:   %.462: type = fn_type_with_self_type %Z.Method.type, %Z.facet [concrete]
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
// CHECK:STDOUT:   %Z.Zero.decl: %Z.Zero.type = fn_decl @Z.Zero [concrete = constants.%Z.Zero] {} {}
// CHECK:STDOUT:   %assoc0: %Z.assoc_type = assoc_entity element0, %Z.Zero.decl [concrete = constants.%assoc0.534]
// CHECK:STDOUT:   %Z.Method.decl: %Z.Method.type = fn_decl @Z.Method [concrete = constants.%Z.Method] {
// CHECK:STDOUT:     %self.patt: @Z.Method.%pattern_type (%pattern_type.a40) = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: @Z.Method.%pattern_type (%pattern_type.a40) = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: @Z.Method.%Self.as_type.loc5_19.1 (%Self.as_type.cd1) = value_param call_param0
// CHECK:STDOUT:     %.loc5_19.1: type = splice_block %.loc5_19.2 [symbolic = %Self.as_type.loc5_19.1 (constants.%Self.as_type.cd1)] {
// CHECK:STDOUT:       %Self.ref: %Z.type = name_ref Self, @Z.%Self [symbolic = %Self (constants.%Self.6e6)]
// CHECK:STDOUT:       %Self.as_type.loc5_19.2: type = facet_access_type %Self.ref [symbolic = %Self.as_type.loc5_19.1 (constants.%Self.as_type.cd1)]
// CHECK:STDOUT:       %.loc5_19.2: type = converted %Self.ref, %Self.as_type.loc5_19.2 [symbolic = %Self.as_type.loc5_19.1 (constants.%Self.as_type.cd1)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %self: @Z.Method.%Self.as_type.loc5_19.1 (%Self.as_type.cd1) = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %assoc1: %Z.assoc_type = assoc_entity element1, %Z.Method.decl [concrete = constants.%assoc1]
// CHECK:STDOUT:   impl_decl @<error>.as.Z.impl [concrete] {} {
// CHECK:STDOUT:     %Z.ref: type = name_ref Z, file.%Z.decl [concrete = constants.%Z.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .Zero = %assoc0
// CHECK:STDOUT:   .Method = %assoc1
// CHECK:STDOUT:   .Z = <poisoned>
// CHECK:STDOUT:   witness = (%Z.Zero.decl, %Z.Method.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @<error>.as.Z.impl(@Z.%Self: %Z.type) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self: %Z.type = bind_symbolic_name Self, 0 [symbolic = %Self (constants.%Self.6e6)]
// CHECK:STDOUT:   %<error>.as.Z.impl.Zero.type: type = fn_type @<error>.as.Z.impl.Zero, @<error>.as.Z.impl(%Self) [symbolic = %<error>.as.Z.impl.Zero.type (constants.%<error>.as.Z.impl.Zero.type)]
// CHECK:STDOUT:   %<error>.as.Z.impl.Zero: @<error>.as.Z.impl.%<error>.as.Z.impl.Zero.type (%<error>.as.Z.impl.Zero.type) = struct_value () [symbolic = %<error>.as.Z.impl.Zero (constants.%<error>.as.Z.impl.Zero)]
// CHECK:STDOUT:   %<error>.as.Z.impl.Method.type: type = fn_type @<error>.as.Z.impl.Method, @<error>.as.Z.impl(%Self) [symbolic = %<error>.as.Z.impl.Method.type (constants.%<error>.as.Z.impl.Method.type)]
// CHECK:STDOUT:   %<error>.as.Z.impl.Method: @<error>.as.Z.impl.%<error>.as.Z.impl.Method.type (%<error>.as.Z.impl.Method.type) = struct_value () [symbolic = %<error>.as.Z.impl.Method (constants.%<error>.as.Z.impl.Method)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: <error> as %Z.ref {
// CHECK:STDOUT:     %<error>.as.Z.impl.Zero.decl: @<error>.as.Z.impl.%<error>.as.Z.impl.Zero.type (%<error>.as.Z.impl.Zero.type) = fn_decl @<error>.as.Z.impl.Zero [symbolic = @<error>.as.Z.impl.%<error>.as.Z.impl.Zero (constants.%<error>.as.Z.impl.Zero)] {} {}
// CHECK:STDOUT:     %<error>.as.Z.impl.Method.decl: @<error>.as.Z.impl.%<error>.as.Z.impl.Method.type (%<error>.as.Z.impl.Method.type) = fn_decl @<error>.as.Z.impl.Method [symbolic = @<error>.as.Z.impl.%<error>.as.Z.impl.Method (constants.%<error>.as.Z.impl.Method)] {
// CHECK:STDOUT:       %self.patt: @<error>.as.Z.impl.Method.%pattern_type (%pattern_type.a40) = binding_pattern self [concrete]
// CHECK:STDOUT:       %self.param_patt: @<error>.as.Z.impl.Method.%pattern_type (%pattern_type.a40) = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %self.param: @<error>.as.Z.impl.Method.%Self.as_type.loc13_21.1 (%Self.as_type.cd1) = value_param call_param0
// CHECK:STDOUT:       %.loc13_21.1: type = splice_block %.loc13_21.2 [symbolic = %Self.as_type.loc13_21.1 (constants.%Self.as_type.cd1)] {
// CHECK:STDOUT:         %Self.ref: %Z.type = name_ref Self, @Z.%Self [symbolic = %Self (constants.%Self.6e6)]
// CHECK:STDOUT:         %Self.as_type.loc13_21.2: type = facet_access_type %Self.ref [symbolic = %Self.as_type.loc13_21.1 (constants.%Self.as_type.cd1)]
// CHECK:STDOUT:         %.loc13_21.2: type = converted %Self.ref, %Self.as_type.loc13_21.2 [symbolic = %Self.as_type.loc13_21.1 (constants.%Self.as_type.cd1)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %self: @<error>.as.Z.impl.Method.%Self.as_type.loc13_21.1 (%Self.as_type.cd1) = bind_name self, %self.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Zero = %<error>.as.Z.impl.Zero.decl
// CHECK:STDOUT:     .Method = %<error>.as.Z.impl.Method.decl
// CHECK:STDOUT:     witness = <error>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @Point.as.Z.impl: %Self.ref as %Z.ref {
// CHECK:STDOUT:   %Point.as.Z.impl.Zero.decl: %Point.as.Z.impl.Zero.type = fn_decl @Point.as.Z.impl.Zero [concrete = constants.%Point.as.Z.impl.Zero] {} {}
// CHECK:STDOUT:   %Point.as.Z.impl.Method.decl: %Point.as.Z.impl.Method.type = fn_decl @Point.as.Z.impl.Method [concrete = constants.%Point.as.Z.impl.Method] {
// CHECK:STDOUT:     %self.patt: %pattern_type.35b = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.35b = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %Point = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%Point [concrete = constants.%Point]
// CHECK:STDOUT:     %self: %Point = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Zero = %Point.as.Z.impl.Zero.decl
// CHECK:STDOUT:   .Method = %Point.as.Z.impl.Method.decl
// CHECK:STDOUT:   witness = @Point.%Z.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @Point.as.Destroy.impl: constants.%Point as constants.%Destroy.type {
// CHECK:STDOUT:   %Point.as.Destroy.impl.Op.decl: %Point.as.Destroy.impl.Op.type = fn_decl @Point.as.Destroy.impl.Op [concrete = constants.%Point.as.Destroy.impl.Op] {
// CHECK:STDOUT:     %self.patt: %pattern_type.bbc = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.bbc = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %.loc17: %pattern_type.f6d = addr_pattern %self.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %ptr.536 = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%Point [concrete = constants.%Point]
// CHECK:STDOUT:     %self: %ptr.536 = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Op = %Point.as.Destroy.impl.Op.decl
// CHECK:STDOUT:   witness = @Point.%Destroy.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Point {
// CHECK:STDOUT:   impl_decl @Point.as.Z.impl [concrete] {} {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%Point [concrete = constants.%Point]
// CHECK:STDOUT:     %Z.ref: type = name_ref Z, file.%Z.decl [concrete = constants.%Z.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Z.impl_witness_table = impl_witness_table (@Point.as.Z.impl.%Point.as.Z.impl.Zero.decl, @Point.as.Z.impl.%Point.as.Z.impl.Method.decl), @Point.as.Z.impl [concrete]
// CHECK:STDOUT:   %Z.impl_witness: <witness> = impl_witness %Z.impl_witness_table [concrete = constants.%Z.impl_witness]
// CHECK:STDOUT:   impl_decl @Point.as.Destroy.impl [concrete] {} {}
// CHECK:STDOUT:   %Destroy.impl_witness_table = impl_witness_table (@Point.as.Destroy.impl.%Point.as.Destroy.impl.Op.decl), @Point.as.Destroy.impl [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness: <witness> = impl_witness %Destroy.impl_witness_table [concrete = constants.%Destroy.impl_witness.e19]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Point
// CHECK:STDOUT:   .Z = <poisoned>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Z.Zero(@Z.%Self: %Z.type) {
// CHECK:STDOUT:   fn();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Z.Method(@Z.%Self: %Z.type) {
// CHECK:STDOUT:   %Self: %Z.type = bind_symbolic_name Self, 0 [symbolic = %Self (constants.%Self.6e6)]
// CHECK:STDOUT:   %Self.as_type.loc5_19.1: type = facet_access_type %Self [symbolic = %Self.as_type.loc5_19.1 (constants.%Self.as_type.cd1)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Self.as_type.loc5_19.1 [symbolic = %pattern_type (constants.%pattern_type.a40)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @Z.Method.%Self.as_type.loc5_19.1 (%Self.as_type.cd1));
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @<error>.as.Z.impl.Zero(@Z.%Self: %Z.type) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @<error>.as.Z.impl.Method(@Z.%Self: %Z.type) {
// CHECK:STDOUT:   %Self: %Z.type = bind_symbolic_name Self, 0 [symbolic = %Self (constants.%Self.6e6)]
// CHECK:STDOUT:   %Self.as_type.loc13_21.1: type = facet_access_type %Self [symbolic = %Self.as_type.loc13_21.1 (constants.%Self.as_type.cd1)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Self.as_type.loc13_21.1 [symbolic = %pattern_type (constants.%pattern_type.a40)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %Self.as_type.loc13_21.1 [symbolic = %require_complete (constants.%require_complete.dbd)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @<error>.as.Z.impl.Method.%Self.as_type.loc13_21.1 (%Self.as_type.cd1)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Point.as.Z.impl.Zero() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Point.as.Z.impl.Method(%self.param: %Point) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Point.as.Destroy.impl.Op(%self.param: %ptr.536) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Point.ref.loc28: type = name_ref Point, file.%Point.decl [concrete = constants.%Point]
// CHECK:STDOUT:   %Z.ref.loc28: type = name_ref Z, file.%Z.decl [concrete = constants.%Z.type]
// CHECK:STDOUT:   %Zero.ref: %Z.assoc_type = name_ref Zero, @Z.%assoc0 [concrete = constants.%assoc0.534]
// CHECK:STDOUT:   %Z.facet: %Z.type = facet_value constants.%Point, (constants.%Z.impl_witness) [concrete = constants.%Z.facet]
// CHECK:STDOUT:   %.loc28: %Z.type = converted %Point.ref.loc28, %Z.facet [concrete = constants.%Z.facet]
// CHECK:STDOUT:   %impl.elem0: %.d52 = impl_witness_access constants.%Z.impl_witness, element0 [concrete = constants.%Point.as.Z.impl.Zero]
// CHECK:STDOUT:   %Point.as.Z.impl.Zero.call: init %empty_tuple.type = call %impl.elem0()
// CHECK:STDOUT:   %.loc29_5.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %Point.ref.loc29: type = name_ref Point, file.%Point.decl [concrete = constants.%Point]
// CHECK:STDOUT:   %.loc29_5.2: ref %Point = temporary_storage
// CHECK:STDOUT:   %.loc29_5.3: init %Point = class_init (), %.loc29_5.2 [concrete = constants.%Point.val]
// CHECK:STDOUT:   %.loc29_5.4: ref %Point = temporary %.loc29_5.2, %.loc29_5.3
// CHECK:STDOUT:   %.loc29_7.1: ref %Point = converted %.loc29_5.1, %.loc29_5.4
// CHECK:STDOUT:   %Z.ref.loc29: type = name_ref Z, file.%Z.decl [concrete = constants.%Z.type]
// CHECK:STDOUT:   %Method.ref: %Z.assoc_type = name_ref Method, @Z.%assoc1 [concrete = constants.%assoc1]
// CHECK:STDOUT:   %impl.elem1: %.462 = impl_witness_access constants.%Z.impl_witness, element1 [concrete = constants.%Point.as.Z.impl.Method]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %.loc29_7.1, %impl.elem1
// CHECK:STDOUT:   %.loc29_7.2: %Point = bind_value %.loc29_7.1
// CHECK:STDOUT:   %Point.as.Z.impl.Method.call: init %empty_tuple.type = call %bound_method(%.loc29_7.2)
// CHECK:STDOUT:   %Point.as.Destroy.impl.Op.bound: <bound method> = bound_method %.loc29_5.2, constants.%Point.as.Destroy.impl.Op
// CHECK:STDOUT:   %addr: %ptr.536 = addr_of %.loc29_5.2
// CHECK:STDOUT:   %Point.as.Destroy.impl.Op.call: init %empty_tuple.type = call %Point.as.Destroy.impl.Op.bound(%addr)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Z.Zero(constants.%Self.6e6) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Z.Method(constants.%Self.6e6) {
// CHECK:STDOUT:   %Self => constants.%Self.6e6
// CHECK:STDOUT:   %Self.as_type.loc5_19.1 => constants.%Self.as_type.cd1
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.a40
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @<error>.as.Z.impl(constants.%Self.6e6) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%Self.6e6
// CHECK:STDOUT:   %<error>.as.Z.impl.Zero.type => constants.%<error>.as.Z.impl.Zero.type
// CHECK:STDOUT:   %<error>.as.Z.impl.Zero => constants.%<error>.as.Z.impl.Zero
// CHECK:STDOUT:   %<error>.as.Z.impl.Method.type => constants.%<error>.as.Z.impl.Method.type
// CHECK:STDOUT:   %<error>.as.Z.impl.Method => constants.%<error>.as.Z.impl.Method
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @<error>.as.Z.impl.Zero(constants.%Self.6e6) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @<error>.as.Z.impl.Method(constants.%Self.6e6) {
// CHECK:STDOUT:   %Self => constants.%Self.6e6
// CHECK:STDOUT:   %Self.as_type.loc13_21.1 => constants.%Self.as_type.cd1
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.a40
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Z.Zero(constants.%Z.facet) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Z.Method(constants.%Z.facet) {
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
// CHECK:STDOUT:   %A.B.type: type = fn_type @A.B [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %A.B: %A.B.type = struct_value () [concrete]
// CHECK:STDOUT:   %A.assoc_type: type = assoc_entity_type @A [concrete]
// CHECK:STDOUT:   %assoc0: %A.assoc_type = assoc_entity element0, @A.%A.B.decl [concrete]
// CHECK:STDOUT:   %C.type: type = facet_type <@C> [concrete]
// CHECK:STDOUT:   %Self.02e: %C.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %X: type = class_type @X [concrete]
// CHECK:STDOUT:   %A.impl_witness: <witness> = impl_witness @X.%A.impl_witness_table [concrete]
// CHECK:STDOUT:   %X.as.A.impl.B.type: type = fn_type @X.as.A.impl.B [concrete]
// CHECK:STDOUT:   %X.as.A.impl.B: %X.as.A.impl.B.type = struct_value () [concrete]
// CHECK:STDOUT:   %A.facet: %A.type = facet_value %X, (%A.impl_witness) [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %pattern_type.f6d: type = pattern_type auto [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness: <witness> = impl_witness @X.%Destroy.impl_witness_table [concrete]
// CHECK:STDOUT:   %ptr.d17: type = ptr_type %X [concrete]
// CHECK:STDOUT:   %pattern_type.1c6: type = pattern_type %ptr.d17 [concrete]
// CHECK:STDOUT:   %X.as.Destroy.impl.Op.type: type = fn_type @X.as.Destroy.impl.Op [concrete]
// CHECK:STDOUT:   %X.as.Destroy.impl.Op: %X.as.Destroy.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
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
// CHECK:STDOUT:   %A.B.decl: %A.B.type = fn_decl @A.B [concrete = constants.%A.B] {} {}
// CHECK:STDOUT:   %assoc0: %A.assoc_type = assoc_entity element0, %A.B.decl [concrete = constants.%assoc0]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .B = %assoc0
// CHECK:STDOUT:   witness = (%A.B.decl)
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
// CHECK:STDOUT: impl @X.as.A.impl: %Self.ref as %A.ref {
// CHECK:STDOUT:   impl_decl @<error>.as.C.impl [concrete] {} {
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %X.as.A.impl.B.decl: %X.as.A.impl.B.type = fn_decl @X.as.A.impl.B [concrete = constants.%X.as.A.impl.B] {} {}
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .C = <poisoned>
// CHECK:STDOUT:   .B = %X.as.A.impl.B.decl
// CHECK:STDOUT:   .G = <poisoned>
// CHECK:STDOUT:   witness = @X.%A.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @<error>.as.C.impl: <error> as %C.ref {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @X.as.Destroy.impl: constants.%X as constants.%Destroy.type {
// CHECK:STDOUT:   %X.as.Destroy.impl.Op.decl: %X.as.Destroy.impl.Op.type = fn_decl @X.as.Destroy.impl.Op [concrete = constants.%X.as.Destroy.impl.Op] {
// CHECK:STDOUT:     %self.patt: %pattern_type.1c6 = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.1c6 = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %.loc11: %pattern_type.f6d = addr_pattern %self.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %ptr.d17 = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%X [concrete = constants.%X]
// CHECK:STDOUT:     %self: %ptr.d17 = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Op = %X.as.Destroy.impl.Op.decl
// CHECK:STDOUT:   witness = @X.%Destroy.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @X {
// CHECK:STDOUT:   impl_decl @X.as.A.impl [concrete] {} {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%X [concrete = constants.%X]
// CHECK:STDOUT:     %A.ref: type = name_ref A, file.%A.decl [concrete = constants.%A.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %A.impl_witness_table = impl_witness_table (@X.as.A.impl.%X.as.A.impl.B.decl), @X.as.A.impl [concrete]
// CHECK:STDOUT:   %A.impl_witness: <witness> = impl_witness %A.impl_witness_table [concrete = constants.%A.impl_witness]
// CHECK:STDOUT:   impl_decl @X.as.Destroy.impl [concrete] {} {}
// CHECK:STDOUT:   %Destroy.impl_witness_table = impl_witness_table (@X.as.Destroy.impl.%X.as.Destroy.impl.Op.decl), @X.as.Destroy.impl [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness: <witness> = impl_witness %Destroy.impl_witness_table [concrete = constants.%Destroy.impl_witness]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%X
// CHECK:STDOUT:   .A = <poisoned>
// CHECK:STDOUT:   .C = <poisoned>
// CHECK:STDOUT:   .G = <poisoned>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @A.B(@A.%Self: %A.type) {
// CHECK:STDOUT:   fn();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @X.as.A.impl.B() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %G.ref: %G.type = name_ref G, file.%G.decl [concrete = constants.%G]
// CHECK:STDOUT:   %G.call: init %empty_tuple.type = call %G.ref()
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @X.as.Destroy.impl.Op(%self.param: %ptr.d17) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A.B(constants.%Self.31d) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A.B(constants.%A.facet) {}
// CHECK:STDOUT:
