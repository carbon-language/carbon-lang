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
  fn Method(self);

   // CHECK:STDERR: fail_impl_as_self_interface.carbon:[[@LINE+4]]:9: error: `impl as` can only be used in a class [ImplAsOutsideClass]
   // CHECK:STDERR:    impl as Z {
   // CHECK:STDERR:         ^~
   // CHECK:STDERR:
   impl as Z {
    fn Zero() {}
    fn Method(unused self) {}
  }
}

class Point {
   impl as Z {
    fn Zero() {}
    fn Method(unused self) {}
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
// CHECK:STDOUT:   %Self: %I.type = symbolic_binding Self, 0 [symbolic]
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
// CHECK:STDOUT:   %Self: %I.type = symbolic_binding Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   %I.WithSelf.decl = interface_with_self_decl @I [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @<error>.as.I.impl: <error> as %I.ref {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   extend %I.ref
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
// CHECK:STDOUT: specific @I.WithSelf(constants.%Self) {}
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_impl_as_function_scope.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %J.type: type = facet_type <@J> [concrete]
// CHECK:STDOUT:   %Self: %J.type = symbolic_binding Self, 0 [symbolic]
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
// CHECK:STDOUT:   %Self: %J.type = symbolic_binding Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   %J.WithSelf.decl = interface_with_self_decl @J [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @<error>.as.J.impl: <error> as %J.ref {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   extend %J.ref
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
// CHECK:STDOUT: specific @J.WithSelf(constants.%Self) {}
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_impl_as_self_interface.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Z.type: type = facet_type <@Z> [concrete]
// CHECK:STDOUT:   %Self.cb6: %Z.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %Z.WithSelf.Zero.type.68b: type = fn_type @Z.WithSelf.Zero, @Z.WithSelf(%Self.cb6) [symbolic]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Z.WithSelf.Zero.087: %Z.WithSelf.Zero.type.68b = struct_value () [symbolic]
// CHECK:STDOUT:   %Z.assoc_type: type = assoc_entity_type @Z [concrete]
// CHECK:STDOUT:   %assoc0.627: %Z.assoc_type = assoc_entity element0, @Z.WithSelf.%Z.WithSelf.Zero.decl [concrete]
// CHECK:STDOUT:   %Self.as_type.0cf: type = facet_access_type %Self.cb6 [symbolic]
// CHECK:STDOUT:   %pattern_type.52a: type = pattern_type %Self.as_type.0cf [symbolic]
// CHECK:STDOUT:   %self.param_patt.241: %pattern_type.52a = value_param_pattern [symbolic]
// CHECK:STDOUT:   %self.patt.104: %pattern_type.52a = at_binding_pattern self, %self.param_patt.241 [symbolic]
// CHECK:STDOUT:   %Z.WithSelf.Method.type.d41: type = fn_type @Z.WithSelf.Method, @Z.WithSelf(%Self.cb6) [symbolic]
// CHECK:STDOUT:   %Z.WithSelf.Method.ead: %Z.WithSelf.Method.type.d41 = struct_value () [symbolic]
// CHECK:STDOUT:   %assoc1: %Z.assoc_type = assoc_entity element1, @Z.WithSelf.%Z.WithSelf.Method.decl [concrete]
// CHECK:STDOUT:   %<error>.as.Z.impl.Zero.type: type = fn_type @<error>.as.Z.impl.Zero, @<error>.as.Z.impl(%Self.cb6) [symbolic]
// CHECK:STDOUT:   %<error>.as.Z.impl.Zero: %<error>.as.Z.impl.Zero.type = struct_value () [symbolic]
// CHECK:STDOUT:   %<error>.as.Z.impl.Method.type: type = fn_type @<error>.as.Z.impl.Method, @<error>.as.Z.impl(%Self.cb6) [symbolic]
// CHECK:STDOUT:   %<error>.as.Z.impl.Method: %<error>.as.Z.impl.Method.type = struct_value () [symbolic]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %Self.as_type.0cf [symbolic]
// CHECK:STDOUT:   %Point: type = class_type @Point [concrete]
// CHECK:STDOUT:   %Z.impl_witness: <witness> = impl_witness @Point.as.Z.impl.%Z.impl_witness_table [concrete]
// CHECK:STDOUT:   %Point.as.Z.impl.Zero.type: type = fn_type @Point.as.Z.impl.Zero [concrete]
// CHECK:STDOUT:   %Point.as.Z.impl.Zero: %Point.as.Z.impl.Zero.type = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.9d8: type = pattern_type %Point [concrete]
// CHECK:STDOUT:   %self.param_patt.b38: %pattern_type.9d8 = value_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.60a: %pattern_type.9d8 = at_binding_pattern self, %self.param_patt.b38 [concrete]
// CHECK:STDOUT:   %Point.as.Z.impl.Method.type: type = fn_type @Point.as.Z.impl.Method [concrete]
// CHECK:STDOUT:   %Point.as.Z.impl.Method: %Point.as.Z.impl.Method.type = struct_value () [concrete]
// CHECK:STDOUT:   %Z.facet: %Z.type = facet_value %Point, (%Z.impl_witness) [concrete]
// CHECK:STDOUT:   %Z.WithSelf.Zero.type.d05: type = fn_type @Z.WithSelf.Zero, @Z.WithSelf(%Z.facet) [concrete]
// CHECK:STDOUT:   %Z.WithSelf.Zero.0d2: %Z.WithSelf.Zero.type.d05 = struct_value () [concrete]
// CHECK:STDOUT:   %Z.WithSelf.Method.type.de2: type = fn_type @Z.WithSelf.Method, @Z.WithSelf(%Z.facet) [concrete]
// CHECK:STDOUT:   %Z.WithSelf.Method.683: %Z.WithSelf.Method.type.de2 = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %.1fe: type = fn_type_with_self_type %Z.WithSelf.Zero.type.d05, %Z.facet [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %Point.val: %Point = struct_value () [concrete]
// CHECK:STDOUT:   %.f5a: type = fn_type_with_self_type %Z.WithSelf.Method.type.de2, %Z.facet [concrete]
// CHECK:STDOUT:   %Point.as.Z.impl.Method.bound: <bound method> = bound_method %Point.val, %Point.as.Z.impl.Method [concrete]
// CHECK:STDOUT:   %.e6b: ref %Point = temporary invalid, %Point.val [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.1d8f74.2: type = fn_type @Destroy.Op.loc29_7.2 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1a2547.2: %Destroy.Op.type.1d8f74.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.Op.bound: <bound method> = bound_method %.e6b, %Destroy.Op.1a2547.2 [concrete]
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
// CHECK:STDOUT:   %Self: %Z.type = symbolic_binding Self, 0 [symbolic = constants.%Self.cb6]
// CHECK:STDOUT:   %Z.WithSelf.decl = interface_with_self_decl @Z [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !with Self:
// CHECK:STDOUT:   %Z.WithSelf.Zero.decl: @Z.WithSelf.%Z.WithSelf.Zero.type (%Z.WithSelf.Zero.type.68b) = fn_decl @Z.WithSelf.Zero [symbolic = @Z.WithSelf.%Z.WithSelf.Zero (constants.%Z.WithSelf.Zero.087)] {} {}
// CHECK:STDOUT:   %assoc0: %Z.assoc_type = assoc_entity element0, %Z.WithSelf.Zero.decl [concrete = constants.%assoc0.627]
// CHECK:STDOUT:   %Z.WithSelf.Method.decl: @Z.WithSelf.%Z.WithSelf.Method.type (%Z.WithSelf.Method.type.d41) = fn_decl @Z.WithSelf.Method [symbolic = @Z.WithSelf.%Z.WithSelf.Method (constants.%Z.WithSelf.Method.ead)] {
// CHECK:STDOUT:     %self.param_patt.loc5_13.1: @Z.WithSelf.Method.%pattern_type (%pattern_type.52a) = value_param_pattern [symbolic = %self.param_patt.loc5_13.2 (constants.%self.param_patt.241)]
// CHECK:STDOUT:     %self.patt.loc5_13.1: @Z.WithSelf.Method.%pattern_type (%pattern_type.52a) = at_binding_pattern self, %self.param_patt.loc5_13.1 [symbolic = %self.patt.loc5_13.2 (constants.%self.patt.104)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: @Z.WithSelf.Method.%Self.as_type.loc5_13.1 (%Self.as_type.0cf) = value_param call_param0
// CHECK:STDOUT:     %.loc5_13.1: type = splice_block %.loc5_13.2 [symbolic = %Self.as_type.loc5_13.1 (constants.%Self.as_type.0cf)] {
// CHECK:STDOUT:       %Self.ref: %Z.type = name_ref Self, @Z.%Self [symbolic = %Self (constants.%Self.cb6)]
// CHECK:STDOUT:       %Self.as_type.loc5_13.2: type = facet_access_type %Self.ref [symbolic = %Self.as_type.loc5_13.1 (constants.%Self.as_type.0cf)]
// CHECK:STDOUT:       %.loc5_13.2: type = converted %Self.ref, %Self.as_type.loc5_13.2 [symbolic = %Self.as_type.loc5_13.1 (constants.%Self.as_type.0cf)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %self: @Z.WithSelf.Method.%Self.as_type.loc5_13.1 (%Self.as_type.0cf) = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %assoc1: %Z.assoc_type = assoc_entity element1, %Z.WithSelf.Method.decl [concrete = constants.%assoc1]
// CHECK:STDOUT:   impl_decl @<error>.as.Z.impl [concrete] {} {
// CHECK:STDOUT:     %Z.ref: type = name_ref Z, file.%Z.decl [concrete = constants.%Z.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .Z = <poisoned>
// CHECK:STDOUT:   .Zero = @Z.WithSelf.%assoc0
// CHECK:STDOUT:   .Method = @Z.WithSelf.%assoc1
// CHECK:STDOUT:   .Z = <poisoned>
// CHECK:STDOUT:   witness = (@Z.WithSelf.%Z.WithSelf.Zero.decl, @Z.WithSelf.%Z.WithSelf.Method.decl)
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @<error>.as.Z.impl(@Z.%Self: %Z.type) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self: %Z.type = symbolic_binding Self, 0 [symbolic = %Self (constants.%Self.cb6)]
// CHECK:STDOUT:   %<error>.as.Z.impl.Zero.type: type = fn_type @<error>.as.Z.impl.Zero, @<error>.as.Z.impl(%Self) [symbolic = %<error>.as.Z.impl.Zero.type (constants.%<error>.as.Z.impl.Zero.type)]
// CHECK:STDOUT:   %<error>.as.Z.impl.Zero: @<error>.as.Z.impl.%<error>.as.Z.impl.Zero.type (%<error>.as.Z.impl.Zero.type) = struct_value () [symbolic = %<error>.as.Z.impl.Zero (constants.%<error>.as.Z.impl.Zero)]
// CHECK:STDOUT:   %<error>.as.Z.impl.Method.type: type = fn_type @<error>.as.Z.impl.Method, @<error>.as.Z.impl(%Self) [symbolic = %<error>.as.Z.impl.Method.type (constants.%<error>.as.Z.impl.Method.type)]
// CHECK:STDOUT:   %<error>.as.Z.impl.Method: @<error>.as.Z.impl.%<error>.as.Z.impl.Method.type (%<error>.as.Z.impl.Method.type) = struct_value () [symbolic = %<error>.as.Z.impl.Method (constants.%<error>.as.Z.impl.Method)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: <error> as %Z.ref {
// CHECK:STDOUT:     %<error>.as.Z.impl.Zero.decl: @<error>.as.Z.impl.%<error>.as.Z.impl.Zero.type (%<error>.as.Z.impl.Zero.type) = fn_decl @<error>.as.Z.impl.Zero [symbolic = @<error>.as.Z.impl.%<error>.as.Z.impl.Zero (constants.%<error>.as.Z.impl.Zero)] {} {}
// CHECK:STDOUT:     %<error>.as.Z.impl.Method.decl: @<error>.as.Z.impl.%<error>.as.Z.impl.Method.type (%<error>.as.Z.impl.Method.type) = fn_decl @<error>.as.Z.impl.Method [symbolic = @<error>.as.Z.impl.%<error>.as.Z.impl.Method (constants.%<error>.as.Z.impl.Method)] {
// CHECK:STDOUT:       %self.param_patt.loc13_22.1: @<error>.as.Z.impl.Method.%pattern_type (%pattern_type.52a) = value_param_pattern [symbolic = %self.param_patt.loc13_22.2 (constants.%self.param_patt.241)]
// CHECK:STDOUT:       %self.patt.loc13_22.1: @<error>.as.Z.impl.Method.%pattern_type (%pattern_type.52a) = at_binding_pattern self, %self.param_patt.loc13_22.1 [symbolic = %self.patt.loc13_22.2 (constants.%self.patt.104)]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %self.param: @<error>.as.Z.impl.Method.%Self.as_type.loc13_22.1 (%Self.as_type.0cf) = value_param call_param0
// CHECK:STDOUT:       %.loc13_22.1: type = splice_block %.loc13_22.2 [symbolic = %Self.as_type.loc13_22.1 (constants.%Self.as_type.0cf)] {
// CHECK:STDOUT:         %Self.ref: %Z.type = name_ref Self, @Z.%Self [symbolic = %Self (constants.%Self.cb6)]
// CHECK:STDOUT:         %Self.as_type.loc13_22.2: type = facet_access_type %Self.ref [symbolic = %Self.as_type.loc13_22.1 (constants.%Self.as_type.0cf)]
// CHECK:STDOUT:         %.loc13_22.2: type = converted %Self.ref, %Self.as_type.loc13_22.2 [symbolic = %Self.as_type.loc13_22.1 (constants.%Self.as_type.0cf)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %self: @<error>.as.Z.impl.Method.%Self.as_type.loc13_22.1 (%Self.as_type.0cf) = wrapper_binding self, %self.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Zero = %<error>.as.Z.impl.Zero.decl
// CHECK:STDOUT:     .Method = %<error>.as.Z.impl.Method.decl
// CHECK:STDOUT:     extend %Z.ref
// CHECK:STDOUT:     witness = <error>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @Point.as.Z.impl: %Self.ref as %Z.ref {
// CHECK:STDOUT:   %Point.as.Z.impl.Zero.decl: %Point.as.Z.impl.Zero.type = fn_decl @Point.as.Z.impl.Zero [concrete = constants.%Point.as.Z.impl.Zero] {} {}
// CHECK:STDOUT:   %Point.as.Z.impl.Method.decl: %Point.as.Z.impl.Method.type = fn_decl @Point.as.Z.impl.Method [concrete = constants.%Point.as.Z.impl.Method] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.9d8 = value_param_pattern [concrete = constants.%self.param_patt.b38]
// CHECK:STDOUT:     %self.patt: %pattern_type.9d8 = at_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.60a]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %Point = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%Point [concrete = constants.%Point]
// CHECK:STDOUT:     %self: %Point = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Z.impl_witness_table = impl_witness_table (%Point.as.Z.impl.Zero.decl, %Point.as.Z.impl.Method.decl), @Point.as.Z.impl [concrete]
// CHECK:STDOUT:   %Z.impl_witness: <witness> = impl_witness %Z.impl_witness_table [concrete = constants.%Z.impl_witness]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Zero = %Point.as.Z.impl.Zero.decl
// CHECK:STDOUT:   .Method = %Point.as.Z.impl.Method.decl
// CHECK:STDOUT:   extend %Z.ref
// CHECK:STDOUT:   witness = %Z.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Point {
// CHECK:STDOUT:   impl_decl @Point.as.Z.impl [concrete] {} {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%Point [concrete = constants.%Point]
// CHECK:STDOUT:     %Z.ref: type = name_ref Z, file.%Z.decl [concrete = constants.%Z.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Point
// CHECK:STDOUT:   .Z = <poisoned>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Z.WithSelf.Zero(@Z.%Self: %Z.type) {
// CHECK:STDOUT:   fn();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Z.WithSelf.Method(@Z.%Self: %Z.type) {
// CHECK:STDOUT:   %Self: %Z.type = symbolic_binding Self, 0 [symbolic = %Self (constants.%Self.cb6)]
// CHECK:STDOUT:   %Self.as_type.loc5_13.1: type = facet_access_type %Self [symbolic = %Self.as_type.loc5_13.1 (constants.%Self.as_type.0cf)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Self.as_type.loc5_13.1 [symbolic = %pattern_type (constants.%pattern_type.52a)]
// CHECK:STDOUT:   %self.param_patt.loc5_13.2: @Z.WithSelf.Method.%pattern_type (%pattern_type.52a) = value_param_pattern [symbolic = %self.param_patt.loc5_13.2 (constants.%self.param_patt.241)]
// CHECK:STDOUT:   %self.patt.loc5_13.2: @Z.WithSelf.Method.%pattern_type (%pattern_type.52a) = at_binding_pattern self, %self.param_patt.loc5_13.2 [symbolic = %self.patt.loc5_13.2 (constants.%self.patt.104)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @Z.WithSelf.Method.%Self.as_type.loc5_13.1 (%Self.as_type.0cf));
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
// CHECK:STDOUT:   %Self: %Z.type = symbolic_binding Self, 0 [symbolic = %Self (constants.%Self.cb6)]
// CHECK:STDOUT:   %Self.as_type.loc13_22.1: type = facet_access_type %Self [symbolic = %Self.as_type.loc13_22.1 (constants.%Self.as_type.0cf)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Self.as_type.loc13_22.1 [symbolic = %pattern_type (constants.%pattern_type.52a)]
// CHECK:STDOUT:   %self.param_patt.loc13_22.2: @<error>.as.Z.impl.Method.%pattern_type (%pattern_type.52a) = value_param_pattern [symbolic = %self.param_patt.loc13_22.2 (constants.%self.param_patt.241)]
// CHECK:STDOUT:   %self.patt.loc13_22.2: @<error>.as.Z.impl.Method.%pattern_type (%pattern_type.52a) = at_binding_pattern self, %self.param_patt.loc13_22.2 [symbolic = %self.patt.loc13_22.2 (constants.%self.patt.104)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %Self.as_type.loc13_22.1 [symbolic = %require_complete (constants.%require_complete)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @<error>.as.Z.impl.Method.%Self.as_type.loc13_22.1 (%Self.as_type.0cf)) {
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
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Point.ref.loc28: type = name_ref Point, file.%Point.decl [concrete = constants.%Point]
// CHECK:STDOUT:   %Z.ref.loc28: type = name_ref Z, file.%Z.decl [concrete = constants.%Z.type]
// CHECK:STDOUT:   %Zero.ref: %Z.assoc_type = name_ref Zero, @Z.WithSelf.%assoc0 [concrete = constants.%assoc0.627]
// CHECK:STDOUT:   %Z.facet: %Z.type = facet_value %Point.ref.loc28, (constants.%Z.impl_witness) [concrete = constants.%Z.facet]
// CHECK:STDOUT:   %.loc28: %Z.type = converted %Point.ref.loc28, %Z.facet [concrete = constants.%Z.facet]
// CHECK:STDOUT:   %impl.elem0: %.1fe = impl_witness_access constants.%Z.impl_witness, element0 [concrete = constants.%Point.as.Z.impl.Zero]
// CHECK:STDOUT:   %Point.as.Z.impl.Zero.call: init %empty_tuple.type = call %impl.elem0()
// CHECK:STDOUT:   %.loc29_5.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %Point.ref.loc29: type = name_ref Point, file.%Point.decl [concrete = constants.%Point]
// CHECK:STDOUT:   %.loc29_5.2: ref %Point = temporary_storage
// CHECK:STDOUT:   %.loc29_5.3: init %Point to %.loc29_5.2 = class_init () [concrete = constants.%Point.val]
// CHECK:STDOUT:   %.loc29_7.1: init %Point = converted %.loc29_5.1, %.loc29_5.3 [concrete = constants.%Point.val]
// CHECK:STDOUT:   %Z.ref.loc29: type = name_ref Z, file.%Z.decl [concrete = constants.%Z.type]
// CHECK:STDOUT:   %Method.ref: %Z.assoc_type = name_ref Method, @Z.WithSelf.%assoc1 [concrete = constants.%assoc1]
// CHECK:STDOUT:   %impl.elem1: %.f5a = impl_witness_access constants.%Z.impl_witness, element1 [concrete = constants.%Point.as.Z.impl.Method]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %.loc29_7.1, %impl.elem1 [concrete = constants.%Point.as.Z.impl.Method.bound]
// CHECK:STDOUT:   %.loc29_7.2: ref %Point = temporary %.loc29_5.2, %.loc29_7.1 [concrete = constants.%.e6b]
// CHECK:STDOUT:   %.loc29_7.3: %Point = acquire_value %.loc29_7.2 [concrete = constants.%Point.val]
// CHECK:STDOUT:   %Point.as.Z.impl.Method.call: init %empty_tuple.type = call %bound_method(%.loc29_7.3)
// CHECK:STDOUT:   %Destroy.Op.call: init %empty_tuple.type = call constants.%Destroy.Op.bound(constants.%.e6b)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc29_7.1(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc29_7.2(%self.param: ref %Point) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Z.WithSelf(constants.%Self.cb6) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%Self.cb6
// CHECK:STDOUT:   %Z.WithSelf.Zero.type => constants.%Z.WithSelf.Zero.type.68b
// CHECK:STDOUT:   %Z.WithSelf.Zero => constants.%Z.WithSelf.Zero.087
// CHECK:STDOUT:   %Z.WithSelf.Method.type => constants.%Z.WithSelf.Method.type.d41
// CHECK:STDOUT:   %Z.WithSelf.Method => constants.%Z.WithSelf.Method.ead
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Z.WithSelf.Zero(constants.%Self.cb6) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Z.WithSelf.Method(constants.%Self.cb6) {
// CHECK:STDOUT:   %Self => constants.%Self.cb6
// CHECK:STDOUT:   %Self.as_type.loc5_13.1 => constants.%Self.as_type.0cf
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.52a
// CHECK:STDOUT:   %self.param_patt.loc5_13.2 => constants.%self.param_patt.241
// CHECK:STDOUT:   %self.patt.loc5_13.2 => constants.%self.patt.104
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @<error>.as.Z.impl(constants.%Self.cb6) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%Self.cb6
// CHECK:STDOUT:   %<error>.as.Z.impl.Zero.type => constants.%<error>.as.Z.impl.Zero.type
// CHECK:STDOUT:   %<error>.as.Z.impl.Zero => constants.%<error>.as.Z.impl.Zero
// CHECK:STDOUT:   %<error>.as.Z.impl.Method.type => constants.%<error>.as.Z.impl.Method.type
// CHECK:STDOUT:   %<error>.as.Z.impl.Method => constants.%<error>.as.Z.impl.Method
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @<error>.as.Z.impl.Zero(constants.%Self.cb6) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @<error>.as.Z.impl.Method(constants.%Self.cb6) {
// CHECK:STDOUT:   %Self => constants.%Self.cb6
// CHECK:STDOUT:   %Self.as_type.loc13_22.1 => constants.%Self.as_type.0cf
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.52a
// CHECK:STDOUT:   %self.param_patt.loc13_22.2 => constants.%self.param_patt.241
// CHECK:STDOUT:   %self.patt.loc13_22.2 => constants.%self.patt.104
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Z.WithSelf(constants.%Z.facet) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%Z.facet
// CHECK:STDOUT:   %Z.WithSelf.Zero.type => constants.%Z.WithSelf.Zero.type.d05
// CHECK:STDOUT:   %Z.WithSelf.Zero => constants.%Z.WithSelf.Zero.0d2
// CHECK:STDOUT:   %Z.WithSelf.Method.type => constants.%Z.WithSelf.Method.type.de2
// CHECK:STDOUT:   %Z.WithSelf.Method => constants.%Z.WithSelf.Method.683
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Z.WithSelf.Zero(constants.%Z.facet) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Z.WithSelf.Method(constants.%Z.facet) {
// CHECK:STDOUT:   %Self => constants.%Z.facet
// CHECK:STDOUT:   %Self.as_type.loc5_13.1 => constants.%Point
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.9d8
// CHECK:STDOUT:   %self.param_patt.loc5_13.2 => constants.%self.param_patt.b38
// CHECK:STDOUT:   %self.patt.loc5_13.2 => constants.%self.patt.60a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_impl_as_other_interface.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %A.type: type = facet_type <@A> [concrete]
// CHECK:STDOUT:   %Self.d0c: %A.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %A.WithSelf.B.type.ab5: type = fn_type @A.WithSelf.B, @A.WithSelf(%Self.d0c) [symbolic]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %A.WithSelf.B.a0b: %A.WithSelf.B.type.ab5 = struct_value () [symbolic]
// CHECK:STDOUT:   %A.assoc_type: type = assoc_entity_type @A [concrete]
// CHECK:STDOUT:   %assoc0: %A.assoc_type = assoc_entity element0, @A.WithSelf.%A.WithSelf.B.decl [concrete]
// CHECK:STDOUT:   %C.type: type = facet_type <@C> [concrete]
// CHECK:STDOUT:   %Self.62a: %C.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %X: type = class_type @X [concrete]
// CHECK:STDOUT:   %A.impl_witness: <witness> = impl_witness @X.as.A.impl.%A.impl_witness_table [concrete]
// CHECK:STDOUT:   %X.as.A.impl.B.type: type = fn_type @X.as.A.impl.B [concrete]
// CHECK:STDOUT:   %X.as.A.impl.B: %X.as.A.impl.B.type = struct_value () [concrete]
// CHECK:STDOUT:   %A.facet: %A.type = facet_value %X, (%A.impl_witness) [concrete]
// CHECK:STDOUT:   %A.WithSelf.B.type.d09: type = fn_type @A.WithSelf.B, @A.WithSelf(%A.facet) [concrete]
// CHECK:STDOUT:   %A.WithSelf.B.e21: %A.WithSelf.B.type.d09 = struct_value () [concrete]
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
// CHECK:STDOUT:   %Self: %A.type = symbolic_binding Self, 0 [symbolic = constants.%Self.d0c]
// CHECK:STDOUT:   %A.WithSelf.decl = interface_with_self_decl @A [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !with Self:
// CHECK:STDOUT:   %A.WithSelf.B.decl: @A.WithSelf.%A.WithSelf.B.type (%A.WithSelf.B.type.ab5) = fn_decl @A.WithSelf.B [symbolic = @A.WithSelf.%A.WithSelf.B (constants.%A.WithSelf.B.a0b)] {} {}
// CHECK:STDOUT:   %assoc0: %A.assoc_type = assoc_entity element0, %A.WithSelf.B.decl [concrete = constants.%assoc0]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .B = @A.WithSelf.%assoc0
// CHECK:STDOUT:   .C = <poisoned>
// CHECK:STDOUT:   .G = <poisoned>
// CHECK:STDOUT:   witness = (@A.WithSelf.%A.WithSelf.B.decl)
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @C {
// CHECK:STDOUT:   %Self: %C.type = symbolic_binding Self, 0 [symbolic = constants.%Self.62a]
// CHECK:STDOUT:   %C.WithSelf.decl = interface_with_self_decl @C [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @X.as.A.impl: %Self.ref as %A.ref {
// CHECK:STDOUT:   impl_decl @<error>.as.C.impl [concrete] {} {
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %X.as.A.impl.B.decl: %X.as.A.impl.B.type = fn_decl @X.as.A.impl.B [concrete = constants.%X.as.A.impl.B] {} {}
// CHECK:STDOUT:   %A.impl_witness_table = impl_witness_table (%X.as.A.impl.B.decl), @X.as.A.impl [concrete]
// CHECK:STDOUT:   %A.impl_witness: <witness> = impl_witness %A.impl_witness_table [concrete = constants.%A.impl_witness]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .C = <poisoned>
// CHECK:STDOUT:   .B = %X.as.A.impl.B.decl
// CHECK:STDOUT:   .G = <poisoned>
// CHECK:STDOUT:   extend %A.ref
// CHECK:STDOUT:   witness = %A.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @<error>.as.C.impl: <error> as %C.ref {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   extend %C.ref
// CHECK:STDOUT:   witness = <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @X {
// CHECK:STDOUT:   impl_decl @X.as.A.impl [concrete] {} {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%X [concrete = constants.%X]
// CHECK:STDOUT:     %A.ref: type = name_ref A, file.%A.decl [concrete = constants.%A.type]
// CHECK:STDOUT:   }
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
// CHECK:STDOUT: generic fn @A.WithSelf.B(@A.%Self: %A.type) {
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
// CHECK:STDOUT: specific @A.WithSelf(constants.%Self.d0c) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%Self.d0c
// CHECK:STDOUT:   %A.WithSelf.B.type => constants.%A.WithSelf.B.type.ab5
// CHECK:STDOUT:   %A.WithSelf.B => constants.%A.WithSelf.B.a0b
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A.WithSelf.B(constants.%Self.d0c) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C.WithSelf(constants.%Self.62a) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A.WithSelf(constants.%A.facet) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%A.facet
// CHECK:STDOUT:   %A.WithSelf.B.type => constants.%A.WithSelf.B.type.d09
// CHECK:STDOUT:   %A.WithSelf.B => constants.%A.WithSelf.B.e21
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A.WithSelf.B(constants.%A.facet) {}
// CHECK:STDOUT:
