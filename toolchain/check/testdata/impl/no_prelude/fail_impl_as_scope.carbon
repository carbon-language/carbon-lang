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

// CHECK:STDERR: fail_impl_as_file_scope.carbon:[[@LINE+4]]:6: error: `impl as` can only be used in a class [ImplAsOutsideClass]
// CHECK:STDERR: impl as I {}
// CHECK:STDERR:      ^~
// CHECK:STDERR:
impl as I {}

// --- fail_impl_as_function_scope.carbon
library "[[@TEST_NAME]]";

interface J {}

fn F() {
  // CHECK:STDERR: fail_impl_as_function_scope.carbon:[[@LINE+4]]:8: error: `impl as` can only be used in a class [ImplAsOutsideClass]
  // CHECK:STDERR:   impl as J {}
  // CHECK:STDERR:        ^~
  // CHECK:STDERR:
  impl as J {}
}

// --- fail_impl_as_self_interface.carbon
library "[[@TEST_NAME]]";

interface Z {
  fn Zero();

   // CHECK:STDERR: fail_impl_as_self_interface.carbon:[[@LINE+4]]:9: error: `impl as` can only be used in a class [ImplAsOutsideClass]
   // CHECK:STDERR:    impl as Z {
   // CHECK:STDERR:         ^~
   // CHECK:STDERR:
   impl as Z {
    fn Zero() {}
  }
}

class Point {
   impl as Z {
    fn Zero() {}
  }
}

fn F() {
  // Even if the `impl` is diagnosed above, we must not add the impl of the
  // interface to itself in a way that allows it to be used during impl lookup,
  // or we end up with infinite impl lookup recursion here.
  ({} as Point).(Z.Zero)();
}

// --- fail_impl_as_other_interface.carbon

library "[[@TEST_NAME]]";

interface A {
  fn B();
}

class X {
  impl as A {
    fn B() {}
    // CHECK:STDERR: fail_impl_as_other_interface.carbon:[[@LINE+8]]:10: error: `impl as` can only be used in a class [ImplAsOutsideClass]
    // CHECK:STDERR:     impl as B {}
    // CHECK:STDERR:          ^~
    // CHECK:STDERR:
    // CHECK:STDERR: fail_impl_as_other_interface.carbon:[[@LINE+4]]:13: error: expression cannot be used as a value [UseOfNonExprAsValue]
    // CHECK:STDERR:     impl as B {}
    // CHECK:STDERR:             ^
    // CHECK:STDERR:
    impl as B {}
  }
}

// CHECK:STDOUT: --- fail_impl_as_file_scope.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [template]
// CHECK:STDOUT:   %Self: %I.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .I = %I.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %I.decl: type = interface_decl @I [template = constants.%I.type] {} {}
// CHECK:STDOUT:   impl_decl @impl [template] {} {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, <error> [template = <error>]
// CHECK:STDOUT:     %I.ref: type = name_ref I, file.%I.decl [template = constants.%I.type]
// CHECK:STDOUT:   }
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
// CHECK:STDOUT: --- fail_impl_as_function_scope.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %J.type: type = facet_type <@J> [template]
// CHECK:STDOUT:   %Self: %J.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .J = %J.decl
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %J.decl: type = interface_decl @J [template = constants.%J.type] {} {}
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [template = constants.%F] {} {}
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
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   impl_decl @impl [template] {} {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, <error> [template = <error>]
// CHECK:STDOUT:     %J.ref: type = name_ref J, file.%J.decl [template = constants.%J.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_impl_as_self_interface.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Z.type: type = facet_type <@Z> [template]
// CHECK:STDOUT:   %Self: %Z.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %Zero.type.822: type = fn_type @Zero.1 [template]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [template]
// CHECK:STDOUT:   %Zero.d14: %Zero.type.822 = struct_value () [template]
// CHECK:STDOUT:   %Z.assoc_type: type = assoc_entity_type %Z.type [template]
// CHECK:STDOUT:   %assoc0: %Z.assoc_type = assoc_entity element0, @Z.%Zero.decl [template]
// CHECK:STDOUT:   %Zero.type.db4: type = fn_type @Zero.2, @impl.1(%Self) [symbolic]
// CHECK:STDOUT:   %Zero.8fb: %Zero.type.db4 = struct_value () [symbolic]
// CHECK:STDOUT:   %Point: type = class_type @Point [template]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (@impl.2.%Zero.decl) [template]
// CHECK:STDOUT:   %Zero.type.e33: type = fn_type @Zero.3 [template]
// CHECK:STDOUT:   %Zero.dec: %Zero.type.e33 = struct_value () [template]
// CHECK:STDOUT:   %Z.facet: %Z.type = facet_value %Point, %impl_witness [template]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [template]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT:   %Point.val: %Point = struct_value () [template]
// CHECK:STDOUT:   %.c37: type = fn_type_with_self_type %Zero.type.822, %Z.facet [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Z = %Z.decl
// CHECK:STDOUT:     .Point = %Point.decl
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Z.decl: type = interface_decl @Z [template = constants.%Z.type] {} {}
// CHECK:STDOUT:   %Point.decl: type = class_decl @Point [template = constants.%Point] {} {}
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [template = constants.%F] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Z {
// CHECK:STDOUT:   %Self: %Z.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   %Zero.decl: %Zero.type.822 = fn_decl @Zero.1 [template = constants.%Zero.d14] {} {}
// CHECK:STDOUT:   %assoc0: %Z.assoc_type = assoc_entity element0, %Zero.decl [template = constants.%assoc0]
// CHECK:STDOUT:   impl_decl @impl.1 [template] {} {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, <error> [template = <error>]
// CHECK:STDOUT:     %Z.ref: type = name_ref Z, file.%Z.decl [template = constants.%Z.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .Zero = %assoc0
// CHECK:STDOUT:   witness = (%Zero.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl.1(@Z.%Self: %Z.type) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self: %Z.type = bind_symbolic_name Self, 0 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:   %Zero.type: type = fn_type @Zero.2, @impl.1(%Self) [symbolic = %Zero.type (constants.%Zero.type.db4)]
// CHECK:STDOUT:   %Zero: @impl.1.%Zero.type (%Zero.type.db4) = struct_value () [symbolic = %Zero (constants.%Zero.8fb)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %Self.ref as %Z.ref {
// CHECK:STDOUT:     %Zero.decl: @impl.1.%Zero.type (%Zero.type.db4) = fn_decl @Zero.2 [symbolic = @impl.1.%Zero (constants.%Zero.8fb)] {} {}
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Zero = %Zero.decl
// CHECK:STDOUT:     witness = <error>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.2: %Self.ref as %Z.ref {
// CHECK:STDOUT:   %Zero.decl: %Zero.type.e33 = fn_decl @Zero.3 [template = constants.%Zero.dec] {} {}
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Zero = %Zero.decl
// CHECK:STDOUT:   witness = @Point.%impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Point {
// CHECK:STDOUT:   impl_decl @impl.2 [template] {} {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%Point [template = constants.%Point]
// CHECK:STDOUT:     %Z.ref: type = name_ref Z, file.%Z.decl [template = constants.%Z.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (@impl.2.%Zero.decl) [template = constants.%impl_witness]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Point
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Zero.1(@Z.%Self: %Z.type) {
// CHECK:STDOUT:   fn();
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
// CHECK:STDOUT: fn @Zero.3() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc25_5.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %Point.ref: type = name_ref Point, file.%Point.decl [template = constants.%Point]
// CHECK:STDOUT:   %.loc25_5.2: ref %Point = temporary_storage
// CHECK:STDOUT:   %.loc25_5.3: init %Point = class_init (), %.loc25_5.2 [template = constants.%Point.val]
// CHECK:STDOUT:   %.loc25_5.4: ref %Point = temporary %.loc25_5.2, %.loc25_5.3
// CHECK:STDOUT:   %.loc25_7: ref %Point = converted %.loc25_5.1, %.loc25_5.4
// CHECK:STDOUT:   %Z.ref: type = name_ref Z, file.%Z.decl [template = constants.%Z.type]
// CHECK:STDOUT:   %Zero.ref: %Z.assoc_type = name_ref Zero, @Z.%assoc0 [template = constants.%assoc0]
// CHECK:STDOUT:   %impl.elem0: %.c37 = impl_witness_access constants.%impl_witness, element0 [template = constants.%Zero.dec]
// CHECK:STDOUT:   %Zero.call: init %empty_tuple.type = call %impl.elem0()
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Zero.1(constants.%Self) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.1(constants.%Self) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT:   %Zero.type => constants.%Zero.type.db4
// CHECK:STDOUT:   %Zero => constants.%Zero.8fb
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Zero.2(constants.%Self) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.1(%Self) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Zero.1(constants.%Z.facet) {}
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_impl_as_other_interface.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %A.type: type = facet_type <@A> [template]
// CHECK:STDOUT:   %Self: %A.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %B.type.1c3: type = fn_type @B.1 [template]
// CHECK:STDOUT:   %B.b08: %B.type.1c3 = struct_value () [template]
// CHECK:STDOUT:   %A.assoc_type: type = assoc_entity_type %A.type [template]
// CHECK:STDOUT:   %assoc0: %A.assoc_type = assoc_entity element0, @A.%B.decl [template]
// CHECK:STDOUT:   %X: type = class_type @X [template]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (@impl.1.%B.decl) [template]
// CHECK:STDOUT:   %B.type.d47: type = fn_type @B.2 [template]
// CHECK:STDOUT:   %B.4af: %B.type.d47 = struct_value () [template]
// CHECK:STDOUT:   %A.facet: %A.type = facet_value %X, %impl_witness [template]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [template]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .A = %A.decl
// CHECK:STDOUT:     .X = %X.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %A.decl: type = interface_decl @A [template = constants.%A.type] {} {}
// CHECK:STDOUT:   %X.decl: type = class_decl @X [template = constants.%X] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @A {
// CHECK:STDOUT:   %Self: %A.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   %B.decl: %B.type.1c3 = fn_decl @B.1 [template = constants.%B.b08] {} {}
// CHECK:STDOUT:   %assoc0: %A.assoc_type = assoc_entity element0, %B.decl [template = constants.%assoc0]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .B = %assoc0
// CHECK:STDOUT:   witness = (%B.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.1: %Self.ref as %A.ref {
// CHECK:STDOUT:   %B.decl: %B.type.d47 = fn_decl @B.2 [template = constants.%B.4af] {} {}
// CHECK:STDOUT:   impl_decl @impl.2 [template] {} {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, <error> [template = <error>]
// CHECK:STDOUT:     %B.ref: %B.type.d47 = name_ref B, @impl.1.%B.decl [template = constants.%B.4af]
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .B = %B.decl
// CHECK:STDOUT:   witness = @X.%impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.2: %Self.ref as <error> {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @X {
// CHECK:STDOUT:   impl_decl @impl.1 [template] {} {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%X [template = constants.%X]
// CHECK:STDOUT:     %A.ref: type = name_ref A, file.%A.decl [template = constants.%A.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (@impl.1.%B.decl) [template = constants.%impl_witness]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%X
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @B.1(@A.%Self: %A.type) {
// CHECK:STDOUT:   fn();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @B.2() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @B.1(constants.%Self) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @B.1(constants.%A.facet) {}
// CHECK:STDOUT:
