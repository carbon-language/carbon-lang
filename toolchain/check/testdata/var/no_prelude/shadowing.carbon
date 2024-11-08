// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/var/no_prelude/shadowing.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/var/no_prelude/shadowing.carbon

namespace NS;

fn Main() {
  var NS: () = ();
  NS = ();

  var x: () = ();
  if (true) {
    var x: () = ();

    // TODO: We should reject this use of the shadowed variable `x`.
    x = ();
  }
}

// CHECK:STDOUT: --- shadowing.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Main.type: type = fn_type @Main [template]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [template]
// CHECK:STDOUT:   %Main: %Main.type = struct_value () [template]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [template]
// CHECK:STDOUT:   %.1: bool = bool_literal true [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .NS = %NS
// CHECK:STDOUT:     .Main = %Main.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %NS: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %Main.decl: %Main.type = fn_decl @Main [template = constants.%Main] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Main() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc14_12.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %.loc14_12.2: type = converted %.loc14_12.1, constants.%empty_tuple.type [template = constants.%empty_tuple.type]
// CHECK:STDOUT:   %NS.var: ref %empty_tuple.type = var NS
// CHECK:STDOUT:   %NS: ref %empty_tuple.type = bind_name NS, %NS.var
// CHECK:STDOUT:   %.loc14_17.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %.loc14_17.2: init %empty_tuple.type = tuple_init () to %NS.var [template = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc14_18: init %empty_tuple.type = converted %.loc14_17.1, %.loc14_17.2 [template = constants.%empty_tuple]
// CHECK:STDOUT:   assign %NS.var, %.loc14_18
// CHECK:STDOUT:   %NS.ref: ref %empty_tuple.type = name_ref NS, %NS
// CHECK:STDOUT:   %.loc15_9.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %.loc15_9.2: init %empty_tuple.type = tuple_init () to %NS.ref [template = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc15_6: init %empty_tuple.type = converted %.loc15_9.1, %.loc15_9.2 [template = constants.%empty_tuple]
// CHECK:STDOUT:   assign %NS.ref, %.loc15_6
// CHECK:STDOUT:   %.loc17_11.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %.loc17_11.2: type = converted %.loc17_11.1, constants.%empty_tuple.type [template = constants.%empty_tuple.type]
// CHECK:STDOUT:   %x.var.loc17: ref %empty_tuple.type = var x
// CHECK:STDOUT:   %x.loc17: ref %empty_tuple.type = bind_name x, %x.var.loc17
// CHECK:STDOUT:   %.loc17_16.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %.loc17_16.2: init %empty_tuple.type = tuple_init () to %x.var.loc17 [template = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc17_17: init %empty_tuple.type = converted %.loc17_16.1, %.loc17_16.2 [template = constants.%empty_tuple]
// CHECK:STDOUT:   assign %x.var.loc17, %.loc17_17
// CHECK:STDOUT:   %.loc18: bool = bool_literal true [template = constants.%.1]
// CHECK:STDOUT:   if %.loc18 br !if.then else br !if.else
// CHECK:STDOUT:
// CHECK:STDOUT: !if.then:
// CHECK:STDOUT:   %.loc19_13.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %.loc19_13.2: type = converted %.loc19_13.1, constants.%empty_tuple.type [template = constants.%empty_tuple.type]
// CHECK:STDOUT:   %x.var.loc19: ref %empty_tuple.type = var x
// CHECK:STDOUT:   %x.loc19: ref %empty_tuple.type = bind_name x, %x.var.loc19
// CHECK:STDOUT:   %.loc19_18.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %.loc19_18.2: init %empty_tuple.type = tuple_init () to %x.var.loc19 [template = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc19_19: init %empty_tuple.type = converted %.loc19_18.1, %.loc19_18.2 [template = constants.%empty_tuple]
// CHECK:STDOUT:   assign %x.var.loc19, %.loc19_19
// CHECK:STDOUT:   %x.ref: ref %empty_tuple.type = name_ref x, %x.loc19
// CHECK:STDOUT:   %.loc22_10.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %.loc22_10.2: init %empty_tuple.type = tuple_init () to %x.ref [template = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc22_7: init %empty_tuple.type = converted %.loc22_10.1, %.loc22_10.2 [template = constants.%empty_tuple]
// CHECK:STDOUT:   assign %x.ref, %.loc22_7
// CHECK:STDOUT:   br !if.else
// CHECK:STDOUT:
// CHECK:STDOUT: !if.else:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
