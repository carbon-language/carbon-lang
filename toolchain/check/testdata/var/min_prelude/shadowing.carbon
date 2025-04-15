// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/min_prelude/destroy.carbon
// EXTRA-ARGS: --custom-core
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/var/min_prelude/shadowing.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/var/min_prelude/shadowing.carbon

// --- basics.carbon

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
// CHECK:STDOUT:   %Main.type: type = fn_type @Main [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Main: %Main.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %true: bool = bool_literal true [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .NS = %NS
// CHECK:STDOUT:     .Main = %Main.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %NS: <namespace> = namespace [concrete] {}
// CHECK:STDOUT:   %Main.decl: %Main.type = fn_decl @Main [concrete = constants.%Main] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Main() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %NS.patt: %empty_tuple.type = binding_pattern NS
// CHECK:STDOUT:     %.loc17_3.1: %empty_tuple.type = var_pattern %NS.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %NS.var: ref %empty_tuple.type = var NS
// CHECK:STDOUT:   %.loc17_17.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %.loc17_17.2: init %empty_tuple.type = tuple_init () to %NS.var [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc17_3.2: init %empty_tuple.type = converted %.loc17_17.1, %.loc17_17.2 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   assign %NS.var, %.loc17_3.2
// CHECK:STDOUT:   %.loc17_12.1: type = splice_block %.loc17_12.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:     %.loc17_12.2: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc17_12.3: type = converted %.loc17_12.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %NS: ref %empty_tuple.type = bind_name NS, %NS.var
// CHECK:STDOUT:   %NS.ref: ref %empty_tuple.type = name_ref NS, %NS
// CHECK:STDOUT:   %.loc18_9.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %.loc18_9.2: init %empty_tuple.type = tuple_init () to %NS.ref [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc18_6: init %empty_tuple.type = converted %.loc18_9.1, %.loc18_9.2 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   assign %NS.ref, %.loc18_6
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt.loc20: %empty_tuple.type = binding_pattern x
// CHECK:STDOUT:     %.loc20_3.1: %empty_tuple.type = var_pattern %x.patt.loc20
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %x.var.loc20: ref %empty_tuple.type = var x
// CHECK:STDOUT:   %.loc20_16.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %.loc20_16.2: init %empty_tuple.type = tuple_init () to %x.var.loc20 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc20_3.2: init %empty_tuple.type = converted %.loc20_16.1, %.loc20_16.2 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   assign %x.var.loc20, %.loc20_3.2
// CHECK:STDOUT:   %.loc20_11.1: type = splice_block %.loc20_11.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:     %.loc20_11.2: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc20_11.3: type = converted %.loc20_11.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %x.loc20: ref %empty_tuple.type = bind_name x, %x.var.loc20
// CHECK:STDOUT:   %true: bool = bool_literal true [concrete = constants.%true]
// CHECK:STDOUT:   if %true br !if.then else br !if.else
// CHECK:STDOUT:
// CHECK:STDOUT: !if.then:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt.loc22: %empty_tuple.type = binding_pattern x
// CHECK:STDOUT:     %.loc22_5.1: %empty_tuple.type = var_pattern %x.patt.loc22
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %x.var.loc22: ref %empty_tuple.type = var x
// CHECK:STDOUT:   %.loc22_18.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %.loc22_18.2: init %empty_tuple.type = tuple_init () to %x.var.loc22 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc22_5.2: init %empty_tuple.type = converted %.loc22_18.1, %.loc22_18.2 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   assign %x.var.loc22, %.loc22_5.2
// CHECK:STDOUT:   %.loc22_13.1: type = splice_block %.loc22_13.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:     %.loc22_13.2: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc22_13.3: type = converted %.loc22_13.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %x.loc22: ref %empty_tuple.type = bind_name x, %x.var.loc22
// CHECK:STDOUT:   %x.ref: ref %empty_tuple.type = name_ref x, %x.loc22
// CHECK:STDOUT:   %.loc25_10.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %.loc25_10.2: init %empty_tuple.type = tuple_init () to %x.ref [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc25_7: init %empty_tuple.type = converted %.loc25_10.1, %.loc25_10.2 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   assign %x.ref, %.loc25_7
// CHECK:STDOUT:   br !if.else
// CHECK:STDOUT:
// CHECK:STDOUT: !if.else:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- include_files/destroy.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
