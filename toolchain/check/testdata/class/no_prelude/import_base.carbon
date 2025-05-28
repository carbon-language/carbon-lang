// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// EXTRA-ARGS: --dump-sem-ir-ranges=only
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/no_prelude/import_base.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/no_prelude/import_base.carbon

// --- a.carbon

library "[[@TEST_NAME]]";

base class Base {
  fn F[self: Self]();
  fn Unused[self: Self]();

  var x: ();
  var unused: ();
}

class Child {
  extend base: Base;
}

// --- b.carbon

library "[[@TEST_NAME]]";

import library "a";

fn Run() {
  var a: Child = {.base = {.x = (), .unused = ()}};
  //@dump-sem-ir-begin
  a.x = ();
  a.F();
  //@dump-sem-ir-end
}

// CHECK:STDOUT: --- b.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Child: type = class_type @Child [concrete]
// CHECK:STDOUT:   %Base: type = class_type @Base [concrete]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %Base.elem: type = unbound_element_type %Base, %empty_tuple.type [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.import_ref.e8f: %F.type = import_ref Main//a, loc5_21, loaded [concrete = constants.%F]
// CHECK:STDOUT:   %Main.import_ref.f99: %Base.elem = import_ref Main//a, loc8_8, loaded [concrete = %.6dd]
// CHECK:STDOUT:   %.6dd: %Base.elem = field_decl x, element0 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Run() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %a.ref.loc9: ref %Child = name_ref a, %a
// CHECK:STDOUT:   %x.ref: %Base.elem = name_ref x, imports.%Main.import_ref.f99 [concrete = imports.%.6dd]
// CHECK:STDOUT:   %.loc9_4.1: ref %Base = class_element_access %a.ref.loc9, element0
// CHECK:STDOUT:   %.loc9_4.2: ref %Base = converted %a.ref.loc9, %.loc9_4.1
// CHECK:STDOUT:   %.loc9_4.3: ref %empty_tuple.type = class_element_access %.loc9_4.2, element0
// CHECK:STDOUT:   %.loc9_10.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %.loc9_10.2: init %empty_tuple.type = tuple_init () to %.loc9_4.3 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc9_7: init %empty_tuple.type = converted %.loc9_10.1, %.loc9_10.2 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   assign %.loc9_4.3, %.loc9_7
// CHECK:STDOUT:   %a.ref.loc10: ref %Child = name_ref a, %a
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, imports.%Main.import_ref.e8f [concrete = constants.%F]
// CHECK:STDOUT:   %F.bound: <bound method> = bound_method %a.ref.loc10, %F.ref
// CHECK:STDOUT:   %.loc10_3.1: ref %Base = class_element_access %a.ref.loc10, element0
// CHECK:STDOUT:   %.loc10_3.2: ref %Base = converted %a.ref.loc10, %.loc10_3.1
// CHECK:STDOUT:   %.loc10_3.3: %Base = bind_value %.loc10_3.2
// CHECK:STDOUT:   %F.call: init %empty_tuple.type = call %F.bound(%.loc10_3.3)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
