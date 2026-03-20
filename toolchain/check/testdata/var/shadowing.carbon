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
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/var/shadowing.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/var/shadowing.carbon

// --- basics.carbon

namespace NS;

fn Main() {
  var NS: () = ();
  NS = ();

  var unused x: () = ();
  if (true) {
    var x: () = ();

    // TODO: We should reject this use of the shadowed variable `x`.
    x = ();
  }
}

// CHECK:STDOUT: --- basics.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Main.type: type = fn_type @Main [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Main: %Main.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %pattern_type.cb1: type = pattern_type %empty_tuple.type [concrete]
// CHECK:STDOUT:   %true: bool = bool_literal true [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %Destroy.Op.type: type = fn_type @Destroy.Op [concrete]
// CHECK:STDOUT:   %Destroy.Op: %Destroy.Op.type = struct_value () [concrete]
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
// CHECK:STDOUT:     %NS.patt: %pattern_type.cb1 = ref_binding_pattern NS [concrete]
// CHECK:STDOUT:     %NS.var_patt: %pattern_type.cb1 = var_pattern %NS.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %NS.var: ref %empty_tuple.type = var %NS.var_patt
// CHECK:STDOUT:   %.loc16.1: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc16.2: init %empty_tuple.type = tuple_init () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc9: init %empty_tuple.type = converted %.loc16.1, %.loc16.2 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   assign %NS.var, %.loc9
// CHECK:STDOUT:   %.loc13.1: type = splice_block %.loc13.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:     %.loc13.2: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc13.3: type = converted %.loc13.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %NS: ref %empty_tuple.type = ref_binding NS, %NS.var
// CHECK:STDOUT:   %NS.ref: ref %empty_tuple.type = name_ref NS, %NS
// CHECK:STDOUT:   %.loc21.1: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc21.2: init %empty_tuple.type = tuple_init () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc19: init %empty_tuple.type = converted %.loc21.1, %.loc21.2 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   assign %NS.ref, %.loc19
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt.loc25: %pattern_type.cb1 = ref_binding_pattern x [concrete]
// CHECK:STDOUT:     %x.var_patt.loc23: %pattern_type.cb1 = var_pattern %x.patt.loc25 [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %x.var.loc23: ref %empty_tuple.type = var %x.var_patt.loc23
// CHECK:STDOUT:   %.loc31.1: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc31.2: init %empty_tuple.type = tuple_init () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc23: init %empty_tuple.type = converted %.loc31.1, %.loc31.2 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   assign %x.var.loc23, %.loc23
// CHECK:STDOUT:   %.loc28.1: type = splice_block %.loc28.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:     %.loc28.2: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc28.3: type = converted %.loc28.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %x.loc25: ref %empty_tuple.type = ref_binding x, %x.var.loc23
// CHECK:STDOUT:   %true: bool = bool_literal true [concrete = constants.%true]
// CHECK:STDOUT:   if %true br !if.then else br !if.else
// CHECK:STDOUT:
// CHECK:STDOUT: !if.then:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt.loc39: %pattern_type.cb1 = ref_binding_pattern x [concrete]
// CHECK:STDOUT:     %x.var_patt.loc38: %pattern_type.cb1 = var_pattern %x.patt.loc39 [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %x.var.loc38: ref %empty_tuple.type = var %x.var_patt.loc38
// CHECK:STDOUT:   %.loc45.1: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc45.2: init %empty_tuple.type = tuple_init () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc38: init %empty_tuple.type = converted %.loc45.1, %.loc45.2 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   assign %x.var.loc38, %.loc38
// CHECK:STDOUT:   %.loc42.1: type = splice_block %.loc42.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:     %.loc42.2: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc42.3: type = converted %.loc42.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %x.loc39: ref %empty_tuple.type = ref_binding x, %x.var.loc38
// CHECK:STDOUT:   %x.ref: ref %empty_tuple.type = name_ref x, %x.loc39
// CHECK:STDOUT:   %.loc50.1: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc50.2: init %empty_tuple.type = tuple_init () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc48: init %empty_tuple.type = converted %.loc50.1, %.loc50.2 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   assign %x.ref, %.loc48
// CHECK:STDOUT:   br !if.else
// CHECK:STDOUT:
// CHECK:STDOUT: !if.else:
// CHECK:STDOUT:   %Destroy.Op.bound.loc38: <bound method> = bound_method %x.var.loc38, constants.%Destroy.Op
// CHECK:STDOUT:   %Destroy.Op.call.loc38: init %empty_tuple.type = call %Destroy.Op.bound.loc38(%x.var.loc38)
// CHECK:STDOUT:   %Destroy.Op.bound.loc23: <bound method> = bound_method %x.var.loc23, constants.%Destroy.Op
// CHECK:STDOUT:   %Destroy.Op.call.loc23: init %empty_tuple.type = call %Destroy.Op.bound.loc23(%x.var.loc23)
// CHECK:STDOUT:   %Destroy.Op.bound.loc9: <bound method> = bound_method %NS.var, constants.%Destroy.Op
// CHECK:STDOUT:   %Destroy.Op.call.loc9: init %empty_tuple.type = call %Destroy.Op.bound.loc9(%NS.var)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op(%self.param: ref %empty_tuple.type) = "no_op";
// CHECK:STDOUT:
