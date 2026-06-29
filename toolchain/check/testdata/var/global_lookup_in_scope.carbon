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
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/var/global_lookup_in_scope.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/var/global_lookup_in_scope.carbon

// --- basic.carbon

var x: {.v: ()} = {.v = ()};

fn Main() {
  var unused y: {.v: ()} = x;
}

// CHECK:STDOUT: --- basic.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %struct_type.v: type = struct_type {.v: %empty_tuple.type} [concrete]
// CHECK:STDOUT:   %pattern_type.c4c: type = pattern_type %struct_type.v [concrete]
// CHECK:STDOUT:   %x.patt: %pattern_type.c4c = ref_binding_pattern x [concrete]
// CHECK:STDOUT:   %x.var_patt: %pattern_type.c4c = var_pattern %x.patt [concrete]
// CHECK:STDOUT:   %struct: %struct_type.v = struct_value (%empty_tuple) [concrete]
// CHECK:STDOUT:   %.267: ref %empty_tuple.type = struct_access file.%x.var, element0 [concrete]
// CHECK:STDOUT:   %Main.type: type = fn_type @Main [concrete]
// CHECK:STDOUT:   %Main: %Main.type = struct_value () [concrete]
// CHECK:STDOUT:   %y.patt: %pattern_type.c4c = ref_binding_pattern y [concrete]
// CHECK:STDOUT:   %y.var_patt: %pattern_type.c4c = var_pattern %y.patt [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.1d8f74.2: type = fn_type @Destroy.Op.loc5_3.2 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1a2547.2: %Destroy.Op.type.1d8f74.2 = struct_value () [concrete]
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
// CHECK:STDOUT:     .x = %x
// CHECK:STDOUT:     .Main = %Main.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %pattern_type.c4c = ref_binding_pattern x [concrete = constants.%x.patt]
// CHECK:STDOUT:     %x.var_patt: %pattern_type.c4c = var_pattern %x.patt [concrete = constants.%x.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %x.var: ref %struct_type.v = var %x.var_patt [concrete]
// CHECK:STDOUT:   %.loc2_15: type = splice_block %struct_type.v [concrete = constants.%struct_type.v] {
// CHECK:STDOUT:     %.loc2_14.1: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc2_14.2: type = converted %.loc2_14.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %struct_type.v: type = struct_type {.v: %empty_tuple.type} [concrete = constants.%struct_type.v]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %x: ref %struct_type.v = wrapper_binding x, %x.var [concrete = %x.var]
// CHECK:STDOUT:   %Main.decl: %Main.type = fn_decl @Main [concrete = constants.%Main] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Main() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %y.patt: %pattern_type.c4c = ref_binding_pattern y [concrete = constants.%y.patt]
// CHECK:STDOUT:     %y.var_patt: %pattern_type.c4c = var_pattern %y.patt [concrete = constants.%y.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %y.var: ref %struct_type.v = var %y.var_patt
// CHECK:STDOUT:   %x.ref: ref %struct_type.v = name_ref x, file.%x [concrete = file.%x.var]
// CHECK:STDOUT:   %.loc5_28.1: ref %empty_tuple.type = struct_access %x.ref, element0 [concrete = constants.%.267]
// CHECK:STDOUT:   %.loc5_28.2: ref %empty_tuple.type = struct_access %y.var, element0
// CHECK:STDOUT:   %.loc5_28.3: init %empty_tuple.type = tuple_init () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc5_28.4: init %empty_tuple.type = converted %.loc5_28.1, %.loc5_28.3 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc5_28.5: init %struct_type.v to %y.var = struct_init (%.loc5_28.4) [concrete = constants.%struct]
// CHECK:STDOUT:   %.loc5_3: init %struct_type.v = converted %x.ref, %.loc5_28.5 [concrete = constants.%struct]
// CHECK:STDOUT:   assign %y.var, %.loc5_3
// CHECK:STDOUT:   %.loc5_24: type = splice_block %struct_type.v [concrete = constants.%struct_type.v] {
// CHECK:STDOUT:     %.loc5_23.1: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc5_23.2: type = converted %.loc5_23.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %struct_type.v: type = struct_type {.v: %empty_tuple.type} [concrete = constants.%struct_type.v]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %y: ref %struct_type.v = wrapper_binding y, %y.var
// CHECK:STDOUT:   %Destroy.Op.bound: <bound method> = bound_method %y.var, constants.%Destroy.Op.1a2547.2
// CHECK:STDOUT:   %Destroy.Op.call: init %empty_tuple.type = call %Destroy.Op.bound(%y.var)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc5_3.1(%self.param: ref %empty_tuple.type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc5_3.2(%self.param: ref %struct_type.v) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc2_26.1: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc2_27.1: %struct_type.v = struct_literal (%.loc2_26.1) [concrete = constants.%struct]
// CHECK:STDOUT:   %.loc2_27.2: ref %empty_tuple.type = struct_access file.%x.var, element0 [concrete = constants.%.267]
// CHECK:STDOUT:   %.loc2_26.2: init %empty_tuple.type = tuple_init () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc2_27.3: init %empty_tuple.type = converted %.loc2_26.1, %.loc2_26.2 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc2_27.4: init %struct_type.v to file.%x.var = struct_init (%.loc2_27.3) [concrete = constants.%struct]
// CHECK:STDOUT:   %.loc2_1: init %struct_type.v = converted %.loc2_27.1, %.loc2_27.4 [concrete = constants.%struct]
// CHECK:STDOUT:   assign file.%x.var, %.loc2_1
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
