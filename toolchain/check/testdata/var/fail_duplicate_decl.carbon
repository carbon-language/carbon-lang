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
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/var/fail_duplicate_decl.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/var/fail_duplicate_decl.carbon

// --- fail_basic.carbon

fn Main() {
  var unused x: () = ();
  // CHECK:STDERR: fail_basic.carbon:[[@LINE+7]]:14: error: duplicate name `x` being declared in the same scope [NameDeclDuplicate]
  // CHECK:STDERR:   var unused x: () = ();
  // CHECK:STDERR:              ^
  // CHECK:STDERR: fail_basic.carbon:[[@LINE-4]]:14: note: name is previously declared here [NameDeclPrevious]
  // CHECK:STDERR:   var unused x: () = ();
  // CHECK:STDERR:              ^~~~~
  // CHECK:STDERR:
  var unused x: () = ();
}

// CHECK:STDOUT: --- fail_basic.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Main.type: type = fn_type @Main [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Main: %Main.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %pattern_type.cb1: type = pattern_type %empty_tuple.type [concrete]
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
// CHECK:STDOUT:     .Main = %Main.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Main.decl: %Main.type = fn_decl @Main [concrete = constants.%Main] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Main() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt.loc3: %pattern_type.cb1 = ref_binding_pattern x [concrete]
// CHECK:STDOUT:     %x.var_patt.loc3: %pattern_type.cb1 = var_pattern %x.patt.loc3 [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %x.var.loc3: ref %empty_tuple.type = var %x.var_patt.loc3
// CHECK:STDOUT:   %.loc3_23.1: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc3_23.2: init %empty_tuple.type = tuple_init () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc3_3: init %empty_tuple.type = converted %.loc3_23.1, %.loc3_23.2 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   assign %x.var.loc3, %.loc3_3
// CHECK:STDOUT:   %.loc3_18.1: type = splice_block %.loc3_18.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:     %.loc3_18.2: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc3_18.3: type = converted %.loc3_18.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %x.loc3: ref %empty_tuple.type = ref_binding x, %x.var.loc3
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt.loc11: %pattern_type.cb1 = ref_binding_pattern x [concrete]
// CHECK:STDOUT:     %x.var_patt.loc11: %pattern_type.cb1 = var_pattern %x.patt.loc11 [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %x.var.loc11: ref %empty_tuple.type = var %x.var_patt.loc11
// CHECK:STDOUT:   %.loc11_23.1: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc11_23.2: init %empty_tuple.type = tuple_init () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc11_3: init %empty_tuple.type = converted %.loc11_23.1, %.loc11_23.2 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   assign %x.var.loc11, %.loc11_3
// CHECK:STDOUT:   %.loc11_18.1: type = splice_block %.loc11_18.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:     %.loc11_18.2: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc11_18.3: type = converted %.loc11_18.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %x.loc11: ref %empty_tuple.type = ref_binding x, %x.var.loc11
// CHECK:STDOUT:   %Destroy.Op.bound.loc11: <bound method> = bound_method %x.var.loc11, constants.%Destroy.Op
// CHECK:STDOUT:   %Destroy.Op.call.loc11: init %empty_tuple.type = call %Destroy.Op.bound.loc11(%x.var.loc11)
// CHECK:STDOUT:   %Destroy.Op.bound.loc3: <bound method> = bound_method %x.var.loc3, constants.%Destroy.Op
// CHECK:STDOUT:   %Destroy.Op.call.loc3: init %empty_tuple.type = call %Destroy.Op.bound.loc3(%x.var.loc3)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op(%self.param: ref %empty_tuple.type) = "no_op";
// CHECK:STDOUT:
