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
  // CHECK:STDERR:        ╭─┤ fail_basic.carbon:[[@LINE+11]]:14
  // CHECK:STDERR:        │   var unused x: () = ();
  // CHECK:STDERR:        ·              ──┬──
  // CHECK:STDERR:        ·                ╰──┤ name is previously declared here [NameDeclPrevious]
  // CHECK:STDERR:        ┆
  // CHECK:STDERR: ->     │   var unused x: () = ();
  // CHECK:STDERR:        ·              ┳
  // CHECK:STDERR:        ·              ┗━━┫ error: duplicate name `x` being declared in the same scope
  // CHECK:STDERR:        ·                 ┃        [NameDeclDuplicate]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  var unused x: () = ();
}

// CHECK:STDOUT: --- fail_basic.carbon = $fail_basic
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %Main.type: type = fn_type @Main [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Main: %Main.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %pattern_type.cb1: type = pattern_type %empty_tuple.type [concrete]
// CHECK:STDOUT:   %x.patt.06838d.1: %pattern_type.cb1 = ref_binding_pattern x [concrete]
// CHECK:STDOUT:   %x.var_patt.28d573.1: %pattern_type.cb1 = var_pattern %x.patt.06838d.1 [concrete]
// CHECK:STDOUT:   %x.patt.06838d.2: %pattern_type.cb1 = ref_binding_pattern x [concrete]
// CHECK:STDOUT:   %x.var_patt.28d573.2: %pattern_type.cb1 = var_pattern %x.patt.06838d.2 [concrete]
// CHECK:STDOUT:   %TrivialDestroyImpl.type: type = facet_type <@TrivialDestroyImpl> [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %DestroyImpl.type: type = facet_type <@DestroyImpl> [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .TrivialDestroyImpl = %Core.TrivialDestroyImpl
// CHECK:STDOUT:     .SubobjectDestroy = %Core.SubobjectDestroy
// CHECK:STDOUT:     .DestroyImpl = %Core.DestroyImpl
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.TrivialDestroyImpl: type = import_ref Core//prelude/parts/destroy, TrivialDestroyImpl, loaded [concrete = constants.%TrivialDestroyImpl.type]
// CHECK:STDOUT:   %Core.SubobjectDestroy: type = import_ref Core//prelude/parts/destroy, SubobjectDestroy, loaded [concrete = constants.%SubobjectDestroy.type]
// CHECK:STDOUT:   %Core.DestroyImpl: type = import_ref Core//prelude/parts/destroy, DestroyImpl, loaded [concrete = constants.%DestroyImpl.type]
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
// CHECK:STDOUT:   %x.var.loc3: ref %empty_tuple.type = var_storage %x.var_patt.loc3
// CHECK:STDOUT:   %.loc3_23.1: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc3_23.2: init %empty_tuple.type = tuple_init () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc3_3: init %empty_tuple.type = converted %.loc3_23.1, %.loc3_23.2 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   assign %x.var.loc3, %.loc3_3
// CHECK:STDOUT:   %.loc3_18.1: type = splice_block %.loc3_18.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:     %.loc3_18.2: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc3_18.3: type = converted %.loc3_18.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %x.loc3: ref %empty_tuple.type = wrapper_binding x, %x.var.loc3
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt.loc3: %pattern_type.cb1 = ref_binding_pattern x [concrete = constants.%x.patt.06838d.1]
// CHECK:STDOUT:     %x.var_patt.loc3: %pattern_type.cb1 = var_pattern %x.patt.loc3 [concrete = constants.%x.var_patt.28d573.1]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %x.var.loc15: ref %empty_tuple.type = var_storage %x.var_patt.loc15
// CHECK:STDOUT:   %.loc15_23.1: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc15_23.2: init %empty_tuple.type = tuple_init () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc15_3: init %empty_tuple.type = converted %.loc15_23.1, %.loc15_23.2 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   assign %x.var.loc15, %.loc15_3
// CHECK:STDOUT:   %.loc15_18.1: type = splice_block %.loc15_18.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:     %.loc15_18.2: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc15_18.3: type = converted %.loc15_18.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %x.loc15: ref %empty_tuple.type = wrapper_binding x, %x.var.loc15
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt.loc15: %pattern_type.cb1 = ref_binding_pattern x [concrete = constants.%x.patt.06838d.2]
// CHECK:STDOUT:     %x.var_patt.loc15: %pattern_type.cb1 = var_pattern %x.patt.loc15 [concrete = constants.%x.var_patt.28d573.2]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc15(%self.param: ref %empty_tuple.type) = "no_op";
// CHECK:STDOUT:
