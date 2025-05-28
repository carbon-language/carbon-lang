// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// EXTRA-ARGS: --dump-sem-ir-ranges=only
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/no_prelude/import_member_cycle.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/no_prelude/import_member_cycle.carbon

// --- a.carbon

library "[[@TEST_NAME]]";

class Cycle {
  var a: Cycle*;
}

// --- b.carbon

library "[[@TEST_NAME]]";

import library "a";

fn Run() {
  //@dump-sem-ir-begin
  var a: Cycle*;
  //@dump-sem-ir-end
}

// CHECK:STDOUT: --- b.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Cycle: type = class_type @Cycle [concrete]
// CHECK:STDOUT:   %ptr: type = ptr_type %Cycle [concrete]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %ptr [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.Cycle: type = import_ref Main//a, Cycle, loaded [concrete = constants.%Cycle]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Run() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type = binding_pattern a [concrete]
// CHECK:STDOUT:     %a.var_patt: %pattern_type = var_pattern %a.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.var: ref %ptr = var %a.var_patt
// CHECK:STDOUT:   %.loc8: type = splice_block %ptr [concrete = constants.%ptr] {
// CHECK:STDOUT:     %Cycle.ref: type = name_ref Cycle, imports.%Main.Cycle [concrete = constants.%Cycle]
// CHECK:STDOUT:     %ptr: type = ptr_type %Cycle.ref [concrete = constants.%ptr]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: ref %ptr = bind_name a, %a.var
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
