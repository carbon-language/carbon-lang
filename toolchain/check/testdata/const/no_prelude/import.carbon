// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// EXTRA-ARGS: --dump-sem-ir-ranges=only
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/const/no_prelude/import.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/const/no_prelude/import.carbon

// --- implicit.carbon
library "[[@TEST_NAME]]";

class C {}

fn F() -> const C;

var a_ref: const C = F();
var a_ptr_ref: const C* = &a_ref;

// --- implicit.impl.carbon
impl library "[[@TEST_NAME]]";

// Take a reference to avoid unsupported copy logic. This still validates the
// `const` is handled.
//@dump-sem-ir-begin
var a: const C* = &a_ref;
var a_ptr: const C* = a_ptr_ref;
//@dump-sem-ir-end

// CHECK:STDOUT: --- implicit.impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %const: type = const_type %C [concrete]
// CHECK:STDOUT:   %ptr.801: type = ptr_type %const [concrete]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %ptr.801 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.C: type = import_ref Main//implicit, C, loaded [concrete = constants.%C]
// CHECK:STDOUT:   %Main.a_ref: ref %const = import_ref Main//implicit, a_ref, loaded
// CHECK:STDOUT:   %Main.a_ptr_ref: ref %ptr.801 = import_ref Main//implicit, a_ptr_ref, loaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type = binding_pattern a [concrete]
// CHECK:STDOUT:     %a.var_patt: %pattern_type = var_pattern %a.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.var: ref %ptr.801 = var %a.var_patt [concrete]
// CHECK:STDOUT:   %.loc6: type = splice_block %ptr.loc6 [concrete = constants.%ptr.801] {
// CHECK:STDOUT:     %C.ref.loc6: type = name_ref C, imports.%Main.C [concrete = constants.%C]
// CHECK:STDOUT:     %const.loc6: type = const_type %C.ref.loc6 [concrete = constants.%const]
// CHECK:STDOUT:     %ptr.loc6: type = ptr_type %const.loc6 [concrete = constants.%ptr.801]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: ref %ptr.801 = bind_name a, %a.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a_ptr.patt: %pattern_type = binding_pattern a_ptr [concrete]
// CHECK:STDOUT:     %a_ptr.var_patt: %pattern_type = var_pattern %a_ptr.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a_ptr.var: ref %ptr.801 = var %a_ptr.var_patt [concrete]
// CHECK:STDOUT:   %.loc7: type = splice_block %ptr.loc7 [concrete = constants.%ptr.801] {
// CHECK:STDOUT:     %C.ref.loc7: type = name_ref C, imports.%Main.C [concrete = constants.%C]
// CHECK:STDOUT:     %const.loc7: type = const_type %C.ref.loc7 [concrete = constants.%const]
// CHECK:STDOUT:     %ptr.loc7: type = ptr_type %const.loc7 [concrete = constants.%ptr.801]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a_ptr: ref %ptr.801 = bind_name a_ptr, %a_ptr.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a_ref.ref: ref %const = name_ref a_ref, imports.%Main.a_ref
// CHECK:STDOUT:   %addr: %ptr.801 = addr_of %a_ref.ref
// CHECK:STDOUT:   assign file.%a.var, %addr
// CHECK:STDOUT:   %a_ptr_ref.ref: ref %ptr.801 = name_ref a_ptr_ref, imports.%Main.a_ptr_ref
// CHECK:STDOUT:   %.loc7: %ptr.801 = bind_value %a_ptr_ref.ref
// CHECK:STDOUT:   assign file.%a_ptr.var, %.loc7
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
