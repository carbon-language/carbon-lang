// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// EXTRA-ARGS: --dump-sem-ir-ranges=only
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/no_prelude/indirect_import_member.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/no_prelude/indirect_import_member.carbon

// ============================================================================
// Setup files
// ============================================================================

// --- a.carbon

library "[[@TEST_NAME]]";

class C {
  fn F() {}
}

// --- b.carbon

library "[[@TEST_NAME]]";

export import library "a";

// --- c.carbon

library "[[@TEST_NAME]]";

import library "b";

export C;

// --- d.carbon

library "[[@TEST_NAME]]";

export import library "c";

// --- e.carbon

library "[[@TEST_NAME]]";

import library "c";

class D {
  alias C = package.C;
}

// --- f.carbon

library "[[@TEST_NAME]]";

export import library "e";

// ============================================================================
// Test files
// ============================================================================

// --- use_b.carbon

library "[[@TEST_NAME]]";

import library "b";

//@dump-sem-ir-begin
var x: () = C.F();
//@dump-sem-ir-end

// --- use_c.carbon

library "[[@TEST_NAME]]";

import library "c";

//@dump-sem-ir-begin
var x: () = C.F();
//@dump-sem-ir-end

// --- use_d.carbon

library "[[@TEST_NAME]]";

import library "d";

//@dump-sem-ir-begin
var x: () = C.F();
//@dump-sem-ir-end

// --- use_e.carbon

library "[[@TEST_NAME]]";

import library "e";

//@dump-sem-ir-begin
var x: () = D.C.F();
//@dump-sem-ir-end

// --- use_f.carbon

library "[[@TEST_NAME]]";

import library "f";

//@dump-sem-ir-begin
var x: () = D.C.F();
//@dump-sem-ir-end

// CHECK:STDOUT: --- use_b.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %empty_tuple.type [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.C: type = import_ref Main//a, C, loaded [concrete = constants.%C]
// CHECK:STDOUT:   %Main.import_ref.943: %F.type = import_ref Main//a, loc5_10, loaded [concrete = constants.%F]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %pattern_type = binding_pattern x [concrete]
// CHECK:STDOUT:     %x.var_patt: %pattern_type = var_pattern %x.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %x.var: ref %empty_tuple.type = var %x.var_patt [concrete]
// CHECK:STDOUT:   %.loc7_9.1: type = splice_block %.loc7_9.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:     %.loc7_9.2: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc7_9.3: type = converted %.loc7_9.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %x: ref %empty_tuple.type = bind_name x, %x.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %C.ref: type = name_ref C, imports.%Main.C [concrete = constants.%C]
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, imports.%Main.import_ref.943 [concrete = constants.%F]
// CHECK:STDOUT:   %F.call: init %empty_tuple.type = call %F.ref()
// CHECK:STDOUT:   assign file.%x.var, %F.call
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- use_c.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %empty_tuple.type [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.C: type = import_ref Main//c, C, loaded [concrete = constants.%C]
// CHECK:STDOUT:   %Main.import_ref.5d3: %F.type = import_ref Main//c, inst23 [indirect], loaded [concrete = constants.%F]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %pattern_type = binding_pattern x [concrete]
// CHECK:STDOUT:     %x.var_patt: %pattern_type = var_pattern %x.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %x.var: ref %empty_tuple.type = var %x.var_patt [concrete]
// CHECK:STDOUT:   %.loc7_9.1: type = splice_block %.loc7_9.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:     %.loc7_9.2: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc7_9.3: type = converted %.loc7_9.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %x: ref %empty_tuple.type = bind_name x, %x.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %C.ref: type = name_ref C, imports.%Main.C [concrete = constants.%C]
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, imports.%Main.import_ref.5d3 [concrete = constants.%F]
// CHECK:STDOUT:   %F.call: init %empty_tuple.type = call %F.ref()
// CHECK:STDOUT:   assign file.%x.var, %F.call
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- use_d.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %empty_tuple.type [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.C: type = import_ref Main//c, C, loaded [concrete = constants.%C]
// CHECK:STDOUT:   %Main.import_ref.5d3: %F.type = import_ref Main//c, inst23 [indirect], loaded [concrete = constants.%F]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %pattern_type = binding_pattern x [concrete]
// CHECK:STDOUT:     %x.var_patt: %pattern_type = var_pattern %x.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %x.var: ref %empty_tuple.type = var %x.var_patt [concrete]
// CHECK:STDOUT:   %.loc7_9.1: type = splice_block %.loc7_9.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:     %.loc7_9.2: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc7_9.3: type = converted %.loc7_9.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %x: ref %empty_tuple.type = bind_name x, %x.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %C.ref: type = name_ref C, imports.%Main.C [concrete = constants.%C]
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, imports.%Main.import_ref.5d3 [concrete = constants.%F]
// CHECK:STDOUT:   %F.call: init %empty_tuple.type = call %F.ref()
// CHECK:STDOUT:   assign file.%x.var, %F.call
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- use_e.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %empty_tuple.type [concrete]
// CHECK:STDOUT:   %D: type = class_type @D [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.D: type = import_ref Main//e, D, loaded [concrete = constants.%D]
// CHECK:STDOUT:   %Main.import_ref.bf1: type = import_ref Main//e, loc7_9, loaded [concrete = constants.%C]
// CHECK:STDOUT:   %Main.import_ref.c85: %F.type = import_ref Main//e, inst26 [indirect], loaded [concrete = constants.%F]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %pattern_type = binding_pattern x [concrete]
// CHECK:STDOUT:     %x.var_patt: %pattern_type = var_pattern %x.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %x.var: ref %empty_tuple.type = var %x.var_patt [concrete]
// CHECK:STDOUT:   %.loc7_9.1: type = splice_block %.loc7_9.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:     %.loc7_9.2: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc7_9.3: type = converted %.loc7_9.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %x: ref %empty_tuple.type = bind_name x, %x.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %D.ref: type = name_ref D, imports.%Main.D [concrete = constants.%D]
// CHECK:STDOUT:   %C.ref: type = name_ref C, imports.%Main.import_ref.bf1 [concrete = constants.%C]
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, imports.%Main.import_ref.c85 [concrete = constants.%F]
// CHECK:STDOUT:   %F.call: init %empty_tuple.type = call %F.ref()
// CHECK:STDOUT:   assign file.%x.var, %F.call
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- use_f.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %empty_tuple.type [concrete]
// CHECK:STDOUT:   %D: type = class_type @D [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.D: type = import_ref Main//e, D, loaded [concrete = constants.%D]
// CHECK:STDOUT:   %Main.import_ref.bf1: type = import_ref Main//e, loc7_9, loaded [concrete = constants.%C]
// CHECK:STDOUT:   %Main.import_ref.c85: %F.type = import_ref Main//e, inst26 [indirect], loaded [concrete = constants.%F]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %pattern_type = binding_pattern x [concrete]
// CHECK:STDOUT:     %x.var_patt: %pattern_type = var_pattern %x.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %x.var: ref %empty_tuple.type = var %x.var_patt [concrete]
// CHECK:STDOUT:   %.loc7_9.1: type = splice_block %.loc7_9.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:     %.loc7_9.2: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc7_9.3: type = converted %.loc7_9.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %x: ref %empty_tuple.type = bind_name x, %x.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %D.ref: type = name_ref D, imports.%Main.D [concrete = constants.%D]
// CHECK:STDOUT:   %C.ref: type = name_ref C, imports.%Main.import_ref.bf1 [concrete = constants.%C]
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, imports.%Main.import_ref.c85 [concrete = constants.%F]
// CHECK:STDOUT:   %F.call: init %empty_tuple.type = call %F.ref()
// CHECK:STDOUT:   assign file.%x.var, %F.call
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
