// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// EXTRA-ARGS: --dump-sem-ir-ranges=only
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/no_prelude/import.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/no_prelude/import.carbon

// --- basics.carbon

library "[[@TEST_NAME]]";

class Empty {
}

class Field {
  var x: ();
}

class ForwardDeclared;

class ForwardDeclared {
  fn F[self: Self]();
  fn G[addr self: Self*]();
}

class Incomplete;

// --- basics_use.carbon

library "[[@TEST_NAME]]";

import library "basics";

fn Run() {
  //@dump-sem-ir-begin
  var a: Empty = {};

  var b: Field = {.x = ()};
  b.x = ();

  var c: ForwardDeclared = {};
  c.F();
  c.G();

  var d: ForwardDeclared* = &c;

  var e: Incomplete*;
  //@dump-sem-ir-end
}

// --- fail_import_misuses.carbon

library "[[@TEST_NAME]]";

import library "basics";

// CHECK:STDERR: fail_import_misuses.carbon:[[@LINE+8]]:1: error: redeclaration of `class Empty` is redundant [RedeclRedundant]
// CHECK:STDERR: class Empty {
// CHECK:STDERR: ^~~~~~~~~~~~~
// CHECK:STDERR: fail_import_misuses.carbon:[[@LINE-5]]:1: in import [InImport]
// CHECK:STDERR: basics.carbon:4:1: note: previously declared here [RedeclPrevDecl]
// CHECK:STDERR: class Empty {
// CHECK:STDERR: ^~~~~~~~~~~~~
// CHECK:STDERR:
class Empty {
}

// CHECK:STDERR: fail_import_misuses.carbon:[[@LINE+8]]:8: error: binding pattern has incomplete type `Incomplete` in name binding declaration [IncompleteTypeInBindingDecl]
// CHECK:STDERR: var a: Incomplete;
// CHECK:STDERR:        ^~~~~~~~~~
// CHECK:STDERR: fail_import_misuses.carbon:[[@LINE-16]]:1: in import [InImport]
// CHECK:STDERR: basics.carbon:18:1: note: class was forward declared here [ClassForwardDeclaredHere]
// CHECK:STDERR: class Incomplete;
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~
// CHECK:STDERR:
var a: Incomplete;

// --- forward_decl.carbon

library "[[@TEST_NAME]]";

class ForwardDecl;

// --- forward_decl.impl.carbon

impl library "[[@TEST_NAME]]";

//@dump-sem-ir-begin
class ForwardDecl {
}
//@dump-sem-ir-end

// CHECK:STDOUT: --- basics_use.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Empty: type = class_type @Empty [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %pattern_type.2f0: type = pattern_type %Empty [concrete]
// CHECK:STDOUT:   %Empty.val: %Empty = struct_value () [concrete]
// CHECK:STDOUT:   %Field: type = class_type @Field [concrete]
// CHECK:STDOUT:   %struct_type.x: type = struct_type {.x: %empty_tuple.type} [concrete]
// CHECK:STDOUT:   %pattern_type.f46: type = pattern_type %Field [concrete]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %Field.val: %Field = struct_value (%empty_tuple) [concrete]
// CHECK:STDOUT:   %Field.elem: type = unbound_element_type %Field, %empty_tuple.type [concrete]
// CHECK:STDOUT:   %ForwardDeclared.7b34f2.1: type = class_type @ForwardDeclared.1 [concrete]
// CHECK:STDOUT:   %pattern_type.1b8: type = pattern_type %ForwardDeclared.7b34f2.1 [concrete]
// CHECK:STDOUT:   %ForwardDeclared.val: %ForwardDeclared.7b34f2.1 = struct_value () [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.6cf: type = ptr_type %ForwardDeclared.7b34f2.1 [concrete]
// CHECK:STDOUT:   %pattern_type.ebb: type = pattern_type %ptr.6cf [concrete]
// CHECK:STDOUT:   %Incomplete: type = class_type @Incomplete [concrete]
// CHECK:STDOUT:   %ptr.c62: type = ptr_type %Incomplete [concrete]
// CHECK:STDOUT:   %pattern_type.275: type = pattern_type %ptr.c62 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.Empty: type = import_ref Main//basics, Empty, loaded [concrete = constants.%Empty]
// CHECK:STDOUT:   %Main.Field: type = import_ref Main//basics, Field, loaded [concrete = constants.%Field]
// CHECK:STDOUT:   %Main.ForwardDeclared: type = import_ref Main//basics, ForwardDeclared, loaded [concrete = constants.%ForwardDeclared.7b34f2.1]
// CHECK:STDOUT:   %Main.Incomplete: type = import_ref Main//basics, Incomplete, loaded [concrete = constants.%Incomplete]
// CHECK:STDOUT:   %Main.import_ref.695: %Field.elem = import_ref Main//basics, loc8_8, loaded [concrete = %.99e]
// CHECK:STDOUT:   %.99e: %Field.elem = field_decl x, element0 [concrete]
// CHECK:STDOUT:   %Main.import_ref.760: %F.type = import_ref Main//basics, loc14_21, loaded [concrete = constants.%F]
// CHECK:STDOUT:   %Main.import_ref.26e: %G.type = import_ref Main//basics, loc15_27, loaded [concrete = constants.%G]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Run() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.2f0 = binding_pattern a [concrete]
// CHECK:STDOUT:     %a.var_patt: %pattern_type.2f0 = var_pattern %a.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.var: ref %Empty = var %a.var_patt
// CHECK:STDOUT:   %.loc8_19.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %.loc8_19.2: init %Empty = class_init (), %a.var [concrete = constants.%Empty.val]
// CHECK:STDOUT:   %.loc8_3: init %Empty = converted %.loc8_19.1, %.loc8_19.2 [concrete = constants.%Empty.val]
// CHECK:STDOUT:   assign %a.var, %.loc8_3
// CHECK:STDOUT:   %Empty.ref: type = name_ref Empty, imports.%Main.Empty [concrete = constants.%Empty]
// CHECK:STDOUT:   %a: ref %Empty = bind_name a, %a.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %pattern_type.f46 = binding_pattern b [concrete]
// CHECK:STDOUT:     %b.var_patt: %pattern_type.f46 = var_pattern %b.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b.var: ref %Field = var %b.var_patt
// CHECK:STDOUT:   %.loc10_25.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %.loc10_26.1: %struct_type.x = struct_literal (%.loc10_25.1)
// CHECK:STDOUT:   %.loc10_26.2: ref %empty_tuple.type = class_element_access %b.var, element0
// CHECK:STDOUT:   %.loc10_25.2: init %empty_tuple.type = tuple_init () to %.loc10_26.2 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc10_26.3: init %empty_tuple.type = converted %.loc10_25.1, %.loc10_25.2 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc10_26.4: init %Field = class_init (%.loc10_26.3), %b.var [concrete = constants.%Field.val]
// CHECK:STDOUT:   %.loc10_3: init %Field = converted %.loc10_26.1, %.loc10_26.4 [concrete = constants.%Field.val]
// CHECK:STDOUT:   assign %b.var, %.loc10_3
// CHECK:STDOUT:   %Field.ref: type = name_ref Field, imports.%Main.Field [concrete = constants.%Field]
// CHECK:STDOUT:   %b: ref %Field = bind_name b, %b.var
// CHECK:STDOUT:   %b.ref: ref %Field = name_ref b, %b
// CHECK:STDOUT:   %x.ref: %Field.elem = name_ref x, imports.%Main.import_ref.695 [concrete = imports.%.99e]
// CHECK:STDOUT:   %.loc11_4: ref %empty_tuple.type = class_element_access %b.ref, element0
// CHECK:STDOUT:   %.loc11_10.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %.loc11_10.2: init %empty_tuple.type = tuple_init () to %.loc11_4 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc11_7: init %empty_tuple.type = converted %.loc11_10.1, %.loc11_10.2 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   assign %.loc11_4, %.loc11_7
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c.patt: %pattern_type.1b8 = binding_pattern c [concrete]
// CHECK:STDOUT:     %c.var_patt: %pattern_type.1b8 = var_pattern %c.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c.var: ref %ForwardDeclared.7b34f2.1 = var %c.var_patt
// CHECK:STDOUT:   %.loc13_29.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %.loc13_29.2: init %ForwardDeclared.7b34f2.1 = class_init (), %c.var [concrete = constants.%ForwardDeclared.val]
// CHECK:STDOUT:   %.loc13_3: init %ForwardDeclared.7b34f2.1 = converted %.loc13_29.1, %.loc13_29.2 [concrete = constants.%ForwardDeclared.val]
// CHECK:STDOUT:   assign %c.var, %.loc13_3
// CHECK:STDOUT:   %ForwardDeclared.ref.loc13: type = name_ref ForwardDeclared, imports.%Main.ForwardDeclared [concrete = constants.%ForwardDeclared.7b34f2.1]
// CHECK:STDOUT:   %c: ref %ForwardDeclared.7b34f2.1 = bind_name c, %c.var
// CHECK:STDOUT:   %c.ref.loc14: ref %ForwardDeclared.7b34f2.1 = name_ref c, %c
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, imports.%Main.import_ref.760 [concrete = constants.%F]
// CHECK:STDOUT:   %F.bound: <bound method> = bound_method %c.ref.loc14, %F.ref
// CHECK:STDOUT:   %.loc14: %ForwardDeclared.7b34f2.1 = bind_value %c.ref.loc14
// CHECK:STDOUT:   %F.call: init %empty_tuple.type = call %F.bound(%.loc14)
// CHECK:STDOUT:   %c.ref.loc15: ref %ForwardDeclared.7b34f2.1 = name_ref c, %c
// CHECK:STDOUT:   %G.ref: %G.type = name_ref G, imports.%Main.import_ref.26e [concrete = constants.%G]
// CHECK:STDOUT:   %G.bound: <bound method> = bound_method %c.ref.loc15, %G.ref
// CHECK:STDOUT:   %addr.loc15: %ptr.6cf = addr_of %c.ref.loc15
// CHECK:STDOUT:   %G.call: init %empty_tuple.type = call %G.bound(%addr.loc15)
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %d.patt: %pattern_type.ebb = binding_pattern d [concrete]
// CHECK:STDOUT:     %d.var_patt: %pattern_type.ebb = var_pattern %d.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %d.var: ref %ptr.6cf = var %d.var_patt
// CHECK:STDOUT:   %c.ref.loc17: ref %ForwardDeclared.7b34f2.1 = name_ref c, %c
// CHECK:STDOUT:   %addr.loc17: %ptr.6cf = addr_of %c.ref.loc17
// CHECK:STDOUT:   assign %d.var, %addr.loc17
// CHECK:STDOUT:   %.loc17: type = splice_block %ptr.loc17 [concrete = constants.%ptr.6cf] {
// CHECK:STDOUT:     %ForwardDeclared.ref.loc17: type = name_ref ForwardDeclared, imports.%Main.ForwardDeclared [concrete = constants.%ForwardDeclared.7b34f2.1]
// CHECK:STDOUT:     %ptr.loc17: type = ptr_type %ForwardDeclared.ref.loc17 [concrete = constants.%ptr.6cf]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %d: ref %ptr.6cf = bind_name d, %d.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %e.patt: %pattern_type.275 = binding_pattern e [concrete]
// CHECK:STDOUT:     %e.var_patt: %pattern_type.275 = var_pattern %e.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %e.var: ref %ptr.c62 = var %e.var_patt
// CHECK:STDOUT:   %.loc19: type = splice_block %ptr.loc19 [concrete = constants.%ptr.c62] {
// CHECK:STDOUT:     %Incomplete.ref: type = name_ref Incomplete, imports.%Main.Incomplete [concrete = constants.%Incomplete]
// CHECK:STDOUT:     %ptr.loc19: type = ptr_type %Incomplete.ref [concrete = constants.%ptr.c62]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %e: ref %ptr.c62 = bind_name e, %e.var
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- forward_decl.impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %ForwardDecl: type = class_type @ForwardDecl [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %ForwardDecl.decl: type = class_decl @ForwardDecl [concrete = constants.%ForwardDecl] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @ForwardDecl {
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%ForwardDecl
// CHECK:STDOUT: }
// CHECK:STDOUT:
