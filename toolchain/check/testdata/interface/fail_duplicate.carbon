// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/none.carbon
// TODO: Add ranges and switch to "--dump-sem-ir-ranges=only".
// EXTRA-ARGS: --dump-sem-ir-ranges=if-present
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interface/fail_duplicate.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interface/fail_duplicate.carbon

// --- fail_redefine_without_dependents.carbon

library "[[@TEST_NAME]]";

interface Interface { }

// CHECK:STDERR:        ╭─┤ fail_redefine_without_dependents.carbon:[[@LINE+10]]:1
// CHECK:STDERR:        │ interface Interface { }
// CHECK:STDERR:        · ──────────┬──────────
// CHECK:STDERR:        ·           ╰──┤ previously defined here [RedeclPrevDef]
// CHECK:STDERR:        ┆
// CHECK:STDERR: ->     │ interface Interface {
// CHECK:STDERR:        · ━━━━━━━━━━┳━━━━━━━━━━
// CHECK:STDERR:        ·           ┗━━┫ error: redefinition of `interface Interface` [RedeclRedef]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
interface Interface {
  fn F();
}

// --- fail_redefine_with_dependents.carbon

library "[[@TEST_NAME]]";

interface Interface {}

// CHECK:STDERR:        ╭─┤ fail_redefine_with_dependents.carbon:[[@LINE+10]]:1
// CHECK:STDERR:        │ interface Interface {}
// CHECK:STDERR:        · ──────────┬──────────
// CHECK:STDERR:        ·           ╰──┤ previously defined here [RedeclPrevDef]
// CHECK:STDERR:        ┆
// CHECK:STDERR: ->     │ interface Interface {
// CHECK:STDERR:        · ━━━━━━━━━━┳━━━━━━━━━━
// CHECK:STDERR:        ·           ┗━━┫ error: redefinition of `interface Interface` [RedeclRedef]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
interface Interface {
  fn F(self);
}

// --- fail_name_conflict_with_fn.carbon

library "[[@TEST_NAME]]";

fn Function();

// CHECK:STDERR:        ╭─┤ fail_name_conflict_with_fn.carbon:[[@LINE+11]]:11
// CHECK:STDERR:        │ fn Function();
// CHECK:STDERR:        · ───────┬──────
// CHECK:STDERR:        ·        ╰──┤ name is previously declared here [NameDeclPrevious]
// CHECK:STDERR:        ┆
// CHECK:STDERR: ->     │ interface Function;
// CHECK:STDERR:        ·           ┳━━━━━━━
// CHECK:STDERR:        ·           ┗━━┫ error: duplicate name `Function` being declared in the same scope
// CHECK:STDERR:        ·              ┃        [NameDeclDuplicate]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
interface Function;

// --- fail_name_conflict_with_class.carbon

class Class;

// CHECK:STDERR:        ╭─┤ fail_name_conflict_with_class.carbon:[[@LINE+11]]:11
// CHECK:STDERR:        │ class Class;
// CHECK:STDERR:        · ──────┬─────
// CHECK:STDERR:        ·       ╰──┤ name is previously declared here [NameDeclPrevious]
// CHECK:STDERR:        ┆
// CHECK:STDERR: ->     │ interface Class { }
// CHECK:STDERR:        ·           ━━┳━━
// CHECK:STDERR:        ·             ┗━━┫ error: duplicate name `Class` being declared in the same scope
// CHECK:STDERR:        ·                ┃        [NameDeclDuplicate]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
interface Class { }


// CHECK:STDOUT: --- fail_redefine_without_dependents.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %Interface.type.283524.1: type = facet_type <@Interface.loc4> [concrete]
// CHECK:STDOUT:   %Self.3ce02d.1: %Interface.type.283524.1 = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %Interface.type.283524.2: type = facet_type <@Interface.loc16> [concrete]
// CHECK:STDOUT:   %Self.3ce02d.2: %Interface.type.283524.2 = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %Interface.WithSelf.F.type: type = fn_type @Interface.WithSelf.F, @Interface.WithSelf.loc16(%Self.3ce02d.2) [symbolic]
// CHECK:STDOUT:   %Interface.WithSelf.F: %Interface.WithSelf.F.type = struct_value () [symbolic]
// CHECK:STDOUT:   %Interface.assoc_type: type = assoc_entity_type @Interface.loc16 [concrete]
// CHECK:STDOUT:   %assoc0: %Interface.assoc_type = assoc_entity element0, @Interface.WithSelf.loc16.%Interface.WithSelf.F.decl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Interface = %Interface.decl.loc4
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Interface.decl.loc4: type = interface_decl @Interface.loc4 [concrete = constants.%Interface.type.283524.1] {} {}
// CHECK:STDOUT:   %Interface.decl.loc16: type = interface_decl @Interface.loc16 [concrete = constants.%Interface.type.283524.2] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Interface.loc4 {
// CHECK:STDOUT:   %Self: %Interface.type.283524.1 = symbolic_binding Self, 0 [symbolic = constants.%Self.3ce02d.1]
// CHECK:STDOUT:   %Interface.WithSelf.decl = interface_with_self_decl @Interface.loc4 [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Interface.loc16 {
// CHECK:STDOUT:   %Self: %Interface.type.283524.2 = symbolic_binding Self, 0 [symbolic = constants.%Self.3ce02d.2]
// CHECK:STDOUT:   %Interface.WithSelf.decl = interface_with_self_decl @Interface.loc16 [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !with Self:
// CHECK:STDOUT:   %Interface.WithSelf.F.decl: @Interface.WithSelf.loc16.%Interface.WithSelf.F.type (%Interface.WithSelf.F.type) = fn_decl @Interface.WithSelf.F [symbolic = @Interface.WithSelf.loc16.%Interface.WithSelf.F (constants.%Interface.WithSelf.F)] {} {}
// CHECK:STDOUT:   %assoc0: %Interface.assoc_type = assoc_entity element0, %Interface.WithSelf.F.decl [concrete = constants.%assoc0]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .F = @Interface.WithSelf.loc16.%assoc0
// CHECK:STDOUT:   witness = (@Interface.WithSelf.loc16.%Interface.WithSelf.F.decl)
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Interface.WithSelf.F(@Interface.loc16.%Self: %Interface.type.283524.2) {
// CHECK:STDOUT:   fn();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Interface.WithSelf.loc4(constants.%Self.3ce02d.1) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Interface.WithSelf.loc16(constants.%Self.3ce02d.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Interface.WithSelf.F(constants.%Self.3ce02d.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_redefine_with_dependents.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %Interface.type.283524.1: type = facet_type <@Interface.loc4> [concrete]
// CHECK:STDOUT:   %Self.3ce02d.1: %Interface.type.283524.1 = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %Interface.type.283524.2: type = facet_type <@Interface.loc16> [concrete]
// CHECK:STDOUT:   %Self.3ce02d.2: %Interface.type.283524.2 = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self.3ce02d.2 [symbolic]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Self.as_type [symbolic]
// CHECK:STDOUT:   %self.param_patt: %pattern_type = value_param_pattern [symbolic]
// CHECK:STDOUT:   %self.patt: %pattern_type = wrapper_binding_pattern self, %self.param_patt [symbolic]
// CHECK:STDOUT:   %Interface.WithSelf.F.type: type = fn_type @Interface.WithSelf.F, @Interface.WithSelf.loc16(%Self.3ce02d.2) [symbolic]
// CHECK:STDOUT:   %Interface.WithSelf.F: %Interface.WithSelf.F.type = struct_value () [symbolic]
// CHECK:STDOUT:   %Interface.assoc_type: type = assoc_entity_type @Interface.loc16 [concrete]
// CHECK:STDOUT:   %assoc0: %Interface.assoc_type = assoc_entity element0, @Interface.WithSelf.loc16.%Interface.WithSelf.F.decl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Interface = %Interface.decl.loc4
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Interface.decl.loc4: type = interface_decl @Interface.loc4 [concrete = constants.%Interface.type.283524.1] {} {}
// CHECK:STDOUT:   %Interface.decl.loc16: type = interface_decl @Interface.loc16 [concrete = constants.%Interface.type.283524.2] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Interface.loc4 {
// CHECK:STDOUT:   %Self: %Interface.type.283524.1 = symbolic_binding Self, 0 [symbolic = constants.%Self.3ce02d.1]
// CHECK:STDOUT:   %Interface.WithSelf.decl = interface_with_self_decl @Interface.loc4 [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Interface.loc16 {
// CHECK:STDOUT:   %Self: %Interface.type.283524.2 = symbolic_binding Self, 0 [symbolic = constants.%Self.3ce02d.2]
// CHECK:STDOUT:   %Interface.WithSelf.decl = interface_with_self_decl @Interface.loc16 [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !with Self:
// CHECK:STDOUT:   %Interface.WithSelf.F.decl: @Interface.WithSelf.loc16.%Interface.WithSelf.F.type (%Interface.WithSelf.F.type) = fn_decl @Interface.WithSelf.F [symbolic = @Interface.WithSelf.loc16.%Interface.WithSelf.F (constants.%Interface.WithSelf.F)] {
// CHECK:STDOUT:     %self.param_patt.loc17_8.1: @Interface.WithSelf.F.%pattern_type (%pattern_type) = value_param_pattern [symbolic = %self.param_patt.loc17_8.2 (constants.%self.param_patt)]
// CHECK:STDOUT:     %self.patt.loc17_8.1: @Interface.WithSelf.F.%pattern_type (%pattern_type) = wrapper_binding_pattern self, %self.param_patt.loc17_8.1 [symbolic = %self.patt.loc17_8.2 (constants.%self.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: @Interface.WithSelf.F.%Self.as_type.loc17_8.1 (%Self.as_type) = value_param call_param0
// CHECK:STDOUT:     %.loc17_8.1: type = splice_block %.loc17_8.2 [symbolic = %Self.as_type.loc17_8.1 (constants.%Self.as_type)] {
// CHECK:STDOUT:       %Self.ref: %Interface.type.283524.2 = name_ref Self, @Interface.loc16.%Self [symbolic = %Self (constants.%Self.3ce02d.2)]
// CHECK:STDOUT:       %Self.as_type.loc17_8.2: type = facet_access_type %Self.ref [symbolic = %Self.as_type.loc17_8.1 (constants.%Self.as_type)]
// CHECK:STDOUT:       %.loc17_8.2: type = converted %Self.ref, %Self.as_type.loc17_8.2 [symbolic = %Self.as_type.loc17_8.1 (constants.%Self.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %self: @Interface.WithSelf.F.%Self.as_type.loc17_8.1 (%Self.as_type) = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %assoc0: %Interface.assoc_type = assoc_entity element0, %Interface.WithSelf.F.decl [concrete = constants.%assoc0]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .F = @Interface.WithSelf.loc16.%assoc0
// CHECK:STDOUT:   witness = (@Interface.WithSelf.loc16.%Interface.WithSelf.F.decl)
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Interface.WithSelf.F(@Interface.loc16.%Self: %Interface.type.283524.2) {
// CHECK:STDOUT:   %Self: %Interface.type.283524.2 = symbolic_binding Self, 0 [symbolic = %Self (constants.%Self.3ce02d.2)]
// CHECK:STDOUT:   %Self.as_type.loc17_8.1: type = facet_access_type %Self [symbolic = %Self.as_type.loc17_8.1 (constants.%Self.as_type)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Self.as_type.loc17_8.1 [symbolic = %pattern_type (constants.%pattern_type)]
// CHECK:STDOUT:   %self.param_patt.loc17_8.2: @Interface.WithSelf.F.%pattern_type (%pattern_type) = value_param_pattern [symbolic = %self.param_patt.loc17_8.2 (constants.%self.param_patt)]
// CHECK:STDOUT:   %self.patt.loc17_8.2: @Interface.WithSelf.F.%pattern_type (%pattern_type) = wrapper_binding_pattern self, %self.param_patt.loc17_8.2 [symbolic = %self.patt.loc17_8.2 (constants.%self.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @Interface.WithSelf.F.%Self.as_type.loc17_8.1 (%Self.as_type));
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Interface.WithSelf.loc4(constants.%Self.3ce02d.1) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Interface.WithSelf.loc16(constants.%Self.3ce02d.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Interface.WithSelf.F(constants.%Self.3ce02d.2) {
// CHECK:STDOUT:   %Self => constants.%Self.3ce02d.2
// CHECK:STDOUT:   %Self.as_type.loc17_8.1 => constants.%Self.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type
// CHECK:STDOUT:   %self.param_patt.loc17_8.2 => constants.%self.param_patt
// CHECK:STDOUT:   %self.patt.loc17_8.2 => constants.%self.patt
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_name_conflict_with_fn.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %Function.type.b68: type = fn_type @Function.loc4 [concrete]
// CHECK:STDOUT:   %Function: %Function.type.b68 = struct_value () [concrete]
// CHECK:STDOUT:   %Function.type.186: type = facet_type <@Function.loc17> [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Function = %Function.decl.loc4
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Function.decl.loc4: %Function.type.b68 = fn_decl @Function.loc4 [concrete = constants.%Function] {} {}
// CHECK:STDOUT:   %Function.decl.loc17: type = interface_decl @Function.loc17 [concrete = constants.%Function.type.186] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Function.loc17;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Function.loc4();
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_name_conflict_with_class.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %Class: type = class_type @Class.loc2 [concrete]
// CHECK:STDOUT:   %Class.type: type = facet_type <@Class.loc15> [concrete]
// CHECK:STDOUT:   %Self: %Class.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Class = %Class.decl.loc2
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Class.decl.loc2: type = class_decl @Class.loc2 [concrete = constants.%Class] {} {}
// CHECK:STDOUT:   %Class.decl.loc15: type = interface_decl @Class.loc15 [concrete = constants.%Class.type] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Class.loc15 {
// CHECK:STDOUT:   %Self: %Class.type = symbolic_binding Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   %Class.WithSelf.decl = interface_with_self_decl @Class.loc15 [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Class.loc2;
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class.WithSelf(constants.%Self) {}
// CHECK:STDOUT:
