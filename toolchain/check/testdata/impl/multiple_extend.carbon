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
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/impl/multiple_extend.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/impl/multiple_extend.carbon

// --- different_impl_member_names.carbon

library "[[@TEST_NAME]]";

interface HasF {
  fn F();
}

interface HasG {
  fn G();
}

class C {
  extend impl as HasF {
    fn F() {}
  }
  extend impl as HasG {
    fn G() {}
  }
}

fn H(c: C) {
  C.F();
  c.F();
  C.G();
  c.G();
}

// --- fail_ambiguous_impls.carbon

library "[[@TEST_NAME]]";

interface HasA1 {
  fn A();
}

interface HasA2 {
  fn A();
}

class D {
  extend impl as HasA1 {
    fn A() {}
  }
  extend impl as HasA2 {
    fn A() {}
  }
}

fn B(d: D) {
  // CHECK:STDERR: fail_ambiguous_impls.carbon:[[@LINE+4]]:3: error: ambiguous use of name `A` found in multiple extended scopes [NameAmbiguousDueToExtend]
  // CHECK:STDERR:   D.A();
  // CHECK:STDERR:   ^~~
  // CHECK:STDERR:
  D.A();
  // CHECK:STDERR: fail_ambiguous_impls.carbon:[[@LINE+4]]:3: error: ambiguous use of name `A` found in multiple extended scopes [NameAmbiguousDueToExtend]
  // CHECK:STDERR:   d.A();
  // CHECK:STDERR:   ^~~
  // CHECK:STDERR:
  d.A();
}

// --- different_impl_and_base.carbon

library "[[@TEST_NAME]]";

interface HasI {
  fn I();
}

base class B {
  fn J() {}
}

class E {
  extend base: B;
  extend impl as HasI {
    fn I() {}
  }
}

fn H(e: E) {
  E.I();
  e.I();
  E.J();
  e.J();
}

// --- fail_ambiguous_impl_and_base.carbon

library "[[@TEST_NAME]]";

base class Base {
  fn K() {}
}

interface HasK {
  fn K();
}

class L {
  extend base: Base;
  extend impl as HasK {
    fn K() {}
  }
}

fn M(l: L) {
   // CHECK:STDERR: fail_ambiguous_impl_and_base.carbon:[[@LINE+4]]:4: error: ambiguous use of name `K` found in multiple extended scopes [NameAmbiguousDueToExtend]
   // CHECK:STDERR:    L.K();
   // CHECK:STDERR:    ^~~
   // CHECK:STDERR:
   L.K();
   // CHECK:STDERR: fail_ambiguous_impl_and_base.carbon:[[@LINE+4]]:4: error: ambiguous use of name `K` found in multiple extended scopes [NameAmbiguousDueToExtend]
   // CHECK:STDERR:    l.K();
   // CHECK:STDERR:    ^~~
   // CHECK:STDERR:
   l.K();
}

// --- ambiguity_hidden.carbon

library "[[@TEST_NAME]]";

base class NBase {
  fn N() {}
}

interface HasN1 {
  fn N();
}

interface HasN2 {
  fn N();
}

class O {
  extend base: NBase;
  extend impl as HasN1 {
    fn N() {}
  }
  extend impl as HasN2 {
    fn N() {}
  }
  fn N();
}

fn P(o: O) {
   O.N();
   o.N();
}

// CHECK:STDOUT: --- different_impl_member_names.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %HasF.type: type = facet_type <@HasF> [concrete]
// CHECK:STDOUT:   %Self.f0c: %HasF.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %HasF.F.type: type = fn_type @HasF.F [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %HasF.F: %HasF.F.type = struct_value () [concrete]
// CHECK:STDOUT:   %HasF.assoc_type: type = assoc_entity_type @HasF [concrete]
// CHECK:STDOUT:   %assoc0.97a: %HasF.assoc_type = assoc_entity element0, @HasF.%HasF.F.decl [concrete]
// CHECK:STDOUT:   %HasG.type: type = facet_type <@HasG> [concrete]
// CHECK:STDOUT:   %Self.d42: %HasG.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %HasG.G.type: type = fn_type @HasG.G [concrete]
// CHECK:STDOUT:   %HasG.G: %HasG.G.type = struct_value () [concrete]
// CHECK:STDOUT:   %HasG.assoc_type: type = assoc_entity_type @HasG [concrete]
// CHECK:STDOUT:   %assoc0.319: %HasG.assoc_type = assoc_entity element0, @HasG.%HasG.G.decl [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %HasF.impl_witness: <witness> = impl_witness @C.%HasF.impl_witness_table [concrete]
// CHECK:STDOUT:   %C.as.HasF.impl.F.type: type = fn_type @C.as.HasF.impl.F [concrete]
// CHECK:STDOUT:   %C.as.HasF.impl.F: %C.as.HasF.impl.F.type = struct_value () [concrete]
// CHECK:STDOUT:   %HasF.facet: %HasF.type = facet_value %C, (%HasF.impl_witness) [concrete]
// CHECK:STDOUT:   %HasG.impl_witness: <witness> = impl_witness @C.%HasG.impl_witness_table [concrete]
// CHECK:STDOUT:   %C.as.HasG.impl.G.type: type = fn_type @C.as.HasG.impl.G [concrete]
// CHECK:STDOUT:   %C.as.HasG.impl.G: %C.as.HasG.impl.G.type = struct_value () [concrete]
// CHECK:STDOUT:   %HasG.facet: %HasG.type = facet_value %C, (%HasG.impl_witness) [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %H.type: type = fn_type @H [concrete]
// CHECK:STDOUT:   %H: %H.type = struct_value () [concrete]
// CHECK:STDOUT:   %.a44: type = fn_type_with_self_type %HasF.F.type, %HasF.facet [concrete]
// CHECK:STDOUT:   %.521: type = fn_type_with_self_type %HasG.G.type, %HasG.facet [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .HasF = %HasF.decl
// CHECK:STDOUT:     .HasG = %HasG.decl
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .H = %H.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %HasF.decl: type = interface_decl @HasF [concrete = constants.%HasF.type] {} {}
// CHECK:STDOUT:   %HasG.decl: type = interface_decl @HasG [concrete = constants.%HasG.type] {} {}
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   %H.decl: %H.type = fn_decl @H [concrete = constants.%H] {
// CHECK:STDOUT:     %c.patt: %pattern_type = binding_pattern c [concrete]
// CHECK:STDOUT:     %c.param_patt: %pattern_type = value_param_pattern %c.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %c.param: %C = value_param call_param0
// CHECK:STDOUT:     %C.ref.loc21: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %c: %C = bind_name c, %c.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @HasF {
// CHECK:STDOUT:   %Self: %HasF.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.f0c]
// CHECK:STDOUT:   %HasF.F.decl: %HasF.F.type = fn_decl @HasF.F [concrete = constants.%HasF.F] {} {}
// CHECK:STDOUT:   %assoc0: %HasF.assoc_type = assoc_entity element0, %HasF.F.decl [concrete = constants.%assoc0.97a]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .F = %assoc0
// CHECK:STDOUT:   .HasG = <poisoned>
// CHECK:STDOUT:   .G = <poisoned>
// CHECK:STDOUT:   witness = (%HasF.F.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @HasG {
// CHECK:STDOUT:   %Self: %HasG.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.d42]
// CHECK:STDOUT:   %HasG.G.decl: %HasG.G.type = fn_decl @HasG.G [concrete = constants.%HasG.G] {} {}
// CHECK:STDOUT:   %assoc0: %HasG.assoc_type = assoc_entity element0, %HasG.G.decl [concrete = constants.%assoc0.319]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .G = %assoc0
// CHECK:STDOUT:   .F = <poisoned>
// CHECK:STDOUT:   witness = (%HasG.G.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @C.as.HasF.impl: %Self.ref as %HasF.ref {
// CHECK:STDOUT:   %C.as.HasF.impl.F.decl: %C.as.HasF.impl.F.type = fn_decl @C.as.HasF.impl.F [concrete = constants.%C.as.HasF.impl.F] {} {}
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %C.as.HasF.impl.F.decl
// CHECK:STDOUT:   witness = @C.%HasF.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @C.as.HasG.impl: %Self.ref as %HasG.ref {
// CHECK:STDOUT:   %C.as.HasG.impl.G.decl: %C.as.HasG.impl.G.type = fn_decl @C.as.HasG.impl.G [concrete = constants.%C.as.HasG.impl.G] {} {}
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .G = %C.as.HasG.impl.G.decl
// CHECK:STDOUT:   witness = @C.%HasG.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   impl_decl @C.as.HasF.impl [concrete] {} {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%C [concrete = constants.%C]
// CHECK:STDOUT:     %HasF.ref: type = name_ref HasF, file.%HasF.decl [concrete = constants.%HasF.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %HasF.impl_witness_table = impl_witness_table (@C.as.HasF.impl.%C.as.HasF.impl.F.decl), @C.as.HasF.impl [concrete]
// CHECK:STDOUT:   %HasF.impl_witness: <witness> = impl_witness %HasF.impl_witness_table [concrete = constants.%HasF.impl_witness]
// CHECK:STDOUT:   impl_decl @C.as.HasG.impl [concrete] {} {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%C [concrete = constants.%C]
// CHECK:STDOUT:     %HasG.ref: type = name_ref HasG, file.%HasG.decl [concrete = constants.%HasG.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %HasG.impl_witness_table = impl_witness_table (@C.as.HasG.impl.%C.as.HasG.impl.G.decl), @C.as.HasG.impl [concrete]
// CHECK:STDOUT:   %HasG.impl_witness: <witness> = impl_witness %HasG.impl_witness_table [concrete = constants.%HasG.impl_witness]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT:   .HasF = <poisoned>
// CHECK:STDOUT:   .HasG = <poisoned>
// CHECK:STDOUT:   .F = <poisoned>
// CHECK:STDOUT:   .G = <poisoned>
// CHECK:STDOUT:   extend @C.as.HasF.impl.%HasF.ref
// CHECK:STDOUT:   extend @C.as.HasG.impl.%HasG.ref
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @HasF.F(@HasF.%Self: %HasF.type) {
// CHECK:STDOUT:   fn();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @HasG.G(@HasG.%Self: %HasG.type) {
// CHECK:STDOUT:   fn();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C.as.HasF.impl.F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C.as.HasG.impl.G() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @H(%c.param: %C) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %C.ref.loc22: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %F.ref.loc22: %HasF.assoc_type = name_ref F, @HasF.%assoc0 [concrete = constants.%assoc0.97a]
// CHECK:STDOUT:   %impl.elem0.loc22: %.a44 = impl_witness_access constants.%HasF.impl_witness, element0 [concrete = constants.%C.as.HasF.impl.F]
// CHECK:STDOUT:   %C.as.HasF.impl.F.call.loc22: init %empty_tuple.type = call %impl.elem0.loc22()
// CHECK:STDOUT:   %c.ref.loc23: %C = name_ref c, %c
// CHECK:STDOUT:   %F.ref.loc23: %HasF.assoc_type = name_ref F, @HasF.%assoc0 [concrete = constants.%assoc0.97a]
// CHECK:STDOUT:   %impl.elem0.loc23: %.a44 = impl_witness_access constants.%HasF.impl_witness, element0 [concrete = constants.%C.as.HasF.impl.F]
// CHECK:STDOUT:   %C.as.HasF.impl.F.call.loc23: init %empty_tuple.type = call %impl.elem0.loc23()
// CHECK:STDOUT:   %C.ref.loc24: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %G.ref.loc24: %HasG.assoc_type = name_ref G, @HasG.%assoc0 [concrete = constants.%assoc0.319]
// CHECK:STDOUT:   %impl.elem0.loc24: %.521 = impl_witness_access constants.%HasG.impl_witness, element0 [concrete = constants.%C.as.HasG.impl.G]
// CHECK:STDOUT:   %C.as.HasG.impl.G.call.loc24: init %empty_tuple.type = call %impl.elem0.loc24()
// CHECK:STDOUT:   %c.ref.loc25: %C = name_ref c, %c
// CHECK:STDOUT:   %G.ref.loc25: %HasG.assoc_type = name_ref G, @HasG.%assoc0 [concrete = constants.%assoc0.319]
// CHECK:STDOUT:   %impl.elem0.loc25: %.521 = impl_witness_access constants.%HasG.impl_witness, element0 [concrete = constants.%C.as.HasG.impl.G]
// CHECK:STDOUT:   %C.as.HasG.impl.G.call.loc25: init %empty_tuple.type = call %impl.elem0.loc25()
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HasF.F(constants.%Self.f0c) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HasG.G(constants.%Self.d42) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HasF.F(constants.%HasF.facet) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HasG.G(constants.%HasG.facet) {}
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_ambiguous_impls.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %HasA1.type: type = facet_type <@HasA1> [concrete]
// CHECK:STDOUT:   %Self.90d: %HasA1.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %HasA1.A.type: type = fn_type @HasA1.A [concrete]
// CHECK:STDOUT:   %HasA1.A: %HasA1.A.type = struct_value () [concrete]
// CHECK:STDOUT:   %HasA1.assoc_type: type = assoc_entity_type @HasA1 [concrete]
// CHECK:STDOUT:   %assoc0.34b: %HasA1.assoc_type = assoc_entity element0, @HasA1.%HasA1.A.decl [concrete]
// CHECK:STDOUT:   %HasA2.type: type = facet_type <@HasA2> [concrete]
// CHECK:STDOUT:   %Self.bdc: %HasA2.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %HasA2.A.type: type = fn_type @HasA2.A [concrete]
// CHECK:STDOUT:   %HasA2.A: %HasA2.A.type = struct_value () [concrete]
// CHECK:STDOUT:   %HasA2.assoc_type: type = assoc_entity_type @HasA2 [concrete]
// CHECK:STDOUT:   %assoc0.316: %HasA2.assoc_type = assoc_entity element0, @HasA2.%HasA2.A.decl [concrete]
// CHECK:STDOUT:   %D: type = class_type @D [concrete]
// CHECK:STDOUT:   %HasA1.impl_witness: <witness> = impl_witness @D.%HasA1.impl_witness_table [concrete]
// CHECK:STDOUT:   %D.as.HasA1.impl.A.type: type = fn_type @D.as.HasA1.impl.A [concrete]
// CHECK:STDOUT:   %D.as.HasA1.impl.A: %D.as.HasA1.impl.A.type = struct_value () [concrete]
// CHECK:STDOUT:   %HasA1.facet: %HasA1.type = facet_value %D, (%HasA1.impl_witness) [concrete]
// CHECK:STDOUT:   %HasA2.impl_witness: <witness> = impl_witness @D.%HasA2.impl_witness_table [concrete]
// CHECK:STDOUT:   %D.as.HasA2.impl.A.type: type = fn_type @D.as.HasA2.impl.A [concrete]
// CHECK:STDOUT:   %D.as.HasA2.impl.A: %D.as.HasA2.impl.A.type = struct_value () [concrete]
// CHECK:STDOUT:   %HasA2.facet: %HasA2.type = facet_value %D, (%HasA2.impl_witness) [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %D [concrete]
// CHECK:STDOUT:   %B.type: type = fn_type @B [concrete]
// CHECK:STDOUT:   %B: %B.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .HasA1 = %HasA1.decl
// CHECK:STDOUT:     .HasA2 = %HasA2.decl
// CHECK:STDOUT:     .D = %D.decl
// CHECK:STDOUT:     .B = %B.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %HasA1.decl: type = interface_decl @HasA1 [concrete = constants.%HasA1.type] {} {}
// CHECK:STDOUT:   %HasA2.decl: type = interface_decl @HasA2 [concrete = constants.%HasA2.type] {} {}
// CHECK:STDOUT:   %D.decl: type = class_decl @D [concrete = constants.%D] {} {}
// CHECK:STDOUT:   %B.decl: %B.type = fn_decl @B [concrete = constants.%B] {
// CHECK:STDOUT:     %d.patt: %pattern_type = binding_pattern d [concrete]
// CHECK:STDOUT:     %d.param_patt: %pattern_type = value_param_pattern %d.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %d.param: %D = value_param call_param0
// CHECK:STDOUT:     %D.ref.loc21: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %d: %D = bind_name d, %d.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @HasA1 {
// CHECK:STDOUT:   %Self: %HasA1.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.90d]
// CHECK:STDOUT:   %HasA1.A.decl: %HasA1.A.type = fn_decl @HasA1.A [concrete = constants.%HasA1.A] {} {}
// CHECK:STDOUT:   %assoc0: %HasA1.assoc_type = assoc_entity element0, %HasA1.A.decl [concrete = constants.%assoc0.34b]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .A = %assoc0
// CHECK:STDOUT:   .HasA2 = <poisoned>
// CHECK:STDOUT:   witness = (%HasA1.A.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @HasA2 {
// CHECK:STDOUT:   %Self: %HasA2.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.bdc]
// CHECK:STDOUT:   %HasA2.A.decl: %HasA2.A.type = fn_decl @HasA2.A [concrete = constants.%HasA2.A] {} {}
// CHECK:STDOUT:   %assoc0: %HasA2.assoc_type = assoc_entity element0, %HasA2.A.decl [concrete = constants.%assoc0.316]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .A = %assoc0
// CHECK:STDOUT:   witness = (%HasA2.A.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @D.as.HasA1.impl: %Self.ref as %HasA1.ref {
// CHECK:STDOUT:   %D.as.HasA1.impl.A.decl: %D.as.HasA1.impl.A.type = fn_decl @D.as.HasA1.impl.A [concrete = constants.%D.as.HasA1.impl.A] {} {}
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .A = %D.as.HasA1.impl.A.decl
// CHECK:STDOUT:   witness = @D.%HasA1.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @D.as.HasA2.impl: %Self.ref as %HasA2.ref {
// CHECK:STDOUT:   %D.as.HasA2.impl.A.decl: %D.as.HasA2.impl.A.type = fn_decl @D.as.HasA2.impl.A [concrete = constants.%D.as.HasA2.impl.A] {} {}
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .A = %D.as.HasA2.impl.A.decl
// CHECK:STDOUT:   witness = @D.%HasA2.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @D {
// CHECK:STDOUT:   impl_decl @D.as.HasA1.impl [concrete] {} {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%D [concrete = constants.%D]
// CHECK:STDOUT:     %HasA1.ref: type = name_ref HasA1, file.%HasA1.decl [concrete = constants.%HasA1.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %HasA1.impl_witness_table = impl_witness_table (@D.as.HasA1.impl.%D.as.HasA1.impl.A.decl), @D.as.HasA1.impl [concrete]
// CHECK:STDOUT:   %HasA1.impl_witness: <witness> = impl_witness %HasA1.impl_witness_table [concrete = constants.%HasA1.impl_witness]
// CHECK:STDOUT:   impl_decl @D.as.HasA2.impl [concrete] {} {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%D [concrete = constants.%D]
// CHECK:STDOUT:     %HasA2.ref: type = name_ref HasA2, file.%HasA2.decl [concrete = constants.%HasA2.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %HasA2.impl_witness_table = impl_witness_table (@D.as.HasA2.impl.%D.as.HasA2.impl.A.decl), @D.as.HasA2.impl [concrete]
// CHECK:STDOUT:   %HasA2.impl_witness: <witness> = impl_witness %HasA2.impl_witness_table [concrete = constants.%HasA2.impl_witness]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%D
// CHECK:STDOUT:   .HasA1 = <poisoned>
// CHECK:STDOUT:   .HasA2 = <poisoned>
// CHECK:STDOUT:   .A = <poisoned>
// CHECK:STDOUT:   extend @D.as.HasA1.impl.%HasA1.ref
// CHECK:STDOUT:   extend @D.as.HasA2.impl.%HasA2.ref
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @HasA1.A(@HasA1.%Self: %HasA1.type) {
// CHECK:STDOUT:   fn();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @HasA2.A(@HasA2.%Self: %HasA2.type) {
// CHECK:STDOUT:   fn();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @D.as.HasA1.impl.A() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @D.as.HasA2.impl.A() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @B(%d.param: %D) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %D.ref.loc26: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:   %A.ref.loc26: <error> = name_ref A, <error> [concrete = <error>]
// CHECK:STDOUT:   %d.ref: %D = name_ref d, %d
// CHECK:STDOUT:   %A.ref.loc31: <error> = name_ref A, <error> [concrete = <error>]
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HasA1.A(constants.%Self.90d) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HasA2.A(constants.%Self.bdc) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HasA1.A(constants.%HasA1.facet) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HasA2.A(constants.%HasA2.facet) {}
// CHECK:STDOUT:
// CHECK:STDOUT: --- different_impl_and_base.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %HasI.type: type = facet_type <@HasI> [concrete]
// CHECK:STDOUT:   %Self: %HasI.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %HasI.I.type: type = fn_type @HasI.I [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %HasI.I: %HasI.I.type = struct_value () [concrete]
// CHECK:STDOUT:   %HasI.assoc_type: type = assoc_entity_type @HasI [concrete]
// CHECK:STDOUT:   %assoc0: %HasI.assoc_type = assoc_entity element0, @HasI.%HasI.I.decl [concrete]
// CHECK:STDOUT:   %B: type = class_type @B [concrete]
// CHECK:STDOUT:   %B.J.type: type = fn_type @B.J [concrete]
// CHECK:STDOUT:   %B.J: %B.J.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %E: type = class_type @E [concrete]
// CHECK:STDOUT:   %E.elem: type = unbound_element_type %E, %B [concrete]
// CHECK:STDOUT:   %HasI.impl_witness: <witness> = impl_witness @E.%HasI.impl_witness_table [concrete]
// CHECK:STDOUT:   %E.as.HasI.impl.I.type: type = fn_type @E.as.HasI.impl.I [concrete]
// CHECK:STDOUT:   %E.as.HasI.impl.I: %E.as.HasI.impl.I.type = struct_value () [concrete]
// CHECK:STDOUT:   %HasI.facet: %HasI.type = facet_value %E, (%HasI.impl_witness) [concrete]
// CHECK:STDOUT:   %struct_type.base.0ff: type = struct_type {.base: %B} [concrete]
// CHECK:STDOUT:   %complete_type.98e: <witness> = complete_type_witness %struct_type.base.0ff [concrete]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %E [concrete]
// CHECK:STDOUT:   %H.type: type = fn_type @H [concrete]
// CHECK:STDOUT:   %H: %H.type = struct_value () [concrete]
// CHECK:STDOUT:   %.06f: type = fn_type_with_self_type %HasI.I.type, %HasI.facet [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .HasI = %HasI.decl
// CHECK:STDOUT:     .B = %B.decl
// CHECK:STDOUT:     .E = %E.decl
// CHECK:STDOUT:     .H = %H.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %HasI.decl: type = interface_decl @HasI [concrete = constants.%HasI.type] {} {}
// CHECK:STDOUT:   %B.decl: type = class_decl @B [concrete = constants.%B] {} {}
// CHECK:STDOUT:   %E.decl: type = class_decl @E [concrete = constants.%E] {} {}
// CHECK:STDOUT:   %H.decl: %H.type = fn_decl @H [concrete = constants.%H] {
// CHECK:STDOUT:     %e.patt: %pattern_type = binding_pattern e [concrete]
// CHECK:STDOUT:     %e.param_patt: %pattern_type = value_param_pattern %e.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %e.param: %E = value_param call_param0
// CHECK:STDOUT:     %E.ref.loc19: type = name_ref E, file.%E.decl [concrete = constants.%E]
// CHECK:STDOUT:     %e: %E = bind_name e, %e.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @HasI {
// CHECK:STDOUT:   %Self: %HasI.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   %HasI.I.decl: %HasI.I.type = fn_decl @HasI.I [concrete = constants.%HasI.I] {} {}
// CHECK:STDOUT:   %assoc0: %HasI.assoc_type = assoc_entity element0, %HasI.I.decl [concrete = constants.%assoc0]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .I = %assoc0
// CHECK:STDOUT:   .J = <poisoned>
// CHECK:STDOUT:   witness = (%HasI.I.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @E.as.HasI.impl: %Self.ref as %HasI.ref {
// CHECK:STDOUT:   %E.as.HasI.impl.I.decl: %E.as.HasI.impl.I.type = fn_decl @E.as.HasI.impl.I [concrete = constants.%E.as.HasI.impl.I] {} {}
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .I = %E.as.HasI.impl.I.decl
// CHECK:STDOUT:   witness = @E.%HasI.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @B {
// CHECK:STDOUT:   %B.J.decl: %B.J.type = fn_decl @B.J [concrete = constants.%B.J] {} {}
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%B
// CHECK:STDOUT:   .J = %B.J.decl
// CHECK:STDOUT:   .HasI = <poisoned>
// CHECK:STDOUT:   .I = <poisoned>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @E {
// CHECK:STDOUT:   %B.ref: type = name_ref B, file.%B.decl [concrete = constants.%B]
// CHECK:STDOUT:   %.loc13: %E.elem = base_decl %B.ref, element0 [concrete]
// CHECK:STDOUT:   impl_decl @E.as.HasI.impl [concrete] {} {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%E [concrete = constants.%E]
// CHECK:STDOUT:     %HasI.ref: type = name_ref HasI, file.%HasI.decl [concrete = constants.%HasI.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %HasI.impl_witness_table = impl_witness_table (@E.as.HasI.impl.%E.as.HasI.impl.I.decl), @E.as.HasI.impl [concrete]
// CHECK:STDOUT:   %HasI.impl_witness: <witness> = impl_witness %HasI.impl_witness_table [concrete = constants.%HasI.impl_witness]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%struct_type.base.0ff [concrete = constants.%complete_type.98e]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%E
// CHECK:STDOUT:   .B = <poisoned>
// CHECK:STDOUT:   .base = %.loc13
// CHECK:STDOUT:   .HasI = <poisoned>
// CHECK:STDOUT:   .I = <poisoned>
// CHECK:STDOUT:   .J = <poisoned>
// CHECK:STDOUT:   extend %B.ref
// CHECK:STDOUT:   extend @E.as.HasI.impl.%HasI.ref
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @HasI.I(@HasI.%Self: %HasI.type) {
// CHECK:STDOUT:   fn();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @B.J() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @E.as.HasI.impl.I() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @H(%e.param: %E) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %E.ref.loc20: type = name_ref E, file.%E.decl [concrete = constants.%E]
// CHECK:STDOUT:   %I.ref.loc20: %HasI.assoc_type = name_ref I, @HasI.%assoc0 [concrete = constants.%assoc0]
// CHECK:STDOUT:   %impl.elem0.loc20: %.06f = impl_witness_access constants.%HasI.impl_witness, element0 [concrete = constants.%E.as.HasI.impl.I]
// CHECK:STDOUT:   %E.as.HasI.impl.I.call.loc20: init %empty_tuple.type = call %impl.elem0.loc20()
// CHECK:STDOUT:   %e.ref.loc21: %E = name_ref e, %e
// CHECK:STDOUT:   %I.ref.loc21: %HasI.assoc_type = name_ref I, @HasI.%assoc0 [concrete = constants.%assoc0]
// CHECK:STDOUT:   %impl.elem0.loc21: %.06f = impl_witness_access constants.%HasI.impl_witness, element0 [concrete = constants.%E.as.HasI.impl.I]
// CHECK:STDOUT:   %E.as.HasI.impl.I.call.loc21: init %empty_tuple.type = call %impl.elem0.loc21()
// CHECK:STDOUT:   %E.ref.loc22: type = name_ref E, file.%E.decl [concrete = constants.%E]
// CHECK:STDOUT:   %J.ref.loc22: %B.J.type = name_ref J, @B.%B.J.decl [concrete = constants.%B.J]
// CHECK:STDOUT:   %B.J.call.loc22: init %empty_tuple.type = call %J.ref.loc22()
// CHECK:STDOUT:   %e.ref.loc23: %E = name_ref e, %e
// CHECK:STDOUT:   %J.ref.loc23: %B.J.type = name_ref J, @B.%B.J.decl [concrete = constants.%B.J]
// CHECK:STDOUT:   %B.J.call.loc23: init %empty_tuple.type = call %J.ref.loc23()
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HasI.I(constants.%Self) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HasI.I(constants.%HasI.facet) {}
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_ambiguous_impl_and_base.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Base: type = class_type @Base [concrete]
// CHECK:STDOUT:   %Base.K.type: type = fn_type @Base.K [concrete]
// CHECK:STDOUT:   %Base.K: %Base.K.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %HasK.type: type = facet_type <@HasK> [concrete]
// CHECK:STDOUT:   %Self: %HasK.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %HasK.K.type: type = fn_type @HasK.K [concrete]
// CHECK:STDOUT:   %HasK.K: %HasK.K.type = struct_value () [concrete]
// CHECK:STDOUT:   %HasK.assoc_type: type = assoc_entity_type @HasK [concrete]
// CHECK:STDOUT:   %assoc0: %HasK.assoc_type = assoc_entity element0, @HasK.%HasK.K.decl [concrete]
// CHECK:STDOUT:   %L: type = class_type @L [concrete]
// CHECK:STDOUT:   %L.elem: type = unbound_element_type %L, %Base [concrete]
// CHECK:STDOUT:   %HasK.impl_witness: <witness> = impl_witness @L.%HasK.impl_witness_table [concrete]
// CHECK:STDOUT:   %L.as.HasK.impl.K.type: type = fn_type @L.as.HasK.impl.K [concrete]
// CHECK:STDOUT:   %L.as.HasK.impl.K: %L.as.HasK.impl.K.type = struct_value () [concrete]
// CHECK:STDOUT:   %HasK.facet: %HasK.type = facet_value %L, (%HasK.impl_witness) [concrete]
// CHECK:STDOUT:   %struct_type.base.b1e: type = struct_type {.base: %Base} [concrete]
// CHECK:STDOUT:   %complete_type.15c: <witness> = complete_type_witness %struct_type.base.b1e [concrete]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %L [concrete]
// CHECK:STDOUT:   %M.type: type = fn_type @M [concrete]
// CHECK:STDOUT:   %M: %M.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Base = %Base.decl
// CHECK:STDOUT:     .HasK = %HasK.decl
// CHECK:STDOUT:     .L = %L.decl
// CHECK:STDOUT:     .M = %M.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Base.decl: type = class_decl @Base [concrete = constants.%Base] {} {}
// CHECK:STDOUT:   %HasK.decl: type = interface_decl @HasK [concrete = constants.%HasK.type] {} {}
// CHECK:STDOUT:   %L.decl: type = class_decl @L [concrete = constants.%L] {} {}
// CHECK:STDOUT:   %M.decl: %M.type = fn_decl @M [concrete = constants.%M] {
// CHECK:STDOUT:     %l.patt: %pattern_type = binding_pattern l [concrete]
// CHECK:STDOUT:     %l.param_patt: %pattern_type = value_param_pattern %l.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %l.param: %L = value_param call_param0
// CHECK:STDOUT:     %L.ref.loc19: type = name_ref L, file.%L.decl [concrete = constants.%L]
// CHECK:STDOUT:     %l: %L = bind_name l, %l.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @HasK {
// CHECK:STDOUT:   %Self: %HasK.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   %HasK.K.decl: %HasK.K.type = fn_decl @HasK.K [concrete = constants.%HasK.K] {} {}
// CHECK:STDOUT:   %assoc0: %HasK.assoc_type = assoc_entity element0, %HasK.K.decl [concrete = constants.%assoc0]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .K = %assoc0
// CHECK:STDOUT:   witness = (%HasK.K.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @L.as.HasK.impl: %Self.ref as %HasK.ref {
// CHECK:STDOUT:   %L.as.HasK.impl.K.decl: %L.as.HasK.impl.K.type = fn_decl @L.as.HasK.impl.K [concrete = constants.%L.as.HasK.impl.K] {} {}
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .K = %L.as.HasK.impl.K.decl
// CHECK:STDOUT:   witness = @L.%HasK.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Base {
// CHECK:STDOUT:   %Base.K.decl: %Base.K.type = fn_decl @Base.K [concrete = constants.%Base.K] {} {}
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Base
// CHECK:STDOUT:   .K = %Base.K.decl
// CHECK:STDOUT:   .HasK = <poisoned>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @L {
// CHECK:STDOUT:   %Base.ref: type = name_ref Base, file.%Base.decl [concrete = constants.%Base]
// CHECK:STDOUT:   %.loc13: %L.elem = base_decl %Base.ref, element0 [concrete]
// CHECK:STDOUT:   impl_decl @L.as.HasK.impl [concrete] {} {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%L [concrete = constants.%L]
// CHECK:STDOUT:     %HasK.ref: type = name_ref HasK, file.%HasK.decl [concrete = constants.%HasK.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %HasK.impl_witness_table = impl_witness_table (@L.as.HasK.impl.%L.as.HasK.impl.K.decl), @L.as.HasK.impl [concrete]
// CHECK:STDOUT:   %HasK.impl_witness: <witness> = impl_witness %HasK.impl_witness_table [concrete = constants.%HasK.impl_witness]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%struct_type.base.b1e [concrete = constants.%complete_type.15c]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%L
// CHECK:STDOUT:   .Base = <poisoned>
// CHECK:STDOUT:   .base = %.loc13
// CHECK:STDOUT:   .HasK = <poisoned>
// CHECK:STDOUT:   .K = <poisoned>
// CHECK:STDOUT:   extend %Base.ref
// CHECK:STDOUT:   extend @L.as.HasK.impl.%HasK.ref
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Base.K() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @HasK.K(@HasK.%Self: %HasK.type) {
// CHECK:STDOUT:   fn();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @L.as.HasK.impl.K() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @M(%l.param: %L) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %L.ref.loc24: type = name_ref L, file.%L.decl [concrete = constants.%L]
// CHECK:STDOUT:   %K.ref.loc24: <error> = name_ref K, <error> [concrete = <error>]
// CHECK:STDOUT:   %l.ref: %L = name_ref l, %l
// CHECK:STDOUT:   %K.ref.loc29: <error> = name_ref K, <error> [concrete = <error>]
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HasK.K(constants.%Self) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HasK.K(constants.%HasK.facet) {}
// CHECK:STDOUT:
// CHECK:STDOUT: --- ambiguity_hidden.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %NBase: type = class_type @NBase [concrete]
// CHECK:STDOUT:   %NBase.N.type: type = fn_type @NBase.N [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %NBase.N: %NBase.N.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %HasN1.type: type = facet_type <@HasN1> [concrete]
// CHECK:STDOUT:   %Self.e0b: %HasN1.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %HasN1.N.type: type = fn_type @HasN1.N [concrete]
// CHECK:STDOUT:   %HasN1.N: %HasN1.N.type = struct_value () [concrete]
// CHECK:STDOUT:   %HasN1.assoc_type: type = assoc_entity_type @HasN1 [concrete]
// CHECK:STDOUT:   %assoc0.c23: %HasN1.assoc_type = assoc_entity element0, @HasN1.%HasN1.N.decl [concrete]
// CHECK:STDOUT:   %HasN2.type: type = facet_type <@HasN2> [concrete]
// CHECK:STDOUT:   %Self.e6e: %HasN2.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %HasN2.N.type: type = fn_type @HasN2.N [concrete]
// CHECK:STDOUT:   %HasN2.N: %HasN2.N.type = struct_value () [concrete]
// CHECK:STDOUT:   %HasN2.assoc_type: type = assoc_entity_type @HasN2 [concrete]
// CHECK:STDOUT:   %assoc0.7d1: %HasN2.assoc_type = assoc_entity element0, @HasN2.%HasN2.N.decl [concrete]
// CHECK:STDOUT:   %O: type = class_type @O [concrete]
// CHECK:STDOUT:   %O.elem: type = unbound_element_type %O, %NBase [concrete]
// CHECK:STDOUT:   %HasN1.impl_witness: <witness> = impl_witness @O.%HasN1.impl_witness_table [concrete]
// CHECK:STDOUT:   %O.as.HasN1.impl.N.type: type = fn_type @O.as.HasN1.impl.N [concrete]
// CHECK:STDOUT:   %O.as.HasN1.impl.N: %O.as.HasN1.impl.N.type = struct_value () [concrete]
// CHECK:STDOUT:   %HasN1.facet: %HasN1.type = facet_value %O, (%HasN1.impl_witness) [concrete]
// CHECK:STDOUT:   %HasN2.impl_witness: <witness> = impl_witness @O.%HasN2.impl_witness_table [concrete]
// CHECK:STDOUT:   %O.as.HasN2.impl.N.type: type = fn_type @O.as.HasN2.impl.N [concrete]
// CHECK:STDOUT:   %O.as.HasN2.impl.N: %O.as.HasN2.impl.N.type = struct_value () [concrete]
// CHECK:STDOUT:   %HasN2.facet: %HasN2.type = facet_value %O, (%HasN2.impl_witness) [concrete]
// CHECK:STDOUT:   %O.N.type: type = fn_type @O.N [concrete]
// CHECK:STDOUT:   %O.N: %O.N.type = struct_value () [concrete]
// CHECK:STDOUT:   %struct_type.base.9c6: type = struct_type {.base: %NBase} [concrete]
// CHECK:STDOUT:   %complete_type.121: <witness> = complete_type_witness %struct_type.base.9c6 [concrete]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %O [concrete]
// CHECK:STDOUT:   %P.type: type = fn_type @P [concrete]
// CHECK:STDOUT:   %P: %P.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .NBase = %NBase.decl
// CHECK:STDOUT:     .HasN1 = %HasN1.decl
// CHECK:STDOUT:     .HasN2 = %HasN2.decl
// CHECK:STDOUT:     .O = %O.decl
// CHECK:STDOUT:     .P = %P.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %NBase.decl: type = class_decl @NBase [concrete = constants.%NBase] {} {}
// CHECK:STDOUT:   %HasN1.decl: type = interface_decl @HasN1 [concrete = constants.%HasN1.type] {} {}
// CHECK:STDOUT:   %HasN2.decl: type = interface_decl @HasN2 [concrete = constants.%HasN2.type] {} {}
// CHECK:STDOUT:   %O.decl: type = class_decl @O [concrete = constants.%O] {} {}
// CHECK:STDOUT:   %P.decl: %P.type = fn_decl @P [concrete = constants.%P] {
// CHECK:STDOUT:     %o.patt: %pattern_type = binding_pattern o [concrete]
// CHECK:STDOUT:     %o.param_patt: %pattern_type = value_param_pattern %o.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %o.param: %O = value_param call_param0
// CHECK:STDOUT:     %O.ref.loc27: type = name_ref O, file.%O.decl [concrete = constants.%O]
// CHECK:STDOUT:     %o: %O = bind_name o, %o.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @HasN1 {
// CHECK:STDOUT:   %Self: %HasN1.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.e0b]
// CHECK:STDOUT:   %HasN1.N.decl: %HasN1.N.type = fn_decl @HasN1.N [concrete = constants.%HasN1.N] {} {}
// CHECK:STDOUT:   %assoc0: %HasN1.assoc_type = assoc_entity element0, %HasN1.N.decl [concrete = constants.%assoc0.c23]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .N = %assoc0
// CHECK:STDOUT:   .HasN2 = <poisoned>
// CHECK:STDOUT:   witness = (%HasN1.N.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @HasN2 {
// CHECK:STDOUT:   %Self: %HasN2.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.e6e]
// CHECK:STDOUT:   %HasN2.N.decl: %HasN2.N.type = fn_decl @HasN2.N [concrete = constants.%HasN2.N] {} {}
// CHECK:STDOUT:   %assoc0: %HasN2.assoc_type = assoc_entity element0, %HasN2.N.decl [concrete = constants.%assoc0.7d1]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .N = %assoc0
// CHECK:STDOUT:   witness = (%HasN2.N.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @O.as.HasN1.impl: %Self.ref as %HasN1.ref {
// CHECK:STDOUT:   %O.as.HasN1.impl.N.decl: %O.as.HasN1.impl.N.type = fn_decl @O.as.HasN1.impl.N [concrete = constants.%O.as.HasN1.impl.N] {} {}
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .N = %O.as.HasN1.impl.N.decl
// CHECK:STDOUT:   witness = @O.%HasN1.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @O.as.HasN2.impl: %Self.ref as %HasN2.ref {
// CHECK:STDOUT:   %O.as.HasN2.impl.N.decl: %O.as.HasN2.impl.N.type = fn_decl @O.as.HasN2.impl.N [concrete = constants.%O.as.HasN2.impl.N] {} {}
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .N = %O.as.HasN2.impl.N.decl
// CHECK:STDOUT:   witness = @O.%HasN2.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @NBase {
// CHECK:STDOUT:   %NBase.N.decl: %NBase.N.type = fn_decl @NBase.N [concrete = constants.%NBase.N] {} {}
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%NBase
// CHECK:STDOUT:   .N = %NBase.N.decl
// CHECK:STDOUT:   .HasN1 = <poisoned>
// CHECK:STDOUT:   .HasN2 = <poisoned>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @O {
// CHECK:STDOUT:   %NBase.ref: type = name_ref NBase, file.%NBase.decl [concrete = constants.%NBase]
// CHECK:STDOUT:   %.loc17: %O.elem = base_decl %NBase.ref, element0 [concrete]
// CHECK:STDOUT:   impl_decl @O.as.HasN1.impl [concrete] {} {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%O [concrete = constants.%O]
// CHECK:STDOUT:     %HasN1.ref: type = name_ref HasN1, file.%HasN1.decl [concrete = constants.%HasN1.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %HasN1.impl_witness_table = impl_witness_table (@O.as.HasN1.impl.%O.as.HasN1.impl.N.decl), @O.as.HasN1.impl [concrete]
// CHECK:STDOUT:   %HasN1.impl_witness: <witness> = impl_witness %HasN1.impl_witness_table [concrete = constants.%HasN1.impl_witness]
// CHECK:STDOUT:   impl_decl @O.as.HasN2.impl [concrete] {} {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%O [concrete = constants.%O]
// CHECK:STDOUT:     %HasN2.ref: type = name_ref HasN2, file.%HasN2.decl [concrete = constants.%HasN2.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %HasN2.impl_witness_table = impl_witness_table (@O.as.HasN2.impl.%O.as.HasN2.impl.N.decl), @O.as.HasN2.impl [concrete]
// CHECK:STDOUT:   %HasN2.impl_witness: <witness> = impl_witness %HasN2.impl_witness_table [concrete = constants.%HasN2.impl_witness]
// CHECK:STDOUT:   %O.N.decl: %O.N.type = fn_decl @O.N [concrete = constants.%O.N] {} {}
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%struct_type.base.9c6 [concrete = constants.%complete_type.121]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%O
// CHECK:STDOUT:   .NBase = <poisoned>
// CHECK:STDOUT:   .base = %.loc17
// CHECK:STDOUT:   .HasN1 = <poisoned>
// CHECK:STDOUT:   .HasN2 = <poisoned>
// CHECK:STDOUT:   .N = %O.N.decl
// CHECK:STDOUT:   extend %NBase.ref
// CHECK:STDOUT:   extend @O.as.HasN1.impl.%HasN1.ref
// CHECK:STDOUT:   extend @O.as.HasN2.impl.%HasN2.ref
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @NBase.N() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @HasN1.N(@HasN1.%Self: %HasN1.type) {
// CHECK:STDOUT:   fn();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @HasN2.N(@HasN2.%Self: %HasN2.type) {
// CHECK:STDOUT:   fn();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @O.as.HasN1.impl.N() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @O.as.HasN2.impl.N() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @O.N();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @P(%o.param: %O) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %O.ref.loc28: type = name_ref O, file.%O.decl [concrete = constants.%O]
// CHECK:STDOUT:   %N.ref.loc28: %O.N.type = name_ref N, @O.%O.N.decl [concrete = constants.%O.N]
// CHECK:STDOUT:   %O.N.call.loc28: init %empty_tuple.type = call %N.ref.loc28()
// CHECK:STDOUT:   %o.ref: %O = name_ref o, %o
// CHECK:STDOUT:   %N.ref.loc29: %O.N.type = name_ref N, @O.%O.N.decl [concrete = constants.%O.N]
// CHECK:STDOUT:   %O.N.call.loc29: init %empty_tuple.type = call %N.ref.loc29()
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HasN1.N(constants.%Self.e0b) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HasN2.N(constants.%Self.e6e) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HasN1.N(constants.%HasN1.facet) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HasN2.N(constants.%HasN2.facet) {}
// CHECK:STDOUT:
