// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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
   // CHECK:STDERR: fail_ambiguous_impl_and_base.carbon:[[@LINE+3]]:4: error: ambiguous use of name `K` found in multiple extended scopes [NameAmbiguousDueToExtend]
   // CHECK:STDERR:    l.K();
   // CHECK:STDERR:    ^~~
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
// CHECK:STDOUT:   %HasF.type: type = facet_type <@HasF> [template]
// CHECK:STDOUT:   %Self.348: %HasF.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %F.type.16d: type = fn_type @F.1 [template]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [template]
// CHECK:STDOUT:   %F.a20: %F.type.16d = struct_value () [template]
// CHECK:STDOUT:   %F.assoc_type: type = assoc_entity_type %HasF.type, %F.type.16d [template]
// CHECK:STDOUT:   %assoc0.472: %F.assoc_type = assoc_entity element0, @HasF.%F.decl [template]
// CHECK:STDOUT:   %HasG.type: type = facet_type <@HasG> [template]
// CHECK:STDOUT:   %Self.4db: %HasG.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %G.type.6fe: type = fn_type @G.1 [template]
// CHECK:STDOUT:   %G.0ea: %G.type.6fe = struct_value () [template]
// CHECK:STDOUT:   %G.assoc_type: type = assoc_entity_type %HasG.type, %G.type.6fe [template]
// CHECK:STDOUT:   %assoc0.8b3: %G.assoc_type = assoc_entity element0, @HasG.%G.decl [template]
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %impl_witness.52d: <witness> = impl_witness (@impl.1.%F.decl) [template]
// CHECK:STDOUT:   %F.type.cf2: type = fn_type @F.2 [template]
// CHECK:STDOUT:   %F.f1c: %F.type.cf2 = struct_value () [template]
// CHECK:STDOUT:   %HasF.facet: %HasF.type = facet_value %C, %impl_witness.52d [template]
// CHECK:STDOUT:   %impl_witness.5f2: <witness> = impl_witness (@impl.2.%G.decl) [template]
// CHECK:STDOUT:   %G.type.892: type = fn_type @G.2 [template]
// CHECK:STDOUT:   %G.90b: %G.type.892 = struct_value () [template]
// CHECK:STDOUT:   %HasG.facet: %HasG.type = facet_value %C, %impl_witness.5f2 [template]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [template]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template]
// CHECK:STDOUT:   %H.type: type = fn_type @H [template]
// CHECK:STDOUT:   %H: %H.type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .HasF = %HasF.decl
// CHECK:STDOUT:     .HasG = %HasG.decl
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .H = %H.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %HasF.decl: type = interface_decl @HasF [template = constants.%HasF.type] {} {}
// CHECK:STDOUT:   %HasG.decl: type = interface_decl @HasG [template = constants.%HasG.type] {} {}
// CHECK:STDOUT:   %C.decl: type = class_decl @C [template = constants.%C] {} {}
// CHECK:STDOUT:   %H.decl: %H.type = fn_decl @H [template = constants.%H] {
// CHECK:STDOUT:     %c.patt: %C = binding_pattern c
// CHECK:STDOUT:     %c.param_patt: %C = value_param_pattern %c.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %c.param: %C = value_param runtime_param0
// CHECK:STDOUT:     %C.ref.loc21: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %c: %C = bind_name c, %c.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @HasF {
// CHECK:STDOUT:   %Self: %HasF.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.348]
// CHECK:STDOUT:   %F.decl: %F.type.16d = fn_decl @F.1 [template = constants.%F.a20] {} {}
// CHECK:STDOUT:   %assoc0: %F.assoc_type = assoc_entity element0, %F.decl [template = constants.%assoc0.472]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .F = %assoc0
// CHECK:STDOUT:   witness = (%F.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @HasG {
// CHECK:STDOUT:   %Self: %HasG.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.4db]
// CHECK:STDOUT:   %G.decl: %G.type.6fe = fn_decl @G.1 [template = constants.%G.0ea] {} {}
// CHECK:STDOUT:   %assoc0: %G.assoc_type = assoc_entity element0, %G.decl [template = constants.%assoc0.8b3]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .G = %assoc0
// CHECK:STDOUT:   witness = (%G.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.1: %Self.ref as %HasF.ref {
// CHECK:STDOUT:   %F.decl: %F.type.cf2 = fn_decl @F.2 [template = constants.%F.f1c] {} {}
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   witness = @C.%impl_witness.loc13
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.2: %Self.ref as %HasG.ref {
// CHECK:STDOUT:   %G.decl: %G.type.892 = fn_decl @G.2 [template = constants.%G.90b] {} {}
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .G = %G.decl
// CHECK:STDOUT:   witness = @C.%impl_witness.loc16
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   impl_decl @impl.1 [template] {} {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%C [template = constants.%C]
// CHECK:STDOUT:     %HasF.ref: type = name_ref HasF, file.%HasF.decl [template = constants.%HasF.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness.loc13: <witness> = impl_witness (@impl.1.%F.decl) [template = constants.%impl_witness.52d]
// CHECK:STDOUT:   impl_decl @impl.2 [template] {} {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%C [template = constants.%C]
// CHECK:STDOUT:     %HasG.ref: type = name_ref HasG, file.%HasG.decl [template = constants.%HasG.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness.loc16: <witness> = impl_witness (@impl.2.%G.decl) [template = constants.%impl_witness.5f2]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template = constants.%complete_type]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT:   extend @impl.1.%HasF.ref
// CHECK:STDOUT:   extend @impl.2.%HasG.ref
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F.1(@HasF.%Self: %HasF.type) {
// CHECK:STDOUT:
// CHECK:STDOUT:   fn();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @G.1(@HasG.%Self: %HasG.type) {
// CHECK:STDOUT:
// CHECK:STDOUT:   fn();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.2() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G.2() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @H(%c.param_patt: %C) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %C.ref.loc22: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:   %F.ref.loc22: %F.assoc_type = name_ref F, @HasF.%assoc0 [template = constants.%assoc0.472]
// CHECK:STDOUT:   %impl.elem0.loc22: %F.type.16d = impl_witness_access constants.%impl_witness.52d, element0 [template = constants.%F.f1c]
// CHECK:STDOUT:   %F.call.loc22: init %empty_tuple.type = call %impl.elem0.loc22()
// CHECK:STDOUT:   %c.ref.loc23: %C = name_ref c, %c
// CHECK:STDOUT:   %F.ref.loc23: %F.assoc_type = name_ref F, @HasF.%assoc0 [template = constants.%assoc0.472]
// CHECK:STDOUT:   %impl.elem0.loc23: %F.type.16d = impl_witness_access constants.%impl_witness.52d, element0 [template = constants.%F.f1c]
// CHECK:STDOUT:   %F.call.loc23: init %empty_tuple.type = call %impl.elem0.loc23()
// CHECK:STDOUT:   %C.ref.loc24: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:   %G.ref.loc24: %G.assoc_type = name_ref G, @HasG.%assoc0 [template = constants.%assoc0.8b3]
// CHECK:STDOUT:   %impl.elem0.loc24: %G.type.6fe = impl_witness_access constants.%impl_witness.5f2, element0 [template = constants.%G.90b]
// CHECK:STDOUT:   %G.call.loc24: init %empty_tuple.type = call %impl.elem0.loc24()
// CHECK:STDOUT:   %c.ref.loc25: %C = name_ref c, %c
// CHECK:STDOUT:   %G.ref.loc25: %G.assoc_type = name_ref G, @HasG.%assoc0 [template = constants.%assoc0.8b3]
// CHECK:STDOUT:   %impl.elem0.loc25: %G.type.6fe = impl_witness_access constants.%impl_witness.5f2, element0 [template = constants.%G.90b]
// CHECK:STDOUT:   %G.call.loc25: init %empty_tuple.type = call %impl.elem0.loc25()
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(constants.%Self.348) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @G.1(constants.%Self.4db) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(constants.%HasF.facet) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @G.1(constants.%HasG.facet) {}
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_ambiguous_impls.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %HasA1.type: type = facet_type <@HasA1> [template]
// CHECK:STDOUT:   %Self.6df: %HasA1.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %A.type.319: type = fn_type @A.1 [template]
// CHECK:STDOUT:   %A.072: %A.type.319 = struct_value () [template]
// CHECK:STDOUT:   %A.assoc_type.f09: type = assoc_entity_type %HasA1.type, %A.type.319 [template]
// CHECK:STDOUT:   %assoc0.680: %A.assoc_type.f09 = assoc_entity element0, @HasA1.%A.decl [template]
// CHECK:STDOUT:   %HasA2.type: type = facet_type <@HasA2> [template]
// CHECK:STDOUT:   %Self.0d0: %HasA2.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %A.type.8fb: type = fn_type @A.2 [template]
// CHECK:STDOUT:   %A.838: %A.type.8fb = struct_value () [template]
// CHECK:STDOUT:   %A.assoc_type.c72: type = assoc_entity_type %HasA2.type, %A.type.8fb [template]
// CHECK:STDOUT:   %assoc0.abb: %A.assoc_type.c72 = assoc_entity element0, @HasA2.%A.decl [template]
// CHECK:STDOUT:   %D: type = class_type @D [template]
// CHECK:STDOUT:   %impl_witness.bb1: <witness> = impl_witness (@impl.1.%A.decl) [template]
// CHECK:STDOUT:   %A.type.79c: type = fn_type @A.3 [template]
// CHECK:STDOUT:   %A.cd0: %A.type.79c = struct_value () [template]
// CHECK:STDOUT:   %HasA1.facet: %HasA1.type = facet_value %D, %impl_witness.bb1 [template]
// CHECK:STDOUT:   %impl_witness.fce: <witness> = impl_witness (@impl.2.%A.decl) [template]
// CHECK:STDOUT:   %A.type.c3a: type = fn_type @A.4 [template]
// CHECK:STDOUT:   %A.cfa: %A.type.c3a = struct_value () [template]
// CHECK:STDOUT:   %HasA2.facet: %HasA2.type = facet_value %D, %impl_witness.fce [template]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [template]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template]
// CHECK:STDOUT:   %B.type: type = fn_type @B [template]
// CHECK:STDOUT:   %B: %B.type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .HasA1 = %HasA1.decl
// CHECK:STDOUT:     .HasA2 = %HasA2.decl
// CHECK:STDOUT:     .D = %D.decl
// CHECK:STDOUT:     .B = %B.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %HasA1.decl: type = interface_decl @HasA1 [template = constants.%HasA1.type] {} {}
// CHECK:STDOUT:   %HasA2.decl: type = interface_decl @HasA2 [template = constants.%HasA2.type] {} {}
// CHECK:STDOUT:   %D.decl: type = class_decl @D [template = constants.%D] {} {}
// CHECK:STDOUT:   %B.decl: %B.type = fn_decl @B [template = constants.%B] {
// CHECK:STDOUT:     %d.patt: %D = binding_pattern d
// CHECK:STDOUT:     %d.param_patt: %D = value_param_pattern %d.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %d.param: %D = value_param runtime_param0
// CHECK:STDOUT:     %D.ref.loc21: type = name_ref D, file.%D.decl [template = constants.%D]
// CHECK:STDOUT:     %d: %D = bind_name d, %d.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @HasA1 {
// CHECK:STDOUT:   %Self: %HasA1.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.6df]
// CHECK:STDOUT:   %A.decl: %A.type.319 = fn_decl @A.1 [template = constants.%A.072] {} {}
// CHECK:STDOUT:   %assoc0: %A.assoc_type.f09 = assoc_entity element0, %A.decl [template = constants.%assoc0.680]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .A = %assoc0
// CHECK:STDOUT:   witness = (%A.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @HasA2 {
// CHECK:STDOUT:   %Self: %HasA2.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.0d0]
// CHECK:STDOUT:   %A.decl: %A.type.8fb = fn_decl @A.2 [template = constants.%A.838] {} {}
// CHECK:STDOUT:   %assoc0: %A.assoc_type.c72 = assoc_entity element0, %A.decl [template = constants.%assoc0.abb]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .A = %assoc0
// CHECK:STDOUT:   witness = (%A.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.1: %Self.ref as %HasA1.ref {
// CHECK:STDOUT:   %A.decl: %A.type.79c = fn_decl @A.3 [template = constants.%A.cd0] {} {}
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .A = %A.decl
// CHECK:STDOUT:   witness = @D.%impl_witness.loc13
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.2: %Self.ref as %HasA2.ref {
// CHECK:STDOUT:   %A.decl: %A.type.c3a = fn_decl @A.4 [template = constants.%A.cfa] {} {}
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .A = %A.decl
// CHECK:STDOUT:   witness = @D.%impl_witness.loc16
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @D {
// CHECK:STDOUT:   impl_decl @impl.1 [template] {} {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%D [template = constants.%D]
// CHECK:STDOUT:     %HasA1.ref: type = name_ref HasA1, file.%HasA1.decl [template = constants.%HasA1.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness.loc13: <witness> = impl_witness (@impl.1.%A.decl) [template = constants.%impl_witness.bb1]
// CHECK:STDOUT:   impl_decl @impl.2 [template] {} {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%D [template = constants.%D]
// CHECK:STDOUT:     %HasA2.ref: type = name_ref HasA2, file.%HasA2.decl [template = constants.%HasA2.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness.loc16: <witness> = impl_witness (@impl.2.%A.decl) [template = constants.%impl_witness.fce]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template = constants.%complete_type]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%D
// CHECK:STDOUT:   extend @impl.1.%HasA1.ref
// CHECK:STDOUT:   extend @impl.2.%HasA2.ref
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @A.1(@HasA1.%Self: %HasA1.type) {
// CHECK:STDOUT:
// CHECK:STDOUT:   fn();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @A.2(@HasA2.%Self: %HasA2.type) {
// CHECK:STDOUT:
// CHECK:STDOUT:   fn();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @A.3() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @A.4() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @B(%d.param_patt: %D) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %D.ref.loc26: type = name_ref D, file.%D.decl [template = constants.%D]
// CHECK:STDOUT:   %A.ref.loc26: <error> = name_ref A, <error> [template = <error>]
// CHECK:STDOUT:   %d.ref: %D = name_ref d, %d
// CHECK:STDOUT:   %A.ref.loc31: <error> = name_ref A, <error> [template = <error>]
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A.1(constants.%Self.6df) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A.2(constants.%Self.0d0) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A.1(constants.%HasA1.facet) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A.2(constants.%HasA2.facet) {}
// CHECK:STDOUT:
// CHECK:STDOUT: --- different_impl_and_base.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %HasI.type: type = facet_type <@HasI> [template]
// CHECK:STDOUT:   %Self: %HasI.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %I.type.ed9: type = fn_type @I.1 [template]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [template]
// CHECK:STDOUT:   %I.f16: %I.type.ed9 = struct_value () [template]
// CHECK:STDOUT:   %I.assoc_type: type = assoc_entity_type %HasI.type, %I.type.ed9 [template]
// CHECK:STDOUT:   %assoc0: %I.assoc_type = assoc_entity element0, @HasI.%I.decl [template]
// CHECK:STDOUT:   %B: type = class_type @B [template]
// CHECK:STDOUT:   %J.type: type = fn_type @J [template]
// CHECK:STDOUT:   %J: %J.type = struct_value () [template]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [template]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [template]
// CHECK:STDOUT:   %E: type = class_type @E [template]
// CHECK:STDOUT:   %E.elem: type = unbound_element_type %E, %B [template]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (@impl.%I.decl) [template]
// CHECK:STDOUT:   %I.type.340: type = fn_type @I.2 [template]
// CHECK:STDOUT:   %I.c36: %I.type.340 = struct_value () [template]
// CHECK:STDOUT:   %HasI.facet: %HasI.type = facet_value %E, %impl_witness [template]
// CHECK:STDOUT:   %struct_type.base.0ff: type = struct_type {.base: %B} [template]
// CHECK:STDOUT:   %complete_type.98e: <witness> = complete_type_witness %struct_type.base.0ff [template]
// CHECK:STDOUT:   %H.type: type = fn_type @H [template]
// CHECK:STDOUT:   %H: %H.type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .HasI = %HasI.decl
// CHECK:STDOUT:     .B = %B.decl
// CHECK:STDOUT:     .E = %E.decl
// CHECK:STDOUT:     .H = %H.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %HasI.decl: type = interface_decl @HasI [template = constants.%HasI.type] {} {}
// CHECK:STDOUT:   %B.decl: type = class_decl @B [template = constants.%B] {} {}
// CHECK:STDOUT:   %E.decl: type = class_decl @E [template = constants.%E] {} {}
// CHECK:STDOUT:   %H.decl: %H.type = fn_decl @H [template = constants.%H] {
// CHECK:STDOUT:     %e.patt: %E = binding_pattern e
// CHECK:STDOUT:     %e.param_patt: %E = value_param_pattern %e.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %e.param: %E = value_param runtime_param0
// CHECK:STDOUT:     %E.ref.loc19: type = name_ref E, file.%E.decl [template = constants.%E]
// CHECK:STDOUT:     %e: %E = bind_name e, %e.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @HasI {
// CHECK:STDOUT:   %Self: %HasI.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   %I.decl: %I.type.ed9 = fn_decl @I.1 [template = constants.%I.f16] {} {}
// CHECK:STDOUT:   %assoc0: %I.assoc_type = assoc_entity element0, %I.decl [template = constants.%assoc0]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .I = %assoc0
// CHECK:STDOUT:   witness = (%I.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl: %Self.ref as %HasI.ref {
// CHECK:STDOUT:   %I.decl: %I.type.340 = fn_decl @I.2 [template = constants.%I.c36] {} {}
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .I = %I.decl
// CHECK:STDOUT:   witness = @E.%impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @B {
// CHECK:STDOUT:   %J.decl: %J.type = fn_decl @J [template = constants.%J] {} {}
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template = constants.%complete_type.357]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%B
// CHECK:STDOUT:   .J = %J.decl
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @E {
// CHECK:STDOUT:   %B.ref: type = name_ref B, file.%B.decl [template = constants.%B]
// CHECK:STDOUT:   %.loc13: %E.elem = base_decl %B.ref, element0 [template]
// CHECK:STDOUT:   impl_decl @impl [template] {} {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%E [template = constants.%E]
// CHECK:STDOUT:     %HasI.ref: type = name_ref HasI, file.%HasI.decl [template = constants.%HasI.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (@impl.%I.decl) [template = constants.%impl_witness]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.base.0ff [template = constants.%complete_type.98e]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%E
// CHECK:STDOUT:   .base = %.loc13
// CHECK:STDOUT:   extend %B.ref
// CHECK:STDOUT:   extend @impl.%HasI.ref
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @I.1(@HasI.%Self: %HasI.type) {
// CHECK:STDOUT:
// CHECK:STDOUT:   fn();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @J() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @I.2() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @H(%e.param_patt: %E) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %E.ref.loc20: type = name_ref E, file.%E.decl [template = constants.%E]
// CHECK:STDOUT:   %I.ref.loc20: %I.assoc_type = name_ref I, @HasI.%assoc0 [template = constants.%assoc0]
// CHECK:STDOUT:   %impl.elem0.loc20: %I.type.ed9 = impl_witness_access constants.%impl_witness, element0 [template = constants.%I.c36]
// CHECK:STDOUT:   %I.call.loc20: init %empty_tuple.type = call %impl.elem0.loc20()
// CHECK:STDOUT:   %e.ref.loc21: %E = name_ref e, %e
// CHECK:STDOUT:   %I.ref.loc21: %I.assoc_type = name_ref I, @HasI.%assoc0 [template = constants.%assoc0]
// CHECK:STDOUT:   %impl.elem0.loc21: %I.type.ed9 = impl_witness_access constants.%impl_witness, element0 [template = constants.%I.c36]
// CHECK:STDOUT:   %I.call.loc21: init %empty_tuple.type = call %impl.elem0.loc21()
// CHECK:STDOUT:   %E.ref.loc22: type = name_ref E, file.%E.decl [template = constants.%E]
// CHECK:STDOUT:   %J.ref.loc22: %J.type = name_ref J, @B.%J.decl [template = constants.%J]
// CHECK:STDOUT:   %J.call.loc22: init %empty_tuple.type = call %J.ref.loc22()
// CHECK:STDOUT:   %e.ref.loc23: %E = name_ref e, %e
// CHECK:STDOUT:   %J.ref.loc23: %J.type = name_ref J, @B.%J.decl [template = constants.%J]
// CHECK:STDOUT:   %J.call.loc23: init %empty_tuple.type = call %J.ref.loc23()
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.1(constants.%Self) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.1(constants.%HasI.facet) {}
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_ambiguous_impl_and_base.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Base: type = class_type @Base [template]
// CHECK:STDOUT:   %K.type.4ac: type = fn_type @K.1 [template]
// CHECK:STDOUT:   %K.5b6: %K.type.4ac = struct_value () [template]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [template]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [template]
// CHECK:STDOUT:   %HasK.type: type = facet_type <@HasK> [template]
// CHECK:STDOUT:   %Self: %HasK.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %K.type.cec: type = fn_type @K.2 [template]
// CHECK:STDOUT:   %K.389: %K.type.cec = struct_value () [template]
// CHECK:STDOUT:   %K.assoc_type: type = assoc_entity_type %HasK.type, %K.type.cec [template]
// CHECK:STDOUT:   %assoc0: %K.assoc_type = assoc_entity element0, @HasK.%K.decl [template]
// CHECK:STDOUT:   %L: type = class_type @L [template]
// CHECK:STDOUT:   %L.elem: type = unbound_element_type %L, %Base [template]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (@impl.%K.decl) [template]
// CHECK:STDOUT:   %K.type.5ab: type = fn_type @K.3 [template]
// CHECK:STDOUT:   %K.d12: %K.type.5ab = struct_value () [template]
// CHECK:STDOUT:   %HasK.facet: %HasK.type = facet_value %L, %impl_witness [template]
// CHECK:STDOUT:   %struct_type.base.b1e: type = struct_type {.base: %Base} [template]
// CHECK:STDOUT:   %complete_type.15c: <witness> = complete_type_witness %struct_type.base.b1e [template]
// CHECK:STDOUT:   %M.type: type = fn_type @M [template]
// CHECK:STDOUT:   %M: %M.type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Base = %Base.decl
// CHECK:STDOUT:     .HasK = %HasK.decl
// CHECK:STDOUT:     .L = %L.decl
// CHECK:STDOUT:     .M = %M.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Base.decl: type = class_decl @Base [template = constants.%Base] {} {}
// CHECK:STDOUT:   %HasK.decl: type = interface_decl @HasK [template = constants.%HasK.type] {} {}
// CHECK:STDOUT:   %L.decl: type = class_decl @L [template = constants.%L] {} {}
// CHECK:STDOUT:   %M.decl: %M.type = fn_decl @M [template = constants.%M] {
// CHECK:STDOUT:     %l.patt: %L = binding_pattern l
// CHECK:STDOUT:     %l.param_patt: %L = value_param_pattern %l.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %l.param: %L = value_param runtime_param0
// CHECK:STDOUT:     %L.ref.loc19: type = name_ref L, file.%L.decl [template = constants.%L]
// CHECK:STDOUT:     %l: %L = bind_name l, %l.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @HasK {
// CHECK:STDOUT:   %Self: %HasK.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   %K.decl: %K.type.cec = fn_decl @K.2 [template = constants.%K.389] {} {}
// CHECK:STDOUT:   %assoc0: %K.assoc_type = assoc_entity element0, %K.decl [template = constants.%assoc0]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .K = %assoc0
// CHECK:STDOUT:   witness = (%K.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl: %Self.ref as %HasK.ref {
// CHECK:STDOUT:   %K.decl: %K.type.5ab = fn_decl @K.3 [template = constants.%K.d12] {} {}
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .K = %K.decl
// CHECK:STDOUT:   witness = @L.%impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Base {
// CHECK:STDOUT:   %K.decl: %K.type.4ac = fn_decl @K.1 [template = constants.%K.5b6] {} {}
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template = constants.%complete_type.357]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Base
// CHECK:STDOUT:   .K = %K.decl
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @L {
// CHECK:STDOUT:   %Base.ref: type = name_ref Base, file.%Base.decl [template = constants.%Base]
// CHECK:STDOUT:   %.loc13: %L.elem = base_decl %Base.ref, element0 [template]
// CHECK:STDOUT:   impl_decl @impl [template] {} {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%L [template = constants.%L]
// CHECK:STDOUT:     %HasK.ref: type = name_ref HasK, file.%HasK.decl [template = constants.%HasK.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (@impl.%K.decl) [template = constants.%impl_witness]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.base.b1e [template = constants.%complete_type.15c]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%L
// CHECK:STDOUT:   .base = %.loc13
// CHECK:STDOUT:   extend %Base.ref
// CHECK:STDOUT:   extend @impl.%HasK.ref
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @K.1() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @K.2(@HasK.%Self: %HasK.type) {
// CHECK:STDOUT:
// CHECK:STDOUT:   fn();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @K.3() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @M(%l.param_patt: %L) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %L.ref.loc24: type = name_ref L, file.%L.decl [template = constants.%L]
// CHECK:STDOUT:   %K.ref.loc24: <error> = name_ref K, <error> [template = <error>]
// CHECK:STDOUT:   %l.ref: %L = name_ref l, %l
// CHECK:STDOUT:   %K.ref.loc28: <error> = name_ref K, <error> [template = <error>]
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @K.2(constants.%Self) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @K.2(constants.%HasK.facet) {}
// CHECK:STDOUT:
// CHECK:STDOUT: --- ambiguity_hidden.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %NBase: type = class_type @NBase [template]
// CHECK:STDOUT:   %N.type.e79: type = fn_type @N.1 [template]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [template]
// CHECK:STDOUT:   %N.950: %N.type.e79 = struct_value () [template]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [template]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [template]
// CHECK:STDOUT:   %HasN1.type: type = facet_type <@HasN1> [template]
// CHECK:STDOUT:   %Self.f9f: %HasN1.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %N.type.4c7: type = fn_type @N.2 [template]
// CHECK:STDOUT:   %N.ba3: %N.type.4c7 = struct_value () [template]
// CHECK:STDOUT:   %N.assoc_type.58f: type = assoc_entity_type %HasN1.type, %N.type.4c7 [template]
// CHECK:STDOUT:   %assoc0.f1a: %N.assoc_type.58f = assoc_entity element0, @HasN1.%N.decl [template]
// CHECK:STDOUT:   %HasN2.type: type = facet_type <@HasN2> [template]
// CHECK:STDOUT:   %Self.82d: %HasN2.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %N.type.06f: type = fn_type @N.3 [template]
// CHECK:STDOUT:   %N.039: %N.type.06f = struct_value () [template]
// CHECK:STDOUT:   %N.assoc_type.fb0: type = assoc_entity_type %HasN2.type, %N.type.06f [template]
// CHECK:STDOUT:   %assoc0.b5a: %N.assoc_type.fb0 = assoc_entity element0, @HasN2.%N.decl [template]
// CHECK:STDOUT:   %O: type = class_type @O [template]
// CHECK:STDOUT:   %O.elem: type = unbound_element_type %O, %NBase [template]
// CHECK:STDOUT:   %impl_witness.d2b: <witness> = impl_witness (@impl.1.%N.decl) [template]
// CHECK:STDOUT:   %N.type.b8d: type = fn_type @N.4 [template]
// CHECK:STDOUT:   %N.cd2: %N.type.b8d = struct_value () [template]
// CHECK:STDOUT:   %HasN1.facet: %HasN1.type = facet_value %O, %impl_witness.d2b [template]
// CHECK:STDOUT:   %impl_witness.93f: <witness> = impl_witness (@impl.2.%N.decl) [template]
// CHECK:STDOUT:   %N.type.afd: type = fn_type @N.5 [template]
// CHECK:STDOUT:   %N.512: %N.type.afd = struct_value () [template]
// CHECK:STDOUT:   %HasN2.facet: %HasN2.type = facet_value %O, %impl_witness.93f [template]
// CHECK:STDOUT:   %N.type.964: type = fn_type @N.6 [template]
// CHECK:STDOUT:   %N.3dd: %N.type.964 = struct_value () [template]
// CHECK:STDOUT:   %struct_type.base.9c6: type = struct_type {.base: %NBase} [template]
// CHECK:STDOUT:   %complete_type.121: <witness> = complete_type_witness %struct_type.base.9c6 [template]
// CHECK:STDOUT:   %P.type: type = fn_type @P [template]
// CHECK:STDOUT:   %P: %P.type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .NBase = %NBase.decl
// CHECK:STDOUT:     .HasN1 = %HasN1.decl
// CHECK:STDOUT:     .HasN2 = %HasN2.decl
// CHECK:STDOUT:     .O = %O.decl
// CHECK:STDOUT:     .P = %P.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %NBase.decl: type = class_decl @NBase [template = constants.%NBase] {} {}
// CHECK:STDOUT:   %HasN1.decl: type = interface_decl @HasN1 [template = constants.%HasN1.type] {} {}
// CHECK:STDOUT:   %HasN2.decl: type = interface_decl @HasN2 [template = constants.%HasN2.type] {} {}
// CHECK:STDOUT:   %O.decl: type = class_decl @O [template = constants.%O] {} {}
// CHECK:STDOUT:   %P.decl: %P.type = fn_decl @P [template = constants.%P] {
// CHECK:STDOUT:     %o.patt: %O = binding_pattern o
// CHECK:STDOUT:     %o.param_patt: %O = value_param_pattern %o.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %o.param: %O = value_param runtime_param0
// CHECK:STDOUT:     %O.ref.loc27: type = name_ref O, file.%O.decl [template = constants.%O]
// CHECK:STDOUT:     %o: %O = bind_name o, %o.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @HasN1 {
// CHECK:STDOUT:   %Self: %HasN1.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.f9f]
// CHECK:STDOUT:   %N.decl: %N.type.4c7 = fn_decl @N.2 [template = constants.%N.ba3] {} {}
// CHECK:STDOUT:   %assoc0: %N.assoc_type.58f = assoc_entity element0, %N.decl [template = constants.%assoc0.f1a]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .N = %assoc0
// CHECK:STDOUT:   witness = (%N.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @HasN2 {
// CHECK:STDOUT:   %Self: %HasN2.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.82d]
// CHECK:STDOUT:   %N.decl: %N.type.06f = fn_decl @N.3 [template = constants.%N.039] {} {}
// CHECK:STDOUT:   %assoc0: %N.assoc_type.fb0 = assoc_entity element0, %N.decl [template = constants.%assoc0.b5a]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .N = %assoc0
// CHECK:STDOUT:   witness = (%N.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.1: %Self.ref as %HasN1.ref {
// CHECK:STDOUT:   %N.decl: %N.type.b8d = fn_decl @N.4 [template = constants.%N.cd2] {} {}
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .N = %N.decl
// CHECK:STDOUT:   witness = @O.%impl_witness.loc18
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.2: %Self.ref as %HasN2.ref {
// CHECK:STDOUT:   %N.decl: %N.type.afd = fn_decl @N.5 [template = constants.%N.512] {} {}
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .N = %N.decl
// CHECK:STDOUT:   witness = @O.%impl_witness.loc21
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @NBase {
// CHECK:STDOUT:   %N.decl: %N.type.e79 = fn_decl @N.1 [template = constants.%N.950] {} {}
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template = constants.%complete_type.357]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%NBase
// CHECK:STDOUT:   .N = %N.decl
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @O {
// CHECK:STDOUT:   %NBase.ref: type = name_ref NBase, file.%NBase.decl [template = constants.%NBase]
// CHECK:STDOUT:   %.loc17: %O.elem = base_decl %NBase.ref, element0 [template]
// CHECK:STDOUT:   impl_decl @impl.1 [template] {} {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%O [template = constants.%O]
// CHECK:STDOUT:     %HasN1.ref: type = name_ref HasN1, file.%HasN1.decl [template = constants.%HasN1.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness.loc18: <witness> = impl_witness (@impl.1.%N.decl) [template = constants.%impl_witness.d2b]
// CHECK:STDOUT:   impl_decl @impl.2 [template] {} {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%O [template = constants.%O]
// CHECK:STDOUT:     %HasN2.ref: type = name_ref HasN2, file.%HasN2.decl [template = constants.%HasN2.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness.loc21: <witness> = impl_witness (@impl.2.%N.decl) [template = constants.%impl_witness.93f]
// CHECK:STDOUT:   %N.decl: %N.type.964 = fn_decl @N.6 [template = constants.%N.3dd] {} {}
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.base.9c6 [template = constants.%complete_type.121]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%O
// CHECK:STDOUT:   .base = %.loc17
// CHECK:STDOUT:   .N = %N.decl
// CHECK:STDOUT:   extend %NBase.ref
// CHECK:STDOUT:   extend @impl.1.%HasN1.ref
// CHECK:STDOUT:   extend @impl.2.%HasN2.ref
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @N.1() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @N.2(@HasN1.%Self: %HasN1.type) {
// CHECK:STDOUT:
// CHECK:STDOUT:   fn();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @N.3(@HasN2.%Self: %HasN2.type) {
// CHECK:STDOUT:
// CHECK:STDOUT:   fn();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @N.4() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @N.5() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @N.6();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @P(%o.param_patt: %O) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %O.ref.loc28: type = name_ref O, file.%O.decl [template = constants.%O]
// CHECK:STDOUT:   %N.ref.loc28: %N.type.964 = name_ref N, @O.%N.decl [template = constants.%N.3dd]
// CHECK:STDOUT:   %N.call.loc28: init %empty_tuple.type = call %N.ref.loc28()
// CHECK:STDOUT:   %o.ref: %O = name_ref o, %o
// CHECK:STDOUT:   %N.ref.loc29: %N.type.964 = name_ref N, @O.%N.decl [template = constants.%N.3dd]
// CHECK:STDOUT:   %N.call.loc29: init %empty_tuple.type = call %N.ref.loc29()
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @N.2(constants.%Self.f9f) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @N.3(constants.%Self.82d) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @N.2(constants.%HasN1.facet) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @N.3(constants.%HasN2.facet) {}
// CHECK:STDOUT:
