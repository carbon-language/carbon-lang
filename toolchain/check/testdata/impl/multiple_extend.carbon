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
// CHECK:STDOUT:   %F.type.b7b: type = fn_type @F.1 [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %F.f50: %F.type.b7b = struct_value () [concrete]
// CHECK:STDOUT:   %HasF.assoc_type: type = assoc_entity_type %HasF.type [concrete]
// CHECK:STDOUT:   %assoc0.992: %HasF.assoc_type = assoc_entity element0, @HasF.%F.decl [concrete]
// CHECK:STDOUT:   %HasG.type: type = facet_type <@HasG> [concrete]
// CHECK:STDOUT:   %Self.d42: %HasG.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %G.type.d27: type = fn_type @G.1 [concrete]
// CHECK:STDOUT:   %G.688: %G.type.d27 = struct_value () [concrete]
// CHECK:STDOUT:   %HasG.assoc_type: type = assoc_entity_type %HasG.type [concrete]
// CHECK:STDOUT:   %assoc0.58a: %HasG.assoc_type = assoc_entity element0, @HasG.%G.decl [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %impl_witness.329: <witness> = impl_witness (@impl.670.%F.decl) [concrete]
// CHECK:STDOUT:   %F.type.a65: type = fn_type @F.2 [concrete]
// CHECK:STDOUT:   %F.ad8: %F.type.a65 = struct_value () [concrete]
// CHECK:STDOUT:   %HasF.facet: %HasF.type = facet_value %C, (%impl_witness.329) [concrete]
// CHECK:STDOUT:   %impl_witness.42a: <witness> = impl_witness (@impl.096.%G.decl) [concrete]
// CHECK:STDOUT:   %G.type.cf6: type = fn_type @G.2 [concrete]
// CHECK:STDOUT:   %G.957: %G.type.cf6 = struct_value () [concrete]
// CHECK:STDOUT:   %HasG.facet: %HasG.type = facet_value %C, (%impl_witness.42a) [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %H.type: type = fn_type @H [concrete]
// CHECK:STDOUT:   %H: %H.type = struct_value () [concrete]
// CHECK:STDOUT:   %.1bc: type = fn_type_with_self_type %F.type.b7b, %HasF.facet [concrete]
// CHECK:STDOUT:   %.bba: type = fn_type_with_self_type %G.type.d27, %HasG.facet [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .HasF = %HasF.decl
// CHECK:STDOUT:     .HasG = %HasG.decl
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .H = %H.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %HasF.decl: type = interface_decl @HasF [concrete = constants.%HasF.type] {} {}
// CHECK:STDOUT:   %HasG.decl: type = interface_decl @HasG [concrete = constants.%HasG.type] {} {}
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   %H.decl: %H.type = fn_decl @H [concrete = constants.%H] {
// CHECK:STDOUT:     %c.patt: %C = binding_pattern c
// CHECK:STDOUT:     %c.param_patt: %C = value_param_pattern %c.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %c.param: %C = value_param call_param0
// CHECK:STDOUT:     %C.ref.loc21: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %c: %C = bind_name c, %c.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @HasF {
// CHECK:STDOUT:   %Self: %HasF.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.f0c]
// CHECK:STDOUT:   %F.decl: %F.type.b7b = fn_decl @F.1 [concrete = constants.%F.f50] {} {}
// CHECK:STDOUT:   %assoc0: %HasF.assoc_type = assoc_entity element0, %F.decl [concrete = constants.%assoc0.992]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .F = %assoc0
// CHECK:STDOUT:   .HasG = <poisoned>
// CHECK:STDOUT:   .G = <poisoned>
// CHECK:STDOUT:   witness = (%F.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @HasG {
// CHECK:STDOUT:   %Self: %HasG.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.d42]
// CHECK:STDOUT:   %G.decl: %G.type.d27 = fn_decl @G.1 [concrete = constants.%G.688] {} {}
// CHECK:STDOUT:   %assoc0: %HasG.assoc_type = assoc_entity element0, %G.decl [concrete = constants.%assoc0.58a]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .G = %assoc0
// CHECK:STDOUT:   .F = <poisoned>
// CHECK:STDOUT:   witness = (%G.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.670: %Self.ref as %HasF.ref {
// CHECK:STDOUT:   %F.decl: %F.type.a65 = fn_decl @F.2 [concrete = constants.%F.ad8] {} {}
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   witness = @C.%impl_witness.loc13
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.096: %Self.ref as %HasG.ref {
// CHECK:STDOUT:   %G.decl: %G.type.cf6 = fn_decl @G.2 [concrete = constants.%G.957] {} {}
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .G = %G.decl
// CHECK:STDOUT:   witness = @C.%impl_witness.loc16
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   impl_decl @impl.670 [concrete] {} {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%C [concrete = constants.%C]
// CHECK:STDOUT:     %HasF.ref: type = name_ref HasF, file.%HasF.decl [concrete = constants.%HasF.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness.loc13: <witness> = impl_witness (@impl.670.%F.decl) [concrete = constants.%impl_witness.329]
// CHECK:STDOUT:   impl_decl @impl.096 [concrete] {} {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%C [concrete = constants.%C]
// CHECK:STDOUT:     %HasG.ref: type = name_ref HasG, file.%HasG.decl [concrete = constants.%HasG.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness.loc16: <witness> = impl_witness (@impl.096.%G.decl) [concrete = constants.%impl_witness.42a]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT:   .HasF = <poisoned>
// CHECK:STDOUT:   .HasG = <poisoned>
// CHECK:STDOUT:   .F = <poisoned>
// CHECK:STDOUT:   .G = <poisoned>
// CHECK:STDOUT:   extend @impl.670.%HasF.ref
// CHECK:STDOUT:   extend @impl.096.%HasG.ref
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F.1(@HasF.%Self: %HasF.type) {
// CHECK:STDOUT:   fn();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @G.1(@HasG.%Self: %HasG.type) {
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
// CHECK:STDOUT:   %C.ref.loc22: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %F.ref.loc22: %HasF.assoc_type = name_ref F, @HasF.%assoc0 [concrete = constants.%assoc0.992]
// CHECK:STDOUT:   %impl.elem0.loc22: %.1bc = impl_witness_access constants.%impl_witness.329, element0 [concrete = constants.%F.ad8]
// CHECK:STDOUT:   %F.call.loc22: init %empty_tuple.type = call %impl.elem0.loc22()
// CHECK:STDOUT:   %c.ref.loc23: %C = name_ref c, %c
// CHECK:STDOUT:   %F.ref.loc23: %HasF.assoc_type = name_ref F, @HasF.%assoc0 [concrete = constants.%assoc0.992]
// CHECK:STDOUT:   %impl.elem0.loc23: %.1bc = impl_witness_access constants.%impl_witness.329, element0 [concrete = constants.%F.ad8]
// CHECK:STDOUT:   %F.call.loc23: init %empty_tuple.type = call %impl.elem0.loc23()
// CHECK:STDOUT:   %C.ref.loc24: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %G.ref.loc24: %HasG.assoc_type = name_ref G, @HasG.%assoc0 [concrete = constants.%assoc0.58a]
// CHECK:STDOUT:   %impl.elem0.loc24: %.bba = impl_witness_access constants.%impl_witness.42a, element0 [concrete = constants.%G.957]
// CHECK:STDOUT:   %G.call.loc24: init %empty_tuple.type = call %impl.elem0.loc24()
// CHECK:STDOUT:   %c.ref.loc25: %C = name_ref c, %c
// CHECK:STDOUT:   %G.ref.loc25: %HasG.assoc_type = name_ref G, @HasG.%assoc0 [concrete = constants.%assoc0.58a]
// CHECK:STDOUT:   %impl.elem0.loc25: %.bba = impl_witness_access constants.%impl_witness.42a, element0 [concrete = constants.%G.957]
// CHECK:STDOUT:   %G.call.loc25: init %empty_tuple.type = call %impl.elem0.loc25()
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(constants.%Self.f0c) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @G.1(constants.%Self.d42) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(constants.%HasF.facet) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @G.1(constants.%HasG.facet) {}
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_ambiguous_impls.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %HasA1.type: type = facet_type <@HasA1> [concrete]
// CHECK:STDOUT:   %Self.90d: %HasA1.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %A.type.9c4: type = fn_type @A.1 [concrete]
// CHECK:STDOUT:   %A.c3c: %A.type.9c4 = struct_value () [concrete]
// CHECK:STDOUT:   %HasA1.assoc_type: type = assoc_entity_type %HasA1.type [concrete]
// CHECK:STDOUT:   %assoc0.b5a: %HasA1.assoc_type = assoc_entity element0, @HasA1.%A.decl [concrete]
// CHECK:STDOUT:   %HasA2.type: type = facet_type <@HasA2> [concrete]
// CHECK:STDOUT:   %Self.bdc: %HasA2.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %A.type.f71: type = fn_type @A.2 [concrete]
// CHECK:STDOUT:   %A.0eb: %A.type.f71 = struct_value () [concrete]
// CHECK:STDOUT:   %HasA2.assoc_type: type = assoc_entity_type %HasA2.type [concrete]
// CHECK:STDOUT:   %assoc0.6ed: %HasA2.assoc_type = assoc_entity element0, @HasA2.%A.decl [concrete]
// CHECK:STDOUT:   %D: type = class_type @D [concrete]
// CHECK:STDOUT:   %impl_witness.73a: <witness> = impl_witness (@impl.909.%A.decl) [concrete]
// CHECK:STDOUT:   %A.type.468: type = fn_type @A.3 [concrete]
// CHECK:STDOUT:   %A.efc: %A.type.468 = struct_value () [concrete]
// CHECK:STDOUT:   %HasA1.facet: %HasA1.type = facet_value %D, (%impl_witness.73a) [concrete]
// CHECK:STDOUT:   %impl_witness.baf: <witness> = impl_witness (@impl.390.%A.decl) [concrete]
// CHECK:STDOUT:   %A.type.938: type = fn_type @A.4 [concrete]
// CHECK:STDOUT:   %A.890: %A.type.938 = struct_value () [concrete]
// CHECK:STDOUT:   %HasA2.facet: %HasA2.type = facet_value %D, (%impl_witness.baf) [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %B.type: type = fn_type @B [concrete]
// CHECK:STDOUT:   %B: %B.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .HasA1 = %HasA1.decl
// CHECK:STDOUT:     .HasA2 = %HasA2.decl
// CHECK:STDOUT:     .D = %D.decl
// CHECK:STDOUT:     .B = %B.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %HasA1.decl: type = interface_decl @HasA1 [concrete = constants.%HasA1.type] {} {}
// CHECK:STDOUT:   %HasA2.decl: type = interface_decl @HasA2 [concrete = constants.%HasA2.type] {} {}
// CHECK:STDOUT:   %D.decl: type = class_decl @D [concrete = constants.%D] {} {}
// CHECK:STDOUT:   %B.decl: %B.type = fn_decl @B [concrete = constants.%B] {
// CHECK:STDOUT:     %d.patt: %D = binding_pattern d
// CHECK:STDOUT:     %d.param_patt: %D = value_param_pattern %d.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %d.param: %D = value_param call_param0
// CHECK:STDOUT:     %D.ref.loc21: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %d: %D = bind_name d, %d.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @HasA1 {
// CHECK:STDOUT:   %Self: %HasA1.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.90d]
// CHECK:STDOUT:   %A.decl: %A.type.9c4 = fn_decl @A.1 [concrete = constants.%A.c3c] {} {}
// CHECK:STDOUT:   %assoc0: %HasA1.assoc_type = assoc_entity element0, %A.decl [concrete = constants.%assoc0.b5a]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .A = %assoc0
// CHECK:STDOUT:   .HasA2 = <poisoned>
// CHECK:STDOUT:   witness = (%A.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @HasA2 {
// CHECK:STDOUT:   %Self: %HasA2.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.bdc]
// CHECK:STDOUT:   %A.decl: %A.type.f71 = fn_decl @A.2 [concrete = constants.%A.0eb] {} {}
// CHECK:STDOUT:   %assoc0: %HasA2.assoc_type = assoc_entity element0, %A.decl [concrete = constants.%assoc0.6ed]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .A = %assoc0
// CHECK:STDOUT:   witness = (%A.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.909: %Self.ref as %HasA1.ref {
// CHECK:STDOUT:   %A.decl: %A.type.468 = fn_decl @A.3 [concrete = constants.%A.efc] {} {}
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .A = %A.decl
// CHECK:STDOUT:   witness = @D.%impl_witness.loc13
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.390: %Self.ref as %HasA2.ref {
// CHECK:STDOUT:   %A.decl: %A.type.938 = fn_decl @A.4 [concrete = constants.%A.890] {} {}
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .A = %A.decl
// CHECK:STDOUT:   witness = @D.%impl_witness.loc16
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @D {
// CHECK:STDOUT:   impl_decl @impl.909 [concrete] {} {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%D [concrete = constants.%D]
// CHECK:STDOUT:     %HasA1.ref: type = name_ref HasA1, file.%HasA1.decl [concrete = constants.%HasA1.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness.loc13: <witness> = impl_witness (@impl.909.%A.decl) [concrete = constants.%impl_witness.73a]
// CHECK:STDOUT:   impl_decl @impl.390 [concrete] {} {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%D [concrete = constants.%D]
// CHECK:STDOUT:     %HasA2.ref: type = name_ref HasA2, file.%HasA2.decl [concrete = constants.%HasA2.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness.loc16: <witness> = impl_witness (@impl.390.%A.decl) [concrete = constants.%impl_witness.baf]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%D
// CHECK:STDOUT:   .HasA1 = <poisoned>
// CHECK:STDOUT:   .HasA2 = <poisoned>
// CHECK:STDOUT:   .A = <poisoned>
// CHECK:STDOUT:   extend @impl.909.%HasA1.ref
// CHECK:STDOUT:   extend @impl.390.%HasA2.ref
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @A.1(@HasA1.%Self: %HasA1.type) {
// CHECK:STDOUT:   fn();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @A.2(@HasA2.%Self: %HasA2.type) {
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
// CHECK:STDOUT:   %D.ref.loc26: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:   %A.ref.loc26: <error> = name_ref A, <error> [concrete = <error>]
// CHECK:STDOUT:   %d.ref: %D = name_ref d, %d
// CHECK:STDOUT:   %A.ref.loc31: <error> = name_ref A, <error> [concrete = <error>]
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A.1(constants.%Self.90d) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A.2(constants.%Self.bdc) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A.1(constants.%HasA1.facet) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A.2(constants.%HasA2.facet) {}
// CHECK:STDOUT:
// CHECK:STDOUT: --- different_impl_and_base.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %HasI.type: type = facet_type <@HasI> [concrete]
// CHECK:STDOUT:   %Self: %HasI.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %I.type.9bd: type = fn_type @I.1 [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %I.0a6: %I.type.9bd = struct_value () [concrete]
// CHECK:STDOUT:   %HasI.assoc_type: type = assoc_entity_type %HasI.type [concrete]
// CHECK:STDOUT:   %assoc0: %HasI.assoc_type = assoc_entity element0, @HasI.%I.decl [concrete]
// CHECK:STDOUT:   %B: type = class_type @B [concrete]
// CHECK:STDOUT:   %J.type: type = fn_type @J [concrete]
// CHECK:STDOUT:   %J: %J.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %E: type = class_type @E [concrete]
// CHECK:STDOUT:   %E.elem: type = unbound_element_type %E, %B [concrete]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (@impl.%I.decl) [concrete]
// CHECK:STDOUT:   %I.type.395: type = fn_type @I.2 [concrete]
// CHECK:STDOUT:   %I.e74: %I.type.395 = struct_value () [concrete]
// CHECK:STDOUT:   %HasI.facet: %HasI.type = facet_value %E, (%impl_witness) [concrete]
// CHECK:STDOUT:   %struct_type.base.0ff: type = struct_type {.base: %B} [concrete]
// CHECK:STDOUT:   %complete_type.98e: <witness> = complete_type_witness %struct_type.base.0ff [concrete]
// CHECK:STDOUT:   %H.type: type = fn_type @H [concrete]
// CHECK:STDOUT:   %H: %H.type = struct_value () [concrete]
// CHECK:STDOUT:   %.52d: type = fn_type_with_self_type %I.type.9bd, %HasI.facet [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .HasI = %HasI.decl
// CHECK:STDOUT:     .B = %B.decl
// CHECK:STDOUT:     .E = %E.decl
// CHECK:STDOUT:     .H = %H.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %HasI.decl: type = interface_decl @HasI [concrete = constants.%HasI.type] {} {}
// CHECK:STDOUT:   %B.decl: type = class_decl @B [concrete = constants.%B] {} {}
// CHECK:STDOUT:   %E.decl: type = class_decl @E [concrete = constants.%E] {} {}
// CHECK:STDOUT:   %H.decl: %H.type = fn_decl @H [concrete = constants.%H] {
// CHECK:STDOUT:     %e.patt: %E = binding_pattern e
// CHECK:STDOUT:     %e.param_patt: %E = value_param_pattern %e.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %e.param: %E = value_param call_param0
// CHECK:STDOUT:     %E.ref.loc19: type = name_ref E, file.%E.decl [concrete = constants.%E]
// CHECK:STDOUT:     %e: %E = bind_name e, %e.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @HasI {
// CHECK:STDOUT:   %Self: %HasI.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   %I.decl: %I.type.9bd = fn_decl @I.1 [concrete = constants.%I.0a6] {} {}
// CHECK:STDOUT:   %assoc0: %HasI.assoc_type = assoc_entity element0, %I.decl [concrete = constants.%assoc0]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .I = %assoc0
// CHECK:STDOUT:   .J = <poisoned>
// CHECK:STDOUT:   witness = (%I.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl: %Self.ref as %HasI.ref {
// CHECK:STDOUT:   %I.decl: %I.type.395 = fn_decl @I.2 [concrete = constants.%I.e74] {} {}
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .I = %I.decl
// CHECK:STDOUT:   witness = @E.%impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @B {
// CHECK:STDOUT:   %J.decl: %J.type = fn_decl @J [concrete = constants.%J] {} {}
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%B
// CHECK:STDOUT:   .J = %J.decl
// CHECK:STDOUT:   .HasI = <poisoned>
// CHECK:STDOUT:   .I = <poisoned>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @E {
// CHECK:STDOUT:   %B.ref: type = name_ref B, file.%B.decl [concrete = constants.%B]
// CHECK:STDOUT:   %.loc13: %E.elem = base_decl %B.ref, element0 [concrete]
// CHECK:STDOUT:   impl_decl @impl [concrete] {} {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%E [concrete = constants.%E]
// CHECK:STDOUT:     %HasI.ref: type = name_ref HasI, file.%HasI.decl [concrete = constants.%HasI.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (@impl.%I.decl) [concrete = constants.%impl_witness]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.base.0ff [concrete = constants.%complete_type.98e]
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
// CHECK:STDOUT:   extend @impl.%HasI.ref
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @I.1(@HasI.%Self: %HasI.type) {
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
// CHECK:STDOUT:   %E.ref.loc20: type = name_ref E, file.%E.decl [concrete = constants.%E]
// CHECK:STDOUT:   %I.ref.loc20: %HasI.assoc_type = name_ref I, @HasI.%assoc0 [concrete = constants.%assoc0]
// CHECK:STDOUT:   %impl.elem0.loc20: %.52d = impl_witness_access constants.%impl_witness, element0 [concrete = constants.%I.e74]
// CHECK:STDOUT:   %I.call.loc20: init %empty_tuple.type = call %impl.elem0.loc20()
// CHECK:STDOUT:   %e.ref.loc21: %E = name_ref e, %e
// CHECK:STDOUT:   %I.ref.loc21: %HasI.assoc_type = name_ref I, @HasI.%assoc0 [concrete = constants.%assoc0]
// CHECK:STDOUT:   %impl.elem0.loc21: %.52d = impl_witness_access constants.%impl_witness, element0 [concrete = constants.%I.e74]
// CHECK:STDOUT:   %I.call.loc21: init %empty_tuple.type = call %impl.elem0.loc21()
// CHECK:STDOUT:   %E.ref.loc22: type = name_ref E, file.%E.decl [concrete = constants.%E]
// CHECK:STDOUT:   %J.ref.loc22: %J.type = name_ref J, @B.%J.decl [concrete = constants.%J]
// CHECK:STDOUT:   %J.call.loc22: init %empty_tuple.type = call %J.ref.loc22()
// CHECK:STDOUT:   %e.ref.loc23: %E = name_ref e, %e
// CHECK:STDOUT:   %J.ref.loc23: %J.type = name_ref J, @B.%J.decl [concrete = constants.%J]
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
// CHECK:STDOUT:   %Base: type = class_type @Base [concrete]
// CHECK:STDOUT:   %K.type.5a4: type = fn_type @K.1 [concrete]
// CHECK:STDOUT:   %K.801: %K.type.5a4 = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %HasK.type: type = facet_type <@HasK> [concrete]
// CHECK:STDOUT:   %Self: %HasK.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %K.type.76f: type = fn_type @K.2 [concrete]
// CHECK:STDOUT:   %K.c0e: %K.type.76f = struct_value () [concrete]
// CHECK:STDOUT:   %HasK.assoc_type: type = assoc_entity_type %HasK.type [concrete]
// CHECK:STDOUT:   %assoc0: %HasK.assoc_type = assoc_entity element0, @HasK.%K.decl [concrete]
// CHECK:STDOUT:   %L: type = class_type @L [concrete]
// CHECK:STDOUT:   %L.elem: type = unbound_element_type %L, %Base [concrete]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (@impl.%K.decl) [concrete]
// CHECK:STDOUT:   %K.type.bc2: type = fn_type @K.3 [concrete]
// CHECK:STDOUT:   %K.bd6: %K.type.bc2 = struct_value () [concrete]
// CHECK:STDOUT:   %HasK.facet: %HasK.type = facet_value %L, (%impl_witness) [concrete]
// CHECK:STDOUT:   %struct_type.base.b1e: type = struct_type {.base: %Base} [concrete]
// CHECK:STDOUT:   %complete_type.15c: <witness> = complete_type_witness %struct_type.base.b1e [concrete]
// CHECK:STDOUT:   %M.type: type = fn_type @M [concrete]
// CHECK:STDOUT:   %M: %M.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Base = %Base.decl
// CHECK:STDOUT:     .HasK = %HasK.decl
// CHECK:STDOUT:     .L = %L.decl
// CHECK:STDOUT:     .M = %M.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Base.decl: type = class_decl @Base [concrete = constants.%Base] {} {}
// CHECK:STDOUT:   %HasK.decl: type = interface_decl @HasK [concrete = constants.%HasK.type] {} {}
// CHECK:STDOUT:   %L.decl: type = class_decl @L [concrete = constants.%L] {} {}
// CHECK:STDOUT:   %M.decl: %M.type = fn_decl @M [concrete = constants.%M] {
// CHECK:STDOUT:     %l.patt: %L = binding_pattern l
// CHECK:STDOUT:     %l.param_patt: %L = value_param_pattern %l.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %l.param: %L = value_param call_param0
// CHECK:STDOUT:     %L.ref.loc19: type = name_ref L, file.%L.decl [concrete = constants.%L]
// CHECK:STDOUT:     %l: %L = bind_name l, %l.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @HasK {
// CHECK:STDOUT:   %Self: %HasK.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   %K.decl: %K.type.76f = fn_decl @K.2 [concrete = constants.%K.c0e] {} {}
// CHECK:STDOUT:   %assoc0: %HasK.assoc_type = assoc_entity element0, %K.decl [concrete = constants.%assoc0]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .K = %assoc0
// CHECK:STDOUT:   witness = (%K.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl: %Self.ref as %HasK.ref {
// CHECK:STDOUT:   %K.decl: %K.type.bc2 = fn_decl @K.3 [concrete = constants.%K.bd6] {} {}
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .K = %K.decl
// CHECK:STDOUT:   witness = @L.%impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Base {
// CHECK:STDOUT:   %K.decl: %K.type.5a4 = fn_decl @K.1 [concrete = constants.%K.801] {} {}
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Base
// CHECK:STDOUT:   .K = %K.decl
// CHECK:STDOUT:   .HasK = <poisoned>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @L {
// CHECK:STDOUT:   %Base.ref: type = name_ref Base, file.%Base.decl [concrete = constants.%Base]
// CHECK:STDOUT:   %.loc13: %L.elem = base_decl %Base.ref, element0 [concrete]
// CHECK:STDOUT:   impl_decl @impl [concrete] {} {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%L [concrete = constants.%L]
// CHECK:STDOUT:     %HasK.ref: type = name_ref HasK, file.%HasK.decl [concrete = constants.%HasK.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (@impl.%K.decl) [concrete = constants.%impl_witness]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.base.b1e [concrete = constants.%complete_type.15c]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%L
// CHECK:STDOUT:   .Base = <poisoned>
// CHECK:STDOUT:   .base = %.loc13
// CHECK:STDOUT:   .HasK = <poisoned>
// CHECK:STDOUT:   .K = <poisoned>
// CHECK:STDOUT:   extend %Base.ref
// CHECK:STDOUT:   extend @impl.%HasK.ref
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @K.1() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @K.2(@HasK.%Self: %HasK.type) {
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
// CHECK:STDOUT:   %L.ref.loc24: type = name_ref L, file.%L.decl [concrete = constants.%L]
// CHECK:STDOUT:   %K.ref.loc24: <error> = name_ref K, <error> [concrete = <error>]
// CHECK:STDOUT:   %l.ref: %L = name_ref l, %l
// CHECK:STDOUT:   %K.ref.loc29: <error> = name_ref K, <error> [concrete = <error>]
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
// CHECK:STDOUT:   %NBase: type = class_type @NBase [concrete]
// CHECK:STDOUT:   %N.type.fbf: type = fn_type @N.1 [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %N.897: %N.type.fbf = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %HasN1.type: type = facet_type <@HasN1> [concrete]
// CHECK:STDOUT:   %Self.e0b: %HasN1.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %N.type.3da: type = fn_type @N.2 [concrete]
// CHECK:STDOUT:   %N.a51: %N.type.3da = struct_value () [concrete]
// CHECK:STDOUT:   %HasN1.assoc_type: type = assoc_entity_type %HasN1.type [concrete]
// CHECK:STDOUT:   %assoc0.46d: %HasN1.assoc_type = assoc_entity element0, @HasN1.%N.decl [concrete]
// CHECK:STDOUT:   %HasN2.type: type = facet_type <@HasN2> [concrete]
// CHECK:STDOUT:   %Self.e6e: %HasN2.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %N.type.0aa: type = fn_type @N.3 [concrete]
// CHECK:STDOUT:   %N.1a4: %N.type.0aa = struct_value () [concrete]
// CHECK:STDOUT:   %HasN2.assoc_type: type = assoc_entity_type %HasN2.type [concrete]
// CHECK:STDOUT:   %assoc0.9ae: %HasN2.assoc_type = assoc_entity element0, @HasN2.%N.decl [concrete]
// CHECK:STDOUT:   %O: type = class_type @O [concrete]
// CHECK:STDOUT:   %O.elem: type = unbound_element_type %O, %NBase [concrete]
// CHECK:STDOUT:   %impl_witness.982: <witness> = impl_witness (@impl.968.%N.decl) [concrete]
// CHECK:STDOUT:   %N.type.ffc: type = fn_type @N.4 [concrete]
// CHECK:STDOUT:   %N.4f5: %N.type.ffc = struct_value () [concrete]
// CHECK:STDOUT:   %HasN1.facet: %HasN1.type = facet_value %O, (%impl_witness.982) [concrete]
// CHECK:STDOUT:   %impl_witness.599: <witness> = impl_witness (@impl.117.%N.decl) [concrete]
// CHECK:STDOUT:   %N.type.8af: type = fn_type @N.5 [concrete]
// CHECK:STDOUT:   %N.308: %N.type.8af = struct_value () [concrete]
// CHECK:STDOUT:   %HasN2.facet: %HasN2.type = facet_value %O, (%impl_witness.599) [concrete]
// CHECK:STDOUT:   %N.type.8b4: type = fn_type @N.6 [concrete]
// CHECK:STDOUT:   %N.cc3: %N.type.8b4 = struct_value () [concrete]
// CHECK:STDOUT:   %struct_type.base.9c6: type = struct_type {.base: %NBase} [concrete]
// CHECK:STDOUT:   %complete_type.121: <witness> = complete_type_witness %struct_type.base.9c6 [concrete]
// CHECK:STDOUT:   %P.type: type = fn_type @P [concrete]
// CHECK:STDOUT:   %P: %P.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .NBase = %NBase.decl
// CHECK:STDOUT:     .HasN1 = %HasN1.decl
// CHECK:STDOUT:     .HasN2 = %HasN2.decl
// CHECK:STDOUT:     .O = %O.decl
// CHECK:STDOUT:     .P = %P.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %NBase.decl: type = class_decl @NBase [concrete = constants.%NBase] {} {}
// CHECK:STDOUT:   %HasN1.decl: type = interface_decl @HasN1 [concrete = constants.%HasN1.type] {} {}
// CHECK:STDOUT:   %HasN2.decl: type = interface_decl @HasN2 [concrete = constants.%HasN2.type] {} {}
// CHECK:STDOUT:   %O.decl: type = class_decl @O [concrete = constants.%O] {} {}
// CHECK:STDOUT:   %P.decl: %P.type = fn_decl @P [concrete = constants.%P] {
// CHECK:STDOUT:     %o.patt: %O = binding_pattern o
// CHECK:STDOUT:     %o.param_patt: %O = value_param_pattern %o.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %o.param: %O = value_param call_param0
// CHECK:STDOUT:     %O.ref.loc27: type = name_ref O, file.%O.decl [concrete = constants.%O]
// CHECK:STDOUT:     %o: %O = bind_name o, %o.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @HasN1 {
// CHECK:STDOUT:   %Self: %HasN1.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.e0b]
// CHECK:STDOUT:   %N.decl: %N.type.3da = fn_decl @N.2 [concrete = constants.%N.a51] {} {}
// CHECK:STDOUT:   %assoc0: %HasN1.assoc_type = assoc_entity element0, %N.decl [concrete = constants.%assoc0.46d]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .N = %assoc0
// CHECK:STDOUT:   .HasN2 = <poisoned>
// CHECK:STDOUT:   witness = (%N.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @HasN2 {
// CHECK:STDOUT:   %Self: %HasN2.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.e6e]
// CHECK:STDOUT:   %N.decl: %N.type.0aa = fn_decl @N.3 [concrete = constants.%N.1a4] {} {}
// CHECK:STDOUT:   %assoc0: %HasN2.assoc_type = assoc_entity element0, %N.decl [concrete = constants.%assoc0.9ae]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .N = %assoc0
// CHECK:STDOUT:   witness = (%N.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.968: %Self.ref as %HasN1.ref {
// CHECK:STDOUT:   %N.decl: %N.type.ffc = fn_decl @N.4 [concrete = constants.%N.4f5] {} {}
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .N = %N.decl
// CHECK:STDOUT:   witness = @O.%impl_witness.loc18
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.117: %Self.ref as %HasN2.ref {
// CHECK:STDOUT:   %N.decl: %N.type.8af = fn_decl @N.5 [concrete = constants.%N.308] {} {}
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .N = %N.decl
// CHECK:STDOUT:   witness = @O.%impl_witness.loc21
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @NBase {
// CHECK:STDOUT:   %N.decl: %N.type.fbf = fn_decl @N.1 [concrete = constants.%N.897] {} {}
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%NBase
// CHECK:STDOUT:   .N = %N.decl
// CHECK:STDOUT:   .HasN1 = <poisoned>
// CHECK:STDOUT:   .HasN2 = <poisoned>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @O {
// CHECK:STDOUT:   %NBase.ref: type = name_ref NBase, file.%NBase.decl [concrete = constants.%NBase]
// CHECK:STDOUT:   %.loc17: %O.elem = base_decl %NBase.ref, element0 [concrete]
// CHECK:STDOUT:   impl_decl @impl.968 [concrete] {} {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%O [concrete = constants.%O]
// CHECK:STDOUT:     %HasN1.ref: type = name_ref HasN1, file.%HasN1.decl [concrete = constants.%HasN1.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness.loc18: <witness> = impl_witness (@impl.968.%N.decl) [concrete = constants.%impl_witness.982]
// CHECK:STDOUT:   impl_decl @impl.117 [concrete] {} {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%O [concrete = constants.%O]
// CHECK:STDOUT:     %HasN2.ref: type = name_ref HasN2, file.%HasN2.decl [concrete = constants.%HasN2.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness.loc21: <witness> = impl_witness (@impl.117.%N.decl) [concrete = constants.%impl_witness.599]
// CHECK:STDOUT:   %N.decl: %N.type.8b4 = fn_decl @N.6 [concrete = constants.%N.cc3] {} {}
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.base.9c6 [concrete = constants.%complete_type.121]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%O
// CHECK:STDOUT:   .NBase = <poisoned>
// CHECK:STDOUT:   .base = %.loc17
// CHECK:STDOUT:   .HasN1 = <poisoned>
// CHECK:STDOUT:   .HasN2 = <poisoned>
// CHECK:STDOUT:   .N = %N.decl
// CHECK:STDOUT:   extend %NBase.ref
// CHECK:STDOUT:   extend @impl.968.%HasN1.ref
// CHECK:STDOUT:   extend @impl.117.%HasN2.ref
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @N.1() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @N.2(@HasN1.%Self: %HasN1.type) {
// CHECK:STDOUT:   fn();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @N.3(@HasN2.%Self: %HasN2.type) {
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
// CHECK:STDOUT:   %O.ref.loc28: type = name_ref O, file.%O.decl [concrete = constants.%O]
// CHECK:STDOUT:   %N.ref.loc28: %N.type.8b4 = name_ref N, @O.%N.decl [concrete = constants.%N.cc3]
// CHECK:STDOUT:   %N.call.loc28: init %empty_tuple.type = call %N.ref.loc28()
// CHECK:STDOUT:   %o.ref: %O = name_ref o, %o
// CHECK:STDOUT:   %N.ref.loc29: %N.type.8b4 = name_ref N, @O.%N.decl [concrete = constants.%N.cc3]
// CHECK:STDOUT:   %N.call.loc29: init %empty_tuple.type = call %N.ref.loc29()
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @N.2(constants.%Self.e0b) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @N.3(constants.%Self.e6e) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @N.2(constants.%HasN1.facet) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @N.3(constants.%HasN2.facet) {}
// CHECK:STDOUT:
