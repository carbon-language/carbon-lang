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
// CHECK:STDOUT:   %Self.1: %HasF.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %F.type.1: type = fn_type @F.1 [template]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [template]
// CHECK:STDOUT:   %F.1: %F.type.1 = struct_value () [template]
// CHECK:STDOUT:   %.1: type = assoc_entity_type %HasF.type, %F.type.1 [template]
// CHECK:STDOUT:   %.2: %.1 = assoc_entity element0, @HasF.%F.decl [template]
// CHECK:STDOUT:   %HasG.type: type = facet_type <@HasG> [template]
// CHECK:STDOUT:   %Self.2: %HasG.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %G.type.1: type = fn_type @G.1 [template]
// CHECK:STDOUT:   %G.1: %G.type.1 = struct_value () [template]
// CHECK:STDOUT:   %.3: type = assoc_entity_type %HasG.type, %G.type.1 [template]
// CHECK:STDOUT:   %.4: %.3 = assoc_entity element0, @HasG.%G.decl [template]
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %F.type.2: type = fn_type @F.2 [template]
// CHECK:STDOUT:   %F.2: %F.type.2 = struct_value () [template]
// CHECK:STDOUT:   %.5: %HasF.type = facet_value %C, %C [template]
// CHECK:STDOUT:   %.6: <witness> = interface_witness (%F.2) [template]
// CHECK:STDOUT:   %G.type.2: type = fn_type @G.2 [template]
// CHECK:STDOUT:   %G.2: %G.type.2 = struct_value () [template]
// CHECK:STDOUT:   %.7: %HasG.type = facet_value %C, %C [template]
// CHECK:STDOUT:   %.8: <witness> = interface_witness (%G.2) [template]
// CHECK:STDOUT:   %.9: type = struct_type {} [template]
// CHECK:STDOUT:   %.10: <witness> = complete_type_witness %.9 [template]
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
// CHECK:STDOUT:     %C.ref.loc21: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %c.param: %C = value_param runtime_param0
// CHECK:STDOUT:     %c: %C = bind_name c, %c.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @HasF {
// CHECK:STDOUT:   %Self: %HasF.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.1]
// CHECK:STDOUT:   %F.decl: %F.type.1 = fn_decl @F.1 [template = constants.%F.1] {} {}
// CHECK:STDOUT:   %.loc5: %.1 = assoc_entity element0, %F.decl [template = constants.%.2]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .F = %.loc5
// CHECK:STDOUT:   witness = (%F.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @HasG {
// CHECK:STDOUT:   %Self: %HasG.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.2]
// CHECK:STDOUT:   %G.decl: %G.type.1 = fn_decl @G.1 [template = constants.%G.1] {} {}
// CHECK:STDOUT:   %.loc9: %.3 = assoc_entity element0, %G.decl [template = constants.%.4]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .G = %.loc9
// CHECK:STDOUT:   witness = (%G.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.1: %Self.ref as %HasF.ref {
// CHECK:STDOUT:   %F.decl: %F.type.2 = fn_decl @F.2 [template = constants.%F.2] {} {}
// CHECK:STDOUT:   %.loc13: <witness> = interface_witness (%F.decl) [template = constants.%.6]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   witness = %.loc13
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.2: %Self.ref as %HasG.ref {
// CHECK:STDOUT:   %G.decl: %G.type.2 = fn_decl @G.2 [template = constants.%G.2] {} {}
// CHECK:STDOUT:   %.loc16: <witness> = interface_witness (%G.decl) [template = constants.%.8]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .G = %G.decl
// CHECK:STDOUT:   witness = %.loc16
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   impl_decl @impl.1 [template] {} {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%C [template = constants.%C]
// CHECK:STDOUT:     %HasF.ref: type = name_ref HasF, file.%HasF.decl [template = constants.%HasF.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @impl.2 [template] {} {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%C [template = constants.%C]
// CHECK:STDOUT:     %HasG.ref: type = name_ref HasG, file.%HasG.decl [template = constants.%HasG.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc19: <witness> = complete_type_witness %.9 [template = constants.%.10]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT:   extend @impl.1.%HasF.ref
// CHECK:STDOUT:   extend @impl.2.%HasG.ref
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
// CHECK:STDOUT:   %F.ref.loc22: %.1 = name_ref F, @HasF.%.loc5 [template = constants.%.2]
// CHECK:STDOUT:   %.loc22: %F.type.1 = interface_witness_access constants.%.6, element0 [template = constants.%F.2]
// CHECK:STDOUT:   %F.call.loc22: init %empty_tuple.type = call %.loc22()
// CHECK:STDOUT:   %c.ref.loc23: %C = name_ref c, %c
// CHECK:STDOUT:   %F.ref.loc23: %.1 = name_ref F, @HasF.%.loc5 [template = constants.%.2]
// CHECK:STDOUT:   %.loc23: %F.type.1 = interface_witness_access constants.%.6, element0 [template = constants.%F.2]
// CHECK:STDOUT:   %F.call.loc23: init %empty_tuple.type = call %.loc23()
// CHECK:STDOUT:   %C.ref.loc24: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:   %G.ref.loc24: %.3 = name_ref G, @HasG.%.loc9 [template = constants.%.4]
// CHECK:STDOUT:   %.loc24: %G.type.1 = interface_witness_access constants.%.8, element0 [template = constants.%G.2]
// CHECK:STDOUT:   %G.call.loc24: init %empty_tuple.type = call %.loc24()
// CHECK:STDOUT:   %c.ref.loc25: %C = name_ref c, %c
// CHECK:STDOUT:   %G.ref.loc25: %.3 = name_ref G, @HasG.%.loc9 [template = constants.%.4]
// CHECK:STDOUT:   %.loc25: %G.type.1 = interface_witness_access constants.%.8, element0 [template = constants.%G.2]
// CHECK:STDOUT:   %G.call.loc25: init %empty_tuple.type = call %.loc25()
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(constants.%Self.1) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @G.1(constants.%Self.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(constants.%.5) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @G.1(constants.%.7) {}
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_ambiguous_impls.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %HasA1.type: type = facet_type <@HasA1> [template]
// CHECK:STDOUT:   %Self.1: %HasA1.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %A.type.1: type = fn_type @A.1 [template]
// CHECK:STDOUT:   %A.1: %A.type.1 = struct_value () [template]
// CHECK:STDOUT:   %.1: type = assoc_entity_type %HasA1.type, %A.type.1 [template]
// CHECK:STDOUT:   %.2: %.1 = assoc_entity element0, @HasA1.%A.decl [template]
// CHECK:STDOUT:   %HasA2.type: type = facet_type <@HasA2> [template]
// CHECK:STDOUT:   %Self.2: %HasA2.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %A.type.2: type = fn_type @A.2 [template]
// CHECK:STDOUT:   %A.2: %A.type.2 = struct_value () [template]
// CHECK:STDOUT:   %.3: type = assoc_entity_type %HasA2.type, %A.type.2 [template]
// CHECK:STDOUT:   %.4: %.3 = assoc_entity element0, @HasA2.%A.decl [template]
// CHECK:STDOUT:   %D: type = class_type @D [template]
// CHECK:STDOUT:   %A.type.3: type = fn_type @A.3 [template]
// CHECK:STDOUT:   %A.3: %A.type.3 = struct_value () [template]
// CHECK:STDOUT:   %.5: %HasA1.type = facet_value %D, %D [template]
// CHECK:STDOUT:   %.6: <witness> = interface_witness (%A.3) [template]
// CHECK:STDOUT:   %A.type.4: type = fn_type @A.4 [template]
// CHECK:STDOUT:   %A.4: %A.type.4 = struct_value () [template]
// CHECK:STDOUT:   %.7: %HasA2.type = facet_value %D, %D [template]
// CHECK:STDOUT:   %.8: <witness> = interface_witness (%A.4) [template]
// CHECK:STDOUT:   %.9: type = struct_type {} [template]
// CHECK:STDOUT:   %.10: <witness> = complete_type_witness %.9 [template]
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
// CHECK:STDOUT:     %D.ref.loc21: type = name_ref D, file.%D.decl [template = constants.%D]
// CHECK:STDOUT:     %d.param: %D = value_param runtime_param0
// CHECK:STDOUT:     %d: %D = bind_name d, %d.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @HasA1 {
// CHECK:STDOUT:   %Self: %HasA1.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.1]
// CHECK:STDOUT:   %A.decl: %A.type.1 = fn_decl @A.1 [template = constants.%A.1] {} {}
// CHECK:STDOUT:   %.loc5: %.1 = assoc_entity element0, %A.decl [template = constants.%.2]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .A = %.loc5
// CHECK:STDOUT:   witness = (%A.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @HasA2 {
// CHECK:STDOUT:   %Self: %HasA2.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.2]
// CHECK:STDOUT:   %A.decl: %A.type.2 = fn_decl @A.2 [template = constants.%A.2] {} {}
// CHECK:STDOUT:   %.loc9: %.3 = assoc_entity element0, %A.decl [template = constants.%.4]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .A = %.loc9
// CHECK:STDOUT:   witness = (%A.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.1: %Self.ref as %HasA1.ref {
// CHECK:STDOUT:   %A.decl: %A.type.3 = fn_decl @A.3 [template = constants.%A.3] {} {}
// CHECK:STDOUT:   %.loc13: <witness> = interface_witness (%A.decl) [template = constants.%.6]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .A = %A.decl
// CHECK:STDOUT:   witness = %.loc13
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.2: %Self.ref as %HasA2.ref {
// CHECK:STDOUT:   %A.decl: %A.type.4 = fn_decl @A.4 [template = constants.%A.4] {} {}
// CHECK:STDOUT:   %.loc16: <witness> = interface_witness (%A.decl) [template = constants.%.8]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .A = %A.decl
// CHECK:STDOUT:   witness = %.loc16
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @D {
// CHECK:STDOUT:   impl_decl @impl.1 [template] {} {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%D [template = constants.%D]
// CHECK:STDOUT:     %HasA1.ref: type = name_ref HasA1, file.%HasA1.decl [template = constants.%HasA1.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @impl.2 [template] {} {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%D [template = constants.%D]
// CHECK:STDOUT:     %HasA2.ref: type = name_ref HasA2, file.%HasA2.decl [template = constants.%HasA2.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc19: <witness> = complete_type_witness %.9 [template = constants.%.10]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%D
// CHECK:STDOUT:   extend @impl.1.%HasA1.ref
// CHECK:STDOUT:   extend @impl.2.%HasA2.ref
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
// CHECK:STDOUT: specific @A.1(constants.%Self.1) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A.2(constants.%Self.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A.1(constants.%.5) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A.2(constants.%.7) {}
// CHECK:STDOUT:
// CHECK:STDOUT: --- different_impl_and_base.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %HasI.type: type = facet_type <@HasI> [template]
// CHECK:STDOUT:   %Self: %HasI.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %I.type.1: type = fn_type @I.1 [template]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [template]
// CHECK:STDOUT:   %I.1: %I.type.1 = struct_value () [template]
// CHECK:STDOUT:   %.1: type = assoc_entity_type %HasI.type, %I.type.1 [template]
// CHECK:STDOUT:   %.2: %.1 = assoc_entity element0, @HasI.%I.decl [template]
// CHECK:STDOUT:   %B: type = class_type @B [template]
// CHECK:STDOUT:   %J.type: type = fn_type @J [template]
// CHECK:STDOUT:   %J: %J.type = struct_value () [template]
// CHECK:STDOUT:   %.3: type = struct_type {} [template]
// CHECK:STDOUT:   %.4: <witness> = complete_type_witness %.3 [template]
// CHECK:STDOUT:   %E: type = class_type @E [template]
// CHECK:STDOUT:   %.6: type = unbound_element_type %E, %B [template]
// CHECK:STDOUT:   %I.type.2: type = fn_type @I.2 [template]
// CHECK:STDOUT:   %I.2: %I.type.2 = struct_value () [template]
// CHECK:STDOUT:   %.7: %HasI.type = facet_value %E, %E [template]
// CHECK:STDOUT:   %.8: <witness> = interface_witness (%I.2) [template]
// CHECK:STDOUT:   %.9: type = struct_type {.base: %B} [template]
// CHECK:STDOUT:   %.10: <witness> = complete_type_witness %.9 [template]
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
// CHECK:STDOUT:     %E.ref.loc19: type = name_ref E, file.%E.decl [template = constants.%E]
// CHECK:STDOUT:     %e.param: %E = value_param runtime_param0
// CHECK:STDOUT:     %e: %E = bind_name e, %e.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @HasI {
// CHECK:STDOUT:   %Self: %HasI.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   %I.decl: %I.type.1 = fn_decl @I.1 [template = constants.%I.1] {} {}
// CHECK:STDOUT:   %.loc5: %.1 = assoc_entity element0, %I.decl [template = constants.%.2]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .I = %.loc5
// CHECK:STDOUT:   witness = (%I.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl: %Self.ref as %HasI.ref {
// CHECK:STDOUT:   %I.decl: %I.type.2 = fn_decl @I.2 [template = constants.%I.2] {} {}
// CHECK:STDOUT:   %.loc14: <witness> = interface_witness (%I.decl) [template = constants.%.8]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .I = %I.decl
// CHECK:STDOUT:   witness = %.loc14
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @B {
// CHECK:STDOUT:   %J.decl: %J.type = fn_decl @J [template = constants.%J] {} {}
// CHECK:STDOUT:   %.loc10: <witness> = complete_type_witness %.3 [template = constants.%.4]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%B
// CHECK:STDOUT:   .J = %J.decl
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @E {
// CHECK:STDOUT:   %B.ref: type = name_ref B, file.%B.decl [template = constants.%B]
// CHECK:STDOUT:   %.loc13: %.6 = base_decl %B, element0 [template]
// CHECK:STDOUT:   impl_decl @impl [template] {} {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%E [template = constants.%E]
// CHECK:STDOUT:     %HasI.ref: type = name_ref HasI, file.%HasI.decl [template = constants.%HasI.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc17: <witness> = complete_type_witness %.9 [template = constants.%.10]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%E
// CHECK:STDOUT:   .base = %.loc13
// CHECK:STDOUT:   extend %B.ref
// CHECK:STDOUT:   extend @impl.%HasI.ref
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
// CHECK:STDOUT:   %I.ref.loc20: %.1 = name_ref I, @HasI.%.loc5 [template = constants.%.2]
// CHECK:STDOUT:   %.loc20: %I.type.1 = interface_witness_access constants.%.8, element0 [template = constants.%I.2]
// CHECK:STDOUT:   %I.call.loc20: init %empty_tuple.type = call %.loc20()
// CHECK:STDOUT:   %e.ref.loc21: %E = name_ref e, %e
// CHECK:STDOUT:   %I.ref.loc21: %.1 = name_ref I, @HasI.%.loc5 [template = constants.%.2]
// CHECK:STDOUT:   %.loc21: %I.type.1 = interface_witness_access constants.%.8, element0 [template = constants.%I.2]
// CHECK:STDOUT:   %I.call.loc21: init %empty_tuple.type = call %.loc21()
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
// CHECK:STDOUT: specific @I.1(constants.%.7) {}
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_ambiguous_impl_and_base.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Base: type = class_type @Base [template]
// CHECK:STDOUT:   %K.type.1: type = fn_type @K.1 [template]
// CHECK:STDOUT:   %K.1: %K.type.1 = struct_value () [template]
// CHECK:STDOUT:   %.1: type = struct_type {} [template]
// CHECK:STDOUT:   %.2: <witness> = complete_type_witness %.1 [template]
// CHECK:STDOUT:   %HasK.type: type = facet_type <@HasK> [template]
// CHECK:STDOUT:   %Self: %HasK.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %K.type.2: type = fn_type @K.2 [template]
// CHECK:STDOUT:   %K.2: %K.type.2 = struct_value () [template]
// CHECK:STDOUT:   %.3: type = assoc_entity_type %HasK.type, %K.type.2 [template]
// CHECK:STDOUT:   %.4: %.3 = assoc_entity element0, @HasK.%K.decl [template]
// CHECK:STDOUT:   %L: type = class_type @L [template]
// CHECK:STDOUT:   %.6: type = unbound_element_type %L, %Base [template]
// CHECK:STDOUT:   %K.type.3: type = fn_type @K.3 [template]
// CHECK:STDOUT:   %K.3: %K.type.3 = struct_value () [template]
// CHECK:STDOUT:   %.7: %HasK.type = facet_value %L, %L [template]
// CHECK:STDOUT:   %.8: <witness> = interface_witness (%K.3) [template]
// CHECK:STDOUT:   %.9: type = struct_type {.base: %Base} [template]
// CHECK:STDOUT:   %.10: <witness> = complete_type_witness %.9 [template]
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
// CHECK:STDOUT:     %L.ref.loc19: type = name_ref L, file.%L.decl [template = constants.%L]
// CHECK:STDOUT:     %l.param: %L = value_param runtime_param0
// CHECK:STDOUT:     %l: %L = bind_name l, %l.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @HasK {
// CHECK:STDOUT:   %Self: %HasK.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   %K.decl: %K.type.2 = fn_decl @K.2 [template = constants.%K.2] {} {}
// CHECK:STDOUT:   %.loc9: %.3 = assoc_entity element0, %K.decl [template = constants.%.4]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .K = %.loc9
// CHECK:STDOUT:   witness = (%K.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl: %Self.ref as %HasK.ref {
// CHECK:STDOUT:   %K.decl: %K.type.3 = fn_decl @K.3 [template = constants.%K.3] {} {}
// CHECK:STDOUT:   %.loc14: <witness> = interface_witness (%K.decl) [template = constants.%.8]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .K = %K.decl
// CHECK:STDOUT:   witness = %.loc14
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Base {
// CHECK:STDOUT:   %K.decl: %K.type.1 = fn_decl @K.1 [template = constants.%K.1] {} {}
// CHECK:STDOUT:   %.loc6: <witness> = complete_type_witness %.1 [template = constants.%.2]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Base
// CHECK:STDOUT:   .K = %K.decl
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @L {
// CHECK:STDOUT:   %Base.ref: type = name_ref Base, file.%Base.decl [template = constants.%Base]
// CHECK:STDOUT:   %.loc13: %.6 = base_decl %Base, element0 [template]
// CHECK:STDOUT:   impl_decl @impl [template] {} {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%L [template = constants.%L]
// CHECK:STDOUT:     %HasK.ref: type = name_ref HasK, file.%HasK.decl [template = constants.%HasK.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc17: <witness> = complete_type_witness %.9 [template = constants.%.10]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%L
// CHECK:STDOUT:   .base = %.loc13
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
// CHECK:STDOUT: specific @K.2(constants.%.7) {}
// CHECK:STDOUT:
// CHECK:STDOUT: --- ambiguity_hidden.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %NBase: type = class_type @NBase [template]
// CHECK:STDOUT:   %N.type.1: type = fn_type @N.1 [template]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [template]
// CHECK:STDOUT:   %N.1: %N.type.1 = struct_value () [template]
// CHECK:STDOUT:   %.1: type = struct_type {} [template]
// CHECK:STDOUT:   %.2: <witness> = complete_type_witness %.1 [template]
// CHECK:STDOUT:   %HasN1.type: type = facet_type <@HasN1> [template]
// CHECK:STDOUT:   %Self.1: %HasN1.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %N.type.2: type = fn_type @N.2 [template]
// CHECK:STDOUT:   %N.2: %N.type.2 = struct_value () [template]
// CHECK:STDOUT:   %.3: type = assoc_entity_type %HasN1.type, %N.type.2 [template]
// CHECK:STDOUT:   %.4: %.3 = assoc_entity element0, @HasN1.%N.decl [template]
// CHECK:STDOUT:   %HasN2.type: type = facet_type <@HasN2> [template]
// CHECK:STDOUT:   %Self.2: %HasN2.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %N.type.3: type = fn_type @N.3 [template]
// CHECK:STDOUT:   %N.3: %N.type.3 = struct_value () [template]
// CHECK:STDOUT:   %.5: type = assoc_entity_type %HasN2.type, %N.type.3 [template]
// CHECK:STDOUT:   %.6: %.5 = assoc_entity element0, @HasN2.%N.decl [template]
// CHECK:STDOUT:   %O: type = class_type @O [template]
// CHECK:STDOUT:   %.8: type = unbound_element_type %O, %NBase [template]
// CHECK:STDOUT:   %N.type.4: type = fn_type @N.4 [template]
// CHECK:STDOUT:   %N.4: %N.type.4 = struct_value () [template]
// CHECK:STDOUT:   %.9: %HasN1.type = facet_value %O, %O [template]
// CHECK:STDOUT:   %.10: <witness> = interface_witness (%N.4) [template]
// CHECK:STDOUT:   %N.type.5: type = fn_type @N.5 [template]
// CHECK:STDOUT:   %N.5: %N.type.5 = struct_value () [template]
// CHECK:STDOUT:   %.11: %HasN2.type = facet_value %O, %O [template]
// CHECK:STDOUT:   %.12: <witness> = interface_witness (%N.5) [template]
// CHECK:STDOUT:   %N.type.6: type = fn_type @N.6 [template]
// CHECK:STDOUT:   %N.6: %N.type.6 = struct_value () [template]
// CHECK:STDOUT:   %.13: type = struct_type {.base: %NBase} [template]
// CHECK:STDOUT:   %.14: <witness> = complete_type_witness %.13 [template]
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
// CHECK:STDOUT:     %O.ref.loc27: type = name_ref O, file.%O.decl [template = constants.%O]
// CHECK:STDOUT:     %o.param: %O = value_param runtime_param0
// CHECK:STDOUT:     %o: %O = bind_name o, %o.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @HasN1 {
// CHECK:STDOUT:   %Self: %HasN1.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.1]
// CHECK:STDOUT:   %N.decl: %N.type.2 = fn_decl @N.2 [template = constants.%N.2] {} {}
// CHECK:STDOUT:   %.loc9: %.3 = assoc_entity element0, %N.decl [template = constants.%.4]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .N = %.loc9
// CHECK:STDOUT:   witness = (%N.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @HasN2 {
// CHECK:STDOUT:   %Self: %HasN2.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.2]
// CHECK:STDOUT:   %N.decl: %N.type.3 = fn_decl @N.3 [template = constants.%N.3] {} {}
// CHECK:STDOUT:   %.loc13: %.5 = assoc_entity element0, %N.decl [template = constants.%.6]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .N = %.loc13
// CHECK:STDOUT:   witness = (%N.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.1: %Self.ref as %HasN1.ref {
// CHECK:STDOUT:   %N.decl: %N.type.4 = fn_decl @N.4 [template = constants.%N.4] {} {}
// CHECK:STDOUT:   %.loc18: <witness> = interface_witness (%N.decl) [template = constants.%.10]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .N = %N.decl
// CHECK:STDOUT:   witness = %.loc18
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.2: %Self.ref as %HasN2.ref {
// CHECK:STDOUT:   %N.decl: %N.type.5 = fn_decl @N.5 [template = constants.%N.5] {} {}
// CHECK:STDOUT:   %.loc21: <witness> = interface_witness (%N.decl) [template = constants.%.12]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .N = %N.decl
// CHECK:STDOUT:   witness = %.loc21
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @NBase {
// CHECK:STDOUT:   %N.decl: %N.type.1 = fn_decl @N.1 [template = constants.%N.1] {} {}
// CHECK:STDOUT:   %.loc6: <witness> = complete_type_witness %.1 [template = constants.%.2]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%NBase
// CHECK:STDOUT:   .N = %N.decl
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @O {
// CHECK:STDOUT:   %NBase.ref: type = name_ref NBase, file.%NBase.decl [template = constants.%NBase]
// CHECK:STDOUT:   %.loc17: %.8 = base_decl %NBase, element0 [template]
// CHECK:STDOUT:   impl_decl @impl.1 [template] {} {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%O [template = constants.%O]
// CHECK:STDOUT:     %HasN1.ref: type = name_ref HasN1, file.%HasN1.decl [template = constants.%HasN1.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @impl.2 [template] {} {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%O [template = constants.%O]
// CHECK:STDOUT:     %HasN2.ref: type = name_ref HasN2, file.%HasN2.decl [template = constants.%HasN2.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %N.decl: %N.type.6 = fn_decl @N.6 [template = constants.%N.6] {} {}
// CHECK:STDOUT:   %.loc25: <witness> = complete_type_witness %.13 [template = constants.%.14]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%O
// CHECK:STDOUT:   .base = %.loc17
// CHECK:STDOUT:   .N = %N.decl
// CHECK:STDOUT:   extend %NBase.ref
// CHECK:STDOUT:   extend @impl.1.%HasN1.ref
// CHECK:STDOUT:   extend @impl.2.%HasN2.ref
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
// CHECK:STDOUT:   %N.ref.loc28: %N.type.6 = name_ref N, @O.%N.decl [template = constants.%N.6]
// CHECK:STDOUT:   %N.call.loc28: init %empty_tuple.type = call %N.ref.loc28()
// CHECK:STDOUT:   %o.ref: %O = name_ref o, %o
// CHECK:STDOUT:   %N.ref.loc29: %N.type.6 = name_ref N, @O.%N.decl [template = constants.%N.6]
// CHECK:STDOUT:   %N.call.loc29: init %empty_tuple.type = call %N.ref.loc29()
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @N.2(constants.%Self.1) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @N.3(constants.%Self.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @N.2(constants.%.9) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @N.3(constants.%.11) {}
// CHECK:STDOUT:
