// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/impl/no_prelude/impl_self_as.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/impl/no_prelude/impl_self_as.carbon

// FIXME: Should this --no-dump-sem-ir?

// --- match.carbon
library "[[@TEST_NAME]]";

interface I1 {}
interface I2 {}
interface J1(T1:! type) {}
interface J2(T2:! type) {}

// `impl Self as` should match `impl as`.

class C1 {
  impl Self as I1;
  impl as I1 {}

  impl as I2;
  impl Self as I2 {}

  impl forall [U:! type] Self as J1(U);
  impl forall [U:! type] as J1(U) {}

  impl forall [V:! type] as J2(V);
  impl forall [V:! type] Self as J2(V) {}
}

class C2(W:! type) {
  impl Self as I1;
  impl as I1 {}

  impl as I2;
  impl Self as I2 {}

  impl forall [X:! type] Self as J1(X);
  impl forall [X:! type] as J1(X) {}

  impl forall [Y:! type] as J2(Y);
  impl forall [Y:! type] Self as J2(Y) {}
}


// --- fail_no_match.carbon
library "[[@TEST_NAME]]";

interface I3 {}
interface I4 {}
interface I5 {}
interface I6 {}
interface J3(T3:! type) {}
interface J4(T4:! type) {}
interface J5(T5:! type) {}
interface J6(T6:! type) {}

// `impl C as` should not match `impl Self as` or `impl as`.

class C3 {
  // CHECK:STDERR: fail_no_match.carbon:[[@LINE+4]]:3: error: impl declared but not defined [MissingImplDefinition]
  // CHECK:STDERR:   impl C3 as I3;
  // CHECK:STDERR:   ^~~~~~~~~~~~~~
  // CHECK:STDERR:
  impl C3 as I3;
  impl as I3 {}

  // CHECK:STDERR: fail_no_match.carbon:[[@LINE+4]]:3: error: impl declared but not defined [MissingImplDefinition]
  // CHECK:STDERR:   impl C3 as I4;
  // CHECK:STDERR:   ^~~~~~~~~~~~~~
  // CHECK:STDERR:
  impl C3 as I4;
  impl Self as I4 {}

  // CHECK:STDERR: fail_no_match.carbon:[[@LINE+4]]:3: error: impl declared but not defined [MissingImplDefinition]
  // CHECK:STDERR:   impl as I5;
  // CHECK:STDERR:   ^~~~~~~~~~~
  // CHECK:STDERR:
  impl as I5;
  impl C3 as I5 {}

  // CHECK:STDERR: fail_no_match.carbon:[[@LINE+4]]:3: error: impl declared but not defined [MissingImplDefinition]
  // CHECK:STDERR:   impl Self as I6;
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  impl Self as I6;
  impl C3 as I6 {}

  // CHECK:STDERR: fail_no_match.carbon:[[@LINE+4]]:3: error: impl declared but not defined [MissingImplDefinition]
  // CHECK:STDERR:   impl forall [Z3:! type] C3 as J3(Z3);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  impl forall [Z3:! type] C3 as J3(Z3);
  impl forall [Z3:! type] as J3(Z3) {}

  // CHECK:STDERR: fail_no_match.carbon:[[@LINE+4]]:3: error: impl declared but not defined [MissingImplDefinition]
  // CHECK:STDERR:   impl forall [Z4:! type] C3 as J4(Z4);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  impl forall [Z4:! type] C3 as J4(Z4);
  impl forall [Z4:! type] Self as J4(Z4) {}

  // CHECK:STDERR: fail_no_match.carbon:[[@LINE+4]]:3: error: impl declared but not defined [MissingImplDefinition]
  // CHECK:STDERR:   impl forall [Z5:! type] as J5(Z5);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  impl forall [Z5:! type] as J5(Z5);
  impl forall [Z5:! type] C3 as J5(Z5) {}

  // CHECK:STDERR: fail_no_match.carbon:[[@LINE+4]]:3: error: impl declared but not defined [MissingImplDefinition]
  // CHECK:STDERR:   impl forall [Z6:! type] Self as J6(Z6);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  impl forall [Z6:! type] Self as J6(Z6);
  impl forall [Z6:! type] C3 as J6(Z6) {}
}

class C4(A:! type) {
  // CHECK:STDERR: fail_no_match.carbon:[[@LINE+4]]:3: error: impl declared but not defined [MissingImplDefinition]
  // CHECK:STDERR:   impl C4(A) as I3;
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  impl C4(A) as I3;
  impl as I3 {}

  // CHECK:STDERR: fail_no_match.carbon:[[@LINE+4]]:3: error: impl declared but not defined [MissingImplDefinition]
  // CHECK:STDERR:   impl C4(A) as I4;
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  impl C4(A) as I4;
  impl Self as I4 {}

  // CHECK:STDERR: fail_no_match.carbon:[[@LINE+4]]:3: error: impl declared but not defined [MissingImplDefinition]
  // CHECK:STDERR:   impl as I5;
  // CHECK:STDERR:   ^~~~~~~~~~~
  // CHECK:STDERR:
  impl as I5;
  impl C4(A) as I5 {}

  // CHECK:STDERR: fail_no_match.carbon:[[@LINE+4]]:3: error: impl declared but not defined [MissingImplDefinition]
  // CHECK:STDERR:   impl Self as I6;
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  impl Self as I6;
  impl C4(A) as I6 {}

  // CHECK:STDERR: fail_no_match.carbon:[[@LINE+4]]:3: error: impl declared but not defined [MissingImplDefinition]
  // CHECK:STDERR:   impl forall [B3:! type] C4(A) as J3(B3);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  impl forall [B3:! type] C4(A) as J3(B3);
  impl forall [B3:! type] as J3(B3) {}

  // CHECK:STDERR: fail_no_match.carbon:[[@LINE+4]]:3: error: impl declared but not defined [MissingImplDefinition]
  // CHECK:STDERR:   impl forall [B4:! type] C4(A) as J4(B4);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  impl forall [B4:! type] C4(A) as J4(B4);
  impl forall [B4:! type] Self as J4(B4) {}

  // CHECK:STDERR: fail_no_match.carbon:[[@LINE+4]]:3: error: impl declared but not defined [MissingImplDefinition]
  // CHECK:STDERR:   impl forall [B5:! type] as J5(B5);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  impl forall [B5:! type] as J5(B5);
  impl forall [B5:! type] C4(A) as J5(B5) {}

  // CHECK:STDERR: fail_no_match.carbon:[[@LINE+3]]:3: error: impl declared but not defined [MissingImplDefinition]
  // CHECK:STDERR:   impl forall [B6:! type] Self as J6(B6);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  impl forall [B6:! type] Self as J6(B6);
  impl forall [B6:! type] C4(A) as J6(B6) {}
}

// CHECK:STDOUT: --- match.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %I1.type: type = facet_type <@I1> [template]
// CHECK:STDOUT:   %Self.1: %I1.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %I2.type: type = facet_type <@I2> [template]
// CHECK:STDOUT:   %Self.2: %I2.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %T1: type = bind_symbolic_name T1, 0 [symbolic]
// CHECK:STDOUT:   %T1.patt: type = symbolic_binding_pattern T1, 0 [symbolic]
// CHECK:STDOUT:   %J1.type.1: type = generic_interface_type @J1 [template]
// CHECK:STDOUT:   %J1.generic: %J1.type.1 = struct_value () [template]
// CHECK:STDOUT:   %J1.type.2: type = facet_type <@J1, @J1(%T1)> [symbolic]
// CHECK:STDOUT:   %Self.3: %J1.type.2 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %T2: type = bind_symbolic_name T2, 0 [symbolic]
// CHECK:STDOUT:   %T2.patt: type = symbolic_binding_pattern T2, 0 [symbolic]
// CHECK:STDOUT:   %J2.type.1: type = generic_interface_type @J2 [template]
// CHECK:STDOUT:   %J2.generic: %J2.type.1 = struct_value () [template]
// CHECK:STDOUT:   %J2.type.2: type = facet_type <@J2, @J2(%T2)> [symbolic]
// CHECK:STDOUT:   %Self.4: %J2.type.2 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %C1: type = class_type @C1 [template]
// CHECK:STDOUT:   %impl_witness.1: <witness> = impl_witness () [template]
// CHECK:STDOUT:   %U: type = bind_symbolic_name U, 0 [symbolic]
// CHECK:STDOUT:   %U.patt: type = symbolic_binding_pattern U, 0 [symbolic]
// CHECK:STDOUT:   %J1.type.3: type = facet_type <@J1, @J1(%U)> [symbolic]
// CHECK:STDOUT:   %require_complete.1: <witness> = require_complete_type %J1.type.3 [symbolic]
// CHECK:STDOUT:   %impl_witness.2: <witness> = impl_witness (), @impl.3(%U) [symbolic]
// CHECK:STDOUT:   %V: type = bind_symbolic_name V, 0 [symbolic]
// CHECK:STDOUT:   %V.patt: type = symbolic_binding_pattern V, 0 [symbolic]
// CHECK:STDOUT:   %J2.type.3: type = facet_type <@J2, @J2(%V)> [symbolic]
// CHECK:STDOUT:   %require_complete.2: <witness> = require_complete_type %J2.type.3 [symbolic]
// CHECK:STDOUT:   %impl_witness.3: <witness> = impl_witness (), @impl.4(%V) [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [template]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template]
// CHECK:STDOUT:   %W: type = bind_symbolic_name W, 0 [symbolic]
// CHECK:STDOUT:   %W.patt: type = symbolic_binding_pattern W, 0 [symbolic]
// CHECK:STDOUT:   %C2.type: type = generic_class_type @C2 [template]
// CHECK:STDOUT:   %C2.generic: %C2.type = struct_value () [template]
// CHECK:STDOUT:   %C2: type = class_type @C2, @C2(%W) [symbolic]
// CHECK:STDOUT:   %impl_witness.4: <witness> = impl_witness (), @impl.5(%W) [symbolic]
// CHECK:STDOUT:   %impl_witness.5: <witness> = impl_witness (), @impl.6(%W) [symbolic]
// CHECK:STDOUT:   %X: type = bind_symbolic_name X, 1 [symbolic]
// CHECK:STDOUT:   %X.patt: type = symbolic_binding_pattern X, 1 [symbolic]
// CHECK:STDOUT:   %J1.type.4: type = facet_type <@J1, @J1(%X)> [symbolic]
// CHECK:STDOUT:   %require_complete.3: <witness> = require_complete_type %J1.type.4 [symbolic]
// CHECK:STDOUT:   %impl_witness.6: <witness> = impl_witness (), @impl.7(%W, %X) [symbolic]
// CHECK:STDOUT:   %Y: type = bind_symbolic_name Y, 1 [symbolic]
// CHECK:STDOUT:   %Y.patt: type = symbolic_binding_pattern Y, 1 [symbolic]
// CHECK:STDOUT:   %J2.type.4: type = facet_type <@J2, @J2(%Y)> [symbolic]
// CHECK:STDOUT:   %require_complete.4: <witness> = require_complete_type %J2.type.4 [symbolic]
// CHECK:STDOUT:   %impl_witness.7: <witness> = impl_witness (), @impl.8(%W, %Y) [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .I1 = %I1.decl
// CHECK:STDOUT:     .I2 = %I2.decl
// CHECK:STDOUT:     .J1 = %J1.decl
// CHECK:STDOUT:     .J2 = %J2.decl
// CHECK:STDOUT:     .C1 = %C1.decl
// CHECK:STDOUT:     .C2 = %C2.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %I1.decl: type = interface_decl @I1 [template = constants.%I1.type] {} {}
// CHECK:STDOUT:   %I2.decl: type = interface_decl @I2 [template = constants.%I2.type] {} {}
// CHECK:STDOUT:   %J1.decl: %J1.type.1 = interface_decl @J1 [template = constants.%J1.generic] {
// CHECK:STDOUT:     %T1.patt.loc5_14.1: type = symbolic_binding_pattern T1, 0 [symbolic = %T1.patt.loc5_14.2 (constants.%T1.patt)]
// CHECK:STDOUT:     %T1.param_patt: type = value_param_pattern %T1.patt.loc5_14.1, runtime_param<invalid> [symbolic = %T1.patt.loc5_14.2 (constants.%T1.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T1.param: type = value_param runtime_param<invalid>
// CHECK:STDOUT:     %T1.loc5_14.1: type = bind_symbolic_name T1, 0, %T1.param [symbolic = %T1.loc5_14.2 (constants.%T1)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %J2.decl: %J2.type.1 = interface_decl @J2 [template = constants.%J2.generic] {
// CHECK:STDOUT:     %T2.patt.loc6_14.1: type = symbolic_binding_pattern T2, 0 [symbolic = %T2.patt.loc6_14.2 (constants.%T2.patt)]
// CHECK:STDOUT:     %T2.param_patt: type = value_param_pattern %T2.patt.loc6_14.1, runtime_param<invalid> [symbolic = %T2.patt.loc6_14.2 (constants.%T2.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T2.param: type = value_param runtime_param<invalid>
// CHECK:STDOUT:     %T2.loc6_14.1: type = bind_symbolic_name T2, 0, %T2.param [symbolic = %T2.loc6_14.2 (constants.%T2)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C1.decl: type = class_decl @C1 [template = constants.%C1] {} {}
// CHECK:STDOUT:   %C2.decl: %C2.type = class_decl @C2 [template = constants.%C2.generic] {
// CHECK:STDOUT:     %W.patt.loc24_10.1: type = symbolic_binding_pattern W, 0 [symbolic = %W.patt.loc24_10.2 (constants.%W.patt)]
// CHECK:STDOUT:     %W.param_patt: type = value_param_pattern %W.patt.loc24_10.1, runtime_param<invalid> [symbolic = %W.patt.loc24_10.2 (constants.%W.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %W.param: type = value_param runtime_param<invalid>
// CHECK:STDOUT:     %W.loc24_10.1: type = bind_symbolic_name W, 0, %W.param [symbolic = %W.loc24_10.2 (constants.%W)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I1 {
// CHECK:STDOUT:   %Self: %I1.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.1]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I2 {
// CHECK:STDOUT:   %Self: %I2.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.2]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @J1(%T1.loc5_14.1: type) {
// CHECK:STDOUT:   %T1.loc5_14.2: type = bind_symbolic_name T1, 0 [symbolic = %T1.loc5_14.2 (constants.%T1)]
// CHECK:STDOUT:   %T1.patt.loc5_14.2: type = symbolic_binding_pattern T1, 0 [symbolic = %T1.patt.loc5_14.2 (constants.%T1.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %J1.type: type = facet_type <@J1, @J1(%T1.loc5_14.2)> [symbolic = %J1.type (constants.%J1.type.2)]
// CHECK:STDOUT:   %Self.2: %J1.type.2 = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self.3)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.1: @J1.%J1.type (%J1.type.2) = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self.3)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.1
// CHECK:STDOUT:     witness = ()
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @J2(%T2.loc6_14.1: type) {
// CHECK:STDOUT:   %T2.loc6_14.2: type = bind_symbolic_name T2, 0 [symbolic = %T2.loc6_14.2 (constants.%T2)]
// CHECK:STDOUT:   %T2.patt.loc6_14.2: type = symbolic_binding_pattern T2, 0 [symbolic = %T2.patt.loc6_14.2 (constants.%T2.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %J2.type: type = facet_type <@J2, @J2(%T2.loc6_14.2)> [symbolic = %J2.type (constants.%J2.type.2)]
// CHECK:STDOUT:   %Self.2: %J2.type.2 = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self.4)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.1: @J2.%J2.type (%J2.type.2) = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self.4)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.1
// CHECK:STDOUT:     witness = ()
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.1: %Self.ref.loc11 as %I1.ref.loc11 {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = @C1.%impl_witness.loc11
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.2: %Self.ref.loc14 as %I2.ref.loc14 {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = @C1.%impl_witness.loc14
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl.3(%U.loc17_16.1: type) {
// CHECK:STDOUT:   %U.loc17_16.2: type = bind_symbolic_name U, 0 [symbolic = %U.loc17_16.2 (constants.%U)]
// CHECK:STDOUT:   %U.patt.loc17_16.2: type = symbolic_binding_pattern U, 0 [symbolic = %U.patt.loc17_16.2 (constants.%U.patt)]
// CHECK:STDOUT:   %J1.type.loc17_38.2: type = facet_type <@J1, @J1(%U.loc17_16.2)> [symbolic = %J1.type.loc17_38.2 (constants.%J1.type.3)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type @impl.3.%J1.type.loc17_38.2 (%J1.type.3) [symbolic = %require_complete (constants.%require_complete.1)]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (), @impl.3(%U.loc17_16.2) [symbolic = %impl_witness (constants.%impl_witness.2)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %Self.ref.loc17 as %J1.type.loc17_38.1 {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = @C1.%impl_witness.loc17
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl.4(%V.loc20_16.1: type) {
// CHECK:STDOUT:   %V.loc20_16.2: type = bind_symbolic_name V, 0 [symbolic = %V.loc20_16.2 (constants.%V)]
// CHECK:STDOUT:   %V.patt.loc20_16.2: type = symbolic_binding_pattern V, 0 [symbolic = %V.patt.loc20_16.2 (constants.%V.patt)]
// CHECK:STDOUT:   %J2.type.loc20_33.2: type = facet_type <@J2, @J2(%V.loc20_16.2)> [symbolic = %J2.type.loc20_33.2 (constants.%J2.type.3)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type @impl.4.%J2.type.loc20_33.2 (%J2.type.3) [symbolic = %require_complete (constants.%require_complete.2)]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (), @impl.4(%V.loc20_16.2) [symbolic = %impl_witness (constants.%impl_witness.3)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %Self.ref.loc20 as %J2.type.loc20_33.1 {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = @C1.%impl_witness.loc20
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl.5(@C2.%W.loc24_10.1: type) {
// CHECK:STDOUT:   %W: type = bind_symbolic_name W, 0 [symbolic = %W (constants.%W)]
// CHECK:STDOUT:   %C2: type = class_type @C2, @C2(%W) [symbolic = %C2 (constants.%C2)]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (), @impl.5(%W) [symbolic = %impl_witness (constants.%impl_witness.4)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %Self.ref.loc25 as %I1.ref.loc25 {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = @C2.%impl_witness.loc25
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl.6(@C2.%W.loc24_10.1: type) {
// CHECK:STDOUT:   %W: type = bind_symbolic_name W, 0 [symbolic = %W (constants.%W)]
// CHECK:STDOUT:   %C2: type = class_type @C2, @C2(%W) [symbolic = %C2 (constants.%C2)]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (), @impl.6(%W) [symbolic = %impl_witness (constants.%impl_witness.5)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %Self.ref.loc28 as %I2.ref.loc28 {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = @C2.%impl_witness.loc28
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl.7(@C2.%W.loc24_10.1: type, %X.loc31_16.1: type) {
// CHECK:STDOUT:   %X.loc31_16.2: type = bind_symbolic_name X, 1 [symbolic = %X.loc31_16.2 (constants.%X)]
// CHECK:STDOUT:   %X.patt.loc31_16.2: type = symbolic_binding_pattern X, 1 [symbolic = %X.patt.loc31_16.2 (constants.%X.patt)]
// CHECK:STDOUT:   %W: type = bind_symbolic_name W, 0 [symbolic = %W (constants.%W)]
// CHECK:STDOUT:   %C2: type = class_type @C2, @C2(%W) [symbolic = %C2 (constants.%C2)]
// CHECK:STDOUT:   %J1.type.loc31_38.2: type = facet_type <@J1, @J1(%X.loc31_16.2)> [symbolic = %J1.type.loc31_38.2 (constants.%J1.type.4)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type @impl.7.%J1.type.loc31_38.2 (%J1.type.4) [symbolic = %require_complete (constants.%require_complete.3)]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (), @impl.7(%W, %X.loc31_16.2) [symbolic = %impl_witness (constants.%impl_witness.6)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %Self.ref.loc31 as %J1.type.loc31_38.1 {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = @C2.%impl_witness.loc31
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl.8(@C2.%W.loc24_10.1: type, %Y.loc34_16.1: type) {
// CHECK:STDOUT:   %Y.loc34_16.2: type = bind_symbolic_name Y, 1 [symbolic = %Y.loc34_16.2 (constants.%Y)]
// CHECK:STDOUT:   %Y.patt.loc34_16.2: type = symbolic_binding_pattern Y, 1 [symbolic = %Y.patt.loc34_16.2 (constants.%Y.patt)]
// CHECK:STDOUT:   %W: type = bind_symbolic_name W, 0 [symbolic = %W (constants.%W)]
// CHECK:STDOUT:   %C2: type = class_type @C2, @C2(%W) [symbolic = %C2 (constants.%C2)]
// CHECK:STDOUT:   %J2.type.loc34_33.2: type = facet_type <@J2, @J2(%Y.loc34_16.2)> [symbolic = %J2.type.loc34_33.2 (constants.%J2.type.4)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type @impl.8.%J2.type.loc34_33.2 (%J2.type.4) [symbolic = %require_complete (constants.%require_complete.4)]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (), @impl.8(%W, %Y.loc34_16.2) [symbolic = %impl_witness (constants.%impl_witness.7)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %Self.ref.loc34 as %J2.type.loc34_33.1 {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = @C2.%impl_witness.loc34
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C1 {
// CHECK:STDOUT:   impl_decl @impl.1 [template] {} {
// CHECK:STDOUT:     %Self.ref.loc11: type = name_ref Self, constants.%C1 [template = constants.%C1]
// CHECK:STDOUT:     %I1.ref.loc11: type = name_ref I1, file.%I1.decl [template = constants.%I1.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness.loc11: <witness> = impl_witness () [template = constants.%impl_witness.1]
// CHECK:STDOUT:   impl_decl @impl.1 [template] {} {
// CHECK:STDOUT:     %Self.ref.loc12: type = name_ref Self, constants.%C1 [template = constants.%C1]
// CHECK:STDOUT:     %I1.ref.loc12: type = name_ref I1, file.%I1.decl [template = constants.%I1.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @impl.2 [template] {} {
// CHECK:STDOUT:     %Self.ref.loc14: type = name_ref Self, constants.%C1 [template = constants.%C1]
// CHECK:STDOUT:     %I2.ref.loc14: type = name_ref I2, file.%I2.decl [template = constants.%I2.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness.loc14: <witness> = impl_witness () [template = constants.%impl_witness.1]
// CHECK:STDOUT:   impl_decl @impl.2 [template] {} {
// CHECK:STDOUT:     %Self.ref.loc15: type = name_ref Self, constants.%C1 [template = constants.%C1]
// CHECK:STDOUT:     %I2.ref.loc15: type = name_ref I2, file.%I2.decl [template = constants.%I2.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @impl.3 [template] {
// CHECK:STDOUT:     %U.patt.loc17_16.1: type = symbolic_binding_pattern U, 0 [symbolic = %U.patt.loc17_16.2 (constants.%U.patt)]
// CHECK:STDOUT:     %U.param_patt: type = value_param_pattern %U.patt.loc17_16.1, runtime_param<invalid> [symbolic = %U.patt.loc17_16.2 (constants.%U.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Self.ref.loc17: type = name_ref Self, constants.%C1 [template = constants.%C1]
// CHECK:STDOUT:     %J1.ref.loc17: %J1.type.1 = name_ref J1, file.%J1.decl [template = constants.%J1.generic]
// CHECK:STDOUT:     %U.ref.loc17: type = name_ref U, %U.loc17_16.1 [symbolic = %U.loc17_16.2 (constants.%U)]
// CHECK:STDOUT:     %J1.type.loc17_38.1: type = facet_type <@J1, @J1(constants.%U)> [symbolic = %J1.type.loc17_38.2 (constants.%J1.type.3)]
// CHECK:STDOUT:     %U.param.loc17: type = value_param runtime_param<invalid>
// CHECK:STDOUT:     %U.loc17_16.1: type = bind_symbolic_name U, 0, %U.param.loc17 [symbolic = %U.loc17_16.2 (constants.%U)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness.loc17: <witness> = impl_witness (), @impl.3(constants.%U) [symbolic = @impl.3.%impl_witness (constants.%impl_witness.2)]
// CHECK:STDOUT:   impl_decl @impl.3 [template] {
// CHECK:STDOUT:     %U.patt.loc17_16.1: type = symbolic_binding_pattern U, 0 [symbolic = %U.patt.loc17_16.2 (constants.%U.patt)]
// CHECK:STDOUT:     %U.param_patt: type = value_param_pattern %U.patt.loc17_16.1, runtime_param<invalid> [symbolic = %U.patt.loc17_16.2 (constants.%U.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Self.ref.loc18: type = name_ref Self, constants.%C1 [template = constants.%C1]
// CHECK:STDOUT:     %J1.ref.loc18: %J1.type.1 = name_ref J1, file.%J1.decl [template = constants.%J1.generic]
// CHECK:STDOUT:     %U.ref.loc18: type = name_ref U, %U.loc18 [symbolic = constants.%U]
// CHECK:STDOUT:     %J1.type.loc18: type = facet_type <@J1, @J1(constants.%U)> [symbolic = constants.%J1.type.3]
// CHECK:STDOUT:     %U.param.loc18: type = value_param runtime_param<invalid>
// CHECK:STDOUT:     %U.loc18: type = bind_symbolic_name U, 0, %U.param.loc18 [symbolic = constants.%U]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @impl.4 [template] {
// CHECK:STDOUT:     %V.patt.loc20_16.1: type = symbolic_binding_pattern V, 0 [symbolic = %V.patt.loc20_16.2 (constants.%V.patt)]
// CHECK:STDOUT:     %V.param_patt: type = value_param_pattern %V.patt.loc20_16.1, runtime_param<invalid> [symbolic = %V.patt.loc20_16.2 (constants.%V.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Self.ref.loc20: type = name_ref Self, constants.%C1 [template = constants.%C1]
// CHECK:STDOUT:     %J2.ref.loc20: %J2.type.1 = name_ref J2, file.%J2.decl [template = constants.%J2.generic]
// CHECK:STDOUT:     %V.ref.loc20: type = name_ref V, %V.loc20_16.1 [symbolic = %V.loc20_16.2 (constants.%V)]
// CHECK:STDOUT:     %J2.type.loc20_33.1: type = facet_type <@J2, @J2(constants.%V)> [symbolic = %J2.type.loc20_33.2 (constants.%J2.type.3)]
// CHECK:STDOUT:     %V.param.loc20: type = value_param runtime_param<invalid>
// CHECK:STDOUT:     %V.loc20_16.1: type = bind_symbolic_name V, 0, %V.param.loc20 [symbolic = %V.loc20_16.2 (constants.%V)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness.loc20: <witness> = impl_witness (), @impl.4(constants.%V) [symbolic = @impl.4.%impl_witness (constants.%impl_witness.3)]
// CHECK:STDOUT:   impl_decl @impl.4 [template] {
// CHECK:STDOUT:     %V.patt.loc20_16.1: type = symbolic_binding_pattern V, 0 [symbolic = %V.patt.loc20_16.2 (constants.%V.patt)]
// CHECK:STDOUT:     %V.param_patt: type = value_param_pattern %V.patt.loc20_16.1, runtime_param<invalid> [symbolic = %V.patt.loc20_16.2 (constants.%V.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Self.ref.loc21: type = name_ref Self, constants.%C1 [template = constants.%C1]
// CHECK:STDOUT:     %J2.ref.loc21: %J2.type.1 = name_ref J2, file.%J2.decl [template = constants.%J2.generic]
// CHECK:STDOUT:     %V.ref.loc21: type = name_ref V, %V.loc21 [symbolic = constants.%V]
// CHECK:STDOUT:     %J2.type.loc21: type = facet_type <@J2, @J2(constants.%V)> [symbolic = constants.%J2.type.3]
// CHECK:STDOUT:     %V.param.loc21: type = value_param runtime_param<invalid>
// CHECK:STDOUT:     %V.loc21: type = bind_symbolic_name V, 0, %V.param.loc21 [symbolic = constants.%V]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template = constants.%complete_type]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C1
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @C2(%W.loc24_10.1: type) {
// CHECK:STDOUT:   %W.loc24_10.2: type = bind_symbolic_name W, 0 [symbolic = %W.loc24_10.2 (constants.%W)]
// CHECK:STDOUT:   %W.patt.loc24_10.2: type = symbolic_binding_pattern W, 0 [symbolic = %W.patt.loc24_10.2 (constants.%W.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     impl_decl @impl.5 [template] {} {
// CHECK:STDOUT:       %.loc25: type = specific_constant constants.%C2, @C2(constants.%W) [symbolic = %C2 (constants.%C2)]
// CHECK:STDOUT:       %Self.ref.loc25: type = name_ref Self, %.loc25 [symbolic = %C2 (constants.%C2)]
// CHECK:STDOUT:       %I1.ref.loc25: type = name_ref I1, file.%I1.decl [template = constants.%I1.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %impl_witness.loc25: <witness> = impl_witness (), @impl.5(constants.%W) [symbolic = @impl.5.%impl_witness (constants.%impl_witness.4)]
// CHECK:STDOUT:     impl_decl @impl.5 [template] {} {
// CHECK:STDOUT:       %Self.ref.loc26: type = name_ref Self, constants.%C2 [symbolic = constants.%C2]
// CHECK:STDOUT:       %I1.ref.loc26: type = name_ref I1, file.%I1.decl [template = constants.%I1.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     impl_decl @impl.6 [template] {} {
// CHECK:STDOUT:       %Self.ref.loc28: type = name_ref Self, constants.%C2 [symbolic = %C2 (constants.%C2)]
// CHECK:STDOUT:       %I2.ref.loc28: type = name_ref I2, file.%I2.decl [template = constants.%I2.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %impl_witness.loc28: <witness> = impl_witness (), @impl.6(constants.%W) [symbolic = @impl.6.%impl_witness (constants.%impl_witness.5)]
// CHECK:STDOUT:     impl_decl @impl.6 [template] {} {
// CHECK:STDOUT:       %.loc29: type = specific_constant constants.%C2, @C2(constants.%W) [symbolic = constants.%C2]
// CHECK:STDOUT:       %Self.ref.loc29: type = name_ref Self, %.loc29 [symbolic = constants.%C2]
// CHECK:STDOUT:       %I2.ref.loc29: type = name_ref I2, file.%I2.decl [template = constants.%I2.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     impl_decl @impl.7 [template] {
// CHECK:STDOUT:       %X.patt.loc31_16.1: type = symbolic_binding_pattern X, 1 [symbolic = %X.patt.loc31_16.2 (constants.%X.patt)]
// CHECK:STDOUT:       %X.param_patt: type = value_param_pattern %X.patt.loc31_16.1, runtime_param<invalid> [symbolic = %X.patt.loc31_16.2 (constants.%X.patt)]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %.loc31: type = specific_constant constants.%C2, @C2(constants.%W) [symbolic = %C2 (constants.%C2)]
// CHECK:STDOUT:       %Self.ref.loc31: type = name_ref Self, %.loc31 [symbolic = %C2 (constants.%C2)]
// CHECK:STDOUT:       %J1.ref.loc31: %J1.type.1 = name_ref J1, file.%J1.decl [template = constants.%J1.generic]
// CHECK:STDOUT:       %X.ref.loc31: type = name_ref X, %X.loc31_16.1 [symbolic = %X.loc31_16.2 (constants.%X)]
// CHECK:STDOUT:       %J1.type.loc31_38.1: type = facet_type <@J1, @J1(constants.%X)> [symbolic = %J1.type.loc31_38.2 (constants.%J1.type.4)]
// CHECK:STDOUT:       %X.param.loc31: type = value_param runtime_param<invalid>
// CHECK:STDOUT:       %X.loc31_16.1: type = bind_symbolic_name X, 1, %X.param.loc31 [symbolic = %X.loc31_16.2 (constants.%X)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %impl_witness.loc31: <witness> = impl_witness (), @impl.7(constants.%W, constants.%X) [symbolic = @impl.7.%impl_witness (constants.%impl_witness.6)]
// CHECK:STDOUT:     impl_decl @impl.7 [template] {
// CHECK:STDOUT:       %X.patt.loc31_16.1: type = symbolic_binding_pattern X, 1 [symbolic = %X.patt.loc31_16.2 (constants.%X.patt)]
// CHECK:STDOUT:       %X.param_patt: type = value_param_pattern %X.patt.loc31_16.1, runtime_param<invalid> [symbolic = %X.patt.loc31_16.2 (constants.%X.patt)]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %Self.ref.loc32: type = name_ref Self, constants.%C2 [symbolic = constants.%C2]
// CHECK:STDOUT:       %J1.ref.loc32: %J1.type.1 = name_ref J1, file.%J1.decl [template = constants.%J1.generic]
// CHECK:STDOUT:       %X.ref.loc32: type = name_ref X, %X.loc32 [symbolic = constants.%X]
// CHECK:STDOUT:       %J1.type.loc32: type = facet_type <@J1, @J1(constants.%X)> [symbolic = constants.%J1.type.4]
// CHECK:STDOUT:       %X.param.loc32: type = value_param runtime_param<invalid>
// CHECK:STDOUT:       %X.loc32: type = bind_symbolic_name X, 1, %X.param.loc32 [symbolic = constants.%X]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     impl_decl @impl.8 [template] {
// CHECK:STDOUT:       %Y.patt.loc34_16.1: type = symbolic_binding_pattern Y, 1 [symbolic = %Y.patt.loc34_16.2 (constants.%Y.patt)]
// CHECK:STDOUT:       %Y.param_patt: type = value_param_pattern %Y.patt.loc34_16.1, runtime_param<invalid> [symbolic = %Y.patt.loc34_16.2 (constants.%Y.patt)]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %Self.ref.loc34: type = name_ref Self, constants.%C2 [symbolic = %C2 (constants.%C2)]
// CHECK:STDOUT:       %J2.ref.loc34: %J2.type.1 = name_ref J2, file.%J2.decl [template = constants.%J2.generic]
// CHECK:STDOUT:       %Y.ref.loc34: type = name_ref Y, %Y.loc34_16.1 [symbolic = %Y.loc34_16.2 (constants.%Y)]
// CHECK:STDOUT:       %J2.type.loc34_33.1: type = facet_type <@J2, @J2(constants.%Y)> [symbolic = %J2.type.loc34_33.2 (constants.%J2.type.4)]
// CHECK:STDOUT:       %Y.param.loc34: type = value_param runtime_param<invalid>
// CHECK:STDOUT:       %Y.loc34_16.1: type = bind_symbolic_name Y, 1, %Y.param.loc34 [symbolic = %Y.loc34_16.2 (constants.%Y)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %impl_witness.loc34: <witness> = impl_witness (), @impl.8(constants.%W, constants.%Y) [symbolic = @impl.8.%impl_witness (constants.%impl_witness.7)]
// CHECK:STDOUT:     impl_decl @impl.8 [template] {
// CHECK:STDOUT:       %Y.patt.loc34_16.1: type = symbolic_binding_pattern Y, 1 [symbolic = %Y.patt.loc34_16.2 (constants.%Y.patt)]
// CHECK:STDOUT:       %Y.param_patt: type = value_param_pattern %Y.patt.loc34_16.1, runtime_param<invalid> [symbolic = %Y.patt.loc34_16.2 (constants.%Y.patt)]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %.loc35: type = specific_constant constants.%C2, @C2(constants.%W) [symbolic = constants.%C2]
// CHECK:STDOUT:       %Self.ref.loc35: type = name_ref Self, %.loc35 [symbolic = constants.%C2]
// CHECK:STDOUT:       %J2.ref.loc35: %J2.type.1 = name_ref J2, file.%J2.decl [template = constants.%J2.generic]
// CHECK:STDOUT:       %Y.ref.loc35: type = name_ref Y, %Y.loc35 [symbolic = constants.%Y]
// CHECK:STDOUT:       %J2.type.loc35: type = facet_type <@J2, @J2(constants.%Y)> [symbolic = constants.%J2.type.4]
// CHECK:STDOUT:       %Y.param.loc35: type = value_param runtime_param<invalid>
// CHECK:STDOUT:       %Y.loc35: type = bind_symbolic_name Y, 1, %Y.param.loc35 [symbolic = constants.%Y]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness %empty_struct_type [template = constants.%complete_type]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%C2
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J1(constants.%T1) {
// CHECK:STDOUT:   %T1.loc5_14.2 => constants.%T1
// CHECK:STDOUT:   %T1.patt.loc5_14.2 => constants.%T1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J1(%T1.loc5_14.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J2(constants.%T2) {
// CHECK:STDOUT:   %T2.loc6_14.2 => constants.%T2
// CHECK:STDOUT:   %T2.patt.loc6_14.2 => constants.%T2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J2(%T2.loc6_14.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J1(constants.%U) {
// CHECK:STDOUT:   %T1.loc5_14.2 => constants.%U
// CHECK:STDOUT:   %T1.patt.loc5_14.2 => constants.%U
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %J1.type => constants.%J1.type.3
// CHECK:STDOUT:   %Self.2 => constants.%Self.3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.3(constants.%U) {
// CHECK:STDOUT:   %U.loc17_16.2 => constants.%U
// CHECK:STDOUT:   %U.patt.loc17_16.2 => constants.%U
// CHECK:STDOUT:   %J1.type.loc17_38.2 => constants.%J1.type.3
// CHECK:STDOUT:   %require_complete => constants.%require_complete.1
// CHECK:STDOUT:   %impl_witness => constants.%impl_witness.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J1(@impl.3.%U.loc17_16.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.3(%U.loc17_16.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J2(constants.%V) {
// CHECK:STDOUT:   %T2.loc6_14.2 => constants.%V
// CHECK:STDOUT:   %T2.patt.loc6_14.2 => constants.%V
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %J2.type => constants.%J2.type.3
// CHECK:STDOUT:   %Self.2 => constants.%Self.4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.4(constants.%V) {
// CHECK:STDOUT:   %V.loc20_16.2 => constants.%V
// CHECK:STDOUT:   %V.patt.loc20_16.2 => constants.%V
// CHECK:STDOUT:   %J2.type.loc20_33.2 => constants.%J2.type.3
// CHECK:STDOUT:   %require_complete => constants.%require_complete.2
// CHECK:STDOUT:   %impl_witness => constants.%impl_witness.3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J2(@impl.4.%V.loc20_16.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.4(%V.loc20_16.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C2(constants.%W) {
// CHECK:STDOUT:   %W.loc24_10.2 => constants.%W
// CHECK:STDOUT:   %W.patt.loc24_10.2 => constants.%W
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.5(constants.%W) {
// CHECK:STDOUT:   %W => constants.%W
// CHECK:STDOUT:   %C2 => constants.%C2
// CHECK:STDOUT:   %impl_witness => constants.%impl_witness.4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C2(@impl.5.%W) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.5(%W) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.6(constants.%W) {
// CHECK:STDOUT:   %W => constants.%W
// CHECK:STDOUT:   %C2 => constants.%C2
// CHECK:STDOUT:   %impl_witness => constants.%impl_witness.5
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C2(@impl.6.%W) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.6(%W) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J1(constants.%X) {
// CHECK:STDOUT:   %T1.loc5_14.2 => constants.%X
// CHECK:STDOUT:   %T1.patt.loc5_14.2 => constants.%X
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %J1.type => constants.%J1.type.4
// CHECK:STDOUT:   %Self.2 => constants.%Self.3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.7(constants.%W, constants.%X) {
// CHECK:STDOUT:   %X.loc31_16.2 => constants.%X
// CHECK:STDOUT:   %X.patt.loc31_16.2 => constants.%X
// CHECK:STDOUT:   %W => constants.%W
// CHECK:STDOUT:   %C2 => constants.%C2
// CHECK:STDOUT:   %J1.type.loc31_38.2 => constants.%J1.type.4
// CHECK:STDOUT:   %require_complete => constants.%require_complete.3
// CHECK:STDOUT:   %impl_witness => constants.%impl_witness.6
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C2(@impl.7.%W) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J1(@impl.7.%X.loc31_16.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.7(%W, %X.loc31_16.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J2(constants.%Y) {
// CHECK:STDOUT:   %T2.loc6_14.2 => constants.%Y
// CHECK:STDOUT:   %T2.patt.loc6_14.2 => constants.%Y
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %J2.type => constants.%J2.type.4
// CHECK:STDOUT:   %Self.2 => constants.%Self.4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.8(constants.%W, constants.%Y) {
// CHECK:STDOUT:   %Y.loc34_16.2 => constants.%Y
// CHECK:STDOUT:   %Y.patt.loc34_16.2 => constants.%Y
// CHECK:STDOUT:   %W => constants.%W
// CHECK:STDOUT:   %C2 => constants.%C2
// CHECK:STDOUT:   %J2.type.loc34_33.2 => constants.%J2.type.4
// CHECK:STDOUT:   %require_complete => constants.%require_complete.4
// CHECK:STDOUT:   %impl_witness => constants.%impl_witness.7
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C2(@impl.8.%W) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J2(@impl.8.%Y.loc34_16.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.8(%W, %Y.loc34_16.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_no_match.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %I3.type: type = facet_type <@I3> [template]
// CHECK:STDOUT:   %Self.1: %I3.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %I4.type: type = facet_type <@I4> [template]
// CHECK:STDOUT:   %Self.2: %I4.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %I5.type: type = facet_type <@I5> [template]
// CHECK:STDOUT:   %Self.3: %I5.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %I6.type: type = facet_type <@I6> [template]
// CHECK:STDOUT:   %Self.4: %I6.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %T3: type = bind_symbolic_name T3, 0 [symbolic]
// CHECK:STDOUT:   %T3.patt: type = symbolic_binding_pattern T3, 0 [symbolic]
// CHECK:STDOUT:   %J3.type.1: type = generic_interface_type @J3 [template]
// CHECK:STDOUT:   %J3.generic: %J3.type.1 = struct_value () [template]
// CHECK:STDOUT:   %J3.type.2: type = facet_type <@J3, @J3(%T3)> [symbolic]
// CHECK:STDOUT:   %Self.5: %J3.type.2 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %T4: type = bind_symbolic_name T4, 0 [symbolic]
// CHECK:STDOUT:   %T4.patt: type = symbolic_binding_pattern T4, 0 [symbolic]
// CHECK:STDOUT:   %J4.type.1: type = generic_interface_type @J4 [template]
// CHECK:STDOUT:   %J4.generic: %J4.type.1 = struct_value () [template]
// CHECK:STDOUT:   %J4.type.2: type = facet_type <@J4, @J4(%T4)> [symbolic]
// CHECK:STDOUT:   %Self.6: %J4.type.2 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %T5: type = bind_symbolic_name T5, 0 [symbolic]
// CHECK:STDOUT:   %T5.patt: type = symbolic_binding_pattern T5, 0 [symbolic]
// CHECK:STDOUT:   %J5.type.1: type = generic_interface_type @J5 [template]
// CHECK:STDOUT:   %J5.generic: %J5.type.1 = struct_value () [template]
// CHECK:STDOUT:   %J5.type.2: type = facet_type <@J5, @J5(%T5)> [symbolic]
// CHECK:STDOUT:   %Self.7: %J5.type.2 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %T6: type = bind_symbolic_name T6, 0 [symbolic]
// CHECK:STDOUT:   %T6.patt: type = symbolic_binding_pattern T6, 0 [symbolic]
// CHECK:STDOUT:   %J6.type.1: type = generic_interface_type @J6 [template]
// CHECK:STDOUT:   %J6.generic: %J6.type.1 = struct_value () [template]
// CHECK:STDOUT:   %J6.type.2: type = facet_type <@J6, @J6(%T6)> [symbolic]
// CHECK:STDOUT:   %Self.8: %J6.type.2 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %C3: type = class_type @C3 [template]
// CHECK:STDOUT:   %impl_witness.1: <witness> = impl_witness () [template]
// CHECK:STDOUT:   %Z3: type = bind_symbolic_name Z3, 0 [symbolic]
// CHECK:STDOUT:   %Z3.patt: type = symbolic_binding_pattern Z3, 0 [symbolic]
// CHECK:STDOUT:   %J3.type.3: type = facet_type <@J3, @J3(%Z3)> [symbolic]
// CHECK:STDOUT:   %require_complete.1: <witness> = require_complete_type %J3.type.3 [symbolic]
// CHECK:STDOUT:   %impl_witness.2: <witness> = impl_witness (), @impl.9(%Z3) [symbolic]
// CHECK:STDOUT:   %impl_witness.3: <witness> = impl_witness (), @impl.10(%Z3) [symbolic]
// CHECK:STDOUT:   %Z4: type = bind_symbolic_name Z4, 0 [symbolic]
// CHECK:STDOUT:   %Z4.patt: type = symbolic_binding_pattern Z4, 0 [symbolic]
// CHECK:STDOUT:   %J4.type.3: type = facet_type <@J4, @J4(%Z4)> [symbolic]
// CHECK:STDOUT:   %require_complete.2: <witness> = require_complete_type %J4.type.3 [symbolic]
// CHECK:STDOUT:   %impl_witness.4: <witness> = impl_witness (), @impl.11(%Z4) [symbolic]
// CHECK:STDOUT:   %impl_witness.5: <witness> = impl_witness (), @impl.12(%Z4) [symbolic]
// CHECK:STDOUT:   %Z5: type = bind_symbolic_name Z5, 0 [symbolic]
// CHECK:STDOUT:   %Z5.patt: type = symbolic_binding_pattern Z5, 0 [symbolic]
// CHECK:STDOUT:   %J5.type.3: type = facet_type <@J5, @J5(%Z5)> [symbolic]
// CHECK:STDOUT:   %require_complete.3: <witness> = require_complete_type %J5.type.3 [symbolic]
// CHECK:STDOUT:   %impl_witness.6: <witness> = impl_witness (), @impl.13(%Z5) [symbolic]
// CHECK:STDOUT:   %impl_witness.7: <witness> = impl_witness (), @impl.14(%Z5) [symbolic]
// CHECK:STDOUT:   %Z6: type = bind_symbolic_name Z6, 0 [symbolic]
// CHECK:STDOUT:   %Z6.patt: type = symbolic_binding_pattern Z6, 0 [symbolic]
// CHECK:STDOUT:   %J6.type.3: type = facet_type <@J6, @J6(%Z6)> [symbolic]
// CHECK:STDOUT:   %require_complete.4: <witness> = require_complete_type %J6.type.3 [symbolic]
// CHECK:STDOUT:   %impl_witness.8: <witness> = impl_witness (), @impl.15(%Z6) [symbolic]
// CHECK:STDOUT:   %impl_witness.9: <witness> = impl_witness (), @impl.16(%Z6) [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [template]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template]
// CHECK:STDOUT:   %A: type = bind_symbolic_name A, 0 [symbolic]
// CHECK:STDOUT:   %A.patt: type = symbolic_binding_pattern A, 0 [symbolic]
// CHECK:STDOUT:   %C4.type: type = generic_class_type @C4 [template]
// CHECK:STDOUT:   %C4.generic: %C4.type = struct_value () [template]
// CHECK:STDOUT:   %C4: type = class_type @C4, @C4(%A) [symbolic]
// CHECK:STDOUT:   %impl_witness.10: <witness> = impl_witness (), @impl.17(%A) [symbolic]
// CHECK:STDOUT:   %impl_witness.11: <witness> = impl_witness (), @impl.18(%A) [symbolic]
// CHECK:STDOUT:   %impl_witness.12: <witness> = impl_witness (), @impl.19(%A) [symbolic]
// CHECK:STDOUT:   %impl_witness.13: <witness> = impl_witness (), @impl.20(%A) [symbolic]
// CHECK:STDOUT:   %impl_witness.14: <witness> = impl_witness (), @impl.21(%A) [symbolic]
// CHECK:STDOUT:   %impl_witness.15: <witness> = impl_witness (), @impl.22(%A) [symbolic]
// CHECK:STDOUT:   %impl_witness.16: <witness> = impl_witness (), @impl.23(%A) [symbolic]
// CHECK:STDOUT:   %impl_witness.17: <witness> = impl_witness (), @impl.24(%A) [symbolic]
// CHECK:STDOUT:   %B3: type = bind_symbolic_name B3, 1 [symbolic]
// CHECK:STDOUT:   %B3.patt: type = symbolic_binding_pattern B3, 1 [symbolic]
// CHECK:STDOUT:   %J3.type.4: type = facet_type <@J3, @J3(%B3)> [symbolic]
// CHECK:STDOUT:   %require_complete.5: <witness> = require_complete_type %J3.type.4 [symbolic]
// CHECK:STDOUT:   %impl_witness.18: <witness> = impl_witness (), @impl.25(%A, %B3) [symbolic]
// CHECK:STDOUT:   %impl_witness.19: <witness> = impl_witness (), @impl.26(%A, %B3) [symbolic]
// CHECK:STDOUT:   %B4: type = bind_symbolic_name B4, 1 [symbolic]
// CHECK:STDOUT:   %B4.patt: type = symbolic_binding_pattern B4, 1 [symbolic]
// CHECK:STDOUT:   %J4.type.4: type = facet_type <@J4, @J4(%B4)> [symbolic]
// CHECK:STDOUT:   %require_complete.6: <witness> = require_complete_type %J4.type.4 [symbolic]
// CHECK:STDOUT:   %impl_witness.20: <witness> = impl_witness (), @impl.27(%A, %B4) [symbolic]
// CHECK:STDOUT:   %impl_witness.21: <witness> = impl_witness (), @impl.28(%A, %B4) [symbolic]
// CHECK:STDOUT:   %B5: type = bind_symbolic_name B5, 1 [symbolic]
// CHECK:STDOUT:   %B5.patt: type = symbolic_binding_pattern B5, 1 [symbolic]
// CHECK:STDOUT:   %J5.type.4: type = facet_type <@J5, @J5(%B5)> [symbolic]
// CHECK:STDOUT:   %require_complete.7: <witness> = require_complete_type %J5.type.4 [symbolic]
// CHECK:STDOUT:   %impl_witness.22: <witness> = impl_witness (), @impl.29(%A, %B5) [symbolic]
// CHECK:STDOUT:   %impl_witness.23: <witness> = impl_witness (), @impl.30(%A, %B5) [symbolic]
// CHECK:STDOUT:   %B6: type = bind_symbolic_name B6, 1 [symbolic]
// CHECK:STDOUT:   %B6.patt: type = symbolic_binding_pattern B6, 1 [symbolic]
// CHECK:STDOUT:   %J6.type.4: type = facet_type <@J6, @J6(%B6)> [symbolic]
// CHECK:STDOUT:   %require_complete.8: <witness> = require_complete_type %J6.type.4 [symbolic]
// CHECK:STDOUT:   %impl_witness.24: <witness> = impl_witness (), @impl.31(%A, %B6) [symbolic]
// CHECK:STDOUT:   %impl_witness.25: <witness> = impl_witness (), @impl.32(%A, %B6) [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .I3 = %I3.decl
// CHECK:STDOUT:     .I4 = %I4.decl
// CHECK:STDOUT:     .I5 = %I5.decl
// CHECK:STDOUT:     .I6 = %I6.decl
// CHECK:STDOUT:     .J3 = %J3.decl
// CHECK:STDOUT:     .J4 = %J4.decl
// CHECK:STDOUT:     .J5 = %J5.decl
// CHECK:STDOUT:     .J6 = %J6.decl
// CHECK:STDOUT:     .C3 = %C3.decl
// CHECK:STDOUT:     .C4 = %C4.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %I3.decl: type = interface_decl @I3 [template = constants.%I3.type] {} {}
// CHECK:STDOUT:   %I4.decl: type = interface_decl @I4 [template = constants.%I4.type] {} {}
// CHECK:STDOUT:   %I5.decl: type = interface_decl @I5 [template = constants.%I5.type] {} {}
// CHECK:STDOUT:   %I6.decl: type = interface_decl @I6 [template = constants.%I6.type] {} {}
// CHECK:STDOUT:   %J3.decl: %J3.type.1 = interface_decl @J3 [template = constants.%J3.generic] {
// CHECK:STDOUT:     %T3.patt.loc7_14.1: type = symbolic_binding_pattern T3, 0 [symbolic = %T3.patt.loc7_14.2 (constants.%T3.patt)]
// CHECK:STDOUT:     %T3.param_patt: type = value_param_pattern %T3.patt.loc7_14.1, runtime_param<invalid> [symbolic = %T3.patt.loc7_14.2 (constants.%T3.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T3.param: type = value_param runtime_param<invalid>
// CHECK:STDOUT:     %T3.loc7_14.1: type = bind_symbolic_name T3, 0, %T3.param [symbolic = %T3.loc7_14.2 (constants.%T3)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %J4.decl: %J4.type.1 = interface_decl @J4 [template = constants.%J4.generic] {
// CHECK:STDOUT:     %T4.patt.loc8_14.1: type = symbolic_binding_pattern T4, 0 [symbolic = %T4.patt.loc8_14.2 (constants.%T4.patt)]
// CHECK:STDOUT:     %T4.param_patt: type = value_param_pattern %T4.patt.loc8_14.1, runtime_param<invalid> [symbolic = %T4.patt.loc8_14.2 (constants.%T4.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T4.param: type = value_param runtime_param<invalid>
// CHECK:STDOUT:     %T4.loc8_14.1: type = bind_symbolic_name T4, 0, %T4.param [symbolic = %T4.loc8_14.2 (constants.%T4)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %J5.decl: %J5.type.1 = interface_decl @J5 [template = constants.%J5.generic] {
// CHECK:STDOUT:     %T5.patt.loc9_14.1: type = symbolic_binding_pattern T5, 0 [symbolic = %T5.patt.loc9_14.2 (constants.%T5.patt)]
// CHECK:STDOUT:     %T5.param_patt: type = value_param_pattern %T5.patt.loc9_14.1, runtime_param<invalid> [symbolic = %T5.patt.loc9_14.2 (constants.%T5.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T5.param: type = value_param runtime_param<invalid>
// CHECK:STDOUT:     %T5.loc9_14.1: type = bind_symbolic_name T5, 0, %T5.param [symbolic = %T5.loc9_14.2 (constants.%T5)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %J6.decl: %J6.type.1 = interface_decl @J6 [template = constants.%J6.generic] {
// CHECK:STDOUT:     %T6.patt.loc10_14.1: type = symbolic_binding_pattern T6, 0 [symbolic = %T6.patt.loc10_14.2 (constants.%T6.patt)]
// CHECK:STDOUT:     %T6.param_patt: type = value_param_pattern %T6.patt.loc10_14.1, runtime_param<invalid> [symbolic = %T6.patt.loc10_14.2 (constants.%T6.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T6.param: type = value_param runtime_param<invalid>
// CHECK:STDOUT:     %T6.loc10_14.1: type = bind_symbolic_name T6, 0, %T6.param [symbolic = %T6.loc10_14.2 (constants.%T6)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C3.decl: type = class_decl @C3 [template = constants.%C3] {} {}
// CHECK:STDOUT:   %C4.decl: %C4.type = class_decl @C4 [template = constants.%C4.generic] {
// CHECK:STDOUT:     %A.patt.loc72_10.1: type = symbolic_binding_pattern A, 0 [symbolic = %A.patt.loc72_10.2 (constants.%A.patt)]
// CHECK:STDOUT:     %A.param_patt: type = value_param_pattern %A.patt.loc72_10.1, runtime_param<invalid> [symbolic = %A.patt.loc72_10.2 (constants.%A.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %A.param: type = value_param runtime_param<invalid>
// CHECK:STDOUT:     %A.loc72_10.1: type = bind_symbolic_name A, 0, %A.param [symbolic = %A.loc72_10.2 (constants.%A)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I3 {
// CHECK:STDOUT:   %Self: %I3.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.1]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I4 {
// CHECK:STDOUT:   %Self: %I4.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.2]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I5 {
// CHECK:STDOUT:   %Self: %I5.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.3]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I6 {
// CHECK:STDOUT:   %Self: %I6.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.4]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @J3(%T3.loc7_14.1: type) {
// CHECK:STDOUT:   %T3.loc7_14.2: type = bind_symbolic_name T3, 0 [symbolic = %T3.loc7_14.2 (constants.%T3)]
// CHECK:STDOUT:   %T3.patt.loc7_14.2: type = symbolic_binding_pattern T3, 0 [symbolic = %T3.patt.loc7_14.2 (constants.%T3.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %J3.type: type = facet_type <@J3, @J3(%T3.loc7_14.2)> [symbolic = %J3.type (constants.%J3.type.2)]
// CHECK:STDOUT:   %Self.2: %J3.type.2 = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self.5)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.1: @J3.%J3.type (%J3.type.2) = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self.5)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.1
// CHECK:STDOUT:     witness = ()
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @J4(%T4.loc8_14.1: type) {
// CHECK:STDOUT:   %T4.loc8_14.2: type = bind_symbolic_name T4, 0 [symbolic = %T4.loc8_14.2 (constants.%T4)]
// CHECK:STDOUT:   %T4.patt.loc8_14.2: type = symbolic_binding_pattern T4, 0 [symbolic = %T4.patt.loc8_14.2 (constants.%T4.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %J4.type: type = facet_type <@J4, @J4(%T4.loc8_14.2)> [symbolic = %J4.type (constants.%J4.type.2)]
// CHECK:STDOUT:   %Self.2: %J4.type.2 = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self.6)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.1: @J4.%J4.type (%J4.type.2) = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self.6)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.1
// CHECK:STDOUT:     witness = ()
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @J5(%T5.loc9_14.1: type) {
// CHECK:STDOUT:   %T5.loc9_14.2: type = bind_symbolic_name T5, 0 [symbolic = %T5.loc9_14.2 (constants.%T5)]
// CHECK:STDOUT:   %T5.patt.loc9_14.2: type = symbolic_binding_pattern T5, 0 [symbolic = %T5.patt.loc9_14.2 (constants.%T5.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %J5.type: type = facet_type <@J5, @J5(%T5.loc9_14.2)> [symbolic = %J5.type (constants.%J5.type.2)]
// CHECK:STDOUT:   %Self.2: %J5.type.2 = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self.7)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.1: @J5.%J5.type (%J5.type.2) = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self.7)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.1
// CHECK:STDOUT:     witness = ()
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @J6(%T6.loc10_14.1: type) {
// CHECK:STDOUT:   %T6.loc10_14.2: type = bind_symbolic_name T6, 0 [symbolic = %T6.loc10_14.2 (constants.%T6)]
// CHECK:STDOUT:   %T6.patt.loc10_14.2: type = symbolic_binding_pattern T6, 0 [symbolic = %T6.patt.loc10_14.2 (constants.%T6.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %J6.type: type = facet_type <@J6, @J6(%T6.loc10_14.2)> [symbolic = %J6.type (constants.%J6.type.2)]
// CHECK:STDOUT:   %Self.2: %J6.type.2 = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self.8)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.1: @J6.%J6.type (%J6.type.2) = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self.8)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.1
// CHECK:STDOUT:     witness = ()
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.1: %C3.ref as %I3.ref;
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.2: %Self.ref as %I3.ref {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = @C3.%impl_witness.loc20
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.3: %C3.ref as %I4.ref;
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.4: %Self.ref as %I4.ref {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = @C3.%impl_witness.loc27
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.5: %Self.ref as %I5.ref;
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.6: %C3.ref as %I5.ref {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = @C3.%impl_witness.loc34
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.7: %Self.ref as %I6.ref;
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.8: %C3.ref as %I6.ref {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = @C3.%impl_witness.loc41
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl.9(%Z3.loc47_16.1: type) {
// CHECK:STDOUT:   %Z3.loc47_16.2: type = bind_symbolic_name Z3, 0 [symbolic = %Z3.loc47_16.2 (constants.%Z3)]
// CHECK:STDOUT:   %Z3.patt.loc47_16.2: type = symbolic_binding_pattern Z3, 0 [symbolic = %Z3.patt.loc47_16.2 (constants.%Z3.patt)]
// CHECK:STDOUT:   %J3.type.loc47_38.2: type = facet_type <@J3, @J3(%Z3.loc47_16.2)> [symbolic = %J3.type.loc47_38.2 (constants.%J3.type.3)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type @impl.9.%J3.type.loc47_38.2 (%J3.type.3) [symbolic = %require_complete (constants.%require_complete.1)]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (), @impl.9(%Z3.loc47_16.2) [symbolic = %impl_witness (constants.%impl_witness.2)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %C3.ref as %J3.type.loc47_38.1;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl.10(%Z3.loc48_16.1: type) {
// CHECK:STDOUT:   %Z3.loc48_16.2: type = bind_symbolic_name Z3, 0 [symbolic = %Z3.loc48_16.2 (constants.%Z3)]
// CHECK:STDOUT:   %Z3.patt.loc48_16.2: type = symbolic_binding_pattern Z3, 0 [symbolic = %Z3.patt.loc48_16.2 (constants.%Z3.patt)]
// CHECK:STDOUT:   %J3.type.loc48_35.2: type = facet_type <@J3, @J3(%Z3.loc48_16.2)> [symbolic = %J3.type.loc48_35.2 (constants.%J3.type.3)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type @impl.10.%J3.type.loc48_35.2 (%J3.type.3) [symbolic = %require_complete (constants.%require_complete.1)]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (), @impl.10(%Z3.loc48_16.2) [symbolic = %impl_witness (constants.%impl_witness.3)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %Self.ref as %J3.type.loc48_35.1 {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = @C3.%impl_witness.loc48
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl.11(%Z4.loc54_16.1: type) {
// CHECK:STDOUT:   %Z4.loc54_16.2: type = bind_symbolic_name Z4, 0 [symbolic = %Z4.loc54_16.2 (constants.%Z4)]
// CHECK:STDOUT:   %Z4.patt.loc54_16.2: type = symbolic_binding_pattern Z4, 0 [symbolic = %Z4.patt.loc54_16.2 (constants.%Z4.patt)]
// CHECK:STDOUT:   %J4.type.loc54_38.2: type = facet_type <@J4, @J4(%Z4.loc54_16.2)> [symbolic = %J4.type.loc54_38.2 (constants.%J4.type.3)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type @impl.11.%J4.type.loc54_38.2 (%J4.type.3) [symbolic = %require_complete (constants.%require_complete.2)]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (), @impl.11(%Z4.loc54_16.2) [symbolic = %impl_witness (constants.%impl_witness.4)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %C3.ref as %J4.type.loc54_38.1;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl.12(%Z4.loc55_16.1: type) {
// CHECK:STDOUT:   %Z4.loc55_16.2: type = bind_symbolic_name Z4, 0 [symbolic = %Z4.loc55_16.2 (constants.%Z4)]
// CHECK:STDOUT:   %Z4.patt.loc55_16.2: type = symbolic_binding_pattern Z4, 0 [symbolic = %Z4.patt.loc55_16.2 (constants.%Z4.patt)]
// CHECK:STDOUT:   %J4.type.loc55_40.2: type = facet_type <@J4, @J4(%Z4.loc55_16.2)> [symbolic = %J4.type.loc55_40.2 (constants.%J4.type.3)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type @impl.12.%J4.type.loc55_40.2 (%J4.type.3) [symbolic = %require_complete (constants.%require_complete.2)]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (), @impl.12(%Z4.loc55_16.2) [symbolic = %impl_witness (constants.%impl_witness.5)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %Self.ref as %J4.type.loc55_40.1 {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = @C3.%impl_witness.loc55
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl.13(%Z5.loc61_16.1: type) {
// CHECK:STDOUT:   %Z5.loc61_16.2: type = bind_symbolic_name Z5, 0 [symbolic = %Z5.loc61_16.2 (constants.%Z5)]
// CHECK:STDOUT:   %Z5.patt.loc61_16.2: type = symbolic_binding_pattern Z5, 0 [symbolic = %Z5.patt.loc61_16.2 (constants.%Z5.patt)]
// CHECK:STDOUT:   %J5.type.loc61_35.2: type = facet_type <@J5, @J5(%Z5.loc61_16.2)> [symbolic = %J5.type.loc61_35.2 (constants.%J5.type.3)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type @impl.13.%J5.type.loc61_35.2 (%J5.type.3) [symbolic = %require_complete (constants.%require_complete.3)]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (), @impl.13(%Z5.loc61_16.2) [symbolic = %impl_witness (constants.%impl_witness.6)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %Self.ref as %J5.type.loc61_35.1;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl.14(%Z5.loc62_16.1: type) {
// CHECK:STDOUT:   %Z5.loc62_16.2: type = bind_symbolic_name Z5, 0 [symbolic = %Z5.loc62_16.2 (constants.%Z5)]
// CHECK:STDOUT:   %Z5.patt.loc62_16.2: type = symbolic_binding_pattern Z5, 0 [symbolic = %Z5.patt.loc62_16.2 (constants.%Z5.patt)]
// CHECK:STDOUT:   %J5.type.loc62_38.2: type = facet_type <@J5, @J5(%Z5.loc62_16.2)> [symbolic = %J5.type.loc62_38.2 (constants.%J5.type.3)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type @impl.14.%J5.type.loc62_38.2 (%J5.type.3) [symbolic = %require_complete (constants.%require_complete.3)]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (), @impl.14(%Z5.loc62_16.2) [symbolic = %impl_witness (constants.%impl_witness.7)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %C3.ref as %J5.type.loc62_38.1 {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = @C3.%impl_witness.loc62
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl.15(%Z6.loc68_16.1: type) {
// CHECK:STDOUT:   %Z6.loc68_16.2: type = bind_symbolic_name Z6, 0 [symbolic = %Z6.loc68_16.2 (constants.%Z6)]
// CHECK:STDOUT:   %Z6.patt.loc68_16.2: type = symbolic_binding_pattern Z6, 0 [symbolic = %Z6.patt.loc68_16.2 (constants.%Z6.patt)]
// CHECK:STDOUT:   %J6.type.loc68_40.2: type = facet_type <@J6, @J6(%Z6.loc68_16.2)> [symbolic = %J6.type.loc68_40.2 (constants.%J6.type.3)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type @impl.15.%J6.type.loc68_40.2 (%J6.type.3) [symbolic = %require_complete (constants.%require_complete.4)]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (), @impl.15(%Z6.loc68_16.2) [symbolic = %impl_witness (constants.%impl_witness.8)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %Self.ref as %J6.type.loc68_40.1;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl.16(%Z6.loc69_16.1: type) {
// CHECK:STDOUT:   %Z6.loc69_16.2: type = bind_symbolic_name Z6, 0 [symbolic = %Z6.loc69_16.2 (constants.%Z6)]
// CHECK:STDOUT:   %Z6.patt.loc69_16.2: type = symbolic_binding_pattern Z6, 0 [symbolic = %Z6.patt.loc69_16.2 (constants.%Z6.patt)]
// CHECK:STDOUT:   %J6.type.loc69_38.2: type = facet_type <@J6, @J6(%Z6.loc69_16.2)> [symbolic = %J6.type.loc69_38.2 (constants.%J6.type.3)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type @impl.16.%J6.type.loc69_38.2 (%J6.type.3) [symbolic = %require_complete (constants.%require_complete.4)]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (), @impl.16(%Z6.loc69_16.2) [symbolic = %impl_witness (constants.%impl_witness.9)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %C3.ref as %J6.type.loc69_38.1 {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = @C3.%impl_witness.loc69
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl.17(@C4.%A.loc72_10.1: type) {
// CHECK:STDOUT:   %A: type = bind_symbolic_name A, 0 [symbolic = %A (constants.%A)]
// CHECK:STDOUT:   %C4.loc77_12.2: type = class_type @C4, @C4(%A) [symbolic = %C4.loc77_12.2 (constants.%C4)]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (), @impl.17(%A) [symbolic = %impl_witness (constants.%impl_witness.10)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %C4.loc77_12.1 as %I3.ref;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl.18(@C4.%A.loc72_10.1: type) {
// CHECK:STDOUT:   %A: type = bind_symbolic_name A, 0 [symbolic = %A (constants.%A)]
// CHECK:STDOUT:   %C4: type = class_type @C4, @C4(%A) [symbolic = %C4 (constants.%C4)]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (), @impl.18(%A) [symbolic = %impl_witness (constants.%impl_witness.11)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %Self.ref as %I3.ref {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = @C4.%impl_witness.loc78
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl.19(@C4.%A.loc72_10.1: type) {
// CHECK:STDOUT:   %A: type = bind_symbolic_name A, 0 [symbolic = %A (constants.%A)]
// CHECK:STDOUT:   %C4.loc84_12.2: type = class_type @C4, @C4(%A) [symbolic = %C4.loc84_12.2 (constants.%C4)]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (), @impl.19(%A) [symbolic = %impl_witness (constants.%impl_witness.12)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %C4.loc84_12.1 as %I4.ref;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl.20(@C4.%A.loc72_10.1: type) {
// CHECK:STDOUT:   %A: type = bind_symbolic_name A, 0 [symbolic = %A (constants.%A)]
// CHECK:STDOUT:   %C4: type = class_type @C4, @C4(%A) [symbolic = %C4 (constants.%C4)]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (), @impl.20(%A) [symbolic = %impl_witness (constants.%impl_witness.13)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %Self.ref as %I4.ref {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = @C4.%impl_witness.loc85
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl.21(@C4.%A.loc72_10.1: type) {
// CHECK:STDOUT:   %A: type = bind_symbolic_name A, 0 [symbolic = %A (constants.%A)]
// CHECK:STDOUT:   %C4: type = class_type @C4, @C4(%A) [symbolic = %C4 (constants.%C4)]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (), @impl.21(%A) [symbolic = %impl_witness (constants.%impl_witness.14)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %Self.ref as %I5.ref;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl.22(@C4.%A.loc72_10.1: type) {
// CHECK:STDOUT:   %A: type = bind_symbolic_name A, 0 [symbolic = %A (constants.%A)]
// CHECK:STDOUT:   %C4.loc92_12.2: type = class_type @C4, @C4(%A) [symbolic = %C4.loc92_12.2 (constants.%C4)]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (), @impl.22(%A) [symbolic = %impl_witness (constants.%impl_witness.15)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %C4.loc92_12.1 as %I5.ref {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = @C4.%impl_witness.loc92
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl.23(@C4.%A.loc72_10.1: type) {
// CHECK:STDOUT:   %A: type = bind_symbolic_name A, 0 [symbolic = %A (constants.%A)]
// CHECK:STDOUT:   %C4: type = class_type @C4, @C4(%A) [symbolic = %C4 (constants.%C4)]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (), @impl.23(%A) [symbolic = %impl_witness (constants.%impl_witness.16)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %Self.ref as %I6.ref;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl.24(@C4.%A.loc72_10.1: type) {
// CHECK:STDOUT:   %A: type = bind_symbolic_name A, 0 [symbolic = %A (constants.%A)]
// CHECK:STDOUT:   %C4.loc99_12.2: type = class_type @C4, @C4(%A) [symbolic = %C4.loc99_12.2 (constants.%C4)]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (), @impl.24(%A) [symbolic = %impl_witness (constants.%impl_witness.17)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %C4.loc99_12.1 as %I6.ref {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = @C4.%impl_witness.loc99
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl.25(@C4.%A.loc72_10.1: type, %B3.loc105_16.1: type) {
// CHECK:STDOUT:   %B3.loc105_16.2: type = bind_symbolic_name B3, 1 [symbolic = %B3.loc105_16.2 (constants.%B3)]
// CHECK:STDOUT:   %B3.patt.loc105_16.2: type = symbolic_binding_pattern B3, 1 [symbolic = %B3.patt.loc105_16.2 (constants.%B3.patt)]
// CHECK:STDOUT:   %A: type = bind_symbolic_name A, 0 [symbolic = %A (constants.%A)]
// CHECK:STDOUT:   %C4.loc105_31.2: type = class_type @C4, @C4(%A) [symbolic = %C4.loc105_31.2 (constants.%C4)]
// CHECK:STDOUT:   %J3.type.loc105_41.2: type = facet_type <@J3, @J3(%B3.loc105_16.2)> [symbolic = %J3.type.loc105_41.2 (constants.%J3.type.4)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type @impl.25.%J3.type.loc105_41.2 (%J3.type.4) [symbolic = %require_complete (constants.%require_complete.5)]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (), @impl.25(%A, %B3.loc105_16.2) [symbolic = %impl_witness (constants.%impl_witness.18)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %C4.loc105_31.1 as %J3.type.loc105_41.1;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl.26(@C4.%A.loc72_10.1: type, %B3.loc106_16.1: type) {
// CHECK:STDOUT:   %B3.loc106_16.2: type = bind_symbolic_name B3, 1 [symbolic = %B3.loc106_16.2 (constants.%B3)]
// CHECK:STDOUT:   %B3.patt.loc106_16.2: type = symbolic_binding_pattern B3, 1 [symbolic = %B3.patt.loc106_16.2 (constants.%B3.patt)]
// CHECK:STDOUT:   %A: type = bind_symbolic_name A, 0 [symbolic = %A (constants.%A)]
// CHECK:STDOUT:   %C4: type = class_type @C4, @C4(%A) [symbolic = %C4 (constants.%C4)]
// CHECK:STDOUT:   %J3.type.loc106_35.2: type = facet_type <@J3, @J3(%B3.loc106_16.2)> [symbolic = %J3.type.loc106_35.2 (constants.%J3.type.4)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type @impl.26.%J3.type.loc106_35.2 (%J3.type.4) [symbolic = %require_complete (constants.%require_complete.5)]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (), @impl.26(%A, %B3.loc106_16.2) [symbolic = %impl_witness (constants.%impl_witness.19)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %Self.ref as %J3.type.loc106_35.1 {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = @C4.%impl_witness.loc106
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl.27(@C4.%A.loc72_10.1: type, %B4.loc112_16.1: type) {
// CHECK:STDOUT:   %B4.loc112_16.2: type = bind_symbolic_name B4, 1 [symbolic = %B4.loc112_16.2 (constants.%B4)]
// CHECK:STDOUT:   %B4.patt.loc112_16.2: type = symbolic_binding_pattern B4, 1 [symbolic = %B4.patt.loc112_16.2 (constants.%B4.patt)]
// CHECK:STDOUT:   %A: type = bind_symbolic_name A, 0 [symbolic = %A (constants.%A)]
// CHECK:STDOUT:   %C4.loc112_31.2: type = class_type @C4, @C4(%A) [symbolic = %C4.loc112_31.2 (constants.%C4)]
// CHECK:STDOUT:   %J4.type.loc112_41.2: type = facet_type <@J4, @J4(%B4.loc112_16.2)> [symbolic = %J4.type.loc112_41.2 (constants.%J4.type.4)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type @impl.27.%J4.type.loc112_41.2 (%J4.type.4) [symbolic = %require_complete (constants.%require_complete.6)]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (), @impl.27(%A, %B4.loc112_16.2) [symbolic = %impl_witness (constants.%impl_witness.20)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %C4.loc112_31.1 as %J4.type.loc112_41.1;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl.28(@C4.%A.loc72_10.1: type, %B4.loc113_16.1: type) {
// CHECK:STDOUT:   %B4.loc113_16.2: type = bind_symbolic_name B4, 1 [symbolic = %B4.loc113_16.2 (constants.%B4)]
// CHECK:STDOUT:   %B4.patt.loc113_16.2: type = symbolic_binding_pattern B4, 1 [symbolic = %B4.patt.loc113_16.2 (constants.%B4.patt)]
// CHECK:STDOUT:   %A: type = bind_symbolic_name A, 0 [symbolic = %A (constants.%A)]
// CHECK:STDOUT:   %C4: type = class_type @C4, @C4(%A) [symbolic = %C4 (constants.%C4)]
// CHECK:STDOUT:   %J4.type.loc113_40.2: type = facet_type <@J4, @J4(%B4.loc113_16.2)> [symbolic = %J4.type.loc113_40.2 (constants.%J4.type.4)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type @impl.28.%J4.type.loc113_40.2 (%J4.type.4) [symbolic = %require_complete (constants.%require_complete.6)]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (), @impl.28(%A, %B4.loc113_16.2) [symbolic = %impl_witness (constants.%impl_witness.21)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %Self.ref as %J4.type.loc113_40.1 {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = @C4.%impl_witness.loc113
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl.29(@C4.%A.loc72_10.1: type, %B5.loc119_16.1: type) {
// CHECK:STDOUT:   %B5.loc119_16.2: type = bind_symbolic_name B5, 1 [symbolic = %B5.loc119_16.2 (constants.%B5)]
// CHECK:STDOUT:   %B5.patt.loc119_16.2: type = symbolic_binding_pattern B5, 1 [symbolic = %B5.patt.loc119_16.2 (constants.%B5.patt)]
// CHECK:STDOUT:   %A: type = bind_symbolic_name A, 0 [symbolic = %A (constants.%A)]
// CHECK:STDOUT:   %C4: type = class_type @C4, @C4(%A) [symbolic = %C4 (constants.%C4)]
// CHECK:STDOUT:   %J5.type.loc119_35.2: type = facet_type <@J5, @J5(%B5.loc119_16.2)> [symbolic = %J5.type.loc119_35.2 (constants.%J5.type.4)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type @impl.29.%J5.type.loc119_35.2 (%J5.type.4) [symbolic = %require_complete (constants.%require_complete.7)]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (), @impl.29(%A, %B5.loc119_16.2) [symbolic = %impl_witness (constants.%impl_witness.22)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %Self.ref as %J5.type.loc119_35.1;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl.30(@C4.%A.loc72_10.1: type, %B5.loc120_16.1: type) {
// CHECK:STDOUT:   %B5.loc120_16.2: type = bind_symbolic_name B5, 1 [symbolic = %B5.loc120_16.2 (constants.%B5)]
// CHECK:STDOUT:   %B5.patt.loc120_16.2: type = symbolic_binding_pattern B5, 1 [symbolic = %B5.patt.loc120_16.2 (constants.%B5.patt)]
// CHECK:STDOUT:   %A: type = bind_symbolic_name A, 0 [symbolic = %A (constants.%A)]
// CHECK:STDOUT:   %C4.loc120_31.2: type = class_type @C4, @C4(%A) [symbolic = %C4.loc120_31.2 (constants.%C4)]
// CHECK:STDOUT:   %J5.type.loc120_41.2: type = facet_type <@J5, @J5(%B5.loc120_16.2)> [symbolic = %J5.type.loc120_41.2 (constants.%J5.type.4)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type @impl.30.%J5.type.loc120_41.2 (%J5.type.4) [symbolic = %require_complete (constants.%require_complete.7)]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (), @impl.30(%A, %B5.loc120_16.2) [symbolic = %impl_witness (constants.%impl_witness.23)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %C4.loc120_31.1 as %J5.type.loc120_41.1 {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = @C4.%impl_witness.loc120
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl.31(@C4.%A.loc72_10.1: type, %B6.loc125_16.1: type) {
// CHECK:STDOUT:   %B6.loc125_16.2: type = bind_symbolic_name B6, 1 [symbolic = %B6.loc125_16.2 (constants.%B6)]
// CHECK:STDOUT:   %B6.patt.loc125_16.2: type = symbolic_binding_pattern B6, 1 [symbolic = %B6.patt.loc125_16.2 (constants.%B6.patt)]
// CHECK:STDOUT:   %A: type = bind_symbolic_name A, 0 [symbolic = %A (constants.%A)]
// CHECK:STDOUT:   %C4: type = class_type @C4, @C4(%A) [symbolic = %C4 (constants.%C4)]
// CHECK:STDOUT:   %J6.type.loc125_40.2: type = facet_type <@J6, @J6(%B6.loc125_16.2)> [symbolic = %J6.type.loc125_40.2 (constants.%J6.type.4)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type @impl.31.%J6.type.loc125_40.2 (%J6.type.4) [symbolic = %require_complete (constants.%require_complete.8)]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (), @impl.31(%A, %B6.loc125_16.2) [symbolic = %impl_witness (constants.%impl_witness.24)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %Self.ref as %J6.type.loc125_40.1;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl.32(@C4.%A.loc72_10.1: type, %B6.loc126_16.1: type) {
// CHECK:STDOUT:   %B6.loc126_16.2: type = bind_symbolic_name B6, 1 [symbolic = %B6.loc126_16.2 (constants.%B6)]
// CHECK:STDOUT:   %B6.patt.loc126_16.2: type = symbolic_binding_pattern B6, 1 [symbolic = %B6.patt.loc126_16.2 (constants.%B6.patt)]
// CHECK:STDOUT:   %A: type = bind_symbolic_name A, 0 [symbolic = %A (constants.%A)]
// CHECK:STDOUT:   %C4.loc126_31.2: type = class_type @C4, @C4(%A) [symbolic = %C4.loc126_31.2 (constants.%C4)]
// CHECK:STDOUT:   %J6.type.loc126_41.2: type = facet_type <@J6, @J6(%B6.loc126_16.2)> [symbolic = %J6.type.loc126_41.2 (constants.%J6.type.4)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type @impl.32.%J6.type.loc126_41.2 (%J6.type.4) [symbolic = %require_complete (constants.%require_complete.8)]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (), @impl.32(%A, %B6.loc126_16.2) [symbolic = %impl_witness (constants.%impl_witness.25)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %C4.loc126_31.1 as %J6.type.loc126_41.1 {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = @C4.%impl_witness.loc126
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C3 {
// CHECK:STDOUT:   impl_decl @impl.1 [template] {} {
// CHECK:STDOUT:     %C3.ref: type = name_ref C3, file.%C3.decl [template = constants.%C3]
// CHECK:STDOUT:     %I3.ref: type = name_ref I3, file.%I3.decl [template = constants.%I3.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness.loc19: <witness> = impl_witness () [template = constants.%impl_witness.1]
// CHECK:STDOUT:   impl_decl @impl.2 [template] {} {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%C3 [template = constants.%C3]
// CHECK:STDOUT:     %I3.ref: type = name_ref I3, file.%I3.decl [template = constants.%I3.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness.loc20: <witness> = impl_witness () [template = constants.%impl_witness.1]
// CHECK:STDOUT:   impl_decl @impl.3 [template] {} {
// CHECK:STDOUT:     %C3.ref: type = name_ref C3, file.%C3.decl [template = constants.%C3]
// CHECK:STDOUT:     %I4.ref: type = name_ref I4, file.%I4.decl [template = constants.%I4.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness.loc26: <witness> = impl_witness () [template = constants.%impl_witness.1]
// CHECK:STDOUT:   impl_decl @impl.4 [template] {} {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%C3 [template = constants.%C3]
// CHECK:STDOUT:     %I4.ref: type = name_ref I4, file.%I4.decl [template = constants.%I4.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness.loc27: <witness> = impl_witness () [template = constants.%impl_witness.1]
// CHECK:STDOUT:   impl_decl @impl.5 [template] {} {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%C3 [template = constants.%C3]
// CHECK:STDOUT:     %I5.ref: type = name_ref I5, file.%I5.decl [template = constants.%I5.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness.loc33: <witness> = impl_witness () [template = constants.%impl_witness.1]
// CHECK:STDOUT:   impl_decl @impl.6 [template] {} {
// CHECK:STDOUT:     %C3.ref: type = name_ref C3, file.%C3.decl [template = constants.%C3]
// CHECK:STDOUT:     %I5.ref: type = name_ref I5, file.%I5.decl [template = constants.%I5.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness.loc34: <witness> = impl_witness () [template = constants.%impl_witness.1]
// CHECK:STDOUT:   impl_decl @impl.7 [template] {} {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%C3 [template = constants.%C3]
// CHECK:STDOUT:     %I6.ref: type = name_ref I6, file.%I6.decl [template = constants.%I6.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness.loc40: <witness> = impl_witness () [template = constants.%impl_witness.1]
// CHECK:STDOUT:   impl_decl @impl.8 [template] {} {
// CHECK:STDOUT:     %C3.ref: type = name_ref C3, file.%C3.decl [template = constants.%C3]
// CHECK:STDOUT:     %I6.ref: type = name_ref I6, file.%I6.decl [template = constants.%I6.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness.loc41: <witness> = impl_witness () [template = constants.%impl_witness.1]
// CHECK:STDOUT:   impl_decl @impl.9 [template] {
// CHECK:STDOUT:     %Z3.patt.loc47_16.1: type = symbolic_binding_pattern Z3, 0 [symbolic = %Z3.patt.loc47_16.2 (constants.%Z3.patt)]
// CHECK:STDOUT:     %Z3.param_patt: type = value_param_pattern %Z3.patt.loc47_16.1, runtime_param<invalid> [symbolic = %Z3.patt.loc47_16.2 (constants.%Z3.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C3.ref: type = name_ref C3, file.%C3.decl [template = constants.%C3]
// CHECK:STDOUT:     %J3.ref: %J3.type.1 = name_ref J3, file.%J3.decl [template = constants.%J3.generic]
// CHECK:STDOUT:     %Z3.ref: type = name_ref Z3, %Z3.loc47_16.1 [symbolic = %Z3.loc47_16.2 (constants.%Z3)]
// CHECK:STDOUT:     %J3.type.loc47_38.1: type = facet_type <@J3, @J3(constants.%Z3)> [symbolic = %J3.type.loc47_38.2 (constants.%J3.type.3)]
// CHECK:STDOUT:     %Z3.param: type = value_param runtime_param<invalid>
// CHECK:STDOUT:     %Z3.loc47_16.1: type = bind_symbolic_name Z3, 0, %Z3.param [symbolic = %Z3.loc47_16.2 (constants.%Z3)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness.loc47: <witness> = impl_witness (), @impl.9(constants.%Z3) [symbolic = @impl.9.%impl_witness (constants.%impl_witness.2)]
// CHECK:STDOUT:   impl_decl @impl.10 [template] {
// CHECK:STDOUT:     %Z3.patt.loc48_16.1: type = symbolic_binding_pattern Z3, 0 [symbolic = %Z3.patt.loc48_16.2 (constants.%Z3.patt)]
// CHECK:STDOUT:     %Z3.param_patt: type = value_param_pattern %Z3.patt.loc48_16.1, runtime_param<invalid> [symbolic = %Z3.patt.loc48_16.2 (constants.%Z3.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%C3 [template = constants.%C3]
// CHECK:STDOUT:     %J3.ref: %J3.type.1 = name_ref J3, file.%J3.decl [template = constants.%J3.generic]
// CHECK:STDOUT:     %Z3.ref: type = name_ref Z3, %Z3.loc48_16.1 [symbolic = %Z3.loc48_16.2 (constants.%Z3)]
// CHECK:STDOUT:     %J3.type.loc48_35.1: type = facet_type <@J3, @J3(constants.%Z3)> [symbolic = %J3.type.loc48_35.2 (constants.%J3.type.3)]
// CHECK:STDOUT:     %Z3.param: type = value_param runtime_param<invalid>
// CHECK:STDOUT:     %Z3.loc48_16.1: type = bind_symbolic_name Z3, 0, %Z3.param [symbolic = %Z3.loc48_16.2 (constants.%Z3)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness.loc48: <witness> = impl_witness (), @impl.10(constants.%Z3) [symbolic = @impl.10.%impl_witness (constants.%impl_witness.3)]
// CHECK:STDOUT:   impl_decl @impl.11 [template] {
// CHECK:STDOUT:     %Z4.patt.loc54_16.1: type = symbolic_binding_pattern Z4, 0 [symbolic = %Z4.patt.loc54_16.2 (constants.%Z4.patt)]
// CHECK:STDOUT:     %Z4.param_patt: type = value_param_pattern %Z4.patt.loc54_16.1, runtime_param<invalid> [symbolic = %Z4.patt.loc54_16.2 (constants.%Z4.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C3.ref: type = name_ref C3, file.%C3.decl [template = constants.%C3]
// CHECK:STDOUT:     %J4.ref: %J4.type.1 = name_ref J4, file.%J4.decl [template = constants.%J4.generic]
// CHECK:STDOUT:     %Z4.ref: type = name_ref Z4, %Z4.loc54_16.1 [symbolic = %Z4.loc54_16.2 (constants.%Z4)]
// CHECK:STDOUT:     %J4.type.loc54_38.1: type = facet_type <@J4, @J4(constants.%Z4)> [symbolic = %J4.type.loc54_38.2 (constants.%J4.type.3)]
// CHECK:STDOUT:     %Z4.param: type = value_param runtime_param<invalid>
// CHECK:STDOUT:     %Z4.loc54_16.1: type = bind_symbolic_name Z4, 0, %Z4.param [symbolic = %Z4.loc54_16.2 (constants.%Z4)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness.loc54: <witness> = impl_witness (), @impl.11(constants.%Z4) [symbolic = @impl.11.%impl_witness (constants.%impl_witness.4)]
// CHECK:STDOUT:   impl_decl @impl.12 [template] {
// CHECK:STDOUT:     %Z4.patt.loc55_16.1: type = symbolic_binding_pattern Z4, 0 [symbolic = %Z4.patt.loc55_16.2 (constants.%Z4.patt)]
// CHECK:STDOUT:     %Z4.param_patt: type = value_param_pattern %Z4.patt.loc55_16.1, runtime_param<invalid> [symbolic = %Z4.patt.loc55_16.2 (constants.%Z4.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%C3 [template = constants.%C3]
// CHECK:STDOUT:     %J4.ref: %J4.type.1 = name_ref J4, file.%J4.decl [template = constants.%J4.generic]
// CHECK:STDOUT:     %Z4.ref: type = name_ref Z4, %Z4.loc55_16.1 [symbolic = %Z4.loc55_16.2 (constants.%Z4)]
// CHECK:STDOUT:     %J4.type.loc55_40.1: type = facet_type <@J4, @J4(constants.%Z4)> [symbolic = %J4.type.loc55_40.2 (constants.%J4.type.3)]
// CHECK:STDOUT:     %Z4.param: type = value_param runtime_param<invalid>
// CHECK:STDOUT:     %Z4.loc55_16.1: type = bind_symbolic_name Z4, 0, %Z4.param [symbolic = %Z4.loc55_16.2 (constants.%Z4)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness.loc55: <witness> = impl_witness (), @impl.12(constants.%Z4) [symbolic = @impl.12.%impl_witness (constants.%impl_witness.5)]
// CHECK:STDOUT:   impl_decl @impl.13 [template] {
// CHECK:STDOUT:     %Z5.patt.loc61_16.1: type = symbolic_binding_pattern Z5, 0 [symbolic = %Z5.patt.loc61_16.2 (constants.%Z5.patt)]
// CHECK:STDOUT:     %Z5.param_patt: type = value_param_pattern %Z5.patt.loc61_16.1, runtime_param<invalid> [symbolic = %Z5.patt.loc61_16.2 (constants.%Z5.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%C3 [template = constants.%C3]
// CHECK:STDOUT:     %J5.ref: %J5.type.1 = name_ref J5, file.%J5.decl [template = constants.%J5.generic]
// CHECK:STDOUT:     %Z5.ref: type = name_ref Z5, %Z5.loc61_16.1 [symbolic = %Z5.loc61_16.2 (constants.%Z5)]
// CHECK:STDOUT:     %J5.type.loc61_35.1: type = facet_type <@J5, @J5(constants.%Z5)> [symbolic = %J5.type.loc61_35.2 (constants.%J5.type.3)]
// CHECK:STDOUT:     %Z5.param: type = value_param runtime_param<invalid>
// CHECK:STDOUT:     %Z5.loc61_16.1: type = bind_symbolic_name Z5, 0, %Z5.param [symbolic = %Z5.loc61_16.2 (constants.%Z5)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness.loc61: <witness> = impl_witness (), @impl.13(constants.%Z5) [symbolic = @impl.13.%impl_witness (constants.%impl_witness.6)]
// CHECK:STDOUT:   impl_decl @impl.14 [template] {
// CHECK:STDOUT:     %Z5.patt.loc62_16.1: type = symbolic_binding_pattern Z5, 0 [symbolic = %Z5.patt.loc62_16.2 (constants.%Z5.patt)]
// CHECK:STDOUT:     %Z5.param_patt: type = value_param_pattern %Z5.patt.loc62_16.1, runtime_param<invalid> [symbolic = %Z5.patt.loc62_16.2 (constants.%Z5.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C3.ref: type = name_ref C3, file.%C3.decl [template = constants.%C3]
// CHECK:STDOUT:     %J5.ref: %J5.type.1 = name_ref J5, file.%J5.decl [template = constants.%J5.generic]
// CHECK:STDOUT:     %Z5.ref: type = name_ref Z5, %Z5.loc62_16.1 [symbolic = %Z5.loc62_16.2 (constants.%Z5)]
// CHECK:STDOUT:     %J5.type.loc62_38.1: type = facet_type <@J5, @J5(constants.%Z5)> [symbolic = %J5.type.loc62_38.2 (constants.%J5.type.3)]
// CHECK:STDOUT:     %Z5.param: type = value_param runtime_param<invalid>
// CHECK:STDOUT:     %Z5.loc62_16.1: type = bind_symbolic_name Z5, 0, %Z5.param [symbolic = %Z5.loc62_16.2 (constants.%Z5)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness.loc62: <witness> = impl_witness (), @impl.14(constants.%Z5) [symbolic = @impl.14.%impl_witness (constants.%impl_witness.7)]
// CHECK:STDOUT:   impl_decl @impl.15 [template] {
// CHECK:STDOUT:     %Z6.patt.loc68_16.1: type = symbolic_binding_pattern Z6, 0 [symbolic = %Z6.patt.loc68_16.2 (constants.%Z6.patt)]
// CHECK:STDOUT:     %Z6.param_patt: type = value_param_pattern %Z6.patt.loc68_16.1, runtime_param<invalid> [symbolic = %Z6.patt.loc68_16.2 (constants.%Z6.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%C3 [template = constants.%C3]
// CHECK:STDOUT:     %J6.ref: %J6.type.1 = name_ref J6, file.%J6.decl [template = constants.%J6.generic]
// CHECK:STDOUT:     %Z6.ref: type = name_ref Z6, %Z6.loc68_16.1 [symbolic = %Z6.loc68_16.2 (constants.%Z6)]
// CHECK:STDOUT:     %J6.type.loc68_40.1: type = facet_type <@J6, @J6(constants.%Z6)> [symbolic = %J6.type.loc68_40.2 (constants.%J6.type.3)]
// CHECK:STDOUT:     %Z6.param: type = value_param runtime_param<invalid>
// CHECK:STDOUT:     %Z6.loc68_16.1: type = bind_symbolic_name Z6, 0, %Z6.param [symbolic = %Z6.loc68_16.2 (constants.%Z6)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness.loc68: <witness> = impl_witness (), @impl.15(constants.%Z6) [symbolic = @impl.15.%impl_witness (constants.%impl_witness.8)]
// CHECK:STDOUT:   impl_decl @impl.16 [template] {
// CHECK:STDOUT:     %Z6.patt.loc69_16.1: type = symbolic_binding_pattern Z6, 0 [symbolic = %Z6.patt.loc69_16.2 (constants.%Z6.patt)]
// CHECK:STDOUT:     %Z6.param_patt: type = value_param_pattern %Z6.patt.loc69_16.1, runtime_param<invalid> [symbolic = %Z6.patt.loc69_16.2 (constants.%Z6.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C3.ref: type = name_ref C3, file.%C3.decl [template = constants.%C3]
// CHECK:STDOUT:     %J6.ref: %J6.type.1 = name_ref J6, file.%J6.decl [template = constants.%J6.generic]
// CHECK:STDOUT:     %Z6.ref: type = name_ref Z6, %Z6.loc69_16.1 [symbolic = %Z6.loc69_16.2 (constants.%Z6)]
// CHECK:STDOUT:     %J6.type.loc69_38.1: type = facet_type <@J6, @J6(constants.%Z6)> [symbolic = %J6.type.loc69_38.2 (constants.%J6.type.3)]
// CHECK:STDOUT:     %Z6.param: type = value_param runtime_param<invalid>
// CHECK:STDOUT:     %Z6.loc69_16.1: type = bind_symbolic_name Z6, 0, %Z6.param [symbolic = %Z6.loc69_16.2 (constants.%Z6)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness.loc69: <witness> = impl_witness (), @impl.16(constants.%Z6) [symbolic = @impl.16.%impl_witness (constants.%impl_witness.9)]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template = constants.%complete_type]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C3
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @C4(%A.loc72_10.1: type) {
// CHECK:STDOUT:   %A.loc72_10.2: type = bind_symbolic_name A, 0 [symbolic = %A.loc72_10.2 (constants.%A)]
// CHECK:STDOUT:   %A.patt.loc72_10.2: type = symbolic_binding_pattern A, 0 [symbolic = %A.patt.loc72_10.2 (constants.%A.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     impl_decl @impl.17 [template] {} {
// CHECK:STDOUT:       %C4.ref: %C4.type = name_ref C4, file.%C4.decl [template = constants.%C4.generic]
// CHECK:STDOUT:       %A.ref: type = name_ref A, @C4.%A.loc72_10.1 [symbolic = %A (constants.%A)]
// CHECK:STDOUT:       %C4.loc77_12.1: type = class_type @C4, @C4(constants.%A) [symbolic = %C4.loc77_12.2 (constants.%C4)]
// CHECK:STDOUT:       %I3.ref: type = name_ref I3, file.%I3.decl [template = constants.%I3.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %impl_witness.loc77: <witness> = impl_witness (), @impl.17(constants.%A) [symbolic = @impl.17.%impl_witness (constants.%impl_witness.10)]
// CHECK:STDOUT:     impl_decl @impl.18 [template] {} {
// CHECK:STDOUT:       %Self.ref: type = name_ref Self, constants.%C4 [symbolic = %C4 (constants.%C4)]
// CHECK:STDOUT:       %I3.ref: type = name_ref I3, file.%I3.decl [template = constants.%I3.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %impl_witness.loc78: <witness> = impl_witness (), @impl.18(constants.%A) [symbolic = @impl.18.%impl_witness (constants.%impl_witness.11)]
// CHECK:STDOUT:     impl_decl @impl.19 [template] {} {
// CHECK:STDOUT:       %C4.ref: %C4.type = name_ref C4, file.%C4.decl [template = constants.%C4.generic]
// CHECK:STDOUT:       %A.ref: type = name_ref A, @C4.%A.loc72_10.1 [symbolic = %A (constants.%A)]
// CHECK:STDOUT:       %C4.loc84_12.1: type = class_type @C4, @C4(constants.%A) [symbolic = %C4.loc84_12.2 (constants.%C4)]
// CHECK:STDOUT:       %I4.ref: type = name_ref I4, file.%I4.decl [template = constants.%I4.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %impl_witness.loc84: <witness> = impl_witness (), @impl.19(constants.%A) [symbolic = @impl.19.%impl_witness (constants.%impl_witness.12)]
// CHECK:STDOUT:     impl_decl @impl.20 [template] {} {
// CHECK:STDOUT:       %.loc85: type = specific_constant constants.%C4, @C4(constants.%A) [symbolic = %C4 (constants.%C4)]
// CHECK:STDOUT:       %Self.ref: type = name_ref Self, %.loc85 [symbolic = %C4 (constants.%C4)]
// CHECK:STDOUT:       %I4.ref: type = name_ref I4, file.%I4.decl [template = constants.%I4.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %impl_witness.loc85: <witness> = impl_witness (), @impl.20(constants.%A) [symbolic = @impl.20.%impl_witness (constants.%impl_witness.13)]
// CHECK:STDOUT:     impl_decl @impl.21 [template] {} {
// CHECK:STDOUT:       %Self.ref: type = name_ref Self, constants.%C4 [symbolic = %C4 (constants.%C4)]
// CHECK:STDOUT:       %I5.ref: type = name_ref I5, file.%I5.decl [template = constants.%I5.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %impl_witness.loc91: <witness> = impl_witness (), @impl.21(constants.%A) [symbolic = @impl.21.%impl_witness (constants.%impl_witness.14)]
// CHECK:STDOUT:     impl_decl @impl.22 [template] {} {
// CHECK:STDOUT:       %C4.ref: %C4.type = name_ref C4, file.%C4.decl [template = constants.%C4.generic]
// CHECK:STDOUT:       %A.ref: type = name_ref A, @C4.%A.loc72_10.1 [symbolic = %A (constants.%A)]
// CHECK:STDOUT:       %C4.loc92_12.1: type = class_type @C4, @C4(constants.%A) [symbolic = %C4.loc92_12.2 (constants.%C4)]
// CHECK:STDOUT:       %I5.ref: type = name_ref I5, file.%I5.decl [template = constants.%I5.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %impl_witness.loc92: <witness> = impl_witness (), @impl.22(constants.%A) [symbolic = @impl.22.%impl_witness (constants.%impl_witness.15)]
// CHECK:STDOUT:     impl_decl @impl.23 [template] {} {
// CHECK:STDOUT:       %.loc98: type = specific_constant constants.%C4, @C4(constants.%A) [symbolic = %C4 (constants.%C4)]
// CHECK:STDOUT:       %Self.ref: type = name_ref Self, %.loc98 [symbolic = %C4 (constants.%C4)]
// CHECK:STDOUT:       %I6.ref: type = name_ref I6, file.%I6.decl [template = constants.%I6.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %impl_witness.loc98: <witness> = impl_witness (), @impl.23(constants.%A) [symbolic = @impl.23.%impl_witness (constants.%impl_witness.16)]
// CHECK:STDOUT:     impl_decl @impl.24 [template] {} {
// CHECK:STDOUT:       %C4.ref: %C4.type = name_ref C4, file.%C4.decl [template = constants.%C4.generic]
// CHECK:STDOUT:       %A.ref: type = name_ref A, @C4.%A.loc72_10.1 [symbolic = %A (constants.%A)]
// CHECK:STDOUT:       %C4.loc99_12.1: type = class_type @C4, @C4(constants.%A) [symbolic = %C4.loc99_12.2 (constants.%C4)]
// CHECK:STDOUT:       %I6.ref: type = name_ref I6, file.%I6.decl [template = constants.%I6.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %impl_witness.loc99: <witness> = impl_witness (), @impl.24(constants.%A) [symbolic = @impl.24.%impl_witness (constants.%impl_witness.17)]
// CHECK:STDOUT:     impl_decl @impl.25 [template] {
// CHECK:STDOUT:       %B3.patt.loc105_16.1: type = symbolic_binding_pattern B3, 1 [symbolic = %B3.patt.loc105_16.2 (constants.%B3.patt)]
// CHECK:STDOUT:       %B3.param_patt: type = value_param_pattern %B3.patt.loc105_16.1, runtime_param<invalid> [symbolic = %B3.patt.loc105_16.2 (constants.%B3.patt)]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %C4.ref: %C4.type = name_ref C4, file.%C4.decl [template = constants.%C4.generic]
// CHECK:STDOUT:       %A.ref: type = name_ref A, @C4.%A.loc72_10.1 [symbolic = %A (constants.%A)]
// CHECK:STDOUT:       %C4.loc105_31.1: type = class_type @C4, @C4(constants.%A) [symbolic = %C4.loc105_31.2 (constants.%C4)]
// CHECK:STDOUT:       %J3.ref: %J3.type.1 = name_ref J3, file.%J3.decl [template = constants.%J3.generic]
// CHECK:STDOUT:       %B3.ref: type = name_ref B3, %B3.loc105_16.1 [symbolic = %B3.loc105_16.2 (constants.%B3)]
// CHECK:STDOUT:       %J3.type.loc105_41.1: type = facet_type <@J3, @J3(constants.%B3)> [symbolic = %J3.type.loc105_41.2 (constants.%J3.type.4)]
// CHECK:STDOUT:       %B3.param: type = value_param runtime_param<invalid>
// CHECK:STDOUT:       %B3.loc105_16.1: type = bind_symbolic_name B3, 1, %B3.param [symbolic = %B3.loc105_16.2 (constants.%B3)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %impl_witness.loc105: <witness> = impl_witness (), @impl.25(constants.%A, constants.%B3) [symbolic = @impl.25.%impl_witness (constants.%impl_witness.18)]
// CHECK:STDOUT:     impl_decl @impl.26 [template] {
// CHECK:STDOUT:       %B3.patt.loc106_16.1: type = symbolic_binding_pattern B3, 1 [symbolic = %B3.patt.loc106_16.2 (constants.%B3.patt)]
// CHECK:STDOUT:       %B3.param_patt: type = value_param_pattern %B3.patt.loc106_16.1, runtime_param<invalid> [symbolic = %B3.patt.loc106_16.2 (constants.%B3.patt)]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %Self.ref: type = name_ref Self, constants.%C4 [symbolic = %C4 (constants.%C4)]
// CHECK:STDOUT:       %J3.ref: %J3.type.1 = name_ref J3, file.%J3.decl [template = constants.%J3.generic]
// CHECK:STDOUT:       %B3.ref: type = name_ref B3, %B3.loc106_16.1 [symbolic = %B3.loc106_16.2 (constants.%B3)]
// CHECK:STDOUT:       %J3.type.loc106_35.1: type = facet_type <@J3, @J3(constants.%B3)> [symbolic = %J3.type.loc106_35.2 (constants.%J3.type.4)]
// CHECK:STDOUT:       %B3.param: type = value_param runtime_param<invalid>
// CHECK:STDOUT:       %B3.loc106_16.1: type = bind_symbolic_name B3, 1, %B3.param [symbolic = %B3.loc106_16.2 (constants.%B3)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %impl_witness.loc106: <witness> = impl_witness (), @impl.26(constants.%A, constants.%B3) [symbolic = @impl.26.%impl_witness (constants.%impl_witness.19)]
// CHECK:STDOUT:     impl_decl @impl.27 [template] {
// CHECK:STDOUT:       %B4.patt.loc112_16.1: type = symbolic_binding_pattern B4, 1 [symbolic = %B4.patt.loc112_16.2 (constants.%B4.patt)]
// CHECK:STDOUT:       %B4.param_patt: type = value_param_pattern %B4.patt.loc112_16.1, runtime_param<invalid> [symbolic = %B4.patt.loc112_16.2 (constants.%B4.patt)]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %C4.ref: %C4.type = name_ref C4, file.%C4.decl [template = constants.%C4.generic]
// CHECK:STDOUT:       %A.ref: type = name_ref A, @C4.%A.loc72_10.1 [symbolic = %A (constants.%A)]
// CHECK:STDOUT:       %C4.loc112_31.1: type = class_type @C4, @C4(constants.%A) [symbolic = %C4.loc112_31.2 (constants.%C4)]
// CHECK:STDOUT:       %J4.ref: %J4.type.1 = name_ref J4, file.%J4.decl [template = constants.%J4.generic]
// CHECK:STDOUT:       %B4.ref: type = name_ref B4, %B4.loc112_16.1 [symbolic = %B4.loc112_16.2 (constants.%B4)]
// CHECK:STDOUT:       %J4.type.loc112_41.1: type = facet_type <@J4, @J4(constants.%B4)> [symbolic = %J4.type.loc112_41.2 (constants.%J4.type.4)]
// CHECK:STDOUT:       %B4.param: type = value_param runtime_param<invalid>
// CHECK:STDOUT:       %B4.loc112_16.1: type = bind_symbolic_name B4, 1, %B4.param [symbolic = %B4.loc112_16.2 (constants.%B4)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %impl_witness.loc112: <witness> = impl_witness (), @impl.27(constants.%A, constants.%B4) [symbolic = @impl.27.%impl_witness (constants.%impl_witness.20)]
// CHECK:STDOUT:     impl_decl @impl.28 [template] {
// CHECK:STDOUT:       %B4.patt.loc113_16.1: type = symbolic_binding_pattern B4, 1 [symbolic = %B4.patt.loc113_16.2 (constants.%B4.patt)]
// CHECK:STDOUT:       %B4.param_patt: type = value_param_pattern %B4.patt.loc113_16.1, runtime_param<invalid> [symbolic = %B4.patt.loc113_16.2 (constants.%B4.patt)]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %.loc113: type = specific_constant constants.%C4, @C4(constants.%A) [symbolic = %C4 (constants.%C4)]
// CHECK:STDOUT:       %Self.ref: type = name_ref Self, %.loc113 [symbolic = %C4 (constants.%C4)]
// CHECK:STDOUT:       %J4.ref: %J4.type.1 = name_ref J4, file.%J4.decl [template = constants.%J4.generic]
// CHECK:STDOUT:       %B4.ref: type = name_ref B4, %B4.loc113_16.1 [symbolic = %B4.loc113_16.2 (constants.%B4)]
// CHECK:STDOUT:       %J4.type.loc113_40.1: type = facet_type <@J4, @J4(constants.%B4)> [symbolic = %J4.type.loc113_40.2 (constants.%J4.type.4)]
// CHECK:STDOUT:       %B4.param: type = value_param runtime_param<invalid>
// CHECK:STDOUT:       %B4.loc113_16.1: type = bind_symbolic_name B4, 1, %B4.param [symbolic = %B4.loc113_16.2 (constants.%B4)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %impl_witness.loc113: <witness> = impl_witness (), @impl.28(constants.%A, constants.%B4) [symbolic = @impl.28.%impl_witness (constants.%impl_witness.21)]
// CHECK:STDOUT:     impl_decl @impl.29 [template] {
// CHECK:STDOUT:       %B5.patt.loc119_16.1: type = symbolic_binding_pattern B5, 1 [symbolic = %B5.patt.loc119_16.2 (constants.%B5.patt)]
// CHECK:STDOUT:       %B5.param_patt: type = value_param_pattern %B5.patt.loc119_16.1, runtime_param<invalid> [symbolic = %B5.patt.loc119_16.2 (constants.%B5.patt)]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %Self.ref: type = name_ref Self, constants.%C4 [symbolic = %C4 (constants.%C4)]
// CHECK:STDOUT:       %J5.ref: %J5.type.1 = name_ref J5, file.%J5.decl [template = constants.%J5.generic]
// CHECK:STDOUT:       %B5.ref: type = name_ref B5, %B5.loc119_16.1 [symbolic = %B5.loc119_16.2 (constants.%B5)]
// CHECK:STDOUT:       %J5.type.loc119_35.1: type = facet_type <@J5, @J5(constants.%B5)> [symbolic = %J5.type.loc119_35.2 (constants.%J5.type.4)]
// CHECK:STDOUT:       %B5.param: type = value_param runtime_param<invalid>
// CHECK:STDOUT:       %B5.loc119_16.1: type = bind_symbolic_name B5, 1, %B5.param [symbolic = %B5.loc119_16.2 (constants.%B5)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %impl_witness.loc119: <witness> = impl_witness (), @impl.29(constants.%A, constants.%B5) [symbolic = @impl.29.%impl_witness (constants.%impl_witness.22)]
// CHECK:STDOUT:     impl_decl @impl.30 [template] {
// CHECK:STDOUT:       %B5.patt.loc120_16.1: type = symbolic_binding_pattern B5, 1 [symbolic = %B5.patt.loc120_16.2 (constants.%B5.patt)]
// CHECK:STDOUT:       %B5.param_patt: type = value_param_pattern %B5.patt.loc120_16.1, runtime_param<invalid> [symbolic = %B5.patt.loc120_16.2 (constants.%B5.patt)]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %C4.ref: %C4.type = name_ref C4, file.%C4.decl [template = constants.%C4.generic]
// CHECK:STDOUT:       %A.ref: type = name_ref A, @C4.%A.loc72_10.1 [symbolic = %A (constants.%A)]
// CHECK:STDOUT:       %C4.loc120_31.1: type = class_type @C4, @C4(constants.%A) [symbolic = %C4.loc120_31.2 (constants.%C4)]
// CHECK:STDOUT:       %J5.ref: %J5.type.1 = name_ref J5, file.%J5.decl [template = constants.%J5.generic]
// CHECK:STDOUT:       %B5.ref: type = name_ref B5, %B5.loc120_16.1 [symbolic = %B5.loc120_16.2 (constants.%B5)]
// CHECK:STDOUT:       %J5.type.loc120_41.1: type = facet_type <@J5, @J5(constants.%B5)> [symbolic = %J5.type.loc120_41.2 (constants.%J5.type.4)]
// CHECK:STDOUT:       %B5.param: type = value_param runtime_param<invalid>
// CHECK:STDOUT:       %B5.loc120_16.1: type = bind_symbolic_name B5, 1, %B5.param [symbolic = %B5.loc120_16.2 (constants.%B5)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %impl_witness.loc120: <witness> = impl_witness (), @impl.30(constants.%A, constants.%B5) [symbolic = @impl.30.%impl_witness (constants.%impl_witness.23)]
// CHECK:STDOUT:     impl_decl @impl.31 [template] {
// CHECK:STDOUT:       %B6.patt.loc125_16.1: type = symbolic_binding_pattern B6, 1 [symbolic = %B6.patt.loc125_16.2 (constants.%B6.patt)]
// CHECK:STDOUT:       %B6.param_patt: type = value_param_pattern %B6.patt.loc125_16.1, runtime_param<invalid> [symbolic = %B6.patt.loc125_16.2 (constants.%B6.patt)]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %.loc125: type = specific_constant constants.%C4, @C4(constants.%A) [symbolic = %C4 (constants.%C4)]
// CHECK:STDOUT:       %Self.ref: type = name_ref Self, %.loc125 [symbolic = %C4 (constants.%C4)]
// CHECK:STDOUT:       %J6.ref: %J6.type.1 = name_ref J6, file.%J6.decl [template = constants.%J6.generic]
// CHECK:STDOUT:       %B6.ref: type = name_ref B6, %B6.loc125_16.1 [symbolic = %B6.loc125_16.2 (constants.%B6)]
// CHECK:STDOUT:       %J6.type.loc125_40.1: type = facet_type <@J6, @J6(constants.%B6)> [symbolic = %J6.type.loc125_40.2 (constants.%J6.type.4)]
// CHECK:STDOUT:       %B6.param: type = value_param runtime_param<invalid>
// CHECK:STDOUT:       %B6.loc125_16.1: type = bind_symbolic_name B6, 1, %B6.param [symbolic = %B6.loc125_16.2 (constants.%B6)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %impl_witness.loc125: <witness> = impl_witness (), @impl.31(constants.%A, constants.%B6) [symbolic = @impl.31.%impl_witness (constants.%impl_witness.24)]
// CHECK:STDOUT:     impl_decl @impl.32 [template] {
// CHECK:STDOUT:       %B6.patt.loc126_16.1: type = symbolic_binding_pattern B6, 1 [symbolic = %B6.patt.loc126_16.2 (constants.%B6.patt)]
// CHECK:STDOUT:       %B6.param_patt: type = value_param_pattern %B6.patt.loc126_16.1, runtime_param<invalid> [symbolic = %B6.patt.loc126_16.2 (constants.%B6.patt)]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %C4.ref: %C4.type = name_ref C4, file.%C4.decl [template = constants.%C4.generic]
// CHECK:STDOUT:       %A.ref: type = name_ref A, @C4.%A.loc72_10.1 [symbolic = %A (constants.%A)]
// CHECK:STDOUT:       %C4.loc126_31.1: type = class_type @C4, @C4(constants.%A) [symbolic = %C4.loc126_31.2 (constants.%C4)]
// CHECK:STDOUT:       %J6.ref: %J6.type.1 = name_ref J6, file.%J6.decl [template = constants.%J6.generic]
// CHECK:STDOUT:       %B6.ref: type = name_ref B6, %B6.loc126_16.1 [symbolic = %B6.loc126_16.2 (constants.%B6)]
// CHECK:STDOUT:       %J6.type.loc126_41.1: type = facet_type <@J6, @J6(constants.%B6)> [symbolic = %J6.type.loc126_41.2 (constants.%J6.type.4)]
// CHECK:STDOUT:       %B6.param: type = value_param runtime_param<invalid>
// CHECK:STDOUT:       %B6.loc126_16.1: type = bind_symbolic_name B6, 1, %B6.param [symbolic = %B6.loc126_16.2 (constants.%B6)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %impl_witness.loc126: <witness> = impl_witness (), @impl.32(constants.%A, constants.%B6) [symbolic = @impl.32.%impl_witness (constants.%impl_witness.25)]
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness %empty_struct_type [template = constants.%complete_type]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%C4
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J3(constants.%T3) {
// CHECK:STDOUT:   %T3.loc7_14.2 => constants.%T3
// CHECK:STDOUT:   %T3.patt.loc7_14.2 => constants.%T3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J3(%T3.loc7_14.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J4(constants.%T4) {
// CHECK:STDOUT:   %T4.loc8_14.2 => constants.%T4
// CHECK:STDOUT:   %T4.patt.loc8_14.2 => constants.%T4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J4(%T4.loc8_14.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J5(constants.%T5) {
// CHECK:STDOUT:   %T5.loc9_14.2 => constants.%T5
// CHECK:STDOUT:   %T5.patt.loc9_14.2 => constants.%T5
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J5(%T5.loc9_14.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J6(constants.%T6) {
// CHECK:STDOUT:   %T6.loc10_14.2 => constants.%T6
// CHECK:STDOUT:   %T6.patt.loc10_14.2 => constants.%T6
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J6(%T6.loc10_14.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J3(constants.%Z3) {
// CHECK:STDOUT:   %T3.loc7_14.2 => constants.%Z3
// CHECK:STDOUT:   %T3.patt.loc7_14.2 => constants.%Z3
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %J3.type => constants.%J3.type.3
// CHECK:STDOUT:   %Self.2 => constants.%Self.5
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.9(constants.%Z3) {
// CHECK:STDOUT:   %Z3.loc47_16.2 => constants.%Z3
// CHECK:STDOUT:   %Z3.patt.loc47_16.2 => constants.%Z3
// CHECK:STDOUT:   %J3.type.loc47_38.2 => constants.%J3.type.3
// CHECK:STDOUT:   %require_complete => constants.%require_complete.1
// CHECK:STDOUT:   %impl_witness => constants.%impl_witness.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J3(@impl.9.%Z3.loc47_16.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.9(%Z3.loc47_16.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.10(constants.%Z3) {
// CHECK:STDOUT:   %Z3.loc48_16.2 => constants.%Z3
// CHECK:STDOUT:   %Z3.patt.loc48_16.2 => constants.%Z3
// CHECK:STDOUT:   %J3.type.loc48_35.2 => constants.%J3.type.3
// CHECK:STDOUT:   %require_complete => constants.%require_complete.1
// CHECK:STDOUT:   %impl_witness => constants.%impl_witness.3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J3(@impl.10.%Z3.loc48_16.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.10(%Z3.loc48_16.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J4(constants.%Z4) {
// CHECK:STDOUT:   %T4.loc8_14.2 => constants.%Z4
// CHECK:STDOUT:   %T4.patt.loc8_14.2 => constants.%Z4
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %J4.type => constants.%J4.type.3
// CHECK:STDOUT:   %Self.2 => constants.%Self.6
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.11(constants.%Z4) {
// CHECK:STDOUT:   %Z4.loc54_16.2 => constants.%Z4
// CHECK:STDOUT:   %Z4.patt.loc54_16.2 => constants.%Z4
// CHECK:STDOUT:   %J4.type.loc54_38.2 => constants.%J4.type.3
// CHECK:STDOUT:   %require_complete => constants.%require_complete.2
// CHECK:STDOUT:   %impl_witness => constants.%impl_witness.4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J4(@impl.11.%Z4.loc54_16.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.11(%Z4.loc54_16.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.12(constants.%Z4) {
// CHECK:STDOUT:   %Z4.loc55_16.2 => constants.%Z4
// CHECK:STDOUT:   %Z4.patt.loc55_16.2 => constants.%Z4
// CHECK:STDOUT:   %J4.type.loc55_40.2 => constants.%J4.type.3
// CHECK:STDOUT:   %require_complete => constants.%require_complete.2
// CHECK:STDOUT:   %impl_witness => constants.%impl_witness.5
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J4(@impl.12.%Z4.loc55_16.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.12(%Z4.loc55_16.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J5(constants.%Z5) {
// CHECK:STDOUT:   %T5.loc9_14.2 => constants.%Z5
// CHECK:STDOUT:   %T5.patt.loc9_14.2 => constants.%Z5
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %J5.type => constants.%J5.type.3
// CHECK:STDOUT:   %Self.2 => constants.%Self.7
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.13(constants.%Z5) {
// CHECK:STDOUT:   %Z5.loc61_16.2 => constants.%Z5
// CHECK:STDOUT:   %Z5.patt.loc61_16.2 => constants.%Z5
// CHECK:STDOUT:   %J5.type.loc61_35.2 => constants.%J5.type.3
// CHECK:STDOUT:   %require_complete => constants.%require_complete.3
// CHECK:STDOUT:   %impl_witness => constants.%impl_witness.6
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J5(@impl.13.%Z5.loc61_16.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.13(%Z5.loc61_16.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.14(constants.%Z5) {
// CHECK:STDOUT:   %Z5.loc62_16.2 => constants.%Z5
// CHECK:STDOUT:   %Z5.patt.loc62_16.2 => constants.%Z5
// CHECK:STDOUT:   %J5.type.loc62_38.2 => constants.%J5.type.3
// CHECK:STDOUT:   %require_complete => constants.%require_complete.3
// CHECK:STDOUT:   %impl_witness => constants.%impl_witness.7
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J5(@impl.14.%Z5.loc62_16.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.14(%Z5.loc62_16.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J6(constants.%Z6) {
// CHECK:STDOUT:   %T6.loc10_14.2 => constants.%Z6
// CHECK:STDOUT:   %T6.patt.loc10_14.2 => constants.%Z6
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %J6.type => constants.%J6.type.3
// CHECK:STDOUT:   %Self.2 => constants.%Self.8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.15(constants.%Z6) {
// CHECK:STDOUT:   %Z6.loc68_16.2 => constants.%Z6
// CHECK:STDOUT:   %Z6.patt.loc68_16.2 => constants.%Z6
// CHECK:STDOUT:   %J6.type.loc68_40.2 => constants.%J6.type.3
// CHECK:STDOUT:   %require_complete => constants.%require_complete.4
// CHECK:STDOUT:   %impl_witness => constants.%impl_witness.8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J6(@impl.15.%Z6.loc68_16.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.15(%Z6.loc68_16.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.16(constants.%Z6) {
// CHECK:STDOUT:   %Z6.loc69_16.2 => constants.%Z6
// CHECK:STDOUT:   %Z6.patt.loc69_16.2 => constants.%Z6
// CHECK:STDOUT:   %J6.type.loc69_38.2 => constants.%J6.type.3
// CHECK:STDOUT:   %require_complete => constants.%require_complete.4
// CHECK:STDOUT:   %impl_witness => constants.%impl_witness.9
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J6(@impl.16.%Z6.loc69_16.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.16(%Z6.loc69_16.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C4(constants.%A) {
// CHECK:STDOUT:   %A.loc72_10.2 => constants.%A
// CHECK:STDOUT:   %A.patt.loc72_10.2 => constants.%A
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.17(constants.%A) {
// CHECK:STDOUT:   %A => constants.%A
// CHECK:STDOUT:   %C4.loc77_12.2 => constants.%C4
// CHECK:STDOUT:   %impl_witness => constants.%impl_witness.10
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C4(@impl.17.%A) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.17(%A) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.18(constants.%A) {
// CHECK:STDOUT:   %A => constants.%A
// CHECK:STDOUT:   %C4 => constants.%C4
// CHECK:STDOUT:   %impl_witness => constants.%impl_witness.11
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C4(@impl.18.%A) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.18(%A) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.19(constants.%A) {
// CHECK:STDOUT:   %A => constants.%A
// CHECK:STDOUT:   %C4.loc84_12.2 => constants.%C4
// CHECK:STDOUT:   %impl_witness => constants.%impl_witness.12
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C4(@impl.19.%A) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.19(%A) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.20(constants.%A) {
// CHECK:STDOUT:   %A => constants.%A
// CHECK:STDOUT:   %C4 => constants.%C4
// CHECK:STDOUT:   %impl_witness => constants.%impl_witness.13
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C4(@impl.20.%A) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.20(%A) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.21(constants.%A) {
// CHECK:STDOUT:   %A => constants.%A
// CHECK:STDOUT:   %C4 => constants.%C4
// CHECK:STDOUT:   %impl_witness => constants.%impl_witness.14
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C4(@impl.21.%A) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.21(%A) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.22(constants.%A) {
// CHECK:STDOUT:   %A => constants.%A
// CHECK:STDOUT:   %C4.loc92_12.2 => constants.%C4
// CHECK:STDOUT:   %impl_witness => constants.%impl_witness.15
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C4(@impl.22.%A) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.22(%A) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.23(constants.%A) {
// CHECK:STDOUT:   %A => constants.%A
// CHECK:STDOUT:   %C4 => constants.%C4
// CHECK:STDOUT:   %impl_witness => constants.%impl_witness.16
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C4(@impl.23.%A) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.23(%A) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.24(constants.%A) {
// CHECK:STDOUT:   %A => constants.%A
// CHECK:STDOUT:   %C4.loc99_12.2 => constants.%C4
// CHECK:STDOUT:   %impl_witness => constants.%impl_witness.17
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C4(@impl.24.%A) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.24(%A) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J3(constants.%B3) {
// CHECK:STDOUT:   %T3.loc7_14.2 => constants.%B3
// CHECK:STDOUT:   %T3.patt.loc7_14.2 => constants.%B3
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %J3.type => constants.%J3.type.4
// CHECK:STDOUT:   %Self.2 => constants.%Self.5
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.25(constants.%A, constants.%B3) {
// CHECK:STDOUT:   %B3.loc105_16.2 => constants.%B3
// CHECK:STDOUT:   %B3.patt.loc105_16.2 => constants.%B3
// CHECK:STDOUT:   %A => constants.%A
// CHECK:STDOUT:   %C4.loc105_31.2 => constants.%C4
// CHECK:STDOUT:   %J3.type.loc105_41.2 => constants.%J3.type.4
// CHECK:STDOUT:   %require_complete => constants.%require_complete.5
// CHECK:STDOUT:   %impl_witness => constants.%impl_witness.18
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C4(@impl.25.%A) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J3(@impl.25.%B3.loc105_16.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.25(%A, %B3.loc105_16.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.26(constants.%A, constants.%B3) {
// CHECK:STDOUT:   %B3.loc106_16.2 => constants.%B3
// CHECK:STDOUT:   %B3.patt.loc106_16.2 => constants.%B3
// CHECK:STDOUT:   %A => constants.%A
// CHECK:STDOUT:   %C4 => constants.%C4
// CHECK:STDOUT:   %J3.type.loc106_35.2 => constants.%J3.type.4
// CHECK:STDOUT:   %require_complete => constants.%require_complete.5
// CHECK:STDOUT:   %impl_witness => constants.%impl_witness.19
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C4(@impl.26.%A) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J3(@impl.26.%B3.loc106_16.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.26(%A, %B3.loc106_16.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J4(constants.%B4) {
// CHECK:STDOUT:   %T4.loc8_14.2 => constants.%B4
// CHECK:STDOUT:   %T4.patt.loc8_14.2 => constants.%B4
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %J4.type => constants.%J4.type.4
// CHECK:STDOUT:   %Self.2 => constants.%Self.6
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.27(constants.%A, constants.%B4) {
// CHECK:STDOUT:   %B4.loc112_16.2 => constants.%B4
// CHECK:STDOUT:   %B4.patt.loc112_16.2 => constants.%B4
// CHECK:STDOUT:   %A => constants.%A
// CHECK:STDOUT:   %C4.loc112_31.2 => constants.%C4
// CHECK:STDOUT:   %J4.type.loc112_41.2 => constants.%J4.type.4
// CHECK:STDOUT:   %require_complete => constants.%require_complete.6
// CHECK:STDOUT:   %impl_witness => constants.%impl_witness.20
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C4(@impl.27.%A) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J4(@impl.27.%B4.loc112_16.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.27(%A, %B4.loc112_16.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.28(constants.%A, constants.%B4) {
// CHECK:STDOUT:   %B4.loc113_16.2 => constants.%B4
// CHECK:STDOUT:   %B4.patt.loc113_16.2 => constants.%B4
// CHECK:STDOUT:   %A => constants.%A
// CHECK:STDOUT:   %C4 => constants.%C4
// CHECK:STDOUT:   %J4.type.loc113_40.2 => constants.%J4.type.4
// CHECK:STDOUT:   %require_complete => constants.%require_complete.6
// CHECK:STDOUT:   %impl_witness => constants.%impl_witness.21
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C4(@impl.28.%A) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J4(@impl.28.%B4.loc113_16.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.28(%A, %B4.loc113_16.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J5(constants.%B5) {
// CHECK:STDOUT:   %T5.loc9_14.2 => constants.%B5
// CHECK:STDOUT:   %T5.patt.loc9_14.2 => constants.%B5
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %J5.type => constants.%J5.type.4
// CHECK:STDOUT:   %Self.2 => constants.%Self.7
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.29(constants.%A, constants.%B5) {
// CHECK:STDOUT:   %B5.loc119_16.2 => constants.%B5
// CHECK:STDOUT:   %B5.patt.loc119_16.2 => constants.%B5
// CHECK:STDOUT:   %A => constants.%A
// CHECK:STDOUT:   %C4 => constants.%C4
// CHECK:STDOUT:   %J5.type.loc119_35.2 => constants.%J5.type.4
// CHECK:STDOUT:   %require_complete => constants.%require_complete.7
// CHECK:STDOUT:   %impl_witness => constants.%impl_witness.22
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C4(@impl.29.%A) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J5(@impl.29.%B5.loc119_16.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.29(%A, %B5.loc119_16.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.30(constants.%A, constants.%B5) {
// CHECK:STDOUT:   %B5.loc120_16.2 => constants.%B5
// CHECK:STDOUT:   %B5.patt.loc120_16.2 => constants.%B5
// CHECK:STDOUT:   %A => constants.%A
// CHECK:STDOUT:   %C4.loc120_31.2 => constants.%C4
// CHECK:STDOUT:   %J5.type.loc120_41.2 => constants.%J5.type.4
// CHECK:STDOUT:   %require_complete => constants.%require_complete.7
// CHECK:STDOUT:   %impl_witness => constants.%impl_witness.23
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C4(@impl.30.%A) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J5(@impl.30.%B5.loc120_16.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.30(%A, %B5.loc120_16.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J6(constants.%B6) {
// CHECK:STDOUT:   %T6.loc10_14.2 => constants.%B6
// CHECK:STDOUT:   %T6.patt.loc10_14.2 => constants.%B6
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %J6.type => constants.%J6.type.4
// CHECK:STDOUT:   %Self.2 => constants.%Self.8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.31(constants.%A, constants.%B6) {
// CHECK:STDOUT:   %B6.loc125_16.2 => constants.%B6
// CHECK:STDOUT:   %B6.patt.loc125_16.2 => constants.%B6
// CHECK:STDOUT:   %A => constants.%A
// CHECK:STDOUT:   %C4 => constants.%C4
// CHECK:STDOUT:   %J6.type.loc125_40.2 => constants.%J6.type.4
// CHECK:STDOUT:   %require_complete => constants.%require_complete.8
// CHECK:STDOUT:   %impl_witness => constants.%impl_witness.24
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C4(@impl.31.%A) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J6(@impl.31.%B6.loc125_16.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.31(%A, %B6.loc125_16.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.32(constants.%A, constants.%B6) {
// CHECK:STDOUT:   %B6.loc126_16.2 => constants.%B6
// CHECK:STDOUT:   %B6.patt.loc126_16.2 => constants.%B6
// CHECK:STDOUT:   %A => constants.%A
// CHECK:STDOUT:   %C4.loc126_31.2 => constants.%C4
// CHECK:STDOUT:   %J6.type.loc126_41.2 => constants.%J6.type.4
// CHECK:STDOUT:   %require_complete => constants.%require_complete.8
// CHECK:STDOUT:   %impl_witness => constants.%impl_witness.25
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C4(@impl.32.%A) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J6(@impl.32.%B6.loc126_16.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.32(%A, %B6.loc126_16.2) {}
// CHECK:STDOUT:
