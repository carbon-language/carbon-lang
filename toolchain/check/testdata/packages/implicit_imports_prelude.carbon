// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// This is just checking an implicit import behavior with the prelude present.
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/packages/implicit_imports_prelude.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/packages/implicit_imports_prelude.carbon

// --- lib.carbon

library "[[@TEST_NAME]]";

var a: i32 = 0;

// --- lib.impl.carbon

impl library "[[@TEST_NAME]]";

var b: i32 = a;

// CHECK:STDOUT: --- lib.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %.1 [template]
// CHECK:STDOUT:   %.2: Core.IntLiteral = int_value 0 [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %Convert.type.14: type = fn_type @Convert.2, @impl.1(%.1) [template]
// CHECK:STDOUT:   %Convert.14: %Convert.type.14 = struct_value () [template]
// CHECK:STDOUT:   %.26: <witness> = interface_witness (%Convert.14) [template]
// CHECK:STDOUT:   %.27: <bound method> = bound_method %.2, %Convert.14 [template]
// CHECK:STDOUT:   %.28: <specific function> = specific_function %.27, @Convert.2(%.1) [template]
// CHECK:STDOUT:   %.29: %i32 = int_value 0 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int = %import_ref.1
// CHECK:STDOUT:     .ImplicitAs = %import_ref.2
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .a = %a
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %.loc4_8.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:   %int.make_type_signed: init type = call constants.%Int(%.loc4_8.1) [template = constants.%i32]
// CHECK:STDOUT:   %.loc4_8.2: type = value_of_initializer %int.make_type_signed [template = constants.%i32]
// CHECK:STDOUT:   %.loc4_8.3: type = converted %int.make_type_signed, %.loc4_8.2 [template = constants.%i32]
// CHECK:STDOUT:   %a.var: ref %i32 = var a
// CHECK:STDOUT:   %a: ref %i32 = bind_name a, %a.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc4_14: Core.IntLiteral = int_value 0 [template = constants.%.2]
// CHECK:STDOUT:   %.loc4_15.1: %Convert.type.2 = interface_witness_access constants.%.26, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc4_15.2: <bound method> = bound_method %.loc4_14, %.loc4_15.1 [template = constants.%.27]
// CHECK:STDOUT:   %.loc4_15.3: <specific function> = specific_function %.loc4_15.2, @Convert.2(constants.%.1) [template = constants.%.28]
// CHECK:STDOUT:   %int.convert_checked: init %i32 = call %.loc4_15.3(%.loc4_14) [template = constants.%.29]
// CHECK:STDOUT:   %.loc4_15.4: init %i32 = converted %.loc4_14, %int.convert_checked [template = constants.%.29]
// CHECK:STDOUT:   assign file.%a.var, %.loc4_15.4
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- lib.impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic]
// CHECK:STDOUT:   %iN: type = int_type signed, %N [symbolic]
// CHECK:STDOUT:   %Dest: type = bind_symbolic_name Dest, 0 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.2: type = facet_type <@ImplicitAs, @ImplicitAs(%Dest)> [symbolic]
// CHECK:STDOUT:   %Self.1: @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.2) = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %Dest.patt: type = symbolic_binding_pattern Dest, 0 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.3: type = facet_type <@ImplicitAs, @ImplicitAs(%iN)> [symbolic]
// CHECK:STDOUT:   %N.patt: Core.IntLiteral = symbolic_binding_pattern N, 0 [symbolic]
// CHECK:STDOUT:   %Self.2: %ImplicitAs.type.2 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %Convert.type.1: type = fn_type @Convert.1, @ImplicitAs(%Dest) [symbolic]
// CHECK:STDOUT:   %Convert.1: %Convert.type.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %.1: type = facet_access_type %Self.2 [symbolic]
// CHECK:STDOUT:   %.2: type = assoc_entity_type %ImplicitAs.type.2, %Convert.type.1 [symbolic]
// CHECK:STDOUT:   %.3: %.2 = assoc_entity element0, imports.%import_ref.8 [symbolic]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.2, @impl.1(%N) [symbolic]
// CHECK:STDOUT:   %Convert.2: %Convert.type.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %.4: <witness> = interface_witness (%Convert.2) [symbolic]
// CHECK:STDOUT:   %.5: type = facet_access_type %Self.1 [symbolic]
// CHECK:STDOUT:   %uN: type = int_type unsigned, %N [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.4: type = facet_type <@ImplicitAs, @ImplicitAs(%uN)> [symbolic]
// CHECK:STDOUT:   %Convert.type.3: type = fn_type @Convert.3, @impl.2(%N) [symbolic]
// CHECK:STDOUT:   %Convert.3: %Convert.type.3 = struct_value () [symbolic]
// CHECK:STDOUT:   %.6: <witness> = interface_witness (%Convert.3) [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.5: type = facet_type <@ImplicitAs, @ImplicitAs(Core.IntLiteral)> [template]
// CHECK:STDOUT:   %Convert.type.4: type = fn_type @Convert.4, @impl.3(%N) [symbolic]
// CHECK:STDOUT:   %Convert.4: %Convert.type.4 = struct_value () [symbolic]
// CHECK:STDOUT:   %.7: <witness> = interface_witness (%Convert.4) [symbolic]
// CHECK:STDOUT:   %Convert.type.5: type = fn_type @Convert.1, @ImplicitAs(Core.IntLiteral) [template]
// CHECK:STDOUT:   %Convert.5: %Convert.type.5 = struct_value () [template]
// CHECK:STDOUT:   %.8: type = assoc_entity_type %ImplicitAs.type.5, %Convert.type.5 [template]
// CHECK:STDOUT:   %.9: %.8 = assoc_entity element0, imports.%import_ref.15 [template]
// CHECK:STDOUT:   %Convert.type.6: type = fn_type @Convert.5, @impl.4(%N) [symbolic]
// CHECK:STDOUT:   %Convert.6: %Convert.type.6 = struct_value () [symbolic]
// CHECK:STDOUT:   %.10: <witness> = interface_witness (%Convert.6) [symbolic]
// CHECK:STDOUT:   %As.type.2: type = facet_type <@As, @As(%Dest)> [symbolic]
// CHECK:STDOUT:   %Self.3: @As.%As.type (%As.type.2) = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %As.type.3: type = facet_type <@As, @As(%iN)> [symbolic]
// CHECK:STDOUT:   %Self.4: %As.type.2 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %Convert.type.7: type = fn_type @Convert.6, @As(%Dest) [symbolic]
// CHECK:STDOUT:   %Convert.7: %Convert.type.7 = struct_value () [symbolic]
// CHECK:STDOUT:   %.11: type = facet_access_type %Self.4 [symbolic]
// CHECK:STDOUT:   %.12: type = assoc_entity_type %As.type.2, %Convert.type.7 [symbolic]
// CHECK:STDOUT:   %.13: %.12 = assoc_entity element0, imports.%import_ref.25 [symbolic]
// CHECK:STDOUT:   %Convert.type.8: type = fn_type @Convert.7, @impl.5(%N) [symbolic]
// CHECK:STDOUT:   %Convert.8: %Convert.type.8 = struct_value () [symbolic]
// CHECK:STDOUT:   %.14: <witness> = interface_witness (%Convert.8) [symbolic]
// CHECK:STDOUT:   %.15: type = facet_access_type %Self.3 [symbolic]
// CHECK:STDOUT:   %As.type.4: type = facet_type <@As, @As(%uN)> [symbolic]
// CHECK:STDOUT:   %Convert.type.9: type = fn_type @Convert.8, @impl.6(%N) [symbolic]
// CHECK:STDOUT:   %Convert.9: %Convert.type.9 = struct_value () [symbolic]
// CHECK:STDOUT:   %.16: <witness> = interface_witness (%Convert.9) [symbolic]
// CHECK:STDOUT:   %As.type.5: type = facet_type <@As, @As(Core.IntLiteral)> [template]
// CHECK:STDOUT:   %Convert.type.10: type = fn_type @Convert.9, @impl.7(%N) [symbolic]
// CHECK:STDOUT:   %Convert.10: %Convert.type.10 = struct_value () [symbolic]
// CHECK:STDOUT:   %.17: <witness> = interface_witness (%Convert.10) [symbolic]
// CHECK:STDOUT:   %Convert.type.11: type = fn_type @Convert.6, @As(Core.IntLiteral) [template]
// CHECK:STDOUT:   %Convert.11: %Convert.type.11 = struct_value () [template]
// CHECK:STDOUT:   %.18: type = assoc_entity_type %As.type.5, %Convert.type.11 [template]
// CHECK:STDOUT:   %.19: %.18 = assoc_entity element0, imports.%import_ref.32 [template]
// CHECK:STDOUT:   %Convert.type.12: type = fn_type @Convert.10, @impl.8(%N) [symbolic]
// CHECK:STDOUT:   %Convert.12: %Convert.type.12 = struct_value () [symbolic]
// CHECK:STDOUT:   %.20: <witness> = interface_witness (%Convert.12) [symbolic]
// CHECK:STDOUT:   %.21: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %.21 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %import_ref.1: ref %i32 = import_ref Main//lib, inst+22, loaded
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int = %import_ref.36
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.2 = import_ref Main//lib, inst+37, unloaded
// CHECK:STDOUT:   %import_ref.3 = import_ref Main//lib, inst+38, unloaded
// CHECK:STDOUT:   %import_ref.4 = import_ref Main//lib, inst+39, unloaded
// CHECK:STDOUT:   %import_ref.5: type = import_ref Main//lib, inst+84, loaded [template = Core.IntLiteral]
// CHECK:STDOUT:   %import_ref.6: type = import_ref Main//lib, inst+85, loaded [symbolic = @impl.1.%ImplicitAs.type (constants.%ImplicitAs.type.3)]
// CHECK:STDOUT:   %import_ref.7 = import_ref Main//lib, inst+86, unloaded
// CHECK:STDOUT:   %import_ref.8 = import_ref Main//lib, inst+55, unloaded
// CHECK:STDOUT:   %import_ref.9: type = import_ref Main//lib, inst+114, loaded [template = Core.IntLiteral]
// CHECK:STDOUT:   %import_ref.10: type = import_ref Main//lib, inst+115, loaded [symbolic = @impl.2.%ImplicitAs.type (constants.%ImplicitAs.type.4)]
// CHECK:STDOUT:   %import_ref.11 = import_ref Main//lib, inst+116, unloaded
// CHECK:STDOUT:   %import_ref.12: type = import_ref Main//lib, inst+143, loaded [symbolic = @impl.3.%iN (constants.%iN)]
// CHECK:STDOUT:   %import_ref.13: type = import_ref Main//lib, inst+144, loaded [template = constants.%ImplicitAs.type.5]
// CHECK:STDOUT:   %import_ref.14 = import_ref Main//lib, inst+145, unloaded
// CHECK:STDOUT:   %import_ref.16: type = import_ref Main//lib, inst+175, loaded [symbolic = @impl.4.%uN (constants.%uN)]
// CHECK:STDOUT:   %import_ref.17: type = import_ref Main//lib, inst+176, loaded [template = constants.%ImplicitAs.type.5]
// CHECK:STDOUT:   %import_ref.18 = import_ref Main//lib, inst+177, unloaded
// CHECK:STDOUT:   %import_ref.19 = import_ref Main//lib, inst+207, unloaded
// CHECK:STDOUT:   %import_ref.20 = import_ref Main//lib, inst+208, unloaded
// CHECK:STDOUT:   %import_ref.21 = import_ref Main//lib, inst+209, unloaded
// CHECK:STDOUT:   %import_ref.22: type = import_ref Main//lib, inst+215, loaded [template = Core.IntLiteral]
// CHECK:STDOUT:   %import_ref.23: type = import_ref Main//lib, inst+216, loaded [symbolic = @impl.5.%As.type (constants.%As.type.3)]
// CHECK:STDOUT:   %import_ref.24 = import_ref Main//lib, inst+217, unloaded
// CHECK:STDOUT:   %import_ref.25 = import_ref Main//lib, inst+233, unloaded
// CHECK:STDOUT:   %import_ref.26: type = import_ref Main//lib, inst+272, loaded [template = Core.IntLiteral]
// CHECK:STDOUT:   %import_ref.27: type = import_ref Main//lib, inst+273, loaded [symbolic = @impl.6.%As.type (constants.%As.type.4)]
// CHECK:STDOUT:   %import_ref.28 = import_ref Main//lib, inst+274, unloaded
// CHECK:STDOUT:   %import_ref.29: type = import_ref Main//lib, inst+301, loaded [symbolic = @impl.7.%iN (constants.%iN)]
// CHECK:STDOUT:   %import_ref.30: type = import_ref Main//lib, inst+302, loaded [template = constants.%As.type.5]
// CHECK:STDOUT:   %import_ref.31 = import_ref Main//lib, inst+303, unloaded
// CHECK:STDOUT:   %import_ref.33: type = import_ref Main//lib, inst+333, loaded [symbolic = @impl.8.%uN (constants.%uN)]
// CHECK:STDOUT:   %import_ref.34: type = import_ref Main//lib, inst+334, loaded [template = constants.%As.type.5]
// CHECK:STDOUT:   %import_ref.35 = import_ref Main//lib, inst+335, unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .a = imports.%import_ref.1
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .b = %b
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %default.import.loc2_6.1 = import <invalid>
// CHECK:STDOUT:   %default.import.loc2_6.2 = import <invalid>
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %.loc4_8.1: Core.IntLiteral = int_value 32 [template = constants.%.21]
// CHECK:STDOUT:   %int.make_type_signed: init type = call constants.%Int(%.loc4_8.1) [template = constants.%i32]
// CHECK:STDOUT:   %.loc4_8.2: type = value_of_initializer %int.make_type_signed [template = constants.%i32]
// CHECK:STDOUT:   %.loc4_8.3: type = converted %int.make_type_signed, %.loc4_8.2 [template = constants.%i32]
// CHECK:STDOUT:   %b.var: ref %i32 = var b
// CHECK:STDOUT:   %b: ref %i32 = bind_name b, %b.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @ImplicitAs(constants.%Dest: type) {
// CHECK:STDOUT:   %Dest: type = bind_symbolic_name Dest, 0 [symbolic = %Dest (constants.%Dest)]
// CHECK:STDOUT:   %Dest.patt: type = symbolic_binding_pattern Dest, 0 [symbolic = %Dest.patt (constants.%Dest.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(%Dest)> [symbolic = %ImplicitAs.type (constants.%ImplicitAs.type.2)]
// CHECK:STDOUT:   %Self: %ImplicitAs.type.2 = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self.2)]
// CHECK:STDOUT:   %Convert.type: type = fn_type @Convert.1, @ImplicitAs(%Dest) [symbolic = %Convert.type (constants.%Convert.type.1)]
// CHECK:STDOUT:   %Convert: @ImplicitAs.%Convert.type (%Convert.type.1) = struct_value () [symbolic = %Convert (constants.%Convert.1)]
// CHECK:STDOUT:   %.1: type = assoc_entity_type @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.2), @ImplicitAs.%Convert.type (%Convert.type.1) [symbolic = %.1 (constants.%.2)]
// CHECK:STDOUT:   %.2: @ImplicitAs.%.1 (%.2) = assoc_entity element0, imports.%import_ref.8 [symbolic = %.2 (constants.%.3)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%import_ref.2
// CHECK:STDOUT:     .Convert = imports.%import_ref.3
// CHECK:STDOUT:     witness = (imports.%import_ref.4)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @As(constants.%Dest: type) {
// CHECK:STDOUT:   %Dest: type = bind_symbolic_name Dest, 0 [symbolic = %Dest (constants.%Dest)]
// CHECK:STDOUT:   %Dest.patt: type = symbolic_binding_pattern Dest, 0 [symbolic = %Dest.patt (constants.%Dest.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %As.type: type = facet_type <@As, @As(%Dest)> [symbolic = %As.type (constants.%As.type.2)]
// CHECK:STDOUT:   %Self: %As.type.2 = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self.4)]
// CHECK:STDOUT:   %Convert.type: type = fn_type @Convert.6, @As(%Dest) [symbolic = %Convert.type (constants.%Convert.type.7)]
// CHECK:STDOUT:   %Convert: @As.%Convert.type (%Convert.type.7) = struct_value () [symbolic = %Convert (constants.%Convert.7)]
// CHECK:STDOUT:   %.1: type = assoc_entity_type @As.%As.type (%As.type.2), @As.%Convert.type (%Convert.type.7) [symbolic = %.1 (constants.%.12)]
// CHECK:STDOUT:   %.2: @As.%.1 (%.12) = assoc_entity element0, imports.%import_ref.25 [symbolic = %.2 (constants.%.13)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%import_ref.19
// CHECK:STDOUT:     .Convert = imports.%import_ref.20
// CHECK:STDOUT:     witness = (imports.%import_ref.21)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl.1(constants.%N: Core.IntLiteral) {
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:   %N.patt: Core.IntLiteral = symbolic_binding_pattern N, 0 [symbolic = %N.patt (constants.%N.patt)]
// CHECK:STDOUT:   %iN: type = int_type signed, %N [symbolic = %iN (constants.%iN)]
// CHECK:STDOUT:   %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(%iN)> [symbolic = %ImplicitAs.type (constants.%ImplicitAs.type.3)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Convert.type: type = fn_type @Convert.2, @impl.1(%N) [symbolic = %Convert.type (constants.%Convert.type.2)]
// CHECK:STDOUT:   %Convert: @impl.1.%Convert.type (%Convert.type.2) = struct_value () [symbolic = %Convert (constants.%Convert.2)]
// CHECK:STDOUT:   %.1: <witness> = interface_witness (%Convert) [symbolic = %.1 (constants.%.4)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: imports.%import_ref.5 as imports.%import_ref.6 {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = imports.%import_ref.7
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl.2(constants.%N: Core.IntLiteral) {
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:   %N.patt: Core.IntLiteral = symbolic_binding_pattern N, 0 [symbolic = %N.patt (constants.%N.patt)]
// CHECK:STDOUT:   %uN: type = int_type unsigned, %N [symbolic = %uN (constants.%uN)]
// CHECK:STDOUT:   %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(%uN)> [symbolic = %ImplicitAs.type (constants.%ImplicitAs.type.4)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Convert.type: type = fn_type @Convert.3, @impl.2(%N) [symbolic = %Convert.type (constants.%Convert.type.3)]
// CHECK:STDOUT:   %Convert: @impl.2.%Convert.type (%Convert.type.3) = struct_value () [symbolic = %Convert (constants.%Convert.3)]
// CHECK:STDOUT:   %.1: <witness> = interface_witness (%Convert) [symbolic = %.1 (constants.%.6)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: imports.%import_ref.9 as imports.%import_ref.10 {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = imports.%import_ref.11
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl.3(constants.%N: Core.IntLiteral) {
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:   %N.patt: Core.IntLiteral = symbolic_binding_pattern N, 0 [symbolic = %N.patt (constants.%N.patt)]
// CHECK:STDOUT:   %iN: type = int_type signed, %N [symbolic = %iN (constants.%iN)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Convert.type: type = fn_type @Convert.4, @impl.3(%N) [symbolic = %Convert.type (constants.%Convert.type.4)]
// CHECK:STDOUT:   %Convert: @impl.3.%Convert.type (%Convert.type.4) = struct_value () [symbolic = %Convert (constants.%Convert.4)]
// CHECK:STDOUT:   %.1: <witness> = interface_witness (%Convert) [symbolic = %.1 (constants.%.7)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: imports.%import_ref.12 as imports.%import_ref.13 {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = imports.%import_ref.14
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl.4(constants.%N: Core.IntLiteral) {
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:   %N.patt: Core.IntLiteral = symbolic_binding_pattern N, 0 [symbolic = %N.patt (constants.%N.patt)]
// CHECK:STDOUT:   %uN: type = int_type unsigned, %N [symbolic = %uN (constants.%uN)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Convert.type: type = fn_type @Convert.5, @impl.4(%N) [symbolic = %Convert.type (constants.%Convert.type.6)]
// CHECK:STDOUT:   %Convert: @impl.4.%Convert.type (%Convert.type.6) = struct_value () [symbolic = %Convert (constants.%Convert.6)]
// CHECK:STDOUT:   %.1: <witness> = interface_witness (%Convert) [symbolic = %.1 (constants.%.10)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: imports.%import_ref.16 as imports.%import_ref.17 {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = imports.%import_ref.18
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl.5(constants.%N: Core.IntLiteral) {
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:   %N.patt: Core.IntLiteral = symbolic_binding_pattern N, 0 [symbolic = %N.patt (constants.%N.patt)]
// CHECK:STDOUT:   %iN: type = int_type signed, %N [symbolic = %iN (constants.%iN)]
// CHECK:STDOUT:   %As.type: type = facet_type <@As, @As(%iN)> [symbolic = %As.type (constants.%As.type.3)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Convert.type: type = fn_type @Convert.7, @impl.5(%N) [symbolic = %Convert.type (constants.%Convert.type.8)]
// CHECK:STDOUT:   %Convert: @impl.5.%Convert.type (%Convert.type.8) = struct_value () [symbolic = %Convert (constants.%Convert.8)]
// CHECK:STDOUT:   %.1: <witness> = interface_witness (%Convert) [symbolic = %.1 (constants.%.14)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: imports.%import_ref.22 as imports.%import_ref.23 {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = imports.%import_ref.24
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl.6(constants.%N: Core.IntLiteral) {
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:   %N.patt: Core.IntLiteral = symbolic_binding_pattern N, 0 [symbolic = %N.patt (constants.%N.patt)]
// CHECK:STDOUT:   %uN: type = int_type unsigned, %N [symbolic = %uN (constants.%uN)]
// CHECK:STDOUT:   %As.type: type = facet_type <@As, @As(%uN)> [symbolic = %As.type (constants.%As.type.4)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Convert.type: type = fn_type @Convert.8, @impl.6(%N) [symbolic = %Convert.type (constants.%Convert.type.9)]
// CHECK:STDOUT:   %Convert: @impl.6.%Convert.type (%Convert.type.9) = struct_value () [symbolic = %Convert (constants.%Convert.9)]
// CHECK:STDOUT:   %.1: <witness> = interface_witness (%Convert) [symbolic = %.1 (constants.%.16)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: imports.%import_ref.26 as imports.%import_ref.27 {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = imports.%import_ref.28
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl.7(constants.%N: Core.IntLiteral) {
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:   %N.patt: Core.IntLiteral = symbolic_binding_pattern N, 0 [symbolic = %N.patt (constants.%N.patt)]
// CHECK:STDOUT:   %iN: type = int_type signed, %N [symbolic = %iN (constants.%iN)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Convert.type: type = fn_type @Convert.9, @impl.7(%N) [symbolic = %Convert.type (constants.%Convert.type.10)]
// CHECK:STDOUT:   %Convert: @impl.7.%Convert.type (%Convert.type.10) = struct_value () [symbolic = %Convert (constants.%Convert.10)]
// CHECK:STDOUT:   %.1: <witness> = interface_witness (%Convert) [symbolic = %.1 (constants.%.17)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: imports.%import_ref.29 as imports.%import_ref.30 {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = imports.%import_ref.31
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl.8(constants.%N: Core.IntLiteral) {
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:   %N.patt: Core.IntLiteral = symbolic_binding_pattern N, 0 [symbolic = %N.patt (constants.%N.patt)]
// CHECK:STDOUT:   %uN: type = int_type unsigned, %N [symbolic = %uN (constants.%uN)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Convert.type: type = fn_type @Convert.10, @impl.8(%N) [symbolic = %Convert.type (constants.%Convert.type.12)]
// CHECK:STDOUT:   %Convert: @impl.8.%Convert.type (%Convert.type.12) = struct_value () [symbolic = %Convert (constants.%Convert.12)]
// CHECK:STDOUT:   %.1: <witness> = interface_witness (%Convert) [symbolic = %.1 (constants.%.20)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: imports.%import_ref.33 as imports.%import_ref.34 {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = imports.%import_ref.35
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Convert.1(constants.%Dest: type, constants.%Self.1: @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.2)) {
// CHECK:STDOUT:   %Dest: type = bind_symbolic_name Dest, 0 [symbolic = %Dest (constants.%Dest)]
// CHECK:STDOUT:   %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(%Dest)> [symbolic = %ImplicitAs.type (constants.%ImplicitAs.type.2)]
// CHECK:STDOUT:   %Self: %ImplicitAs.type.2 = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self.2)]
// CHECK:STDOUT:   %.1: type = facet_access_type %Self [symbolic = %.1 (constants.%.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @Convert.1.%.1 (%.1)]() -> @Convert.1.%Dest (%Dest);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Convert.2(constants.%N: Core.IntLiteral) {
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:   %iN: type = int_type signed, %N [symbolic = %iN (constants.%iN)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: Core.IntLiteral]() -> @Convert.2.%iN (%iN) = "int.convert_checked";
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Convert.3(constants.%N: Core.IntLiteral) {
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:   %uN: type = int_type unsigned, %N [symbolic = %uN (constants.%uN)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: Core.IntLiteral]() -> @Convert.3.%uN (%uN) = "int.convert_checked";
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Convert.4(constants.%N: Core.IntLiteral) {
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:   %iN: type = int_type signed, %N [symbolic = %iN (constants.%iN)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @Convert.4.%iN (%iN)]() -> Core.IntLiteral = "int.convert_checked";
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Convert.5(constants.%N: Core.IntLiteral) {
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:   %uN: type = int_type unsigned, %N [symbolic = %uN (constants.%uN)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @Convert.5.%uN (%uN)]() -> Core.IntLiteral = "int.convert_checked";
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Convert.6(constants.%Dest: type, constants.%Self.3: @As.%As.type (%As.type.2)) {
// CHECK:STDOUT:   %Dest: type = bind_symbolic_name Dest, 0 [symbolic = %Dest (constants.%Dest)]
// CHECK:STDOUT:   %As.type: type = facet_type <@As, @As(%Dest)> [symbolic = %As.type (constants.%As.type.2)]
// CHECK:STDOUT:   %Self: %As.type.2 = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self.4)]
// CHECK:STDOUT:   %.1: type = facet_access_type %Self [symbolic = %.1 (constants.%.11)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @Convert.6.%.1 (%.11)]() -> @Convert.6.%Dest (%Dest);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Convert.7(constants.%N: Core.IntLiteral) {
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:   %iN: type = int_type signed, %N [symbolic = %iN (constants.%iN)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: Core.IntLiteral]() -> @Convert.7.%iN (%iN) = "int.convert_checked";
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Convert.8(constants.%N: Core.IntLiteral) {
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:   %uN: type = int_type unsigned, %N [symbolic = %uN (constants.%uN)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: Core.IntLiteral]() -> @Convert.8.%uN (%uN) = "int.convert_checked";
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Convert.9(constants.%N: Core.IntLiteral) {
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:   %iN: type = int_type signed, %N [symbolic = %iN (constants.%iN)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @Convert.9.%iN (%iN)]() -> Core.IntLiteral = "int.convert_checked";
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Convert.10(constants.%N: Core.IntLiteral) {
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:   %uN: type = int_type unsigned, %N [symbolic = %uN (constants.%uN)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @Convert.10.%uN (%uN)]() -> Core.IntLiteral = "int.convert_checked";
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: ref %i32 = name_ref a, imports.%import_ref.1
// CHECK:STDOUT:   %.loc4: %i32 = bind_value %a.ref
// CHECK:STDOUT:   assign file.%b.var, %.loc4
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(constants.%Dest) {
// CHECK:STDOUT:   %Dest => constants.%Dest
// CHECK:STDOUT:   %Dest.patt => constants.%Dest
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(constants.%iN) {
// CHECK:STDOUT:   %Dest => constants.%iN
// CHECK:STDOUT:   %Dest.patt => constants.%iN
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(%Dest) {
// CHECK:STDOUT:   %Dest => constants.%Dest
// CHECK:STDOUT:   %Dest.patt => constants.%Dest
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(@impl.1.%iN) {
// CHECK:STDOUT:   %Dest => constants.%iN
// CHECK:STDOUT:   %Dest.patt => constants.%iN
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.1(constants.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %N.patt => constants.%N
// CHECK:STDOUT:   %iN => constants.%iN
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.1(%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %N.patt => constants.%N
// CHECK:STDOUT:   %iN => constants.%iN
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(@Convert.1.%Dest) {
// CHECK:STDOUT:   %Dest => constants.%Dest
// CHECK:STDOUT:   %Dest.patt => constants.%Dest
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert.1(constants.%Dest, constants.%Self.1) {
// CHECK:STDOUT:   %Dest => constants.%Dest
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.2
// CHECK:STDOUT:   %Self => constants.%Self.1
// CHECK:STDOUT:   %.1 => constants.%.5
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert.2(constants.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %iN => constants.%iN
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(constants.%uN) {
// CHECK:STDOUT:   %Dest => constants.%uN
// CHECK:STDOUT:   %Dest.patt => constants.%uN
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(@impl.2.%uN) {
// CHECK:STDOUT:   %Dest => constants.%uN
// CHECK:STDOUT:   %Dest.patt => constants.%uN
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.2(constants.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %N.patt => constants.%N
// CHECK:STDOUT:   %uN => constants.%uN
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.2(%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %N.patt => constants.%N
// CHECK:STDOUT:   %uN => constants.%uN
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert.3(constants.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %uN => constants.%uN
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(Core.IntLiteral) {
// CHECK:STDOUT:   %Dest => Core.IntLiteral
// CHECK:STDOUT:   %Dest.patt => Core.IntLiteral
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.5
// CHECK:STDOUT:   %Self => constants.%Self.2
// CHECK:STDOUT:   %Convert.type => constants.%Convert.type.5
// CHECK:STDOUT:   %Convert => constants.%Convert.5
// CHECK:STDOUT:   %.1 => constants.%.8
// CHECK:STDOUT:   %.2 => constants.%.9
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.3(constants.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %N.patt => constants.%N
// CHECK:STDOUT:   %iN => constants.%iN
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.3(%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %N.patt => constants.%N
// CHECK:STDOUT:   %iN => constants.%iN
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert.4(constants.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %iN => constants.%iN
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.4(constants.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %N.patt => constants.%N
// CHECK:STDOUT:   %uN => constants.%uN
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.4(%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %N.patt => constants.%N
// CHECK:STDOUT:   %uN => constants.%uN
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert.5(constants.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %uN => constants.%uN
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @As(constants.%Dest) {
// CHECK:STDOUT:   %Dest => constants.%Dest
// CHECK:STDOUT:   %Dest.patt => constants.%Dest
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @As(constants.%iN) {
// CHECK:STDOUT:   %Dest => constants.%iN
// CHECK:STDOUT:   %Dest.patt => constants.%iN
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @As(%Dest) {
// CHECK:STDOUT:   %Dest => constants.%Dest
// CHECK:STDOUT:   %Dest.patt => constants.%Dest
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @As(@impl.5.%iN) {
// CHECK:STDOUT:   %Dest => constants.%iN
// CHECK:STDOUT:   %Dest.patt => constants.%iN
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.5(constants.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %N.patt => constants.%N
// CHECK:STDOUT:   %iN => constants.%iN
// CHECK:STDOUT:   %As.type => constants.%As.type.3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.5(%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %N.patt => constants.%N
// CHECK:STDOUT:   %iN => constants.%iN
// CHECK:STDOUT:   %As.type => constants.%As.type.3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @As(@Convert.6.%Dest) {
// CHECK:STDOUT:   %Dest => constants.%Dest
// CHECK:STDOUT:   %Dest.patt => constants.%Dest
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert.6(constants.%Dest, constants.%Self.3) {
// CHECK:STDOUT:   %Dest => constants.%Dest
// CHECK:STDOUT:   %As.type => constants.%As.type.2
// CHECK:STDOUT:   %Self => constants.%Self.3
// CHECK:STDOUT:   %.1 => constants.%.15
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert.7(constants.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %iN => constants.%iN
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @As(constants.%uN) {
// CHECK:STDOUT:   %Dest => constants.%uN
// CHECK:STDOUT:   %Dest.patt => constants.%uN
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @As(@impl.6.%uN) {
// CHECK:STDOUT:   %Dest => constants.%uN
// CHECK:STDOUT:   %Dest.patt => constants.%uN
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.6(constants.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %N.patt => constants.%N
// CHECK:STDOUT:   %uN => constants.%uN
// CHECK:STDOUT:   %As.type => constants.%As.type.4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.6(%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %N.patt => constants.%N
// CHECK:STDOUT:   %uN => constants.%uN
// CHECK:STDOUT:   %As.type => constants.%As.type.4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert.8(constants.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %uN => constants.%uN
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @As(Core.IntLiteral) {
// CHECK:STDOUT:   %Dest => Core.IntLiteral
// CHECK:STDOUT:   %Dest.patt => Core.IntLiteral
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %As.type => constants.%As.type.5
// CHECK:STDOUT:   %Self => constants.%Self.4
// CHECK:STDOUT:   %Convert.type => constants.%Convert.type.11
// CHECK:STDOUT:   %Convert => constants.%Convert.11
// CHECK:STDOUT:   %.1 => constants.%.18
// CHECK:STDOUT:   %.2 => constants.%.19
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.7(constants.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %N.patt => constants.%N
// CHECK:STDOUT:   %iN => constants.%iN
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.7(%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %N.patt => constants.%N
// CHECK:STDOUT:   %iN => constants.%iN
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert.9(constants.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %iN => constants.%iN
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.8(constants.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %N.patt => constants.%N
// CHECK:STDOUT:   %uN => constants.%uN
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.8(%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %N.patt => constants.%N
// CHECK:STDOUT:   %uN => constants.%uN
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert.10(constants.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %uN => constants.%uN
// CHECK:STDOUT: }
// CHECK:STDOUT:
