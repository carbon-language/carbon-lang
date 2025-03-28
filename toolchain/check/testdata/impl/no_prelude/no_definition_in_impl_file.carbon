// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/impl/no_prelude/no_definition_in_impl_file.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/impl/no_prelude/no_definition_in_impl_file.carbon

// --- fail_decl_in_api_definition_in_impl.carbon

library "[[@TEST_NAME]]";

interface A {}

// CHECK:STDERR: fail_decl_in_api_definition_in_impl.carbon:[[@LINE+4]]:1: error: impl declared but not defined [ImplMissingDefinition]
// CHECK:STDERR: impl () as A;
// CHECK:STDERR: ^~~~~~~~~~~~~
// CHECK:STDERR:
impl () as A;

// --- fail_decl_in_api_definition_in_impl.impl.carbon

impl library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_decl_in_api_definition_in_impl.impl.carbon:[[@LINE+4]]:1: error: redeclaration of imported impl [RedeclImportedImpl]
// CHECK:STDERR: impl () as A;
// CHECK:STDERR: ^~~~~~~~~~~~~
// CHECK:STDERR:
impl () as A;

// CHECK:STDERR: fail_decl_in_api_definition_in_impl.impl.carbon:[[@LINE+4]]:1: error: redeclaration of imported impl [RedeclImportedImpl]
// CHECK:STDERR: impl () as A {}
// CHECK:STDERR: ^~~~~~~~~~~~~~
// CHECK:STDERR:
impl () as A {}

// --- use_decl_in_api.carbon

library "[[@TEST_NAME]]";

// --- use_decl_in_api.impl.carbon

impl library "[[@TEST_NAME]]";

import library "decl_in_api_definition_in_impl";

// --- fail_decl_only_in_api.carbon

library "[[@TEST_NAME]]";

interface B {}

// CHECK:STDERR: fail_decl_only_in_api.carbon:[[@LINE+4]]:1: error: impl declared but not defined [ImplMissingDefinition]
// CHECK:STDERR: impl () as B;
// CHECK:STDERR: ^~~~~~~~~~~~~
// CHECK:STDERR:
impl () as B;

// --- decl_only_in_api.impl.carbon

impl library "[[@TEST_NAME]]";

// --- fail_decl_in_api_decl_in_impl.carbon

library "[[@TEST_NAME]]";

interface C {}

// CHECK:STDERR: fail_decl_in_api_decl_in_impl.carbon:[[@LINE+4]]:1: error: impl declared but not defined [ImplMissingDefinition]
// CHECK:STDERR: impl () as C;
// CHECK:STDERR: ^~~~~~~~~~~~~
// CHECK:STDERR:
impl () as C;

// --- fail_decl_in_api_decl_in_impl.impl.carbon

impl library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_decl_in_api_decl_in_impl.impl.carbon:[[@LINE+4]]:1: error: redeclaration of imported impl [RedeclImportedImpl]
// CHECK:STDERR: impl () as C;
// CHECK:STDERR: ^~~~~~~~~~~~~
// CHECK:STDERR:
impl () as C;

// --- decl_only_in_impl.carbon

library "[[@TEST_NAME]]";

// --- fail_decl_only_in_impl.impl.carbon

impl library "[[@TEST_NAME]]";

interface D {}

// CHECK:STDERR: fail_decl_only_in_impl.impl.carbon:[[@LINE+4]]:1: error: impl declared but not defined [ImplMissingDefinition]
// CHECK:STDERR: impl () as D;
// CHECK:STDERR: ^~~~~~~~~~~~~
// CHECK:STDERR:
impl () as D;

// CHECK:STDOUT: --- fail_decl_in_api_definition_in_impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %A.type: type = facet_type <@A> [concrete]
// CHECK:STDOUT:   %Self: %A.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .A = %A.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %A.decl: type = interface_decl @A [concrete = constants.%A.type] {} {}
// CHECK:STDOUT:   impl_decl @impl [concrete] {} {
// CHECK:STDOUT:     %.loc10_7.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc10_7.2: type = converted %.loc10_7.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %A.ref: type = name_ref A, file.%A.decl [concrete = constants.%A.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness () [concrete = constants.%impl_witness]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @A {
// CHECK:STDOUT:   %Self: %A.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl: %.loc10_7.2 as %A.ref;
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_decl_in_api_definition_in_impl.impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %A.type: type = facet_type <@A> [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %impl_witness.1bcebf.1: <witness> = impl_witness () [concrete]
// CHECK:STDOUT:   %impl_witness.1bcebf.2: <witness> = impl_witness () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.A: type = import_ref Main//decl_in_api_definition_in_impl, A, loaded [concrete = constants.%A.type]
// CHECK:STDOUT:   %Main.import_ref.b61 = import_ref Main//decl_in_api_definition_in_impl, inst16 [no loc], unloaded
// CHECK:STDOUT:   %Main.import_ref.e5c: type = import_ref Main//decl_in_api_definition_in_impl, loc10_7, loaded [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   %Main.import_ref.831: type = import_ref Main//decl_in_api_definition_in_impl, loc10_12, loaded [concrete = constants.%A.type]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .A = imports.%Main.A
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %default.import.loc2_46.1 = import <none>
// CHECK:STDOUT:   %default.import.loc2_46.2 = import <none>
// CHECK:STDOUT:   impl_decl @impl.064930.2 [concrete] {} {
// CHECK:STDOUT:     %.loc8_7.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc8_7.2: type = converted %.loc8_7.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %A.ref: type = name_ref A, imports.%Main.A [concrete = constants.%A.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness.loc8: <witness> = impl_witness () [concrete = constants.%impl_witness.1bcebf.1]
// CHECK:STDOUT:   impl_decl @impl.064930.3 [concrete] {} {
// CHECK:STDOUT:     %.loc14_7.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc14_7.2: type = converted %.loc14_7.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %A.ref: type = name_ref A, imports.%Main.A [concrete = constants.%A.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness.loc14: <witness> = impl_witness () [concrete = constants.%impl_witness.1bcebf.2]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @A [from "fail_decl_in_api_definition_in_impl.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.b61
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.064930.1: imports.%Main.import_ref.e5c as imports.%Main.import_ref.831 [from "fail_decl_in_api_definition_in_impl.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.064930.2: %.loc8_7.2 as %A.ref;
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.064930.3: %.loc14_7.2 as %A.ref {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = file.%impl_witness.loc14
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- use_decl_in_api.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- use_decl_in_api.impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.A = import_ref Main//decl_in_api_definition_in_impl, A, unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .A = imports.%Main.A
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %default.import.loc2_31.1 = import <none>
// CHECK:STDOUT:   %default.import.loc2_31.2 = import <none>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_decl_only_in_api.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %B.type: type = facet_type <@B> [concrete]
// CHECK:STDOUT:   %Self: %B.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .B = %B.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %B.decl: type = interface_decl @B [concrete = constants.%B.type] {} {}
// CHECK:STDOUT:   impl_decl @impl [concrete] {} {
// CHECK:STDOUT:     %.loc10_7.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc10_7.2: type = converted %.loc10_7.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %B.ref: type = name_ref B, file.%B.decl [concrete = constants.%B.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness () [concrete = constants.%impl_witness]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @B {
// CHECK:STDOUT:   %Self: %B.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl: %.loc10_7.2 as %B.ref;
// CHECK:STDOUT:
// CHECK:STDOUT: --- decl_only_in_api.impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %B.type: type = facet_type <@B> [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.B = import_ref Main//decl_only_in_api, B, unloaded
// CHECK:STDOUT:   %Main.import_ref.420 = import_ref Main//decl_only_in_api, inst16 [no loc], unloaded
// CHECK:STDOUT:   %Main.import_ref.e5c: type = import_ref Main//decl_only_in_api, loc10_7, loaded [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   %Main.import_ref.171: type = import_ref Main//decl_only_in_api, loc10_12, loaded [concrete = constants.%B.type]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .B = imports.%Main.B
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %default.import.loc2_32.1 = import <none>
// CHECK:STDOUT:   %default.import.loc2_32.2 = import <none>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @B [from "fail_decl_only_in_api.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.420
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl: imports.%Main.import_ref.e5c as imports.%Main.import_ref.171 [from "fail_decl_only_in_api.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_decl_in_api_decl_in_impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C.type: type = facet_type <@C> [concrete]
// CHECK:STDOUT:   %Self: %C.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = interface_decl @C [concrete = constants.%C.type] {} {}
// CHECK:STDOUT:   impl_decl @impl [concrete] {} {
// CHECK:STDOUT:     %.loc10_7.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc10_7.2: type = converted %.loc10_7.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness () [concrete = constants.%impl_witness]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @C {
// CHECK:STDOUT:   %Self: %C.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl: %.loc10_7.2 as %C.ref;
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_decl_in_api_decl_in_impl.impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C.type: type = facet_type <@C> [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.C: type = import_ref Main//decl_in_api_decl_in_impl, C, loaded [concrete = constants.%C.type]
// CHECK:STDOUT:   %Main.import_ref.721 = import_ref Main//decl_in_api_decl_in_impl, inst16 [no loc], unloaded
// CHECK:STDOUT:   %Main.import_ref.e5c: type = import_ref Main//decl_in_api_decl_in_impl, loc10_7, loaded [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   %Main.import_ref.653: type = import_ref Main//decl_in_api_decl_in_impl, loc10_12, loaded [concrete = constants.%C.type]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .C = imports.%Main.C
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %default.import.loc2_40.1 = import <none>
// CHECK:STDOUT:   %default.import.loc2_40.2 = import <none>
// CHECK:STDOUT:   impl_decl @impl.590fe9.2 [concrete] {} {
// CHECK:STDOUT:     %.loc8_7.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc8_7.2: type = converted %.loc8_7.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %C.ref: type = name_ref C, imports.%Main.C [concrete = constants.%C.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness () [concrete = constants.%impl_witness]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @C [from "fail_decl_in_api_decl_in_impl.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.721
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.590fe9.1: imports.%Main.import_ref.e5c as imports.%Main.import_ref.653 [from "fail_decl_in_api_decl_in_impl.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.590fe9.2: %.loc8_7.2 as %C.ref;
// CHECK:STDOUT:
// CHECK:STDOUT: --- decl_only_in_impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_decl_only_in_impl.impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %D.type: type = facet_type <@D> [concrete]
// CHECK:STDOUT:   %Self: %D.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .D = %D.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %default.import.loc2_33.1 = import <none>
// CHECK:STDOUT:   %default.import.loc2_33.2 = import <none>
// CHECK:STDOUT:   %D.decl: type = interface_decl @D [concrete = constants.%D.type] {} {}
// CHECK:STDOUT:   impl_decl @impl [concrete] {} {
// CHECK:STDOUT:     %.loc10_7.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc10_7.2: type = converted %.loc10_7.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [concrete = constants.%D.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness () [concrete = constants.%impl_witness]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @D {
// CHECK:STDOUT:   %Self: %D.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl: %.loc10_7.2 as %D.ref;
// CHECK:STDOUT:
