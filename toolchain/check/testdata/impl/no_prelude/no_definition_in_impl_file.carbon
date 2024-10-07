// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/impl/no_prelude/no_definition_in_impl_file.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/impl/no_prelude/no_definition_in_impl_file.carbon

// --- decl_in_api_definition_in_impl.carbon

library "[[@TEST_NAME]]";

interface A {};

impl () as A;

// --- decl_in_api_definition_in_impl.impl.carbon

impl library "[[@TEST_NAME]]";

impl () as A;

impl () as A {}

// --- use_decl_in_api.carbon

library "[[@TEST_NAME]]";

// --- use_decl_in_api.impl.carbon

impl library "[[@TEST_NAME]]";

import library "decl_in_api_definition_in_impl";

// --- decl_only_in_api.carbon

library "[[@TEST_NAME]]";

interface B {};

impl () as B;

// --- decl_only_in_api.impl.carbon

impl library "[[@TEST_NAME]]";

// --- decl_in_api_decl_in_impl.carbon

library "[[@TEST_NAME]]";

interface C {};

impl () as C;

// --- fail_decl_in_api_decl_in_impl.impl.carbon

impl library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_decl_in_api_decl_in_impl.impl.carbon:[[@LINE+4]]:1: error: no definition found for declaration in impl file
// CHECK:STDERR: impl () as C;
// CHECK:STDERR: ^~~~~~~~~~~~~
// CHECK:STDERR:
impl () as C;

// --- decl_only_in_impl.carbon

library "[[@TEST_NAME]]";

// --- fail_decl_only_in_impl.impl.carbon

impl library "[[@TEST_NAME]]";

interface D {};

// CHECK:STDERR: fail_decl_only_in_impl.impl.carbon:[[@LINE+3]]:1: error: no definition found for declaration in impl file
// CHECK:STDERR: impl () as D;
// CHECK:STDERR: ^~~~~~~~~~~~~
impl () as D;

// CHECK:STDOUT: --- decl_in_api_definition_in_impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %A.type: type = interface_type @A [template]
// CHECK:STDOUT:   %Self: %A.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .A = %A.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %A.decl: type = interface_decl @A [template = constants.%A.type] {} {}
// CHECK:STDOUT:   impl_decl @impl [template] {} {
// CHECK:STDOUT:     %.loc6_7.1: %.1 = tuple_literal ()
// CHECK:STDOUT:     %.loc6_7.2: type = converted %.loc6_7.1, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT:     %A.ref: type = name_ref A, file.%A.decl [template = constants.%A.type]
// CHECK:STDOUT:   }
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
// CHECK:STDOUT: impl @impl: %.loc6_7.2 as %A.ref;
// CHECK:STDOUT:
// CHECK:STDOUT: --- decl_in_api_definition_in_impl.impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %A.type: type = interface_type @A [template]
// CHECK:STDOUT:   %Self: %A.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %.2: <witness> = interface_witness () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %import_ref.1: type = import_ref Main//decl_in_api_definition_in_impl, inst+1, loaded [template = constants.%A.type]
// CHECK:STDOUT:   %import_ref.2 = import_ref Main//decl_in_api_definition_in_impl, inst+3, unloaded
// CHECK:STDOUT:   %import_ref.3: type = import_ref Main//decl_in_api_definition_in_impl, inst+7, loaded [template = constants.%.1]
// CHECK:STDOUT:   %import_ref.4: type = import_ref Main//decl_in_api_definition_in_impl, inst+8, loaded [template = constants.%A.type]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .A = imports.%import_ref.1
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %default.import.loc2_6.1 = import <invalid>
// CHECK:STDOUT:   %default.import.loc2_6.2 = import <invalid>
// CHECK:STDOUT:   impl_decl @impl [template] {} {
// CHECK:STDOUT:     %.loc4_7.1: %.1 = tuple_literal ()
// CHECK:STDOUT:     %.loc4_7.2: type = converted %.loc4_7.1, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT:     %A.ref.loc4: type = name_ref A, imports.%import_ref.1 [template = constants.%A.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @impl [template] {} {
// CHECK:STDOUT:     %.loc6_7.1: %.1 = tuple_literal ()
// CHECK:STDOUT:     %.loc6_7.2: type = converted %.loc6_7.1, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT:     %A.ref.loc6: type = name_ref A, imports.%import_ref.1 [template = constants.%A.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @A {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%import_ref.2
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl: imports.%import_ref.3 as imports.%import_ref.4 {
// CHECK:STDOUT:   %.loc6_14: <witness> = interface_witness () [template = constants.%.2]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = %.loc6_14
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- use_decl_in_api.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- use_decl_in_api.impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %A.type: type = interface_type @A [template]
// CHECK:STDOUT:   %Self: %A.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %import_ref.1 = import_ref Main//decl_in_api_definition_in_impl, inst+1, unloaded
// CHECK:STDOUT:   %import_ref.2 = import_ref Main//decl_in_api_definition_in_impl, inst+3, unloaded
// CHECK:STDOUT:   %import_ref.3: type = import_ref Main//decl_in_api_definition_in_impl, inst+7, loaded [template = constants.%.1]
// CHECK:STDOUT:   %import_ref.4: type = import_ref Main//decl_in_api_definition_in_impl, inst+8, loaded [template = constants.%A.type]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .A = imports.%import_ref.1
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %default.import.loc2_6.1 = import <invalid>
// CHECK:STDOUT:   %default.import.loc2_6.2 = import <invalid>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @A {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%import_ref.2
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl: imports.%import_ref.3 as imports.%import_ref.4;
// CHECK:STDOUT:
// CHECK:STDOUT: --- decl_only_in_api.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %B.type: type = interface_type @B [template]
// CHECK:STDOUT:   %Self: %B.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .B = %B.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %B.decl: type = interface_decl @B [template = constants.%B.type] {} {}
// CHECK:STDOUT:   impl_decl @impl [template] {} {
// CHECK:STDOUT:     %.loc6_7.1: %.1 = tuple_literal ()
// CHECK:STDOUT:     %.loc6_7.2: type = converted %.loc6_7.1, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT:     %B.ref: type = name_ref B, file.%B.decl [template = constants.%B.type]
// CHECK:STDOUT:   }
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
// CHECK:STDOUT: impl @impl: %.loc6_7.2 as %B.ref;
// CHECK:STDOUT:
// CHECK:STDOUT: --- decl_only_in_api.impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %B.type: type = interface_type @B [template]
// CHECK:STDOUT:   %Self: %B.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %import_ref.1 = import_ref Main//decl_only_in_api, inst+1, unloaded
// CHECK:STDOUT:   %import_ref.2 = import_ref Main//decl_only_in_api, inst+3, unloaded
// CHECK:STDOUT:   %import_ref.3: type = import_ref Main//decl_only_in_api, inst+7, loaded [template = constants.%.1]
// CHECK:STDOUT:   %import_ref.4: type = import_ref Main//decl_only_in_api, inst+8, loaded [template = constants.%B.type]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .B = imports.%import_ref.1
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %default.import.loc2_6.1 = import <invalid>
// CHECK:STDOUT:   %default.import.loc2_6.2 = import <invalid>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @B {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%import_ref.2
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl: imports.%import_ref.3 as imports.%import_ref.4;
// CHECK:STDOUT:
// CHECK:STDOUT: --- decl_in_api_decl_in_impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C.type: type = interface_type @C [template]
// CHECK:STDOUT:   %Self: %C.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = interface_decl @C [template = constants.%C.type] {} {}
// CHECK:STDOUT:   impl_decl @impl [template] {} {
// CHECK:STDOUT:     %.loc6_7.1: %.1 = tuple_literal ()
// CHECK:STDOUT:     %.loc6_7.2: type = converted %.loc6_7.1, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [template = constants.%C.type]
// CHECK:STDOUT:   }
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
// CHECK:STDOUT: impl @impl: %.loc6_7.2 as %C.ref;
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_decl_in_api_decl_in_impl.impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C.type: type = interface_type @C [template]
// CHECK:STDOUT:   %Self: %C.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %import_ref.1: type = import_ref Main//decl_in_api_decl_in_impl, inst+1, loaded [template = constants.%C.type]
// CHECK:STDOUT:   %import_ref.2 = import_ref Main//decl_in_api_decl_in_impl, inst+3, unloaded
// CHECK:STDOUT:   %import_ref.3: type = import_ref Main//decl_in_api_decl_in_impl, inst+7, loaded [template = constants.%.1]
// CHECK:STDOUT:   %import_ref.4: type = import_ref Main//decl_in_api_decl_in_impl, inst+8, loaded [template = constants.%C.type]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .C = imports.%import_ref.1
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %default.import.loc2_6.1 = import <invalid>
// CHECK:STDOUT:   %default.import.loc2_6.2 = import <invalid>
// CHECK:STDOUT:   impl_decl @impl [template] {} {
// CHECK:STDOUT:     %.loc8_7.1: %.1 = tuple_literal ()
// CHECK:STDOUT:     %.loc8_7.2: type = converted %.loc8_7.1, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT:     %C.ref: type = name_ref C, imports.%import_ref.1 [template = constants.%C.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @C {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%import_ref.2
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl: imports.%import_ref.3 as imports.%import_ref.4;
// CHECK:STDOUT:
// CHECK:STDOUT: --- decl_only_in_impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_decl_only_in_impl.impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %D.type: type = interface_type @D [template]
// CHECK:STDOUT:   %Self: %D.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .D = %D.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %default.import.loc2_6.1 = import <invalid>
// CHECK:STDOUT:   %default.import.loc2_6.2 = import <invalid>
// CHECK:STDOUT:   %D.decl: type = interface_decl @D [template = constants.%D.type] {} {}
// CHECK:STDOUT:   impl_decl @impl [template] {} {
// CHECK:STDOUT:     %.loc9_7.1: %.1 = tuple_literal ()
// CHECK:STDOUT:     %.loc9_7.2: type = converted %.loc9_7.1, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [template = constants.%D.type]
// CHECK:STDOUT:   }
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
// CHECK:STDOUT: impl @impl: %.loc9_7.2 as %D.ref;
// CHECK:STDOUT:
