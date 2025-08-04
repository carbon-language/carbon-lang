// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
// TODO: Add ranges and switch to "--dump-sem-ir-ranges=only".
// EXTRA-ARGS: --dump-sem-ir-ranges=if-present
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/let/associated_const.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/let/associated_const.carbon

interface Xyz {
	let hello:! String;
	let helloWithInitializer:! String = "Hello world";
}

// CHECK:STDOUT: --- associated_const.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Xyz.type: type = facet_type <@Xyz> [concrete]
// CHECK:STDOUT:   %Self: %Xyz.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %Xyz.assoc_type: type = assoc_entity_type @Xyz [concrete]
// CHECK:STDOUT:   %assoc0: %Xyz.assoc_type = assoc_entity element0, @Xyz.%hello [concrete]
// CHECK:STDOUT:   %assoc1: %Xyz.assoc_type = assoc_entity element1, @Xyz.%helloWithInitializer [concrete]
// CHECK:STDOUT:   %str: String = string_literal "Hello world" [concrete]
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
// CHECK:STDOUT:     .Xyz = %Xyz.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Xyz.decl: type = interface_decl @Xyz [concrete = constants.%Xyz.type] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Xyz {
// CHECK:STDOUT:   %Self: %Xyz.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   %hello: String = assoc_const_decl @hello [concrete] {
// CHECK:STDOUT:     %assoc0: %Xyz.assoc_type = assoc_entity element0, @Xyz.%hello [concrete = constants.%assoc0]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %helloWithInitializer: String = assoc_const_decl @helloWithInitializer [concrete] {
// CHECK:STDOUT:     %assoc1: %Xyz.assoc_type = assoc_entity element1, @Xyz.%helloWithInitializer [concrete = constants.%assoc1]
// CHECK:STDOUT:     %str: String = string_literal "Hello world" [concrete = constants.%str]
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .hello = @hello.%assoc0
// CHECK:STDOUT:   .helloWithInitializer = @helloWithInitializer.%assoc1
// CHECK:STDOUT:   witness = (%hello, %helloWithInitializer)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic assoc_const @hello(@Xyz.%Self: %Xyz.type) {
// CHECK:STDOUT:   assoc_const hello:! String;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic assoc_const @helloWithInitializer(@Xyz.%Self: %Xyz.type) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   assoc_const helloWithInitializer:! String = %str;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @hello(constants.%Self) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @helloWithInitializer(constants.%Self) {}
// CHECK:STDOUT:
