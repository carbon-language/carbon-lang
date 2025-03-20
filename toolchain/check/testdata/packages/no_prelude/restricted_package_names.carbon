// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/packages/no_prelude/restricted_package_names.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/packages/no_prelude/restricted_package_names.carbon

// --- lowercase_main.carbon

// The restriction is case-sensitive.
package main;

// --- fail_main.carbon

// CHECK:STDERR: fail_main.carbon:[[@LINE+4]]:1: error: `Main//default` must omit `package` declaration [ExplicitMainPackage]
// CHECK:STDERR: package Main;
// CHECK:STDERR: ^~~~~~~~~~~~~
// CHECK:STDERR:
package Main;

// --- fail_main_impl.carbon

// CHECK:STDERR: fail_main_impl.carbon:[[@LINE+4]]:1: error: `Main//default` must omit `package` declaration [ExplicitMainPackage]
// CHECK:STDERR: impl package Main;
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
impl package Main;

// --- fail_raw_main.carbon

// `Main` isn't a keyword, so this fails the same way.
// CHECK:STDERR: fail_raw_main.carbon:[[@LINE+4]]:1: error: `Main//default` must omit `package` declaration [ExplicitMainPackage]
// CHECK:STDERR: package r#Main;
// CHECK:STDERR: ^~~~~~~~~~~~~~~
// CHECK:STDERR:
package r#Main;

// --- fail_main_lib.carbon

// CHECK:STDERR: fail_main_lib.carbon:[[@LINE+4]]:1: error: use `library` declaration in `Main` package libraries [ExplicitMainLibrary]
// CHECK:STDERR: package Main library "main_lib";
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
package Main library "[[@TEST_NAME]]";

// --- lowercase_cpp.carbon

// The restriction is case-sensitive.
package cpp;

// --- fail_cpp.carbon

// CHECK:STDERR: fail_cpp.carbon:[[@LINE+4]]:1: error: `Cpp` cannot be used by a `package` declaration [CppPackageDeclaration]
// CHECK:STDERR: package Cpp;
// CHECK:STDERR: ^~~~~~~~~~~~
// CHECK:STDERR:
package Cpp;

// --- fail_cpp.impl.carbon

// CHECK:STDERR: fail_cpp.impl.carbon:[[@LINE+8]]:1: error: `Cpp` cannot be used by a `package` declaration [CppPackageDeclaration]
// CHECK:STDERR: impl package Cpp;
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_cpp.impl.carbon:[[@LINE+4]]:1: error: `Cpp` import missing library [CppInteropMissingLibrary]
// CHECK:STDERR: impl package Cpp;
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~
// CHECK:STDERR:
impl package Cpp;

// --- fail_raw_cpp.carbon

// `cpp` isn't a keyword, so this fails the same way.
// CHECK:STDERR: fail_raw_cpp.carbon:[[@LINE+4]]:1: error: `Cpp` cannot be used by a `package` declaration [CppPackageDeclaration]
// CHECK:STDERR: package r#Cpp;
// CHECK:STDERR: ^~~~~~~~~~~~~~
// CHECK:STDERR:
package r#Cpp;

// --- fail_cpp_lib.carbon

// CHECK:STDERR: fail_cpp_lib.carbon:[[@LINE+4]]:1: error: `Cpp` cannot be used by a `package` declaration [CppPackageDeclaration]
// CHECK:STDERR: package Cpp library "cpp_lib";
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
package Cpp library "[[@TEST_NAME]]";

// CHECK:STDOUT: --- lowercase_main.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_main.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_main_impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_raw_main.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_main_lib.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- lowercase_cpp.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_cpp.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_cpp.impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_raw_cpp.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_cpp_lib.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
