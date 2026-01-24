// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/convert.carbon
// EXTRA-ARGS: --clang-arg=-fms-compatibility
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interop/cpp/enum/incomplete.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interop/cpp/enum/incomplete.carbon

// --- declaration.h

// This is not valid in standard C++, but MSVC allows it as an extension, and so
// Clang allows it in MSVC compatibility mode.
enum UnsizedEnum;

// --- import_declaration.carbon

library "[[@TEST_NAME]]";

// CHECK:STDERR: import_declaration.carbon:[[@LINE+5]]:10: in file included here [InCppInclude]
// CHECK:STDERR: ./declaration.h:4:6: warning: forward references to 'enum' types are a Microsoft extension [CppInteropParseWarning]
// CHECK:STDERR:     4 | enum UnsizedEnum;
// CHECK:STDERR:       |      ^
// CHECK:STDERR:
import Cpp library "declaration.h";

//@dump-sem-ir-begin
fn F(e: Cpp.UnsizedEnum);
//@dump-sem-ir-end

// --- fail_use_declaration_as_definition.carbon

library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_use_declaration_as_definition.carbon:[[@LINE+5]]:10: in file included here [InCppInclude]
// CHECK:STDERR: ./declaration.h:4:6: warning: forward references to 'enum' types are a Microsoft extension [CppInteropParseWarning]
// CHECK:STDERR:     4 | enum UnsizedEnum;
// CHECK:STDERR:       |      ^
// CHECK:STDERR:
import Cpp library "declaration.h";

fn MyF() {
  // TODO: It's a bit surprising to refer to this type as a class, even though
  // we model it as a class in Carbon. Consider customizing the diagnostic.
  // CHECK:STDERR: fail_use_declaration_as_definition.carbon:[[@LINE+8]]:10: error: binding pattern has incomplete type `UnsizedEnum` in name binding declaration [IncompleteTypeInBindingDecl]
  // CHECK:STDERR:   var e: Cpp.UnsizedEnum;
  // CHECK:STDERR:          ^~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_use_declaration_as_definition.carbon:[[@LINE-8]]:10: in file included here [InCppInclude]
  // CHECK:STDERR: ./declaration.h:4:6: note: class was forward declared here [ClassForwardDeclaredHere]
  // CHECK:STDERR: enum UnsizedEnum;
  // CHECK:STDERR:      ^
  // CHECK:STDERR:
  var e: Cpp.UnsizedEnum;
}

// CHECK:STDOUT: --- import_declaration.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %UnsizedEnum: type = class_type @UnsizedEnum [concrete]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %UnsizedEnum [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .UnsizedEnum = %UnsizedEnum.decl
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %UnsizedEnum.decl: type = class_decl @UnsizedEnum [concrete = constants.%UnsizedEnum] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %e.patt: %pattern_type = value_binding_pattern e [concrete]
// CHECK:STDOUT:     %e.param_patt: %pattern_type = value_param_pattern %e.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %e.param: %UnsizedEnum = value_param call_param0
// CHECK:STDOUT:     %.loc12: type = splice_block %UnsizedEnum.ref [concrete = constants.%UnsizedEnum] {
// CHECK:STDOUT:       %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:       %UnsizedEnum.ref: type = name_ref UnsizedEnum, imports.%UnsizedEnum.decl [concrete = constants.%UnsizedEnum]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %e: %UnsizedEnum = value_binding e, %e.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%e.param: %UnsizedEnum);
// CHECK:STDOUT:
