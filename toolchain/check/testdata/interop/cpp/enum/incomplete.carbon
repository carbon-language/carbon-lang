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
// CHECK:STDERR:        ╷ ╭── included from: import_declaration.carbon [InCppInclude]
// CHECK:STDERR:        ├─┤ ./declaration.h:[[@LINE+15]]:6
// CHECK:STDERR: ->     │ enum UnsizedEnum;
// CHECK:STDERR:        ·      ┳
// CHECK:STDERR:        ·      ┗━━┫ warning: forward references to 'enum' types are a Microsoft extension
// CHECK:STDERR:        ·         ┃          [CppInteropParseWarning]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
// CHECK:STDERR:        ╷ ╭── included from: fail_use_declaration_as_definition.carbon [InCppInclude]
// CHECK:STDERR:        ├─┤ ./declaration.h:[[@LINE+7]]:6
// CHECK:STDERR: ->     │ enum UnsizedEnum;
// CHECK:STDERR:        ·      ┳
// CHECK:STDERR:        ·      ┗━━┫ warning: forward references to 'enum' types are a Microsoft extension
// CHECK:STDERR:        ·         ┃          [CppInteropParseWarning]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
enum UnsizedEnum;

// --- import_declaration.carbon

library "[[@TEST_NAME]]";

import Cpp library "declaration.h";

//@dump-sem-ir-begin
fn F(e: Cpp.UnsizedEnum);
//@dump-sem-ir-end

// --- fail_use_declaration_as_definition.carbon

library "[[@TEST_NAME]]";

import Cpp library "declaration.h";

fn MyF() {
  // TODO: It's a bit surprising to refer to this type as a class, even though
  // we model it as a class in Carbon. Consider customizing the diagnostic.
  // CHECK:STDERR:        ╭─┤ fail_use_declaration_as_definition.carbon:[[@LINE+14]]:17
  // CHECK:STDERR: ->     │   var unused e: Cpp.UnsizedEnum;
  // CHECK:STDERR:        ·                 ┳━━━━━━━━━━━━━━
  // CHECK:STDERR:        ·                 ┗━━┫ error: binding pattern has incomplete type `UnsizedEnum` in
  // CHECK:STDERR:        ·                    ┃        name binding declaration
  // CHECK:STDERR:        ·                    ┃        [IncompleteTypeInBindingDecl]
  // CHECK:STDERR:        ┆
  // CHECK:STDERR:        │ ╭── included from: fail_use_declaration_as_definition.carbon [InCppInclude]
  // CHECK:STDERR:        ├─┤ ./declaration.h
  // CHECK:STDERR:        │ enum UnsizedEnum;
  // CHECK:STDERR:        ·      ┬
  // CHECK:STDERR:        ·      ╰──┤ class was forward declared here [ClassForwardDeclaredHere]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  var unused e: Cpp.UnsizedEnum;
}

// CHECK:STDOUT: --- import_declaration.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %UnsizedEnum: type = class_type @UnsizedEnum [concrete]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %UnsizedEnum [concrete]
// CHECK:STDOUT:   %e.param_patt: %pattern_type = value_param_pattern [concrete]
// CHECK:STDOUT:   %e.patt: %pattern_type = wrapper_binding_pattern e, %e.param_patt [concrete]
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
// CHECK:STDOUT:     %e.param_patt: %pattern_type = value_param_pattern [concrete = constants.%e.param_patt]
// CHECK:STDOUT:     %e.patt: %pattern_type = wrapper_binding_pattern e, %e.param_patt [concrete = constants.%e.patt]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %e.param: %UnsizedEnum = value_param call_param0
// CHECK:STDOUT:     %.loc7: type = splice_block %UnsizedEnum.ref [concrete = constants.%UnsizedEnum] {
// CHECK:STDOUT:       %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:       %UnsizedEnum.ref: type = name_ref UnsizedEnum, imports.%UnsizedEnum.decl [concrete = constants.%UnsizedEnum]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %e: %UnsizedEnum = wrapper_binding e, %e.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%e.param: %UnsizedEnum);
// CHECK:STDOUT:
