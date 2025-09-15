// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/destroy.carbon
// EXTRA-ARGS: --dump-sem-ir-ranges=if-present
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interop/cpp/globals.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interop/cpp/globals.carbon

// ============================================================================
// Global scope
// ============================================================================

// --- global_scope.h

class C {};
C global;

// --- import_global_scope.carbon

library "[[@TEST_NAME]]";

import Cpp library "global_scope.h";

fn MyF() {
  let local: Cpp.C = Cpp.global;
}

// ============================================================================
// Namespace
// ============================================================================

// --- namespace.h

namespace N {
class C {};
C global;
}  // namespace N

// --- import_namespace.carbon

library "[[@TEST_NAME]]";

import Cpp library "namespace.h";

fn MyF() {
  let local: Cpp.N.C = Cpp.N.global;
}

// ============================================================================
// Unsupported global type
// ============================================================================

// --- unsupported_type.h

using UnusupportedType = _BitInt(23);
UnusupportedType global;

// --- fail_import_unsupported_type.carbon

library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_import_unsupported_type.carbon:[[@LINE+4]]:10: in file included here [InCppInclude]
// CHECK:STDERR: ./unsupported_type.h:2:7: error: semantics TODO: `Unsupported: Type declaration: UnusupportedType` [SemanticsTodo]
// CHECK:STDERR: using UnusupportedType = _BitInt(23);
// CHECK:STDERR:       ^
import Cpp library "unsupported_type.h";

fn MyF() {
  // CHECK:STDERR: fail_import_unsupported_type.carbon:[[@LINE+11]]:14: note: in `Cpp` name lookup for `UnusupportedType` [InCppNameLookup]
  // CHECK:STDERR:   let local: Cpp.UnusupportedType = Cpp.global;
  // CHECK:STDERR:              ^~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: fail_import_unsupported_type.carbon:[[@LINE+7]]:37: error: semantics TODO: `Unsupported: var type: UnusupportedType` [SemanticsTodo]
  // CHECK:STDERR:   let local: Cpp.UnusupportedType = Cpp.global;
  // CHECK:STDERR:                                     ^~~~~~~~~~
  // CHECK:STDERR: fail_import_unsupported_type.carbon:[[@LINE+4]]:37: note: in `Cpp` name lookup for `global` [InCppNameLookup]
  // CHECK:STDERR:   let local: Cpp.UnusupportedType = Cpp.global;
  // CHECK:STDERR:                                     ^~~~~~~~~~
  // CHECK:STDERR:
  let local: Cpp.UnusupportedType = Cpp.global;
}

// CHECK:STDOUT: --- import_global_scope.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %MyF.type: type = fn_type @MyF [concrete]
// CHECK:STDOUT:   %MyF: %MyF.type = struct_value () [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %C [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .global = %global.var
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   %global.var: ref %C = var %global.var_patt [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Cpp = imports.%Cpp
// CHECK:STDOUT:     .MyF = %MyF.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Cpp.import_cpp = import_cpp {
// CHECK:STDOUT:     import Cpp "global_scope.h"
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %MyF.decl: %MyF.type = fn_decl @MyF [concrete = constants.%MyF] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   import Cpp//...
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @MyF() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %local.patt: %pattern_type = binding_pattern local [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc7_22: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %global.ref: ref %C = name_ref global, imports.%global.var [concrete = imports.%global.var]
// CHECK:STDOUT:   %.loc7_17: type = splice_block %C.ref [concrete = constants.%C] {
// CHECK:STDOUT:     %Cpp.ref.loc7_14: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %C.ref: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc7_25: %C = bind_value %global.ref
// CHECK:STDOUT:   %local: %C = bind_name local, %.loc7_25
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_namespace.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %MyF.type: type = fn_type @MyF [concrete]
// CHECK:STDOUT:   %MyF: %MyF.type = struct_value () [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %C [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .N = %N
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %N: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .global = %global.var
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   %global.var: ref %C = var %global.var_patt [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Cpp = imports.%Cpp
// CHECK:STDOUT:     .MyF = %MyF.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Cpp.import_cpp = import_cpp {
// CHECK:STDOUT:     import Cpp "namespace.h"
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %MyF.decl: %MyF.type = fn_decl @MyF [concrete = constants.%MyF] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   import Cpp//...
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @MyF() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %local.patt: %pattern_type = binding_pattern local [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc7_24: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %N.ref.loc7_27: <namespace> = name_ref N, imports.%N [concrete = imports.%N]
// CHECK:STDOUT:   %global.ref: ref %C = name_ref global, imports.%global.var [concrete = imports.%global.var]
// CHECK:STDOUT:   %.loc7_19: type = splice_block %C.ref [concrete = constants.%C] {
// CHECK:STDOUT:     %Cpp.ref.loc7_14: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %N.ref.loc7_17: <namespace> = name_ref N, imports.%N [concrete = imports.%N]
// CHECK:STDOUT:     %C.ref: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc7_29: %C = bind_value %global.ref
// CHECK:STDOUT:   %local: %C = bind_name local, %.loc7_29
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_import_unsupported_type.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %MyF.type: type = fn_type @MyF [concrete]
// CHECK:STDOUT:   %MyF: %MyF.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .UnusupportedType = <error>
// CHECK:STDOUT:     .global = <error>
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Cpp = imports.%Cpp
// CHECK:STDOUT:     .MyF = %MyF.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Cpp.import_cpp = import_cpp {
// CHECK:STDOUT:     import Cpp "unsupported_type.h"
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %MyF.decl: %MyF.type = fn_decl @MyF [concrete = constants.%MyF] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @MyF() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %local.patt: <error> = binding_pattern local [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc22_37: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %global.ref: <error> = name_ref global, <error> [concrete = <error>]
// CHECK:STDOUT:   %.1: <error> = splice_block <error> [concrete = <error>] {
// CHECK:STDOUT:     %Cpp.ref.loc22_14: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %UnusupportedType.ref: <error> = name_ref UnusupportedType, <error> [concrete = <error>]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %local: <error> = bind_name local, <error> [concrete = <error>]
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
