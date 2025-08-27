// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/convert.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interop/cpp/enum/member.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interop/cpp/enum/member.carbon

// --- enum_member.h

class A {
public:
  enum E { a, b, c };
};

// --- import_enum_member.carbon

library "[[@TEST_NAME]]";

import Cpp library "enum_member.h";

//@dump-sem-ir-begin
fn F() {
  let a: Cpp.A.E = Cpp.A.a;
  let b: Cpp.A.E = Cpp.A.E.b;
}
//@dump-sem-ir-end

// CHECK:STDOUT: --- import_enum_member.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %A: type = class_type @A [concrete]
// CHECK:STDOUT:   %E: type = class_type @E [concrete]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %E [concrete]
// CHECK:STDOUT:   %int_0: %E = int_value 0 [concrete]
// CHECK:STDOUT:   %int_1: %E = int_value 1 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .A = %A.decl
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %A.decl: type = class_decl @A [concrete = constants.%A] {} {}
// CHECK:STDOUT:   %E.decl: type = class_decl @E [concrete = constants.%E] {} {}
// CHECK:STDOUT:   %int_0: %E = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT:   %int_1: %E = int_value 1 [concrete = constants.%int_1]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type = binding_pattern a [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc8_20: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %A.ref.loc8_23: type = name_ref A, imports.%A.decl [concrete = constants.%A]
// CHECK:STDOUT:   %a.ref: %E = name_ref a, imports.%int_0 [concrete = constants.%int_0]
// CHECK:STDOUT:   %.loc8: type = splice_block %E.ref.loc8 [concrete = constants.%E] {
// CHECK:STDOUT:     %Cpp.ref.loc8_10: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %A.ref.loc8_13: type = name_ref A, imports.%A.decl [concrete = constants.%A]
// CHECK:STDOUT:     %E.ref.loc8: type = name_ref E, imports.%E.decl [concrete = constants.%E]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: %E = bind_name a, %a.ref
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %pattern_type = binding_pattern b [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc9_20: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %A.ref.loc9_23: type = name_ref A, imports.%A.decl [concrete = constants.%A]
// CHECK:STDOUT:   %E.ref.loc9_25: type = name_ref E, imports.%E.decl [concrete = constants.%E]
// CHECK:STDOUT:   %b.ref: %E = name_ref b, imports.%int_1 [concrete = constants.%int_1]
// CHECK:STDOUT:   %.loc9: type = splice_block %E.ref.loc9_15 [concrete = constants.%E] {
// CHECK:STDOUT:     %Cpp.ref.loc9_10: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %A.ref.loc9_13: type = name_ref A, imports.%A.decl [concrete = constants.%A]
// CHECK:STDOUT:     %E.ref.loc9_15: type = name_ref E, imports.%E.decl [concrete = constants.%E]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b: %E = bind_name b, %b.ref
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
