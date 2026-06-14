// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/convert.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interop/cpp/enum/unscoped.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interop/cpp/enum/unscoped.carbon

// --- enum.h

enum Enum { a, b, c };

enum Other { d };

// --- import_enum.carbon

library "[[@TEST_NAME]]";

import Cpp library "enum.h";

//@dump-sem-ir-begin
fn F() {
  let unused a: Cpp.Enum = Cpp.Enum.a;
  let unused b: Cpp.Enum = Cpp.b;
}
//@dump-sem-ir-end

// --- fail_wrong_enum.carbon

library "[[@TEST_NAME]]";

import Cpp library "enum.h";

fn F() {
  // CHECK:STDERR: fail_wrong_enum.carbon:[[@LINE+7]]:29: error: cannot implicitly convert expression of type `Cpp.Enum` to `Cpp.Other` [ConversionFailure]
  // CHECK:STDERR:   let unused c: Cpp.Other = Cpp.Enum.c;
  // CHECK:STDERR:                             ^~~~~~~~~~
  // CHECK:STDERR: fail_wrong_enum.carbon:[[@LINE+4]]:29: note: type `Cpp.Enum` does not implement interface `Core.ImplicitAs(Cpp.Other)` [MissingImplInMemberAccessInContext]
  // CHECK:STDERR:   let unused c: Cpp.Other = Cpp.Enum.c;
  // CHECK:STDERR:                             ^~~~~~~~~~
  // CHECK:STDERR:
  let unused c: Cpp.Other = Cpp.Enum.c;

  // CHECK:STDERR: fail_wrong_enum.carbon:[[@LINE+7]]:30: error: cannot implicitly convert expression of type `Cpp.Enum` to `Cpp.Other` [ConversionFailure]
  // CHECK:STDERR:   let unused c2: Cpp.Other = Cpp.c;
  // CHECK:STDERR:                              ^~~~~
  // CHECK:STDERR: fail_wrong_enum.carbon:[[@LINE+4]]:30: note: type `Cpp.Enum` does not implement interface `Core.ImplicitAs(Cpp.Other)` [MissingImplInMemberAccessInContext]
  // CHECK:STDERR:   let unused c2: Cpp.Other = Cpp.c;
  // CHECK:STDERR:                              ^~~~~
  // CHECK:STDERR:
  let unused c2: Cpp.Other = Cpp.c;
}

// --- fail_non_enum_member.carbon

library "[[@TEST_NAME]]";

import Cpp library "enum.h";

fn F() {
  // CHECK:STDERR: fail_non_enum_member.carbon:[[@LINE+4]]:29: error: member name `d` not found in `Cpp.Enum` [MemberNameNotFoundInInstScope]
  // CHECK:STDERR:   let unused c: Cpp.Other = Cpp.Enum.d;
  // CHECK:STDERR:                             ^~~~~~~~~~
  // CHECK:STDERR:
  let unused c: Cpp.Other = Cpp.Enum.d;
}

// CHECK:STDOUT: --- import_enum.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %Enum: type = class_type @Enum [concrete]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Enum [concrete]
// CHECK:STDOUT:   %a.patt: %pattern_type = value_binding_pattern a [concrete]
// CHECK:STDOUT:   %int_0: %Enum = int_value 0 [concrete]
// CHECK:STDOUT:   %b.patt: %pattern_type = value_binding_pattern b [concrete]
// CHECK:STDOUT:   %int_1: %Enum = int_value 1 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .Enum = %Enum.decl
// CHECK:STDOUT:     .b = %int_1
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Enum.decl: type = class_decl @Enum [concrete = constants.%Enum] {} {}
// CHECK:STDOUT:   %int_0: %Enum = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT:   %int_1: %Enum = int_value 1 [concrete = constants.%int_1]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type = value_binding_pattern a [concrete = constants.%a.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc8_28: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %Enum.ref.loc8_31: type = name_ref Enum, imports.%Enum.decl [concrete = constants.%Enum]
// CHECK:STDOUT:   %a.ref: %Enum = name_ref a, imports.%int_0 [concrete = constants.%int_0]
// CHECK:STDOUT:   %.loc8: type = splice_block %Enum.ref.loc8_20 [concrete = constants.%Enum] {
// CHECK:STDOUT:     %Cpp.ref.loc8_17: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %Enum.ref.loc8_20: type = name_ref Enum, imports.%Enum.decl [concrete = constants.%Enum]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: %Enum = value_binding a, %a.ref
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %pattern_type = value_binding_pattern b [concrete = constants.%b.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc9_28: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %b.ref: %Enum = name_ref b, imports.%int_1 [concrete = constants.%int_1]
// CHECK:STDOUT:   %.loc9: type = splice_block %Enum.ref.loc9 [concrete = constants.%Enum] {
// CHECK:STDOUT:     %Cpp.ref.loc9_17: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %Enum.ref.loc9: type = name_ref Enum, imports.%Enum.decl [concrete = constants.%Enum]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b: %Enum = value_binding b, %b.ref
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
