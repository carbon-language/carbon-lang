// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interop/cpp/enum/convert.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interop/cpp/enum/convert.carbon

// --- enum.h

enum Enum : short { a, b, c };

enum Other : short {};

enum DifferentSize : long {};

// --- convert_enum.carbon

library "[[@TEST_NAME]]";

import Cpp library "enum.h";

//@dump-sem-ir-begin
fn F() {
  let unused a: i16 = Cpp.Enum.a as i16;
  let unused b: Cpp.Enum = (42 as i16) as Cpp.Enum;
  let unused c: Cpp.Other = Cpp.a as Cpp.Other;
}
//@dump-sem-ir-end

// --- fail_wrong_type.carbon

library "[[@TEST_NAME]]";

import Cpp library "enum.h";

fn F() {
  // CHECK:STDERR: fail_wrong_type.carbon:[[@LINE+7]]:3: error: cannot convert expression of type `Cpp.Enum` to `i8` with `as` [ConversionFailure]
  // CHECK:STDERR:   Cpp.Enum.a as i8;
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_wrong_type.carbon:[[@LINE+4]]:3: note: type `Cpp.Enum` does not implement interface `Core.As(i8)` [MissingImplInMemberAccessInContext]
  // CHECK:STDERR:   Cpp.Enum.a as i8;
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  Cpp.Enum.a as i8;

  // CHECK:STDERR: fail_wrong_type.carbon:[[@LINE+7]]:3: error: cannot convert expression of type `Cpp.Enum` to `i32` with `as` [ConversionFailure]
  // CHECK:STDERR:   Cpp.Enum.a as i32;
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_wrong_type.carbon:[[@LINE+4]]:3: note: type `Cpp.Enum` does not implement interface `Core.As(i32)` [MissingImplInMemberAccessInContext]
  // CHECK:STDERR:   Cpp.Enum.a as i32;
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  Cpp.Enum.a as i32;

  // CHECK:STDERR: fail_wrong_type.carbon:[[@LINE+7]]:3: error: cannot convert expression of type `i8` to `Cpp.Enum` with `as` [ConversionFailure]
  // CHECK:STDERR:   (42 as i8) as Cpp.Enum;
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_wrong_type.carbon:[[@LINE+4]]:3: note: type `i8` does not implement interface `Core.As(Cpp.Enum)` [MissingImplInMemberAccessInContext]
  // CHECK:STDERR:   (42 as i8) as Cpp.Enum;
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  (42 as i8) as Cpp.Enum;

  // CHECK:STDERR: fail_wrong_type.carbon:[[@LINE+7]]:3: error: cannot convert expression of type `i32` to `Cpp.Enum` with `as` [ConversionFailure]
  // CHECK:STDERR:   (42 as i32) as Cpp.Enum;
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_wrong_type.carbon:[[@LINE+4]]:3: note: type `i32` does not implement interface `Core.As(Cpp.Enum)` [MissingImplInMemberAccessInContext]
  // CHECK:STDERR:   (42 as i32) as Cpp.Enum;
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  (42 as i32) as Cpp.Enum;

  // CHECK:STDERR: fail_wrong_type.carbon:[[@LINE+7]]:3: error: cannot convert expression of type `Cpp.Enum` to `Cpp.DifferentSize` with `as` [ConversionFailure]
  // CHECK:STDERR:   Cpp.a as Cpp.DifferentSize;
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_wrong_type.carbon:[[@LINE+4]]:3: note: type `Cpp.Enum` does not implement interface `Core.As(Cpp.DifferentSize)` [MissingImplInMemberAccessInContext]
  // CHECK:STDERR:   Cpp.a as Cpp.DifferentSize;
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  Cpp.a as Cpp.DifferentSize;
}

// CHECK:STDOUT: --- convert_enum.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %int_16: Core.IntLiteral = int_value 16 [concrete]
// CHECK:STDOUT:   %i16: type = class_type @Int, @Int(%int_16) [concrete]
// CHECK:STDOUT:   %pattern_type.2f8: type = pattern_type %i16 [concrete]
// CHECK:STDOUT:   %Enum: type = class_type @Enum [concrete]
// CHECK:STDOUT:   %int_0.420: %Enum = int_value 0 [concrete]
// CHECK:STDOUT:   %int_0.320: %i16 = int_value 0 [concrete]
// CHECK:STDOUT:   %pattern_type.ebf: type = pattern_type %Enum [concrete]
// CHECK:STDOUT:   %int_42.20e: Core.IntLiteral = int_value 42 [concrete]
// CHECK:STDOUT:   %As.type.359: type = facet_type <@As, @As(%i16)> [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.type.09e: type = fn_type @Core.IntLiteral.as.As.impl.Convert, @Core.IntLiteral.as.As.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.dbe: %Core.IntLiteral.as.As.impl.Convert.type.09e = struct_value () [symbolic]
// CHECK:STDOUT:   %As.impl_witness.b61: <witness> = impl_witness imports.%As.impl_witness_table.9fc, @Core.IntLiteral.as.As.impl(%int_16) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.type.c60: type = fn_type @Core.IntLiteral.as.As.impl.Convert, @Core.IntLiteral.as.As.impl(%int_16) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.a42: %Core.IntLiteral.as.As.impl.Convert.type.c60 = struct_value () [concrete]
// CHECK:STDOUT:   %As.facet: %As.type.359 = facet_value Core.IntLiteral, (%As.impl_witness.b61) [concrete]
// CHECK:STDOUT:   %As.WithSelf.Convert.type.2c0: type = fn_type @As.WithSelf.Convert, @As(%i16, %As.facet) [concrete]
// CHECK:STDOUT:   %.70c: type = fn_type_with_self_type %As.WithSelf.Convert.type.2c0, %As.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.bound: <bound method> = bound_method %int_42.20e, %Core.IntLiteral.as.As.impl.Convert.a42 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.As.impl.Convert.a42, @Core.IntLiteral.as.As.impl.Convert(%int_16) [concrete]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %int_42.20e, %Core.IntLiteral.as.As.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_42.a0e: %i16 = int_value 42 [concrete]
// CHECK:STDOUT:   %int_42.c3a: %Enum = int_value 42 [concrete]
// CHECK:STDOUT:   %Other: type = class_type @Other [concrete]
// CHECK:STDOUT:   %pattern_type.c06: type = pattern_type %Other [concrete]
// CHECK:STDOUT:   %int_0.fa2: %Other = int_value 0 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .Enum = %Enum.decl
// CHECK:STDOUT:     .Other = %Other.decl
// CHECK:STDOUT:     .a = %int_0
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Enum.decl: type = class_decl @Enum [concrete = constants.%Enum] {} {}
// CHECK:STDOUT:   %int_0: %Enum = int_value 0 [concrete = constants.%int_0.420]
// CHECK:STDOUT:   %Core.import_ref.ca0: @Core.IntLiteral.as.As.impl.%Core.IntLiteral.as.As.impl.Convert.type (%Core.IntLiteral.as.As.impl.Convert.type.09e) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.As.impl.%Core.IntLiteral.as.As.impl.Convert (constants.%Core.IntLiteral.as.As.impl.Convert.dbe)]
// CHECK:STDOUT:   %As.impl_witness_table.9fc = impl_witness_table (%Core.import_ref.ca0), @Core.IntLiteral.as.As.impl [concrete]
// CHECK:STDOUT:   %Other.decl: type = class_decl @Other [concrete = constants.%Other] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.2f8 = value_binding_pattern a [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc8: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %Enum.ref.loc8: type = name_ref Enum, imports.%Enum.decl [concrete = constants.%Enum]
// CHECK:STDOUT:   %a.ref.loc8: %Enum = name_ref a, imports.%int_0 [concrete = constants.%int_0.420]
// CHECK:STDOUT:   %int_16.loc8_37: Core.IntLiteral = int_value 16 [concrete = constants.%int_16]
// CHECK:STDOUT:   %i16.loc8_37: type = class_type @Int, @Int(constants.%int_16) [concrete = constants.%i16]
// CHECK:STDOUT:   %.loc8_34.1: %i16 = as_compatible %a.ref.loc8 [concrete = constants.%int_0.320]
// CHECK:STDOUT:   %.loc8_34.2: %i16 = converted %a.ref.loc8, %.loc8_34.1 [concrete = constants.%int_0.320]
// CHECK:STDOUT:   %.loc8_17: type = splice_block %i16.loc8_17 [concrete = constants.%i16] {
// CHECK:STDOUT:     %int_16.loc8_17: Core.IntLiteral = int_value 16 [concrete = constants.%int_16]
// CHECK:STDOUT:     %i16.loc8_17: type = class_type @Int, @Int(constants.%int_16) [concrete = constants.%i16]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: %i16 = value_binding a, %.loc8_34.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %pattern_type.ebf = value_binding_pattern b [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int_42: Core.IntLiteral = int_value 42 [concrete = constants.%int_42.20e]
// CHECK:STDOUT:   %int_16.loc9: Core.IntLiteral = int_value 16 [concrete = constants.%int_16]
// CHECK:STDOUT:   %i16.loc9: type = class_type @Int, @Int(constants.%int_16) [concrete = constants.%i16]
// CHECK:STDOUT:   %impl.elem0: %.70c = impl_witness_access constants.%As.impl_witness.b61, element0 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.a42]
// CHECK:STDOUT:   %bound_method.loc9_32.1: <bound method> = bound_method %int_42, %impl.elem0 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @Core.IntLiteral.as.As.impl.Convert(constants.%int_16) [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc9_32.2: <bound method> = bound_method %int_42, %specific_fn [concrete = constants.%bound_method]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.call: init %i16 = call %bound_method.loc9_32.2(%int_42) [concrete = constants.%int_42.a0e]
// CHECK:STDOUT:   %.loc9_32.1: %i16 = value_of_initializer %Core.IntLiteral.as.As.impl.Convert.call [concrete = constants.%int_42.a0e]
// CHECK:STDOUT:   %.loc9_32.2: %i16 = converted %int_42, %.loc9_32.1 [concrete = constants.%int_42.a0e]
// CHECK:STDOUT:   %Cpp.ref.loc9_43: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %Enum.ref.loc9_46: type = name_ref Enum, imports.%Enum.decl [concrete = constants.%Enum]
// CHECK:STDOUT:   %.loc9_40.1: %Enum = as_compatible %.loc9_32.2 [concrete = constants.%int_42.c3a]
// CHECK:STDOUT:   %.loc9_40.2: %Enum = converted %.loc9_32.2, %.loc9_40.1 [concrete = constants.%int_42.c3a]
// CHECK:STDOUT:   %.loc9_20: type = splice_block %Enum.ref.loc9_20 [concrete = constants.%Enum] {
// CHECK:STDOUT:     %Cpp.ref.loc9_17: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %Enum.ref.loc9_20: type = name_ref Enum, imports.%Enum.decl [concrete = constants.%Enum]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b: %Enum = value_binding b, %.loc9_40.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c.patt: %pattern_type.c06 = value_binding_pattern c [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc10_29: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %a.ref.loc10: %Enum = name_ref a, imports.%int_0 [concrete = constants.%int_0.420]
// CHECK:STDOUT:   %Cpp.ref.loc10_38: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %Other.ref.loc10_41: type = name_ref Other, imports.%Other.decl [concrete = constants.%Other]
// CHECK:STDOUT:   %.loc10_35.1: %Other = as_compatible %a.ref.loc10 [concrete = constants.%int_0.fa2]
// CHECK:STDOUT:   %.loc10_35.2: %Other = converted %a.ref.loc10, %.loc10_35.1 [concrete = constants.%int_0.fa2]
// CHECK:STDOUT:   %.loc10_20: type = splice_block %Other.ref.loc10_20 [concrete = constants.%Other] {
// CHECK:STDOUT:     %Cpp.ref.loc10_17: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %Other.ref.loc10_20: type = name_ref Other, imports.%Other.decl [concrete = constants.%Other]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c: %Other = value_binding c, %.loc10_35.2
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
