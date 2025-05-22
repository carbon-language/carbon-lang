// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// EXTRA-ARGS: --dump-sem-ir-ranges=only
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/adapter/init_adapt.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/adapter/init_adapt.carbon

// --- init_adapt.carbon

library "[[@TEST_NAME]]";

class C {
  var a: i32;
  var b: i32;
}

//@dump-sem-ir-begin
class AdaptC {
  adapt C;
}
//@dump-sem-ir-end

let a: C = {.a = 1, .b = 2};

//@dump-sem-ir-begin
let b: AdaptC = a as AdaptC;
//@dump-sem-ir-end

let c: C = b as C;

fn MakeC() -> C;

fn MakeAdaptC() -> AdaptC;

var d: AdaptC = MakeC() as AdaptC;

var e: C = MakeAdaptC() as C;

// --- fail_not_implicit.carbon

library "[[@TEST_NAME]]";

class C {
  var a: i32;
  var b: i32;
}

class AdaptC {
  adapt C;
}

let a: C = {.a = 1, .b = 2};

// Cannot implicitly convert between a type and an adapter for the type.

// CHECK:STDERR: fail_not_implicit.carbon:[[@LINE+7]]:17: error: cannot implicitly convert expression of type `C` to `AdaptC` [ConversionFailure]
// CHECK:STDERR: let b: AdaptC = a;
// CHECK:STDERR:                 ^
// CHECK:STDERR: fail_not_implicit.carbon:[[@LINE+4]]:17: note: type `C` does not implement interface `Core.ImplicitAs(AdaptC)` [MissingImplInMemberAccessNote]
// CHECK:STDERR: let b: AdaptC = a;
// CHECK:STDERR:                 ^
// CHECK:STDERR:
let b: AdaptC = a;

// CHECK:STDERR: fail_not_implicit.carbon:[[@LINE+7]]:12: error: cannot implicitly convert expression of type `AdaptC` to `C` [ConversionFailure]
// CHECK:STDERR: let c: C = b;
// CHECK:STDERR:            ^
// CHECK:STDERR: fail_not_implicit.carbon:[[@LINE+4]]:12: note: type `AdaptC` does not implement interface `Core.ImplicitAs(C)` [MissingImplInMemberAccessNote]
// CHECK:STDERR: let c: C = b;
// CHECK:STDERR:            ^
// CHECK:STDERR:
let c: C = b;

fn MakeC() -> C;

fn MakeAdaptC() -> AdaptC;

// CHECK:STDERR: fail_not_implicit.carbon:[[@LINE+7]]:1: error: cannot implicitly convert expression of type `C` to `AdaptC` [ConversionFailure]
// CHECK:STDERR: var d: AdaptC = MakeC();
// CHECK:STDERR: ^~~~~~~~~~~~~
// CHECK:STDERR: fail_not_implicit.carbon:[[@LINE+4]]:1: note: type `C` does not implement interface `Core.ImplicitAs(AdaptC)` [MissingImplInMemberAccessNote]
// CHECK:STDERR: var d: AdaptC = MakeC();
// CHECK:STDERR: ^~~~~~~~~~~~~
// CHECK:STDERR:
var d: AdaptC = MakeC();

// CHECK:STDERR: fail_not_implicit.carbon:[[@LINE+7]]:1: error: cannot implicitly convert expression of type `AdaptC` to `C` [ConversionFailure]
// CHECK:STDERR: var e: C = MakeAdaptC();
// CHECK:STDERR: ^~~~~~~~
// CHECK:STDERR: fail_not_implicit.carbon:[[@LINE+4]]:1: note: type `AdaptC` does not implement interface `Core.ImplicitAs(C)` [MissingImplInMemberAccessNote]
// CHECK:STDERR: var e: C = MakeAdaptC();
// CHECK:STDERR: ^~~~~~~~
// CHECK:STDERR:
var e: C = MakeAdaptC();

// CHECK:STDOUT: --- init_adapt.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %struct_type.a.b.501: type = struct_type {.a: %i32, .b: %i32} [concrete]
// CHECK:STDOUT:   %complete_type.705: <witness> = complete_type_witness %struct_type.a.b.501 [concrete]
// CHECK:STDOUT:   %AdaptC: type = class_type @AdaptC [concrete]
// CHECK:STDOUT:   %pattern_type.a1a: type = pattern_type %AdaptC [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %AdaptC.decl: type = class_decl @AdaptC [concrete = constants.%AdaptC] {} {}
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %pattern_type.a1a = binding_pattern b [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %AdaptC.ref.loc18: type = name_ref AdaptC, %AdaptC.decl [concrete = constants.%AdaptC]
// CHECK:STDOUT:   %b: ref %AdaptC = bind_name b, @__global_init.%.loc18_19.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @AdaptC {
// CHECK:STDOUT:   %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   adapt_decl %C.ref [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%struct_type.a.b.501 [concrete = constants.%complete_type.705]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%AdaptC
// CHECK:STDOUT:   .C = <poisoned>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %a.ref: ref %C = name_ref a, file.%a
// CHECK:STDOUT:   %AdaptC.ref.loc18: type = name_ref AdaptC, file.%AdaptC.decl [concrete = constants.%AdaptC]
// CHECK:STDOUT:   %.loc18_19.1: ref %AdaptC = as_compatible %a.ref
// CHECK:STDOUT:   %.loc18_19.2: ref %AdaptC = converted %a.ref, %.loc18_19.1
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
