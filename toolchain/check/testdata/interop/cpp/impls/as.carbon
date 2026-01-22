// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/primitives.carbon
// EXTRA-ARGS: --target=x86_64-linux-gnu --clang-arg=-std=c++20
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interop/cpp/impls/as.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interop/cpp/impls/as.carbon

// --- conversions.h

struct Dest {};

struct Source {
  operator Dest() const;
};

struct Source2 {
  int x;
};
struct Dest2 {
  Dest2(Source2);
};

struct ExplicitConstructor {
  explicit ExplicitConstructor(Source);
};

struct ExplicitConversion {
  explicit operator Dest() const;
};

template<bool B> struct ConditionallyExplicit {
  explicit(B) ConditionallyExplicit(Source);
  explicit(B) operator Dest() const;
};

// --- conversions.carbon

library "[[@TEST_NAME]]";

import Cpp library "conversions.h";

fn UserConversion(s: Cpp.Source) {
  //@dump-sem-ir-begin
  s as Cpp.Dest;
  //@dump-sem-ir-end
}

fn ConstructorConversion(s: Cpp.Source2) {
  //@dump-sem-ir-begin
  // CHECK:STDERR: conversions.carbon:[[@LINE+13]]:3: error: semantics TODO: `Unsupported initialization sequence:
  // CHECK:STDERR: Normal sequence: constructor initialization [struct Dest2]
  // CHECK:STDERR: ` [SemanticsTodo]
  // CHECK:STDERR:   s as Cpp.Dest2;
  // CHECK:STDERR:   ^~~~~~~~~~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: conversions.carbon:[[@LINE+7]]:3: error: cannot convert expression of type `Cpp.Source2` to `Cpp.Dest2` with `as` [ConversionFailure]
  // CHECK:STDERR:   s as Cpp.Dest2;
  // CHECK:STDERR:   ^~~~~~~~~~~~~~
  // CHECK:STDERR: conversions.carbon:[[@LINE+4]]:3: note: type `Cpp.Source2` does not implement interface `Core.As(Cpp.Dest2)` [MissingImplInMemberAccessNote]
  // CHECK:STDERR:   s as Cpp.Dest2;
  // CHECK:STDERR:   ^~~~~~~~~~~~~~
  // CHECK:STDERR:
  s as Cpp.Dest2;
  //@dump-sem-ir-end
}

fn ExplicitConstructorTest(s: Cpp.Source) {
  //@dump-sem-ir-begin
  // CHECK:STDERR: conversions.carbon:[[@LINE+13]]:3: error: semantics TODO: `Unsupported initialization sequence:
  // CHECK:STDERR: Normal sequence: constructor initialization [struct ExplicitConstructor]
  // CHECK:STDERR: ` [SemanticsTodo]
  // CHECK:STDERR:   s as Cpp.ExplicitConstructor;
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: conversions.carbon:[[@LINE+7]]:3: error: cannot convert expression of type `Cpp.Source` to `Cpp.ExplicitConstructor` with `as` [ConversionFailure]
  // CHECK:STDERR:   s as Cpp.ExplicitConstructor;
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: conversions.carbon:[[@LINE+4]]:3: note: type `Cpp.Source` does not implement interface `Core.As(Cpp.ExplicitConstructor)` [MissingImplInMemberAccessNote]
  // CHECK:STDERR:   s as Cpp.ExplicitConstructor;
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  s as Cpp.ExplicitConstructor;
  //@dump-sem-ir-end
}

fn ExplicitConversionTest(s: Cpp.ExplicitConversion) {
  //@dump-sem-ir-begin
  s as Cpp.Dest;
  //@dump-sem-ir-end
}

// --- conditionally_explicit.carbon

library "[[@TEST_NAME]]";

import Cpp library "conversions.h";

fn ConstructorNotExplicit(s: Cpp.Source) {
  //@dump-sem-ir-begin
  // CHECK:STDERR: conditionally_explicit.carbon:[[@LINE+4]]:36: error: semantics TODO: `argument for non-type template parameter` [SemanticsTodo]
  // CHECK:STDERR:   let _: Cpp.ConditionallyExplicit(false) = s;
  // CHECK:STDERR:                                    ^~~~~
  // CHECK:STDERR:
  let _: Cpp.ConditionallyExplicit(false) = s;
  // CHECK:STDERR: conditionally_explicit.carbon:[[@LINE+4]]:34: error: semantics TODO: `argument for non-type template parameter` [SemanticsTodo]
  // CHECK:STDERR:   s as Cpp.ConditionallyExplicit(false);
  // CHECK:STDERR:                                  ^~~~~
  // CHECK:STDERR:
  s as Cpp.ConditionallyExplicit(false);
  //@dump-sem-ir-end
}

fn ConstructorExplicit(s: Cpp.Source) {
  //@dump-sem-ir-begin
  // CHECK:STDERR: conditionally_explicit.carbon:[[@LINE+4]]:34: error: semantics TODO: `argument for non-type template parameter` [SemanticsTodo]
  // CHECK:STDERR:   s as Cpp.ConditionallyExplicit(true);
  // CHECK:STDERR:                                  ^~~~
  // CHECK:STDERR:
  s as Cpp.ConditionallyExplicit(true);
  //@dump-sem-ir-end
}

// CHECK:STDERR: conditionally_explicit.carbon:[[@LINE+4]]:55: error: semantics TODO: `argument for non-type template parameter` [SemanticsTodo]
// CHECK:STDERR: fn ConversionNotExplicit(s: Cpp.ConditionallyExplicit(false)) {
// CHECK:STDERR:                                                       ^~~~~
// CHECK:STDERR:
fn ConversionNotExplicit(s: Cpp.ConditionallyExplicit(false)) {
  //@dump-sem-ir-begin
  let _: Cpp.Dest = s;
  s as Cpp.Dest;
  //@dump-sem-ir-end
}

// CHECK:STDERR: conditionally_explicit.carbon:[[@LINE+4]]:52: error: semantics TODO: `argument for non-type template parameter` [SemanticsTodo]
// CHECK:STDERR: fn ConversionExplicit(s: Cpp.ConditionallyExplicit(false)) {
// CHECK:STDERR:                                                    ^~~~~
// CHECK:STDERR:
fn ConversionExplicit(s: Cpp.ConditionallyExplicit(false)) {
  //@dump-sem-ir-begin
  s as Cpp.Dest;
  //@dump-sem-ir-end
}

// --- fail_conditionally_explicit_implicit.carbon

library "[[@TEST_NAME]]";

import Cpp library "conversions.h";

fn ConstructorExplicit(s: Cpp.Source) {
  //@dump-sem-ir-begin
  // CHECK:STDERR: fail_conditionally_explicit_implicit.carbon:[[@LINE+4]]:36: error: semantics TODO: `argument for non-type template parameter` [SemanticsTodo]
  // CHECK:STDERR:   let _: Cpp.ConditionallyExplicit(true) = s;
  // CHECK:STDERR:                                    ^~~~
  // CHECK:STDERR:
  let _: Cpp.ConditionallyExplicit(true) = s;
  //@dump-sem-ir-end
}

// CHECK:STDERR: fail_conditionally_explicit_implicit.carbon:[[@LINE+4]]:52: error: semantics TODO: `argument for non-type template parameter` [SemanticsTodo]
// CHECK:STDERR: fn ConversionExplicit(s: Cpp.ConditionallyExplicit(false)) {
// CHECK:STDERR:                                                    ^~~~~
// CHECK:STDERR:
fn ConversionExplicit(s: Cpp.ConditionallyExplicit(false)) {
  //@dump-sem-ir-begin
  let _: Cpp.Dest = s;
  //@dump-sem-ir-end
}

// CHECK:STDOUT: --- conversions.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Source: type = class_type @Source [concrete]
// CHECK:STDOUT:   %pattern_type.78c: type = pattern_type %Source [concrete]
// CHECK:STDOUT:   %Dest.5e7: type = class_type @Dest [concrete]
// CHECK:STDOUT:   %Source.cpp_operator.type: type = fn_type @Source.cpp_operator [concrete]
// CHECK:STDOUT:   %Source.cpp_operator: %Source.cpp_operator.type = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.551: type = ptr_type %Dest.5e7 [concrete]
// CHECK:STDOUT:   %Dest__carbon_thunk.type.2ffc9e.1: type = fn_type @Dest__carbon_thunk.1 [concrete]
// CHECK:STDOUT:   %Dest__carbon_thunk.f8fa06.1: %Dest__carbon_thunk.type.2ffc9e.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Dest.cpp_destructor.type: type = fn_type @Dest.cpp_destructor [concrete]
// CHECK:STDOUT:   %Dest.cpp_destructor: %Dest.cpp_destructor.type = struct_value () [concrete]
// CHECK:STDOUT:   %Source2: type = class_type @Source2 [concrete]
// CHECK:STDOUT:   %Dest2: type = class_type @Dest2 [concrete]
// CHECK:STDOUT:   %ExplicitConstructor: type = class_type @ExplicitConstructor [concrete]
// CHECK:STDOUT:   %ExplicitConversion: type = class_type @ExplicitConversion [concrete]
// CHECK:STDOUT:   %pattern_type.a70: type = pattern_type %ExplicitConversion [concrete]
// CHECK:STDOUT:   %ExplicitConversion.cpp_operator.type: type = fn_type @ExplicitConversion.cpp_operator [concrete]
// CHECK:STDOUT:   %ExplicitConversion.cpp_operator: %ExplicitConversion.cpp_operator.type = struct_value () [concrete]
// CHECK:STDOUT:   %Dest__carbon_thunk.type.2ffc9e.2: type = fn_type @Dest__carbon_thunk.2 [concrete]
// CHECK:STDOUT:   %Dest__carbon_thunk.f8fa06.2: %Dest__carbon_thunk.type.2ffc9e.2 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .Source = %Source.decl
// CHECK:STDOUT:     .Dest = %Dest.decl
// CHECK:STDOUT:     .Source2 = %Source2.decl
// CHECK:STDOUT:     .Dest2 = %Dest2.decl
// CHECK:STDOUT:     .ExplicitConstructor = %ExplicitConstructor.decl
// CHECK:STDOUT:     .ExplicitConversion = %ExplicitConversion.decl
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Source.decl: type = class_decl @Source [concrete = constants.%Source] {} {}
// CHECK:STDOUT:   %Dest.decl: type = class_decl @Dest [concrete = constants.%Dest.5e7] {} {}
// CHECK:STDOUT:   %Source.cpp_operator.decl: %Source.cpp_operator.type = fn_decl @Source.cpp_operator [concrete = constants.%Source.cpp_operator] {
// CHECK:STDOUT:     %self.patt: %pattern_type.78c = value_binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.78c = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %Source = value_param call_param0
// CHECK:STDOUT:     %self: %Source = value_binding self, %self.param
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Dest__carbon_thunk.decl.b58ecd.1: %Dest__carbon_thunk.type.2ffc9e.1 = fn_decl @Dest__carbon_thunk.1 [concrete = constants.%Dest__carbon_thunk.f8fa06.1] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Source2.decl: type = class_decl @Source2 [concrete = constants.%Source2] {} {}
// CHECK:STDOUT:   %Dest2.decl: type = class_decl @Dest2 [concrete = constants.%Dest2] {} {}
// CHECK:STDOUT:   %ExplicitConstructor.decl: type = class_decl @ExplicitConstructor [concrete = constants.%ExplicitConstructor] {} {}
// CHECK:STDOUT:   %ExplicitConversion.decl: type = class_decl @ExplicitConversion [concrete = constants.%ExplicitConversion] {} {}
// CHECK:STDOUT:   %ExplicitConversion.cpp_operator.decl: %ExplicitConversion.cpp_operator.type = fn_decl @ExplicitConversion.cpp_operator [concrete = constants.%ExplicitConversion.cpp_operator] {
// CHECK:STDOUT:     %self.patt: %pattern_type.a70 = value_binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.a70 = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %ExplicitConversion = value_param call_param0
// CHECK:STDOUT:     %self: %ExplicitConversion = value_binding self, %self.param
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Dest__carbon_thunk.decl.b58ecd.2: %Dest__carbon_thunk.type.2ffc9e.2 = fn_decl @Dest__carbon_thunk.2 [concrete = constants.%Dest__carbon_thunk.f8fa06.2] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @UserConversion(%s.param: %Source) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %s.ref: %Source = name_ref s, %s
// CHECK:STDOUT:   %Cpp.ref.loc8: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %Dest.ref: type = name_ref Dest, imports.%Dest.decl [concrete = constants.%Dest.5e7]
// CHECK:STDOUT:   %Source.cpp_operator.bound: <bound method> = bound_method %s.ref, imports.%Source.cpp_operator.decl
// CHECK:STDOUT:   %.loc8_5.1: ref %Dest.5e7 = temporary_storage
// CHECK:STDOUT:   %addr: %ptr.551 = addr_of %.loc8_5.1
// CHECK:STDOUT:   %Dest__carbon_thunk.call: init %empty_tuple.type = call imports.%Dest__carbon_thunk.decl.b58ecd.1(%s.ref, %addr)
// CHECK:STDOUT:   %.loc8_5.2: init %Dest.5e7 to %.loc8_5.1 = in_place_init %Dest__carbon_thunk.call
// CHECK:STDOUT:   %.loc8_5.3: init %Dest.5e7 = converted %s.ref, %.loc8_5.2
// CHECK:STDOUT:   %.loc8_5.4: ref %Dest.5e7 = temporary %.loc8_5.1, %.loc8_5.3
// CHECK:STDOUT:   %Dest.cpp_destructor.bound: <bound method> = bound_method %.loc8_5.4, constants.%Dest.cpp_destructor
// CHECK:STDOUT:   %Dest.cpp_destructor.call: init %empty_tuple.type = call %Dest.cpp_destructor.bound(%.loc8_5.4)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @DestroyOp(%self.param: %Dest.5e7) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ConstructorConversion(%s.param: %Source2) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %s.ref: %Source2 = name_ref s, %s
// CHECK:STDOUT:   %Cpp.ref.loc27: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %Dest2.ref: type = name_ref Dest2, imports.%Dest2.decl [concrete = constants.%Dest2]
// CHECK:STDOUT:   %.loc27: %Dest2 = converted %s.ref, <error> [concrete = <error>]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ExplicitConstructorTest(%s.param: %Source) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %s.ref: %Source = name_ref s, %s
// CHECK:STDOUT:   %Cpp.ref.loc46: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %ExplicitConstructor.ref: type = name_ref ExplicitConstructor, imports.%ExplicitConstructor.decl [concrete = constants.%ExplicitConstructor]
// CHECK:STDOUT:   %.loc46: %ExplicitConstructor = converted %s.ref, <error> [concrete = <error>]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ExplicitConversionTest(%s.param: %ExplicitConversion) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %s.ref: %ExplicitConversion = name_ref s, %s
// CHECK:STDOUT:   %Cpp.ref.loc52: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %Dest.ref: type = name_ref Dest, imports.%Dest.decl [concrete = constants.%Dest.5e7]
// CHECK:STDOUT:   %ExplicitConversion.cpp_operator.bound: <bound method> = bound_method %s.ref, imports.%ExplicitConversion.cpp_operator.decl
// CHECK:STDOUT:   %.loc52_5.1: ref %Dest.5e7 = temporary_storage
// CHECK:STDOUT:   %addr: %ptr.551 = addr_of %.loc52_5.1
// CHECK:STDOUT:   %Dest__carbon_thunk.call: init %empty_tuple.type = call imports.%Dest__carbon_thunk.decl.b58ecd.2(%s.ref, %addr)
// CHECK:STDOUT:   %.loc52_5.2: init %Dest.5e7 to %.loc52_5.1 = in_place_init %Dest__carbon_thunk.call
// CHECK:STDOUT:   %.loc52_5.3: init %Dest.5e7 = converted %s.ref, %.loc52_5.2
// CHECK:STDOUT:   %.loc52_5.4: ref %Dest.5e7 = temporary %.loc52_5.1, %.loc52_5.3
// CHECK:STDOUT:   %Dest.cpp_destructor.bound: <bound method> = bound_method %.loc52_5.4, constants.%Dest.cpp_destructor
// CHECK:STDOUT:   %Dest.cpp_destructor.call: init %empty_tuple.type = call %Dest.cpp_destructor.bound(%.loc52_5.4)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- conditionally_explicit.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Source: type = class_type @Source [concrete]
// CHECK:STDOUT:   %ConditionallyExplicit.type: type = cpp_type_template_type ConditionallyExplicit [concrete]
// CHECK:STDOUT:   %ConditionallyExplicit.template: %ConditionallyExplicit.type = struct_value () [concrete]
// CHECK:STDOUT:   %false: bool = bool_literal false [concrete]
// CHECK:STDOUT:   %true: bool = bool_literal true [concrete]
// CHECK:STDOUT:   %Dest: type = class_type @Dest [concrete]
// CHECK:STDOUT:   %pattern_type.69a: type = pattern_type %Dest [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .Source = %Source.decl
// CHECK:STDOUT:     .ConditionallyExplicit = %ConditionallyExplicit.template
// CHECK:STDOUT:     .Dest = %Dest.decl
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Source.decl: type = class_decl @Source [concrete = constants.%Source] {} {}
// CHECK:STDOUT:   %ConditionallyExplicit.template: %ConditionallyExplicit.type = struct_value () [concrete = constants.%ConditionallyExplicit.template]
// CHECK:STDOUT:   %Dest.decl: type = class_decl @Dest [concrete = constants.%Dest] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ConstructorNotExplicit(%s.param: %Source) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %_.patt: <error> = value_binding_pattern _ [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %s.ref.loc12: %Source = name_ref s, %s
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %_: <error> = value_binding _, <error> [concrete = <error>]
// CHECK:STDOUT:   %s.ref.loc17: %Source = name_ref s, %s
// CHECK:STDOUT:   %Cpp.ref.loc17: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %ConditionallyExplicit.ref.loc17: %ConditionallyExplicit.type = name_ref ConditionallyExplicit, imports.%ConditionallyExplicit.template [concrete = constants.%ConditionallyExplicit.template]
// CHECK:STDOUT:   %false.loc17: bool = bool_literal false [concrete = constants.%false]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ConstructorExplicit(%s.param: %Source) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %s.ref: %Source = name_ref s, %s
// CHECK:STDOUT:   %Cpp.ref.loc27: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %ConditionallyExplicit.ref: %ConditionallyExplicit.type = name_ref ConditionallyExplicit, imports.%ConditionallyExplicit.template [concrete = constants.%ConditionallyExplicit.template]
// CHECK:STDOUT:   %true: bool = bool_literal true [concrete = constants.%true]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ConversionNotExplicit(%s.param: <error>) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %_.patt: %pattern_type.69a = value_binding_pattern _ [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %s.ref.loc37: <error> = name_ref s, %s
// CHECK:STDOUT:   %.loc37: type = splice_block %Dest.ref.loc37 [concrete = constants.%Dest] {
// CHECK:STDOUT:     %Cpp.ref.loc37: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %Dest.ref.loc37: type = name_ref Dest, imports.%Dest.decl [concrete = constants.%Dest]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %_: %Dest = value_binding _, <error> [concrete = <error>]
// CHECK:STDOUT:   %s.ref.loc38: <error> = name_ref s, %s
// CHECK:STDOUT:   %Cpp.ref.loc38: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %Dest.ref.loc38: type = name_ref Dest, imports.%Dest.decl [concrete = constants.%Dest]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ConversionExplicit(%s.param: <error>) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %s.ref: <error> = name_ref s, %s
// CHECK:STDOUT:   %Cpp.ref.loc48: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %Dest.ref: type = name_ref Dest, imports.%Dest.decl [concrete = constants.%Dest]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_conditionally_explicit_implicit.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Source: type = class_type @Source [concrete]
// CHECK:STDOUT:   %ConditionallyExplicit.type: type = cpp_type_template_type ConditionallyExplicit [concrete]
// CHECK:STDOUT:   %ConditionallyExplicit.template: %ConditionallyExplicit.type = struct_value () [concrete]
// CHECK:STDOUT:   %Dest: type = class_type @Dest [concrete]
// CHECK:STDOUT:   %pattern_type.69a: type = pattern_type %Dest [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .Source = %Source.decl
// CHECK:STDOUT:     .ConditionallyExplicit = %ConditionallyExplicit.template
// CHECK:STDOUT:     .Dest = %Dest.decl
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Source.decl: type = class_decl @Source [concrete = constants.%Source] {} {}
// CHECK:STDOUT:   %ConditionallyExplicit.template: %ConditionallyExplicit.type = struct_value () [concrete = constants.%ConditionallyExplicit.template]
// CHECK:STDOUT:   %Dest.decl: type = class_decl @Dest [concrete = constants.%Dest] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ConstructorExplicit(%s.param: %Source) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %_.patt: <error> = value_binding_pattern _ [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %s.ref: %Source = name_ref s, %s
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %_: <error> = value_binding _, <error> [concrete = <error>]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ConversionExplicit(%s.param: <error>) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %_.patt: %pattern_type.69a = value_binding_pattern _ [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %s.ref: <error> = name_ref s, %s
// CHECK:STDOUT:   %.loc22: type = splice_block %Dest.ref [concrete = constants.%Dest] {
// CHECK:STDOUT:     %Cpp.ref.loc22: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %Dest.ref: type = name_ref Dest, imports.%Dest.decl [concrete = constants.%Dest]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %_: %Dest = value_binding _, <error> [concrete = <error>]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
