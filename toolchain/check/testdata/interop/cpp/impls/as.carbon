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

using ConditionallyExplicitFalse = ConditionallyExplicit<false>;
using ConditionallyExplicitTrue = ConditionallyExplicit<true>;

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
  s as Cpp.Dest2;
  //@dump-sem-ir-end
}

fn ExplicitConstructorTest(s: Cpp.Source) {
  //@dump-sem-ir-begin
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
  let _: Cpp.ConditionallyExplicitFalse = s;
  s as Cpp.ConditionallyExplicitFalse;
  //@dump-sem-ir-end
}

fn ConstructorExplicit(s: Cpp.Source) {
  //@dump-sem-ir-begin
  s as Cpp.ConditionallyExplicitTrue;
  //@dump-sem-ir-end
}

fn ConversionNotExplicit(s: Cpp.ConditionallyExplicitFalse) {
  //@dump-sem-ir-begin
  let _: Cpp.Dest = s;
  s as Cpp.Dest;
  //@dump-sem-ir-end
}

fn ConversionExplicit(s: Cpp.ConditionallyExplicitFalse) {
  //@dump-sem-ir-begin
  s as Cpp.Dest;
  //@dump-sem-ir-end
}

// --- fail_conditionally_explicit_implicit.carbon

library "[[@TEST_NAME]]";

import Cpp library "conversions.h";

fn ConstructorExplicit(s: Cpp.Source) {
  //@dump-sem-ir-begin
  // CHECK:STDERR: fail_conditionally_explicit_implicit.carbon:[[@LINE+7]]:42: error: cannot implicitly convert expression of type `Cpp.Source` to `Cpp.ConditionallyExplicit` [ConversionFailure]
  // CHECK:STDERR:   let _: Cpp.ConditionallyExplicitTrue = s;
  // CHECK:STDERR:                                          ^
  // CHECK:STDERR: fail_conditionally_explicit_implicit.carbon:[[@LINE+4]]:42: note: type `Cpp.Source` does not implement interface `Core.ImplicitAs(Cpp.ConditionallyExplicit)` [MissingImplInMemberAccessInContext]
  // CHECK:STDERR:   let _: Cpp.ConditionallyExplicitTrue = s;
  // CHECK:STDERR:                                          ^
  // CHECK:STDERR:
  let _: Cpp.ConditionallyExplicitTrue = s;
  //@dump-sem-ir-end
}

fn ConversionExplicit(s: Cpp.ConditionallyExplicitFalse) {
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
// CHECK:STDOUT:   %Dest: type = class_type @Dest [concrete]
// CHECK:STDOUT:   %pattern_type.69a: type = pattern_type %Dest [concrete]
// CHECK:STDOUT:   %Source.cpp_operator.type: type = fn_type @Source.cpp_operator [concrete]
// CHECK:STDOUT:   %Source.cpp_operator: %Source.cpp_operator.type = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.551: type = ptr_type %Dest [concrete]
// CHECK:STDOUT:   %Dest__carbon_thunk.type.2ffc9e.1: type = fn_type @Dest__carbon_thunk.1 [concrete]
// CHECK:STDOUT:   %Dest__carbon_thunk.f8fa06.1: %Dest__carbon_thunk.type.2ffc9e.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Dest.cpp_destructor.type: type = fn_type @Dest.cpp_destructor [concrete]
// CHECK:STDOUT:   %Dest.cpp_destructor: %Dest.cpp_destructor.type = struct_value () [concrete]
// CHECK:STDOUT:   %Dest.Op.type: type = fn_type @Dest.Op [concrete]
// CHECK:STDOUT:   %Dest.Op: %Dest.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %Source2: type = class_type @Source2 [concrete]
// CHECK:STDOUT:   %Dest2: type = class_type @Dest2 [concrete]
// CHECK:STDOUT:   %pattern_type.39d: type = pattern_type %Dest2 [concrete]
// CHECK:STDOUT:   %ptr.472: type = ptr_type %Source2 [concrete]
// CHECK:STDOUT:   %ptr.9ae: type = ptr_type %Dest2 [concrete]
// CHECK:STDOUT:   %Dest2__carbon_thunk.type: type = fn_type @Dest2__carbon_thunk [concrete]
// CHECK:STDOUT:   %Dest2__carbon_thunk: %Dest2__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %Dest2.cpp_destructor.type: type = fn_type @Dest2.cpp_destructor [concrete]
// CHECK:STDOUT:   %Dest2.cpp_destructor: %Dest2.cpp_destructor.type = struct_value () [concrete]
// CHECK:STDOUT:   %Dest2.Op.type: type = fn_type @Dest2.Op [concrete]
// CHECK:STDOUT:   %Dest2.Op: %Dest2.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %ExplicitConstructor: type = class_type @ExplicitConstructor [concrete]
// CHECK:STDOUT:   %pattern_type.1eb: type = pattern_type %ExplicitConstructor [concrete]
// CHECK:STDOUT:   %ptr.1fc: type = ptr_type %Source [concrete]
// CHECK:STDOUT:   %ptr.5b5: type = ptr_type %ExplicitConstructor [concrete]
// CHECK:STDOUT:   %ExplicitConstructor__carbon_thunk.type: type = fn_type @ExplicitConstructor__carbon_thunk [concrete]
// CHECK:STDOUT:   %ExplicitConstructor__carbon_thunk: %ExplicitConstructor__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %ExplicitConstructor.cpp_destructor.type: type = fn_type @ExplicitConstructor.cpp_destructor [concrete]
// CHECK:STDOUT:   %ExplicitConstructor.cpp_destructor: %ExplicitConstructor.cpp_destructor.type = struct_value () [concrete]
// CHECK:STDOUT:   %ExplicitConstructor.Op.type: type = fn_type @ExplicitConstructor.Op [concrete]
// CHECK:STDOUT:   %ExplicitConstructor.Op: %ExplicitConstructor.Op.type = struct_value () [concrete]
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
// CHECK:STDOUT:   %Dest.decl: type = class_decl @Dest [concrete = constants.%Dest] {} {}
// CHECK:STDOUT:   %Source.cpp_operator.decl: %Source.cpp_operator.type = fn_decl @Source.cpp_operator [concrete = constants.%Source.cpp_operator] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.78c = value_param_pattern [concrete]
// CHECK:STDOUT:     %self.patt: %pattern_type.78c = at_binding_pattern self, %self.param_patt [concrete]
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
// CHECK:STDOUT:   %Dest.cpp_destructor.decl: %Dest.cpp_destructor.type = fn_decl @Dest.cpp_destructor [concrete = constants.%Dest.cpp_destructor] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.69a = ref_param_pattern [concrete]
// CHECK:STDOUT:     %self.patt: %pattern_type.69a = at_binding_pattern self, %self.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %Dest = ref_param call_param0
// CHECK:STDOUT:     %self: ref %Dest = ref_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Source2.decl: type = class_decl @Source2 [concrete = constants.%Source2] {} {}
// CHECK:STDOUT:   %Dest2.decl: type = class_decl @Dest2 [concrete = constants.%Dest2] {} {}
// CHECK:STDOUT:   %Dest2__carbon_thunk.decl: %Dest2__carbon_thunk.type = fn_decl @Dest2__carbon_thunk [concrete = constants.%Dest2__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Dest2.cpp_destructor.decl: %Dest2.cpp_destructor.type = fn_decl @Dest2.cpp_destructor [concrete = constants.%Dest2.cpp_destructor] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.39d = ref_param_pattern [concrete]
// CHECK:STDOUT:     %self.patt: %pattern_type.39d = at_binding_pattern self, %self.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %Dest2 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %Dest2 = ref_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ExplicitConstructor.decl: type = class_decl @ExplicitConstructor [concrete = constants.%ExplicitConstructor] {} {}
// CHECK:STDOUT:   %ExplicitConstructor__carbon_thunk.decl: %ExplicitConstructor__carbon_thunk.type = fn_decl @ExplicitConstructor__carbon_thunk [concrete = constants.%ExplicitConstructor__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ExplicitConstructor.cpp_destructor.decl: %ExplicitConstructor.cpp_destructor.type = fn_decl @ExplicitConstructor.cpp_destructor [concrete = constants.%ExplicitConstructor.cpp_destructor] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.1eb = ref_param_pattern [concrete]
// CHECK:STDOUT:     %self.patt: %pattern_type.1eb = at_binding_pattern self, %self.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %ExplicitConstructor = ref_param call_param0
// CHECK:STDOUT:     %self: ref %ExplicitConstructor = ref_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ExplicitConversion.decl: type = class_decl @ExplicitConversion [concrete = constants.%ExplicitConversion] {} {}
// CHECK:STDOUT:   %ExplicitConversion.cpp_operator.decl: %ExplicitConversion.cpp_operator.type = fn_decl @ExplicitConversion.cpp_operator [concrete = constants.%ExplicitConversion.cpp_operator] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.a70 = value_param_pattern [concrete]
// CHECK:STDOUT:     %self.patt: %pattern_type.a70 = at_binding_pattern self, %self.param_patt [concrete]
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
// CHECK:STDOUT:   %Cpp.ref.loc21: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %Dest.ref: type = name_ref Dest, imports.%Dest.decl [concrete = constants.%Dest]
// CHECK:STDOUT:   %Source.cpp_operator.bound: <bound method> = bound_method %s.ref, imports.%Source.cpp_operator.decl
// CHECK:STDOUT:   %.loc20.1: ref %Dest = temporary_storage
// CHECK:STDOUT:   %addr: %ptr.551 = addr_of %.loc20.1
// CHECK:STDOUT:   %Dest__carbon_thunk.call: init %empty_tuple.type = call imports.%Dest__carbon_thunk.decl.b58ecd.1(%s.ref, %addr)
// CHECK:STDOUT:   %.loc20.2: init %Dest to %.loc20.1 = mark_in_place_init %Dest__carbon_thunk.call
// CHECK:STDOUT:   %.loc20.3: init %Dest = converted %s.ref, %.loc20.2
// CHECK:STDOUT:   %.loc20.4: ref %Dest = temporary %.loc20.1, %.loc20.3
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %Dest.Op.bound: <bound method> = bound_method %.loc20.4, constants.%Dest.Op
// CHECK:STDOUT:   %Op.ref: %Dest.cpp_destructor.type = name_ref Op, imports.%Dest.cpp_destructor.decl [concrete = constants.%Dest.cpp_destructor]
// CHECK:STDOUT:   %Dest.cpp_destructor.bound: <bound method> = bound_method %.loc20.4, %Op.ref
// CHECK:STDOUT:   %Dest.cpp_destructor.call: init %empty_tuple.type = call %Dest.cpp_destructor.bound(%.loc20.4)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ConstructorConversion(%s.param: %Source2) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %s.ref: %Source2 = name_ref s, %s
// CHECK:STDOUT:   %Cpp.ref.loc38: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %Dest2.ref: type = name_ref Dest2, imports.%Dest2.decl [concrete = constants.%Dest2]
// CHECK:STDOUT:   %.loc37.1: ref %Dest2 = temporary_storage
// CHECK:STDOUT:   %.loc36: ref %Source2 = value_as_ref %s.ref
// CHECK:STDOUT:   %addr.loc37.1: %ptr.472 = addr_of %.loc36
// CHECK:STDOUT:   %addr.loc37.2: %ptr.9ae = addr_of %.loc37.1
// CHECK:STDOUT:   %Dest2__carbon_thunk.call: init %empty_tuple.type = call imports.%Dest2__carbon_thunk.decl(%addr.loc37.1, %addr.loc37.2)
// CHECK:STDOUT:   %.loc37.2: init %Dest2 to %.loc37.1 = mark_in_place_init %Dest2__carbon_thunk.call
// CHECK:STDOUT:   %.loc37.3: init %Dest2 = converted %s.ref, %.loc37.2
// CHECK:STDOUT:   %.loc37.4: ref %Dest2 = temporary %.loc37.1, %.loc37.3
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %Dest2.Op.bound: <bound method> = bound_method %.loc37.4, constants.%Dest2.Op
// CHECK:STDOUT:   %Op.ref: %Dest2.cpp_destructor.type = name_ref Op, imports.%Dest2.cpp_destructor.decl [concrete = constants.%Dest2.cpp_destructor]
// CHECK:STDOUT:   %Dest2.cpp_destructor.bound: <bound method> = bound_method %.loc37.4, %Op.ref
// CHECK:STDOUT:   %Dest2.cpp_destructor.call: init %empty_tuple.type = call %Dest2.cpp_destructor.bound(%.loc37.4)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ExplicitConstructorTest(%s.param: %Source) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %s.ref: %Source = name_ref s, %s
// CHECK:STDOUT:   %Cpp.ref.loc55: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %ExplicitConstructor.ref: type = name_ref ExplicitConstructor, imports.%ExplicitConstructor.decl [concrete = constants.%ExplicitConstructor]
// CHECK:STDOUT:   %.loc54.1: ref %ExplicitConstructor = temporary_storage
// CHECK:STDOUT:   %.loc53: ref %Source = value_as_ref %s.ref
// CHECK:STDOUT:   %addr.loc54.1: %ptr.1fc = addr_of %.loc53
// CHECK:STDOUT:   %addr.loc54.2: %ptr.5b5 = addr_of %.loc54.1
// CHECK:STDOUT:   %ExplicitConstructor__carbon_thunk.call: init %empty_tuple.type = call imports.%ExplicitConstructor__carbon_thunk.decl(%addr.loc54.1, %addr.loc54.2)
// CHECK:STDOUT:   %.loc54.2: init %ExplicitConstructor to %.loc54.1 = mark_in_place_init %ExplicitConstructor__carbon_thunk.call
// CHECK:STDOUT:   %.loc54.3: init %ExplicitConstructor = converted %s.ref, %.loc54.2
// CHECK:STDOUT:   %.loc54.4: ref %ExplicitConstructor = temporary %.loc54.1, %.loc54.3
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %ExplicitConstructor.Op.bound: <bound method> = bound_method %.loc54.4, constants.%ExplicitConstructor.Op
// CHECK:STDOUT:   %Op.ref: %ExplicitConstructor.cpp_destructor.type = name_ref Op, imports.%ExplicitConstructor.cpp_destructor.decl [concrete = constants.%ExplicitConstructor.cpp_destructor]
// CHECK:STDOUT:   %ExplicitConstructor.cpp_destructor.bound: <bound method> = bound_method %.loc54.4, %Op.ref
// CHECK:STDOUT:   %ExplicitConstructor.cpp_destructor.call: init %empty_tuple.type = call %ExplicitConstructor.cpp_destructor.bound(%.loc54.4)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ExplicitConversionTest(%s.param: %ExplicitConversion) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %s.ref: %ExplicitConversion = name_ref s, %s
// CHECK:STDOUT:   %Cpp.ref.loc72: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %Dest.ref: type = name_ref Dest, imports.%Dest.decl [concrete = constants.%Dest]
// CHECK:STDOUT:   %ExplicitConversion.cpp_operator.bound: <bound method> = bound_method %s.ref, imports.%ExplicitConversion.cpp_operator.decl
// CHECK:STDOUT:   %.loc71.1: ref %Dest = temporary_storage
// CHECK:STDOUT:   %addr: %ptr.551 = addr_of %.loc71.1
// CHECK:STDOUT:   %Dest__carbon_thunk.call: init %empty_tuple.type = call imports.%Dest__carbon_thunk.decl.b58ecd.2(%s.ref, %addr)
// CHECK:STDOUT:   %.loc71.2: init %Dest to %.loc71.1 = mark_in_place_init %Dest__carbon_thunk.call
// CHECK:STDOUT:   %.loc71.3: init %Dest = converted %s.ref, %.loc71.2
// CHECK:STDOUT:   %.loc71.4: ref %Dest = temporary %.loc71.1, %.loc71.3
// CHECK:STDOUT:   %Dest.Op.bound: <bound method> = bound_method %.loc71.4, constants.%Dest.Op
// CHECK:STDOUT:   %Op.ref: %Dest.cpp_destructor.type = name_ref Op, imports.%Dest.cpp_destructor.decl [concrete = constants.%Dest.cpp_destructor]
// CHECK:STDOUT:   %Dest.cpp_destructor.bound: <bound method> = bound_method %.loc71.4, %Op.ref
// CHECK:STDOUT:   %Dest.cpp_destructor.call: init %empty_tuple.type = call %Dest.cpp_destructor.bound(%.loc71.4)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- conditionally_explicit.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Source: type = class_type @Source [concrete]
// CHECK:STDOUT:   %ConditionallyExplicit.c52b91.1: type = class_type @ConditionallyExplicit.1 [concrete]
// CHECK:STDOUT:   %pattern_type.285f84.1: type = pattern_type %ConditionallyExplicit.c52b91.1 [concrete]
// CHECK:STDOUT:   %ptr.1fc: type = ptr_type %Source [concrete]
// CHECK:STDOUT:   %ptr.ca4178.1: type = ptr_type %ConditionallyExplicit.c52b91.1 [concrete]
// CHECK:STDOUT:   %ConditionallyExplicit__carbon_thunk.type.805b57.1: type = fn_type @ConditionallyExplicit__carbon_thunk.1 [concrete]
// CHECK:STDOUT:   %ConditionallyExplicit__carbon_thunk.b68c64.1: %ConditionallyExplicit__carbon_thunk.type.805b57.1 = struct_value () [concrete]
// CHECK:STDOUT:   %ConditionallyExplicit.cpp_destructor.type.a865b6.1: type = fn_type @ConditionallyExplicit.cpp_destructor.1 [concrete]
// CHECK:STDOUT:   %ConditionallyExplicit.cpp_destructor.1ba031.1: %ConditionallyExplicit.cpp_destructor.type.a865b6.1 = struct_value () [concrete]
// CHECK:STDOUT:   %ConditionallyExplicit.Op.type.6a678d.1: type = fn_type @ConditionallyExplicit.Op.1 [concrete]
// CHECK:STDOUT:   %ConditionallyExplicit.Op.c4e51d.1: %ConditionallyExplicit.Op.type.6a678d.1 = struct_value () [concrete]
// CHECK:STDOUT:   %ConditionallyExplicit.c52b91.2: type = class_type @ConditionallyExplicit.2 [concrete]
// CHECK:STDOUT:   %pattern_type.285f84.2: type = pattern_type %ConditionallyExplicit.c52b91.2 [concrete]
// CHECK:STDOUT:   %ptr.ca4178.2: type = ptr_type %ConditionallyExplicit.c52b91.2 [concrete]
// CHECK:STDOUT:   %ConditionallyExplicit__carbon_thunk.type.805b57.2: type = fn_type @ConditionallyExplicit__carbon_thunk.2 [concrete]
// CHECK:STDOUT:   %ConditionallyExplicit__carbon_thunk.b68c64.2: %ConditionallyExplicit__carbon_thunk.type.805b57.2 = struct_value () [concrete]
// CHECK:STDOUT:   %ConditionallyExplicit.cpp_destructor.type.a865b6.2: type = fn_type @ConditionallyExplicit.cpp_destructor.2 [concrete]
// CHECK:STDOUT:   %ConditionallyExplicit.cpp_destructor.1ba031.2: %ConditionallyExplicit.cpp_destructor.type.a865b6.2 = struct_value () [concrete]
// CHECK:STDOUT:   %ConditionallyExplicit.Op.type.6a678d.2: type = fn_type @ConditionallyExplicit.Op.2 [concrete]
// CHECK:STDOUT:   %ConditionallyExplicit.Op.c4e51d.2: %ConditionallyExplicit.Op.type.6a678d.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Dest: type = class_type @Dest [concrete]
// CHECK:STDOUT:   %pattern_type.69a: type = pattern_type %Dest [concrete]
// CHECK:STDOUT:   %ConditionallyExplicit.cpp_operator.type: type = fn_type @ConditionallyExplicit.cpp_operator [concrete]
// CHECK:STDOUT:   %ConditionallyExplicit.cpp_operator: %ConditionallyExplicit.cpp_operator.type = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.551: type = ptr_type %Dest [concrete]
// CHECK:STDOUT:   %Dest__carbon_thunk.type: type = fn_type @Dest__carbon_thunk [concrete]
// CHECK:STDOUT:   %Dest__carbon_thunk: %Dest__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %Dest.cpp_destructor.type: type = fn_type @Dest.cpp_destructor [concrete]
// CHECK:STDOUT:   %Dest.cpp_destructor: %Dest.cpp_destructor.type = struct_value () [concrete]
// CHECK:STDOUT:   %Dest.Op.type: type = fn_type @Dest.Op [concrete]
// CHECK:STDOUT:   %Dest.Op: %Dest.Op.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .Source = %Source.decl
// CHECK:STDOUT:     .ConditionallyExplicitFalse = %ConditionallyExplicit.decl.7acffe.1
// CHECK:STDOUT:     .ConditionallyExplicitTrue = %ConditionallyExplicit.decl.7acffe.2
// CHECK:STDOUT:     .Dest = %Dest.decl
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Source.decl: type = class_decl @Source [concrete = constants.%Source] {} {}
// CHECK:STDOUT:   %ConditionallyExplicit.decl.7acffe.1: type = class_decl @ConditionallyExplicit.1 [concrete = constants.%ConditionallyExplicit.c52b91.1] {} {}
// CHECK:STDOUT:   %ConditionallyExplicit__carbon_thunk.decl.22daaa.1: %ConditionallyExplicit__carbon_thunk.type.805b57.1 = fn_decl @ConditionallyExplicit__carbon_thunk.1 [concrete = constants.%ConditionallyExplicit__carbon_thunk.b68c64.1] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ConditionallyExplicit.cpp_destructor.decl.a8c6bd.1: %ConditionallyExplicit.cpp_destructor.type.a865b6.1 = fn_decl @ConditionallyExplicit.cpp_destructor.1 [concrete = constants.%ConditionallyExplicit.cpp_destructor.1ba031.1] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.285f84.1 = ref_param_pattern [concrete]
// CHECK:STDOUT:     %self.patt: %pattern_type.285f84.1 = at_binding_pattern self, %self.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %ConditionallyExplicit.c52b91.1 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %ConditionallyExplicit.c52b91.1 = ref_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ConditionallyExplicit.decl.7acffe.2: type = class_decl @ConditionallyExplicit.2 [concrete = constants.%ConditionallyExplicit.c52b91.2] {} {}
// CHECK:STDOUT:   %ConditionallyExplicit__carbon_thunk.decl.22daaa.2: %ConditionallyExplicit__carbon_thunk.type.805b57.2 = fn_decl @ConditionallyExplicit__carbon_thunk.2 [concrete = constants.%ConditionallyExplicit__carbon_thunk.b68c64.2] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ConditionallyExplicit.cpp_destructor.decl.a8c6bd.2: %ConditionallyExplicit.cpp_destructor.type.a865b6.2 = fn_decl @ConditionallyExplicit.cpp_destructor.2 [concrete = constants.%ConditionallyExplicit.cpp_destructor.1ba031.2] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.285f84.2 = ref_param_pattern [concrete]
// CHECK:STDOUT:     %self.patt: %pattern_type.285f84.2 = at_binding_pattern self, %self.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %ConditionallyExplicit.c52b91.2 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %ConditionallyExplicit.c52b91.2 = ref_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Dest.decl: type = class_decl @Dest [concrete = constants.%Dest] {} {}
// CHECK:STDOUT:   %ConditionallyExplicit.cpp_operator.decl: %ConditionallyExplicit.cpp_operator.type = fn_decl @ConditionallyExplicit.cpp_operator [concrete = constants.%ConditionallyExplicit.cpp_operator] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.285f84.1 = value_param_pattern [concrete]
// CHECK:STDOUT:     %self.patt: %pattern_type.285f84.1 = at_binding_pattern self, %self.param_patt [concrete]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %ConditionallyExplicit.c52b91.1 = value_param call_param0
// CHECK:STDOUT:     %self: %ConditionallyExplicit.c52b91.1 = value_binding self, %self.param
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Dest__carbon_thunk.decl: %Dest__carbon_thunk.type = fn_decl @Dest__carbon_thunk [concrete = constants.%Dest__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Dest.cpp_destructor.decl: %Dest.cpp_destructor.type = fn_decl @Dest.cpp_destructor [concrete = constants.%Dest.cpp_destructor] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.69a = ref_param_pattern [concrete]
// CHECK:STDOUT:     %self.patt: %pattern_type.69a = at_binding_pattern self, %self.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %Dest = ref_param call_param0
// CHECK:STDOUT:     %self: ref %Dest = ref_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ConstructorNotExplicit(%s.param: %Source) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %_.patt: %pattern_type.285f84.1 = value_binding_pattern _ [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %s.ref.loc26: %Source = name_ref s, %s
// CHECK:STDOUT:   %.loc23: type = splice_block %ConditionallyExplicitFalse.ref.loc23 [concrete = constants.%ConditionallyExplicit.c52b91.1] {
// CHECK:STDOUT:     %Cpp.ref.loc22: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %ConditionallyExplicitFalse.ref.loc23: type = name_ref ConditionallyExplicitFalse, imports.%ConditionallyExplicit.decl.7acffe.1 [concrete = constants.%ConditionallyExplicit.c52b91.1]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc26.1: ref %ConditionallyExplicit.c52b91.1 = temporary_storage
// CHECK:STDOUT:   %.loc26.2: ref %Source = value_as_ref %s.ref.loc26
// CHECK:STDOUT:   %addr.loc26.1: %ptr.1fc = addr_of %.loc26.2
// CHECK:STDOUT:   %addr.loc26.2: %ptr.ca4178.1 = addr_of %.loc26.1
// CHECK:STDOUT:   %ConditionallyExplicit__carbon_thunk.call.loc26: init %empty_tuple.type = call imports.%ConditionallyExplicit__carbon_thunk.decl.22daaa.1(%addr.loc26.1, %addr.loc26.2)
// CHECK:STDOUT:   %.loc26.3: init %ConditionallyExplicit.c52b91.1 to %.loc26.1 = mark_in_place_init %ConditionallyExplicit__carbon_thunk.call.loc26
// CHECK:STDOUT:   %.loc26.4: init %ConditionallyExplicit.c52b91.1 = converted %s.ref.loc26, %.loc26.3
// CHECK:STDOUT:   %.loc26.5: ref %ConditionallyExplicit.c52b91.1 = temporary %.loc26.1, %.loc26.4
// CHECK:STDOUT:   %.loc26.6: %ConditionallyExplicit.c52b91.1 = acquire_value %.loc26.5
// CHECK:STDOUT:   %_: %ConditionallyExplicit.c52b91.1 = value_binding _, %.loc26.6
// CHECK:STDOUT:   %s.ref.loc28: %Source = name_ref s, %s
// CHECK:STDOUT:   %Cpp.ref.loc30: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %ConditionallyExplicitFalse.ref.loc31: type = name_ref ConditionallyExplicitFalse, imports.%ConditionallyExplicit.decl.7acffe.1 [concrete = constants.%ConditionallyExplicit.c52b91.1]
// CHECK:STDOUT:   %.loc29.1: ref %ConditionallyExplicit.c52b91.1 = temporary_storage
// CHECK:STDOUT:   %.loc28: ref %Source = value_as_ref %s.ref.loc28
// CHECK:STDOUT:   %addr.loc29.1: %ptr.1fc = addr_of %.loc28
// CHECK:STDOUT:   %addr.loc29.2: %ptr.ca4178.1 = addr_of %.loc29.1
// CHECK:STDOUT:   %ConditionallyExplicit__carbon_thunk.call.loc29: init %empty_tuple.type = call imports.%ConditionallyExplicit__carbon_thunk.decl.22daaa.1(%addr.loc29.1, %addr.loc29.2)
// CHECK:STDOUT:   %.loc29.2: init %ConditionallyExplicit.c52b91.1 to %.loc29.1 = mark_in_place_init %ConditionallyExplicit__carbon_thunk.call.loc29
// CHECK:STDOUT:   %.loc29.3: init %ConditionallyExplicit.c52b91.1 = converted %s.ref.loc28, %.loc29.2
// CHECK:STDOUT:   %.loc29.4: ref %ConditionallyExplicit.c52b91.1 = temporary %.loc29.1, %.loc29.3
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %ConditionallyExplicit.Op.bound.loc29: <bound method> = bound_method %.loc29.4, constants.%ConditionallyExplicit.Op.c4e51d.1
// CHECK:STDOUT:   %Op.ref.loc29: %ConditionallyExplicit.cpp_destructor.type.a865b6.1 = name_ref Op, imports.%ConditionallyExplicit.cpp_destructor.decl.a8c6bd.1 [concrete = constants.%ConditionallyExplicit.cpp_destructor.1ba031.1]
// CHECK:STDOUT:   %ConditionallyExplicit.cpp_destructor.bound.loc29: <bound method> = bound_method %.loc29.4, %Op.ref.loc29
// CHECK:STDOUT:   %ConditionallyExplicit.cpp_destructor.call.loc29: init %empty_tuple.type = call %ConditionallyExplicit.cpp_destructor.bound.loc29(%.loc29.4)
// CHECK:STDOUT:   %ConditionallyExplicit.Op.bound.loc26: <bound method> = bound_method %.loc26.5, constants.%ConditionallyExplicit.Op.c4e51d.1
// CHECK:STDOUT:   %Op.ref.loc26: %ConditionallyExplicit.cpp_destructor.type.a865b6.1 = name_ref Op, imports.%ConditionallyExplicit.cpp_destructor.decl.a8c6bd.1 [concrete = constants.%ConditionallyExplicit.cpp_destructor.1ba031.1]
// CHECK:STDOUT:   %ConditionallyExplicit.cpp_destructor.bound.loc26: <bound method> = bound_method %.loc26.5, %Op.ref.loc26
// CHECK:STDOUT:   %ConditionallyExplicit.cpp_destructor.call.loc26: init %empty_tuple.type = call %ConditionallyExplicit.cpp_destructor.bound.loc26(%.loc26.5)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ConstructorExplicit(%s.param: %Source) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %s.ref: %Source = name_ref s, %s
// CHECK:STDOUT:   %Cpp.ref.loc47: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %ConditionallyExplicitTrue.ref: type = name_ref ConditionallyExplicitTrue, imports.%ConditionallyExplicit.decl.7acffe.2 [concrete = constants.%ConditionallyExplicit.c52b91.2]
// CHECK:STDOUT:   %.loc46.1: ref %ConditionallyExplicit.c52b91.2 = temporary_storage
// CHECK:STDOUT:   %.loc45: ref %Source = value_as_ref %s.ref
// CHECK:STDOUT:   %addr.loc46.1: %ptr.1fc = addr_of %.loc45
// CHECK:STDOUT:   %addr.loc46.2: %ptr.ca4178.2 = addr_of %.loc46.1
// CHECK:STDOUT:   %ConditionallyExplicit__carbon_thunk.call: init %empty_tuple.type = call imports.%ConditionallyExplicit__carbon_thunk.decl.22daaa.2(%addr.loc46.1, %addr.loc46.2)
// CHECK:STDOUT:   %.loc46.2: init %ConditionallyExplicit.c52b91.2 to %.loc46.1 = mark_in_place_init %ConditionallyExplicit__carbon_thunk.call
// CHECK:STDOUT:   %.loc46.3: init %ConditionallyExplicit.c52b91.2 = converted %s.ref, %.loc46.2
// CHECK:STDOUT:   %.loc46.4: ref %ConditionallyExplicit.c52b91.2 = temporary %.loc46.1, %.loc46.3
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %ConditionallyExplicit.Op.bound: <bound method> = bound_method %.loc46.4, constants.%ConditionallyExplicit.Op.c4e51d.2
// CHECK:STDOUT:   %Op.ref: %ConditionallyExplicit.cpp_destructor.type.a865b6.2 = name_ref Op, imports.%ConditionallyExplicit.cpp_destructor.decl.a8c6bd.2 [concrete = constants.%ConditionallyExplicit.cpp_destructor.1ba031.2]
// CHECK:STDOUT:   %ConditionallyExplicit.cpp_destructor.bound: <bound method> = bound_method %.loc46.4, %Op.ref
// CHECK:STDOUT:   %ConditionallyExplicit.cpp_destructor.call: init %empty_tuple.type = call %ConditionallyExplicit.cpp_destructor.bound(%.loc46.4)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ConversionNotExplicit(%s.param: %ConditionallyExplicit.c52b91.1) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %_.patt: %pattern_type.69a = value_binding_pattern _ [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %s.ref.loc69: %ConditionallyExplicit.c52b91.1 = name_ref s, %s
// CHECK:STDOUT:   %.loc66: type = splice_block %Dest.ref.loc66 [concrete = constants.%Dest] {
// CHECK:STDOUT:     %Cpp.ref.loc65: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %Dest.ref.loc66: type = name_ref Dest, imports.%Dest.decl [concrete = constants.%Dest]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ConditionallyExplicit.cpp_operator.bound.loc69: <bound method> = bound_method %s.ref.loc69, imports.%ConditionallyExplicit.cpp_operator.decl
// CHECK:STDOUT:   %.loc69.1: ref %Dest = temporary_storage
// CHECK:STDOUT:   %addr.loc69: %ptr.551 = addr_of %.loc69.1
// CHECK:STDOUT:   %Dest__carbon_thunk.call.loc69: init %empty_tuple.type = call imports.%Dest__carbon_thunk.decl(%s.ref.loc69, %addr.loc69)
// CHECK:STDOUT:   %.loc69.2: init %Dest to %.loc69.1 = mark_in_place_init %Dest__carbon_thunk.call.loc69
// CHECK:STDOUT:   %.loc69.3: init %Dest = converted %s.ref.loc69, %.loc69.2
// CHECK:STDOUT:   %.loc69.4: ref %Dest = temporary %.loc69.1, %.loc69.3
// CHECK:STDOUT:   %.loc69.5: %Dest = acquire_value %.loc69.4
// CHECK:STDOUT:   %_: %Dest = value_binding _, %.loc69.5
// CHECK:STDOUT:   %s.ref.loc71: %ConditionallyExplicit.c52b91.1 = name_ref s, %s
// CHECK:STDOUT:   %Cpp.ref.loc73: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %Dest.ref.loc74: type = name_ref Dest, imports.%Dest.decl [concrete = constants.%Dest]
// CHECK:STDOUT:   %ConditionallyExplicit.cpp_operator.bound.loc72: <bound method> = bound_method %s.ref.loc71, imports.%ConditionallyExplicit.cpp_operator.decl
// CHECK:STDOUT:   %.loc72.1: ref %Dest = temporary_storage
// CHECK:STDOUT:   %addr.loc72: %ptr.551 = addr_of %.loc72.1
// CHECK:STDOUT:   %Dest__carbon_thunk.call.loc72: init %empty_tuple.type = call imports.%Dest__carbon_thunk.decl(%s.ref.loc71, %addr.loc72)
// CHECK:STDOUT:   %.loc72.2: init %Dest to %.loc72.1 = mark_in_place_init %Dest__carbon_thunk.call.loc72
// CHECK:STDOUT:   %.loc72.3: init %Dest = converted %s.ref.loc71, %.loc72.2
// CHECK:STDOUT:   %.loc72.4: ref %Dest = temporary %.loc72.1, %.loc72.3
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %Dest.Op.bound.loc72: <bound method> = bound_method %.loc72.4, constants.%Dest.Op
// CHECK:STDOUT:   %Op.ref.loc72: %Dest.cpp_destructor.type = name_ref Op, imports.%Dest.cpp_destructor.decl [concrete = constants.%Dest.cpp_destructor]
// CHECK:STDOUT:   %Dest.cpp_destructor.bound.loc72: <bound method> = bound_method %.loc72.4, %Op.ref.loc72
// CHECK:STDOUT:   %Dest.cpp_destructor.call.loc72: init %empty_tuple.type = call %Dest.cpp_destructor.bound.loc72(%.loc72.4)
// CHECK:STDOUT:   %Dest.Op.bound.loc69: <bound method> = bound_method %.loc69.4, constants.%Dest.Op
// CHECK:STDOUT:   %Op.ref.loc69: %Dest.cpp_destructor.type = name_ref Op, imports.%Dest.cpp_destructor.decl [concrete = constants.%Dest.cpp_destructor]
// CHECK:STDOUT:   %Dest.cpp_destructor.bound.loc69: <bound method> = bound_method %.loc69.4, %Op.ref.loc69
// CHECK:STDOUT:   %Dest.cpp_destructor.call.loc69: init %empty_tuple.type = call %Dest.cpp_destructor.bound.loc69(%.loc69.4)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ConversionExplicit(%s.param: %ConditionallyExplicit.c52b91.1) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %s.ref: %ConditionallyExplicit.c52b91.1 = name_ref s, %s
// CHECK:STDOUT:   %Cpp.ref.loc90: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %Dest.ref: type = name_ref Dest, imports.%Dest.decl [concrete = constants.%Dest]
// CHECK:STDOUT:   %ConditionallyExplicit.cpp_operator.bound: <bound method> = bound_method %s.ref, imports.%ConditionallyExplicit.cpp_operator.decl
// CHECK:STDOUT:   %.loc89.1: ref %Dest = temporary_storage
// CHECK:STDOUT:   %addr: %ptr.551 = addr_of %.loc89.1
// CHECK:STDOUT:   %Dest__carbon_thunk.call: init %empty_tuple.type = call imports.%Dest__carbon_thunk.decl(%s.ref, %addr)
// CHECK:STDOUT:   %.loc89.2: init %Dest to %.loc89.1 = mark_in_place_init %Dest__carbon_thunk.call
// CHECK:STDOUT:   %.loc89.3: init %Dest = converted %s.ref, %.loc89.2
// CHECK:STDOUT:   %.loc89.4: ref %Dest = temporary %.loc89.1, %.loc89.3
// CHECK:STDOUT:   %Dest.Op.bound: <bound method> = bound_method %.loc89.4, constants.%Dest.Op
// CHECK:STDOUT:   %Op.ref: %Dest.cpp_destructor.type = name_ref Op, imports.%Dest.cpp_destructor.decl [concrete = constants.%Dest.cpp_destructor]
// CHECK:STDOUT:   %Dest.cpp_destructor.bound: <bound method> = bound_method %.loc89.4, %Op.ref
// CHECK:STDOUT:   %Dest.cpp_destructor.call: init %empty_tuple.type = call %Dest.cpp_destructor.bound(%.loc89.4)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_conditionally_explicit_implicit.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Source: type = class_type @Source [concrete]
// CHECK:STDOUT:   %ConditionallyExplicit.c52b91.1: type = class_type @ConditionallyExplicit.1 [concrete]
// CHECK:STDOUT:   %pattern_type.285f84.1: type = pattern_type %ConditionallyExplicit.c52b91.1 [concrete]
// CHECK:STDOUT:   %ConditionallyExplicit.c52b91.2: type = class_type @ConditionallyExplicit.2 [concrete]
// CHECK:STDOUT:   %pattern_type.285f84.2: type = pattern_type %ConditionallyExplicit.c52b91.2 [concrete]
// CHECK:STDOUT:   %Dest.5e7: type = class_type @Dest [concrete]
// CHECK:STDOUT:   %pattern_type.69a: type = pattern_type %Dest.5e7 [concrete]
// CHECK:STDOUT:   %ConditionallyExplicit.cpp_operator.type: type = fn_type @ConditionallyExplicit.cpp_operator [concrete]
// CHECK:STDOUT:   %ConditionallyExplicit.cpp_operator: %ConditionallyExplicit.cpp_operator.type = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.551: type = ptr_type %Dest.5e7 [concrete]
// CHECK:STDOUT:   %Dest__carbon_thunk.type: type = fn_type @Dest__carbon_thunk [concrete]
// CHECK:STDOUT:   %Dest__carbon_thunk: %Dest__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %Dest.cpp_destructor.type: type = fn_type @Dest.cpp_destructor [concrete]
// CHECK:STDOUT:   %Dest.cpp_destructor: %Dest.cpp_destructor.type = struct_value () [concrete]
// CHECK:STDOUT:   %Dest.Op.type: type = fn_type @Dest.Op [concrete]
// CHECK:STDOUT:   %Dest.Op: %Dest.Op.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .Source = %Source.decl
// CHECK:STDOUT:     .ConditionallyExplicitTrue = %ConditionallyExplicit.decl.7acffe.1
// CHECK:STDOUT:     .ConditionallyExplicitFalse = %ConditionallyExplicit.decl.7acffe.2
// CHECK:STDOUT:     .Dest = %Dest.decl
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Source.decl: type = class_decl @Source [concrete = constants.%Source] {} {}
// CHECK:STDOUT:   %ConditionallyExplicit.decl.7acffe.1: type = class_decl @ConditionallyExplicit.1 [concrete = constants.%ConditionallyExplicit.c52b91.1] {} {}
// CHECK:STDOUT:   %ConditionallyExplicit.decl.7acffe.2: type = class_decl @ConditionallyExplicit.2 [concrete = constants.%ConditionallyExplicit.c52b91.2] {} {}
// CHECK:STDOUT:   %Dest.decl: type = class_decl @Dest [concrete = constants.%Dest.5e7] {} {}
// CHECK:STDOUT:   %ConditionallyExplicit.cpp_operator.decl: %ConditionallyExplicit.cpp_operator.type = fn_decl @ConditionallyExplicit.cpp_operator [concrete = constants.%ConditionallyExplicit.cpp_operator] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.285f84.2 = value_param_pattern [concrete]
// CHECK:STDOUT:     %self.patt: %pattern_type.285f84.2 = at_binding_pattern self, %self.param_patt [concrete]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %ConditionallyExplicit.c52b91.2 = value_param call_param0
// CHECK:STDOUT:     %self: %ConditionallyExplicit.c52b91.2 = value_binding self, %self.param
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Dest__carbon_thunk.decl: %Dest__carbon_thunk.type = fn_decl @Dest__carbon_thunk [concrete = constants.%Dest__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Dest.cpp_destructor.decl: %Dest.cpp_destructor.type = fn_decl @Dest.cpp_destructor [concrete = constants.%Dest.cpp_destructor] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.69a = ref_param_pattern [concrete]
// CHECK:STDOUT:     %self.patt: %pattern_type.69a = at_binding_pattern self, %self.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %Dest.5e7 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %Dest.5e7 = ref_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ConstructorExplicit(%s.param: %Source) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %_.patt: %pattern_type.285f84.1 = value_binding_pattern _ [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %s.ref: %Source = name_ref s, %s
// CHECK:STDOUT:   %.loc23: type = splice_block %ConditionallyExplicitTrue.ref [concrete = constants.%ConditionallyExplicit.c52b91.1] {
// CHECK:STDOUT:     %Cpp.ref.loc22: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %ConditionallyExplicitTrue.ref: type = name_ref ConditionallyExplicitTrue, imports.%ConditionallyExplicit.decl.7acffe.1 [concrete = constants.%ConditionallyExplicit.c52b91.1]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc26: %ConditionallyExplicit.c52b91.1 = converted %s.ref, <error> [concrete = <error>]
// CHECK:STDOUT:   %_: %ConditionallyExplicit.c52b91.1 = value_binding _, <error> [concrete = <error>]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ConversionExplicit(%s.param: %ConditionallyExplicit.c52b91.2) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %_.patt: %pattern_type.69a = value_binding_pattern _ [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %s.ref: %ConditionallyExplicit.c52b91.2 = name_ref s, %s
// CHECK:STDOUT:   %.loc43: type = splice_block %Dest.ref [concrete = constants.%Dest.5e7] {
// CHECK:STDOUT:     %Cpp.ref.loc42: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %Dest.ref: type = name_ref Dest, imports.%Dest.decl [concrete = constants.%Dest.5e7]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ConditionallyExplicit.cpp_operator.bound: <bound method> = bound_method %s.ref, imports.%ConditionallyExplicit.cpp_operator.decl
// CHECK:STDOUT:   %.loc46.1: ref %Dest.5e7 = temporary_storage
// CHECK:STDOUT:   %addr: %ptr.551 = addr_of %.loc46.1
// CHECK:STDOUT:   %Dest__carbon_thunk.call: init %empty_tuple.type = call imports.%Dest__carbon_thunk.decl(%s.ref, %addr)
// CHECK:STDOUT:   %.loc46.2: init %Dest.5e7 to %.loc46.1 = mark_in_place_init %Dest__carbon_thunk.call
// CHECK:STDOUT:   %.loc46.3: init %Dest.5e7 = converted %s.ref, %.loc46.2
// CHECK:STDOUT:   %.loc46.4: ref %Dest.5e7 = temporary %.loc46.1, %.loc46.3
// CHECK:STDOUT:   %.loc46.5: %Dest.5e7 = acquire_value %.loc46.4
// CHECK:STDOUT:   %_: %Dest.5e7 = value_binding _, %.loc46.5
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %Dest.Op.bound: <bound method> = bound_method %.loc46.4, constants.%Dest.Op
// CHECK:STDOUT:   %Op.ref: %Dest.cpp_destructor.type = name_ref Op, imports.%Dest.cpp_destructor.decl [concrete = constants.%Dest.cpp_destructor]
// CHECK:STDOUT:   %Dest.cpp_destructor.bound: <bound method> = bound_method %.loc46.4, %Op.ref
// CHECK:STDOUT:   %Dest.cpp_destructor.call: init %empty_tuple.type = call %Dest.cpp_destructor.bound(%.loc46.4)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
