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
// CHECK:STDOUT:   %Cpp.ref.loc53: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %Dest.ref: type = name_ref Dest, imports.%Dest.decl [concrete = constants.%Dest]
// CHECK:STDOUT:   %Source.cpp_operator.bound: <bound method> = bound_method %s.ref, imports.%Source.cpp_operator.decl
// CHECK:STDOUT:   %.loc53_5.1: ref %Dest = temporary_storage
// CHECK:STDOUT:   %addr: %ptr.551 = addr_of %.loc53_5.1
// CHECK:STDOUT:   %Dest__carbon_thunk.call: init %empty_tuple.type = call imports.%Dest__carbon_thunk.decl.b58ecd.1(%s.ref, %addr)
// CHECK:STDOUT:   %.loc53_5.2: init %Dest to %.loc53_5.1 = mark_in_place_init %Dest__carbon_thunk.call
// CHECK:STDOUT:   %.loc53_5.3: init %Dest = converted %s.ref, %.loc53_5.2
// CHECK:STDOUT:   %.loc53_5.4: ref %Dest = temporary %.loc53_5.1, %.loc53_5.3
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %Dest.Op.bound: <bound method> = bound_method %.loc53_5.4, constants.%Dest.Op
// CHECK:STDOUT:   %Op.ref: %Dest.cpp_destructor.type = name_ref Op, imports.%Dest.cpp_destructor.decl [concrete = constants.%Dest.cpp_destructor]
// CHECK:STDOUT:   %Dest.cpp_destructor.bound: <bound method> = bound_method %.loc53_5.4, %Op.ref
// CHECK:STDOUT:   %Dest.cpp_destructor.call: init %empty_tuple.type = call %Dest.cpp_destructor.bound(%.loc53_5.4)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ConstructorConversion(%s.param: %Source2) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %s.ref: %Source2 = name_ref s, %s
// CHECK:STDOUT:   %Cpp.ref.loc59: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %Dest2.ref: type = name_ref Dest2, imports.%Dest2.decl [concrete = constants.%Dest2]
// CHECK:STDOUT:   %.loc59_5.1: ref %Dest2 = temporary_storage
// CHECK:STDOUT:   %.loc59_3: ref %Source2 = value_as_ref %s.ref
// CHECK:STDOUT:   %addr.loc59_5.1: %ptr.472 = addr_of %.loc59_3
// CHECK:STDOUT:   %addr.loc59_5.2: %ptr.9ae = addr_of %.loc59_5.1
// CHECK:STDOUT:   %Dest2__carbon_thunk.call: init %empty_tuple.type = call imports.%Dest2__carbon_thunk.decl(%addr.loc59_5.1, %addr.loc59_5.2)
// CHECK:STDOUT:   %.loc59_5.2: init %Dest2 to %.loc59_5.1 = mark_in_place_init %Dest2__carbon_thunk.call
// CHECK:STDOUT:   %.loc59_5.3: init %Dest2 = converted %s.ref, %.loc59_5.2
// CHECK:STDOUT:   %.loc59_5.4: ref %Dest2 = temporary %.loc59_5.1, %.loc59_5.3
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %Dest2.Op.bound: <bound method> = bound_method %.loc59_5.4, constants.%Dest2.Op
// CHECK:STDOUT:   %Op.ref: %Dest2.cpp_destructor.type = name_ref Op, imports.%Dest2.cpp_destructor.decl [concrete = constants.%Dest2.cpp_destructor]
// CHECK:STDOUT:   %Dest2.cpp_destructor.bound: <bound method> = bound_method %.loc59_5.4, %Op.ref
// CHECK:STDOUT:   %Dest2.cpp_destructor.call: init %empty_tuple.type = call %Dest2.cpp_destructor.bound(%.loc59_5.4)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ExplicitConstructorTest(%s.param: %Source) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %s.ref: %Source = name_ref s, %s
// CHECK:STDOUT:   %Cpp.ref.loc65: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %ExplicitConstructor.ref: type = name_ref ExplicitConstructor, imports.%ExplicitConstructor.decl [concrete = constants.%ExplicitConstructor]
// CHECK:STDOUT:   %.loc65_5.1: ref %ExplicitConstructor = temporary_storage
// CHECK:STDOUT:   %.loc65_3: ref %Source = value_as_ref %s.ref
// CHECK:STDOUT:   %addr.loc65_5.1: %ptr.1fc = addr_of %.loc65_3
// CHECK:STDOUT:   %addr.loc65_5.2: %ptr.5b5 = addr_of %.loc65_5.1
// CHECK:STDOUT:   %ExplicitConstructor__carbon_thunk.call: init %empty_tuple.type = call imports.%ExplicitConstructor__carbon_thunk.decl(%addr.loc65_5.1, %addr.loc65_5.2)
// CHECK:STDOUT:   %.loc65_5.2: init %ExplicitConstructor to %.loc65_5.1 = mark_in_place_init %ExplicitConstructor__carbon_thunk.call
// CHECK:STDOUT:   %.loc65_5.3: init %ExplicitConstructor = converted %s.ref, %.loc65_5.2
// CHECK:STDOUT:   %.loc65_5.4: ref %ExplicitConstructor = temporary %.loc65_5.1, %.loc65_5.3
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %ExplicitConstructor.Op.bound: <bound method> = bound_method %.loc65_5.4, constants.%ExplicitConstructor.Op
// CHECK:STDOUT:   %Op.ref: %ExplicitConstructor.cpp_destructor.type = name_ref Op, imports.%ExplicitConstructor.cpp_destructor.decl [concrete = constants.%ExplicitConstructor.cpp_destructor]
// CHECK:STDOUT:   %ExplicitConstructor.cpp_destructor.bound: <bound method> = bound_method %.loc65_5.4, %Op.ref
// CHECK:STDOUT:   %ExplicitConstructor.cpp_destructor.call: init %empty_tuple.type = call %ExplicitConstructor.cpp_destructor.bound(%.loc65_5.4)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ExplicitConversionTest(%s.param: %ExplicitConversion) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %s.ref: %ExplicitConversion = name_ref s, %s
// CHECK:STDOUT:   %Cpp.ref.loc71: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %Dest.ref: type = name_ref Dest, imports.%Dest.decl [concrete = constants.%Dest]
// CHECK:STDOUT:   %ExplicitConversion.cpp_operator.bound: <bound method> = bound_method %s.ref, imports.%ExplicitConversion.cpp_operator.decl
// CHECK:STDOUT:   %.loc71_5.1: ref %Dest = temporary_storage
// CHECK:STDOUT:   %addr: %ptr.551 = addr_of %.loc71_5.1
// CHECK:STDOUT:   %Dest__carbon_thunk.call: init %empty_tuple.type = call imports.%Dest__carbon_thunk.decl.b58ecd.2(%s.ref, %addr)
// CHECK:STDOUT:   %.loc71_5.2: init %Dest to %.loc71_5.1 = mark_in_place_init %Dest__carbon_thunk.call
// CHECK:STDOUT:   %.loc71_5.3: init %Dest = converted %s.ref, %.loc71_5.2
// CHECK:STDOUT:   %.loc71_5.4: ref %Dest = temporary %.loc71_5.1, %.loc71_5.3
// CHECK:STDOUT:   %Dest.Op.bound: <bound method> = bound_method %.loc71_5.4, constants.%Dest.Op
// CHECK:STDOUT:   %Op.ref: %Dest.cpp_destructor.type = name_ref Op, imports.%Dest.cpp_destructor.decl [concrete = constants.%Dest.cpp_destructor]
// CHECK:STDOUT:   %Dest.cpp_destructor.bound: <bound method> = bound_method %.loc71_5.4, %Op.ref
// CHECK:STDOUT:   %Dest.cpp_destructor.call: init %empty_tuple.type = call %Dest.cpp_destructor.bound(%.loc71_5.4)
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
// CHECK:STDOUT:   %s.ref.loc83: %Source = name_ref s, %s
// CHECK:STDOUT:   %.loc83_13: type = splice_block %ConditionallyExplicitFalse.ref.loc83 [concrete = constants.%ConditionallyExplicit.c52b91.1] {
// CHECK:STDOUT:     %Cpp.ref.loc83: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %ConditionallyExplicitFalse.ref.loc83: type = name_ref ConditionallyExplicitFalse, imports.%ConditionallyExplicit.decl.7acffe.1 [concrete = constants.%ConditionallyExplicit.c52b91.1]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc83_43.1: ref %ConditionallyExplicit.c52b91.1 = temporary_storage
// CHECK:STDOUT:   %.loc83_43.2: ref %Source = value_as_ref %s.ref.loc83
// CHECK:STDOUT:   %addr.loc83_43.1: %ptr.1fc = addr_of %.loc83_43.2
// CHECK:STDOUT:   %addr.loc83_43.2: %ptr.ca4178.1 = addr_of %.loc83_43.1
// CHECK:STDOUT:   %ConditionallyExplicit__carbon_thunk.call.loc83: init %empty_tuple.type = call imports.%ConditionallyExplicit__carbon_thunk.decl.22daaa.1(%addr.loc83_43.1, %addr.loc83_43.2)
// CHECK:STDOUT:   %.loc83_43.3: init %ConditionallyExplicit.c52b91.1 to %.loc83_43.1 = mark_in_place_init %ConditionallyExplicit__carbon_thunk.call.loc83
// CHECK:STDOUT:   %.loc83_43.4: init %ConditionallyExplicit.c52b91.1 = converted %s.ref.loc83, %.loc83_43.3
// CHECK:STDOUT:   %.loc83_43.5: ref %ConditionallyExplicit.c52b91.1 = temporary %.loc83_43.1, %.loc83_43.4
// CHECK:STDOUT:   %.loc83_43.6: %ConditionallyExplicit.c52b91.1 = acquire_value %.loc83_43.5
// CHECK:STDOUT:   %_: %ConditionallyExplicit.c52b91.1 = value_binding _, %.loc83_43.6
// CHECK:STDOUT:   %s.ref.loc84: %Source = name_ref s, %s
// CHECK:STDOUT:   %Cpp.ref.loc84: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %ConditionallyExplicitFalse.ref.loc84: type = name_ref ConditionallyExplicitFalse, imports.%ConditionallyExplicit.decl.7acffe.1 [concrete = constants.%ConditionallyExplicit.c52b91.1]
// CHECK:STDOUT:   %.loc84_5.1: ref %ConditionallyExplicit.c52b91.1 = temporary_storage
// CHECK:STDOUT:   %.loc84_3: ref %Source = value_as_ref %s.ref.loc84
// CHECK:STDOUT:   %addr.loc84_5.1: %ptr.1fc = addr_of %.loc84_3
// CHECK:STDOUT:   %addr.loc84_5.2: %ptr.ca4178.1 = addr_of %.loc84_5.1
// CHECK:STDOUT:   %ConditionallyExplicit__carbon_thunk.call.loc84: init %empty_tuple.type = call imports.%ConditionallyExplicit__carbon_thunk.decl.22daaa.1(%addr.loc84_5.1, %addr.loc84_5.2)
// CHECK:STDOUT:   %.loc84_5.2: init %ConditionallyExplicit.c52b91.1 to %.loc84_5.1 = mark_in_place_init %ConditionallyExplicit__carbon_thunk.call.loc84
// CHECK:STDOUT:   %.loc84_5.3: init %ConditionallyExplicit.c52b91.1 = converted %s.ref.loc84, %.loc84_5.2
// CHECK:STDOUT:   %.loc84_5.4: ref %ConditionallyExplicit.c52b91.1 = temporary %.loc84_5.1, %.loc84_5.3
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %ConditionallyExplicit.Op.bound.loc84: <bound method> = bound_method %.loc84_5.4, constants.%ConditionallyExplicit.Op.c4e51d.1
// CHECK:STDOUT:   %Op.ref.loc84: %ConditionallyExplicit.cpp_destructor.type.a865b6.1 = name_ref Op, imports.%ConditionallyExplicit.cpp_destructor.decl.a8c6bd.1 [concrete = constants.%ConditionallyExplicit.cpp_destructor.1ba031.1]
// CHECK:STDOUT:   %ConditionallyExplicit.cpp_destructor.bound.loc84: <bound method> = bound_method %.loc84_5.4, %Op.ref.loc84
// CHECK:STDOUT:   %ConditionallyExplicit.cpp_destructor.call.loc84: init %empty_tuple.type = call %ConditionallyExplicit.cpp_destructor.bound.loc84(%.loc84_5.4)
// CHECK:STDOUT:   %ConditionallyExplicit.Op.bound.loc83: <bound method> = bound_method %.loc83_43.5, constants.%ConditionallyExplicit.Op.c4e51d.1
// CHECK:STDOUT:   %Op.ref.loc83: %ConditionallyExplicit.cpp_destructor.type.a865b6.1 = name_ref Op, imports.%ConditionallyExplicit.cpp_destructor.decl.a8c6bd.1 [concrete = constants.%ConditionallyExplicit.cpp_destructor.1ba031.1]
// CHECK:STDOUT:   %ConditionallyExplicit.cpp_destructor.bound.loc83: <bound method> = bound_method %.loc83_43.5, %Op.ref.loc83
// CHECK:STDOUT:   %ConditionallyExplicit.cpp_destructor.call.loc83: init %empty_tuple.type = call %ConditionallyExplicit.cpp_destructor.bound.loc83(%.loc83_43.5)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ConstructorExplicit(%s.param: %Source) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %s.ref: %Source = name_ref s, %s
// CHECK:STDOUT:   %Cpp.ref.loc90: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %ConditionallyExplicitTrue.ref: type = name_ref ConditionallyExplicitTrue, imports.%ConditionallyExplicit.decl.7acffe.2 [concrete = constants.%ConditionallyExplicit.c52b91.2]
// CHECK:STDOUT:   %.loc90_5.1: ref %ConditionallyExplicit.c52b91.2 = temporary_storage
// CHECK:STDOUT:   %.loc90_3: ref %Source = value_as_ref %s.ref
// CHECK:STDOUT:   %addr.loc90_5.1: %ptr.1fc = addr_of %.loc90_3
// CHECK:STDOUT:   %addr.loc90_5.2: %ptr.ca4178.2 = addr_of %.loc90_5.1
// CHECK:STDOUT:   %ConditionallyExplicit__carbon_thunk.call: init %empty_tuple.type = call imports.%ConditionallyExplicit__carbon_thunk.decl.22daaa.2(%addr.loc90_5.1, %addr.loc90_5.2)
// CHECK:STDOUT:   %.loc90_5.2: init %ConditionallyExplicit.c52b91.2 to %.loc90_5.1 = mark_in_place_init %ConditionallyExplicit__carbon_thunk.call
// CHECK:STDOUT:   %.loc90_5.3: init %ConditionallyExplicit.c52b91.2 = converted %s.ref, %.loc90_5.2
// CHECK:STDOUT:   %.loc90_5.4: ref %ConditionallyExplicit.c52b91.2 = temporary %.loc90_5.1, %.loc90_5.3
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %ConditionallyExplicit.Op.bound: <bound method> = bound_method %.loc90_5.4, constants.%ConditionallyExplicit.Op.c4e51d.2
// CHECK:STDOUT:   %Op.ref: %ConditionallyExplicit.cpp_destructor.type.a865b6.2 = name_ref Op, imports.%ConditionallyExplicit.cpp_destructor.decl.a8c6bd.2 [concrete = constants.%ConditionallyExplicit.cpp_destructor.1ba031.2]
// CHECK:STDOUT:   %ConditionallyExplicit.cpp_destructor.bound: <bound method> = bound_method %.loc90_5.4, %Op.ref
// CHECK:STDOUT:   %ConditionallyExplicit.cpp_destructor.call: init %empty_tuple.type = call %ConditionallyExplicit.cpp_destructor.bound(%.loc90_5.4)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ConversionNotExplicit(%s.param: %ConditionallyExplicit.c52b91.1) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %_.patt: %pattern_type.69a = value_binding_pattern _ [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %s.ref.loc96: %ConditionallyExplicit.c52b91.1 = name_ref s, %s
// CHECK:STDOUT:   %.loc96_13: type = splice_block %Dest.ref.loc96 [concrete = constants.%Dest] {
// CHECK:STDOUT:     %Cpp.ref.loc96: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %Dest.ref.loc96: type = name_ref Dest, imports.%Dest.decl [concrete = constants.%Dest]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ConditionallyExplicit.cpp_operator.bound.loc96: <bound method> = bound_method %s.ref.loc96, imports.%ConditionallyExplicit.cpp_operator.decl
// CHECK:STDOUT:   %.loc96_21.1: ref %Dest = temporary_storage
// CHECK:STDOUT:   %addr.loc96: %ptr.551 = addr_of %.loc96_21.1
// CHECK:STDOUT:   %Dest__carbon_thunk.call.loc96: init %empty_tuple.type = call imports.%Dest__carbon_thunk.decl(%s.ref.loc96, %addr.loc96)
// CHECK:STDOUT:   %.loc96_21.2: init %Dest to %.loc96_21.1 = mark_in_place_init %Dest__carbon_thunk.call.loc96
// CHECK:STDOUT:   %.loc96_21.3: init %Dest = converted %s.ref.loc96, %.loc96_21.2
// CHECK:STDOUT:   %.loc96_21.4: ref %Dest = temporary %.loc96_21.1, %.loc96_21.3
// CHECK:STDOUT:   %.loc96_21.5: %Dest = acquire_value %.loc96_21.4
// CHECK:STDOUT:   %_: %Dest = value_binding _, %.loc96_21.5
// CHECK:STDOUT:   %s.ref.loc97: %ConditionallyExplicit.c52b91.1 = name_ref s, %s
// CHECK:STDOUT:   %Cpp.ref.loc97: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %Dest.ref.loc97: type = name_ref Dest, imports.%Dest.decl [concrete = constants.%Dest]
// CHECK:STDOUT:   %ConditionallyExplicit.cpp_operator.bound.loc97: <bound method> = bound_method %s.ref.loc97, imports.%ConditionallyExplicit.cpp_operator.decl
// CHECK:STDOUT:   %.loc97_5.1: ref %Dest = temporary_storage
// CHECK:STDOUT:   %addr.loc97: %ptr.551 = addr_of %.loc97_5.1
// CHECK:STDOUT:   %Dest__carbon_thunk.call.loc97: init %empty_tuple.type = call imports.%Dest__carbon_thunk.decl(%s.ref.loc97, %addr.loc97)
// CHECK:STDOUT:   %.loc97_5.2: init %Dest to %.loc97_5.1 = mark_in_place_init %Dest__carbon_thunk.call.loc97
// CHECK:STDOUT:   %.loc97_5.3: init %Dest = converted %s.ref.loc97, %.loc97_5.2
// CHECK:STDOUT:   %.loc97_5.4: ref %Dest = temporary %.loc97_5.1, %.loc97_5.3
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %Dest.Op.bound.loc97: <bound method> = bound_method %.loc97_5.4, constants.%Dest.Op
// CHECK:STDOUT:   %Op.ref.loc97: %Dest.cpp_destructor.type = name_ref Op, imports.%Dest.cpp_destructor.decl [concrete = constants.%Dest.cpp_destructor]
// CHECK:STDOUT:   %Dest.cpp_destructor.bound.loc97: <bound method> = bound_method %.loc97_5.4, %Op.ref.loc97
// CHECK:STDOUT:   %Dest.cpp_destructor.call.loc97: init %empty_tuple.type = call %Dest.cpp_destructor.bound.loc97(%.loc97_5.4)
// CHECK:STDOUT:   %Dest.Op.bound.loc96: <bound method> = bound_method %.loc96_21.4, constants.%Dest.Op
// CHECK:STDOUT:   %Op.ref.loc96: %Dest.cpp_destructor.type = name_ref Op, imports.%Dest.cpp_destructor.decl [concrete = constants.%Dest.cpp_destructor]
// CHECK:STDOUT:   %Dest.cpp_destructor.bound.loc96: <bound method> = bound_method %.loc96_21.4, %Op.ref.loc96
// CHECK:STDOUT:   %Dest.cpp_destructor.call.loc96: init %empty_tuple.type = call %Dest.cpp_destructor.bound.loc96(%.loc96_21.4)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ConversionExplicit(%s.param: %ConditionallyExplicit.c52b91.1) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %s.ref: %ConditionallyExplicit.c52b91.1 = name_ref s, %s
// CHECK:STDOUT:   %Cpp.ref.loc103: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %Dest.ref: type = name_ref Dest, imports.%Dest.decl [concrete = constants.%Dest]
// CHECK:STDOUT:   %ConditionallyExplicit.cpp_operator.bound: <bound method> = bound_method %s.ref, imports.%ConditionallyExplicit.cpp_operator.decl
// CHECK:STDOUT:   %.loc103_5.1: ref %Dest = temporary_storage
// CHECK:STDOUT:   %addr: %ptr.551 = addr_of %.loc103_5.1
// CHECK:STDOUT:   %Dest__carbon_thunk.call: init %empty_tuple.type = call imports.%Dest__carbon_thunk.decl(%s.ref, %addr)
// CHECK:STDOUT:   %.loc103_5.2: init %Dest to %.loc103_5.1 = mark_in_place_init %Dest__carbon_thunk.call
// CHECK:STDOUT:   %.loc103_5.3: init %Dest = converted %s.ref, %.loc103_5.2
// CHECK:STDOUT:   %.loc103_5.4: ref %Dest = temporary %.loc103_5.1, %.loc103_5.3
// CHECK:STDOUT:   %Dest.Op.bound: <bound method> = bound_method %.loc103_5.4, constants.%Dest.Op
// CHECK:STDOUT:   %Op.ref: %Dest.cpp_destructor.type = name_ref Op, imports.%Dest.cpp_destructor.decl [concrete = constants.%Dest.cpp_destructor]
// CHECK:STDOUT:   %Dest.cpp_destructor.bound: <bound method> = bound_method %.loc103_5.4, %Op.ref
// CHECK:STDOUT:   %Dest.cpp_destructor.call: init %empty_tuple.type = call %Dest.cpp_destructor.bound(%.loc103_5.4)
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
// CHECK:STDOUT:   %.loc122_13: type = splice_block %ConditionallyExplicitTrue.ref [concrete = constants.%ConditionallyExplicit.c52b91.1] {
// CHECK:STDOUT:     %Cpp.ref.loc122: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %ConditionallyExplicitTrue.ref: type = name_ref ConditionallyExplicitTrue, imports.%ConditionallyExplicit.decl.7acffe.1 [concrete = constants.%ConditionallyExplicit.c52b91.1]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc122_42: %ConditionallyExplicit.c52b91.1 = converted %s.ref, <error> [concrete = <error>]
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
// CHECK:STDOUT:   %.loc128_13: type = splice_block %Dest.ref [concrete = constants.%Dest.5e7] {
// CHECK:STDOUT:     %Cpp.ref.loc128: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %Dest.ref: type = name_ref Dest, imports.%Dest.decl [concrete = constants.%Dest.5e7]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ConditionallyExplicit.cpp_operator.bound: <bound method> = bound_method %s.ref, imports.%ConditionallyExplicit.cpp_operator.decl
// CHECK:STDOUT:   %.loc128_21.1: ref %Dest.5e7 = temporary_storage
// CHECK:STDOUT:   %addr: %ptr.551 = addr_of %.loc128_21.1
// CHECK:STDOUT:   %Dest__carbon_thunk.call: init %empty_tuple.type = call imports.%Dest__carbon_thunk.decl(%s.ref, %addr)
// CHECK:STDOUT:   %.loc128_21.2: init %Dest.5e7 to %.loc128_21.1 = mark_in_place_init %Dest__carbon_thunk.call
// CHECK:STDOUT:   %.loc128_21.3: init %Dest.5e7 = converted %s.ref, %.loc128_21.2
// CHECK:STDOUT:   %.loc128_21.4: ref %Dest.5e7 = temporary %.loc128_21.1, %.loc128_21.3
// CHECK:STDOUT:   %.loc128_21.5: %Dest.5e7 = acquire_value %.loc128_21.4
// CHECK:STDOUT:   %_: %Dest.5e7 = value_binding _, %.loc128_21.5
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %Dest.Op.bound: <bound method> = bound_method %.loc128_21.4, constants.%Dest.Op
// CHECK:STDOUT:   %Op.ref: %Dest.cpp_destructor.type = name_ref Op, imports.%Dest.cpp_destructor.decl [concrete = constants.%Dest.cpp_destructor]
// CHECK:STDOUT:   %Dest.cpp_destructor.bound: <bound method> = bound_method %.loc128_21.4, %Op.ref
// CHECK:STDOUT:   %Dest.cpp_destructor.call: init %empty_tuple.type = call %Dest.cpp_destructor.bound(%.loc128_21.4)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
