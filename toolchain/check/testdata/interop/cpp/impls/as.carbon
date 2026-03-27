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
// CHECK:STDOUT:   %.d95: ref %Dest = temporary invalid, @UserConversion.%.loc12_5.3 [concrete]
// CHECK:STDOUT:   %Dest.cpp_destructor.type: type = fn_type @Dest.cpp_destructor [concrete]
// CHECK:STDOUT:   %Dest.cpp_destructor: %Dest.cpp_destructor.type = struct_value () [concrete]
// CHECK:STDOUT:   %Source2: type = class_type @Source2 [concrete]
// CHECK:STDOUT:   %Dest2: type = class_type @Dest2 [concrete]
// CHECK:STDOUT:   %pattern_type.39d: type = pattern_type %Dest2 [concrete]
// CHECK:STDOUT:   %ptr.472: type = ptr_type %Source2 [concrete]
// CHECK:STDOUT:   %ptr.9ae: type = ptr_type %Dest2 [concrete]
// CHECK:STDOUT:   %Dest2__carbon_thunk.type: type = fn_type @Dest2__carbon_thunk [concrete]
// CHECK:STDOUT:   %Dest2__carbon_thunk: %Dest2__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %.7a8: ref %Dest2 = temporary invalid, @ConstructorConversion.%.loc22_5.3 [concrete]
// CHECK:STDOUT:   %Dest2.cpp_destructor.type: type = fn_type @Dest2.cpp_destructor [concrete]
// CHECK:STDOUT:   %Dest2.cpp_destructor: %Dest2.cpp_destructor.type = struct_value () [concrete]
// CHECK:STDOUT:   %ExplicitConstructor: type = class_type @ExplicitConstructor [concrete]
// CHECK:STDOUT:   %pattern_type.1eb: type = pattern_type %ExplicitConstructor [concrete]
// CHECK:STDOUT:   %ptr.1fc: type = ptr_type %Source [concrete]
// CHECK:STDOUT:   %ptr.5b5: type = ptr_type %ExplicitConstructor [concrete]
// CHECK:STDOUT:   %ExplicitConstructor__carbon_thunk.type: type = fn_type @ExplicitConstructor__carbon_thunk [concrete]
// CHECK:STDOUT:   %ExplicitConstructor__carbon_thunk: %ExplicitConstructor__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %.291: ref %ExplicitConstructor = temporary invalid, @ExplicitConstructorTest.%.loc32_5.3 [concrete]
// CHECK:STDOUT:   %ExplicitConstructor.cpp_destructor.type: type = fn_type @ExplicitConstructor.cpp_destructor [concrete]
// CHECK:STDOUT:   %ExplicitConstructor.cpp_destructor: %ExplicitConstructor.cpp_destructor.type = struct_value () [concrete]
// CHECK:STDOUT:   %ExplicitConversion: type = class_type @ExplicitConversion [concrete]
// CHECK:STDOUT:   %pattern_type.a70: type = pattern_type %ExplicitConversion [concrete]
// CHECK:STDOUT:   %ExplicitConversion.cpp_operator.type: type = fn_type @ExplicitConversion.cpp_operator [concrete]
// CHECK:STDOUT:   %ExplicitConversion.cpp_operator: %ExplicitConversion.cpp_operator.type = struct_value () [concrete]
// CHECK:STDOUT:   %Dest__carbon_thunk.type.2ffc9e.2: type = fn_type @Dest__carbon_thunk.2 [concrete]
// CHECK:STDOUT:   %Dest__carbon_thunk.f8fa06.2: %Dest__carbon_thunk.type.2ffc9e.2 = struct_value () [concrete]
// CHECK:STDOUT:   %.4f0: ref %Dest = temporary invalid, @ExplicitConversionTest.%.loc42_5.3 [concrete]
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
// CHECK:STDOUT:   %Cpp.ref.loc12: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %Dest.ref: type = name_ref Dest, imports.%Dest.decl [concrete = constants.%Dest]
// CHECK:STDOUT:   %Source.cpp_operator.bound: <bound method> = bound_method %s.ref, imports.%Source.cpp_operator.decl
// CHECK:STDOUT:   %.loc12_5.1: ref %Dest = temporary_storage
// CHECK:STDOUT:   %addr: %ptr.551 = addr_of %.loc12_5.1
// CHECK:STDOUT:   %Dest__carbon_thunk.call: init %empty_tuple.type = call imports.%Dest__carbon_thunk.decl.b58ecd.1(%s.ref, %addr)
// CHECK:STDOUT:   %.loc12_5.2: init %Dest to %.loc12_5.1 = mark_in_place_init %Dest__carbon_thunk.call
// CHECK:STDOUT:   %.loc12_5.3: init %Dest = converted %s.ref, %.loc12_5.2
// CHECK:STDOUT:   %.loc12_5.4: ref %Dest = temporary %.loc12_5.1, %.loc12_5.3 [concrete = constants.%.d95]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %Op.ref: %Dest.cpp_destructor.type = name_ref Op, imports.%Dest.cpp_destructor.decl [concrete = constants.%Dest.cpp_destructor]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ConstructorConversion(%s.param: %Source2) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %s.ref: %Source2 = name_ref s, %s
// CHECK:STDOUT:   %Cpp.ref.loc22: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %Dest2.ref: type = name_ref Dest2, imports.%Dest2.decl [concrete = constants.%Dest2]
// CHECK:STDOUT:   %.loc22_5.1: ref %Dest2 = temporary_storage
// CHECK:STDOUT:   %.loc22_3: ref %Source2 = value_as_ref %s.ref
// CHECK:STDOUT:   %addr.loc22_5.1: %ptr.472 = addr_of %.loc22_3
// CHECK:STDOUT:   %addr.loc22_5.2: %ptr.9ae = addr_of %.loc22_5.1
// CHECK:STDOUT:   %Dest2__carbon_thunk.call: init %empty_tuple.type = call imports.%Dest2__carbon_thunk.decl(%addr.loc22_5.1, %addr.loc22_5.2)
// CHECK:STDOUT:   %.loc22_5.2: init %Dest2 to %.loc22_5.1 = mark_in_place_init %Dest2__carbon_thunk.call
// CHECK:STDOUT:   %.loc22_5.3: init %Dest2 = converted %s.ref, %.loc22_5.2
// CHECK:STDOUT:   %.loc22_5.4: ref %Dest2 = temporary %.loc22_5.1, %.loc22_5.3 [concrete = constants.%.7a8]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %Op.ref: %Dest2.cpp_destructor.type = name_ref Op, imports.%Dest2.cpp_destructor.decl [concrete = constants.%Dest2.cpp_destructor]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ExplicitConstructorTest(%s.param: %Source) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %s.ref: %Source = name_ref s, %s
// CHECK:STDOUT:   %Cpp.ref.loc32: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %ExplicitConstructor.ref: type = name_ref ExplicitConstructor, imports.%ExplicitConstructor.decl [concrete = constants.%ExplicitConstructor]
// CHECK:STDOUT:   %.loc32_5.1: ref %ExplicitConstructor = temporary_storage
// CHECK:STDOUT:   %.loc32_3: ref %Source = value_as_ref %s.ref
// CHECK:STDOUT:   %addr.loc32_5.1: %ptr.1fc = addr_of %.loc32_3
// CHECK:STDOUT:   %addr.loc32_5.2: %ptr.5b5 = addr_of %.loc32_5.1
// CHECK:STDOUT:   %ExplicitConstructor__carbon_thunk.call: init %empty_tuple.type = call imports.%ExplicitConstructor__carbon_thunk.decl(%addr.loc32_5.1, %addr.loc32_5.2)
// CHECK:STDOUT:   %.loc32_5.2: init %ExplicitConstructor to %.loc32_5.1 = mark_in_place_init %ExplicitConstructor__carbon_thunk.call
// CHECK:STDOUT:   %.loc32_5.3: init %ExplicitConstructor = converted %s.ref, %.loc32_5.2
// CHECK:STDOUT:   %.loc32_5.4: ref %ExplicitConstructor = temporary %.loc32_5.1, %.loc32_5.3 [concrete = constants.%.291]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %Op.ref: %ExplicitConstructor.cpp_destructor.type = name_ref Op, imports.%ExplicitConstructor.cpp_destructor.decl [concrete = constants.%ExplicitConstructor.cpp_destructor]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ExplicitConversionTest(%s.param: %ExplicitConversion) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %s.ref: %ExplicitConversion = name_ref s, %s
// CHECK:STDOUT:   %Cpp.ref.loc42: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %Dest.ref: type = name_ref Dest, imports.%Dest.decl [concrete = constants.%Dest]
// CHECK:STDOUT:   %ExplicitConversion.cpp_operator.bound: <bound method> = bound_method %s.ref, imports.%ExplicitConversion.cpp_operator.decl
// CHECK:STDOUT:   %.loc42_5.1: ref %Dest = temporary_storage
// CHECK:STDOUT:   %addr: %ptr.551 = addr_of %.loc42_5.1
// CHECK:STDOUT:   %Dest__carbon_thunk.call: init %empty_tuple.type = call imports.%Dest__carbon_thunk.decl.b58ecd.2(%s.ref, %addr)
// CHECK:STDOUT:   %.loc42_5.2: init %Dest to %.loc42_5.1 = mark_in_place_init %Dest__carbon_thunk.call
// CHECK:STDOUT:   %.loc42_5.3: init %Dest = converted %s.ref, %.loc42_5.2
// CHECK:STDOUT:   %.loc42_5.4: ref %Dest = temporary %.loc42_5.1, %.loc42_5.3 [concrete = constants.%.4f0]
// CHECK:STDOUT:   %Op.ref: %Dest.cpp_destructor.type = name_ref Op, imports.%Dest.cpp_destructor.decl [concrete = constants.%Dest.cpp_destructor]
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
// CHECK:STDOUT:   %.8ca805.1: ref %ConditionallyExplicit.c52b91.1 = temporary invalid, @ConstructorNotExplicit.%.loc12_43.4 [concrete]
// CHECK:STDOUT:   %.8ca805.2: ref %ConditionallyExplicit.c52b91.1 = temporary invalid, @ConstructorNotExplicit.%.loc17_5.3 [concrete]
// CHECK:STDOUT:   %ConditionallyExplicit.cpp_destructor.type.a865b6.1: type = fn_type @ConditionallyExplicit.cpp_destructor.1 [concrete]
// CHECK:STDOUT:   %ConditionallyExplicit.cpp_destructor.1ba031.1: %ConditionallyExplicit.cpp_destructor.type.a865b6.1 = struct_value () [concrete]
// CHECK:STDOUT:   %ConditionallyExplicit.c52b91.2: type = class_type @ConditionallyExplicit.2 [concrete]
// CHECK:STDOUT:   %pattern_type.285f84.2: type = pattern_type %ConditionallyExplicit.c52b91.2 [concrete]
// CHECK:STDOUT:   %ptr.ca4178.2: type = ptr_type %ConditionallyExplicit.c52b91.2 [concrete]
// CHECK:STDOUT:   %ConditionallyExplicit__carbon_thunk.type.805b57.2: type = fn_type @ConditionallyExplicit__carbon_thunk.2 [concrete]
// CHECK:STDOUT:   %ConditionallyExplicit__carbon_thunk.b68c64.2: %ConditionallyExplicit__carbon_thunk.type.805b57.2 = struct_value () [concrete]
// CHECK:STDOUT:   %.8ca805.3: ref %ConditionallyExplicit.c52b91.2 = temporary invalid, @ConstructorExplicit.%.loc27_5.3 [concrete]
// CHECK:STDOUT:   %ConditionallyExplicit.cpp_destructor.type.a865b6.2: type = fn_type @ConditionallyExplicit.cpp_destructor.2 [concrete]
// CHECK:STDOUT:   %ConditionallyExplicit.cpp_destructor.1ba031.2: %ConditionallyExplicit.cpp_destructor.type.a865b6.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Dest: type = class_type @Dest [concrete]
// CHECK:STDOUT:   %pattern_type.69a: type = pattern_type %Dest [concrete]
// CHECK:STDOUT:   %ConditionallyExplicit.cpp_operator.type: type = fn_type @ConditionallyExplicit.cpp_operator [concrete]
// CHECK:STDOUT:   %ConditionallyExplicit.cpp_operator: %ConditionallyExplicit.cpp_operator.type = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.551: type = ptr_type %Dest [concrete]
// CHECK:STDOUT:   %Dest__carbon_thunk.type: type = fn_type @Dest__carbon_thunk [concrete]
// CHECK:STDOUT:   %Dest__carbon_thunk: %Dest__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %.21cafd.1: ref %Dest = temporary invalid, @ConversionNotExplicit.%.loc37_21.3 [concrete]
// CHECK:STDOUT:   %.21cafd.2: ref %Dest = temporary invalid, @ConversionNotExplicit.%.loc42_5.3 [concrete]
// CHECK:STDOUT:   %Dest.cpp_destructor.type: type = fn_type @Dest.cpp_destructor [concrete]
// CHECK:STDOUT:   %Dest.cpp_destructor: %Dest.cpp_destructor.type = struct_value () [concrete]
// CHECK:STDOUT:   %.21cafd.3: ref %Dest = temporary invalid, @ConversionExplicit.%.loc52_5.3 [concrete]
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
// CHECK:STDOUT:   %s.ref.loc12: %Source = name_ref s, %s
// CHECK:STDOUT:   %.loc12_13: type = splice_block %ConditionallyExplicitFalse.ref.loc12 [concrete = constants.%ConditionallyExplicit.c52b91.1] {
// CHECK:STDOUT:     %Cpp.ref.loc12: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %ConditionallyExplicitFalse.ref.loc12: type = name_ref ConditionallyExplicitFalse, imports.%ConditionallyExplicit.decl.7acffe.1 [concrete = constants.%ConditionallyExplicit.c52b91.1]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc12_43.1: ref %ConditionallyExplicit.c52b91.1 = temporary_storage
// CHECK:STDOUT:   %.loc12_43.2: ref %Source = value_as_ref %s.ref.loc12
// CHECK:STDOUT:   %addr.loc12_43.1: %ptr.1fc = addr_of %.loc12_43.2
// CHECK:STDOUT:   %addr.loc12_43.2: %ptr.ca4178.1 = addr_of %.loc12_43.1
// CHECK:STDOUT:   %ConditionallyExplicit__carbon_thunk.call.loc12: init %empty_tuple.type = call imports.%ConditionallyExplicit__carbon_thunk.decl.22daaa.1(%addr.loc12_43.1, %addr.loc12_43.2)
// CHECK:STDOUT:   %.loc12_43.3: init %ConditionallyExplicit.c52b91.1 to %.loc12_43.1 = mark_in_place_init %ConditionallyExplicit__carbon_thunk.call.loc12
// CHECK:STDOUT:   %.loc12_43.4: init %ConditionallyExplicit.c52b91.1 = converted %s.ref.loc12, %.loc12_43.3
// CHECK:STDOUT:   %.loc12_43.5: ref %ConditionallyExplicit.c52b91.1 = temporary %.loc12_43.1, %.loc12_43.4 [concrete = constants.%.8ca805.1]
// CHECK:STDOUT:   %.loc12_43.6: %ConditionallyExplicit.c52b91.1 = acquire_value %.loc12_43.5
// CHECK:STDOUT:   %_: %ConditionallyExplicit.c52b91.1 = value_binding _, %.loc12_43.6
// CHECK:STDOUT:   %s.ref.loc17: %Source = name_ref s, %s
// CHECK:STDOUT:   %Cpp.ref.loc17: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %ConditionallyExplicitFalse.ref.loc17: type = name_ref ConditionallyExplicitFalse, imports.%ConditionallyExplicit.decl.7acffe.1 [concrete = constants.%ConditionallyExplicit.c52b91.1]
// CHECK:STDOUT:   %.loc17_5.1: ref %ConditionallyExplicit.c52b91.1 = temporary_storage
// CHECK:STDOUT:   %.loc17_3: ref %Source = value_as_ref %s.ref.loc17
// CHECK:STDOUT:   %addr.loc17_5.1: %ptr.1fc = addr_of %.loc17_3
// CHECK:STDOUT:   %addr.loc17_5.2: %ptr.ca4178.1 = addr_of %.loc17_5.1
// CHECK:STDOUT:   %ConditionallyExplicit__carbon_thunk.call.loc17: init %empty_tuple.type = call imports.%ConditionallyExplicit__carbon_thunk.decl.22daaa.1(%addr.loc17_5.1, %addr.loc17_5.2)
// CHECK:STDOUT:   %.loc17_5.2: init %ConditionallyExplicit.c52b91.1 to %.loc17_5.1 = mark_in_place_init %ConditionallyExplicit__carbon_thunk.call.loc17
// CHECK:STDOUT:   %.loc17_5.3: init %ConditionallyExplicit.c52b91.1 = converted %s.ref.loc17, %.loc17_5.2
// CHECK:STDOUT:   %.loc17_5.4: ref %ConditionallyExplicit.c52b91.1 = temporary %.loc17_5.1, %.loc17_5.3 [concrete = constants.%.8ca805.2]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %Op.ref.loc17: %ConditionallyExplicit.cpp_destructor.type.a865b6.1 = name_ref Op, imports.%ConditionallyExplicit.cpp_destructor.decl.a8c6bd.1 [concrete = constants.%ConditionallyExplicit.cpp_destructor.1ba031.1]
// CHECK:STDOUT:   %Op.ref.loc12: %ConditionallyExplicit.cpp_destructor.type.a865b6.1 = name_ref Op, imports.%ConditionallyExplicit.cpp_destructor.decl.a8c6bd.1 [concrete = constants.%ConditionallyExplicit.cpp_destructor.1ba031.1]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ConstructorExplicit(%s.param: %Source) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %s.ref: %Source = name_ref s, %s
// CHECK:STDOUT:   %Cpp.ref.loc27: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %ConditionallyExplicitTrue.ref: type = name_ref ConditionallyExplicitTrue, imports.%ConditionallyExplicit.decl.7acffe.2 [concrete = constants.%ConditionallyExplicit.c52b91.2]
// CHECK:STDOUT:   %.loc27_5.1: ref %ConditionallyExplicit.c52b91.2 = temporary_storage
// CHECK:STDOUT:   %.loc27_3: ref %Source = value_as_ref %s.ref
// CHECK:STDOUT:   %addr.loc27_5.1: %ptr.1fc = addr_of %.loc27_3
// CHECK:STDOUT:   %addr.loc27_5.2: %ptr.ca4178.2 = addr_of %.loc27_5.1
// CHECK:STDOUT:   %ConditionallyExplicit__carbon_thunk.call: init %empty_tuple.type = call imports.%ConditionallyExplicit__carbon_thunk.decl.22daaa.2(%addr.loc27_5.1, %addr.loc27_5.2)
// CHECK:STDOUT:   %.loc27_5.2: init %ConditionallyExplicit.c52b91.2 to %.loc27_5.1 = mark_in_place_init %ConditionallyExplicit__carbon_thunk.call
// CHECK:STDOUT:   %.loc27_5.3: init %ConditionallyExplicit.c52b91.2 = converted %s.ref, %.loc27_5.2
// CHECK:STDOUT:   %.loc27_5.4: ref %ConditionallyExplicit.c52b91.2 = temporary %.loc27_5.1, %.loc27_5.3 [concrete = constants.%.8ca805.3]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %Op.ref: %ConditionallyExplicit.cpp_destructor.type.a865b6.2 = name_ref Op, imports.%ConditionallyExplicit.cpp_destructor.decl.a8c6bd.2 [concrete = constants.%ConditionallyExplicit.cpp_destructor.1ba031.2]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ConversionNotExplicit(%s.param: %ConditionallyExplicit.c52b91.1) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %_.patt: %pattern_type.69a = value_binding_pattern _ [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %s.ref.loc37: %ConditionallyExplicit.c52b91.1 = name_ref s, %s
// CHECK:STDOUT:   %.loc37_13: type = splice_block %Dest.ref.loc37 [concrete = constants.%Dest] {
// CHECK:STDOUT:     %Cpp.ref.loc37: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %Dest.ref.loc37: type = name_ref Dest, imports.%Dest.decl [concrete = constants.%Dest]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ConditionallyExplicit.cpp_operator.bound.loc37: <bound method> = bound_method %s.ref.loc37, imports.%ConditionallyExplicit.cpp_operator.decl
// CHECK:STDOUT:   %.loc37_21.1: ref %Dest = temporary_storage
// CHECK:STDOUT:   %addr.loc37: %ptr.551 = addr_of %.loc37_21.1
// CHECK:STDOUT:   %Dest__carbon_thunk.call.loc37: init %empty_tuple.type = call imports.%Dest__carbon_thunk.decl(%s.ref.loc37, %addr.loc37)
// CHECK:STDOUT:   %.loc37_21.2: init %Dest to %.loc37_21.1 = mark_in_place_init %Dest__carbon_thunk.call.loc37
// CHECK:STDOUT:   %.loc37_21.3: init %Dest = converted %s.ref.loc37, %.loc37_21.2
// CHECK:STDOUT:   %.loc37_21.4: ref %Dest = temporary %.loc37_21.1, %.loc37_21.3 [concrete = constants.%.21cafd.1]
// CHECK:STDOUT:   %.loc37_21.5: %Dest = acquire_value %.loc37_21.4
// CHECK:STDOUT:   %_: %Dest = value_binding _, %.loc37_21.5
// CHECK:STDOUT:   %s.ref.loc42: %ConditionallyExplicit.c52b91.1 = name_ref s, %s
// CHECK:STDOUT:   %Cpp.ref.loc42: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %Dest.ref.loc42: type = name_ref Dest, imports.%Dest.decl [concrete = constants.%Dest]
// CHECK:STDOUT:   %ConditionallyExplicit.cpp_operator.bound.loc42: <bound method> = bound_method %s.ref.loc42, imports.%ConditionallyExplicit.cpp_operator.decl
// CHECK:STDOUT:   %.loc42_5.1: ref %Dest = temporary_storage
// CHECK:STDOUT:   %addr.loc42: %ptr.551 = addr_of %.loc42_5.1
// CHECK:STDOUT:   %Dest__carbon_thunk.call.loc42: init %empty_tuple.type = call imports.%Dest__carbon_thunk.decl(%s.ref.loc42, %addr.loc42)
// CHECK:STDOUT:   %.loc42_5.2: init %Dest to %.loc42_5.1 = mark_in_place_init %Dest__carbon_thunk.call.loc42
// CHECK:STDOUT:   %.loc42_5.3: init %Dest = converted %s.ref.loc42, %.loc42_5.2
// CHECK:STDOUT:   %.loc42_5.4: ref %Dest = temporary %.loc42_5.1, %.loc42_5.3 [concrete = constants.%.21cafd.2]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %Op.ref.loc42: %Dest.cpp_destructor.type = name_ref Op, imports.%Dest.cpp_destructor.decl [concrete = constants.%Dest.cpp_destructor]
// CHECK:STDOUT:   %Op.ref.loc37: %Dest.cpp_destructor.type = name_ref Op, imports.%Dest.cpp_destructor.decl [concrete = constants.%Dest.cpp_destructor]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ConversionExplicit(%s.param: %ConditionallyExplicit.c52b91.1) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %s.ref: %ConditionallyExplicit.c52b91.1 = name_ref s, %s
// CHECK:STDOUT:   %Cpp.ref.loc52: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %Dest.ref: type = name_ref Dest, imports.%Dest.decl [concrete = constants.%Dest]
// CHECK:STDOUT:   %ConditionallyExplicit.cpp_operator.bound: <bound method> = bound_method %s.ref, imports.%ConditionallyExplicit.cpp_operator.decl
// CHECK:STDOUT:   %.loc52_5.1: ref %Dest = temporary_storage
// CHECK:STDOUT:   %addr: %ptr.551 = addr_of %.loc52_5.1
// CHECK:STDOUT:   %Dest__carbon_thunk.call: init %empty_tuple.type = call imports.%Dest__carbon_thunk.decl(%s.ref, %addr)
// CHECK:STDOUT:   %.loc52_5.2: init %Dest to %.loc52_5.1 = mark_in_place_init %Dest__carbon_thunk.call
// CHECK:STDOUT:   %.loc52_5.3: init %Dest = converted %s.ref, %.loc52_5.2
// CHECK:STDOUT:   %.loc52_5.4: ref %Dest = temporary %.loc52_5.1, %.loc52_5.3 [concrete = constants.%.21cafd.3]
// CHECK:STDOUT:   %Op.ref: %Dest.cpp_destructor.type = name_ref Op, imports.%Dest.cpp_destructor.decl [concrete = constants.%Dest.cpp_destructor]
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
// CHECK:STDOUT:   %.21c: ref %Dest.5e7 = temporary invalid, @ConversionExplicit.%.loc25_21.3 [concrete]
// CHECK:STDOUT:   %Dest.cpp_destructor.type: type = fn_type @Dest.cpp_destructor [concrete]
// CHECK:STDOUT:   %Dest.cpp_destructor: %Dest.cpp_destructor.type = struct_value () [concrete]
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
// CHECK:STDOUT:   %.loc15_13: type = splice_block %ConditionallyExplicitTrue.ref [concrete = constants.%ConditionallyExplicit.c52b91.1] {
// CHECK:STDOUT:     %Cpp.ref.loc15: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %ConditionallyExplicitTrue.ref: type = name_ref ConditionallyExplicitTrue, imports.%ConditionallyExplicit.decl.7acffe.1 [concrete = constants.%ConditionallyExplicit.c52b91.1]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc15_42: %ConditionallyExplicit.c52b91.1 = converted %s.ref, <error> [concrete = <error>]
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
// CHECK:STDOUT:   %.loc25_13: type = splice_block %Dest.ref [concrete = constants.%Dest.5e7] {
// CHECK:STDOUT:     %Cpp.ref.loc25: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %Dest.ref: type = name_ref Dest, imports.%Dest.decl [concrete = constants.%Dest.5e7]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ConditionallyExplicit.cpp_operator.bound: <bound method> = bound_method %s.ref, imports.%ConditionallyExplicit.cpp_operator.decl
// CHECK:STDOUT:   %.loc25_21.1: ref %Dest.5e7 = temporary_storage
// CHECK:STDOUT:   %addr: %ptr.551 = addr_of %.loc25_21.1
// CHECK:STDOUT:   %Dest__carbon_thunk.call: init %empty_tuple.type = call imports.%Dest__carbon_thunk.decl(%s.ref, %addr)
// CHECK:STDOUT:   %.loc25_21.2: init %Dest.5e7 to %.loc25_21.1 = mark_in_place_init %Dest__carbon_thunk.call
// CHECK:STDOUT:   %.loc25_21.3: init %Dest.5e7 = converted %s.ref, %.loc25_21.2
// CHECK:STDOUT:   %.loc25_21.4: ref %Dest.5e7 = temporary %.loc25_21.1, %.loc25_21.3 [concrete = constants.%.21c]
// CHECK:STDOUT:   %.loc25_21.5: %Dest.5e7 = acquire_value %.loc25_21.4
// CHECK:STDOUT:   %_: %Dest.5e7 = value_binding _, %.loc25_21.5
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %Op.ref: %Dest.cpp_destructor.type = name_ref Op, imports.%Dest.cpp_destructor.decl [concrete = constants.%Dest.cpp_destructor]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
