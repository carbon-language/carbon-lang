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
// CHECK:STDOUT:   %Source: type = class_type @Source [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Dest: type = class_type @Dest [concrete]
// CHECK:STDOUT:   %ptr.ca8: type = ptr_type %Dest [concrete]
// CHECK:STDOUT:   %Dest__carbon_thunk.type.eb4c5e.1: type = fn_type @Dest__carbon_thunk.1 [concrete]
// CHECK:STDOUT:   %Dest__carbon_thunk.c8e433.1: %Dest__carbon_thunk.type.eb4c5e.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Dest.cpp_destructor.type: type = fn_type @Dest.cpp_destructor [concrete]
// CHECK:STDOUT:   %Dest.cpp_destructor: %Dest.cpp_destructor.type = struct_value () [concrete]
// CHECK:STDOUT:   %Dest.Op.type: type = fn_type @Dest.Op [concrete]
// CHECK:STDOUT:   %Dest.Op: %Dest.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %Source2: type = class_type @Source2 [concrete]
// CHECK:STDOUT:   %Dest2: type = class_type @Dest2 [concrete]
// CHECK:STDOUT:   %ptr.0db: type = ptr_type %Source2 [concrete]
// CHECK:STDOUT:   %ptr.802: type = ptr_type %Dest2 [concrete]
// CHECK:STDOUT:   %Dest2__carbon_thunk.type: type = fn_type @Dest2__carbon_thunk [concrete]
// CHECK:STDOUT:   %Dest2__carbon_thunk: %Dest2__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %Dest2.cpp_destructor.type: type = fn_type @Dest2.cpp_destructor [concrete]
// CHECK:STDOUT:   %Dest2.cpp_destructor: %Dest2.cpp_destructor.type = struct_value () [concrete]
// CHECK:STDOUT:   %Dest2.Op.type: type = fn_type @Dest2.Op [concrete]
// CHECK:STDOUT:   %Dest2.Op: %Dest2.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %ExplicitConstructor: type = class_type @ExplicitConstructor [concrete]
// CHECK:STDOUT:   %ptr.5f6: type = ptr_type %Source [concrete]
// CHECK:STDOUT:   %ptr.44f: type = ptr_type %ExplicitConstructor [concrete]
// CHECK:STDOUT:   %ExplicitConstructor__carbon_thunk.type: type = fn_type @ExplicitConstructor__carbon_thunk [concrete]
// CHECK:STDOUT:   %ExplicitConstructor__carbon_thunk: %ExplicitConstructor__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %ExplicitConstructor.cpp_destructor.type: type = fn_type @ExplicitConstructor.cpp_destructor [concrete]
// CHECK:STDOUT:   %ExplicitConstructor.cpp_destructor: %ExplicitConstructor.cpp_destructor.type = struct_value () [concrete]
// CHECK:STDOUT:   %ExplicitConstructor.Op.type: type = fn_type @ExplicitConstructor.Op [concrete]
// CHECK:STDOUT:   %ExplicitConstructor.Op: %ExplicitConstructor.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %ExplicitConversion: type = class_type @ExplicitConversion [concrete]
// CHECK:STDOUT:   %Dest__carbon_thunk.type.eb4c5e.2: type = fn_type @Dest__carbon_thunk.2 [concrete]
// CHECK:STDOUT:   %Dest__carbon_thunk.c8e433.2: %Dest__carbon_thunk.type.eb4c5e.2 = struct_value () [concrete]
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
// CHECK:STDOUT:   %Dest__carbon_thunk.decl.a8fbc1.1: %Dest__carbon_thunk.type.eb4c5e.1 = fn_decl @Dest__carbon_thunk.1 [concrete = constants.%Dest__carbon_thunk.c8e433.1] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Dest.cpp_destructor.decl: %Dest.cpp_destructor.type = fn_decl @Dest.cpp_destructor [concrete = constants.%Dest.cpp_destructor] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Source2.decl: type = class_decl @Source2 [concrete = constants.%Source2] {} {}
// CHECK:STDOUT:   %Dest2.decl: type = class_decl @Dest2 [concrete = constants.%Dest2] {} {}
// CHECK:STDOUT:   %Dest2__carbon_thunk.decl: %Dest2__carbon_thunk.type = fn_decl @Dest2__carbon_thunk [concrete = constants.%Dest2__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Dest2.cpp_destructor.decl: %Dest2.cpp_destructor.type = fn_decl @Dest2.cpp_destructor [concrete = constants.%Dest2.cpp_destructor] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ExplicitConstructor.decl: type = class_decl @ExplicitConstructor [concrete = constants.%ExplicitConstructor] {} {}
// CHECK:STDOUT:   %ExplicitConstructor__carbon_thunk.decl: %ExplicitConstructor__carbon_thunk.type = fn_decl @ExplicitConstructor__carbon_thunk [concrete = constants.%ExplicitConstructor__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ExplicitConstructor.cpp_destructor.decl: %ExplicitConstructor.cpp_destructor.type = fn_decl @ExplicitConstructor.cpp_destructor [concrete = constants.%ExplicitConstructor.cpp_destructor] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ExplicitConversion.decl: type = class_decl @ExplicitConversion [concrete = constants.%ExplicitConversion] {} {}
// CHECK:STDOUT:   %Dest__carbon_thunk.decl.a8fbc1.2: %Dest__carbon_thunk.type.eb4c5e.2 = fn_decl @Dest__carbon_thunk.2 [concrete = constants.%Dest__carbon_thunk.c8e433.2] {
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
// CHECK:STDOUT:   %Dest.ref: type = name_ref Dest, imports.%Dest.decl [concrete = constants.%Dest]
// CHECK:STDOUT:   %.loc8_5.1: ref %Dest = temporary_storage
// CHECK:STDOUT:   %addr: %ptr.ca8 = addr_of %.loc8_5.1
// CHECK:STDOUT:   %Dest__carbon_thunk.call: init %empty_tuple.type = call imports.%Dest__carbon_thunk.decl.a8fbc1.1(%s.ref, %addr)
// CHECK:STDOUT:   %.loc8_5.2: init %Dest to %.loc8_5.1 = mark_in_place_init %Dest__carbon_thunk.call
// CHECK:STDOUT:   %.loc8_5.3: init %Dest = converted %s.ref, %.loc8_5.2
// CHECK:STDOUT:   %.loc8_5.4: ref %Dest = temporary %.loc8_5.1, %.loc8_5.3
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %Dest.Op.bound: <bound method> = bound_method %.loc8_5.4, constants.%Dest.Op
// CHECK:STDOUT:   %Op.ref: %Dest.cpp_destructor.type = name_ref Op, imports.%Dest.cpp_destructor.decl [concrete = constants.%Dest.cpp_destructor]
// CHECK:STDOUT:   %Dest.cpp_destructor.bound: <bound method> = bound_method %.loc8_5.4, %Op.ref
// CHECK:STDOUT:   %Dest.cpp_destructor.call: init %empty_tuple.type = call %Dest.cpp_destructor.bound(%.loc8_5.4)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ConstructorConversion(%s.param: %Source2) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %s.ref: %Source2 = name_ref s, %s
// CHECK:STDOUT:   %Cpp.ref.loc14: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %Dest2.ref: type = name_ref Dest2, imports.%Dest2.decl [concrete = constants.%Dest2]
// CHECK:STDOUT:   %.loc14_5.1: ref %Dest2 = temporary_storage
// CHECK:STDOUT:   %.loc14_3: ref %Source2 = value_as_ref %s.ref
// CHECK:STDOUT:   %addr.loc14_5.1: %ptr.0db = addr_of %.loc14_3
// CHECK:STDOUT:   %addr.loc14_5.2: %ptr.802 = addr_of %.loc14_5.1
// CHECK:STDOUT:   %Dest2__carbon_thunk.call: init %empty_tuple.type = call imports.%Dest2__carbon_thunk.decl(%addr.loc14_5.1, %addr.loc14_5.2)
// CHECK:STDOUT:   %.loc14_5.2: init %Dest2 to %.loc14_5.1 = mark_in_place_init %Dest2__carbon_thunk.call
// CHECK:STDOUT:   %.loc14_5.3: init %Dest2 = converted %s.ref, %.loc14_5.2
// CHECK:STDOUT:   %.loc14_5.4: ref %Dest2 = temporary %.loc14_5.1, %.loc14_5.3
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %Dest2.Op.bound: <bound method> = bound_method %.loc14_5.4, constants.%Dest2.Op
// CHECK:STDOUT:   %Op.ref: %Dest2.cpp_destructor.type = name_ref Op, imports.%Dest2.cpp_destructor.decl [concrete = constants.%Dest2.cpp_destructor]
// CHECK:STDOUT:   %Dest2.cpp_destructor.bound: <bound method> = bound_method %.loc14_5.4, %Op.ref
// CHECK:STDOUT:   %Dest2.cpp_destructor.call: init %empty_tuple.type = call %Dest2.cpp_destructor.bound(%.loc14_5.4)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ExplicitConstructorTest(%s.param: %Source) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %s.ref: %Source = name_ref s, %s
// CHECK:STDOUT:   %Cpp.ref.loc20: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %ExplicitConstructor.ref: type = name_ref ExplicitConstructor, imports.%ExplicitConstructor.decl [concrete = constants.%ExplicitConstructor]
// CHECK:STDOUT:   %.loc20_5.1: ref %ExplicitConstructor = temporary_storage
// CHECK:STDOUT:   %.loc20_3: ref %Source = value_as_ref %s.ref
// CHECK:STDOUT:   %addr.loc20_5.1: %ptr.5f6 = addr_of %.loc20_3
// CHECK:STDOUT:   %addr.loc20_5.2: %ptr.44f = addr_of %.loc20_5.1
// CHECK:STDOUT:   %ExplicitConstructor__carbon_thunk.call: init %empty_tuple.type = call imports.%ExplicitConstructor__carbon_thunk.decl(%addr.loc20_5.1, %addr.loc20_5.2)
// CHECK:STDOUT:   %.loc20_5.2: init %ExplicitConstructor to %.loc20_5.1 = mark_in_place_init %ExplicitConstructor__carbon_thunk.call
// CHECK:STDOUT:   %.loc20_5.3: init %ExplicitConstructor = converted %s.ref, %.loc20_5.2
// CHECK:STDOUT:   %.loc20_5.4: ref %ExplicitConstructor = temporary %.loc20_5.1, %.loc20_5.3
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %ExplicitConstructor.Op.bound: <bound method> = bound_method %.loc20_5.4, constants.%ExplicitConstructor.Op
// CHECK:STDOUT:   %Op.ref: %ExplicitConstructor.cpp_destructor.type = name_ref Op, imports.%ExplicitConstructor.cpp_destructor.decl [concrete = constants.%ExplicitConstructor.cpp_destructor]
// CHECK:STDOUT:   %ExplicitConstructor.cpp_destructor.bound: <bound method> = bound_method %.loc20_5.4, %Op.ref
// CHECK:STDOUT:   %ExplicitConstructor.cpp_destructor.call: init %empty_tuple.type = call %ExplicitConstructor.cpp_destructor.bound(%.loc20_5.4)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ExplicitConversionTest(%s.param: %ExplicitConversion) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %s.ref: %ExplicitConversion = name_ref s, %s
// CHECK:STDOUT:   %Cpp.ref.loc26: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %Dest.ref: type = name_ref Dest, imports.%Dest.decl [concrete = constants.%Dest]
// CHECK:STDOUT:   %.loc26_5.1: ref %Dest = temporary_storage
// CHECK:STDOUT:   %addr: %ptr.ca8 = addr_of %.loc26_5.1
// CHECK:STDOUT:   %Dest__carbon_thunk.call: init %empty_tuple.type = call imports.%Dest__carbon_thunk.decl.a8fbc1.2(%s.ref, %addr)
// CHECK:STDOUT:   %.loc26_5.2: init %Dest to %.loc26_5.1 = mark_in_place_init %Dest__carbon_thunk.call
// CHECK:STDOUT:   %.loc26_5.3: init %Dest = converted %s.ref, %.loc26_5.2
// CHECK:STDOUT:   %.loc26_5.4: ref %Dest = temporary %.loc26_5.1, %.loc26_5.3
// CHECK:STDOUT:   %Dest.Op.bound: <bound method> = bound_method %.loc26_5.4, constants.%Dest.Op
// CHECK:STDOUT:   %Op.ref: %Dest.cpp_destructor.type = name_ref Op, imports.%Dest.cpp_destructor.decl [concrete = constants.%Dest.cpp_destructor]
// CHECK:STDOUT:   %Dest.cpp_destructor.bound: <bound method> = bound_method %.loc26_5.4, %Op.ref
// CHECK:STDOUT:   %Dest.cpp_destructor.call: init %empty_tuple.type = call %Dest.cpp_destructor.bound(%.loc26_5.4)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- conditionally_explicit.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Source: type = class_type @Source [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %ConditionallyExplicit.3cd: type = class_type @ConditionallyExplicit.1 [concrete]
// CHECK:STDOUT:   %pattern_type.be2: type = pattern_type %ConditionallyExplicit.3cd [concrete]
// CHECK:STDOUT:   %_.patt.3ad: %pattern_type.be2 = value_binding_pattern _ [concrete]
// CHECK:STDOUT:   %ptr.5f6: type = ptr_type %Source [concrete]
// CHECK:STDOUT:   %ptr.bb5: type = ptr_type %ConditionallyExplicit.3cd [concrete]
// CHECK:STDOUT:   %ConditionallyExplicit__carbon_thunk.type.c9ab94.1: type = fn_type @ConditionallyExplicit__carbon_thunk.1 [concrete]
// CHECK:STDOUT:   %ConditionallyExplicit__carbon_thunk.924633.1: %ConditionallyExplicit__carbon_thunk.type.c9ab94.1 = struct_value () [concrete]
// CHECK:STDOUT:   %ConditionallyExplicit.cpp_destructor.type.634098.1: type = fn_type @ConditionallyExplicit.cpp_destructor.1 [concrete]
// CHECK:STDOUT:   %ConditionallyExplicit.cpp_destructor.5dc792.1: %ConditionallyExplicit.cpp_destructor.type.634098.1 = struct_value () [concrete]
// CHECK:STDOUT:   %ConditionallyExplicit.Op.type.83af6e.1: type = fn_type @ConditionallyExplicit.Op.1 [concrete]
// CHECK:STDOUT:   %ConditionallyExplicit.Op.4dca31.1: %ConditionallyExplicit.Op.type.83af6e.1 = struct_value () [concrete]
// CHECK:STDOUT:   %ConditionallyExplicit.9d4: type = class_type @ConditionallyExplicit.2 [concrete]
// CHECK:STDOUT:   %ptr.4c0: type = ptr_type %ConditionallyExplicit.9d4 [concrete]
// CHECK:STDOUT:   %ConditionallyExplicit__carbon_thunk.type.c9ab94.2: type = fn_type @ConditionallyExplicit__carbon_thunk.2 [concrete]
// CHECK:STDOUT:   %ConditionallyExplicit__carbon_thunk.924633.2: %ConditionallyExplicit__carbon_thunk.type.c9ab94.2 = struct_value () [concrete]
// CHECK:STDOUT:   %ConditionallyExplicit.cpp_destructor.type.634098.2: type = fn_type @ConditionallyExplicit.cpp_destructor.2 [concrete]
// CHECK:STDOUT:   %ConditionallyExplicit.cpp_destructor.5dc792.2: %ConditionallyExplicit.cpp_destructor.type.634098.2 = struct_value () [concrete]
// CHECK:STDOUT:   %ConditionallyExplicit.Op.type.83af6e.2: type = fn_type @ConditionallyExplicit.Op.2 [concrete]
// CHECK:STDOUT:   %ConditionallyExplicit.Op.4dca31.2: %ConditionallyExplicit.Op.type.83af6e.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Dest: type = class_type @Dest [concrete]
// CHECK:STDOUT:   %pattern_type.b72: type = pattern_type %Dest [concrete]
// CHECK:STDOUT:   %_.patt.664: %pattern_type.b72 = value_binding_pattern _ [concrete]
// CHECK:STDOUT:   %ptr.ca8: type = ptr_type %Dest [concrete]
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
// CHECK:STDOUT:     .ConditionallyExplicitFalse = %ConditionallyExplicit.decl.996
// CHECK:STDOUT:     .ConditionallyExplicitTrue = %ConditionallyExplicit.decl.6ab
// CHECK:STDOUT:     .Dest = %Dest.decl
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Source.decl: type = class_decl @Source [concrete = constants.%Source] {} {}
// CHECK:STDOUT:   %ConditionallyExplicit.decl.996: type = class_decl @ConditionallyExplicit.1 [concrete = constants.%ConditionallyExplicit.3cd] {} {}
// CHECK:STDOUT:   %ConditionallyExplicit__carbon_thunk.decl.761528.1: %ConditionallyExplicit__carbon_thunk.type.c9ab94.1 = fn_decl @ConditionallyExplicit__carbon_thunk.1 [concrete = constants.%ConditionallyExplicit__carbon_thunk.924633.1] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ConditionallyExplicit.cpp_destructor.decl.a7962a.1: %ConditionallyExplicit.cpp_destructor.type.634098.1 = fn_decl @ConditionallyExplicit.cpp_destructor.1 [concrete = constants.%ConditionallyExplicit.cpp_destructor.5dc792.1] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ConditionallyExplicit.decl.6ab: type = class_decl @ConditionallyExplicit.2 [concrete = constants.%ConditionallyExplicit.9d4] {} {}
// CHECK:STDOUT:   %ConditionallyExplicit__carbon_thunk.decl.761528.2: %ConditionallyExplicit__carbon_thunk.type.c9ab94.2 = fn_decl @ConditionallyExplicit__carbon_thunk.2 [concrete = constants.%ConditionallyExplicit__carbon_thunk.924633.2] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ConditionallyExplicit.cpp_destructor.decl.a7962a.2: %ConditionallyExplicit.cpp_destructor.type.634098.2 = fn_decl @ConditionallyExplicit.cpp_destructor.2 [concrete = constants.%ConditionallyExplicit.cpp_destructor.5dc792.2] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Dest.decl: type = class_decl @Dest [concrete = constants.%Dest] {} {}
// CHECK:STDOUT:   %Dest__carbon_thunk.decl: %Dest__carbon_thunk.type = fn_decl @Dest__carbon_thunk [concrete = constants.%Dest__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Dest.cpp_destructor.decl: %Dest.cpp_destructor.type = fn_decl @Dest.cpp_destructor [concrete = constants.%Dest.cpp_destructor] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ConstructorNotExplicit(%s.param: %Source) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %_.patt: %pattern_type.be2 = value_binding_pattern _ [concrete = constants.%_.patt.3ad]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %s.ref.loc8: %Source = name_ref s, %s
// CHECK:STDOUT:   %.loc8_43.1: ref %ConditionallyExplicit.3cd = temporary_storage
// CHECK:STDOUT:   %.loc8_43.2: ref %Source = value_as_ref %s.ref.loc8
// CHECK:STDOUT:   %addr.loc8_43.1: %ptr.5f6 = addr_of %.loc8_43.2
// CHECK:STDOUT:   %addr.loc8_43.2: %ptr.bb5 = addr_of %.loc8_43.1
// CHECK:STDOUT:   %ConditionallyExplicit__carbon_thunk.call.loc8: init %empty_tuple.type = call imports.%ConditionallyExplicit__carbon_thunk.decl.761528.1(%addr.loc8_43.1, %addr.loc8_43.2)
// CHECK:STDOUT:   %.loc8_43.3: init %ConditionallyExplicit.3cd to %.loc8_43.1 = mark_in_place_init %ConditionallyExplicit__carbon_thunk.call.loc8
// CHECK:STDOUT:   %.loc8_43.4: init %ConditionallyExplicit.3cd = converted %s.ref.loc8, %.loc8_43.3
// CHECK:STDOUT:   %.loc8_43.5: ref %ConditionallyExplicit.3cd = temporary %.loc8_43.1, %.loc8_43.4
// CHECK:STDOUT:   %.loc8_43.6: %ConditionallyExplicit.3cd = acquire_value %.loc8_43.5
// CHECK:STDOUT:   %.loc8_13: type = splice_block %ConditionallyExplicitFalse.ref.loc8 [concrete = constants.%ConditionallyExplicit.3cd] {
// CHECK:STDOUT:     %Cpp.ref.loc8: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %ConditionallyExplicitFalse.ref.loc8: type = name_ref ConditionallyExplicitFalse, imports.%ConditionallyExplicit.decl.996 [concrete = constants.%ConditionallyExplicit.3cd]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %_: %ConditionallyExplicit.3cd = wrapper_binding _, %.loc8_43.6
// CHECK:STDOUT:   %s.ref.loc9: %Source = name_ref s, %s
// CHECK:STDOUT:   %Cpp.ref.loc9: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %ConditionallyExplicitFalse.ref.loc9: type = name_ref ConditionallyExplicitFalse, imports.%ConditionallyExplicit.decl.996 [concrete = constants.%ConditionallyExplicit.3cd]
// CHECK:STDOUT:   %.loc9_5.1: ref %ConditionallyExplicit.3cd = temporary_storage
// CHECK:STDOUT:   %.loc9_3: ref %Source = value_as_ref %s.ref.loc9
// CHECK:STDOUT:   %addr.loc9_5.1: %ptr.5f6 = addr_of %.loc9_3
// CHECK:STDOUT:   %addr.loc9_5.2: %ptr.bb5 = addr_of %.loc9_5.1
// CHECK:STDOUT:   %ConditionallyExplicit__carbon_thunk.call.loc9: init %empty_tuple.type = call imports.%ConditionallyExplicit__carbon_thunk.decl.761528.1(%addr.loc9_5.1, %addr.loc9_5.2)
// CHECK:STDOUT:   %.loc9_5.2: init %ConditionallyExplicit.3cd to %.loc9_5.1 = mark_in_place_init %ConditionallyExplicit__carbon_thunk.call.loc9
// CHECK:STDOUT:   %.loc9_5.3: init %ConditionallyExplicit.3cd = converted %s.ref.loc9, %.loc9_5.2
// CHECK:STDOUT:   %.loc9_5.4: ref %ConditionallyExplicit.3cd = temporary %.loc9_5.1, %.loc9_5.3
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %ConditionallyExplicit.Op.bound.loc9: <bound method> = bound_method %.loc9_5.4, constants.%ConditionallyExplicit.Op.4dca31.1
// CHECK:STDOUT:   %Op.ref.loc9: %ConditionallyExplicit.cpp_destructor.type.634098.1 = name_ref Op, imports.%ConditionallyExplicit.cpp_destructor.decl.a7962a.1 [concrete = constants.%ConditionallyExplicit.cpp_destructor.5dc792.1]
// CHECK:STDOUT:   %ConditionallyExplicit.cpp_destructor.bound.loc9: <bound method> = bound_method %.loc9_5.4, %Op.ref.loc9
// CHECK:STDOUT:   %ConditionallyExplicit.cpp_destructor.call.loc9: init %empty_tuple.type = call %ConditionallyExplicit.cpp_destructor.bound.loc9(%.loc9_5.4)
// CHECK:STDOUT:   %ConditionallyExplicit.Op.bound.loc8: <bound method> = bound_method %.loc8_43.5, constants.%ConditionallyExplicit.Op.4dca31.1
// CHECK:STDOUT:   %Op.ref.loc8: %ConditionallyExplicit.cpp_destructor.type.634098.1 = name_ref Op, imports.%ConditionallyExplicit.cpp_destructor.decl.a7962a.1 [concrete = constants.%ConditionallyExplicit.cpp_destructor.5dc792.1]
// CHECK:STDOUT:   %ConditionallyExplicit.cpp_destructor.bound.loc8: <bound method> = bound_method %.loc8_43.5, %Op.ref.loc8
// CHECK:STDOUT:   %ConditionallyExplicit.cpp_destructor.call.loc8: init %empty_tuple.type = call %ConditionallyExplicit.cpp_destructor.bound.loc8(%.loc8_43.5)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ConstructorExplicit(%s.param: %Source) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %s.ref: %Source = name_ref s, %s
// CHECK:STDOUT:   %Cpp.ref.loc15: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %ConditionallyExplicitTrue.ref: type = name_ref ConditionallyExplicitTrue, imports.%ConditionallyExplicit.decl.6ab [concrete = constants.%ConditionallyExplicit.9d4]
// CHECK:STDOUT:   %.loc15_5.1: ref %ConditionallyExplicit.9d4 = temporary_storage
// CHECK:STDOUT:   %.loc15_3: ref %Source = value_as_ref %s.ref
// CHECK:STDOUT:   %addr.loc15_5.1: %ptr.5f6 = addr_of %.loc15_3
// CHECK:STDOUT:   %addr.loc15_5.2: %ptr.4c0 = addr_of %.loc15_5.1
// CHECK:STDOUT:   %ConditionallyExplicit__carbon_thunk.call: init %empty_tuple.type = call imports.%ConditionallyExplicit__carbon_thunk.decl.761528.2(%addr.loc15_5.1, %addr.loc15_5.2)
// CHECK:STDOUT:   %.loc15_5.2: init %ConditionallyExplicit.9d4 to %.loc15_5.1 = mark_in_place_init %ConditionallyExplicit__carbon_thunk.call
// CHECK:STDOUT:   %.loc15_5.3: init %ConditionallyExplicit.9d4 = converted %s.ref, %.loc15_5.2
// CHECK:STDOUT:   %.loc15_5.4: ref %ConditionallyExplicit.9d4 = temporary %.loc15_5.1, %.loc15_5.3
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %ConditionallyExplicit.Op.bound: <bound method> = bound_method %.loc15_5.4, constants.%ConditionallyExplicit.Op.4dca31.2
// CHECK:STDOUT:   %Op.ref: %ConditionallyExplicit.cpp_destructor.type.634098.2 = name_ref Op, imports.%ConditionallyExplicit.cpp_destructor.decl.a7962a.2 [concrete = constants.%ConditionallyExplicit.cpp_destructor.5dc792.2]
// CHECK:STDOUT:   %ConditionallyExplicit.cpp_destructor.bound: <bound method> = bound_method %.loc15_5.4, %Op.ref
// CHECK:STDOUT:   %ConditionallyExplicit.cpp_destructor.call: init %empty_tuple.type = call %ConditionallyExplicit.cpp_destructor.bound(%.loc15_5.4)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ConversionNotExplicit(%s.param: %ConditionallyExplicit.3cd) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %_.patt: %pattern_type.b72 = value_binding_pattern _ [concrete = constants.%_.patt.664]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %s.ref.loc21: %ConditionallyExplicit.3cd = name_ref s, %s
// CHECK:STDOUT:   %.loc21_21.1: ref %Dest = temporary_storage
// CHECK:STDOUT:   %addr.loc21: %ptr.ca8 = addr_of %.loc21_21.1
// CHECK:STDOUT:   %Dest__carbon_thunk.call.loc21: init %empty_tuple.type = call imports.%Dest__carbon_thunk.decl(%s.ref.loc21, %addr.loc21)
// CHECK:STDOUT:   %.loc21_21.2: init %Dest to %.loc21_21.1 = mark_in_place_init %Dest__carbon_thunk.call.loc21
// CHECK:STDOUT:   %.loc21_21.3: init %Dest = converted %s.ref.loc21, %.loc21_21.2
// CHECK:STDOUT:   %.loc21_21.4: ref %Dest = temporary %.loc21_21.1, %.loc21_21.3
// CHECK:STDOUT:   %.loc21_21.5: %Dest = acquire_value %.loc21_21.4
// CHECK:STDOUT:   %.loc21_13: type = splice_block %Dest.ref.loc21 [concrete = constants.%Dest] {
// CHECK:STDOUT:     %Cpp.ref.loc21: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %Dest.ref.loc21: type = name_ref Dest, imports.%Dest.decl [concrete = constants.%Dest]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %_: %Dest = wrapper_binding _, %.loc21_21.5
// CHECK:STDOUT:   %s.ref.loc22: %ConditionallyExplicit.3cd = name_ref s, %s
// CHECK:STDOUT:   %Cpp.ref.loc22: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %Dest.ref.loc22: type = name_ref Dest, imports.%Dest.decl [concrete = constants.%Dest]
// CHECK:STDOUT:   %.loc22_5.1: ref %Dest = temporary_storage
// CHECK:STDOUT:   %addr.loc22: %ptr.ca8 = addr_of %.loc22_5.1
// CHECK:STDOUT:   %Dest__carbon_thunk.call.loc22: init %empty_tuple.type = call imports.%Dest__carbon_thunk.decl(%s.ref.loc22, %addr.loc22)
// CHECK:STDOUT:   %.loc22_5.2: init %Dest to %.loc22_5.1 = mark_in_place_init %Dest__carbon_thunk.call.loc22
// CHECK:STDOUT:   %.loc22_5.3: init %Dest = converted %s.ref.loc22, %.loc22_5.2
// CHECK:STDOUT:   %.loc22_5.4: ref %Dest = temporary %.loc22_5.1, %.loc22_5.3
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %Dest.Op.bound.loc22: <bound method> = bound_method %.loc22_5.4, constants.%Dest.Op
// CHECK:STDOUT:   %Op.ref.loc22: %Dest.cpp_destructor.type = name_ref Op, imports.%Dest.cpp_destructor.decl [concrete = constants.%Dest.cpp_destructor]
// CHECK:STDOUT:   %Dest.cpp_destructor.bound.loc22: <bound method> = bound_method %.loc22_5.4, %Op.ref.loc22
// CHECK:STDOUT:   %Dest.cpp_destructor.call.loc22: init %empty_tuple.type = call %Dest.cpp_destructor.bound.loc22(%.loc22_5.4)
// CHECK:STDOUT:   %Dest.Op.bound.loc21: <bound method> = bound_method %.loc21_21.4, constants.%Dest.Op
// CHECK:STDOUT:   %Op.ref.loc21: %Dest.cpp_destructor.type = name_ref Op, imports.%Dest.cpp_destructor.decl [concrete = constants.%Dest.cpp_destructor]
// CHECK:STDOUT:   %Dest.cpp_destructor.bound.loc21: <bound method> = bound_method %.loc21_21.4, %Op.ref.loc21
// CHECK:STDOUT:   %Dest.cpp_destructor.call.loc21: init %empty_tuple.type = call %Dest.cpp_destructor.bound.loc21(%.loc21_21.4)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ConversionExplicit(%s.param: %ConditionallyExplicit.3cd) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %s.ref: %ConditionallyExplicit.3cd = name_ref s, %s
// CHECK:STDOUT:   %Cpp.ref.loc28: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %Dest.ref: type = name_ref Dest, imports.%Dest.decl [concrete = constants.%Dest]
// CHECK:STDOUT:   %.loc28_5.1: ref %Dest = temporary_storage
// CHECK:STDOUT:   %addr: %ptr.ca8 = addr_of %.loc28_5.1
// CHECK:STDOUT:   %Dest__carbon_thunk.call: init %empty_tuple.type = call imports.%Dest__carbon_thunk.decl(%s.ref, %addr)
// CHECK:STDOUT:   %.loc28_5.2: init %Dest to %.loc28_5.1 = mark_in_place_init %Dest__carbon_thunk.call
// CHECK:STDOUT:   %.loc28_5.3: init %Dest = converted %s.ref, %.loc28_5.2
// CHECK:STDOUT:   %.loc28_5.4: ref %Dest = temporary %.loc28_5.1, %.loc28_5.3
// CHECK:STDOUT:   %Dest.Op.bound: <bound method> = bound_method %.loc28_5.4, constants.%Dest.Op
// CHECK:STDOUT:   %Op.ref: %Dest.cpp_destructor.type = name_ref Op, imports.%Dest.cpp_destructor.decl [concrete = constants.%Dest.cpp_destructor]
// CHECK:STDOUT:   %Dest.cpp_destructor.bound: <bound method> = bound_method %.loc28_5.4, %Op.ref
// CHECK:STDOUT:   %Dest.cpp_destructor.call: init %empty_tuple.type = call %Dest.cpp_destructor.bound(%.loc28_5.4)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_conditionally_explicit_implicit.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Source: type = class_type @Source [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %ConditionallyExplicit.9d4: type = class_type @ConditionallyExplicit.1 [concrete]
// CHECK:STDOUT:   %pattern_type.04c: type = pattern_type %ConditionallyExplicit.9d4 [concrete]
// CHECK:STDOUT:   %_.patt.d67: %pattern_type.04c = value_binding_pattern _ [concrete]
// CHECK:STDOUT:   %ConditionallyExplicit.3cd: type = class_type @ConditionallyExplicit.2 [concrete]
// CHECK:STDOUT:   %Dest.980: type = class_type @Dest [concrete]
// CHECK:STDOUT:   %pattern_type.b72: type = pattern_type %Dest.980 [concrete]
// CHECK:STDOUT:   %_.patt.664: %pattern_type.b72 = value_binding_pattern _ [concrete]
// CHECK:STDOUT:   %ptr.ca8: type = ptr_type %Dest.980 [concrete]
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
// CHECK:STDOUT:     .ConditionallyExplicitTrue = %ConditionallyExplicit.decl.6ab
// CHECK:STDOUT:     .ConditionallyExplicitFalse = %ConditionallyExplicit.decl.996
// CHECK:STDOUT:     .Dest = %Dest.decl
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Source.decl: type = class_decl @Source [concrete = constants.%Source] {} {}
// CHECK:STDOUT:   %ConditionallyExplicit.decl.6ab: type = class_decl @ConditionallyExplicit.1 [concrete = constants.%ConditionallyExplicit.9d4] {} {}
// CHECK:STDOUT:   %ConditionallyExplicit.decl.996: type = class_decl @ConditionallyExplicit.2 [concrete = constants.%ConditionallyExplicit.3cd] {} {}
// CHECK:STDOUT:   %Dest.decl: type = class_decl @Dest [concrete = constants.%Dest.980] {} {}
// CHECK:STDOUT:   %Dest__carbon_thunk.decl: %Dest__carbon_thunk.type = fn_decl @Dest__carbon_thunk [concrete = constants.%Dest__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Dest.cpp_destructor.decl: %Dest.cpp_destructor.type = fn_decl @Dest.cpp_destructor [concrete = constants.%Dest.cpp_destructor] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ConstructorExplicit(%s.param: %Source) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %_.patt: %pattern_type.04c = value_binding_pattern _ [concrete = constants.%_.patt.d67]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %s.ref: %Source = name_ref s, %s
// CHECK:STDOUT:   %.loc15_42: %ConditionallyExplicit.9d4 = converted %s.ref, <error> [concrete = <error>]
// CHECK:STDOUT:   %.loc15_13: type = splice_block %ConditionallyExplicitTrue.ref [concrete = constants.%ConditionallyExplicit.9d4] {
// CHECK:STDOUT:     %Cpp.ref.loc15: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %ConditionallyExplicitTrue.ref: type = name_ref ConditionallyExplicitTrue, imports.%ConditionallyExplicit.decl.6ab [concrete = constants.%ConditionallyExplicit.9d4]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %_: %ConditionallyExplicit.9d4 = wrapper_binding _, <error> [concrete = <error>]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ConversionExplicit(%s.param: %ConditionallyExplicit.3cd) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %_.patt: %pattern_type.b72 = value_binding_pattern _ [concrete = constants.%_.patt.664]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %s.ref: %ConditionallyExplicit.3cd = name_ref s, %s
// CHECK:STDOUT:   %.loc21_21.1: ref %Dest.980 = temporary_storage
// CHECK:STDOUT:   %addr: %ptr.ca8 = addr_of %.loc21_21.1
// CHECK:STDOUT:   %Dest__carbon_thunk.call: init %empty_tuple.type = call imports.%Dest__carbon_thunk.decl(%s.ref, %addr)
// CHECK:STDOUT:   %.loc21_21.2: init %Dest.980 to %.loc21_21.1 = mark_in_place_init %Dest__carbon_thunk.call
// CHECK:STDOUT:   %.loc21_21.3: init %Dest.980 = converted %s.ref, %.loc21_21.2
// CHECK:STDOUT:   %.loc21_21.4: ref %Dest.980 = temporary %.loc21_21.1, %.loc21_21.3
// CHECK:STDOUT:   %.loc21_21.5: %Dest.980 = acquire_value %.loc21_21.4
// CHECK:STDOUT:   %.loc21_13: type = splice_block %Dest.ref [concrete = constants.%Dest.980] {
// CHECK:STDOUT:     %Cpp.ref.loc21: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %Dest.ref: type = name_ref Dest, imports.%Dest.decl [concrete = constants.%Dest.980]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %_: %Dest.980 = wrapper_binding _, %.loc21_21.5
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %Dest.Op.bound: <bound method> = bound_method %.loc21_21.4, constants.%Dest.Op
// CHECK:STDOUT:   %Op.ref: %Dest.cpp_destructor.type = name_ref Op, imports.%Dest.cpp_destructor.decl [concrete = constants.%Dest.cpp_destructor]
// CHECK:STDOUT:   %Dest.cpp_destructor.bound: <bound method> = bound_method %.loc21_21.4, %Op.ref
// CHECK:STDOUT:   %Dest.cpp_destructor.call: init %empty_tuple.type = call %Dest.cpp_destructor.bound(%.loc21_21.4)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
