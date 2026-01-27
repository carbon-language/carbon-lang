// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/primitives.carbon
// EXTRA-ARGS: --target=x86_64-linux-gnu --clang-arg=-std=c++20
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interop/cpp/impls/implicit_as.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interop/cpp/impls/implicit_as.carbon

// --- implicit_conversions.h

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

struct NonConstConversion {
  operator Dest();
};

struct InaccessibleConstructor {
private:
  InaccessibleConstructor(Source);
};

struct InaccessibleConversion {
private:
  operator Dest() const;
};

struct DeletedConstructor {
  DeletedConstructor(Source) = delete;
};

struct DeletedConversion {
  operator Dest() const = delete;
};

struct IntConstructor {
  IntConstructor(int);
};

struct ExplicitConstructor {
  explicit ExplicitConstructor(Source);
};

struct ExplicitConversion {
  explicit operator Dest() const;
};

// --- implicit_conversions.carbon

library "[[@TEST_NAME]]";

import Cpp library "implicit_conversions.h";

fn UserConversion(s: Cpp.Source) {
  //@dump-sem-ir-begin
  let _: Cpp.Dest = s;
  //@dump-sem-ir-end
}

fn NonConstConversion(ref s: Cpp.NonConstConversion) {
  //@dump-sem-ir-begin
  let _: Cpp.Dest = s;
  //@dump-sem-ir-end
}

fn ConstructorConversion(s: Cpp.Source2) {
  //@dump-sem-ir-begin
  let _: Cpp.Dest2 = s;
  //@dump-sem-ir-end
}

// --- fail_expr_category.carbon

library "[[@TEST_NAME]]";

import Cpp library "implicit_conversions.h";

fn NonConstConversionTest(s: Cpp.NonConstConversion) {
  //@dump-sem-ir-begin
  // CHECK:STDERR: fail_expr_category.carbon:[[@LINE+7]]:21: error: value expression passed to reference parameter [ValueForRefParam]
  // CHECK:STDERR:   let _: Cpp.Dest = s;
  // CHECK:STDERR:                     ^
  // CHECK:STDERR: fail_expr_category.carbon:[[@LINE+4]]:21: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR:   let _: Cpp.Dest = s;
  // CHECK:STDERR:                     ^
  // CHECK:STDERR:
  let _: Cpp.Dest = s;
  //@dump-sem-ir-end
}

// --- fail_access.carbon

library "[[@TEST_NAME]]";

import Cpp library "implicit_conversions.h";

fn InaccessibleConstructorTest(s: Cpp.Source) {
  //@dump-sem-ir-begin
  // CHECK:STDERR: fail_access.carbon:[[@LINE+8]]:40: error: cannot access private member `InaccessibleConstructor` of type `Cpp.InaccessibleConstructor` [ClassInvalidMemberAccess]
  // CHECK:STDERR:   let _: Cpp.InaccessibleConstructor = s;
  // CHECK:STDERR:                                        ^
  // CHECK:STDERR: fail_access.carbon:[[@LINE-7]]:10: in file included here [InCppInclude]
  // CHECK:STDERR: ./implicit_conversions.h:21:3: note: declared here [ClassMemberDeclaration]
  // CHECK:STDERR:   InaccessibleConstructor(Source);
  // CHECK:STDERR:   ^
  // CHECK:STDERR:
  let _: Cpp.InaccessibleConstructor = s;
  //@dump-sem-ir-end
}

fn InaccessibleConversionTest(s: Cpp.InaccessibleConversion) {
  //@dump-sem-ir-begin
  // CHECK:STDERR: fail_access.carbon:[[@LINE+8]]:21: error: cannot access private member `<C++ operator>` of type `Cpp.InaccessibleConversion` [ClassInvalidMemberAccess]
  // CHECK:STDERR:   let _: Cpp.Dest = s;
  // CHECK:STDERR:                     ^
  // CHECK:STDERR: fail_access.carbon:[[@LINE-21]]:10: in file included here [InCppInclude]
  // CHECK:STDERR: ./implicit_conversions.h:26:3: note: declared here [ClassMemberDeclaration]
  // CHECK:STDERR:   operator Dest() const;
  // CHECK:STDERR:   ^
  // CHECK:STDERR:
  let _: Cpp.Dest = s;
  //@dump-sem-ir-end
}

// --- fail_deleted.carbon

library "[[@TEST_NAME]]";

import Cpp library "implicit_conversions.h";

fn DeletedConstructorTest(s: Cpp.Source) {
  //@dump-sem-ir-begin
  // CHECK:STDERR: fail_deleted.carbon:[[@LINE+7]]:35: error: cannot implicitly convert expression of type `Cpp.Source` to `Cpp.DeletedConstructor` [ConversionFailure]
  // CHECK:STDERR:   let _: Cpp.DeletedConstructor = s;
  // CHECK:STDERR:                                   ^
  // CHECK:STDERR: fail_deleted.carbon:[[@LINE+4]]:35: note: type `Cpp.Source` does not implement interface `Core.ImplicitAs(Cpp.DeletedConstructor)` [MissingImplInMemberAccessNote]
  // CHECK:STDERR:   let _: Cpp.DeletedConstructor = s;
  // CHECK:STDERR:                                   ^
  // CHECK:STDERR:
  let _: Cpp.DeletedConstructor = s;
  //@dump-sem-ir-end
}

fn DeletedConversionTest(s: Cpp.DeletedConversion) {
  //@dump-sem-ir-begin
  // CHECK:STDERR: fail_deleted.carbon:[[@LINE+7]]:21: error: cannot implicitly convert expression of type `Cpp.DeletedConversion` to `Cpp.Dest` [ConversionFailure]
  // CHECK:STDERR:   let _: Cpp.Dest = s;
  // CHECK:STDERR:                     ^
  // CHECK:STDERR: fail_deleted.carbon:[[@LINE+4]]:21: note: type `Cpp.DeletedConversion` does not implement interface `Core.ImplicitAs(Cpp.Dest)` [MissingImplInMemberAccessNote]
  // CHECK:STDERR:   let _: Cpp.Dest = s;
  // CHECK:STDERR:                     ^
  // CHECK:STDERR:
  let _: Cpp.Dest = s;
  //@dump-sem-ir-end
}

// --- fail_explicit.carbon

library "[[@TEST_NAME]]";

import Cpp library "implicit_conversions.h";

fn ExplicitConstructorTest(s: Cpp.Source) {
  //@dump-sem-ir-begin
  // CHECK:STDERR: fail_explicit.carbon:[[@LINE+7]]:36: error: cannot implicitly convert expression of type `Cpp.Source` to `Cpp.ExplicitConstructor` [ConversionFailure]
  // CHECK:STDERR:   let _: Cpp.ExplicitConstructor = s;
  // CHECK:STDERR:                                    ^
  // CHECK:STDERR: fail_explicit.carbon:[[@LINE+4]]:36: note: type `Cpp.Source` does not implement interface `Core.ImplicitAs(Cpp.ExplicitConstructor)` [MissingImplInMemberAccessNote]
  // CHECK:STDERR:   let _: Cpp.ExplicitConstructor = s;
  // CHECK:STDERR:                                    ^
  // CHECK:STDERR:
  let _: Cpp.ExplicitConstructor = s;
  //@dump-sem-ir-end
}

fn ExplicitConversionTest(s: Cpp.ExplicitConversion) {
  //@dump-sem-ir-begin
  // CHECK:STDERR: fail_explicit.carbon:[[@LINE+7]]:21: error: cannot implicitly convert expression of type `Cpp.ExplicitConversion` to `Cpp.Dest` [ConversionFailure]
  // CHECK:STDERR:   let _: Cpp.Dest = s;
  // CHECK:STDERR:                     ^
  // CHECK:STDERR: fail_explicit.carbon:[[@LINE+4]]:21: note: type `Cpp.ExplicitConversion` does not implement interface `Core.ImplicitAs(Cpp.Dest)` [MissingImplInMemberAccessNote]
  // CHECK:STDERR:   let _: Cpp.Dest = s;
  // CHECK:STDERR:                     ^
  // CHECK:STDERR:
  let _: Cpp.Dest = s;
  //@dump-sem-ir-end
}

// --- i32_to_int.carbon

library "[[@TEST_NAME]]";

import Cpp library "implicit_conversions.h";

fn IntConstructor(i: i32) {
  //@dump-sem-ir-begin
  let _: Cpp.IntConstructor = i;
  //@dump-sem-ir-end
}

// --- fail_no_u32_to_int_conversion_in_carbon.carbon

library "[[@TEST_NAME]]";

import Cpp library "implicit_conversions.h";

fn IntConstructorTest(u: u32) {
  //@dump-sem-ir-begin
  // CHECK:STDERR: fail_no_u32_to_int_conversion_in_carbon.carbon:[[@LINE+11]]:31: error: cannot implicitly convert expression of type `u32` to `i32` [ConversionFailure]
  // CHECK:STDERR:   let _: Cpp.IntConstructor = u;
  // CHECK:STDERR:                               ^
  // CHECK:STDERR: fail_no_u32_to_int_conversion_in_carbon.carbon:[[@LINE+8]]:31: note: type `u32` does not implement interface `Core.ImplicitAs(i32)` [MissingImplInMemberAccessNote]
  // CHECK:STDERR:   let _: Cpp.IntConstructor = u;
  // CHECK:STDERR:                               ^
  // CHECK:STDERR: fail_no_u32_to_int_conversion_in_carbon.carbon:[[@LINE-10]]:10: in file included here [InCppInclude]
  // CHECK:STDERR: ./implicit_conversions.h:38:21: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR:   IntConstructor(int);
  // CHECK:STDERR:                     ^
  // CHECK:STDERR:
  let _: Cpp.IntConstructor = u;
  //@dump-sem-ir-end
}

// --- multi_argument.h

struct Two {
  Two(int, int);
};

// --- fail_todo_construct_multi_argument.carbon

library "[[@TEST_NAME]]";

import Cpp library "multi_argument.h";

fn ImplicitConvert() {
  //@dump-sem-ir-begin
  // CHECK:STDERR: fail_todo_construct_multi_argument.carbon:[[@LINE+6]]:20: error: semantics TODO: `Unsupported initialization sequence:
  // CHECK:STDERR: Normal sequence: list initialization via constructor [struct Two]
  // CHECK:STDERR: ` [SemanticsTodo]
  // CHECK:STDERR:   let _: Cpp.Two = (1, 2);
  // CHECK:STDERR:                    ^~~~~~
  // CHECK:STDERR:
  let _: Cpp.Two = (1, 2);
  //@dump-sem-ir-end
}

// CHECK:STDOUT: --- implicit_conversions.carbon
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
// CHECK:STDOUT:   %NonConstConversion.480: type = class_type @NonConstConversion.1 [concrete]
// CHECK:STDOUT:   %pattern_type.b08: type = pattern_type %NonConstConversion.480 [concrete]
// CHECK:STDOUT:   %NonConstConversion.cpp_operator.type: type = fn_type @NonConstConversion.cpp_operator [concrete]
// CHECK:STDOUT:   %NonConstConversion.cpp_operator: %NonConstConversion.cpp_operator.type = struct_value () [concrete]
// CHECK:STDOUT:   %Dest__carbon_thunk.type.2ffc9e.2: type = fn_type @Dest__carbon_thunk.2 [concrete]
// CHECK:STDOUT:   %Dest__carbon_thunk.f8fa06.2: %Dest__carbon_thunk.type.2ffc9e.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Source2: type = class_type @Source2 [concrete]
// CHECK:STDOUT:   %Dest2: type = class_type @Dest2 [concrete]
// CHECK:STDOUT:   %pattern_type.39d: type = pattern_type %Dest2 [concrete]
// CHECK:STDOUT:   %ptr.472: type = ptr_type %Source2 [concrete]
// CHECK:STDOUT:   %ptr.9ae: type = ptr_type %Dest2 [concrete]
// CHECK:STDOUT:   %Dest2__carbon_thunk.type: type = fn_type @Dest2__carbon_thunk [concrete]
// CHECK:STDOUT:   %Dest2__carbon_thunk: %Dest2__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %Dest2.cpp_destructor.type: type = fn_type @Dest2.cpp_destructor [concrete]
// CHECK:STDOUT:   %Dest2.cpp_destructor: %Dest2.cpp_destructor.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .Source = %Source.decl
// CHECK:STDOUT:     .Dest = %Dest.decl
// CHECK:STDOUT:     .NonConstConversion = %NonConstConversion.decl
// CHECK:STDOUT:     .Source2 = %Source2.decl
// CHECK:STDOUT:     .Dest2 = %Dest2.decl
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Source.decl: type = class_decl @Source [concrete = constants.%Source] {} {}
// CHECK:STDOUT:   %Dest.decl: type = class_decl @Dest [concrete = constants.%Dest] {} {}
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
// CHECK:STDOUT:   %NonConstConversion.decl: type = class_decl @NonConstConversion.1 [concrete = constants.%NonConstConversion.480] {} {}
// CHECK:STDOUT:   %NonConstConversion.cpp_operator.decl: %NonConstConversion.cpp_operator.type = fn_decl @NonConstConversion.cpp_operator [concrete = constants.%NonConstConversion.cpp_operator] {
// CHECK:STDOUT:     %self.patt: %pattern_type.b08 = ref_binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.b08 = ref_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %NonConstConversion.480 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %NonConstConversion.480 = ref_binding self, %self.param
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Dest__carbon_thunk.decl.b58ecd.2: %Dest__carbon_thunk.type.2ffc9e.2 = fn_decl @Dest__carbon_thunk.2 [concrete = constants.%Dest__carbon_thunk.f8fa06.2] {
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
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @UserConversion(%s.param: %Source) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %_.patt: %pattern_type.69a = value_binding_pattern _ [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %s.ref: %Source = name_ref s, %s
// CHECK:STDOUT:   %.loc8_13: type = splice_block %Dest.ref [concrete = constants.%Dest] {
// CHECK:STDOUT:     %Cpp.ref.loc8: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %Dest.ref: type = name_ref Dest, imports.%Dest.decl [concrete = constants.%Dest]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Source.cpp_operator.bound: <bound method> = bound_method %s.ref, imports.%Source.cpp_operator.decl
// CHECK:STDOUT:   %.loc8_21.1: ref %Dest = temporary_storage
// CHECK:STDOUT:   %addr: %ptr.551 = addr_of %.loc8_21.1
// CHECK:STDOUT:   %Dest__carbon_thunk.call: init %empty_tuple.type = call imports.%Dest__carbon_thunk.decl.b58ecd.1(%s.ref, %addr)
// CHECK:STDOUT:   %.loc8_21.2: init %Dest to %.loc8_21.1 = in_place_init %Dest__carbon_thunk.call
// CHECK:STDOUT:   %.loc8_21.3: init %Dest = converted %s.ref, %.loc8_21.2
// CHECK:STDOUT:   %.loc8_21.4: ref %Dest = temporary %.loc8_21.1, %.loc8_21.3
// CHECK:STDOUT:   %.loc8_21.5: %Dest = acquire_value %.loc8_21.4
// CHECK:STDOUT:   %_: %Dest = value_binding _, %.loc8_21.5
// CHECK:STDOUT:   %Dest.cpp_destructor.bound: <bound method> = bound_method %.loc8_21.4, constants.%Dest.cpp_destructor
// CHECK:STDOUT:   %Dest.cpp_destructor.call: init %empty_tuple.type = call %Dest.cpp_destructor.bound(%.loc8_21.4)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @DestroyOp.loc8(%self.param: %Dest) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @NonConstConversion.loc12(%s.param: %NonConstConversion.480) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %_.patt: %pattern_type.69a = value_binding_pattern _ [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %s.ref: ref %NonConstConversion.480 = name_ref s, %s
// CHECK:STDOUT:   %.loc14_13: type = splice_block %Dest.ref [concrete = constants.%Dest] {
// CHECK:STDOUT:     %Cpp.ref.loc14: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %Dest.ref: type = name_ref Dest, imports.%Dest.decl [concrete = constants.%Dest]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %NonConstConversion.cpp_operator.bound: <bound method> = bound_method %s.ref, imports.%NonConstConversion.cpp_operator.decl
// CHECK:STDOUT:   %.loc14_21.1: ref %Dest = temporary_storage
// CHECK:STDOUT:   %addr: %ptr.551 = addr_of %.loc14_21.1
// CHECK:STDOUT:   %Dest__carbon_thunk.call: init %empty_tuple.type = call imports.%Dest__carbon_thunk.decl.b58ecd.2(%s.ref, %addr)
// CHECK:STDOUT:   %.loc14_21.2: init %Dest to %.loc14_21.1 = in_place_init %Dest__carbon_thunk.call
// CHECK:STDOUT:   %.loc14_21.3: init %Dest = converted %s.ref, %.loc14_21.2
// CHECK:STDOUT:   %.loc14_21.4: ref %Dest = temporary %.loc14_21.1, %.loc14_21.3
// CHECK:STDOUT:   %.loc14_21.5: %Dest = acquire_value %.loc14_21.4
// CHECK:STDOUT:   %_: %Dest = value_binding _, %.loc14_21.5
// CHECK:STDOUT:   %Dest.cpp_destructor.bound: <bound method> = bound_method %.loc14_21.4, constants.%Dest.cpp_destructor
// CHECK:STDOUT:   %Dest.cpp_destructor.call: init %empty_tuple.type = call %Dest.cpp_destructor.bound(%.loc14_21.4)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ConstructorConversion(%s.param: %Source2) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %_.patt: %pattern_type.39d = value_binding_pattern _ [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %s.ref: %Source2 = name_ref s, %s
// CHECK:STDOUT:   %.loc20_13: type = splice_block %Dest2.ref [concrete = constants.%Dest2] {
// CHECK:STDOUT:     %Cpp.ref.loc20: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %Dest2.ref: type = name_ref Dest2, imports.%Dest2.decl [concrete = constants.%Dest2]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc20_22.1: ref %Dest2 = temporary_storage
// CHECK:STDOUT:   %.loc20_22.2: ref %Source2 = value_as_ref %s.ref
// CHECK:STDOUT:   %addr.loc20_22.1: %ptr.472 = addr_of %.loc20_22.2
// CHECK:STDOUT:   %addr.loc20_22.2: %ptr.9ae = addr_of %.loc20_22.1
// CHECK:STDOUT:   %Dest2__carbon_thunk.call: init %empty_tuple.type = call imports.%Dest2__carbon_thunk.decl(%addr.loc20_22.1, %addr.loc20_22.2)
// CHECK:STDOUT:   %.loc20_22.3: init %Dest2 to %.loc20_22.1 = in_place_init %Dest2__carbon_thunk.call
// CHECK:STDOUT:   %.loc20_22.4: init %Dest2 = converted %s.ref, %.loc20_22.3
// CHECK:STDOUT:   %.loc20_22.5: ref %Dest2 = temporary %.loc20_22.1, %.loc20_22.4
// CHECK:STDOUT:   %.loc20_22.6: %Dest2 = acquire_value %.loc20_22.5
// CHECK:STDOUT:   %_: %Dest2 = value_binding _, %.loc20_22.6
// CHECK:STDOUT:   %Dest2.cpp_destructor.bound: <bound method> = bound_method %.loc20_22.5, constants.%Dest2.cpp_destructor
// CHECK:STDOUT:   %Dest2.cpp_destructor.call: init %empty_tuple.type = call %Dest2.cpp_destructor.bound(%.loc20_22.5)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @DestroyOp.loc20(%self.param: %Dest2) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_expr_category.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %NonConstConversion: type = class_type @NonConstConversion [concrete]
// CHECK:STDOUT:   %pattern_type.b08: type = pattern_type %NonConstConversion [concrete]
// CHECK:STDOUT:   %Dest: type = class_type @Dest [concrete]
// CHECK:STDOUT:   %pattern_type.69a: type = pattern_type %Dest [concrete]
// CHECK:STDOUT:   %NonConstConversion.cpp_operator.type: type = fn_type @NonConstConversion.cpp_operator [concrete]
// CHECK:STDOUT:   %NonConstConversion.cpp_operator: %NonConstConversion.cpp_operator.type = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.551: type = ptr_type %Dest [concrete]
// CHECK:STDOUT:   %Dest__carbon_thunk.type: type = fn_type @Dest__carbon_thunk [concrete]
// CHECK:STDOUT:   %Dest__carbon_thunk: %Dest__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %Dest.cpp_destructor.type: type = fn_type @Dest.cpp_destructor [concrete]
// CHECK:STDOUT:   %Dest.cpp_destructor: %Dest.cpp_destructor.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .NonConstConversion = %NonConstConversion.decl
// CHECK:STDOUT:     .Dest = %Dest.decl
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %NonConstConversion.decl: type = class_decl @NonConstConversion [concrete = constants.%NonConstConversion] {} {}
// CHECK:STDOUT:   %Dest.decl: type = class_decl @Dest [concrete = constants.%Dest] {} {}
// CHECK:STDOUT:   %NonConstConversion.cpp_operator.decl: %NonConstConversion.cpp_operator.type = fn_decl @NonConstConversion.cpp_operator [concrete = constants.%NonConstConversion.cpp_operator] {
// CHECK:STDOUT:     %self.patt: %pattern_type.b08 = ref_binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.b08 = ref_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %NonConstConversion = ref_param call_param0
// CHECK:STDOUT:     %self: ref %NonConstConversion = ref_binding self, %self.param
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Dest__carbon_thunk.decl: %Dest__carbon_thunk.type = fn_decl @Dest__carbon_thunk [concrete = constants.%Dest__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @NonConstConversionTest(%s.param: %NonConstConversion) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %_.patt: %pattern_type.69a = value_binding_pattern _ [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %s.ref: %NonConstConversion = name_ref s, %s
// CHECK:STDOUT:   %.loc15_13: type = splice_block %Dest.ref [concrete = constants.%Dest] {
// CHECK:STDOUT:     %Cpp.ref.loc15: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %Dest.ref: type = name_ref Dest, imports.%Dest.decl [concrete = constants.%Dest]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %NonConstConversion.cpp_operator.bound: <bound method> = bound_method %s.ref, imports.%NonConstConversion.cpp_operator.decl
// CHECK:STDOUT:   %.loc15_21.1: ref %Dest = temporary_storage
// CHECK:STDOUT:   %addr: %ptr.551 = addr_of %.loc15_21.1
// CHECK:STDOUT:   %Dest__carbon_thunk.call: init %empty_tuple.type = call imports.%Dest__carbon_thunk.decl(<error>, %addr)
// CHECK:STDOUT:   %.loc15_21.2: init %Dest to %.loc15_21.1 = in_place_init %Dest__carbon_thunk.call
// CHECK:STDOUT:   %.loc15_21.3: init %Dest = converted %s.ref, %.loc15_21.2
// CHECK:STDOUT:   %.loc15_21.4: ref %Dest = temporary %.loc15_21.1, %.loc15_21.3
// CHECK:STDOUT:   %.loc15_21.5: %Dest = acquire_value %.loc15_21.4
// CHECK:STDOUT:   %_: %Dest = value_binding _, %.loc15_21.5
// CHECK:STDOUT:   %Dest.cpp_destructor.bound: <bound method> = bound_method %.loc15_21.4, constants.%Dest.cpp_destructor
// CHECK:STDOUT:   %Dest.cpp_destructor.call: init %empty_tuple.type = call %Dest.cpp_destructor.bound(%.loc15_21.4)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @DestroyOp(%self.param: %Dest) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_access.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Source: type = class_type @Source [concrete]
// CHECK:STDOUT:   %InaccessibleConstructor: type = class_type @InaccessibleConstructor [concrete]
// CHECK:STDOUT:   %pattern_type.b73: type = pattern_type %InaccessibleConstructor [concrete]
// CHECK:STDOUT:   %ptr.1fc: type = ptr_type %Source [concrete]
// CHECK:STDOUT:   %ptr.12d: type = ptr_type %InaccessibleConstructor [concrete]
// CHECK:STDOUT:   %InaccessibleConstructor__carbon_thunk.type: type = fn_type @InaccessibleConstructor__carbon_thunk [concrete]
// CHECK:STDOUT:   %InaccessibleConstructor__carbon_thunk: %InaccessibleConstructor__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %InaccessibleConstructor.cpp_destructor.type: type = fn_type @InaccessibleConstructor.cpp_destructor [concrete]
// CHECK:STDOUT:   %InaccessibleConstructor.cpp_destructor: %InaccessibleConstructor.cpp_destructor.type = struct_value () [concrete]
// CHECK:STDOUT:   %InaccessibleConversion: type = class_type @InaccessibleConversion [concrete]
// CHECK:STDOUT:   %pattern_type.510: type = pattern_type %InaccessibleConversion [concrete]
// CHECK:STDOUT:   %Dest: type = class_type @Dest [concrete]
// CHECK:STDOUT:   %pattern_type.69a: type = pattern_type %Dest [concrete]
// CHECK:STDOUT:   %InaccessibleConversion.cpp_operator.type: type = fn_type @InaccessibleConversion.cpp_operator [concrete]
// CHECK:STDOUT:   %InaccessibleConversion.cpp_operator: %InaccessibleConversion.cpp_operator.type = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.551: type = ptr_type %Dest [concrete]
// CHECK:STDOUT:   %Dest__carbon_thunk.type: type = fn_type @Dest__carbon_thunk [concrete]
// CHECK:STDOUT:   %Dest__carbon_thunk: %Dest__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %Dest.cpp_destructor.type: type = fn_type @Dest.cpp_destructor [concrete]
// CHECK:STDOUT:   %Dest.cpp_destructor: %Dest.cpp_destructor.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .Source = %Source.decl
// CHECK:STDOUT:     .InaccessibleConstructor = %InaccessibleConstructor.decl
// CHECK:STDOUT:     .InaccessibleConversion = %InaccessibleConversion.decl
// CHECK:STDOUT:     .Dest = %Dest.decl
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Source.decl: type = class_decl @Source [concrete = constants.%Source] {} {}
// CHECK:STDOUT:   %InaccessibleConstructor.decl: type = class_decl @InaccessibleConstructor [concrete = constants.%InaccessibleConstructor] {} {}
// CHECK:STDOUT:   %InaccessibleConstructor__carbon_thunk.decl: %InaccessibleConstructor__carbon_thunk.type = fn_decl @InaccessibleConstructor__carbon_thunk [concrete = constants.%InaccessibleConstructor__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %InaccessibleConversion.decl: type = class_decl @InaccessibleConversion [concrete = constants.%InaccessibleConversion] {} {}
// CHECK:STDOUT:   %Dest.decl: type = class_decl @Dest [concrete = constants.%Dest] {} {}
// CHECK:STDOUT:   %InaccessibleConversion.cpp_operator.decl: %InaccessibleConversion.cpp_operator.type = fn_decl @InaccessibleConversion.cpp_operator [concrete = constants.%InaccessibleConversion.cpp_operator] {
// CHECK:STDOUT:     %self.patt: %pattern_type.510 = value_binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.510 = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %InaccessibleConversion = value_param call_param0
// CHECK:STDOUT:     %self: %InaccessibleConversion = value_binding self, %self.param
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Dest__carbon_thunk.decl: %Dest__carbon_thunk.type = fn_decl @Dest__carbon_thunk [concrete = constants.%Dest__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @InaccessibleConstructorTest(%s.param: %Source) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %_.patt: %pattern_type.b73 = value_binding_pattern _ [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %s.ref: %Source = name_ref s, %s
// CHECK:STDOUT:   %.loc16_13: type = splice_block %InaccessibleConstructor.ref [concrete = constants.%InaccessibleConstructor] {
// CHECK:STDOUT:     %Cpp.ref.loc16: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %InaccessibleConstructor.ref: type = name_ref InaccessibleConstructor, imports.%InaccessibleConstructor.decl [concrete = constants.%InaccessibleConstructor]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc16_40.1: ref %InaccessibleConstructor = temporary_storage
// CHECK:STDOUT:   %.loc16_40.2: ref %Source = value_as_ref %s.ref
// CHECK:STDOUT:   %addr.loc16_40.1: %ptr.1fc = addr_of %.loc16_40.2
// CHECK:STDOUT:   %addr.loc16_40.2: %ptr.12d = addr_of %.loc16_40.1
// CHECK:STDOUT:   %InaccessibleConstructor__carbon_thunk.call: init %empty_tuple.type = call imports.%InaccessibleConstructor__carbon_thunk.decl(%addr.loc16_40.1, %addr.loc16_40.2)
// CHECK:STDOUT:   %.loc16_40.3: init %InaccessibleConstructor to %.loc16_40.1 = in_place_init %InaccessibleConstructor__carbon_thunk.call
// CHECK:STDOUT:   %.loc16_40.4: init %InaccessibleConstructor = converted %s.ref, %.loc16_40.3
// CHECK:STDOUT:   %.loc16_40.5: ref %InaccessibleConstructor = temporary %.loc16_40.1, %.loc16_40.4
// CHECK:STDOUT:   %.loc16_40.6: %InaccessibleConstructor = acquire_value %.loc16_40.5
// CHECK:STDOUT:   %_: %InaccessibleConstructor = value_binding _, %.loc16_40.6
// CHECK:STDOUT:   %InaccessibleConstructor.cpp_destructor.bound: <bound method> = bound_method %.loc16_40.5, constants.%InaccessibleConstructor.cpp_destructor
// CHECK:STDOUT:   %InaccessibleConstructor.cpp_destructor.call: init %empty_tuple.type = call %InaccessibleConstructor.cpp_destructor.bound(%.loc16_40.5)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @DestroyOp.loc16(%self.param: %InaccessibleConstructor) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @InaccessibleConversionTest(%s.param: %InaccessibleConversion) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %_.patt: %pattern_type.69a = value_binding_pattern _ [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %s.ref: %InaccessibleConversion = name_ref s, %s
// CHECK:STDOUT:   %.loc30_13: type = splice_block %Dest.ref [concrete = constants.%Dest] {
// CHECK:STDOUT:     %Cpp.ref.loc30: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %Dest.ref: type = name_ref Dest, imports.%Dest.decl [concrete = constants.%Dest]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %InaccessibleConversion.cpp_operator.bound: <bound method> = bound_method %s.ref, imports.%InaccessibleConversion.cpp_operator.decl
// CHECK:STDOUT:   %.loc30_21.1: ref %Dest = temporary_storage
// CHECK:STDOUT:   %addr: %ptr.551 = addr_of %.loc30_21.1
// CHECK:STDOUT:   %Dest__carbon_thunk.call: init %empty_tuple.type = call imports.%Dest__carbon_thunk.decl(%s.ref, %addr)
// CHECK:STDOUT:   %.loc30_21.2: init %Dest to %.loc30_21.1 = in_place_init %Dest__carbon_thunk.call
// CHECK:STDOUT:   %.loc30_21.3: init %Dest = converted %s.ref, %.loc30_21.2
// CHECK:STDOUT:   %.loc30_21.4: ref %Dest = temporary %.loc30_21.1, %.loc30_21.3
// CHECK:STDOUT:   %.loc30_21.5: %Dest = acquire_value %.loc30_21.4
// CHECK:STDOUT:   %_: %Dest = value_binding _, %.loc30_21.5
// CHECK:STDOUT:   %Dest.cpp_destructor.bound: <bound method> = bound_method %.loc30_21.4, constants.%Dest.cpp_destructor
// CHECK:STDOUT:   %Dest.cpp_destructor.call: init %empty_tuple.type = call %Dest.cpp_destructor.bound(%.loc30_21.4)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @DestroyOp.loc30(%self.param: %Dest) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_deleted.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Source: type = class_type @Source [concrete]
// CHECK:STDOUT:   %DeletedConstructor: type = class_type @DeletedConstructor [concrete]
// CHECK:STDOUT:   %pattern_type.158: type = pattern_type %DeletedConstructor [concrete]
// CHECK:STDOUT:   %DeletedConversion: type = class_type @DeletedConversion [concrete]
// CHECK:STDOUT:   %Dest.5e7: type = class_type @Dest [concrete]
// CHECK:STDOUT:   %pattern_type.69a: type = pattern_type %Dest.5e7 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .Source = %Source.decl
// CHECK:STDOUT:     .DeletedConstructor = %DeletedConstructor.decl
// CHECK:STDOUT:     .DeletedConversion = %DeletedConversion.decl
// CHECK:STDOUT:     .Dest = %Dest.decl
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Source.decl: type = class_decl @Source [concrete = constants.%Source] {} {}
// CHECK:STDOUT:   %DeletedConstructor.decl: type = class_decl @DeletedConstructor [concrete = constants.%DeletedConstructor] {} {}
// CHECK:STDOUT:   %DeletedConversion.decl: type = class_decl @DeletedConversion [concrete = constants.%DeletedConversion] {} {}
// CHECK:STDOUT:   %Dest.decl: type = class_decl @Dest [concrete = constants.%Dest.5e7] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @DeletedConstructorTest(%s.param: %Source) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %_.patt: %pattern_type.158 = value_binding_pattern _ [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %s.ref: %Source = name_ref s, %s
// CHECK:STDOUT:   %.loc15_13: type = splice_block %DeletedConstructor.ref [concrete = constants.%DeletedConstructor] {
// CHECK:STDOUT:     %Cpp.ref.loc15: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %DeletedConstructor.ref: type = name_ref DeletedConstructor, imports.%DeletedConstructor.decl [concrete = constants.%DeletedConstructor]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc15_35: %DeletedConstructor = converted %s.ref, <error> [concrete = <error>]
// CHECK:STDOUT:   %_: %DeletedConstructor = value_binding _, <error> [concrete = <error>]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @DeletedConversionTest(%s.param: %DeletedConversion) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %_.patt: %pattern_type.69a = value_binding_pattern _ [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %s.ref: %DeletedConversion = name_ref s, %s
// CHECK:STDOUT:   %.loc28_13: type = splice_block %Dest.ref [concrete = constants.%Dest.5e7] {
// CHECK:STDOUT:     %Cpp.ref.loc28: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %Dest.ref: type = name_ref Dest, imports.%Dest.decl [concrete = constants.%Dest.5e7]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc28_21: %Dest.5e7 = converted %s.ref, <error> [concrete = <error>]
// CHECK:STDOUT:   %_: %Dest.5e7 = value_binding _, <error> [concrete = <error>]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_explicit.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Source: type = class_type @Source [concrete]
// CHECK:STDOUT:   %ExplicitConstructor: type = class_type @ExplicitConstructor [concrete]
// CHECK:STDOUT:   %pattern_type.1eb: type = pattern_type %ExplicitConstructor [concrete]
// CHECK:STDOUT:   %ExplicitConversion: type = class_type @ExplicitConversion [concrete]
// CHECK:STDOUT:   %Dest.5e7: type = class_type @Dest [concrete]
// CHECK:STDOUT:   %pattern_type.69a: type = pattern_type %Dest.5e7 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .Source = %Source.decl
// CHECK:STDOUT:     .ExplicitConstructor = %ExplicitConstructor.decl
// CHECK:STDOUT:     .ExplicitConversion = %ExplicitConversion.decl
// CHECK:STDOUT:     .Dest = %Dest.decl
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Source.decl: type = class_decl @Source [concrete = constants.%Source] {} {}
// CHECK:STDOUT:   %ExplicitConstructor.decl: type = class_decl @ExplicitConstructor [concrete = constants.%ExplicitConstructor] {} {}
// CHECK:STDOUT:   %ExplicitConversion.decl: type = class_decl @ExplicitConversion [concrete = constants.%ExplicitConversion] {} {}
// CHECK:STDOUT:   %Dest.decl: type = class_decl @Dest [concrete = constants.%Dest.5e7] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ExplicitConstructorTest(%s.param: %Source) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %_.patt: %pattern_type.1eb = value_binding_pattern _ [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %s.ref: %Source = name_ref s, %s
// CHECK:STDOUT:   %.loc15_13: type = splice_block %ExplicitConstructor.ref [concrete = constants.%ExplicitConstructor] {
// CHECK:STDOUT:     %Cpp.ref.loc15: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %ExplicitConstructor.ref: type = name_ref ExplicitConstructor, imports.%ExplicitConstructor.decl [concrete = constants.%ExplicitConstructor]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc15_36: %ExplicitConstructor = converted %s.ref, <error> [concrete = <error>]
// CHECK:STDOUT:   %_: %ExplicitConstructor = value_binding _, <error> [concrete = <error>]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ExplicitConversionTest(%s.param: %ExplicitConversion) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %_.patt: %pattern_type.69a = value_binding_pattern _ [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %s.ref: %ExplicitConversion = name_ref s, %s
// CHECK:STDOUT:   %.loc28_13: type = splice_block %Dest.ref [concrete = constants.%Dest.5e7] {
// CHECK:STDOUT:     %Cpp.ref.loc28: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %Dest.ref: type = name_ref Dest, imports.%Dest.decl [concrete = constants.%Dest.5e7]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc28_21: %Dest.5e7 = converted %s.ref, <error> [concrete = <error>]
// CHECK:STDOUT:   %_: %Dest.5e7 = value_binding _, <error> [concrete = <error>]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- i32_to_int.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %IntConstructor.f49: type = class_type @IntConstructor.1 [concrete]
// CHECK:STDOUT:   %pattern_type.3c3: type = pattern_type %IntConstructor.f49 [concrete]
// CHECK:STDOUT:   %ptr.d59: type = ptr_type %IntConstructor.f49 [concrete]
// CHECK:STDOUT:   %IntConstructor__carbon_thunk.type: type = fn_type @IntConstructor__carbon_thunk [concrete]
// CHECK:STDOUT:   %IntConstructor__carbon_thunk: %IntConstructor__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %IntConstructor.cpp_destructor.type: type = fn_type @IntConstructor.cpp_destructor [concrete]
// CHECK:STDOUT:   %IntConstructor.cpp_destructor: %IntConstructor.cpp_destructor.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .IntConstructor = %IntConstructor.decl
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %IntConstructor.decl: type = class_decl @IntConstructor.1 [concrete = constants.%IntConstructor.f49] {} {}
// CHECK:STDOUT:   %IntConstructor__carbon_thunk.decl: %IntConstructor__carbon_thunk.type = fn_decl @IntConstructor__carbon_thunk [concrete = constants.%IntConstructor__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @IntConstructor.loc6(%i.param: %i32) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %_.patt: %pattern_type.3c3 = value_binding_pattern _ [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %i.ref: %i32 = name_ref i, %i
// CHECK:STDOUT:   %.loc8_13: type = splice_block %IntConstructor.ref [concrete = constants.%IntConstructor.f49] {
// CHECK:STDOUT:     %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %IntConstructor.ref: type = name_ref IntConstructor, imports.%IntConstructor.decl [concrete = constants.%IntConstructor.f49]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc8_31.1: ref %IntConstructor.f49 = temporary_storage
// CHECK:STDOUT:   %addr: %ptr.d59 = addr_of %.loc8_31.1
// CHECK:STDOUT:   %IntConstructor__carbon_thunk.call: init %empty_tuple.type = call imports.%IntConstructor__carbon_thunk.decl(%i.ref, %addr)
// CHECK:STDOUT:   %.loc8_31.2: init %IntConstructor.f49 to %.loc8_31.1 = in_place_init %IntConstructor__carbon_thunk.call
// CHECK:STDOUT:   %.loc8_31.3: init %IntConstructor.f49 = converted %i.ref, %.loc8_31.2
// CHECK:STDOUT:   %.loc8_31.4: ref %IntConstructor.f49 = temporary %.loc8_31.1, %.loc8_31.3
// CHECK:STDOUT:   %.loc8_31.5: %IntConstructor.f49 = acquire_value %.loc8_31.4
// CHECK:STDOUT:   %_: %IntConstructor.f49 = value_binding _, %.loc8_31.5
// CHECK:STDOUT:   %IntConstructor.cpp_destructor.bound: <bound method> = bound_method %.loc8_31.4, constants.%IntConstructor.cpp_destructor
// CHECK:STDOUT:   %IntConstructor.cpp_destructor.call: init %empty_tuple.type = call %IntConstructor.cpp_destructor.bound(%.loc8_31.4)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @DestroyOp(%self.param: %IntConstructor.f49) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_no_u32_to_int_conversion_in_carbon.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %u32: type = class_type @UInt, @UInt(%int_32) [concrete]
// CHECK:STDOUT:   %IntConstructor: type = class_type @IntConstructor [concrete]
// CHECK:STDOUT:   %pattern_type.3c3: type = pattern_type %IntConstructor [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %ptr.d59: type = ptr_type %IntConstructor [concrete]
// CHECK:STDOUT:   %IntConstructor__carbon_thunk.type: type = fn_type @IntConstructor__carbon_thunk [concrete]
// CHECK:STDOUT:   %IntConstructor__carbon_thunk: %IntConstructor__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %IntConstructor.cpp_destructor.type: type = fn_type @IntConstructor.cpp_destructor [concrete]
// CHECK:STDOUT:   %IntConstructor.cpp_destructor: %IntConstructor.cpp_destructor.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .IntConstructor = %IntConstructor.decl
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %IntConstructor.decl: type = class_decl @IntConstructor [concrete = constants.%IntConstructor] {} {}
// CHECK:STDOUT:   %IntConstructor__carbon_thunk.decl: %IntConstructor__carbon_thunk.type = fn_decl @IntConstructor__carbon_thunk [concrete = constants.%IntConstructor__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @IntConstructorTest(%u.param: %u32) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %_.patt: %pattern_type.3c3 = value_binding_pattern _ [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %u.ref: %u32 = name_ref u, %u
// CHECK:STDOUT:   %.loc19_13: type = splice_block %IntConstructor.ref [concrete = constants.%IntConstructor] {
// CHECK:STDOUT:     %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %IntConstructor.ref: type = name_ref IntConstructor, imports.%IntConstructor.decl [concrete = constants.%IntConstructor]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc19_31.1: ref %IntConstructor = temporary_storage
// CHECK:STDOUT:   %.loc19_31.2: %i32 = converted %u.ref, <error> [concrete = <error>]
// CHECK:STDOUT:   %addr: %ptr.d59 = addr_of %.loc19_31.1
// CHECK:STDOUT:   %IntConstructor__carbon_thunk.call: init %empty_tuple.type = call imports.%IntConstructor__carbon_thunk.decl(<error>, %addr)
// CHECK:STDOUT:   %.loc19_31.3: init %IntConstructor to %.loc19_31.1 = in_place_init %IntConstructor__carbon_thunk.call
// CHECK:STDOUT:   %.loc19_31.4: init %IntConstructor = converted %u.ref, %.loc19_31.3
// CHECK:STDOUT:   %.loc19_31.5: ref %IntConstructor = temporary %.loc19_31.1, %.loc19_31.4
// CHECK:STDOUT:   %.loc19_31.6: %IntConstructor = acquire_value %.loc19_31.5
// CHECK:STDOUT:   %_: %IntConstructor = value_binding _, %.loc19_31.6
// CHECK:STDOUT:   %IntConstructor.cpp_destructor.bound: <bound method> = bound_method %.loc19_31.5, constants.%IntConstructor.cpp_destructor
// CHECK:STDOUT:   %IntConstructor.cpp_destructor.call: init %empty_tuple.type = call %IntConstructor.cpp_destructor.bound(%.loc19_31.5)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @DestroyOp(%self.param: %IntConstructor) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_todo_construct_multi_argument.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Two: type = class_type @Two [concrete]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Two [concrete]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [concrete]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (Core.IntLiteral, Core.IntLiteral) [concrete]
// CHECK:STDOUT:   %tuple: %tuple.type = tuple_value (%int_1, %int_2) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .Two = %Two.decl
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Two.decl: type = class_decl @Two [concrete = constants.%Two] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ImplicitConvert() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %_.patt: %pattern_type = value_binding_pattern _ [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1]
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [concrete = constants.%int_2]
// CHECK:STDOUT:   %.loc14_25.1: %tuple.type = tuple_literal (%int_1, %int_2) [concrete = constants.%tuple]
// CHECK:STDOUT:   %.loc14_13: type = splice_block %Two.ref [concrete = constants.%Two] {
// CHECK:STDOUT:     %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %Two.ref: type = name_ref Two, imports.%Two.decl [concrete = constants.%Two]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc14_25.2: %Two = converted %.loc14_25.1, <error> [concrete = <error>]
// CHECK:STDOUT:   %_: %Two = value_binding _, <error> [concrete = <error>]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
