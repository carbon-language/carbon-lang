// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/full.carbon
// EXTRA-ARGS: --target=x86_64-linux-gnu
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interop/cpp/function/operators.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interop/cpp/function/operators.carbon

// ============================================================================
// Negate
// ============================================================================

// --- negate.h

class C {};
auto operator-(C operand) -> C;

// --- fail_todo_import_negate.carbon

library "[[@TEST_NAME]]";

import Cpp library "negate.h";

fn F() {
  //@dump-sem-ir-begin
  let c1: Cpp.C = Cpp.C.C();
  // CHECK:STDERR: fail_todo_import_negate.carbon:[[@LINE+4]]:19: error: cannot access member of interface `Core.Negate` in type `Cpp.C` that does not implement that interface [MissingImplInMemberAccess]
  // CHECK:STDERR:   let c2: Cpp.C = -c1;
  // CHECK:STDERR:                   ^~~
  // CHECK:STDERR:
  let c2: Cpp.C = -c1;
  //@dump-sem-ir-end
}

// ============================================================================
// AddWith
// ============================================================================

// --- add_with.h

class C {};
auto operator+(C lhs, C rhs) -> C;

// --- import_add_with.carbon

library "[[@TEST_NAME]]";

import Cpp library "add_with.h";

fn F() {
  //@dump-sem-ir-begin
  let c1: Cpp.C = Cpp.C.C();
  let c2: Cpp.C = Cpp.C.C();
  let c3: Cpp.C = c1 + c2;
  //@dump-sem-ir-end
}

// --- multiple_calls.carbon

library "[[@TEST_NAME]]";

import Cpp library "add_with.h";

fn F() {
  //@dump-sem-ir-begin
  let c1: Cpp.C = Cpp.C.C();
  let c2: Cpp.C = Cpp.C.C();
  let c3: Cpp.C = c1 + c2;
  let c4: Cpp.C = c1 + c3;
  let c5: Cpp.C = c4 + c3;
  //@dump-sem-ir-end
}

// --- fail_call_with_wrong_type.carbon

library "[[@TEST_NAME]]";

import Cpp library "add_with.h";

fn F() {
  let c1: Cpp.C = Cpp.C.C();
  // CHECK:STDERR: fail_call_with_wrong_type.carbon:[[@LINE+8]]:24: error: cannot implicitly convert expression of type `Core.IntLiteral` to `Cpp.C` [ConversionFailure]
  // CHECK:STDERR:   let c2: Cpp.C = c1 + 5;
  // CHECK:STDERR:                        ^
  // CHECK:STDERR: fail_call_with_wrong_type.carbon:[[@LINE+5]]:24: note: type `Core.IntLiteral` does not implement interface `Core.ImplicitAs(Cpp.C)` [MissingImplInMemberAccessNote]
  // CHECK:STDERR:   let c2: Cpp.C = c1 + 5;
  // CHECK:STDERR:                        ^
  // CHECK:STDERR: fail_call_with_wrong_type.carbon: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR:
  let c2: Cpp.C = c1 + 5;
  // CHECK:STDERR: fail_call_with_wrong_type.carbon:[[@LINE+8]]:19: error: cannot implicitly convert expression of type `Core.IntLiteral` to `Cpp.C` [ConversionFailure]
  // CHECK:STDERR:   let c3: Cpp.C = 6 + c1;
  // CHECK:STDERR:                   ^
  // CHECK:STDERR: fail_call_with_wrong_type.carbon:[[@LINE+5]]:19: note: type `Core.IntLiteral` does not implement interface `Core.ImplicitAs(Cpp.C)` [MissingImplInMemberAccessNote]
  // CHECK:STDERR:   let c3: Cpp.C = 6 + c1;
  // CHECK:STDERR:                   ^
  // CHECK:STDERR: fail_call_with_wrong_type.carbon: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR:
  let c3: Cpp.C = 6 + c1;
}

// ============================================================================
// One of two operands conversion
// ============================================================================

// --- plus_with_int_conversion.h

class C {
 public:
  C(int);
};
auto operator+(C lhs, C rhs) -> C;

// --- fail_todo_plus_with_int_conversion.carbon

library "[[@TEST_NAME]]";

import Cpp library "plus_with_int_conversion.h";

fn F() {
  let c1: Cpp.C = Cpp.C.C(4);
  // CHECK:STDERR: fail_todo_plus_with_int_conversion.carbon:[[@LINE+8]]:24: error: cannot implicitly convert expression of type `Core.IntLiteral` to `Cpp.C` [ConversionFailure]
  // CHECK:STDERR:   let c2: Cpp.C = c1 + 5;
  // CHECK:STDERR:                        ^
  // CHECK:STDERR: fail_todo_plus_with_int_conversion.carbon:[[@LINE+5]]:24: note: type `Core.IntLiteral` does not implement interface `Core.ImplicitAs(Cpp.C)` [MissingImplInMemberAccessNote]
  // CHECK:STDERR:   let c2: Cpp.C = c1 + 5;
  // CHECK:STDERR:                        ^
  // CHECK:STDERR: fail_todo_plus_with_int_conversion.carbon: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR:
  let c2: Cpp.C = c1 + 5;
  // CHECK:STDERR: fail_todo_plus_with_int_conversion.carbon:[[@LINE+8]]:19: error: cannot implicitly convert expression of type `Core.IntLiteral` to `Cpp.C` [ConversionFailure]
  // CHECK:STDERR:   let c3: Cpp.C = 6 + c1;
  // CHECK:STDERR:                   ^
  // CHECK:STDERR: fail_todo_plus_with_int_conversion.carbon:[[@LINE+5]]:19: note: type `Core.IntLiteral` does not implement interface `Core.ImplicitAs(Cpp.C)` [MissingImplInMemberAccessNote]
  // CHECK:STDERR:   let c3: Cpp.C = 6 + c1;
  // CHECK:STDERR:                   ^
  // CHECK:STDERR: fail_todo_plus_with_int_conversion.carbon: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR:
  let c3: Cpp.C = 6 + c1;
}

// ============================================================================
// All operands conversion
// ============================================================================

// --- plus_with_string_view_conversion.h

namespace std {
  using size_t = __SIZE_TYPE__;

  inline namespace __1 {
    template<typename T> struct char_traits {};

    template<typename CharT, typename Traits = char_traits<CharT>>
    class basic_string_view {
     public:
      basic_string_view() = default;
      size_t size() const { return size_; }

     private:
      const CharT* data_;
      size_t size_;
    };

    using string_view = basic_string_view<char>;
  }
}

class C {
 public:
  C(std::string_view s);
};
auto operator+(C lhs, C rhs) -> C;

// --- fail_todo_import_plus_with_string_view_conversion.carbon

library "[[@TEST_NAME]]";

import Cpp library "plus_with_string_view_conversion.h";

fn F() {
  let s1: str = "hello";
  let s2: str = "world";
  // CHECK:STDERR: fail_todo_import_plus_with_string_view_conversion.carbon:[[@LINE+4]]:18: error: cannot access member of interface `Core.AddWith(Core.String)` in type `Core.String` that does not implement that interface [MissingImplInMemberAccess]
  // CHECK:STDERR:   let c: Cpp.C = s1 + s2;
  // CHECK:STDERR:                  ^~~~~~~
  // CHECK:STDERR:
  let c: Cpp.C = s1 + s2;
}

// ============================================================================
// SubWith
// ============================================================================

// --- sub_with.h

class C {};
auto operator-(C lhs, C rhs) -> C;

// --- fail_todo_import_sub_with.carbon

library "[[@TEST_NAME]]";

import Cpp library "sub_with.h";

fn F() {
  //@dump-sem-ir-begin
  let c1: Cpp.C = Cpp.C.C();
  let c2: Cpp.C = Cpp.C.C();
  // CHECK:STDERR: fail_todo_import_sub_with.carbon:[[@LINE+11]]:19: error: semantics TODO: `Unsupported operator interface `SubWith`` [SemanticsTodo]
  // CHECK:STDERR:   let c3: Cpp.C = c1 - c2;
  // CHECK:STDERR:                   ^~~~~~~
  // CHECK:STDERR: fail_todo_import_sub_with.carbon:[[@LINE+8]]:19: note: in `Cpp` operator `SubWith` lookup [InCppOperatorLookup]
  // CHECK:STDERR:   let c3: Cpp.C = c1 - c2;
  // CHECK:STDERR:                   ^~~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: fail_todo_import_sub_with.carbon:[[@LINE+4]]:19: error: cannot access member of interface `Core.SubWith(Cpp.C)` in type `Cpp.C` that does not implement that interface [MissingImplInMemberAccess]
  // CHECK:STDERR:   let c3: Cpp.C = c1 - c2;
  // CHECK:STDERR:                   ^~~~~~~
  // CHECK:STDERR:
  let c3: Cpp.C = c1 - c2;
  //@dump-sem-ir-end
}

// ============================================================================
// Member operator
// ============================================================================

// --- member_add_with.h

class C {
 public:
  auto operator+(C rhs) -> C;
};

// --- fail_todo_import_member__add_with.carbon

library "[[@TEST_NAME]]";

import Cpp library "member_add_with.h";

fn F() {
  //@dump-sem-ir-begin
  let c1: Cpp.C = Cpp.C.C();
  let c2: Cpp.C = Cpp.C.C();
  // CHECK:STDERR: fail_todo_import_member__add_with.carbon:[[@LINE+4]]:19: error: cannot access member of interface `Core.AddWith(Cpp.C)` in type `Cpp.C` that does not implement that interface [MissingImplInMemberAccess]
  // CHECK:STDERR:   let c3: Cpp.C = c1 + c2;
  // CHECK:STDERR:                   ^~~~~~~
  // CHECK:STDERR:
  let c3: Cpp.C = c1 + c2;
  //@dump-sem-ir-end
}

// ============================================================================
// Satisfying constraints
// ============================================================================

// --- fail_todo_constraints.carbon

library "[[@TEST_NAME]]";

import Cpp inline '''
class X {};
X operator+(X, X);
''';

// TODO: Use Core.Add() when it is available.
// TODO: `Sum()` should be declared this way when generics implementation is ready for that:
// fn Sum[T:! Core.AddWith(.Self) where .Result = .Self](a: T, b: T) -> T
fn Sum[U:! type, T:! Core.AddWith(U) where .Result = U](a: T, b: U) -> U {
  return a + b;
}
fn Call(x: Cpp.X) -> Cpp.X {
  // CHECK:STDERR: fail_todo_constraints.carbon:[[@LINE+7]]:10: error: cannot convert type `Cpp.X` into type implementing `Core.AddWith(Cpp.X) where .(Core.AddWith(Cpp.X).Core.Result) = Cpp.X` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   return Sum(x, x);
  // CHECK:STDERR:          ^~~~~~~~~
  // CHECK:STDERR: fail_todo_constraints.carbon:[[@LINE-7]]:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn Sum[U:! type, T:! Core.AddWith(U) where .Result = U](a: T, b: U) -> U {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  return Sum(x, x);
}

// ============================================================================
// Operator not found
// ============================================================================

// --- not_found.h

class C {};

// --- fail_import_not_found.carbon

library "[[@TEST_NAME]]";

import Cpp library "not_found.h";

fn F() {
  let c1: Cpp.C = Cpp.C.C();
  let c2: Cpp.C = Cpp.C.C();
  // CHECK:STDERR: fail_import_not_found.carbon:[[@LINE+4]]:19: error: cannot access member of interface `Core.AddWith(Cpp.C)` in type `Cpp.C` that does not implement that interface [MissingImplInMemberAccess]
  // CHECK:STDERR:   let c3: Cpp.C = c1 + c2;
  // CHECK:STDERR:                   ^~~~~~~
  // CHECK:STDERR:
  let c3: Cpp.C = c1 + c2;
}

// ============================================================================
// Operator overloading
// ============================================================================

// --- overloading.h

class C {};
auto operator+(C lhs, C rhs) -> C;
class D {};
auto operator+(D lhs, D rhs) -> D;

// --- fail_todo_import_overloading.carbon

library "[[@TEST_NAME]]";

import Cpp library "overloading.h";

fn F() {
  let c1: Cpp.C = Cpp.C.C();
  let c2: Cpp.C = Cpp.C.C();
  // CHECK:STDERR: fail_todo_import_overloading.carbon:[[@LINE+7]]:19: error: semantics TODO: `Unsupported: Lookup succeeded but couldn't find a single result; LookupResultKind: 3` [SemanticsTodo]
  // CHECK:STDERR:   let c3: Cpp.C = c1 + c2;
  // CHECK:STDERR:                   ^~~~~~~
  // CHECK:STDERR: fail_todo_import_overloading.carbon:[[@LINE+4]]:19: note: in `Cpp` operator `AddWith` lookup [InCppOperatorLookup]
  // CHECK:STDERR:   let c3: Cpp.C = c1 + c2;
  // CHECK:STDERR:                   ^~~~~~~
  // CHECK:STDERR:
  let c3: Cpp.C = c1 + c2;
}

// CHECK:STDOUT: --- fail_todo_import_negate.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %pattern_type.217: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %C.C.type: type = fn_type @C.C [concrete]
// CHECK:STDOUT:   %C.C: %C.C.type = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.d9e: type = ptr_type %C [concrete]
// CHECK:STDOUT:   %C__carbon_thunk.type: type = fn_type @C__carbon_thunk [concrete]
// CHECK:STDOUT:   %C__carbon_thunk: %C__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.type.1b3: type = fn_type @T.as.Destroy.impl.Op, @T.as.Destroy.impl(%C) [concrete]
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.21b: %T.as.Destroy.impl.Op.type.1b3 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   %C.C.decl: %C.C.type = fn_decl @C.C [concrete = constants.%C.C] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C__carbon_thunk.decl: %C__carbon_thunk.type = fn_decl @C__carbon_thunk [concrete = constants.%C__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c1.patt: %pattern_type.217 = binding_pattern c1 [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc8_19: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %C.ref.loc8_22: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %C.ref.loc8_24: %C.C.type = name_ref C, imports.%C.C.decl [concrete = constants.%C.C]
// CHECK:STDOUT:   %.loc8_27.1: ref %C = temporary_storage
// CHECK:STDOUT:   %addr.loc8_27.1: %ptr.d9e = addr_of %.loc8_27.1
// CHECK:STDOUT:   %C__carbon_thunk.call: init %empty_tuple.type = call imports.%C__carbon_thunk.decl(%addr.loc8_27.1)
// CHECK:STDOUT:   %.loc8_27.2: init %C = in_place_init %C__carbon_thunk.call, %.loc8_27.1
// CHECK:STDOUT:   %.loc8_14: type = splice_block %C.ref.loc8_14 [concrete = constants.%C] {
// CHECK:STDOUT:     %Cpp.ref.loc8_11: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %C.ref.loc8_14: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc8_27.3: ref %C = temporary %.loc8_27.1, %.loc8_27.2
// CHECK:STDOUT:   %.loc8_27.4: %C = bind_value %.loc8_27.3
// CHECK:STDOUT:   %c1: %C = bind_name c1, %.loc8_27.4
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c2.patt: %pattern_type.217 = binding_pattern c2 [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c1.ref: %C = name_ref c1, %c1
// CHECK:STDOUT:   %.loc13: type = splice_block %C.ref.loc13 [concrete = constants.%C] {
// CHECK:STDOUT:     %Cpp.ref.loc13: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %C.ref.loc13: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c2: %C = bind_name c2, <error> [concrete = <error>]
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.bound: <bound method> = bound_method %.loc8_27.3, constants.%T.as.Destroy.impl.Op.21b
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %.loc8_27.3, %T.as.Destroy.impl.Op.specific_fn
// CHECK:STDOUT:   %addr.loc8_27.2: %ptr.d9e = addr_of %.loc8_27.3
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.call: init %empty_tuple.type = call %bound_method(%addr.loc8_27.2)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_add_with.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %pattern_type.217: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %C.C.type: type = fn_type @C.C [concrete]
// CHECK:STDOUT:   %C.C: %C.C.type = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.d9e: type = ptr_type %C [concrete]
// CHECK:STDOUT:   %C__carbon_thunk.type: type = fn_type @C__carbon_thunk [concrete]
// CHECK:STDOUT:   %C__carbon_thunk: %C__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %operator+__carbon_thunk.type: type = fn_type @operator+__carbon_thunk [concrete]
// CHECK:STDOUT:   %operator+__carbon_thunk: %operator+__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.type.1b3: type = fn_type @T.as.Destroy.impl.Op, @T.as.Destroy.impl(%C) [concrete]
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.21b: %T.as.Destroy.impl.Op.type.1b3 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   %C.C.decl: %C.C.type = fn_decl @C.C [concrete = constants.%C.C] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C__carbon_thunk.decl: %C__carbon_thunk.type = fn_decl @C__carbon_thunk [concrete = constants.%C__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %operator+__carbon_thunk.decl: %operator+__carbon_thunk.type = fn_decl @operator+__carbon_thunk [concrete = constants.%operator+__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c1.patt: %pattern_type.217 = binding_pattern c1 [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc8_19: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %C.ref.loc8_22: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %C.ref.loc8_24: %C.C.type = name_ref C, imports.%C.C.decl [concrete = constants.%C.C]
// CHECK:STDOUT:   %.loc8_27.1: ref %C = temporary_storage
// CHECK:STDOUT:   %addr.loc8_27.1: %ptr.d9e = addr_of %.loc8_27.1
// CHECK:STDOUT:   %C__carbon_thunk.call.loc8: init %empty_tuple.type = call imports.%C__carbon_thunk.decl(%addr.loc8_27.1)
// CHECK:STDOUT:   %.loc8_27.2: init %C = in_place_init %C__carbon_thunk.call.loc8, %.loc8_27.1
// CHECK:STDOUT:   %.loc8_14: type = splice_block %C.ref.loc8_14 [concrete = constants.%C] {
// CHECK:STDOUT:     %Cpp.ref.loc8_11: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %C.ref.loc8_14: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc8_27.3: ref %C = temporary %.loc8_27.1, %.loc8_27.2
// CHECK:STDOUT:   %.loc8_27.4: %C = bind_value %.loc8_27.3
// CHECK:STDOUT:   %c1: %C = bind_name c1, %.loc8_27.4
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c2.patt: %pattern_type.217 = binding_pattern c2 [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc9_19: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %C.ref.loc9_22: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %C.ref.loc9_24: %C.C.type = name_ref C, imports.%C.C.decl [concrete = constants.%C.C]
// CHECK:STDOUT:   %.loc9_27.1: ref %C = temporary_storage
// CHECK:STDOUT:   %addr.loc9_27.1: %ptr.d9e = addr_of %.loc9_27.1
// CHECK:STDOUT:   %C__carbon_thunk.call.loc9: init %empty_tuple.type = call imports.%C__carbon_thunk.decl(%addr.loc9_27.1)
// CHECK:STDOUT:   %.loc9_27.2: init %C = in_place_init %C__carbon_thunk.call.loc9, %.loc9_27.1
// CHECK:STDOUT:   %.loc9_14: type = splice_block %C.ref.loc9_14 [concrete = constants.%C] {
// CHECK:STDOUT:     %Cpp.ref.loc9_11: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %C.ref.loc9_14: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc9_27.3: ref %C = temporary %.loc9_27.1, %.loc9_27.2
// CHECK:STDOUT:   %.loc9_27.4: %C = bind_value %.loc9_27.3
// CHECK:STDOUT:   %c2: %C = bind_name c2, %.loc9_27.4
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c3.patt: %pattern_type.217 = binding_pattern c3 [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c1.ref: %C = name_ref c1, %c1
// CHECK:STDOUT:   %c2.ref: %C = name_ref c2, %c2
// CHECK:STDOUT:   %.loc10_22.1: ref %C = temporary_storage
// CHECK:STDOUT:   %.loc10_19: ref %C = value_as_ref %c1.ref
// CHECK:STDOUT:   %addr.loc10_22.1: %ptr.d9e = addr_of %.loc10_19
// CHECK:STDOUT:   %.loc10_24: ref %C = value_as_ref %c2.ref
// CHECK:STDOUT:   %addr.loc10_22.2: %ptr.d9e = addr_of %.loc10_24
// CHECK:STDOUT:   %addr.loc10_22.3: %ptr.d9e = addr_of %.loc10_22.1
// CHECK:STDOUT:   %operator+__carbon_thunk.call: init %empty_tuple.type = call imports.%operator+__carbon_thunk.decl(%addr.loc10_22.1, %addr.loc10_22.2, %addr.loc10_22.3)
// CHECK:STDOUT:   %.loc10_22.2: init %C = in_place_init %operator+__carbon_thunk.call, %.loc10_22.1
// CHECK:STDOUT:   %.loc10_14: type = splice_block %C.ref.loc10 [concrete = constants.%C] {
// CHECK:STDOUT:     %Cpp.ref.loc10: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %C.ref.loc10: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc10_22.3: ref %C = temporary %.loc10_22.1, %.loc10_22.2
// CHECK:STDOUT:   %.loc10_22.4: %C = bind_value %.loc10_22.3
// CHECK:STDOUT:   %c3: %C = bind_name c3, %.loc10_22.4
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.bound.loc10: <bound method> = bound_method %.loc10_22.3, constants.%T.as.Destroy.impl.Op.21b
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method.loc10: <bound method> = bound_method %.loc10_22.3, %T.as.Destroy.impl.Op.specific_fn.1
// CHECK:STDOUT:   %addr.loc10_22.4: %ptr.d9e = addr_of %.loc10_22.3
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.call.loc10: init %empty_tuple.type = call %bound_method.loc10(%addr.loc10_22.4)
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.bound.loc9: <bound method> = bound_method %.loc9_27.3, constants.%T.as.Destroy.impl.Op.21b
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method.loc9: <bound method> = bound_method %.loc9_27.3, %T.as.Destroy.impl.Op.specific_fn.2
// CHECK:STDOUT:   %addr.loc9_27.2: %ptr.d9e = addr_of %.loc9_27.3
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.call.loc9: init %empty_tuple.type = call %bound_method.loc9(%addr.loc9_27.2)
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.bound.loc8: <bound method> = bound_method %.loc8_27.3, constants.%T.as.Destroy.impl.Op.21b
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method.loc8: <bound method> = bound_method %.loc8_27.3, %T.as.Destroy.impl.Op.specific_fn.3
// CHECK:STDOUT:   %addr.loc8_27.2: %ptr.d9e = addr_of %.loc8_27.3
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.call.loc8: init %empty_tuple.type = call %bound_method.loc8(%addr.loc8_27.2)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- multiple_calls.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %pattern_type.217: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %C.C.type: type = fn_type @C.C [concrete]
// CHECK:STDOUT:   %C.C: %C.C.type = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.d9e: type = ptr_type %C [concrete]
// CHECK:STDOUT:   %C__carbon_thunk.type: type = fn_type @C__carbon_thunk [concrete]
// CHECK:STDOUT:   %C__carbon_thunk: %C__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %operator+__carbon_thunk.type: type = fn_type @operator+__carbon_thunk [concrete]
// CHECK:STDOUT:   %operator+__carbon_thunk: %operator+__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.type.1b3: type = fn_type @T.as.Destroy.impl.Op, @T.as.Destroy.impl(%C) [concrete]
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.21b: %T.as.Destroy.impl.Op.type.1b3 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   %C.C.decl: %C.C.type = fn_decl @C.C [concrete = constants.%C.C] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C__carbon_thunk.decl: %C__carbon_thunk.type = fn_decl @C__carbon_thunk [concrete = constants.%C__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %operator+__carbon_thunk.decl: %operator+__carbon_thunk.type = fn_decl @operator+__carbon_thunk [concrete = constants.%operator+__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c1.patt: %pattern_type.217 = binding_pattern c1 [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc8_19: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %C.ref.loc8_22: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %C.ref.loc8_24: %C.C.type = name_ref C, imports.%C.C.decl [concrete = constants.%C.C]
// CHECK:STDOUT:   %.loc8_27.1: ref %C = temporary_storage
// CHECK:STDOUT:   %addr.loc8_27.1: %ptr.d9e = addr_of %.loc8_27.1
// CHECK:STDOUT:   %C__carbon_thunk.call.loc8: init %empty_tuple.type = call imports.%C__carbon_thunk.decl(%addr.loc8_27.1)
// CHECK:STDOUT:   %.loc8_27.2: init %C = in_place_init %C__carbon_thunk.call.loc8, %.loc8_27.1
// CHECK:STDOUT:   %.loc8_14: type = splice_block %C.ref.loc8_14 [concrete = constants.%C] {
// CHECK:STDOUT:     %Cpp.ref.loc8_11: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %C.ref.loc8_14: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc8_27.3: ref %C = temporary %.loc8_27.1, %.loc8_27.2
// CHECK:STDOUT:   %.loc8_27.4: %C = bind_value %.loc8_27.3
// CHECK:STDOUT:   %c1: %C = bind_name c1, %.loc8_27.4
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c2.patt: %pattern_type.217 = binding_pattern c2 [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc9_19: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %C.ref.loc9_22: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %C.ref.loc9_24: %C.C.type = name_ref C, imports.%C.C.decl [concrete = constants.%C.C]
// CHECK:STDOUT:   %.loc9_27.1: ref %C = temporary_storage
// CHECK:STDOUT:   %addr.loc9_27.1: %ptr.d9e = addr_of %.loc9_27.1
// CHECK:STDOUT:   %C__carbon_thunk.call.loc9: init %empty_tuple.type = call imports.%C__carbon_thunk.decl(%addr.loc9_27.1)
// CHECK:STDOUT:   %.loc9_27.2: init %C = in_place_init %C__carbon_thunk.call.loc9, %.loc9_27.1
// CHECK:STDOUT:   %.loc9_14: type = splice_block %C.ref.loc9_14 [concrete = constants.%C] {
// CHECK:STDOUT:     %Cpp.ref.loc9_11: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %C.ref.loc9_14: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc9_27.3: ref %C = temporary %.loc9_27.1, %.loc9_27.2
// CHECK:STDOUT:   %.loc9_27.4: %C = bind_value %.loc9_27.3
// CHECK:STDOUT:   %c2: %C = bind_name c2, %.loc9_27.4
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c3.patt: %pattern_type.217 = binding_pattern c3 [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c1.ref.loc10: %C = name_ref c1, %c1
// CHECK:STDOUT:   %c2.ref: %C = name_ref c2, %c2
// CHECK:STDOUT:   %.loc10_22.1: ref %C = temporary_storage
// CHECK:STDOUT:   %.loc10_19: ref %C = value_as_ref %c1.ref.loc10
// CHECK:STDOUT:   %addr.loc10_22.1: %ptr.d9e = addr_of %.loc10_19
// CHECK:STDOUT:   %.loc10_24: ref %C = value_as_ref %c2.ref
// CHECK:STDOUT:   %addr.loc10_22.2: %ptr.d9e = addr_of %.loc10_24
// CHECK:STDOUT:   %addr.loc10_22.3: %ptr.d9e = addr_of %.loc10_22.1
// CHECK:STDOUT:   %operator+__carbon_thunk.call.loc10: init %empty_tuple.type = call imports.%operator+__carbon_thunk.decl(%addr.loc10_22.1, %addr.loc10_22.2, %addr.loc10_22.3)
// CHECK:STDOUT:   %.loc10_22.2: init %C = in_place_init %operator+__carbon_thunk.call.loc10, %.loc10_22.1
// CHECK:STDOUT:   %.loc10_14: type = splice_block %C.ref.loc10 [concrete = constants.%C] {
// CHECK:STDOUT:     %Cpp.ref.loc10: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %C.ref.loc10: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc10_22.3: ref %C = temporary %.loc10_22.1, %.loc10_22.2
// CHECK:STDOUT:   %.loc10_22.4: %C = bind_value %.loc10_22.3
// CHECK:STDOUT:   %c3: %C = bind_name c3, %.loc10_22.4
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c4.patt: %pattern_type.217 = binding_pattern c4 [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c1.ref.loc11: %C = name_ref c1, %c1
// CHECK:STDOUT:   %c3.ref.loc11: %C = name_ref c3, %c3
// CHECK:STDOUT:   %.loc11_22.1: ref %C = temporary_storage
// CHECK:STDOUT:   %.loc11_19: ref %C = value_as_ref %c1.ref.loc11
// CHECK:STDOUT:   %addr.loc11_22.1: %ptr.d9e = addr_of %.loc11_19
// CHECK:STDOUT:   %.loc11_24: ref %C = value_as_ref %c3.ref.loc11
// CHECK:STDOUT:   %addr.loc11_22.2: %ptr.d9e = addr_of %.loc11_24
// CHECK:STDOUT:   %addr.loc11_22.3: %ptr.d9e = addr_of %.loc11_22.1
// CHECK:STDOUT:   %operator+__carbon_thunk.call.loc11: init %empty_tuple.type = call imports.%operator+__carbon_thunk.decl(%addr.loc11_22.1, %addr.loc11_22.2, %addr.loc11_22.3)
// CHECK:STDOUT:   %.loc11_22.2: init %C = in_place_init %operator+__carbon_thunk.call.loc11, %.loc11_22.1
// CHECK:STDOUT:   %.loc11_14: type = splice_block %C.ref.loc11 [concrete = constants.%C] {
// CHECK:STDOUT:     %Cpp.ref.loc11: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %C.ref.loc11: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc11_22.3: ref %C = temporary %.loc11_22.1, %.loc11_22.2
// CHECK:STDOUT:   %.loc11_22.4: %C = bind_value %.loc11_22.3
// CHECK:STDOUT:   %c4: %C = bind_name c4, %.loc11_22.4
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c5.patt: %pattern_type.217 = binding_pattern c5 [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c4.ref: %C = name_ref c4, %c4
// CHECK:STDOUT:   %c3.ref.loc12: %C = name_ref c3, %c3
// CHECK:STDOUT:   %.loc12_22.1: ref %C = temporary_storage
// CHECK:STDOUT:   %.loc12_19: ref %C = value_as_ref %c4.ref
// CHECK:STDOUT:   %addr.loc12_22.1: %ptr.d9e = addr_of %.loc12_19
// CHECK:STDOUT:   %.loc12_24: ref %C = value_as_ref %c3.ref.loc12
// CHECK:STDOUT:   %addr.loc12_22.2: %ptr.d9e = addr_of %.loc12_24
// CHECK:STDOUT:   %addr.loc12_22.3: %ptr.d9e = addr_of %.loc12_22.1
// CHECK:STDOUT:   %operator+__carbon_thunk.call.loc12: init %empty_tuple.type = call imports.%operator+__carbon_thunk.decl(%addr.loc12_22.1, %addr.loc12_22.2, %addr.loc12_22.3)
// CHECK:STDOUT:   %.loc12_22.2: init %C = in_place_init %operator+__carbon_thunk.call.loc12, %.loc12_22.1
// CHECK:STDOUT:   %.loc12_14: type = splice_block %C.ref.loc12 [concrete = constants.%C] {
// CHECK:STDOUT:     %Cpp.ref.loc12: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %C.ref.loc12: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc12_22.3: ref %C = temporary %.loc12_22.1, %.loc12_22.2
// CHECK:STDOUT:   %.loc12_22.4: %C = bind_value %.loc12_22.3
// CHECK:STDOUT:   %c5: %C = bind_name c5, %.loc12_22.4
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.bound.loc12: <bound method> = bound_method %.loc12_22.3, constants.%T.as.Destroy.impl.Op.21b
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method.loc12: <bound method> = bound_method %.loc12_22.3, %T.as.Destroy.impl.Op.specific_fn.1
// CHECK:STDOUT:   %addr.loc12_22.4: %ptr.d9e = addr_of %.loc12_22.3
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.call.loc12: init %empty_tuple.type = call %bound_method.loc12(%addr.loc12_22.4)
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.bound.loc11: <bound method> = bound_method %.loc11_22.3, constants.%T.as.Destroy.impl.Op.21b
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method.loc11: <bound method> = bound_method %.loc11_22.3, %T.as.Destroy.impl.Op.specific_fn.2
// CHECK:STDOUT:   %addr.loc11_22.4: %ptr.d9e = addr_of %.loc11_22.3
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.call.loc11: init %empty_tuple.type = call %bound_method.loc11(%addr.loc11_22.4)
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.bound.loc10: <bound method> = bound_method %.loc10_22.3, constants.%T.as.Destroy.impl.Op.21b
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method.loc10: <bound method> = bound_method %.loc10_22.3, %T.as.Destroy.impl.Op.specific_fn.3
// CHECK:STDOUT:   %addr.loc10_22.4: %ptr.d9e = addr_of %.loc10_22.3
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.call.loc10: init %empty_tuple.type = call %bound_method.loc10(%addr.loc10_22.4)
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.bound.loc9: <bound method> = bound_method %.loc9_27.3, constants.%T.as.Destroy.impl.Op.21b
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method.loc9: <bound method> = bound_method %.loc9_27.3, %T.as.Destroy.impl.Op.specific_fn.4
// CHECK:STDOUT:   %addr.loc9_27.2: %ptr.d9e = addr_of %.loc9_27.3
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.call.loc9: init %empty_tuple.type = call %bound_method.loc9(%addr.loc9_27.2)
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.bound.loc8: <bound method> = bound_method %.loc8_27.3, constants.%T.as.Destroy.impl.Op.21b
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method.loc8: <bound method> = bound_method %.loc8_27.3, %T.as.Destroy.impl.Op.specific_fn.5
// CHECK:STDOUT:   %addr.loc8_27.2: %ptr.d9e = addr_of %.loc8_27.3
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.call.loc8: init %empty_tuple.type = call %bound_method.loc8(%addr.loc8_27.2)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_todo_import_sub_with.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %pattern_type.217: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %C.C.type: type = fn_type @C.C [concrete]
// CHECK:STDOUT:   %C.C: %C.C.type = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.d9e: type = ptr_type %C [concrete]
// CHECK:STDOUT:   %C__carbon_thunk.type: type = fn_type @C__carbon_thunk [concrete]
// CHECK:STDOUT:   %C__carbon_thunk: %C__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.type.1b3: type = fn_type @T.as.Destroy.impl.Op, @T.as.Destroy.impl(%C) [concrete]
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.21b: %T.as.Destroy.impl.Op.type.1b3 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   %C.C.decl: %C.C.type = fn_decl @C.C [concrete = constants.%C.C] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C__carbon_thunk.decl: %C__carbon_thunk.type = fn_decl @C__carbon_thunk [concrete = constants.%C__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c1.patt: %pattern_type.217 = binding_pattern c1 [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc8_19: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %C.ref.loc8_22: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %C.ref.loc8_24: %C.C.type = name_ref C, imports.%C.C.decl [concrete = constants.%C.C]
// CHECK:STDOUT:   %.loc8_27.1: ref %C = temporary_storage
// CHECK:STDOUT:   %addr.loc8_27.1: %ptr.d9e = addr_of %.loc8_27.1
// CHECK:STDOUT:   %C__carbon_thunk.call.loc8: init %empty_tuple.type = call imports.%C__carbon_thunk.decl(%addr.loc8_27.1)
// CHECK:STDOUT:   %.loc8_27.2: init %C = in_place_init %C__carbon_thunk.call.loc8, %.loc8_27.1
// CHECK:STDOUT:   %.loc8_14: type = splice_block %C.ref.loc8_14 [concrete = constants.%C] {
// CHECK:STDOUT:     %Cpp.ref.loc8_11: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %C.ref.loc8_14: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc8_27.3: ref %C = temporary %.loc8_27.1, %.loc8_27.2
// CHECK:STDOUT:   %.loc8_27.4: %C = bind_value %.loc8_27.3
// CHECK:STDOUT:   %c1: %C = bind_name c1, %.loc8_27.4
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c2.patt: %pattern_type.217 = binding_pattern c2 [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc9_19: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %C.ref.loc9_22: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %C.ref.loc9_24: %C.C.type = name_ref C, imports.%C.C.decl [concrete = constants.%C.C]
// CHECK:STDOUT:   %.loc9_27.1: ref %C = temporary_storage
// CHECK:STDOUT:   %addr.loc9_27.1: %ptr.d9e = addr_of %.loc9_27.1
// CHECK:STDOUT:   %C__carbon_thunk.call.loc9: init %empty_tuple.type = call imports.%C__carbon_thunk.decl(%addr.loc9_27.1)
// CHECK:STDOUT:   %.loc9_27.2: init %C = in_place_init %C__carbon_thunk.call.loc9, %.loc9_27.1
// CHECK:STDOUT:   %.loc9_14: type = splice_block %C.ref.loc9_14 [concrete = constants.%C] {
// CHECK:STDOUT:     %Cpp.ref.loc9_11: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %C.ref.loc9_14: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc9_27.3: ref %C = temporary %.loc9_27.1, %.loc9_27.2
// CHECK:STDOUT:   %.loc9_27.4: %C = bind_value %.loc9_27.3
// CHECK:STDOUT:   %c2: %C = bind_name c2, %.loc9_27.4
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c3.patt: %pattern_type.217 = binding_pattern c3 [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c1.ref: %C = name_ref c1, %c1
// CHECK:STDOUT:   %c2.ref: %C = name_ref c2, %c2
// CHECK:STDOUT:   %.loc21: type = splice_block %C.ref.loc21 [concrete = constants.%C] {
// CHECK:STDOUT:     %Cpp.ref.loc21: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %C.ref.loc21: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c3: %C = bind_name c3, <error> [concrete = <error>]
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.bound.loc9: <bound method> = bound_method %.loc9_27.3, constants.%T.as.Destroy.impl.Op.21b
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method.loc9: <bound method> = bound_method %.loc9_27.3, %T.as.Destroy.impl.Op.specific_fn.1
// CHECK:STDOUT:   %addr.loc9_27.2: %ptr.d9e = addr_of %.loc9_27.3
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.call.loc9: init %empty_tuple.type = call %bound_method.loc9(%addr.loc9_27.2)
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.bound.loc8: <bound method> = bound_method %.loc8_27.3, constants.%T.as.Destroy.impl.Op.21b
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method.loc8: <bound method> = bound_method %.loc8_27.3, %T.as.Destroy.impl.Op.specific_fn.2
// CHECK:STDOUT:   %addr.loc8_27.2: %ptr.d9e = addr_of %.loc8_27.3
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.call.loc8: init %empty_tuple.type = call %bound_method.loc8(%addr.loc8_27.2)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_todo_import_member__add_with.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %pattern_type.217: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %C.C.type: type = fn_type @C.C [concrete]
// CHECK:STDOUT:   %C.C: %C.C.type = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.d9e: type = ptr_type %C [concrete]
// CHECK:STDOUT:   %C__carbon_thunk.type: type = fn_type @C__carbon_thunk [concrete]
// CHECK:STDOUT:   %C__carbon_thunk: %C__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.type.1b3: type = fn_type @T.as.Destroy.impl.Op, @T.as.Destroy.impl(%C) [concrete]
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.21b: %T.as.Destroy.impl.Op.type.1b3 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   %C.C.decl: %C.C.type = fn_decl @C.C [concrete = constants.%C.C] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C__carbon_thunk.decl: %C__carbon_thunk.type = fn_decl @C__carbon_thunk [concrete = constants.%C__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c1.patt: %pattern_type.217 = binding_pattern c1 [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc8_19: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %C.ref.loc8_22: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %C.ref.loc8_24: %C.C.type = name_ref C, imports.%C.C.decl [concrete = constants.%C.C]
// CHECK:STDOUT:   %.loc8_27.1: ref %C = temporary_storage
// CHECK:STDOUT:   %addr.loc8_27.1: %ptr.d9e = addr_of %.loc8_27.1
// CHECK:STDOUT:   %C__carbon_thunk.call.loc8: init %empty_tuple.type = call imports.%C__carbon_thunk.decl(%addr.loc8_27.1)
// CHECK:STDOUT:   %.loc8_27.2: init %C = in_place_init %C__carbon_thunk.call.loc8, %.loc8_27.1
// CHECK:STDOUT:   %.loc8_14: type = splice_block %C.ref.loc8_14 [concrete = constants.%C] {
// CHECK:STDOUT:     %Cpp.ref.loc8_11: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %C.ref.loc8_14: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc8_27.3: ref %C = temporary %.loc8_27.1, %.loc8_27.2
// CHECK:STDOUT:   %.loc8_27.4: %C = bind_value %.loc8_27.3
// CHECK:STDOUT:   %c1: %C = bind_name c1, %.loc8_27.4
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c2.patt: %pattern_type.217 = binding_pattern c2 [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc9_19: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %C.ref.loc9_22: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %C.ref.loc9_24: %C.C.type = name_ref C, imports.%C.C.decl [concrete = constants.%C.C]
// CHECK:STDOUT:   %.loc9_27.1: ref %C = temporary_storage
// CHECK:STDOUT:   %addr.loc9_27.1: %ptr.d9e = addr_of %.loc9_27.1
// CHECK:STDOUT:   %C__carbon_thunk.call.loc9: init %empty_tuple.type = call imports.%C__carbon_thunk.decl(%addr.loc9_27.1)
// CHECK:STDOUT:   %.loc9_27.2: init %C = in_place_init %C__carbon_thunk.call.loc9, %.loc9_27.1
// CHECK:STDOUT:   %.loc9_14: type = splice_block %C.ref.loc9_14 [concrete = constants.%C] {
// CHECK:STDOUT:     %Cpp.ref.loc9_11: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %C.ref.loc9_14: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc9_27.3: ref %C = temporary %.loc9_27.1, %.loc9_27.2
// CHECK:STDOUT:   %.loc9_27.4: %C = bind_value %.loc9_27.3
// CHECK:STDOUT:   %c2: %C = bind_name c2, %.loc9_27.4
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c3.patt: %pattern_type.217 = binding_pattern c3 [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c1.ref: %C = name_ref c1, %c1
// CHECK:STDOUT:   %c2.ref: %C = name_ref c2, %c2
// CHECK:STDOUT:   %.loc14: type = splice_block %C.ref.loc14 [concrete = constants.%C] {
// CHECK:STDOUT:     %Cpp.ref.loc14: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %C.ref.loc14: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c3: %C = bind_name c3, <error> [concrete = <error>]
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.bound.loc9: <bound method> = bound_method %.loc9_27.3, constants.%T.as.Destroy.impl.Op.21b
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method.loc9: <bound method> = bound_method %.loc9_27.3, %T.as.Destroy.impl.Op.specific_fn.1
// CHECK:STDOUT:   %addr.loc9_27.2: %ptr.d9e = addr_of %.loc9_27.3
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.call.loc9: init %empty_tuple.type = call %bound_method.loc9(%addr.loc9_27.2)
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.bound.loc8: <bound method> = bound_method %.loc8_27.3, constants.%T.as.Destroy.impl.Op.21b
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method.loc8: <bound method> = bound_method %.loc8_27.3, %T.as.Destroy.impl.Op.specific_fn.2
// CHECK:STDOUT:   %addr.loc8_27.2: %ptr.d9e = addr_of %.loc8_27.3
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.call.loc8: init %empty_tuple.type = call %bound_method.loc8(%addr.loc8_27.2)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
