// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/full.carbon
// EXTRA-ARGS: --target=x86_64-linux-gnu --clang-arg=-std=c++20
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interop/cpp/function/operators.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interop/cpp/function/operators.carbon

// ============================================================================
// Unary operators
// ============================================================================

// --- unary_operators.h

class C {};

// Increment and Decrement.
// TODO: Change return value to reference when it is supported.
auto operator++(C& operand) -> C;
auto operator--(C& operand) -> C;

// Arithmetic.
auto operator-(C operand) -> C;

// --- import_unary_operators.carbon

library "[[@TEST_NAME]]";

import Cpp library "unary_operators.h";

fn F() {
  //@dump-sem-ir-begin
  var c: Cpp.C = Cpp.C.C();

  // Increment and Decrement.
  ++c;
  --c;

  // Arithmetic.
  let minus: Cpp.C = -c;
  //@dump-sem-ir-end
}

// ============================================================================
// Prefix increment and decrement vs postfix increment and decrement
// ============================================================================

// --- prefix_inc_and_dec.h

class Prefix {};
// TODO: Change return value to reference when it is supported.
auto operator++(Prefix& operand) -> Prefix;
auto operator--(Prefix& operand) -> Prefix;

// --- fail_prefix_calling_postfix.carbon

library "[[@TEST_NAME]]";

import Cpp library "prefix_inc_and_dec.h";

fn F() {
  var prefix: Cpp.Prefix = Cpp.Prefix.Prefix();
  // CHECK:STDERR: fail_prefix_calling_postfix.carbon:[[@LINE+8]]:9: error: expected `;` after expression statement [ExpectedExprSemi]
  // CHECK:STDERR:   prefix++;
  // CHECK:STDERR:         ^~
  // CHECK:STDERR:
  // CHECK:STDERR: fail_prefix_calling_postfix.carbon:[[@LINE+4]]:3: error: semantics TODO: `handle invalid parse trees in `check`` [SemanticsTodo]
  // CHECK:STDERR:   prefix++;
  // CHECK:STDERR:   ^~~~~~~~~
  // CHECK:STDERR:
  prefix++;
  // CHECK:STDERR: fail_prefix_calling_postfix.carbon:[[@LINE+4]]:9: error: expected `;` after expression statement [ExpectedExprSemi]
  // CHECK:STDERR:   prefix--;
  // CHECK:STDERR:         ^~
  // CHECK:STDERR:
  prefix--;
}

// --- postfix_inc_and_dec.h

class Postfix {};
// TODO: Change return value to reference when it is supported.
auto operator++(Postfix& operand, int) -> Postfix;
auto operator--(Postfix& operand, int) -> Postfix;

// --- fail_postfix_calling_prefix.carbon

library "[[@TEST_NAME]]";

import Cpp library "postfix_inc_and_dec.h";

fn F() {
  var postfix: Cpp.Postfix = Cpp.Postfix.Postfix();
  // CHECK:STDERR: fail_postfix_calling_prefix.carbon:[[@LINE+8]]:3: error: no matching function for call to '<C++ operator>' [CppInteropParseError]
  // CHECK:STDERR:    16 |   ++postfix;
  // CHECK:STDERR:       |   ^
  // CHECK:STDERR: fail_postfix_calling_prefix.carbon:[[@LINE-7]]:10: in file included here [InCppInclude]
  // CHECK:STDERR: ./postfix_inc_and_dec.h:4:6: note: candidate function not viable: requires 2 arguments, but 1 was provided [CppInteropParseNote]
  // CHECK:STDERR:     4 | auto operator++(Postfix& operand, int) -> Postfix;
  // CHECK:STDERR:       |      ^          ~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  ++postfix;
  // CHECK:STDERR: fail_postfix_calling_prefix.carbon:[[@LINE+8]]:3: error: no matching function for call to '<C++ operator>' [CppInteropParseError]
  // CHECK:STDERR:    25 |   --postfix;
  // CHECK:STDERR:       |   ^
  // CHECK:STDERR: fail_postfix_calling_prefix.carbon:[[@LINE-16]]:10: in file included here [InCppInclude]
  // CHECK:STDERR: ./postfix_inc_and_dec.h:5:6: note: candidate function not viable: requires 2 arguments, but 1 was provided [CppInteropParseNote]
  // CHECK:STDERR:     5 | auto operator--(Postfix& operand, int) -> Postfix;
  // CHECK:STDERR:       |      ^          ~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  --postfix;
}

// ============================================================================
// Binary operators
// ============================================================================

// --- binary_operators.h

class C {};

// Arithmetic.
auto operator+(C lhs, C rhs) -> C;
auto operator-(C lhs, C rhs) -> C;
auto operator*(C lhs, C rhs) -> C;
auto operator/(C lhs, C rhs) -> C;
auto operator%(C lhs, C rhs) -> C;

// Bitwise.
auto operator&(C lhs, C rhs) -> C;
auto operator|(C lhs, C rhs) -> C;
auto operator^(C lhs, C rhs) -> C;
auto operator<<(C lhs, int num_bits) -> C;
auto operator>>(C lhs, int num_bits) -> C;

// Compound Assignment Arithmetic.
// TODO: Change the return value to reference when it is supported.
auto operator+=(C& lhs, C rhs) -> C;
auto operator-=(C& lhs, C rhs) -> C;
auto operator*=(C& lhs, C rhs) -> C;
auto operator/=(C& lhs, C rhs) -> C;
auto operator%=(C& lhs, C rhs) -> C;

// Compound Assignment Bitwuse.
// TODO: Change the return value to reference when it is supported.
auto operator&=(C& lhs, C rhs) -> C;
auto operator|=(C& lhs, C rhs) -> C;
auto operator^=(C& lhs, C rhs) -> C;
// TODO: Add <<= and >>= when references are supported.

// Relational.
auto operator==(C lhs, C rhs) -> bool;
auto operator!=(C lhs, C rhs) -> bool;
auto operator<(C lhs, C rhs) -> bool;
auto operator>(C lhs, C rhs) -> bool;
auto operator<=(C lhs, C rhs) -> bool;
auto operator>=(C lhs, C rhs) -> bool;

// --- import_binary_operators.carbon

library "[[@TEST_NAME]]";

import Cpp library "binary_operators.h";

fn F() {
  //@dump-sem-ir-begin
  var c1: Cpp.C = Cpp.C.C();
  var c2: Cpp.C = Cpp.C.C();

  // Arithmetic.
  let addition: Cpp.C = c1 + c2;
  let subtraction: Cpp.C = c1 - c2;
  let multiplication: Cpp.C = c1 * c2;
  let division: Cpp.C = c1 / c2;
  let modulo: Cpp.C = c1 % c2;

  // Bitwise.
  let bitwise_and: Cpp.C = c1 & c2;
  let bitwise_or: Cpp.C = c1 | c2;
  let bitwise_xor: Cpp.C = c1 ^ c2;
  let left_shift: Cpp.C = c1 << 3;
  let right_shift: Cpp.C = c1 >> 5;

  // Compound Assignment Arithmetic.
  c1 += c2;
  c1 -= c2;
  c1 *= c2;
  c1 /= c2;
  c1 %= c2;

  // Compound Assignment Bitwise.
  c1 &= c2;
  c1 |= c2;
  c1 ^= c2;

  // Relational.
  let equal: bool = c1 == c2;
  let not_equal: bool = c1 != c2;
  let greater_than: bool = c1 > c2;
  let less_than: bool = c1 < c2;
  let greater_than_or_equal: bool = c1 >= c2;
  let less_than_or_equal: bool = c1 <= c2;
  //@dump-sem-ir-end
}

// --- multiple_calls.carbon

library "[[@TEST_NAME]]";

import Cpp library "binary_operators.h";

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

import Cpp library "binary_operators.h";

fn F() {
  let c1: Cpp.C = Cpp.C.C();
  // CHECK:STDERR: fail_call_with_wrong_type.carbon:[[@LINE+8]]:22: error: no matching function for call to '<C++ operator>' [CppInteropParseError]
  // CHECK:STDERR:    16 |   let c2: Cpp.C = c1 + 5;
  // CHECK:STDERR:       |                      ^
  // CHECK:STDERR: fail_call_with_wrong_type.carbon:[[@LINE-7]]:10: in file included here [InCppInclude]
  // CHECK:STDERR: ./binary_operators.h:5:6: note: candidate function not viable: no known conversion from 'int' to 'C' for 2nd argument [CppInteropParseNote]
  // CHECK:STDERR:     5 | auto operator+(C lhs, C rhs) -> C;
  // CHECK:STDERR:       |      ^                ~~~~~
  // CHECK:STDERR:
  let c2: Cpp.C = c1 + 5;
  // CHECK:STDERR: fail_call_with_wrong_type.carbon:[[@LINE+8]]:21: error: no matching function for call to '<C++ operator>' [CppInteropParseError]
  // CHECK:STDERR:    25 |   let c3: Cpp.C = 6 + c1;
  // CHECK:STDERR:       |                     ^
  // CHECK:STDERR: fail_call_with_wrong_type.carbon:[[@LINE-16]]:10: in file included here [InCppInclude]
  // CHECK:STDERR: ./binary_operators.h:5:6: note: candidate function not viable: no known conversion from 'int' to 'C' for 1st argument [CppInteropParseNote]
  // CHECK:STDERR:     5 | auto operator+(C lhs, C rhs) -> C;
  // CHECK:STDERR:       |      ^         ~~~~~
  // CHECK:STDERR:
  let c3: Cpp.C = 6 + c1;
}

// ============================================================================
// Rewrite using the spaceship operator
// ============================================================================

// --- fail_todo_rewrite_spaceship.carbon

library "[[@TEST_NAME]]";

import Cpp inline '''
class C {};

namespace std { class strong_ordering {}; }
auto operator<=>(C lhs, C rhs) -> std::strong_ordering;
auto operator>(C lhs, C rhs) -> bool;
auto operator<=(C lhs, C rhs) -> bool;
''';

fn F() {
  //@dump-sem-ir-begin
  var c1: Cpp.C = Cpp.C.C();
  var c2: Cpp.C = Cpp.C.C();

  // No rewrite.
  let greater_than: bool = c1 > c2;

  // Rewrite.
  // CHECK:STDERR: fail_todo_rewrite_spaceship.carbon:[[@LINE+4]]:25: error: semantics TODO: `Rewriting operator< using operator<=> is not supported` [SemanticsTodo]
  // CHECK:STDERR:   let less_than: bool = c1 < c2;
  // CHECK:STDERR:                         ^~~~~~~
  // CHECK:STDERR:
  let less_than: bool = c1 < c2;

  // Rewrite.
  // CHECK:STDERR: fail_todo_rewrite_spaceship.carbon:[[@LINE+4]]:37: error: semantics TODO: `Rewriting operator>= using operator<=> is not supported` [SemanticsTodo]
  // CHECK:STDERR:   let greater_than_or_equal: bool = c1 >= c2;
  // CHECK:STDERR:                                     ^~~~~~~~
  // CHECK:STDERR:
  let greater_than_or_equal: bool = c1 >= c2;

  // No rewrite.
  let less_than_or_equal: bool = c1 <= c2;
  //@dump-sem-ir-end
}

// ============================================================================
// Rewrite using the equal operator
// ============================================================================

// --- fail_todo_rewrite_equal.carbon

library "[[@TEST_NAME]]";

import Cpp inline '''
class C {};

auto operator==(C lhs, C rhs) -> bool;
''';

fn F() {
  //@dump-sem-ir-begin
  var c1: Cpp.C = Cpp.C.C();
  var c2: Cpp.C = Cpp.C.C();

  // No rewrite.
  let equal: bool = c1 == c2;

  // Rewrite.
  // CHECK:STDERR: fail_todo_rewrite_equal.carbon:[[@LINE+4]]:25: error: semantics TODO: `Rewriting operator!= using operator== is not supported` [SemanticsTodo]
  // CHECK:STDERR:   let not_equal: bool = c1 != c2;
  // CHECK:STDERR:                         ^~~~~~~~
  // CHECK:STDERR:
  let not_equal: bool = c1 != c2;
  //@dump-sem-ir-end
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
  // CHECK:STDERR: fail_todo_plus_with_int_conversion.carbon:[[@LINE+11]]:24: error: cannot implicitly convert expression of type `Core.IntLiteral` to `Cpp.C` [ConversionFailure]
  // CHECK:STDERR:   let c2: Cpp.C = c1 + 5;
  // CHECK:STDERR:                        ^
  // CHECK:STDERR: fail_todo_plus_with_int_conversion.carbon:[[@LINE+8]]:24: note: type `Core.IntLiteral` does not implement interface `Core.ImplicitAs(Cpp.C)` [MissingImplInMemberAccessNote]
  // CHECK:STDERR:   let c2: Cpp.C = c1 + 5;
  // CHECK:STDERR:                        ^
  // CHECK:STDERR: fail_todo_plus_with_int_conversion.carbon:[[@LINE-10]]:10: in file included here [InCppInclude]
  // CHECK:STDERR: ./plus_with_int_conversion.h:6:25: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR: auto operator+(C lhs, C rhs) -> C;
  // CHECK:STDERR:                         ^
  // CHECK:STDERR:
  let c2: Cpp.C = c1 + 5;
  // CHECK:STDERR: fail_todo_plus_with_int_conversion.carbon:[[@LINE+11]]:19: error: cannot implicitly convert expression of type `Core.IntLiteral` to `Cpp.C` [ConversionFailure]
  // CHECK:STDERR:   let c3: Cpp.C = 6 + c1;
  // CHECK:STDERR:                   ^
  // CHECK:STDERR: fail_todo_plus_with_int_conversion.carbon:[[@LINE+8]]:19: note: type `Core.IntLiteral` does not implement interface `Core.ImplicitAs(Cpp.C)` [MissingImplInMemberAccessNote]
  // CHECK:STDERR:   let c3: Cpp.C = 6 + c1;
  // CHECK:STDERR:                   ^
  // CHECK:STDERR: fail_todo_plus_with_int_conversion.carbon:[[@LINE-22]]:10: in file included here [InCppInclude]
  // CHECK:STDERR: ./plus_with_int_conversion.h:6:18: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR: auto operator+(C lhs, C rhs) -> C;
  // CHECK:STDERR:                  ^
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
  // CHECK:STDERR: fail_todo_import_plus_with_string_view_conversion.carbon:[[@LINE+4]]:18: error: cannot access member of interface `Core.AddWith(str)` in type `str` that does not implement that interface [MissingImplInMemberAccess]
  // CHECK:STDERR:   let c: Cpp.C = s1 + s2;
  // CHECK:STDERR:                  ^~~~~~~
  // CHECK:STDERR:
  let c: Cpp.C = s1 + s2;
}

// ============================================================================
// Unsupported unary operators
// ============================================================================

// --- unsupported_unary_operators.h

class C {};

// Arithmetic.
auto operator+(C operand) -> C;

// Increment and Decrement.
// TODO: Change the return value to reference when it is supported.
auto operator++(C& operand, int) -> C;
auto operator--(C& operand, int) -> C;

// Bitwise.
auto operator~(C operand) -> C;

// Logical.
auto operator!(C operand) -> C;

// Pointer and Memory.
class P {
 public:
  auto foo() -> P;
  auto operator->() const -> P* _Nonnull;
};
auto operator*(P operand) -> P;
auto operator&(P operand) -> P;

// --- fail_todo_import_unsupported_unary_operators.carbon

library "[[@TEST_NAME]]";

import Cpp library "unsupported_unary_operators.h";

fn F() {
  var c: Cpp.C = Cpp.C.C();

  // Arithmetic.
  // CHECK:STDERR: fail_todo_import_unsupported_unary_operators.carbon:[[@LINE+12]]:21: error: expected expression [ExpectedExpr]
  // CHECK:STDERR:   let plus: Cpp.C = +c;
  // CHECK:STDERR:                     ^
  // CHECK:STDERR:
  // CHECK:STDERR: fail_todo_import_unsupported_unary_operators.carbon:[[@LINE+8]]:21: error: whitespace missing after binary operator [BinaryOperatorRequiresWhitespace]
  // CHECK:STDERR:   let plus: Cpp.C = +c;
  // CHECK:STDERR:                     ^
  // CHECK:STDERR:
  // CHECK:STDERR: fail_todo_import_unsupported_unary_operators.carbon:[[@LINE+4]]:21: error: semantics TODO: `handle invalid parse trees in `check`` [SemanticsTodo]
  // CHECK:STDERR:   let plus: Cpp.C = +c;
  // CHECK:STDERR:                     ^
  // CHECK:STDERR:
  let plus: Cpp.C = +c;

  // Increment and Decrement.
  // CHECK:STDERR: fail_todo_import_unsupported_unary_operators.carbon:[[@LINE+4]]:4: error: expected `;` after expression statement [ExpectedExprSemi]
  // CHECK:STDERR:   c++;
  // CHECK:STDERR:    ^~
  // CHECK:STDERR:
  c++;
  // CHECK:STDERR: fail_todo_import_unsupported_unary_operators.carbon:[[@LINE+4]]:4: error: expected `;` after expression statement [ExpectedExprSemi]
  // CHECK:STDERR:   c--;
  // CHECK:STDERR:    ^~
  // CHECK:STDERR:
  c--;

  // Bitwise.
  // CHECK:STDERR: fail_todo_import_unsupported_unary_operators.carbon:[[@LINE+8]]:26: error: expected expression [ExpectedExpr]
  // CHECK:STDERR:   let not_value: Cpp.C = ~c;
  // CHECK:STDERR:                          ^
  // CHECK:STDERR:
  // CHECK:STDERR: fail_todo_import_unsupported_unary_operators.carbon:[[@LINE+4]]:26: error: `let` declarations must end with a `;` [ExpectedDeclSemi]
  // CHECK:STDERR:   let not_value: Cpp.C = ~c;
  // CHECK:STDERR:                          ^
  // CHECK:STDERR:
  let not_value: Cpp.C = ~c;

  // Logical.
  // CHECK:STDERR: fail_todo_import_unsupported_unary_operators.carbon:[[@LINE+4]]:35: error: `let` declarations must end with a `;` [ExpectedDeclSemi]
  // CHECK:STDERR:   let not_result: Cpp.C = operator!(c1);
  // CHECK:STDERR:                                   ^
  // CHECK:STDERR:
  let not_result: Cpp.C = operator!(c1);

  // Pointer and Memory.
  var p: Cpp.P = Cpp.P.P();

  // Pointer and Memory.
  let dereference: Cpp.P = *p;
  let address: Cpp.P = operator&(p);
  let call_result: Cpp.P = p->foo();
}

// ============================================================================
// Unsupported binary operators
// ============================================================================

// --- unsupported_binary_operators.h

class C {};

// Bitwise.
// TODO: Change the return value to reference when it is supported.
auto operator<<=(C& lhs, int num_bits) -> C;
auto operator>>=(C& lhs, int num_bits) -> C;

// Logical.
auto operator&&(C lhs, C rhs) -> C;
auto operator||(C lhs, C rhs) -> C;

// --- fail_todo_import_unsupported_binary_operators.carbon

library "[[@TEST_NAME]]";

import Cpp library "unsupported_binary_operators.h";

fn F() {
  var c1: Cpp.C = Cpp.C.C();

  // Bitwise.
  // CHECK:STDERR: fail_todo_import_unsupported_binary_operators.carbon:[[@LINE+8]]:3: error: semantics TODO: `Unsupported operator interface `LeftShiftAssignWith`` [SemanticsTodo]
  // CHECK:STDERR:   c1 <<= 1;
  // CHECK:STDERR:   ^~~~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: fail_todo_import_unsupported_binary_operators.carbon:[[@LINE+4]]:3: error: cannot access member of interface `Core.LeftShiftAssignWith(Core.IntLiteral)` in type `Cpp.C` that does not implement that interface [MissingImplInMemberAccess]
  // CHECK:STDERR:   c1 <<= 1;
  // CHECK:STDERR:   ^~~~~~~~
  // CHECK:STDERR:
  c1 <<= 1;
  // CHECK:STDERR: fail_todo_import_unsupported_binary_operators.carbon:[[@LINE+8]]:3: error: semantics TODO: `Unsupported operator interface `RightShiftAssignWith`` [SemanticsTodo]
  // CHECK:STDERR:   c1 >>= 2;
  // CHECK:STDERR:   ^~~~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: fail_todo_import_unsupported_binary_operators.carbon:[[@LINE+4]]:3: error: cannot access member of interface `Core.RightShiftAssignWith(Core.IntLiteral)` in type `Cpp.C` that does not implement that interface [MissingImplInMemberAccess]
  // CHECK:STDERR:   c1 >>= 2;
  // CHECK:STDERR:   ^~~~~~~~
  // CHECK:STDERR:
  c1 >>= 2;

  // Logical.
  // CHECK:STDERR: fail_todo_import_unsupported_binary_operators.carbon:[[@LINE+16]]:27: error: name `operator` not found [NameNotFound]
  // CHECK:STDERR:   let and_result: Cpp.C = operator&&(c1, c2);
  // CHECK:STDERR:                           ^~~~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: fail_todo_import_unsupported_binary_operators.carbon:[[@LINE+12]]:35: error: whitespace missing around binary operator [BinaryOperatorRequiresWhitespace]
  // CHECK:STDERR:   let and_result: Cpp.C = operator&&(c1, c2);
  // CHECK:STDERR:                                   ^
  // CHECK:STDERR:
  // CHECK:STDERR: fail_todo_import_unsupported_binary_operators.carbon:[[@LINE+8]]:42: error: name `c2` not found [NameNotFound]
  // CHECK:STDERR:   let and_result: Cpp.C = operator&&(c1, c2);
  // CHECK:STDERR:                                          ^~
  // CHECK:STDERR:
  // CHECK:STDERR: fail_todo_import_unsupported_binary_operators.carbon:[[@LINE+4]]:36: error: cannot take the address of non-reference expression [AddrOfNonRef]
  // CHECK:STDERR:   let and_result: Cpp.C = operator&&(c1, c2);
  // CHECK:STDERR:                                    ^
  // CHECK:STDERR:
  let and_result: Cpp.C = operator&&(c1, c2);
  // CHECK:STDERR: fail_todo_import_unsupported_binary_operators.carbon:[[@LINE+20]]:26: error: name `operator` not found [NameNotFound]
  // CHECK:STDERR:   let or_result: Cpp.C = operator||(c1, c2);
  // CHECK:STDERR:                          ^~~~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: fail_todo_import_unsupported_binary_operators.carbon:[[@LINE+16]]:34: error: whitespace missing around binary operator [BinaryOperatorRequiresWhitespace]
  // CHECK:STDERR:   let or_result: Cpp.C = operator||(c1, c2);
  // CHECK:STDERR:                                  ^
  // CHECK:STDERR:
  // CHECK:STDERR: fail_todo_import_unsupported_binary_operators.carbon:[[@LINE+12]]:35: error: expected expression [ExpectedExpr]
  // CHECK:STDERR:   let or_result: Cpp.C = operator||(c1, c2);
  // CHECK:STDERR:                                   ^
  // CHECK:STDERR:
  // CHECK:STDERR: fail_todo_import_unsupported_binary_operators.carbon:[[@LINE+8]]:35: error: whitespace missing around binary operator [BinaryOperatorRequiresWhitespace]
  // CHECK:STDERR:   let or_result: Cpp.C = operator||(c1, c2);
  // CHECK:STDERR:                                   ^
  // CHECK:STDERR:
  // CHECK:STDERR: fail_todo_import_unsupported_binary_operators.carbon:[[@LINE+4]]:35: error: semantics TODO: `handle invalid parse trees in `check`` [SemanticsTodo]
  // CHECK:STDERR:   let or_result: Cpp.C = operator||(c1, c2);
  // CHECK:STDERR:                                   ^
  // CHECK:STDERR:
  let or_result: Cpp.C = operator||(c1, c2);
}

// ============================================================================
// Operator and operands in a single namespace
// ============================================================================

// --- single_namespace.h

namespace N {
class C {};
auto operator+(C lhs, C rhs) -> C;
}  // namespace N

// --- import_single_namespace.carbon

library "[[@TEST_NAME]]";

import Cpp library "single_namespace.h";

fn F() {
  //@dump-sem-ir-begin
  let c1: Cpp.N.C = Cpp.N.C.C();
  let c2: Cpp.N.C = Cpp.N.C.C();
  let c3: Cpp.N.C = c1 + c2;
  //@dump-sem-ir-end
}

// ============================================================================
// Operator and operands in a different namespaces
// ============================================================================

// --- multiple_namespaces.h

namespace N1 {
class C1 {};
}  // namespace N1
namespace N2 {
class C2 {};
auto operator+(N1::C1 lhs, C2 rhs) -> C2;
auto operator-(C2 lhs, N1::C1 rhs) -> C2;
}  // namespace N2

// --- import_multiple_namespaces.carbon

library "[[@TEST_NAME]]";

import Cpp library "multiple_namespaces.h";

fn F() {
  //@dump-sem-ir-begin
  let c1: Cpp.N1.C1 = Cpp.N1.C1.C1();
  let c2: Cpp.N2.C2 = Cpp.N2.C2.C2();
  let c3: Cpp.N2.C2 = c1 + c2;
  let c4: Cpp.N2.C2 = c2 - c1;
  //@dump-sem-ir-end
}

// ============================================================================
// Operands in namespace, operator in global namespace
// ============================================================================

// --- operands_in_namespace_operator_in_global.h

namespace N {
class C {};
}  // namespace N
auto operator+(N::C lhs, N::C rhs) -> N::C;

void foo() {
  N::C() + N::C();
}

// --- fail_todo_import_operands_in_namespace_operator_in_global.carbon

library "[[@TEST_NAME]]";

import Cpp library "operands_in_namespace_operator_in_global.h";

fn F() {
  //@dump-sem-ir-begin
  let c1: Cpp.N.C = Cpp.N.C.C();
  let c2: Cpp.N.C = Cpp.N.C.C();
  // CHECK:STDERR: fail_todo_import_operands_in_namespace_operator_in_global.carbon:[[@LINE+4]]:24: error: no matching function for call to '<C++ operator>' [CppInteropParseError]
  // CHECK:STDERR:    14 |   let c3: Cpp.N.C = c1 + c2;
  // CHECK:STDERR:       |                        ^
  // CHECK:STDERR:
  let c3: Cpp.N.C = c1 + c2;
  //@dump-sem-ir-end
}

// ============================================================================
// Operand is an inner class
// ============================================================================

// --- inner_class.h

class O {
 public:
  class C {};
};
auto operator+(O::C lhs, O::C rhs) -> O::C;

// --- import_inner_class.carbon

library "[[@TEST_NAME]]";

import Cpp library "inner_class.h";

fn F() {
  //@dump-sem-ir-begin
  let c1: Cpp.O.C = Cpp.O.C.C();
  let c2: Cpp.O.C = Cpp.O.C.C();
  let c3: Cpp.O.C = c1 + c2;
  //@dump-sem-ir-end
}

// ============================================================================
// Operand is an inner class in a namespace
// ============================================================================

// --- inner_class_in_namespace.h

namespace N {
class O {
 public:
  class C {};
};
auto operator+(O::C lhs, O::C rhs) -> O::C;
}  // namespace N

// --- import_inner_class_in_namespace.carbon

library "[[@TEST_NAME]]";

import Cpp library "inner_class_in_namespace.h";

fn F() {
  //@dump-sem-ir-begin
  let c1: Cpp.N.O.C = Cpp.N.O.C.C();
  let c2: Cpp.N.O.C = Cpp.N.O.C.C();
  let c3: Cpp.N.O.C = c1 + c2;
  //@dump-sem-ir-end
}

// ============================================================================
// Member operator
// ============================================================================

// --- member_add_with.h

class C {
 public:
  // Unary.
  auto operator-() -> C;

  // Binary.
  auto operator+(C rhs) -> C;
};

// --- import_member_add_with.carbon

library "[[@TEST_NAME]]";

import Cpp library "member_add_with.h";

fn F() {
  //@dump-sem-ir-begin
  var c1: Cpp.C = Cpp.C.C();
  var c2: Cpp.C = -c1;
  var c3: Cpp.C = c1 + c2;
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
  // CHECK:STDERR: fail_import_not_found.carbon:[[@LINE+4]]:22: error: no matching function for call to '<C++ operator>' [CppInteropParseError]
  // CHECK:STDERR:    13 |   let c3: Cpp.C = c1 + c2;
  // CHECK:STDERR:       |                      ^
  // CHECK:STDERR:
  let c3: Cpp.C = c1 + c2;
}

// ============================================================================
// Incomplete operand C++ type
// ============================================================================

// --- incomplete.h

class Incomplete;
class Complete {};

auto CreateIncomplete() -> Incomplete* _Nonnull;

// --- fail_import_incomplete_unary.carbon

library "[[@TEST_NAME]]";

import Cpp library "incomplete.h";

fn F() {
  // CHECK:STDERR: fail_import_incomplete_unary.carbon:[[@LINE+8]]:27: error: looking up a C++ operator with incomplete operand type `Cpp.Incomplete` [IncompleteOperandTypeInCppOperatorLookup]
  // CHECK:STDERR:   let result_unary: i32 = -*Cpp.CreateIncomplete();
  // CHECK:STDERR:                           ^~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_import_incomplete_unary.carbon:[[@LINE-6]]:10: in file included here [InCppInclude]
  // CHECK:STDERR: ./incomplete.h:2:7: note: class was forward declared here [ClassForwardDeclaredHere]
  // CHECK:STDERR: class Incomplete;
  // CHECK:STDERR:       ^
  // CHECK:STDERR:
  let result_unary: i32 = -*Cpp.CreateIncomplete();
}

// --- fail_import_incomplete_binary.carbon

library "[[@TEST_NAME]]";

import Cpp library "incomplete.h";

fn F() {
  var complete: Cpp.Complete = Cpp.Complete.Complete();
  // CHECK:STDERR: fail_import_incomplete_binary.carbon:[[@LINE+8]]:28: error: looking up a C++ operator with incomplete operand type `Cpp.Incomplete` [IncompleteOperandTypeInCppOperatorLookup]
  // CHECK:STDERR:   let result_binary: i32 = complete + *Cpp.CreateIncomplete();
  // CHECK:STDERR:                            ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_import_incomplete_binary.carbon:[[@LINE-7]]:10: in file included here [InCppInclude]
  // CHECK:STDERR: ./incomplete.h:2:7: note: class was forward declared here [ClassForwardDeclaredHere]
  // CHECK:STDERR: class Incomplete;
  // CHECK:STDERR:       ^
  // CHECK:STDERR:
  let result_binary: i32 = complete + *Cpp.CreateIncomplete();
}

// ============================================================================
// Incomplete operand Carbon type
// ============================================================================

// --- complete.h

struct Complete {};

// --- fail_incomplete_operand_carbon_type.carbon

library "[[@TEST_NAME]]";

import Cpp library "complete.h";

class Incomplete;
fn CreateIncomplete() -> Incomplete*;

fn F() {
  var complete: Cpp.Complete = Cpp.Complete.Complete();
  // CHECK:STDERR: fail_incomplete_operand_carbon_type.carbon:[[@LINE+7]]:21: error: looking up a C++ operator with incomplete operand type `Incomplete` [IncompleteOperandTypeInCppOperatorLookup]
  // CHECK:STDERR:   let result: i32 = *CreateIncomplete() + complete;
  // CHECK:STDERR:                     ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_incomplete_operand_carbon_type.carbon:[[@LINE-8]]:1: note: class was forward declared here [ClassForwardDeclaredHere]
  // CHECK:STDERR: class Incomplete;
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  let result: i32 = *CreateIncomplete() + complete;
}

// ============================================================================
// Unsupported operand type in instantiation
// ============================================================================

// --- unsupported_in_instantiation.h

struct Supported {};

template<typename T>
struct Unsupported : public virtual Supported {};
using UnsupportedAlias = Unsupported<int>;
extern UnsupportedAlias unsupported;

// --- fail_import_unsupported_in_instantiation_unary.carbon

library "[[@TEST_NAME]]";

import Cpp library "unsupported_in_instantiation.h";

fn F() {
  // CHECK:STDERR: fail_import_unsupported_in_instantiation_unary.carbon:[[@LINE+7]]:21: error: semantics TODO: `class with virtual bases` [SemanticsTodo]
  // CHECK:STDERR:   let result: i32 = -Cpp.unsupported;
  // CHECK:STDERR:                     ^~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_import_unsupported_in_instantiation_unary.carbon:[[@LINE+4]]:21: note: while completing C++ type `Cpp.Unsupported` [InCppTypeCompletion]
  // CHECK:STDERR:   let result: i32 = -Cpp.unsupported;
  // CHECK:STDERR:                     ^~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  let result: i32 = -Cpp.unsupported;
}

// --- fail_import_unsupported_in_instantiation_binary.carbon

library "[[@TEST_NAME]]";

import Cpp library "unsupported_in_instantiation.h";

fn F() {
  var supported: Cpp.Supported = Cpp.Supported.Supported();
  // CHECK:STDERR: fail_import_unsupported_in_instantiation_binary.carbon:[[@LINE+7]]:21: error: semantics TODO: `class with virtual bases` [SemanticsTodo]
  // CHECK:STDERR:   let result: i32 = supported + Cpp.unsupported;
  // CHECK:STDERR:                     ^~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_import_unsupported_in_instantiation_binary.carbon:[[@LINE+4]]:21: note: while completing C++ type `Cpp.Unsupported` [InCppTypeCompletion]
  // CHECK:STDERR:   let result: i32 = supported + Cpp.unsupported;
  // CHECK:STDERR:                     ^~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  let result: i32 = supported + Cpp.unsupported;
}

// ============================================================================
// Indirect template instantiation
// ============================================================================

// --- indirect_template_instantiation.carbon

library "[[@TEST_NAME]]";

import Cpp inline '''
template<typename T> struct B {};
template<typename T> struct A {};

using X = B<A<int>>;
int operator+(X, X);
''';

fn F(x: Cpp.X) -> i32 {
  //@dump-sem-ir-begin
  return x + x;
  //@dump-sem-ir-end
}

// ============================================================================
// Indirect template instantiation error
// ============================================================================

// --- fail_indirect_template_instantiation_error.carbon

library "[[@TEST_NAME]]";

import Cpp inline '''
template<typename T> struct B {};
template<typename T> struct A {
  // CHECK:STDERR: fail_indirect_template_instantiation_error.carbon:[[@LINE+3]]:19: error: type 'int' cannot be used prior to '::' because it has no members [CppInteropParseError]
  // CHECK:STDERR:    10 |   friend typename T::error operator+(B<A>, B<A>);
  // CHECK:STDERR:       |                   ^
  friend typename T::error operator+(B<A>, B<A>);
};

using X = B<A<int>>;
''';

fn F(x: Cpp.X) -> i32 {
  // CHECK:STDERR: fail_indirect_template_instantiation_error.carbon:[[@LINE+8]]:12: note: in instantiation of template class 'A<int>' requested here [CppInteropParseNote]
  // CHECK:STDERR:    25 |   return x + x;
  // CHECK:STDERR:       |            ^
  // CHECK:STDERR:
  // CHECK:STDERR: fail_indirect_template_instantiation_error.carbon:[[@LINE+4]]:12: error: no matching function for call to '<C++ operator>' [CppInteropParseError]
  // CHECK:STDERR:    25 |   return x + x;
  // CHECK:STDERR:       |            ^
  // CHECK:STDERR:
  return x + x;
}

// ============================================================================
// Operator overloading
// ============================================================================

// --- overloading.h

class C {};
auto operator+(C lhs, C rhs) -> C;
class D {};
auto operator+(D lhs, D rhs) -> D;

// --- import_overloading.carbon

library "[[@TEST_NAME]]";

import Cpp library "overloading.h";

fn F() {
  let c1: Cpp.C = Cpp.C.C();
  let c2: Cpp.C = Cpp.C.C();
  //@dump-sem-ir-begin
  let c3: Cpp.C = c1 + c2;
  //@dump-sem-ir-end
}

// CHECK:STDOUT: --- import_unary_operators.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %pattern_type.217: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %C.C.cpp_overload_set.type: type = cpp_overload_set_type @C.C.cpp_overload_set [concrete]
// CHECK:STDOUT:   %C.C.cpp_overload_set.value: %C.C.cpp_overload_set.type = cpp_overload_set_value @C.C.cpp_overload_set [concrete]
// CHECK:STDOUT:   %ptr.d9e: type = ptr_type %C [concrete]
// CHECK:STDOUT:   %C__carbon_thunk.type: type = fn_type @C__carbon_thunk [concrete]
// CHECK:STDOUT:   %C__carbon_thunk: %C__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %operator++__carbon_thunk.type: type = fn_type @operator++__carbon_thunk [concrete]
// CHECK:STDOUT:   %operator++__carbon_thunk: %operator++__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %operator--__carbon_thunk.type: type = fn_type @operator--__carbon_thunk [concrete]
// CHECK:STDOUT:   %operator--__carbon_thunk: %operator--__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %operator-__carbon_thunk.type: type = fn_type @operator-__carbon_thunk [concrete]
// CHECK:STDOUT:   %operator-__carbon_thunk: %operator-__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %type_where: type = facet_type <type where .Self impls <CanDestroy>> [concrete]
// CHECK:STDOUT:   %facet_value: %type_where = facet_value %C, () [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.type.b92: type = fn_type @DestroyT.binding.as_type.as.Destroy.impl.Op, @DestroyT.binding.as_type.as.Destroy.impl(%facet_value) [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.841: %DestroyT.binding.as_type.as.Destroy.impl.Op.type.b92 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   %C.C.cpp_overload_set.value: %C.C.cpp_overload_set.type = cpp_overload_set_value @C.C.cpp_overload_set [concrete = constants.%C.C.cpp_overload_set.value]
// CHECK:STDOUT:   %C__carbon_thunk.decl: %C__carbon_thunk.type = fn_decl @C__carbon_thunk [concrete = constants.%C__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %operator++__carbon_thunk.decl: %operator++__carbon_thunk.type = fn_decl @operator++__carbon_thunk [concrete = constants.%operator++__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %operator--__carbon_thunk.decl: %operator--__carbon_thunk.type = fn_decl @operator--__carbon_thunk [concrete = constants.%operator--__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %operator-__carbon_thunk.decl: %operator-__carbon_thunk.type = fn_decl @operator-__carbon_thunk [concrete = constants.%operator-__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c.patt: %pattern_type.217 = binding_pattern c [concrete]
// CHECK:STDOUT:     %c.var_patt: %pattern_type.217 = var_pattern %c.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c.var: ref %C = var %c.var_patt
// CHECK:STDOUT:   %Cpp.ref.loc8_18: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %C.ref.loc8_21: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %C.ref.loc8_23: %C.C.cpp_overload_set.type = name_ref C, imports.%C.C.cpp_overload_set.value [concrete = constants.%C.C.cpp_overload_set.value]
// CHECK:STDOUT:   %.loc8_3: ref %C = splice_block %c.var {}
// CHECK:STDOUT:   %addr.loc8_26: %ptr.d9e = addr_of %.loc8_3
// CHECK:STDOUT:   %C__carbon_thunk.call: init %empty_tuple.type = call imports.%C__carbon_thunk.decl(%addr.loc8_26)
// CHECK:STDOUT:   %.loc8_26: init %C = in_place_init %C__carbon_thunk.call, %.loc8_3
// CHECK:STDOUT:   assign %c.var, %.loc8_26
// CHECK:STDOUT:   %.loc8_13: type = splice_block %C.ref.loc8_13 [concrete = constants.%C] {
// CHECK:STDOUT:     %Cpp.ref.loc8_10: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %C.ref.loc8_13: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c: ref %C = bind_name c, %c.var
// CHECK:STDOUT:   %c.ref.loc11: ref %C = name_ref c, %c
// CHECK:STDOUT:   %.loc11_3.1: ref %C = temporary_storage
// CHECK:STDOUT:   %addr.loc11_5: %ptr.d9e = addr_of %c.ref.loc11
// CHECK:STDOUT:   %addr.loc11_3.1: %ptr.d9e = addr_of %.loc11_3.1
// CHECK:STDOUT:   %operator++__carbon_thunk.call: init %empty_tuple.type = call imports.%operator++__carbon_thunk.decl(%addr.loc11_5, %addr.loc11_3.1)
// CHECK:STDOUT:   %.loc11_3.2: init %C = in_place_init %operator++__carbon_thunk.call, %.loc11_3.1
// CHECK:STDOUT:   %.loc11_3.3: ref %C = temporary %.loc11_3.1, %.loc11_3.2
// CHECK:STDOUT:   %c.ref.loc12: ref %C = name_ref c, %c
// CHECK:STDOUT:   %.loc12_3.1: ref %C = temporary_storage
// CHECK:STDOUT:   %addr.loc12_5: %ptr.d9e = addr_of %c.ref.loc12
// CHECK:STDOUT:   %addr.loc12_3.1: %ptr.d9e = addr_of %.loc12_3.1
// CHECK:STDOUT:   %operator--__carbon_thunk.call: init %empty_tuple.type = call imports.%operator--__carbon_thunk.decl(%addr.loc12_5, %addr.loc12_3.1)
// CHECK:STDOUT:   %.loc12_3.2: init %C = in_place_init %operator--__carbon_thunk.call, %.loc12_3.1
// CHECK:STDOUT:   %.loc12_3.3: ref %C = temporary %.loc12_3.1, %.loc12_3.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %minus.patt: %pattern_type.217 = binding_pattern minus [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c.ref.loc15: ref %C = name_ref c, %c
// CHECK:STDOUT:   %.loc15_22.1: ref %C = temporary_storage
// CHECK:STDOUT:   %.loc15_23.1: %C = bind_value %c.ref.loc15
// CHECK:STDOUT:   %.loc15_23.2: ref %C = value_as_ref %.loc15_23.1
// CHECK:STDOUT:   %addr.loc15_22.1: %ptr.d9e = addr_of %.loc15_23.2
// CHECK:STDOUT:   %addr.loc15_22.2: %ptr.d9e = addr_of %.loc15_22.1
// CHECK:STDOUT:   %operator-__carbon_thunk.call: init %empty_tuple.type = call imports.%operator-__carbon_thunk.decl(%addr.loc15_22.1, %addr.loc15_22.2)
// CHECK:STDOUT:   %.loc15_22.2: init %C = in_place_init %operator-__carbon_thunk.call, %.loc15_22.1
// CHECK:STDOUT:   %.loc15_17: type = splice_block %C.ref.loc15 [concrete = constants.%C] {
// CHECK:STDOUT:     %Cpp.ref.loc15: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %C.ref.loc15: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc15_22.3: ref %C = temporary %.loc15_22.1, %.loc15_22.2
// CHECK:STDOUT:   %.loc15_22.4: %C = bind_value %.loc15_22.3
// CHECK:STDOUT:   %minus: %C = bind_name minus, %.loc15_22.4
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.bound.loc15: <bound method> = bound_method %.loc15_22.3, constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.841
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method.loc15: <bound method> = bound_method %.loc15_22.3, %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.1
// CHECK:STDOUT:   %addr.loc15_22.3: %ptr.d9e = addr_of %.loc15_22.3
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.call.loc15: init %empty_tuple.type = call %bound_method.loc15(%addr.loc15_22.3)
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.bound.loc12: <bound method> = bound_method %.loc12_3.3, constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.841
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method.loc12: <bound method> = bound_method %.loc12_3.3, %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.2
// CHECK:STDOUT:   %addr.loc12_3.2: %ptr.d9e = addr_of %.loc12_3.3
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.call.loc12: init %empty_tuple.type = call %bound_method.loc12(%addr.loc12_3.2)
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.bound.loc11: <bound method> = bound_method %.loc11_3.3, constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.841
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method.loc11: <bound method> = bound_method %.loc11_3.3, %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.3
// CHECK:STDOUT:   %addr.loc11_3.2: %ptr.d9e = addr_of %.loc11_3.3
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.call.loc11: init %empty_tuple.type = call %bound_method.loc11(%addr.loc11_3.2)
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.bound.loc8: <bound method> = bound_method %c.var, constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.841
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method.loc8: <bound method> = bound_method %c.var, %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.4
// CHECK:STDOUT:   %addr.loc8_3: %ptr.d9e = addr_of %c.var
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.call.loc8: init %empty_tuple.type = call %bound_method.loc8(%addr.loc8_3)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_binary_operators.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %pattern_type.217: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %C.C.cpp_overload_set.type: type = cpp_overload_set_type @C.C.cpp_overload_set [concrete]
// CHECK:STDOUT:   %C.C.cpp_overload_set.value: %C.C.cpp_overload_set.type = cpp_overload_set_value @C.C.cpp_overload_set [concrete]
// CHECK:STDOUT:   %ptr.d9e: type = ptr_type %C [concrete]
// CHECK:STDOUT:   %C__carbon_thunk.type: type = fn_type @C__carbon_thunk [concrete]
// CHECK:STDOUT:   %C__carbon_thunk: %C__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %operator+__carbon_thunk.type: type = fn_type @operator+__carbon_thunk [concrete]
// CHECK:STDOUT:   %operator+__carbon_thunk: %operator+__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %operator-__carbon_thunk.type: type = fn_type @operator-__carbon_thunk [concrete]
// CHECK:STDOUT:   %operator-__carbon_thunk: %operator-__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %operator*__carbon_thunk.type: type = fn_type @operator*__carbon_thunk [concrete]
// CHECK:STDOUT:   %operator*__carbon_thunk: %operator*__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %operator/__carbon_thunk.type: type = fn_type @operator/__carbon_thunk [concrete]
// CHECK:STDOUT:   %operator/__carbon_thunk: %operator/__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %operator%__carbon_thunk.type: type = fn_type @operator%__carbon_thunk [concrete]
// CHECK:STDOUT:   %operator%__carbon_thunk: %operator%__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %operator&__carbon_thunk.type: type = fn_type @operator&__carbon_thunk [concrete]
// CHECK:STDOUT:   %operator&__carbon_thunk: %operator&__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %operator|__carbon_thunk.type: type = fn_type @operator|__carbon_thunk [concrete]
// CHECK:STDOUT:   %operator|__carbon_thunk: %operator|__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %operator^__carbon_thunk.type: type = fn_type @operator^__carbon_thunk [concrete]
// CHECK:STDOUT:   %operator^__carbon_thunk: %operator^__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %int_3.1ba: Core.IntLiteral = int_value 3 [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %operator<<__carbon_thunk.type: type = fn_type @operator<<__carbon_thunk [concrete]
// CHECK:STDOUT:   %operator<<__carbon_thunk: %operator<<__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.d14: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.Convert.type.1b6: type = fn_type @ImplicitAs.Convert, @ImplicitAs(%i32) [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = bind_symbolic_name To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.340: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.1c0: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.340 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.204: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.9e9, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.584: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.0f0: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.584 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.d14 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.204) [concrete]
// CHECK:STDOUT:   %.1df: type = fn_type_with_self_type %ImplicitAs.Convert.type.1b6, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.595: <bound method> = bound_method %int_3.1ba, %Core.IntLiteral.as.ImplicitAs.impl.Convert.0f0 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.0f0, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.f36: <bound method> = bound_method %int_3.1ba, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_3.822: %i32 = int_value 3 [concrete]
// CHECK:STDOUT:   %int_5.64b: Core.IntLiteral = int_value 5 [concrete]
// CHECK:STDOUT:   %operator>>__carbon_thunk.type: type = fn_type @operator>>__carbon_thunk [concrete]
// CHECK:STDOUT:   %operator>>__carbon_thunk: %operator>>__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.23d: <bound method> = bound_method %int_5.64b, %Core.IntLiteral.as.ImplicitAs.impl.Convert.0f0 [concrete]
// CHECK:STDOUT:   %bound_method.724: <bound method> = bound_method %int_5.64b, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_5.0f6: %i32 = int_value 5 [concrete]
// CHECK:STDOUT:   %operator+=__carbon_thunk.type: type = fn_type @operator+=__carbon_thunk [concrete]
// CHECK:STDOUT:   %operator+=__carbon_thunk: %operator+=__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %operator-=__carbon_thunk.type: type = fn_type @operator-=__carbon_thunk [concrete]
// CHECK:STDOUT:   %operator-=__carbon_thunk: %operator-=__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %operator*=__carbon_thunk.type: type = fn_type @operator*=__carbon_thunk [concrete]
// CHECK:STDOUT:   %operator*=__carbon_thunk: %operator*=__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %operator/=__carbon_thunk.type: type = fn_type @operator/=__carbon_thunk [concrete]
// CHECK:STDOUT:   %operator/=__carbon_thunk: %operator/=__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %operator%=__carbon_thunk.type: type = fn_type @operator%=__carbon_thunk [concrete]
// CHECK:STDOUT:   %operator%=__carbon_thunk: %operator%=__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %operator&=__carbon_thunk.type: type = fn_type @operator&=__carbon_thunk [concrete]
// CHECK:STDOUT:   %operator&=__carbon_thunk: %operator&=__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %operator|=__carbon_thunk.type: type = fn_type @operator|=__carbon_thunk [concrete]
// CHECK:STDOUT:   %operator|=__carbon_thunk: %operator|=__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %operator^=__carbon_thunk.type: type = fn_type @operator^=__carbon_thunk [concrete]
// CHECK:STDOUT:   %operator^=__carbon_thunk: %operator^=__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %Bool.type: type = fn_type @Bool [concrete]
// CHECK:STDOUT:   %Bool: %Bool.type = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.831: type = pattern_type bool [concrete]
// CHECK:STDOUT:   %ptr.bb2: type = ptr_type bool [concrete]
// CHECK:STDOUT:   %operator==__carbon_thunk.type: type = fn_type @operator==__carbon_thunk [concrete]
// CHECK:STDOUT:   %operator==__carbon_thunk: %operator==__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %operator!=__carbon_thunk.type: type = fn_type @operator!=__carbon_thunk [concrete]
// CHECK:STDOUT:   %operator!=__carbon_thunk: %operator!=__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %operator>__carbon_thunk.type: type = fn_type @operator>__carbon_thunk [concrete]
// CHECK:STDOUT:   %operator>__carbon_thunk: %operator>__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %operator<__carbon_thunk.type: type = fn_type @operator<__carbon_thunk [concrete]
// CHECK:STDOUT:   %operator<__carbon_thunk: %operator<__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %operator>=__carbon_thunk.type: type = fn_type @operator>=__carbon_thunk [concrete]
// CHECK:STDOUT:   %operator>=__carbon_thunk: %operator>=__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %operator<=__carbon_thunk.type: type = fn_type @operator<=__carbon_thunk [concrete]
// CHECK:STDOUT:   %operator<=__carbon_thunk: %operator<=__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %type_where: type = facet_type <type where .Self impls <CanDestroy>> [concrete]
// CHECK:STDOUT:   %facet_value: %type_where = facet_value %C, () [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.type.b92: type = fn_type @DestroyT.binding.as_type.as.Destroy.impl.Op, @DestroyT.binding.as_type.as.Destroy.impl(%facet_value) [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.841: %DestroyT.binding.as_type.as.Destroy.impl.Op.type.b92 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   %C.C.cpp_overload_set.value: %C.C.cpp_overload_set.type = cpp_overload_set_value @C.C.cpp_overload_set [concrete = constants.%C.C.cpp_overload_set.value]
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
// CHECK:STDOUT:   %operator-__carbon_thunk.decl: %operator-__carbon_thunk.type = fn_decl @operator-__carbon_thunk [concrete = constants.%operator-__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %operator*__carbon_thunk.decl: %operator*__carbon_thunk.type = fn_decl @operator*__carbon_thunk [concrete = constants.%operator*__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %operator/__carbon_thunk.decl: %operator/__carbon_thunk.type = fn_decl @operator/__carbon_thunk [concrete = constants.%operator/__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %operator%__carbon_thunk.decl: %operator%__carbon_thunk.type = fn_decl @operator%__carbon_thunk [concrete = constants.%operator%__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %operator&__carbon_thunk.decl: %operator&__carbon_thunk.type = fn_decl @operator&__carbon_thunk [concrete = constants.%operator&__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %operator|__carbon_thunk.decl: %operator|__carbon_thunk.type = fn_decl @operator|__carbon_thunk [concrete = constants.%operator|__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %operator^__carbon_thunk.decl: %operator^__carbon_thunk.type = fn_decl @operator^__carbon_thunk [concrete = constants.%operator^__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %operator<<__carbon_thunk.decl: %operator<<__carbon_thunk.type = fn_decl @operator<<__carbon_thunk [concrete = constants.%operator<<__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.ee7: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.340) = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.1c0)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.9e9 = impl_witness_table (%Core.import_ref.ee7), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %operator>>__carbon_thunk.decl: %operator>>__carbon_thunk.type = fn_decl @operator>>__carbon_thunk [concrete = constants.%operator>>__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %operator+=__carbon_thunk.decl: %operator+=__carbon_thunk.type = fn_decl @operator+=__carbon_thunk [concrete = constants.%operator+=__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %operator-=__carbon_thunk.decl: %operator-=__carbon_thunk.type = fn_decl @operator-=__carbon_thunk [concrete = constants.%operator-=__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %operator*=__carbon_thunk.decl: %operator*=__carbon_thunk.type = fn_decl @operator*=__carbon_thunk [concrete = constants.%operator*=__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %operator/=__carbon_thunk.decl: %operator/=__carbon_thunk.type = fn_decl @operator/=__carbon_thunk [concrete = constants.%operator/=__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %operator%=__carbon_thunk.decl: %operator%=__carbon_thunk.type = fn_decl @operator%=__carbon_thunk [concrete = constants.%operator%=__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %operator&=__carbon_thunk.decl: %operator&=__carbon_thunk.type = fn_decl @operator&=__carbon_thunk [concrete = constants.%operator&=__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %operator|=__carbon_thunk.decl: %operator|=__carbon_thunk.type = fn_decl @operator|=__carbon_thunk [concrete = constants.%operator|=__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %operator^=__carbon_thunk.decl: %operator^=__carbon_thunk.type = fn_decl @operator^=__carbon_thunk [concrete = constants.%operator^=__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %operator==__carbon_thunk.decl: %operator==__carbon_thunk.type = fn_decl @operator==__carbon_thunk [concrete = constants.%operator==__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %operator!=__carbon_thunk.decl: %operator!=__carbon_thunk.type = fn_decl @operator!=__carbon_thunk [concrete = constants.%operator!=__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %operator>__carbon_thunk.decl: %operator>__carbon_thunk.type = fn_decl @operator>__carbon_thunk [concrete = constants.%operator>__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %operator<__carbon_thunk.decl: %operator<__carbon_thunk.type = fn_decl @operator<__carbon_thunk [concrete = constants.%operator<__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %operator>=__carbon_thunk.decl: %operator>=__carbon_thunk.type = fn_decl @operator>=__carbon_thunk [concrete = constants.%operator>=__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %operator<=__carbon_thunk.decl: %operator<=__carbon_thunk.type = fn_decl @operator<=__carbon_thunk [concrete = constants.%operator<=__carbon_thunk] {
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
// CHECK:STDOUT:     %c1.var_patt: %pattern_type.217 = var_pattern %c1.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c1.var: ref %C = var %c1.var_patt
// CHECK:STDOUT:   %Cpp.ref.loc8_19: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %C.ref.loc8_22: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %C.ref.loc8_24: %C.C.cpp_overload_set.type = name_ref C, imports.%C.C.cpp_overload_set.value [concrete = constants.%C.C.cpp_overload_set.value]
// CHECK:STDOUT:   %.loc8_3: ref %C = splice_block %c1.var {}
// CHECK:STDOUT:   %addr.loc8_27: %ptr.d9e = addr_of %.loc8_3
// CHECK:STDOUT:   %C__carbon_thunk.call.loc8: init %empty_tuple.type = call imports.%C__carbon_thunk.decl(%addr.loc8_27)
// CHECK:STDOUT:   %.loc8_27: init %C = in_place_init %C__carbon_thunk.call.loc8, %.loc8_3
// CHECK:STDOUT:   assign %c1.var, %.loc8_27
// CHECK:STDOUT:   %.loc8_14: type = splice_block %C.ref.loc8_14 [concrete = constants.%C] {
// CHECK:STDOUT:     %Cpp.ref.loc8_11: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %C.ref.loc8_14: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c1: ref %C = bind_name c1, %c1.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c2.patt: %pattern_type.217 = binding_pattern c2 [concrete]
// CHECK:STDOUT:     %c2.var_patt: %pattern_type.217 = var_pattern %c2.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c2.var: ref %C = var %c2.var_patt
// CHECK:STDOUT:   %Cpp.ref.loc9_19: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %C.ref.loc9_22: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %C.ref.loc9_24: %C.C.cpp_overload_set.type = name_ref C, imports.%C.C.cpp_overload_set.value [concrete = constants.%C.C.cpp_overload_set.value]
// CHECK:STDOUT:   %.loc9_3: ref %C = splice_block %c2.var {}
// CHECK:STDOUT:   %addr.loc9_27: %ptr.d9e = addr_of %.loc9_3
// CHECK:STDOUT:   %C__carbon_thunk.call.loc9: init %empty_tuple.type = call imports.%C__carbon_thunk.decl(%addr.loc9_27)
// CHECK:STDOUT:   %.loc9_27: init %C = in_place_init %C__carbon_thunk.call.loc9, %.loc9_3
// CHECK:STDOUT:   assign %c2.var, %.loc9_27
// CHECK:STDOUT:   %.loc9_14: type = splice_block %C.ref.loc9_14 [concrete = constants.%C] {
// CHECK:STDOUT:     %Cpp.ref.loc9_11: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %C.ref.loc9_14: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c2: ref %C = bind_name c2, %c2.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %addition.patt: %pattern_type.217 = binding_pattern addition [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c1.ref.loc12: ref %C = name_ref c1, %c1
// CHECK:STDOUT:   %c2.ref.loc12: ref %C = name_ref c2, %c2
// CHECK:STDOUT:   %.loc12_28.1: ref %C = temporary_storage
// CHECK:STDOUT:   %.loc12_25.1: %C = bind_value %c1.ref.loc12
// CHECK:STDOUT:   %.loc12_30.1: %C = bind_value %c2.ref.loc12
// CHECK:STDOUT:   %.loc12_25.2: ref %C = value_as_ref %.loc12_25.1
// CHECK:STDOUT:   %addr.loc12_28.1: %ptr.d9e = addr_of %.loc12_25.2
// CHECK:STDOUT:   %.loc12_30.2: ref %C = value_as_ref %.loc12_30.1
// CHECK:STDOUT:   %addr.loc12_28.2: %ptr.d9e = addr_of %.loc12_30.2
// CHECK:STDOUT:   %addr.loc12_28.3: %ptr.d9e = addr_of %.loc12_28.1
// CHECK:STDOUT:   %operator+__carbon_thunk.call: init %empty_tuple.type = call imports.%operator+__carbon_thunk.decl(%addr.loc12_28.1, %addr.loc12_28.2, %addr.loc12_28.3)
// CHECK:STDOUT:   %.loc12_28.2: init %C = in_place_init %operator+__carbon_thunk.call, %.loc12_28.1
// CHECK:STDOUT:   %.loc12_20: type = splice_block %C.ref.loc12 [concrete = constants.%C] {
// CHECK:STDOUT:     %Cpp.ref.loc12: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %C.ref.loc12: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc12_28.3: ref %C = temporary %.loc12_28.1, %.loc12_28.2
// CHECK:STDOUT:   %.loc12_28.4: %C = bind_value %.loc12_28.3
// CHECK:STDOUT:   %addition: %C = bind_name addition, %.loc12_28.4
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %subtraction.patt: %pattern_type.217 = binding_pattern subtraction [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c1.ref.loc13: ref %C = name_ref c1, %c1
// CHECK:STDOUT:   %c2.ref.loc13: ref %C = name_ref c2, %c2
// CHECK:STDOUT:   %.loc13_31.1: ref %C = temporary_storage
// CHECK:STDOUT:   %.loc13_28.1: %C = bind_value %c1.ref.loc13
// CHECK:STDOUT:   %.loc13_33.1: %C = bind_value %c2.ref.loc13
// CHECK:STDOUT:   %.loc13_28.2: ref %C = value_as_ref %.loc13_28.1
// CHECK:STDOUT:   %addr.loc13_31.1: %ptr.d9e = addr_of %.loc13_28.2
// CHECK:STDOUT:   %.loc13_33.2: ref %C = value_as_ref %.loc13_33.1
// CHECK:STDOUT:   %addr.loc13_31.2: %ptr.d9e = addr_of %.loc13_33.2
// CHECK:STDOUT:   %addr.loc13_31.3: %ptr.d9e = addr_of %.loc13_31.1
// CHECK:STDOUT:   %operator-__carbon_thunk.call: init %empty_tuple.type = call imports.%operator-__carbon_thunk.decl(%addr.loc13_31.1, %addr.loc13_31.2, %addr.loc13_31.3)
// CHECK:STDOUT:   %.loc13_31.2: init %C = in_place_init %operator-__carbon_thunk.call, %.loc13_31.1
// CHECK:STDOUT:   %.loc13_23: type = splice_block %C.ref.loc13 [concrete = constants.%C] {
// CHECK:STDOUT:     %Cpp.ref.loc13: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %C.ref.loc13: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc13_31.3: ref %C = temporary %.loc13_31.1, %.loc13_31.2
// CHECK:STDOUT:   %.loc13_31.4: %C = bind_value %.loc13_31.3
// CHECK:STDOUT:   %subtraction: %C = bind_name subtraction, %.loc13_31.4
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %multiplication.patt: %pattern_type.217 = binding_pattern multiplication [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c1.ref.loc14: ref %C = name_ref c1, %c1
// CHECK:STDOUT:   %c2.ref.loc14: ref %C = name_ref c2, %c2
// CHECK:STDOUT:   %.loc14_34.1: ref %C = temporary_storage
// CHECK:STDOUT:   %.loc14_31.1: %C = bind_value %c1.ref.loc14
// CHECK:STDOUT:   %.loc14_36.1: %C = bind_value %c2.ref.loc14
// CHECK:STDOUT:   %.loc14_31.2: ref %C = value_as_ref %.loc14_31.1
// CHECK:STDOUT:   %addr.loc14_34.1: %ptr.d9e = addr_of %.loc14_31.2
// CHECK:STDOUT:   %.loc14_36.2: ref %C = value_as_ref %.loc14_36.1
// CHECK:STDOUT:   %addr.loc14_34.2: %ptr.d9e = addr_of %.loc14_36.2
// CHECK:STDOUT:   %addr.loc14_34.3: %ptr.d9e = addr_of %.loc14_34.1
// CHECK:STDOUT:   %operator*__carbon_thunk.call: init %empty_tuple.type = call imports.%operator*__carbon_thunk.decl(%addr.loc14_34.1, %addr.loc14_34.2, %addr.loc14_34.3)
// CHECK:STDOUT:   %.loc14_34.2: init %C = in_place_init %operator*__carbon_thunk.call, %.loc14_34.1
// CHECK:STDOUT:   %.loc14_26: type = splice_block %C.ref.loc14 [concrete = constants.%C] {
// CHECK:STDOUT:     %Cpp.ref.loc14: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %C.ref.loc14: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc14_34.3: ref %C = temporary %.loc14_34.1, %.loc14_34.2
// CHECK:STDOUT:   %.loc14_34.4: %C = bind_value %.loc14_34.3
// CHECK:STDOUT:   %multiplication: %C = bind_name multiplication, %.loc14_34.4
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %division.patt: %pattern_type.217 = binding_pattern division [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c1.ref.loc15: ref %C = name_ref c1, %c1
// CHECK:STDOUT:   %c2.ref.loc15: ref %C = name_ref c2, %c2
// CHECK:STDOUT:   %.loc15_28.1: ref %C = temporary_storage
// CHECK:STDOUT:   %.loc15_25.1: %C = bind_value %c1.ref.loc15
// CHECK:STDOUT:   %.loc15_30.1: %C = bind_value %c2.ref.loc15
// CHECK:STDOUT:   %.loc15_25.2: ref %C = value_as_ref %.loc15_25.1
// CHECK:STDOUT:   %addr.loc15_28.1: %ptr.d9e = addr_of %.loc15_25.2
// CHECK:STDOUT:   %.loc15_30.2: ref %C = value_as_ref %.loc15_30.1
// CHECK:STDOUT:   %addr.loc15_28.2: %ptr.d9e = addr_of %.loc15_30.2
// CHECK:STDOUT:   %addr.loc15_28.3: %ptr.d9e = addr_of %.loc15_28.1
// CHECK:STDOUT:   %operator/__carbon_thunk.call: init %empty_tuple.type = call imports.%operator/__carbon_thunk.decl(%addr.loc15_28.1, %addr.loc15_28.2, %addr.loc15_28.3)
// CHECK:STDOUT:   %.loc15_28.2: init %C = in_place_init %operator/__carbon_thunk.call, %.loc15_28.1
// CHECK:STDOUT:   %.loc15_20: type = splice_block %C.ref.loc15 [concrete = constants.%C] {
// CHECK:STDOUT:     %Cpp.ref.loc15: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %C.ref.loc15: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc15_28.3: ref %C = temporary %.loc15_28.1, %.loc15_28.2
// CHECK:STDOUT:   %.loc15_28.4: %C = bind_value %.loc15_28.3
// CHECK:STDOUT:   %division: %C = bind_name division, %.loc15_28.4
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %modulo.patt: %pattern_type.217 = binding_pattern modulo [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c1.ref.loc16: ref %C = name_ref c1, %c1
// CHECK:STDOUT:   %c2.ref.loc16: ref %C = name_ref c2, %c2
// CHECK:STDOUT:   %.loc16_26.1: ref %C = temporary_storage
// CHECK:STDOUT:   %.loc16_23.1: %C = bind_value %c1.ref.loc16
// CHECK:STDOUT:   %.loc16_28.1: %C = bind_value %c2.ref.loc16
// CHECK:STDOUT:   %.loc16_23.2: ref %C = value_as_ref %.loc16_23.1
// CHECK:STDOUT:   %addr.loc16_26.1: %ptr.d9e = addr_of %.loc16_23.2
// CHECK:STDOUT:   %.loc16_28.2: ref %C = value_as_ref %.loc16_28.1
// CHECK:STDOUT:   %addr.loc16_26.2: %ptr.d9e = addr_of %.loc16_28.2
// CHECK:STDOUT:   %addr.loc16_26.3: %ptr.d9e = addr_of %.loc16_26.1
// CHECK:STDOUT:   %operator%__carbon_thunk.call: init %empty_tuple.type = call imports.%operator%__carbon_thunk.decl(%addr.loc16_26.1, %addr.loc16_26.2, %addr.loc16_26.3)
// CHECK:STDOUT:   %.loc16_26.2: init %C = in_place_init %operator%__carbon_thunk.call, %.loc16_26.1
// CHECK:STDOUT:   %.loc16_18: type = splice_block %C.ref.loc16 [concrete = constants.%C] {
// CHECK:STDOUT:     %Cpp.ref.loc16: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %C.ref.loc16: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc16_26.3: ref %C = temporary %.loc16_26.1, %.loc16_26.2
// CHECK:STDOUT:   %.loc16_26.4: %C = bind_value %.loc16_26.3
// CHECK:STDOUT:   %modulo: %C = bind_name modulo, %.loc16_26.4
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %bitwise_and.patt: %pattern_type.217 = binding_pattern bitwise_and [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c1.ref.loc19: ref %C = name_ref c1, %c1
// CHECK:STDOUT:   %c2.ref.loc19: ref %C = name_ref c2, %c2
// CHECK:STDOUT:   %.loc19_31.1: ref %C = temporary_storage
// CHECK:STDOUT:   %.loc19_28.1: %C = bind_value %c1.ref.loc19
// CHECK:STDOUT:   %.loc19_33.1: %C = bind_value %c2.ref.loc19
// CHECK:STDOUT:   %.loc19_28.2: ref %C = value_as_ref %.loc19_28.1
// CHECK:STDOUT:   %addr.loc19_31.1: %ptr.d9e = addr_of %.loc19_28.2
// CHECK:STDOUT:   %.loc19_33.2: ref %C = value_as_ref %.loc19_33.1
// CHECK:STDOUT:   %addr.loc19_31.2: %ptr.d9e = addr_of %.loc19_33.2
// CHECK:STDOUT:   %addr.loc19_31.3: %ptr.d9e = addr_of %.loc19_31.1
// CHECK:STDOUT:   %operator&__carbon_thunk.call: init %empty_tuple.type = call imports.%operator&__carbon_thunk.decl(%addr.loc19_31.1, %addr.loc19_31.2, %addr.loc19_31.3)
// CHECK:STDOUT:   %.loc19_31.2: init %C = in_place_init %operator&__carbon_thunk.call, %.loc19_31.1
// CHECK:STDOUT:   %.loc19_23: type = splice_block %C.ref.loc19 [concrete = constants.%C] {
// CHECK:STDOUT:     %Cpp.ref.loc19: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %C.ref.loc19: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc19_31.3: ref %C = temporary %.loc19_31.1, %.loc19_31.2
// CHECK:STDOUT:   %.loc19_31.4: %C = bind_value %.loc19_31.3
// CHECK:STDOUT:   %bitwise_and: %C = bind_name bitwise_and, %.loc19_31.4
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %bitwise_or.patt: %pattern_type.217 = binding_pattern bitwise_or [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c1.ref.loc20: ref %C = name_ref c1, %c1
// CHECK:STDOUT:   %c2.ref.loc20: ref %C = name_ref c2, %c2
// CHECK:STDOUT:   %.loc20_30.1: ref %C = temporary_storage
// CHECK:STDOUT:   %.loc20_27.1: %C = bind_value %c1.ref.loc20
// CHECK:STDOUT:   %.loc20_32.1: %C = bind_value %c2.ref.loc20
// CHECK:STDOUT:   %.loc20_27.2: ref %C = value_as_ref %.loc20_27.1
// CHECK:STDOUT:   %addr.loc20_30.1: %ptr.d9e = addr_of %.loc20_27.2
// CHECK:STDOUT:   %.loc20_32.2: ref %C = value_as_ref %.loc20_32.1
// CHECK:STDOUT:   %addr.loc20_30.2: %ptr.d9e = addr_of %.loc20_32.2
// CHECK:STDOUT:   %addr.loc20_30.3: %ptr.d9e = addr_of %.loc20_30.1
// CHECK:STDOUT:   %operator|__carbon_thunk.call: init %empty_tuple.type = call imports.%operator|__carbon_thunk.decl(%addr.loc20_30.1, %addr.loc20_30.2, %addr.loc20_30.3)
// CHECK:STDOUT:   %.loc20_30.2: init %C = in_place_init %operator|__carbon_thunk.call, %.loc20_30.1
// CHECK:STDOUT:   %.loc20_22: type = splice_block %C.ref.loc20 [concrete = constants.%C] {
// CHECK:STDOUT:     %Cpp.ref.loc20: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %C.ref.loc20: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc20_30.3: ref %C = temporary %.loc20_30.1, %.loc20_30.2
// CHECK:STDOUT:   %.loc20_30.4: %C = bind_value %.loc20_30.3
// CHECK:STDOUT:   %bitwise_or: %C = bind_name bitwise_or, %.loc20_30.4
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %bitwise_xor.patt: %pattern_type.217 = binding_pattern bitwise_xor [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c1.ref.loc21: ref %C = name_ref c1, %c1
// CHECK:STDOUT:   %c2.ref.loc21: ref %C = name_ref c2, %c2
// CHECK:STDOUT:   %.loc21_31.1: ref %C = temporary_storage
// CHECK:STDOUT:   %.loc21_28.1: %C = bind_value %c1.ref.loc21
// CHECK:STDOUT:   %.loc21_33.1: %C = bind_value %c2.ref.loc21
// CHECK:STDOUT:   %.loc21_28.2: ref %C = value_as_ref %.loc21_28.1
// CHECK:STDOUT:   %addr.loc21_31.1: %ptr.d9e = addr_of %.loc21_28.2
// CHECK:STDOUT:   %.loc21_33.2: ref %C = value_as_ref %.loc21_33.1
// CHECK:STDOUT:   %addr.loc21_31.2: %ptr.d9e = addr_of %.loc21_33.2
// CHECK:STDOUT:   %addr.loc21_31.3: %ptr.d9e = addr_of %.loc21_31.1
// CHECK:STDOUT:   %operator^__carbon_thunk.call: init %empty_tuple.type = call imports.%operator^__carbon_thunk.decl(%addr.loc21_31.1, %addr.loc21_31.2, %addr.loc21_31.3)
// CHECK:STDOUT:   %.loc21_31.2: init %C = in_place_init %operator^__carbon_thunk.call, %.loc21_31.1
// CHECK:STDOUT:   %.loc21_23: type = splice_block %C.ref.loc21 [concrete = constants.%C] {
// CHECK:STDOUT:     %Cpp.ref.loc21: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %C.ref.loc21: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc21_31.3: ref %C = temporary %.loc21_31.1, %.loc21_31.2
// CHECK:STDOUT:   %.loc21_31.4: %C = bind_value %.loc21_31.3
// CHECK:STDOUT:   %bitwise_xor: %C = bind_name bitwise_xor, %.loc21_31.4
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %left_shift.patt: %pattern_type.217 = binding_pattern left_shift [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c1.ref.loc22: ref %C = name_ref c1, %c1
// CHECK:STDOUT:   %int_3: Core.IntLiteral = int_value 3 [concrete = constants.%int_3.1ba]
// CHECK:STDOUT:   %.loc22_30.1: ref %C = temporary_storage
// CHECK:STDOUT:   %.loc22_27.1: %C = bind_value %c1.ref.loc22
// CHECK:STDOUT:   %impl.elem0.loc22: %.1df = impl_witness_access constants.%ImplicitAs.impl_witness.204, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0f0]
// CHECK:STDOUT:   %bound_method.loc22_33.1: <bound method> = bound_method %int_3, %impl.elem0.loc22 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.595]
// CHECK:STDOUT:   %specific_fn.loc22: <specific function> = specific_function %impl.elem0.loc22, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc22_33.2: <bound method> = bound_method %int_3, %specific_fn.loc22 [concrete = constants.%bound_method.f36]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc22: init %i32 = call %bound_method.loc22_33.2(%int_3) [concrete = constants.%int_3.822]
// CHECK:STDOUT:   %.loc22_33.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc22 [concrete = constants.%int_3.822]
// CHECK:STDOUT:   %.loc22_33.2: %i32 = converted %int_3, %.loc22_33.1 [concrete = constants.%int_3.822]
// CHECK:STDOUT:   %.loc22_27.2: ref %C = value_as_ref %.loc22_27.1
// CHECK:STDOUT:   %addr.loc22_30.1: %ptr.d9e = addr_of %.loc22_27.2
// CHECK:STDOUT:   %addr.loc22_30.2: %ptr.d9e = addr_of %.loc22_30.1
// CHECK:STDOUT:   %operator<<__carbon_thunk.call: init %empty_tuple.type = call imports.%operator<<__carbon_thunk.decl(%addr.loc22_30.1, %.loc22_33.2, %addr.loc22_30.2)
// CHECK:STDOUT:   %.loc22_30.2: init %C = in_place_init %operator<<__carbon_thunk.call, %.loc22_30.1
// CHECK:STDOUT:   %.loc22_22: type = splice_block %C.ref.loc22 [concrete = constants.%C] {
// CHECK:STDOUT:     %Cpp.ref.loc22: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %C.ref.loc22: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc22_30.3: ref %C = temporary %.loc22_30.1, %.loc22_30.2
// CHECK:STDOUT:   %.loc22_30.4: %C = bind_value %.loc22_30.3
// CHECK:STDOUT:   %left_shift: %C = bind_name left_shift, %.loc22_30.4
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %right_shift.patt: %pattern_type.217 = binding_pattern right_shift [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c1.ref.loc23: ref %C = name_ref c1, %c1
// CHECK:STDOUT:   %int_5: Core.IntLiteral = int_value 5 [concrete = constants.%int_5.64b]
// CHECK:STDOUT:   %.loc23_31.1: ref %C = temporary_storage
// CHECK:STDOUT:   %.loc23_28.1: %C = bind_value %c1.ref.loc23
// CHECK:STDOUT:   %impl.elem0.loc23: %.1df = impl_witness_access constants.%ImplicitAs.impl_witness.204, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0f0]
// CHECK:STDOUT:   %bound_method.loc23_34.1: <bound method> = bound_method %int_5, %impl.elem0.loc23 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.23d]
// CHECK:STDOUT:   %specific_fn.loc23: <specific function> = specific_function %impl.elem0.loc23, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc23_34.2: <bound method> = bound_method %int_5, %specific_fn.loc23 [concrete = constants.%bound_method.724]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc23: init %i32 = call %bound_method.loc23_34.2(%int_5) [concrete = constants.%int_5.0f6]
// CHECK:STDOUT:   %.loc23_34.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc23 [concrete = constants.%int_5.0f6]
// CHECK:STDOUT:   %.loc23_34.2: %i32 = converted %int_5, %.loc23_34.1 [concrete = constants.%int_5.0f6]
// CHECK:STDOUT:   %.loc23_28.2: ref %C = value_as_ref %.loc23_28.1
// CHECK:STDOUT:   %addr.loc23_31.1: %ptr.d9e = addr_of %.loc23_28.2
// CHECK:STDOUT:   %addr.loc23_31.2: %ptr.d9e = addr_of %.loc23_31.1
// CHECK:STDOUT:   %operator>>__carbon_thunk.call: init %empty_tuple.type = call imports.%operator>>__carbon_thunk.decl(%addr.loc23_31.1, %.loc23_34.2, %addr.loc23_31.2)
// CHECK:STDOUT:   %.loc23_31.2: init %C = in_place_init %operator>>__carbon_thunk.call, %.loc23_31.1
// CHECK:STDOUT:   %.loc23_23: type = splice_block %C.ref.loc23 [concrete = constants.%C] {
// CHECK:STDOUT:     %Cpp.ref.loc23: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %C.ref.loc23: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc23_31.3: ref %C = temporary %.loc23_31.1, %.loc23_31.2
// CHECK:STDOUT:   %.loc23_31.4: %C = bind_value %.loc23_31.3
// CHECK:STDOUT:   %right_shift: %C = bind_name right_shift, %.loc23_31.4
// CHECK:STDOUT:   %c1.ref.loc26: ref %C = name_ref c1, %c1
// CHECK:STDOUT:   %c2.ref.loc26: ref %C = name_ref c2, %c2
// CHECK:STDOUT:   %.loc26_6.1: ref %C = temporary_storage
// CHECK:STDOUT:   %addr.loc26_3: %ptr.d9e = addr_of %c1.ref.loc26
// CHECK:STDOUT:   %.loc26_9.1: %C = bind_value %c2.ref.loc26
// CHECK:STDOUT:   %.loc26_9.2: ref %C = value_as_ref %.loc26_9.1
// CHECK:STDOUT:   %addr.loc26_6.1: %ptr.d9e = addr_of %.loc26_9.2
// CHECK:STDOUT:   %addr.loc26_6.2: %ptr.d9e = addr_of %.loc26_6.1
// CHECK:STDOUT:   %operator+=__carbon_thunk.call: init %empty_tuple.type = call imports.%operator+=__carbon_thunk.decl(%addr.loc26_3, %addr.loc26_6.1, %addr.loc26_6.2)
// CHECK:STDOUT:   %.loc26_6.2: init %C = in_place_init %operator+=__carbon_thunk.call, %.loc26_6.1
// CHECK:STDOUT:   %.loc26_6.3: ref %C = temporary %.loc26_6.1, %.loc26_6.2
// CHECK:STDOUT:   %c1.ref.loc27: ref %C = name_ref c1, %c1
// CHECK:STDOUT:   %c2.ref.loc27: ref %C = name_ref c2, %c2
// CHECK:STDOUT:   %.loc27_6.1: ref %C = temporary_storage
// CHECK:STDOUT:   %addr.loc27_3: %ptr.d9e = addr_of %c1.ref.loc27
// CHECK:STDOUT:   %.loc27_9.1: %C = bind_value %c2.ref.loc27
// CHECK:STDOUT:   %.loc27_9.2: ref %C = value_as_ref %.loc27_9.1
// CHECK:STDOUT:   %addr.loc27_6.1: %ptr.d9e = addr_of %.loc27_9.2
// CHECK:STDOUT:   %addr.loc27_6.2: %ptr.d9e = addr_of %.loc27_6.1
// CHECK:STDOUT:   %operator-=__carbon_thunk.call: init %empty_tuple.type = call imports.%operator-=__carbon_thunk.decl(%addr.loc27_3, %addr.loc27_6.1, %addr.loc27_6.2)
// CHECK:STDOUT:   %.loc27_6.2: init %C = in_place_init %operator-=__carbon_thunk.call, %.loc27_6.1
// CHECK:STDOUT:   %.loc27_6.3: ref %C = temporary %.loc27_6.1, %.loc27_6.2
// CHECK:STDOUT:   %c1.ref.loc28: ref %C = name_ref c1, %c1
// CHECK:STDOUT:   %c2.ref.loc28: ref %C = name_ref c2, %c2
// CHECK:STDOUT:   %.loc28_6.1: ref %C = temporary_storage
// CHECK:STDOUT:   %addr.loc28_3: %ptr.d9e = addr_of %c1.ref.loc28
// CHECK:STDOUT:   %.loc28_9.1: %C = bind_value %c2.ref.loc28
// CHECK:STDOUT:   %.loc28_9.2: ref %C = value_as_ref %.loc28_9.1
// CHECK:STDOUT:   %addr.loc28_6.1: %ptr.d9e = addr_of %.loc28_9.2
// CHECK:STDOUT:   %addr.loc28_6.2: %ptr.d9e = addr_of %.loc28_6.1
// CHECK:STDOUT:   %operator*=__carbon_thunk.call: init %empty_tuple.type = call imports.%operator*=__carbon_thunk.decl(%addr.loc28_3, %addr.loc28_6.1, %addr.loc28_6.2)
// CHECK:STDOUT:   %.loc28_6.2: init %C = in_place_init %operator*=__carbon_thunk.call, %.loc28_6.1
// CHECK:STDOUT:   %.loc28_6.3: ref %C = temporary %.loc28_6.1, %.loc28_6.2
// CHECK:STDOUT:   %c1.ref.loc29: ref %C = name_ref c1, %c1
// CHECK:STDOUT:   %c2.ref.loc29: ref %C = name_ref c2, %c2
// CHECK:STDOUT:   %.loc29_6.1: ref %C = temporary_storage
// CHECK:STDOUT:   %addr.loc29_3: %ptr.d9e = addr_of %c1.ref.loc29
// CHECK:STDOUT:   %.loc29_9.1: %C = bind_value %c2.ref.loc29
// CHECK:STDOUT:   %.loc29_9.2: ref %C = value_as_ref %.loc29_9.1
// CHECK:STDOUT:   %addr.loc29_6.1: %ptr.d9e = addr_of %.loc29_9.2
// CHECK:STDOUT:   %addr.loc29_6.2: %ptr.d9e = addr_of %.loc29_6.1
// CHECK:STDOUT:   %operator/=__carbon_thunk.call: init %empty_tuple.type = call imports.%operator/=__carbon_thunk.decl(%addr.loc29_3, %addr.loc29_6.1, %addr.loc29_6.2)
// CHECK:STDOUT:   %.loc29_6.2: init %C = in_place_init %operator/=__carbon_thunk.call, %.loc29_6.1
// CHECK:STDOUT:   %.loc29_6.3: ref %C = temporary %.loc29_6.1, %.loc29_6.2
// CHECK:STDOUT:   %c1.ref.loc30: ref %C = name_ref c1, %c1
// CHECK:STDOUT:   %c2.ref.loc30: ref %C = name_ref c2, %c2
// CHECK:STDOUT:   %.loc30_6.1: ref %C = temporary_storage
// CHECK:STDOUT:   %addr.loc30_3: %ptr.d9e = addr_of %c1.ref.loc30
// CHECK:STDOUT:   %.loc30_9.1: %C = bind_value %c2.ref.loc30
// CHECK:STDOUT:   %.loc30_9.2: ref %C = value_as_ref %.loc30_9.1
// CHECK:STDOUT:   %addr.loc30_6.1: %ptr.d9e = addr_of %.loc30_9.2
// CHECK:STDOUT:   %addr.loc30_6.2: %ptr.d9e = addr_of %.loc30_6.1
// CHECK:STDOUT:   %operator%=__carbon_thunk.call: init %empty_tuple.type = call imports.%operator%=__carbon_thunk.decl(%addr.loc30_3, %addr.loc30_6.1, %addr.loc30_6.2)
// CHECK:STDOUT:   %.loc30_6.2: init %C = in_place_init %operator%=__carbon_thunk.call, %.loc30_6.1
// CHECK:STDOUT:   %.loc30_6.3: ref %C = temporary %.loc30_6.1, %.loc30_6.2
// CHECK:STDOUT:   %c1.ref.loc33: ref %C = name_ref c1, %c1
// CHECK:STDOUT:   %c2.ref.loc33: ref %C = name_ref c2, %c2
// CHECK:STDOUT:   %.loc33_6.1: ref %C = temporary_storage
// CHECK:STDOUT:   %addr.loc33_3: %ptr.d9e = addr_of %c1.ref.loc33
// CHECK:STDOUT:   %.loc33_9.1: %C = bind_value %c2.ref.loc33
// CHECK:STDOUT:   %.loc33_9.2: ref %C = value_as_ref %.loc33_9.1
// CHECK:STDOUT:   %addr.loc33_6.1: %ptr.d9e = addr_of %.loc33_9.2
// CHECK:STDOUT:   %addr.loc33_6.2: %ptr.d9e = addr_of %.loc33_6.1
// CHECK:STDOUT:   %operator&=__carbon_thunk.call: init %empty_tuple.type = call imports.%operator&=__carbon_thunk.decl(%addr.loc33_3, %addr.loc33_6.1, %addr.loc33_6.2)
// CHECK:STDOUT:   %.loc33_6.2: init %C = in_place_init %operator&=__carbon_thunk.call, %.loc33_6.1
// CHECK:STDOUT:   %.loc33_6.3: ref %C = temporary %.loc33_6.1, %.loc33_6.2
// CHECK:STDOUT:   %c1.ref.loc34: ref %C = name_ref c1, %c1
// CHECK:STDOUT:   %c2.ref.loc34: ref %C = name_ref c2, %c2
// CHECK:STDOUT:   %.loc34_6.1: ref %C = temporary_storage
// CHECK:STDOUT:   %addr.loc34_3: %ptr.d9e = addr_of %c1.ref.loc34
// CHECK:STDOUT:   %.loc34_9.1: %C = bind_value %c2.ref.loc34
// CHECK:STDOUT:   %.loc34_9.2: ref %C = value_as_ref %.loc34_9.1
// CHECK:STDOUT:   %addr.loc34_6.1: %ptr.d9e = addr_of %.loc34_9.2
// CHECK:STDOUT:   %addr.loc34_6.2: %ptr.d9e = addr_of %.loc34_6.1
// CHECK:STDOUT:   %operator|=__carbon_thunk.call: init %empty_tuple.type = call imports.%operator|=__carbon_thunk.decl(%addr.loc34_3, %addr.loc34_6.1, %addr.loc34_6.2)
// CHECK:STDOUT:   %.loc34_6.2: init %C = in_place_init %operator|=__carbon_thunk.call, %.loc34_6.1
// CHECK:STDOUT:   %.loc34_6.3: ref %C = temporary %.loc34_6.1, %.loc34_6.2
// CHECK:STDOUT:   %c1.ref.loc35: ref %C = name_ref c1, %c1
// CHECK:STDOUT:   %c2.ref.loc35: ref %C = name_ref c2, %c2
// CHECK:STDOUT:   %.loc35_6.1: ref %C = temporary_storage
// CHECK:STDOUT:   %addr.loc35_3: %ptr.d9e = addr_of %c1.ref.loc35
// CHECK:STDOUT:   %.loc35_9.1: %C = bind_value %c2.ref.loc35
// CHECK:STDOUT:   %.loc35_9.2: ref %C = value_as_ref %.loc35_9.1
// CHECK:STDOUT:   %addr.loc35_6.1: %ptr.d9e = addr_of %.loc35_9.2
// CHECK:STDOUT:   %addr.loc35_6.2: %ptr.d9e = addr_of %.loc35_6.1
// CHECK:STDOUT:   %operator^=__carbon_thunk.call: init %empty_tuple.type = call imports.%operator^=__carbon_thunk.decl(%addr.loc35_3, %addr.loc35_6.1, %addr.loc35_6.2)
// CHECK:STDOUT:   %.loc35_6.2: init %C = in_place_init %operator^=__carbon_thunk.call, %.loc35_6.1
// CHECK:STDOUT:   %.loc35_6.3: ref %C = temporary %.loc35_6.1, %.loc35_6.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %equal.patt: %pattern_type.831 = binding_pattern equal [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c1.ref.loc38: ref %C = name_ref c1, %c1
// CHECK:STDOUT:   %c2.ref.loc38: ref %C = name_ref c2, %c2
// CHECK:STDOUT:   %.loc38_21.1: %C = bind_value %c1.ref.loc38
// CHECK:STDOUT:   %.loc38_27.1: %C = bind_value %c2.ref.loc38
// CHECK:STDOUT:   %.loc38_21.2: ref %C = value_as_ref %.loc38_21.1
// CHECK:STDOUT:   %addr.loc38_24.1: %ptr.d9e = addr_of %.loc38_21.2
// CHECK:STDOUT:   %.loc38_27.2: ref %C = value_as_ref %.loc38_27.1
// CHECK:STDOUT:   %addr.loc38_24.2: %ptr.d9e = addr_of %.loc38_27.2
// CHECK:STDOUT:   %.loc38_24.1: ref bool = temporary_storage
// CHECK:STDOUT:   %addr.loc38_24.3: %ptr.bb2 = addr_of %.loc38_24.1
// CHECK:STDOUT:   %operator==__carbon_thunk.call: init %empty_tuple.type = call imports.%operator==__carbon_thunk.decl(%addr.loc38_24.1, %addr.loc38_24.2, %addr.loc38_24.3)
// CHECK:STDOUT:   %.loc38_24.2: init bool = in_place_init %operator==__carbon_thunk.call, %.loc38_24.1
// CHECK:STDOUT:   %.loc38_14.1: type = splice_block %.loc38_14.3 [concrete = bool] {
// CHECK:STDOUT:     %Bool.call.loc38: init type = call constants.%Bool() [concrete = bool]
// CHECK:STDOUT:     %.loc38_14.2: type = value_of_initializer %Bool.call.loc38 [concrete = bool]
// CHECK:STDOUT:     %.loc38_14.3: type = converted %Bool.call.loc38, %.loc38_14.2 [concrete = bool]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc38_24.3: bool = value_of_initializer %.loc38_24.2
// CHECK:STDOUT:   %.loc38_24.4: bool = converted %.loc38_24.2, %.loc38_24.3
// CHECK:STDOUT:   %equal: bool = bind_name equal, %.loc38_24.4
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %not_equal.patt: %pattern_type.831 = binding_pattern not_equal [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c1.ref.loc39: ref %C = name_ref c1, %c1
// CHECK:STDOUT:   %c2.ref.loc39: ref %C = name_ref c2, %c2
// CHECK:STDOUT:   %.loc39_25.1: %C = bind_value %c1.ref.loc39
// CHECK:STDOUT:   %.loc39_31.1: %C = bind_value %c2.ref.loc39
// CHECK:STDOUT:   %.loc39_25.2: ref %C = value_as_ref %.loc39_25.1
// CHECK:STDOUT:   %addr.loc39_28.1: %ptr.d9e = addr_of %.loc39_25.2
// CHECK:STDOUT:   %.loc39_31.2: ref %C = value_as_ref %.loc39_31.1
// CHECK:STDOUT:   %addr.loc39_28.2: %ptr.d9e = addr_of %.loc39_31.2
// CHECK:STDOUT:   %.loc39_28.1: ref bool = temporary_storage
// CHECK:STDOUT:   %addr.loc39_28.3: %ptr.bb2 = addr_of %.loc39_28.1
// CHECK:STDOUT:   %operator!=__carbon_thunk.call: init %empty_tuple.type = call imports.%operator!=__carbon_thunk.decl(%addr.loc39_28.1, %addr.loc39_28.2, %addr.loc39_28.3)
// CHECK:STDOUT:   %.loc39_28.2: init bool = in_place_init %operator!=__carbon_thunk.call, %.loc39_28.1
// CHECK:STDOUT:   %.loc39_18.1: type = splice_block %.loc39_18.3 [concrete = bool] {
// CHECK:STDOUT:     %Bool.call.loc39: init type = call constants.%Bool() [concrete = bool]
// CHECK:STDOUT:     %.loc39_18.2: type = value_of_initializer %Bool.call.loc39 [concrete = bool]
// CHECK:STDOUT:     %.loc39_18.3: type = converted %Bool.call.loc39, %.loc39_18.2 [concrete = bool]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc39_28.3: bool = value_of_initializer %.loc39_28.2
// CHECK:STDOUT:   %.loc39_28.4: bool = converted %.loc39_28.2, %.loc39_28.3
// CHECK:STDOUT:   %not_equal: bool = bind_name not_equal, %.loc39_28.4
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %greater_than.patt: %pattern_type.831 = binding_pattern greater_than [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c1.ref.loc40: ref %C = name_ref c1, %c1
// CHECK:STDOUT:   %c2.ref.loc40: ref %C = name_ref c2, %c2
// CHECK:STDOUT:   %.loc40_28.1: %C = bind_value %c1.ref.loc40
// CHECK:STDOUT:   %.loc40_33.1: %C = bind_value %c2.ref.loc40
// CHECK:STDOUT:   %.loc40_28.2: ref %C = value_as_ref %.loc40_28.1
// CHECK:STDOUT:   %addr.loc40_31.1: %ptr.d9e = addr_of %.loc40_28.2
// CHECK:STDOUT:   %.loc40_33.2: ref %C = value_as_ref %.loc40_33.1
// CHECK:STDOUT:   %addr.loc40_31.2: %ptr.d9e = addr_of %.loc40_33.2
// CHECK:STDOUT:   %.loc40_31.1: ref bool = temporary_storage
// CHECK:STDOUT:   %addr.loc40_31.3: %ptr.bb2 = addr_of %.loc40_31.1
// CHECK:STDOUT:   %operator>__carbon_thunk.call: init %empty_tuple.type = call imports.%operator>__carbon_thunk.decl(%addr.loc40_31.1, %addr.loc40_31.2, %addr.loc40_31.3)
// CHECK:STDOUT:   %.loc40_31.2: init bool = in_place_init %operator>__carbon_thunk.call, %.loc40_31.1
// CHECK:STDOUT:   %.loc40_21.1: type = splice_block %.loc40_21.3 [concrete = bool] {
// CHECK:STDOUT:     %Bool.call.loc40: init type = call constants.%Bool() [concrete = bool]
// CHECK:STDOUT:     %.loc40_21.2: type = value_of_initializer %Bool.call.loc40 [concrete = bool]
// CHECK:STDOUT:     %.loc40_21.3: type = converted %Bool.call.loc40, %.loc40_21.2 [concrete = bool]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc40_31.3: bool = value_of_initializer %.loc40_31.2
// CHECK:STDOUT:   %.loc40_31.4: bool = converted %.loc40_31.2, %.loc40_31.3
// CHECK:STDOUT:   %greater_than: bool = bind_name greater_than, %.loc40_31.4
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %less_than.patt: %pattern_type.831 = binding_pattern less_than [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c1.ref.loc41: ref %C = name_ref c1, %c1
// CHECK:STDOUT:   %c2.ref.loc41: ref %C = name_ref c2, %c2
// CHECK:STDOUT:   %.loc41_25.1: %C = bind_value %c1.ref.loc41
// CHECK:STDOUT:   %.loc41_30.1: %C = bind_value %c2.ref.loc41
// CHECK:STDOUT:   %.loc41_25.2: ref %C = value_as_ref %.loc41_25.1
// CHECK:STDOUT:   %addr.loc41_28.1: %ptr.d9e = addr_of %.loc41_25.2
// CHECK:STDOUT:   %.loc41_30.2: ref %C = value_as_ref %.loc41_30.1
// CHECK:STDOUT:   %addr.loc41_28.2: %ptr.d9e = addr_of %.loc41_30.2
// CHECK:STDOUT:   %.loc41_28.1: ref bool = temporary_storage
// CHECK:STDOUT:   %addr.loc41_28.3: %ptr.bb2 = addr_of %.loc41_28.1
// CHECK:STDOUT:   %operator<__carbon_thunk.call: init %empty_tuple.type = call imports.%operator<__carbon_thunk.decl(%addr.loc41_28.1, %addr.loc41_28.2, %addr.loc41_28.3)
// CHECK:STDOUT:   %.loc41_28.2: init bool = in_place_init %operator<__carbon_thunk.call, %.loc41_28.1
// CHECK:STDOUT:   %.loc41_18.1: type = splice_block %.loc41_18.3 [concrete = bool] {
// CHECK:STDOUT:     %Bool.call.loc41: init type = call constants.%Bool() [concrete = bool]
// CHECK:STDOUT:     %.loc41_18.2: type = value_of_initializer %Bool.call.loc41 [concrete = bool]
// CHECK:STDOUT:     %.loc41_18.3: type = converted %Bool.call.loc41, %.loc41_18.2 [concrete = bool]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc41_28.3: bool = value_of_initializer %.loc41_28.2
// CHECK:STDOUT:   %.loc41_28.4: bool = converted %.loc41_28.2, %.loc41_28.3
// CHECK:STDOUT:   %less_than: bool = bind_name less_than, %.loc41_28.4
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %greater_than_or_equal.patt: %pattern_type.831 = binding_pattern greater_than_or_equal [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c1.ref.loc42: ref %C = name_ref c1, %c1
// CHECK:STDOUT:   %c2.ref.loc42: ref %C = name_ref c2, %c2
// CHECK:STDOUT:   %.loc42_37.1: %C = bind_value %c1.ref.loc42
// CHECK:STDOUT:   %.loc42_43.1: %C = bind_value %c2.ref.loc42
// CHECK:STDOUT:   %.loc42_37.2: ref %C = value_as_ref %.loc42_37.1
// CHECK:STDOUT:   %addr.loc42_40.1: %ptr.d9e = addr_of %.loc42_37.2
// CHECK:STDOUT:   %.loc42_43.2: ref %C = value_as_ref %.loc42_43.1
// CHECK:STDOUT:   %addr.loc42_40.2: %ptr.d9e = addr_of %.loc42_43.2
// CHECK:STDOUT:   %.loc42_40.1: ref bool = temporary_storage
// CHECK:STDOUT:   %addr.loc42_40.3: %ptr.bb2 = addr_of %.loc42_40.1
// CHECK:STDOUT:   %operator>=__carbon_thunk.call: init %empty_tuple.type = call imports.%operator>=__carbon_thunk.decl(%addr.loc42_40.1, %addr.loc42_40.2, %addr.loc42_40.3)
// CHECK:STDOUT:   %.loc42_40.2: init bool = in_place_init %operator>=__carbon_thunk.call, %.loc42_40.1
// CHECK:STDOUT:   %.loc42_30.1: type = splice_block %.loc42_30.3 [concrete = bool] {
// CHECK:STDOUT:     %Bool.call.loc42: init type = call constants.%Bool() [concrete = bool]
// CHECK:STDOUT:     %.loc42_30.2: type = value_of_initializer %Bool.call.loc42 [concrete = bool]
// CHECK:STDOUT:     %.loc42_30.3: type = converted %Bool.call.loc42, %.loc42_30.2 [concrete = bool]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc42_40.3: bool = value_of_initializer %.loc42_40.2
// CHECK:STDOUT:   %.loc42_40.4: bool = converted %.loc42_40.2, %.loc42_40.3
// CHECK:STDOUT:   %greater_than_or_equal: bool = bind_name greater_than_or_equal, %.loc42_40.4
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %less_than_or_equal.patt: %pattern_type.831 = binding_pattern less_than_or_equal [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c1.ref.loc43: ref %C = name_ref c1, %c1
// CHECK:STDOUT:   %c2.ref.loc43: ref %C = name_ref c2, %c2
// CHECK:STDOUT:   %.loc43_34.1: %C = bind_value %c1.ref.loc43
// CHECK:STDOUT:   %.loc43_40.1: %C = bind_value %c2.ref.loc43
// CHECK:STDOUT:   %.loc43_34.2: ref %C = value_as_ref %.loc43_34.1
// CHECK:STDOUT:   %addr.loc43_37.1: %ptr.d9e = addr_of %.loc43_34.2
// CHECK:STDOUT:   %.loc43_40.2: ref %C = value_as_ref %.loc43_40.1
// CHECK:STDOUT:   %addr.loc43_37.2: %ptr.d9e = addr_of %.loc43_40.2
// CHECK:STDOUT:   %.loc43_37.1: ref bool = temporary_storage
// CHECK:STDOUT:   %addr.loc43_37.3: %ptr.bb2 = addr_of %.loc43_37.1
// CHECK:STDOUT:   %operator<=__carbon_thunk.call: init %empty_tuple.type = call imports.%operator<=__carbon_thunk.decl(%addr.loc43_37.1, %addr.loc43_37.2, %addr.loc43_37.3)
// CHECK:STDOUT:   %.loc43_37.2: init bool = in_place_init %operator<=__carbon_thunk.call, %.loc43_37.1
// CHECK:STDOUT:   %.loc43_27.1: type = splice_block %.loc43_27.3 [concrete = bool] {
// CHECK:STDOUT:     %Bool.call.loc43: init type = call constants.%Bool() [concrete = bool]
// CHECK:STDOUT:     %.loc43_27.2: type = value_of_initializer %Bool.call.loc43 [concrete = bool]
// CHECK:STDOUT:     %.loc43_27.3: type = converted %Bool.call.loc43, %.loc43_27.2 [concrete = bool]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc43_37.3: bool = value_of_initializer %.loc43_37.2
// CHECK:STDOUT:   %.loc43_37.4: bool = converted %.loc43_37.2, %.loc43_37.3
// CHECK:STDOUT:   %less_than_or_equal: bool = bind_name less_than_or_equal, %.loc43_37.4
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.bound.loc35: <bound method> = bound_method %.loc35_6.3, constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.841
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method.loc35: <bound method> = bound_method %.loc35_6.3, %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.1
// CHECK:STDOUT:   %addr.loc35_6.3: %ptr.d9e = addr_of %.loc35_6.3
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.call.loc35: init %empty_tuple.type = call %bound_method.loc35(%addr.loc35_6.3)
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.bound.loc34: <bound method> = bound_method %.loc34_6.3, constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.841
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method.loc34: <bound method> = bound_method %.loc34_6.3, %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.2
// CHECK:STDOUT:   %addr.loc34_6.3: %ptr.d9e = addr_of %.loc34_6.3
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.call.loc34: init %empty_tuple.type = call %bound_method.loc34(%addr.loc34_6.3)
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.bound.loc33: <bound method> = bound_method %.loc33_6.3, constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.841
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method.loc33: <bound method> = bound_method %.loc33_6.3, %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.3
// CHECK:STDOUT:   %addr.loc33_6.3: %ptr.d9e = addr_of %.loc33_6.3
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.call.loc33: init %empty_tuple.type = call %bound_method.loc33(%addr.loc33_6.3)
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.bound.loc30: <bound method> = bound_method %.loc30_6.3, constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.841
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method.loc30: <bound method> = bound_method %.loc30_6.3, %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.4
// CHECK:STDOUT:   %addr.loc30_6.3: %ptr.d9e = addr_of %.loc30_6.3
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.call.loc30: init %empty_tuple.type = call %bound_method.loc30(%addr.loc30_6.3)
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.bound.loc29: <bound method> = bound_method %.loc29_6.3, constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.841
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method.loc29: <bound method> = bound_method %.loc29_6.3, %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.5
// CHECK:STDOUT:   %addr.loc29_6.3: %ptr.d9e = addr_of %.loc29_6.3
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.call.loc29: init %empty_tuple.type = call %bound_method.loc29(%addr.loc29_6.3)
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.bound.loc28: <bound method> = bound_method %.loc28_6.3, constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.841
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method.loc28: <bound method> = bound_method %.loc28_6.3, %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.6
// CHECK:STDOUT:   %addr.loc28_6.3: %ptr.d9e = addr_of %.loc28_6.3
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.call.loc28: init %empty_tuple.type = call %bound_method.loc28(%addr.loc28_6.3)
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.bound.loc27: <bound method> = bound_method %.loc27_6.3, constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.841
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method.loc27: <bound method> = bound_method %.loc27_6.3, %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.7
// CHECK:STDOUT:   %addr.loc27_6.3: %ptr.d9e = addr_of %.loc27_6.3
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.call.loc27: init %empty_tuple.type = call %bound_method.loc27(%addr.loc27_6.3)
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.bound.loc26: <bound method> = bound_method %.loc26_6.3, constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.841
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method.loc26: <bound method> = bound_method %.loc26_6.3, %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.8
// CHECK:STDOUT:   %addr.loc26_6.3: %ptr.d9e = addr_of %.loc26_6.3
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.call.loc26: init %empty_tuple.type = call %bound_method.loc26(%addr.loc26_6.3)
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.bound.loc23: <bound method> = bound_method %.loc23_31.3, constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.841
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method.loc23_31: <bound method> = bound_method %.loc23_31.3, %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.9
// CHECK:STDOUT:   %addr.loc23_31.3: %ptr.d9e = addr_of %.loc23_31.3
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.call.loc23: init %empty_tuple.type = call %bound_method.loc23_31(%addr.loc23_31.3)
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.bound.loc22: <bound method> = bound_method %.loc22_30.3, constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.841
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method.loc22_30: <bound method> = bound_method %.loc22_30.3, %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.10
// CHECK:STDOUT:   %addr.loc22_30.3: %ptr.d9e = addr_of %.loc22_30.3
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.call.loc22: init %empty_tuple.type = call %bound_method.loc22_30(%addr.loc22_30.3)
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.bound.loc21: <bound method> = bound_method %.loc21_31.3, constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.841
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method.loc21: <bound method> = bound_method %.loc21_31.3, %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.11
// CHECK:STDOUT:   %addr.loc21_31.4: %ptr.d9e = addr_of %.loc21_31.3
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.call.loc21: init %empty_tuple.type = call %bound_method.loc21(%addr.loc21_31.4)
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.bound.loc20: <bound method> = bound_method %.loc20_30.3, constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.841
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method.loc20: <bound method> = bound_method %.loc20_30.3, %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.12
// CHECK:STDOUT:   %addr.loc20_30.4: %ptr.d9e = addr_of %.loc20_30.3
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.call.loc20: init %empty_tuple.type = call %bound_method.loc20(%addr.loc20_30.4)
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.bound.loc19: <bound method> = bound_method %.loc19_31.3, constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.841
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method.loc19: <bound method> = bound_method %.loc19_31.3, %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.13
// CHECK:STDOUT:   %addr.loc19_31.4: %ptr.d9e = addr_of %.loc19_31.3
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.call.loc19: init %empty_tuple.type = call %bound_method.loc19(%addr.loc19_31.4)
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.bound.loc16: <bound method> = bound_method %.loc16_26.3, constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.841
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method.loc16: <bound method> = bound_method %.loc16_26.3, %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.14
// CHECK:STDOUT:   %addr.loc16_26.4: %ptr.d9e = addr_of %.loc16_26.3
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.call.loc16: init %empty_tuple.type = call %bound_method.loc16(%addr.loc16_26.4)
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.bound.loc15: <bound method> = bound_method %.loc15_28.3, constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.841
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method.loc15: <bound method> = bound_method %.loc15_28.3, %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.15
// CHECK:STDOUT:   %addr.loc15_28.4: %ptr.d9e = addr_of %.loc15_28.3
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.call.loc15: init %empty_tuple.type = call %bound_method.loc15(%addr.loc15_28.4)
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.bound.loc14: <bound method> = bound_method %.loc14_34.3, constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.841
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method.loc14: <bound method> = bound_method %.loc14_34.3, %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.16
// CHECK:STDOUT:   %addr.loc14_34.4: %ptr.d9e = addr_of %.loc14_34.3
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.call.loc14: init %empty_tuple.type = call %bound_method.loc14(%addr.loc14_34.4)
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.bound.loc13: <bound method> = bound_method %.loc13_31.3, constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.841
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method.loc13: <bound method> = bound_method %.loc13_31.3, %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.17
// CHECK:STDOUT:   %addr.loc13_31.4: %ptr.d9e = addr_of %.loc13_31.3
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.call.loc13: init %empty_tuple.type = call %bound_method.loc13(%addr.loc13_31.4)
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.bound.loc12: <bound method> = bound_method %.loc12_28.3, constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.841
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method.loc12: <bound method> = bound_method %.loc12_28.3, %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.18
// CHECK:STDOUT:   %addr.loc12_28.4: %ptr.d9e = addr_of %.loc12_28.3
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.call.loc12: init %empty_tuple.type = call %bound_method.loc12(%addr.loc12_28.4)
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.bound.loc9: <bound method> = bound_method %c2.var, constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.841
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method.loc9: <bound method> = bound_method %c2.var, %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.19
// CHECK:STDOUT:   %addr.loc9_3: %ptr.d9e = addr_of %c2.var
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.call.loc9: init %empty_tuple.type = call %bound_method.loc9(%addr.loc9_3)
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.bound.loc8: <bound method> = bound_method %c1.var, constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.841
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method.loc8: <bound method> = bound_method %c1.var, %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.20
// CHECK:STDOUT:   %addr.loc8_3: %ptr.d9e = addr_of %c1.var
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.call.loc8: init %empty_tuple.type = call %bound_method.loc8(%addr.loc8_3)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- multiple_calls.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %pattern_type.217: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %C.C.cpp_overload_set.type: type = cpp_overload_set_type @C.C.cpp_overload_set [concrete]
// CHECK:STDOUT:   %C.C.cpp_overload_set.value: %C.C.cpp_overload_set.type = cpp_overload_set_value @C.C.cpp_overload_set [concrete]
// CHECK:STDOUT:   %ptr.d9e: type = ptr_type %C [concrete]
// CHECK:STDOUT:   %C__carbon_thunk.type: type = fn_type @C__carbon_thunk [concrete]
// CHECK:STDOUT:   %C__carbon_thunk: %C__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %operator+__carbon_thunk.type: type = fn_type @operator+__carbon_thunk [concrete]
// CHECK:STDOUT:   %operator+__carbon_thunk: %operator+__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %type_where: type = facet_type <type where .Self impls <CanDestroy>> [concrete]
// CHECK:STDOUT:   %facet_value: %type_where = facet_value %C, () [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.type.b92: type = fn_type @DestroyT.binding.as_type.as.Destroy.impl.Op, @DestroyT.binding.as_type.as.Destroy.impl(%facet_value) [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.841: %DestroyT.binding.as_type.as.Destroy.impl.Op.type.b92 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   %C.C.cpp_overload_set.value: %C.C.cpp_overload_set.type = cpp_overload_set_value @C.C.cpp_overload_set [concrete = constants.%C.C.cpp_overload_set.value]
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
// CHECK:STDOUT:   %C.ref.loc8_24: %C.C.cpp_overload_set.type = name_ref C, imports.%C.C.cpp_overload_set.value [concrete = constants.%C.C.cpp_overload_set.value]
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
// CHECK:STDOUT:   %C.ref.loc9_24: %C.C.cpp_overload_set.type = name_ref C, imports.%C.C.cpp_overload_set.value [concrete = constants.%C.C.cpp_overload_set.value]
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
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.bound.loc12: <bound method> = bound_method %.loc12_22.3, constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.841
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method.loc12: <bound method> = bound_method %.loc12_22.3, %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.1
// CHECK:STDOUT:   %addr.loc12_22.4: %ptr.d9e = addr_of %.loc12_22.3
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.call.loc12: init %empty_tuple.type = call %bound_method.loc12(%addr.loc12_22.4)
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.bound.loc11: <bound method> = bound_method %.loc11_22.3, constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.841
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method.loc11: <bound method> = bound_method %.loc11_22.3, %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.2
// CHECK:STDOUT:   %addr.loc11_22.4: %ptr.d9e = addr_of %.loc11_22.3
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.call.loc11: init %empty_tuple.type = call %bound_method.loc11(%addr.loc11_22.4)
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.bound.loc10: <bound method> = bound_method %.loc10_22.3, constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.841
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method.loc10: <bound method> = bound_method %.loc10_22.3, %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.3
// CHECK:STDOUT:   %addr.loc10_22.4: %ptr.d9e = addr_of %.loc10_22.3
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.call.loc10: init %empty_tuple.type = call %bound_method.loc10(%addr.loc10_22.4)
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.bound.loc9: <bound method> = bound_method %.loc9_27.3, constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.841
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method.loc9: <bound method> = bound_method %.loc9_27.3, %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.4
// CHECK:STDOUT:   %addr.loc9_27.2: %ptr.d9e = addr_of %.loc9_27.3
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.call.loc9: init %empty_tuple.type = call %bound_method.loc9(%addr.loc9_27.2)
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.bound.loc8: <bound method> = bound_method %.loc8_27.3, constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.841
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method.loc8: <bound method> = bound_method %.loc8_27.3, %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.5
// CHECK:STDOUT:   %addr.loc8_27.2: %ptr.d9e = addr_of %.loc8_27.3
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.call.loc8: init %empty_tuple.type = call %bound_method.loc8(%addr.loc8_27.2)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_todo_rewrite_spaceship.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %pattern_type.217: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %C.C.cpp_overload_set.type: type = cpp_overload_set_type @C.C.cpp_overload_set [concrete]
// CHECK:STDOUT:   %C.C.cpp_overload_set.value: %C.C.cpp_overload_set.type = cpp_overload_set_value @C.C.cpp_overload_set [concrete]
// CHECK:STDOUT:   %ptr.d9e: type = ptr_type %C [concrete]
// CHECK:STDOUT:   %C__carbon_thunk.type: type = fn_type @C__carbon_thunk [concrete]
// CHECK:STDOUT:   %C__carbon_thunk: %C__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %Bool.type: type = fn_type @Bool [concrete]
// CHECK:STDOUT:   %Bool: %Bool.type = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.831: type = pattern_type bool [concrete]
// CHECK:STDOUT:   %ptr.bb2: type = ptr_type bool [concrete]
// CHECK:STDOUT:   %operator>__carbon_thunk.type: type = fn_type @operator>__carbon_thunk [concrete]
// CHECK:STDOUT:   %operator>__carbon_thunk: %operator>__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %operator<=__carbon_thunk.type: type = fn_type @operator<=__carbon_thunk [concrete]
// CHECK:STDOUT:   %operator<=__carbon_thunk: %operator<=__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %type_where: type = facet_type <type where .Self impls <CanDestroy>> [concrete]
// CHECK:STDOUT:   %facet_value: %type_where = facet_value %C, () [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.type.b92: type = fn_type @DestroyT.binding.as_type.as.Destroy.impl.Op, @DestroyT.binding.as_type.as.Destroy.impl(%facet_value) [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.841: %DestroyT.binding.as_type.as.Destroy.impl.Op.type.b92 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   %C.C.cpp_overload_set.value: %C.C.cpp_overload_set.type = cpp_overload_set_value @C.C.cpp_overload_set [concrete = constants.%C.C.cpp_overload_set.value]
// CHECK:STDOUT:   %C__carbon_thunk.decl: %C__carbon_thunk.type = fn_decl @C__carbon_thunk [concrete = constants.%C__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %operator>__carbon_thunk.decl: %operator>__carbon_thunk.type = fn_decl @operator>__carbon_thunk [concrete = constants.%operator>__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %operator<=__carbon_thunk.decl: %operator<=__carbon_thunk.type = fn_decl @operator<=__carbon_thunk [concrete = constants.%operator<=__carbon_thunk] {
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
// CHECK:STDOUT:     %c1.var_patt: %pattern_type.217 = var_pattern %c1.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c1.var: ref %C = var %c1.var_patt
// CHECK:STDOUT:   %Cpp.ref.loc15_19: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %C.ref.loc15_22: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %C.ref.loc15_24: %C.C.cpp_overload_set.type = name_ref C, imports.%C.C.cpp_overload_set.value [concrete = constants.%C.C.cpp_overload_set.value]
// CHECK:STDOUT:   %.loc15_3: ref %C = splice_block %c1.var {}
// CHECK:STDOUT:   %addr.loc15_27: %ptr.d9e = addr_of %.loc15_3
// CHECK:STDOUT:   %C__carbon_thunk.call.loc15: init %empty_tuple.type = call imports.%C__carbon_thunk.decl(%addr.loc15_27)
// CHECK:STDOUT:   %.loc15_27: init %C = in_place_init %C__carbon_thunk.call.loc15, %.loc15_3
// CHECK:STDOUT:   assign %c1.var, %.loc15_27
// CHECK:STDOUT:   %.loc15_14: type = splice_block %C.ref.loc15_14 [concrete = constants.%C] {
// CHECK:STDOUT:     %Cpp.ref.loc15_11: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %C.ref.loc15_14: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c1: ref %C = bind_name c1, %c1.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c2.patt: %pattern_type.217 = binding_pattern c2 [concrete]
// CHECK:STDOUT:     %c2.var_patt: %pattern_type.217 = var_pattern %c2.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c2.var: ref %C = var %c2.var_patt
// CHECK:STDOUT:   %Cpp.ref.loc16_19: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %C.ref.loc16_22: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %C.ref.loc16_24: %C.C.cpp_overload_set.type = name_ref C, imports.%C.C.cpp_overload_set.value [concrete = constants.%C.C.cpp_overload_set.value]
// CHECK:STDOUT:   %.loc16_3: ref %C = splice_block %c2.var {}
// CHECK:STDOUT:   %addr.loc16_27: %ptr.d9e = addr_of %.loc16_3
// CHECK:STDOUT:   %C__carbon_thunk.call.loc16: init %empty_tuple.type = call imports.%C__carbon_thunk.decl(%addr.loc16_27)
// CHECK:STDOUT:   %.loc16_27: init %C = in_place_init %C__carbon_thunk.call.loc16, %.loc16_3
// CHECK:STDOUT:   assign %c2.var, %.loc16_27
// CHECK:STDOUT:   %.loc16_14: type = splice_block %C.ref.loc16_14 [concrete = constants.%C] {
// CHECK:STDOUT:     %Cpp.ref.loc16_11: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %C.ref.loc16_14: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c2: ref %C = bind_name c2, %c2.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %greater_than.patt: %pattern_type.831 = binding_pattern greater_than [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c1.ref.loc19: ref %C = name_ref c1, %c1
// CHECK:STDOUT:   %c2.ref.loc19: ref %C = name_ref c2, %c2
// CHECK:STDOUT:   %.loc19_28.1: %C = bind_value %c1.ref.loc19
// CHECK:STDOUT:   %.loc19_33.1: %C = bind_value %c2.ref.loc19
// CHECK:STDOUT:   %.loc19_28.2: ref %C = value_as_ref %.loc19_28.1
// CHECK:STDOUT:   %addr.loc19_31.1: %ptr.d9e = addr_of %.loc19_28.2
// CHECK:STDOUT:   %.loc19_33.2: ref %C = value_as_ref %.loc19_33.1
// CHECK:STDOUT:   %addr.loc19_31.2: %ptr.d9e = addr_of %.loc19_33.2
// CHECK:STDOUT:   %.loc19_31.1: ref bool = temporary_storage
// CHECK:STDOUT:   %addr.loc19_31.3: %ptr.bb2 = addr_of %.loc19_31.1
// CHECK:STDOUT:   %operator>__carbon_thunk.call: init %empty_tuple.type = call imports.%operator>__carbon_thunk.decl(%addr.loc19_31.1, %addr.loc19_31.2, %addr.loc19_31.3)
// CHECK:STDOUT:   %.loc19_31.2: init bool = in_place_init %operator>__carbon_thunk.call, %.loc19_31.1
// CHECK:STDOUT:   %.loc19_21.1: type = splice_block %.loc19_21.3 [concrete = bool] {
// CHECK:STDOUT:     %Bool.call.loc19: init type = call constants.%Bool() [concrete = bool]
// CHECK:STDOUT:     %.loc19_21.2: type = value_of_initializer %Bool.call.loc19 [concrete = bool]
// CHECK:STDOUT:     %.loc19_21.3: type = converted %Bool.call.loc19, %.loc19_21.2 [concrete = bool]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc19_31.3: bool = value_of_initializer %.loc19_31.2
// CHECK:STDOUT:   %.loc19_31.4: bool = converted %.loc19_31.2, %.loc19_31.3
// CHECK:STDOUT:   %greater_than: bool = bind_name greater_than, %.loc19_31.4
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %less_than.patt: %pattern_type.831 = binding_pattern less_than [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c1.ref.loc26: ref %C = name_ref c1, %c1
// CHECK:STDOUT:   %c2.ref.loc26: ref %C = name_ref c2, %c2
// CHECK:STDOUT:   %.loc26_18.1: type = splice_block %.loc26_18.3 [concrete = bool] {
// CHECK:STDOUT:     %Bool.call.loc26: init type = call constants.%Bool() [concrete = bool]
// CHECK:STDOUT:     %.loc26_18.2: type = value_of_initializer %Bool.call.loc26 [concrete = bool]
// CHECK:STDOUT:     %.loc26_18.3: type = converted %Bool.call.loc26, %.loc26_18.2 [concrete = bool]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %less_than: bool = bind_name less_than, <error> [concrete = <error>]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %greater_than_or_equal.patt: %pattern_type.831 = binding_pattern greater_than_or_equal [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c1.ref.loc33: ref %C = name_ref c1, %c1
// CHECK:STDOUT:   %c2.ref.loc33: ref %C = name_ref c2, %c2
// CHECK:STDOUT:   %.loc33_30.1: type = splice_block %.loc33_30.3 [concrete = bool] {
// CHECK:STDOUT:     %Bool.call.loc33: init type = call constants.%Bool() [concrete = bool]
// CHECK:STDOUT:     %.loc33_30.2: type = value_of_initializer %Bool.call.loc33 [concrete = bool]
// CHECK:STDOUT:     %.loc33_30.3: type = converted %Bool.call.loc33, %.loc33_30.2 [concrete = bool]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %greater_than_or_equal: bool = bind_name greater_than_or_equal, <error> [concrete = <error>]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %less_than_or_equal.patt: %pattern_type.831 = binding_pattern less_than_or_equal [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c1.ref.loc36: ref %C = name_ref c1, %c1
// CHECK:STDOUT:   %c2.ref.loc36: ref %C = name_ref c2, %c2
// CHECK:STDOUT:   %.loc36_34.1: %C = bind_value %c1.ref.loc36
// CHECK:STDOUT:   %.loc36_40.1: %C = bind_value %c2.ref.loc36
// CHECK:STDOUT:   %.loc36_34.2: ref %C = value_as_ref %.loc36_34.1
// CHECK:STDOUT:   %addr.loc36_37.1: %ptr.d9e = addr_of %.loc36_34.2
// CHECK:STDOUT:   %.loc36_40.2: ref %C = value_as_ref %.loc36_40.1
// CHECK:STDOUT:   %addr.loc36_37.2: %ptr.d9e = addr_of %.loc36_40.2
// CHECK:STDOUT:   %.loc36_37.1: ref bool = temporary_storage
// CHECK:STDOUT:   %addr.loc36_37.3: %ptr.bb2 = addr_of %.loc36_37.1
// CHECK:STDOUT:   %operator<=__carbon_thunk.call: init %empty_tuple.type = call imports.%operator<=__carbon_thunk.decl(%addr.loc36_37.1, %addr.loc36_37.2, %addr.loc36_37.3)
// CHECK:STDOUT:   %.loc36_37.2: init bool = in_place_init %operator<=__carbon_thunk.call, %.loc36_37.1
// CHECK:STDOUT:   %.loc36_27.1: type = splice_block %.loc36_27.3 [concrete = bool] {
// CHECK:STDOUT:     %Bool.call.loc36: init type = call constants.%Bool() [concrete = bool]
// CHECK:STDOUT:     %.loc36_27.2: type = value_of_initializer %Bool.call.loc36 [concrete = bool]
// CHECK:STDOUT:     %.loc36_27.3: type = converted %Bool.call.loc36, %.loc36_27.2 [concrete = bool]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc36_37.3: bool = value_of_initializer %.loc36_37.2
// CHECK:STDOUT:   %.loc36_37.4: bool = converted %.loc36_37.2, %.loc36_37.3
// CHECK:STDOUT:   %less_than_or_equal: bool = bind_name less_than_or_equal, %.loc36_37.4
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.bound.loc16: <bound method> = bound_method %c2.var, constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.841
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method.loc16: <bound method> = bound_method %c2.var, %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.1
// CHECK:STDOUT:   %addr.loc16_3: %ptr.d9e = addr_of %c2.var
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.call.loc16: init %empty_tuple.type = call %bound_method.loc16(%addr.loc16_3)
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.bound.loc15: <bound method> = bound_method %c1.var, constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.841
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method.loc15: <bound method> = bound_method %c1.var, %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.2
// CHECK:STDOUT:   %addr.loc15_3: %ptr.d9e = addr_of %c1.var
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.call.loc15: init %empty_tuple.type = call %bound_method.loc15(%addr.loc15_3)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_todo_rewrite_equal.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %pattern_type.217: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %C.C.cpp_overload_set.type: type = cpp_overload_set_type @C.C.cpp_overload_set [concrete]
// CHECK:STDOUT:   %C.C.cpp_overload_set.value: %C.C.cpp_overload_set.type = cpp_overload_set_value @C.C.cpp_overload_set [concrete]
// CHECK:STDOUT:   %ptr.d9e: type = ptr_type %C [concrete]
// CHECK:STDOUT:   %C__carbon_thunk.type: type = fn_type @C__carbon_thunk [concrete]
// CHECK:STDOUT:   %C__carbon_thunk: %C__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %Bool.type: type = fn_type @Bool [concrete]
// CHECK:STDOUT:   %Bool: %Bool.type = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.831: type = pattern_type bool [concrete]
// CHECK:STDOUT:   %ptr.bb2: type = ptr_type bool [concrete]
// CHECK:STDOUT:   %operator==__carbon_thunk.type: type = fn_type @operator==__carbon_thunk [concrete]
// CHECK:STDOUT:   %operator==__carbon_thunk: %operator==__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %type_where: type = facet_type <type where .Self impls <CanDestroy>> [concrete]
// CHECK:STDOUT:   %facet_value: %type_where = facet_value %C, () [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.type.b92: type = fn_type @DestroyT.binding.as_type.as.Destroy.impl.Op, @DestroyT.binding.as_type.as.Destroy.impl(%facet_value) [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.841: %DestroyT.binding.as_type.as.Destroy.impl.Op.type.b92 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   %C.C.cpp_overload_set.value: %C.C.cpp_overload_set.type = cpp_overload_set_value @C.C.cpp_overload_set [concrete = constants.%C.C.cpp_overload_set.value]
// CHECK:STDOUT:   %C__carbon_thunk.decl: %C__carbon_thunk.type = fn_decl @C__carbon_thunk [concrete = constants.%C__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %operator==__carbon_thunk.decl: %operator==__carbon_thunk.type = fn_decl @operator==__carbon_thunk [concrete = constants.%operator==__carbon_thunk] {
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
// CHECK:STDOUT:     %c1.var_patt: %pattern_type.217 = var_pattern %c1.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c1.var: ref %C = var %c1.var_patt
// CHECK:STDOUT:   %Cpp.ref.loc12_19: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %C.ref.loc12_22: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %C.ref.loc12_24: %C.C.cpp_overload_set.type = name_ref C, imports.%C.C.cpp_overload_set.value [concrete = constants.%C.C.cpp_overload_set.value]
// CHECK:STDOUT:   %.loc12_3: ref %C = splice_block %c1.var {}
// CHECK:STDOUT:   %addr.loc12_27: %ptr.d9e = addr_of %.loc12_3
// CHECK:STDOUT:   %C__carbon_thunk.call.loc12: init %empty_tuple.type = call imports.%C__carbon_thunk.decl(%addr.loc12_27)
// CHECK:STDOUT:   %.loc12_27: init %C = in_place_init %C__carbon_thunk.call.loc12, %.loc12_3
// CHECK:STDOUT:   assign %c1.var, %.loc12_27
// CHECK:STDOUT:   %.loc12_14: type = splice_block %C.ref.loc12_14 [concrete = constants.%C] {
// CHECK:STDOUT:     %Cpp.ref.loc12_11: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %C.ref.loc12_14: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c1: ref %C = bind_name c1, %c1.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c2.patt: %pattern_type.217 = binding_pattern c2 [concrete]
// CHECK:STDOUT:     %c2.var_patt: %pattern_type.217 = var_pattern %c2.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c2.var: ref %C = var %c2.var_patt
// CHECK:STDOUT:   %Cpp.ref.loc13_19: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %C.ref.loc13_22: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %C.ref.loc13_24: %C.C.cpp_overload_set.type = name_ref C, imports.%C.C.cpp_overload_set.value [concrete = constants.%C.C.cpp_overload_set.value]
// CHECK:STDOUT:   %.loc13_3: ref %C = splice_block %c2.var {}
// CHECK:STDOUT:   %addr.loc13_27: %ptr.d9e = addr_of %.loc13_3
// CHECK:STDOUT:   %C__carbon_thunk.call.loc13: init %empty_tuple.type = call imports.%C__carbon_thunk.decl(%addr.loc13_27)
// CHECK:STDOUT:   %.loc13_27: init %C = in_place_init %C__carbon_thunk.call.loc13, %.loc13_3
// CHECK:STDOUT:   assign %c2.var, %.loc13_27
// CHECK:STDOUT:   %.loc13_14: type = splice_block %C.ref.loc13_14 [concrete = constants.%C] {
// CHECK:STDOUT:     %Cpp.ref.loc13_11: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %C.ref.loc13_14: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c2: ref %C = bind_name c2, %c2.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %equal.patt: %pattern_type.831 = binding_pattern equal [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c1.ref.loc16: ref %C = name_ref c1, %c1
// CHECK:STDOUT:   %c2.ref.loc16: ref %C = name_ref c2, %c2
// CHECK:STDOUT:   %.loc16_21.1: %C = bind_value %c1.ref.loc16
// CHECK:STDOUT:   %.loc16_27.1: %C = bind_value %c2.ref.loc16
// CHECK:STDOUT:   %.loc16_21.2: ref %C = value_as_ref %.loc16_21.1
// CHECK:STDOUT:   %addr.loc16_24.1: %ptr.d9e = addr_of %.loc16_21.2
// CHECK:STDOUT:   %.loc16_27.2: ref %C = value_as_ref %.loc16_27.1
// CHECK:STDOUT:   %addr.loc16_24.2: %ptr.d9e = addr_of %.loc16_27.2
// CHECK:STDOUT:   %.loc16_24.1: ref bool = temporary_storage
// CHECK:STDOUT:   %addr.loc16_24.3: %ptr.bb2 = addr_of %.loc16_24.1
// CHECK:STDOUT:   %operator==__carbon_thunk.call: init %empty_tuple.type = call imports.%operator==__carbon_thunk.decl(%addr.loc16_24.1, %addr.loc16_24.2, %addr.loc16_24.3)
// CHECK:STDOUT:   %.loc16_24.2: init bool = in_place_init %operator==__carbon_thunk.call, %.loc16_24.1
// CHECK:STDOUT:   %.loc16_14.1: type = splice_block %.loc16_14.3 [concrete = bool] {
// CHECK:STDOUT:     %Bool.call.loc16: init type = call constants.%Bool() [concrete = bool]
// CHECK:STDOUT:     %.loc16_14.2: type = value_of_initializer %Bool.call.loc16 [concrete = bool]
// CHECK:STDOUT:     %.loc16_14.3: type = converted %Bool.call.loc16, %.loc16_14.2 [concrete = bool]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc16_24.3: bool = value_of_initializer %.loc16_24.2
// CHECK:STDOUT:   %.loc16_24.4: bool = converted %.loc16_24.2, %.loc16_24.3
// CHECK:STDOUT:   %equal: bool = bind_name equal, %.loc16_24.4
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %not_equal.patt: %pattern_type.831 = binding_pattern not_equal [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c1.ref.loc23: ref %C = name_ref c1, %c1
// CHECK:STDOUT:   %c2.ref.loc23: ref %C = name_ref c2, %c2
// CHECK:STDOUT:   %.loc23_18.1: type = splice_block %.loc23_18.3 [concrete = bool] {
// CHECK:STDOUT:     %Bool.call.loc23: init type = call constants.%Bool() [concrete = bool]
// CHECK:STDOUT:     %.loc23_18.2: type = value_of_initializer %Bool.call.loc23 [concrete = bool]
// CHECK:STDOUT:     %.loc23_18.3: type = converted %Bool.call.loc23, %.loc23_18.2 [concrete = bool]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %not_equal: bool = bind_name not_equal, <error> [concrete = <error>]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.bound.loc13: <bound method> = bound_method %c2.var, constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.841
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method.loc13: <bound method> = bound_method %c2.var, %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.1
// CHECK:STDOUT:   %addr.loc13_3: %ptr.d9e = addr_of %c2.var
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.call.loc13: init %empty_tuple.type = call %bound_method.loc13(%addr.loc13_3)
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.bound.loc12: <bound method> = bound_method %c1.var, constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.841
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method.loc12: <bound method> = bound_method %c1.var, %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.2
// CHECK:STDOUT:   %addr.loc12_3: %ptr.d9e = addr_of %c1.var
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.call.loc12: init %empty_tuple.type = call %bound_method.loc12(%addr.loc12_3)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_single_namespace.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %pattern_type.69f: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %C.C.cpp_overload_set.type: type = cpp_overload_set_type @C.C.cpp_overload_set [concrete]
// CHECK:STDOUT:   %C.C.cpp_overload_set.value: %C.C.cpp_overload_set.type = cpp_overload_set_value @C.C.cpp_overload_set [concrete]
// CHECK:STDOUT:   %ptr.838: type = ptr_type %C [concrete]
// CHECK:STDOUT:   %C__carbon_thunk.type: type = fn_type @C__carbon_thunk [concrete]
// CHECK:STDOUT:   %C__carbon_thunk: %C__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %operator+__carbon_thunk.type: type = fn_type @operator+__carbon_thunk [concrete]
// CHECK:STDOUT:   %operator+__carbon_thunk: %operator+__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %type_where: type = facet_type <type where .Self impls <CanDestroy>> [concrete]
// CHECK:STDOUT:   %facet_value: %type_where = facet_value %C, () [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.type.368: type = fn_type @DestroyT.binding.as_type.as.Destroy.impl.Op, @DestroyT.binding.as_type.as.Destroy.impl(%facet_value) [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.b69: %DestroyT.binding.as_type.as.Destroy.impl.Op.type.368 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .N = %N
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %N: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   %C.C.cpp_overload_set.value: %C.C.cpp_overload_set.type = cpp_overload_set_value @C.C.cpp_overload_set [concrete = constants.%C.C.cpp_overload_set.value]
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
// CHECK:STDOUT:     %c1.patt: %pattern_type.69f = binding_pattern c1 [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc8_21: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %N.ref.loc8_24: <namespace> = name_ref N, imports.%N [concrete = imports.%N]
// CHECK:STDOUT:   %C.ref.loc8_26: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %C.ref.loc8_28: %C.C.cpp_overload_set.type = name_ref C, imports.%C.C.cpp_overload_set.value [concrete = constants.%C.C.cpp_overload_set.value]
// CHECK:STDOUT:   %.loc8_31.1: ref %C = temporary_storage
// CHECK:STDOUT:   %addr.loc8_31.1: %ptr.838 = addr_of %.loc8_31.1
// CHECK:STDOUT:   %C__carbon_thunk.call.loc8: init %empty_tuple.type = call imports.%C__carbon_thunk.decl(%addr.loc8_31.1)
// CHECK:STDOUT:   %.loc8_31.2: init %C = in_place_init %C__carbon_thunk.call.loc8, %.loc8_31.1
// CHECK:STDOUT:   %.loc8_16: type = splice_block %C.ref.loc8_16 [concrete = constants.%C] {
// CHECK:STDOUT:     %Cpp.ref.loc8_11: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %N.ref.loc8_14: <namespace> = name_ref N, imports.%N [concrete = imports.%N]
// CHECK:STDOUT:     %C.ref.loc8_16: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc8_31.3: ref %C = temporary %.loc8_31.1, %.loc8_31.2
// CHECK:STDOUT:   %.loc8_31.4: %C = bind_value %.loc8_31.3
// CHECK:STDOUT:   %c1: %C = bind_name c1, %.loc8_31.4
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c2.patt: %pattern_type.69f = binding_pattern c2 [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc9_21: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %N.ref.loc9_24: <namespace> = name_ref N, imports.%N [concrete = imports.%N]
// CHECK:STDOUT:   %C.ref.loc9_26: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %C.ref.loc9_28: %C.C.cpp_overload_set.type = name_ref C, imports.%C.C.cpp_overload_set.value [concrete = constants.%C.C.cpp_overload_set.value]
// CHECK:STDOUT:   %.loc9_31.1: ref %C = temporary_storage
// CHECK:STDOUT:   %addr.loc9_31.1: %ptr.838 = addr_of %.loc9_31.1
// CHECK:STDOUT:   %C__carbon_thunk.call.loc9: init %empty_tuple.type = call imports.%C__carbon_thunk.decl(%addr.loc9_31.1)
// CHECK:STDOUT:   %.loc9_31.2: init %C = in_place_init %C__carbon_thunk.call.loc9, %.loc9_31.1
// CHECK:STDOUT:   %.loc9_16: type = splice_block %C.ref.loc9_16 [concrete = constants.%C] {
// CHECK:STDOUT:     %Cpp.ref.loc9_11: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %N.ref.loc9_14: <namespace> = name_ref N, imports.%N [concrete = imports.%N]
// CHECK:STDOUT:     %C.ref.loc9_16: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc9_31.3: ref %C = temporary %.loc9_31.1, %.loc9_31.2
// CHECK:STDOUT:   %.loc9_31.4: %C = bind_value %.loc9_31.3
// CHECK:STDOUT:   %c2: %C = bind_name c2, %.loc9_31.4
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c3.patt: %pattern_type.69f = binding_pattern c3 [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c1.ref: %C = name_ref c1, %c1
// CHECK:STDOUT:   %c2.ref: %C = name_ref c2, %c2
// CHECK:STDOUT:   %.loc10_24.1: ref %C = temporary_storage
// CHECK:STDOUT:   %.loc10_21: ref %C = value_as_ref %c1.ref
// CHECK:STDOUT:   %addr.loc10_24.1: %ptr.838 = addr_of %.loc10_21
// CHECK:STDOUT:   %.loc10_26: ref %C = value_as_ref %c2.ref
// CHECK:STDOUT:   %addr.loc10_24.2: %ptr.838 = addr_of %.loc10_26
// CHECK:STDOUT:   %addr.loc10_24.3: %ptr.838 = addr_of %.loc10_24.1
// CHECK:STDOUT:   %operator+__carbon_thunk.call: init %empty_tuple.type = call imports.%operator+__carbon_thunk.decl(%addr.loc10_24.1, %addr.loc10_24.2, %addr.loc10_24.3)
// CHECK:STDOUT:   %.loc10_24.2: init %C = in_place_init %operator+__carbon_thunk.call, %.loc10_24.1
// CHECK:STDOUT:   %.loc10_16: type = splice_block %C.ref.loc10 [concrete = constants.%C] {
// CHECK:STDOUT:     %Cpp.ref.loc10: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %N.ref.loc10: <namespace> = name_ref N, imports.%N [concrete = imports.%N]
// CHECK:STDOUT:     %C.ref.loc10: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc10_24.3: ref %C = temporary %.loc10_24.1, %.loc10_24.2
// CHECK:STDOUT:   %.loc10_24.4: %C = bind_value %.loc10_24.3
// CHECK:STDOUT:   %c3: %C = bind_name c3, %.loc10_24.4
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.bound.loc10: <bound method> = bound_method %.loc10_24.3, constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.b69
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method.loc10: <bound method> = bound_method %.loc10_24.3, %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.1
// CHECK:STDOUT:   %addr.loc10_24.4: %ptr.838 = addr_of %.loc10_24.3
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.call.loc10: init %empty_tuple.type = call %bound_method.loc10(%addr.loc10_24.4)
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.bound.loc9: <bound method> = bound_method %.loc9_31.3, constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.b69
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method.loc9: <bound method> = bound_method %.loc9_31.3, %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.2
// CHECK:STDOUT:   %addr.loc9_31.2: %ptr.838 = addr_of %.loc9_31.3
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.call.loc9: init %empty_tuple.type = call %bound_method.loc9(%addr.loc9_31.2)
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.bound.loc8: <bound method> = bound_method %.loc8_31.3, constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.b69
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method.loc8: <bound method> = bound_method %.loc8_31.3, %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.3
// CHECK:STDOUT:   %addr.loc8_31.2: %ptr.838 = addr_of %.loc8_31.3
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.call.loc8: init %empty_tuple.type = call %bound_method.loc8(%addr.loc8_31.2)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_multiple_namespaces.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %C1: type = class_type @C1 [concrete]
// CHECK:STDOUT:   %pattern_type.20f: type = pattern_type %C1 [concrete]
// CHECK:STDOUT:   %C1.C1.cpp_overload_set.type: type = cpp_overload_set_type @C1.C1.cpp_overload_set [concrete]
// CHECK:STDOUT:   %C1.C1.cpp_overload_set.value: %C1.C1.cpp_overload_set.type = cpp_overload_set_value @C1.C1.cpp_overload_set [concrete]
// CHECK:STDOUT:   %ptr.087: type = ptr_type %C1 [concrete]
// CHECK:STDOUT:   %C1__carbon_thunk.type: type = fn_type @C1__carbon_thunk [concrete]
// CHECK:STDOUT:   %C1__carbon_thunk: %C1__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %C2: type = class_type @C2 [concrete]
// CHECK:STDOUT:   %pattern_type.846: type = pattern_type %C2 [concrete]
// CHECK:STDOUT:   %C2.C2.cpp_overload_set.type: type = cpp_overload_set_type @C2.C2.cpp_overload_set [concrete]
// CHECK:STDOUT:   %C2.C2.cpp_overload_set.value: %C2.C2.cpp_overload_set.type = cpp_overload_set_value @C2.C2.cpp_overload_set [concrete]
// CHECK:STDOUT:   %ptr.51f: type = ptr_type %C2 [concrete]
// CHECK:STDOUT:   %C2__carbon_thunk.type: type = fn_type @C2__carbon_thunk [concrete]
// CHECK:STDOUT:   %C2__carbon_thunk: %C2__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %operator+__carbon_thunk.type: type = fn_type @operator+__carbon_thunk [concrete]
// CHECK:STDOUT:   %operator+__carbon_thunk: %operator+__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %operator-__carbon_thunk.type: type = fn_type @operator-__carbon_thunk [concrete]
// CHECK:STDOUT:   %operator-__carbon_thunk: %operator-__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %type_where: type = facet_type <type where .Self impls <CanDestroy>> [concrete]
// CHECK:STDOUT:   %facet_value.d7a: %type_where = facet_value %C2, () [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.type.cf4: type = fn_type @DestroyT.binding.as_type.as.Destroy.impl.Op, @DestroyT.binding.as_type.as.Destroy.impl(%facet_value.d7a) [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.553: %DestroyT.binding.as_type.as.Destroy.impl.Op.type.cf4 = struct_value () [concrete]
// CHECK:STDOUT:   %facet_value.e5d: %type_where = facet_value %C1, () [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.type.e1f: type = fn_type @DestroyT.binding.as_type.as.Destroy.impl.Op, @DestroyT.binding.as_type.as.Destroy.impl(%facet_value.e5d) [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.b74: %DestroyT.binding.as_type.as.Destroy.impl.Op.type.e1f = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .N1 = %N1
// CHECK:STDOUT:     .N2 = %N2
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %N1: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .C1 = %C1.decl
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C1.decl: type = class_decl @C1 [concrete = constants.%C1] {} {}
// CHECK:STDOUT:   %C1.C1.cpp_overload_set.value: %C1.C1.cpp_overload_set.type = cpp_overload_set_value @C1.C1.cpp_overload_set [concrete = constants.%C1.C1.cpp_overload_set.value]
// CHECK:STDOUT:   %C1__carbon_thunk.decl: %C1__carbon_thunk.type = fn_decl @C1__carbon_thunk [concrete = constants.%C1__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %N2: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .C2 = %C2.decl
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C2.decl: type = class_decl @C2 [concrete = constants.%C2] {} {}
// CHECK:STDOUT:   %C2.C2.cpp_overload_set.value: %C2.C2.cpp_overload_set.type = cpp_overload_set_value @C2.C2.cpp_overload_set [concrete = constants.%C2.C2.cpp_overload_set.value]
// CHECK:STDOUT:   %C2__carbon_thunk.decl: %C2__carbon_thunk.type = fn_decl @C2__carbon_thunk [concrete = constants.%C2__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %operator+__carbon_thunk.decl: %operator+__carbon_thunk.type = fn_decl @operator+__carbon_thunk [concrete = constants.%operator+__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %operator-__carbon_thunk.decl: %operator-__carbon_thunk.type = fn_decl @operator-__carbon_thunk [concrete = constants.%operator-__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c1.patt: %pattern_type.20f = binding_pattern c1 [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc8_23: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %N1.ref.loc8_26: <namespace> = name_ref N1, imports.%N1 [concrete = imports.%N1]
// CHECK:STDOUT:   %C1.ref.loc8_29: type = name_ref C1, imports.%C1.decl [concrete = constants.%C1]
// CHECK:STDOUT:   %C1.ref.loc8_32: %C1.C1.cpp_overload_set.type = name_ref C1, imports.%C1.C1.cpp_overload_set.value [concrete = constants.%C1.C1.cpp_overload_set.value]
// CHECK:STDOUT:   %.loc8_36.1: ref %C1 = temporary_storage
// CHECK:STDOUT:   %addr.loc8_36.1: %ptr.087 = addr_of %.loc8_36.1
// CHECK:STDOUT:   %C1__carbon_thunk.call: init %empty_tuple.type = call imports.%C1__carbon_thunk.decl(%addr.loc8_36.1)
// CHECK:STDOUT:   %.loc8_36.2: init %C1 = in_place_init %C1__carbon_thunk.call, %.loc8_36.1
// CHECK:STDOUT:   %.loc8_17: type = splice_block %C1.ref.loc8_17 [concrete = constants.%C1] {
// CHECK:STDOUT:     %Cpp.ref.loc8_11: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %N1.ref.loc8_14: <namespace> = name_ref N1, imports.%N1 [concrete = imports.%N1]
// CHECK:STDOUT:     %C1.ref.loc8_17: type = name_ref C1, imports.%C1.decl [concrete = constants.%C1]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc8_36.3: ref %C1 = temporary %.loc8_36.1, %.loc8_36.2
// CHECK:STDOUT:   %.loc8_36.4: %C1 = bind_value %.loc8_36.3
// CHECK:STDOUT:   %c1: %C1 = bind_name c1, %.loc8_36.4
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c2.patt: %pattern_type.846 = binding_pattern c2 [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc9_23: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %N2.ref.loc9_26: <namespace> = name_ref N2, imports.%N2 [concrete = imports.%N2]
// CHECK:STDOUT:   %C2.ref.loc9_29: type = name_ref C2, imports.%C2.decl [concrete = constants.%C2]
// CHECK:STDOUT:   %C2.ref.loc9_32: %C2.C2.cpp_overload_set.type = name_ref C2, imports.%C2.C2.cpp_overload_set.value [concrete = constants.%C2.C2.cpp_overload_set.value]
// CHECK:STDOUT:   %.loc9_36.1: ref %C2 = temporary_storage
// CHECK:STDOUT:   %addr.loc9_36.1: %ptr.51f = addr_of %.loc9_36.1
// CHECK:STDOUT:   %C2__carbon_thunk.call: init %empty_tuple.type = call imports.%C2__carbon_thunk.decl(%addr.loc9_36.1)
// CHECK:STDOUT:   %.loc9_36.2: init %C2 = in_place_init %C2__carbon_thunk.call, %.loc9_36.1
// CHECK:STDOUT:   %.loc9_17: type = splice_block %C2.ref.loc9_17 [concrete = constants.%C2] {
// CHECK:STDOUT:     %Cpp.ref.loc9_11: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %N2.ref.loc9_14: <namespace> = name_ref N2, imports.%N2 [concrete = imports.%N2]
// CHECK:STDOUT:     %C2.ref.loc9_17: type = name_ref C2, imports.%C2.decl [concrete = constants.%C2]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc9_36.3: ref %C2 = temporary %.loc9_36.1, %.loc9_36.2
// CHECK:STDOUT:   %.loc9_36.4: %C2 = bind_value %.loc9_36.3
// CHECK:STDOUT:   %c2: %C2 = bind_name c2, %.loc9_36.4
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c3.patt: %pattern_type.846 = binding_pattern c3 [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c1.ref.loc10: %C1 = name_ref c1, %c1
// CHECK:STDOUT:   %c2.ref.loc10: %C2 = name_ref c2, %c2
// CHECK:STDOUT:   %.loc10_26.1: ref %C2 = temporary_storage
// CHECK:STDOUT:   %.loc10_23: ref %C1 = value_as_ref %c1.ref.loc10
// CHECK:STDOUT:   %addr.loc10_26.1: %ptr.087 = addr_of %.loc10_23
// CHECK:STDOUT:   %.loc10_28: ref %C2 = value_as_ref %c2.ref.loc10
// CHECK:STDOUT:   %addr.loc10_26.2: %ptr.51f = addr_of %.loc10_28
// CHECK:STDOUT:   %addr.loc10_26.3: %ptr.51f = addr_of %.loc10_26.1
// CHECK:STDOUT:   %operator+__carbon_thunk.call: init %empty_tuple.type = call imports.%operator+__carbon_thunk.decl(%addr.loc10_26.1, %addr.loc10_26.2, %addr.loc10_26.3)
// CHECK:STDOUT:   %.loc10_26.2: init %C2 = in_place_init %operator+__carbon_thunk.call, %.loc10_26.1
// CHECK:STDOUT:   %.loc10_17: type = splice_block %C2.ref.loc10 [concrete = constants.%C2] {
// CHECK:STDOUT:     %Cpp.ref.loc10: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %N2.ref.loc10: <namespace> = name_ref N2, imports.%N2 [concrete = imports.%N2]
// CHECK:STDOUT:     %C2.ref.loc10: type = name_ref C2, imports.%C2.decl [concrete = constants.%C2]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc10_26.3: ref %C2 = temporary %.loc10_26.1, %.loc10_26.2
// CHECK:STDOUT:   %.loc10_26.4: %C2 = bind_value %.loc10_26.3
// CHECK:STDOUT:   %c3: %C2 = bind_name c3, %.loc10_26.4
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c4.patt: %pattern_type.846 = binding_pattern c4 [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c2.ref.loc11: %C2 = name_ref c2, %c2
// CHECK:STDOUT:   %c1.ref.loc11: %C1 = name_ref c1, %c1
// CHECK:STDOUT:   %.loc11_26.1: ref %C2 = temporary_storage
// CHECK:STDOUT:   %.loc11_23: ref %C2 = value_as_ref %c2.ref.loc11
// CHECK:STDOUT:   %addr.loc11_26.1: %ptr.51f = addr_of %.loc11_23
// CHECK:STDOUT:   %.loc11_28: ref %C1 = value_as_ref %c1.ref.loc11
// CHECK:STDOUT:   %addr.loc11_26.2: %ptr.087 = addr_of %.loc11_28
// CHECK:STDOUT:   %addr.loc11_26.3: %ptr.51f = addr_of %.loc11_26.1
// CHECK:STDOUT:   %operator-__carbon_thunk.call: init %empty_tuple.type = call imports.%operator-__carbon_thunk.decl(%addr.loc11_26.1, %addr.loc11_26.2, %addr.loc11_26.3)
// CHECK:STDOUT:   %.loc11_26.2: init %C2 = in_place_init %operator-__carbon_thunk.call, %.loc11_26.1
// CHECK:STDOUT:   %.loc11_17: type = splice_block %C2.ref.loc11 [concrete = constants.%C2] {
// CHECK:STDOUT:     %Cpp.ref.loc11: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %N2.ref.loc11: <namespace> = name_ref N2, imports.%N2 [concrete = imports.%N2]
// CHECK:STDOUT:     %C2.ref.loc11: type = name_ref C2, imports.%C2.decl [concrete = constants.%C2]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc11_26.3: ref %C2 = temporary %.loc11_26.1, %.loc11_26.2
// CHECK:STDOUT:   %.loc11_26.4: %C2 = bind_value %.loc11_26.3
// CHECK:STDOUT:   %c4: %C2 = bind_name c4, %.loc11_26.4
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.bound.loc11: <bound method> = bound_method %.loc11_26.3, constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.553
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method.loc11: <bound method> = bound_method %.loc11_26.3, %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.1
// CHECK:STDOUT:   %addr.loc11_26.4: %ptr.51f = addr_of %.loc11_26.3
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.call.loc11: init %empty_tuple.type = call %bound_method.loc11(%addr.loc11_26.4)
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.bound.loc10: <bound method> = bound_method %.loc10_26.3, constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.553
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method.loc10: <bound method> = bound_method %.loc10_26.3, %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.2
// CHECK:STDOUT:   %addr.loc10_26.4: %ptr.51f = addr_of %.loc10_26.3
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.call.loc10: init %empty_tuple.type = call %bound_method.loc10(%addr.loc10_26.4)
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.bound.loc9: <bound method> = bound_method %.loc9_36.3, constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.553
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method.loc9: <bound method> = bound_method %.loc9_36.3, %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.3
// CHECK:STDOUT:   %addr.loc9_36.2: %ptr.51f = addr_of %.loc9_36.3
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.call.loc9: init %empty_tuple.type = call %bound_method.loc9(%addr.loc9_36.2)
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.bound.loc8: <bound method> = bound_method %.loc8_36.3, constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.b74
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method.loc8: <bound method> = bound_method %.loc8_36.3, %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.4
// CHECK:STDOUT:   %addr.loc8_36.2: %ptr.087 = addr_of %.loc8_36.3
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.call.loc8: init %empty_tuple.type = call %bound_method.loc8(%addr.loc8_36.2)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_todo_import_operands_in_namespace_operator_in_global.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %pattern_type.69f: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %C.C.cpp_overload_set.type: type = cpp_overload_set_type @C.C.cpp_overload_set [concrete]
// CHECK:STDOUT:   %C.C.cpp_overload_set.value: %C.C.cpp_overload_set.type = cpp_overload_set_value @C.C.cpp_overload_set [concrete]
// CHECK:STDOUT:   %ptr.838: type = ptr_type %C [concrete]
// CHECK:STDOUT:   %C__carbon_thunk.type: type = fn_type @C__carbon_thunk [concrete]
// CHECK:STDOUT:   %C__carbon_thunk: %C__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %type_where: type = facet_type <type where .Self impls <CanDestroy>> [concrete]
// CHECK:STDOUT:   %facet_value: %type_where = facet_value %C, () [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.type.368: type = fn_type @DestroyT.binding.as_type.as.Destroy.impl.Op, @DestroyT.binding.as_type.as.Destroy.impl(%facet_value) [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.b69: %DestroyT.binding.as_type.as.Destroy.impl.Op.type.368 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .N = %N
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %N: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   %C.C.cpp_overload_set.value: %C.C.cpp_overload_set.type = cpp_overload_set_value @C.C.cpp_overload_set [concrete = constants.%C.C.cpp_overload_set.value]
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
// CHECK:STDOUT:     %c1.patt: %pattern_type.69f = binding_pattern c1 [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc8_21: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %N.ref.loc8_24: <namespace> = name_ref N, imports.%N [concrete = imports.%N]
// CHECK:STDOUT:   %C.ref.loc8_26: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %C.ref.loc8_28: %C.C.cpp_overload_set.type = name_ref C, imports.%C.C.cpp_overload_set.value [concrete = constants.%C.C.cpp_overload_set.value]
// CHECK:STDOUT:   %.loc8_31.1: ref %C = temporary_storage
// CHECK:STDOUT:   %addr.loc8_31.1: %ptr.838 = addr_of %.loc8_31.1
// CHECK:STDOUT:   %C__carbon_thunk.call.loc8: init %empty_tuple.type = call imports.%C__carbon_thunk.decl(%addr.loc8_31.1)
// CHECK:STDOUT:   %.loc8_31.2: init %C = in_place_init %C__carbon_thunk.call.loc8, %.loc8_31.1
// CHECK:STDOUT:   %.loc8_16: type = splice_block %C.ref.loc8_16 [concrete = constants.%C] {
// CHECK:STDOUT:     %Cpp.ref.loc8_11: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %N.ref.loc8_14: <namespace> = name_ref N, imports.%N [concrete = imports.%N]
// CHECK:STDOUT:     %C.ref.loc8_16: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc8_31.3: ref %C = temporary %.loc8_31.1, %.loc8_31.2
// CHECK:STDOUT:   %.loc8_31.4: %C = bind_value %.loc8_31.3
// CHECK:STDOUT:   %c1: %C = bind_name c1, %.loc8_31.4
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c2.patt: %pattern_type.69f = binding_pattern c2 [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc9_21: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %N.ref.loc9_24: <namespace> = name_ref N, imports.%N [concrete = imports.%N]
// CHECK:STDOUT:   %C.ref.loc9_26: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %C.ref.loc9_28: %C.C.cpp_overload_set.type = name_ref C, imports.%C.C.cpp_overload_set.value [concrete = constants.%C.C.cpp_overload_set.value]
// CHECK:STDOUT:   %.loc9_31.1: ref %C = temporary_storage
// CHECK:STDOUT:   %addr.loc9_31.1: %ptr.838 = addr_of %.loc9_31.1
// CHECK:STDOUT:   %C__carbon_thunk.call.loc9: init %empty_tuple.type = call imports.%C__carbon_thunk.decl(%addr.loc9_31.1)
// CHECK:STDOUT:   %.loc9_31.2: init %C = in_place_init %C__carbon_thunk.call.loc9, %.loc9_31.1
// CHECK:STDOUT:   %.loc9_16: type = splice_block %C.ref.loc9_16 [concrete = constants.%C] {
// CHECK:STDOUT:     %Cpp.ref.loc9_11: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %N.ref.loc9_14: <namespace> = name_ref N, imports.%N [concrete = imports.%N]
// CHECK:STDOUT:     %C.ref.loc9_16: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc9_31.3: ref %C = temporary %.loc9_31.1, %.loc9_31.2
// CHECK:STDOUT:   %.loc9_31.4: %C = bind_value %.loc9_31.3
// CHECK:STDOUT:   %c2: %C = bind_name c2, %.loc9_31.4
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c3.patt: %pattern_type.69f = binding_pattern c3 [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c1.ref: %C = name_ref c1, %c1
// CHECK:STDOUT:   %c2.ref: %C = name_ref c2, %c2
// CHECK:STDOUT:   %.loc14: type = splice_block %C.ref.loc14 [concrete = constants.%C] {
// CHECK:STDOUT:     %Cpp.ref.loc14: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %N.ref.loc14: <namespace> = name_ref N, imports.%N [concrete = imports.%N]
// CHECK:STDOUT:     %C.ref.loc14: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c3: %C = bind_name c3, <error> [concrete = <error>]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.bound.loc9: <bound method> = bound_method %.loc9_31.3, constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.b69
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method.loc9: <bound method> = bound_method %.loc9_31.3, %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.1
// CHECK:STDOUT:   %addr.loc9_31.2: %ptr.838 = addr_of %.loc9_31.3
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.call.loc9: init %empty_tuple.type = call %bound_method.loc9(%addr.loc9_31.2)
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.bound.loc8: <bound method> = bound_method %.loc8_31.3, constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.b69
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method.loc8: <bound method> = bound_method %.loc8_31.3, %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.2
// CHECK:STDOUT:   %addr.loc8_31.2: %ptr.838 = addr_of %.loc8_31.3
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.call.loc8: init %empty_tuple.type = call %bound_method.loc8(%addr.loc8_31.2)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_inner_class.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %O: type = class_type @O [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %pattern_type.b28: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %C.C.cpp_overload_set.type: type = cpp_overload_set_type @C.C.cpp_overload_set [concrete]
// CHECK:STDOUT:   %C.C.cpp_overload_set.value: %C.C.cpp_overload_set.type = cpp_overload_set_value @C.C.cpp_overload_set [concrete]
// CHECK:STDOUT:   %ptr.de2: type = ptr_type %C [concrete]
// CHECK:STDOUT:   %C__carbon_thunk.type: type = fn_type @C__carbon_thunk [concrete]
// CHECK:STDOUT:   %C__carbon_thunk: %C__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %operator+__carbon_thunk.type: type = fn_type @operator+__carbon_thunk [concrete]
// CHECK:STDOUT:   %operator+__carbon_thunk: %operator+__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %type_where: type = facet_type <type where .Self impls <CanDestroy>> [concrete]
// CHECK:STDOUT:   %facet_value: %type_where = facet_value %C, () [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.type.075: type = fn_type @DestroyT.binding.as_type.as.Destroy.impl.Op, @DestroyT.binding.as_type.as.Destroy.impl(%facet_value) [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.ae5: %DestroyT.binding.as_type.as.Destroy.impl.Op.type.075 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .O = %O.decl
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %O.decl: type = class_decl @O [concrete = constants.%O] {} {}
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   %C.C.cpp_overload_set.value: %C.C.cpp_overload_set.type = cpp_overload_set_value @C.C.cpp_overload_set [concrete = constants.%C.C.cpp_overload_set.value]
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
// CHECK:STDOUT:     %c1.patt: %pattern_type.b28 = binding_pattern c1 [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc8_21: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %O.ref.loc8_24: type = name_ref O, imports.%O.decl [concrete = constants.%O]
// CHECK:STDOUT:   %C.ref.loc8_26: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %C.ref.loc8_28: %C.C.cpp_overload_set.type = name_ref C, imports.%C.C.cpp_overload_set.value [concrete = constants.%C.C.cpp_overload_set.value]
// CHECK:STDOUT:   %.loc8_31.1: ref %C = temporary_storage
// CHECK:STDOUT:   %addr.loc8_31.1: %ptr.de2 = addr_of %.loc8_31.1
// CHECK:STDOUT:   %C__carbon_thunk.call.loc8: init %empty_tuple.type = call imports.%C__carbon_thunk.decl(%addr.loc8_31.1)
// CHECK:STDOUT:   %.loc8_31.2: init %C = in_place_init %C__carbon_thunk.call.loc8, %.loc8_31.1
// CHECK:STDOUT:   %.loc8_16: type = splice_block %C.ref.loc8_16 [concrete = constants.%C] {
// CHECK:STDOUT:     %Cpp.ref.loc8_11: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %O.ref.loc8_14: type = name_ref O, imports.%O.decl [concrete = constants.%O]
// CHECK:STDOUT:     %C.ref.loc8_16: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc8_31.3: ref %C = temporary %.loc8_31.1, %.loc8_31.2
// CHECK:STDOUT:   %.loc8_31.4: %C = bind_value %.loc8_31.3
// CHECK:STDOUT:   %c1: %C = bind_name c1, %.loc8_31.4
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c2.patt: %pattern_type.b28 = binding_pattern c2 [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc9_21: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %O.ref.loc9_24: type = name_ref O, imports.%O.decl [concrete = constants.%O]
// CHECK:STDOUT:   %C.ref.loc9_26: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %C.ref.loc9_28: %C.C.cpp_overload_set.type = name_ref C, imports.%C.C.cpp_overload_set.value [concrete = constants.%C.C.cpp_overload_set.value]
// CHECK:STDOUT:   %.loc9_31.1: ref %C = temporary_storage
// CHECK:STDOUT:   %addr.loc9_31.1: %ptr.de2 = addr_of %.loc9_31.1
// CHECK:STDOUT:   %C__carbon_thunk.call.loc9: init %empty_tuple.type = call imports.%C__carbon_thunk.decl(%addr.loc9_31.1)
// CHECK:STDOUT:   %.loc9_31.2: init %C = in_place_init %C__carbon_thunk.call.loc9, %.loc9_31.1
// CHECK:STDOUT:   %.loc9_16: type = splice_block %C.ref.loc9_16 [concrete = constants.%C] {
// CHECK:STDOUT:     %Cpp.ref.loc9_11: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %O.ref.loc9_14: type = name_ref O, imports.%O.decl [concrete = constants.%O]
// CHECK:STDOUT:     %C.ref.loc9_16: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc9_31.3: ref %C = temporary %.loc9_31.1, %.loc9_31.2
// CHECK:STDOUT:   %.loc9_31.4: %C = bind_value %.loc9_31.3
// CHECK:STDOUT:   %c2: %C = bind_name c2, %.loc9_31.4
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c3.patt: %pattern_type.b28 = binding_pattern c3 [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c1.ref: %C = name_ref c1, %c1
// CHECK:STDOUT:   %c2.ref: %C = name_ref c2, %c2
// CHECK:STDOUT:   %.loc10_24.1: ref %C = temporary_storage
// CHECK:STDOUT:   %.loc10_21: ref %C = value_as_ref %c1.ref
// CHECK:STDOUT:   %addr.loc10_24.1: %ptr.de2 = addr_of %.loc10_21
// CHECK:STDOUT:   %.loc10_26: ref %C = value_as_ref %c2.ref
// CHECK:STDOUT:   %addr.loc10_24.2: %ptr.de2 = addr_of %.loc10_26
// CHECK:STDOUT:   %addr.loc10_24.3: %ptr.de2 = addr_of %.loc10_24.1
// CHECK:STDOUT:   %operator+__carbon_thunk.call: init %empty_tuple.type = call imports.%operator+__carbon_thunk.decl(%addr.loc10_24.1, %addr.loc10_24.2, %addr.loc10_24.3)
// CHECK:STDOUT:   %.loc10_24.2: init %C = in_place_init %operator+__carbon_thunk.call, %.loc10_24.1
// CHECK:STDOUT:   %.loc10_16: type = splice_block %C.ref.loc10 [concrete = constants.%C] {
// CHECK:STDOUT:     %Cpp.ref.loc10: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %O.ref.loc10: type = name_ref O, imports.%O.decl [concrete = constants.%O]
// CHECK:STDOUT:     %C.ref.loc10: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc10_24.3: ref %C = temporary %.loc10_24.1, %.loc10_24.2
// CHECK:STDOUT:   %.loc10_24.4: %C = bind_value %.loc10_24.3
// CHECK:STDOUT:   %c3: %C = bind_name c3, %.loc10_24.4
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.bound.loc10: <bound method> = bound_method %.loc10_24.3, constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.ae5
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method.loc10: <bound method> = bound_method %.loc10_24.3, %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.1
// CHECK:STDOUT:   %addr.loc10_24.4: %ptr.de2 = addr_of %.loc10_24.3
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.call.loc10: init %empty_tuple.type = call %bound_method.loc10(%addr.loc10_24.4)
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.bound.loc9: <bound method> = bound_method %.loc9_31.3, constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.ae5
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method.loc9: <bound method> = bound_method %.loc9_31.3, %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.2
// CHECK:STDOUT:   %addr.loc9_31.2: %ptr.de2 = addr_of %.loc9_31.3
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.call.loc9: init %empty_tuple.type = call %bound_method.loc9(%addr.loc9_31.2)
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.bound.loc8: <bound method> = bound_method %.loc8_31.3, constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.ae5
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method.loc8: <bound method> = bound_method %.loc8_31.3, %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.3
// CHECK:STDOUT:   %addr.loc8_31.2: %ptr.de2 = addr_of %.loc8_31.3
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.call.loc8: init %empty_tuple.type = call %bound_method.loc8(%addr.loc8_31.2)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_inner_class_in_namespace.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %O: type = class_type @O [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %pattern_type.84b: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %C.C.cpp_overload_set.type: type = cpp_overload_set_type @C.C.cpp_overload_set [concrete]
// CHECK:STDOUT:   %C.C.cpp_overload_set.value: %C.C.cpp_overload_set.type = cpp_overload_set_value @C.C.cpp_overload_set [concrete]
// CHECK:STDOUT:   %ptr.4b2: type = ptr_type %C [concrete]
// CHECK:STDOUT:   %C__carbon_thunk.type: type = fn_type @C__carbon_thunk [concrete]
// CHECK:STDOUT:   %C__carbon_thunk: %C__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %operator+__carbon_thunk.type: type = fn_type @operator+__carbon_thunk [concrete]
// CHECK:STDOUT:   %operator+__carbon_thunk: %operator+__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %type_where: type = facet_type <type where .Self impls <CanDestroy>> [concrete]
// CHECK:STDOUT:   %facet_value: %type_where = facet_value %C, () [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.type.c9b: type = fn_type @DestroyT.binding.as_type.as.Destroy.impl.Op, @DestroyT.binding.as_type.as.Destroy.impl(%facet_value) [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.971: %DestroyT.binding.as_type.as.Destroy.impl.Op.type.c9b = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .N = %N
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %N: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .O = %O.decl
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %O.decl: type = class_decl @O [concrete = constants.%O] {} {}
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   %C.C.cpp_overload_set.value: %C.C.cpp_overload_set.type = cpp_overload_set_value @C.C.cpp_overload_set [concrete = constants.%C.C.cpp_overload_set.value]
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
// CHECK:STDOUT:     %c1.patt: %pattern_type.84b = binding_pattern c1 [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc8_23: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %N.ref.loc8_26: <namespace> = name_ref N, imports.%N [concrete = imports.%N]
// CHECK:STDOUT:   %O.ref.loc8_28: type = name_ref O, imports.%O.decl [concrete = constants.%O]
// CHECK:STDOUT:   %C.ref.loc8_30: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %C.ref.loc8_32: %C.C.cpp_overload_set.type = name_ref C, imports.%C.C.cpp_overload_set.value [concrete = constants.%C.C.cpp_overload_set.value]
// CHECK:STDOUT:   %.loc8_35.1: ref %C = temporary_storage
// CHECK:STDOUT:   %addr.loc8_35.1: %ptr.4b2 = addr_of %.loc8_35.1
// CHECK:STDOUT:   %C__carbon_thunk.call.loc8: init %empty_tuple.type = call imports.%C__carbon_thunk.decl(%addr.loc8_35.1)
// CHECK:STDOUT:   %.loc8_35.2: init %C = in_place_init %C__carbon_thunk.call.loc8, %.loc8_35.1
// CHECK:STDOUT:   %.loc8_18: type = splice_block %C.ref.loc8_18 [concrete = constants.%C] {
// CHECK:STDOUT:     %Cpp.ref.loc8_11: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %N.ref.loc8_14: <namespace> = name_ref N, imports.%N [concrete = imports.%N]
// CHECK:STDOUT:     %O.ref.loc8_16: type = name_ref O, imports.%O.decl [concrete = constants.%O]
// CHECK:STDOUT:     %C.ref.loc8_18: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc8_35.3: ref %C = temporary %.loc8_35.1, %.loc8_35.2
// CHECK:STDOUT:   %.loc8_35.4: %C = bind_value %.loc8_35.3
// CHECK:STDOUT:   %c1: %C = bind_name c1, %.loc8_35.4
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c2.patt: %pattern_type.84b = binding_pattern c2 [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc9_23: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %N.ref.loc9_26: <namespace> = name_ref N, imports.%N [concrete = imports.%N]
// CHECK:STDOUT:   %O.ref.loc9_28: type = name_ref O, imports.%O.decl [concrete = constants.%O]
// CHECK:STDOUT:   %C.ref.loc9_30: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %C.ref.loc9_32: %C.C.cpp_overload_set.type = name_ref C, imports.%C.C.cpp_overload_set.value [concrete = constants.%C.C.cpp_overload_set.value]
// CHECK:STDOUT:   %.loc9_35.1: ref %C = temporary_storage
// CHECK:STDOUT:   %addr.loc9_35.1: %ptr.4b2 = addr_of %.loc9_35.1
// CHECK:STDOUT:   %C__carbon_thunk.call.loc9: init %empty_tuple.type = call imports.%C__carbon_thunk.decl(%addr.loc9_35.1)
// CHECK:STDOUT:   %.loc9_35.2: init %C = in_place_init %C__carbon_thunk.call.loc9, %.loc9_35.1
// CHECK:STDOUT:   %.loc9_18: type = splice_block %C.ref.loc9_18 [concrete = constants.%C] {
// CHECK:STDOUT:     %Cpp.ref.loc9_11: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %N.ref.loc9_14: <namespace> = name_ref N, imports.%N [concrete = imports.%N]
// CHECK:STDOUT:     %O.ref.loc9_16: type = name_ref O, imports.%O.decl [concrete = constants.%O]
// CHECK:STDOUT:     %C.ref.loc9_18: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc9_35.3: ref %C = temporary %.loc9_35.1, %.loc9_35.2
// CHECK:STDOUT:   %.loc9_35.4: %C = bind_value %.loc9_35.3
// CHECK:STDOUT:   %c2: %C = bind_name c2, %.loc9_35.4
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c3.patt: %pattern_type.84b = binding_pattern c3 [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c1.ref: %C = name_ref c1, %c1
// CHECK:STDOUT:   %c2.ref: %C = name_ref c2, %c2
// CHECK:STDOUT:   %.loc10_26.1: ref %C = temporary_storage
// CHECK:STDOUT:   %.loc10_23: ref %C = value_as_ref %c1.ref
// CHECK:STDOUT:   %addr.loc10_26.1: %ptr.4b2 = addr_of %.loc10_23
// CHECK:STDOUT:   %.loc10_28: ref %C = value_as_ref %c2.ref
// CHECK:STDOUT:   %addr.loc10_26.2: %ptr.4b2 = addr_of %.loc10_28
// CHECK:STDOUT:   %addr.loc10_26.3: %ptr.4b2 = addr_of %.loc10_26.1
// CHECK:STDOUT:   %operator+__carbon_thunk.call: init %empty_tuple.type = call imports.%operator+__carbon_thunk.decl(%addr.loc10_26.1, %addr.loc10_26.2, %addr.loc10_26.3)
// CHECK:STDOUT:   %.loc10_26.2: init %C = in_place_init %operator+__carbon_thunk.call, %.loc10_26.1
// CHECK:STDOUT:   %.loc10_18: type = splice_block %C.ref.loc10 [concrete = constants.%C] {
// CHECK:STDOUT:     %Cpp.ref.loc10: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %N.ref.loc10: <namespace> = name_ref N, imports.%N [concrete = imports.%N]
// CHECK:STDOUT:     %O.ref.loc10: type = name_ref O, imports.%O.decl [concrete = constants.%O]
// CHECK:STDOUT:     %C.ref.loc10: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc10_26.3: ref %C = temporary %.loc10_26.1, %.loc10_26.2
// CHECK:STDOUT:   %.loc10_26.4: %C = bind_value %.loc10_26.3
// CHECK:STDOUT:   %c3: %C = bind_name c3, %.loc10_26.4
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.bound.loc10: <bound method> = bound_method %.loc10_26.3, constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.971
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method.loc10: <bound method> = bound_method %.loc10_26.3, %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.1
// CHECK:STDOUT:   %addr.loc10_26.4: %ptr.4b2 = addr_of %.loc10_26.3
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.call.loc10: init %empty_tuple.type = call %bound_method.loc10(%addr.loc10_26.4)
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.bound.loc9: <bound method> = bound_method %.loc9_35.3, constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.971
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method.loc9: <bound method> = bound_method %.loc9_35.3, %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.2
// CHECK:STDOUT:   %addr.loc9_35.2: %ptr.4b2 = addr_of %.loc9_35.3
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.call.loc9: init %empty_tuple.type = call %bound_method.loc9(%addr.loc9_35.2)
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.bound.loc8: <bound method> = bound_method %.loc8_35.3, constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.971
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method.loc8: <bound method> = bound_method %.loc8_35.3, %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.3
// CHECK:STDOUT:   %addr.loc8_35.2: %ptr.4b2 = addr_of %.loc8_35.3
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.call.loc8: init %empty_tuple.type = call %bound_method.loc8(%addr.loc8_35.2)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_member_add_with.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %pattern_type.217: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %C.C.cpp_overload_set.type: type = cpp_overload_set_type @C.C.cpp_overload_set [concrete]
// CHECK:STDOUT:   %C.C.cpp_overload_set.value: %C.C.cpp_overload_set.type = cpp_overload_set_value @C.C.cpp_overload_set [concrete]
// CHECK:STDOUT:   %ptr.d9e: type = ptr_type %C [concrete]
// CHECK:STDOUT:   %C__carbon_thunk.type: type = fn_type @C__carbon_thunk [concrete]
// CHECK:STDOUT:   %C__carbon_thunk: %C__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %operator-__carbon_thunk.type: type = fn_type @operator-__carbon_thunk [concrete]
// CHECK:STDOUT:   %operator-__carbon_thunk: %operator-__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %operator+__carbon_thunk.type: type = fn_type @operator+__carbon_thunk [concrete]
// CHECK:STDOUT:   %operator+__carbon_thunk: %operator+__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %type_where: type = facet_type <type where .Self impls <CanDestroy>> [concrete]
// CHECK:STDOUT:   %facet_value: %type_where = facet_value %C, () [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.type.b92: type = fn_type @DestroyT.binding.as_type.as.Destroy.impl.Op, @DestroyT.binding.as_type.as.Destroy.impl(%facet_value) [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.841: %DestroyT.binding.as_type.as.Destroy.impl.Op.type.b92 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   %C.C.cpp_overload_set.value: %C.C.cpp_overload_set.type = cpp_overload_set_value @C.C.cpp_overload_set [concrete = constants.%C.C.cpp_overload_set.value]
// CHECK:STDOUT:   %C__carbon_thunk.decl: %C__carbon_thunk.type = fn_decl @C__carbon_thunk [concrete = constants.%C__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %operator-__carbon_thunk.decl: %operator-__carbon_thunk.type = fn_decl @operator-__carbon_thunk [concrete = constants.%operator-__carbon_thunk] {
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
// CHECK:STDOUT:     %c1.var_patt: %pattern_type.217 = var_pattern %c1.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c1.var: ref %C = var %c1.var_patt
// CHECK:STDOUT:   %Cpp.ref.loc8_19: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %C.ref.loc8_22: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %C.ref.loc8_24: %C.C.cpp_overload_set.type = name_ref C, imports.%C.C.cpp_overload_set.value [concrete = constants.%C.C.cpp_overload_set.value]
// CHECK:STDOUT:   %.loc8_3: ref %C = splice_block %c1.var {}
// CHECK:STDOUT:   %addr.loc8_27: %ptr.d9e = addr_of %.loc8_3
// CHECK:STDOUT:   %C__carbon_thunk.call: init %empty_tuple.type = call imports.%C__carbon_thunk.decl(%addr.loc8_27)
// CHECK:STDOUT:   %.loc8_27: init %C = in_place_init %C__carbon_thunk.call, %.loc8_3
// CHECK:STDOUT:   assign %c1.var, %.loc8_27
// CHECK:STDOUT:   %.loc8_14: type = splice_block %C.ref.loc8_14 [concrete = constants.%C] {
// CHECK:STDOUT:     %Cpp.ref.loc8_11: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %C.ref.loc8_14: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c1: ref %C = bind_name c1, %c1.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c2.patt: %pattern_type.217 = binding_pattern c2 [concrete]
// CHECK:STDOUT:     %c2.var_patt: %pattern_type.217 = var_pattern %c2.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c2.var: ref %C = var %c2.var_patt
// CHECK:STDOUT:   %c1.ref.loc9: ref %C = name_ref c1, %c1
// CHECK:STDOUT:   %.loc9_3: ref %C = splice_block %c2.var {}
// CHECK:STDOUT:   %addr.loc9_20: %ptr.d9e = addr_of %c1.ref.loc9
// CHECK:STDOUT:   %addr.loc9_19: %ptr.d9e = addr_of %.loc9_3
// CHECK:STDOUT:   %operator-__carbon_thunk.call: init %empty_tuple.type = call imports.%operator-__carbon_thunk.decl(%addr.loc9_20, %addr.loc9_19)
// CHECK:STDOUT:   %.loc9_19: init %C = in_place_init %operator-__carbon_thunk.call, %.loc9_3
// CHECK:STDOUT:   assign %c2.var, %.loc9_19
// CHECK:STDOUT:   %.loc9_14: type = splice_block %C.ref.loc9 [concrete = constants.%C] {
// CHECK:STDOUT:     %Cpp.ref.loc9: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %C.ref.loc9: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c2: ref %C = bind_name c2, %c2.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c3.patt: %pattern_type.217 = binding_pattern c3 [concrete]
// CHECK:STDOUT:     %c3.var_patt: %pattern_type.217 = var_pattern %c3.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c3.var: ref %C = var %c3.var_patt
// CHECK:STDOUT:   %c1.ref.loc10: ref %C = name_ref c1, %c1
// CHECK:STDOUT:   %c2.ref: ref %C = name_ref c2, %c2
// CHECK:STDOUT:   %.loc10_3: ref %C = splice_block %c3.var {}
// CHECK:STDOUT:   %addr.loc10_19: %ptr.d9e = addr_of %c1.ref.loc10
// CHECK:STDOUT:   %.loc10_24.1: %C = bind_value %c2.ref
// CHECK:STDOUT:   %.loc10_24.2: ref %C = value_as_ref %.loc10_24.1
// CHECK:STDOUT:   %addr.loc10_22.1: %ptr.d9e = addr_of %.loc10_24.2
// CHECK:STDOUT:   %addr.loc10_22.2: %ptr.d9e = addr_of %.loc10_3
// CHECK:STDOUT:   %operator+__carbon_thunk.call: init %empty_tuple.type = call imports.%operator+__carbon_thunk.decl(%addr.loc10_19, %addr.loc10_22.1, %addr.loc10_22.2)
// CHECK:STDOUT:   %.loc10_22: init %C = in_place_init %operator+__carbon_thunk.call, %.loc10_3
// CHECK:STDOUT:   assign %c3.var, %.loc10_22
// CHECK:STDOUT:   %.loc10_14: type = splice_block %C.ref.loc10 [concrete = constants.%C] {
// CHECK:STDOUT:     %Cpp.ref.loc10: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %C.ref.loc10: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c3: ref %C = bind_name c3, %c3.var
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.bound.loc10: <bound method> = bound_method %c3.var, constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.841
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method.loc10: <bound method> = bound_method %c3.var, %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.1
// CHECK:STDOUT:   %addr.loc10_3: %ptr.d9e = addr_of %c3.var
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.call.loc10: init %empty_tuple.type = call %bound_method.loc10(%addr.loc10_3)
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.bound.loc9: <bound method> = bound_method %c2.var, constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.841
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method.loc9: <bound method> = bound_method %c2.var, %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.2
// CHECK:STDOUT:   %addr.loc9_3: %ptr.d9e = addr_of %c2.var
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.call.loc9: init %empty_tuple.type = call %bound_method.loc9(%addr.loc9_3)
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.bound.loc8: <bound method> = bound_method %c1.var, constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.841
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method.loc8: <bound method> = bound_method %c1.var, %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.3
// CHECK:STDOUT:   %addr.loc8_3: %ptr.d9e = addr_of %c1.var
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.call.loc8: init %empty_tuple.type = call %bound_method.loc8(%addr.loc8_3)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- indirect_template_instantiation.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %B: type = class_type @B [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %ptr.a04: type = ptr_type %B [concrete]
// CHECK:STDOUT:   %operator+__carbon_thunk.type: type = fn_type @operator+__carbon_thunk [concrete]
// CHECK:STDOUT:   %operator+__carbon_thunk: %operator+__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %operator+__carbon_thunk.decl: %operator+__carbon_thunk.type = fn_decl @operator+__carbon_thunk [concrete = constants.%operator+__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%x.param: %B) -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %x.ref.loc14_10: %B = name_ref x, %x
// CHECK:STDOUT:   %x.ref.loc14_14: %B = name_ref x, %x
// CHECK:STDOUT:   %.loc14_10: ref %B = value_as_ref %x.ref.loc14_10
// CHECK:STDOUT:   %addr.loc14_12.1: %ptr.a04 = addr_of %.loc14_10
// CHECK:STDOUT:   %.loc14_14: ref %B = value_as_ref %x.ref.loc14_14
// CHECK:STDOUT:   %addr.loc14_12.2: %ptr.a04 = addr_of %.loc14_14
// CHECK:STDOUT:   %operator+__carbon_thunk.call: init %i32 = call imports.%operator+__carbon_thunk.decl(%addr.loc14_12.1, %addr.loc14_12.2)
// CHECK:STDOUT:   return %operator+__carbon_thunk.call to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_overloading.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %pattern_type.217: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %ptr.d9e: type = ptr_type %C [concrete]
// CHECK:STDOUT:   %operator+__carbon_thunk.type: type = fn_type @operator+__carbon_thunk [concrete]
// CHECK:STDOUT:   %operator+__carbon_thunk: %operator+__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %type_where: type = facet_type <type where .Self impls <CanDestroy>> [concrete]
// CHECK:STDOUT:   %facet_value: %type_where = facet_value %C, () [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.type.b92: type = fn_type @DestroyT.binding.as_type.as.Destroy.impl.Op, @DestroyT.binding.as_type.as.Destroy.impl(%facet_value) [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.841: %DestroyT.binding.as_type.as.Destroy.impl.Op.type.b92 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   %operator+__carbon_thunk.decl: %operator+__carbon_thunk.type = fn_decl @operator+__carbon_thunk [concrete = constants.%operator+__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   <elided>
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
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.bound.loc10: <bound method> = bound_method %.loc10_22.3, constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.841
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method.loc10: <bound method> = bound_method %.loc10_22.3, %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.1
// CHECK:STDOUT:   %addr.loc10_22.4: %ptr.d9e = addr_of %.loc10_22.3
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.call.loc10: init %empty_tuple.type = call %bound_method.loc10(%addr.loc10_22.4)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
