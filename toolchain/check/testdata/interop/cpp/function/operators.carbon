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
auto operator++(C& operand) -> C&;
auto operator--(C& operand) -> C&;

// Arithmetic.
auto operator-(C operand) -> C;

// Bitwise.
auto operator~(C operand) -> C;

// --- import_unary_operators.carbon

library "[[@TEST_NAME]]";

import Cpp library "unary_operators.h";

fn F() {
  //@dump-sem-ir-begin
  // CHECK:STDERR: import_unary_operators.carbon:[[@LINE+4]]:18: error: failed to evaluate constexpr function call as a constant [CppConstexprEval]
  // CHECK:STDERR:   var c: Cpp.C = Cpp.C.C();
  // CHECK:STDERR:                  ^~~~~~~~~
  // CHECK:STDERR:
  var c: Cpp.C = Cpp.C.C();

  // Increment and Decrement.
  ++c;
  --c;

  // Arithmetic.
  let unused minus: Cpp.C = -c;

  // Bitwise.
  let unused complement: Cpp.C = ^c;
  //@dump-sem-ir-end
}

// ============================================================================
// Prefix increment and decrement vs postfix increment and decrement
// ============================================================================

// --- prefix_inc_and_dec.h

class Prefix {};
auto operator++(Prefix& operand) -> Prefix&;
auto operator--(Prefix& operand) -> Prefix&;

// --- fail_prefix_calling_postfix.carbon

library "[[@TEST_NAME]]";

import Cpp library "prefix_inc_and_dec.h";

fn F() {
  // CHECK:STDERR: fail_prefix_calling_postfix.carbon:[[@LINE+4]]:28: error: failed to evaluate constexpr function call as a constant [CppConstexprEval]
  // CHECK:STDERR:   var prefix: Cpp.Prefix = Cpp.Prefix.Prefix();
  // CHECK:STDERR:                            ^~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
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
auto operator++(Postfix& operand, int) -> Postfix&;
auto operator--(Postfix& operand, int) -> Postfix&;

// --- fail_postfix_calling_prefix.carbon

library "[[@TEST_NAME]]";

import Cpp library "postfix_inc_and_dec.h";

fn F() {
  // CHECK:STDERR: fail_postfix_calling_prefix.carbon:[[@LINE+4]]:30: error: failed to evaluate constexpr function call as a constant [CppConstexprEval]
  // CHECK:STDERR:   var postfix: Cpp.Postfix = Cpp.Postfix.Postfix();
  // CHECK:STDERR:                              ^~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  var postfix: Cpp.Postfix = Cpp.Postfix.Postfix();
  // CHECK:STDERR: fail_postfix_calling_prefix.carbon:[[@LINE+4]]:3: error: cannot access member of interface `Core.Inc` in type `Cpp.Postfix` that does not implement that interface [MissingImplInMemberAccess]
  // CHECK:STDERR:   ++postfix;
  // CHECK:STDERR:   ^~~~~~~~~
  // CHECK:STDERR:
  ++postfix;
  // CHECK:STDERR: fail_postfix_calling_prefix.carbon:[[@LINE+4]]:3: error: cannot access member of interface `Core.Dec` in type `Cpp.Postfix` that does not implement that interface [MissingImplInMemberAccess]
  // CHECK:STDERR:   --postfix;
  // CHECK:STDERR:   ^~~~~~~~~
  // CHECK:STDERR:
  --postfix;
}

// ============================================================================
// Binary operators
// ============================================================================

// --- binary_operators.h

class C {
 public:
  void operator=(int);
  int operator[](int);
};

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
auto operator+=(C& lhs, C rhs) -> C&;
auto operator-=(C& lhs, C rhs) -> C&;
auto operator*=(C& lhs, C rhs) -> C&;
auto operator/=(C& lhs, C rhs) -> C&;
auto operator%=(C& lhs, C rhs) -> C&;

// Compound Assignment Bitwuse.
auto operator&=(C& lhs, C rhs) -> C&;
auto operator|=(C& lhs, C rhs) -> C&;
auto operator^=(C& lhs, C rhs) -> C&;
auto operator<<=(C& lhs, int num_bites) -> C&;
auto operator>>=(C& lhs, int num_bites) -> C&;

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
  // CHECK:STDERR: import_binary_operators.carbon:[[@LINE+4]]:19: error: failed to evaluate constexpr function call as a constant [CppConstexprEval]
  // CHECK:STDERR:   var c1: Cpp.C = Cpp.C.C();
  // CHECK:STDERR:                   ^~~~~~~~~
  // CHECK:STDERR:
  var c1: Cpp.C = Cpp.C.C();
  // CHECK:STDERR: import_binary_operators.carbon:[[@LINE+4]]:19: error: failed to evaluate constexpr function call as a constant [CppConstexprEval]
  // CHECK:STDERR:   var c2: Cpp.C = Cpp.C.C();
  // CHECK:STDERR:                   ^~~~~~~~~
  // CHECK:STDERR:
  var c2: Cpp.C = Cpp.C.C();

  // Arithmetic.
  let unused addition: Cpp.C = c1 + c2;
  let unused subtraction: Cpp.C = c1 - c2;
  let unused multiplication: Cpp.C = c1 * c2;
  let unused division: Cpp.C = c1 / c2;
  let unused modulo: Cpp.C = c1 % c2;

  // Bitwise.
  let unused bitwise_and: Cpp.C = c1 & c2;
  let unused bitwise_or: Cpp.C = c1 | c2;
  let unused bitwise_xor: Cpp.C = c1 ^ c2;
  let unused left_shift: Cpp.C = c1 << 3;
  let unused right_shift: Cpp.C = c1 >> 5;

  // Compound assignment, arithmetic.
  c1 += c2;
  c1 -= c2;
  c1 *= c2;
  c1 /= c2;
  c1 %= c2;

  // Compound assignment, bitwise.
  c1 &= c2;
  c1 |= c2;
  c1 ^= c2;
  c1 <<= 3;
  c1 >>= 5;

  // Relational.
  let unused equal: bool = c1 == c2;
  let unused not_equal: bool = c1 != c2;
  let unused greater_than: bool = c1 > c2;
  let unused less_than: bool = c1 < c2;
  let unused greater_than_or_equal: bool = c1 >= c2;
  let unused less_than_or_equal: bool = c1 <= c2;

  // Indexing.
  let unused index: i32 = c1[42];
  //@dump-sem-ir-end
}

// --- fail_todo_import_assignment.carbon

library "[[@TEST_NAME]]";

import Cpp library "binary_operators.h";

fn F() {
  var c: Cpp.C = {};

  // Simple assignment.
  // CHECK:STDERR: fail_todo_import_assignment.carbon:[[@LINE+7]]:3: error: cannot implicitly convert expression of type `Core.IntLiteral` to `Cpp.C` [ConversionFailure]
  // CHECK:STDERR:   c = 42;
  // CHECK:STDERR:   ^~~~~~
  // CHECK:STDERR: fail_todo_import_assignment.carbon:[[@LINE+4]]:3: note: type `Core.IntLiteral` does not implement interface `Core.ImplicitAs(Cpp.C)` [MissingImplInMemberAccessInContext]
  // CHECK:STDERR:   c = 42;
  // CHECK:STDERR:   ^~~~~~
  // CHECK:STDERR:
  c = 42;
}

// --- multiple_calls.carbon

library "[[@TEST_NAME]]";

import Cpp library "binary_operators.h";

fn F() {
  //@dump-sem-ir-begin
  // CHECK:STDERR: multiple_calls.carbon:[[@LINE+4]]:19: error: failed to evaluate constexpr function call as a constant [CppConstexprEval]
  // CHECK:STDERR:   let c1: Cpp.C = Cpp.C.C();
  // CHECK:STDERR:                   ^~~~~~~~~
  // CHECK:STDERR:
  let c1: Cpp.C = Cpp.C.C();
  // CHECK:STDERR: multiple_calls.carbon:[[@LINE+4]]:19: error: failed to evaluate constexpr function call as a constant [CppConstexprEval]
  // CHECK:STDERR:   let c2: Cpp.C = Cpp.C.C();
  // CHECK:STDERR:                   ^~~~~~~~~
  // CHECK:STDERR:
  let c2: Cpp.C = Cpp.C.C();
  let c3: Cpp.C = c1 + c2;
  let c4: Cpp.C = c1 + c3;
  let unused c5: Cpp.C = c4 + c3;
  //@dump-sem-ir-end
}

// --- fail_call_with_wrong_type.carbon

library "[[@TEST_NAME]]";

import Cpp library "binary_operators.h";

fn F() {
  // CHECK:STDERR: fail_call_with_wrong_type.carbon:[[@LINE+4]]:19: error: failed to evaluate constexpr function call as a constant [CppConstexprEval]
  // CHECK:STDERR:   let c1: Cpp.C = Cpp.C.C();
  // CHECK:STDERR:                   ^~~~~~~~~
  // CHECK:STDERR:
  let c1: Cpp.C = Cpp.C.C();
  // TODO: Include a list of non-viable candidate operators with this diagnostic.
  // CHECK:STDERR: fail_call_with_wrong_type.carbon:[[@LINE+4]]:26: error: cannot access member of interface `Core.AddWith(Core.IntLiteral)` in type `Cpp.C` that does not implement that interface [MissingImplInMemberAccess]
  // CHECK:STDERR:   let unused c2: Cpp.C = c1 + 5;
  // CHECK:STDERR:                          ^~~~~~
  // CHECK:STDERR:
  let unused c2: Cpp.C = c1 + 5;
  // CHECK:STDERR: fail_call_with_wrong_type.carbon:[[@LINE+4]]:26: error: cannot access member of interface `Core.AddWith(Cpp.C)` in type `Core.IntLiteral` that does not implement that interface [MissingImplInMemberAccess]
  // CHECK:STDERR:   let unused c3: Cpp.C = 6 + c1;
  // CHECK:STDERR:                          ^~~~~~
  // CHECK:STDERR:
  let unused c3: Cpp.C = 6 + c1;
}

// ============================================================================
// Overload resolution failures
// ============================================================================

// --- fail_ambiguous.carbon

library "[[@TEST_NAME]]";

import Cpp inline '''
class C {};
void operator+(C, char);
void operator+(C, short);
''';

fn Test(c: Cpp.C) {
  // CHECK:STDERR: fail_ambiguous.carbon:[[@LINE+10]]:5: error: use of overloaded operator '+' is ambiguous (with operand types 'const C' and 'int') [CppInteropParseError]
  // CHECK:STDERR:    21 |   c + 1;
  // CHECK:STDERR:       |     ^
  // CHECK:STDERR: fail_ambiguous.carbon:[[@LINE-8]]:6: note: candidate function [CppInteropParseNote]
  // CHECK:STDERR:     6 | void operator+(C, char);
  // CHECK:STDERR:       |      ^
  // CHECK:STDERR: fail_ambiguous.carbon:[[@LINE-10]]:6: note: candidate function [CppInteropParseNote]
  // CHECK:STDERR:     7 | void operator+(C, short);
  // CHECK:STDERR:       |      ^
  // CHECK:STDERR:
  c + 1;
}

// --- fail_deleted.carbon

library "[[@TEST_NAME]]";

import Cpp inline '''
class C {};
void operator+(C, int) = delete;
void operator+(C, void*) = delete;
''';

fn Test(c: Cpp.C) {
  // CHECK:STDERR: fail_deleted.carbon:[[@LINE+10]]:5: error: overload resolution selected deleted operator '+' [CppInteropParseError]
  // CHECK:STDERR:    21 |   c + 1;
  // CHECK:STDERR:       |     ^
  // CHECK:STDERR: fail_deleted.carbon:[[@LINE-8]]:6: note: candidate function has been explicitly deleted [CppInteropParseNote]
  // CHECK:STDERR:     6 | void operator+(C, int) = delete;
  // CHECK:STDERR:       |      ^
  // CHECK:STDERR: fail_deleted.carbon:[[@LINE-10]]:6: note: candidate function not viable: no known conversion from 'int' to 'void *' for 2nd argument [CppInteropParseNote]
  // CHECK:STDERR:     7 | void operator+(C, void*) = delete;
  // CHECK:STDERR:       |      ^            ~~~~~
  // CHECK:STDERR:
  c + 1;
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
  // CHECK:STDERR: fail_todo_rewrite_spaceship.carbon:[[@LINE+4]]:19: error: failed to evaluate constexpr function call as a constant [CppConstexprEval]
  // CHECK:STDERR:   var c1: Cpp.C = Cpp.C.C();
  // CHECK:STDERR:                   ^~~~~~~~~
  // CHECK:STDERR:
  var c1: Cpp.C = Cpp.C.C();
  // CHECK:STDERR: fail_todo_rewrite_spaceship.carbon:[[@LINE+4]]:19: error: failed to evaluate constexpr function call as a constant [CppConstexprEval]
  // CHECK:STDERR:   var c2: Cpp.C = Cpp.C.C();
  // CHECK:STDERR:                   ^~~~~~~~~
  // CHECK:STDERR:
  var c2: Cpp.C = Cpp.C.C();

  // No rewrite.
  let unused greater_than: bool = c1 > c2;

  // Rewrite.
  // CHECK:STDERR: fail_todo_rewrite_spaceship.carbon:[[@LINE+4]]:32: error: semantics TODO: `Rewriting operator< using operator<=> is not supported` [SemanticsTodo]
  // CHECK:STDERR:   let unused less_than: bool = c1 < c2;
  // CHECK:STDERR:                                ^~~~~~~
  // CHECK:STDERR:
  let unused less_than: bool = c1 < c2;

  // Rewrite.
  // CHECK:STDERR: fail_todo_rewrite_spaceship.carbon:[[@LINE+4]]:44: error: semantics TODO: `Rewriting operator>= using operator<=> is not supported` [SemanticsTodo]
  // CHECK:STDERR:   let unused greater_than_or_equal: bool = c1 >= c2;
  // CHECK:STDERR:                                            ^~~~~~~~
  // CHECK:STDERR:
  let unused greater_than_or_equal: bool = c1 >= c2;

  // No rewrite.
  let unused less_than_or_equal: bool = c1 <= c2;
  //@dump-sem-ir-end
}

// --- fail_rewrite_deleted.carbon

library "[[@TEST_NAME]]";

import Cpp inline '''
class B {};
class C : public B {};

namespace std { class strong_ordering {}; }
auto operator<=>(C lhs, C rhs) -> std::strong_ordering = delete;
auto operator<(C lhs, B rhs) -> bool;
''';

fn F(c: Cpp.C) -> bool {
  // Note that we mention `operator '<=>'` in the diagnostic.
  // CHECK:STDERR: fail_rewrite_deleted.carbon:[[@LINE+10]]:12: error: overload resolution selected deleted operator '<=>' [CppInteropParseError]
  // CHECK:STDERR:    25 |   return c < c;
  // CHECK:STDERR:       |            ^
  // CHECK:STDERR: fail_rewrite_deleted.carbon:[[@LINE-9]]:6: note: candidate function has been explicitly deleted [CppInteropParseNote]
  // CHECK:STDERR:     9 | auto operator<=>(C lhs, C rhs) -> std::strong_ordering = delete;
  // CHECK:STDERR:       |      ^
  // CHECK:STDERR: fail_rewrite_deleted.carbon:[[@LINE-11]]:6: note: candidate function [CppInteropParseNote]
  // CHECK:STDERR:    10 | auto operator<(C lhs, B rhs) -> bool;
  // CHECK:STDERR:       |      ^
  // CHECK:STDERR:
  return c < c;
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
  // CHECK:STDERR: fail_todo_rewrite_equal.carbon:[[@LINE+4]]:19: error: failed to evaluate constexpr function call as a constant [CppConstexprEval]
  // CHECK:STDERR:   var c1: Cpp.C = Cpp.C.C();
  // CHECK:STDERR:                   ^~~~~~~~~
  // CHECK:STDERR:
  var c1: Cpp.C = Cpp.C.C();
  // CHECK:STDERR: fail_todo_rewrite_equal.carbon:[[@LINE+4]]:19: error: failed to evaluate constexpr function call as a constant [CppConstexprEval]
  // CHECK:STDERR:   var c2: Cpp.C = Cpp.C.C();
  // CHECK:STDERR:                   ^~~~~~~~~
  // CHECK:STDERR:
  var c2: Cpp.C = Cpp.C.C();

  // No rewrite.
  let unused equal: bool = c1 == c2;

  // Rewrite.
  // CHECK:STDERR: fail_todo_rewrite_equal.carbon:[[@LINE+4]]:32: error: semantics TODO: `Rewriting operator!= using operator== is not supported` [SemanticsTodo]
  // CHECK:STDERR:   let unused not_equal: bool = c1 != c2;
  // CHECK:STDERR:                                ^~~~~~~~
  // CHECK:STDERR:
  let unused not_equal: bool = c1 != c2;
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

// --- plus_with_int_conversion.carbon

library "[[@TEST_NAME]]";

import Cpp library "plus_with_int_conversion.h";

fn F() {
  let c1: Cpp.C = Cpp.C.C(4);
  let unused c2: Cpp.C = c1 + 5;
  let unused c3: Cpp.C = 6 + c1;
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
  // CHECK:STDERR: fail_todo_import_plus_with_string_view_conversion.carbon:[[@LINE+4]]:25: error: cannot access member of interface `Core.AddWith(str)` in type `str` that does not implement that interface [MissingImplInMemberAccess]
  // CHECK:STDERR:   let unused c: Cpp.C = s1 + s2;
  // CHECK:STDERR:                         ^~~~~~~
  // CHECK:STDERR:
  let unused c: Cpp.C = s1 + s2;
}

// ============================================================================
// Unsupported unary operators
// ============================================================================

// --- unsupported_unary_operators.h

class C {};

// Arithmetic.
auto operator+(C operand) -> C;

// Increment and Decrement.
auto operator++(C& operand, int) -> C&;
auto operator--(C& operand, int) -> C&;

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
  // CHECK:STDERR: fail_todo_import_unsupported_unary_operators.carbon:[[@LINE+4]]:18: error: failed to evaluate constexpr function call as a constant [CppConstexprEval]
  // CHECK:STDERR:   var c: Cpp.C = Cpp.C.C();
  // CHECK:STDERR:                  ^~~~~~~~~
  // CHECK:STDERR:
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

// Logical.
auto operator&&(C lhs, C rhs) -> C;
auto operator||(C lhs, C rhs) -> C;

// --- fail_todo_import_unsupported_binary_operators.carbon

library "[[@TEST_NAME]]";

import Cpp library "unsupported_binary_operators.h";

fn F() {
  // CHECK:STDERR: fail_todo_import_unsupported_binary_operators.carbon:[[@LINE+4]]:19: error: failed to evaluate constexpr function call as a constant [CppConstexprEval]
  // CHECK:STDERR:   var c1: Cpp.C = Cpp.C.C();
  // CHECK:STDERR:                   ^~~~~~~~~
  // CHECK:STDERR:
  var c1: Cpp.C = Cpp.C.C();

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
  // CHECK:STDERR: import_single_namespace.carbon:[[@LINE+4]]:21: error: failed to evaluate constexpr function call as a constant [CppConstexprEval]
  // CHECK:STDERR:   let c1: Cpp.N.C = Cpp.N.C.C();
  // CHECK:STDERR:                     ^~~~~~~~~~~
  // CHECK:STDERR:
  let c1: Cpp.N.C = Cpp.N.C.C();
  // CHECK:STDERR: import_single_namespace.carbon:[[@LINE+4]]:21: error: failed to evaluate constexpr function call as a constant [CppConstexprEval]
  // CHECK:STDERR:   let c2: Cpp.N.C = Cpp.N.C.C();
  // CHECK:STDERR:                     ^~~~~~~~~~~
  // CHECK:STDERR:
  let c2: Cpp.N.C = Cpp.N.C.C();
  let unused c3: Cpp.N.C = c1 + c2;
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
  // CHECK:STDERR: import_multiple_namespaces.carbon:[[@LINE+4]]:23: error: failed to evaluate constexpr function call as a constant [CppConstexprEval]
  // CHECK:STDERR:   let c1: Cpp.N1.C1 = Cpp.N1.C1.C1();
  // CHECK:STDERR:                       ^~~~~~~~~~~~~~
  // CHECK:STDERR:
  let c1: Cpp.N1.C1 = Cpp.N1.C1.C1();
  // CHECK:STDERR: import_multiple_namespaces.carbon:[[@LINE+4]]:23: error: failed to evaluate constexpr function call as a constant [CppConstexprEval]
  // CHECK:STDERR:   let c2: Cpp.N2.C2 = Cpp.N2.C2.C2();
  // CHECK:STDERR:                       ^~~~~~~~~~~~~~
  // CHECK:STDERR:
  let c2: Cpp.N2.C2 = Cpp.N2.C2.C2();
  let unused c3: Cpp.N2.C2 = c1 + c2;
  let unused c4: Cpp.N2.C2 = c2 - c1;
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
  // CHECK:STDERR: fail_todo_import_operands_in_namespace_operator_in_global.carbon:[[@LINE+4]]:21: error: failed to evaluate constexpr function call as a constant [CppConstexprEval]
  // CHECK:STDERR:   let c1: Cpp.N.C = Cpp.N.C.C();
  // CHECK:STDERR:                     ^~~~~~~~~~~
  // CHECK:STDERR:
  let c1: Cpp.N.C = Cpp.N.C.C();
  // CHECK:STDERR: fail_todo_import_operands_in_namespace_operator_in_global.carbon:[[@LINE+4]]:21: error: failed to evaluate constexpr function call as a constant [CppConstexprEval]
  // CHECK:STDERR:   let c2: Cpp.N.C = Cpp.N.C.C();
  // CHECK:STDERR:                     ^~~~~~~~~~~
  // CHECK:STDERR:
  let c2: Cpp.N.C = Cpp.N.C.C();
  // CHECK:STDERR: fail_todo_import_operands_in_namespace_operator_in_global.carbon:[[@LINE+4]]:28: error: cannot access member of interface `Core.AddWith(Cpp.N.C)` in type `Cpp.N.C` that does not implement that interface [MissingImplInMemberAccess]
  // CHECK:STDERR:   let unused c3: Cpp.N.C = c1 + c2;
  // CHECK:STDERR:                            ^~~~~~~
  // CHECK:STDERR:
  let unused c3: Cpp.N.C = c1 + c2;
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
  // CHECK:STDERR: import_inner_class.carbon:[[@LINE+4]]:21: error: failed to evaluate constexpr function call as a constant [CppConstexprEval]
  // CHECK:STDERR:   let c1: Cpp.O.C = Cpp.O.C.C();
  // CHECK:STDERR:                     ^~~~~~~~~~~
  // CHECK:STDERR:
  let c1: Cpp.O.C = Cpp.O.C.C();
  // CHECK:STDERR: import_inner_class.carbon:[[@LINE+4]]:21: error: failed to evaluate constexpr function call as a constant [CppConstexprEval]
  // CHECK:STDERR:   let c2: Cpp.O.C = Cpp.O.C.C();
  // CHECK:STDERR:                     ^~~~~~~~~~~
  // CHECK:STDERR:
  let c2: Cpp.O.C = Cpp.O.C.C();
  let unused c3: Cpp.O.C = c1 + c2;
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
  // CHECK:STDERR: import_inner_class_in_namespace.carbon:[[@LINE+4]]:23: error: failed to evaluate constexpr function call as a constant [CppConstexprEval]
  // CHECK:STDERR:   let c1: Cpp.N.O.C = Cpp.N.O.C.C();
  // CHECK:STDERR:                       ^~~~~~~~~~~~~
  // CHECK:STDERR:
  let c1: Cpp.N.O.C = Cpp.N.O.C.C();
  // CHECK:STDERR: import_inner_class_in_namespace.carbon:[[@LINE+4]]:23: error: failed to evaluate constexpr function call as a constant [CppConstexprEval]
  // CHECK:STDERR:   let c2: Cpp.N.O.C = Cpp.N.O.C.C();
  // CHECK:STDERR:                       ^~~~~~~~~~~~~
  // CHECK:STDERR:
  let c2: Cpp.N.O.C = Cpp.N.O.C.C();
  let unused c3: Cpp.N.O.C = c1 + c2;
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
  // CHECK:STDERR: import_member_add_with.carbon:[[@LINE+4]]:19: error: failed to evaluate constexpr function call as a constant [CppConstexprEval]
  // CHECK:STDERR:   var c1: Cpp.C = Cpp.C.C();
  // CHECK:STDERR:                   ^~~~~~~~~
  // CHECK:STDERR:
  var c1: Cpp.C = Cpp.C.C();
  var c2: Cpp.C = -c1;
  var unused c3: Cpp.C = c1 + c2;
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
  // CHECK:STDERR: fail_import_not_found.carbon:[[@LINE+4]]:19: error: failed to evaluate constexpr function call as a constant [CppConstexprEval]
  // CHECK:STDERR:   let c1: Cpp.C = Cpp.C.C();
  // CHECK:STDERR:                   ^~~~~~~~~
  // CHECK:STDERR:
  let c1: Cpp.C = Cpp.C.C();
  // CHECK:STDERR: fail_import_not_found.carbon:[[@LINE+4]]:19: error: failed to evaluate constexpr function call as a constant [CppConstexprEval]
  // CHECK:STDERR:   let c2: Cpp.C = Cpp.C.C();
  // CHECK:STDERR:                   ^~~~~~~~~
  // CHECK:STDERR:
  let c2: Cpp.C = Cpp.C.C();
  // CHECK:STDERR: fail_import_not_found.carbon:[[@LINE+4]]:26: error: cannot access member of interface `Core.AddWith(Cpp.C)` in type `Cpp.C` that does not implement that interface [MissingImplInMemberAccess]
  // CHECK:STDERR:   let unused c3: Cpp.C = c1 + c2;
  // CHECK:STDERR:                          ^~~~~~~
  // CHECK:STDERR:
  let unused c3: Cpp.C = c1 + c2;
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
  // CHECK:STDERR: fail_import_incomplete_unary.carbon:[[@LINE+8]]:34: error: looking up a C++ operator with incomplete operand type `Cpp.Incomplete` [IncompleteOperandTypeInCppOperatorLookup]
  // CHECK:STDERR:   let unused result_unary: i32 = -*Cpp.CreateIncomplete();
  // CHECK:STDERR:                                  ^~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_import_incomplete_unary.carbon:[[@LINE-6]]:10: in file included here [InCppInclude]
  // CHECK:STDERR: ./incomplete.h:2:7: note: class was forward declared here [ClassForwardDeclaredHere]
  // CHECK:STDERR: class Incomplete;
  // CHECK:STDERR:       ^
  // CHECK:STDERR:
  let unused result_unary: i32 = -*Cpp.CreateIncomplete();
}

// --- fail_import_incomplete_binary.carbon

library "[[@TEST_NAME]]";

import Cpp library "incomplete.h";

fn F() {
  // CHECK:STDERR: fail_import_incomplete_binary.carbon:[[@LINE+4]]:32: error: failed to evaluate constexpr function call as a constant [CppConstexprEval]
  // CHECK:STDERR:   var complete: Cpp.Complete = Cpp.Complete.Complete();
  // CHECK:STDERR:                                ^~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  var complete: Cpp.Complete = Cpp.Complete.Complete();
  // CHECK:STDERR: fail_import_incomplete_binary.carbon:[[@LINE+8]]:35: error: looking up a C++ operator with incomplete operand type `Cpp.Incomplete` [IncompleteOperandTypeInCppOperatorLookup]
  // CHECK:STDERR:   let unused result_binary: i32 = complete + *Cpp.CreateIncomplete();
  // CHECK:STDERR:                                   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_import_incomplete_binary.carbon:[[@LINE-11]]:10: in file included here [InCppInclude]
  // CHECK:STDERR: ./incomplete.h:2:7: note: class was forward declared here [ClassForwardDeclaredHere]
  // CHECK:STDERR: class Incomplete;
  // CHECK:STDERR:       ^
  // CHECK:STDERR:
  let unused result_binary: i32 = complete + *Cpp.CreateIncomplete();
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
  // CHECK:STDERR: fail_incomplete_operand_carbon_type.carbon:[[@LINE+4]]:32: error: failed to evaluate constexpr function call as a constant [CppConstexprEval]
  // CHECK:STDERR:   var complete: Cpp.Complete = Cpp.Complete.Complete();
  // CHECK:STDERR:                                ^~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  var complete: Cpp.Complete = Cpp.Complete.Complete();
  // CHECK:STDERR: fail_incomplete_operand_carbon_type.carbon:[[@LINE+7]]:28: error: looking up a C++ operator with incomplete operand type `Incomplete` [IncompleteOperandTypeInCppOperatorLookup]
  // CHECK:STDERR:   let unused result: i32 = *CreateIncomplete() + complete;
  // CHECK:STDERR:                            ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_incomplete_operand_carbon_type.carbon:[[@LINE-12]]:1: note: class was forward declared here [ClassForwardDeclaredHere]
  // CHECK:STDERR: class Incomplete;
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  let unused result: i32 = *CreateIncomplete() + complete;
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
  // CHECK:STDERR: fail_indirect_template_instantiation_error.carbon:[[@LINE+4]]:10: error: cannot access member of interface `Core.AddWith(Cpp.B)` in type `Cpp.B` that does not implement that interface [MissingImplInMemberAccess]
  // CHECK:STDERR:   return x + x;
  // CHECK:STDERR:          ^~~~~
  // CHECK:STDERR:
  return x + x;
}

// ============================================================================
// Templated operators
// ============================================================================

// --- templated_operator.h

template<typename T> struct X {};
template<typename T> X<T> operator+(X<T>, X<T>);
using Xint = X<int>;

// --- templated_operator.carbon

library "[[@TEST_NAME]]";

import Cpp library "templated_operator.h";

fn Test(x: Cpp.Xint) -> Cpp.Xint {
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
  // CHECK:STDERR: import_overloading.carbon:[[@LINE+4]]:19: error: failed to evaluate constexpr function call as a constant [CppConstexprEval]
  // CHECK:STDERR:   let c1: Cpp.C = Cpp.C.C();
  // CHECK:STDERR:                   ^~~~~~~~~
  // CHECK:STDERR:
  let c1: Cpp.C = Cpp.C.C();
  // CHECK:STDERR: import_overloading.carbon:[[@LINE+4]]:19: error: failed to evaluate constexpr function call as a constant [CppConstexprEval]
  // CHECK:STDERR:   let c2: Cpp.C = Cpp.C.C();
  // CHECK:STDERR:                   ^~~~~~~~~
  // CHECK:STDERR:
  let c2: Cpp.C = Cpp.C.C();
  //@dump-sem-ir-begin
  let unused c3: Cpp.C = c1 + c2;
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
// CHECK:STDOUT:   %cpp_operator.type.1ea478.1: type = fn_type @cpp_operator.1 [concrete]
// CHECK:STDOUT:   %cpp_operator.0a3797.1: %cpp_operator.type.1ea478.1 = struct_value () [concrete]
// CHECK:STDOUT:   %cpp_operator.type.1ea478.2: type = fn_type @cpp_operator.2 [concrete]
// CHECK:STDOUT:   %cpp_operator.0a3797.2: %cpp_operator.type.1ea478.2 = struct_value () [concrete]
// CHECK:STDOUT:   %operator_Minus__carbon_thunk.type: type = fn_type @operator_Minus__carbon_thunk [concrete]
// CHECK:STDOUT:   %operator_Minus__carbon_thunk: %operator_Minus__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %operator_Tilde__carbon_thunk.type: type = fn_type @operator_Tilde__carbon_thunk [concrete]
// CHECK:STDOUT:   %operator_Tilde__carbon_thunk: %operator_Tilde__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %C.cpp_destructor.type: type = fn_type @C.cpp_destructor [concrete]
// CHECK:STDOUT:   %C.cpp_destructor: %C.cpp_destructor.type = struct_value () [concrete]
// CHECK:STDOUT:   %C.Op.type: type = fn_type @C.Op [concrete]
// CHECK:STDOUT:   %C.Op: %C.Op.type = struct_value () [concrete]
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
// CHECK:STDOUT:   %cpp_operator.decl.4206c9.1: %cpp_operator.type.1ea478.1 = fn_decl @cpp_operator.1 [concrete = constants.%cpp_operator.0a3797.1] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %cpp_operator.decl.4206c9.2: %cpp_operator.type.1ea478.2 = fn_decl @cpp_operator.2 [concrete = constants.%cpp_operator.0a3797.2] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %operator_Minus__carbon_thunk.decl: %operator_Minus__carbon_thunk.type = fn_decl @operator_Minus__carbon_thunk [concrete = constants.%operator_Minus__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %operator_Tilde__carbon_thunk.decl: %operator_Tilde__carbon_thunk.type = fn_decl @operator_Tilde__carbon_thunk [concrete = constants.%operator_Tilde__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.cpp_destructor.decl: %C.cpp_destructor.type = fn_decl @C.cpp_destructor [concrete = constants.%C.cpp_destructor] {
// CHECK:STDOUT:     %self.patt: %pattern_type.217 = ref_binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.217 = ref_param_pattern %self.patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %C = ref_param call_param0
// CHECK:STDOUT:     %self: ref %C = ref_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c.patt: %pattern_type.217 = ref_binding_pattern c [concrete]
// CHECK:STDOUT:     %c.var_patt: %pattern_type.217 = var_pattern %c.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c.var: ref %C = var %c.var_patt
// CHECK:STDOUT:   %Cpp.ref.loc12_18: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %C.ref.loc12_21: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %C.ref.loc12_23: %C.C.cpp_overload_set.type = name_ref C, imports.%C.C.cpp_overload_set.value [concrete = constants.%C.C.cpp_overload_set.value]
// CHECK:STDOUT:   %.loc12_3: ref %C = splice_block %c.var {}
// CHECK:STDOUT:   %addr.loc12: %ptr.d9e = addr_of %.loc12_3
// CHECK:STDOUT:   %C__carbon_thunk.call: init %empty_tuple.type = call imports.%C__carbon_thunk.decl(%addr.loc12) [concrete = <error>]
// CHECK:STDOUT:   %.loc12_26: init %C to %.loc12_3 = mark_in_place_init %C__carbon_thunk.call [concrete = <error>]
// CHECK:STDOUT:   assign %c.var, %.loc12_26
// CHECK:STDOUT:   %.loc12_13: type = splice_block %C.ref.loc12_13 [concrete = constants.%C] {
// CHECK:STDOUT:     %Cpp.ref.loc12_10: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %C.ref.loc12_13: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c: ref %C = ref_binding c, %c.var
// CHECK:STDOUT:   %c.ref.loc15: ref %C = name_ref c, %c
// CHECK:STDOUT:   %cpp_operator.call.loc15: ref %C = call imports.%cpp_operator.decl.4206c9.1(%c.ref.loc15)
// CHECK:STDOUT:   %c.ref.loc16: ref %C = name_ref c, %c
// CHECK:STDOUT:   %cpp_operator.call.loc16: ref %C = call imports.%cpp_operator.decl.4206c9.2(%c.ref.loc16)
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %minus.patt: %pattern_type.217 = value_binding_pattern minus [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c.ref.loc19: ref %C = name_ref c, %c
// CHECK:STDOUT:   %.loc19_29.1: ref %C = temporary_storage
// CHECK:STDOUT:   %.loc19_30.1: %C = acquire_value %c.ref.loc19
// CHECK:STDOUT:   %.loc19_30.2: ref %C = value_as_ref %.loc19_30.1
// CHECK:STDOUT:   %addr.loc19_29.1: %ptr.d9e = addr_of %.loc19_30.2
// CHECK:STDOUT:   %addr.loc19_29.2: %ptr.d9e = addr_of %.loc19_29.1
// CHECK:STDOUT:   %operator_Minus__carbon_thunk.call: init %empty_tuple.type = call imports.%operator_Minus__carbon_thunk.decl(%addr.loc19_29.1, %addr.loc19_29.2)
// CHECK:STDOUT:   %.loc19_29.2: init %C to %.loc19_29.1 = mark_in_place_init %operator_Minus__carbon_thunk.call
// CHECK:STDOUT:   %.loc19_24: type = splice_block %C.ref.loc19 [concrete = constants.%C] {
// CHECK:STDOUT:     %Cpp.ref.loc19: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %C.ref.loc19: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc19_29.3: ref %C = temporary %.loc19_29.1, %.loc19_29.2
// CHECK:STDOUT:   %.loc19_29.4: %C = acquire_value %.loc19_29.3
// CHECK:STDOUT:   %minus: %C = value_binding minus, %.loc19_29.4
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %complement.patt: %pattern_type.217 = value_binding_pattern complement [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c.ref.loc22: ref %C = name_ref c, %c
// CHECK:STDOUT:   %.loc22_34.1: ref %C = temporary_storage
// CHECK:STDOUT:   %.loc22_35.1: %C = acquire_value %c.ref.loc22
// CHECK:STDOUT:   %.loc22_35.2: ref %C = value_as_ref %.loc22_35.1
// CHECK:STDOUT:   %addr.loc22_34.1: %ptr.d9e = addr_of %.loc22_35.2
// CHECK:STDOUT:   %addr.loc22_34.2: %ptr.d9e = addr_of %.loc22_34.1
// CHECK:STDOUT:   %operator_Tilde__carbon_thunk.call: init %empty_tuple.type = call imports.%operator_Tilde__carbon_thunk.decl(%addr.loc22_34.1, %addr.loc22_34.2)
// CHECK:STDOUT:   %.loc22_34.2: init %C to %.loc22_34.1 = mark_in_place_init %operator_Tilde__carbon_thunk.call
// CHECK:STDOUT:   %.loc22_29: type = splice_block %C.ref.loc22 [concrete = constants.%C] {
// CHECK:STDOUT:     %Cpp.ref.loc22: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %C.ref.loc22: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc22_34.3: ref %C = temporary %.loc22_34.1, %.loc22_34.2
// CHECK:STDOUT:   %.loc22_34.4: %C = acquire_value %.loc22_34.3
// CHECK:STDOUT:   %complement: %C = value_binding complement, %.loc22_34.4
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %C.Op.bound.loc22: <bound method> = bound_method %.loc22_34.3, constants.%C.Op
// CHECK:STDOUT:   %Op.ref.loc22: %C.cpp_destructor.type = name_ref Op, imports.%C.cpp_destructor.decl [concrete = constants.%C.cpp_destructor]
// CHECK:STDOUT:   %C.cpp_destructor.bound.loc22: <bound method> = bound_method %.loc22_34.3, %Op.ref.loc22
// CHECK:STDOUT:   %C.cpp_destructor.call.loc22: init %empty_tuple.type = call %C.cpp_destructor.bound.loc22(%.loc22_34.3)
// CHECK:STDOUT:   %C.Op.bound.loc19: <bound method> = bound_method %.loc19_29.3, constants.%C.Op
// CHECK:STDOUT:   %Op.ref.loc19: %C.cpp_destructor.type = name_ref Op, imports.%C.cpp_destructor.decl [concrete = constants.%C.cpp_destructor]
// CHECK:STDOUT:   %C.cpp_destructor.bound.loc19: <bound method> = bound_method %.loc19_29.3, %Op.ref.loc19
// CHECK:STDOUT:   %C.cpp_destructor.call.loc19: init %empty_tuple.type = call %C.cpp_destructor.bound.loc19(%.loc19_29.3)
// CHECK:STDOUT:   %C.Op.bound.loc12: <bound method> = bound_method %c.var, constants.%C.Op
// CHECK:STDOUT:   %Op.ref.loc12: %C.cpp_destructor.type = name_ref Op, imports.%C.cpp_destructor.decl [concrete = constants.%C.cpp_destructor]
// CHECK:STDOUT:   %C.cpp_destructor.bound.loc12: <bound method> = bound_method %c.var, %Op.ref.loc12
// CHECK:STDOUT:   %C.cpp_destructor.call.loc12: init %empty_tuple.type = call %C.cpp_destructor.bound.loc12(%c.var)
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
// CHECK:STDOUT:   %operator_Plus__carbon_thunk.type: type = fn_type @operator_Plus__carbon_thunk [concrete]
// CHECK:STDOUT:   %operator_Plus__carbon_thunk: %operator_Plus__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %operator_Minus__carbon_thunk.type: type = fn_type @operator_Minus__carbon_thunk [concrete]
// CHECK:STDOUT:   %operator_Minus__carbon_thunk: %operator_Minus__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %operator_Star__carbon_thunk.type: type = fn_type @operator_Star__carbon_thunk [concrete]
// CHECK:STDOUT:   %operator_Star__carbon_thunk: %operator_Star__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %operator_Slash__carbon_thunk.type: type = fn_type @operator_Slash__carbon_thunk [concrete]
// CHECK:STDOUT:   %operator_Slash__carbon_thunk: %operator_Slash__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %operator_Percent__carbon_thunk.type: type = fn_type @operator_Percent__carbon_thunk [concrete]
// CHECK:STDOUT:   %operator_Percent__carbon_thunk: %operator_Percent__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %operator_Amp__carbon_thunk.type: type = fn_type @operator_Amp__carbon_thunk [concrete]
// CHECK:STDOUT:   %operator_Amp__carbon_thunk: %operator_Amp__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %operator_Pipe__carbon_thunk.type: type = fn_type @operator_Pipe__carbon_thunk [concrete]
// CHECK:STDOUT:   %operator_Pipe__carbon_thunk: %operator_Pipe__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %operator_Caret__carbon_thunk.type: type = fn_type @operator_Caret__carbon_thunk [concrete]
// CHECK:STDOUT:   %operator_Caret__carbon_thunk: %operator_Caret__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %int_3.1ba: Core.IntLiteral = int_value 3 [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %pattern_type.7ce: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %operator_LessLess__carbon_thunk.type: type = fn_type @operator_LessLess__carbon_thunk [concrete]
// CHECK:STDOUT:   %operator_LessLess__carbon_thunk: %operator_LessLess__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.e8c: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.4e6: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.3c2: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.4e6 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.6bc: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.74f, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.e0d: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.e0d = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.e8c = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.6bc) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.b37: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%i32, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.545: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.b37, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.061: <bound method> = bound_method %int_3.1ba, %Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.fa7: <bound method> = bound_method %int_3.1ba, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_3.822: %i32 = int_value 3 [concrete]
// CHECK:STDOUT:   %int_5.64b: Core.IntLiteral = int_value 5 [concrete]
// CHECK:STDOUT:   %operator_GreaterGreater__carbon_thunk.type: type = fn_type @operator_GreaterGreater__carbon_thunk [concrete]
// CHECK:STDOUT:   %operator_GreaterGreater__carbon_thunk: %operator_GreaterGreater__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.005: <bound method> = bound_method %int_5.64b, %Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5 [concrete]
// CHECK:STDOUT:   %bound_method.e9d: <bound method> = bound_method %int_5.64b, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_5.0f6: %i32 = int_value 5 [concrete]
// CHECK:STDOUT:   %operator_PlusEqual__carbon_thunk.type: type = fn_type @operator_PlusEqual__carbon_thunk [concrete]
// CHECK:STDOUT:   %operator_PlusEqual__carbon_thunk: %operator_PlusEqual__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %operator_MinusEqual__carbon_thunk.type: type = fn_type @operator_MinusEqual__carbon_thunk [concrete]
// CHECK:STDOUT:   %operator_MinusEqual__carbon_thunk: %operator_MinusEqual__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %operator_StarEqual__carbon_thunk.type: type = fn_type @operator_StarEqual__carbon_thunk [concrete]
// CHECK:STDOUT:   %operator_StarEqual__carbon_thunk: %operator_StarEqual__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %operator_SlashEqual__carbon_thunk.type: type = fn_type @operator_SlashEqual__carbon_thunk [concrete]
// CHECK:STDOUT:   %operator_SlashEqual__carbon_thunk: %operator_SlashEqual__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %operator_PercentEqual__carbon_thunk.type: type = fn_type @operator_PercentEqual__carbon_thunk [concrete]
// CHECK:STDOUT:   %operator_PercentEqual__carbon_thunk: %operator_PercentEqual__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %operator_AmpEqual__carbon_thunk.type: type = fn_type @operator_AmpEqual__carbon_thunk [concrete]
// CHECK:STDOUT:   %operator_AmpEqual__carbon_thunk: %operator_AmpEqual__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %operator_PipeEqual__carbon_thunk.type: type = fn_type @operator_PipeEqual__carbon_thunk [concrete]
// CHECK:STDOUT:   %operator_PipeEqual__carbon_thunk: %operator_PipeEqual__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %operator_CaretEqual__carbon_thunk.type: type = fn_type @operator_CaretEqual__carbon_thunk [concrete]
// CHECK:STDOUT:   %operator_CaretEqual__carbon_thunk: %operator_CaretEqual__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %cpp_operator.type.1ea478.19: type = fn_type @cpp_operator.19 [concrete]
// CHECK:STDOUT:   %cpp_operator.0a3797.19: %cpp_operator.type.1ea478.19 = struct_value () [concrete]
// CHECK:STDOUT:   %cpp_operator.type.1ea478.20: type = fn_type @cpp_operator.20 [concrete]
// CHECK:STDOUT:   %cpp_operator.0a3797.20: %cpp_operator.type.1ea478.20 = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.831: type = pattern_type bool [concrete]
// CHECK:STDOUT:   %ptr.bb2: type = ptr_type bool [concrete]
// CHECK:STDOUT:   %operator_EqualEqual__carbon_thunk.type: type = fn_type @operator_EqualEqual__carbon_thunk [concrete]
// CHECK:STDOUT:   %operator_EqualEqual__carbon_thunk: %operator_EqualEqual__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %operator_ExclaimEqual__carbon_thunk.type: type = fn_type @operator_ExclaimEqual__carbon_thunk [concrete]
// CHECK:STDOUT:   %operator_ExclaimEqual__carbon_thunk: %operator_ExclaimEqual__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %operator_Greater__carbon_thunk.type: type = fn_type @operator_Greater__carbon_thunk [concrete]
// CHECK:STDOUT:   %operator_Greater__carbon_thunk: %operator_Greater__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %operator_Less__carbon_thunk.type: type = fn_type @operator_Less__carbon_thunk [concrete]
// CHECK:STDOUT:   %operator_Less__carbon_thunk: %operator_Less__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %operator_GreaterEqual__carbon_thunk.type: type = fn_type @operator_GreaterEqual__carbon_thunk [concrete]
// CHECK:STDOUT:   %operator_GreaterEqual__carbon_thunk: %operator_GreaterEqual__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %operator_LessEqual__carbon_thunk.type: type = fn_type @operator_LessEqual__carbon_thunk [concrete]
// CHECK:STDOUT:   %operator_LessEqual__carbon_thunk: %operator_LessEqual__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %int_42.20e: Core.IntLiteral = int_value 42 [concrete]
// CHECK:STDOUT:   %C.cpp_operator.type: type = fn_type @C.cpp_operator [concrete]
// CHECK:STDOUT:   %C.cpp_operator: %C.cpp_operator.type = struct_value () [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.c83: <bound method> = bound_method %int_42.20e, %Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5 [concrete]
// CHECK:STDOUT:   %bound_method.cb9: <bound method> = bound_method %int_42.20e, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_42.c68: %i32 = int_value 42 [concrete]
// CHECK:STDOUT:   %Destroy.Op.type: type = fn_type @Destroy.Op [concrete]
// CHECK:STDOUT:   %Destroy.Op: %Destroy.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %C.cpp_destructor.type: type = fn_type @C.cpp_destructor [concrete]
// CHECK:STDOUT:   %C.cpp_destructor: %C.cpp_destructor.type = struct_value () [concrete]
// CHECK:STDOUT:   %C.Op.type: type = fn_type @C.Op [concrete]
// CHECK:STDOUT:   %C.Op: %C.Op.type = struct_value () [concrete]
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
// CHECK:STDOUT:   %operator_Plus__carbon_thunk.decl: %operator_Plus__carbon_thunk.type = fn_decl @operator_Plus__carbon_thunk [concrete = constants.%operator_Plus__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %operator_Minus__carbon_thunk.decl: %operator_Minus__carbon_thunk.type = fn_decl @operator_Minus__carbon_thunk [concrete = constants.%operator_Minus__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %operator_Star__carbon_thunk.decl: %operator_Star__carbon_thunk.type = fn_decl @operator_Star__carbon_thunk [concrete = constants.%operator_Star__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %operator_Slash__carbon_thunk.decl: %operator_Slash__carbon_thunk.type = fn_decl @operator_Slash__carbon_thunk [concrete = constants.%operator_Slash__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %operator_Percent__carbon_thunk.decl: %operator_Percent__carbon_thunk.type = fn_decl @operator_Percent__carbon_thunk [concrete = constants.%operator_Percent__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %operator_Amp__carbon_thunk.decl: %operator_Amp__carbon_thunk.type = fn_decl @operator_Amp__carbon_thunk [concrete = constants.%operator_Amp__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %operator_Pipe__carbon_thunk.decl: %operator_Pipe__carbon_thunk.type = fn_decl @operator_Pipe__carbon_thunk [concrete = constants.%operator_Pipe__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %operator_Caret__carbon_thunk.decl: %operator_Caret__carbon_thunk.type = fn_decl @operator_Caret__carbon_thunk [concrete = constants.%operator_Caret__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %operator_LessLess__carbon_thunk.decl: %operator_LessLess__carbon_thunk.type = fn_decl @operator_LessLess__carbon_thunk [concrete = constants.%operator_LessLess__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.42d: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.4e6) = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3c2)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.74f = impl_witness_table (%Core.import_ref.42d), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %operator_GreaterGreater__carbon_thunk.decl: %operator_GreaterGreater__carbon_thunk.type = fn_decl @operator_GreaterGreater__carbon_thunk [concrete = constants.%operator_GreaterGreater__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %operator_PlusEqual__carbon_thunk.decl: %operator_PlusEqual__carbon_thunk.type = fn_decl @operator_PlusEqual__carbon_thunk [concrete = constants.%operator_PlusEqual__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %operator_MinusEqual__carbon_thunk.decl: %operator_MinusEqual__carbon_thunk.type = fn_decl @operator_MinusEqual__carbon_thunk [concrete = constants.%operator_MinusEqual__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %operator_StarEqual__carbon_thunk.decl: %operator_StarEqual__carbon_thunk.type = fn_decl @operator_StarEqual__carbon_thunk [concrete = constants.%operator_StarEqual__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %operator_SlashEqual__carbon_thunk.decl: %operator_SlashEqual__carbon_thunk.type = fn_decl @operator_SlashEqual__carbon_thunk [concrete = constants.%operator_SlashEqual__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %operator_PercentEqual__carbon_thunk.decl: %operator_PercentEqual__carbon_thunk.type = fn_decl @operator_PercentEqual__carbon_thunk [concrete = constants.%operator_PercentEqual__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %operator_AmpEqual__carbon_thunk.decl: %operator_AmpEqual__carbon_thunk.type = fn_decl @operator_AmpEqual__carbon_thunk [concrete = constants.%operator_AmpEqual__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %operator_PipeEqual__carbon_thunk.decl: %operator_PipeEqual__carbon_thunk.type = fn_decl @operator_PipeEqual__carbon_thunk [concrete = constants.%operator_PipeEqual__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %operator_CaretEqual__carbon_thunk.decl: %operator_CaretEqual__carbon_thunk.type = fn_decl @operator_CaretEqual__carbon_thunk [concrete = constants.%operator_CaretEqual__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %cpp_operator.decl.4206c9.19: %cpp_operator.type.1ea478.19 = fn_decl @cpp_operator.19 [concrete = constants.%cpp_operator.0a3797.19] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %cpp_operator.decl.4206c9.20: %cpp_operator.type.1ea478.20 = fn_decl @cpp_operator.20 [concrete = constants.%cpp_operator.0a3797.20] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %operator_EqualEqual__carbon_thunk.decl: %operator_EqualEqual__carbon_thunk.type = fn_decl @operator_EqualEqual__carbon_thunk [concrete = constants.%operator_EqualEqual__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %operator_ExclaimEqual__carbon_thunk.decl: %operator_ExclaimEqual__carbon_thunk.type = fn_decl @operator_ExclaimEqual__carbon_thunk [concrete = constants.%operator_ExclaimEqual__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %operator_Greater__carbon_thunk.decl: %operator_Greater__carbon_thunk.type = fn_decl @operator_Greater__carbon_thunk [concrete = constants.%operator_Greater__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %operator_Less__carbon_thunk.decl: %operator_Less__carbon_thunk.type = fn_decl @operator_Less__carbon_thunk [concrete = constants.%operator_Less__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %operator_GreaterEqual__carbon_thunk.decl: %operator_GreaterEqual__carbon_thunk.type = fn_decl @operator_GreaterEqual__carbon_thunk [concrete = constants.%operator_GreaterEqual__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %operator_LessEqual__carbon_thunk.decl: %operator_LessEqual__carbon_thunk.type = fn_decl @operator_LessEqual__carbon_thunk [concrete = constants.%operator_LessEqual__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.cpp_operator.decl: %C.cpp_operator.type = fn_decl @C.cpp_operator [concrete = constants.%C.cpp_operator] {
// CHECK:STDOUT:     %self.patt: %pattern_type.217 = ref_binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.217 = ref_param_pattern %self.patt [concrete]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %self.param: ref %C = ref_param call_param0
// CHECK:STDOUT:     %self: ref %C = ref_binding self, %self.param
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.cpp_destructor.decl: %C.cpp_destructor.type = fn_decl @C.cpp_destructor [concrete = constants.%C.cpp_destructor] {
// CHECK:STDOUT:     %self.patt: %pattern_type.217 = ref_binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.217 = ref_param_pattern %self.patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %C = ref_param call_param0
// CHECK:STDOUT:     %self: ref %C = ref_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c1.patt: %pattern_type.217 = ref_binding_pattern c1 [concrete]
// CHECK:STDOUT:     %c1.var_patt: %pattern_type.217 = var_pattern %c1.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c1.var: ref %C = var %c1.var_patt
// CHECK:STDOUT:   %Cpp.ref.loc12_19: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %C.ref.loc12_22: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %C.ref.loc12_24: %C.C.cpp_overload_set.type = name_ref C, imports.%C.C.cpp_overload_set.value [concrete = constants.%C.C.cpp_overload_set.value]
// CHECK:STDOUT:   %.loc12_3: ref %C = splice_block %c1.var {}
// CHECK:STDOUT:   %addr.loc12: %ptr.d9e = addr_of %.loc12_3
// CHECK:STDOUT:   %C__carbon_thunk.call.loc12: init %empty_tuple.type = call imports.%C__carbon_thunk.decl(%addr.loc12) [concrete = <error>]
// CHECK:STDOUT:   %.loc12_27: init %C to %.loc12_3 = mark_in_place_init %C__carbon_thunk.call.loc12 [concrete = <error>]
// CHECK:STDOUT:   assign %c1.var, %.loc12_27
// CHECK:STDOUT:   %.loc12_14: type = splice_block %C.ref.loc12_14 [concrete = constants.%C] {
// CHECK:STDOUT:     %Cpp.ref.loc12_11: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %C.ref.loc12_14: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c1: ref %C = ref_binding c1, %c1.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c2.patt: %pattern_type.217 = ref_binding_pattern c2 [concrete]
// CHECK:STDOUT:     %c2.var_patt: %pattern_type.217 = var_pattern %c2.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c2.var: ref %C = var %c2.var_patt
// CHECK:STDOUT:   %Cpp.ref.loc17_19: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %C.ref.loc17_22: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %C.ref.loc17_24: %C.C.cpp_overload_set.type = name_ref C, imports.%C.C.cpp_overload_set.value [concrete = constants.%C.C.cpp_overload_set.value]
// CHECK:STDOUT:   %.loc17_3: ref %C = splice_block %c2.var {}
// CHECK:STDOUT:   %addr.loc17: %ptr.d9e = addr_of %.loc17_3
// CHECK:STDOUT:   %C__carbon_thunk.call.loc17: init %empty_tuple.type = call imports.%C__carbon_thunk.decl(%addr.loc17) [concrete = <error>]
// CHECK:STDOUT:   %.loc17_27: init %C to %.loc17_3 = mark_in_place_init %C__carbon_thunk.call.loc17 [concrete = <error>]
// CHECK:STDOUT:   assign %c2.var, %.loc17_27
// CHECK:STDOUT:   %.loc17_14: type = splice_block %C.ref.loc17_14 [concrete = constants.%C] {
// CHECK:STDOUT:     %Cpp.ref.loc17_11: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %C.ref.loc17_14: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c2: ref %C = ref_binding c2, %c2.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %addition.patt: %pattern_type.217 = value_binding_pattern addition [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c1.ref.loc20: ref %C = name_ref c1, %c1
// CHECK:STDOUT:   %c2.ref.loc20: ref %C = name_ref c2, %c2
// CHECK:STDOUT:   %.loc20_35.1: ref %C = temporary_storage
// CHECK:STDOUT:   %.loc20_32.1: %C = acquire_value %c1.ref.loc20
// CHECK:STDOUT:   %.loc20_37.1: %C = acquire_value %c2.ref.loc20
// CHECK:STDOUT:   %.loc20_32.2: ref %C = value_as_ref %.loc20_32.1
// CHECK:STDOUT:   %addr.loc20_35.1: %ptr.d9e = addr_of %.loc20_32.2
// CHECK:STDOUT:   %.loc20_37.2: ref %C = value_as_ref %.loc20_37.1
// CHECK:STDOUT:   %addr.loc20_35.2: %ptr.d9e = addr_of %.loc20_37.2
// CHECK:STDOUT:   %addr.loc20_35.3: %ptr.d9e = addr_of %.loc20_35.1
// CHECK:STDOUT:   %operator_Plus__carbon_thunk.call: init %empty_tuple.type = call imports.%operator_Plus__carbon_thunk.decl(%addr.loc20_35.1, %addr.loc20_35.2, %addr.loc20_35.3)
// CHECK:STDOUT:   %.loc20_35.2: init %C to %.loc20_35.1 = mark_in_place_init %operator_Plus__carbon_thunk.call
// CHECK:STDOUT:   %.loc20_27: type = splice_block %C.ref.loc20 [concrete = constants.%C] {
// CHECK:STDOUT:     %Cpp.ref.loc20: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %C.ref.loc20: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc20_35.3: ref %C = temporary %.loc20_35.1, %.loc20_35.2
// CHECK:STDOUT:   %.loc20_35.4: %C = acquire_value %.loc20_35.3
// CHECK:STDOUT:   %addition: %C = value_binding addition, %.loc20_35.4
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %subtraction.patt: %pattern_type.217 = value_binding_pattern subtraction [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c1.ref.loc21: ref %C = name_ref c1, %c1
// CHECK:STDOUT:   %c2.ref.loc21: ref %C = name_ref c2, %c2
// CHECK:STDOUT:   %.loc21_38.1: ref %C = temporary_storage
// CHECK:STDOUT:   %.loc21_35.1: %C = acquire_value %c1.ref.loc21
// CHECK:STDOUT:   %.loc21_40.1: %C = acquire_value %c2.ref.loc21
// CHECK:STDOUT:   %.loc21_35.2: ref %C = value_as_ref %.loc21_35.1
// CHECK:STDOUT:   %addr.loc21_38.1: %ptr.d9e = addr_of %.loc21_35.2
// CHECK:STDOUT:   %.loc21_40.2: ref %C = value_as_ref %.loc21_40.1
// CHECK:STDOUT:   %addr.loc21_38.2: %ptr.d9e = addr_of %.loc21_40.2
// CHECK:STDOUT:   %addr.loc21_38.3: %ptr.d9e = addr_of %.loc21_38.1
// CHECK:STDOUT:   %operator_Minus__carbon_thunk.call: init %empty_tuple.type = call imports.%operator_Minus__carbon_thunk.decl(%addr.loc21_38.1, %addr.loc21_38.2, %addr.loc21_38.3)
// CHECK:STDOUT:   %.loc21_38.2: init %C to %.loc21_38.1 = mark_in_place_init %operator_Minus__carbon_thunk.call
// CHECK:STDOUT:   %.loc21_30: type = splice_block %C.ref.loc21 [concrete = constants.%C] {
// CHECK:STDOUT:     %Cpp.ref.loc21: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %C.ref.loc21: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc21_38.3: ref %C = temporary %.loc21_38.1, %.loc21_38.2
// CHECK:STDOUT:   %.loc21_38.4: %C = acquire_value %.loc21_38.3
// CHECK:STDOUT:   %subtraction: %C = value_binding subtraction, %.loc21_38.4
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %multiplication.patt: %pattern_type.217 = value_binding_pattern multiplication [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c1.ref.loc22: ref %C = name_ref c1, %c1
// CHECK:STDOUT:   %c2.ref.loc22: ref %C = name_ref c2, %c2
// CHECK:STDOUT:   %.loc22_41.1: ref %C = temporary_storage
// CHECK:STDOUT:   %.loc22_38.1: %C = acquire_value %c1.ref.loc22
// CHECK:STDOUT:   %.loc22_43.1: %C = acquire_value %c2.ref.loc22
// CHECK:STDOUT:   %.loc22_38.2: ref %C = value_as_ref %.loc22_38.1
// CHECK:STDOUT:   %addr.loc22_41.1: %ptr.d9e = addr_of %.loc22_38.2
// CHECK:STDOUT:   %.loc22_43.2: ref %C = value_as_ref %.loc22_43.1
// CHECK:STDOUT:   %addr.loc22_41.2: %ptr.d9e = addr_of %.loc22_43.2
// CHECK:STDOUT:   %addr.loc22_41.3: %ptr.d9e = addr_of %.loc22_41.1
// CHECK:STDOUT:   %operator_Star__carbon_thunk.call: init %empty_tuple.type = call imports.%operator_Star__carbon_thunk.decl(%addr.loc22_41.1, %addr.loc22_41.2, %addr.loc22_41.3)
// CHECK:STDOUT:   %.loc22_41.2: init %C to %.loc22_41.1 = mark_in_place_init %operator_Star__carbon_thunk.call
// CHECK:STDOUT:   %.loc22_33: type = splice_block %C.ref.loc22 [concrete = constants.%C] {
// CHECK:STDOUT:     %Cpp.ref.loc22: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %C.ref.loc22: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc22_41.3: ref %C = temporary %.loc22_41.1, %.loc22_41.2
// CHECK:STDOUT:   %.loc22_41.4: %C = acquire_value %.loc22_41.3
// CHECK:STDOUT:   %multiplication: %C = value_binding multiplication, %.loc22_41.4
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %division.patt: %pattern_type.217 = value_binding_pattern division [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c1.ref.loc23: ref %C = name_ref c1, %c1
// CHECK:STDOUT:   %c2.ref.loc23: ref %C = name_ref c2, %c2
// CHECK:STDOUT:   %.loc23_35.1: ref %C = temporary_storage
// CHECK:STDOUT:   %.loc23_32.1: %C = acquire_value %c1.ref.loc23
// CHECK:STDOUT:   %.loc23_37.1: %C = acquire_value %c2.ref.loc23
// CHECK:STDOUT:   %.loc23_32.2: ref %C = value_as_ref %.loc23_32.1
// CHECK:STDOUT:   %addr.loc23_35.1: %ptr.d9e = addr_of %.loc23_32.2
// CHECK:STDOUT:   %.loc23_37.2: ref %C = value_as_ref %.loc23_37.1
// CHECK:STDOUT:   %addr.loc23_35.2: %ptr.d9e = addr_of %.loc23_37.2
// CHECK:STDOUT:   %addr.loc23_35.3: %ptr.d9e = addr_of %.loc23_35.1
// CHECK:STDOUT:   %operator_Slash__carbon_thunk.call: init %empty_tuple.type = call imports.%operator_Slash__carbon_thunk.decl(%addr.loc23_35.1, %addr.loc23_35.2, %addr.loc23_35.3)
// CHECK:STDOUT:   %.loc23_35.2: init %C to %.loc23_35.1 = mark_in_place_init %operator_Slash__carbon_thunk.call
// CHECK:STDOUT:   %.loc23_27: type = splice_block %C.ref.loc23 [concrete = constants.%C] {
// CHECK:STDOUT:     %Cpp.ref.loc23: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %C.ref.loc23: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc23_35.3: ref %C = temporary %.loc23_35.1, %.loc23_35.2
// CHECK:STDOUT:   %.loc23_35.4: %C = acquire_value %.loc23_35.3
// CHECK:STDOUT:   %division: %C = value_binding division, %.loc23_35.4
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %modulo.patt: %pattern_type.217 = value_binding_pattern modulo [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c1.ref.loc24: ref %C = name_ref c1, %c1
// CHECK:STDOUT:   %c2.ref.loc24: ref %C = name_ref c2, %c2
// CHECK:STDOUT:   %.loc24_33.1: ref %C = temporary_storage
// CHECK:STDOUT:   %.loc24_30.1: %C = acquire_value %c1.ref.loc24
// CHECK:STDOUT:   %.loc24_35.1: %C = acquire_value %c2.ref.loc24
// CHECK:STDOUT:   %.loc24_30.2: ref %C = value_as_ref %.loc24_30.1
// CHECK:STDOUT:   %addr.loc24_33.1: %ptr.d9e = addr_of %.loc24_30.2
// CHECK:STDOUT:   %.loc24_35.2: ref %C = value_as_ref %.loc24_35.1
// CHECK:STDOUT:   %addr.loc24_33.2: %ptr.d9e = addr_of %.loc24_35.2
// CHECK:STDOUT:   %addr.loc24_33.3: %ptr.d9e = addr_of %.loc24_33.1
// CHECK:STDOUT:   %operator_Percent__carbon_thunk.call: init %empty_tuple.type = call imports.%operator_Percent__carbon_thunk.decl(%addr.loc24_33.1, %addr.loc24_33.2, %addr.loc24_33.3)
// CHECK:STDOUT:   %.loc24_33.2: init %C to %.loc24_33.1 = mark_in_place_init %operator_Percent__carbon_thunk.call
// CHECK:STDOUT:   %.loc24_25: type = splice_block %C.ref.loc24 [concrete = constants.%C] {
// CHECK:STDOUT:     %Cpp.ref.loc24: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %C.ref.loc24: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc24_33.3: ref %C = temporary %.loc24_33.1, %.loc24_33.2
// CHECK:STDOUT:   %.loc24_33.4: %C = acquire_value %.loc24_33.3
// CHECK:STDOUT:   %modulo: %C = value_binding modulo, %.loc24_33.4
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %bitwise_and.patt: %pattern_type.217 = value_binding_pattern bitwise_and [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c1.ref.loc27: ref %C = name_ref c1, %c1
// CHECK:STDOUT:   %c2.ref.loc27: ref %C = name_ref c2, %c2
// CHECK:STDOUT:   %.loc27_38.1: ref %C = temporary_storage
// CHECK:STDOUT:   %.loc27_35.1: %C = acquire_value %c1.ref.loc27
// CHECK:STDOUT:   %.loc27_40.1: %C = acquire_value %c2.ref.loc27
// CHECK:STDOUT:   %.loc27_35.2: ref %C = value_as_ref %.loc27_35.1
// CHECK:STDOUT:   %addr.loc27_38.1: %ptr.d9e = addr_of %.loc27_35.2
// CHECK:STDOUT:   %.loc27_40.2: ref %C = value_as_ref %.loc27_40.1
// CHECK:STDOUT:   %addr.loc27_38.2: %ptr.d9e = addr_of %.loc27_40.2
// CHECK:STDOUT:   %addr.loc27_38.3: %ptr.d9e = addr_of %.loc27_38.1
// CHECK:STDOUT:   %operator_Amp__carbon_thunk.call: init %empty_tuple.type = call imports.%operator_Amp__carbon_thunk.decl(%addr.loc27_38.1, %addr.loc27_38.2, %addr.loc27_38.3)
// CHECK:STDOUT:   %.loc27_38.2: init %C to %.loc27_38.1 = mark_in_place_init %operator_Amp__carbon_thunk.call
// CHECK:STDOUT:   %.loc27_30: type = splice_block %C.ref.loc27 [concrete = constants.%C] {
// CHECK:STDOUT:     %Cpp.ref.loc27: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %C.ref.loc27: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc27_38.3: ref %C = temporary %.loc27_38.1, %.loc27_38.2
// CHECK:STDOUT:   %.loc27_38.4: %C = acquire_value %.loc27_38.3
// CHECK:STDOUT:   %bitwise_and: %C = value_binding bitwise_and, %.loc27_38.4
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %bitwise_or.patt: %pattern_type.217 = value_binding_pattern bitwise_or [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c1.ref.loc28: ref %C = name_ref c1, %c1
// CHECK:STDOUT:   %c2.ref.loc28: ref %C = name_ref c2, %c2
// CHECK:STDOUT:   %.loc28_37.1: ref %C = temporary_storage
// CHECK:STDOUT:   %.loc28_34.1: %C = acquire_value %c1.ref.loc28
// CHECK:STDOUT:   %.loc28_39.1: %C = acquire_value %c2.ref.loc28
// CHECK:STDOUT:   %.loc28_34.2: ref %C = value_as_ref %.loc28_34.1
// CHECK:STDOUT:   %addr.loc28_37.1: %ptr.d9e = addr_of %.loc28_34.2
// CHECK:STDOUT:   %.loc28_39.2: ref %C = value_as_ref %.loc28_39.1
// CHECK:STDOUT:   %addr.loc28_37.2: %ptr.d9e = addr_of %.loc28_39.2
// CHECK:STDOUT:   %addr.loc28_37.3: %ptr.d9e = addr_of %.loc28_37.1
// CHECK:STDOUT:   %operator_Pipe__carbon_thunk.call: init %empty_tuple.type = call imports.%operator_Pipe__carbon_thunk.decl(%addr.loc28_37.1, %addr.loc28_37.2, %addr.loc28_37.3)
// CHECK:STDOUT:   %.loc28_37.2: init %C to %.loc28_37.1 = mark_in_place_init %operator_Pipe__carbon_thunk.call
// CHECK:STDOUT:   %.loc28_29: type = splice_block %C.ref.loc28 [concrete = constants.%C] {
// CHECK:STDOUT:     %Cpp.ref.loc28: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %C.ref.loc28: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc28_37.3: ref %C = temporary %.loc28_37.1, %.loc28_37.2
// CHECK:STDOUT:   %.loc28_37.4: %C = acquire_value %.loc28_37.3
// CHECK:STDOUT:   %bitwise_or: %C = value_binding bitwise_or, %.loc28_37.4
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %bitwise_xor.patt: %pattern_type.217 = value_binding_pattern bitwise_xor [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c1.ref.loc29: ref %C = name_ref c1, %c1
// CHECK:STDOUT:   %c2.ref.loc29: ref %C = name_ref c2, %c2
// CHECK:STDOUT:   %.loc29_38.1: ref %C = temporary_storage
// CHECK:STDOUT:   %.loc29_35.1: %C = acquire_value %c1.ref.loc29
// CHECK:STDOUT:   %.loc29_40.1: %C = acquire_value %c2.ref.loc29
// CHECK:STDOUT:   %.loc29_35.2: ref %C = value_as_ref %.loc29_35.1
// CHECK:STDOUT:   %addr.loc29_38.1: %ptr.d9e = addr_of %.loc29_35.2
// CHECK:STDOUT:   %.loc29_40.2: ref %C = value_as_ref %.loc29_40.1
// CHECK:STDOUT:   %addr.loc29_38.2: %ptr.d9e = addr_of %.loc29_40.2
// CHECK:STDOUT:   %addr.loc29_38.3: %ptr.d9e = addr_of %.loc29_38.1
// CHECK:STDOUT:   %operator_Caret__carbon_thunk.call: init %empty_tuple.type = call imports.%operator_Caret__carbon_thunk.decl(%addr.loc29_38.1, %addr.loc29_38.2, %addr.loc29_38.3)
// CHECK:STDOUT:   %.loc29_38.2: init %C to %.loc29_38.1 = mark_in_place_init %operator_Caret__carbon_thunk.call
// CHECK:STDOUT:   %.loc29_30: type = splice_block %C.ref.loc29 [concrete = constants.%C] {
// CHECK:STDOUT:     %Cpp.ref.loc29: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %C.ref.loc29: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc29_38.3: ref %C = temporary %.loc29_38.1, %.loc29_38.2
// CHECK:STDOUT:   %.loc29_38.4: %C = acquire_value %.loc29_38.3
// CHECK:STDOUT:   %bitwise_xor: %C = value_binding bitwise_xor, %.loc29_38.4
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %left_shift.patt: %pattern_type.217 = value_binding_pattern left_shift [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c1.ref.loc30: ref %C = name_ref c1, %c1
// CHECK:STDOUT:   %int_3.loc30: Core.IntLiteral = int_value 3 [concrete = constants.%int_3.1ba]
// CHECK:STDOUT:   %.loc30_37.1: ref %C = temporary_storage
// CHECK:STDOUT:   %.loc30_34.1: %C = acquire_value %c1.ref.loc30
// CHECK:STDOUT:   %impl.elem0.loc30: %.545 = impl_witness_access constants.%ImplicitAs.impl_witness.6bc, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5]
// CHECK:STDOUT:   %bound_method.loc30_40.1: <bound method> = bound_method %int_3.loc30, %impl.elem0.loc30 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.061]
// CHECK:STDOUT:   %specific_fn.loc30: <specific function> = specific_function %impl.elem0.loc30, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc30_40.2: <bound method> = bound_method %int_3.loc30, %specific_fn.loc30 [concrete = constants.%bound_method.fa7]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc30: init %i32 = call %bound_method.loc30_40.2(%int_3.loc30) [concrete = constants.%int_3.822]
// CHECK:STDOUT:   %.loc30_40.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc30 [concrete = constants.%int_3.822]
// CHECK:STDOUT:   %.loc30_40.2: %i32 = converted %int_3.loc30, %.loc30_40.1 [concrete = constants.%int_3.822]
// CHECK:STDOUT:   %.loc30_34.2: ref %C = value_as_ref %.loc30_34.1
// CHECK:STDOUT:   %addr.loc30_37.1: %ptr.d9e = addr_of %.loc30_34.2
// CHECK:STDOUT:   %addr.loc30_37.2: %ptr.d9e = addr_of %.loc30_37.1
// CHECK:STDOUT:   %operator_LessLess__carbon_thunk.call: init %empty_tuple.type = call imports.%operator_LessLess__carbon_thunk.decl(%addr.loc30_37.1, %.loc30_40.2, %addr.loc30_37.2)
// CHECK:STDOUT:   %.loc30_37.2: init %C to %.loc30_37.1 = mark_in_place_init %operator_LessLess__carbon_thunk.call
// CHECK:STDOUT:   %.loc30_29: type = splice_block %C.ref.loc30 [concrete = constants.%C] {
// CHECK:STDOUT:     %Cpp.ref.loc30: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %C.ref.loc30: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc30_37.3: ref %C = temporary %.loc30_37.1, %.loc30_37.2
// CHECK:STDOUT:   %.loc30_37.4: %C = acquire_value %.loc30_37.3
// CHECK:STDOUT:   %left_shift: %C = value_binding left_shift, %.loc30_37.4
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %right_shift.patt: %pattern_type.217 = value_binding_pattern right_shift [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c1.ref.loc31: ref %C = name_ref c1, %c1
// CHECK:STDOUT:   %int_5.loc31: Core.IntLiteral = int_value 5 [concrete = constants.%int_5.64b]
// CHECK:STDOUT:   %.loc31_38.1: ref %C = temporary_storage
// CHECK:STDOUT:   %.loc31_35.1: %C = acquire_value %c1.ref.loc31
// CHECK:STDOUT:   %impl.elem0.loc31: %.545 = impl_witness_access constants.%ImplicitAs.impl_witness.6bc, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5]
// CHECK:STDOUT:   %bound_method.loc31_41.1: <bound method> = bound_method %int_5.loc31, %impl.elem0.loc31 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.005]
// CHECK:STDOUT:   %specific_fn.loc31: <specific function> = specific_function %impl.elem0.loc31, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc31_41.2: <bound method> = bound_method %int_5.loc31, %specific_fn.loc31 [concrete = constants.%bound_method.e9d]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc31: init %i32 = call %bound_method.loc31_41.2(%int_5.loc31) [concrete = constants.%int_5.0f6]
// CHECK:STDOUT:   %.loc31_41.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc31 [concrete = constants.%int_5.0f6]
// CHECK:STDOUT:   %.loc31_41.2: %i32 = converted %int_5.loc31, %.loc31_41.1 [concrete = constants.%int_5.0f6]
// CHECK:STDOUT:   %.loc31_35.2: ref %C = value_as_ref %.loc31_35.1
// CHECK:STDOUT:   %addr.loc31_38.1: %ptr.d9e = addr_of %.loc31_35.2
// CHECK:STDOUT:   %addr.loc31_38.2: %ptr.d9e = addr_of %.loc31_38.1
// CHECK:STDOUT:   %operator_GreaterGreater__carbon_thunk.call: init %empty_tuple.type = call imports.%operator_GreaterGreater__carbon_thunk.decl(%addr.loc31_38.1, %.loc31_41.2, %addr.loc31_38.2)
// CHECK:STDOUT:   %.loc31_38.2: init %C to %.loc31_38.1 = mark_in_place_init %operator_GreaterGreater__carbon_thunk.call
// CHECK:STDOUT:   %.loc31_30: type = splice_block %C.ref.loc31 [concrete = constants.%C] {
// CHECK:STDOUT:     %Cpp.ref.loc31: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %C.ref.loc31: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc31_38.3: ref %C = temporary %.loc31_38.1, %.loc31_38.2
// CHECK:STDOUT:   %.loc31_38.4: %C = acquire_value %.loc31_38.3
// CHECK:STDOUT:   %right_shift: %C = value_binding right_shift, %.loc31_38.4
// CHECK:STDOUT:   %c1.ref.loc34: ref %C = name_ref c1, %c1
// CHECK:STDOUT:   %c2.ref.loc34: ref %C = name_ref c2, %c2
// CHECK:STDOUT:   %.loc34_9.1: %C = acquire_value %c2.ref.loc34
// CHECK:STDOUT:   %.loc34_9.2: ref %C = value_as_ref %.loc34_9.1
// CHECK:STDOUT:   %addr.loc34: %ptr.d9e = addr_of %.loc34_9.2
// CHECK:STDOUT:   %operator_PlusEqual__carbon_thunk.call: ref %C = call imports.%operator_PlusEqual__carbon_thunk.decl(%c1.ref.loc34, %addr.loc34)
// CHECK:STDOUT:   %c1.ref.loc35: ref %C = name_ref c1, %c1
// CHECK:STDOUT:   %c2.ref.loc35: ref %C = name_ref c2, %c2
// CHECK:STDOUT:   %.loc35_9.1: %C = acquire_value %c2.ref.loc35
// CHECK:STDOUT:   %.loc35_9.2: ref %C = value_as_ref %.loc35_9.1
// CHECK:STDOUT:   %addr.loc35: %ptr.d9e = addr_of %.loc35_9.2
// CHECK:STDOUT:   %operator_MinusEqual__carbon_thunk.call: ref %C = call imports.%operator_MinusEqual__carbon_thunk.decl(%c1.ref.loc35, %addr.loc35)
// CHECK:STDOUT:   %c1.ref.loc36: ref %C = name_ref c1, %c1
// CHECK:STDOUT:   %c2.ref.loc36: ref %C = name_ref c2, %c2
// CHECK:STDOUT:   %.loc36_9.1: %C = acquire_value %c2.ref.loc36
// CHECK:STDOUT:   %.loc36_9.2: ref %C = value_as_ref %.loc36_9.1
// CHECK:STDOUT:   %addr.loc36: %ptr.d9e = addr_of %.loc36_9.2
// CHECK:STDOUT:   %operator_StarEqual__carbon_thunk.call: ref %C = call imports.%operator_StarEqual__carbon_thunk.decl(%c1.ref.loc36, %addr.loc36)
// CHECK:STDOUT:   %c1.ref.loc37: ref %C = name_ref c1, %c1
// CHECK:STDOUT:   %c2.ref.loc37: ref %C = name_ref c2, %c2
// CHECK:STDOUT:   %.loc37_9.1: %C = acquire_value %c2.ref.loc37
// CHECK:STDOUT:   %.loc37_9.2: ref %C = value_as_ref %.loc37_9.1
// CHECK:STDOUT:   %addr.loc37: %ptr.d9e = addr_of %.loc37_9.2
// CHECK:STDOUT:   %operator_SlashEqual__carbon_thunk.call: ref %C = call imports.%operator_SlashEqual__carbon_thunk.decl(%c1.ref.loc37, %addr.loc37)
// CHECK:STDOUT:   %c1.ref.loc38: ref %C = name_ref c1, %c1
// CHECK:STDOUT:   %c2.ref.loc38: ref %C = name_ref c2, %c2
// CHECK:STDOUT:   %.loc38_9.1: %C = acquire_value %c2.ref.loc38
// CHECK:STDOUT:   %.loc38_9.2: ref %C = value_as_ref %.loc38_9.1
// CHECK:STDOUT:   %addr.loc38: %ptr.d9e = addr_of %.loc38_9.2
// CHECK:STDOUT:   %operator_PercentEqual__carbon_thunk.call: ref %C = call imports.%operator_PercentEqual__carbon_thunk.decl(%c1.ref.loc38, %addr.loc38)
// CHECK:STDOUT:   %c1.ref.loc41: ref %C = name_ref c1, %c1
// CHECK:STDOUT:   %c2.ref.loc41: ref %C = name_ref c2, %c2
// CHECK:STDOUT:   %.loc41_9.1: %C = acquire_value %c2.ref.loc41
// CHECK:STDOUT:   %.loc41_9.2: ref %C = value_as_ref %.loc41_9.1
// CHECK:STDOUT:   %addr.loc41: %ptr.d9e = addr_of %.loc41_9.2
// CHECK:STDOUT:   %operator_AmpEqual__carbon_thunk.call: ref %C = call imports.%operator_AmpEqual__carbon_thunk.decl(%c1.ref.loc41, %addr.loc41)
// CHECK:STDOUT:   %c1.ref.loc42: ref %C = name_ref c1, %c1
// CHECK:STDOUT:   %c2.ref.loc42: ref %C = name_ref c2, %c2
// CHECK:STDOUT:   %.loc42_9.1: %C = acquire_value %c2.ref.loc42
// CHECK:STDOUT:   %.loc42_9.2: ref %C = value_as_ref %.loc42_9.1
// CHECK:STDOUT:   %addr.loc42: %ptr.d9e = addr_of %.loc42_9.2
// CHECK:STDOUT:   %operator_PipeEqual__carbon_thunk.call: ref %C = call imports.%operator_PipeEqual__carbon_thunk.decl(%c1.ref.loc42, %addr.loc42)
// CHECK:STDOUT:   %c1.ref.loc43: ref %C = name_ref c1, %c1
// CHECK:STDOUT:   %c2.ref.loc43: ref %C = name_ref c2, %c2
// CHECK:STDOUT:   %.loc43_9.1: %C = acquire_value %c2.ref.loc43
// CHECK:STDOUT:   %.loc43_9.2: ref %C = value_as_ref %.loc43_9.1
// CHECK:STDOUT:   %addr.loc43: %ptr.d9e = addr_of %.loc43_9.2
// CHECK:STDOUT:   %operator_CaretEqual__carbon_thunk.call: ref %C = call imports.%operator_CaretEqual__carbon_thunk.decl(%c1.ref.loc43, %addr.loc43)
// CHECK:STDOUT:   %c1.ref.loc44: ref %C = name_ref c1, %c1
// CHECK:STDOUT:   %int_3.loc44: Core.IntLiteral = int_value 3 [concrete = constants.%int_3.1ba]
// CHECK:STDOUT:   %impl.elem0.loc44: %.545 = impl_witness_access constants.%ImplicitAs.impl_witness.6bc, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5]
// CHECK:STDOUT:   %bound_method.loc44_10.1: <bound method> = bound_method %int_3.loc44, %impl.elem0.loc44 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.061]
// CHECK:STDOUT:   %specific_fn.loc44: <specific function> = specific_function %impl.elem0.loc44, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc44_10.2: <bound method> = bound_method %int_3.loc44, %specific_fn.loc44 [concrete = constants.%bound_method.fa7]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc44: init %i32 = call %bound_method.loc44_10.2(%int_3.loc44) [concrete = constants.%int_3.822]
// CHECK:STDOUT:   %.loc44_10.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc44 [concrete = constants.%int_3.822]
// CHECK:STDOUT:   %.loc44_10.2: %i32 = converted %int_3.loc44, %.loc44_10.1 [concrete = constants.%int_3.822]
// CHECK:STDOUT:   %cpp_operator.call.loc44: ref %C = call imports.%cpp_operator.decl.4206c9.19(%c1.ref.loc44, %.loc44_10.2)
// CHECK:STDOUT:   %c1.ref.loc45: ref %C = name_ref c1, %c1
// CHECK:STDOUT:   %int_5.loc45: Core.IntLiteral = int_value 5 [concrete = constants.%int_5.64b]
// CHECK:STDOUT:   %impl.elem0.loc45: %.545 = impl_witness_access constants.%ImplicitAs.impl_witness.6bc, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5]
// CHECK:STDOUT:   %bound_method.loc45_10.1: <bound method> = bound_method %int_5.loc45, %impl.elem0.loc45 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.005]
// CHECK:STDOUT:   %specific_fn.loc45: <specific function> = specific_function %impl.elem0.loc45, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc45_10.2: <bound method> = bound_method %int_5.loc45, %specific_fn.loc45 [concrete = constants.%bound_method.e9d]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc45: init %i32 = call %bound_method.loc45_10.2(%int_5.loc45) [concrete = constants.%int_5.0f6]
// CHECK:STDOUT:   %.loc45_10.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc45 [concrete = constants.%int_5.0f6]
// CHECK:STDOUT:   %.loc45_10.2: %i32 = converted %int_5.loc45, %.loc45_10.1 [concrete = constants.%int_5.0f6]
// CHECK:STDOUT:   %cpp_operator.call.loc45: ref %C = call imports.%cpp_operator.decl.4206c9.20(%c1.ref.loc45, %.loc45_10.2)
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %equal.patt: %pattern_type.831 = value_binding_pattern equal [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c1.ref.loc48: ref %C = name_ref c1, %c1
// CHECK:STDOUT:   %c2.ref.loc48: ref %C = name_ref c2, %c2
// CHECK:STDOUT:   %.loc48_28.1: %C = acquire_value %c1.ref.loc48
// CHECK:STDOUT:   %.loc48_34.1: %C = acquire_value %c2.ref.loc48
// CHECK:STDOUT:   %.loc48_28.2: ref %C = value_as_ref %.loc48_28.1
// CHECK:STDOUT:   %addr.loc48_31.1: %ptr.d9e = addr_of %.loc48_28.2
// CHECK:STDOUT:   %.loc48_34.2: ref %C = value_as_ref %.loc48_34.1
// CHECK:STDOUT:   %addr.loc48_31.2: %ptr.d9e = addr_of %.loc48_34.2
// CHECK:STDOUT:   %.loc48_31.1: ref bool = temporary_storage
// CHECK:STDOUT:   %addr.loc48_31.3: %ptr.bb2 = addr_of %.loc48_31.1
// CHECK:STDOUT:   %operator_EqualEqual__carbon_thunk.call: init %empty_tuple.type = call imports.%operator_EqualEqual__carbon_thunk.decl(%addr.loc48_31.1, %addr.loc48_31.2, %addr.loc48_31.3)
// CHECK:STDOUT:   %.loc48_31.2: init bool to %.loc48_31.1 = mark_in_place_init %operator_EqualEqual__carbon_thunk.call
// CHECK:STDOUT:   %.loc48_21: type = type_literal bool [concrete = bool]
// CHECK:STDOUT:   %.loc48_31.3: ref bool = temporary %.loc48_31.1, %.loc48_31.2
// CHECK:STDOUT:   %.loc48_31.4: bool = acquire_value %.loc48_31.3
// CHECK:STDOUT:   %equal: bool = value_binding equal, %.loc48_31.4
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %not_equal.patt: %pattern_type.831 = value_binding_pattern not_equal [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c1.ref.loc49: ref %C = name_ref c1, %c1
// CHECK:STDOUT:   %c2.ref.loc49: ref %C = name_ref c2, %c2
// CHECK:STDOUT:   %.loc49_32.1: %C = acquire_value %c1.ref.loc49
// CHECK:STDOUT:   %.loc49_38.1: %C = acquire_value %c2.ref.loc49
// CHECK:STDOUT:   %.loc49_32.2: ref %C = value_as_ref %.loc49_32.1
// CHECK:STDOUT:   %addr.loc49_35.1: %ptr.d9e = addr_of %.loc49_32.2
// CHECK:STDOUT:   %.loc49_38.2: ref %C = value_as_ref %.loc49_38.1
// CHECK:STDOUT:   %addr.loc49_35.2: %ptr.d9e = addr_of %.loc49_38.2
// CHECK:STDOUT:   %.loc49_35.1: ref bool = temporary_storage
// CHECK:STDOUT:   %addr.loc49_35.3: %ptr.bb2 = addr_of %.loc49_35.1
// CHECK:STDOUT:   %operator_ExclaimEqual__carbon_thunk.call: init %empty_tuple.type = call imports.%operator_ExclaimEqual__carbon_thunk.decl(%addr.loc49_35.1, %addr.loc49_35.2, %addr.loc49_35.3)
// CHECK:STDOUT:   %.loc49_35.2: init bool to %.loc49_35.1 = mark_in_place_init %operator_ExclaimEqual__carbon_thunk.call
// CHECK:STDOUT:   %.loc49_25: type = type_literal bool [concrete = bool]
// CHECK:STDOUT:   %.loc49_35.3: ref bool = temporary %.loc49_35.1, %.loc49_35.2
// CHECK:STDOUT:   %.loc49_35.4: bool = acquire_value %.loc49_35.3
// CHECK:STDOUT:   %not_equal: bool = value_binding not_equal, %.loc49_35.4
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %greater_than.patt: %pattern_type.831 = value_binding_pattern greater_than [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c1.ref.loc50: ref %C = name_ref c1, %c1
// CHECK:STDOUT:   %c2.ref.loc50: ref %C = name_ref c2, %c2
// CHECK:STDOUT:   %.loc50_35.1: %C = acquire_value %c1.ref.loc50
// CHECK:STDOUT:   %.loc50_40.1: %C = acquire_value %c2.ref.loc50
// CHECK:STDOUT:   %.loc50_35.2: ref %C = value_as_ref %.loc50_35.1
// CHECK:STDOUT:   %addr.loc50_38.1: %ptr.d9e = addr_of %.loc50_35.2
// CHECK:STDOUT:   %.loc50_40.2: ref %C = value_as_ref %.loc50_40.1
// CHECK:STDOUT:   %addr.loc50_38.2: %ptr.d9e = addr_of %.loc50_40.2
// CHECK:STDOUT:   %.loc50_38.1: ref bool = temporary_storage
// CHECK:STDOUT:   %addr.loc50_38.3: %ptr.bb2 = addr_of %.loc50_38.1
// CHECK:STDOUT:   %operator_Greater__carbon_thunk.call: init %empty_tuple.type = call imports.%operator_Greater__carbon_thunk.decl(%addr.loc50_38.1, %addr.loc50_38.2, %addr.loc50_38.3)
// CHECK:STDOUT:   %.loc50_38.2: init bool to %.loc50_38.1 = mark_in_place_init %operator_Greater__carbon_thunk.call
// CHECK:STDOUT:   %.loc50_28: type = type_literal bool [concrete = bool]
// CHECK:STDOUT:   %.loc50_38.3: ref bool = temporary %.loc50_38.1, %.loc50_38.2
// CHECK:STDOUT:   %.loc50_38.4: bool = acquire_value %.loc50_38.3
// CHECK:STDOUT:   %greater_than: bool = value_binding greater_than, %.loc50_38.4
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %less_than.patt: %pattern_type.831 = value_binding_pattern less_than [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c1.ref.loc51: ref %C = name_ref c1, %c1
// CHECK:STDOUT:   %c2.ref.loc51: ref %C = name_ref c2, %c2
// CHECK:STDOUT:   %.loc51_32.1: %C = acquire_value %c1.ref.loc51
// CHECK:STDOUT:   %.loc51_37.1: %C = acquire_value %c2.ref.loc51
// CHECK:STDOUT:   %.loc51_32.2: ref %C = value_as_ref %.loc51_32.1
// CHECK:STDOUT:   %addr.loc51_35.1: %ptr.d9e = addr_of %.loc51_32.2
// CHECK:STDOUT:   %.loc51_37.2: ref %C = value_as_ref %.loc51_37.1
// CHECK:STDOUT:   %addr.loc51_35.2: %ptr.d9e = addr_of %.loc51_37.2
// CHECK:STDOUT:   %.loc51_35.1: ref bool = temporary_storage
// CHECK:STDOUT:   %addr.loc51_35.3: %ptr.bb2 = addr_of %.loc51_35.1
// CHECK:STDOUT:   %operator_Less__carbon_thunk.call: init %empty_tuple.type = call imports.%operator_Less__carbon_thunk.decl(%addr.loc51_35.1, %addr.loc51_35.2, %addr.loc51_35.3)
// CHECK:STDOUT:   %.loc51_35.2: init bool to %.loc51_35.1 = mark_in_place_init %operator_Less__carbon_thunk.call
// CHECK:STDOUT:   %.loc51_25: type = type_literal bool [concrete = bool]
// CHECK:STDOUT:   %.loc51_35.3: ref bool = temporary %.loc51_35.1, %.loc51_35.2
// CHECK:STDOUT:   %.loc51_35.4: bool = acquire_value %.loc51_35.3
// CHECK:STDOUT:   %less_than: bool = value_binding less_than, %.loc51_35.4
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %greater_than_or_equal.patt: %pattern_type.831 = value_binding_pattern greater_than_or_equal [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c1.ref.loc52: ref %C = name_ref c1, %c1
// CHECK:STDOUT:   %c2.ref.loc52: ref %C = name_ref c2, %c2
// CHECK:STDOUT:   %.loc52_44.1: %C = acquire_value %c1.ref.loc52
// CHECK:STDOUT:   %.loc52_50.1: %C = acquire_value %c2.ref.loc52
// CHECK:STDOUT:   %.loc52_44.2: ref %C = value_as_ref %.loc52_44.1
// CHECK:STDOUT:   %addr.loc52_47.1: %ptr.d9e = addr_of %.loc52_44.2
// CHECK:STDOUT:   %.loc52_50.2: ref %C = value_as_ref %.loc52_50.1
// CHECK:STDOUT:   %addr.loc52_47.2: %ptr.d9e = addr_of %.loc52_50.2
// CHECK:STDOUT:   %.loc52_47.1: ref bool = temporary_storage
// CHECK:STDOUT:   %addr.loc52_47.3: %ptr.bb2 = addr_of %.loc52_47.1
// CHECK:STDOUT:   %operator_GreaterEqual__carbon_thunk.call: init %empty_tuple.type = call imports.%operator_GreaterEqual__carbon_thunk.decl(%addr.loc52_47.1, %addr.loc52_47.2, %addr.loc52_47.3)
// CHECK:STDOUT:   %.loc52_47.2: init bool to %.loc52_47.1 = mark_in_place_init %operator_GreaterEqual__carbon_thunk.call
// CHECK:STDOUT:   %.loc52_37: type = type_literal bool [concrete = bool]
// CHECK:STDOUT:   %.loc52_47.3: ref bool = temporary %.loc52_47.1, %.loc52_47.2
// CHECK:STDOUT:   %.loc52_47.4: bool = acquire_value %.loc52_47.3
// CHECK:STDOUT:   %greater_than_or_equal: bool = value_binding greater_than_or_equal, %.loc52_47.4
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %less_than_or_equal.patt: %pattern_type.831 = value_binding_pattern less_than_or_equal [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c1.ref.loc53: ref %C = name_ref c1, %c1
// CHECK:STDOUT:   %c2.ref.loc53: ref %C = name_ref c2, %c2
// CHECK:STDOUT:   %.loc53_41.1: %C = acquire_value %c1.ref.loc53
// CHECK:STDOUT:   %.loc53_47.1: %C = acquire_value %c2.ref.loc53
// CHECK:STDOUT:   %.loc53_41.2: ref %C = value_as_ref %.loc53_41.1
// CHECK:STDOUT:   %addr.loc53_44.1: %ptr.d9e = addr_of %.loc53_41.2
// CHECK:STDOUT:   %.loc53_47.2: ref %C = value_as_ref %.loc53_47.1
// CHECK:STDOUT:   %addr.loc53_44.2: %ptr.d9e = addr_of %.loc53_47.2
// CHECK:STDOUT:   %.loc53_44.1: ref bool = temporary_storage
// CHECK:STDOUT:   %addr.loc53_44.3: %ptr.bb2 = addr_of %.loc53_44.1
// CHECK:STDOUT:   %operator_LessEqual__carbon_thunk.call: init %empty_tuple.type = call imports.%operator_LessEqual__carbon_thunk.decl(%addr.loc53_44.1, %addr.loc53_44.2, %addr.loc53_44.3)
// CHECK:STDOUT:   %.loc53_44.2: init bool to %.loc53_44.1 = mark_in_place_init %operator_LessEqual__carbon_thunk.call
// CHECK:STDOUT:   %.loc53_34: type = type_literal bool [concrete = bool]
// CHECK:STDOUT:   %.loc53_44.3: ref bool = temporary %.loc53_44.1, %.loc53_44.2
// CHECK:STDOUT:   %.loc53_44.4: bool = acquire_value %.loc53_44.3
// CHECK:STDOUT:   %less_than_or_equal: bool = value_binding less_than_or_equal, %.loc53_44.4
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %index.patt: %pattern_type.7ce = value_binding_pattern index [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c1.ref.loc56: ref %C = name_ref c1, %c1
// CHECK:STDOUT:   %int_42: Core.IntLiteral = int_value 42 [concrete = constants.%int_42.20e]
// CHECK:STDOUT:   %C.cpp_operator.bound: <bound method> = bound_method %c1.ref.loc56, imports.%C.cpp_operator.decl
// CHECK:STDOUT:   %impl.elem0.loc56: %.545 = impl_witness_access constants.%ImplicitAs.impl_witness.6bc, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5]
// CHECK:STDOUT:   %bound_method.loc56_30.1: <bound method> = bound_method %int_42, %impl.elem0.loc56 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.c83]
// CHECK:STDOUT:   %specific_fn.loc56: <specific function> = specific_function %impl.elem0.loc56, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc56_30.2: <bound method> = bound_method %int_42, %specific_fn.loc56 [concrete = constants.%bound_method.cb9]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc56: init %i32 = call %bound_method.loc56_30.2(%int_42) [concrete = constants.%int_42.c68]
// CHECK:STDOUT:   %.loc56_30.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc56 [concrete = constants.%int_42.c68]
// CHECK:STDOUT:   %.loc56_30.2: %i32 = converted %int_42, %.loc56_30.1 [concrete = constants.%int_42.c68]
// CHECK:STDOUT:   %C.cpp_operator.call: init %i32 = call %C.cpp_operator.bound(%c1.ref.loc56, %.loc56_30.2)
// CHECK:STDOUT:   %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %.loc56_32.1: %i32 = value_of_initializer %C.cpp_operator.call
// CHECK:STDOUT:   %.loc56_32.2: %i32 = converted %C.cpp_operator.call, %.loc56_32.1
// CHECK:STDOUT:   %index: %i32 = value_binding index, %.loc56_32.2
// CHECK:STDOUT:   %Destroy.Op.bound.loc53: <bound method> = bound_method %.loc53_44.3, constants.%Destroy.Op
// CHECK:STDOUT:   %Destroy.Op.call.loc53: init %empty_tuple.type = call %Destroy.Op.bound.loc53(%.loc53_44.3)
// CHECK:STDOUT:   %Destroy.Op.bound.loc52: <bound method> = bound_method %.loc52_47.3, constants.%Destroy.Op
// CHECK:STDOUT:   %Destroy.Op.call.loc52: init %empty_tuple.type = call %Destroy.Op.bound.loc52(%.loc52_47.3)
// CHECK:STDOUT:   %Destroy.Op.bound.loc51: <bound method> = bound_method %.loc51_35.3, constants.%Destroy.Op
// CHECK:STDOUT:   %Destroy.Op.call.loc51: init %empty_tuple.type = call %Destroy.Op.bound.loc51(%.loc51_35.3)
// CHECK:STDOUT:   %Destroy.Op.bound.loc50: <bound method> = bound_method %.loc50_38.3, constants.%Destroy.Op
// CHECK:STDOUT:   %Destroy.Op.call.loc50: init %empty_tuple.type = call %Destroy.Op.bound.loc50(%.loc50_38.3)
// CHECK:STDOUT:   %Destroy.Op.bound.loc49: <bound method> = bound_method %.loc49_35.3, constants.%Destroy.Op
// CHECK:STDOUT:   %Destroy.Op.call.loc49: init %empty_tuple.type = call %Destroy.Op.bound.loc49(%.loc49_35.3)
// CHECK:STDOUT:   %Destroy.Op.bound.loc48: <bound method> = bound_method %.loc48_31.3, constants.%Destroy.Op
// CHECK:STDOUT:   %Destroy.Op.call.loc48: init %empty_tuple.type = call %Destroy.Op.bound.loc48(%.loc48_31.3)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %C.Op.bound.loc31: <bound method> = bound_method %.loc31_38.3, constants.%C.Op
// CHECK:STDOUT:   %Op.ref.loc31: %C.cpp_destructor.type = name_ref Op, imports.%C.cpp_destructor.decl [concrete = constants.%C.cpp_destructor]
// CHECK:STDOUT:   %C.cpp_destructor.bound.loc31: <bound method> = bound_method %.loc31_38.3, %Op.ref.loc31
// CHECK:STDOUT:   %C.cpp_destructor.call.loc31: init %empty_tuple.type = call %C.cpp_destructor.bound.loc31(%.loc31_38.3)
// CHECK:STDOUT:   %C.Op.bound.loc30: <bound method> = bound_method %.loc30_37.3, constants.%C.Op
// CHECK:STDOUT:   %Op.ref.loc30: %C.cpp_destructor.type = name_ref Op, imports.%C.cpp_destructor.decl [concrete = constants.%C.cpp_destructor]
// CHECK:STDOUT:   %C.cpp_destructor.bound.loc30: <bound method> = bound_method %.loc30_37.3, %Op.ref.loc30
// CHECK:STDOUT:   %C.cpp_destructor.call.loc30: init %empty_tuple.type = call %C.cpp_destructor.bound.loc30(%.loc30_37.3)
// CHECK:STDOUT:   %C.Op.bound.loc29: <bound method> = bound_method %.loc29_38.3, constants.%C.Op
// CHECK:STDOUT:   %Op.ref.loc29: %C.cpp_destructor.type = name_ref Op, imports.%C.cpp_destructor.decl [concrete = constants.%C.cpp_destructor]
// CHECK:STDOUT:   %C.cpp_destructor.bound.loc29: <bound method> = bound_method %.loc29_38.3, %Op.ref.loc29
// CHECK:STDOUT:   %C.cpp_destructor.call.loc29: init %empty_tuple.type = call %C.cpp_destructor.bound.loc29(%.loc29_38.3)
// CHECK:STDOUT:   %C.Op.bound.loc28: <bound method> = bound_method %.loc28_37.3, constants.%C.Op
// CHECK:STDOUT:   %Op.ref.loc28: %C.cpp_destructor.type = name_ref Op, imports.%C.cpp_destructor.decl [concrete = constants.%C.cpp_destructor]
// CHECK:STDOUT:   %C.cpp_destructor.bound.loc28: <bound method> = bound_method %.loc28_37.3, %Op.ref.loc28
// CHECK:STDOUT:   %C.cpp_destructor.call.loc28: init %empty_tuple.type = call %C.cpp_destructor.bound.loc28(%.loc28_37.3)
// CHECK:STDOUT:   %C.Op.bound.loc27: <bound method> = bound_method %.loc27_38.3, constants.%C.Op
// CHECK:STDOUT:   %Op.ref.loc27: %C.cpp_destructor.type = name_ref Op, imports.%C.cpp_destructor.decl [concrete = constants.%C.cpp_destructor]
// CHECK:STDOUT:   %C.cpp_destructor.bound.loc27: <bound method> = bound_method %.loc27_38.3, %Op.ref.loc27
// CHECK:STDOUT:   %C.cpp_destructor.call.loc27: init %empty_tuple.type = call %C.cpp_destructor.bound.loc27(%.loc27_38.3)
// CHECK:STDOUT:   %C.Op.bound.loc24: <bound method> = bound_method %.loc24_33.3, constants.%C.Op
// CHECK:STDOUT:   %Op.ref.loc24: %C.cpp_destructor.type = name_ref Op, imports.%C.cpp_destructor.decl [concrete = constants.%C.cpp_destructor]
// CHECK:STDOUT:   %C.cpp_destructor.bound.loc24: <bound method> = bound_method %.loc24_33.3, %Op.ref.loc24
// CHECK:STDOUT:   %C.cpp_destructor.call.loc24: init %empty_tuple.type = call %C.cpp_destructor.bound.loc24(%.loc24_33.3)
// CHECK:STDOUT:   %C.Op.bound.loc23: <bound method> = bound_method %.loc23_35.3, constants.%C.Op
// CHECK:STDOUT:   %Op.ref.loc23: %C.cpp_destructor.type = name_ref Op, imports.%C.cpp_destructor.decl [concrete = constants.%C.cpp_destructor]
// CHECK:STDOUT:   %C.cpp_destructor.bound.loc23: <bound method> = bound_method %.loc23_35.3, %Op.ref.loc23
// CHECK:STDOUT:   %C.cpp_destructor.call.loc23: init %empty_tuple.type = call %C.cpp_destructor.bound.loc23(%.loc23_35.3)
// CHECK:STDOUT:   %C.Op.bound.loc22: <bound method> = bound_method %.loc22_41.3, constants.%C.Op
// CHECK:STDOUT:   %Op.ref.loc22: %C.cpp_destructor.type = name_ref Op, imports.%C.cpp_destructor.decl [concrete = constants.%C.cpp_destructor]
// CHECK:STDOUT:   %C.cpp_destructor.bound.loc22: <bound method> = bound_method %.loc22_41.3, %Op.ref.loc22
// CHECK:STDOUT:   %C.cpp_destructor.call.loc22: init %empty_tuple.type = call %C.cpp_destructor.bound.loc22(%.loc22_41.3)
// CHECK:STDOUT:   %C.Op.bound.loc21: <bound method> = bound_method %.loc21_38.3, constants.%C.Op
// CHECK:STDOUT:   %Op.ref.loc21: %C.cpp_destructor.type = name_ref Op, imports.%C.cpp_destructor.decl [concrete = constants.%C.cpp_destructor]
// CHECK:STDOUT:   %C.cpp_destructor.bound.loc21: <bound method> = bound_method %.loc21_38.3, %Op.ref.loc21
// CHECK:STDOUT:   %C.cpp_destructor.call.loc21: init %empty_tuple.type = call %C.cpp_destructor.bound.loc21(%.loc21_38.3)
// CHECK:STDOUT:   %C.Op.bound.loc20: <bound method> = bound_method %.loc20_35.3, constants.%C.Op
// CHECK:STDOUT:   %Op.ref.loc20: %C.cpp_destructor.type = name_ref Op, imports.%C.cpp_destructor.decl [concrete = constants.%C.cpp_destructor]
// CHECK:STDOUT:   %C.cpp_destructor.bound.loc20: <bound method> = bound_method %.loc20_35.3, %Op.ref.loc20
// CHECK:STDOUT:   %C.cpp_destructor.call.loc20: init %empty_tuple.type = call %C.cpp_destructor.bound.loc20(%.loc20_35.3)
// CHECK:STDOUT:   %C.Op.bound.loc17: <bound method> = bound_method %c2.var, constants.%C.Op
// CHECK:STDOUT:   %Op.ref.loc17: %C.cpp_destructor.type = name_ref Op, imports.%C.cpp_destructor.decl [concrete = constants.%C.cpp_destructor]
// CHECK:STDOUT:   %C.cpp_destructor.bound.loc17: <bound method> = bound_method %c2.var, %Op.ref.loc17
// CHECK:STDOUT:   %C.cpp_destructor.call.loc17: init %empty_tuple.type = call %C.cpp_destructor.bound.loc17(%c2.var)
// CHECK:STDOUT:   %C.Op.bound.loc12: <bound method> = bound_method %c1.var, constants.%C.Op
// CHECK:STDOUT:   %Op.ref.loc12: %C.cpp_destructor.type = name_ref Op, imports.%C.cpp_destructor.decl [concrete = constants.%C.cpp_destructor]
// CHECK:STDOUT:   %C.cpp_destructor.bound.loc12: <bound method> = bound_method %c1.var, %Op.ref.loc12
// CHECK:STDOUT:   %C.cpp_destructor.call.loc12: init %empty_tuple.type = call %C.cpp_destructor.bound.loc12(%c1.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op(%self.param: ref bool) = "no_op";
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
// CHECK:STDOUT:   %operator_Plus__carbon_thunk.type: type = fn_type @operator_Plus__carbon_thunk [concrete]
// CHECK:STDOUT:   %operator_Plus__carbon_thunk: %operator_Plus__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %C.cpp_destructor.type: type = fn_type @C.cpp_destructor [concrete]
// CHECK:STDOUT:   %C.cpp_destructor: %C.cpp_destructor.type = struct_value () [concrete]
// CHECK:STDOUT:   %C.Op.type: type = fn_type @C.Op [concrete]
// CHECK:STDOUT:   %C.Op: %C.Op.type = struct_value () [concrete]
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
// CHECK:STDOUT:   %operator_Plus__carbon_thunk.decl: %operator_Plus__carbon_thunk.type = fn_decl @operator_Plus__carbon_thunk [concrete = constants.%operator_Plus__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.cpp_destructor.decl: %C.cpp_destructor.type = fn_decl @C.cpp_destructor [concrete = constants.%C.cpp_destructor] {
// CHECK:STDOUT:     %self.patt: %pattern_type.217 = ref_binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.217 = ref_param_pattern %self.patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %C = ref_param call_param0
// CHECK:STDOUT:     %self: ref %C = ref_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c1.patt: %pattern_type.217 = value_binding_pattern c1 [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc12_19: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %C.ref.loc12_22: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %C.ref.loc12_24: %C.C.cpp_overload_set.type = name_ref C, imports.%C.C.cpp_overload_set.value [concrete = constants.%C.C.cpp_overload_set.value]
// CHECK:STDOUT:   %.loc12_27.1: ref %C = temporary_storage
// CHECK:STDOUT:   %addr.loc12: %ptr.d9e = addr_of %.loc12_27.1
// CHECK:STDOUT:   %C__carbon_thunk.call.loc12: init %empty_tuple.type = call imports.%C__carbon_thunk.decl(%addr.loc12) [concrete = <error>]
// CHECK:STDOUT:   %.loc12_27.2: init %C to %.loc12_27.1 = mark_in_place_init %C__carbon_thunk.call.loc12 [concrete = <error>]
// CHECK:STDOUT:   %.loc12_14: type = splice_block %C.ref.loc12_14 [concrete = constants.%C] {
// CHECK:STDOUT:     %Cpp.ref.loc12_11: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %C.ref.loc12_14: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc12_27.3: ref %C = temporary %.loc12_27.1, %.loc12_27.2 [concrete = <error>]
// CHECK:STDOUT:   %.loc12_27.4: %C = acquire_value %.loc12_27.3 [concrete = <error>]
// CHECK:STDOUT:   %c1: %C = value_binding c1, %.loc12_27.4 [concrete = <error>]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c2.patt: %pattern_type.217 = value_binding_pattern c2 [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc17_19: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %C.ref.loc17_22: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %C.ref.loc17_24: %C.C.cpp_overload_set.type = name_ref C, imports.%C.C.cpp_overload_set.value [concrete = constants.%C.C.cpp_overload_set.value]
// CHECK:STDOUT:   %.loc17_27.1: ref %C = temporary_storage
// CHECK:STDOUT:   %addr.loc17: %ptr.d9e = addr_of %.loc17_27.1
// CHECK:STDOUT:   %C__carbon_thunk.call.loc17: init %empty_tuple.type = call imports.%C__carbon_thunk.decl(%addr.loc17) [concrete = <error>]
// CHECK:STDOUT:   %.loc17_27.2: init %C to %.loc17_27.1 = mark_in_place_init %C__carbon_thunk.call.loc17 [concrete = <error>]
// CHECK:STDOUT:   %.loc17_14: type = splice_block %C.ref.loc17_14 [concrete = constants.%C] {
// CHECK:STDOUT:     %Cpp.ref.loc17_11: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %C.ref.loc17_14: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc17_27.3: ref %C = temporary %.loc17_27.1, %.loc17_27.2 [concrete = <error>]
// CHECK:STDOUT:   %.loc17_27.4: %C = acquire_value %.loc17_27.3 [concrete = <error>]
// CHECK:STDOUT:   %c2: %C = value_binding c2, %.loc17_27.4 [concrete = <error>]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c3.patt: %pattern_type.217 = value_binding_pattern c3 [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c1.ref.loc18: %C = name_ref c1, %c1 [concrete = <error>]
// CHECK:STDOUT:   %c2.ref: %C = name_ref c2, %c2 [concrete = <error>]
// CHECK:STDOUT:   %.loc18_22.1: ref %C = temporary_storage
// CHECK:STDOUT:   %.loc18_19: ref %C = value_as_ref %c1.ref.loc18
// CHECK:STDOUT:   %addr.loc18_22.1: %ptr.d9e = addr_of %.loc18_19
// CHECK:STDOUT:   %.loc18_24: ref %C = value_as_ref %c2.ref
// CHECK:STDOUT:   %addr.loc18_22.2: %ptr.d9e = addr_of %.loc18_24
// CHECK:STDOUT:   %addr.loc18_22.3: %ptr.d9e = addr_of %.loc18_22.1
// CHECK:STDOUT:   %operator_Plus__carbon_thunk.call.loc18: init %empty_tuple.type = call imports.%operator_Plus__carbon_thunk.decl(%addr.loc18_22.1, %addr.loc18_22.2, %addr.loc18_22.3)
// CHECK:STDOUT:   %.loc18_22.2: init %C to %.loc18_22.1 = mark_in_place_init %operator_Plus__carbon_thunk.call.loc18
// CHECK:STDOUT:   %.loc18_14: type = splice_block %C.ref.loc18 [concrete = constants.%C] {
// CHECK:STDOUT:     %Cpp.ref.loc18: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %C.ref.loc18: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc18_22.3: ref %C = temporary %.loc18_22.1, %.loc18_22.2
// CHECK:STDOUT:   %.loc18_22.4: %C = acquire_value %.loc18_22.3
// CHECK:STDOUT:   %c3: %C = value_binding c3, %.loc18_22.4
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c4.patt: %pattern_type.217 = value_binding_pattern c4 [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c1.ref.loc19: %C = name_ref c1, %c1 [concrete = <error>]
// CHECK:STDOUT:   %c3.ref.loc19: %C = name_ref c3, %c3
// CHECK:STDOUT:   %.loc19_22.1: ref %C = temporary_storage
// CHECK:STDOUT:   %.loc19_19: ref %C = value_as_ref %c1.ref.loc19
// CHECK:STDOUT:   %addr.loc19_22.1: %ptr.d9e = addr_of %.loc19_19
// CHECK:STDOUT:   %.loc19_24: ref %C = value_as_ref %c3.ref.loc19
// CHECK:STDOUT:   %addr.loc19_22.2: %ptr.d9e = addr_of %.loc19_24
// CHECK:STDOUT:   %addr.loc19_22.3: %ptr.d9e = addr_of %.loc19_22.1
// CHECK:STDOUT:   %operator_Plus__carbon_thunk.call.loc19: init %empty_tuple.type = call imports.%operator_Plus__carbon_thunk.decl(%addr.loc19_22.1, %addr.loc19_22.2, %addr.loc19_22.3)
// CHECK:STDOUT:   %.loc19_22.2: init %C to %.loc19_22.1 = mark_in_place_init %operator_Plus__carbon_thunk.call.loc19
// CHECK:STDOUT:   %.loc19_14: type = splice_block %C.ref.loc19 [concrete = constants.%C] {
// CHECK:STDOUT:     %Cpp.ref.loc19: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %C.ref.loc19: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc19_22.3: ref %C = temporary %.loc19_22.1, %.loc19_22.2
// CHECK:STDOUT:   %.loc19_22.4: %C = acquire_value %.loc19_22.3
// CHECK:STDOUT:   %c4: %C = value_binding c4, %.loc19_22.4
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c5.patt: %pattern_type.217 = value_binding_pattern c5 [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c4.ref: %C = name_ref c4, %c4
// CHECK:STDOUT:   %c3.ref.loc20: %C = name_ref c3, %c3
// CHECK:STDOUT:   %.loc20_29.1: ref %C = temporary_storage
// CHECK:STDOUT:   %.loc20_26: ref %C = value_as_ref %c4.ref
// CHECK:STDOUT:   %addr.loc20_29.1: %ptr.d9e = addr_of %.loc20_26
// CHECK:STDOUT:   %.loc20_31: ref %C = value_as_ref %c3.ref.loc20
// CHECK:STDOUT:   %addr.loc20_29.2: %ptr.d9e = addr_of %.loc20_31
// CHECK:STDOUT:   %addr.loc20_29.3: %ptr.d9e = addr_of %.loc20_29.1
// CHECK:STDOUT:   %operator_Plus__carbon_thunk.call.loc20: init %empty_tuple.type = call imports.%operator_Plus__carbon_thunk.decl(%addr.loc20_29.1, %addr.loc20_29.2, %addr.loc20_29.3)
// CHECK:STDOUT:   %.loc20_29.2: init %C to %.loc20_29.1 = mark_in_place_init %operator_Plus__carbon_thunk.call.loc20
// CHECK:STDOUT:   %.loc20_21: type = splice_block %C.ref.loc20 [concrete = constants.%C] {
// CHECK:STDOUT:     %Cpp.ref.loc20: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %C.ref.loc20: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc20_29.3: ref %C = temporary %.loc20_29.1, %.loc20_29.2
// CHECK:STDOUT:   %.loc20_29.4: %C = acquire_value %.loc20_29.3
// CHECK:STDOUT:   %c5: %C = value_binding c5, %.loc20_29.4
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %C.Op.bound.loc20: <bound method> = bound_method %.loc20_29.3, constants.%C.Op
// CHECK:STDOUT:   %Op.ref.loc20: %C.cpp_destructor.type = name_ref Op, imports.%C.cpp_destructor.decl [concrete = constants.%C.cpp_destructor]
// CHECK:STDOUT:   %C.cpp_destructor.bound.loc20: <bound method> = bound_method %.loc20_29.3, %Op.ref.loc20
// CHECK:STDOUT:   %C.cpp_destructor.call.loc20: init %empty_tuple.type = call %C.cpp_destructor.bound.loc20(%.loc20_29.3)
// CHECK:STDOUT:   %C.Op.bound.loc19: <bound method> = bound_method %.loc19_22.3, constants.%C.Op
// CHECK:STDOUT:   %Op.ref.loc19: %C.cpp_destructor.type = name_ref Op, imports.%C.cpp_destructor.decl [concrete = constants.%C.cpp_destructor]
// CHECK:STDOUT:   %C.cpp_destructor.bound.loc19: <bound method> = bound_method %.loc19_22.3, %Op.ref.loc19
// CHECK:STDOUT:   %C.cpp_destructor.call.loc19: init %empty_tuple.type = call %C.cpp_destructor.bound.loc19(%.loc19_22.3)
// CHECK:STDOUT:   %C.Op.bound.loc18: <bound method> = bound_method %.loc18_22.3, constants.%C.Op
// CHECK:STDOUT:   %Op.ref.loc18: %C.cpp_destructor.type = name_ref Op, imports.%C.cpp_destructor.decl [concrete = constants.%C.cpp_destructor]
// CHECK:STDOUT:   %C.cpp_destructor.bound.loc18: <bound method> = bound_method %.loc18_22.3, %Op.ref.loc18
// CHECK:STDOUT:   %C.cpp_destructor.call.loc18: init %empty_tuple.type = call %C.cpp_destructor.bound.loc18(%.loc18_22.3)
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
// CHECK:STDOUT:   %pattern_type.831: type = pattern_type bool [concrete]
// CHECK:STDOUT:   %ptr.bb2: type = ptr_type bool [concrete]
// CHECK:STDOUT:   %operator_Greater__carbon_thunk.type: type = fn_type @operator_Greater__carbon_thunk [concrete]
// CHECK:STDOUT:   %operator_Greater__carbon_thunk: %operator_Greater__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %operator_LessEqual__carbon_thunk.type: type = fn_type @operator_LessEqual__carbon_thunk [concrete]
// CHECK:STDOUT:   %operator_LessEqual__carbon_thunk: %operator_LessEqual__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.Op.type: type = fn_type @Destroy.Op [concrete]
// CHECK:STDOUT:   %Destroy.Op: %Destroy.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %C.cpp_destructor.type: type = fn_type @C.cpp_destructor [concrete]
// CHECK:STDOUT:   %C.cpp_destructor: %C.cpp_destructor.type = struct_value () [concrete]
// CHECK:STDOUT:   %C.Op.type: type = fn_type @C.Op [concrete]
// CHECK:STDOUT:   %C.Op: %C.Op.type = struct_value () [concrete]
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
// CHECK:STDOUT:   %operator_Greater__carbon_thunk.decl: %operator_Greater__carbon_thunk.type = fn_decl @operator_Greater__carbon_thunk [concrete = constants.%operator_Greater__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %operator_LessEqual__carbon_thunk.decl: %operator_LessEqual__carbon_thunk.type = fn_decl @operator_LessEqual__carbon_thunk [concrete = constants.%operator_LessEqual__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.cpp_destructor.decl: %C.cpp_destructor.type = fn_decl @C.cpp_destructor [concrete = constants.%C.cpp_destructor] {
// CHECK:STDOUT:     %self.patt: %pattern_type.217 = ref_binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.217 = ref_param_pattern %self.patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %C = ref_param call_param0
// CHECK:STDOUT:     %self: ref %C = ref_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c1.patt: %pattern_type.217 = ref_binding_pattern c1 [concrete]
// CHECK:STDOUT:     %c1.var_patt: %pattern_type.217 = var_pattern %c1.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c1.var: ref %C = var %c1.var_patt
// CHECK:STDOUT:   %Cpp.ref.loc19_19: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %C.ref.loc19_22: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %C.ref.loc19_24: %C.C.cpp_overload_set.type = name_ref C, imports.%C.C.cpp_overload_set.value [concrete = constants.%C.C.cpp_overload_set.value]
// CHECK:STDOUT:   %.loc19_3: ref %C = splice_block %c1.var {}
// CHECK:STDOUT:   %addr.loc19: %ptr.d9e = addr_of %.loc19_3
// CHECK:STDOUT:   %C__carbon_thunk.call.loc19: init %empty_tuple.type = call imports.%C__carbon_thunk.decl(%addr.loc19) [concrete = <error>]
// CHECK:STDOUT:   %.loc19_27: init %C to %.loc19_3 = mark_in_place_init %C__carbon_thunk.call.loc19 [concrete = <error>]
// CHECK:STDOUT:   assign %c1.var, %.loc19_27
// CHECK:STDOUT:   %.loc19_14: type = splice_block %C.ref.loc19_14 [concrete = constants.%C] {
// CHECK:STDOUT:     %Cpp.ref.loc19_11: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %C.ref.loc19_14: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c1: ref %C = ref_binding c1, %c1.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c2.patt: %pattern_type.217 = ref_binding_pattern c2 [concrete]
// CHECK:STDOUT:     %c2.var_patt: %pattern_type.217 = var_pattern %c2.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c2.var: ref %C = var %c2.var_patt
// CHECK:STDOUT:   %Cpp.ref.loc24_19: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %C.ref.loc24_22: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %C.ref.loc24_24: %C.C.cpp_overload_set.type = name_ref C, imports.%C.C.cpp_overload_set.value [concrete = constants.%C.C.cpp_overload_set.value]
// CHECK:STDOUT:   %.loc24_3: ref %C = splice_block %c2.var {}
// CHECK:STDOUT:   %addr.loc24: %ptr.d9e = addr_of %.loc24_3
// CHECK:STDOUT:   %C__carbon_thunk.call.loc24: init %empty_tuple.type = call imports.%C__carbon_thunk.decl(%addr.loc24) [concrete = <error>]
// CHECK:STDOUT:   %.loc24_27: init %C to %.loc24_3 = mark_in_place_init %C__carbon_thunk.call.loc24 [concrete = <error>]
// CHECK:STDOUT:   assign %c2.var, %.loc24_27
// CHECK:STDOUT:   %.loc24_14: type = splice_block %C.ref.loc24_14 [concrete = constants.%C] {
// CHECK:STDOUT:     %Cpp.ref.loc24_11: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %C.ref.loc24_14: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c2: ref %C = ref_binding c2, %c2.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %greater_than.patt: %pattern_type.831 = value_binding_pattern greater_than [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c1.ref.loc27: ref %C = name_ref c1, %c1
// CHECK:STDOUT:   %c2.ref.loc27: ref %C = name_ref c2, %c2
// CHECK:STDOUT:   %.loc27_35.1: %C = acquire_value %c1.ref.loc27
// CHECK:STDOUT:   %.loc27_40.1: %C = acquire_value %c2.ref.loc27
// CHECK:STDOUT:   %.loc27_35.2: ref %C = value_as_ref %.loc27_35.1
// CHECK:STDOUT:   %addr.loc27_38.1: %ptr.d9e = addr_of %.loc27_35.2
// CHECK:STDOUT:   %.loc27_40.2: ref %C = value_as_ref %.loc27_40.1
// CHECK:STDOUT:   %addr.loc27_38.2: %ptr.d9e = addr_of %.loc27_40.2
// CHECK:STDOUT:   %.loc27_38.1: ref bool = temporary_storage
// CHECK:STDOUT:   %addr.loc27_38.3: %ptr.bb2 = addr_of %.loc27_38.1
// CHECK:STDOUT:   %operator_Greater__carbon_thunk.call: init %empty_tuple.type = call imports.%operator_Greater__carbon_thunk.decl(%addr.loc27_38.1, %addr.loc27_38.2, %addr.loc27_38.3)
// CHECK:STDOUT:   %.loc27_38.2: init bool to %.loc27_38.1 = mark_in_place_init %operator_Greater__carbon_thunk.call
// CHECK:STDOUT:   %.loc27_28: type = type_literal bool [concrete = bool]
// CHECK:STDOUT:   %.loc27_38.3: ref bool = temporary %.loc27_38.1, %.loc27_38.2
// CHECK:STDOUT:   %.loc27_38.4: bool = acquire_value %.loc27_38.3
// CHECK:STDOUT:   %greater_than: bool = value_binding greater_than, %.loc27_38.4
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %less_than.patt: %pattern_type.831 = value_binding_pattern less_than [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c1.ref.loc34: ref %C = name_ref c1, %c1
// CHECK:STDOUT:   %c2.ref.loc34: ref %C = name_ref c2, %c2
// CHECK:STDOUT:   %.loc34: type = type_literal bool [concrete = bool]
// CHECK:STDOUT:   %less_than: bool = value_binding less_than, <error> [concrete = <error>]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %greater_than_or_equal.patt: %pattern_type.831 = value_binding_pattern greater_than_or_equal [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c1.ref.loc41: ref %C = name_ref c1, %c1
// CHECK:STDOUT:   %c2.ref.loc41: ref %C = name_ref c2, %c2
// CHECK:STDOUT:   %.loc41: type = type_literal bool [concrete = bool]
// CHECK:STDOUT:   %greater_than_or_equal: bool = value_binding greater_than_or_equal, <error> [concrete = <error>]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %less_than_or_equal.patt: %pattern_type.831 = value_binding_pattern less_than_or_equal [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c1.ref.loc44: ref %C = name_ref c1, %c1
// CHECK:STDOUT:   %c2.ref.loc44: ref %C = name_ref c2, %c2
// CHECK:STDOUT:   %.loc44_41.1: %C = acquire_value %c1.ref.loc44
// CHECK:STDOUT:   %.loc44_47.1: %C = acquire_value %c2.ref.loc44
// CHECK:STDOUT:   %.loc44_41.2: ref %C = value_as_ref %.loc44_41.1
// CHECK:STDOUT:   %addr.loc44_44.1: %ptr.d9e = addr_of %.loc44_41.2
// CHECK:STDOUT:   %.loc44_47.2: ref %C = value_as_ref %.loc44_47.1
// CHECK:STDOUT:   %addr.loc44_44.2: %ptr.d9e = addr_of %.loc44_47.2
// CHECK:STDOUT:   %.loc44_44.1: ref bool = temporary_storage
// CHECK:STDOUT:   %addr.loc44_44.3: %ptr.bb2 = addr_of %.loc44_44.1
// CHECK:STDOUT:   %operator_LessEqual__carbon_thunk.call: init %empty_tuple.type = call imports.%operator_LessEqual__carbon_thunk.decl(%addr.loc44_44.1, %addr.loc44_44.2, %addr.loc44_44.3)
// CHECK:STDOUT:   %.loc44_44.2: init bool to %.loc44_44.1 = mark_in_place_init %operator_LessEqual__carbon_thunk.call
// CHECK:STDOUT:   %.loc44_34: type = type_literal bool [concrete = bool]
// CHECK:STDOUT:   %.loc44_44.3: ref bool = temporary %.loc44_44.1, %.loc44_44.2
// CHECK:STDOUT:   %.loc44_44.4: bool = acquire_value %.loc44_44.3
// CHECK:STDOUT:   %less_than_or_equal: bool = value_binding less_than_or_equal, %.loc44_44.4
// CHECK:STDOUT:   %Destroy.Op.bound.loc44: <bound method> = bound_method %.loc44_44.3, constants.%Destroy.Op
// CHECK:STDOUT:   %Destroy.Op.call.loc44: init %empty_tuple.type = call %Destroy.Op.bound.loc44(%.loc44_44.3)
// CHECK:STDOUT:   %Destroy.Op.bound.loc27: <bound method> = bound_method %.loc27_38.3, constants.%Destroy.Op
// CHECK:STDOUT:   %Destroy.Op.call.loc27: init %empty_tuple.type = call %Destroy.Op.bound.loc27(%.loc27_38.3)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %C.Op.bound.loc24: <bound method> = bound_method %c2.var, constants.%C.Op
// CHECK:STDOUT:   %Op.ref.loc24: %C.cpp_destructor.type = name_ref Op, imports.%C.cpp_destructor.decl [concrete = constants.%C.cpp_destructor]
// CHECK:STDOUT:   %C.cpp_destructor.bound.loc24: <bound method> = bound_method %c2.var, %Op.ref.loc24
// CHECK:STDOUT:   %C.cpp_destructor.call.loc24: init %empty_tuple.type = call %C.cpp_destructor.bound.loc24(%c2.var)
// CHECK:STDOUT:   %C.Op.bound.loc19: <bound method> = bound_method %c1.var, constants.%C.Op
// CHECK:STDOUT:   %Op.ref.loc19: %C.cpp_destructor.type = name_ref Op, imports.%C.cpp_destructor.decl [concrete = constants.%C.cpp_destructor]
// CHECK:STDOUT:   %C.cpp_destructor.bound.loc19: <bound method> = bound_method %c1.var, %Op.ref.loc19
// CHECK:STDOUT:   %C.cpp_destructor.call.loc19: init %empty_tuple.type = call %C.cpp_destructor.bound.loc19(%c1.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op(%self.param: ref bool) = "no_op";
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
// CHECK:STDOUT:   %pattern_type.831: type = pattern_type bool [concrete]
// CHECK:STDOUT:   %ptr.bb2: type = ptr_type bool [concrete]
// CHECK:STDOUT:   %operator_EqualEqual__carbon_thunk.type: type = fn_type @operator_EqualEqual__carbon_thunk [concrete]
// CHECK:STDOUT:   %operator_EqualEqual__carbon_thunk: %operator_EqualEqual__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.Op.type: type = fn_type @Destroy.Op [concrete]
// CHECK:STDOUT:   %Destroy.Op: %Destroy.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %C.cpp_destructor.type: type = fn_type @C.cpp_destructor [concrete]
// CHECK:STDOUT:   %C.cpp_destructor: %C.cpp_destructor.type = struct_value () [concrete]
// CHECK:STDOUT:   %C.Op.type: type = fn_type @C.Op [concrete]
// CHECK:STDOUT:   %C.Op: %C.Op.type = struct_value () [concrete]
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
// CHECK:STDOUT:   %operator_EqualEqual__carbon_thunk.decl: %operator_EqualEqual__carbon_thunk.type = fn_decl @operator_EqualEqual__carbon_thunk [concrete = constants.%operator_EqualEqual__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.cpp_destructor.decl: %C.cpp_destructor.type = fn_decl @C.cpp_destructor [concrete = constants.%C.cpp_destructor] {
// CHECK:STDOUT:     %self.patt: %pattern_type.217 = ref_binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.217 = ref_param_pattern %self.patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %C = ref_param call_param0
// CHECK:STDOUT:     %self: ref %C = ref_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c1.patt: %pattern_type.217 = ref_binding_pattern c1 [concrete]
// CHECK:STDOUT:     %c1.var_patt: %pattern_type.217 = var_pattern %c1.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c1.var: ref %C = var %c1.var_patt
// CHECK:STDOUT:   %Cpp.ref.loc16_19: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %C.ref.loc16_22: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %C.ref.loc16_24: %C.C.cpp_overload_set.type = name_ref C, imports.%C.C.cpp_overload_set.value [concrete = constants.%C.C.cpp_overload_set.value]
// CHECK:STDOUT:   %.loc16_3: ref %C = splice_block %c1.var {}
// CHECK:STDOUT:   %addr.loc16: %ptr.d9e = addr_of %.loc16_3
// CHECK:STDOUT:   %C__carbon_thunk.call.loc16: init %empty_tuple.type = call imports.%C__carbon_thunk.decl(%addr.loc16) [concrete = <error>]
// CHECK:STDOUT:   %.loc16_27: init %C to %.loc16_3 = mark_in_place_init %C__carbon_thunk.call.loc16 [concrete = <error>]
// CHECK:STDOUT:   assign %c1.var, %.loc16_27
// CHECK:STDOUT:   %.loc16_14: type = splice_block %C.ref.loc16_14 [concrete = constants.%C] {
// CHECK:STDOUT:     %Cpp.ref.loc16_11: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %C.ref.loc16_14: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c1: ref %C = ref_binding c1, %c1.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c2.patt: %pattern_type.217 = ref_binding_pattern c2 [concrete]
// CHECK:STDOUT:     %c2.var_patt: %pattern_type.217 = var_pattern %c2.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c2.var: ref %C = var %c2.var_patt
// CHECK:STDOUT:   %Cpp.ref.loc21_19: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %C.ref.loc21_22: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %C.ref.loc21_24: %C.C.cpp_overload_set.type = name_ref C, imports.%C.C.cpp_overload_set.value [concrete = constants.%C.C.cpp_overload_set.value]
// CHECK:STDOUT:   %.loc21_3: ref %C = splice_block %c2.var {}
// CHECK:STDOUT:   %addr.loc21: %ptr.d9e = addr_of %.loc21_3
// CHECK:STDOUT:   %C__carbon_thunk.call.loc21: init %empty_tuple.type = call imports.%C__carbon_thunk.decl(%addr.loc21) [concrete = <error>]
// CHECK:STDOUT:   %.loc21_27: init %C to %.loc21_3 = mark_in_place_init %C__carbon_thunk.call.loc21 [concrete = <error>]
// CHECK:STDOUT:   assign %c2.var, %.loc21_27
// CHECK:STDOUT:   %.loc21_14: type = splice_block %C.ref.loc21_14 [concrete = constants.%C] {
// CHECK:STDOUT:     %Cpp.ref.loc21_11: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %C.ref.loc21_14: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c2: ref %C = ref_binding c2, %c2.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %equal.patt: %pattern_type.831 = value_binding_pattern equal [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c1.ref.loc24: ref %C = name_ref c1, %c1
// CHECK:STDOUT:   %c2.ref.loc24: ref %C = name_ref c2, %c2
// CHECK:STDOUT:   %.loc24_28.1: %C = acquire_value %c1.ref.loc24
// CHECK:STDOUT:   %.loc24_34.1: %C = acquire_value %c2.ref.loc24
// CHECK:STDOUT:   %.loc24_28.2: ref %C = value_as_ref %.loc24_28.1
// CHECK:STDOUT:   %addr.loc24_31.1: %ptr.d9e = addr_of %.loc24_28.2
// CHECK:STDOUT:   %.loc24_34.2: ref %C = value_as_ref %.loc24_34.1
// CHECK:STDOUT:   %addr.loc24_31.2: %ptr.d9e = addr_of %.loc24_34.2
// CHECK:STDOUT:   %.loc24_31.1: ref bool = temporary_storage
// CHECK:STDOUT:   %addr.loc24_31.3: %ptr.bb2 = addr_of %.loc24_31.1
// CHECK:STDOUT:   %operator_EqualEqual__carbon_thunk.call: init %empty_tuple.type = call imports.%operator_EqualEqual__carbon_thunk.decl(%addr.loc24_31.1, %addr.loc24_31.2, %addr.loc24_31.3)
// CHECK:STDOUT:   %.loc24_31.2: init bool to %.loc24_31.1 = mark_in_place_init %operator_EqualEqual__carbon_thunk.call
// CHECK:STDOUT:   %.loc24_21: type = type_literal bool [concrete = bool]
// CHECK:STDOUT:   %.loc24_31.3: ref bool = temporary %.loc24_31.1, %.loc24_31.2
// CHECK:STDOUT:   %.loc24_31.4: bool = acquire_value %.loc24_31.3
// CHECK:STDOUT:   %equal: bool = value_binding equal, %.loc24_31.4
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %not_equal.patt: %pattern_type.831 = value_binding_pattern not_equal [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c1.ref.loc31: ref %C = name_ref c1, %c1
// CHECK:STDOUT:   %c2.ref.loc31: ref %C = name_ref c2, %c2
// CHECK:STDOUT:   %.loc31: type = type_literal bool [concrete = bool]
// CHECK:STDOUT:   %not_equal: bool = value_binding not_equal, <error> [concrete = <error>]
// CHECK:STDOUT:   %Destroy.Op.bound: <bound method> = bound_method %.loc24_31.3, constants.%Destroy.Op
// CHECK:STDOUT:   %Destroy.Op.call: init %empty_tuple.type = call %Destroy.Op.bound(%.loc24_31.3)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %C.Op.bound.loc21: <bound method> = bound_method %c2.var, constants.%C.Op
// CHECK:STDOUT:   %Op.ref.loc21: %C.cpp_destructor.type = name_ref Op, imports.%C.cpp_destructor.decl [concrete = constants.%C.cpp_destructor]
// CHECK:STDOUT:   %C.cpp_destructor.bound.loc21: <bound method> = bound_method %c2.var, %Op.ref.loc21
// CHECK:STDOUT:   %C.cpp_destructor.call.loc21: init %empty_tuple.type = call %C.cpp_destructor.bound.loc21(%c2.var)
// CHECK:STDOUT:   %C.Op.bound.loc16: <bound method> = bound_method %c1.var, constants.%C.Op
// CHECK:STDOUT:   %Op.ref.loc16: %C.cpp_destructor.type = name_ref Op, imports.%C.cpp_destructor.decl [concrete = constants.%C.cpp_destructor]
// CHECK:STDOUT:   %C.cpp_destructor.bound.loc16: <bound method> = bound_method %c1.var, %Op.ref.loc16
// CHECK:STDOUT:   %C.cpp_destructor.call.loc16: init %empty_tuple.type = call %C.cpp_destructor.bound.loc16(%c1.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op(%self.param: ref bool) = "no_op";
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
// CHECK:STDOUT:   %operator_Plus__carbon_thunk.type: type = fn_type @operator_Plus__carbon_thunk [concrete]
// CHECK:STDOUT:   %operator_Plus__carbon_thunk: %operator_Plus__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %C.cpp_destructor.type: type = fn_type @C.cpp_destructor [concrete]
// CHECK:STDOUT:   %C.cpp_destructor: %C.cpp_destructor.type = struct_value () [concrete]
// CHECK:STDOUT:   %C.Op.type: type = fn_type @C.Op [concrete]
// CHECK:STDOUT:   %C.Op: %C.Op.type = struct_value () [concrete]
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
// CHECK:STDOUT:   %operator_Plus__carbon_thunk.decl: %operator_Plus__carbon_thunk.type = fn_decl @operator_Plus__carbon_thunk [concrete = constants.%operator_Plus__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.cpp_destructor.decl: %C.cpp_destructor.type = fn_decl @C.cpp_destructor [concrete = constants.%C.cpp_destructor] {
// CHECK:STDOUT:     %self.patt: %pattern_type.69f = ref_binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.69f = ref_param_pattern %self.patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %C = ref_param call_param0
// CHECK:STDOUT:     %self: ref %C = ref_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c1.patt: %pattern_type.69f = value_binding_pattern c1 [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc12_21: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %N.ref.loc12_24: <namespace> = name_ref N, imports.%N [concrete = imports.%N]
// CHECK:STDOUT:   %C.ref.loc12_26: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %C.ref.loc12_28: %C.C.cpp_overload_set.type = name_ref C, imports.%C.C.cpp_overload_set.value [concrete = constants.%C.C.cpp_overload_set.value]
// CHECK:STDOUT:   %.loc12_31.1: ref %C = temporary_storage
// CHECK:STDOUT:   %addr.loc12: %ptr.838 = addr_of %.loc12_31.1
// CHECK:STDOUT:   %C__carbon_thunk.call.loc12: init %empty_tuple.type = call imports.%C__carbon_thunk.decl(%addr.loc12) [concrete = <error>]
// CHECK:STDOUT:   %.loc12_31.2: init %C to %.loc12_31.1 = mark_in_place_init %C__carbon_thunk.call.loc12 [concrete = <error>]
// CHECK:STDOUT:   %.loc12_16: type = splice_block %C.ref.loc12_16 [concrete = constants.%C] {
// CHECK:STDOUT:     %Cpp.ref.loc12_11: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %N.ref.loc12_14: <namespace> = name_ref N, imports.%N [concrete = imports.%N]
// CHECK:STDOUT:     %C.ref.loc12_16: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc12_31.3: ref %C = temporary %.loc12_31.1, %.loc12_31.2 [concrete = <error>]
// CHECK:STDOUT:   %.loc12_31.4: %C = acquire_value %.loc12_31.3 [concrete = <error>]
// CHECK:STDOUT:   %c1: %C = value_binding c1, %.loc12_31.4 [concrete = <error>]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c2.patt: %pattern_type.69f = value_binding_pattern c2 [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc17_21: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %N.ref.loc17_24: <namespace> = name_ref N, imports.%N [concrete = imports.%N]
// CHECK:STDOUT:   %C.ref.loc17_26: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %C.ref.loc17_28: %C.C.cpp_overload_set.type = name_ref C, imports.%C.C.cpp_overload_set.value [concrete = constants.%C.C.cpp_overload_set.value]
// CHECK:STDOUT:   %.loc17_31.1: ref %C = temporary_storage
// CHECK:STDOUT:   %addr.loc17: %ptr.838 = addr_of %.loc17_31.1
// CHECK:STDOUT:   %C__carbon_thunk.call.loc17: init %empty_tuple.type = call imports.%C__carbon_thunk.decl(%addr.loc17) [concrete = <error>]
// CHECK:STDOUT:   %.loc17_31.2: init %C to %.loc17_31.1 = mark_in_place_init %C__carbon_thunk.call.loc17 [concrete = <error>]
// CHECK:STDOUT:   %.loc17_16: type = splice_block %C.ref.loc17_16 [concrete = constants.%C] {
// CHECK:STDOUT:     %Cpp.ref.loc17_11: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %N.ref.loc17_14: <namespace> = name_ref N, imports.%N [concrete = imports.%N]
// CHECK:STDOUT:     %C.ref.loc17_16: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc17_31.3: ref %C = temporary %.loc17_31.1, %.loc17_31.2 [concrete = <error>]
// CHECK:STDOUT:   %.loc17_31.4: %C = acquire_value %.loc17_31.3 [concrete = <error>]
// CHECK:STDOUT:   %c2: %C = value_binding c2, %.loc17_31.4 [concrete = <error>]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c3.patt: %pattern_type.69f = value_binding_pattern c3 [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c1.ref: %C = name_ref c1, %c1 [concrete = <error>]
// CHECK:STDOUT:   %c2.ref: %C = name_ref c2, %c2 [concrete = <error>]
// CHECK:STDOUT:   %.loc18_31.1: ref %C = temporary_storage
// CHECK:STDOUT:   %.loc18_28: ref %C = value_as_ref %c1.ref
// CHECK:STDOUT:   %addr.loc18_31.1: %ptr.838 = addr_of %.loc18_28
// CHECK:STDOUT:   %.loc18_33: ref %C = value_as_ref %c2.ref
// CHECK:STDOUT:   %addr.loc18_31.2: %ptr.838 = addr_of %.loc18_33
// CHECK:STDOUT:   %addr.loc18_31.3: %ptr.838 = addr_of %.loc18_31.1
// CHECK:STDOUT:   %operator_Plus__carbon_thunk.call: init %empty_tuple.type = call imports.%operator_Plus__carbon_thunk.decl(%addr.loc18_31.1, %addr.loc18_31.2, %addr.loc18_31.3)
// CHECK:STDOUT:   %.loc18_31.2: init %C to %.loc18_31.1 = mark_in_place_init %operator_Plus__carbon_thunk.call
// CHECK:STDOUT:   %.loc18_23: type = splice_block %C.ref.loc18 [concrete = constants.%C] {
// CHECK:STDOUT:     %Cpp.ref.loc18: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %N.ref.loc18: <namespace> = name_ref N, imports.%N [concrete = imports.%N]
// CHECK:STDOUT:     %C.ref.loc18: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc18_31.3: ref %C = temporary %.loc18_31.1, %.loc18_31.2
// CHECK:STDOUT:   %.loc18_31.4: %C = acquire_value %.loc18_31.3
// CHECK:STDOUT:   %c3: %C = value_binding c3, %.loc18_31.4
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %C.Op.bound: <bound method> = bound_method %.loc18_31.3, constants.%C.Op
// CHECK:STDOUT:   %Op.ref: %C.cpp_destructor.type = name_ref Op, imports.%C.cpp_destructor.decl [concrete = constants.%C.cpp_destructor]
// CHECK:STDOUT:   %C.cpp_destructor.bound: <bound method> = bound_method %.loc18_31.3, %Op.ref
// CHECK:STDOUT:   %C.cpp_destructor.call: init %empty_tuple.type = call %C.cpp_destructor.bound(%.loc18_31.3)
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
// CHECK:STDOUT:   %operator_Plus__carbon_thunk.type: type = fn_type @operator_Plus__carbon_thunk [concrete]
// CHECK:STDOUT:   %operator_Plus__carbon_thunk: %operator_Plus__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %operator_Minus__carbon_thunk.type: type = fn_type @operator_Minus__carbon_thunk [concrete]
// CHECK:STDOUT:   %operator_Minus__carbon_thunk: %operator_Minus__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %C2.cpp_destructor.type: type = fn_type @C2.cpp_destructor [concrete]
// CHECK:STDOUT:   %C2.cpp_destructor: %C2.cpp_destructor.type = struct_value () [concrete]
// CHECK:STDOUT:   %C2.Op.type: type = fn_type @C2.Op [concrete]
// CHECK:STDOUT:   %C2.Op: %C2.Op.type = struct_value () [concrete]
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
// CHECK:STDOUT:   %operator_Plus__carbon_thunk.decl: %operator_Plus__carbon_thunk.type = fn_decl @operator_Plus__carbon_thunk [concrete = constants.%operator_Plus__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %operator_Minus__carbon_thunk.decl: %operator_Minus__carbon_thunk.type = fn_decl @operator_Minus__carbon_thunk [concrete = constants.%operator_Minus__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C2.cpp_destructor.decl: %C2.cpp_destructor.type = fn_decl @C2.cpp_destructor [concrete = constants.%C2.cpp_destructor] {
// CHECK:STDOUT:     %self.patt: %pattern_type.846 = ref_binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.846 = ref_param_pattern %self.patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %C2 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %C2 = ref_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c1.patt: %pattern_type.20f = value_binding_pattern c1 [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc12_23: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %N1.ref.loc12_26: <namespace> = name_ref N1, imports.%N1 [concrete = imports.%N1]
// CHECK:STDOUT:   %C1.ref.loc12_29: type = name_ref C1, imports.%C1.decl [concrete = constants.%C1]
// CHECK:STDOUT:   %C1.ref.loc12_32: %C1.C1.cpp_overload_set.type = name_ref C1, imports.%C1.C1.cpp_overload_set.value [concrete = constants.%C1.C1.cpp_overload_set.value]
// CHECK:STDOUT:   %.loc12_36.1: ref %C1 = temporary_storage
// CHECK:STDOUT:   %addr.loc12: %ptr.087 = addr_of %.loc12_36.1
// CHECK:STDOUT:   %C1__carbon_thunk.call: init %empty_tuple.type = call imports.%C1__carbon_thunk.decl(%addr.loc12) [concrete = <error>]
// CHECK:STDOUT:   %.loc12_36.2: init %C1 to %.loc12_36.1 = mark_in_place_init %C1__carbon_thunk.call [concrete = <error>]
// CHECK:STDOUT:   %.loc12_17: type = splice_block %C1.ref.loc12_17 [concrete = constants.%C1] {
// CHECK:STDOUT:     %Cpp.ref.loc12_11: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %N1.ref.loc12_14: <namespace> = name_ref N1, imports.%N1 [concrete = imports.%N1]
// CHECK:STDOUT:     %C1.ref.loc12_17: type = name_ref C1, imports.%C1.decl [concrete = constants.%C1]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc12_36.3: ref %C1 = temporary %.loc12_36.1, %.loc12_36.2 [concrete = <error>]
// CHECK:STDOUT:   %.loc12_36.4: %C1 = acquire_value %.loc12_36.3 [concrete = <error>]
// CHECK:STDOUT:   %c1: %C1 = value_binding c1, %.loc12_36.4 [concrete = <error>]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c2.patt: %pattern_type.846 = value_binding_pattern c2 [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc17_23: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %N2.ref.loc17_26: <namespace> = name_ref N2, imports.%N2 [concrete = imports.%N2]
// CHECK:STDOUT:   %C2.ref.loc17_29: type = name_ref C2, imports.%C2.decl [concrete = constants.%C2]
// CHECK:STDOUT:   %C2.ref.loc17_32: %C2.C2.cpp_overload_set.type = name_ref C2, imports.%C2.C2.cpp_overload_set.value [concrete = constants.%C2.C2.cpp_overload_set.value]
// CHECK:STDOUT:   %.loc17_36.1: ref %C2 = temporary_storage
// CHECK:STDOUT:   %addr.loc17: %ptr.51f = addr_of %.loc17_36.1
// CHECK:STDOUT:   %C2__carbon_thunk.call: init %empty_tuple.type = call imports.%C2__carbon_thunk.decl(%addr.loc17) [concrete = <error>]
// CHECK:STDOUT:   %.loc17_36.2: init %C2 to %.loc17_36.1 = mark_in_place_init %C2__carbon_thunk.call [concrete = <error>]
// CHECK:STDOUT:   %.loc17_17: type = splice_block %C2.ref.loc17_17 [concrete = constants.%C2] {
// CHECK:STDOUT:     %Cpp.ref.loc17_11: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %N2.ref.loc17_14: <namespace> = name_ref N2, imports.%N2 [concrete = imports.%N2]
// CHECK:STDOUT:     %C2.ref.loc17_17: type = name_ref C2, imports.%C2.decl [concrete = constants.%C2]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc17_36.3: ref %C2 = temporary %.loc17_36.1, %.loc17_36.2 [concrete = <error>]
// CHECK:STDOUT:   %.loc17_36.4: %C2 = acquire_value %.loc17_36.3 [concrete = <error>]
// CHECK:STDOUT:   %c2: %C2 = value_binding c2, %.loc17_36.4 [concrete = <error>]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c3.patt: %pattern_type.846 = value_binding_pattern c3 [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c1.ref.loc18: %C1 = name_ref c1, %c1 [concrete = <error>]
// CHECK:STDOUT:   %c2.ref.loc18: %C2 = name_ref c2, %c2 [concrete = <error>]
// CHECK:STDOUT:   %.loc18_33.1: ref %C2 = temporary_storage
// CHECK:STDOUT:   %.loc18_30: ref %C1 = value_as_ref %c1.ref.loc18
// CHECK:STDOUT:   %addr.loc18_33.1: %ptr.087 = addr_of %.loc18_30
// CHECK:STDOUT:   %.loc18_35: ref %C2 = value_as_ref %c2.ref.loc18
// CHECK:STDOUT:   %addr.loc18_33.2: %ptr.51f = addr_of %.loc18_35
// CHECK:STDOUT:   %addr.loc18_33.3: %ptr.51f = addr_of %.loc18_33.1
// CHECK:STDOUT:   %operator_Plus__carbon_thunk.call: init %empty_tuple.type = call imports.%operator_Plus__carbon_thunk.decl(%addr.loc18_33.1, %addr.loc18_33.2, %addr.loc18_33.3)
// CHECK:STDOUT:   %.loc18_33.2: init %C2 to %.loc18_33.1 = mark_in_place_init %operator_Plus__carbon_thunk.call
// CHECK:STDOUT:   %.loc18_24: type = splice_block %C2.ref.loc18 [concrete = constants.%C2] {
// CHECK:STDOUT:     %Cpp.ref.loc18: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %N2.ref.loc18: <namespace> = name_ref N2, imports.%N2 [concrete = imports.%N2]
// CHECK:STDOUT:     %C2.ref.loc18: type = name_ref C2, imports.%C2.decl [concrete = constants.%C2]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc18_33.3: ref %C2 = temporary %.loc18_33.1, %.loc18_33.2
// CHECK:STDOUT:   %.loc18_33.4: %C2 = acquire_value %.loc18_33.3
// CHECK:STDOUT:   %c3: %C2 = value_binding c3, %.loc18_33.4
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c4.patt: %pattern_type.846 = value_binding_pattern c4 [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c2.ref.loc19: %C2 = name_ref c2, %c2 [concrete = <error>]
// CHECK:STDOUT:   %c1.ref.loc19: %C1 = name_ref c1, %c1 [concrete = <error>]
// CHECK:STDOUT:   %.loc19_33.1: ref %C2 = temporary_storage
// CHECK:STDOUT:   %.loc19_30: ref %C2 = value_as_ref %c2.ref.loc19
// CHECK:STDOUT:   %addr.loc19_33.1: %ptr.51f = addr_of %.loc19_30
// CHECK:STDOUT:   %.loc19_35: ref %C1 = value_as_ref %c1.ref.loc19
// CHECK:STDOUT:   %addr.loc19_33.2: %ptr.087 = addr_of %.loc19_35
// CHECK:STDOUT:   %addr.loc19_33.3: %ptr.51f = addr_of %.loc19_33.1
// CHECK:STDOUT:   %operator_Minus__carbon_thunk.call: init %empty_tuple.type = call imports.%operator_Minus__carbon_thunk.decl(%addr.loc19_33.1, %addr.loc19_33.2, %addr.loc19_33.3)
// CHECK:STDOUT:   %.loc19_33.2: init %C2 to %.loc19_33.1 = mark_in_place_init %operator_Minus__carbon_thunk.call
// CHECK:STDOUT:   %.loc19_24: type = splice_block %C2.ref.loc19 [concrete = constants.%C2] {
// CHECK:STDOUT:     %Cpp.ref.loc19: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %N2.ref.loc19: <namespace> = name_ref N2, imports.%N2 [concrete = imports.%N2]
// CHECK:STDOUT:     %C2.ref.loc19: type = name_ref C2, imports.%C2.decl [concrete = constants.%C2]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc19_33.3: ref %C2 = temporary %.loc19_33.1, %.loc19_33.2
// CHECK:STDOUT:   %.loc19_33.4: %C2 = acquire_value %.loc19_33.3
// CHECK:STDOUT:   %c4: %C2 = value_binding c4, %.loc19_33.4
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %C2.Op.bound.loc19: <bound method> = bound_method %.loc19_33.3, constants.%C2.Op
// CHECK:STDOUT:   %Op.ref.loc19: %C2.cpp_destructor.type = name_ref Op, imports.%C2.cpp_destructor.decl [concrete = constants.%C2.cpp_destructor]
// CHECK:STDOUT:   %C2.cpp_destructor.bound.loc19: <bound method> = bound_method %.loc19_33.3, %Op.ref.loc19
// CHECK:STDOUT:   %C2.cpp_destructor.call.loc19: init %empty_tuple.type = call %C2.cpp_destructor.bound.loc19(%.loc19_33.3)
// CHECK:STDOUT:   %C2.Op.bound.loc18: <bound method> = bound_method %.loc18_33.3, constants.%C2.Op
// CHECK:STDOUT:   %Op.ref.loc18: %C2.cpp_destructor.type = name_ref Op, imports.%C2.cpp_destructor.decl [concrete = constants.%C2.cpp_destructor]
// CHECK:STDOUT:   %C2.cpp_destructor.bound.loc18: <bound method> = bound_method %.loc18_33.3, %Op.ref.loc18
// CHECK:STDOUT:   %C2.cpp_destructor.call.loc18: init %empty_tuple.type = call %C2.cpp_destructor.bound.loc18(%.loc18_33.3)
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
// CHECK:STDOUT:     %c1.patt: %pattern_type.69f = value_binding_pattern c1 [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc12_21: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %N.ref.loc12_24: <namespace> = name_ref N, imports.%N [concrete = imports.%N]
// CHECK:STDOUT:   %C.ref.loc12_26: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %C.ref.loc12_28: %C.C.cpp_overload_set.type = name_ref C, imports.%C.C.cpp_overload_set.value [concrete = constants.%C.C.cpp_overload_set.value]
// CHECK:STDOUT:   %.loc12_31.1: ref %C = temporary_storage
// CHECK:STDOUT:   %addr.loc12: %ptr.838 = addr_of %.loc12_31.1
// CHECK:STDOUT:   %C__carbon_thunk.call.loc12: init %empty_tuple.type = call imports.%C__carbon_thunk.decl(%addr.loc12) [concrete = <error>]
// CHECK:STDOUT:   %.loc12_31.2: init %C to %.loc12_31.1 = mark_in_place_init %C__carbon_thunk.call.loc12 [concrete = <error>]
// CHECK:STDOUT:   %.loc12_16: type = splice_block %C.ref.loc12_16 [concrete = constants.%C] {
// CHECK:STDOUT:     %Cpp.ref.loc12_11: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %N.ref.loc12_14: <namespace> = name_ref N, imports.%N [concrete = imports.%N]
// CHECK:STDOUT:     %C.ref.loc12_16: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc12_31.3: ref %C = temporary %.loc12_31.1, %.loc12_31.2 [concrete = <error>]
// CHECK:STDOUT:   %.loc12_31.4: %C = acquire_value %.loc12_31.3 [concrete = <error>]
// CHECK:STDOUT:   %c1: %C = value_binding c1, %.loc12_31.4 [concrete = <error>]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c2.patt: %pattern_type.69f = value_binding_pattern c2 [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc17_21: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %N.ref.loc17_24: <namespace> = name_ref N, imports.%N [concrete = imports.%N]
// CHECK:STDOUT:   %C.ref.loc17_26: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %C.ref.loc17_28: %C.C.cpp_overload_set.type = name_ref C, imports.%C.C.cpp_overload_set.value [concrete = constants.%C.C.cpp_overload_set.value]
// CHECK:STDOUT:   %.loc17_31.1: ref %C = temporary_storage
// CHECK:STDOUT:   %addr.loc17: %ptr.838 = addr_of %.loc17_31.1
// CHECK:STDOUT:   %C__carbon_thunk.call.loc17: init %empty_tuple.type = call imports.%C__carbon_thunk.decl(%addr.loc17) [concrete = <error>]
// CHECK:STDOUT:   %.loc17_31.2: init %C to %.loc17_31.1 = mark_in_place_init %C__carbon_thunk.call.loc17 [concrete = <error>]
// CHECK:STDOUT:   %.loc17_16: type = splice_block %C.ref.loc17_16 [concrete = constants.%C] {
// CHECK:STDOUT:     %Cpp.ref.loc17_11: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %N.ref.loc17_14: <namespace> = name_ref N, imports.%N [concrete = imports.%N]
// CHECK:STDOUT:     %C.ref.loc17_16: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc17_31.3: ref %C = temporary %.loc17_31.1, %.loc17_31.2 [concrete = <error>]
// CHECK:STDOUT:   %.loc17_31.4: %C = acquire_value %.loc17_31.3 [concrete = <error>]
// CHECK:STDOUT:   %c2: %C = value_binding c2, %.loc17_31.4 [concrete = <error>]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c3.patt: %pattern_type.69f = value_binding_pattern c3 [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c1.ref: %C = name_ref c1, %c1 [concrete = <error>]
// CHECK:STDOUT:   %c2.ref: %C = name_ref c2, %c2 [concrete = <error>]
// CHECK:STDOUT:   %.loc22: type = splice_block %C.ref.loc22 [concrete = constants.%C] {
// CHECK:STDOUT:     %Cpp.ref.loc22: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %N.ref.loc22: <namespace> = name_ref N, imports.%N [concrete = imports.%N]
// CHECK:STDOUT:     %C.ref.loc22: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c3: %C = value_binding c3, <error> [concrete = <error>]
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
// CHECK:STDOUT:   %operator_Plus__carbon_thunk.type: type = fn_type @operator_Plus__carbon_thunk [concrete]
// CHECK:STDOUT:   %operator_Plus__carbon_thunk: %operator_Plus__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %C.cpp_destructor.type: type = fn_type @C.cpp_destructor [concrete]
// CHECK:STDOUT:   %C.cpp_destructor: %C.cpp_destructor.type = struct_value () [concrete]
// CHECK:STDOUT:   %C.Op.type: type = fn_type @C.Op [concrete]
// CHECK:STDOUT:   %C.Op: %C.Op.type = struct_value () [concrete]
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
// CHECK:STDOUT:   %operator_Plus__carbon_thunk.decl: %operator_Plus__carbon_thunk.type = fn_decl @operator_Plus__carbon_thunk [concrete = constants.%operator_Plus__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.cpp_destructor.decl: %C.cpp_destructor.type = fn_decl @C.cpp_destructor [concrete = constants.%C.cpp_destructor] {
// CHECK:STDOUT:     %self.patt: %pattern_type.b28 = ref_binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.b28 = ref_param_pattern %self.patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %C = ref_param call_param0
// CHECK:STDOUT:     %self: ref %C = ref_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c1.patt: %pattern_type.b28 = value_binding_pattern c1 [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc12_21: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %O.ref.loc12_24: type = name_ref O, imports.%O.decl [concrete = constants.%O]
// CHECK:STDOUT:   %C.ref.loc12_26: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %C.ref.loc12_28: %C.C.cpp_overload_set.type = name_ref C, imports.%C.C.cpp_overload_set.value [concrete = constants.%C.C.cpp_overload_set.value]
// CHECK:STDOUT:   %.loc12_31.1: ref %C = temporary_storage
// CHECK:STDOUT:   %addr.loc12: %ptr.de2 = addr_of %.loc12_31.1
// CHECK:STDOUT:   %C__carbon_thunk.call.loc12: init %empty_tuple.type = call imports.%C__carbon_thunk.decl(%addr.loc12) [concrete = <error>]
// CHECK:STDOUT:   %.loc12_31.2: init %C to %.loc12_31.1 = mark_in_place_init %C__carbon_thunk.call.loc12 [concrete = <error>]
// CHECK:STDOUT:   %.loc12_16: type = splice_block %C.ref.loc12_16 [concrete = constants.%C] {
// CHECK:STDOUT:     %Cpp.ref.loc12_11: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %O.ref.loc12_14: type = name_ref O, imports.%O.decl [concrete = constants.%O]
// CHECK:STDOUT:     %C.ref.loc12_16: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc12_31.3: ref %C = temporary %.loc12_31.1, %.loc12_31.2 [concrete = <error>]
// CHECK:STDOUT:   %.loc12_31.4: %C = acquire_value %.loc12_31.3 [concrete = <error>]
// CHECK:STDOUT:   %c1: %C = value_binding c1, %.loc12_31.4 [concrete = <error>]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c2.patt: %pattern_type.b28 = value_binding_pattern c2 [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc17_21: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %O.ref.loc17_24: type = name_ref O, imports.%O.decl [concrete = constants.%O]
// CHECK:STDOUT:   %C.ref.loc17_26: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %C.ref.loc17_28: %C.C.cpp_overload_set.type = name_ref C, imports.%C.C.cpp_overload_set.value [concrete = constants.%C.C.cpp_overload_set.value]
// CHECK:STDOUT:   %.loc17_31.1: ref %C = temporary_storage
// CHECK:STDOUT:   %addr.loc17: %ptr.de2 = addr_of %.loc17_31.1
// CHECK:STDOUT:   %C__carbon_thunk.call.loc17: init %empty_tuple.type = call imports.%C__carbon_thunk.decl(%addr.loc17) [concrete = <error>]
// CHECK:STDOUT:   %.loc17_31.2: init %C to %.loc17_31.1 = mark_in_place_init %C__carbon_thunk.call.loc17 [concrete = <error>]
// CHECK:STDOUT:   %.loc17_16: type = splice_block %C.ref.loc17_16 [concrete = constants.%C] {
// CHECK:STDOUT:     %Cpp.ref.loc17_11: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %O.ref.loc17_14: type = name_ref O, imports.%O.decl [concrete = constants.%O]
// CHECK:STDOUT:     %C.ref.loc17_16: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc17_31.3: ref %C = temporary %.loc17_31.1, %.loc17_31.2 [concrete = <error>]
// CHECK:STDOUT:   %.loc17_31.4: %C = acquire_value %.loc17_31.3 [concrete = <error>]
// CHECK:STDOUT:   %c2: %C = value_binding c2, %.loc17_31.4 [concrete = <error>]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c3.patt: %pattern_type.b28 = value_binding_pattern c3 [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c1.ref: %C = name_ref c1, %c1 [concrete = <error>]
// CHECK:STDOUT:   %c2.ref: %C = name_ref c2, %c2 [concrete = <error>]
// CHECK:STDOUT:   %.loc18_31.1: ref %C = temporary_storage
// CHECK:STDOUT:   %.loc18_28: ref %C = value_as_ref %c1.ref
// CHECK:STDOUT:   %addr.loc18_31.1: %ptr.de2 = addr_of %.loc18_28
// CHECK:STDOUT:   %.loc18_33: ref %C = value_as_ref %c2.ref
// CHECK:STDOUT:   %addr.loc18_31.2: %ptr.de2 = addr_of %.loc18_33
// CHECK:STDOUT:   %addr.loc18_31.3: %ptr.de2 = addr_of %.loc18_31.1
// CHECK:STDOUT:   %operator_Plus__carbon_thunk.call: init %empty_tuple.type = call imports.%operator_Plus__carbon_thunk.decl(%addr.loc18_31.1, %addr.loc18_31.2, %addr.loc18_31.3)
// CHECK:STDOUT:   %.loc18_31.2: init %C to %.loc18_31.1 = mark_in_place_init %operator_Plus__carbon_thunk.call
// CHECK:STDOUT:   %.loc18_23: type = splice_block %C.ref.loc18 [concrete = constants.%C] {
// CHECK:STDOUT:     %Cpp.ref.loc18: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %O.ref.loc18: type = name_ref O, imports.%O.decl [concrete = constants.%O]
// CHECK:STDOUT:     %C.ref.loc18: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc18_31.3: ref %C = temporary %.loc18_31.1, %.loc18_31.2
// CHECK:STDOUT:   %.loc18_31.4: %C = acquire_value %.loc18_31.3
// CHECK:STDOUT:   %c3: %C = value_binding c3, %.loc18_31.4
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %C.Op.bound: <bound method> = bound_method %.loc18_31.3, constants.%C.Op
// CHECK:STDOUT:   %Op.ref: %C.cpp_destructor.type = name_ref Op, imports.%C.cpp_destructor.decl [concrete = constants.%C.cpp_destructor]
// CHECK:STDOUT:   %C.cpp_destructor.bound: <bound method> = bound_method %.loc18_31.3, %Op.ref
// CHECK:STDOUT:   %C.cpp_destructor.call: init %empty_tuple.type = call %C.cpp_destructor.bound(%.loc18_31.3)
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
// CHECK:STDOUT:   %operator_Plus__carbon_thunk.type: type = fn_type @operator_Plus__carbon_thunk [concrete]
// CHECK:STDOUT:   %operator_Plus__carbon_thunk: %operator_Plus__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %C.cpp_destructor.type: type = fn_type @C.cpp_destructor [concrete]
// CHECK:STDOUT:   %C.cpp_destructor: %C.cpp_destructor.type = struct_value () [concrete]
// CHECK:STDOUT:   %C.Op.type: type = fn_type @C.Op [concrete]
// CHECK:STDOUT:   %C.Op: %C.Op.type = struct_value () [concrete]
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
// CHECK:STDOUT:   %operator_Plus__carbon_thunk.decl: %operator_Plus__carbon_thunk.type = fn_decl @operator_Plus__carbon_thunk [concrete = constants.%operator_Plus__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.cpp_destructor.decl: %C.cpp_destructor.type = fn_decl @C.cpp_destructor [concrete = constants.%C.cpp_destructor] {
// CHECK:STDOUT:     %self.patt: %pattern_type.84b = ref_binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.84b = ref_param_pattern %self.patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %C = ref_param call_param0
// CHECK:STDOUT:     %self: ref %C = ref_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c1.patt: %pattern_type.84b = value_binding_pattern c1 [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc12_23: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %N.ref.loc12_26: <namespace> = name_ref N, imports.%N [concrete = imports.%N]
// CHECK:STDOUT:   %O.ref.loc12_28: type = name_ref O, imports.%O.decl [concrete = constants.%O]
// CHECK:STDOUT:   %C.ref.loc12_30: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %C.ref.loc12_32: %C.C.cpp_overload_set.type = name_ref C, imports.%C.C.cpp_overload_set.value [concrete = constants.%C.C.cpp_overload_set.value]
// CHECK:STDOUT:   %.loc12_35.1: ref %C = temporary_storage
// CHECK:STDOUT:   %addr.loc12: %ptr.4b2 = addr_of %.loc12_35.1
// CHECK:STDOUT:   %C__carbon_thunk.call.loc12: init %empty_tuple.type = call imports.%C__carbon_thunk.decl(%addr.loc12) [concrete = <error>]
// CHECK:STDOUT:   %.loc12_35.2: init %C to %.loc12_35.1 = mark_in_place_init %C__carbon_thunk.call.loc12 [concrete = <error>]
// CHECK:STDOUT:   %.loc12_18: type = splice_block %C.ref.loc12_18 [concrete = constants.%C] {
// CHECK:STDOUT:     %Cpp.ref.loc12_11: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %N.ref.loc12_14: <namespace> = name_ref N, imports.%N [concrete = imports.%N]
// CHECK:STDOUT:     %O.ref.loc12_16: type = name_ref O, imports.%O.decl [concrete = constants.%O]
// CHECK:STDOUT:     %C.ref.loc12_18: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc12_35.3: ref %C = temporary %.loc12_35.1, %.loc12_35.2 [concrete = <error>]
// CHECK:STDOUT:   %.loc12_35.4: %C = acquire_value %.loc12_35.3 [concrete = <error>]
// CHECK:STDOUT:   %c1: %C = value_binding c1, %.loc12_35.4 [concrete = <error>]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c2.patt: %pattern_type.84b = value_binding_pattern c2 [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc17_23: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %N.ref.loc17_26: <namespace> = name_ref N, imports.%N [concrete = imports.%N]
// CHECK:STDOUT:   %O.ref.loc17_28: type = name_ref O, imports.%O.decl [concrete = constants.%O]
// CHECK:STDOUT:   %C.ref.loc17_30: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %C.ref.loc17_32: %C.C.cpp_overload_set.type = name_ref C, imports.%C.C.cpp_overload_set.value [concrete = constants.%C.C.cpp_overload_set.value]
// CHECK:STDOUT:   %.loc17_35.1: ref %C = temporary_storage
// CHECK:STDOUT:   %addr.loc17: %ptr.4b2 = addr_of %.loc17_35.1
// CHECK:STDOUT:   %C__carbon_thunk.call.loc17: init %empty_tuple.type = call imports.%C__carbon_thunk.decl(%addr.loc17) [concrete = <error>]
// CHECK:STDOUT:   %.loc17_35.2: init %C to %.loc17_35.1 = mark_in_place_init %C__carbon_thunk.call.loc17 [concrete = <error>]
// CHECK:STDOUT:   %.loc17_18: type = splice_block %C.ref.loc17_18 [concrete = constants.%C] {
// CHECK:STDOUT:     %Cpp.ref.loc17_11: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %N.ref.loc17_14: <namespace> = name_ref N, imports.%N [concrete = imports.%N]
// CHECK:STDOUT:     %O.ref.loc17_16: type = name_ref O, imports.%O.decl [concrete = constants.%O]
// CHECK:STDOUT:     %C.ref.loc17_18: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc17_35.3: ref %C = temporary %.loc17_35.1, %.loc17_35.2 [concrete = <error>]
// CHECK:STDOUT:   %.loc17_35.4: %C = acquire_value %.loc17_35.3 [concrete = <error>]
// CHECK:STDOUT:   %c2: %C = value_binding c2, %.loc17_35.4 [concrete = <error>]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c3.patt: %pattern_type.84b = value_binding_pattern c3 [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c1.ref: %C = name_ref c1, %c1 [concrete = <error>]
// CHECK:STDOUT:   %c2.ref: %C = name_ref c2, %c2 [concrete = <error>]
// CHECK:STDOUT:   %.loc18_33.1: ref %C = temporary_storage
// CHECK:STDOUT:   %.loc18_30: ref %C = value_as_ref %c1.ref
// CHECK:STDOUT:   %addr.loc18_33.1: %ptr.4b2 = addr_of %.loc18_30
// CHECK:STDOUT:   %.loc18_35: ref %C = value_as_ref %c2.ref
// CHECK:STDOUT:   %addr.loc18_33.2: %ptr.4b2 = addr_of %.loc18_35
// CHECK:STDOUT:   %addr.loc18_33.3: %ptr.4b2 = addr_of %.loc18_33.1
// CHECK:STDOUT:   %operator_Plus__carbon_thunk.call: init %empty_tuple.type = call imports.%operator_Plus__carbon_thunk.decl(%addr.loc18_33.1, %addr.loc18_33.2, %addr.loc18_33.3)
// CHECK:STDOUT:   %.loc18_33.2: init %C to %.loc18_33.1 = mark_in_place_init %operator_Plus__carbon_thunk.call
// CHECK:STDOUT:   %.loc18_25: type = splice_block %C.ref.loc18 [concrete = constants.%C] {
// CHECK:STDOUT:     %Cpp.ref.loc18: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %N.ref.loc18: <namespace> = name_ref N, imports.%N [concrete = imports.%N]
// CHECK:STDOUT:     %O.ref.loc18: type = name_ref O, imports.%O.decl [concrete = constants.%O]
// CHECK:STDOUT:     %C.ref.loc18: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc18_33.3: ref %C = temporary %.loc18_33.1, %.loc18_33.2
// CHECK:STDOUT:   %.loc18_33.4: %C = acquire_value %.loc18_33.3
// CHECK:STDOUT:   %c3: %C = value_binding c3, %.loc18_33.4
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %C.Op.bound: <bound method> = bound_method %.loc18_33.3, constants.%C.Op
// CHECK:STDOUT:   %Op.ref: %C.cpp_destructor.type = name_ref Op, imports.%C.cpp_destructor.decl [concrete = constants.%C.cpp_destructor]
// CHECK:STDOUT:   %C.cpp_destructor.bound: <bound method> = bound_method %.loc18_33.3, %Op.ref
// CHECK:STDOUT:   %C.cpp_destructor.call: init %empty_tuple.type = call %C.cpp_destructor.bound(%.loc18_33.3)
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
// CHECK:STDOUT:   %C.cpp_operator.type.dab96a.1: type = fn_type @C.cpp_operator.1 [concrete]
// CHECK:STDOUT:   %C.cpp_operator.d21c75.1: %C.cpp_operator.type.dab96a.1 = struct_value () [concrete]
// CHECK:STDOUT:   %operator_Minus__carbon_thunk.type: type = fn_type @operator_Minus__carbon_thunk [concrete]
// CHECK:STDOUT:   %operator_Minus__carbon_thunk: %operator_Minus__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %C.cpp_operator.type.dab96a.2: type = fn_type @C.cpp_operator.2 [concrete]
// CHECK:STDOUT:   %C.cpp_operator.d21c75.2: %C.cpp_operator.type.dab96a.2 = struct_value () [concrete]
// CHECK:STDOUT:   %operator_Plus__carbon_thunk.type: type = fn_type @operator_Plus__carbon_thunk [concrete]
// CHECK:STDOUT:   %operator_Plus__carbon_thunk: %operator_Plus__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %C.cpp_destructor.type: type = fn_type @C.cpp_destructor [concrete]
// CHECK:STDOUT:   %C.cpp_destructor: %C.cpp_destructor.type = struct_value () [concrete]
// CHECK:STDOUT:   %C.Op.type: type = fn_type @C.Op [concrete]
// CHECK:STDOUT:   %C.Op: %C.Op.type = struct_value () [concrete]
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
// CHECK:STDOUT:   %C.cpp_operator.decl.828f43.1: %C.cpp_operator.type.dab96a.1 = fn_decl @C.cpp_operator.1 [concrete = constants.%C.cpp_operator.d21c75.1] {
// CHECK:STDOUT:     %self.patt: %pattern_type.217 = ref_binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.217 = ref_param_pattern %self.patt [concrete]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %C = ref_param call_param0
// CHECK:STDOUT:     %self: ref %C = ref_binding self, %self.param
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %operator_Minus__carbon_thunk.decl: %operator_Minus__carbon_thunk.type = fn_decl @operator_Minus__carbon_thunk [concrete = constants.%operator_Minus__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.cpp_operator.decl.828f43.2: %C.cpp_operator.type.dab96a.2 = fn_decl @C.cpp_operator.2 [concrete = constants.%C.cpp_operator.d21c75.2] {
// CHECK:STDOUT:     %self.patt: %pattern_type.217 = ref_binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.217 = ref_param_pattern %self.patt [concrete]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %C = ref_param call_param0
// CHECK:STDOUT:     %self: ref %C = ref_binding self, %self.param
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %operator_Plus__carbon_thunk.decl: %operator_Plus__carbon_thunk.type = fn_decl @operator_Plus__carbon_thunk [concrete = constants.%operator_Plus__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.cpp_destructor.decl: %C.cpp_destructor.type = fn_decl @C.cpp_destructor [concrete = constants.%C.cpp_destructor] {
// CHECK:STDOUT:     %self.patt: %pattern_type.217 = ref_binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.217 = ref_param_pattern %self.patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %C = ref_param call_param0
// CHECK:STDOUT:     %self: ref %C = ref_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c1.patt: %pattern_type.217 = ref_binding_pattern c1 [concrete]
// CHECK:STDOUT:     %c1.var_patt: %pattern_type.217 = var_pattern %c1.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c1.var: ref %C = var %c1.var_patt
// CHECK:STDOUT:   %Cpp.ref.loc12_19: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %C.ref.loc12_22: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %C.ref.loc12_24: %C.C.cpp_overload_set.type = name_ref C, imports.%C.C.cpp_overload_set.value [concrete = constants.%C.C.cpp_overload_set.value]
// CHECK:STDOUT:   %.loc12_3: ref %C = splice_block %c1.var {}
// CHECK:STDOUT:   %addr.loc12: %ptr.d9e = addr_of %.loc12_3
// CHECK:STDOUT:   %C__carbon_thunk.call: init %empty_tuple.type = call imports.%C__carbon_thunk.decl(%addr.loc12) [concrete = <error>]
// CHECK:STDOUT:   %.loc12_27: init %C to %.loc12_3 = mark_in_place_init %C__carbon_thunk.call [concrete = <error>]
// CHECK:STDOUT:   assign %c1.var, %.loc12_27
// CHECK:STDOUT:   %.loc12_14: type = splice_block %C.ref.loc12_14 [concrete = constants.%C] {
// CHECK:STDOUT:     %Cpp.ref.loc12_11: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %C.ref.loc12_14: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c1: ref %C = ref_binding c1, %c1.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c2.patt: %pattern_type.217 = ref_binding_pattern c2 [concrete]
// CHECK:STDOUT:     %c2.var_patt: %pattern_type.217 = var_pattern %c2.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c2.var: ref %C = var %c2.var_patt
// CHECK:STDOUT:   %c1.ref.loc13: ref %C = name_ref c1, %c1
// CHECK:STDOUT:   %C.cpp_operator.bound.loc13: <bound method> = bound_method %c1.ref.loc13, imports.%C.cpp_operator.decl.828f43.1
// CHECK:STDOUT:   %.loc13_3: ref %C = splice_block %c2.var {}
// CHECK:STDOUT:   %addr.loc13: %ptr.d9e = addr_of %.loc13_3
// CHECK:STDOUT:   %operator_Minus__carbon_thunk.call: init %empty_tuple.type = call imports.%operator_Minus__carbon_thunk.decl(%c1.ref.loc13, %addr.loc13)
// CHECK:STDOUT:   %.loc13_19: init %C to %.loc13_3 = mark_in_place_init %operator_Minus__carbon_thunk.call
// CHECK:STDOUT:   assign %c2.var, %.loc13_19
// CHECK:STDOUT:   %.loc13_14: type = splice_block %C.ref.loc13 [concrete = constants.%C] {
// CHECK:STDOUT:     %Cpp.ref.loc13: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %C.ref.loc13: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c2: ref %C = ref_binding c2, %c2.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c3.patt: %pattern_type.217 = ref_binding_pattern c3 [concrete]
// CHECK:STDOUT:     %c3.var_patt: %pattern_type.217 = var_pattern %c3.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c3.var: ref %C = var %c3.var_patt
// CHECK:STDOUT:   %c1.ref.loc14: ref %C = name_ref c1, %c1
// CHECK:STDOUT:   %c2.ref: ref %C = name_ref c2, %c2
// CHECK:STDOUT:   %C.cpp_operator.bound.loc14: <bound method> = bound_method %c1.ref.loc14, imports.%C.cpp_operator.decl.828f43.2
// CHECK:STDOUT:   %.loc14_3: ref %C = splice_block %c3.var {}
// CHECK:STDOUT:   %.loc14_31.1: %C = acquire_value %c2.ref
// CHECK:STDOUT:   %.loc14_31.2: ref %C = value_as_ref %.loc14_31.1
// CHECK:STDOUT:   %addr.loc14_29.1: %ptr.d9e = addr_of %.loc14_31.2
// CHECK:STDOUT:   %addr.loc14_29.2: %ptr.d9e = addr_of %.loc14_3
// CHECK:STDOUT:   %operator_Plus__carbon_thunk.call: init %empty_tuple.type = call imports.%operator_Plus__carbon_thunk.decl(%c1.ref.loc14, %addr.loc14_29.1, %addr.loc14_29.2)
// CHECK:STDOUT:   %.loc14_29: init %C to %.loc14_3 = mark_in_place_init %operator_Plus__carbon_thunk.call
// CHECK:STDOUT:   assign %c3.var, %.loc14_29
// CHECK:STDOUT:   %.loc14_21: type = splice_block %C.ref.loc14 [concrete = constants.%C] {
// CHECK:STDOUT:     %Cpp.ref.loc14: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %C.ref.loc14: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c3: ref %C = ref_binding c3, %c3.var
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %C.Op.bound.loc14: <bound method> = bound_method %c3.var, constants.%C.Op
// CHECK:STDOUT:   %Op.ref.loc14: %C.cpp_destructor.type = name_ref Op, imports.%C.cpp_destructor.decl [concrete = constants.%C.cpp_destructor]
// CHECK:STDOUT:   %C.cpp_destructor.bound.loc14: <bound method> = bound_method %c3.var, %Op.ref.loc14
// CHECK:STDOUT:   %C.cpp_destructor.call.loc14: init %empty_tuple.type = call %C.cpp_destructor.bound.loc14(%c3.var)
// CHECK:STDOUT:   %C.Op.bound.loc13: <bound method> = bound_method %c2.var, constants.%C.Op
// CHECK:STDOUT:   %Op.ref.loc13: %C.cpp_destructor.type = name_ref Op, imports.%C.cpp_destructor.decl [concrete = constants.%C.cpp_destructor]
// CHECK:STDOUT:   %C.cpp_destructor.bound.loc13: <bound method> = bound_method %c2.var, %Op.ref.loc13
// CHECK:STDOUT:   %C.cpp_destructor.call.loc13: init %empty_tuple.type = call %C.cpp_destructor.bound.loc13(%c2.var)
// CHECK:STDOUT:   %C.Op.bound.loc12: <bound method> = bound_method %c1.var, constants.%C.Op
// CHECK:STDOUT:   %Op.ref.loc12: %C.cpp_destructor.type = name_ref Op, imports.%C.cpp_destructor.decl [concrete = constants.%C.cpp_destructor]
// CHECK:STDOUT:   %C.cpp_destructor.bound.loc12: <bound method> = bound_method %c1.var, %Op.ref.loc12
// CHECK:STDOUT:   %C.cpp_destructor.call.loc12: init %empty_tuple.type = call %C.cpp_destructor.bound.loc12(%c1.var)
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
// CHECK:STDOUT:   %operator_Plus__carbon_thunk.type: type = fn_type @operator_Plus__carbon_thunk [concrete]
// CHECK:STDOUT:   %operator_Plus__carbon_thunk: %operator_Plus__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %operator_Plus__carbon_thunk.decl: %operator_Plus__carbon_thunk.type = fn_decl @operator_Plus__carbon_thunk [concrete = constants.%operator_Plus__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%x.param: %B) -> out %return.param: %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %x.ref.loc14_10: %B = name_ref x, %x
// CHECK:STDOUT:   %x.ref.loc14_14: %B = name_ref x, %x
// CHECK:STDOUT:   %.loc14_10: ref %B = value_as_ref %x.ref.loc14_10
// CHECK:STDOUT:   %addr.loc14_12.1: %ptr.a04 = addr_of %.loc14_10
// CHECK:STDOUT:   %.loc14_14: ref %B = value_as_ref %x.ref.loc14_14
// CHECK:STDOUT:   %addr.loc14_12.2: %ptr.a04 = addr_of %.loc14_14
// CHECK:STDOUT:   %operator_Plus__carbon_thunk.call: init %i32 = call imports.%operator_Plus__carbon_thunk.decl(%addr.loc14_12.1, %addr.loc14_12.2)
// CHECK:STDOUT:   return %operator_Plus__carbon_thunk.call
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_overloading.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %pattern_type.217: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %ptr.d9e: type = ptr_type %C [concrete]
// CHECK:STDOUT:   %operator_Plus__carbon_thunk.type: type = fn_type @operator_Plus__carbon_thunk [concrete]
// CHECK:STDOUT:   %operator_Plus__carbon_thunk: %operator_Plus__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %C.cpp_destructor.type: type = fn_type @C.cpp_destructor [concrete]
// CHECK:STDOUT:   %C.cpp_destructor: %C.cpp_destructor.type = struct_value () [concrete]
// CHECK:STDOUT:   %C.Op.type: type = fn_type @C.Op [concrete]
// CHECK:STDOUT:   %C.Op: %C.Op.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   %operator_Plus__carbon_thunk.decl: %operator_Plus__carbon_thunk.type = fn_decl @operator_Plus__carbon_thunk [concrete = constants.%operator_Plus__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.cpp_destructor.decl: %C.cpp_destructor.type = fn_decl @C.cpp_destructor [concrete = constants.%C.cpp_destructor] {
// CHECK:STDOUT:     %self.patt: %pattern_type.217 = ref_binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.217 = ref_param_pattern %self.patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %C = ref_param call_param0
// CHECK:STDOUT:     %self: ref %C = ref_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c3.patt: %pattern_type.217 = value_binding_pattern c3 [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c1.ref: %C = name_ref c1, %c1 [concrete = <error>]
// CHECK:STDOUT:   %c2.ref: %C = name_ref c2, %c2 [concrete = <error>]
// CHECK:STDOUT:   %.loc18_29.1: ref %C = temporary_storage
// CHECK:STDOUT:   %.loc18_26: ref %C = value_as_ref %c1.ref
// CHECK:STDOUT:   %addr.loc18_29.1: %ptr.d9e = addr_of %.loc18_26
// CHECK:STDOUT:   %.loc18_31: ref %C = value_as_ref %c2.ref
// CHECK:STDOUT:   %addr.loc18_29.2: %ptr.d9e = addr_of %.loc18_31
// CHECK:STDOUT:   %addr.loc18_29.3: %ptr.d9e = addr_of %.loc18_29.1
// CHECK:STDOUT:   %operator_Plus__carbon_thunk.call: init %empty_tuple.type = call imports.%operator_Plus__carbon_thunk.decl(%addr.loc18_29.1, %addr.loc18_29.2, %addr.loc18_29.3)
// CHECK:STDOUT:   %.loc18_29.2: init %C to %.loc18_29.1 = mark_in_place_init %operator_Plus__carbon_thunk.call
// CHECK:STDOUT:   %.loc18_21: type = splice_block %C.ref.loc18 [concrete = constants.%C] {
// CHECK:STDOUT:     %Cpp.ref.loc18: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %C.ref.loc18: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc18_29.3: ref %C = temporary %.loc18_29.1, %.loc18_29.2
// CHECK:STDOUT:   %.loc18_29.4: %C = acquire_value %.loc18_29.3
// CHECK:STDOUT:   %c3: %C = value_binding c3, %.loc18_29.4
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %C.Op.bound: <bound method> = bound_method %.loc18_29.3, constants.%C.Op
// CHECK:STDOUT:   %Op.ref: %C.cpp_destructor.type = name_ref Op, imports.%C.cpp_destructor.decl [concrete = constants.%C.cpp_destructor]
// CHECK:STDOUT:   %C.cpp_destructor.bound: <bound method> = bound_method %.loc18_29.3, %Op.ref
// CHECK:STDOUT:   %C.cpp_destructor.call: init %empty_tuple.type = call %C.cpp_destructor.bound(%.loc18_29.3)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
