// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/full.carbon
// EXTRA-ARGS: --target=x86_64-linux-gnu --clang-arg=-std=c++20
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interop/cpp/operators/operators.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interop/cpp/operators/operators.carbon

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
  var c1: Cpp.C = Cpp.C.C();
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
  let c1: Cpp.C = Cpp.C.C();
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
  var c1: Cpp.C = Cpp.C.C();
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
  var c1: Cpp.C = Cpp.C.C();
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
  let c1: Cpp.N.C = Cpp.N.C.C();
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
  let c1: Cpp.N1.C1 = Cpp.N1.C1.C1();
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
  let c1: Cpp.N.C = Cpp.N.C.C();
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
  let c1: Cpp.O.C = Cpp.O.C.C();
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
  let c1: Cpp.N.O.C = Cpp.N.O.C.C();
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
  var c1: Cpp.C = Cpp.C.C();
  var c2: Cpp.C = -c1;
  var unused c3: Cpp.C = c1 + c2;
  //@dump-sem-ir-end
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
  var complete: Cpp.Complete = Cpp.Complete.Complete();
  // CHECK:STDERR: fail_import_incomplete_binary.carbon:[[@LINE+8]]:35: error: looking up a C++ operator with incomplete operand type `Cpp.Incomplete` [IncompleteOperandTypeInCppOperatorLookup]
  // CHECK:STDERR:   let unused result_binary: i32 = complete + *Cpp.CreateIncomplete();
  // CHECK:STDERR:                                   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_import_incomplete_binary.carbon:[[@LINE-7]]:10: in file included here [InCppInclude]
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
  var complete: Cpp.Complete = Cpp.Complete.Complete();
  // CHECK:STDERR: fail_incomplete_operand_carbon_type.carbon:[[@LINE+7]]:28: error: looking up a C++ operator with incomplete operand type `Incomplete` [IncompleteOperandTypeInCppOperatorLookup]
  // CHECK:STDERR:   let unused result: i32 = *CreateIncomplete() + complete;
  // CHECK:STDERR:                            ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_incomplete_operand_carbon_type.carbon:[[@LINE-8]]:1: note: class was forward declared here [ClassForwardDeclaredHere]
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
  let c1: Cpp.C = Cpp.C.C();
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
// CHECK:STDOUT:   %C__carbon_thunk.type.65f120.1: type = fn_type @C__carbon_thunk.1 [concrete]
// CHECK:STDOUT:   %C__carbon_thunk.d98342.1: %C__carbon_thunk.type.65f120.1 = struct_value () [concrete]
// CHECK:STDOUT:   %operator_PlusPlus__carbon_thunk.type: type = fn_type @operator_PlusPlus__carbon_thunk [concrete]
// CHECK:STDOUT:   %operator_PlusPlus__carbon_thunk: %operator_PlusPlus__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %operator_MinusMinus__carbon_thunk.type: type = fn_type @operator_MinusMinus__carbon_thunk [concrete]
// CHECK:STDOUT:   %operator_MinusMinus__carbon_thunk: %operator_MinusMinus__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %operator_Minus__carbon_thunk.type: type = fn_type @operator_Minus__carbon_thunk [concrete]
// CHECK:STDOUT:   %operator_Minus__carbon_thunk: %operator_Minus__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %C.C.type.065f0c.2: type = fn_type @C.C.2 [concrete]
// CHECK:STDOUT:   %C.C.6522d8.2: %C.C.type.065f0c.2 = struct_value () [concrete]
// CHECK:STDOUT:   %const.057: type = const_type %C [concrete]
// CHECK:STDOUT:   %ptr.c92: type = ptr_type %const.057 [concrete]
// CHECK:STDOUT:   %C__carbon_thunk.type.65f120.2: type = fn_type @C__carbon_thunk.2 [concrete]
// CHECK:STDOUT:   %C__carbon_thunk.d98342.2: %C__carbon_thunk.type.65f120.2 = struct_value () [concrete]
// CHECK:STDOUT:   %C.Op.type.ea1bc9.1: type = fn_type @C.Op.1 [concrete]
// CHECK:STDOUT:   %C.Op.068411.1: %C.Op.type.ea1bc9.1 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.681: <witness> = custom_witness (%C.Op.068411.1), @Copy [concrete]
// CHECK:STDOUT:   %Copy.facet.3b9: %Copy.type = facet_value %C, (%custom_witness.681) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.530: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet.3b9) [concrete]
// CHECK:STDOUT:   %.a81: type = fn_type_with_self_type %Copy.WithSelf.Op.type.530, %Copy.facet.3b9 [concrete]
// CHECK:STDOUT:   %operator_Tilde__carbon_thunk.type: type = fn_type @operator_Tilde__carbon_thunk [concrete]
// CHECK:STDOUT:   %operator_Tilde__carbon_thunk: %operator_Tilde__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %C.cpp_destructor.type: type = fn_type @C.cpp_destructor [concrete]
// CHECK:STDOUT:   %C.cpp_destructor: %C.cpp_destructor.type = struct_value () [concrete]
// CHECK:STDOUT:   %C.Op.type.ea1bc9.2: type = fn_type @C.Op.2 [concrete]
// CHECK:STDOUT:   %C.Op.068411.2: %C.Op.type.ea1bc9.2 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   %C.C.cpp_overload_set.value: %C.C.cpp_overload_set.type = cpp_overload_set_value @C.C.cpp_overload_set [concrete = constants.%C.C.cpp_overload_set.value]
// CHECK:STDOUT:   %C__carbon_thunk.decl.8acdfe.1: %C__carbon_thunk.type.65f120.1 = fn_decl @C__carbon_thunk.1 [concrete = constants.%C__carbon_thunk.d98342.1] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %operator_PlusPlus__carbon_thunk.decl: %operator_PlusPlus__carbon_thunk.type = fn_decl @operator_PlusPlus__carbon_thunk [concrete = constants.%operator_PlusPlus__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %operator_MinusMinus__carbon_thunk.decl: %operator_MinusMinus__carbon_thunk.type = fn_decl @operator_MinusMinus__carbon_thunk [concrete = constants.%operator_MinusMinus__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %operator_Minus__carbon_thunk.decl: %operator_Minus__carbon_thunk.type = fn_decl @operator_Minus__carbon_thunk [concrete = constants.%operator_Minus__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.C.decl.041267.2: %C.C.type.065f0c.2 = fn_decl @C.C.2 [concrete = constants.%C.C.6522d8.2] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C__carbon_thunk.decl.8acdfe.2: %C__carbon_thunk.type.65f120.2 = fn_decl @C__carbon_thunk.2 [concrete = constants.%C__carbon_thunk.d98342.2] {
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
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
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
// CHECK:STDOUT:   %Cpp.ref.loc8_18: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %C.ref.loc8_21: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %C.ref.loc8_23: %C.C.cpp_overload_set.type = name_ref C, imports.%C.C.cpp_overload_set.value [concrete = constants.%C.C.cpp_overload_set.value]
// CHECK:STDOUT:   %.loc8_3: ref %C = splice_block %c.var {}
// CHECK:STDOUT:   %addr.loc8: %ptr.d9e = addr_of %.loc8_3
// CHECK:STDOUT:   %C__carbon_thunk.call.loc8: init %empty_tuple.type = call imports.%C__carbon_thunk.decl.8acdfe.1(%addr.loc8)
// CHECK:STDOUT:   %.loc8_26: init %C to %.loc8_3 = mark_in_place_init %C__carbon_thunk.call.loc8
// CHECK:STDOUT:   assign %c.var, %.loc8_26
// CHECK:STDOUT:   %.loc8_13: type = splice_block %C.ref.loc8_13 [concrete = constants.%C] {
// CHECK:STDOUT:     %Cpp.ref.loc8_10: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %C.ref.loc8_13: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c: ref %C = ref_binding c, %c.var
// CHECK:STDOUT:   %c.ref.loc11: ref %C = name_ref c, %c
// CHECK:STDOUT:   %operator_PlusPlus__carbon_thunk.call: ref %C = call imports.%operator_PlusPlus__carbon_thunk.decl(%c.ref.loc11)
// CHECK:STDOUT:   %c.ref.loc12: ref %C = name_ref c, %c
// CHECK:STDOUT:   %operator_MinusMinus__carbon_thunk.call: ref %C = call imports.%operator_MinusMinus__carbon_thunk.decl(%c.ref.loc12)
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %minus.patt: %pattern_type.217 = value_binding_pattern minus [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c.ref.loc15: ref %C = name_ref c, %c
// CHECK:STDOUT:   %.loc15_29.1: ref %C = temporary_storage
// CHECK:STDOUT:   %.loc15_30.1: %C = acquire_value %c.ref.loc15
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %impl.elem0.loc15: %.a81 = impl_witness_access constants.%custom_witness.681, element0 [concrete = constants.%C.Op.068411.1]
// CHECK:STDOUT:   %bound_method.loc15: <bound method> = bound_method %.loc15_30.1, %impl.elem0.loc15
// CHECK:STDOUT:   %.loc15_30.2: ref %C = temporary_storage
// CHECK:STDOUT:   %Op.ref.loc15_30: %C.C.type.065f0c.2 = name_ref Op, imports.%C.C.decl.041267.2 [concrete = constants.%C.C.6522d8.2]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %.loc15_30.3: ref %C = value_as_ref %.loc15_30.1
// CHECK:STDOUT:   %addr.loc15_30.1: %ptr.d9e = addr_of %.loc15_30.3
// CHECK:STDOUT:   %.loc15_30.4: %ptr.c92 = as_compatible %addr.loc15_30.1
// CHECK:STDOUT:   %.loc15_30.5: %ptr.c92 = converted %addr.loc15_30.1, %.loc15_30.4
// CHECK:STDOUT:   %addr.loc15_30.2: %ptr.d9e = addr_of %operand.var.1
// CHECK:STDOUT:   %C__carbon_thunk.call.loc15: init %empty_tuple.type = call imports.%C__carbon_thunk.decl.8acdfe.2(%.loc15_30.5, %addr.loc15_30.2)
// CHECK:STDOUT:   %.loc15_30.6: init %C to %operand.var.1 = mark_in_place_init %C__carbon_thunk.call.loc15
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %addr.loc15_29.1: %ptr.d9e = addr_of %operand.var.1
// CHECK:STDOUT:   %addr.loc15_29.2: %ptr.d9e = addr_of %.loc15_29.1
// CHECK:STDOUT:   %operator_Minus__carbon_thunk.call: init %empty_tuple.type = call imports.%operator_Minus__carbon_thunk.decl(%addr.loc15_29.1, %addr.loc15_29.2)
// CHECK:STDOUT:   %.loc15_29.2: init %C to %.loc15_29.1 = mark_in_place_init %operator_Minus__carbon_thunk.call
// CHECK:STDOUT:   %.loc15_24: type = splice_block %C.ref.loc15 [concrete = constants.%C] {
// CHECK:STDOUT:     %Cpp.ref.loc15: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %C.ref.loc15: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc15_29.3: ref %C = temporary %.loc15_29.1, %.loc15_29.2
// CHECK:STDOUT:   %.loc15_29.4: %C = acquire_value %.loc15_29.3
// CHECK:STDOUT:   %minus: %C = value_binding minus, %.loc15_29.4
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %complement.patt: %pattern_type.217 = value_binding_pattern complement [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c.ref.loc18: ref %C = name_ref c, %c
// CHECK:STDOUT:   %.loc18_34.1: ref %C = temporary_storage
// CHECK:STDOUT:   %.loc18_35.1: %C = acquire_value %c.ref.loc18
// CHECK:STDOUT:   %impl.elem0.loc18: %.a81 = impl_witness_access constants.%custom_witness.681, element0 [concrete = constants.%C.Op.068411.1]
// CHECK:STDOUT:   %bound_method.loc18: <bound method> = bound_method %.loc18_35.1, %impl.elem0.loc18
// CHECK:STDOUT:   %.loc18_35.2: ref %C = temporary_storage
// CHECK:STDOUT:   %Op.ref.loc18_35: %C.C.type.065f0c.2 = name_ref Op, imports.%C.C.decl.041267.2 [concrete = constants.%C.C.6522d8.2]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %.loc18_35.3: ref %C = value_as_ref %.loc18_35.1
// CHECK:STDOUT:   %addr.loc18_35.1: %ptr.d9e = addr_of %.loc18_35.3
// CHECK:STDOUT:   %.loc18_35.4: %ptr.c92 = as_compatible %addr.loc18_35.1
// CHECK:STDOUT:   %.loc18_35.5: %ptr.c92 = converted %addr.loc18_35.1, %.loc18_35.4
// CHECK:STDOUT:   %addr.loc18_35.2: %ptr.d9e = addr_of %operand.var.2
// CHECK:STDOUT:   %C__carbon_thunk.call.loc18: init %empty_tuple.type = call imports.%C__carbon_thunk.decl.8acdfe.2(%.loc18_35.5, %addr.loc18_35.2)
// CHECK:STDOUT:   %.loc18_35.6: init %C to %operand.var.2 = mark_in_place_init %C__carbon_thunk.call.loc18
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %addr.loc18_34.1: %ptr.d9e = addr_of %operand.var.2
// CHECK:STDOUT:   %addr.loc18_34.2: %ptr.d9e = addr_of %.loc18_34.1
// CHECK:STDOUT:   %operator_Tilde__carbon_thunk.call: init %empty_tuple.type = call imports.%operator_Tilde__carbon_thunk.decl(%addr.loc18_34.1, %addr.loc18_34.2)
// CHECK:STDOUT:   %.loc18_34.2: init %C to %.loc18_34.1 = mark_in_place_init %operator_Tilde__carbon_thunk.call
// CHECK:STDOUT:   %.loc18_29: type = splice_block %C.ref.loc18 [concrete = constants.%C] {
// CHECK:STDOUT:     %Cpp.ref.loc18: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %C.ref.loc18: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc18_34.3: ref %C = temporary %.loc18_34.1, %.loc18_34.2
// CHECK:STDOUT:   %.loc18_34.4: %C = acquire_value %.loc18_34.3
// CHECK:STDOUT:   %complement: %C = value_binding complement, %.loc18_34.4
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %C.Op.bound.loc18: <bound method> = bound_method %.loc18_34.3, constants.%C.Op.068411.2
// CHECK:STDOUT:   %Op.ref.loc18_34: %C.cpp_destructor.type = name_ref Op, imports.%C.cpp_destructor.decl [concrete = constants.%C.cpp_destructor]
// CHECK:STDOUT:   %C.cpp_destructor.bound.loc18: <bound method> = bound_method %.loc18_34.3, %Op.ref.loc18_34
// CHECK:STDOUT:   %C.cpp_destructor.call.loc18: init %empty_tuple.type = call %C.cpp_destructor.bound.loc18(%.loc18_34.3)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %C.Op.bound.loc15: <bound method> = bound_method %.loc15_29.3, constants.%C.Op.068411.2
// CHECK:STDOUT:   %Op.ref.loc15_29: %C.cpp_destructor.type = name_ref Op, imports.%C.cpp_destructor.decl [concrete = constants.%C.cpp_destructor]
// CHECK:STDOUT:   %C.cpp_destructor.bound.loc15: <bound method> = bound_method %.loc15_29.3, %Op.ref.loc15_29
// CHECK:STDOUT:   %C.cpp_destructor.call.loc15: init %empty_tuple.type = call %C.cpp_destructor.bound.loc15(%.loc15_29.3)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %C.Op.bound.loc8: <bound method> = bound_method %c.var, constants.%C.Op.068411.2
// CHECK:STDOUT:   %Op.ref.loc8: %C.cpp_destructor.type = name_ref Op, imports.%C.cpp_destructor.decl [concrete = constants.%C.cpp_destructor]
// CHECK:STDOUT:   %C.cpp_destructor.bound.loc8: <bound method> = bound_method %c.var, %Op.ref.loc8
// CHECK:STDOUT:   %C.cpp_destructor.call.loc8: init %empty_tuple.type = call %C.cpp_destructor.bound.loc8(%c.var)
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
// CHECK:STDOUT:   %C__carbon_thunk.type.65f120.1: type = fn_type @C__carbon_thunk.1 [concrete]
// CHECK:STDOUT:   %C__carbon_thunk.d98342.1: %C__carbon_thunk.type.65f120.1 = struct_value () [concrete]
// CHECK:STDOUT:   %operator_Plus__carbon_thunk.type: type = fn_type @operator_Plus__carbon_thunk [concrete]
// CHECK:STDOUT:   %operator_Plus__carbon_thunk: %operator_Plus__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %C.C.type.065f0c.2: type = fn_type @C.C.2 [concrete]
// CHECK:STDOUT:   %C.C.6522d8.2: %C.C.type.065f0c.2 = struct_value () [concrete]
// CHECK:STDOUT:   %const.057: type = const_type %C [concrete]
// CHECK:STDOUT:   %ptr.c92: type = ptr_type %const.057 [concrete]
// CHECK:STDOUT:   %C__carbon_thunk.type.65f120.2: type = fn_type @C__carbon_thunk.2 [concrete]
// CHECK:STDOUT:   %C__carbon_thunk.d98342.2: %C__carbon_thunk.type.65f120.2 = struct_value () [concrete]
// CHECK:STDOUT:   %C.Op.type.ea1bc9.1: type = fn_type @C.Op.1 [concrete]
// CHECK:STDOUT:   %C.Op.068411.1: %C.Op.type.ea1bc9.1 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.681: <witness> = custom_witness (%C.Op.068411.1), @Copy [concrete]
// CHECK:STDOUT:   %Copy.facet.3b9: %Copy.type = facet_value %C, (%custom_witness.681) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.530: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet.3b9) [concrete]
// CHECK:STDOUT:   %.a81: type = fn_type_with_self_type %Copy.WithSelf.Op.type.530, %Copy.facet.3b9 [concrete]
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
// CHECK:STDOUT:   %int_5.64b: Core.IntLiteral = int_value 5 [concrete]
// CHECK:STDOUT:   %operator_GreaterGreater__carbon_thunk.type: type = fn_type @operator_GreaterGreater__carbon_thunk [concrete]
// CHECK:STDOUT:   %operator_GreaterGreater__carbon_thunk: %operator_GreaterGreater__carbon_thunk.type = struct_value () [concrete]
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
// CHECK:STDOUT:   %operator_LessLessEqual__carbon_thunk.type: type = fn_type @operator_LessLessEqual__carbon_thunk [concrete]
// CHECK:STDOUT:   %operator_LessLessEqual__carbon_thunk: %operator_LessLessEqual__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %operator_GreaterGreaterEqual__carbon_thunk.type: type = fn_type @operator_GreaterGreaterEqual__carbon_thunk [concrete]
// CHECK:STDOUT:   %operator_GreaterGreaterEqual__carbon_thunk: %operator_GreaterGreaterEqual__carbon_thunk.type = struct_value () [concrete]
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
// CHECK:STDOUT:   %operator_Subscript__carbon_thunk.type: type = fn_type @operator_Subscript__carbon_thunk [concrete]
// CHECK:STDOUT:   %operator_Subscript__carbon_thunk: %operator_Subscript__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.bae255.3: type = fn_type @Destroy.Op.loc45 [concrete]
// CHECK:STDOUT:   %Destroy.Op.651ba6.3: %Destroy.Op.type.bae255.3 = struct_value () [concrete]
// CHECK:STDOUT:   %C.cpp_destructor.type: type = fn_type @C.cpp_destructor [concrete]
// CHECK:STDOUT:   %C.cpp_destructor: %C.cpp_destructor.type = struct_value () [concrete]
// CHECK:STDOUT:   %C.Op.type.ea1bc9.2: type = fn_type @C.Op.2 [concrete]
// CHECK:STDOUT:   %C.Op.068411.2: %C.Op.type.ea1bc9.2 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   %C.C.cpp_overload_set.value: %C.C.cpp_overload_set.type = cpp_overload_set_value @C.C.cpp_overload_set [concrete = constants.%C.C.cpp_overload_set.value]
// CHECK:STDOUT:   %C__carbon_thunk.decl.8acdfe.1: %C__carbon_thunk.type.65f120.1 = fn_decl @C__carbon_thunk.1 [concrete = constants.%C__carbon_thunk.d98342.1] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %operator_Plus__carbon_thunk.decl: %operator_Plus__carbon_thunk.type = fn_decl @operator_Plus__carbon_thunk [concrete = constants.%operator_Plus__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.C.decl.041267.2: %C.C.type.065f0c.2 = fn_decl @C.C.2 [concrete = constants.%C.C.6522d8.2] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C__carbon_thunk.decl.8acdfe.2: %C__carbon_thunk.type.65f120.2 = fn_decl @C__carbon_thunk.2 [concrete = constants.%C__carbon_thunk.d98342.2] {
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
// CHECK:STDOUT:   %operator_LessLessEqual__carbon_thunk.decl: %operator_LessLessEqual__carbon_thunk.type = fn_decl @operator_LessLessEqual__carbon_thunk [concrete = constants.%operator_LessLessEqual__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %operator_GreaterGreaterEqual__carbon_thunk.decl: %operator_GreaterGreaterEqual__carbon_thunk.type = fn_decl @operator_GreaterGreaterEqual__carbon_thunk [concrete = constants.%operator_GreaterGreaterEqual__carbon_thunk] {
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
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %operator_Subscript__carbon_thunk.decl: %operator_Subscript__carbon_thunk.type = fn_decl @operator_Subscript__carbon_thunk [concrete = constants.%operator_Subscript__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.cpp_destructor.decl: %C.cpp_destructor.type = fn_decl @C.cpp_destructor [concrete = constants.%C.cpp_destructor] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
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
// CHECK:STDOUT:   %Cpp.ref.loc8_19: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %C.ref.loc8_22: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %C.ref.loc8_24: %C.C.cpp_overload_set.type = name_ref C, imports.%C.C.cpp_overload_set.value [concrete = constants.%C.C.cpp_overload_set.value]
// CHECK:STDOUT:   %.loc8_3: ref %C = splice_block %c1.var {}
// CHECK:STDOUT:   %addr.loc8: %ptr.d9e = addr_of %.loc8_3
// CHECK:STDOUT:   %C__carbon_thunk.call.loc8: init %empty_tuple.type = call imports.%C__carbon_thunk.decl.8acdfe.1(%addr.loc8)
// CHECK:STDOUT:   %.loc8_27: init %C to %.loc8_3 = mark_in_place_init %C__carbon_thunk.call.loc8
// CHECK:STDOUT:   assign %c1.var, %.loc8_27
// CHECK:STDOUT:   %.loc8_14: type = splice_block %C.ref.loc8_14 [concrete = constants.%C] {
// CHECK:STDOUT:     %Cpp.ref.loc8_11: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %C.ref.loc8_14: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c1: ref %C = ref_binding c1, %c1.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c2.patt: %pattern_type.217 = ref_binding_pattern c2 [concrete]
// CHECK:STDOUT:     %c2.var_patt: %pattern_type.217 = var_pattern %c2.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c2.var: ref %C = var %c2.var_patt
// CHECK:STDOUT:   %Cpp.ref.loc9_19: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %C.ref.loc9_22: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %C.ref.loc9_24: %C.C.cpp_overload_set.type = name_ref C, imports.%C.C.cpp_overload_set.value [concrete = constants.%C.C.cpp_overload_set.value]
// CHECK:STDOUT:   %.loc9_3: ref %C = splice_block %c2.var {}
// CHECK:STDOUT:   %addr.loc9: %ptr.d9e = addr_of %.loc9_3
// CHECK:STDOUT:   %C__carbon_thunk.call.loc9: init %empty_tuple.type = call imports.%C__carbon_thunk.decl.8acdfe.1(%addr.loc9)
// CHECK:STDOUT:   %.loc9_27: init %C to %.loc9_3 = mark_in_place_init %C__carbon_thunk.call.loc9
// CHECK:STDOUT:   assign %c2.var, %.loc9_27
// CHECK:STDOUT:   %.loc9_14: type = splice_block %C.ref.loc9_14 [concrete = constants.%C] {
// CHECK:STDOUT:     %Cpp.ref.loc9_11: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %C.ref.loc9_14: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c2: ref %C = ref_binding c2, %c2.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %addition.patt: %pattern_type.217 = value_binding_pattern addition [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c1.ref.loc12: ref %C = name_ref c1, %c1
// CHECK:STDOUT:   %c2.ref.loc12: ref %C = name_ref c2, %c2
// CHECK:STDOUT:   %.loc12_35.1: ref %C = temporary_storage
// CHECK:STDOUT:   %.loc12_32.1: %C = acquire_value %c1.ref.loc12
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %impl.elem0.loc12_32: %.a81 = impl_witness_access constants.%custom_witness.681, element0 [concrete = constants.%C.Op.068411.1]
// CHECK:STDOUT:   %bound_method.loc12_32: <bound method> = bound_method %.loc12_32.1, %impl.elem0.loc12_32
// CHECK:STDOUT:   %.loc12_32.2: ref %C = temporary_storage
// CHECK:STDOUT:   %Op.ref.loc12_32: %C.C.type.065f0c.2 = name_ref Op, imports.%C.C.decl.041267.2 [concrete = constants.%C.C.6522d8.2]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %.loc12_32.3: ref %C = value_as_ref %.loc12_32.1
// CHECK:STDOUT:   %addr.loc12_32.1: %ptr.d9e = addr_of %.loc12_32.3
// CHECK:STDOUT:   %.loc12_32.4: %ptr.c92 = as_compatible %addr.loc12_32.1
// CHECK:STDOUT:   %.loc12_32.5: %ptr.c92 = converted %addr.loc12_32.1, %.loc12_32.4
// CHECK:STDOUT:   %addr.loc12_32.2: %ptr.d9e = addr_of %lhs.var.1
// CHECK:STDOUT:   %C__carbon_thunk.call.loc12_32: init %empty_tuple.type = call imports.%C__carbon_thunk.decl.8acdfe.2(%.loc12_32.5, %addr.loc12_32.2)
// CHECK:STDOUT:   %.loc12_32.6: init %C to %lhs.var.1 = mark_in_place_init %C__carbon_thunk.call.loc12_32
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %.loc12_37.1: %C = acquire_value %c2.ref.loc12
// CHECK:STDOUT:   %impl.elem0.loc12_37: %.a81 = impl_witness_access constants.%custom_witness.681, element0 [concrete = constants.%C.Op.068411.1]
// CHECK:STDOUT:   %bound_method.loc12_37: <bound method> = bound_method %.loc12_37.1, %impl.elem0.loc12_37
// CHECK:STDOUT:   %.loc12_37.2: ref %C = temporary_storage
// CHECK:STDOUT:   %Op.ref.loc12_37: %C.C.type.065f0c.2 = name_ref Op, imports.%C.C.decl.041267.2 [concrete = constants.%C.C.6522d8.2]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %.loc12_37.3: ref %C = value_as_ref %.loc12_37.1
// CHECK:STDOUT:   %addr.loc12_37.1: %ptr.d9e = addr_of %.loc12_37.3
// CHECK:STDOUT:   %.loc12_37.4: %ptr.c92 = as_compatible %addr.loc12_37.1
// CHECK:STDOUT:   %.loc12_37.5: %ptr.c92 = converted %addr.loc12_37.1, %.loc12_37.4
// CHECK:STDOUT:   %addr.loc12_37.2: %ptr.d9e = addr_of %rhs.var.1
// CHECK:STDOUT:   %C__carbon_thunk.call.loc12_37: init %empty_tuple.type = call imports.%C__carbon_thunk.decl.8acdfe.2(%.loc12_37.5, %addr.loc12_37.2)
// CHECK:STDOUT:   %.loc12_37.6: init %C to %rhs.var.1 = mark_in_place_init %C__carbon_thunk.call.loc12_37
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %addr.loc12_35.1: %ptr.d9e = addr_of %lhs.var.1
// CHECK:STDOUT:   %addr.loc12_35.2: %ptr.d9e = addr_of %rhs.var.1
// CHECK:STDOUT:   %addr.loc12_35.3: %ptr.d9e = addr_of %.loc12_35.1
// CHECK:STDOUT:   %operator_Plus__carbon_thunk.call: init %empty_tuple.type = call imports.%operator_Plus__carbon_thunk.decl(%addr.loc12_35.1, %addr.loc12_35.2, %addr.loc12_35.3)
// CHECK:STDOUT:   %.loc12_35.2: init %C to %.loc12_35.1 = mark_in_place_init %operator_Plus__carbon_thunk.call
// CHECK:STDOUT:   %.loc12_27: type = splice_block %C.ref.loc12 [concrete = constants.%C] {
// CHECK:STDOUT:     %Cpp.ref.loc12: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %C.ref.loc12: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc12_35.3: ref %C = temporary %.loc12_35.1, %.loc12_35.2
// CHECK:STDOUT:   %.loc12_35.4: %C = acquire_value %.loc12_35.3
// CHECK:STDOUT:   %addition: %C = value_binding addition, %.loc12_35.4
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %subtraction.patt: %pattern_type.217 = value_binding_pattern subtraction [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c1.ref.loc13: ref %C = name_ref c1, %c1
// CHECK:STDOUT:   %c2.ref.loc13: ref %C = name_ref c2, %c2
// CHECK:STDOUT:   %.loc13_38.1: ref %C = temporary_storage
// CHECK:STDOUT:   %.loc13_35.1: %C = acquire_value %c1.ref.loc13
// CHECK:STDOUT:   %impl.elem0.loc13_35: %.a81 = impl_witness_access constants.%custom_witness.681, element0 [concrete = constants.%C.Op.068411.1]
// CHECK:STDOUT:   %bound_method.loc13_35: <bound method> = bound_method %.loc13_35.1, %impl.elem0.loc13_35
// CHECK:STDOUT:   %.loc13_35.2: ref %C = temporary_storage
// CHECK:STDOUT:   %Op.ref.loc13_35: %C.C.type.065f0c.2 = name_ref Op, imports.%C.C.decl.041267.2 [concrete = constants.%C.C.6522d8.2]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %.loc13_35.3: ref %C = value_as_ref %.loc13_35.1
// CHECK:STDOUT:   %addr.loc13_35.1: %ptr.d9e = addr_of %.loc13_35.3
// CHECK:STDOUT:   %.loc13_35.4: %ptr.c92 = as_compatible %addr.loc13_35.1
// CHECK:STDOUT:   %.loc13_35.5: %ptr.c92 = converted %addr.loc13_35.1, %.loc13_35.4
// CHECK:STDOUT:   %addr.loc13_35.2: %ptr.d9e = addr_of %lhs.var.2
// CHECK:STDOUT:   %C__carbon_thunk.call.loc13_35: init %empty_tuple.type = call imports.%C__carbon_thunk.decl.8acdfe.2(%.loc13_35.5, %addr.loc13_35.2)
// CHECK:STDOUT:   %.loc13_35.6: init %C to %lhs.var.2 = mark_in_place_init %C__carbon_thunk.call.loc13_35
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %.loc13_40.1: %C = acquire_value %c2.ref.loc13
// CHECK:STDOUT:   %impl.elem0.loc13_40: %.a81 = impl_witness_access constants.%custom_witness.681, element0 [concrete = constants.%C.Op.068411.1]
// CHECK:STDOUT:   %bound_method.loc13_40: <bound method> = bound_method %.loc13_40.1, %impl.elem0.loc13_40
// CHECK:STDOUT:   %.loc13_40.2: ref %C = temporary_storage
// CHECK:STDOUT:   %Op.ref.loc13_40: %C.C.type.065f0c.2 = name_ref Op, imports.%C.C.decl.041267.2 [concrete = constants.%C.C.6522d8.2]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %.loc13_40.3: ref %C = value_as_ref %.loc13_40.1
// CHECK:STDOUT:   %addr.loc13_40.1: %ptr.d9e = addr_of %.loc13_40.3
// CHECK:STDOUT:   %.loc13_40.4: %ptr.c92 = as_compatible %addr.loc13_40.1
// CHECK:STDOUT:   %.loc13_40.5: %ptr.c92 = converted %addr.loc13_40.1, %.loc13_40.4
// CHECK:STDOUT:   %addr.loc13_40.2: %ptr.d9e = addr_of %rhs.var.2
// CHECK:STDOUT:   %C__carbon_thunk.call.loc13_40: init %empty_tuple.type = call imports.%C__carbon_thunk.decl.8acdfe.2(%.loc13_40.5, %addr.loc13_40.2)
// CHECK:STDOUT:   %.loc13_40.6: init %C to %rhs.var.2 = mark_in_place_init %C__carbon_thunk.call.loc13_40
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %addr.loc13_38.1: %ptr.d9e = addr_of %lhs.var.2
// CHECK:STDOUT:   %addr.loc13_38.2: %ptr.d9e = addr_of %rhs.var.2
// CHECK:STDOUT:   %addr.loc13_38.3: %ptr.d9e = addr_of %.loc13_38.1
// CHECK:STDOUT:   %operator_Minus__carbon_thunk.call: init %empty_tuple.type = call imports.%operator_Minus__carbon_thunk.decl(%addr.loc13_38.1, %addr.loc13_38.2, %addr.loc13_38.3)
// CHECK:STDOUT:   %.loc13_38.2: init %C to %.loc13_38.1 = mark_in_place_init %operator_Minus__carbon_thunk.call
// CHECK:STDOUT:   %.loc13_30: type = splice_block %C.ref.loc13 [concrete = constants.%C] {
// CHECK:STDOUT:     %Cpp.ref.loc13: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %C.ref.loc13: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc13_38.3: ref %C = temporary %.loc13_38.1, %.loc13_38.2
// CHECK:STDOUT:   %.loc13_38.4: %C = acquire_value %.loc13_38.3
// CHECK:STDOUT:   %subtraction: %C = value_binding subtraction, %.loc13_38.4
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %multiplication.patt: %pattern_type.217 = value_binding_pattern multiplication [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c1.ref.loc14: ref %C = name_ref c1, %c1
// CHECK:STDOUT:   %c2.ref.loc14: ref %C = name_ref c2, %c2
// CHECK:STDOUT:   %.loc14_41.1: ref %C = temporary_storage
// CHECK:STDOUT:   %.loc14_38.1: %C = acquire_value %c1.ref.loc14
// CHECK:STDOUT:   %impl.elem0.loc14_38: %.a81 = impl_witness_access constants.%custom_witness.681, element0 [concrete = constants.%C.Op.068411.1]
// CHECK:STDOUT:   %bound_method.loc14_38: <bound method> = bound_method %.loc14_38.1, %impl.elem0.loc14_38
// CHECK:STDOUT:   %.loc14_38.2: ref %C = temporary_storage
// CHECK:STDOUT:   %Op.ref.loc14_38: %C.C.type.065f0c.2 = name_ref Op, imports.%C.C.decl.041267.2 [concrete = constants.%C.C.6522d8.2]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %.loc14_38.3: ref %C = value_as_ref %.loc14_38.1
// CHECK:STDOUT:   %addr.loc14_38.1: %ptr.d9e = addr_of %.loc14_38.3
// CHECK:STDOUT:   %.loc14_38.4: %ptr.c92 = as_compatible %addr.loc14_38.1
// CHECK:STDOUT:   %.loc14_38.5: %ptr.c92 = converted %addr.loc14_38.1, %.loc14_38.4
// CHECK:STDOUT:   %addr.loc14_38.2: %ptr.d9e = addr_of %lhs.var.3
// CHECK:STDOUT:   %C__carbon_thunk.call.loc14_38: init %empty_tuple.type = call imports.%C__carbon_thunk.decl.8acdfe.2(%.loc14_38.5, %addr.loc14_38.2)
// CHECK:STDOUT:   %.loc14_38.6: init %C to %lhs.var.3 = mark_in_place_init %C__carbon_thunk.call.loc14_38
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %.loc14_43.1: %C = acquire_value %c2.ref.loc14
// CHECK:STDOUT:   %impl.elem0.loc14_43: %.a81 = impl_witness_access constants.%custom_witness.681, element0 [concrete = constants.%C.Op.068411.1]
// CHECK:STDOUT:   %bound_method.loc14_43: <bound method> = bound_method %.loc14_43.1, %impl.elem0.loc14_43
// CHECK:STDOUT:   %.loc14_43.2: ref %C = temporary_storage
// CHECK:STDOUT:   %Op.ref.loc14_43: %C.C.type.065f0c.2 = name_ref Op, imports.%C.C.decl.041267.2 [concrete = constants.%C.C.6522d8.2]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %.loc14_43.3: ref %C = value_as_ref %.loc14_43.1
// CHECK:STDOUT:   %addr.loc14_43.1: %ptr.d9e = addr_of %.loc14_43.3
// CHECK:STDOUT:   %.loc14_43.4: %ptr.c92 = as_compatible %addr.loc14_43.1
// CHECK:STDOUT:   %.loc14_43.5: %ptr.c92 = converted %addr.loc14_43.1, %.loc14_43.4
// CHECK:STDOUT:   %addr.loc14_43.2: %ptr.d9e = addr_of %rhs.var.3
// CHECK:STDOUT:   %C__carbon_thunk.call.loc14_43: init %empty_tuple.type = call imports.%C__carbon_thunk.decl.8acdfe.2(%.loc14_43.5, %addr.loc14_43.2)
// CHECK:STDOUT:   %.loc14_43.6: init %C to %rhs.var.3 = mark_in_place_init %C__carbon_thunk.call.loc14_43
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %addr.loc14_41.1: %ptr.d9e = addr_of %lhs.var.3
// CHECK:STDOUT:   %addr.loc14_41.2: %ptr.d9e = addr_of %rhs.var.3
// CHECK:STDOUT:   %addr.loc14_41.3: %ptr.d9e = addr_of %.loc14_41.1
// CHECK:STDOUT:   %operator_Star__carbon_thunk.call: init %empty_tuple.type = call imports.%operator_Star__carbon_thunk.decl(%addr.loc14_41.1, %addr.loc14_41.2, %addr.loc14_41.3)
// CHECK:STDOUT:   %.loc14_41.2: init %C to %.loc14_41.1 = mark_in_place_init %operator_Star__carbon_thunk.call
// CHECK:STDOUT:   %.loc14_33: type = splice_block %C.ref.loc14 [concrete = constants.%C] {
// CHECK:STDOUT:     %Cpp.ref.loc14: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %C.ref.loc14: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc14_41.3: ref %C = temporary %.loc14_41.1, %.loc14_41.2
// CHECK:STDOUT:   %.loc14_41.4: %C = acquire_value %.loc14_41.3
// CHECK:STDOUT:   %multiplication: %C = value_binding multiplication, %.loc14_41.4
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %division.patt: %pattern_type.217 = value_binding_pattern division [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c1.ref.loc15: ref %C = name_ref c1, %c1
// CHECK:STDOUT:   %c2.ref.loc15: ref %C = name_ref c2, %c2
// CHECK:STDOUT:   %.loc15_35.1: ref %C = temporary_storage
// CHECK:STDOUT:   %.loc15_32.1: %C = acquire_value %c1.ref.loc15
// CHECK:STDOUT:   %impl.elem0.loc15_32: %.a81 = impl_witness_access constants.%custom_witness.681, element0 [concrete = constants.%C.Op.068411.1]
// CHECK:STDOUT:   %bound_method.loc15_32: <bound method> = bound_method %.loc15_32.1, %impl.elem0.loc15_32
// CHECK:STDOUT:   %.loc15_32.2: ref %C = temporary_storage
// CHECK:STDOUT:   %Op.ref.loc15_32: %C.C.type.065f0c.2 = name_ref Op, imports.%C.C.decl.041267.2 [concrete = constants.%C.C.6522d8.2]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %.loc15_32.3: ref %C = value_as_ref %.loc15_32.1
// CHECK:STDOUT:   %addr.loc15_32.1: %ptr.d9e = addr_of %.loc15_32.3
// CHECK:STDOUT:   %.loc15_32.4: %ptr.c92 = as_compatible %addr.loc15_32.1
// CHECK:STDOUT:   %.loc15_32.5: %ptr.c92 = converted %addr.loc15_32.1, %.loc15_32.4
// CHECK:STDOUT:   %addr.loc15_32.2: %ptr.d9e = addr_of %lhs.var.4
// CHECK:STDOUT:   %C__carbon_thunk.call.loc15_32: init %empty_tuple.type = call imports.%C__carbon_thunk.decl.8acdfe.2(%.loc15_32.5, %addr.loc15_32.2)
// CHECK:STDOUT:   %.loc15_32.6: init %C to %lhs.var.4 = mark_in_place_init %C__carbon_thunk.call.loc15_32
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %.loc15_37.1: %C = acquire_value %c2.ref.loc15
// CHECK:STDOUT:   %impl.elem0.loc15_37: %.a81 = impl_witness_access constants.%custom_witness.681, element0 [concrete = constants.%C.Op.068411.1]
// CHECK:STDOUT:   %bound_method.loc15_37: <bound method> = bound_method %.loc15_37.1, %impl.elem0.loc15_37
// CHECK:STDOUT:   %.loc15_37.2: ref %C = temporary_storage
// CHECK:STDOUT:   %Op.ref.loc15_37: %C.C.type.065f0c.2 = name_ref Op, imports.%C.C.decl.041267.2 [concrete = constants.%C.C.6522d8.2]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %.loc15_37.3: ref %C = value_as_ref %.loc15_37.1
// CHECK:STDOUT:   %addr.loc15_37.1: %ptr.d9e = addr_of %.loc15_37.3
// CHECK:STDOUT:   %.loc15_37.4: %ptr.c92 = as_compatible %addr.loc15_37.1
// CHECK:STDOUT:   %.loc15_37.5: %ptr.c92 = converted %addr.loc15_37.1, %.loc15_37.4
// CHECK:STDOUT:   %addr.loc15_37.2: %ptr.d9e = addr_of %rhs.var.4
// CHECK:STDOUT:   %C__carbon_thunk.call.loc15_37: init %empty_tuple.type = call imports.%C__carbon_thunk.decl.8acdfe.2(%.loc15_37.5, %addr.loc15_37.2)
// CHECK:STDOUT:   %.loc15_37.6: init %C to %rhs.var.4 = mark_in_place_init %C__carbon_thunk.call.loc15_37
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %addr.loc15_35.1: %ptr.d9e = addr_of %lhs.var.4
// CHECK:STDOUT:   %addr.loc15_35.2: %ptr.d9e = addr_of %rhs.var.4
// CHECK:STDOUT:   %addr.loc15_35.3: %ptr.d9e = addr_of %.loc15_35.1
// CHECK:STDOUT:   %operator_Slash__carbon_thunk.call: init %empty_tuple.type = call imports.%operator_Slash__carbon_thunk.decl(%addr.loc15_35.1, %addr.loc15_35.2, %addr.loc15_35.3)
// CHECK:STDOUT:   %.loc15_35.2: init %C to %.loc15_35.1 = mark_in_place_init %operator_Slash__carbon_thunk.call
// CHECK:STDOUT:   %.loc15_27: type = splice_block %C.ref.loc15 [concrete = constants.%C] {
// CHECK:STDOUT:     %Cpp.ref.loc15: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %C.ref.loc15: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc15_35.3: ref %C = temporary %.loc15_35.1, %.loc15_35.2
// CHECK:STDOUT:   %.loc15_35.4: %C = acquire_value %.loc15_35.3
// CHECK:STDOUT:   %division: %C = value_binding division, %.loc15_35.4
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %modulo.patt: %pattern_type.217 = value_binding_pattern modulo [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c1.ref.loc16: ref %C = name_ref c1, %c1
// CHECK:STDOUT:   %c2.ref.loc16: ref %C = name_ref c2, %c2
// CHECK:STDOUT:   %.loc16_33.1: ref %C = temporary_storage
// CHECK:STDOUT:   %.loc16_30.1: %C = acquire_value %c1.ref.loc16
// CHECK:STDOUT:   %impl.elem0.loc16_30: %.a81 = impl_witness_access constants.%custom_witness.681, element0 [concrete = constants.%C.Op.068411.1]
// CHECK:STDOUT:   %bound_method.loc16_30: <bound method> = bound_method %.loc16_30.1, %impl.elem0.loc16_30
// CHECK:STDOUT:   %.loc16_30.2: ref %C = temporary_storage
// CHECK:STDOUT:   %Op.ref.loc16_30: %C.C.type.065f0c.2 = name_ref Op, imports.%C.C.decl.041267.2 [concrete = constants.%C.C.6522d8.2]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %.loc16_30.3: ref %C = value_as_ref %.loc16_30.1
// CHECK:STDOUT:   %addr.loc16_30.1: %ptr.d9e = addr_of %.loc16_30.3
// CHECK:STDOUT:   %.loc16_30.4: %ptr.c92 = as_compatible %addr.loc16_30.1
// CHECK:STDOUT:   %.loc16_30.5: %ptr.c92 = converted %addr.loc16_30.1, %.loc16_30.4
// CHECK:STDOUT:   %addr.loc16_30.2: %ptr.d9e = addr_of %lhs.var.5
// CHECK:STDOUT:   %C__carbon_thunk.call.loc16_30: init %empty_tuple.type = call imports.%C__carbon_thunk.decl.8acdfe.2(%.loc16_30.5, %addr.loc16_30.2)
// CHECK:STDOUT:   %.loc16_30.6: init %C to %lhs.var.5 = mark_in_place_init %C__carbon_thunk.call.loc16_30
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %.loc16_35.1: %C = acquire_value %c2.ref.loc16
// CHECK:STDOUT:   %impl.elem0.loc16_35: %.a81 = impl_witness_access constants.%custom_witness.681, element0 [concrete = constants.%C.Op.068411.1]
// CHECK:STDOUT:   %bound_method.loc16_35: <bound method> = bound_method %.loc16_35.1, %impl.elem0.loc16_35
// CHECK:STDOUT:   %.loc16_35.2: ref %C = temporary_storage
// CHECK:STDOUT:   %Op.ref.loc16_35: %C.C.type.065f0c.2 = name_ref Op, imports.%C.C.decl.041267.2 [concrete = constants.%C.C.6522d8.2]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %.loc16_35.3: ref %C = value_as_ref %.loc16_35.1
// CHECK:STDOUT:   %addr.loc16_35.1: %ptr.d9e = addr_of %.loc16_35.3
// CHECK:STDOUT:   %.loc16_35.4: %ptr.c92 = as_compatible %addr.loc16_35.1
// CHECK:STDOUT:   %.loc16_35.5: %ptr.c92 = converted %addr.loc16_35.1, %.loc16_35.4
// CHECK:STDOUT:   %addr.loc16_35.2: %ptr.d9e = addr_of %rhs.var.5
// CHECK:STDOUT:   %C__carbon_thunk.call.loc16_35: init %empty_tuple.type = call imports.%C__carbon_thunk.decl.8acdfe.2(%.loc16_35.5, %addr.loc16_35.2)
// CHECK:STDOUT:   %.loc16_35.6: init %C to %rhs.var.5 = mark_in_place_init %C__carbon_thunk.call.loc16_35
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %addr.loc16_33.1: %ptr.d9e = addr_of %lhs.var.5
// CHECK:STDOUT:   %addr.loc16_33.2: %ptr.d9e = addr_of %rhs.var.5
// CHECK:STDOUT:   %addr.loc16_33.3: %ptr.d9e = addr_of %.loc16_33.1
// CHECK:STDOUT:   %operator_Percent__carbon_thunk.call: init %empty_tuple.type = call imports.%operator_Percent__carbon_thunk.decl(%addr.loc16_33.1, %addr.loc16_33.2, %addr.loc16_33.3)
// CHECK:STDOUT:   %.loc16_33.2: init %C to %.loc16_33.1 = mark_in_place_init %operator_Percent__carbon_thunk.call
// CHECK:STDOUT:   %.loc16_25: type = splice_block %C.ref.loc16 [concrete = constants.%C] {
// CHECK:STDOUT:     %Cpp.ref.loc16: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %C.ref.loc16: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc16_33.3: ref %C = temporary %.loc16_33.1, %.loc16_33.2
// CHECK:STDOUT:   %.loc16_33.4: %C = acquire_value %.loc16_33.3
// CHECK:STDOUT:   %modulo: %C = value_binding modulo, %.loc16_33.4
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %bitwise_and.patt: %pattern_type.217 = value_binding_pattern bitwise_and [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c1.ref.loc19: ref %C = name_ref c1, %c1
// CHECK:STDOUT:   %c2.ref.loc19: ref %C = name_ref c2, %c2
// CHECK:STDOUT:   %.loc19_38.1: ref %C = temporary_storage
// CHECK:STDOUT:   %.loc19_35.1: %C = acquire_value %c1.ref.loc19
// CHECK:STDOUT:   %impl.elem0.loc19_35: %.a81 = impl_witness_access constants.%custom_witness.681, element0 [concrete = constants.%C.Op.068411.1]
// CHECK:STDOUT:   %bound_method.loc19_35: <bound method> = bound_method %.loc19_35.1, %impl.elem0.loc19_35
// CHECK:STDOUT:   %.loc19_35.2: ref %C = temporary_storage
// CHECK:STDOUT:   %Op.ref.loc19_35: %C.C.type.065f0c.2 = name_ref Op, imports.%C.C.decl.041267.2 [concrete = constants.%C.C.6522d8.2]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %.loc19_35.3: ref %C = value_as_ref %.loc19_35.1
// CHECK:STDOUT:   %addr.loc19_35.1: %ptr.d9e = addr_of %.loc19_35.3
// CHECK:STDOUT:   %.loc19_35.4: %ptr.c92 = as_compatible %addr.loc19_35.1
// CHECK:STDOUT:   %.loc19_35.5: %ptr.c92 = converted %addr.loc19_35.1, %.loc19_35.4
// CHECK:STDOUT:   %addr.loc19_35.2: %ptr.d9e = addr_of %lhs.var.6
// CHECK:STDOUT:   %C__carbon_thunk.call.loc19_35: init %empty_tuple.type = call imports.%C__carbon_thunk.decl.8acdfe.2(%.loc19_35.5, %addr.loc19_35.2)
// CHECK:STDOUT:   %.loc19_35.6: init %C to %lhs.var.6 = mark_in_place_init %C__carbon_thunk.call.loc19_35
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %.loc19_40.1: %C = acquire_value %c2.ref.loc19
// CHECK:STDOUT:   %impl.elem0.loc19_40: %.a81 = impl_witness_access constants.%custom_witness.681, element0 [concrete = constants.%C.Op.068411.1]
// CHECK:STDOUT:   %bound_method.loc19_40: <bound method> = bound_method %.loc19_40.1, %impl.elem0.loc19_40
// CHECK:STDOUT:   %.loc19_40.2: ref %C = temporary_storage
// CHECK:STDOUT:   %Op.ref.loc19_40: %C.C.type.065f0c.2 = name_ref Op, imports.%C.C.decl.041267.2 [concrete = constants.%C.C.6522d8.2]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %.loc19_40.3: ref %C = value_as_ref %.loc19_40.1
// CHECK:STDOUT:   %addr.loc19_40.1: %ptr.d9e = addr_of %.loc19_40.3
// CHECK:STDOUT:   %.loc19_40.4: %ptr.c92 = as_compatible %addr.loc19_40.1
// CHECK:STDOUT:   %.loc19_40.5: %ptr.c92 = converted %addr.loc19_40.1, %.loc19_40.4
// CHECK:STDOUT:   %addr.loc19_40.2: %ptr.d9e = addr_of %rhs.var.6
// CHECK:STDOUT:   %C__carbon_thunk.call.loc19_40: init %empty_tuple.type = call imports.%C__carbon_thunk.decl.8acdfe.2(%.loc19_40.5, %addr.loc19_40.2)
// CHECK:STDOUT:   %.loc19_40.6: init %C to %rhs.var.6 = mark_in_place_init %C__carbon_thunk.call.loc19_40
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %addr.loc19_38.1: %ptr.d9e = addr_of %lhs.var.6
// CHECK:STDOUT:   %addr.loc19_38.2: %ptr.d9e = addr_of %rhs.var.6
// CHECK:STDOUT:   %addr.loc19_38.3: %ptr.d9e = addr_of %.loc19_38.1
// CHECK:STDOUT:   %operator_Amp__carbon_thunk.call: init %empty_tuple.type = call imports.%operator_Amp__carbon_thunk.decl(%addr.loc19_38.1, %addr.loc19_38.2, %addr.loc19_38.3)
// CHECK:STDOUT:   %.loc19_38.2: init %C to %.loc19_38.1 = mark_in_place_init %operator_Amp__carbon_thunk.call
// CHECK:STDOUT:   %.loc19_30: type = splice_block %C.ref.loc19 [concrete = constants.%C] {
// CHECK:STDOUT:     %Cpp.ref.loc19: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %C.ref.loc19: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc19_38.3: ref %C = temporary %.loc19_38.1, %.loc19_38.2
// CHECK:STDOUT:   %.loc19_38.4: %C = acquire_value %.loc19_38.3
// CHECK:STDOUT:   %bitwise_and: %C = value_binding bitwise_and, %.loc19_38.4
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %bitwise_or.patt: %pattern_type.217 = value_binding_pattern bitwise_or [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c1.ref.loc20: ref %C = name_ref c1, %c1
// CHECK:STDOUT:   %c2.ref.loc20: ref %C = name_ref c2, %c2
// CHECK:STDOUT:   %.loc20_37.1: ref %C = temporary_storage
// CHECK:STDOUT:   %.loc20_34.1: %C = acquire_value %c1.ref.loc20
// CHECK:STDOUT:   %impl.elem0.loc20_34: %.a81 = impl_witness_access constants.%custom_witness.681, element0 [concrete = constants.%C.Op.068411.1]
// CHECK:STDOUT:   %bound_method.loc20_34: <bound method> = bound_method %.loc20_34.1, %impl.elem0.loc20_34
// CHECK:STDOUT:   %.loc20_34.2: ref %C = temporary_storage
// CHECK:STDOUT:   %Op.ref.loc20_34: %C.C.type.065f0c.2 = name_ref Op, imports.%C.C.decl.041267.2 [concrete = constants.%C.C.6522d8.2]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %.loc20_34.3: ref %C = value_as_ref %.loc20_34.1
// CHECK:STDOUT:   %addr.loc20_34.1: %ptr.d9e = addr_of %.loc20_34.3
// CHECK:STDOUT:   %.loc20_34.4: %ptr.c92 = as_compatible %addr.loc20_34.1
// CHECK:STDOUT:   %.loc20_34.5: %ptr.c92 = converted %addr.loc20_34.1, %.loc20_34.4
// CHECK:STDOUT:   %addr.loc20_34.2: %ptr.d9e = addr_of %lhs.var.7
// CHECK:STDOUT:   %C__carbon_thunk.call.loc20_34: init %empty_tuple.type = call imports.%C__carbon_thunk.decl.8acdfe.2(%.loc20_34.5, %addr.loc20_34.2)
// CHECK:STDOUT:   %.loc20_34.6: init %C to %lhs.var.7 = mark_in_place_init %C__carbon_thunk.call.loc20_34
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %.loc20_39.1: %C = acquire_value %c2.ref.loc20
// CHECK:STDOUT:   %impl.elem0.loc20_39: %.a81 = impl_witness_access constants.%custom_witness.681, element0 [concrete = constants.%C.Op.068411.1]
// CHECK:STDOUT:   %bound_method.loc20_39: <bound method> = bound_method %.loc20_39.1, %impl.elem0.loc20_39
// CHECK:STDOUT:   %.loc20_39.2: ref %C = temporary_storage
// CHECK:STDOUT:   %Op.ref.loc20_39: %C.C.type.065f0c.2 = name_ref Op, imports.%C.C.decl.041267.2 [concrete = constants.%C.C.6522d8.2]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %.loc20_39.3: ref %C = value_as_ref %.loc20_39.1
// CHECK:STDOUT:   %addr.loc20_39.1: %ptr.d9e = addr_of %.loc20_39.3
// CHECK:STDOUT:   %.loc20_39.4: %ptr.c92 = as_compatible %addr.loc20_39.1
// CHECK:STDOUT:   %.loc20_39.5: %ptr.c92 = converted %addr.loc20_39.1, %.loc20_39.4
// CHECK:STDOUT:   %addr.loc20_39.2: %ptr.d9e = addr_of %rhs.var.7
// CHECK:STDOUT:   %C__carbon_thunk.call.loc20_39: init %empty_tuple.type = call imports.%C__carbon_thunk.decl.8acdfe.2(%.loc20_39.5, %addr.loc20_39.2)
// CHECK:STDOUT:   %.loc20_39.6: init %C to %rhs.var.7 = mark_in_place_init %C__carbon_thunk.call.loc20_39
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %addr.loc20_37.1: %ptr.d9e = addr_of %lhs.var.7
// CHECK:STDOUT:   %addr.loc20_37.2: %ptr.d9e = addr_of %rhs.var.7
// CHECK:STDOUT:   %addr.loc20_37.3: %ptr.d9e = addr_of %.loc20_37.1
// CHECK:STDOUT:   %operator_Pipe__carbon_thunk.call: init %empty_tuple.type = call imports.%operator_Pipe__carbon_thunk.decl(%addr.loc20_37.1, %addr.loc20_37.2, %addr.loc20_37.3)
// CHECK:STDOUT:   %.loc20_37.2: init %C to %.loc20_37.1 = mark_in_place_init %operator_Pipe__carbon_thunk.call
// CHECK:STDOUT:   %.loc20_29: type = splice_block %C.ref.loc20 [concrete = constants.%C] {
// CHECK:STDOUT:     %Cpp.ref.loc20: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %C.ref.loc20: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc20_37.3: ref %C = temporary %.loc20_37.1, %.loc20_37.2
// CHECK:STDOUT:   %.loc20_37.4: %C = acquire_value %.loc20_37.3
// CHECK:STDOUT:   %bitwise_or: %C = value_binding bitwise_or, %.loc20_37.4
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %bitwise_xor.patt: %pattern_type.217 = value_binding_pattern bitwise_xor [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c1.ref.loc21: ref %C = name_ref c1, %c1
// CHECK:STDOUT:   %c2.ref.loc21: ref %C = name_ref c2, %c2
// CHECK:STDOUT:   %.loc21_38.1: ref %C = temporary_storage
// CHECK:STDOUT:   %.loc21_35.1: %C = acquire_value %c1.ref.loc21
// CHECK:STDOUT:   %impl.elem0.loc21_35: %.a81 = impl_witness_access constants.%custom_witness.681, element0 [concrete = constants.%C.Op.068411.1]
// CHECK:STDOUT:   %bound_method.loc21_35: <bound method> = bound_method %.loc21_35.1, %impl.elem0.loc21_35
// CHECK:STDOUT:   %.loc21_35.2: ref %C = temporary_storage
// CHECK:STDOUT:   %Op.ref.loc21_35: %C.C.type.065f0c.2 = name_ref Op, imports.%C.C.decl.041267.2 [concrete = constants.%C.C.6522d8.2]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %.loc21_35.3: ref %C = value_as_ref %.loc21_35.1
// CHECK:STDOUT:   %addr.loc21_35.1: %ptr.d9e = addr_of %.loc21_35.3
// CHECK:STDOUT:   %.loc21_35.4: %ptr.c92 = as_compatible %addr.loc21_35.1
// CHECK:STDOUT:   %.loc21_35.5: %ptr.c92 = converted %addr.loc21_35.1, %.loc21_35.4
// CHECK:STDOUT:   %addr.loc21_35.2: %ptr.d9e = addr_of %lhs.var.8
// CHECK:STDOUT:   %C__carbon_thunk.call.loc21_35: init %empty_tuple.type = call imports.%C__carbon_thunk.decl.8acdfe.2(%.loc21_35.5, %addr.loc21_35.2)
// CHECK:STDOUT:   %.loc21_35.6: init %C to %lhs.var.8 = mark_in_place_init %C__carbon_thunk.call.loc21_35
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %.loc21_40.1: %C = acquire_value %c2.ref.loc21
// CHECK:STDOUT:   %impl.elem0.loc21_40: %.a81 = impl_witness_access constants.%custom_witness.681, element0 [concrete = constants.%C.Op.068411.1]
// CHECK:STDOUT:   %bound_method.loc21_40: <bound method> = bound_method %.loc21_40.1, %impl.elem0.loc21_40
// CHECK:STDOUT:   %.loc21_40.2: ref %C = temporary_storage
// CHECK:STDOUT:   %Op.ref.loc21_40: %C.C.type.065f0c.2 = name_ref Op, imports.%C.C.decl.041267.2 [concrete = constants.%C.C.6522d8.2]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %.loc21_40.3: ref %C = value_as_ref %.loc21_40.1
// CHECK:STDOUT:   %addr.loc21_40.1: %ptr.d9e = addr_of %.loc21_40.3
// CHECK:STDOUT:   %.loc21_40.4: %ptr.c92 = as_compatible %addr.loc21_40.1
// CHECK:STDOUT:   %.loc21_40.5: %ptr.c92 = converted %addr.loc21_40.1, %.loc21_40.4
// CHECK:STDOUT:   %addr.loc21_40.2: %ptr.d9e = addr_of %rhs.var.8
// CHECK:STDOUT:   %C__carbon_thunk.call.loc21_40: init %empty_tuple.type = call imports.%C__carbon_thunk.decl.8acdfe.2(%.loc21_40.5, %addr.loc21_40.2)
// CHECK:STDOUT:   %.loc21_40.6: init %C to %rhs.var.8 = mark_in_place_init %C__carbon_thunk.call.loc21_40
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %addr.loc21_38.1: %ptr.d9e = addr_of %lhs.var.8
// CHECK:STDOUT:   %addr.loc21_38.2: %ptr.d9e = addr_of %rhs.var.8
// CHECK:STDOUT:   %addr.loc21_38.3: %ptr.d9e = addr_of %.loc21_38.1
// CHECK:STDOUT:   %operator_Caret__carbon_thunk.call: init %empty_tuple.type = call imports.%operator_Caret__carbon_thunk.decl(%addr.loc21_38.1, %addr.loc21_38.2, %addr.loc21_38.3)
// CHECK:STDOUT:   %.loc21_38.2: init %C to %.loc21_38.1 = mark_in_place_init %operator_Caret__carbon_thunk.call
// CHECK:STDOUT:   %.loc21_30: type = splice_block %C.ref.loc21 [concrete = constants.%C] {
// CHECK:STDOUT:     %Cpp.ref.loc21: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %C.ref.loc21: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc21_38.3: ref %C = temporary %.loc21_38.1, %.loc21_38.2
// CHECK:STDOUT:   %.loc21_38.4: %C = acquire_value %.loc21_38.3
// CHECK:STDOUT:   %bitwise_xor: %C = value_binding bitwise_xor, %.loc21_38.4
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %left_shift.patt: %pattern_type.217 = value_binding_pattern left_shift [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c1.ref.loc22: ref %C = name_ref c1, %c1
// CHECK:STDOUT:   %int_3.loc22: Core.IntLiteral = int_value 3 [concrete = constants.%int_3.1ba]
// CHECK:STDOUT:   %.loc22_37.1: ref %C = temporary_storage
// CHECK:STDOUT:   %.loc22_34.1: %C = acquire_value %c1.ref.loc22
// CHECK:STDOUT:   %impl.elem0.loc22: %.a81 = impl_witness_access constants.%custom_witness.681, element0 [concrete = constants.%C.Op.068411.1]
// CHECK:STDOUT:   %bound_method.loc22: <bound method> = bound_method %.loc22_34.1, %impl.elem0.loc22
// CHECK:STDOUT:   %.loc22_34.2: ref %C = temporary_storage
// CHECK:STDOUT:   %Op.ref.loc22_34: %C.C.type.065f0c.2 = name_ref Op, imports.%C.C.decl.041267.2 [concrete = constants.%C.C.6522d8.2]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %.loc22_34.3: ref %C = value_as_ref %.loc22_34.1
// CHECK:STDOUT:   %addr.loc22_34.1: %ptr.d9e = addr_of %.loc22_34.3
// CHECK:STDOUT:   %.loc22_34.4: %ptr.c92 = as_compatible %addr.loc22_34.1
// CHECK:STDOUT:   %.loc22_34.5: %ptr.c92 = converted %addr.loc22_34.1, %.loc22_34.4
// CHECK:STDOUT:   %addr.loc22_34.2: %ptr.d9e = addr_of %lhs.var.9
// CHECK:STDOUT:   %C__carbon_thunk.call.loc22: init %empty_tuple.type = call imports.%C__carbon_thunk.decl.8acdfe.2(%.loc22_34.5, %addr.loc22_34.2)
// CHECK:STDOUT:   %.loc22_34.6: init %C to %lhs.var.9 = mark_in_place_init %C__carbon_thunk.call.loc22
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %addr.loc22_37.1: %ptr.d9e = addr_of %lhs.var.9
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %addr.loc22_37.2: %ptr.d9e = addr_of %.loc22_37.1
// CHECK:STDOUT:   %operator_LessLess__carbon_thunk.call: init %empty_tuple.type = call imports.%operator_LessLess__carbon_thunk.decl(%addr.loc22_37.1, %.5, %addr.loc22_37.2)
// CHECK:STDOUT:   %.loc22_37.2: init %C to %.loc22_37.1 = mark_in_place_init %operator_LessLess__carbon_thunk.call
// CHECK:STDOUT:   %.loc22_29: type = splice_block %C.ref.loc22 [concrete = constants.%C] {
// CHECK:STDOUT:     %Cpp.ref.loc22: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %C.ref.loc22: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc22_37.3: ref %C = temporary %.loc22_37.1, %.loc22_37.2
// CHECK:STDOUT:   %.loc22_37.4: %C = acquire_value %.loc22_37.3
// CHECK:STDOUT:   %left_shift: %C = value_binding left_shift, %.loc22_37.4
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %right_shift.patt: %pattern_type.217 = value_binding_pattern right_shift [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c1.ref.loc23: ref %C = name_ref c1, %c1
// CHECK:STDOUT:   %int_5.loc23: Core.IntLiteral = int_value 5 [concrete = constants.%int_5.64b]
// CHECK:STDOUT:   %.loc23_38.1: ref %C = temporary_storage
// CHECK:STDOUT:   %.loc23_35.1: %C = acquire_value %c1.ref.loc23
// CHECK:STDOUT:   %impl.elem0.loc23: %.a81 = impl_witness_access constants.%custom_witness.681, element0 [concrete = constants.%C.Op.068411.1]
// CHECK:STDOUT:   %bound_method.loc23: <bound method> = bound_method %.loc23_35.1, %impl.elem0.loc23
// CHECK:STDOUT:   %.loc23_35.2: ref %C = temporary_storage
// CHECK:STDOUT:   %Op.ref.loc23_35: %C.C.type.065f0c.2 = name_ref Op, imports.%C.C.decl.041267.2 [concrete = constants.%C.C.6522d8.2]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %.loc23_35.3: ref %C = value_as_ref %.loc23_35.1
// CHECK:STDOUT:   %addr.loc23_35.1: %ptr.d9e = addr_of %.loc23_35.3
// CHECK:STDOUT:   %.loc23_35.4: %ptr.c92 = as_compatible %addr.loc23_35.1
// CHECK:STDOUT:   %.loc23_35.5: %ptr.c92 = converted %addr.loc23_35.1, %.loc23_35.4
// CHECK:STDOUT:   %addr.loc23_35.2: %ptr.d9e = addr_of %lhs.var.10
// CHECK:STDOUT:   %C__carbon_thunk.call.loc23: init %empty_tuple.type = call imports.%C__carbon_thunk.decl.8acdfe.2(%.loc23_35.5, %addr.loc23_35.2)
// CHECK:STDOUT:   %.loc23_35.6: init %C to %lhs.var.10 = mark_in_place_init %C__carbon_thunk.call.loc23
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %addr.loc23_38.1: %ptr.d9e = addr_of %lhs.var.10
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %addr.loc23_38.2: %ptr.d9e = addr_of %.loc23_38.1
// CHECK:STDOUT:   %operator_GreaterGreater__carbon_thunk.call: init %empty_tuple.type = call imports.%operator_GreaterGreater__carbon_thunk.decl(%addr.loc23_38.1, %.8, %addr.loc23_38.2)
// CHECK:STDOUT:   %.loc23_38.2: init %C to %.loc23_38.1 = mark_in_place_init %operator_GreaterGreater__carbon_thunk.call
// CHECK:STDOUT:   %.loc23_30: type = splice_block %C.ref.loc23 [concrete = constants.%C] {
// CHECK:STDOUT:     %Cpp.ref.loc23: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %C.ref.loc23: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc23_38.3: ref %C = temporary %.loc23_38.1, %.loc23_38.2
// CHECK:STDOUT:   %.loc23_38.4: %C = acquire_value %.loc23_38.3
// CHECK:STDOUT:   %right_shift: %C = value_binding right_shift, %.loc23_38.4
// CHECK:STDOUT:   %c1.ref.loc26: ref %C = name_ref c1, %c1
// CHECK:STDOUT:   %c2.ref.loc26: ref %C = name_ref c2, %c2
// CHECK:STDOUT:   %.loc26_9.1: %C = acquire_value %c2.ref.loc26
// CHECK:STDOUT:   %impl.elem0.loc26: %.a81 = impl_witness_access constants.%custom_witness.681, element0 [concrete = constants.%C.Op.068411.1]
// CHECK:STDOUT:   %bound_method.loc26: <bound method> = bound_method %.loc26_9.1, %impl.elem0.loc26
// CHECK:STDOUT:   %.loc26_9.2: ref %C = temporary_storage
// CHECK:STDOUT:   %Op.ref.loc26: %C.C.type.065f0c.2 = name_ref Op, imports.%C.C.decl.041267.2 [concrete = constants.%C.C.6522d8.2]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %.loc26_9.3: ref %C = value_as_ref %.loc26_9.1
// CHECK:STDOUT:   %addr.loc26_9.1: %ptr.d9e = addr_of %.loc26_9.3
// CHECK:STDOUT:   %.loc26_9.4: %ptr.c92 = as_compatible %addr.loc26_9.1
// CHECK:STDOUT:   %.loc26_9.5: %ptr.c92 = converted %addr.loc26_9.1, %.loc26_9.4
// CHECK:STDOUT:   %addr.loc26_9.2: %ptr.d9e = addr_of %rhs.var.9
// CHECK:STDOUT:   %C__carbon_thunk.call.loc26: init %empty_tuple.type = call imports.%C__carbon_thunk.decl.8acdfe.2(%.loc26_9.5, %addr.loc26_9.2)
// CHECK:STDOUT:   %.loc26_9.6: init %C to %rhs.var.9 = mark_in_place_init %C__carbon_thunk.call.loc26
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %addr.loc26_6: %ptr.d9e = addr_of %rhs.var.9
// CHECK:STDOUT:   %operator_PlusEqual__carbon_thunk.call: ref %C = call imports.%operator_PlusEqual__carbon_thunk.decl(%c1.ref.loc26, %addr.loc26_6)
// CHECK:STDOUT:   %c1.ref.loc27: ref %C = name_ref c1, %c1
// CHECK:STDOUT:   %c2.ref.loc27: ref %C = name_ref c2, %c2
// CHECK:STDOUT:   %.loc27_9.1: %C = acquire_value %c2.ref.loc27
// CHECK:STDOUT:   %impl.elem0.loc27: %.a81 = impl_witness_access constants.%custom_witness.681, element0 [concrete = constants.%C.Op.068411.1]
// CHECK:STDOUT:   %bound_method.loc27: <bound method> = bound_method %.loc27_9.1, %impl.elem0.loc27
// CHECK:STDOUT:   %.loc27_9.2: ref %C = temporary_storage
// CHECK:STDOUT:   %Op.ref.loc27: %C.C.type.065f0c.2 = name_ref Op, imports.%C.C.decl.041267.2 [concrete = constants.%C.C.6522d8.2]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %.loc27_9.3: ref %C = value_as_ref %.loc27_9.1
// CHECK:STDOUT:   %addr.loc27_9.1: %ptr.d9e = addr_of %.loc27_9.3
// CHECK:STDOUT:   %.loc27_9.4: %ptr.c92 = as_compatible %addr.loc27_9.1
// CHECK:STDOUT:   %.loc27_9.5: %ptr.c92 = converted %addr.loc27_9.1, %.loc27_9.4
// CHECK:STDOUT:   %addr.loc27_9.2: %ptr.d9e = addr_of %rhs.var.10
// CHECK:STDOUT:   %C__carbon_thunk.call.loc27: init %empty_tuple.type = call imports.%C__carbon_thunk.decl.8acdfe.2(%.loc27_9.5, %addr.loc27_9.2)
// CHECK:STDOUT:   %.loc27_9.6: init %C to %rhs.var.10 = mark_in_place_init %C__carbon_thunk.call.loc27
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %addr.loc27_6: %ptr.d9e = addr_of %rhs.var.10
// CHECK:STDOUT:   %operator_MinusEqual__carbon_thunk.call: ref %C = call imports.%operator_MinusEqual__carbon_thunk.decl(%c1.ref.loc27, %addr.loc27_6)
// CHECK:STDOUT:   %c1.ref.loc28: ref %C = name_ref c1, %c1
// CHECK:STDOUT:   %c2.ref.loc28: ref %C = name_ref c2, %c2
// CHECK:STDOUT:   %.loc28_9.1: %C = acquire_value %c2.ref.loc28
// CHECK:STDOUT:   %impl.elem0.loc28: %.a81 = impl_witness_access constants.%custom_witness.681, element0 [concrete = constants.%C.Op.068411.1]
// CHECK:STDOUT:   %bound_method.loc28: <bound method> = bound_method %.loc28_9.1, %impl.elem0.loc28
// CHECK:STDOUT:   %.loc28_9.2: ref %C = temporary_storage
// CHECK:STDOUT:   %Op.ref.loc28: %C.C.type.065f0c.2 = name_ref Op, imports.%C.C.decl.041267.2 [concrete = constants.%C.C.6522d8.2]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %.loc28_9.3: ref %C = value_as_ref %.loc28_9.1
// CHECK:STDOUT:   %addr.loc28_9.1: %ptr.d9e = addr_of %.loc28_9.3
// CHECK:STDOUT:   %.loc28_9.4: %ptr.c92 = as_compatible %addr.loc28_9.1
// CHECK:STDOUT:   %.loc28_9.5: %ptr.c92 = converted %addr.loc28_9.1, %.loc28_9.4
// CHECK:STDOUT:   %addr.loc28_9.2: %ptr.d9e = addr_of %rhs.var.11
// CHECK:STDOUT:   %C__carbon_thunk.call.loc28: init %empty_tuple.type = call imports.%C__carbon_thunk.decl.8acdfe.2(%.loc28_9.5, %addr.loc28_9.2)
// CHECK:STDOUT:   %.loc28_9.6: init %C to %rhs.var.11 = mark_in_place_init %C__carbon_thunk.call.loc28
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %addr.loc28_6: %ptr.d9e = addr_of %rhs.var.11
// CHECK:STDOUT:   %operator_StarEqual__carbon_thunk.call: ref %C = call imports.%operator_StarEqual__carbon_thunk.decl(%c1.ref.loc28, %addr.loc28_6)
// CHECK:STDOUT:   %c1.ref.loc29: ref %C = name_ref c1, %c1
// CHECK:STDOUT:   %c2.ref.loc29: ref %C = name_ref c2, %c2
// CHECK:STDOUT:   %.loc29_9.1: %C = acquire_value %c2.ref.loc29
// CHECK:STDOUT:   %impl.elem0.loc29: %.a81 = impl_witness_access constants.%custom_witness.681, element0 [concrete = constants.%C.Op.068411.1]
// CHECK:STDOUT:   %bound_method.loc29: <bound method> = bound_method %.loc29_9.1, %impl.elem0.loc29
// CHECK:STDOUT:   %.loc29_9.2: ref %C = temporary_storage
// CHECK:STDOUT:   %Op.ref.loc29: %C.C.type.065f0c.2 = name_ref Op, imports.%C.C.decl.041267.2 [concrete = constants.%C.C.6522d8.2]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %.loc29_9.3: ref %C = value_as_ref %.loc29_9.1
// CHECK:STDOUT:   %addr.loc29_9.1: %ptr.d9e = addr_of %.loc29_9.3
// CHECK:STDOUT:   %.loc29_9.4: %ptr.c92 = as_compatible %addr.loc29_9.1
// CHECK:STDOUT:   %.loc29_9.5: %ptr.c92 = converted %addr.loc29_9.1, %.loc29_9.4
// CHECK:STDOUT:   %addr.loc29_9.2: %ptr.d9e = addr_of %rhs.var.12
// CHECK:STDOUT:   %C__carbon_thunk.call.loc29: init %empty_tuple.type = call imports.%C__carbon_thunk.decl.8acdfe.2(%.loc29_9.5, %addr.loc29_9.2)
// CHECK:STDOUT:   %.loc29_9.6: init %C to %rhs.var.12 = mark_in_place_init %C__carbon_thunk.call.loc29
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %addr.loc29_6: %ptr.d9e = addr_of %rhs.var.12
// CHECK:STDOUT:   %operator_SlashEqual__carbon_thunk.call: ref %C = call imports.%operator_SlashEqual__carbon_thunk.decl(%c1.ref.loc29, %addr.loc29_6)
// CHECK:STDOUT:   %c1.ref.loc30: ref %C = name_ref c1, %c1
// CHECK:STDOUT:   %c2.ref.loc30: ref %C = name_ref c2, %c2
// CHECK:STDOUT:   %.loc30_9.1: %C = acquire_value %c2.ref.loc30
// CHECK:STDOUT:   %impl.elem0.loc30: %.a81 = impl_witness_access constants.%custom_witness.681, element0 [concrete = constants.%C.Op.068411.1]
// CHECK:STDOUT:   %bound_method.loc30: <bound method> = bound_method %.loc30_9.1, %impl.elem0.loc30
// CHECK:STDOUT:   %.loc30_9.2: ref %C = temporary_storage
// CHECK:STDOUT:   %Op.ref.loc30: %C.C.type.065f0c.2 = name_ref Op, imports.%C.C.decl.041267.2 [concrete = constants.%C.C.6522d8.2]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %.loc30_9.3: ref %C = value_as_ref %.loc30_9.1
// CHECK:STDOUT:   %addr.loc30_9.1: %ptr.d9e = addr_of %.loc30_9.3
// CHECK:STDOUT:   %.loc30_9.4: %ptr.c92 = as_compatible %addr.loc30_9.1
// CHECK:STDOUT:   %.loc30_9.5: %ptr.c92 = converted %addr.loc30_9.1, %.loc30_9.4
// CHECK:STDOUT:   %addr.loc30_9.2: %ptr.d9e = addr_of %rhs.var.13
// CHECK:STDOUT:   %C__carbon_thunk.call.loc30: init %empty_tuple.type = call imports.%C__carbon_thunk.decl.8acdfe.2(%.loc30_9.5, %addr.loc30_9.2)
// CHECK:STDOUT:   %.loc30_9.6: init %C to %rhs.var.13 = mark_in_place_init %C__carbon_thunk.call.loc30
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %addr.loc30_6: %ptr.d9e = addr_of %rhs.var.13
// CHECK:STDOUT:   %operator_PercentEqual__carbon_thunk.call: ref %C = call imports.%operator_PercentEqual__carbon_thunk.decl(%c1.ref.loc30, %addr.loc30_6)
// CHECK:STDOUT:   %c1.ref.loc33: ref %C = name_ref c1, %c1
// CHECK:STDOUT:   %c2.ref.loc33: ref %C = name_ref c2, %c2
// CHECK:STDOUT:   %.loc33_9.1: %C = acquire_value %c2.ref.loc33
// CHECK:STDOUT:   %impl.elem0.loc33: %.a81 = impl_witness_access constants.%custom_witness.681, element0 [concrete = constants.%C.Op.068411.1]
// CHECK:STDOUT:   %bound_method.loc33: <bound method> = bound_method %.loc33_9.1, %impl.elem0.loc33
// CHECK:STDOUT:   %.loc33_9.2: ref %C = temporary_storage
// CHECK:STDOUT:   %Op.ref.loc33: %C.C.type.065f0c.2 = name_ref Op, imports.%C.C.decl.041267.2 [concrete = constants.%C.C.6522d8.2]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %.loc33_9.3: ref %C = value_as_ref %.loc33_9.1
// CHECK:STDOUT:   %addr.loc33_9.1: %ptr.d9e = addr_of %.loc33_9.3
// CHECK:STDOUT:   %.loc33_9.4: %ptr.c92 = as_compatible %addr.loc33_9.1
// CHECK:STDOUT:   %.loc33_9.5: %ptr.c92 = converted %addr.loc33_9.1, %.loc33_9.4
// CHECK:STDOUT:   %addr.loc33_9.2: %ptr.d9e = addr_of %rhs.var.14
// CHECK:STDOUT:   %C__carbon_thunk.call.loc33: init %empty_tuple.type = call imports.%C__carbon_thunk.decl.8acdfe.2(%.loc33_9.5, %addr.loc33_9.2)
// CHECK:STDOUT:   %.loc33_9.6: init %C to %rhs.var.14 = mark_in_place_init %C__carbon_thunk.call.loc33
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %addr.loc33_6: %ptr.d9e = addr_of %rhs.var.14
// CHECK:STDOUT:   %operator_AmpEqual__carbon_thunk.call: ref %C = call imports.%operator_AmpEqual__carbon_thunk.decl(%c1.ref.loc33, %addr.loc33_6)
// CHECK:STDOUT:   %c1.ref.loc34: ref %C = name_ref c1, %c1
// CHECK:STDOUT:   %c2.ref.loc34: ref %C = name_ref c2, %c2
// CHECK:STDOUT:   %.loc34_9.1: %C = acquire_value %c2.ref.loc34
// CHECK:STDOUT:   %impl.elem0.loc34: %.a81 = impl_witness_access constants.%custom_witness.681, element0 [concrete = constants.%C.Op.068411.1]
// CHECK:STDOUT:   %bound_method.loc34: <bound method> = bound_method %.loc34_9.1, %impl.elem0.loc34
// CHECK:STDOUT:   %.loc34_9.2: ref %C = temporary_storage
// CHECK:STDOUT:   %Op.ref.loc34: %C.C.type.065f0c.2 = name_ref Op, imports.%C.C.decl.041267.2 [concrete = constants.%C.C.6522d8.2]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %.loc34_9.3: ref %C = value_as_ref %.loc34_9.1
// CHECK:STDOUT:   %addr.loc34_9.1: %ptr.d9e = addr_of %.loc34_9.3
// CHECK:STDOUT:   %.loc34_9.4: %ptr.c92 = as_compatible %addr.loc34_9.1
// CHECK:STDOUT:   %.loc34_9.5: %ptr.c92 = converted %addr.loc34_9.1, %.loc34_9.4
// CHECK:STDOUT:   %addr.loc34_9.2: %ptr.d9e = addr_of %rhs.var.15
// CHECK:STDOUT:   %C__carbon_thunk.call.loc34: init %empty_tuple.type = call imports.%C__carbon_thunk.decl.8acdfe.2(%.loc34_9.5, %addr.loc34_9.2)
// CHECK:STDOUT:   %.loc34_9.6: init %C to %rhs.var.15 = mark_in_place_init %C__carbon_thunk.call.loc34
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %addr.loc34_6: %ptr.d9e = addr_of %rhs.var.15
// CHECK:STDOUT:   %operator_PipeEqual__carbon_thunk.call: ref %C = call imports.%operator_PipeEqual__carbon_thunk.decl(%c1.ref.loc34, %addr.loc34_6)
// CHECK:STDOUT:   %c1.ref.loc35: ref %C = name_ref c1, %c1
// CHECK:STDOUT:   %c2.ref.loc35: ref %C = name_ref c2, %c2
// CHECK:STDOUT:   %.loc35_9.1: %C = acquire_value %c2.ref.loc35
// CHECK:STDOUT:   %impl.elem0.loc35: %.a81 = impl_witness_access constants.%custom_witness.681, element0 [concrete = constants.%C.Op.068411.1]
// CHECK:STDOUT:   %bound_method.loc35: <bound method> = bound_method %.loc35_9.1, %impl.elem0.loc35
// CHECK:STDOUT:   %.loc35_9.2: ref %C = temporary_storage
// CHECK:STDOUT:   %Op.ref.loc35: %C.C.type.065f0c.2 = name_ref Op, imports.%C.C.decl.041267.2 [concrete = constants.%C.C.6522d8.2]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %.loc35_9.3: ref %C = value_as_ref %.loc35_9.1
// CHECK:STDOUT:   %addr.loc35_9.1: %ptr.d9e = addr_of %.loc35_9.3
// CHECK:STDOUT:   %.loc35_9.4: %ptr.c92 = as_compatible %addr.loc35_9.1
// CHECK:STDOUT:   %.loc35_9.5: %ptr.c92 = converted %addr.loc35_9.1, %.loc35_9.4
// CHECK:STDOUT:   %addr.loc35_9.2: %ptr.d9e = addr_of %rhs.var.16
// CHECK:STDOUT:   %C__carbon_thunk.call.loc35: init %empty_tuple.type = call imports.%C__carbon_thunk.decl.8acdfe.2(%.loc35_9.5, %addr.loc35_9.2)
// CHECK:STDOUT:   %.loc35_9.6: init %C to %rhs.var.16 = mark_in_place_init %C__carbon_thunk.call.loc35
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %addr.loc35_6: %ptr.d9e = addr_of %rhs.var.16
// CHECK:STDOUT:   %operator_CaretEqual__carbon_thunk.call: ref %C = call imports.%operator_CaretEqual__carbon_thunk.decl(%c1.ref.loc35, %addr.loc35_6)
// CHECK:STDOUT:   %c1.ref.loc36: ref %C = name_ref c1, %c1
// CHECK:STDOUT:   %int_3.loc36: Core.IntLiteral = int_value 3 [concrete = constants.%int_3.1ba]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %operator_LessLessEqual__carbon_thunk.call: ref %C = call imports.%operator_LessLessEqual__carbon_thunk.decl(%c1.ref.loc36, %.11)
// CHECK:STDOUT:   %c1.ref.loc37: ref %C = name_ref c1, %c1
// CHECK:STDOUT:   %int_5.loc37: Core.IntLiteral = int_value 5 [concrete = constants.%int_5.64b]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %operator_GreaterGreaterEqual__carbon_thunk.call: ref %C = call imports.%operator_GreaterGreaterEqual__carbon_thunk.decl(%c1.ref.loc37, %.14)
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %equal.patt: %pattern_type.831 = value_binding_pattern equal [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c1.ref.loc40: ref %C = name_ref c1, %c1
// CHECK:STDOUT:   %c2.ref.loc40: ref %C = name_ref c2, %c2
// CHECK:STDOUT:   %.loc40_28.1: %C = acquire_value %c1.ref.loc40
// CHECK:STDOUT:   %impl.elem0.loc40_28: %.a81 = impl_witness_access constants.%custom_witness.681, element0 [concrete = constants.%C.Op.068411.1]
// CHECK:STDOUT:   %bound_method.loc40_28: <bound method> = bound_method %.loc40_28.1, %impl.elem0.loc40_28
// CHECK:STDOUT:   %.loc40_28.2: ref %C = temporary_storage
// CHECK:STDOUT:   %Op.ref.loc40_28: %C.C.type.065f0c.2 = name_ref Op, imports.%C.C.decl.041267.2 [concrete = constants.%C.C.6522d8.2]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %.loc40_28.3: ref %C = value_as_ref %.loc40_28.1
// CHECK:STDOUT:   %addr.loc40_28.1: %ptr.d9e = addr_of %.loc40_28.3
// CHECK:STDOUT:   %.loc40_28.4: %ptr.c92 = as_compatible %addr.loc40_28.1
// CHECK:STDOUT:   %.loc40_28.5: %ptr.c92 = converted %addr.loc40_28.1, %.loc40_28.4
// CHECK:STDOUT:   %addr.loc40_28.2: %ptr.d9e = addr_of %lhs.var.11
// CHECK:STDOUT:   %C__carbon_thunk.call.loc40_28: init %empty_tuple.type = call imports.%C__carbon_thunk.decl.8acdfe.2(%.loc40_28.5, %addr.loc40_28.2)
// CHECK:STDOUT:   %.loc40_28.6: init %C to %lhs.var.11 = mark_in_place_init %C__carbon_thunk.call.loc40_28
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %.loc40_34.1: %C = acquire_value %c2.ref.loc40
// CHECK:STDOUT:   %impl.elem0.loc40_34: %.a81 = impl_witness_access constants.%custom_witness.681, element0 [concrete = constants.%C.Op.068411.1]
// CHECK:STDOUT:   %bound_method.loc40_34: <bound method> = bound_method %.loc40_34.1, %impl.elem0.loc40_34
// CHECK:STDOUT:   %.loc40_34.2: ref %C = temporary_storage
// CHECK:STDOUT:   %Op.ref.loc40_34: %C.C.type.065f0c.2 = name_ref Op, imports.%C.C.decl.041267.2 [concrete = constants.%C.C.6522d8.2]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %.loc40_34.3: ref %C = value_as_ref %.loc40_34.1
// CHECK:STDOUT:   %addr.loc40_34.1: %ptr.d9e = addr_of %.loc40_34.3
// CHECK:STDOUT:   %.loc40_34.4: %ptr.c92 = as_compatible %addr.loc40_34.1
// CHECK:STDOUT:   %.loc40_34.5: %ptr.c92 = converted %addr.loc40_34.1, %.loc40_34.4
// CHECK:STDOUT:   %addr.loc40_34.2: %ptr.d9e = addr_of %rhs.var.17
// CHECK:STDOUT:   %C__carbon_thunk.call.loc40_34: init %empty_tuple.type = call imports.%C__carbon_thunk.decl.8acdfe.2(%.loc40_34.5, %addr.loc40_34.2)
// CHECK:STDOUT:   %.loc40_34.6: init %C to %rhs.var.17 = mark_in_place_init %C__carbon_thunk.call.loc40_34
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %addr.loc40_31.1: %ptr.d9e = addr_of %lhs.var.11
// CHECK:STDOUT:   %addr.loc40_31.2: %ptr.d9e = addr_of %rhs.var.17
// CHECK:STDOUT:   %.loc40_31.1: ref bool = temporary_storage
// CHECK:STDOUT:   %addr.loc40_31.3: %ptr.bb2 = addr_of %.loc40_31.1
// CHECK:STDOUT:   %operator_EqualEqual__carbon_thunk.call: init %empty_tuple.type = call imports.%operator_EqualEqual__carbon_thunk.decl(%addr.loc40_31.1, %addr.loc40_31.2, %addr.loc40_31.3)
// CHECK:STDOUT:   %.loc40_31.2: init bool to %.loc40_31.1 = mark_in_place_init %operator_EqualEqual__carbon_thunk.call
// CHECK:STDOUT:   %.loc40_21: type = type_literal bool [concrete = bool]
// CHECK:STDOUT:   %.loc40_31.3: ref bool = temporary %.loc40_31.1, %.loc40_31.2
// CHECK:STDOUT:   %.loc40_31.4: bool = acquire_value %.loc40_31.3
// CHECK:STDOUT:   %equal: bool = value_binding equal, %.loc40_31.4
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %not_equal.patt: %pattern_type.831 = value_binding_pattern not_equal [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c1.ref.loc41: ref %C = name_ref c1, %c1
// CHECK:STDOUT:   %c2.ref.loc41: ref %C = name_ref c2, %c2
// CHECK:STDOUT:   %.loc41_32.1: %C = acquire_value %c1.ref.loc41
// CHECK:STDOUT:   %impl.elem0.loc41_32: %.a81 = impl_witness_access constants.%custom_witness.681, element0 [concrete = constants.%C.Op.068411.1]
// CHECK:STDOUT:   %bound_method.loc41_32: <bound method> = bound_method %.loc41_32.1, %impl.elem0.loc41_32
// CHECK:STDOUT:   %.loc41_32.2: ref %C = temporary_storage
// CHECK:STDOUT:   %Op.ref.loc41_32: %C.C.type.065f0c.2 = name_ref Op, imports.%C.C.decl.041267.2 [concrete = constants.%C.C.6522d8.2]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %.loc41_32.3: ref %C = value_as_ref %.loc41_32.1
// CHECK:STDOUT:   %addr.loc41_32.1: %ptr.d9e = addr_of %.loc41_32.3
// CHECK:STDOUT:   %.loc41_32.4: %ptr.c92 = as_compatible %addr.loc41_32.1
// CHECK:STDOUT:   %.loc41_32.5: %ptr.c92 = converted %addr.loc41_32.1, %.loc41_32.4
// CHECK:STDOUT:   %addr.loc41_32.2: %ptr.d9e = addr_of %lhs.var.12
// CHECK:STDOUT:   %C__carbon_thunk.call.loc41_32: init %empty_tuple.type = call imports.%C__carbon_thunk.decl.8acdfe.2(%.loc41_32.5, %addr.loc41_32.2)
// CHECK:STDOUT:   %.loc41_32.6: init %C to %lhs.var.12 = mark_in_place_init %C__carbon_thunk.call.loc41_32
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %.loc41_38.1: %C = acquire_value %c2.ref.loc41
// CHECK:STDOUT:   %impl.elem0.loc41_38: %.a81 = impl_witness_access constants.%custom_witness.681, element0 [concrete = constants.%C.Op.068411.1]
// CHECK:STDOUT:   %bound_method.loc41_38: <bound method> = bound_method %.loc41_38.1, %impl.elem0.loc41_38
// CHECK:STDOUT:   %.loc41_38.2: ref %C = temporary_storage
// CHECK:STDOUT:   %Op.ref.loc41_38: %C.C.type.065f0c.2 = name_ref Op, imports.%C.C.decl.041267.2 [concrete = constants.%C.C.6522d8.2]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %.loc41_38.3: ref %C = value_as_ref %.loc41_38.1
// CHECK:STDOUT:   %addr.loc41_38.1: %ptr.d9e = addr_of %.loc41_38.3
// CHECK:STDOUT:   %.loc41_38.4: %ptr.c92 = as_compatible %addr.loc41_38.1
// CHECK:STDOUT:   %.loc41_38.5: %ptr.c92 = converted %addr.loc41_38.1, %.loc41_38.4
// CHECK:STDOUT:   %addr.loc41_38.2: %ptr.d9e = addr_of %rhs.var.18
// CHECK:STDOUT:   %C__carbon_thunk.call.loc41_38: init %empty_tuple.type = call imports.%C__carbon_thunk.decl.8acdfe.2(%.loc41_38.5, %addr.loc41_38.2)
// CHECK:STDOUT:   %.loc41_38.6: init %C to %rhs.var.18 = mark_in_place_init %C__carbon_thunk.call.loc41_38
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %addr.loc41_35.1: %ptr.d9e = addr_of %lhs.var.12
// CHECK:STDOUT:   %addr.loc41_35.2: %ptr.d9e = addr_of %rhs.var.18
// CHECK:STDOUT:   %.loc41_35.1: ref bool = temporary_storage
// CHECK:STDOUT:   %addr.loc41_35.3: %ptr.bb2 = addr_of %.loc41_35.1
// CHECK:STDOUT:   %operator_ExclaimEqual__carbon_thunk.call: init %empty_tuple.type = call imports.%operator_ExclaimEqual__carbon_thunk.decl(%addr.loc41_35.1, %addr.loc41_35.2, %addr.loc41_35.3)
// CHECK:STDOUT:   %.loc41_35.2: init bool to %.loc41_35.1 = mark_in_place_init %operator_ExclaimEqual__carbon_thunk.call
// CHECK:STDOUT:   %.loc41_25: type = type_literal bool [concrete = bool]
// CHECK:STDOUT:   %.loc41_35.3: ref bool = temporary %.loc41_35.1, %.loc41_35.2
// CHECK:STDOUT:   %.loc41_35.4: bool = acquire_value %.loc41_35.3
// CHECK:STDOUT:   %not_equal: bool = value_binding not_equal, %.loc41_35.4
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %greater_than.patt: %pattern_type.831 = value_binding_pattern greater_than [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c1.ref.loc42: ref %C = name_ref c1, %c1
// CHECK:STDOUT:   %c2.ref.loc42: ref %C = name_ref c2, %c2
// CHECK:STDOUT:   %.loc42_35.1: %C = acquire_value %c1.ref.loc42
// CHECK:STDOUT:   %impl.elem0.loc42_35: %.a81 = impl_witness_access constants.%custom_witness.681, element0 [concrete = constants.%C.Op.068411.1]
// CHECK:STDOUT:   %bound_method.loc42_35: <bound method> = bound_method %.loc42_35.1, %impl.elem0.loc42_35
// CHECK:STDOUT:   %.loc42_35.2: ref %C = temporary_storage
// CHECK:STDOUT:   %Op.ref.loc42_35: %C.C.type.065f0c.2 = name_ref Op, imports.%C.C.decl.041267.2 [concrete = constants.%C.C.6522d8.2]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %.loc42_35.3: ref %C = value_as_ref %.loc42_35.1
// CHECK:STDOUT:   %addr.loc42_35.1: %ptr.d9e = addr_of %.loc42_35.3
// CHECK:STDOUT:   %.loc42_35.4: %ptr.c92 = as_compatible %addr.loc42_35.1
// CHECK:STDOUT:   %.loc42_35.5: %ptr.c92 = converted %addr.loc42_35.1, %.loc42_35.4
// CHECK:STDOUT:   %addr.loc42_35.2: %ptr.d9e = addr_of %lhs.var.13
// CHECK:STDOUT:   %C__carbon_thunk.call.loc42_35: init %empty_tuple.type = call imports.%C__carbon_thunk.decl.8acdfe.2(%.loc42_35.5, %addr.loc42_35.2)
// CHECK:STDOUT:   %.loc42_35.6: init %C to %lhs.var.13 = mark_in_place_init %C__carbon_thunk.call.loc42_35
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %.loc42_40.1: %C = acquire_value %c2.ref.loc42
// CHECK:STDOUT:   %impl.elem0.loc42_40: %.a81 = impl_witness_access constants.%custom_witness.681, element0 [concrete = constants.%C.Op.068411.1]
// CHECK:STDOUT:   %bound_method.loc42_40: <bound method> = bound_method %.loc42_40.1, %impl.elem0.loc42_40
// CHECK:STDOUT:   %.loc42_40.2: ref %C = temporary_storage
// CHECK:STDOUT:   %Op.ref.loc42_40: %C.C.type.065f0c.2 = name_ref Op, imports.%C.C.decl.041267.2 [concrete = constants.%C.C.6522d8.2]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %.loc42_40.3: ref %C = value_as_ref %.loc42_40.1
// CHECK:STDOUT:   %addr.loc42_40.1: %ptr.d9e = addr_of %.loc42_40.3
// CHECK:STDOUT:   %.loc42_40.4: %ptr.c92 = as_compatible %addr.loc42_40.1
// CHECK:STDOUT:   %.loc42_40.5: %ptr.c92 = converted %addr.loc42_40.1, %.loc42_40.4
// CHECK:STDOUT:   %addr.loc42_40.2: %ptr.d9e = addr_of %rhs.var.19
// CHECK:STDOUT:   %C__carbon_thunk.call.loc42_40: init %empty_tuple.type = call imports.%C__carbon_thunk.decl.8acdfe.2(%.loc42_40.5, %addr.loc42_40.2)
// CHECK:STDOUT:   %.loc42_40.6: init %C to %rhs.var.19 = mark_in_place_init %C__carbon_thunk.call.loc42_40
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %addr.loc42_38.1: %ptr.d9e = addr_of %lhs.var.13
// CHECK:STDOUT:   %addr.loc42_38.2: %ptr.d9e = addr_of %rhs.var.19
// CHECK:STDOUT:   %.loc42_38.1: ref bool = temporary_storage
// CHECK:STDOUT:   %addr.loc42_38.3: %ptr.bb2 = addr_of %.loc42_38.1
// CHECK:STDOUT:   %operator_Greater__carbon_thunk.call: init %empty_tuple.type = call imports.%operator_Greater__carbon_thunk.decl(%addr.loc42_38.1, %addr.loc42_38.2, %addr.loc42_38.3)
// CHECK:STDOUT:   %.loc42_38.2: init bool to %.loc42_38.1 = mark_in_place_init %operator_Greater__carbon_thunk.call
// CHECK:STDOUT:   %.loc42_28: type = type_literal bool [concrete = bool]
// CHECK:STDOUT:   %.loc42_38.3: ref bool = temporary %.loc42_38.1, %.loc42_38.2
// CHECK:STDOUT:   %.loc42_38.4: bool = acquire_value %.loc42_38.3
// CHECK:STDOUT:   %greater_than: bool = value_binding greater_than, %.loc42_38.4
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %less_than.patt: %pattern_type.831 = value_binding_pattern less_than [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c1.ref.loc43: ref %C = name_ref c1, %c1
// CHECK:STDOUT:   %c2.ref.loc43: ref %C = name_ref c2, %c2
// CHECK:STDOUT:   %.loc43_32.1: %C = acquire_value %c1.ref.loc43
// CHECK:STDOUT:   %impl.elem0.loc43_32: %.a81 = impl_witness_access constants.%custom_witness.681, element0 [concrete = constants.%C.Op.068411.1]
// CHECK:STDOUT:   %bound_method.loc43_32: <bound method> = bound_method %.loc43_32.1, %impl.elem0.loc43_32
// CHECK:STDOUT:   %.loc43_32.2: ref %C = temporary_storage
// CHECK:STDOUT:   %Op.ref.loc43_32: %C.C.type.065f0c.2 = name_ref Op, imports.%C.C.decl.041267.2 [concrete = constants.%C.C.6522d8.2]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %.loc43_32.3: ref %C = value_as_ref %.loc43_32.1
// CHECK:STDOUT:   %addr.loc43_32.1: %ptr.d9e = addr_of %.loc43_32.3
// CHECK:STDOUT:   %.loc43_32.4: %ptr.c92 = as_compatible %addr.loc43_32.1
// CHECK:STDOUT:   %.loc43_32.5: %ptr.c92 = converted %addr.loc43_32.1, %.loc43_32.4
// CHECK:STDOUT:   %addr.loc43_32.2: %ptr.d9e = addr_of %lhs.var.14
// CHECK:STDOUT:   %C__carbon_thunk.call.loc43_32: init %empty_tuple.type = call imports.%C__carbon_thunk.decl.8acdfe.2(%.loc43_32.5, %addr.loc43_32.2)
// CHECK:STDOUT:   %.loc43_32.6: init %C to %lhs.var.14 = mark_in_place_init %C__carbon_thunk.call.loc43_32
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %.loc43_37.1: %C = acquire_value %c2.ref.loc43
// CHECK:STDOUT:   %impl.elem0.loc43_37: %.a81 = impl_witness_access constants.%custom_witness.681, element0 [concrete = constants.%C.Op.068411.1]
// CHECK:STDOUT:   %bound_method.loc43_37: <bound method> = bound_method %.loc43_37.1, %impl.elem0.loc43_37
// CHECK:STDOUT:   %.loc43_37.2: ref %C = temporary_storage
// CHECK:STDOUT:   %Op.ref.loc43_37: %C.C.type.065f0c.2 = name_ref Op, imports.%C.C.decl.041267.2 [concrete = constants.%C.C.6522d8.2]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %.loc43_37.3: ref %C = value_as_ref %.loc43_37.1
// CHECK:STDOUT:   %addr.loc43_37.1: %ptr.d9e = addr_of %.loc43_37.3
// CHECK:STDOUT:   %.loc43_37.4: %ptr.c92 = as_compatible %addr.loc43_37.1
// CHECK:STDOUT:   %.loc43_37.5: %ptr.c92 = converted %addr.loc43_37.1, %.loc43_37.4
// CHECK:STDOUT:   %addr.loc43_37.2: %ptr.d9e = addr_of %rhs.var.20
// CHECK:STDOUT:   %C__carbon_thunk.call.loc43_37: init %empty_tuple.type = call imports.%C__carbon_thunk.decl.8acdfe.2(%.loc43_37.5, %addr.loc43_37.2)
// CHECK:STDOUT:   %.loc43_37.6: init %C to %rhs.var.20 = mark_in_place_init %C__carbon_thunk.call.loc43_37
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %addr.loc43_35.1: %ptr.d9e = addr_of %lhs.var.14
// CHECK:STDOUT:   %addr.loc43_35.2: %ptr.d9e = addr_of %rhs.var.20
// CHECK:STDOUT:   %.loc43_35.1: ref bool = temporary_storage
// CHECK:STDOUT:   %addr.loc43_35.3: %ptr.bb2 = addr_of %.loc43_35.1
// CHECK:STDOUT:   %operator_Less__carbon_thunk.call: init %empty_tuple.type = call imports.%operator_Less__carbon_thunk.decl(%addr.loc43_35.1, %addr.loc43_35.2, %addr.loc43_35.3)
// CHECK:STDOUT:   %.loc43_35.2: init bool to %.loc43_35.1 = mark_in_place_init %operator_Less__carbon_thunk.call
// CHECK:STDOUT:   %.loc43_25: type = type_literal bool [concrete = bool]
// CHECK:STDOUT:   %.loc43_35.3: ref bool = temporary %.loc43_35.1, %.loc43_35.2
// CHECK:STDOUT:   %.loc43_35.4: bool = acquire_value %.loc43_35.3
// CHECK:STDOUT:   %less_than: bool = value_binding less_than, %.loc43_35.4
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %greater_than_or_equal.patt: %pattern_type.831 = value_binding_pattern greater_than_or_equal [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c1.ref.loc44: ref %C = name_ref c1, %c1
// CHECK:STDOUT:   %c2.ref.loc44: ref %C = name_ref c2, %c2
// CHECK:STDOUT:   %.loc44_44.1: %C = acquire_value %c1.ref.loc44
// CHECK:STDOUT:   %impl.elem0.loc44_44: %.a81 = impl_witness_access constants.%custom_witness.681, element0 [concrete = constants.%C.Op.068411.1]
// CHECK:STDOUT:   %bound_method.loc44_44: <bound method> = bound_method %.loc44_44.1, %impl.elem0.loc44_44
// CHECK:STDOUT:   %.loc44_44.2: ref %C = temporary_storage
// CHECK:STDOUT:   %Op.ref.loc44_44: %C.C.type.065f0c.2 = name_ref Op, imports.%C.C.decl.041267.2 [concrete = constants.%C.C.6522d8.2]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %.loc44_44.3: ref %C = value_as_ref %.loc44_44.1
// CHECK:STDOUT:   %addr.loc44_44.1: %ptr.d9e = addr_of %.loc44_44.3
// CHECK:STDOUT:   %.loc44_44.4: %ptr.c92 = as_compatible %addr.loc44_44.1
// CHECK:STDOUT:   %.loc44_44.5: %ptr.c92 = converted %addr.loc44_44.1, %.loc44_44.4
// CHECK:STDOUT:   %addr.loc44_44.2: %ptr.d9e = addr_of %lhs.var.15
// CHECK:STDOUT:   %C__carbon_thunk.call.loc44_44: init %empty_tuple.type = call imports.%C__carbon_thunk.decl.8acdfe.2(%.loc44_44.5, %addr.loc44_44.2)
// CHECK:STDOUT:   %.loc44_44.6: init %C to %lhs.var.15 = mark_in_place_init %C__carbon_thunk.call.loc44_44
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %.loc44_50.1: %C = acquire_value %c2.ref.loc44
// CHECK:STDOUT:   %impl.elem0.loc44_50: %.a81 = impl_witness_access constants.%custom_witness.681, element0 [concrete = constants.%C.Op.068411.1]
// CHECK:STDOUT:   %bound_method.loc44_50: <bound method> = bound_method %.loc44_50.1, %impl.elem0.loc44_50
// CHECK:STDOUT:   %.loc44_50.2: ref %C = temporary_storage
// CHECK:STDOUT:   %Op.ref.loc44_50: %C.C.type.065f0c.2 = name_ref Op, imports.%C.C.decl.041267.2 [concrete = constants.%C.C.6522d8.2]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %.loc44_50.3: ref %C = value_as_ref %.loc44_50.1
// CHECK:STDOUT:   %addr.loc44_50.1: %ptr.d9e = addr_of %.loc44_50.3
// CHECK:STDOUT:   %.loc44_50.4: %ptr.c92 = as_compatible %addr.loc44_50.1
// CHECK:STDOUT:   %.loc44_50.5: %ptr.c92 = converted %addr.loc44_50.1, %.loc44_50.4
// CHECK:STDOUT:   %addr.loc44_50.2: %ptr.d9e = addr_of %rhs.var.21
// CHECK:STDOUT:   %C__carbon_thunk.call.loc44_50: init %empty_tuple.type = call imports.%C__carbon_thunk.decl.8acdfe.2(%.loc44_50.5, %addr.loc44_50.2)
// CHECK:STDOUT:   %.loc44_50.6: init %C to %rhs.var.21 = mark_in_place_init %C__carbon_thunk.call.loc44_50
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %addr.loc44_47.1: %ptr.d9e = addr_of %lhs.var.15
// CHECK:STDOUT:   %addr.loc44_47.2: %ptr.d9e = addr_of %rhs.var.21
// CHECK:STDOUT:   %.loc44_47.1: ref bool = temporary_storage
// CHECK:STDOUT:   %addr.loc44_47.3: %ptr.bb2 = addr_of %.loc44_47.1
// CHECK:STDOUT:   %operator_GreaterEqual__carbon_thunk.call: init %empty_tuple.type = call imports.%operator_GreaterEqual__carbon_thunk.decl(%addr.loc44_47.1, %addr.loc44_47.2, %addr.loc44_47.3)
// CHECK:STDOUT:   %.loc44_47.2: init bool to %.loc44_47.1 = mark_in_place_init %operator_GreaterEqual__carbon_thunk.call
// CHECK:STDOUT:   %.loc44_37: type = type_literal bool [concrete = bool]
// CHECK:STDOUT:   %.loc44_47.3: ref bool = temporary %.loc44_47.1, %.loc44_47.2
// CHECK:STDOUT:   %.loc44_47.4: bool = acquire_value %.loc44_47.3
// CHECK:STDOUT:   %greater_than_or_equal: bool = value_binding greater_than_or_equal, %.loc44_47.4
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %less_than_or_equal.patt: %pattern_type.831 = value_binding_pattern less_than_or_equal [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c1.ref.loc45: ref %C = name_ref c1, %c1
// CHECK:STDOUT:   %c2.ref.loc45: ref %C = name_ref c2, %c2
// CHECK:STDOUT:   %.loc45_41.1: %C = acquire_value %c1.ref.loc45
// CHECK:STDOUT:   %impl.elem0.loc45_41: %.a81 = impl_witness_access constants.%custom_witness.681, element0 [concrete = constants.%C.Op.068411.1]
// CHECK:STDOUT:   %bound_method.loc45_41: <bound method> = bound_method %.loc45_41.1, %impl.elem0.loc45_41
// CHECK:STDOUT:   %.loc45_41.2: ref %C = temporary_storage
// CHECK:STDOUT:   %Op.ref.loc45_41: %C.C.type.065f0c.2 = name_ref Op, imports.%C.C.decl.041267.2 [concrete = constants.%C.C.6522d8.2]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %.loc45_41.3: ref %C = value_as_ref %.loc45_41.1
// CHECK:STDOUT:   %addr.loc45_41.1: %ptr.d9e = addr_of %.loc45_41.3
// CHECK:STDOUT:   %.loc45_41.4: %ptr.c92 = as_compatible %addr.loc45_41.1
// CHECK:STDOUT:   %.loc45_41.5: %ptr.c92 = converted %addr.loc45_41.1, %.loc45_41.4
// CHECK:STDOUT:   %addr.loc45_41.2: %ptr.d9e = addr_of %lhs.var.16
// CHECK:STDOUT:   %C__carbon_thunk.call.loc45_41: init %empty_tuple.type = call imports.%C__carbon_thunk.decl.8acdfe.2(%.loc45_41.5, %addr.loc45_41.2)
// CHECK:STDOUT:   %.loc45_41.6: init %C to %lhs.var.16 = mark_in_place_init %C__carbon_thunk.call.loc45_41
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %.loc45_47.1: %C = acquire_value %c2.ref.loc45
// CHECK:STDOUT:   %impl.elem0.loc45_47: %.a81 = impl_witness_access constants.%custom_witness.681, element0 [concrete = constants.%C.Op.068411.1]
// CHECK:STDOUT:   %bound_method.loc45_47: <bound method> = bound_method %.loc45_47.1, %impl.elem0.loc45_47
// CHECK:STDOUT:   %.loc45_47.2: ref %C = temporary_storage
// CHECK:STDOUT:   %Op.ref.loc45_47: %C.C.type.065f0c.2 = name_ref Op, imports.%C.C.decl.041267.2 [concrete = constants.%C.C.6522d8.2]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %.loc45_47.3: ref %C = value_as_ref %.loc45_47.1
// CHECK:STDOUT:   %addr.loc45_47.1: %ptr.d9e = addr_of %.loc45_47.3
// CHECK:STDOUT:   %.loc45_47.4: %ptr.c92 = as_compatible %addr.loc45_47.1
// CHECK:STDOUT:   %.loc45_47.5: %ptr.c92 = converted %addr.loc45_47.1, %.loc45_47.4
// CHECK:STDOUT:   %addr.loc45_47.2: %ptr.d9e = addr_of %rhs.var.22
// CHECK:STDOUT:   %C__carbon_thunk.call.loc45_47: init %empty_tuple.type = call imports.%C__carbon_thunk.decl.8acdfe.2(%.loc45_47.5, %addr.loc45_47.2)
// CHECK:STDOUT:   %.loc45_47.6: init %C to %rhs.var.22 = mark_in_place_init %C__carbon_thunk.call.loc45_47
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %addr.loc45_44.1: %ptr.d9e = addr_of %lhs.var.16
// CHECK:STDOUT:   %addr.loc45_44.2: %ptr.d9e = addr_of %rhs.var.22
// CHECK:STDOUT:   %.loc45_44.1: ref bool = temporary_storage
// CHECK:STDOUT:   %addr.loc45_44.3: %ptr.bb2 = addr_of %.loc45_44.1
// CHECK:STDOUT:   %operator_LessEqual__carbon_thunk.call: init %empty_tuple.type = call imports.%operator_LessEqual__carbon_thunk.decl(%addr.loc45_44.1, %addr.loc45_44.2, %addr.loc45_44.3)
// CHECK:STDOUT:   %.loc45_44.2: init bool to %.loc45_44.1 = mark_in_place_init %operator_LessEqual__carbon_thunk.call
// CHECK:STDOUT:   %.loc45_34: type = type_literal bool [concrete = bool]
// CHECK:STDOUT:   %.loc45_44.3: ref bool = temporary %.loc45_44.1, %.loc45_44.2
// CHECK:STDOUT:   %.loc45_44.4: bool = acquire_value %.loc45_44.3
// CHECK:STDOUT:   %less_than_or_equal: bool = value_binding less_than_or_equal, %.loc45_44.4
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %index.patt: %pattern_type.7ce = value_binding_pattern index [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c1.ref.loc48: ref %C = name_ref c1, %c1
// CHECK:STDOUT:   %int_42: Core.IntLiteral = int_value 42 [concrete = constants.%int_42.20e]
// CHECK:STDOUT:   %C.cpp_operator.bound: <bound method> = bound_method %c1.ref.loc48, imports.%C.cpp_operator.decl
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %operator_Subscript__carbon_thunk.call: init %i32 = call imports.%operator_Subscript__carbon_thunk.decl(%c1.ref.loc48, %.17)
// CHECK:STDOUT:   %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %.loc48_32.1: %i32 = value_of_initializer %operator_Subscript__carbon_thunk.call
// CHECK:STDOUT:   %.loc48_32.2: %i32 = converted %operator_Subscript__carbon_thunk.call, %.loc48_32.1
// CHECK:STDOUT:   %index: %i32 = value_binding index, %.loc48_32.2
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %Destroy.Op.bound.loc45: <bound method> = bound_method %.loc45_44.3, constants.%Destroy.Op.651ba6.3
// CHECK:STDOUT:   %Destroy.Op.call.loc45: init %empty_tuple.type = call %Destroy.Op.bound.loc45(%.loc45_44.3)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %Destroy.Op.bound.loc44: <bound method> = bound_method %.loc44_47.3, constants.%Destroy.Op.651ba6.3
// CHECK:STDOUT:   %Destroy.Op.call.loc44: init %empty_tuple.type = call %Destroy.Op.bound.loc44(%.loc44_47.3)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %Destroy.Op.bound.loc43: <bound method> = bound_method %.loc43_35.3, constants.%Destroy.Op.651ba6.3
// CHECK:STDOUT:   %Destroy.Op.call.loc43: init %empty_tuple.type = call %Destroy.Op.bound.loc43(%.loc43_35.3)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %Destroy.Op.bound.loc42: <bound method> = bound_method %.loc42_38.3, constants.%Destroy.Op.651ba6.3
// CHECK:STDOUT:   %Destroy.Op.call.loc42: init %empty_tuple.type = call %Destroy.Op.bound.loc42(%.loc42_38.3)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %Destroy.Op.bound.loc41: <bound method> = bound_method %.loc41_35.3, constants.%Destroy.Op.651ba6.3
// CHECK:STDOUT:   %Destroy.Op.call.loc41: init %empty_tuple.type = call %Destroy.Op.bound.loc41(%.loc41_35.3)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %Destroy.Op.bound.loc40: <bound method> = bound_method %.loc40_31.3, constants.%Destroy.Op.651ba6.3
// CHECK:STDOUT:   %Destroy.Op.call.loc40: init %empty_tuple.type = call %Destroy.Op.bound.loc40(%.loc40_31.3)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %C.Op.bound.loc23: <bound method> = bound_method %.loc23_38.3, constants.%C.Op.068411.2
// CHECK:STDOUT:   %Op.ref.loc23_38: %C.cpp_destructor.type = name_ref Op, imports.%C.cpp_destructor.decl [concrete = constants.%C.cpp_destructor]
// CHECK:STDOUT:   %C.cpp_destructor.bound.loc23: <bound method> = bound_method %.loc23_38.3, %Op.ref.loc23_38
// CHECK:STDOUT:   %C.cpp_destructor.call.loc23: init %empty_tuple.type = call %C.cpp_destructor.bound.loc23(%.loc23_38.3)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %C.Op.bound.loc22: <bound method> = bound_method %.loc22_37.3, constants.%C.Op.068411.2
// CHECK:STDOUT:   %Op.ref.loc22_37: %C.cpp_destructor.type = name_ref Op, imports.%C.cpp_destructor.decl [concrete = constants.%C.cpp_destructor]
// CHECK:STDOUT:   %C.cpp_destructor.bound.loc22: <bound method> = bound_method %.loc22_37.3, %Op.ref.loc22_37
// CHECK:STDOUT:   %C.cpp_destructor.call.loc22: init %empty_tuple.type = call %C.cpp_destructor.bound.loc22(%.loc22_37.3)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %C.Op.bound.loc21: <bound method> = bound_method %.loc21_38.3, constants.%C.Op.068411.2
// CHECK:STDOUT:   %Op.ref.loc21_38: %C.cpp_destructor.type = name_ref Op, imports.%C.cpp_destructor.decl [concrete = constants.%C.cpp_destructor]
// CHECK:STDOUT:   %C.cpp_destructor.bound.loc21: <bound method> = bound_method %.loc21_38.3, %Op.ref.loc21_38
// CHECK:STDOUT:   %C.cpp_destructor.call.loc21: init %empty_tuple.type = call %C.cpp_destructor.bound.loc21(%.loc21_38.3)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %C.Op.bound.loc20: <bound method> = bound_method %.loc20_37.3, constants.%C.Op.068411.2
// CHECK:STDOUT:   %Op.ref.loc20_37: %C.cpp_destructor.type = name_ref Op, imports.%C.cpp_destructor.decl [concrete = constants.%C.cpp_destructor]
// CHECK:STDOUT:   %C.cpp_destructor.bound.loc20: <bound method> = bound_method %.loc20_37.3, %Op.ref.loc20_37
// CHECK:STDOUT:   %C.cpp_destructor.call.loc20: init %empty_tuple.type = call %C.cpp_destructor.bound.loc20(%.loc20_37.3)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %C.Op.bound.loc19: <bound method> = bound_method %.loc19_38.3, constants.%C.Op.068411.2
// CHECK:STDOUT:   %Op.ref.loc19_38: %C.cpp_destructor.type = name_ref Op, imports.%C.cpp_destructor.decl [concrete = constants.%C.cpp_destructor]
// CHECK:STDOUT:   %C.cpp_destructor.bound.loc19: <bound method> = bound_method %.loc19_38.3, %Op.ref.loc19_38
// CHECK:STDOUT:   %C.cpp_destructor.call.loc19: init %empty_tuple.type = call %C.cpp_destructor.bound.loc19(%.loc19_38.3)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %C.Op.bound.loc16: <bound method> = bound_method %.loc16_33.3, constants.%C.Op.068411.2
// CHECK:STDOUT:   %Op.ref.loc16_33: %C.cpp_destructor.type = name_ref Op, imports.%C.cpp_destructor.decl [concrete = constants.%C.cpp_destructor]
// CHECK:STDOUT:   %C.cpp_destructor.bound.loc16: <bound method> = bound_method %.loc16_33.3, %Op.ref.loc16_33
// CHECK:STDOUT:   %C.cpp_destructor.call.loc16: init %empty_tuple.type = call %C.cpp_destructor.bound.loc16(%.loc16_33.3)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %C.Op.bound.loc15: <bound method> = bound_method %.loc15_35.3, constants.%C.Op.068411.2
// CHECK:STDOUT:   %Op.ref.loc15_35: %C.cpp_destructor.type = name_ref Op, imports.%C.cpp_destructor.decl [concrete = constants.%C.cpp_destructor]
// CHECK:STDOUT:   %C.cpp_destructor.bound.loc15: <bound method> = bound_method %.loc15_35.3, %Op.ref.loc15_35
// CHECK:STDOUT:   %C.cpp_destructor.call.loc15: init %empty_tuple.type = call %C.cpp_destructor.bound.loc15(%.loc15_35.3)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %C.Op.bound.loc14: <bound method> = bound_method %.loc14_41.3, constants.%C.Op.068411.2
// CHECK:STDOUT:   %Op.ref.loc14_41: %C.cpp_destructor.type = name_ref Op, imports.%C.cpp_destructor.decl [concrete = constants.%C.cpp_destructor]
// CHECK:STDOUT:   %C.cpp_destructor.bound.loc14: <bound method> = bound_method %.loc14_41.3, %Op.ref.loc14_41
// CHECK:STDOUT:   %C.cpp_destructor.call.loc14: init %empty_tuple.type = call %C.cpp_destructor.bound.loc14(%.loc14_41.3)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %C.Op.bound.loc13: <bound method> = bound_method %.loc13_38.3, constants.%C.Op.068411.2
// CHECK:STDOUT:   %Op.ref.loc13_38: %C.cpp_destructor.type = name_ref Op, imports.%C.cpp_destructor.decl [concrete = constants.%C.cpp_destructor]
// CHECK:STDOUT:   %C.cpp_destructor.bound.loc13: <bound method> = bound_method %.loc13_38.3, %Op.ref.loc13_38
// CHECK:STDOUT:   %C.cpp_destructor.call.loc13: init %empty_tuple.type = call %C.cpp_destructor.bound.loc13(%.loc13_38.3)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %C.Op.bound.loc12: <bound method> = bound_method %.loc12_35.3, constants.%C.Op.068411.2
// CHECK:STDOUT:   %Op.ref.loc12_35: %C.cpp_destructor.type = name_ref Op, imports.%C.cpp_destructor.decl [concrete = constants.%C.cpp_destructor]
// CHECK:STDOUT:   %C.cpp_destructor.bound.loc12: <bound method> = bound_method %.loc12_35.3, %Op.ref.loc12_35
// CHECK:STDOUT:   %C.cpp_destructor.call.loc12: init %empty_tuple.type = call %C.cpp_destructor.bound.loc12(%.loc12_35.3)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %C.Op.bound.loc9: <bound method> = bound_method %c2.var, constants.%C.Op.068411.2
// CHECK:STDOUT:   %Op.ref.loc9: %C.cpp_destructor.type = name_ref Op, imports.%C.cpp_destructor.decl [concrete = constants.%C.cpp_destructor]
// CHECK:STDOUT:   %C.cpp_destructor.bound.loc9: <bound method> = bound_method %c2.var, %Op.ref.loc9
// CHECK:STDOUT:   %C.cpp_destructor.call.loc9: init %empty_tuple.type = call %C.cpp_destructor.bound.loc9(%c2.var)
// CHECK:STDOUT:   %C.Op.bound.loc8: <bound method> = bound_method %c1.var, constants.%C.Op.068411.2
// CHECK:STDOUT:   %Op.ref.loc8: %C.cpp_destructor.type = name_ref Op, imports.%C.cpp_destructor.decl [concrete = constants.%C.cpp_destructor]
// CHECK:STDOUT:   %C.cpp_destructor.bound.loc8: <bound method> = bound_method %c1.var, %Op.ref.loc8
// CHECK:STDOUT:   %C.cpp_destructor.call.loc8: init %empty_tuple.type = call %C.cpp_destructor.bound.loc8(%c1.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc45(%self.param: ref bool) = "no_op";
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
// CHECK:STDOUT:   %C__carbon_thunk.type.65f120.1: type = fn_type @C__carbon_thunk.1 [concrete]
// CHECK:STDOUT:   %C__carbon_thunk.d98342.1: %C__carbon_thunk.type.65f120.1 = struct_value () [concrete]
// CHECK:STDOUT:   %operator_Plus__carbon_thunk.type: type = fn_type @operator_Plus__carbon_thunk [concrete]
// CHECK:STDOUT:   %operator_Plus__carbon_thunk: %operator_Plus__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %C.C.type.065f0c.2: type = fn_type @C.C.2 [concrete]
// CHECK:STDOUT:   %C.C.6522d8.2: %C.C.type.065f0c.2 = struct_value () [concrete]
// CHECK:STDOUT:   %const.057: type = const_type %C [concrete]
// CHECK:STDOUT:   %ptr.c92: type = ptr_type %const.057 [concrete]
// CHECK:STDOUT:   %C__carbon_thunk.type.65f120.2: type = fn_type @C__carbon_thunk.2 [concrete]
// CHECK:STDOUT:   %C__carbon_thunk.d98342.2: %C__carbon_thunk.type.65f120.2 = struct_value () [concrete]
// CHECK:STDOUT:   %C.Op.type.ea1bc9.1: type = fn_type @C.Op.1 [concrete]
// CHECK:STDOUT:   %C.Op.068411.1: %C.Op.type.ea1bc9.1 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.681: <witness> = custom_witness (%C.Op.068411.1), @Copy [concrete]
// CHECK:STDOUT:   %Copy.facet.3b9: %Copy.type = facet_value %C, (%custom_witness.681) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.530: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet.3b9) [concrete]
// CHECK:STDOUT:   %.a81: type = fn_type_with_self_type %Copy.WithSelf.Op.type.530, %Copy.facet.3b9 [concrete]
// CHECK:STDOUT:   %C.cpp_destructor.type: type = fn_type @C.cpp_destructor [concrete]
// CHECK:STDOUT:   %C.cpp_destructor: %C.cpp_destructor.type = struct_value () [concrete]
// CHECK:STDOUT:   %C.Op.type.ea1bc9.2: type = fn_type @C.Op.2 [concrete]
// CHECK:STDOUT:   %C.Op.068411.2: %C.Op.type.ea1bc9.2 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   %C.C.cpp_overload_set.value: %C.C.cpp_overload_set.type = cpp_overload_set_value @C.C.cpp_overload_set [concrete = constants.%C.C.cpp_overload_set.value]
// CHECK:STDOUT:   %C__carbon_thunk.decl.8acdfe.1: %C__carbon_thunk.type.65f120.1 = fn_decl @C__carbon_thunk.1 [concrete = constants.%C__carbon_thunk.d98342.1] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %operator_Plus__carbon_thunk.decl: %operator_Plus__carbon_thunk.type = fn_decl @operator_Plus__carbon_thunk [concrete = constants.%operator_Plus__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.C.decl.041267.2: %C.C.type.065f0c.2 = fn_decl @C.C.2 [concrete = constants.%C.C.6522d8.2] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C__carbon_thunk.decl.8acdfe.2: %C__carbon_thunk.type.65f120.2 = fn_decl @C__carbon_thunk.2 [concrete = constants.%C__carbon_thunk.d98342.2] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.cpp_destructor.decl: %C.cpp_destructor.type = fn_decl @C.cpp_destructor [concrete = constants.%C.cpp_destructor] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c1.patt: %pattern_type.217 = value_binding_pattern c1 [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc8_19: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %C.ref.loc8_22: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %C.ref.loc8_24: %C.C.cpp_overload_set.type = name_ref C, imports.%C.C.cpp_overload_set.value [concrete = constants.%C.C.cpp_overload_set.value]
// CHECK:STDOUT:   %.loc8_27.1: ref %C = temporary_storage
// CHECK:STDOUT:   %addr.loc8: %ptr.d9e = addr_of %.loc8_27.1
// CHECK:STDOUT:   %C__carbon_thunk.call.loc8: init %empty_tuple.type = call imports.%C__carbon_thunk.decl.8acdfe.1(%addr.loc8)
// CHECK:STDOUT:   %.loc8_27.2: init %C to %.loc8_27.1 = mark_in_place_init %C__carbon_thunk.call.loc8
// CHECK:STDOUT:   %.loc8_14: type = splice_block %C.ref.loc8_14 [concrete = constants.%C] {
// CHECK:STDOUT:     %Cpp.ref.loc8_11: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %C.ref.loc8_14: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc8_27.3: ref %C = temporary %.loc8_27.1, %.loc8_27.2
// CHECK:STDOUT:   %.loc8_27.4: %C = acquire_value %.loc8_27.3
// CHECK:STDOUT:   %c1: %C = value_binding c1, %.loc8_27.4
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c2.patt: %pattern_type.217 = value_binding_pattern c2 [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc9_19: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %C.ref.loc9_22: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %C.ref.loc9_24: %C.C.cpp_overload_set.type = name_ref C, imports.%C.C.cpp_overload_set.value [concrete = constants.%C.C.cpp_overload_set.value]
// CHECK:STDOUT:   %.loc9_27.1: ref %C = temporary_storage
// CHECK:STDOUT:   %addr.loc9: %ptr.d9e = addr_of %.loc9_27.1
// CHECK:STDOUT:   %C__carbon_thunk.call.loc9: init %empty_tuple.type = call imports.%C__carbon_thunk.decl.8acdfe.1(%addr.loc9)
// CHECK:STDOUT:   %.loc9_27.2: init %C to %.loc9_27.1 = mark_in_place_init %C__carbon_thunk.call.loc9
// CHECK:STDOUT:   %.loc9_14: type = splice_block %C.ref.loc9_14 [concrete = constants.%C] {
// CHECK:STDOUT:     %Cpp.ref.loc9_11: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %C.ref.loc9_14: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc9_27.3: ref %C = temporary %.loc9_27.1, %.loc9_27.2
// CHECK:STDOUT:   %.loc9_27.4: %C = acquire_value %.loc9_27.3
// CHECK:STDOUT:   %c2: %C = value_binding c2, %.loc9_27.4
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c3.patt: %pattern_type.217 = value_binding_pattern c3 [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c1.ref.loc10: %C = name_ref c1, %c1
// CHECK:STDOUT:   %c2.ref: %C = name_ref c2, %c2
// CHECK:STDOUT:   %.loc10_22.1: ref %C = temporary_storage
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %impl.elem0.loc10_19: %.a81 = impl_witness_access constants.%custom_witness.681, element0 [concrete = constants.%C.Op.068411.1]
// CHECK:STDOUT:   %bound_method.loc10_19: <bound method> = bound_method %c1.ref.loc10, %impl.elem0.loc10_19
// CHECK:STDOUT:   %.loc10_19.1: ref %C = temporary_storage
// CHECK:STDOUT:   %Op.ref.loc10_19: %C.C.type.065f0c.2 = name_ref Op, imports.%C.C.decl.041267.2 [concrete = constants.%C.C.6522d8.2]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %.loc10_19.2: ref %C = value_as_ref %c1.ref.loc10
// CHECK:STDOUT:   %addr.loc10_19.1: %ptr.d9e = addr_of %.loc10_19.2
// CHECK:STDOUT:   %.loc10_19.3: %ptr.c92 = as_compatible %addr.loc10_19.1
// CHECK:STDOUT:   %.loc10_19.4: %ptr.c92 = converted %addr.loc10_19.1, %.loc10_19.3
// CHECK:STDOUT:   %addr.loc10_19.2: %ptr.d9e = addr_of %lhs.var.1
// CHECK:STDOUT:   %C__carbon_thunk.call.loc10_19: init %empty_tuple.type = call imports.%C__carbon_thunk.decl.8acdfe.2(%.loc10_19.4, %addr.loc10_19.2)
// CHECK:STDOUT:   %.loc10_19.5: init %C to %lhs.var.1 = mark_in_place_init %C__carbon_thunk.call.loc10_19
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %impl.elem0.loc10_24: %.a81 = impl_witness_access constants.%custom_witness.681, element0 [concrete = constants.%C.Op.068411.1]
// CHECK:STDOUT:   %bound_method.loc10_24: <bound method> = bound_method %c2.ref, %impl.elem0.loc10_24
// CHECK:STDOUT:   %.loc10_24.1: ref %C = temporary_storage
// CHECK:STDOUT:   %Op.ref.loc10_24: %C.C.type.065f0c.2 = name_ref Op, imports.%C.C.decl.041267.2 [concrete = constants.%C.C.6522d8.2]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %.loc10_24.2: ref %C = value_as_ref %c2.ref
// CHECK:STDOUT:   %addr.loc10_24.1: %ptr.d9e = addr_of %.loc10_24.2
// CHECK:STDOUT:   %.loc10_24.3: %ptr.c92 = as_compatible %addr.loc10_24.1
// CHECK:STDOUT:   %.loc10_24.4: %ptr.c92 = converted %addr.loc10_24.1, %.loc10_24.3
// CHECK:STDOUT:   %addr.loc10_24.2: %ptr.d9e = addr_of %rhs.var.1
// CHECK:STDOUT:   %C__carbon_thunk.call.loc10_24: init %empty_tuple.type = call imports.%C__carbon_thunk.decl.8acdfe.2(%.loc10_24.4, %addr.loc10_24.2)
// CHECK:STDOUT:   %.loc10_24.5: init %C to %rhs.var.1 = mark_in_place_init %C__carbon_thunk.call.loc10_24
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %addr.loc10_22.1: %ptr.d9e = addr_of %lhs.var.1
// CHECK:STDOUT:   %addr.loc10_22.2: %ptr.d9e = addr_of %rhs.var.1
// CHECK:STDOUT:   %addr.loc10_22.3: %ptr.d9e = addr_of %.loc10_22.1
// CHECK:STDOUT:   %operator_Plus__carbon_thunk.call.loc10: init %empty_tuple.type = call imports.%operator_Plus__carbon_thunk.decl(%addr.loc10_22.1, %addr.loc10_22.2, %addr.loc10_22.3)
// CHECK:STDOUT:   %.loc10_22.2: init %C to %.loc10_22.1 = mark_in_place_init %operator_Plus__carbon_thunk.call.loc10
// CHECK:STDOUT:   %.loc10_14: type = splice_block %C.ref.loc10 [concrete = constants.%C] {
// CHECK:STDOUT:     %Cpp.ref.loc10: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %C.ref.loc10: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc10_22.3: ref %C = temporary %.loc10_22.1, %.loc10_22.2
// CHECK:STDOUT:   %.loc10_22.4: %C = acquire_value %.loc10_22.3
// CHECK:STDOUT:   %c3: %C = value_binding c3, %.loc10_22.4
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c4.patt: %pattern_type.217 = value_binding_pattern c4 [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c1.ref.loc11: %C = name_ref c1, %c1
// CHECK:STDOUT:   %c3.ref.loc11: %C = name_ref c3, %c3
// CHECK:STDOUT:   %.loc11_22.1: ref %C = temporary_storage
// CHECK:STDOUT:   %impl.elem0.loc11_19: %.a81 = impl_witness_access constants.%custom_witness.681, element0 [concrete = constants.%C.Op.068411.1]
// CHECK:STDOUT:   %bound_method.loc11_19: <bound method> = bound_method %c1.ref.loc11, %impl.elem0.loc11_19
// CHECK:STDOUT:   %.loc11_19.1: ref %C = temporary_storage
// CHECK:STDOUT:   %Op.ref.loc11_19: %C.C.type.065f0c.2 = name_ref Op, imports.%C.C.decl.041267.2 [concrete = constants.%C.C.6522d8.2]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %.loc11_19.2: ref %C = value_as_ref %c1.ref.loc11
// CHECK:STDOUT:   %addr.loc11_19.1: %ptr.d9e = addr_of %.loc11_19.2
// CHECK:STDOUT:   %.loc11_19.3: %ptr.c92 = as_compatible %addr.loc11_19.1
// CHECK:STDOUT:   %.loc11_19.4: %ptr.c92 = converted %addr.loc11_19.1, %.loc11_19.3
// CHECK:STDOUT:   %addr.loc11_19.2: %ptr.d9e = addr_of %lhs.var.2
// CHECK:STDOUT:   %C__carbon_thunk.call.loc11_19: init %empty_tuple.type = call imports.%C__carbon_thunk.decl.8acdfe.2(%.loc11_19.4, %addr.loc11_19.2)
// CHECK:STDOUT:   %.loc11_19.5: init %C to %lhs.var.2 = mark_in_place_init %C__carbon_thunk.call.loc11_19
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %impl.elem0.loc11_24: %.a81 = impl_witness_access constants.%custom_witness.681, element0 [concrete = constants.%C.Op.068411.1]
// CHECK:STDOUT:   %bound_method.loc11_24: <bound method> = bound_method %c3.ref.loc11, %impl.elem0.loc11_24
// CHECK:STDOUT:   %.loc11_24.1: ref %C = temporary_storage
// CHECK:STDOUT:   %Op.ref.loc11_24: %C.C.type.065f0c.2 = name_ref Op, imports.%C.C.decl.041267.2 [concrete = constants.%C.C.6522d8.2]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %.loc11_24.2: ref %C = value_as_ref %c3.ref.loc11
// CHECK:STDOUT:   %addr.loc11_24.1: %ptr.d9e = addr_of %.loc11_24.2
// CHECK:STDOUT:   %.loc11_24.3: %ptr.c92 = as_compatible %addr.loc11_24.1
// CHECK:STDOUT:   %.loc11_24.4: %ptr.c92 = converted %addr.loc11_24.1, %.loc11_24.3
// CHECK:STDOUT:   %addr.loc11_24.2: %ptr.d9e = addr_of %rhs.var.2
// CHECK:STDOUT:   %C__carbon_thunk.call.loc11_24: init %empty_tuple.type = call imports.%C__carbon_thunk.decl.8acdfe.2(%.loc11_24.4, %addr.loc11_24.2)
// CHECK:STDOUT:   %.loc11_24.5: init %C to %rhs.var.2 = mark_in_place_init %C__carbon_thunk.call.loc11_24
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %addr.loc11_22.1: %ptr.d9e = addr_of %lhs.var.2
// CHECK:STDOUT:   %addr.loc11_22.2: %ptr.d9e = addr_of %rhs.var.2
// CHECK:STDOUT:   %addr.loc11_22.3: %ptr.d9e = addr_of %.loc11_22.1
// CHECK:STDOUT:   %operator_Plus__carbon_thunk.call.loc11: init %empty_tuple.type = call imports.%operator_Plus__carbon_thunk.decl(%addr.loc11_22.1, %addr.loc11_22.2, %addr.loc11_22.3)
// CHECK:STDOUT:   %.loc11_22.2: init %C to %.loc11_22.1 = mark_in_place_init %operator_Plus__carbon_thunk.call.loc11
// CHECK:STDOUT:   %.loc11_14: type = splice_block %C.ref.loc11 [concrete = constants.%C] {
// CHECK:STDOUT:     %Cpp.ref.loc11: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %C.ref.loc11: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc11_22.3: ref %C = temporary %.loc11_22.1, %.loc11_22.2
// CHECK:STDOUT:   %.loc11_22.4: %C = acquire_value %.loc11_22.3
// CHECK:STDOUT:   %c4: %C = value_binding c4, %.loc11_22.4
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c5.patt: %pattern_type.217 = value_binding_pattern c5 [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c4.ref: %C = name_ref c4, %c4
// CHECK:STDOUT:   %c3.ref.loc12: %C = name_ref c3, %c3
// CHECK:STDOUT:   %.loc12_29.1: ref %C = temporary_storage
// CHECK:STDOUT:   %impl.elem0.loc12_26: %.a81 = impl_witness_access constants.%custom_witness.681, element0 [concrete = constants.%C.Op.068411.1]
// CHECK:STDOUT:   %bound_method.loc12_26: <bound method> = bound_method %c4.ref, %impl.elem0.loc12_26
// CHECK:STDOUT:   %.loc12_26.1: ref %C = temporary_storage
// CHECK:STDOUT:   %Op.ref.loc12_26: %C.C.type.065f0c.2 = name_ref Op, imports.%C.C.decl.041267.2 [concrete = constants.%C.C.6522d8.2]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %.loc12_26.2: ref %C = value_as_ref %c4.ref
// CHECK:STDOUT:   %addr.loc12_26.1: %ptr.d9e = addr_of %.loc12_26.2
// CHECK:STDOUT:   %.loc12_26.3: %ptr.c92 = as_compatible %addr.loc12_26.1
// CHECK:STDOUT:   %.loc12_26.4: %ptr.c92 = converted %addr.loc12_26.1, %.loc12_26.3
// CHECK:STDOUT:   %addr.loc12_26.2: %ptr.d9e = addr_of %lhs.var.3
// CHECK:STDOUT:   %C__carbon_thunk.call.loc12_26: init %empty_tuple.type = call imports.%C__carbon_thunk.decl.8acdfe.2(%.loc12_26.4, %addr.loc12_26.2)
// CHECK:STDOUT:   %.loc12_26.5: init %C to %lhs.var.3 = mark_in_place_init %C__carbon_thunk.call.loc12_26
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %impl.elem0.loc12_31: %.a81 = impl_witness_access constants.%custom_witness.681, element0 [concrete = constants.%C.Op.068411.1]
// CHECK:STDOUT:   %bound_method.loc12_31: <bound method> = bound_method %c3.ref.loc12, %impl.elem0.loc12_31
// CHECK:STDOUT:   %.loc12_31.1: ref %C = temporary_storage
// CHECK:STDOUT:   %Op.ref.loc12_31: %C.C.type.065f0c.2 = name_ref Op, imports.%C.C.decl.041267.2 [concrete = constants.%C.C.6522d8.2]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %.loc12_31.2: ref %C = value_as_ref %c3.ref.loc12
// CHECK:STDOUT:   %addr.loc12_31.1: %ptr.d9e = addr_of %.loc12_31.2
// CHECK:STDOUT:   %.loc12_31.3: %ptr.c92 = as_compatible %addr.loc12_31.1
// CHECK:STDOUT:   %.loc12_31.4: %ptr.c92 = converted %addr.loc12_31.1, %.loc12_31.3
// CHECK:STDOUT:   %addr.loc12_31.2: %ptr.d9e = addr_of %rhs.var.3
// CHECK:STDOUT:   %C__carbon_thunk.call.loc12_31: init %empty_tuple.type = call imports.%C__carbon_thunk.decl.8acdfe.2(%.loc12_31.4, %addr.loc12_31.2)
// CHECK:STDOUT:   %.loc12_31.5: init %C to %rhs.var.3 = mark_in_place_init %C__carbon_thunk.call.loc12_31
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %addr.loc12_29.1: %ptr.d9e = addr_of %lhs.var.3
// CHECK:STDOUT:   %addr.loc12_29.2: %ptr.d9e = addr_of %rhs.var.3
// CHECK:STDOUT:   %addr.loc12_29.3: %ptr.d9e = addr_of %.loc12_29.1
// CHECK:STDOUT:   %operator_Plus__carbon_thunk.call.loc12: init %empty_tuple.type = call imports.%operator_Plus__carbon_thunk.decl(%addr.loc12_29.1, %addr.loc12_29.2, %addr.loc12_29.3)
// CHECK:STDOUT:   %.loc12_29.2: init %C to %.loc12_29.1 = mark_in_place_init %operator_Plus__carbon_thunk.call.loc12
// CHECK:STDOUT:   %.loc12_21: type = splice_block %C.ref.loc12 [concrete = constants.%C] {
// CHECK:STDOUT:     %Cpp.ref.loc12: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %C.ref.loc12: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc12_29.3: ref %C = temporary %.loc12_29.1, %.loc12_29.2
// CHECK:STDOUT:   %.loc12_29.4: %C = acquire_value %.loc12_29.3
// CHECK:STDOUT:   %c5: %C = value_binding c5, %.loc12_29.4
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %C.Op.bound.loc12: <bound method> = bound_method %.loc12_29.3, constants.%C.Op.068411.2
// CHECK:STDOUT:   %Op.ref.loc12_29: %C.cpp_destructor.type = name_ref Op, imports.%C.cpp_destructor.decl [concrete = constants.%C.cpp_destructor]
// CHECK:STDOUT:   %C.cpp_destructor.bound.loc12: <bound method> = bound_method %.loc12_29.3, %Op.ref.loc12_29
// CHECK:STDOUT:   %C.cpp_destructor.call.loc12: init %empty_tuple.type = call %C.cpp_destructor.bound.loc12(%.loc12_29.3)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %C.Op.bound.loc11: <bound method> = bound_method %.loc11_22.3, constants.%C.Op.068411.2
// CHECK:STDOUT:   %Op.ref.loc11_22: %C.cpp_destructor.type = name_ref Op, imports.%C.cpp_destructor.decl [concrete = constants.%C.cpp_destructor]
// CHECK:STDOUT:   %C.cpp_destructor.bound.loc11: <bound method> = bound_method %.loc11_22.3, %Op.ref.loc11_22
// CHECK:STDOUT:   %C.cpp_destructor.call.loc11: init %empty_tuple.type = call %C.cpp_destructor.bound.loc11(%.loc11_22.3)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %C.Op.bound.loc10: <bound method> = bound_method %.loc10_22.3, constants.%C.Op.068411.2
// CHECK:STDOUT:   %Op.ref.loc10_22: %C.cpp_destructor.type = name_ref Op, imports.%C.cpp_destructor.decl [concrete = constants.%C.cpp_destructor]
// CHECK:STDOUT:   %C.cpp_destructor.bound.loc10: <bound method> = bound_method %.loc10_22.3, %Op.ref.loc10_22
// CHECK:STDOUT:   %C.cpp_destructor.call.loc10: init %empty_tuple.type = call %C.cpp_destructor.bound.loc10(%.loc10_22.3)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %C.Op.bound.loc9: <bound method> = bound_method %.loc9_27.3, constants.%C.Op.068411.2
// CHECK:STDOUT:   %Op.ref.loc9: %C.cpp_destructor.type = name_ref Op, imports.%C.cpp_destructor.decl [concrete = constants.%C.cpp_destructor]
// CHECK:STDOUT:   %C.cpp_destructor.bound.loc9: <bound method> = bound_method %.loc9_27.3, %Op.ref.loc9
// CHECK:STDOUT:   %C.cpp_destructor.call.loc9: init %empty_tuple.type = call %C.cpp_destructor.bound.loc9(%.loc9_27.3)
// CHECK:STDOUT:   %C.Op.bound.loc8: <bound method> = bound_method %.loc8_27.3, constants.%C.Op.068411.2
// CHECK:STDOUT:   %Op.ref.loc8: %C.cpp_destructor.type = name_ref Op, imports.%C.cpp_destructor.decl [concrete = constants.%C.cpp_destructor]
// CHECK:STDOUT:   %C.cpp_destructor.bound.loc8: <bound method> = bound_method %.loc8_27.3, %Op.ref.loc8
// CHECK:STDOUT:   %C.cpp_destructor.call.loc8: init %empty_tuple.type = call %C.cpp_destructor.bound.loc8(%.loc8_27.3)
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
// CHECK:STDOUT:   %C__carbon_thunk.type.65f120.1: type = fn_type @C__carbon_thunk.1 [concrete]
// CHECK:STDOUT:   %C__carbon_thunk.d98342.1: %C__carbon_thunk.type.65f120.1 = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.831: type = pattern_type bool [concrete]
// CHECK:STDOUT:   %ptr.bb2: type = ptr_type bool [concrete]
// CHECK:STDOUT:   %operator_Greater__carbon_thunk.type: type = fn_type @operator_Greater__carbon_thunk [concrete]
// CHECK:STDOUT:   %operator_Greater__carbon_thunk: %operator_Greater__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %C.C.type.065f0c.2: type = fn_type @C.C.2 [concrete]
// CHECK:STDOUT:   %C.C.6522d8.2: %C.C.type.065f0c.2 = struct_value () [concrete]
// CHECK:STDOUT:   %const.057: type = const_type %C [concrete]
// CHECK:STDOUT:   %ptr.c92: type = ptr_type %const.057 [concrete]
// CHECK:STDOUT:   %C__carbon_thunk.type.65f120.2: type = fn_type @C__carbon_thunk.2 [concrete]
// CHECK:STDOUT:   %C__carbon_thunk.d98342.2: %C__carbon_thunk.type.65f120.2 = struct_value () [concrete]
// CHECK:STDOUT:   %C.Op.type.ea1bc9.1: type = fn_type @C.Op.1 [concrete]
// CHECK:STDOUT:   %C.Op.068411.1: %C.Op.type.ea1bc9.1 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.681: <witness> = custom_witness (%C.Op.068411.1), @Copy [concrete]
// CHECK:STDOUT:   %Copy.facet.3b9: %Copy.type = facet_value %C, (%custom_witness.681) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.530: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet.3b9) [concrete]
// CHECK:STDOUT:   %.a81: type = fn_type_with_self_type %Copy.WithSelf.Op.type.530, %Copy.facet.3b9 [concrete]
// CHECK:STDOUT:   %operator_LessEqual__carbon_thunk.type: type = fn_type @operator_LessEqual__carbon_thunk [concrete]
// CHECK:STDOUT:   %operator_LessEqual__carbon_thunk: %operator_LessEqual__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.Op.type: type = fn_type @Destroy.Op [concrete]
// CHECK:STDOUT:   %Destroy.Op: %Destroy.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %C.cpp_destructor.type: type = fn_type @C.cpp_destructor [concrete]
// CHECK:STDOUT:   %C.cpp_destructor: %C.cpp_destructor.type = struct_value () [concrete]
// CHECK:STDOUT:   %C.Op.type.ea1bc9.2: type = fn_type @C.Op.2 [concrete]
// CHECK:STDOUT:   %C.Op.068411.2: %C.Op.type.ea1bc9.2 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   %C.C.cpp_overload_set.value: %C.C.cpp_overload_set.type = cpp_overload_set_value @C.C.cpp_overload_set [concrete = constants.%C.C.cpp_overload_set.value]
// CHECK:STDOUT:   %C__carbon_thunk.decl.8acdfe.1: %C__carbon_thunk.type.65f120.1 = fn_decl @C__carbon_thunk.1 [concrete = constants.%C__carbon_thunk.d98342.1] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %operator_Greater__carbon_thunk.decl: %operator_Greater__carbon_thunk.type = fn_decl @operator_Greater__carbon_thunk [concrete = constants.%operator_Greater__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.C.decl.041267.2: %C.C.type.065f0c.2 = fn_decl @C.C.2 [concrete = constants.%C.C.6522d8.2] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C__carbon_thunk.decl.8acdfe.2: %C__carbon_thunk.type.65f120.2 = fn_decl @C__carbon_thunk.2 [concrete = constants.%C__carbon_thunk.d98342.2] {
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
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
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
// CHECK:STDOUT:   %Cpp.ref.loc15_19: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %C.ref.loc15_22: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %C.ref.loc15_24: %C.C.cpp_overload_set.type = name_ref C, imports.%C.C.cpp_overload_set.value [concrete = constants.%C.C.cpp_overload_set.value]
// CHECK:STDOUT:   %.loc15_3: ref %C = splice_block %c1.var {}
// CHECK:STDOUT:   %addr.loc15: %ptr.d9e = addr_of %.loc15_3
// CHECK:STDOUT:   %C__carbon_thunk.call.loc15: init %empty_tuple.type = call imports.%C__carbon_thunk.decl.8acdfe.1(%addr.loc15)
// CHECK:STDOUT:   %.loc15_27: init %C to %.loc15_3 = mark_in_place_init %C__carbon_thunk.call.loc15
// CHECK:STDOUT:   assign %c1.var, %.loc15_27
// CHECK:STDOUT:   %.loc15_14: type = splice_block %C.ref.loc15_14 [concrete = constants.%C] {
// CHECK:STDOUT:     %Cpp.ref.loc15_11: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %C.ref.loc15_14: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c1: ref %C = ref_binding c1, %c1.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c2.patt: %pattern_type.217 = ref_binding_pattern c2 [concrete]
// CHECK:STDOUT:     %c2.var_patt: %pattern_type.217 = var_pattern %c2.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c2.var: ref %C = var %c2.var_patt
// CHECK:STDOUT:   %Cpp.ref.loc16_19: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %C.ref.loc16_22: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %C.ref.loc16_24: %C.C.cpp_overload_set.type = name_ref C, imports.%C.C.cpp_overload_set.value [concrete = constants.%C.C.cpp_overload_set.value]
// CHECK:STDOUT:   %.loc16_3: ref %C = splice_block %c2.var {}
// CHECK:STDOUT:   %addr.loc16: %ptr.d9e = addr_of %.loc16_3
// CHECK:STDOUT:   %C__carbon_thunk.call.loc16: init %empty_tuple.type = call imports.%C__carbon_thunk.decl.8acdfe.1(%addr.loc16)
// CHECK:STDOUT:   %.loc16_27: init %C to %.loc16_3 = mark_in_place_init %C__carbon_thunk.call.loc16
// CHECK:STDOUT:   assign %c2.var, %.loc16_27
// CHECK:STDOUT:   %.loc16_14: type = splice_block %C.ref.loc16_14 [concrete = constants.%C] {
// CHECK:STDOUT:     %Cpp.ref.loc16_11: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %C.ref.loc16_14: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c2: ref %C = ref_binding c2, %c2.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %greater_than.patt: %pattern_type.831 = value_binding_pattern greater_than [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c1.ref.loc19: ref %C = name_ref c1, %c1
// CHECK:STDOUT:   %c2.ref.loc19: ref %C = name_ref c2, %c2
// CHECK:STDOUT:   %.loc19_35.1: %C = acquire_value %c1.ref.loc19
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %impl.elem0.loc19_35: %.a81 = impl_witness_access constants.%custom_witness.681, element0 [concrete = constants.%C.Op.068411.1]
// CHECK:STDOUT:   %bound_method.loc19_35: <bound method> = bound_method %.loc19_35.1, %impl.elem0.loc19_35
// CHECK:STDOUT:   %.loc19_35.2: ref %C = temporary_storage
// CHECK:STDOUT:   %Op.ref.loc19_35: %C.C.type.065f0c.2 = name_ref Op, imports.%C.C.decl.041267.2 [concrete = constants.%C.C.6522d8.2]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %.loc19_35.3: ref %C = value_as_ref %.loc19_35.1
// CHECK:STDOUT:   %addr.loc19_35.1: %ptr.d9e = addr_of %.loc19_35.3
// CHECK:STDOUT:   %.loc19_35.4: %ptr.c92 = as_compatible %addr.loc19_35.1
// CHECK:STDOUT:   %.loc19_35.5: %ptr.c92 = converted %addr.loc19_35.1, %.loc19_35.4
// CHECK:STDOUT:   %addr.loc19_35.2: %ptr.d9e = addr_of %lhs.var.1
// CHECK:STDOUT:   %C__carbon_thunk.call.loc19_35: init %empty_tuple.type = call imports.%C__carbon_thunk.decl.8acdfe.2(%.loc19_35.5, %addr.loc19_35.2)
// CHECK:STDOUT:   %.loc19_35.6: init %C to %lhs.var.1 = mark_in_place_init %C__carbon_thunk.call.loc19_35
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %.loc19_40.1: %C = acquire_value %c2.ref.loc19
// CHECK:STDOUT:   %impl.elem0.loc19_40: %.a81 = impl_witness_access constants.%custom_witness.681, element0 [concrete = constants.%C.Op.068411.1]
// CHECK:STDOUT:   %bound_method.loc19_40: <bound method> = bound_method %.loc19_40.1, %impl.elem0.loc19_40
// CHECK:STDOUT:   %.loc19_40.2: ref %C = temporary_storage
// CHECK:STDOUT:   %Op.ref.loc19_40: %C.C.type.065f0c.2 = name_ref Op, imports.%C.C.decl.041267.2 [concrete = constants.%C.C.6522d8.2]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %.loc19_40.3: ref %C = value_as_ref %.loc19_40.1
// CHECK:STDOUT:   %addr.loc19_40.1: %ptr.d9e = addr_of %.loc19_40.3
// CHECK:STDOUT:   %.loc19_40.4: %ptr.c92 = as_compatible %addr.loc19_40.1
// CHECK:STDOUT:   %.loc19_40.5: %ptr.c92 = converted %addr.loc19_40.1, %.loc19_40.4
// CHECK:STDOUT:   %addr.loc19_40.2: %ptr.d9e = addr_of %rhs.var.1
// CHECK:STDOUT:   %C__carbon_thunk.call.loc19_40: init %empty_tuple.type = call imports.%C__carbon_thunk.decl.8acdfe.2(%.loc19_40.5, %addr.loc19_40.2)
// CHECK:STDOUT:   %.loc19_40.6: init %C to %rhs.var.1 = mark_in_place_init %C__carbon_thunk.call.loc19_40
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %addr.loc19_38.1: %ptr.d9e = addr_of %lhs.var.1
// CHECK:STDOUT:   %addr.loc19_38.2: %ptr.d9e = addr_of %rhs.var.1
// CHECK:STDOUT:   %.loc19_38.1: ref bool = temporary_storage
// CHECK:STDOUT:   %addr.loc19_38.3: %ptr.bb2 = addr_of %.loc19_38.1
// CHECK:STDOUT:   %operator_Greater__carbon_thunk.call: init %empty_tuple.type = call imports.%operator_Greater__carbon_thunk.decl(%addr.loc19_38.1, %addr.loc19_38.2, %addr.loc19_38.3)
// CHECK:STDOUT:   %.loc19_38.2: init bool to %.loc19_38.1 = mark_in_place_init %operator_Greater__carbon_thunk.call
// CHECK:STDOUT:   %.loc19_28: type = type_literal bool [concrete = bool]
// CHECK:STDOUT:   %.loc19_38.3: ref bool = temporary %.loc19_38.1, %.loc19_38.2
// CHECK:STDOUT:   %.loc19_38.4: bool = acquire_value %.loc19_38.3
// CHECK:STDOUT:   %greater_than: bool = value_binding greater_than, %.loc19_38.4
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %less_than.patt: %pattern_type.831 = value_binding_pattern less_than [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c1.ref.loc26: ref %C = name_ref c1, %c1
// CHECK:STDOUT:   %c2.ref.loc26: ref %C = name_ref c2, %c2
// CHECK:STDOUT:   %.loc26: type = type_literal bool [concrete = bool]
// CHECK:STDOUT:   %less_than: bool = value_binding less_than, <error> [concrete = <error>]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %greater_than_or_equal.patt: %pattern_type.831 = value_binding_pattern greater_than_or_equal [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c1.ref.loc33: ref %C = name_ref c1, %c1
// CHECK:STDOUT:   %c2.ref.loc33: ref %C = name_ref c2, %c2
// CHECK:STDOUT:   %.loc33: type = type_literal bool [concrete = bool]
// CHECK:STDOUT:   %greater_than_or_equal: bool = value_binding greater_than_or_equal, <error> [concrete = <error>]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %less_than_or_equal.patt: %pattern_type.831 = value_binding_pattern less_than_or_equal [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c1.ref.loc36: ref %C = name_ref c1, %c1
// CHECK:STDOUT:   %c2.ref.loc36: ref %C = name_ref c2, %c2
// CHECK:STDOUT:   %.loc36_41.1: %C = acquire_value %c1.ref.loc36
// CHECK:STDOUT:   %impl.elem0.loc36_41: %.a81 = impl_witness_access constants.%custom_witness.681, element0 [concrete = constants.%C.Op.068411.1]
// CHECK:STDOUT:   %bound_method.loc36_41: <bound method> = bound_method %.loc36_41.1, %impl.elem0.loc36_41
// CHECK:STDOUT:   %.loc36_41.2: ref %C = temporary_storage
// CHECK:STDOUT:   %Op.ref.loc36_41: %C.C.type.065f0c.2 = name_ref Op, imports.%C.C.decl.041267.2 [concrete = constants.%C.C.6522d8.2]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %.loc36_41.3: ref %C = value_as_ref %.loc36_41.1
// CHECK:STDOUT:   %addr.loc36_41.1: %ptr.d9e = addr_of %.loc36_41.3
// CHECK:STDOUT:   %.loc36_41.4: %ptr.c92 = as_compatible %addr.loc36_41.1
// CHECK:STDOUT:   %.loc36_41.5: %ptr.c92 = converted %addr.loc36_41.1, %.loc36_41.4
// CHECK:STDOUT:   %addr.loc36_41.2: %ptr.d9e = addr_of %lhs.var.2
// CHECK:STDOUT:   %C__carbon_thunk.call.loc36_41: init %empty_tuple.type = call imports.%C__carbon_thunk.decl.8acdfe.2(%.loc36_41.5, %addr.loc36_41.2)
// CHECK:STDOUT:   %.loc36_41.6: init %C to %lhs.var.2 = mark_in_place_init %C__carbon_thunk.call.loc36_41
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %.loc36_47.1: %C = acquire_value %c2.ref.loc36
// CHECK:STDOUT:   %impl.elem0.loc36_47: %.a81 = impl_witness_access constants.%custom_witness.681, element0 [concrete = constants.%C.Op.068411.1]
// CHECK:STDOUT:   %bound_method.loc36_47: <bound method> = bound_method %.loc36_47.1, %impl.elem0.loc36_47
// CHECK:STDOUT:   %.loc36_47.2: ref %C = temporary_storage
// CHECK:STDOUT:   %Op.ref.loc36_47: %C.C.type.065f0c.2 = name_ref Op, imports.%C.C.decl.041267.2 [concrete = constants.%C.C.6522d8.2]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %.loc36_47.3: ref %C = value_as_ref %.loc36_47.1
// CHECK:STDOUT:   %addr.loc36_47.1: %ptr.d9e = addr_of %.loc36_47.3
// CHECK:STDOUT:   %.loc36_47.4: %ptr.c92 = as_compatible %addr.loc36_47.1
// CHECK:STDOUT:   %.loc36_47.5: %ptr.c92 = converted %addr.loc36_47.1, %.loc36_47.4
// CHECK:STDOUT:   %addr.loc36_47.2: %ptr.d9e = addr_of %rhs.var.2
// CHECK:STDOUT:   %C__carbon_thunk.call.loc36_47: init %empty_tuple.type = call imports.%C__carbon_thunk.decl.8acdfe.2(%.loc36_47.5, %addr.loc36_47.2)
// CHECK:STDOUT:   %.loc36_47.6: init %C to %rhs.var.2 = mark_in_place_init %C__carbon_thunk.call.loc36_47
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %addr.loc36_44.1: %ptr.d9e = addr_of %lhs.var.2
// CHECK:STDOUT:   %addr.loc36_44.2: %ptr.d9e = addr_of %rhs.var.2
// CHECK:STDOUT:   %.loc36_44.1: ref bool = temporary_storage
// CHECK:STDOUT:   %addr.loc36_44.3: %ptr.bb2 = addr_of %.loc36_44.1
// CHECK:STDOUT:   %operator_LessEqual__carbon_thunk.call: init %empty_tuple.type = call imports.%operator_LessEqual__carbon_thunk.decl(%addr.loc36_44.1, %addr.loc36_44.2, %addr.loc36_44.3)
// CHECK:STDOUT:   %.loc36_44.2: init bool to %.loc36_44.1 = mark_in_place_init %operator_LessEqual__carbon_thunk.call
// CHECK:STDOUT:   %.loc36_34: type = type_literal bool [concrete = bool]
// CHECK:STDOUT:   %.loc36_44.3: ref bool = temporary %.loc36_44.1, %.loc36_44.2
// CHECK:STDOUT:   %.loc36_44.4: bool = acquire_value %.loc36_44.3
// CHECK:STDOUT:   %less_than_or_equal: bool = value_binding less_than_or_equal, %.loc36_44.4
// CHECK:STDOUT:   %Destroy.Op.bound.loc36: <bound method> = bound_method %.loc36_44.3, constants.%Destroy.Op
// CHECK:STDOUT:   %Destroy.Op.call.loc36: init %empty_tuple.type = call %Destroy.Op.bound.loc36(%.loc36_44.3)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %Destroy.Op.bound.loc19: <bound method> = bound_method %.loc19_38.3, constants.%Destroy.Op
// CHECK:STDOUT:   %Destroy.Op.call.loc19: init %empty_tuple.type = call %Destroy.Op.bound.loc19(%.loc19_38.3)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %C.Op.bound.loc16: <bound method> = bound_method %c2.var, constants.%C.Op.068411.2
// CHECK:STDOUT:   %Op.ref.loc16: %C.cpp_destructor.type = name_ref Op, imports.%C.cpp_destructor.decl [concrete = constants.%C.cpp_destructor]
// CHECK:STDOUT:   %C.cpp_destructor.bound.loc16: <bound method> = bound_method %c2.var, %Op.ref.loc16
// CHECK:STDOUT:   %C.cpp_destructor.call.loc16: init %empty_tuple.type = call %C.cpp_destructor.bound.loc16(%c2.var)
// CHECK:STDOUT:   %C.Op.bound.loc15: <bound method> = bound_method %c1.var, constants.%C.Op.068411.2
// CHECK:STDOUT:   %Op.ref.loc15: %C.cpp_destructor.type = name_ref Op, imports.%C.cpp_destructor.decl [concrete = constants.%C.cpp_destructor]
// CHECK:STDOUT:   %C.cpp_destructor.bound.loc15: <bound method> = bound_method %c1.var, %Op.ref.loc15
// CHECK:STDOUT:   %C.cpp_destructor.call.loc15: init %empty_tuple.type = call %C.cpp_destructor.bound.loc15(%c1.var)
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
// CHECK:STDOUT:   %C__carbon_thunk.type.65f120.1: type = fn_type @C__carbon_thunk.1 [concrete]
// CHECK:STDOUT:   %C__carbon_thunk.d98342.1: %C__carbon_thunk.type.65f120.1 = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.831: type = pattern_type bool [concrete]
// CHECK:STDOUT:   %ptr.bb2: type = ptr_type bool [concrete]
// CHECK:STDOUT:   %operator_EqualEqual__carbon_thunk.type: type = fn_type @operator_EqualEqual__carbon_thunk [concrete]
// CHECK:STDOUT:   %operator_EqualEqual__carbon_thunk: %operator_EqualEqual__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %C.C.type.065f0c.2: type = fn_type @C.C.2 [concrete]
// CHECK:STDOUT:   %C.C.6522d8.2: %C.C.type.065f0c.2 = struct_value () [concrete]
// CHECK:STDOUT:   %const.057: type = const_type %C [concrete]
// CHECK:STDOUT:   %ptr.c92: type = ptr_type %const.057 [concrete]
// CHECK:STDOUT:   %C__carbon_thunk.type.65f120.2: type = fn_type @C__carbon_thunk.2 [concrete]
// CHECK:STDOUT:   %C__carbon_thunk.d98342.2: %C__carbon_thunk.type.65f120.2 = struct_value () [concrete]
// CHECK:STDOUT:   %C.Op.type.ea1bc9.1: type = fn_type @C.Op.1 [concrete]
// CHECK:STDOUT:   %C.Op.068411.1: %C.Op.type.ea1bc9.1 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.681: <witness> = custom_witness (%C.Op.068411.1), @Copy [concrete]
// CHECK:STDOUT:   %Copy.facet.3b9: %Copy.type = facet_value %C, (%custom_witness.681) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.530: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet.3b9) [concrete]
// CHECK:STDOUT:   %.a81: type = fn_type_with_self_type %Copy.WithSelf.Op.type.530, %Copy.facet.3b9 [concrete]
// CHECK:STDOUT:   %Destroy.Op.type: type = fn_type @Destroy.Op [concrete]
// CHECK:STDOUT:   %Destroy.Op: %Destroy.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %C.cpp_destructor.type: type = fn_type @C.cpp_destructor [concrete]
// CHECK:STDOUT:   %C.cpp_destructor: %C.cpp_destructor.type = struct_value () [concrete]
// CHECK:STDOUT:   %C.Op.type.ea1bc9.2: type = fn_type @C.Op.2 [concrete]
// CHECK:STDOUT:   %C.Op.068411.2: %C.Op.type.ea1bc9.2 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   %C.C.cpp_overload_set.value: %C.C.cpp_overload_set.type = cpp_overload_set_value @C.C.cpp_overload_set [concrete = constants.%C.C.cpp_overload_set.value]
// CHECK:STDOUT:   %C__carbon_thunk.decl.8acdfe.1: %C__carbon_thunk.type.65f120.1 = fn_decl @C__carbon_thunk.1 [concrete = constants.%C__carbon_thunk.d98342.1] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %operator_EqualEqual__carbon_thunk.decl: %operator_EqualEqual__carbon_thunk.type = fn_decl @operator_EqualEqual__carbon_thunk [concrete = constants.%operator_EqualEqual__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.C.decl.041267.2: %C.C.type.065f0c.2 = fn_decl @C.C.2 [concrete = constants.%C.C.6522d8.2] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C__carbon_thunk.decl.8acdfe.2: %C__carbon_thunk.type.65f120.2 = fn_decl @C__carbon_thunk.2 [concrete = constants.%C__carbon_thunk.d98342.2] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.cpp_destructor.decl: %C.cpp_destructor.type = fn_decl @C.cpp_destructor [concrete = constants.%C.cpp_destructor] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
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
// CHECK:STDOUT:   %C__carbon_thunk.call.loc12: init %empty_tuple.type = call imports.%C__carbon_thunk.decl.8acdfe.1(%addr.loc12)
// CHECK:STDOUT:   %.loc12_27: init %C to %.loc12_3 = mark_in_place_init %C__carbon_thunk.call.loc12
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
// CHECK:STDOUT:   %Cpp.ref.loc13_19: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %C.ref.loc13_22: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %C.ref.loc13_24: %C.C.cpp_overload_set.type = name_ref C, imports.%C.C.cpp_overload_set.value [concrete = constants.%C.C.cpp_overload_set.value]
// CHECK:STDOUT:   %.loc13_3: ref %C = splice_block %c2.var {}
// CHECK:STDOUT:   %addr.loc13: %ptr.d9e = addr_of %.loc13_3
// CHECK:STDOUT:   %C__carbon_thunk.call.loc13: init %empty_tuple.type = call imports.%C__carbon_thunk.decl.8acdfe.1(%addr.loc13)
// CHECK:STDOUT:   %.loc13_27: init %C to %.loc13_3 = mark_in_place_init %C__carbon_thunk.call.loc13
// CHECK:STDOUT:   assign %c2.var, %.loc13_27
// CHECK:STDOUT:   %.loc13_14: type = splice_block %C.ref.loc13_14 [concrete = constants.%C] {
// CHECK:STDOUT:     %Cpp.ref.loc13_11: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %C.ref.loc13_14: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c2: ref %C = ref_binding c2, %c2.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %equal.patt: %pattern_type.831 = value_binding_pattern equal [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c1.ref.loc16: ref %C = name_ref c1, %c1
// CHECK:STDOUT:   %c2.ref.loc16: ref %C = name_ref c2, %c2
// CHECK:STDOUT:   %.loc16_28.1: %C = acquire_value %c1.ref.loc16
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %impl.elem0.loc16_28: %.a81 = impl_witness_access constants.%custom_witness.681, element0 [concrete = constants.%C.Op.068411.1]
// CHECK:STDOUT:   %bound_method.loc16_28: <bound method> = bound_method %.loc16_28.1, %impl.elem0.loc16_28
// CHECK:STDOUT:   %.loc16_28.2: ref %C = temporary_storage
// CHECK:STDOUT:   %Op.ref.loc16_28: %C.C.type.065f0c.2 = name_ref Op, imports.%C.C.decl.041267.2 [concrete = constants.%C.C.6522d8.2]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %.loc16_28.3: ref %C = value_as_ref %.loc16_28.1
// CHECK:STDOUT:   %addr.loc16_28.1: %ptr.d9e = addr_of %.loc16_28.3
// CHECK:STDOUT:   %.loc16_28.4: %ptr.c92 = as_compatible %addr.loc16_28.1
// CHECK:STDOUT:   %.loc16_28.5: %ptr.c92 = converted %addr.loc16_28.1, %.loc16_28.4
// CHECK:STDOUT:   %addr.loc16_28.2: %ptr.d9e = addr_of %lhs.var
// CHECK:STDOUT:   %C__carbon_thunk.call.loc16_28: init %empty_tuple.type = call imports.%C__carbon_thunk.decl.8acdfe.2(%.loc16_28.5, %addr.loc16_28.2)
// CHECK:STDOUT:   %.loc16_28.6: init %C to %lhs.var = mark_in_place_init %C__carbon_thunk.call.loc16_28
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %.loc16_34.1: %C = acquire_value %c2.ref.loc16
// CHECK:STDOUT:   %impl.elem0.loc16_34: %.a81 = impl_witness_access constants.%custom_witness.681, element0 [concrete = constants.%C.Op.068411.1]
// CHECK:STDOUT:   %bound_method.loc16_34: <bound method> = bound_method %.loc16_34.1, %impl.elem0.loc16_34
// CHECK:STDOUT:   %.loc16_34.2: ref %C = temporary_storage
// CHECK:STDOUT:   %Op.ref.loc16_34: %C.C.type.065f0c.2 = name_ref Op, imports.%C.C.decl.041267.2 [concrete = constants.%C.C.6522d8.2]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %.loc16_34.3: ref %C = value_as_ref %.loc16_34.1
// CHECK:STDOUT:   %addr.loc16_34.1: %ptr.d9e = addr_of %.loc16_34.3
// CHECK:STDOUT:   %.loc16_34.4: %ptr.c92 = as_compatible %addr.loc16_34.1
// CHECK:STDOUT:   %.loc16_34.5: %ptr.c92 = converted %addr.loc16_34.1, %.loc16_34.4
// CHECK:STDOUT:   %addr.loc16_34.2: %ptr.d9e = addr_of %rhs.var
// CHECK:STDOUT:   %C__carbon_thunk.call.loc16_34: init %empty_tuple.type = call imports.%C__carbon_thunk.decl.8acdfe.2(%.loc16_34.5, %addr.loc16_34.2)
// CHECK:STDOUT:   %.loc16_34.6: init %C to %rhs.var = mark_in_place_init %C__carbon_thunk.call.loc16_34
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %addr.loc16_31.1: %ptr.d9e = addr_of %lhs.var
// CHECK:STDOUT:   %addr.loc16_31.2: %ptr.d9e = addr_of %rhs.var
// CHECK:STDOUT:   %.loc16_31.1: ref bool = temporary_storage
// CHECK:STDOUT:   %addr.loc16_31.3: %ptr.bb2 = addr_of %.loc16_31.1
// CHECK:STDOUT:   %operator_EqualEqual__carbon_thunk.call: init %empty_tuple.type = call imports.%operator_EqualEqual__carbon_thunk.decl(%addr.loc16_31.1, %addr.loc16_31.2, %addr.loc16_31.3)
// CHECK:STDOUT:   %.loc16_31.2: init bool to %.loc16_31.1 = mark_in_place_init %operator_EqualEqual__carbon_thunk.call
// CHECK:STDOUT:   %.loc16_21: type = type_literal bool [concrete = bool]
// CHECK:STDOUT:   %.loc16_31.3: ref bool = temporary %.loc16_31.1, %.loc16_31.2
// CHECK:STDOUT:   %.loc16_31.4: bool = acquire_value %.loc16_31.3
// CHECK:STDOUT:   %equal: bool = value_binding equal, %.loc16_31.4
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %not_equal.patt: %pattern_type.831 = value_binding_pattern not_equal [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c1.ref.loc23: ref %C = name_ref c1, %c1
// CHECK:STDOUT:   %c2.ref.loc23: ref %C = name_ref c2, %c2
// CHECK:STDOUT:   %.loc23: type = type_literal bool [concrete = bool]
// CHECK:STDOUT:   %not_equal: bool = value_binding not_equal, <error> [concrete = <error>]
// CHECK:STDOUT:   %Destroy.Op.bound: <bound method> = bound_method %.loc16_31.3, constants.%Destroy.Op
// CHECK:STDOUT:   %Destroy.Op.call: init %empty_tuple.type = call %Destroy.Op.bound(%.loc16_31.3)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %C.Op.bound.loc13: <bound method> = bound_method %c2.var, constants.%C.Op.068411.2
// CHECK:STDOUT:   %Op.ref.loc13: %C.cpp_destructor.type = name_ref Op, imports.%C.cpp_destructor.decl [concrete = constants.%C.cpp_destructor]
// CHECK:STDOUT:   %C.cpp_destructor.bound.loc13: <bound method> = bound_method %c2.var, %Op.ref.loc13
// CHECK:STDOUT:   %C.cpp_destructor.call.loc13: init %empty_tuple.type = call %C.cpp_destructor.bound.loc13(%c2.var)
// CHECK:STDOUT:   %C.Op.bound.loc12: <bound method> = bound_method %c1.var, constants.%C.Op.068411.2
// CHECK:STDOUT:   %Op.ref.loc12: %C.cpp_destructor.type = name_ref Op, imports.%C.cpp_destructor.decl [concrete = constants.%C.cpp_destructor]
// CHECK:STDOUT:   %C.cpp_destructor.bound.loc12: <bound method> = bound_method %c1.var, %Op.ref.loc12
// CHECK:STDOUT:   %C.cpp_destructor.call.loc12: init %empty_tuple.type = call %C.cpp_destructor.bound.loc12(%c1.var)
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
// CHECK:STDOUT:   %C__carbon_thunk.type.65f120.1: type = fn_type @C__carbon_thunk.1 [concrete]
// CHECK:STDOUT:   %C__carbon_thunk.d98342.1: %C__carbon_thunk.type.65f120.1 = struct_value () [concrete]
// CHECK:STDOUT:   %operator_Plus__carbon_thunk.type: type = fn_type @operator_Plus__carbon_thunk [concrete]
// CHECK:STDOUT:   %operator_Plus__carbon_thunk: %operator_Plus__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %C.C.type.4eb029.2: type = fn_type @C.C.2 [concrete]
// CHECK:STDOUT:   %C.C.d3a740.2: %C.C.type.4eb029.2 = struct_value () [concrete]
// CHECK:STDOUT:   %const.756: type = const_type %C [concrete]
// CHECK:STDOUT:   %ptr.7a2: type = ptr_type %const.756 [concrete]
// CHECK:STDOUT:   %C__carbon_thunk.type.65f120.2: type = fn_type @C__carbon_thunk.2 [concrete]
// CHECK:STDOUT:   %C__carbon_thunk.d98342.2: %C__carbon_thunk.type.65f120.2 = struct_value () [concrete]
// CHECK:STDOUT:   %C.Op.type.96359b.1: type = fn_type @C.Op.1 [concrete]
// CHECK:STDOUT:   %C.Op.bdd863.1: %C.Op.type.96359b.1 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.19f: <witness> = custom_witness (%C.Op.bdd863.1), @Copy [concrete]
// CHECK:STDOUT:   %Copy.facet.064: %Copy.type = facet_value %C, (%custom_witness.19f) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.e1e: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet.064) [concrete]
// CHECK:STDOUT:   %.591: type = fn_type_with_self_type %Copy.WithSelf.Op.type.e1e, %Copy.facet.064 [concrete]
// CHECK:STDOUT:   %C.cpp_destructor.type: type = fn_type @C.cpp_destructor [concrete]
// CHECK:STDOUT:   %C.cpp_destructor: %C.cpp_destructor.type = struct_value () [concrete]
// CHECK:STDOUT:   %C.Op.type.96359b.2: type = fn_type @C.Op.2 [concrete]
// CHECK:STDOUT:   %C.Op.bdd863.2: %C.Op.type.96359b.2 = struct_value () [concrete]
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
// CHECK:STDOUT:   %C__carbon_thunk.decl.8acdfe.1: %C__carbon_thunk.type.65f120.1 = fn_decl @C__carbon_thunk.1 [concrete = constants.%C__carbon_thunk.d98342.1] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %operator_Plus__carbon_thunk.decl: %operator_Plus__carbon_thunk.type = fn_decl @operator_Plus__carbon_thunk [concrete = constants.%operator_Plus__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.C.decl.5fd4df.2: %C.C.type.4eb029.2 = fn_decl @C.C.2 [concrete = constants.%C.C.d3a740.2] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C__carbon_thunk.decl.8acdfe.2: %C__carbon_thunk.type.65f120.2 = fn_decl @C__carbon_thunk.2 [concrete = constants.%C__carbon_thunk.d98342.2] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.cpp_destructor.decl: %C.cpp_destructor.type = fn_decl @C.cpp_destructor [concrete = constants.%C.cpp_destructor] {
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
// CHECK:STDOUT:   %Cpp.ref.loc8_21: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %N.ref.loc8_24: <namespace> = name_ref N, imports.%N [concrete = imports.%N]
// CHECK:STDOUT:   %C.ref.loc8_26: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %C.ref.loc8_28: %C.C.cpp_overload_set.type = name_ref C, imports.%C.C.cpp_overload_set.value [concrete = constants.%C.C.cpp_overload_set.value]
// CHECK:STDOUT:   %.loc8_31.1: ref %C = temporary_storage
// CHECK:STDOUT:   %addr.loc8: %ptr.838 = addr_of %.loc8_31.1
// CHECK:STDOUT:   %C__carbon_thunk.call.loc8: init %empty_tuple.type = call imports.%C__carbon_thunk.decl.8acdfe.1(%addr.loc8)
// CHECK:STDOUT:   %.loc8_31.2: init %C to %.loc8_31.1 = mark_in_place_init %C__carbon_thunk.call.loc8
// CHECK:STDOUT:   %.loc8_16: type = splice_block %C.ref.loc8_16 [concrete = constants.%C] {
// CHECK:STDOUT:     %Cpp.ref.loc8_11: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %N.ref.loc8_14: <namespace> = name_ref N, imports.%N [concrete = imports.%N]
// CHECK:STDOUT:     %C.ref.loc8_16: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc8_31.3: ref %C = temporary %.loc8_31.1, %.loc8_31.2
// CHECK:STDOUT:   %.loc8_31.4: %C = acquire_value %.loc8_31.3
// CHECK:STDOUT:   %c1: %C = value_binding c1, %.loc8_31.4
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c2.patt: %pattern_type.69f = value_binding_pattern c2 [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc9_21: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %N.ref.loc9_24: <namespace> = name_ref N, imports.%N [concrete = imports.%N]
// CHECK:STDOUT:   %C.ref.loc9_26: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %C.ref.loc9_28: %C.C.cpp_overload_set.type = name_ref C, imports.%C.C.cpp_overload_set.value [concrete = constants.%C.C.cpp_overload_set.value]
// CHECK:STDOUT:   %.loc9_31.1: ref %C = temporary_storage
// CHECK:STDOUT:   %addr.loc9: %ptr.838 = addr_of %.loc9_31.1
// CHECK:STDOUT:   %C__carbon_thunk.call.loc9: init %empty_tuple.type = call imports.%C__carbon_thunk.decl.8acdfe.1(%addr.loc9)
// CHECK:STDOUT:   %.loc9_31.2: init %C to %.loc9_31.1 = mark_in_place_init %C__carbon_thunk.call.loc9
// CHECK:STDOUT:   %.loc9_16: type = splice_block %C.ref.loc9_16 [concrete = constants.%C] {
// CHECK:STDOUT:     %Cpp.ref.loc9_11: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %N.ref.loc9_14: <namespace> = name_ref N, imports.%N [concrete = imports.%N]
// CHECK:STDOUT:     %C.ref.loc9_16: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc9_31.3: ref %C = temporary %.loc9_31.1, %.loc9_31.2
// CHECK:STDOUT:   %.loc9_31.4: %C = acquire_value %.loc9_31.3
// CHECK:STDOUT:   %c2: %C = value_binding c2, %.loc9_31.4
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c3.patt: %pattern_type.69f = value_binding_pattern c3 [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c1.ref: %C = name_ref c1, %c1
// CHECK:STDOUT:   %c2.ref: %C = name_ref c2, %c2
// CHECK:STDOUT:   %.loc10_31.1: ref %C = temporary_storage
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %impl.elem0.loc10_28: %.591 = impl_witness_access constants.%custom_witness.19f, element0 [concrete = constants.%C.Op.bdd863.1]
// CHECK:STDOUT:   %bound_method.loc10_28: <bound method> = bound_method %c1.ref, %impl.elem0.loc10_28
// CHECK:STDOUT:   %.loc10_28.1: ref %C = temporary_storage
// CHECK:STDOUT:   %Op.ref.loc10_28: %C.C.type.4eb029.2 = name_ref Op, imports.%C.C.decl.5fd4df.2 [concrete = constants.%C.C.d3a740.2]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %.loc10_28.2: ref %C = value_as_ref %c1.ref
// CHECK:STDOUT:   %addr.loc10_28.1: %ptr.838 = addr_of %.loc10_28.2
// CHECK:STDOUT:   %.loc10_28.3: %ptr.7a2 = as_compatible %addr.loc10_28.1
// CHECK:STDOUT:   %.loc10_28.4: %ptr.7a2 = converted %addr.loc10_28.1, %.loc10_28.3
// CHECK:STDOUT:   %addr.loc10_28.2: %ptr.838 = addr_of %lhs.var
// CHECK:STDOUT:   %C__carbon_thunk.call.loc10_28: init %empty_tuple.type = call imports.%C__carbon_thunk.decl.8acdfe.2(%.loc10_28.4, %addr.loc10_28.2)
// CHECK:STDOUT:   %.loc10_28.5: init %C to %lhs.var = mark_in_place_init %C__carbon_thunk.call.loc10_28
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %impl.elem0.loc10_33: %.591 = impl_witness_access constants.%custom_witness.19f, element0 [concrete = constants.%C.Op.bdd863.1]
// CHECK:STDOUT:   %bound_method.loc10_33: <bound method> = bound_method %c2.ref, %impl.elem0.loc10_33
// CHECK:STDOUT:   %.loc10_33.1: ref %C = temporary_storage
// CHECK:STDOUT:   %Op.ref.loc10_33: %C.C.type.4eb029.2 = name_ref Op, imports.%C.C.decl.5fd4df.2 [concrete = constants.%C.C.d3a740.2]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %.loc10_33.2: ref %C = value_as_ref %c2.ref
// CHECK:STDOUT:   %addr.loc10_33.1: %ptr.838 = addr_of %.loc10_33.2
// CHECK:STDOUT:   %.loc10_33.3: %ptr.7a2 = as_compatible %addr.loc10_33.1
// CHECK:STDOUT:   %.loc10_33.4: %ptr.7a2 = converted %addr.loc10_33.1, %.loc10_33.3
// CHECK:STDOUT:   %addr.loc10_33.2: %ptr.838 = addr_of %rhs.var
// CHECK:STDOUT:   %C__carbon_thunk.call.loc10_33: init %empty_tuple.type = call imports.%C__carbon_thunk.decl.8acdfe.2(%.loc10_33.4, %addr.loc10_33.2)
// CHECK:STDOUT:   %.loc10_33.5: init %C to %rhs.var = mark_in_place_init %C__carbon_thunk.call.loc10_33
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %addr.loc10_31.1: %ptr.838 = addr_of %lhs.var
// CHECK:STDOUT:   %addr.loc10_31.2: %ptr.838 = addr_of %rhs.var
// CHECK:STDOUT:   %addr.loc10_31.3: %ptr.838 = addr_of %.loc10_31.1
// CHECK:STDOUT:   %operator_Plus__carbon_thunk.call: init %empty_tuple.type = call imports.%operator_Plus__carbon_thunk.decl(%addr.loc10_31.1, %addr.loc10_31.2, %addr.loc10_31.3)
// CHECK:STDOUT:   %.loc10_31.2: init %C to %.loc10_31.1 = mark_in_place_init %operator_Plus__carbon_thunk.call
// CHECK:STDOUT:   %.loc10_23: type = splice_block %C.ref.loc10 [concrete = constants.%C] {
// CHECK:STDOUT:     %Cpp.ref.loc10: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %N.ref.loc10: <namespace> = name_ref N, imports.%N [concrete = imports.%N]
// CHECK:STDOUT:     %C.ref.loc10: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc10_31.3: ref %C = temporary %.loc10_31.1, %.loc10_31.2
// CHECK:STDOUT:   %.loc10_31.4: %C = acquire_value %.loc10_31.3
// CHECK:STDOUT:   %c3: %C = value_binding c3, %.loc10_31.4
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %C.Op.bound.loc10: <bound method> = bound_method %.loc10_31.3, constants.%C.Op.bdd863.2
// CHECK:STDOUT:   %Op.ref.loc10_31: %C.cpp_destructor.type = name_ref Op, imports.%C.cpp_destructor.decl [concrete = constants.%C.cpp_destructor]
// CHECK:STDOUT:   %C.cpp_destructor.bound.loc10: <bound method> = bound_method %.loc10_31.3, %Op.ref.loc10_31
// CHECK:STDOUT:   %C.cpp_destructor.call.loc10: init %empty_tuple.type = call %C.cpp_destructor.bound.loc10(%.loc10_31.3)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %C.Op.bound.loc9: <bound method> = bound_method %.loc9_31.3, constants.%C.Op.bdd863.2
// CHECK:STDOUT:   %Op.ref.loc9: %C.cpp_destructor.type = name_ref Op, imports.%C.cpp_destructor.decl [concrete = constants.%C.cpp_destructor]
// CHECK:STDOUT:   %C.cpp_destructor.bound.loc9: <bound method> = bound_method %.loc9_31.3, %Op.ref.loc9
// CHECK:STDOUT:   %C.cpp_destructor.call.loc9: init %empty_tuple.type = call %C.cpp_destructor.bound.loc9(%.loc9_31.3)
// CHECK:STDOUT:   %C.Op.bound.loc8: <bound method> = bound_method %.loc8_31.3, constants.%C.Op.bdd863.2
// CHECK:STDOUT:   %Op.ref.loc8: %C.cpp_destructor.type = name_ref Op, imports.%C.cpp_destructor.decl [concrete = constants.%C.cpp_destructor]
// CHECK:STDOUT:   %C.cpp_destructor.bound.loc8: <bound method> = bound_method %.loc8_31.3, %Op.ref.loc8
// CHECK:STDOUT:   %C.cpp_destructor.call.loc8: init %empty_tuple.type = call %C.cpp_destructor.bound.loc8(%.loc8_31.3)
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
// CHECK:STDOUT:   %C1__carbon_thunk.type.7cf9ba.1: type = fn_type @C1__carbon_thunk.1 [concrete]
// CHECK:STDOUT:   %C1__carbon_thunk.d154f5.1: %C1__carbon_thunk.type.7cf9ba.1 = struct_value () [concrete]
// CHECK:STDOUT:   %C2: type = class_type @C2 [concrete]
// CHECK:STDOUT:   %pattern_type.846: type = pattern_type %C2 [concrete]
// CHECK:STDOUT:   %C2.C2.cpp_overload_set.type: type = cpp_overload_set_type @C2.C2.cpp_overload_set [concrete]
// CHECK:STDOUT:   %C2.C2.cpp_overload_set.value: %C2.C2.cpp_overload_set.type = cpp_overload_set_value @C2.C2.cpp_overload_set [concrete]
// CHECK:STDOUT:   %ptr.51f: type = ptr_type %C2 [concrete]
// CHECK:STDOUT:   %C2__carbon_thunk.type.38b4d6.1: type = fn_type @C2__carbon_thunk.1 [concrete]
// CHECK:STDOUT:   %C2__carbon_thunk.097a09.1: %C2__carbon_thunk.type.38b4d6.1 = struct_value () [concrete]
// CHECK:STDOUT:   %operator_Plus__carbon_thunk.type: type = fn_type @operator_Plus__carbon_thunk [concrete]
// CHECK:STDOUT:   %operator_Plus__carbon_thunk: %operator_Plus__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %C1.C1.type.d6cd8a.2: type = fn_type @C1.C1.2 [concrete]
// CHECK:STDOUT:   %C1.C1.85437f.2: %C1.C1.type.d6cd8a.2 = struct_value () [concrete]
// CHECK:STDOUT:   %const.4b2: type = const_type %C1 [concrete]
// CHECK:STDOUT:   %ptr.7bf: type = ptr_type %const.4b2 [concrete]
// CHECK:STDOUT:   %C1__carbon_thunk.type.7cf9ba.2: type = fn_type @C1__carbon_thunk.2 [concrete]
// CHECK:STDOUT:   %C1__carbon_thunk.d154f5.2: %C1__carbon_thunk.type.7cf9ba.2 = struct_value () [concrete]
// CHECK:STDOUT:   %C1.Op.type.008c97.1: type = fn_type @C1.Op.1 [concrete]
// CHECK:STDOUT:   %C1.Op.cf510a.1: %C1.Op.type.008c97.1 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.3e9: <witness> = custom_witness (%C1.Op.cf510a.1), @Copy [concrete]
// CHECK:STDOUT:   %Copy.facet.0dc: %Copy.type = facet_value %C1, (%custom_witness.3e9) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.188: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet.0dc) [concrete]
// CHECK:STDOUT:   %.bd4: type = fn_type_with_self_type %Copy.WithSelf.Op.type.188, %Copy.facet.0dc [concrete]
// CHECK:STDOUT:   %C2.C2.type.e0898a.2: type = fn_type @C2.C2.2 [concrete]
// CHECK:STDOUT:   %C2.C2.e9be96.2: %C2.C2.type.e0898a.2 = struct_value () [concrete]
// CHECK:STDOUT:   %const.b95: type = const_type %C2 [concrete]
// CHECK:STDOUT:   %ptr.388: type = ptr_type %const.b95 [concrete]
// CHECK:STDOUT:   %C2__carbon_thunk.type.38b4d6.2: type = fn_type @C2__carbon_thunk.2 [concrete]
// CHECK:STDOUT:   %C2__carbon_thunk.097a09.2: %C2__carbon_thunk.type.38b4d6.2 = struct_value () [concrete]
// CHECK:STDOUT:   %C2.Op.type.a06894.1: type = fn_type @C2.Op.1 [concrete]
// CHECK:STDOUT:   %C2.Op.65ae22.1: %C2.Op.type.a06894.1 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.4cf: <witness> = custom_witness (%C2.Op.65ae22.1), @Copy [concrete]
// CHECK:STDOUT:   %Copy.facet.a7c: %Copy.type = facet_value %C2, (%custom_witness.4cf) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.bd6: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet.a7c) [concrete]
// CHECK:STDOUT:   %.f89: type = fn_type_with_self_type %Copy.WithSelf.Op.type.bd6, %Copy.facet.a7c [concrete]
// CHECK:STDOUT:   %operator_Minus__carbon_thunk.type: type = fn_type @operator_Minus__carbon_thunk [concrete]
// CHECK:STDOUT:   %operator_Minus__carbon_thunk: %operator_Minus__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %C2.cpp_destructor.type: type = fn_type @C2.cpp_destructor [concrete]
// CHECK:STDOUT:   %C2.cpp_destructor: %C2.cpp_destructor.type = struct_value () [concrete]
// CHECK:STDOUT:   %C2.Op.type.a06894.2: type = fn_type @C2.Op.2 [concrete]
// CHECK:STDOUT:   %C2.Op.65ae22.2: %C2.Op.type.a06894.2 = struct_value () [concrete]
// CHECK:STDOUT:   %C1.cpp_destructor.type: type = fn_type @C1.cpp_destructor [concrete]
// CHECK:STDOUT:   %C1.cpp_destructor: %C1.cpp_destructor.type = struct_value () [concrete]
// CHECK:STDOUT:   %C1.Op.type.008c97.2: type = fn_type @C1.Op.2 [concrete]
// CHECK:STDOUT:   %C1.Op.cf510a.2: %C1.Op.type.008c97.2 = struct_value () [concrete]
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
// CHECK:STDOUT:   %C1__carbon_thunk.decl.daa3a2.1: %C1__carbon_thunk.type.7cf9ba.1 = fn_decl @C1__carbon_thunk.1 [concrete = constants.%C1__carbon_thunk.d154f5.1] {
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
// CHECK:STDOUT:   %C2__carbon_thunk.decl.ae9495.1: %C2__carbon_thunk.type.38b4d6.1 = fn_decl @C2__carbon_thunk.1 [concrete = constants.%C2__carbon_thunk.097a09.1] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %operator_Plus__carbon_thunk.decl: %operator_Plus__carbon_thunk.type = fn_decl @operator_Plus__carbon_thunk [concrete = constants.%operator_Plus__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C1.C1.decl.516672.2: %C1.C1.type.d6cd8a.2 = fn_decl @C1.C1.2 [concrete = constants.%C1.C1.85437f.2] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C1__carbon_thunk.decl.daa3a2.2: %C1__carbon_thunk.type.7cf9ba.2 = fn_decl @C1__carbon_thunk.2 [concrete = constants.%C1__carbon_thunk.d154f5.2] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C2.C2.decl.250fcf.2: %C2.C2.type.e0898a.2 = fn_decl @C2.C2.2 [concrete = constants.%C2.C2.e9be96.2] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C2__carbon_thunk.decl.ae9495.2: %C2__carbon_thunk.type.38b4d6.2 = fn_decl @C2__carbon_thunk.2 [concrete = constants.%C2__carbon_thunk.097a09.2] {
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
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C1.cpp_destructor.decl: %C1.cpp_destructor.type = fn_decl @C1.cpp_destructor [concrete = constants.%C1.cpp_destructor] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c1.patt: %pattern_type.20f = value_binding_pattern c1 [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc8_23: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %N1.ref.loc8_26: <namespace> = name_ref N1, imports.%N1 [concrete = imports.%N1]
// CHECK:STDOUT:   %C1.ref.loc8_29: type = name_ref C1, imports.%C1.decl [concrete = constants.%C1]
// CHECK:STDOUT:   %C1.ref.loc8_32: %C1.C1.cpp_overload_set.type = name_ref C1, imports.%C1.C1.cpp_overload_set.value [concrete = constants.%C1.C1.cpp_overload_set.value]
// CHECK:STDOUT:   %.loc8_36.1: ref %C1 = temporary_storage
// CHECK:STDOUT:   %addr.loc8: %ptr.087 = addr_of %.loc8_36.1
// CHECK:STDOUT:   %C1__carbon_thunk.call.loc8: init %empty_tuple.type = call imports.%C1__carbon_thunk.decl.daa3a2.1(%addr.loc8)
// CHECK:STDOUT:   %.loc8_36.2: init %C1 to %.loc8_36.1 = mark_in_place_init %C1__carbon_thunk.call.loc8
// CHECK:STDOUT:   %.loc8_17: type = splice_block %C1.ref.loc8_17 [concrete = constants.%C1] {
// CHECK:STDOUT:     %Cpp.ref.loc8_11: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %N1.ref.loc8_14: <namespace> = name_ref N1, imports.%N1 [concrete = imports.%N1]
// CHECK:STDOUT:     %C1.ref.loc8_17: type = name_ref C1, imports.%C1.decl [concrete = constants.%C1]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc8_36.3: ref %C1 = temporary %.loc8_36.1, %.loc8_36.2
// CHECK:STDOUT:   %.loc8_36.4: %C1 = acquire_value %.loc8_36.3
// CHECK:STDOUT:   %c1: %C1 = value_binding c1, %.loc8_36.4
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c2.patt: %pattern_type.846 = value_binding_pattern c2 [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc9_23: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %N2.ref.loc9_26: <namespace> = name_ref N2, imports.%N2 [concrete = imports.%N2]
// CHECK:STDOUT:   %C2.ref.loc9_29: type = name_ref C2, imports.%C2.decl [concrete = constants.%C2]
// CHECK:STDOUT:   %C2.ref.loc9_32: %C2.C2.cpp_overload_set.type = name_ref C2, imports.%C2.C2.cpp_overload_set.value [concrete = constants.%C2.C2.cpp_overload_set.value]
// CHECK:STDOUT:   %.loc9_36.1: ref %C2 = temporary_storage
// CHECK:STDOUT:   %addr.loc9: %ptr.51f = addr_of %.loc9_36.1
// CHECK:STDOUT:   %C2__carbon_thunk.call.loc9: init %empty_tuple.type = call imports.%C2__carbon_thunk.decl.ae9495.1(%addr.loc9)
// CHECK:STDOUT:   %.loc9_36.2: init %C2 to %.loc9_36.1 = mark_in_place_init %C2__carbon_thunk.call.loc9
// CHECK:STDOUT:   %.loc9_17: type = splice_block %C2.ref.loc9_17 [concrete = constants.%C2] {
// CHECK:STDOUT:     %Cpp.ref.loc9_11: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %N2.ref.loc9_14: <namespace> = name_ref N2, imports.%N2 [concrete = imports.%N2]
// CHECK:STDOUT:     %C2.ref.loc9_17: type = name_ref C2, imports.%C2.decl [concrete = constants.%C2]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc9_36.3: ref %C2 = temporary %.loc9_36.1, %.loc9_36.2
// CHECK:STDOUT:   %.loc9_36.4: %C2 = acquire_value %.loc9_36.3
// CHECK:STDOUT:   %c2: %C2 = value_binding c2, %.loc9_36.4
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c3.patt: %pattern_type.846 = value_binding_pattern c3 [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c1.ref.loc10: %C1 = name_ref c1, %c1
// CHECK:STDOUT:   %c2.ref.loc10: %C2 = name_ref c2, %c2
// CHECK:STDOUT:   %.loc10_33.1: ref %C2 = temporary_storage
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %impl.elem0.loc10_30: %.bd4 = impl_witness_access constants.%custom_witness.3e9, element0 [concrete = constants.%C1.Op.cf510a.1]
// CHECK:STDOUT:   %bound_method.loc10_30: <bound method> = bound_method %c1.ref.loc10, %impl.elem0.loc10_30
// CHECK:STDOUT:   %.loc10_30.1: ref %C1 = temporary_storage
// CHECK:STDOUT:   %Op.ref.loc10_30: %C1.C1.type.d6cd8a.2 = name_ref Op, imports.%C1.C1.decl.516672.2 [concrete = constants.%C1.C1.85437f.2]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %.loc10_30.2: ref %C1 = value_as_ref %c1.ref.loc10
// CHECK:STDOUT:   %addr.loc10_30.1: %ptr.087 = addr_of %.loc10_30.2
// CHECK:STDOUT:   %.loc10_30.3: %ptr.7bf = as_compatible %addr.loc10_30.1
// CHECK:STDOUT:   %.loc10_30.4: %ptr.7bf = converted %addr.loc10_30.1, %.loc10_30.3
// CHECK:STDOUT:   %addr.loc10_30.2: %ptr.087 = addr_of %lhs.var.1
// CHECK:STDOUT:   %C1__carbon_thunk.call.loc10: init %empty_tuple.type = call imports.%C1__carbon_thunk.decl.daa3a2.2(%.loc10_30.4, %addr.loc10_30.2)
// CHECK:STDOUT:   %.loc10_30.5: init %C1 to %lhs.var.1 = mark_in_place_init %C1__carbon_thunk.call.loc10
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %impl.elem0.loc10_35: %.f89 = impl_witness_access constants.%custom_witness.4cf, element0 [concrete = constants.%C2.Op.65ae22.1]
// CHECK:STDOUT:   %bound_method.loc10_35: <bound method> = bound_method %c2.ref.loc10, %impl.elem0.loc10_35
// CHECK:STDOUT:   %.loc10_35.1: ref %C2 = temporary_storage
// CHECK:STDOUT:   %Op.ref.loc10_35: %C2.C2.type.e0898a.2 = name_ref Op, imports.%C2.C2.decl.250fcf.2 [concrete = constants.%C2.C2.e9be96.2]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %.loc10_35.2: ref %C2 = value_as_ref %c2.ref.loc10
// CHECK:STDOUT:   %addr.loc10_35.1: %ptr.51f = addr_of %.loc10_35.2
// CHECK:STDOUT:   %.loc10_35.3: %ptr.388 = as_compatible %addr.loc10_35.1
// CHECK:STDOUT:   %.loc10_35.4: %ptr.388 = converted %addr.loc10_35.1, %.loc10_35.3
// CHECK:STDOUT:   %addr.loc10_35.2: %ptr.51f = addr_of %rhs.var.1
// CHECK:STDOUT:   %C2__carbon_thunk.call.loc10: init %empty_tuple.type = call imports.%C2__carbon_thunk.decl.ae9495.2(%.loc10_35.4, %addr.loc10_35.2)
// CHECK:STDOUT:   %.loc10_35.5: init %C2 to %rhs.var.1 = mark_in_place_init %C2__carbon_thunk.call.loc10
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %addr.loc10_33.1: %ptr.087 = addr_of %lhs.var.1
// CHECK:STDOUT:   %addr.loc10_33.2: %ptr.51f = addr_of %rhs.var.1
// CHECK:STDOUT:   %addr.loc10_33.3: %ptr.51f = addr_of %.loc10_33.1
// CHECK:STDOUT:   %operator_Plus__carbon_thunk.call: init %empty_tuple.type = call imports.%operator_Plus__carbon_thunk.decl(%addr.loc10_33.1, %addr.loc10_33.2, %addr.loc10_33.3)
// CHECK:STDOUT:   %.loc10_33.2: init %C2 to %.loc10_33.1 = mark_in_place_init %operator_Plus__carbon_thunk.call
// CHECK:STDOUT:   %.loc10_24: type = splice_block %C2.ref.loc10 [concrete = constants.%C2] {
// CHECK:STDOUT:     %Cpp.ref.loc10: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %N2.ref.loc10: <namespace> = name_ref N2, imports.%N2 [concrete = imports.%N2]
// CHECK:STDOUT:     %C2.ref.loc10: type = name_ref C2, imports.%C2.decl [concrete = constants.%C2]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc10_33.3: ref %C2 = temporary %.loc10_33.1, %.loc10_33.2
// CHECK:STDOUT:   %.loc10_33.4: %C2 = acquire_value %.loc10_33.3
// CHECK:STDOUT:   %c3: %C2 = value_binding c3, %.loc10_33.4
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c4.patt: %pattern_type.846 = value_binding_pattern c4 [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c2.ref.loc11: %C2 = name_ref c2, %c2
// CHECK:STDOUT:   %c1.ref.loc11: %C1 = name_ref c1, %c1
// CHECK:STDOUT:   %.loc11_33.1: ref %C2 = temporary_storage
// CHECK:STDOUT:   %impl.elem0.loc11_30: %.f89 = impl_witness_access constants.%custom_witness.4cf, element0 [concrete = constants.%C2.Op.65ae22.1]
// CHECK:STDOUT:   %bound_method.loc11_30: <bound method> = bound_method %c2.ref.loc11, %impl.elem0.loc11_30
// CHECK:STDOUT:   %.loc11_30.1: ref %C2 = temporary_storage
// CHECK:STDOUT:   %Op.ref.loc11_30: %C2.C2.type.e0898a.2 = name_ref Op, imports.%C2.C2.decl.250fcf.2 [concrete = constants.%C2.C2.e9be96.2]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %.loc11_30.2: ref %C2 = value_as_ref %c2.ref.loc11
// CHECK:STDOUT:   %addr.loc11_30.1: %ptr.51f = addr_of %.loc11_30.2
// CHECK:STDOUT:   %.loc11_30.3: %ptr.388 = as_compatible %addr.loc11_30.1
// CHECK:STDOUT:   %.loc11_30.4: %ptr.388 = converted %addr.loc11_30.1, %.loc11_30.3
// CHECK:STDOUT:   %addr.loc11_30.2: %ptr.51f = addr_of %lhs.var.2
// CHECK:STDOUT:   %C2__carbon_thunk.call.loc11: init %empty_tuple.type = call imports.%C2__carbon_thunk.decl.ae9495.2(%.loc11_30.4, %addr.loc11_30.2)
// CHECK:STDOUT:   %.loc11_30.5: init %C2 to %lhs.var.2 = mark_in_place_init %C2__carbon_thunk.call.loc11
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %impl.elem0.loc11_35: %.bd4 = impl_witness_access constants.%custom_witness.3e9, element0 [concrete = constants.%C1.Op.cf510a.1]
// CHECK:STDOUT:   %bound_method.loc11_35: <bound method> = bound_method %c1.ref.loc11, %impl.elem0.loc11_35
// CHECK:STDOUT:   %.loc11_35.1: ref %C1 = temporary_storage
// CHECK:STDOUT:   %Op.ref.loc11_35: %C1.C1.type.d6cd8a.2 = name_ref Op, imports.%C1.C1.decl.516672.2 [concrete = constants.%C1.C1.85437f.2]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %.loc11_35.2: ref %C1 = value_as_ref %c1.ref.loc11
// CHECK:STDOUT:   %addr.loc11_35.1: %ptr.087 = addr_of %.loc11_35.2
// CHECK:STDOUT:   %.loc11_35.3: %ptr.7bf = as_compatible %addr.loc11_35.1
// CHECK:STDOUT:   %.loc11_35.4: %ptr.7bf = converted %addr.loc11_35.1, %.loc11_35.3
// CHECK:STDOUT:   %addr.loc11_35.2: %ptr.087 = addr_of %rhs.var.2
// CHECK:STDOUT:   %C1__carbon_thunk.call.loc11: init %empty_tuple.type = call imports.%C1__carbon_thunk.decl.daa3a2.2(%.loc11_35.4, %addr.loc11_35.2)
// CHECK:STDOUT:   %.loc11_35.5: init %C1 to %rhs.var.2 = mark_in_place_init %C1__carbon_thunk.call.loc11
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %addr.loc11_33.1: %ptr.51f = addr_of %lhs.var.2
// CHECK:STDOUT:   %addr.loc11_33.2: %ptr.087 = addr_of %rhs.var.2
// CHECK:STDOUT:   %addr.loc11_33.3: %ptr.51f = addr_of %.loc11_33.1
// CHECK:STDOUT:   %operator_Minus__carbon_thunk.call: init %empty_tuple.type = call imports.%operator_Minus__carbon_thunk.decl(%addr.loc11_33.1, %addr.loc11_33.2, %addr.loc11_33.3)
// CHECK:STDOUT:   %.loc11_33.2: init %C2 to %.loc11_33.1 = mark_in_place_init %operator_Minus__carbon_thunk.call
// CHECK:STDOUT:   %.loc11_24: type = splice_block %C2.ref.loc11 [concrete = constants.%C2] {
// CHECK:STDOUT:     %Cpp.ref.loc11: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %N2.ref.loc11: <namespace> = name_ref N2, imports.%N2 [concrete = imports.%N2]
// CHECK:STDOUT:     %C2.ref.loc11: type = name_ref C2, imports.%C2.decl [concrete = constants.%C2]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc11_33.3: ref %C2 = temporary %.loc11_33.1, %.loc11_33.2
// CHECK:STDOUT:   %.loc11_33.4: %C2 = acquire_value %.loc11_33.3
// CHECK:STDOUT:   %c4: %C2 = value_binding c4, %.loc11_33.4
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %C2.Op.bound.loc11: <bound method> = bound_method %.loc11_33.3, constants.%C2.Op.65ae22.2
// CHECK:STDOUT:   %Op.ref.loc11_33: %C2.cpp_destructor.type = name_ref Op, imports.%C2.cpp_destructor.decl [concrete = constants.%C2.cpp_destructor]
// CHECK:STDOUT:   %C2.cpp_destructor.bound.loc11: <bound method> = bound_method %.loc11_33.3, %Op.ref.loc11_33
// CHECK:STDOUT:   %C2.cpp_destructor.call.loc11: init %empty_tuple.type = call %C2.cpp_destructor.bound.loc11(%.loc11_33.3)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %C2.Op.bound.loc10: <bound method> = bound_method %.loc10_33.3, constants.%C2.Op.65ae22.2
// CHECK:STDOUT:   %Op.ref.loc10_33: %C2.cpp_destructor.type = name_ref Op, imports.%C2.cpp_destructor.decl [concrete = constants.%C2.cpp_destructor]
// CHECK:STDOUT:   %C2.cpp_destructor.bound.loc10: <bound method> = bound_method %.loc10_33.3, %Op.ref.loc10_33
// CHECK:STDOUT:   %C2.cpp_destructor.call.loc10: init %empty_tuple.type = call %C2.cpp_destructor.bound.loc10(%.loc10_33.3)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %C2.Op.bound.loc9: <bound method> = bound_method %.loc9_36.3, constants.%C2.Op.65ae22.2
// CHECK:STDOUT:   %Op.ref.loc9: %C2.cpp_destructor.type = name_ref Op, imports.%C2.cpp_destructor.decl [concrete = constants.%C2.cpp_destructor]
// CHECK:STDOUT:   %C2.cpp_destructor.bound.loc9: <bound method> = bound_method %.loc9_36.3, %Op.ref.loc9
// CHECK:STDOUT:   %C2.cpp_destructor.call.loc9: init %empty_tuple.type = call %C2.cpp_destructor.bound.loc9(%.loc9_36.3)
// CHECK:STDOUT:   %C1.Op.bound: <bound method> = bound_method %.loc8_36.3, constants.%C1.Op.cf510a.2
// CHECK:STDOUT:   %Op.ref.loc8: %C1.cpp_destructor.type = name_ref Op, imports.%C1.cpp_destructor.decl [concrete = constants.%C1.cpp_destructor]
// CHECK:STDOUT:   %C1.cpp_destructor.bound.loc8: <bound method> = bound_method %.loc8_36.3, %Op.ref.loc8
// CHECK:STDOUT:   %C1.cpp_destructor.call.loc8: init %empty_tuple.type = call %C1.cpp_destructor.bound.loc8(%.loc8_36.3)
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
// CHECK:STDOUT:   %C.cpp_destructor.decl: %C.cpp_destructor.type = fn_decl @C.cpp_destructor [concrete = constants.%C.cpp_destructor] {
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
// CHECK:STDOUT:   %Cpp.ref.loc8_21: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %N.ref.loc8_24: <namespace> = name_ref N, imports.%N [concrete = imports.%N]
// CHECK:STDOUT:   %C.ref.loc8_26: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %C.ref.loc8_28: %C.C.cpp_overload_set.type = name_ref C, imports.%C.C.cpp_overload_set.value [concrete = constants.%C.C.cpp_overload_set.value]
// CHECK:STDOUT:   %.loc8_31.1: ref %C = temporary_storage
// CHECK:STDOUT:   %addr.loc8: %ptr.838 = addr_of %.loc8_31.1
// CHECK:STDOUT:   %C__carbon_thunk.call.loc8: init %empty_tuple.type = call imports.%C__carbon_thunk.decl(%addr.loc8)
// CHECK:STDOUT:   %.loc8_31.2: init %C to %.loc8_31.1 = mark_in_place_init %C__carbon_thunk.call.loc8
// CHECK:STDOUT:   %.loc8_16: type = splice_block %C.ref.loc8_16 [concrete = constants.%C] {
// CHECK:STDOUT:     %Cpp.ref.loc8_11: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %N.ref.loc8_14: <namespace> = name_ref N, imports.%N [concrete = imports.%N]
// CHECK:STDOUT:     %C.ref.loc8_16: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc8_31.3: ref %C = temporary %.loc8_31.1, %.loc8_31.2
// CHECK:STDOUT:   %.loc8_31.4: %C = acquire_value %.loc8_31.3
// CHECK:STDOUT:   %c1: %C = value_binding c1, %.loc8_31.4
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c2.patt: %pattern_type.69f = value_binding_pattern c2 [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc9_21: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %N.ref.loc9_24: <namespace> = name_ref N, imports.%N [concrete = imports.%N]
// CHECK:STDOUT:   %C.ref.loc9_26: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %C.ref.loc9_28: %C.C.cpp_overload_set.type = name_ref C, imports.%C.C.cpp_overload_set.value [concrete = constants.%C.C.cpp_overload_set.value]
// CHECK:STDOUT:   %.loc9_31.1: ref %C = temporary_storage
// CHECK:STDOUT:   %addr.loc9: %ptr.838 = addr_of %.loc9_31.1
// CHECK:STDOUT:   %C__carbon_thunk.call.loc9: init %empty_tuple.type = call imports.%C__carbon_thunk.decl(%addr.loc9)
// CHECK:STDOUT:   %.loc9_31.2: init %C to %.loc9_31.1 = mark_in_place_init %C__carbon_thunk.call.loc9
// CHECK:STDOUT:   %.loc9_16: type = splice_block %C.ref.loc9_16 [concrete = constants.%C] {
// CHECK:STDOUT:     %Cpp.ref.loc9_11: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %N.ref.loc9_14: <namespace> = name_ref N, imports.%N [concrete = imports.%N]
// CHECK:STDOUT:     %C.ref.loc9_16: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc9_31.3: ref %C = temporary %.loc9_31.1, %.loc9_31.2
// CHECK:STDOUT:   %.loc9_31.4: %C = acquire_value %.loc9_31.3
// CHECK:STDOUT:   %c2: %C = value_binding c2, %.loc9_31.4
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c3.patt: %pattern_type.69f = value_binding_pattern c3 [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c1.ref: %C = name_ref c1, %c1
// CHECK:STDOUT:   %c2.ref: %C = name_ref c2, %c2
// CHECK:STDOUT:   %.loc14: type = splice_block %C.ref.loc14 [concrete = constants.%C] {
// CHECK:STDOUT:     %Cpp.ref.loc14: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %N.ref.loc14: <namespace> = name_ref N, imports.%N [concrete = imports.%N]
// CHECK:STDOUT:     %C.ref.loc14: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c3: %C = value_binding c3, <error> [concrete = <error>]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %C.Op.bound.loc9: <bound method> = bound_method %.loc9_31.3, constants.%C.Op
// CHECK:STDOUT:   %Op.ref.loc9: %C.cpp_destructor.type = name_ref Op, imports.%C.cpp_destructor.decl [concrete = constants.%C.cpp_destructor]
// CHECK:STDOUT:   %C.cpp_destructor.bound.loc9: <bound method> = bound_method %.loc9_31.3, %Op.ref.loc9
// CHECK:STDOUT:   %C.cpp_destructor.call.loc9: init %empty_tuple.type = call %C.cpp_destructor.bound.loc9(%.loc9_31.3)
// CHECK:STDOUT:   %C.Op.bound.loc8: <bound method> = bound_method %.loc8_31.3, constants.%C.Op
// CHECK:STDOUT:   %Op.ref.loc8: %C.cpp_destructor.type = name_ref Op, imports.%C.cpp_destructor.decl [concrete = constants.%C.cpp_destructor]
// CHECK:STDOUT:   %C.cpp_destructor.bound.loc8: <bound method> = bound_method %.loc8_31.3, %Op.ref.loc8
// CHECK:STDOUT:   %C.cpp_destructor.call.loc8: init %empty_tuple.type = call %C.cpp_destructor.bound.loc8(%.loc8_31.3)
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
// CHECK:STDOUT:   %C__carbon_thunk.type.65f120.1: type = fn_type @C__carbon_thunk.1 [concrete]
// CHECK:STDOUT:   %C__carbon_thunk.d98342.1: %C__carbon_thunk.type.65f120.1 = struct_value () [concrete]
// CHECK:STDOUT:   %operator_Plus__carbon_thunk.type: type = fn_type @operator_Plus__carbon_thunk [concrete]
// CHECK:STDOUT:   %operator_Plus__carbon_thunk: %operator_Plus__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %C.C.type.8a7395.2: type = fn_type @C.C.2 [concrete]
// CHECK:STDOUT:   %C.C.85bfb1.2: %C.C.type.8a7395.2 = struct_value () [concrete]
// CHECK:STDOUT:   %const.7e1: type = const_type %C [concrete]
// CHECK:STDOUT:   %ptr.f09: type = ptr_type %const.7e1 [concrete]
// CHECK:STDOUT:   %C__carbon_thunk.type.65f120.2: type = fn_type @C__carbon_thunk.2 [concrete]
// CHECK:STDOUT:   %C__carbon_thunk.d98342.2: %C__carbon_thunk.type.65f120.2 = struct_value () [concrete]
// CHECK:STDOUT:   %C.Op.type.2f8cac.1: type = fn_type @C.Op.1 [concrete]
// CHECK:STDOUT:   %C.Op.30d0ef.1: %C.Op.type.2f8cac.1 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.8ae: <witness> = custom_witness (%C.Op.30d0ef.1), @Copy [concrete]
// CHECK:STDOUT:   %Copy.facet.bd6: %Copy.type = facet_value %C, (%custom_witness.8ae) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.3cc: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet.bd6) [concrete]
// CHECK:STDOUT:   %.724: type = fn_type_with_self_type %Copy.WithSelf.Op.type.3cc, %Copy.facet.bd6 [concrete]
// CHECK:STDOUT:   %C.cpp_destructor.type: type = fn_type @C.cpp_destructor [concrete]
// CHECK:STDOUT:   %C.cpp_destructor: %C.cpp_destructor.type = struct_value () [concrete]
// CHECK:STDOUT:   %C.Op.type.2f8cac.2: type = fn_type @C.Op.2 [concrete]
// CHECK:STDOUT:   %C.Op.30d0ef.2: %C.Op.type.2f8cac.2 = struct_value () [concrete]
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
// CHECK:STDOUT:   %C__carbon_thunk.decl.8acdfe.1: %C__carbon_thunk.type.65f120.1 = fn_decl @C__carbon_thunk.1 [concrete = constants.%C__carbon_thunk.d98342.1] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %operator_Plus__carbon_thunk.decl: %operator_Plus__carbon_thunk.type = fn_decl @operator_Plus__carbon_thunk [concrete = constants.%operator_Plus__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.C.decl.b59a3a.2: %C.C.type.8a7395.2 = fn_decl @C.C.2 [concrete = constants.%C.C.85bfb1.2] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C__carbon_thunk.decl.8acdfe.2: %C__carbon_thunk.type.65f120.2 = fn_decl @C__carbon_thunk.2 [concrete = constants.%C__carbon_thunk.d98342.2] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.cpp_destructor.decl: %C.cpp_destructor.type = fn_decl @C.cpp_destructor [concrete = constants.%C.cpp_destructor] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c1.patt: %pattern_type.b28 = value_binding_pattern c1 [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc8_21: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %O.ref.loc8_24: type = name_ref O, imports.%O.decl [concrete = constants.%O]
// CHECK:STDOUT:   %C.ref.loc8_26: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %C.ref.loc8_28: %C.C.cpp_overload_set.type = name_ref C, imports.%C.C.cpp_overload_set.value [concrete = constants.%C.C.cpp_overload_set.value]
// CHECK:STDOUT:   %.loc8_31.1: ref %C = temporary_storage
// CHECK:STDOUT:   %addr.loc8: %ptr.de2 = addr_of %.loc8_31.1
// CHECK:STDOUT:   %C__carbon_thunk.call.loc8: init %empty_tuple.type = call imports.%C__carbon_thunk.decl.8acdfe.1(%addr.loc8)
// CHECK:STDOUT:   %.loc8_31.2: init %C to %.loc8_31.1 = mark_in_place_init %C__carbon_thunk.call.loc8
// CHECK:STDOUT:   %.loc8_16: type = splice_block %C.ref.loc8_16 [concrete = constants.%C] {
// CHECK:STDOUT:     %Cpp.ref.loc8_11: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %O.ref.loc8_14: type = name_ref O, imports.%O.decl [concrete = constants.%O]
// CHECK:STDOUT:     %C.ref.loc8_16: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc8_31.3: ref %C = temporary %.loc8_31.1, %.loc8_31.2
// CHECK:STDOUT:   %.loc8_31.4: %C = acquire_value %.loc8_31.3
// CHECK:STDOUT:   %c1: %C = value_binding c1, %.loc8_31.4
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c2.patt: %pattern_type.b28 = value_binding_pattern c2 [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc9_21: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %O.ref.loc9_24: type = name_ref O, imports.%O.decl [concrete = constants.%O]
// CHECK:STDOUT:   %C.ref.loc9_26: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %C.ref.loc9_28: %C.C.cpp_overload_set.type = name_ref C, imports.%C.C.cpp_overload_set.value [concrete = constants.%C.C.cpp_overload_set.value]
// CHECK:STDOUT:   %.loc9_31.1: ref %C = temporary_storage
// CHECK:STDOUT:   %addr.loc9: %ptr.de2 = addr_of %.loc9_31.1
// CHECK:STDOUT:   %C__carbon_thunk.call.loc9: init %empty_tuple.type = call imports.%C__carbon_thunk.decl.8acdfe.1(%addr.loc9)
// CHECK:STDOUT:   %.loc9_31.2: init %C to %.loc9_31.1 = mark_in_place_init %C__carbon_thunk.call.loc9
// CHECK:STDOUT:   %.loc9_16: type = splice_block %C.ref.loc9_16 [concrete = constants.%C] {
// CHECK:STDOUT:     %Cpp.ref.loc9_11: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %O.ref.loc9_14: type = name_ref O, imports.%O.decl [concrete = constants.%O]
// CHECK:STDOUT:     %C.ref.loc9_16: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc9_31.3: ref %C = temporary %.loc9_31.1, %.loc9_31.2
// CHECK:STDOUT:   %.loc9_31.4: %C = acquire_value %.loc9_31.3
// CHECK:STDOUT:   %c2: %C = value_binding c2, %.loc9_31.4
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c3.patt: %pattern_type.b28 = value_binding_pattern c3 [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c1.ref: %C = name_ref c1, %c1
// CHECK:STDOUT:   %c2.ref: %C = name_ref c2, %c2
// CHECK:STDOUT:   %.loc10_31.1: ref %C = temporary_storage
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %impl.elem0.loc10_28: %.724 = impl_witness_access constants.%custom_witness.8ae, element0 [concrete = constants.%C.Op.30d0ef.1]
// CHECK:STDOUT:   %bound_method.loc10_28: <bound method> = bound_method %c1.ref, %impl.elem0.loc10_28
// CHECK:STDOUT:   %.loc10_28.1: ref %C = temporary_storage
// CHECK:STDOUT:   %Op.ref.loc10_28: %C.C.type.8a7395.2 = name_ref Op, imports.%C.C.decl.b59a3a.2 [concrete = constants.%C.C.85bfb1.2]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %.loc10_28.2: ref %C = value_as_ref %c1.ref
// CHECK:STDOUT:   %addr.loc10_28.1: %ptr.de2 = addr_of %.loc10_28.2
// CHECK:STDOUT:   %.loc10_28.3: %ptr.f09 = as_compatible %addr.loc10_28.1
// CHECK:STDOUT:   %.loc10_28.4: %ptr.f09 = converted %addr.loc10_28.1, %.loc10_28.3
// CHECK:STDOUT:   %addr.loc10_28.2: %ptr.de2 = addr_of %lhs.var
// CHECK:STDOUT:   %C__carbon_thunk.call.loc10_28: init %empty_tuple.type = call imports.%C__carbon_thunk.decl.8acdfe.2(%.loc10_28.4, %addr.loc10_28.2)
// CHECK:STDOUT:   %.loc10_28.5: init %C to %lhs.var = mark_in_place_init %C__carbon_thunk.call.loc10_28
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %impl.elem0.loc10_33: %.724 = impl_witness_access constants.%custom_witness.8ae, element0 [concrete = constants.%C.Op.30d0ef.1]
// CHECK:STDOUT:   %bound_method.loc10_33: <bound method> = bound_method %c2.ref, %impl.elem0.loc10_33
// CHECK:STDOUT:   %.loc10_33.1: ref %C = temporary_storage
// CHECK:STDOUT:   %Op.ref.loc10_33: %C.C.type.8a7395.2 = name_ref Op, imports.%C.C.decl.b59a3a.2 [concrete = constants.%C.C.85bfb1.2]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %.loc10_33.2: ref %C = value_as_ref %c2.ref
// CHECK:STDOUT:   %addr.loc10_33.1: %ptr.de2 = addr_of %.loc10_33.2
// CHECK:STDOUT:   %.loc10_33.3: %ptr.f09 = as_compatible %addr.loc10_33.1
// CHECK:STDOUT:   %.loc10_33.4: %ptr.f09 = converted %addr.loc10_33.1, %.loc10_33.3
// CHECK:STDOUT:   %addr.loc10_33.2: %ptr.de2 = addr_of %rhs.var
// CHECK:STDOUT:   %C__carbon_thunk.call.loc10_33: init %empty_tuple.type = call imports.%C__carbon_thunk.decl.8acdfe.2(%.loc10_33.4, %addr.loc10_33.2)
// CHECK:STDOUT:   %.loc10_33.5: init %C to %rhs.var = mark_in_place_init %C__carbon_thunk.call.loc10_33
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %addr.loc10_31.1: %ptr.de2 = addr_of %lhs.var
// CHECK:STDOUT:   %addr.loc10_31.2: %ptr.de2 = addr_of %rhs.var
// CHECK:STDOUT:   %addr.loc10_31.3: %ptr.de2 = addr_of %.loc10_31.1
// CHECK:STDOUT:   %operator_Plus__carbon_thunk.call: init %empty_tuple.type = call imports.%operator_Plus__carbon_thunk.decl(%addr.loc10_31.1, %addr.loc10_31.2, %addr.loc10_31.3)
// CHECK:STDOUT:   %.loc10_31.2: init %C to %.loc10_31.1 = mark_in_place_init %operator_Plus__carbon_thunk.call
// CHECK:STDOUT:   %.loc10_23: type = splice_block %C.ref.loc10 [concrete = constants.%C] {
// CHECK:STDOUT:     %Cpp.ref.loc10: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %O.ref.loc10: type = name_ref O, imports.%O.decl [concrete = constants.%O]
// CHECK:STDOUT:     %C.ref.loc10: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc10_31.3: ref %C = temporary %.loc10_31.1, %.loc10_31.2
// CHECK:STDOUT:   %.loc10_31.4: %C = acquire_value %.loc10_31.3
// CHECK:STDOUT:   %c3: %C = value_binding c3, %.loc10_31.4
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %C.Op.bound.loc10: <bound method> = bound_method %.loc10_31.3, constants.%C.Op.30d0ef.2
// CHECK:STDOUT:   %Op.ref.loc10_31: %C.cpp_destructor.type = name_ref Op, imports.%C.cpp_destructor.decl [concrete = constants.%C.cpp_destructor]
// CHECK:STDOUT:   %C.cpp_destructor.bound.loc10: <bound method> = bound_method %.loc10_31.3, %Op.ref.loc10_31
// CHECK:STDOUT:   %C.cpp_destructor.call.loc10: init %empty_tuple.type = call %C.cpp_destructor.bound.loc10(%.loc10_31.3)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %C.Op.bound.loc9: <bound method> = bound_method %.loc9_31.3, constants.%C.Op.30d0ef.2
// CHECK:STDOUT:   %Op.ref.loc9: %C.cpp_destructor.type = name_ref Op, imports.%C.cpp_destructor.decl [concrete = constants.%C.cpp_destructor]
// CHECK:STDOUT:   %C.cpp_destructor.bound.loc9: <bound method> = bound_method %.loc9_31.3, %Op.ref.loc9
// CHECK:STDOUT:   %C.cpp_destructor.call.loc9: init %empty_tuple.type = call %C.cpp_destructor.bound.loc9(%.loc9_31.3)
// CHECK:STDOUT:   %C.Op.bound.loc8: <bound method> = bound_method %.loc8_31.3, constants.%C.Op.30d0ef.2
// CHECK:STDOUT:   %Op.ref.loc8: %C.cpp_destructor.type = name_ref Op, imports.%C.cpp_destructor.decl [concrete = constants.%C.cpp_destructor]
// CHECK:STDOUT:   %C.cpp_destructor.bound.loc8: <bound method> = bound_method %.loc8_31.3, %Op.ref.loc8
// CHECK:STDOUT:   %C.cpp_destructor.call.loc8: init %empty_tuple.type = call %C.cpp_destructor.bound.loc8(%.loc8_31.3)
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
// CHECK:STDOUT:   %C__carbon_thunk.type.65f120.1: type = fn_type @C__carbon_thunk.1 [concrete]
// CHECK:STDOUT:   %C__carbon_thunk.d98342.1: %C__carbon_thunk.type.65f120.1 = struct_value () [concrete]
// CHECK:STDOUT:   %operator_Plus__carbon_thunk.type: type = fn_type @operator_Plus__carbon_thunk [concrete]
// CHECK:STDOUT:   %operator_Plus__carbon_thunk: %operator_Plus__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %C.C.type.9dcb25.2: type = fn_type @C.C.2 [concrete]
// CHECK:STDOUT:   %C.C.0dc538.2: %C.C.type.9dcb25.2 = struct_value () [concrete]
// CHECK:STDOUT:   %const.d2c: type = const_type %C [concrete]
// CHECK:STDOUT:   %ptr.2af: type = ptr_type %const.d2c [concrete]
// CHECK:STDOUT:   %C__carbon_thunk.type.65f120.2: type = fn_type @C__carbon_thunk.2 [concrete]
// CHECK:STDOUT:   %C__carbon_thunk.d98342.2: %C__carbon_thunk.type.65f120.2 = struct_value () [concrete]
// CHECK:STDOUT:   %C.Op.type.4e4020.1: type = fn_type @C.Op.1 [concrete]
// CHECK:STDOUT:   %C.Op.ceeb9e.1: %C.Op.type.4e4020.1 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.0c9: <witness> = custom_witness (%C.Op.ceeb9e.1), @Copy [concrete]
// CHECK:STDOUT:   %Copy.facet.fa4: %Copy.type = facet_value %C, (%custom_witness.0c9) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.4d3: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet.fa4) [concrete]
// CHECK:STDOUT:   %.3d6: type = fn_type_with_self_type %Copy.WithSelf.Op.type.4d3, %Copy.facet.fa4 [concrete]
// CHECK:STDOUT:   %C.cpp_destructor.type: type = fn_type @C.cpp_destructor [concrete]
// CHECK:STDOUT:   %C.cpp_destructor: %C.cpp_destructor.type = struct_value () [concrete]
// CHECK:STDOUT:   %C.Op.type.4e4020.2: type = fn_type @C.Op.2 [concrete]
// CHECK:STDOUT:   %C.Op.ceeb9e.2: %C.Op.type.4e4020.2 = struct_value () [concrete]
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
// CHECK:STDOUT:   %C__carbon_thunk.decl.8acdfe.1: %C__carbon_thunk.type.65f120.1 = fn_decl @C__carbon_thunk.1 [concrete = constants.%C__carbon_thunk.d98342.1] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %operator_Plus__carbon_thunk.decl: %operator_Plus__carbon_thunk.type = fn_decl @operator_Plus__carbon_thunk [concrete = constants.%operator_Plus__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.C.decl.240814.2: %C.C.type.9dcb25.2 = fn_decl @C.C.2 [concrete = constants.%C.C.0dc538.2] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C__carbon_thunk.decl.8acdfe.2: %C__carbon_thunk.type.65f120.2 = fn_decl @C__carbon_thunk.2 [concrete = constants.%C__carbon_thunk.d98342.2] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.cpp_destructor.decl: %C.cpp_destructor.type = fn_decl @C.cpp_destructor [concrete = constants.%C.cpp_destructor] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c1.patt: %pattern_type.84b = value_binding_pattern c1 [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc8_23: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %N.ref.loc8_26: <namespace> = name_ref N, imports.%N [concrete = imports.%N]
// CHECK:STDOUT:   %O.ref.loc8_28: type = name_ref O, imports.%O.decl [concrete = constants.%O]
// CHECK:STDOUT:   %C.ref.loc8_30: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %C.ref.loc8_32: %C.C.cpp_overload_set.type = name_ref C, imports.%C.C.cpp_overload_set.value [concrete = constants.%C.C.cpp_overload_set.value]
// CHECK:STDOUT:   %.loc8_35.1: ref %C = temporary_storage
// CHECK:STDOUT:   %addr.loc8: %ptr.4b2 = addr_of %.loc8_35.1
// CHECK:STDOUT:   %C__carbon_thunk.call.loc8: init %empty_tuple.type = call imports.%C__carbon_thunk.decl.8acdfe.1(%addr.loc8)
// CHECK:STDOUT:   %.loc8_35.2: init %C to %.loc8_35.1 = mark_in_place_init %C__carbon_thunk.call.loc8
// CHECK:STDOUT:   %.loc8_18: type = splice_block %C.ref.loc8_18 [concrete = constants.%C] {
// CHECK:STDOUT:     %Cpp.ref.loc8_11: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %N.ref.loc8_14: <namespace> = name_ref N, imports.%N [concrete = imports.%N]
// CHECK:STDOUT:     %O.ref.loc8_16: type = name_ref O, imports.%O.decl [concrete = constants.%O]
// CHECK:STDOUT:     %C.ref.loc8_18: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc8_35.3: ref %C = temporary %.loc8_35.1, %.loc8_35.2
// CHECK:STDOUT:   %.loc8_35.4: %C = acquire_value %.loc8_35.3
// CHECK:STDOUT:   %c1: %C = value_binding c1, %.loc8_35.4
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c2.patt: %pattern_type.84b = value_binding_pattern c2 [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc9_23: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %N.ref.loc9_26: <namespace> = name_ref N, imports.%N [concrete = imports.%N]
// CHECK:STDOUT:   %O.ref.loc9_28: type = name_ref O, imports.%O.decl [concrete = constants.%O]
// CHECK:STDOUT:   %C.ref.loc9_30: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %C.ref.loc9_32: %C.C.cpp_overload_set.type = name_ref C, imports.%C.C.cpp_overload_set.value [concrete = constants.%C.C.cpp_overload_set.value]
// CHECK:STDOUT:   %.loc9_35.1: ref %C = temporary_storage
// CHECK:STDOUT:   %addr.loc9: %ptr.4b2 = addr_of %.loc9_35.1
// CHECK:STDOUT:   %C__carbon_thunk.call.loc9: init %empty_tuple.type = call imports.%C__carbon_thunk.decl.8acdfe.1(%addr.loc9)
// CHECK:STDOUT:   %.loc9_35.2: init %C to %.loc9_35.1 = mark_in_place_init %C__carbon_thunk.call.loc9
// CHECK:STDOUT:   %.loc9_18: type = splice_block %C.ref.loc9_18 [concrete = constants.%C] {
// CHECK:STDOUT:     %Cpp.ref.loc9_11: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %N.ref.loc9_14: <namespace> = name_ref N, imports.%N [concrete = imports.%N]
// CHECK:STDOUT:     %O.ref.loc9_16: type = name_ref O, imports.%O.decl [concrete = constants.%O]
// CHECK:STDOUT:     %C.ref.loc9_18: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc9_35.3: ref %C = temporary %.loc9_35.1, %.loc9_35.2
// CHECK:STDOUT:   %.loc9_35.4: %C = acquire_value %.loc9_35.3
// CHECK:STDOUT:   %c2: %C = value_binding c2, %.loc9_35.4
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c3.patt: %pattern_type.84b = value_binding_pattern c3 [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c1.ref: %C = name_ref c1, %c1
// CHECK:STDOUT:   %c2.ref: %C = name_ref c2, %c2
// CHECK:STDOUT:   %.loc10_33.1: ref %C = temporary_storage
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %impl.elem0.loc10_30: %.3d6 = impl_witness_access constants.%custom_witness.0c9, element0 [concrete = constants.%C.Op.ceeb9e.1]
// CHECK:STDOUT:   %bound_method.loc10_30: <bound method> = bound_method %c1.ref, %impl.elem0.loc10_30
// CHECK:STDOUT:   %.loc10_30.1: ref %C = temporary_storage
// CHECK:STDOUT:   %Op.ref.loc10_30: %C.C.type.9dcb25.2 = name_ref Op, imports.%C.C.decl.240814.2 [concrete = constants.%C.C.0dc538.2]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %.loc10_30.2: ref %C = value_as_ref %c1.ref
// CHECK:STDOUT:   %addr.loc10_30.1: %ptr.4b2 = addr_of %.loc10_30.2
// CHECK:STDOUT:   %.loc10_30.3: %ptr.2af = as_compatible %addr.loc10_30.1
// CHECK:STDOUT:   %.loc10_30.4: %ptr.2af = converted %addr.loc10_30.1, %.loc10_30.3
// CHECK:STDOUT:   %addr.loc10_30.2: %ptr.4b2 = addr_of %lhs.var
// CHECK:STDOUT:   %C__carbon_thunk.call.loc10_30: init %empty_tuple.type = call imports.%C__carbon_thunk.decl.8acdfe.2(%.loc10_30.4, %addr.loc10_30.2)
// CHECK:STDOUT:   %.loc10_30.5: init %C to %lhs.var = mark_in_place_init %C__carbon_thunk.call.loc10_30
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %impl.elem0.loc10_35: %.3d6 = impl_witness_access constants.%custom_witness.0c9, element0 [concrete = constants.%C.Op.ceeb9e.1]
// CHECK:STDOUT:   %bound_method.loc10_35: <bound method> = bound_method %c2.ref, %impl.elem0.loc10_35
// CHECK:STDOUT:   %.loc10_35.1: ref %C = temporary_storage
// CHECK:STDOUT:   %Op.ref.loc10_35: %C.C.type.9dcb25.2 = name_ref Op, imports.%C.C.decl.240814.2 [concrete = constants.%C.C.0dc538.2]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %.loc10_35.2: ref %C = value_as_ref %c2.ref
// CHECK:STDOUT:   %addr.loc10_35.1: %ptr.4b2 = addr_of %.loc10_35.2
// CHECK:STDOUT:   %.loc10_35.3: %ptr.2af = as_compatible %addr.loc10_35.1
// CHECK:STDOUT:   %.loc10_35.4: %ptr.2af = converted %addr.loc10_35.1, %.loc10_35.3
// CHECK:STDOUT:   %addr.loc10_35.2: %ptr.4b2 = addr_of %rhs.var
// CHECK:STDOUT:   %C__carbon_thunk.call.loc10_35: init %empty_tuple.type = call imports.%C__carbon_thunk.decl.8acdfe.2(%.loc10_35.4, %addr.loc10_35.2)
// CHECK:STDOUT:   %.loc10_35.5: init %C to %rhs.var = mark_in_place_init %C__carbon_thunk.call.loc10_35
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %addr.loc10_33.1: %ptr.4b2 = addr_of %lhs.var
// CHECK:STDOUT:   %addr.loc10_33.2: %ptr.4b2 = addr_of %rhs.var
// CHECK:STDOUT:   %addr.loc10_33.3: %ptr.4b2 = addr_of %.loc10_33.1
// CHECK:STDOUT:   %operator_Plus__carbon_thunk.call: init %empty_tuple.type = call imports.%operator_Plus__carbon_thunk.decl(%addr.loc10_33.1, %addr.loc10_33.2, %addr.loc10_33.3)
// CHECK:STDOUT:   %.loc10_33.2: init %C to %.loc10_33.1 = mark_in_place_init %operator_Plus__carbon_thunk.call
// CHECK:STDOUT:   %.loc10_25: type = splice_block %C.ref.loc10 [concrete = constants.%C] {
// CHECK:STDOUT:     %Cpp.ref.loc10: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %N.ref.loc10: <namespace> = name_ref N, imports.%N [concrete = imports.%N]
// CHECK:STDOUT:     %O.ref.loc10: type = name_ref O, imports.%O.decl [concrete = constants.%O]
// CHECK:STDOUT:     %C.ref.loc10: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc10_33.3: ref %C = temporary %.loc10_33.1, %.loc10_33.2
// CHECK:STDOUT:   %.loc10_33.4: %C = acquire_value %.loc10_33.3
// CHECK:STDOUT:   %c3: %C = value_binding c3, %.loc10_33.4
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %C.Op.bound.loc10: <bound method> = bound_method %.loc10_33.3, constants.%C.Op.ceeb9e.2
// CHECK:STDOUT:   %Op.ref.loc10_33: %C.cpp_destructor.type = name_ref Op, imports.%C.cpp_destructor.decl [concrete = constants.%C.cpp_destructor]
// CHECK:STDOUT:   %C.cpp_destructor.bound.loc10: <bound method> = bound_method %.loc10_33.3, %Op.ref.loc10_33
// CHECK:STDOUT:   %C.cpp_destructor.call.loc10: init %empty_tuple.type = call %C.cpp_destructor.bound.loc10(%.loc10_33.3)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %C.Op.bound.loc9: <bound method> = bound_method %.loc9_35.3, constants.%C.Op.ceeb9e.2
// CHECK:STDOUT:   %Op.ref.loc9: %C.cpp_destructor.type = name_ref Op, imports.%C.cpp_destructor.decl [concrete = constants.%C.cpp_destructor]
// CHECK:STDOUT:   %C.cpp_destructor.bound.loc9: <bound method> = bound_method %.loc9_35.3, %Op.ref.loc9
// CHECK:STDOUT:   %C.cpp_destructor.call.loc9: init %empty_tuple.type = call %C.cpp_destructor.bound.loc9(%.loc9_35.3)
// CHECK:STDOUT:   %C.Op.bound.loc8: <bound method> = bound_method %.loc8_35.3, constants.%C.Op.ceeb9e.2
// CHECK:STDOUT:   %Op.ref.loc8: %C.cpp_destructor.type = name_ref Op, imports.%C.cpp_destructor.decl [concrete = constants.%C.cpp_destructor]
// CHECK:STDOUT:   %C.cpp_destructor.bound.loc8: <bound method> = bound_method %.loc8_35.3, %Op.ref.loc8
// CHECK:STDOUT:   %C.cpp_destructor.call.loc8: init %empty_tuple.type = call %C.cpp_destructor.bound.loc8(%.loc8_35.3)
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
// CHECK:STDOUT:   %C__carbon_thunk.type.65f120.1: type = fn_type @C__carbon_thunk.1 [concrete]
// CHECK:STDOUT:   %C__carbon_thunk.d98342.1: %C__carbon_thunk.type.65f120.1 = struct_value () [concrete]
// CHECK:STDOUT:   %C.cpp_operator.type.dab96a.1: type = fn_type @C.cpp_operator.1 [concrete]
// CHECK:STDOUT:   %C.cpp_operator.d21c75.1: %C.cpp_operator.type.dab96a.1 = struct_value () [concrete]
// CHECK:STDOUT:   %operator_Minus__carbon_thunk.type: type = fn_type @operator_Minus__carbon_thunk [concrete]
// CHECK:STDOUT:   %operator_Minus__carbon_thunk: %operator_Minus__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %C.cpp_operator.type.dab96a.2: type = fn_type @C.cpp_operator.2 [concrete]
// CHECK:STDOUT:   %C.cpp_operator.d21c75.2: %C.cpp_operator.type.dab96a.2 = struct_value () [concrete]
// CHECK:STDOUT:   %operator_Plus__carbon_thunk.type: type = fn_type @operator_Plus__carbon_thunk [concrete]
// CHECK:STDOUT:   %operator_Plus__carbon_thunk: %operator_Plus__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %C.C.type.065f0c.2: type = fn_type @C.C.2 [concrete]
// CHECK:STDOUT:   %C.C.6522d8.2: %C.C.type.065f0c.2 = struct_value () [concrete]
// CHECK:STDOUT:   %const.057: type = const_type %C [concrete]
// CHECK:STDOUT:   %ptr.c92: type = ptr_type %const.057 [concrete]
// CHECK:STDOUT:   %C__carbon_thunk.type.65f120.2: type = fn_type @C__carbon_thunk.2 [concrete]
// CHECK:STDOUT:   %C__carbon_thunk.d98342.2: %C__carbon_thunk.type.65f120.2 = struct_value () [concrete]
// CHECK:STDOUT:   %C.Op.type.ea1bc9.1: type = fn_type @C.Op.1 [concrete]
// CHECK:STDOUT:   %C.Op.068411.1: %C.Op.type.ea1bc9.1 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.681: <witness> = custom_witness (%C.Op.068411.1), @Copy [concrete]
// CHECK:STDOUT:   %Copy.facet.3b9: %Copy.type = facet_value %C, (%custom_witness.681) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.530: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet.3b9) [concrete]
// CHECK:STDOUT:   %.a81: type = fn_type_with_self_type %Copy.WithSelf.Op.type.530, %Copy.facet.3b9 [concrete]
// CHECK:STDOUT:   %C.cpp_destructor.type: type = fn_type @C.cpp_destructor [concrete]
// CHECK:STDOUT:   %C.cpp_destructor: %C.cpp_destructor.type = struct_value () [concrete]
// CHECK:STDOUT:   %C.Op.type.ea1bc9.2: type = fn_type @C.Op.2 [concrete]
// CHECK:STDOUT:   %C.Op.068411.2: %C.Op.type.ea1bc9.2 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   %C.C.cpp_overload_set.value: %C.C.cpp_overload_set.type = cpp_overload_set_value @C.C.cpp_overload_set [concrete = constants.%C.C.cpp_overload_set.value]
// CHECK:STDOUT:   %C__carbon_thunk.decl.8acdfe.1: %C__carbon_thunk.type.65f120.1 = fn_decl @C__carbon_thunk.1 [concrete = constants.%C__carbon_thunk.d98342.1] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.cpp_operator.decl.828f43.1: %C.cpp_operator.type.dab96a.1 = fn_decl @C.cpp_operator.1 [concrete = constants.%C.cpp_operator.d21c75.1] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %operator_Minus__carbon_thunk.decl: %operator_Minus__carbon_thunk.type = fn_decl @operator_Minus__carbon_thunk [concrete = constants.%operator_Minus__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.cpp_operator.decl.828f43.2: %C.cpp_operator.type.dab96a.2 = fn_decl @C.cpp_operator.2 [concrete = constants.%C.cpp_operator.d21c75.2] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %operator_Plus__carbon_thunk.decl: %operator_Plus__carbon_thunk.type = fn_decl @operator_Plus__carbon_thunk [concrete = constants.%operator_Plus__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.C.decl.041267.2: %C.C.type.065f0c.2 = fn_decl @C.C.2 [concrete = constants.%C.C.6522d8.2] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C__carbon_thunk.decl.8acdfe.2: %C__carbon_thunk.type.65f120.2 = fn_decl @C__carbon_thunk.2 [concrete = constants.%C__carbon_thunk.d98342.2] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.cpp_destructor.decl: %C.cpp_destructor.type = fn_decl @C.cpp_destructor [concrete = constants.%C.cpp_destructor] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
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
// CHECK:STDOUT:   %Cpp.ref.loc8_19: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %C.ref.loc8_22: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %C.ref.loc8_24: %C.C.cpp_overload_set.type = name_ref C, imports.%C.C.cpp_overload_set.value [concrete = constants.%C.C.cpp_overload_set.value]
// CHECK:STDOUT:   %.loc8_3: ref %C = splice_block %c1.var {}
// CHECK:STDOUT:   %addr.loc8: %ptr.d9e = addr_of %.loc8_3
// CHECK:STDOUT:   %C__carbon_thunk.call.loc8: init %empty_tuple.type = call imports.%C__carbon_thunk.decl.8acdfe.1(%addr.loc8)
// CHECK:STDOUT:   %.loc8_27: init %C to %.loc8_3 = mark_in_place_init %C__carbon_thunk.call.loc8
// CHECK:STDOUT:   assign %c1.var, %.loc8_27
// CHECK:STDOUT:   %.loc8_14: type = splice_block %C.ref.loc8_14 [concrete = constants.%C] {
// CHECK:STDOUT:     %Cpp.ref.loc8_11: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %C.ref.loc8_14: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c1: ref %C = ref_binding c1, %c1.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c2.patt: %pattern_type.217 = ref_binding_pattern c2 [concrete]
// CHECK:STDOUT:     %c2.var_patt: %pattern_type.217 = var_pattern %c2.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c2.var: ref %C = var %c2.var_patt
// CHECK:STDOUT:   %c1.ref.loc9: ref %C = name_ref c1, %c1
// CHECK:STDOUT:   %C.cpp_operator.bound.loc9: <bound method> = bound_method %c1.ref.loc9, imports.%C.cpp_operator.decl.828f43.1
// CHECK:STDOUT:   %.loc9_3: ref %C = splice_block %c2.var {}
// CHECK:STDOUT:   %addr.loc9: %ptr.d9e = addr_of %.loc9_3
// CHECK:STDOUT:   %operator_Minus__carbon_thunk.call: init %empty_tuple.type = call imports.%operator_Minus__carbon_thunk.decl(%c1.ref.loc9, %addr.loc9)
// CHECK:STDOUT:   %.loc9_19: init %C to %.loc9_3 = mark_in_place_init %operator_Minus__carbon_thunk.call
// CHECK:STDOUT:   assign %c2.var, %.loc9_19
// CHECK:STDOUT:   %.loc9_14: type = splice_block %C.ref.loc9 [concrete = constants.%C] {
// CHECK:STDOUT:     %Cpp.ref.loc9: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %C.ref.loc9: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c2: ref %C = ref_binding c2, %c2.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c3.patt: %pattern_type.217 = ref_binding_pattern c3 [concrete]
// CHECK:STDOUT:     %c3.var_patt: %pattern_type.217 = var_pattern %c3.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c3.var: ref %C = var %c3.var_patt
// CHECK:STDOUT:   %c1.ref.loc10: ref %C = name_ref c1, %c1
// CHECK:STDOUT:   %c2.ref: ref %C = name_ref c2, %c2
// CHECK:STDOUT:   %C.cpp_operator.bound.loc10: <bound method> = bound_method %c1.ref.loc10, imports.%C.cpp_operator.decl.828f43.2
// CHECK:STDOUT:   %.loc10_3: ref %C = splice_block %c3.var {}
// CHECK:STDOUT:   %.loc10_31.1: %C = acquire_value %c2.ref
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %impl.elem0.loc10: %.a81 = impl_witness_access constants.%custom_witness.681, element0 [concrete = constants.%C.Op.068411.1]
// CHECK:STDOUT:   %bound_method.loc10: <bound method> = bound_method %.loc10_31.1, %impl.elem0.loc10
// CHECK:STDOUT:   %.loc10_31.2: ref %C = temporary_storage
// CHECK:STDOUT:   %Op.ref.loc10_31: %C.C.type.065f0c.2 = name_ref Op, imports.%C.C.decl.041267.2 [concrete = constants.%C.C.6522d8.2]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %.loc10_31.3: ref %C = value_as_ref %.loc10_31.1
// CHECK:STDOUT:   %addr.loc10_31.1: %ptr.d9e = addr_of %.loc10_31.3
// CHECK:STDOUT:   %.loc10_31.4: %ptr.c92 = as_compatible %addr.loc10_31.1
// CHECK:STDOUT:   %.loc10_31.5: %ptr.c92 = converted %addr.loc10_31.1, %.loc10_31.4
// CHECK:STDOUT:   %addr.loc10_31.2: %ptr.d9e = addr_of %rhs.var
// CHECK:STDOUT:   %C__carbon_thunk.call.loc10: init %empty_tuple.type = call imports.%C__carbon_thunk.decl.8acdfe.2(%.loc10_31.5, %addr.loc10_31.2)
// CHECK:STDOUT:   %.loc10_31.6: init %C to %rhs.var = mark_in_place_init %C__carbon_thunk.call.loc10
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %addr.loc10_29.1: %ptr.d9e = addr_of %rhs.var
// CHECK:STDOUT:   %addr.loc10_29.2: %ptr.d9e = addr_of %.loc10_3
// CHECK:STDOUT:   %operator_Plus__carbon_thunk.call: init %empty_tuple.type = call imports.%operator_Plus__carbon_thunk.decl(%c1.ref.loc10, %addr.loc10_29.1, %addr.loc10_29.2)
// CHECK:STDOUT:   %.loc10_29: init %C to %.loc10_3 = mark_in_place_init %operator_Plus__carbon_thunk.call
// CHECK:STDOUT:   assign %c3.var, %.loc10_29
// CHECK:STDOUT:   %.loc10_21: type = splice_block %C.ref.loc10 [concrete = constants.%C] {
// CHECK:STDOUT:     %Cpp.ref.loc10: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %C.ref.loc10: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c3: ref %C = ref_binding c3, %c3.var
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %C.Op.bound.loc10: <bound method> = bound_method %c3.var, constants.%C.Op.068411.2
// CHECK:STDOUT:   %Op.ref.loc10_3: %C.cpp_destructor.type = name_ref Op, imports.%C.cpp_destructor.decl [concrete = constants.%C.cpp_destructor]
// CHECK:STDOUT:   %C.cpp_destructor.bound.loc10: <bound method> = bound_method %c3.var, %Op.ref.loc10_3
// CHECK:STDOUT:   %C.cpp_destructor.call.loc10: init %empty_tuple.type = call %C.cpp_destructor.bound.loc10(%c3.var)
// CHECK:STDOUT:   %C.Op.bound.loc9: <bound method> = bound_method %c2.var, constants.%C.Op.068411.2
// CHECK:STDOUT:   %Op.ref.loc9: %C.cpp_destructor.type = name_ref Op, imports.%C.cpp_destructor.decl [concrete = constants.%C.cpp_destructor]
// CHECK:STDOUT:   %C.cpp_destructor.bound.loc9: <bound method> = bound_method %c2.var, %Op.ref.loc9
// CHECK:STDOUT:   %C.cpp_destructor.call.loc9: init %empty_tuple.type = call %C.cpp_destructor.bound.loc9(%c2.var)
// CHECK:STDOUT:   %C.Op.bound.loc8: <bound method> = bound_method %c1.var, constants.%C.Op.068411.2
// CHECK:STDOUT:   %Op.ref.loc8: %C.cpp_destructor.type = name_ref Op, imports.%C.cpp_destructor.decl [concrete = constants.%C.cpp_destructor]
// CHECK:STDOUT:   %C.cpp_destructor.bound.loc8: <bound method> = bound_method %c1.var, %Op.ref.loc8
// CHECK:STDOUT:   %C.cpp_destructor.call.loc8: init %empty_tuple.type = call %C.cpp_destructor.bound.loc8(%c1.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- indirect_template_instantiation.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %B: type = class_type @B [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %ptr.a04: type = ptr_type %B [concrete]
// CHECK:STDOUT:   %operator_Plus__carbon_thunk.type: type = fn_type @operator_Plus__carbon_thunk [concrete]
// CHECK:STDOUT:   %operator_Plus__carbon_thunk: %operator_Plus__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %B.B.type: type = fn_type @B.B [concrete]
// CHECK:STDOUT:   %B.B: %B.B.type = struct_value () [concrete]
// CHECK:STDOUT:   %const.f76: type = const_type %B [concrete]
// CHECK:STDOUT:   %ptr.fe8: type = ptr_type %const.f76 [concrete]
// CHECK:STDOUT:   %B__carbon_thunk.type: type = fn_type @B__carbon_thunk [concrete]
// CHECK:STDOUT:   %B__carbon_thunk: %B__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %B.Op.type.8c681d.1: type = fn_type @B.Op.1 [concrete]
// CHECK:STDOUT:   %B.Op.1be686.1: %B.Op.type.8c681d.1 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.589: <witness> = custom_witness (%B.Op.1be686.1), @Copy [concrete]
// CHECK:STDOUT:   %Copy.facet.fa3: %Copy.type = facet_value %B, (%custom_witness.589) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.aef: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet.fa3) [concrete]
// CHECK:STDOUT:   %.73b: type = fn_type_with_self_type %Copy.WithSelf.Op.type.aef, %Copy.facet.fa3 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %operator_Plus__carbon_thunk.decl: %operator_Plus__carbon_thunk.type = fn_decl @operator_Plus__carbon_thunk [concrete = constants.%operator_Plus__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %B.B.decl: %B.B.type = fn_decl @B.B [concrete = constants.%B.B] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %B__carbon_thunk.decl: %B__carbon_thunk.type = fn_decl @B__carbon_thunk [concrete = constants.%B__carbon_thunk] {
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
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %impl.elem0.loc14_10: %.73b = impl_witness_access constants.%custom_witness.589, element0 [concrete = constants.%B.Op.1be686.1]
// CHECK:STDOUT:   %bound_method.loc14_10: <bound method> = bound_method %x.ref.loc14_10, %impl.elem0.loc14_10
// CHECK:STDOUT:   %.loc14_10.1: ref %B = temporary_storage
// CHECK:STDOUT:   %Op.ref.loc14_10: %B.B.type = name_ref Op, imports.%B.B.decl [concrete = constants.%B.B]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %.loc14_10.2: ref %B = value_as_ref %x.ref.loc14_10
// CHECK:STDOUT:   %addr.loc14_10.1: %ptr.a04 = addr_of %.loc14_10.2
// CHECK:STDOUT:   %.loc14_10.3: %ptr.fe8 = as_compatible %addr.loc14_10.1
// CHECK:STDOUT:   %.loc14_10.4: %ptr.fe8 = converted %addr.loc14_10.1, %.loc14_10.3
// CHECK:STDOUT:   %addr.loc14_10.2: %ptr.a04 = addr_of %_.var.1
// CHECK:STDOUT:   %B__carbon_thunk.call.loc14_10: init %empty_tuple.type = call imports.%B__carbon_thunk.decl(%.loc14_10.4, %addr.loc14_10.2)
// CHECK:STDOUT:   %.loc14_10.5: init %B to %_.var.1 = mark_in_place_init %B__carbon_thunk.call.loc14_10
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %impl.elem0.loc14_14: %.73b = impl_witness_access constants.%custom_witness.589, element0 [concrete = constants.%B.Op.1be686.1]
// CHECK:STDOUT:   %bound_method.loc14_14: <bound method> = bound_method %x.ref.loc14_14, %impl.elem0.loc14_14
// CHECK:STDOUT:   %.loc14_14.1: ref %B = temporary_storage
// CHECK:STDOUT:   %Op.ref.loc14_14: %B.B.type = name_ref Op, imports.%B.B.decl [concrete = constants.%B.B]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %.loc14_14.2: ref %B = value_as_ref %x.ref.loc14_14
// CHECK:STDOUT:   %addr.loc14_14.1: %ptr.a04 = addr_of %.loc14_14.2
// CHECK:STDOUT:   %.loc14_14.3: %ptr.fe8 = as_compatible %addr.loc14_14.1
// CHECK:STDOUT:   %.loc14_14.4: %ptr.fe8 = converted %addr.loc14_14.1, %.loc14_14.3
// CHECK:STDOUT:   %addr.loc14_14.2: %ptr.a04 = addr_of %_.var.2
// CHECK:STDOUT:   %B__carbon_thunk.call.loc14_14: init %empty_tuple.type = call imports.%B__carbon_thunk.decl(%.loc14_14.4, %addr.loc14_14.2)
// CHECK:STDOUT:   %.loc14_14.5: init %B to %_.var.2 = mark_in_place_init %B__carbon_thunk.call.loc14_14
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %addr.loc14_12.1: %ptr.a04 = addr_of %_.var.1
// CHECK:STDOUT:   %addr.loc14_12.2: %ptr.a04 = addr_of %_.var.2
// CHECK:STDOUT:   %operator_Plus__carbon_thunk.call: init %i32 = call imports.%operator_Plus__carbon_thunk.decl(%addr.loc14_12.1, %addr.loc14_12.2)
// CHECK:STDOUT:   <elided>
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
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %C.C.type.065f0c.2: type = fn_type @C.C.2 [concrete]
// CHECK:STDOUT:   %C.C.6522d8.2: %C.C.type.065f0c.2 = struct_value () [concrete]
// CHECK:STDOUT:   %const.057: type = const_type %C [concrete]
// CHECK:STDOUT:   %ptr.c92: type = ptr_type %const.057 [concrete]
// CHECK:STDOUT:   %C__carbon_thunk.type.65f120.2: type = fn_type @C__carbon_thunk.2 [concrete]
// CHECK:STDOUT:   %C__carbon_thunk.d98342.2: %C__carbon_thunk.type.65f120.2 = struct_value () [concrete]
// CHECK:STDOUT:   %C.Op.type.ea1bc9.1: type = fn_type @C.Op.1 [concrete]
// CHECK:STDOUT:   %C.Op.068411.1: %C.Op.type.ea1bc9.1 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.681: <witness> = custom_witness (%C.Op.068411.1), @Copy [concrete]
// CHECK:STDOUT:   %Copy.facet.3b9: %Copy.type = facet_value %C, (%custom_witness.681) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.530: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet.3b9) [concrete]
// CHECK:STDOUT:   %.a81: type = fn_type_with_self_type %Copy.WithSelf.Op.type.530, %Copy.facet.3b9 [concrete]
// CHECK:STDOUT:   %C.cpp_destructor.type: type = fn_type @C.cpp_destructor [concrete]
// CHECK:STDOUT:   %C.cpp_destructor: %C.cpp_destructor.type = struct_value () [concrete]
// CHECK:STDOUT:   %C.Op.type.ea1bc9.2: type = fn_type @C.Op.2 [concrete]
// CHECK:STDOUT:   %C.Op.068411.2: %C.Op.type.ea1bc9.2 = struct_value () [concrete]
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
// CHECK:STDOUT:   %C.C.decl.041267.2: %C.C.type.065f0c.2 = fn_decl @C.C.2 [concrete = constants.%C.C.6522d8.2] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C__carbon_thunk.decl.8acdfe.2: %C__carbon_thunk.type.65f120.2 = fn_decl @C__carbon_thunk.2 [concrete = constants.%C__carbon_thunk.d98342.2] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.cpp_destructor.decl: %C.cpp_destructor.type = fn_decl @C.cpp_destructor [concrete = constants.%C.cpp_destructor] {
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
// CHECK:STDOUT:     %c3.patt: %pattern_type.217 = value_binding_pattern c3 [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c1.ref: %C = name_ref c1, %c1
// CHECK:STDOUT:   %c2.ref: %C = name_ref c2, %c2
// CHECK:STDOUT:   %.loc10_29.1: ref %C = temporary_storage
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %impl.elem0.loc10_26: %.a81 = impl_witness_access constants.%custom_witness.681, element0 [concrete = constants.%C.Op.068411.1]
// CHECK:STDOUT:   %bound_method.loc10_26: <bound method> = bound_method %c1.ref, %impl.elem0.loc10_26
// CHECK:STDOUT:   %.loc10_26.1: ref %C = temporary_storage
// CHECK:STDOUT:   %Op.ref.loc10_26: %C.C.type.065f0c.2 = name_ref Op, imports.%C.C.decl.041267.2 [concrete = constants.%C.C.6522d8.2]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %.loc10_26.2: ref %C = value_as_ref %c1.ref
// CHECK:STDOUT:   %addr.loc10_26.1: %ptr.d9e = addr_of %.loc10_26.2
// CHECK:STDOUT:   %.loc10_26.3: %ptr.c92 = as_compatible %addr.loc10_26.1
// CHECK:STDOUT:   %.loc10_26.4: %ptr.c92 = converted %addr.loc10_26.1, %.loc10_26.3
// CHECK:STDOUT:   %addr.loc10_26.2: %ptr.d9e = addr_of %lhs.var
// CHECK:STDOUT:   %C__carbon_thunk.call.loc10_26: init %empty_tuple.type = call imports.%C__carbon_thunk.decl.8acdfe.2(%.loc10_26.4, %addr.loc10_26.2)
// CHECK:STDOUT:   %.loc10_26.5: init %C to %lhs.var = mark_in_place_init %C__carbon_thunk.call.loc10_26
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %impl.elem0.loc10_31: %.a81 = impl_witness_access constants.%custom_witness.681, element0 [concrete = constants.%C.Op.068411.1]
// CHECK:STDOUT:   %bound_method.loc10_31: <bound method> = bound_method %c2.ref, %impl.elem0.loc10_31
// CHECK:STDOUT:   %.loc10_31.1: ref %C = temporary_storage
// CHECK:STDOUT:   %Op.ref.loc10_31: %C.C.type.065f0c.2 = name_ref Op, imports.%C.C.decl.041267.2 [concrete = constants.%C.C.6522d8.2]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %.loc10_31.2: ref %C = value_as_ref %c2.ref
// CHECK:STDOUT:   %addr.loc10_31.1: %ptr.d9e = addr_of %.loc10_31.2
// CHECK:STDOUT:   %.loc10_31.3: %ptr.c92 = as_compatible %addr.loc10_31.1
// CHECK:STDOUT:   %.loc10_31.4: %ptr.c92 = converted %addr.loc10_31.1, %.loc10_31.3
// CHECK:STDOUT:   %addr.loc10_31.2: %ptr.d9e = addr_of %rhs.var
// CHECK:STDOUT:   %C__carbon_thunk.call.loc10_31: init %empty_tuple.type = call imports.%C__carbon_thunk.decl.8acdfe.2(%.loc10_31.4, %addr.loc10_31.2)
// CHECK:STDOUT:   %.loc10_31.5: init %C to %rhs.var = mark_in_place_init %C__carbon_thunk.call.loc10_31
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %addr.loc10_29.1: %ptr.d9e = addr_of %lhs.var
// CHECK:STDOUT:   %addr.loc10_29.2: %ptr.d9e = addr_of %rhs.var
// CHECK:STDOUT:   %addr.loc10_29.3: %ptr.d9e = addr_of %.loc10_29.1
// CHECK:STDOUT:   %operator_Plus__carbon_thunk.call: init %empty_tuple.type = call imports.%operator_Plus__carbon_thunk.decl(%addr.loc10_29.1, %addr.loc10_29.2, %addr.loc10_29.3)
// CHECK:STDOUT:   %.loc10_29.2: init %C to %.loc10_29.1 = mark_in_place_init %operator_Plus__carbon_thunk.call
// CHECK:STDOUT:   %.loc10_21: type = splice_block %C.ref.loc10 [concrete = constants.%C] {
// CHECK:STDOUT:     %Cpp.ref.loc10: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %C.ref.loc10: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc10_29.3: ref %C = temporary %.loc10_29.1, %.loc10_29.2
// CHECK:STDOUT:   %.loc10_29.4: %C = acquire_value %.loc10_29.3
// CHECK:STDOUT:   %c3: %C = value_binding c3, %.loc10_29.4
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %C.Op.bound.loc10: <bound method> = bound_method %.loc10_29.3, constants.%C.Op.068411.2
// CHECK:STDOUT:   %Op.ref.loc10_29: %C.cpp_destructor.type = name_ref Op, imports.%C.cpp_destructor.decl [concrete = constants.%C.cpp_destructor]
// CHECK:STDOUT:   %C.cpp_destructor.bound.loc10: <bound method> = bound_method %.loc10_29.3, %Op.ref.loc10_29
// CHECK:STDOUT:   %C.cpp_destructor.call.loc10: init %empty_tuple.type = call %C.cpp_destructor.bound.loc10(%.loc10_29.3)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
