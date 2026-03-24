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
  // CHECK:STDERR:   296 |   c + 1;
  // CHECK:STDERR:       |     ^
  // CHECK:STDERR: fail_ambiguous.carbon:[[@LINE-8]]:6: note: candidate function [CppInteropParseNote]
  // CHECK:STDERR:   281 | void operator+(C, char);
  // CHECK:STDERR:       |      ^
  // CHECK:STDERR: fail_ambiguous.carbon:[[@LINE-10]]:6: note: candidate function [CppInteropParseNote]
  // CHECK:STDERR:   282 | void operator+(C, short);
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
  // CHECK:STDERR:   320 |   c + 1;
  // CHECK:STDERR:       |     ^
  // CHECK:STDERR: fail_deleted.carbon:[[@LINE-8]]:6: note: candidate function has been explicitly deleted [CppInteropParseNote]
  // CHECK:STDERR:   305 | void operator+(C, int) = delete;
  // CHECK:STDERR:       |      ^
  // CHECK:STDERR: fail_deleted.carbon:[[@LINE-10]]:6: note: candidate function not viable: no known conversion from 'int' to 'void *' for 2nd argument [CppInteropParseNote]
  // CHECK:STDERR:   306 | void operator+(C, void*) = delete;
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
  // CHECK:STDERR:   392 |   return c < c;
  // CHECK:STDERR:       |            ^
  // CHECK:STDERR: fail_rewrite_deleted.carbon:[[@LINE-9]]:6: note: candidate function has been explicitly deleted [CppInteropParseNote]
  // CHECK:STDERR:   376 | auto operator<=>(C lhs, C rhs) -> std::strong_ordering = delete;
  // CHECK:STDERR:       |      ^
  // CHECK:STDERR: fail_rewrite_deleted.carbon:[[@LINE-11]]:6: note: candidate function [CppInteropParseNote]
  // CHECK:STDERR:   377 | auto operator<(C lhs, B rhs) -> bool;
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
  // CHECK:STDERR: ./incomplete.h:885:7: note: class was forward declared here [ClassForwardDeclaredHere]
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
  // CHECK:STDERR: ./incomplete.h:885:7: note: class was forward declared here [ClassForwardDeclaredHere]
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
  // CHECK:STDERR:   988 |   friend typename T::error operator+(B<A>, B<A>);
  // CHECK:STDERR:       |                   ^
  friend typename T::error operator+(B<A>, B<A>);
};

using X = B<A<int>>;
''';

fn F(x: Cpp.X) -> i32 {
  // CHECK:STDERR: fail_indirect_template_instantiation_error.carbon:[[@LINE+8]]:12: note: in instantiation of template class 'A<int>' requested here [CppInteropParseNote]
  // CHECK:STDERR:  1003 |   return x + x;
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
// CHECK:STDOUT:     %self.param_patt: %pattern_type.217 = ref_param_pattern [concrete]
// CHECK:STDOUT:     %self.patt: %pattern_type.217 = at_binding_pattern self, %self.param_patt [concrete]
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
// CHECK:STDOUT:   %Cpp.ref.loc40_18: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %C.ref.loc40_21: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %C.ref.loc40_23: %C.C.cpp_overload_set.type = name_ref C, imports.%C.C.cpp_overload_set.value [concrete = constants.%C.C.cpp_overload_set.value]
// CHECK:STDOUT:   %.loc40_3: ref %C = splice_block %c.var {}
// CHECK:STDOUT:   %addr.loc40: %ptr.d9e = addr_of %.loc40_3
// CHECK:STDOUT:   %C__carbon_thunk.call: init %empty_tuple.type = call imports.%C__carbon_thunk.decl(%addr.loc40)
// CHECK:STDOUT:   %.loc40_26: init %C to %.loc40_3 = mark_in_place_init %C__carbon_thunk.call
// CHECK:STDOUT:   assign %c.var, %.loc40_26
// CHECK:STDOUT:   %.loc40_13: type = splice_block %C.ref.loc40_13 [concrete = constants.%C] {
// CHECK:STDOUT:     %Cpp.ref.loc40_10: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %C.ref.loc40_13: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c: ref %C = ref_binding c, %c.var
// CHECK:STDOUT:   %c.ref.loc43: ref %C = name_ref c, %c
// CHECK:STDOUT:   %cpp_operator.call.loc43: ref %C = call imports.%cpp_operator.decl.4206c9.1(%c.ref.loc43)
// CHECK:STDOUT:   %c.ref.loc44: ref %C = name_ref c, %c
// CHECK:STDOUT:   %cpp_operator.call.loc44: ref %C = call imports.%cpp_operator.decl.4206c9.2(%c.ref.loc44)
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %minus.patt: %pattern_type.217 = value_binding_pattern minus [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c.ref.loc47: ref %C = name_ref c, %c
// CHECK:STDOUT:   %.loc47_29.1: ref %C = temporary_storage
// CHECK:STDOUT:   %.loc47_30.1: %C = acquire_value %c.ref.loc47
// CHECK:STDOUT:   %.loc47_30.2: ref %C = value_as_ref %.loc47_30.1
// CHECK:STDOUT:   %addr.loc47_29.1: %ptr.d9e = addr_of %.loc47_30.2
// CHECK:STDOUT:   %addr.loc47_29.2: %ptr.d9e = addr_of %.loc47_29.1
// CHECK:STDOUT:   %operator_Minus__carbon_thunk.call: init %empty_tuple.type = call imports.%operator_Minus__carbon_thunk.decl(%addr.loc47_29.1, %addr.loc47_29.2)
// CHECK:STDOUT:   %.loc47_29.2: init %C to %.loc47_29.1 = mark_in_place_init %operator_Minus__carbon_thunk.call
// CHECK:STDOUT:   %.loc47_24: type = splice_block %C.ref.loc47 [concrete = constants.%C] {
// CHECK:STDOUT:     %Cpp.ref.loc47: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %C.ref.loc47: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc47_29.3: ref %C = temporary %.loc47_29.1, %.loc47_29.2
// CHECK:STDOUT:   %.loc47_29.4: %C = acquire_value %.loc47_29.3
// CHECK:STDOUT:   %minus: %C = value_binding minus, %.loc47_29.4
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %complement.patt: %pattern_type.217 = value_binding_pattern complement [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c.ref.loc50: ref %C = name_ref c, %c
// CHECK:STDOUT:   %.loc50_34.1: ref %C = temporary_storage
// CHECK:STDOUT:   %.loc50_35.1: %C = acquire_value %c.ref.loc50
// CHECK:STDOUT:   %.loc50_35.2: ref %C = value_as_ref %.loc50_35.1
// CHECK:STDOUT:   %addr.loc50_34.1: %ptr.d9e = addr_of %.loc50_35.2
// CHECK:STDOUT:   %addr.loc50_34.2: %ptr.d9e = addr_of %.loc50_34.1
// CHECK:STDOUT:   %operator_Tilde__carbon_thunk.call: init %empty_tuple.type = call imports.%operator_Tilde__carbon_thunk.decl(%addr.loc50_34.1, %addr.loc50_34.2)
// CHECK:STDOUT:   %.loc50_34.2: init %C to %.loc50_34.1 = mark_in_place_init %operator_Tilde__carbon_thunk.call
// CHECK:STDOUT:   %.loc50_29: type = splice_block %C.ref.loc50 [concrete = constants.%C] {
// CHECK:STDOUT:     %Cpp.ref.loc50: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %C.ref.loc50: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc50_34.3: ref %C = temporary %.loc50_34.1, %.loc50_34.2
// CHECK:STDOUT:   %.loc50_34.4: %C = acquire_value %.loc50_34.3
// CHECK:STDOUT:   %complement: %C = value_binding complement, %.loc50_34.4
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %C.Op.bound.loc50: <bound method> = bound_method %.loc50_34.3, constants.%C.Op
// CHECK:STDOUT:   %Op.ref.loc50: %C.cpp_destructor.type = name_ref Op, imports.%C.cpp_destructor.decl [concrete = constants.%C.cpp_destructor]
// CHECK:STDOUT:   %C.cpp_destructor.bound.loc50: <bound method> = bound_method %.loc50_34.3, %Op.ref.loc50
// CHECK:STDOUT:   %C.cpp_destructor.call.loc50: init %empty_tuple.type = call %C.cpp_destructor.bound.loc50(%.loc50_34.3)
// CHECK:STDOUT:   %C.Op.bound.loc47: <bound method> = bound_method %.loc47_29.3, constants.%C.Op
// CHECK:STDOUT:   %Op.ref.loc47: %C.cpp_destructor.type = name_ref Op, imports.%C.cpp_destructor.decl [concrete = constants.%C.cpp_destructor]
// CHECK:STDOUT:   %C.cpp_destructor.bound.loc47: <bound method> = bound_method %.loc47_29.3, %Op.ref.loc47
// CHECK:STDOUT:   %C.cpp_destructor.call.loc47: init %empty_tuple.type = call %C.cpp_destructor.bound.loc47(%.loc47_29.3)
// CHECK:STDOUT:   %C.Op.bound.loc40: <bound method> = bound_method %c.var, constants.%C.Op
// CHECK:STDOUT:   %Op.ref.loc40: %C.cpp_destructor.type = name_ref Op, imports.%C.cpp_destructor.decl [concrete = constants.%C.cpp_destructor]
// CHECK:STDOUT:   %C.cpp_destructor.bound.loc40: <bound method> = bound_method %c.var, %Op.ref.loc40
// CHECK:STDOUT:   %C.cpp_destructor.call.loc40: init %empty_tuple.type = call %C.cpp_destructor.bound.loc40(%c.var)
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
// CHECK:STDOUT:     %self.param_patt: %pattern_type.217 = ref_param_pattern [concrete]
// CHECK:STDOUT:     %self.patt: %pattern_type.217 = at_binding_pattern self, %self.param_patt [concrete]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %self.param: ref %C = ref_param call_param0
// CHECK:STDOUT:     %self: ref %C = ref_binding self, %self.param
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.cpp_destructor.decl: %C.cpp_destructor.type = fn_decl @C.cpp_destructor [concrete = constants.%C.cpp_destructor] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.217 = ref_param_pattern [concrete]
// CHECK:STDOUT:     %self.patt: %pattern_type.217 = at_binding_pattern self, %self.param_patt [concrete]
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
// CHECK:STDOUT:   %Cpp.ref.loc170_19: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %C.ref.loc170_22: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %C.ref.loc170_24: %C.C.cpp_overload_set.type = name_ref C, imports.%C.C.cpp_overload_set.value [concrete = constants.%C.C.cpp_overload_set.value]
// CHECK:STDOUT:   %.loc170_3: ref %C = splice_block %c1.var {}
// CHECK:STDOUT:   %addr.loc170: %ptr.d9e = addr_of %.loc170_3
// CHECK:STDOUT:   %C__carbon_thunk.call.loc170: init %empty_tuple.type = call imports.%C__carbon_thunk.decl(%addr.loc170)
// CHECK:STDOUT:   %.loc170_27: init %C to %.loc170_3 = mark_in_place_init %C__carbon_thunk.call.loc170
// CHECK:STDOUT:   assign %c1.var, %.loc170_27
// CHECK:STDOUT:   %.loc170_14: type = splice_block %C.ref.loc170_14 [concrete = constants.%C] {
// CHECK:STDOUT:     %Cpp.ref.loc170_11: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %C.ref.loc170_14: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c1: ref %C = ref_binding c1, %c1.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c2.patt: %pattern_type.217 = ref_binding_pattern c2 [concrete]
// CHECK:STDOUT:     %c2.var_patt: %pattern_type.217 = var_pattern %c2.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c2.var: ref %C = var %c2.var_patt
// CHECK:STDOUT:   %Cpp.ref.loc171_19: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %C.ref.loc171_22: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %C.ref.loc171_24: %C.C.cpp_overload_set.type = name_ref C, imports.%C.C.cpp_overload_set.value [concrete = constants.%C.C.cpp_overload_set.value]
// CHECK:STDOUT:   %.loc171_3: ref %C = splice_block %c2.var {}
// CHECK:STDOUT:   %addr.loc171: %ptr.d9e = addr_of %.loc171_3
// CHECK:STDOUT:   %C__carbon_thunk.call.loc171: init %empty_tuple.type = call imports.%C__carbon_thunk.decl(%addr.loc171)
// CHECK:STDOUT:   %.loc171_27: init %C to %.loc171_3 = mark_in_place_init %C__carbon_thunk.call.loc171
// CHECK:STDOUT:   assign %c2.var, %.loc171_27
// CHECK:STDOUT:   %.loc171_14: type = splice_block %C.ref.loc171_14 [concrete = constants.%C] {
// CHECK:STDOUT:     %Cpp.ref.loc171_11: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %C.ref.loc171_14: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c2: ref %C = ref_binding c2, %c2.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %addition.patt: %pattern_type.217 = value_binding_pattern addition [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c1.ref.loc174: ref %C = name_ref c1, %c1
// CHECK:STDOUT:   %c2.ref.loc174: ref %C = name_ref c2, %c2
// CHECK:STDOUT:   %.loc174_35.1: ref %C = temporary_storage
// CHECK:STDOUT:   %.loc174_32.1: %C = acquire_value %c1.ref.loc174
// CHECK:STDOUT:   %.loc174_37.1: %C = acquire_value %c2.ref.loc174
// CHECK:STDOUT:   %.loc174_32.2: ref %C = value_as_ref %.loc174_32.1
// CHECK:STDOUT:   %addr.loc174_35.1: %ptr.d9e = addr_of %.loc174_32.2
// CHECK:STDOUT:   %.loc174_37.2: ref %C = value_as_ref %.loc174_37.1
// CHECK:STDOUT:   %addr.loc174_35.2: %ptr.d9e = addr_of %.loc174_37.2
// CHECK:STDOUT:   %addr.loc174_35.3: %ptr.d9e = addr_of %.loc174_35.1
// CHECK:STDOUT:   %operator_Plus__carbon_thunk.call: init %empty_tuple.type = call imports.%operator_Plus__carbon_thunk.decl(%addr.loc174_35.1, %addr.loc174_35.2, %addr.loc174_35.3)
// CHECK:STDOUT:   %.loc174_35.2: init %C to %.loc174_35.1 = mark_in_place_init %operator_Plus__carbon_thunk.call
// CHECK:STDOUT:   %.loc174_27: type = splice_block %C.ref.loc174 [concrete = constants.%C] {
// CHECK:STDOUT:     %Cpp.ref.loc174: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %C.ref.loc174: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc174_35.3: ref %C = temporary %.loc174_35.1, %.loc174_35.2
// CHECK:STDOUT:   %.loc174_35.4: %C = acquire_value %.loc174_35.3
// CHECK:STDOUT:   %addition: %C = value_binding addition, %.loc174_35.4
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %subtraction.patt: %pattern_type.217 = value_binding_pattern subtraction [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c1.ref.loc175: ref %C = name_ref c1, %c1
// CHECK:STDOUT:   %c2.ref.loc175: ref %C = name_ref c2, %c2
// CHECK:STDOUT:   %.loc175_38.1: ref %C = temporary_storage
// CHECK:STDOUT:   %.loc175_35.1: %C = acquire_value %c1.ref.loc175
// CHECK:STDOUT:   %.loc175_40.1: %C = acquire_value %c2.ref.loc175
// CHECK:STDOUT:   %.loc175_35.2: ref %C = value_as_ref %.loc175_35.1
// CHECK:STDOUT:   %addr.loc175_38.1: %ptr.d9e = addr_of %.loc175_35.2
// CHECK:STDOUT:   %.loc175_40.2: ref %C = value_as_ref %.loc175_40.1
// CHECK:STDOUT:   %addr.loc175_38.2: %ptr.d9e = addr_of %.loc175_40.2
// CHECK:STDOUT:   %addr.loc175_38.3: %ptr.d9e = addr_of %.loc175_38.1
// CHECK:STDOUT:   %operator_Minus__carbon_thunk.call: init %empty_tuple.type = call imports.%operator_Minus__carbon_thunk.decl(%addr.loc175_38.1, %addr.loc175_38.2, %addr.loc175_38.3)
// CHECK:STDOUT:   %.loc175_38.2: init %C to %.loc175_38.1 = mark_in_place_init %operator_Minus__carbon_thunk.call
// CHECK:STDOUT:   %.loc175_30: type = splice_block %C.ref.loc175 [concrete = constants.%C] {
// CHECK:STDOUT:     %Cpp.ref.loc175: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %C.ref.loc175: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc175_38.3: ref %C = temporary %.loc175_38.1, %.loc175_38.2
// CHECK:STDOUT:   %.loc175_38.4: %C = acquire_value %.loc175_38.3
// CHECK:STDOUT:   %subtraction: %C = value_binding subtraction, %.loc175_38.4
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %multiplication.patt: %pattern_type.217 = value_binding_pattern multiplication [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c1.ref.loc176: ref %C = name_ref c1, %c1
// CHECK:STDOUT:   %c2.ref.loc176: ref %C = name_ref c2, %c2
// CHECK:STDOUT:   %.loc176_41.1: ref %C = temporary_storage
// CHECK:STDOUT:   %.loc176_38.1: %C = acquire_value %c1.ref.loc176
// CHECK:STDOUT:   %.loc176_43.1: %C = acquire_value %c2.ref.loc176
// CHECK:STDOUT:   %.loc176_38.2: ref %C = value_as_ref %.loc176_38.1
// CHECK:STDOUT:   %addr.loc176_41.1: %ptr.d9e = addr_of %.loc176_38.2
// CHECK:STDOUT:   %.loc176_43.2: ref %C = value_as_ref %.loc176_43.1
// CHECK:STDOUT:   %addr.loc176_41.2: %ptr.d9e = addr_of %.loc176_43.2
// CHECK:STDOUT:   %addr.loc176_41.3: %ptr.d9e = addr_of %.loc176_41.1
// CHECK:STDOUT:   %operator_Star__carbon_thunk.call: init %empty_tuple.type = call imports.%operator_Star__carbon_thunk.decl(%addr.loc176_41.1, %addr.loc176_41.2, %addr.loc176_41.3)
// CHECK:STDOUT:   %.loc176_41.2: init %C to %.loc176_41.1 = mark_in_place_init %operator_Star__carbon_thunk.call
// CHECK:STDOUT:   %.loc176_33: type = splice_block %C.ref.loc176 [concrete = constants.%C] {
// CHECK:STDOUT:     %Cpp.ref.loc176: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %C.ref.loc176: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc176_41.3: ref %C = temporary %.loc176_41.1, %.loc176_41.2
// CHECK:STDOUT:   %.loc176_41.4: %C = acquire_value %.loc176_41.3
// CHECK:STDOUT:   %multiplication: %C = value_binding multiplication, %.loc176_41.4
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %division.patt: %pattern_type.217 = value_binding_pattern division [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c1.ref.loc177: ref %C = name_ref c1, %c1
// CHECK:STDOUT:   %c2.ref.loc177: ref %C = name_ref c2, %c2
// CHECK:STDOUT:   %.loc177_35.1: ref %C = temporary_storage
// CHECK:STDOUT:   %.loc177_32.1: %C = acquire_value %c1.ref.loc177
// CHECK:STDOUT:   %.loc177_37.1: %C = acquire_value %c2.ref.loc177
// CHECK:STDOUT:   %.loc177_32.2: ref %C = value_as_ref %.loc177_32.1
// CHECK:STDOUT:   %addr.loc177_35.1: %ptr.d9e = addr_of %.loc177_32.2
// CHECK:STDOUT:   %.loc177_37.2: ref %C = value_as_ref %.loc177_37.1
// CHECK:STDOUT:   %addr.loc177_35.2: %ptr.d9e = addr_of %.loc177_37.2
// CHECK:STDOUT:   %addr.loc177_35.3: %ptr.d9e = addr_of %.loc177_35.1
// CHECK:STDOUT:   %operator_Slash__carbon_thunk.call: init %empty_tuple.type = call imports.%operator_Slash__carbon_thunk.decl(%addr.loc177_35.1, %addr.loc177_35.2, %addr.loc177_35.3)
// CHECK:STDOUT:   %.loc177_35.2: init %C to %.loc177_35.1 = mark_in_place_init %operator_Slash__carbon_thunk.call
// CHECK:STDOUT:   %.loc177_27: type = splice_block %C.ref.loc177 [concrete = constants.%C] {
// CHECK:STDOUT:     %Cpp.ref.loc177: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %C.ref.loc177: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc177_35.3: ref %C = temporary %.loc177_35.1, %.loc177_35.2
// CHECK:STDOUT:   %.loc177_35.4: %C = acquire_value %.loc177_35.3
// CHECK:STDOUT:   %division: %C = value_binding division, %.loc177_35.4
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %modulo.patt: %pattern_type.217 = value_binding_pattern modulo [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c1.ref.loc178: ref %C = name_ref c1, %c1
// CHECK:STDOUT:   %c2.ref.loc178: ref %C = name_ref c2, %c2
// CHECK:STDOUT:   %.loc178_33.1: ref %C = temporary_storage
// CHECK:STDOUT:   %.loc178_30.1: %C = acquire_value %c1.ref.loc178
// CHECK:STDOUT:   %.loc178_35.1: %C = acquire_value %c2.ref.loc178
// CHECK:STDOUT:   %.loc178_30.2: ref %C = value_as_ref %.loc178_30.1
// CHECK:STDOUT:   %addr.loc178_33.1: %ptr.d9e = addr_of %.loc178_30.2
// CHECK:STDOUT:   %.loc178_35.2: ref %C = value_as_ref %.loc178_35.1
// CHECK:STDOUT:   %addr.loc178_33.2: %ptr.d9e = addr_of %.loc178_35.2
// CHECK:STDOUT:   %addr.loc178_33.3: %ptr.d9e = addr_of %.loc178_33.1
// CHECK:STDOUT:   %operator_Percent__carbon_thunk.call: init %empty_tuple.type = call imports.%operator_Percent__carbon_thunk.decl(%addr.loc178_33.1, %addr.loc178_33.2, %addr.loc178_33.3)
// CHECK:STDOUT:   %.loc178_33.2: init %C to %.loc178_33.1 = mark_in_place_init %operator_Percent__carbon_thunk.call
// CHECK:STDOUT:   %.loc178_25: type = splice_block %C.ref.loc178 [concrete = constants.%C] {
// CHECK:STDOUT:     %Cpp.ref.loc178: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %C.ref.loc178: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc178_33.3: ref %C = temporary %.loc178_33.1, %.loc178_33.2
// CHECK:STDOUT:   %.loc178_33.4: %C = acquire_value %.loc178_33.3
// CHECK:STDOUT:   %modulo: %C = value_binding modulo, %.loc178_33.4
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %bitwise_and.patt: %pattern_type.217 = value_binding_pattern bitwise_and [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c1.ref.loc181: ref %C = name_ref c1, %c1
// CHECK:STDOUT:   %c2.ref.loc181: ref %C = name_ref c2, %c2
// CHECK:STDOUT:   %.loc181_38.1: ref %C = temporary_storage
// CHECK:STDOUT:   %.loc181_35.1: %C = acquire_value %c1.ref.loc181
// CHECK:STDOUT:   %.loc181_40.1: %C = acquire_value %c2.ref.loc181
// CHECK:STDOUT:   %.loc181_35.2: ref %C = value_as_ref %.loc181_35.1
// CHECK:STDOUT:   %addr.loc181_38.1: %ptr.d9e = addr_of %.loc181_35.2
// CHECK:STDOUT:   %.loc181_40.2: ref %C = value_as_ref %.loc181_40.1
// CHECK:STDOUT:   %addr.loc181_38.2: %ptr.d9e = addr_of %.loc181_40.2
// CHECK:STDOUT:   %addr.loc181_38.3: %ptr.d9e = addr_of %.loc181_38.1
// CHECK:STDOUT:   %operator_Amp__carbon_thunk.call: init %empty_tuple.type = call imports.%operator_Amp__carbon_thunk.decl(%addr.loc181_38.1, %addr.loc181_38.2, %addr.loc181_38.3)
// CHECK:STDOUT:   %.loc181_38.2: init %C to %.loc181_38.1 = mark_in_place_init %operator_Amp__carbon_thunk.call
// CHECK:STDOUT:   %.loc181_30: type = splice_block %C.ref.loc181 [concrete = constants.%C] {
// CHECK:STDOUT:     %Cpp.ref.loc181: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %C.ref.loc181: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc181_38.3: ref %C = temporary %.loc181_38.1, %.loc181_38.2
// CHECK:STDOUT:   %.loc181_38.4: %C = acquire_value %.loc181_38.3
// CHECK:STDOUT:   %bitwise_and: %C = value_binding bitwise_and, %.loc181_38.4
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %bitwise_or.patt: %pattern_type.217 = value_binding_pattern bitwise_or [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c1.ref.loc182: ref %C = name_ref c1, %c1
// CHECK:STDOUT:   %c2.ref.loc182: ref %C = name_ref c2, %c2
// CHECK:STDOUT:   %.loc182_37.1: ref %C = temporary_storage
// CHECK:STDOUT:   %.loc182_34.1: %C = acquire_value %c1.ref.loc182
// CHECK:STDOUT:   %.loc182_39.1: %C = acquire_value %c2.ref.loc182
// CHECK:STDOUT:   %.loc182_34.2: ref %C = value_as_ref %.loc182_34.1
// CHECK:STDOUT:   %addr.loc182_37.1: %ptr.d9e = addr_of %.loc182_34.2
// CHECK:STDOUT:   %.loc182_39.2: ref %C = value_as_ref %.loc182_39.1
// CHECK:STDOUT:   %addr.loc182_37.2: %ptr.d9e = addr_of %.loc182_39.2
// CHECK:STDOUT:   %addr.loc182_37.3: %ptr.d9e = addr_of %.loc182_37.1
// CHECK:STDOUT:   %operator_Pipe__carbon_thunk.call: init %empty_tuple.type = call imports.%operator_Pipe__carbon_thunk.decl(%addr.loc182_37.1, %addr.loc182_37.2, %addr.loc182_37.3)
// CHECK:STDOUT:   %.loc182_37.2: init %C to %.loc182_37.1 = mark_in_place_init %operator_Pipe__carbon_thunk.call
// CHECK:STDOUT:   %.loc182_29: type = splice_block %C.ref.loc182 [concrete = constants.%C] {
// CHECK:STDOUT:     %Cpp.ref.loc182: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %C.ref.loc182: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc182_37.3: ref %C = temporary %.loc182_37.1, %.loc182_37.2
// CHECK:STDOUT:   %.loc182_37.4: %C = acquire_value %.loc182_37.3
// CHECK:STDOUT:   %bitwise_or: %C = value_binding bitwise_or, %.loc182_37.4
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %bitwise_xor.patt: %pattern_type.217 = value_binding_pattern bitwise_xor [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c1.ref.loc183: ref %C = name_ref c1, %c1
// CHECK:STDOUT:   %c2.ref.loc183: ref %C = name_ref c2, %c2
// CHECK:STDOUT:   %.loc183_38.1: ref %C = temporary_storage
// CHECK:STDOUT:   %.loc183_35.1: %C = acquire_value %c1.ref.loc183
// CHECK:STDOUT:   %.loc183_40.1: %C = acquire_value %c2.ref.loc183
// CHECK:STDOUT:   %.loc183_35.2: ref %C = value_as_ref %.loc183_35.1
// CHECK:STDOUT:   %addr.loc183_38.1: %ptr.d9e = addr_of %.loc183_35.2
// CHECK:STDOUT:   %.loc183_40.2: ref %C = value_as_ref %.loc183_40.1
// CHECK:STDOUT:   %addr.loc183_38.2: %ptr.d9e = addr_of %.loc183_40.2
// CHECK:STDOUT:   %addr.loc183_38.3: %ptr.d9e = addr_of %.loc183_38.1
// CHECK:STDOUT:   %operator_Caret__carbon_thunk.call: init %empty_tuple.type = call imports.%operator_Caret__carbon_thunk.decl(%addr.loc183_38.1, %addr.loc183_38.2, %addr.loc183_38.3)
// CHECK:STDOUT:   %.loc183_38.2: init %C to %.loc183_38.1 = mark_in_place_init %operator_Caret__carbon_thunk.call
// CHECK:STDOUT:   %.loc183_30: type = splice_block %C.ref.loc183 [concrete = constants.%C] {
// CHECK:STDOUT:     %Cpp.ref.loc183: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %C.ref.loc183: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc183_38.3: ref %C = temporary %.loc183_38.1, %.loc183_38.2
// CHECK:STDOUT:   %.loc183_38.4: %C = acquire_value %.loc183_38.3
// CHECK:STDOUT:   %bitwise_xor: %C = value_binding bitwise_xor, %.loc183_38.4
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %left_shift.patt: %pattern_type.217 = value_binding_pattern left_shift [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c1.ref.loc184: ref %C = name_ref c1, %c1
// CHECK:STDOUT:   %int_3.loc184: Core.IntLiteral = int_value 3 [concrete = constants.%int_3.1ba]
// CHECK:STDOUT:   %.loc184_37.1: ref %C = temporary_storage
// CHECK:STDOUT:   %.loc184_34.1: %C = acquire_value %c1.ref.loc184
// CHECK:STDOUT:   %impl.elem0.loc184: %.545 = impl_witness_access constants.%ImplicitAs.impl_witness.6bc, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5]
// CHECK:STDOUT:   %bound_method.loc184_40.1: <bound method> = bound_method %int_3.loc184, %impl.elem0.loc184 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.061]
// CHECK:STDOUT:   %specific_fn.loc184: <specific function> = specific_function %impl.elem0.loc184, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc184_40.2: <bound method> = bound_method %int_3.loc184, %specific_fn.loc184 [concrete = constants.%bound_method.fa7]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc184: init %i32 = call %bound_method.loc184_40.2(%int_3.loc184) [concrete = constants.%int_3.822]
// CHECK:STDOUT:   %.loc184_40.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc184 [concrete = constants.%int_3.822]
// CHECK:STDOUT:   %.loc184_40.2: %i32 = converted %int_3.loc184, %.loc184_40.1 [concrete = constants.%int_3.822]
// CHECK:STDOUT:   %.loc184_34.2: ref %C = value_as_ref %.loc184_34.1
// CHECK:STDOUT:   %addr.loc184_37.1: %ptr.d9e = addr_of %.loc184_34.2
// CHECK:STDOUT:   %addr.loc184_37.2: %ptr.d9e = addr_of %.loc184_37.1
// CHECK:STDOUT:   %operator_LessLess__carbon_thunk.call: init %empty_tuple.type = call imports.%operator_LessLess__carbon_thunk.decl(%addr.loc184_37.1, %.loc184_40.2, %addr.loc184_37.2)
// CHECK:STDOUT:   %.loc184_37.2: init %C to %.loc184_37.1 = mark_in_place_init %operator_LessLess__carbon_thunk.call
// CHECK:STDOUT:   %.loc184_29: type = splice_block %C.ref.loc184 [concrete = constants.%C] {
// CHECK:STDOUT:     %Cpp.ref.loc184: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %C.ref.loc184: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc184_37.3: ref %C = temporary %.loc184_37.1, %.loc184_37.2
// CHECK:STDOUT:   %.loc184_37.4: %C = acquire_value %.loc184_37.3
// CHECK:STDOUT:   %left_shift: %C = value_binding left_shift, %.loc184_37.4
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %right_shift.patt: %pattern_type.217 = value_binding_pattern right_shift [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c1.ref.loc185: ref %C = name_ref c1, %c1
// CHECK:STDOUT:   %int_5.loc185: Core.IntLiteral = int_value 5 [concrete = constants.%int_5.64b]
// CHECK:STDOUT:   %.loc185_38.1: ref %C = temporary_storage
// CHECK:STDOUT:   %.loc185_35.1: %C = acquire_value %c1.ref.loc185
// CHECK:STDOUT:   %impl.elem0.loc185: %.545 = impl_witness_access constants.%ImplicitAs.impl_witness.6bc, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5]
// CHECK:STDOUT:   %bound_method.loc185_41.1: <bound method> = bound_method %int_5.loc185, %impl.elem0.loc185 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.005]
// CHECK:STDOUT:   %specific_fn.loc185: <specific function> = specific_function %impl.elem0.loc185, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc185_41.2: <bound method> = bound_method %int_5.loc185, %specific_fn.loc185 [concrete = constants.%bound_method.e9d]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc185: init %i32 = call %bound_method.loc185_41.2(%int_5.loc185) [concrete = constants.%int_5.0f6]
// CHECK:STDOUT:   %.loc185_41.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc185 [concrete = constants.%int_5.0f6]
// CHECK:STDOUT:   %.loc185_41.2: %i32 = converted %int_5.loc185, %.loc185_41.1 [concrete = constants.%int_5.0f6]
// CHECK:STDOUT:   %.loc185_35.2: ref %C = value_as_ref %.loc185_35.1
// CHECK:STDOUT:   %addr.loc185_38.1: %ptr.d9e = addr_of %.loc185_35.2
// CHECK:STDOUT:   %addr.loc185_38.2: %ptr.d9e = addr_of %.loc185_38.1
// CHECK:STDOUT:   %operator_GreaterGreater__carbon_thunk.call: init %empty_tuple.type = call imports.%operator_GreaterGreater__carbon_thunk.decl(%addr.loc185_38.1, %.loc185_41.2, %addr.loc185_38.2)
// CHECK:STDOUT:   %.loc185_38.2: init %C to %.loc185_38.1 = mark_in_place_init %operator_GreaterGreater__carbon_thunk.call
// CHECK:STDOUT:   %.loc185_30: type = splice_block %C.ref.loc185 [concrete = constants.%C] {
// CHECK:STDOUT:     %Cpp.ref.loc185: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %C.ref.loc185: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc185_38.3: ref %C = temporary %.loc185_38.1, %.loc185_38.2
// CHECK:STDOUT:   %.loc185_38.4: %C = acquire_value %.loc185_38.3
// CHECK:STDOUT:   %right_shift: %C = value_binding right_shift, %.loc185_38.4
// CHECK:STDOUT:   %c1.ref.loc188: ref %C = name_ref c1, %c1
// CHECK:STDOUT:   %c2.ref.loc188: ref %C = name_ref c2, %c2
// CHECK:STDOUT:   %.loc188_9.1: %C = acquire_value %c2.ref.loc188
// CHECK:STDOUT:   %.loc188_9.2: ref %C = value_as_ref %.loc188_9.1
// CHECK:STDOUT:   %addr.loc188: %ptr.d9e = addr_of %.loc188_9.2
// CHECK:STDOUT:   %operator_PlusEqual__carbon_thunk.call: ref %C = call imports.%operator_PlusEqual__carbon_thunk.decl(%c1.ref.loc188, %addr.loc188)
// CHECK:STDOUT:   %c1.ref.loc189: ref %C = name_ref c1, %c1
// CHECK:STDOUT:   %c2.ref.loc189: ref %C = name_ref c2, %c2
// CHECK:STDOUT:   %.loc189_9.1: %C = acquire_value %c2.ref.loc189
// CHECK:STDOUT:   %.loc189_9.2: ref %C = value_as_ref %.loc189_9.1
// CHECK:STDOUT:   %addr.loc189: %ptr.d9e = addr_of %.loc189_9.2
// CHECK:STDOUT:   %operator_MinusEqual__carbon_thunk.call: ref %C = call imports.%operator_MinusEqual__carbon_thunk.decl(%c1.ref.loc189, %addr.loc189)
// CHECK:STDOUT:   %c1.ref.loc190: ref %C = name_ref c1, %c1
// CHECK:STDOUT:   %c2.ref.loc190: ref %C = name_ref c2, %c2
// CHECK:STDOUT:   %.loc190_9.1: %C = acquire_value %c2.ref.loc190
// CHECK:STDOUT:   %.loc190_9.2: ref %C = value_as_ref %.loc190_9.1
// CHECK:STDOUT:   %addr.loc190: %ptr.d9e = addr_of %.loc190_9.2
// CHECK:STDOUT:   %operator_StarEqual__carbon_thunk.call: ref %C = call imports.%operator_StarEqual__carbon_thunk.decl(%c1.ref.loc190, %addr.loc190)
// CHECK:STDOUT:   %c1.ref.loc191: ref %C = name_ref c1, %c1
// CHECK:STDOUT:   %c2.ref.loc191: ref %C = name_ref c2, %c2
// CHECK:STDOUT:   %.loc191_9.1: %C = acquire_value %c2.ref.loc191
// CHECK:STDOUT:   %.loc191_9.2: ref %C = value_as_ref %.loc191_9.1
// CHECK:STDOUT:   %addr.loc191: %ptr.d9e = addr_of %.loc191_9.2
// CHECK:STDOUT:   %operator_SlashEqual__carbon_thunk.call: ref %C = call imports.%operator_SlashEqual__carbon_thunk.decl(%c1.ref.loc191, %addr.loc191)
// CHECK:STDOUT:   %c1.ref.loc192: ref %C = name_ref c1, %c1
// CHECK:STDOUT:   %c2.ref.loc192: ref %C = name_ref c2, %c2
// CHECK:STDOUT:   %.loc192_9.1: %C = acquire_value %c2.ref.loc192
// CHECK:STDOUT:   %.loc192_9.2: ref %C = value_as_ref %.loc192_9.1
// CHECK:STDOUT:   %addr.loc192: %ptr.d9e = addr_of %.loc192_9.2
// CHECK:STDOUT:   %operator_PercentEqual__carbon_thunk.call: ref %C = call imports.%operator_PercentEqual__carbon_thunk.decl(%c1.ref.loc192, %addr.loc192)
// CHECK:STDOUT:   %c1.ref.loc195: ref %C = name_ref c1, %c1
// CHECK:STDOUT:   %c2.ref.loc195: ref %C = name_ref c2, %c2
// CHECK:STDOUT:   %.loc195_9.1: %C = acquire_value %c2.ref.loc195
// CHECK:STDOUT:   %.loc195_9.2: ref %C = value_as_ref %.loc195_9.1
// CHECK:STDOUT:   %addr.loc195: %ptr.d9e = addr_of %.loc195_9.2
// CHECK:STDOUT:   %operator_AmpEqual__carbon_thunk.call: ref %C = call imports.%operator_AmpEqual__carbon_thunk.decl(%c1.ref.loc195, %addr.loc195)
// CHECK:STDOUT:   %c1.ref.loc196: ref %C = name_ref c1, %c1
// CHECK:STDOUT:   %c2.ref.loc196: ref %C = name_ref c2, %c2
// CHECK:STDOUT:   %.loc196_9.1: %C = acquire_value %c2.ref.loc196
// CHECK:STDOUT:   %.loc196_9.2: ref %C = value_as_ref %.loc196_9.1
// CHECK:STDOUT:   %addr.loc196: %ptr.d9e = addr_of %.loc196_9.2
// CHECK:STDOUT:   %operator_PipeEqual__carbon_thunk.call: ref %C = call imports.%operator_PipeEqual__carbon_thunk.decl(%c1.ref.loc196, %addr.loc196)
// CHECK:STDOUT:   %c1.ref.loc197: ref %C = name_ref c1, %c1
// CHECK:STDOUT:   %c2.ref.loc197: ref %C = name_ref c2, %c2
// CHECK:STDOUT:   %.loc197_9.1: %C = acquire_value %c2.ref.loc197
// CHECK:STDOUT:   %.loc197_9.2: ref %C = value_as_ref %.loc197_9.1
// CHECK:STDOUT:   %addr.loc197: %ptr.d9e = addr_of %.loc197_9.2
// CHECK:STDOUT:   %operator_CaretEqual__carbon_thunk.call: ref %C = call imports.%operator_CaretEqual__carbon_thunk.decl(%c1.ref.loc197, %addr.loc197)
// CHECK:STDOUT:   %c1.ref.loc198: ref %C = name_ref c1, %c1
// CHECK:STDOUT:   %int_3.loc198: Core.IntLiteral = int_value 3 [concrete = constants.%int_3.1ba]
// CHECK:STDOUT:   %impl.elem0.loc198: %.545 = impl_witness_access constants.%ImplicitAs.impl_witness.6bc, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5]
// CHECK:STDOUT:   %bound_method.loc198_10.1: <bound method> = bound_method %int_3.loc198, %impl.elem0.loc198 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.061]
// CHECK:STDOUT:   %specific_fn.loc198: <specific function> = specific_function %impl.elem0.loc198, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc198_10.2: <bound method> = bound_method %int_3.loc198, %specific_fn.loc198 [concrete = constants.%bound_method.fa7]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc198: init %i32 = call %bound_method.loc198_10.2(%int_3.loc198) [concrete = constants.%int_3.822]
// CHECK:STDOUT:   %.loc198_10.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc198 [concrete = constants.%int_3.822]
// CHECK:STDOUT:   %.loc198_10.2: %i32 = converted %int_3.loc198, %.loc198_10.1 [concrete = constants.%int_3.822]
// CHECK:STDOUT:   %cpp_operator.call.loc198: ref %C = call imports.%cpp_operator.decl.4206c9.19(%c1.ref.loc198, %.loc198_10.2)
// CHECK:STDOUT:   %c1.ref.loc199: ref %C = name_ref c1, %c1
// CHECK:STDOUT:   %int_5.loc199: Core.IntLiteral = int_value 5 [concrete = constants.%int_5.64b]
// CHECK:STDOUT:   %impl.elem0.loc199: %.545 = impl_witness_access constants.%ImplicitAs.impl_witness.6bc, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5]
// CHECK:STDOUT:   %bound_method.loc199_10.1: <bound method> = bound_method %int_5.loc199, %impl.elem0.loc199 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.005]
// CHECK:STDOUT:   %specific_fn.loc199: <specific function> = specific_function %impl.elem0.loc199, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc199_10.2: <bound method> = bound_method %int_5.loc199, %specific_fn.loc199 [concrete = constants.%bound_method.e9d]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc199: init %i32 = call %bound_method.loc199_10.2(%int_5.loc199) [concrete = constants.%int_5.0f6]
// CHECK:STDOUT:   %.loc199_10.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc199 [concrete = constants.%int_5.0f6]
// CHECK:STDOUT:   %.loc199_10.2: %i32 = converted %int_5.loc199, %.loc199_10.1 [concrete = constants.%int_5.0f6]
// CHECK:STDOUT:   %cpp_operator.call.loc199: ref %C = call imports.%cpp_operator.decl.4206c9.20(%c1.ref.loc199, %.loc199_10.2)
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %equal.patt: %pattern_type.831 = value_binding_pattern equal [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c1.ref.loc202: ref %C = name_ref c1, %c1
// CHECK:STDOUT:   %c2.ref.loc202: ref %C = name_ref c2, %c2
// CHECK:STDOUT:   %.loc202_28.1: %C = acquire_value %c1.ref.loc202
// CHECK:STDOUT:   %.loc202_34.1: %C = acquire_value %c2.ref.loc202
// CHECK:STDOUT:   %.loc202_28.2: ref %C = value_as_ref %.loc202_28.1
// CHECK:STDOUT:   %addr.loc202_31.1: %ptr.d9e = addr_of %.loc202_28.2
// CHECK:STDOUT:   %.loc202_34.2: ref %C = value_as_ref %.loc202_34.1
// CHECK:STDOUT:   %addr.loc202_31.2: %ptr.d9e = addr_of %.loc202_34.2
// CHECK:STDOUT:   %.loc202_31.1: ref bool = temporary_storage
// CHECK:STDOUT:   %addr.loc202_31.3: %ptr.bb2 = addr_of %.loc202_31.1
// CHECK:STDOUT:   %operator_EqualEqual__carbon_thunk.call: init %empty_tuple.type = call imports.%operator_EqualEqual__carbon_thunk.decl(%addr.loc202_31.1, %addr.loc202_31.2, %addr.loc202_31.3)
// CHECK:STDOUT:   %.loc202_31.2: init bool to %.loc202_31.1 = mark_in_place_init %operator_EqualEqual__carbon_thunk.call
// CHECK:STDOUT:   %.loc202_21: type = type_literal bool [concrete = bool]
// CHECK:STDOUT:   %.loc202_31.3: ref bool = temporary %.loc202_31.1, %.loc202_31.2
// CHECK:STDOUT:   %.loc202_31.4: bool = acquire_value %.loc202_31.3
// CHECK:STDOUT:   %equal: bool = value_binding equal, %.loc202_31.4
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %not_equal.patt: %pattern_type.831 = value_binding_pattern not_equal [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c1.ref.loc203: ref %C = name_ref c1, %c1
// CHECK:STDOUT:   %c2.ref.loc203: ref %C = name_ref c2, %c2
// CHECK:STDOUT:   %.loc203_32.1: %C = acquire_value %c1.ref.loc203
// CHECK:STDOUT:   %.loc203_38.1: %C = acquire_value %c2.ref.loc203
// CHECK:STDOUT:   %.loc203_32.2: ref %C = value_as_ref %.loc203_32.1
// CHECK:STDOUT:   %addr.loc203_35.1: %ptr.d9e = addr_of %.loc203_32.2
// CHECK:STDOUT:   %.loc203_38.2: ref %C = value_as_ref %.loc203_38.1
// CHECK:STDOUT:   %addr.loc203_35.2: %ptr.d9e = addr_of %.loc203_38.2
// CHECK:STDOUT:   %.loc203_35.1: ref bool = temporary_storage
// CHECK:STDOUT:   %addr.loc203_35.3: %ptr.bb2 = addr_of %.loc203_35.1
// CHECK:STDOUT:   %operator_ExclaimEqual__carbon_thunk.call: init %empty_tuple.type = call imports.%operator_ExclaimEqual__carbon_thunk.decl(%addr.loc203_35.1, %addr.loc203_35.2, %addr.loc203_35.3)
// CHECK:STDOUT:   %.loc203_35.2: init bool to %.loc203_35.1 = mark_in_place_init %operator_ExclaimEqual__carbon_thunk.call
// CHECK:STDOUT:   %.loc203_25: type = type_literal bool [concrete = bool]
// CHECK:STDOUT:   %.loc203_35.3: ref bool = temporary %.loc203_35.1, %.loc203_35.2
// CHECK:STDOUT:   %.loc203_35.4: bool = acquire_value %.loc203_35.3
// CHECK:STDOUT:   %not_equal: bool = value_binding not_equal, %.loc203_35.4
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %greater_than.patt: %pattern_type.831 = value_binding_pattern greater_than [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c1.ref.loc204: ref %C = name_ref c1, %c1
// CHECK:STDOUT:   %c2.ref.loc204: ref %C = name_ref c2, %c2
// CHECK:STDOUT:   %.loc204_35.1: %C = acquire_value %c1.ref.loc204
// CHECK:STDOUT:   %.loc204_40.1: %C = acquire_value %c2.ref.loc204
// CHECK:STDOUT:   %.loc204_35.2: ref %C = value_as_ref %.loc204_35.1
// CHECK:STDOUT:   %addr.loc204_38.1: %ptr.d9e = addr_of %.loc204_35.2
// CHECK:STDOUT:   %.loc204_40.2: ref %C = value_as_ref %.loc204_40.1
// CHECK:STDOUT:   %addr.loc204_38.2: %ptr.d9e = addr_of %.loc204_40.2
// CHECK:STDOUT:   %.loc204_38.1: ref bool = temporary_storage
// CHECK:STDOUT:   %addr.loc204_38.3: %ptr.bb2 = addr_of %.loc204_38.1
// CHECK:STDOUT:   %operator_Greater__carbon_thunk.call: init %empty_tuple.type = call imports.%operator_Greater__carbon_thunk.decl(%addr.loc204_38.1, %addr.loc204_38.2, %addr.loc204_38.3)
// CHECK:STDOUT:   %.loc204_38.2: init bool to %.loc204_38.1 = mark_in_place_init %operator_Greater__carbon_thunk.call
// CHECK:STDOUT:   %.loc204_28: type = type_literal bool [concrete = bool]
// CHECK:STDOUT:   %.loc204_38.3: ref bool = temporary %.loc204_38.1, %.loc204_38.2
// CHECK:STDOUT:   %.loc204_38.4: bool = acquire_value %.loc204_38.3
// CHECK:STDOUT:   %greater_than: bool = value_binding greater_than, %.loc204_38.4
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %less_than.patt: %pattern_type.831 = value_binding_pattern less_than [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c1.ref.loc205: ref %C = name_ref c1, %c1
// CHECK:STDOUT:   %c2.ref.loc205: ref %C = name_ref c2, %c2
// CHECK:STDOUT:   %.loc205_32.1: %C = acquire_value %c1.ref.loc205
// CHECK:STDOUT:   %.loc205_37.1: %C = acquire_value %c2.ref.loc205
// CHECK:STDOUT:   %.loc205_32.2: ref %C = value_as_ref %.loc205_32.1
// CHECK:STDOUT:   %addr.loc205_35.1: %ptr.d9e = addr_of %.loc205_32.2
// CHECK:STDOUT:   %.loc205_37.2: ref %C = value_as_ref %.loc205_37.1
// CHECK:STDOUT:   %addr.loc205_35.2: %ptr.d9e = addr_of %.loc205_37.2
// CHECK:STDOUT:   %.loc205_35.1: ref bool = temporary_storage
// CHECK:STDOUT:   %addr.loc205_35.3: %ptr.bb2 = addr_of %.loc205_35.1
// CHECK:STDOUT:   %operator_Less__carbon_thunk.call: init %empty_tuple.type = call imports.%operator_Less__carbon_thunk.decl(%addr.loc205_35.1, %addr.loc205_35.2, %addr.loc205_35.3)
// CHECK:STDOUT:   %.loc205_35.2: init bool to %.loc205_35.1 = mark_in_place_init %operator_Less__carbon_thunk.call
// CHECK:STDOUT:   %.loc205_25: type = type_literal bool [concrete = bool]
// CHECK:STDOUT:   %.loc205_35.3: ref bool = temporary %.loc205_35.1, %.loc205_35.2
// CHECK:STDOUT:   %.loc205_35.4: bool = acquire_value %.loc205_35.3
// CHECK:STDOUT:   %less_than: bool = value_binding less_than, %.loc205_35.4
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %greater_than_or_equal.patt: %pattern_type.831 = value_binding_pattern greater_than_or_equal [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c1.ref.loc206: ref %C = name_ref c1, %c1
// CHECK:STDOUT:   %c2.ref.loc206: ref %C = name_ref c2, %c2
// CHECK:STDOUT:   %.loc206_44.1: %C = acquire_value %c1.ref.loc206
// CHECK:STDOUT:   %.loc206_50.1: %C = acquire_value %c2.ref.loc206
// CHECK:STDOUT:   %.loc206_44.2: ref %C = value_as_ref %.loc206_44.1
// CHECK:STDOUT:   %addr.loc206_47.1: %ptr.d9e = addr_of %.loc206_44.2
// CHECK:STDOUT:   %.loc206_50.2: ref %C = value_as_ref %.loc206_50.1
// CHECK:STDOUT:   %addr.loc206_47.2: %ptr.d9e = addr_of %.loc206_50.2
// CHECK:STDOUT:   %.loc206_47.1: ref bool = temporary_storage
// CHECK:STDOUT:   %addr.loc206_47.3: %ptr.bb2 = addr_of %.loc206_47.1
// CHECK:STDOUT:   %operator_GreaterEqual__carbon_thunk.call: init %empty_tuple.type = call imports.%operator_GreaterEqual__carbon_thunk.decl(%addr.loc206_47.1, %addr.loc206_47.2, %addr.loc206_47.3)
// CHECK:STDOUT:   %.loc206_47.2: init bool to %.loc206_47.1 = mark_in_place_init %operator_GreaterEqual__carbon_thunk.call
// CHECK:STDOUT:   %.loc206_37: type = type_literal bool [concrete = bool]
// CHECK:STDOUT:   %.loc206_47.3: ref bool = temporary %.loc206_47.1, %.loc206_47.2
// CHECK:STDOUT:   %.loc206_47.4: bool = acquire_value %.loc206_47.3
// CHECK:STDOUT:   %greater_than_or_equal: bool = value_binding greater_than_or_equal, %.loc206_47.4
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %less_than_or_equal.patt: %pattern_type.831 = value_binding_pattern less_than_or_equal [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c1.ref.loc207: ref %C = name_ref c1, %c1
// CHECK:STDOUT:   %c2.ref.loc207: ref %C = name_ref c2, %c2
// CHECK:STDOUT:   %.loc207_41.1: %C = acquire_value %c1.ref.loc207
// CHECK:STDOUT:   %.loc207_47.1: %C = acquire_value %c2.ref.loc207
// CHECK:STDOUT:   %.loc207_41.2: ref %C = value_as_ref %.loc207_41.1
// CHECK:STDOUT:   %addr.loc207_44.1: %ptr.d9e = addr_of %.loc207_41.2
// CHECK:STDOUT:   %.loc207_47.2: ref %C = value_as_ref %.loc207_47.1
// CHECK:STDOUT:   %addr.loc207_44.2: %ptr.d9e = addr_of %.loc207_47.2
// CHECK:STDOUT:   %.loc207_44.1: ref bool = temporary_storage
// CHECK:STDOUT:   %addr.loc207_44.3: %ptr.bb2 = addr_of %.loc207_44.1
// CHECK:STDOUT:   %operator_LessEqual__carbon_thunk.call: init %empty_tuple.type = call imports.%operator_LessEqual__carbon_thunk.decl(%addr.loc207_44.1, %addr.loc207_44.2, %addr.loc207_44.3)
// CHECK:STDOUT:   %.loc207_44.2: init bool to %.loc207_44.1 = mark_in_place_init %operator_LessEqual__carbon_thunk.call
// CHECK:STDOUT:   %.loc207_34: type = type_literal bool [concrete = bool]
// CHECK:STDOUT:   %.loc207_44.3: ref bool = temporary %.loc207_44.1, %.loc207_44.2
// CHECK:STDOUT:   %.loc207_44.4: bool = acquire_value %.loc207_44.3
// CHECK:STDOUT:   %less_than_or_equal: bool = value_binding less_than_or_equal, %.loc207_44.4
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %index.patt: %pattern_type.7ce = value_binding_pattern index [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c1.ref.loc210: ref %C = name_ref c1, %c1
// CHECK:STDOUT:   %int_42: Core.IntLiteral = int_value 42 [concrete = constants.%int_42.20e]
// CHECK:STDOUT:   %C.cpp_operator.bound: <bound method> = bound_method %c1.ref.loc210, imports.%C.cpp_operator.decl
// CHECK:STDOUT:   %impl.elem0.loc210: %.545 = impl_witness_access constants.%ImplicitAs.impl_witness.6bc, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5]
// CHECK:STDOUT:   %bound_method.loc210_30.1: <bound method> = bound_method %int_42, %impl.elem0.loc210 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.c83]
// CHECK:STDOUT:   %specific_fn.loc210: <specific function> = specific_function %impl.elem0.loc210, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc210_30.2: <bound method> = bound_method %int_42, %specific_fn.loc210 [concrete = constants.%bound_method.cb9]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc210: init %i32 = call %bound_method.loc210_30.2(%int_42) [concrete = constants.%int_42.c68]
// CHECK:STDOUT:   %.loc210_30.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc210 [concrete = constants.%int_42.c68]
// CHECK:STDOUT:   %.loc210_30.2: %i32 = converted %int_42, %.loc210_30.1 [concrete = constants.%int_42.c68]
// CHECK:STDOUT:   %C.cpp_operator.call: init %i32 = call %C.cpp_operator.bound(%c1.ref.loc210, %.loc210_30.2)
// CHECK:STDOUT:   %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %.loc210_32.1: %i32 = value_of_initializer %C.cpp_operator.call
// CHECK:STDOUT:   %.loc210_32.2: %i32 = converted %C.cpp_operator.call, %.loc210_32.1
// CHECK:STDOUT:   %index: %i32 = value_binding index, %.loc210_32.2
// CHECK:STDOUT:   %Destroy.Op.bound.loc207: <bound method> = bound_method %.loc207_44.3, constants.%Destroy.Op
// CHECK:STDOUT:   %Destroy.Op.call.loc207: init %empty_tuple.type = call %Destroy.Op.bound.loc207(%.loc207_44.3)
// CHECK:STDOUT:   %Destroy.Op.bound.loc206: <bound method> = bound_method %.loc206_47.3, constants.%Destroy.Op
// CHECK:STDOUT:   %Destroy.Op.call.loc206: init %empty_tuple.type = call %Destroy.Op.bound.loc206(%.loc206_47.3)
// CHECK:STDOUT:   %Destroy.Op.bound.loc205: <bound method> = bound_method %.loc205_35.3, constants.%Destroy.Op
// CHECK:STDOUT:   %Destroy.Op.call.loc205: init %empty_tuple.type = call %Destroy.Op.bound.loc205(%.loc205_35.3)
// CHECK:STDOUT:   %Destroy.Op.bound.loc204: <bound method> = bound_method %.loc204_38.3, constants.%Destroy.Op
// CHECK:STDOUT:   %Destroy.Op.call.loc204: init %empty_tuple.type = call %Destroy.Op.bound.loc204(%.loc204_38.3)
// CHECK:STDOUT:   %Destroy.Op.bound.loc203: <bound method> = bound_method %.loc203_35.3, constants.%Destroy.Op
// CHECK:STDOUT:   %Destroy.Op.call.loc203: init %empty_tuple.type = call %Destroy.Op.bound.loc203(%.loc203_35.3)
// CHECK:STDOUT:   %Destroy.Op.bound.loc202: <bound method> = bound_method %.loc202_31.3, constants.%Destroy.Op
// CHECK:STDOUT:   %Destroy.Op.call.loc202: init %empty_tuple.type = call %Destroy.Op.bound.loc202(%.loc202_31.3)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %C.Op.bound.loc185: <bound method> = bound_method %.loc185_38.3, constants.%C.Op
// CHECK:STDOUT:   %Op.ref.loc185: %C.cpp_destructor.type = name_ref Op, imports.%C.cpp_destructor.decl [concrete = constants.%C.cpp_destructor]
// CHECK:STDOUT:   %C.cpp_destructor.bound.loc185: <bound method> = bound_method %.loc185_38.3, %Op.ref.loc185
// CHECK:STDOUT:   %C.cpp_destructor.call.loc185: init %empty_tuple.type = call %C.cpp_destructor.bound.loc185(%.loc185_38.3)
// CHECK:STDOUT:   %C.Op.bound.loc184: <bound method> = bound_method %.loc184_37.3, constants.%C.Op
// CHECK:STDOUT:   %Op.ref.loc184: %C.cpp_destructor.type = name_ref Op, imports.%C.cpp_destructor.decl [concrete = constants.%C.cpp_destructor]
// CHECK:STDOUT:   %C.cpp_destructor.bound.loc184: <bound method> = bound_method %.loc184_37.3, %Op.ref.loc184
// CHECK:STDOUT:   %C.cpp_destructor.call.loc184: init %empty_tuple.type = call %C.cpp_destructor.bound.loc184(%.loc184_37.3)
// CHECK:STDOUT:   %C.Op.bound.loc183: <bound method> = bound_method %.loc183_38.3, constants.%C.Op
// CHECK:STDOUT:   %Op.ref.loc183: %C.cpp_destructor.type = name_ref Op, imports.%C.cpp_destructor.decl [concrete = constants.%C.cpp_destructor]
// CHECK:STDOUT:   %C.cpp_destructor.bound.loc183: <bound method> = bound_method %.loc183_38.3, %Op.ref.loc183
// CHECK:STDOUT:   %C.cpp_destructor.call.loc183: init %empty_tuple.type = call %C.cpp_destructor.bound.loc183(%.loc183_38.3)
// CHECK:STDOUT:   %C.Op.bound.loc182: <bound method> = bound_method %.loc182_37.3, constants.%C.Op
// CHECK:STDOUT:   %Op.ref.loc182: %C.cpp_destructor.type = name_ref Op, imports.%C.cpp_destructor.decl [concrete = constants.%C.cpp_destructor]
// CHECK:STDOUT:   %C.cpp_destructor.bound.loc182: <bound method> = bound_method %.loc182_37.3, %Op.ref.loc182
// CHECK:STDOUT:   %C.cpp_destructor.call.loc182: init %empty_tuple.type = call %C.cpp_destructor.bound.loc182(%.loc182_37.3)
// CHECK:STDOUT:   %C.Op.bound.loc181: <bound method> = bound_method %.loc181_38.3, constants.%C.Op
// CHECK:STDOUT:   %Op.ref.loc181: %C.cpp_destructor.type = name_ref Op, imports.%C.cpp_destructor.decl [concrete = constants.%C.cpp_destructor]
// CHECK:STDOUT:   %C.cpp_destructor.bound.loc181: <bound method> = bound_method %.loc181_38.3, %Op.ref.loc181
// CHECK:STDOUT:   %C.cpp_destructor.call.loc181: init %empty_tuple.type = call %C.cpp_destructor.bound.loc181(%.loc181_38.3)
// CHECK:STDOUT:   %C.Op.bound.loc178: <bound method> = bound_method %.loc178_33.3, constants.%C.Op
// CHECK:STDOUT:   %Op.ref.loc178: %C.cpp_destructor.type = name_ref Op, imports.%C.cpp_destructor.decl [concrete = constants.%C.cpp_destructor]
// CHECK:STDOUT:   %C.cpp_destructor.bound.loc178: <bound method> = bound_method %.loc178_33.3, %Op.ref.loc178
// CHECK:STDOUT:   %C.cpp_destructor.call.loc178: init %empty_tuple.type = call %C.cpp_destructor.bound.loc178(%.loc178_33.3)
// CHECK:STDOUT:   %C.Op.bound.loc177: <bound method> = bound_method %.loc177_35.3, constants.%C.Op
// CHECK:STDOUT:   %Op.ref.loc177: %C.cpp_destructor.type = name_ref Op, imports.%C.cpp_destructor.decl [concrete = constants.%C.cpp_destructor]
// CHECK:STDOUT:   %C.cpp_destructor.bound.loc177: <bound method> = bound_method %.loc177_35.3, %Op.ref.loc177
// CHECK:STDOUT:   %C.cpp_destructor.call.loc177: init %empty_tuple.type = call %C.cpp_destructor.bound.loc177(%.loc177_35.3)
// CHECK:STDOUT:   %C.Op.bound.loc176: <bound method> = bound_method %.loc176_41.3, constants.%C.Op
// CHECK:STDOUT:   %Op.ref.loc176: %C.cpp_destructor.type = name_ref Op, imports.%C.cpp_destructor.decl [concrete = constants.%C.cpp_destructor]
// CHECK:STDOUT:   %C.cpp_destructor.bound.loc176: <bound method> = bound_method %.loc176_41.3, %Op.ref.loc176
// CHECK:STDOUT:   %C.cpp_destructor.call.loc176: init %empty_tuple.type = call %C.cpp_destructor.bound.loc176(%.loc176_41.3)
// CHECK:STDOUT:   %C.Op.bound.loc175: <bound method> = bound_method %.loc175_38.3, constants.%C.Op
// CHECK:STDOUT:   %Op.ref.loc175: %C.cpp_destructor.type = name_ref Op, imports.%C.cpp_destructor.decl [concrete = constants.%C.cpp_destructor]
// CHECK:STDOUT:   %C.cpp_destructor.bound.loc175: <bound method> = bound_method %.loc175_38.3, %Op.ref.loc175
// CHECK:STDOUT:   %C.cpp_destructor.call.loc175: init %empty_tuple.type = call %C.cpp_destructor.bound.loc175(%.loc175_38.3)
// CHECK:STDOUT:   %C.Op.bound.loc174: <bound method> = bound_method %.loc174_35.3, constants.%C.Op
// CHECK:STDOUT:   %Op.ref.loc174: %C.cpp_destructor.type = name_ref Op, imports.%C.cpp_destructor.decl [concrete = constants.%C.cpp_destructor]
// CHECK:STDOUT:   %C.cpp_destructor.bound.loc174: <bound method> = bound_method %.loc174_35.3, %Op.ref.loc174
// CHECK:STDOUT:   %C.cpp_destructor.call.loc174: init %empty_tuple.type = call %C.cpp_destructor.bound.loc174(%.loc174_35.3)
// CHECK:STDOUT:   %C.Op.bound.loc171: <bound method> = bound_method %c2.var, constants.%C.Op
// CHECK:STDOUT:   %Op.ref.loc171: %C.cpp_destructor.type = name_ref Op, imports.%C.cpp_destructor.decl [concrete = constants.%C.cpp_destructor]
// CHECK:STDOUT:   %C.cpp_destructor.bound.loc171: <bound method> = bound_method %c2.var, %Op.ref.loc171
// CHECK:STDOUT:   %C.cpp_destructor.call.loc171: init %empty_tuple.type = call %C.cpp_destructor.bound.loc171(%c2.var)
// CHECK:STDOUT:   %C.Op.bound.loc170: <bound method> = bound_method %c1.var, constants.%C.Op
// CHECK:STDOUT:   %Op.ref.loc170: %C.cpp_destructor.type = name_ref Op, imports.%C.cpp_destructor.decl [concrete = constants.%C.cpp_destructor]
// CHECK:STDOUT:   %C.cpp_destructor.bound.loc170: <bound method> = bound_method %c1.var, %Op.ref.loc170
// CHECK:STDOUT:   %C.cpp_destructor.call.loc170: init %empty_tuple.type = call %C.cpp_destructor.bound.loc170(%c1.var)
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
// CHECK:STDOUT:     %self.param_patt: %pattern_type.217 = ref_param_pattern [concrete]
// CHECK:STDOUT:     %self.patt: %pattern_type.217 = at_binding_pattern self, %self.param_patt [concrete]
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
// CHECK:STDOUT:   %Cpp.ref.loc242_19: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %C.ref.loc242_22: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %C.ref.loc242_24: %C.C.cpp_overload_set.type = name_ref C, imports.%C.C.cpp_overload_set.value [concrete = constants.%C.C.cpp_overload_set.value]
// CHECK:STDOUT:   %.loc242_27.1: ref %C = temporary_storage
// CHECK:STDOUT:   %addr.loc242: %ptr.d9e = addr_of %.loc242_27.1
// CHECK:STDOUT:   %C__carbon_thunk.call.loc242: init %empty_tuple.type = call imports.%C__carbon_thunk.decl(%addr.loc242)
// CHECK:STDOUT:   %.loc242_27.2: init %C to %.loc242_27.1 = mark_in_place_init %C__carbon_thunk.call.loc242
// CHECK:STDOUT:   %.loc242_14: type = splice_block %C.ref.loc242_14 [concrete = constants.%C] {
// CHECK:STDOUT:     %Cpp.ref.loc242_11: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %C.ref.loc242_14: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc242_27.3: ref %C = temporary %.loc242_27.1, %.loc242_27.2
// CHECK:STDOUT:   %.loc242_27.4: %C = acquire_value %.loc242_27.3
// CHECK:STDOUT:   %c1: %C = value_binding c1, %.loc242_27.4
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c2.patt: %pattern_type.217 = value_binding_pattern c2 [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc243_19: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %C.ref.loc243_22: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %C.ref.loc243_24: %C.C.cpp_overload_set.type = name_ref C, imports.%C.C.cpp_overload_set.value [concrete = constants.%C.C.cpp_overload_set.value]
// CHECK:STDOUT:   %.loc243_27.1: ref %C = temporary_storage
// CHECK:STDOUT:   %addr.loc243: %ptr.d9e = addr_of %.loc243_27.1
// CHECK:STDOUT:   %C__carbon_thunk.call.loc243: init %empty_tuple.type = call imports.%C__carbon_thunk.decl(%addr.loc243)
// CHECK:STDOUT:   %.loc243_27.2: init %C to %.loc243_27.1 = mark_in_place_init %C__carbon_thunk.call.loc243
// CHECK:STDOUT:   %.loc243_14: type = splice_block %C.ref.loc243_14 [concrete = constants.%C] {
// CHECK:STDOUT:     %Cpp.ref.loc243_11: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %C.ref.loc243_14: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc243_27.3: ref %C = temporary %.loc243_27.1, %.loc243_27.2
// CHECK:STDOUT:   %.loc243_27.4: %C = acquire_value %.loc243_27.3
// CHECK:STDOUT:   %c2: %C = value_binding c2, %.loc243_27.4
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c3.patt: %pattern_type.217 = value_binding_pattern c3 [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c1.ref.loc244: %C = name_ref c1, %c1
// CHECK:STDOUT:   %c2.ref: %C = name_ref c2, %c2
// CHECK:STDOUT:   %.loc244_22.1: ref %C = temporary_storage
// CHECK:STDOUT:   %.loc244_19: ref %C = value_as_ref %c1.ref.loc244
// CHECK:STDOUT:   %addr.loc244_22.1: %ptr.d9e = addr_of %.loc244_19
// CHECK:STDOUT:   %.loc244_24: ref %C = value_as_ref %c2.ref
// CHECK:STDOUT:   %addr.loc244_22.2: %ptr.d9e = addr_of %.loc244_24
// CHECK:STDOUT:   %addr.loc244_22.3: %ptr.d9e = addr_of %.loc244_22.1
// CHECK:STDOUT:   %operator_Plus__carbon_thunk.call.loc244: init %empty_tuple.type = call imports.%operator_Plus__carbon_thunk.decl(%addr.loc244_22.1, %addr.loc244_22.2, %addr.loc244_22.3)
// CHECK:STDOUT:   %.loc244_22.2: init %C to %.loc244_22.1 = mark_in_place_init %operator_Plus__carbon_thunk.call.loc244
// CHECK:STDOUT:   %.loc244_14: type = splice_block %C.ref.loc244 [concrete = constants.%C] {
// CHECK:STDOUT:     %Cpp.ref.loc244: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %C.ref.loc244: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc244_22.3: ref %C = temporary %.loc244_22.1, %.loc244_22.2
// CHECK:STDOUT:   %.loc244_22.4: %C = acquire_value %.loc244_22.3
// CHECK:STDOUT:   %c3: %C = value_binding c3, %.loc244_22.4
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c4.patt: %pattern_type.217 = value_binding_pattern c4 [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c1.ref.loc245: %C = name_ref c1, %c1
// CHECK:STDOUT:   %c3.ref.loc245: %C = name_ref c3, %c3
// CHECK:STDOUT:   %.loc245_22.1: ref %C = temporary_storage
// CHECK:STDOUT:   %.loc245_19: ref %C = value_as_ref %c1.ref.loc245
// CHECK:STDOUT:   %addr.loc245_22.1: %ptr.d9e = addr_of %.loc245_19
// CHECK:STDOUT:   %.loc245_24: ref %C = value_as_ref %c3.ref.loc245
// CHECK:STDOUT:   %addr.loc245_22.2: %ptr.d9e = addr_of %.loc245_24
// CHECK:STDOUT:   %addr.loc245_22.3: %ptr.d9e = addr_of %.loc245_22.1
// CHECK:STDOUT:   %operator_Plus__carbon_thunk.call.loc245: init %empty_tuple.type = call imports.%operator_Plus__carbon_thunk.decl(%addr.loc245_22.1, %addr.loc245_22.2, %addr.loc245_22.3)
// CHECK:STDOUT:   %.loc245_22.2: init %C to %.loc245_22.1 = mark_in_place_init %operator_Plus__carbon_thunk.call.loc245
// CHECK:STDOUT:   %.loc245_14: type = splice_block %C.ref.loc245 [concrete = constants.%C] {
// CHECK:STDOUT:     %Cpp.ref.loc245: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %C.ref.loc245: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc245_22.3: ref %C = temporary %.loc245_22.1, %.loc245_22.2
// CHECK:STDOUT:   %.loc245_22.4: %C = acquire_value %.loc245_22.3
// CHECK:STDOUT:   %c4: %C = value_binding c4, %.loc245_22.4
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c5.patt: %pattern_type.217 = value_binding_pattern c5 [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c4.ref: %C = name_ref c4, %c4
// CHECK:STDOUT:   %c3.ref.loc246: %C = name_ref c3, %c3
// CHECK:STDOUT:   %.loc246_29.1: ref %C = temporary_storage
// CHECK:STDOUT:   %.loc246_26: ref %C = value_as_ref %c4.ref
// CHECK:STDOUT:   %addr.loc246_29.1: %ptr.d9e = addr_of %.loc246_26
// CHECK:STDOUT:   %.loc246_31: ref %C = value_as_ref %c3.ref.loc246
// CHECK:STDOUT:   %addr.loc246_29.2: %ptr.d9e = addr_of %.loc246_31
// CHECK:STDOUT:   %addr.loc246_29.3: %ptr.d9e = addr_of %.loc246_29.1
// CHECK:STDOUT:   %operator_Plus__carbon_thunk.call.loc246: init %empty_tuple.type = call imports.%operator_Plus__carbon_thunk.decl(%addr.loc246_29.1, %addr.loc246_29.2, %addr.loc246_29.3)
// CHECK:STDOUT:   %.loc246_29.2: init %C to %.loc246_29.1 = mark_in_place_init %operator_Plus__carbon_thunk.call.loc246
// CHECK:STDOUT:   %.loc246_21: type = splice_block %C.ref.loc246 [concrete = constants.%C] {
// CHECK:STDOUT:     %Cpp.ref.loc246: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %C.ref.loc246: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc246_29.3: ref %C = temporary %.loc246_29.1, %.loc246_29.2
// CHECK:STDOUT:   %.loc246_29.4: %C = acquire_value %.loc246_29.3
// CHECK:STDOUT:   %c5: %C = value_binding c5, %.loc246_29.4
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %C.Op.bound.loc246: <bound method> = bound_method %.loc246_29.3, constants.%C.Op
// CHECK:STDOUT:   %Op.ref.loc246: %C.cpp_destructor.type = name_ref Op, imports.%C.cpp_destructor.decl [concrete = constants.%C.cpp_destructor]
// CHECK:STDOUT:   %C.cpp_destructor.bound.loc246: <bound method> = bound_method %.loc246_29.3, %Op.ref.loc246
// CHECK:STDOUT:   %C.cpp_destructor.call.loc246: init %empty_tuple.type = call %C.cpp_destructor.bound.loc246(%.loc246_29.3)
// CHECK:STDOUT:   %C.Op.bound.loc245: <bound method> = bound_method %.loc245_22.3, constants.%C.Op
// CHECK:STDOUT:   %Op.ref.loc245: %C.cpp_destructor.type = name_ref Op, imports.%C.cpp_destructor.decl [concrete = constants.%C.cpp_destructor]
// CHECK:STDOUT:   %C.cpp_destructor.bound.loc245: <bound method> = bound_method %.loc245_22.3, %Op.ref.loc245
// CHECK:STDOUT:   %C.cpp_destructor.call.loc245: init %empty_tuple.type = call %C.cpp_destructor.bound.loc245(%.loc245_22.3)
// CHECK:STDOUT:   %C.Op.bound.loc244: <bound method> = bound_method %.loc244_22.3, constants.%C.Op
// CHECK:STDOUT:   %Op.ref.loc244: %C.cpp_destructor.type = name_ref Op, imports.%C.cpp_destructor.decl [concrete = constants.%C.cpp_destructor]
// CHECK:STDOUT:   %C.cpp_destructor.bound.loc244: <bound method> = bound_method %.loc244_22.3, %Op.ref.loc244
// CHECK:STDOUT:   %C.cpp_destructor.call.loc244: init %empty_tuple.type = call %C.cpp_destructor.bound.loc244(%.loc244_22.3)
// CHECK:STDOUT:   %C.Op.bound.loc243: <bound method> = bound_method %.loc243_27.3, constants.%C.Op
// CHECK:STDOUT:   %Op.ref.loc243: %C.cpp_destructor.type = name_ref Op, imports.%C.cpp_destructor.decl [concrete = constants.%C.cpp_destructor]
// CHECK:STDOUT:   %C.cpp_destructor.bound.loc243: <bound method> = bound_method %.loc243_27.3, %Op.ref.loc243
// CHECK:STDOUT:   %C.cpp_destructor.call.loc243: init %empty_tuple.type = call %C.cpp_destructor.bound.loc243(%.loc243_27.3)
// CHECK:STDOUT:   %C.Op.bound.loc242: <bound method> = bound_method %.loc242_27.3, constants.%C.Op
// CHECK:STDOUT:   %Op.ref.loc242: %C.cpp_destructor.type = name_ref Op, imports.%C.cpp_destructor.decl [concrete = constants.%C.cpp_destructor]
// CHECK:STDOUT:   %C.cpp_destructor.bound.loc242: <bound method> = bound_method %.loc242_27.3, %Op.ref.loc242
// CHECK:STDOUT:   %C.cpp_destructor.call.loc242: init %empty_tuple.type = call %C.cpp_destructor.bound.loc242(%.loc242_27.3)
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
// CHECK:STDOUT:     %self.param_patt: %pattern_type.217 = ref_param_pattern [concrete]
// CHECK:STDOUT:     %self.patt: %pattern_type.217 = at_binding_pattern self, %self.param_patt [concrete]
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
// CHECK:STDOUT:   %Cpp.ref.loc342_19: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %C.ref.loc342_22: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %C.ref.loc342_24: %C.C.cpp_overload_set.type = name_ref C, imports.%C.C.cpp_overload_set.value [concrete = constants.%C.C.cpp_overload_set.value]
// CHECK:STDOUT:   %.loc342_3: ref %C = splice_block %c1.var {}
// CHECK:STDOUT:   %addr.loc342: %ptr.d9e = addr_of %.loc342_3
// CHECK:STDOUT:   %C__carbon_thunk.call.loc342: init %empty_tuple.type = call imports.%C__carbon_thunk.decl(%addr.loc342)
// CHECK:STDOUT:   %.loc342_27: init %C to %.loc342_3 = mark_in_place_init %C__carbon_thunk.call.loc342
// CHECK:STDOUT:   assign %c1.var, %.loc342_27
// CHECK:STDOUT:   %.loc342_14: type = splice_block %C.ref.loc342_14 [concrete = constants.%C] {
// CHECK:STDOUT:     %Cpp.ref.loc342_11: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %C.ref.loc342_14: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c1: ref %C = ref_binding c1, %c1.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c2.patt: %pattern_type.217 = ref_binding_pattern c2 [concrete]
// CHECK:STDOUT:     %c2.var_patt: %pattern_type.217 = var_pattern %c2.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c2.var: ref %C = var %c2.var_patt
// CHECK:STDOUT:   %Cpp.ref.loc343_19: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %C.ref.loc343_22: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %C.ref.loc343_24: %C.C.cpp_overload_set.type = name_ref C, imports.%C.C.cpp_overload_set.value [concrete = constants.%C.C.cpp_overload_set.value]
// CHECK:STDOUT:   %.loc343_3: ref %C = splice_block %c2.var {}
// CHECK:STDOUT:   %addr.loc343: %ptr.d9e = addr_of %.loc343_3
// CHECK:STDOUT:   %C__carbon_thunk.call.loc343: init %empty_tuple.type = call imports.%C__carbon_thunk.decl(%addr.loc343)
// CHECK:STDOUT:   %.loc343_27: init %C to %.loc343_3 = mark_in_place_init %C__carbon_thunk.call.loc343
// CHECK:STDOUT:   assign %c2.var, %.loc343_27
// CHECK:STDOUT:   %.loc343_14: type = splice_block %C.ref.loc343_14 [concrete = constants.%C] {
// CHECK:STDOUT:     %Cpp.ref.loc343_11: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %C.ref.loc343_14: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c2: ref %C = ref_binding c2, %c2.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %greater_than.patt: %pattern_type.831 = value_binding_pattern greater_than [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c1.ref.loc346: ref %C = name_ref c1, %c1
// CHECK:STDOUT:   %c2.ref.loc346: ref %C = name_ref c2, %c2
// CHECK:STDOUT:   %.loc346_35.1: %C = acquire_value %c1.ref.loc346
// CHECK:STDOUT:   %.loc346_40.1: %C = acquire_value %c2.ref.loc346
// CHECK:STDOUT:   %.loc346_35.2: ref %C = value_as_ref %.loc346_35.1
// CHECK:STDOUT:   %addr.loc346_38.1: %ptr.d9e = addr_of %.loc346_35.2
// CHECK:STDOUT:   %.loc346_40.2: ref %C = value_as_ref %.loc346_40.1
// CHECK:STDOUT:   %addr.loc346_38.2: %ptr.d9e = addr_of %.loc346_40.2
// CHECK:STDOUT:   %.loc346_38.1: ref bool = temporary_storage
// CHECK:STDOUT:   %addr.loc346_38.3: %ptr.bb2 = addr_of %.loc346_38.1
// CHECK:STDOUT:   %operator_Greater__carbon_thunk.call: init %empty_tuple.type = call imports.%operator_Greater__carbon_thunk.decl(%addr.loc346_38.1, %addr.loc346_38.2, %addr.loc346_38.3)
// CHECK:STDOUT:   %.loc346_38.2: init bool to %.loc346_38.1 = mark_in_place_init %operator_Greater__carbon_thunk.call
// CHECK:STDOUT:   %.loc346_28: type = type_literal bool [concrete = bool]
// CHECK:STDOUT:   %.loc346_38.3: ref bool = temporary %.loc346_38.1, %.loc346_38.2
// CHECK:STDOUT:   %.loc346_38.4: bool = acquire_value %.loc346_38.3
// CHECK:STDOUT:   %greater_than: bool = value_binding greater_than, %.loc346_38.4
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %less_than.patt: %pattern_type.831 = value_binding_pattern less_than [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c1.ref.loc353: ref %C = name_ref c1, %c1
// CHECK:STDOUT:   %c2.ref.loc353: ref %C = name_ref c2, %c2
// CHECK:STDOUT:   %.loc353: type = type_literal bool [concrete = bool]
// CHECK:STDOUT:   %less_than: bool = value_binding less_than, <error> [concrete = <error>]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %greater_than_or_equal.patt: %pattern_type.831 = value_binding_pattern greater_than_or_equal [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c1.ref.loc360: ref %C = name_ref c1, %c1
// CHECK:STDOUT:   %c2.ref.loc360: ref %C = name_ref c2, %c2
// CHECK:STDOUT:   %.loc360: type = type_literal bool [concrete = bool]
// CHECK:STDOUT:   %greater_than_or_equal: bool = value_binding greater_than_or_equal, <error> [concrete = <error>]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %less_than_or_equal.patt: %pattern_type.831 = value_binding_pattern less_than_or_equal [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c1.ref.loc363: ref %C = name_ref c1, %c1
// CHECK:STDOUT:   %c2.ref.loc363: ref %C = name_ref c2, %c2
// CHECK:STDOUT:   %.loc363_41.1: %C = acquire_value %c1.ref.loc363
// CHECK:STDOUT:   %.loc363_47.1: %C = acquire_value %c2.ref.loc363
// CHECK:STDOUT:   %.loc363_41.2: ref %C = value_as_ref %.loc363_41.1
// CHECK:STDOUT:   %addr.loc363_44.1: %ptr.d9e = addr_of %.loc363_41.2
// CHECK:STDOUT:   %.loc363_47.2: ref %C = value_as_ref %.loc363_47.1
// CHECK:STDOUT:   %addr.loc363_44.2: %ptr.d9e = addr_of %.loc363_47.2
// CHECK:STDOUT:   %.loc363_44.1: ref bool = temporary_storage
// CHECK:STDOUT:   %addr.loc363_44.3: %ptr.bb2 = addr_of %.loc363_44.1
// CHECK:STDOUT:   %operator_LessEqual__carbon_thunk.call: init %empty_tuple.type = call imports.%operator_LessEqual__carbon_thunk.decl(%addr.loc363_44.1, %addr.loc363_44.2, %addr.loc363_44.3)
// CHECK:STDOUT:   %.loc363_44.2: init bool to %.loc363_44.1 = mark_in_place_init %operator_LessEqual__carbon_thunk.call
// CHECK:STDOUT:   %.loc363_34: type = type_literal bool [concrete = bool]
// CHECK:STDOUT:   %.loc363_44.3: ref bool = temporary %.loc363_44.1, %.loc363_44.2
// CHECK:STDOUT:   %.loc363_44.4: bool = acquire_value %.loc363_44.3
// CHECK:STDOUT:   %less_than_or_equal: bool = value_binding less_than_or_equal, %.loc363_44.4
// CHECK:STDOUT:   %Destroy.Op.bound.loc363: <bound method> = bound_method %.loc363_44.3, constants.%Destroy.Op
// CHECK:STDOUT:   %Destroy.Op.call.loc363: init %empty_tuple.type = call %Destroy.Op.bound.loc363(%.loc363_44.3)
// CHECK:STDOUT:   %Destroy.Op.bound.loc346: <bound method> = bound_method %.loc346_38.3, constants.%Destroy.Op
// CHECK:STDOUT:   %Destroy.Op.call.loc346: init %empty_tuple.type = call %Destroy.Op.bound.loc346(%.loc346_38.3)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %C.Op.bound.loc343: <bound method> = bound_method %c2.var, constants.%C.Op
// CHECK:STDOUT:   %Op.ref.loc343: %C.cpp_destructor.type = name_ref Op, imports.%C.cpp_destructor.decl [concrete = constants.%C.cpp_destructor]
// CHECK:STDOUT:   %C.cpp_destructor.bound.loc343: <bound method> = bound_method %c2.var, %Op.ref.loc343
// CHECK:STDOUT:   %C.cpp_destructor.call.loc343: init %empty_tuple.type = call %C.cpp_destructor.bound.loc343(%c2.var)
// CHECK:STDOUT:   %C.Op.bound.loc342: <bound method> = bound_method %c1.var, constants.%C.Op
// CHECK:STDOUT:   %Op.ref.loc342: %C.cpp_destructor.type = name_ref Op, imports.%C.cpp_destructor.decl [concrete = constants.%C.cpp_destructor]
// CHECK:STDOUT:   %C.cpp_destructor.bound.loc342: <bound method> = bound_method %c1.var, %Op.ref.loc342
// CHECK:STDOUT:   %C.cpp_destructor.call.loc342: init %empty_tuple.type = call %C.cpp_destructor.bound.loc342(%c1.var)
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
// CHECK:STDOUT:     %self.param_patt: %pattern_type.217 = ref_param_pattern [concrete]
// CHECK:STDOUT:     %self.patt: %pattern_type.217 = at_binding_pattern self, %self.param_patt [concrete]
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
// CHECK:STDOUT:   %Cpp.ref.loc411_19: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %C.ref.loc411_22: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %C.ref.loc411_24: %C.C.cpp_overload_set.type = name_ref C, imports.%C.C.cpp_overload_set.value [concrete = constants.%C.C.cpp_overload_set.value]
// CHECK:STDOUT:   %.loc411_3: ref %C = splice_block %c1.var {}
// CHECK:STDOUT:   %addr.loc411: %ptr.d9e = addr_of %.loc411_3
// CHECK:STDOUT:   %C__carbon_thunk.call.loc411: init %empty_tuple.type = call imports.%C__carbon_thunk.decl(%addr.loc411)
// CHECK:STDOUT:   %.loc411_27: init %C to %.loc411_3 = mark_in_place_init %C__carbon_thunk.call.loc411
// CHECK:STDOUT:   assign %c1.var, %.loc411_27
// CHECK:STDOUT:   %.loc411_14: type = splice_block %C.ref.loc411_14 [concrete = constants.%C] {
// CHECK:STDOUT:     %Cpp.ref.loc411_11: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %C.ref.loc411_14: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c1: ref %C = ref_binding c1, %c1.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c2.patt: %pattern_type.217 = ref_binding_pattern c2 [concrete]
// CHECK:STDOUT:     %c2.var_patt: %pattern_type.217 = var_pattern %c2.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c2.var: ref %C = var %c2.var_patt
// CHECK:STDOUT:   %Cpp.ref.loc412_19: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %C.ref.loc412_22: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %C.ref.loc412_24: %C.C.cpp_overload_set.type = name_ref C, imports.%C.C.cpp_overload_set.value [concrete = constants.%C.C.cpp_overload_set.value]
// CHECK:STDOUT:   %.loc412_3: ref %C = splice_block %c2.var {}
// CHECK:STDOUT:   %addr.loc412: %ptr.d9e = addr_of %.loc412_3
// CHECK:STDOUT:   %C__carbon_thunk.call.loc412: init %empty_tuple.type = call imports.%C__carbon_thunk.decl(%addr.loc412)
// CHECK:STDOUT:   %.loc412_27: init %C to %.loc412_3 = mark_in_place_init %C__carbon_thunk.call.loc412
// CHECK:STDOUT:   assign %c2.var, %.loc412_27
// CHECK:STDOUT:   %.loc412_14: type = splice_block %C.ref.loc412_14 [concrete = constants.%C] {
// CHECK:STDOUT:     %Cpp.ref.loc412_11: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %C.ref.loc412_14: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c2: ref %C = ref_binding c2, %c2.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %equal.patt: %pattern_type.831 = value_binding_pattern equal [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c1.ref.loc415: ref %C = name_ref c1, %c1
// CHECK:STDOUT:   %c2.ref.loc415: ref %C = name_ref c2, %c2
// CHECK:STDOUT:   %.loc415_28.1: %C = acquire_value %c1.ref.loc415
// CHECK:STDOUT:   %.loc415_34.1: %C = acquire_value %c2.ref.loc415
// CHECK:STDOUT:   %.loc415_28.2: ref %C = value_as_ref %.loc415_28.1
// CHECK:STDOUT:   %addr.loc415_31.1: %ptr.d9e = addr_of %.loc415_28.2
// CHECK:STDOUT:   %.loc415_34.2: ref %C = value_as_ref %.loc415_34.1
// CHECK:STDOUT:   %addr.loc415_31.2: %ptr.d9e = addr_of %.loc415_34.2
// CHECK:STDOUT:   %.loc415_31.1: ref bool = temporary_storage
// CHECK:STDOUT:   %addr.loc415_31.3: %ptr.bb2 = addr_of %.loc415_31.1
// CHECK:STDOUT:   %operator_EqualEqual__carbon_thunk.call: init %empty_tuple.type = call imports.%operator_EqualEqual__carbon_thunk.decl(%addr.loc415_31.1, %addr.loc415_31.2, %addr.loc415_31.3)
// CHECK:STDOUT:   %.loc415_31.2: init bool to %.loc415_31.1 = mark_in_place_init %operator_EqualEqual__carbon_thunk.call
// CHECK:STDOUT:   %.loc415_21: type = type_literal bool [concrete = bool]
// CHECK:STDOUT:   %.loc415_31.3: ref bool = temporary %.loc415_31.1, %.loc415_31.2
// CHECK:STDOUT:   %.loc415_31.4: bool = acquire_value %.loc415_31.3
// CHECK:STDOUT:   %equal: bool = value_binding equal, %.loc415_31.4
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %not_equal.patt: %pattern_type.831 = value_binding_pattern not_equal [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c1.ref.loc422: ref %C = name_ref c1, %c1
// CHECK:STDOUT:   %c2.ref.loc422: ref %C = name_ref c2, %c2
// CHECK:STDOUT:   %.loc422: type = type_literal bool [concrete = bool]
// CHECK:STDOUT:   %not_equal: bool = value_binding not_equal, <error> [concrete = <error>]
// CHECK:STDOUT:   %Destroy.Op.bound: <bound method> = bound_method %.loc415_31.3, constants.%Destroy.Op
// CHECK:STDOUT:   %Destroy.Op.call: init %empty_tuple.type = call %Destroy.Op.bound(%.loc415_31.3)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %C.Op.bound.loc412: <bound method> = bound_method %c2.var, constants.%C.Op
// CHECK:STDOUT:   %Op.ref.loc412: %C.cpp_destructor.type = name_ref Op, imports.%C.cpp_destructor.decl [concrete = constants.%C.cpp_destructor]
// CHECK:STDOUT:   %C.cpp_destructor.bound.loc412: <bound method> = bound_method %c2.var, %Op.ref.loc412
// CHECK:STDOUT:   %C.cpp_destructor.call.loc412: init %empty_tuple.type = call %C.cpp_destructor.bound.loc412(%c2.var)
// CHECK:STDOUT:   %C.Op.bound.loc411: <bound method> = bound_method %c1.var, constants.%C.Op
// CHECK:STDOUT:   %Op.ref.loc411: %C.cpp_destructor.type = name_ref Op, imports.%C.cpp_destructor.decl [concrete = constants.%C.cpp_destructor]
// CHECK:STDOUT:   %C.cpp_destructor.bound.loc411: <bound method> = bound_method %c1.var, %Op.ref.loc411
// CHECK:STDOUT:   %C.cpp_destructor.call.loc411: init %empty_tuple.type = call %C.cpp_destructor.bound.loc411(%c1.var)
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
// CHECK:STDOUT:     %self.param_patt: %pattern_type.69f = ref_param_pattern [concrete]
// CHECK:STDOUT:     %self.patt: %pattern_type.69f = at_binding_pattern self, %self.param_patt [concrete]
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
// CHECK:STDOUT:   %Cpp.ref.loc673_21: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %N.ref.loc673_24: <namespace> = name_ref N, imports.%N [concrete = imports.%N]
// CHECK:STDOUT:   %C.ref.loc673_26: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %C.ref.loc673_28: %C.C.cpp_overload_set.type = name_ref C, imports.%C.C.cpp_overload_set.value [concrete = constants.%C.C.cpp_overload_set.value]
// CHECK:STDOUT:   %.loc673_31.1: ref %C = temporary_storage
// CHECK:STDOUT:   %addr.loc673: %ptr.838 = addr_of %.loc673_31.1
// CHECK:STDOUT:   %C__carbon_thunk.call.loc673: init %empty_tuple.type = call imports.%C__carbon_thunk.decl(%addr.loc673)
// CHECK:STDOUT:   %.loc673_31.2: init %C to %.loc673_31.1 = mark_in_place_init %C__carbon_thunk.call.loc673
// CHECK:STDOUT:   %.loc673_16: type = splice_block %C.ref.loc673_16 [concrete = constants.%C] {
// CHECK:STDOUT:     %Cpp.ref.loc673_11: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %N.ref.loc673_14: <namespace> = name_ref N, imports.%N [concrete = imports.%N]
// CHECK:STDOUT:     %C.ref.loc673_16: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc673_31.3: ref %C = temporary %.loc673_31.1, %.loc673_31.2
// CHECK:STDOUT:   %.loc673_31.4: %C = acquire_value %.loc673_31.3
// CHECK:STDOUT:   %c1: %C = value_binding c1, %.loc673_31.4
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c2.patt: %pattern_type.69f = value_binding_pattern c2 [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc674_21: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %N.ref.loc674_24: <namespace> = name_ref N, imports.%N [concrete = imports.%N]
// CHECK:STDOUT:   %C.ref.loc674_26: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %C.ref.loc674_28: %C.C.cpp_overload_set.type = name_ref C, imports.%C.C.cpp_overload_set.value [concrete = constants.%C.C.cpp_overload_set.value]
// CHECK:STDOUT:   %.loc674_31.1: ref %C = temporary_storage
// CHECK:STDOUT:   %addr.loc674: %ptr.838 = addr_of %.loc674_31.1
// CHECK:STDOUT:   %C__carbon_thunk.call.loc674: init %empty_tuple.type = call imports.%C__carbon_thunk.decl(%addr.loc674)
// CHECK:STDOUT:   %.loc674_31.2: init %C to %.loc674_31.1 = mark_in_place_init %C__carbon_thunk.call.loc674
// CHECK:STDOUT:   %.loc674_16: type = splice_block %C.ref.loc674_16 [concrete = constants.%C] {
// CHECK:STDOUT:     %Cpp.ref.loc674_11: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %N.ref.loc674_14: <namespace> = name_ref N, imports.%N [concrete = imports.%N]
// CHECK:STDOUT:     %C.ref.loc674_16: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc674_31.3: ref %C = temporary %.loc674_31.1, %.loc674_31.2
// CHECK:STDOUT:   %.loc674_31.4: %C = acquire_value %.loc674_31.3
// CHECK:STDOUT:   %c2: %C = value_binding c2, %.loc674_31.4
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c3.patt: %pattern_type.69f = value_binding_pattern c3 [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c1.ref: %C = name_ref c1, %c1
// CHECK:STDOUT:   %c2.ref: %C = name_ref c2, %c2
// CHECK:STDOUT:   %.loc675_31.1: ref %C = temporary_storage
// CHECK:STDOUT:   %.loc675_28: ref %C = value_as_ref %c1.ref
// CHECK:STDOUT:   %addr.loc675_31.1: %ptr.838 = addr_of %.loc675_28
// CHECK:STDOUT:   %.loc675_33: ref %C = value_as_ref %c2.ref
// CHECK:STDOUT:   %addr.loc675_31.2: %ptr.838 = addr_of %.loc675_33
// CHECK:STDOUT:   %addr.loc675_31.3: %ptr.838 = addr_of %.loc675_31.1
// CHECK:STDOUT:   %operator_Plus__carbon_thunk.call: init %empty_tuple.type = call imports.%operator_Plus__carbon_thunk.decl(%addr.loc675_31.1, %addr.loc675_31.2, %addr.loc675_31.3)
// CHECK:STDOUT:   %.loc675_31.2: init %C to %.loc675_31.1 = mark_in_place_init %operator_Plus__carbon_thunk.call
// CHECK:STDOUT:   %.loc675_23: type = splice_block %C.ref.loc675 [concrete = constants.%C] {
// CHECK:STDOUT:     %Cpp.ref.loc675: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %N.ref.loc675: <namespace> = name_ref N, imports.%N [concrete = imports.%N]
// CHECK:STDOUT:     %C.ref.loc675: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc675_31.3: ref %C = temporary %.loc675_31.1, %.loc675_31.2
// CHECK:STDOUT:   %.loc675_31.4: %C = acquire_value %.loc675_31.3
// CHECK:STDOUT:   %c3: %C = value_binding c3, %.loc675_31.4
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %C.Op.bound.loc675: <bound method> = bound_method %.loc675_31.3, constants.%C.Op
// CHECK:STDOUT:   %Op.ref.loc675: %C.cpp_destructor.type = name_ref Op, imports.%C.cpp_destructor.decl [concrete = constants.%C.cpp_destructor]
// CHECK:STDOUT:   %C.cpp_destructor.bound.loc675: <bound method> = bound_method %.loc675_31.3, %Op.ref.loc675
// CHECK:STDOUT:   %C.cpp_destructor.call.loc675: init %empty_tuple.type = call %C.cpp_destructor.bound.loc675(%.loc675_31.3)
// CHECK:STDOUT:   %C.Op.bound.loc674: <bound method> = bound_method %.loc674_31.3, constants.%C.Op
// CHECK:STDOUT:   %Op.ref.loc674: %C.cpp_destructor.type = name_ref Op, imports.%C.cpp_destructor.decl [concrete = constants.%C.cpp_destructor]
// CHECK:STDOUT:   %C.cpp_destructor.bound.loc674: <bound method> = bound_method %.loc674_31.3, %Op.ref.loc674
// CHECK:STDOUT:   %C.cpp_destructor.call.loc674: init %empty_tuple.type = call %C.cpp_destructor.bound.loc674(%.loc674_31.3)
// CHECK:STDOUT:   %C.Op.bound.loc673: <bound method> = bound_method %.loc673_31.3, constants.%C.Op
// CHECK:STDOUT:   %Op.ref.loc673: %C.cpp_destructor.type = name_ref Op, imports.%C.cpp_destructor.decl [concrete = constants.%C.cpp_destructor]
// CHECK:STDOUT:   %C.cpp_destructor.bound.loc673: <bound method> = bound_method %.loc673_31.3, %Op.ref.loc673
// CHECK:STDOUT:   %C.cpp_destructor.call.loc673: init %empty_tuple.type = call %C.cpp_destructor.bound.loc673(%.loc673_31.3)
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
// CHECK:STDOUT:   %C1.cpp_destructor.type: type = fn_type @C1.cpp_destructor [concrete]
// CHECK:STDOUT:   %C1.cpp_destructor: %C1.cpp_destructor.type = struct_value () [concrete]
// CHECK:STDOUT:   %C1.Op.type: type = fn_type @C1.Op [concrete]
// CHECK:STDOUT:   %C1.Op: %C1.Op.type = struct_value () [concrete]
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
// CHECK:STDOUT:     %self.param_patt: %pattern_type.846 = ref_param_pattern [concrete]
// CHECK:STDOUT:     %self.patt: %pattern_type.846 = at_binding_pattern self, %self.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %C2 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %C2 = ref_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C1.cpp_destructor.decl: %C1.cpp_destructor.type = fn_decl @C1.cpp_destructor [concrete = constants.%C1.cpp_destructor] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.20f = ref_param_pattern [concrete]
// CHECK:STDOUT:     %self.patt: %pattern_type.20f = at_binding_pattern self, %self.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %C1 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %C1 = ref_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c1.patt: %pattern_type.20f = value_binding_pattern c1 [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc702_23: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %N1.ref.loc702_26: <namespace> = name_ref N1, imports.%N1 [concrete = imports.%N1]
// CHECK:STDOUT:   %C1.ref.loc702_29: type = name_ref C1, imports.%C1.decl [concrete = constants.%C1]
// CHECK:STDOUT:   %C1.ref.loc702_32: %C1.C1.cpp_overload_set.type = name_ref C1, imports.%C1.C1.cpp_overload_set.value [concrete = constants.%C1.C1.cpp_overload_set.value]
// CHECK:STDOUT:   %.loc702_36.1: ref %C1 = temporary_storage
// CHECK:STDOUT:   %addr.loc702: %ptr.087 = addr_of %.loc702_36.1
// CHECK:STDOUT:   %C1__carbon_thunk.call: init %empty_tuple.type = call imports.%C1__carbon_thunk.decl(%addr.loc702)
// CHECK:STDOUT:   %.loc702_36.2: init %C1 to %.loc702_36.1 = mark_in_place_init %C1__carbon_thunk.call
// CHECK:STDOUT:   %.loc702_17: type = splice_block %C1.ref.loc702_17 [concrete = constants.%C1] {
// CHECK:STDOUT:     %Cpp.ref.loc702_11: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %N1.ref.loc702_14: <namespace> = name_ref N1, imports.%N1 [concrete = imports.%N1]
// CHECK:STDOUT:     %C1.ref.loc702_17: type = name_ref C1, imports.%C1.decl [concrete = constants.%C1]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc702_36.3: ref %C1 = temporary %.loc702_36.1, %.loc702_36.2
// CHECK:STDOUT:   %.loc702_36.4: %C1 = acquire_value %.loc702_36.3
// CHECK:STDOUT:   %c1: %C1 = value_binding c1, %.loc702_36.4
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c2.patt: %pattern_type.846 = value_binding_pattern c2 [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc703_23: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %N2.ref.loc703_26: <namespace> = name_ref N2, imports.%N2 [concrete = imports.%N2]
// CHECK:STDOUT:   %C2.ref.loc703_29: type = name_ref C2, imports.%C2.decl [concrete = constants.%C2]
// CHECK:STDOUT:   %C2.ref.loc703_32: %C2.C2.cpp_overload_set.type = name_ref C2, imports.%C2.C2.cpp_overload_set.value [concrete = constants.%C2.C2.cpp_overload_set.value]
// CHECK:STDOUT:   %.loc703_36.1: ref %C2 = temporary_storage
// CHECK:STDOUT:   %addr.loc703: %ptr.51f = addr_of %.loc703_36.1
// CHECK:STDOUT:   %C2__carbon_thunk.call: init %empty_tuple.type = call imports.%C2__carbon_thunk.decl(%addr.loc703)
// CHECK:STDOUT:   %.loc703_36.2: init %C2 to %.loc703_36.1 = mark_in_place_init %C2__carbon_thunk.call
// CHECK:STDOUT:   %.loc703_17: type = splice_block %C2.ref.loc703_17 [concrete = constants.%C2] {
// CHECK:STDOUT:     %Cpp.ref.loc703_11: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %N2.ref.loc703_14: <namespace> = name_ref N2, imports.%N2 [concrete = imports.%N2]
// CHECK:STDOUT:     %C2.ref.loc703_17: type = name_ref C2, imports.%C2.decl [concrete = constants.%C2]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc703_36.3: ref %C2 = temporary %.loc703_36.1, %.loc703_36.2
// CHECK:STDOUT:   %.loc703_36.4: %C2 = acquire_value %.loc703_36.3
// CHECK:STDOUT:   %c2: %C2 = value_binding c2, %.loc703_36.4
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c3.patt: %pattern_type.846 = value_binding_pattern c3 [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c1.ref.loc704: %C1 = name_ref c1, %c1
// CHECK:STDOUT:   %c2.ref.loc704: %C2 = name_ref c2, %c2
// CHECK:STDOUT:   %.loc704_33.1: ref %C2 = temporary_storage
// CHECK:STDOUT:   %.loc704_30: ref %C1 = value_as_ref %c1.ref.loc704
// CHECK:STDOUT:   %addr.loc704_33.1: %ptr.087 = addr_of %.loc704_30
// CHECK:STDOUT:   %.loc704_35: ref %C2 = value_as_ref %c2.ref.loc704
// CHECK:STDOUT:   %addr.loc704_33.2: %ptr.51f = addr_of %.loc704_35
// CHECK:STDOUT:   %addr.loc704_33.3: %ptr.51f = addr_of %.loc704_33.1
// CHECK:STDOUT:   %operator_Plus__carbon_thunk.call: init %empty_tuple.type = call imports.%operator_Plus__carbon_thunk.decl(%addr.loc704_33.1, %addr.loc704_33.2, %addr.loc704_33.3)
// CHECK:STDOUT:   %.loc704_33.2: init %C2 to %.loc704_33.1 = mark_in_place_init %operator_Plus__carbon_thunk.call
// CHECK:STDOUT:   %.loc704_24: type = splice_block %C2.ref.loc704 [concrete = constants.%C2] {
// CHECK:STDOUT:     %Cpp.ref.loc704: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %N2.ref.loc704: <namespace> = name_ref N2, imports.%N2 [concrete = imports.%N2]
// CHECK:STDOUT:     %C2.ref.loc704: type = name_ref C2, imports.%C2.decl [concrete = constants.%C2]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc704_33.3: ref %C2 = temporary %.loc704_33.1, %.loc704_33.2
// CHECK:STDOUT:   %.loc704_33.4: %C2 = acquire_value %.loc704_33.3
// CHECK:STDOUT:   %c3: %C2 = value_binding c3, %.loc704_33.4
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c4.patt: %pattern_type.846 = value_binding_pattern c4 [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c2.ref.loc705: %C2 = name_ref c2, %c2
// CHECK:STDOUT:   %c1.ref.loc705: %C1 = name_ref c1, %c1
// CHECK:STDOUT:   %.loc705_33.1: ref %C2 = temporary_storage
// CHECK:STDOUT:   %.loc705_30: ref %C2 = value_as_ref %c2.ref.loc705
// CHECK:STDOUT:   %addr.loc705_33.1: %ptr.51f = addr_of %.loc705_30
// CHECK:STDOUT:   %.loc705_35: ref %C1 = value_as_ref %c1.ref.loc705
// CHECK:STDOUT:   %addr.loc705_33.2: %ptr.087 = addr_of %.loc705_35
// CHECK:STDOUT:   %addr.loc705_33.3: %ptr.51f = addr_of %.loc705_33.1
// CHECK:STDOUT:   %operator_Minus__carbon_thunk.call: init %empty_tuple.type = call imports.%operator_Minus__carbon_thunk.decl(%addr.loc705_33.1, %addr.loc705_33.2, %addr.loc705_33.3)
// CHECK:STDOUT:   %.loc705_33.2: init %C2 to %.loc705_33.1 = mark_in_place_init %operator_Minus__carbon_thunk.call
// CHECK:STDOUT:   %.loc705_24: type = splice_block %C2.ref.loc705 [concrete = constants.%C2] {
// CHECK:STDOUT:     %Cpp.ref.loc705: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %N2.ref.loc705: <namespace> = name_ref N2, imports.%N2 [concrete = imports.%N2]
// CHECK:STDOUT:     %C2.ref.loc705: type = name_ref C2, imports.%C2.decl [concrete = constants.%C2]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc705_33.3: ref %C2 = temporary %.loc705_33.1, %.loc705_33.2
// CHECK:STDOUT:   %.loc705_33.4: %C2 = acquire_value %.loc705_33.3
// CHECK:STDOUT:   %c4: %C2 = value_binding c4, %.loc705_33.4
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %C2.Op.bound.loc705: <bound method> = bound_method %.loc705_33.3, constants.%C2.Op
// CHECK:STDOUT:   %Op.ref.loc705: %C2.cpp_destructor.type = name_ref Op, imports.%C2.cpp_destructor.decl [concrete = constants.%C2.cpp_destructor]
// CHECK:STDOUT:   %C2.cpp_destructor.bound.loc705: <bound method> = bound_method %.loc705_33.3, %Op.ref.loc705
// CHECK:STDOUT:   %C2.cpp_destructor.call.loc705: init %empty_tuple.type = call %C2.cpp_destructor.bound.loc705(%.loc705_33.3)
// CHECK:STDOUT:   %C2.Op.bound.loc704: <bound method> = bound_method %.loc704_33.3, constants.%C2.Op
// CHECK:STDOUT:   %Op.ref.loc704: %C2.cpp_destructor.type = name_ref Op, imports.%C2.cpp_destructor.decl [concrete = constants.%C2.cpp_destructor]
// CHECK:STDOUT:   %C2.cpp_destructor.bound.loc704: <bound method> = bound_method %.loc704_33.3, %Op.ref.loc704
// CHECK:STDOUT:   %C2.cpp_destructor.call.loc704: init %empty_tuple.type = call %C2.cpp_destructor.bound.loc704(%.loc704_33.3)
// CHECK:STDOUT:   %C2.Op.bound.loc703: <bound method> = bound_method %.loc703_36.3, constants.%C2.Op
// CHECK:STDOUT:   %Op.ref.loc703: %C2.cpp_destructor.type = name_ref Op, imports.%C2.cpp_destructor.decl [concrete = constants.%C2.cpp_destructor]
// CHECK:STDOUT:   %C2.cpp_destructor.bound.loc703: <bound method> = bound_method %.loc703_36.3, %Op.ref.loc703
// CHECK:STDOUT:   %C2.cpp_destructor.call.loc703: init %empty_tuple.type = call %C2.cpp_destructor.bound.loc703(%.loc703_36.3)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %C1.Op.bound: <bound method> = bound_method %.loc702_36.3, constants.%C1.Op
// CHECK:STDOUT:   %Op.ref.loc702: %C1.cpp_destructor.type = name_ref Op, imports.%C1.cpp_destructor.decl [concrete = constants.%C1.cpp_destructor]
// CHECK:STDOUT:   %C1.cpp_destructor.bound: <bound method> = bound_method %.loc702_36.3, %Op.ref.loc702
// CHECK:STDOUT:   %C1.cpp_destructor.call: init %empty_tuple.type = call %C1.cpp_destructor.bound(%.loc702_36.3)
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
// CHECK:STDOUT:     %self.param_patt: %pattern_type.69f = ref_param_pattern [concrete]
// CHECK:STDOUT:     %self.patt: %pattern_type.69f = at_binding_pattern self, %self.param_patt [concrete]
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
// CHECK:STDOUT:   %Cpp.ref.loc732_21: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %N.ref.loc732_24: <namespace> = name_ref N, imports.%N [concrete = imports.%N]
// CHECK:STDOUT:   %C.ref.loc732_26: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %C.ref.loc732_28: %C.C.cpp_overload_set.type = name_ref C, imports.%C.C.cpp_overload_set.value [concrete = constants.%C.C.cpp_overload_set.value]
// CHECK:STDOUT:   %.loc732_31.1: ref %C = temporary_storage
// CHECK:STDOUT:   %addr.loc732: %ptr.838 = addr_of %.loc732_31.1
// CHECK:STDOUT:   %C__carbon_thunk.call.loc732: init %empty_tuple.type = call imports.%C__carbon_thunk.decl(%addr.loc732)
// CHECK:STDOUT:   %.loc732_31.2: init %C to %.loc732_31.1 = mark_in_place_init %C__carbon_thunk.call.loc732
// CHECK:STDOUT:   %.loc732_16: type = splice_block %C.ref.loc732_16 [concrete = constants.%C] {
// CHECK:STDOUT:     %Cpp.ref.loc732_11: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %N.ref.loc732_14: <namespace> = name_ref N, imports.%N [concrete = imports.%N]
// CHECK:STDOUT:     %C.ref.loc732_16: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc732_31.3: ref %C = temporary %.loc732_31.1, %.loc732_31.2
// CHECK:STDOUT:   %.loc732_31.4: %C = acquire_value %.loc732_31.3
// CHECK:STDOUT:   %c1: %C = value_binding c1, %.loc732_31.4
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c2.patt: %pattern_type.69f = value_binding_pattern c2 [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc733_21: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %N.ref.loc733_24: <namespace> = name_ref N, imports.%N [concrete = imports.%N]
// CHECK:STDOUT:   %C.ref.loc733_26: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %C.ref.loc733_28: %C.C.cpp_overload_set.type = name_ref C, imports.%C.C.cpp_overload_set.value [concrete = constants.%C.C.cpp_overload_set.value]
// CHECK:STDOUT:   %.loc733_31.1: ref %C = temporary_storage
// CHECK:STDOUT:   %addr.loc733: %ptr.838 = addr_of %.loc733_31.1
// CHECK:STDOUT:   %C__carbon_thunk.call.loc733: init %empty_tuple.type = call imports.%C__carbon_thunk.decl(%addr.loc733)
// CHECK:STDOUT:   %.loc733_31.2: init %C to %.loc733_31.1 = mark_in_place_init %C__carbon_thunk.call.loc733
// CHECK:STDOUT:   %.loc733_16: type = splice_block %C.ref.loc733_16 [concrete = constants.%C] {
// CHECK:STDOUT:     %Cpp.ref.loc733_11: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %N.ref.loc733_14: <namespace> = name_ref N, imports.%N [concrete = imports.%N]
// CHECK:STDOUT:     %C.ref.loc733_16: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc733_31.3: ref %C = temporary %.loc733_31.1, %.loc733_31.2
// CHECK:STDOUT:   %.loc733_31.4: %C = acquire_value %.loc733_31.3
// CHECK:STDOUT:   %c2: %C = value_binding c2, %.loc733_31.4
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c3.patt: %pattern_type.69f = value_binding_pattern c3 [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c1.ref: %C = name_ref c1, %c1
// CHECK:STDOUT:   %c2.ref: %C = name_ref c2, %c2
// CHECK:STDOUT:   %.loc738: type = splice_block %C.ref.loc738 [concrete = constants.%C] {
// CHECK:STDOUT:     %Cpp.ref.loc738: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %N.ref.loc738: <namespace> = name_ref N, imports.%N [concrete = imports.%N]
// CHECK:STDOUT:     %C.ref.loc738: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c3: %C = value_binding c3, <error> [concrete = <error>]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %C.Op.bound.loc733: <bound method> = bound_method %.loc733_31.3, constants.%C.Op
// CHECK:STDOUT:   %Op.ref.loc733: %C.cpp_destructor.type = name_ref Op, imports.%C.cpp_destructor.decl [concrete = constants.%C.cpp_destructor]
// CHECK:STDOUT:   %C.cpp_destructor.bound.loc733: <bound method> = bound_method %.loc733_31.3, %Op.ref.loc733
// CHECK:STDOUT:   %C.cpp_destructor.call.loc733: init %empty_tuple.type = call %C.cpp_destructor.bound.loc733(%.loc733_31.3)
// CHECK:STDOUT:   %C.Op.bound.loc732: <bound method> = bound_method %.loc732_31.3, constants.%C.Op
// CHECK:STDOUT:   %Op.ref.loc732: %C.cpp_destructor.type = name_ref Op, imports.%C.cpp_destructor.decl [concrete = constants.%C.cpp_destructor]
// CHECK:STDOUT:   %C.cpp_destructor.bound.loc732: <bound method> = bound_method %.loc732_31.3, %Op.ref.loc732
// CHECK:STDOUT:   %C.cpp_destructor.call.loc732: init %empty_tuple.type = call %C.cpp_destructor.bound.loc732(%.loc732_31.3)
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
// CHECK:STDOUT:     %self.param_patt: %pattern_type.b28 = ref_param_pattern [concrete]
// CHECK:STDOUT:     %self.patt: %pattern_type.b28 = at_binding_pattern self, %self.param_patt [concrete]
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
// CHECK:STDOUT:   %Cpp.ref.loc762_21: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %O.ref.loc762_24: type = name_ref O, imports.%O.decl [concrete = constants.%O]
// CHECK:STDOUT:   %C.ref.loc762_26: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %C.ref.loc762_28: %C.C.cpp_overload_set.type = name_ref C, imports.%C.C.cpp_overload_set.value [concrete = constants.%C.C.cpp_overload_set.value]
// CHECK:STDOUT:   %.loc762_31.1: ref %C = temporary_storage
// CHECK:STDOUT:   %addr.loc762: %ptr.de2 = addr_of %.loc762_31.1
// CHECK:STDOUT:   %C__carbon_thunk.call.loc762: init %empty_tuple.type = call imports.%C__carbon_thunk.decl(%addr.loc762)
// CHECK:STDOUT:   %.loc762_31.2: init %C to %.loc762_31.1 = mark_in_place_init %C__carbon_thunk.call.loc762
// CHECK:STDOUT:   %.loc762_16: type = splice_block %C.ref.loc762_16 [concrete = constants.%C] {
// CHECK:STDOUT:     %Cpp.ref.loc762_11: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %O.ref.loc762_14: type = name_ref O, imports.%O.decl [concrete = constants.%O]
// CHECK:STDOUT:     %C.ref.loc762_16: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc762_31.3: ref %C = temporary %.loc762_31.1, %.loc762_31.2
// CHECK:STDOUT:   %.loc762_31.4: %C = acquire_value %.loc762_31.3
// CHECK:STDOUT:   %c1: %C = value_binding c1, %.loc762_31.4
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c2.patt: %pattern_type.b28 = value_binding_pattern c2 [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc763_21: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %O.ref.loc763_24: type = name_ref O, imports.%O.decl [concrete = constants.%O]
// CHECK:STDOUT:   %C.ref.loc763_26: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %C.ref.loc763_28: %C.C.cpp_overload_set.type = name_ref C, imports.%C.C.cpp_overload_set.value [concrete = constants.%C.C.cpp_overload_set.value]
// CHECK:STDOUT:   %.loc763_31.1: ref %C = temporary_storage
// CHECK:STDOUT:   %addr.loc763: %ptr.de2 = addr_of %.loc763_31.1
// CHECK:STDOUT:   %C__carbon_thunk.call.loc763: init %empty_tuple.type = call imports.%C__carbon_thunk.decl(%addr.loc763)
// CHECK:STDOUT:   %.loc763_31.2: init %C to %.loc763_31.1 = mark_in_place_init %C__carbon_thunk.call.loc763
// CHECK:STDOUT:   %.loc763_16: type = splice_block %C.ref.loc763_16 [concrete = constants.%C] {
// CHECK:STDOUT:     %Cpp.ref.loc763_11: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %O.ref.loc763_14: type = name_ref O, imports.%O.decl [concrete = constants.%O]
// CHECK:STDOUT:     %C.ref.loc763_16: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc763_31.3: ref %C = temporary %.loc763_31.1, %.loc763_31.2
// CHECK:STDOUT:   %.loc763_31.4: %C = acquire_value %.loc763_31.3
// CHECK:STDOUT:   %c2: %C = value_binding c2, %.loc763_31.4
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c3.patt: %pattern_type.b28 = value_binding_pattern c3 [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c1.ref: %C = name_ref c1, %c1
// CHECK:STDOUT:   %c2.ref: %C = name_ref c2, %c2
// CHECK:STDOUT:   %.loc764_31.1: ref %C = temporary_storage
// CHECK:STDOUT:   %.loc764_28: ref %C = value_as_ref %c1.ref
// CHECK:STDOUT:   %addr.loc764_31.1: %ptr.de2 = addr_of %.loc764_28
// CHECK:STDOUT:   %.loc764_33: ref %C = value_as_ref %c2.ref
// CHECK:STDOUT:   %addr.loc764_31.2: %ptr.de2 = addr_of %.loc764_33
// CHECK:STDOUT:   %addr.loc764_31.3: %ptr.de2 = addr_of %.loc764_31.1
// CHECK:STDOUT:   %operator_Plus__carbon_thunk.call: init %empty_tuple.type = call imports.%operator_Plus__carbon_thunk.decl(%addr.loc764_31.1, %addr.loc764_31.2, %addr.loc764_31.3)
// CHECK:STDOUT:   %.loc764_31.2: init %C to %.loc764_31.1 = mark_in_place_init %operator_Plus__carbon_thunk.call
// CHECK:STDOUT:   %.loc764_23: type = splice_block %C.ref.loc764 [concrete = constants.%C] {
// CHECK:STDOUT:     %Cpp.ref.loc764: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %O.ref.loc764: type = name_ref O, imports.%O.decl [concrete = constants.%O]
// CHECK:STDOUT:     %C.ref.loc764: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc764_31.3: ref %C = temporary %.loc764_31.1, %.loc764_31.2
// CHECK:STDOUT:   %.loc764_31.4: %C = acquire_value %.loc764_31.3
// CHECK:STDOUT:   %c3: %C = value_binding c3, %.loc764_31.4
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %C.Op.bound.loc764: <bound method> = bound_method %.loc764_31.3, constants.%C.Op
// CHECK:STDOUT:   %Op.ref.loc764: %C.cpp_destructor.type = name_ref Op, imports.%C.cpp_destructor.decl [concrete = constants.%C.cpp_destructor]
// CHECK:STDOUT:   %C.cpp_destructor.bound.loc764: <bound method> = bound_method %.loc764_31.3, %Op.ref.loc764
// CHECK:STDOUT:   %C.cpp_destructor.call.loc764: init %empty_tuple.type = call %C.cpp_destructor.bound.loc764(%.loc764_31.3)
// CHECK:STDOUT:   %C.Op.bound.loc763: <bound method> = bound_method %.loc763_31.3, constants.%C.Op
// CHECK:STDOUT:   %Op.ref.loc763: %C.cpp_destructor.type = name_ref Op, imports.%C.cpp_destructor.decl [concrete = constants.%C.cpp_destructor]
// CHECK:STDOUT:   %C.cpp_destructor.bound.loc763: <bound method> = bound_method %.loc763_31.3, %Op.ref.loc763
// CHECK:STDOUT:   %C.cpp_destructor.call.loc763: init %empty_tuple.type = call %C.cpp_destructor.bound.loc763(%.loc763_31.3)
// CHECK:STDOUT:   %C.Op.bound.loc762: <bound method> = bound_method %.loc762_31.3, constants.%C.Op
// CHECK:STDOUT:   %Op.ref.loc762: %C.cpp_destructor.type = name_ref Op, imports.%C.cpp_destructor.decl [concrete = constants.%C.cpp_destructor]
// CHECK:STDOUT:   %C.cpp_destructor.bound.loc762: <bound method> = bound_method %.loc762_31.3, %Op.ref.loc762
// CHECK:STDOUT:   %C.cpp_destructor.call.loc762: init %empty_tuple.type = call %C.cpp_destructor.bound.loc762(%.loc762_31.3)
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
// CHECK:STDOUT:     %self.param_patt: %pattern_type.84b = ref_param_pattern [concrete]
// CHECK:STDOUT:     %self.patt: %pattern_type.84b = at_binding_pattern self, %self.param_patt [concrete]
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
// CHECK:STDOUT:   %Cpp.ref.loc790_23: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %N.ref.loc790_26: <namespace> = name_ref N, imports.%N [concrete = imports.%N]
// CHECK:STDOUT:   %O.ref.loc790_28: type = name_ref O, imports.%O.decl [concrete = constants.%O]
// CHECK:STDOUT:   %C.ref.loc790_30: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %C.ref.loc790_32: %C.C.cpp_overload_set.type = name_ref C, imports.%C.C.cpp_overload_set.value [concrete = constants.%C.C.cpp_overload_set.value]
// CHECK:STDOUT:   %.loc790_35.1: ref %C = temporary_storage
// CHECK:STDOUT:   %addr.loc790: %ptr.4b2 = addr_of %.loc790_35.1
// CHECK:STDOUT:   %C__carbon_thunk.call.loc790: init %empty_tuple.type = call imports.%C__carbon_thunk.decl(%addr.loc790)
// CHECK:STDOUT:   %.loc790_35.2: init %C to %.loc790_35.1 = mark_in_place_init %C__carbon_thunk.call.loc790
// CHECK:STDOUT:   %.loc790_18: type = splice_block %C.ref.loc790_18 [concrete = constants.%C] {
// CHECK:STDOUT:     %Cpp.ref.loc790_11: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %N.ref.loc790_14: <namespace> = name_ref N, imports.%N [concrete = imports.%N]
// CHECK:STDOUT:     %O.ref.loc790_16: type = name_ref O, imports.%O.decl [concrete = constants.%O]
// CHECK:STDOUT:     %C.ref.loc790_18: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc790_35.3: ref %C = temporary %.loc790_35.1, %.loc790_35.2
// CHECK:STDOUT:   %.loc790_35.4: %C = acquire_value %.loc790_35.3
// CHECK:STDOUT:   %c1: %C = value_binding c1, %.loc790_35.4
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c2.patt: %pattern_type.84b = value_binding_pattern c2 [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc791_23: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %N.ref.loc791_26: <namespace> = name_ref N, imports.%N [concrete = imports.%N]
// CHECK:STDOUT:   %O.ref.loc791_28: type = name_ref O, imports.%O.decl [concrete = constants.%O]
// CHECK:STDOUT:   %C.ref.loc791_30: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %C.ref.loc791_32: %C.C.cpp_overload_set.type = name_ref C, imports.%C.C.cpp_overload_set.value [concrete = constants.%C.C.cpp_overload_set.value]
// CHECK:STDOUT:   %.loc791_35.1: ref %C = temporary_storage
// CHECK:STDOUT:   %addr.loc791: %ptr.4b2 = addr_of %.loc791_35.1
// CHECK:STDOUT:   %C__carbon_thunk.call.loc791: init %empty_tuple.type = call imports.%C__carbon_thunk.decl(%addr.loc791)
// CHECK:STDOUT:   %.loc791_35.2: init %C to %.loc791_35.1 = mark_in_place_init %C__carbon_thunk.call.loc791
// CHECK:STDOUT:   %.loc791_18: type = splice_block %C.ref.loc791_18 [concrete = constants.%C] {
// CHECK:STDOUT:     %Cpp.ref.loc791_11: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %N.ref.loc791_14: <namespace> = name_ref N, imports.%N [concrete = imports.%N]
// CHECK:STDOUT:     %O.ref.loc791_16: type = name_ref O, imports.%O.decl [concrete = constants.%O]
// CHECK:STDOUT:     %C.ref.loc791_18: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc791_35.3: ref %C = temporary %.loc791_35.1, %.loc791_35.2
// CHECK:STDOUT:   %.loc791_35.4: %C = acquire_value %.loc791_35.3
// CHECK:STDOUT:   %c2: %C = value_binding c2, %.loc791_35.4
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c3.patt: %pattern_type.84b = value_binding_pattern c3 [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c1.ref: %C = name_ref c1, %c1
// CHECK:STDOUT:   %c2.ref: %C = name_ref c2, %c2
// CHECK:STDOUT:   %.loc792_33.1: ref %C = temporary_storage
// CHECK:STDOUT:   %.loc792_30: ref %C = value_as_ref %c1.ref
// CHECK:STDOUT:   %addr.loc792_33.1: %ptr.4b2 = addr_of %.loc792_30
// CHECK:STDOUT:   %.loc792_35: ref %C = value_as_ref %c2.ref
// CHECK:STDOUT:   %addr.loc792_33.2: %ptr.4b2 = addr_of %.loc792_35
// CHECK:STDOUT:   %addr.loc792_33.3: %ptr.4b2 = addr_of %.loc792_33.1
// CHECK:STDOUT:   %operator_Plus__carbon_thunk.call: init %empty_tuple.type = call imports.%operator_Plus__carbon_thunk.decl(%addr.loc792_33.1, %addr.loc792_33.2, %addr.loc792_33.3)
// CHECK:STDOUT:   %.loc792_33.2: init %C to %.loc792_33.1 = mark_in_place_init %operator_Plus__carbon_thunk.call
// CHECK:STDOUT:   %.loc792_25: type = splice_block %C.ref.loc792 [concrete = constants.%C] {
// CHECK:STDOUT:     %Cpp.ref.loc792: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %N.ref.loc792: <namespace> = name_ref N, imports.%N [concrete = imports.%N]
// CHECK:STDOUT:     %O.ref.loc792: type = name_ref O, imports.%O.decl [concrete = constants.%O]
// CHECK:STDOUT:     %C.ref.loc792: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc792_33.3: ref %C = temporary %.loc792_33.1, %.loc792_33.2
// CHECK:STDOUT:   %.loc792_33.4: %C = acquire_value %.loc792_33.3
// CHECK:STDOUT:   %c3: %C = value_binding c3, %.loc792_33.4
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %C.Op.bound.loc792: <bound method> = bound_method %.loc792_33.3, constants.%C.Op
// CHECK:STDOUT:   %Op.ref.loc792: %C.cpp_destructor.type = name_ref Op, imports.%C.cpp_destructor.decl [concrete = constants.%C.cpp_destructor]
// CHECK:STDOUT:   %C.cpp_destructor.bound.loc792: <bound method> = bound_method %.loc792_33.3, %Op.ref.loc792
// CHECK:STDOUT:   %C.cpp_destructor.call.loc792: init %empty_tuple.type = call %C.cpp_destructor.bound.loc792(%.loc792_33.3)
// CHECK:STDOUT:   %C.Op.bound.loc791: <bound method> = bound_method %.loc791_35.3, constants.%C.Op
// CHECK:STDOUT:   %Op.ref.loc791: %C.cpp_destructor.type = name_ref Op, imports.%C.cpp_destructor.decl [concrete = constants.%C.cpp_destructor]
// CHECK:STDOUT:   %C.cpp_destructor.bound.loc791: <bound method> = bound_method %.loc791_35.3, %Op.ref.loc791
// CHECK:STDOUT:   %C.cpp_destructor.call.loc791: init %empty_tuple.type = call %C.cpp_destructor.bound.loc791(%.loc791_35.3)
// CHECK:STDOUT:   %C.Op.bound.loc790: <bound method> = bound_method %.loc790_35.3, constants.%C.Op
// CHECK:STDOUT:   %Op.ref.loc790: %C.cpp_destructor.type = name_ref Op, imports.%C.cpp_destructor.decl [concrete = constants.%C.cpp_destructor]
// CHECK:STDOUT:   %C.cpp_destructor.bound.loc790: <bound method> = bound_method %.loc790_35.3, %Op.ref.loc790
// CHECK:STDOUT:   %C.cpp_destructor.call.loc790: init %empty_tuple.type = call %C.cpp_destructor.bound.loc790(%.loc790_35.3)
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
// CHECK:STDOUT:     %self.param_patt: %pattern_type.217 = ref_param_pattern [concrete]
// CHECK:STDOUT:     %self.patt: %pattern_type.217 = at_binding_pattern self, %self.param_patt [concrete]
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
// CHECK:STDOUT:     %self.param_patt: %pattern_type.217 = ref_param_pattern [concrete]
// CHECK:STDOUT:     %self.patt: %pattern_type.217 = at_binding_pattern self, %self.param_patt [concrete]
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
// CHECK:STDOUT:     %self.param_patt: %pattern_type.217 = ref_param_pattern [concrete]
// CHECK:STDOUT:     %self.patt: %pattern_type.217 = at_binding_pattern self, %self.param_patt [concrete]
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
// CHECK:STDOUT:   %Cpp.ref.loc819_19: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %C.ref.loc819_22: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %C.ref.loc819_24: %C.C.cpp_overload_set.type = name_ref C, imports.%C.C.cpp_overload_set.value [concrete = constants.%C.C.cpp_overload_set.value]
// CHECK:STDOUT:   %.loc819_3: ref %C = splice_block %c1.var {}
// CHECK:STDOUT:   %addr.loc819: %ptr.d9e = addr_of %.loc819_3
// CHECK:STDOUT:   %C__carbon_thunk.call: init %empty_tuple.type = call imports.%C__carbon_thunk.decl(%addr.loc819)
// CHECK:STDOUT:   %.loc819_27: init %C to %.loc819_3 = mark_in_place_init %C__carbon_thunk.call
// CHECK:STDOUT:   assign %c1.var, %.loc819_27
// CHECK:STDOUT:   %.loc819_14: type = splice_block %C.ref.loc819_14 [concrete = constants.%C] {
// CHECK:STDOUT:     %Cpp.ref.loc819_11: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %C.ref.loc819_14: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c1: ref %C = ref_binding c1, %c1.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c2.patt: %pattern_type.217 = ref_binding_pattern c2 [concrete]
// CHECK:STDOUT:     %c2.var_patt: %pattern_type.217 = var_pattern %c2.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c2.var: ref %C = var %c2.var_patt
// CHECK:STDOUT:   %c1.ref.loc820: ref %C = name_ref c1, %c1
// CHECK:STDOUT:   %C.cpp_operator.bound.loc820: <bound method> = bound_method %c1.ref.loc820, imports.%C.cpp_operator.decl.828f43.1
// CHECK:STDOUT:   %.loc820_3: ref %C = splice_block %c2.var {}
// CHECK:STDOUT:   %addr.loc820: %ptr.d9e = addr_of %.loc820_3
// CHECK:STDOUT:   %operator_Minus__carbon_thunk.call: init %empty_tuple.type = call imports.%operator_Minus__carbon_thunk.decl(%c1.ref.loc820, %addr.loc820)
// CHECK:STDOUT:   %.loc820_19: init %C to %.loc820_3 = mark_in_place_init %operator_Minus__carbon_thunk.call
// CHECK:STDOUT:   assign %c2.var, %.loc820_19
// CHECK:STDOUT:   %.loc820_14: type = splice_block %C.ref.loc820 [concrete = constants.%C] {
// CHECK:STDOUT:     %Cpp.ref.loc820: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %C.ref.loc820: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c2: ref %C = ref_binding c2, %c2.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c3.patt: %pattern_type.217 = ref_binding_pattern c3 [concrete]
// CHECK:STDOUT:     %c3.var_patt: %pattern_type.217 = var_pattern %c3.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c3.var: ref %C = var %c3.var_patt
// CHECK:STDOUT:   %c1.ref.loc821: ref %C = name_ref c1, %c1
// CHECK:STDOUT:   %c2.ref: ref %C = name_ref c2, %c2
// CHECK:STDOUT:   %C.cpp_operator.bound.loc821: <bound method> = bound_method %c1.ref.loc821, imports.%C.cpp_operator.decl.828f43.2
// CHECK:STDOUT:   %.loc821_3: ref %C = splice_block %c3.var {}
// CHECK:STDOUT:   %.loc821_31.1: %C = acquire_value %c2.ref
// CHECK:STDOUT:   %.loc821_31.2: ref %C = value_as_ref %.loc821_31.1
// CHECK:STDOUT:   %addr.loc821_29.1: %ptr.d9e = addr_of %.loc821_31.2
// CHECK:STDOUT:   %addr.loc821_29.2: %ptr.d9e = addr_of %.loc821_3
// CHECK:STDOUT:   %operator_Plus__carbon_thunk.call: init %empty_tuple.type = call imports.%operator_Plus__carbon_thunk.decl(%c1.ref.loc821, %addr.loc821_29.1, %addr.loc821_29.2)
// CHECK:STDOUT:   %.loc821_29: init %C to %.loc821_3 = mark_in_place_init %operator_Plus__carbon_thunk.call
// CHECK:STDOUT:   assign %c3.var, %.loc821_29
// CHECK:STDOUT:   %.loc821_21: type = splice_block %C.ref.loc821 [concrete = constants.%C] {
// CHECK:STDOUT:     %Cpp.ref.loc821: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %C.ref.loc821: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c3: ref %C = ref_binding c3, %c3.var
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %C.Op.bound.loc821: <bound method> = bound_method %c3.var, constants.%C.Op
// CHECK:STDOUT:   %Op.ref.loc821: %C.cpp_destructor.type = name_ref Op, imports.%C.cpp_destructor.decl [concrete = constants.%C.cpp_destructor]
// CHECK:STDOUT:   %C.cpp_destructor.bound.loc821: <bound method> = bound_method %c3.var, %Op.ref.loc821
// CHECK:STDOUT:   %C.cpp_destructor.call.loc821: init %empty_tuple.type = call %C.cpp_destructor.bound.loc821(%c3.var)
// CHECK:STDOUT:   %C.Op.bound.loc820: <bound method> = bound_method %c2.var, constants.%C.Op
// CHECK:STDOUT:   %Op.ref.loc820: %C.cpp_destructor.type = name_ref Op, imports.%C.cpp_destructor.decl [concrete = constants.%C.cpp_destructor]
// CHECK:STDOUT:   %C.cpp_destructor.bound.loc820: <bound method> = bound_method %c2.var, %Op.ref.loc820
// CHECK:STDOUT:   %C.cpp_destructor.call.loc820: init %empty_tuple.type = call %C.cpp_destructor.bound.loc820(%c2.var)
// CHECK:STDOUT:   %C.Op.bound.loc819: <bound method> = bound_method %c1.var, constants.%C.Op
// CHECK:STDOUT:   %Op.ref.loc819: %C.cpp_destructor.type = name_ref Op, imports.%C.cpp_destructor.decl [concrete = constants.%C.cpp_destructor]
// CHECK:STDOUT:   %C.cpp_destructor.bound.loc819: <bound method> = bound_method %c1.var, %Op.ref.loc819
// CHECK:STDOUT:   %C.cpp_destructor.call.loc819: init %empty_tuple.type = call %C.cpp_destructor.bound.loc819(%c1.var)
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
// CHECK:STDOUT:   %x.ref.loc974_10: %B = name_ref x, %x
// CHECK:STDOUT:   %x.ref.loc974_14: %B = name_ref x, %x
// CHECK:STDOUT:   %.loc974_10: ref %B = value_as_ref %x.ref.loc974_10
// CHECK:STDOUT:   %addr.loc974_12.1: %ptr.a04 = addr_of %.loc974_10
// CHECK:STDOUT:   %.loc974_14: ref %B = value_as_ref %x.ref.loc974_14
// CHECK:STDOUT:   %addr.loc974_12.2: %ptr.a04 = addr_of %.loc974_14
// CHECK:STDOUT:   %operator_Plus__carbon_thunk.call: init %i32 = call imports.%operator_Plus__carbon_thunk.decl(%addr.loc974_12.1, %addr.loc974_12.2)
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
// CHECK:STDOUT:     %self.param_patt: %pattern_type.217 = ref_param_pattern [concrete]
// CHECK:STDOUT:     %self.patt: %pattern_type.217 = at_binding_pattern self, %self.param_patt [concrete]
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
// CHECK:STDOUT:   %c1.ref: %C = name_ref c1, %c1
// CHECK:STDOUT:   %c2.ref: %C = name_ref c2, %c2
// CHECK:STDOUT:   %.loc1047_29.1: ref %C = temporary_storage
// CHECK:STDOUT:   %.loc1047_26: ref %C = value_as_ref %c1.ref
// CHECK:STDOUT:   %addr.loc1047_29.1: %ptr.d9e = addr_of %.loc1047_26
// CHECK:STDOUT:   %.loc1047_31: ref %C = value_as_ref %c2.ref
// CHECK:STDOUT:   %addr.loc1047_29.2: %ptr.d9e = addr_of %.loc1047_31
// CHECK:STDOUT:   %addr.loc1047_29.3: %ptr.d9e = addr_of %.loc1047_29.1
// CHECK:STDOUT:   %operator_Plus__carbon_thunk.call: init %empty_tuple.type = call imports.%operator_Plus__carbon_thunk.decl(%addr.loc1047_29.1, %addr.loc1047_29.2, %addr.loc1047_29.3)
// CHECK:STDOUT:   %.loc1047_29.2: init %C to %.loc1047_29.1 = mark_in_place_init %operator_Plus__carbon_thunk.call
// CHECK:STDOUT:   %.loc1047_21: type = splice_block %C.ref.loc1047 [concrete = constants.%C] {
// CHECK:STDOUT:     %Cpp.ref.loc1047: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %C.ref.loc1047: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc1047_29.3: ref %C = temporary %.loc1047_29.1, %.loc1047_29.2
// CHECK:STDOUT:   %.loc1047_29.4: %C = acquire_value %.loc1047_29.3
// CHECK:STDOUT:   %c3: %C = value_binding c3, %.loc1047_29.4
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %C.Op.bound.loc1047: <bound method> = bound_method %.loc1047_29.3, constants.%C.Op
// CHECK:STDOUT:   %Op.ref.loc1047: %C.cpp_destructor.type = name_ref Op, imports.%C.cpp_destructor.decl [concrete = constants.%C.cpp_destructor]
// CHECK:STDOUT:   %C.cpp_destructor.bound.loc1047: <bound method> = bound_method %.loc1047_29.3, %Op.ref.loc1047
// CHECK:STDOUT:   %C.cpp_destructor.call.loc1047: init %empty_tuple.type = call %C.cpp_destructor.bound.loc1047(%.loc1047_29.3)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
