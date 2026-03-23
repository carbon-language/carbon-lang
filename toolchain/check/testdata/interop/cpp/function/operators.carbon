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
// CHECK:STDOUT:   %Cpp.ref.loc21: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %C.ref.loc22: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %C.ref.loc24: %C.C.cpp_overload_set.type = name_ref C, imports.%C.C.cpp_overload_set.value [concrete = constants.%C.C.cpp_overload_set.value]
// CHECK:STDOUT:   %.loc14: ref %C = splice_block %c.var {}
// CHECK:STDOUT:   %addr.loc27: %ptr.d9e = addr_of %.loc14
// CHECK:STDOUT:   %C__carbon_thunk.call: init %empty_tuple.type = call imports.%C__carbon_thunk.decl(%addr.loc27)
// CHECK:STDOUT:   %.loc27: init %C to %.loc14 = mark_in_place_init %C__carbon_thunk.call
// CHECK:STDOUT:   assign %c.var, %.loc27
// CHECK:STDOUT:   %.loc18: type = splice_block %C.ref.loc18 [concrete = constants.%C] {
// CHECK:STDOUT:     %Cpp.ref.loc17: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %C.ref.loc18: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c: ref %C = ref_binding c, %c.var
// CHECK:STDOUT:   %c.ref.loc30: ref %C = name_ref c, %c
// CHECK:STDOUT:   %cpp_operator.call.loc29: ref %C = call imports.%cpp_operator.decl.4206c9.1(%c.ref.loc30)
// CHECK:STDOUT:   %c.ref.loc33: ref %C = name_ref c, %c
// CHECK:STDOUT:   %cpp_operator.call.loc32: ref %C = call imports.%cpp_operator.decl.4206c9.2(%c.ref.loc33)
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %minus.patt: %pattern_type.217 = value_binding_pattern minus [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c.ref.loc44: ref %C = name_ref c, %c
// CHECK:STDOUT:   %.loc43.1: ref %C = temporary_storage
// CHECK:STDOUT:   %.loc44.1: %C = acquire_value %c.ref.loc44
// CHECK:STDOUT:   %.loc44.2: ref %C = value_as_ref %.loc44.1
// CHECK:STDOUT:   %addr.loc43.1: %ptr.d9e = addr_of %.loc44.2
// CHECK:STDOUT:   %addr.loc43.2: %ptr.d9e = addr_of %.loc43.1
// CHECK:STDOUT:   %operator_Minus__carbon_thunk.call: init %empty_tuple.type = call imports.%operator_Minus__carbon_thunk.decl(%addr.loc43.1, %addr.loc43.2)
// CHECK:STDOUT:   %.loc43.2: init %C to %.loc43.1 = mark_in_place_init %operator_Minus__carbon_thunk.call
// CHECK:STDOUT:   %.loc40: type = splice_block %C.ref.loc40 [concrete = constants.%C] {
// CHECK:STDOUT:     %Cpp.ref.loc39: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %C.ref.loc40: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc43.3: ref %C = temporary %.loc43.1, %.loc43.2
// CHECK:STDOUT:   %.loc43.4: %C = acquire_value %.loc43.3
// CHECK:STDOUT:   %minus: %C = value_binding minus, %.loc43.4
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %complement.patt: %pattern_type.217 = value_binding_pattern complement [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c.ref.loc55: ref %C = name_ref c, %c
// CHECK:STDOUT:   %.loc54.1: ref %C = temporary_storage
// CHECK:STDOUT:   %.loc55.1: %C = acquire_value %c.ref.loc55
// CHECK:STDOUT:   %.loc55.2: ref %C = value_as_ref %.loc55.1
// CHECK:STDOUT:   %addr.loc54.1: %ptr.d9e = addr_of %.loc55.2
// CHECK:STDOUT:   %addr.loc54.2: %ptr.d9e = addr_of %.loc54.1
// CHECK:STDOUT:   %operator_Tilde__carbon_thunk.call: init %empty_tuple.type = call imports.%operator_Tilde__carbon_thunk.decl(%addr.loc54.1, %addr.loc54.2)
// CHECK:STDOUT:   %.loc54.2: init %C to %.loc54.1 = mark_in_place_init %operator_Tilde__carbon_thunk.call
// CHECK:STDOUT:   %.loc51: type = splice_block %C.ref.loc51 [concrete = constants.%C] {
// CHECK:STDOUT:     %Cpp.ref.loc50: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %C.ref.loc51: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc54.3: ref %C = temporary %.loc54.1, %.loc54.2
// CHECK:STDOUT:   %.loc54.4: %C = acquire_value %.loc54.3
// CHECK:STDOUT:   %complement: %C = value_binding complement, %.loc54.4
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %C.Op.bound.loc54: <bound method> = bound_method %.loc54.3, constants.%C.Op
// CHECK:STDOUT:   %Op.ref.loc54: %C.cpp_destructor.type = name_ref Op, imports.%C.cpp_destructor.decl [concrete = constants.%C.cpp_destructor]
// CHECK:STDOUT:   %C.cpp_destructor.bound.loc54: <bound method> = bound_method %.loc54.3, %Op.ref.loc54
// CHECK:STDOUT:   %C.cpp_destructor.call.loc54: init %empty_tuple.type = call %C.cpp_destructor.bound.loc54(%.loc54.3)
// CHECK:STDOUT:   %C.Op.bound.loc43: <bound method> = bound_method %.loc43.3, constants.%C.Op
// CHECK:STDOUT:   %Op.ref.loc43: %C.cpp_destructor.type = name_ref Op, imports.%C.cpp_destructor.decl [concrete = constants.%C.cpp_destructor]
// CHECK:STDOUT:   %C.cpp_destructor.bound.loc43: <bound method> = bound_method %.loc43.3, %Op.ref.loc43
// CHECK:STDOUT:   %C.cpp_destructor.call.loc43: init %empty_tuple.type = call %C.cpp_destructor.bound.loc43(%.loc43.3)
// CHECK:STDOUT:   %C.Op.bound.loc14: <bound method> = bound_method %c.var, constants.%C.Op
// CHECK:STDOUT:   %Op.ref.loc14: %C.cpp_destructor.type = name_ref Op, imports.%C.cpp_destructor.decl [concrete = constants.%C.cpp_destructor]
// CHECK:STDOUT:   %C.cpp_destructor.bound.loc14: <bound method> = bound_method %c.var, %Op.ref.loc14
// CHECK:STDOUT:   %C.cpp_destructor.call.loc14: init %empty_tuple.type = call %C.cpp_destructor.bound.loc14(%c.var)
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
// CHECK:STDOUT:   %Core.import_ref.42d: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.4e6) = import_ref Core//prelude/types/int, loc141, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3c2)]
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
// CHECK:STDOUT:   %Cpp.ref.loc21: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %C.ref.loc22: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %C.ref.loc24: %C.C.cpp_overload_set.type = name_ref C, imports.%C.C.cpp_overload_set.value [concrete = constants.%C.C.cpp_overload_set.value]
// CHECK:STDOUT:   %.loc14: ref %C = splice_block %c1.var {}
// CHECK:STDOUT:   %addr.loc27: %ptr.d9e = addr_of %.loc14
// CHECK:STDOUT:   %C__carbon_thunk.call.loc27: init %empty_tuple.type = call imports.%C__carbon_thunk.decl(%addr.loc27)
// CHECK:STDOUT:   %.loc27: init %C to %.loc14 = mark_in_place_init %C__carbon_thunk.call.loc27
// CHECK:STDOUT:   assign %c1.var, %.loc27
// CHECK:STDOUT:   %.loc18: type = splice_block %C.ref.loc18 [concrete = constants.%C] {
// CHECK:STDOUT:     %Cpp.ref.loc17: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %C.ref.loc18: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c1: ref %C = ref_binding c1, %c1.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c2.patt: %pattern_type.217 = ref_binding_pattern c2 [concrete]
// CHECK:STDOUT:     %c2.var_patt: %pattern_type.217 = var_pattern %c2.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c2.var: ref %C = var %c2.var_patt
// CHECK:STDOUT:   %Cpp.ref.loc36: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %C.ref.loc37: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %C.ref.loc39: %C.C.cpp_overload_set.type = name_ref C, imports.%C.C.cpp_overload_set.value [concrete = constants.%C.C.cpp_overload_set.value]
// CHECK:STDOUT:   %.loc29: ref %C = splice_block %c2.var {}
// CHECK:STDOUT:   %addr.loc42: %ptr.d9e = addr_of %.loc29
// CHECK:STDOUT:   %C__carbon_thunk.call.loc42: init %empty_tuple.type = call imports.%C__carbon_thunk.decl(%addr.loc42)
// CHECK:STDOUT:   %.loc42: init %C to %.loc29 = mark_in_place_init %C__carbon_thunk.call.loc42
// CHECK:STDOUT:   assign %c2.var, %.loc42
// CHECK:STDOUT:   %.loc33: type = splice_block %C.ref.loc33 [concrete = constants.%C] {
// CHECK:STDOUT:     %Cpp.ref.loc32: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %C.ref.loc33: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c2: ref %C = ref_binding c2, %c2.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %addition.patt: %pattern_type.217 = value_binding_pattern addition [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c1.ref.loc52: ref %C = name_ref c1, %c1
// CHECK:STDOUT:   %c2.ref.loc54: ref %C = name_ref c2, %c2
// CHECK:STDOUT:   %.loc53.1: ref %C = temporary_storage
// CHECK:STDOUT:   %.loc52.1: %C = acquire_value %c1.ref.loc52
// CHECK:STDOUT:   %.loc54.1: %C = acquire_value %c2.ref.loc54
// CHECK:STDOUT:   %.loc52.2: ref %C = value_as_ref %.loc52.1
// CHECK:STDOUT:   %addr.loc53.1: %ptr.d9e = addr_of %.loc52.2
// CHECK:STDOUT:   %.loc54.2: ref %C = value_as_ref %.loc54.1
// CHECK:STDOUT:   %addr.loc53.2: %ptr.d9e = addr_of %.loc54.2
// CHECK:STDOUT:   %addr.loc53.3: %ptr.d9e = addr_of %.loc53.1
// CHECK:STDOUT:   %operator_Plus__carbon_thunk.call: init %empty_tuple.type = call imports.%operator_Plus__carbon_thunk.decl(%addr.loc53.1, %addr.loc53.2, %addr.loc53.3)
// CHECK:STDOUT:   %.loc53.2: init %C to %.loc53.1 = mark_in_place_init %operator_Plus__carbon_thunk.call
// CHECK:STDOUT:   %.loc49: type = splice_block %C.ref.loc49 [concrete = constants.%C] {
// CHECK:STDOUT:     %Cpp.ref.loc48: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %C.ref.loc49: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc53.3: ref %C = temporary %.loc53.1, %.loc53.2
// CHECK:STDOUT:   %.loc53.4: %C = acquire_value %.loc53.3
// CHECK:STDOUT:   %addition: %C = value_binding addition, %.loc53.4
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %subtraction.patt: %pattern_type.217 = value_binding_pattern subtraction [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c1.ref.loc64: ref %C = name_ref c1, %c1
// CHECK:STDOUT:   %c2.ref.loc66: ref %C = name_ref c2, %c2
// CHECK:STDOUT:   %.loc65.1: ref %C = temporary_storage
// CHECK:STDOUT:   %.loc64.1: %C = acquire_value %c1.ref.loc64
// CHECK:STDOUT:   %.loc66.1: %C = acquire_value %c2.ref.loc66
// CHECK:STDOUT:   %.loc64.2: ref %C = value_as_ref %.loc64.1
// CHECK:STDOUT:   %addr.loc65.1: %ptr.d9e = addr_of %.loc64.2
// CHECK:STDOUT:   %.loc66.2: ref %C = value_as_ref %.loc66.1
// CHECK:STDOUT:   %addr.loc65.2: %ptr.d9e = addr_of %.loc66.2
// CHECK:STDOUT:   %addr.loc65.3: %ptr.d9e = addr_of %.loc65.1
// CHECK:STDOUT:   %operator_Minus__carbon_thunk.call: init %empty_tuple.type = call imports.%operator_Minus__carbon_thunk.decl(%addr.loc65.1, %addr.loc65.2, %addr.loc65.3)
// CHECK:STDOUT:   %.loc65.2: init %C to %.loc65.1 = mark_in_place_init %operator_Minus__carbon_thunk.call
// CHECK:STDOUT:   %.loc61: type = splice_block %C.ref.loc61 [concrete = constants.%C] {
// CHECK:STDOUT:     %Cpp.ref.loc60: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %C.ref.loc61: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc65.3: ref %C = temporary %.loc65.1, %.loc65.2
// CHECK:STDOUT:   %.loc65.4: %C = acquire_value %.loc65.3
// CHECK:STDOUT:   %subtraction: %C = value_binding subtraction, %.loc65.4
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %multiplication.patt: %pattern_type.217 = value_binding_pattern multiplication [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c1.ref.loc76: ref %C = name_ref c1, %c1
// CHECK:STDOUT:   %c2.ref.loc78: ref %C = name_ref c2, %c2
// CHECK:STDOUT:   %.loc77.1: ref %C = temporary_storage
// CHECK:STDOUT:   %.loc76.1: %C = acquire_value %c1.ref.loc76
// CHECK:STDOUT:   %.loc78.1: %C = acquire_value %c2.ref.loc78
// CHECK:STDOUT:   %.loc76.2: ref %C = value_as_ref %.loc76.1
// CHECK:STDOUT:   %addr.loc77.1: %ptr.d9e = addr_of %.loc76.2
// CHECK:STDOUT:   %.loc78.2: ref %C = value_as_ref %.loc78.1
// CHECK:STDOUT:   %addr.loc77.2: %ptr.d9e = addr_of %.loc78.2
// CHECK:STDOUT:   %addr.loc77.3: %ptr.d9e = addr_of %.loc77.1
// CHECK:STDOUT:   %operator_Star__carbon_thunk.call: init %empty_tuple.type = call imports.%operator_Star__carbon_thunk.decl(%addr.loc77.1, %addr.loc77.2, %addr.loc77.3)
// CHECK:STDOUT:   %.loc77.2: init %C to %.loc77.1 = mark_in_place_init %operator_Star__carbon_thunk.call
// CHECK:STDOUT:   %.loc73: type = splice_block %C.ref.loc73 [concrete = constants.%C] {
// CHECK:STDOUT:     %Cpp.ref.loc72: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %C.ref.loc73: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc77.3: ref %C = temporary %.loc77.1, %.loc77.2
// CHECK:STDOUT:   %.loc77.4: %C = acquire_value %.loc77.3
// CHECK:STDOUT:   %multiplication: %C = value_binding multiplication, %.loc77.4
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %division.patt: %pattern_type.217 = value_binding_pattern division [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c1.ref.loc88: ref %C = name_ref c1, %c1
// CHECK:STDOUT:   %c2.ref.loc90: ref %C = name_ref c2, %c2
// CHECK:STDOUT:   %.loc89.1: ref %C = temporary_storage
// CHECK:STDOUT:   %.loc88.1: %C = acquire_value %c1.ref.loc88
// CHECK:STDOUT:   %.loc90.1: %C = acquire_value %c2.ref.loc90
// CHECK:STDOUT:   %.loc88.2: ref %C = value_as_ref %.loc88.1
// CHECK:STDOUT:   %addr.loc89.1: %ptr.d9e = addr_of %.loc88.2
// CHECK:STDOUT:   %.loc90.2: ref %C = value_as_ref %.loc90.1
// CHECK:STDOUT:   %addr.loc89.2: %ptr.d9e = addr_of %.loc90.2
// CHECK:STDOUT:   %addr.loc89.3: %ptr.d9e = addr_of %.loc89.1
// CHECK:STDOUT:   %operator_Slash__carbon_thunk.call: init %empty_tuple.type = call imports.%operator_Slash__carbon_thunk.decl(%addr.loc89.1, %addr.loc89.2, %addr.loc89.3)
// CHECK:STDOUT:   %.loc89.2: init %C to %.loc89.1 = mark_in_place_init %operator_Slash__carbon_thunk.call
// CHECK:STDOUT:   %.loc85: type = splice_block %C.ref.loc85 [concrete = constants.%C] {
// CHECK:STDOUT:     %Cpp.ref.loc84: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %C.ref.loc85: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc89.3: ref %C = temporary %.loc89.1, %.loc89.2
// CHECK:STDOUT:   %.loc89.4: %C = acquire_value %.loc89.3
// CHECK:STDOUT:   %division: %C = value_binding division, %.loc89.4
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %modulo.patt: %pattern_type.217 = value_binding_pattern modulo [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c1.ref.loc100: ref %C = name_ref c1, %c1
// CHECK:STDOUT:   %c2.ref.loc102: ref %C = name_ref c2, %c2
// CHECK:STDOUT:   %.loc101.1: ref %C = temporary_storage
// CHECK:STDOUT:   %.loc100.1: %C = acquire_value %c1.ref.loc100
// CHECK:STDOUT:   %.loc102.1: %C = acquire_value %c2.ref.loc102
// CHECK:STDOUT:   %.loc100.2: ref %C = value_as_ref %.loc100.1
// CHECK:STDOUT:   %addr.loc101.1: %ptr.d9e = addr_of %.loc100.2
// CHECK:STDOUT:   %.loc102.2: ref %C = value_as_ref %.loc102.1
// CHECK:STDOUT:   %addr.loc101.2: %ptr.d9e = addr_of %.loc102.2
// CHECK:STDOUT:   %addr.loc101.3: %ptr.d9e = addr_of %.loc101.1
// CHECK:STDOUT:   %operator_Percent__carbon_thunk.call: init %empty_tuple.type = call imports.%operator_Percent__carbon_thunk.decl(%addr.loc101.1, %addr.loc101.2, %addr.loc101.3)
// CHECK:STDOUT:   %.loc101.2: init %C to %.loc101.1 = mark_in_place_init %operator_Percent__carbon_thunk.call
// CHECK:STDOUT:   %.loc97: type = splice_block %C.ref.loc97 [concrete = constants.%C] {
// CHECK:STDOUT:     %Cpp.ref.loc96: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %C.ref.loc97: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc101.3: ref %C = temporary %.loc101.1, %.loc101.2
// CHECK:STDOUT:   %.loc101.4: %C = acquire_value %.loc101.3
// CHECK:STDOUT:   %modulo: %C = value_binding modulo, %.loc101.4
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %bitwise_and.patt: %pattern_type.217 = value_binding_pattern bitwise_and [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c1.ref.loc112: ref %C = name_ref c1, %c1
// CHECK:STDOUT:   %c2.ref.loc114: ref %C = name_ref c2, %c2
// CHECK:STDOUT:   %.loc113.1: ref %C = temporary_storage
// CHECK:STDOUT:   %.loc112.1: %C = acquire_value %c1.ref.loc112
// CHECK:STDOUT:   %.loc114.1: %C = acquire_value %c2.ref.loc114
// CHECK:STDOUT:   %.loc112.2: ref %C = value_as_ref %.loc112.1
// CHECK:STDOUT:   %addr.loc113.1: %ptr.d9e = addr_of %.loc112.2
// CHECK:STDOUT:   %.loc114.2: ref %C = value_as_ref %.loc114.1
// CHECK:STDOUT:   %addr.loc113.2: %ptr.d9e = addr_of %.loc114.2
// CHECK:STDOUT:   %addr.loc113.3: %ptr.d9e = addr_of %.loc113.1
// CHECK:STDOUT:   %operator_Amp__carbon_thunk.call: init %empty_tuple.type = call imports.%operator_Amp__carbon_thunk.decl(%addr.loc113.1, %addr.loc113.2, %addr.loc113.3)
// CHECK:STDOUT:   %.loc113.2: init %C to %.loc113.1 = mark_in_place_init %operator_Amp__carbon_thunk.call
// CHECK:STDOUT:   %.loc109: type = splice_block %C.ref.loc109 [concrete = constants.%C] {
// CHECK:STDOUT:     %Cpp.ref.loc108: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %C.ref.loc109: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc113.3: ref %C = temporary %.loc113.1, %.loc113.2
// CHECK:STDOUT:   %.loc113.4: %C = acquire_value %.loc113.3
// CHECK:STDOUT:   %bitwise_and: %C = value_binding bitwise_and, %.loc113.4
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %bitwise_or.patt: %pattern_type.217 = value_binding_pattern bitwise_or [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c1.ref.loc124: ref %C = name_ref c1, %c1
// CHECK:STDOUT:   %c2.ref.loc126: ref %C = name_ref c2, %c2
// CHECK:STDOUT:   %.loc125.1: ref %C = temporary_storage
// CHECK:STDOUT:   %.loc124.1: %C = acquire_value %c1.ref.loc124
// CHECK:STDOUT:   %.loc126.1: %C = acquire_value %c2.ref.loc126
// CHECK:STDOUT:   %.loc124.2: ref %C = value_as_ref %.loc124.1
// CHECK:STDOUT:   %addr.loc125.1: %ptr.d9e = addr_of %.loc124.2
// CHECK:STDOUT:   %.loc126.2: ref %C = value_as_ref %.loc126.1
// CHECK:STDOUT:   %addr.loc125.2: %ptr.d9e = addr_of %.loc126.2
// CHECK:STDOUT:   %addr.loc125.3: %ptr.d9e = addr_of %.loc125.1
// CHECK:STDOUT:   %operator_Pipe__carbon_thunk.call: init %empty_tuple.type = call imports.%operator_Pipe__carbon_thunk.decl(%addr.loc125.1, %addr.loc125.2, %addr.loc125.3)
// CHECK:STDOUT:   %.loc125.2: init %C to %.loc125.1 = mark_in_place_init %operator_Pipe__carbon_thunk.call
// CHECK:STDOUT:   %.loc121: type = splice_block %C.ref.loc121 [concrete = constants.%C] {
// CHECK:STDOUT:     %Cpp.ref.loc120: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %C.ref.loc121: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc125.3: ref %C = temporary %.loc125.1, %.loc125.2
// CHECK:STDOUT:   %.loc125.4: %C = acquire_value %.loc125.3
// CHECK:STDOUT:   %bitwise_or: %C = value_binding bitwise_or, %.loc125.4
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %bitwise_xor.patt: %pattern_type.217 = value_binding_pattern bitwise_xor [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c1.ref.loc136: ref %C = name_ref c1, %c1
// CHECK:STDOUT:   %c2.ref.loc138: ref %C = name_ref c2, %c2
// CHECK:STDOUT:   %.loc137.1: ref %C = temporary_storage
// CHECK:STDOUT:   %.loc136.1: %C = acquire_value %c1.ref.loc136
// CHECK:STDOUT:   %.loc138.1: %C = acquire_value %c2.ref.loc138
// CHECK:STDOUT:   %.loc136.2: ref %C = value_as_ref %.loc136.1
// CHECK:STDOUT:   %addr.loc137.1: %ptr.d9e = addr_of %.loc136.2
// CHECK:STDOUT:   %.loc138.2: ref %C = value_as_ref %.loc138.1
// CHECK:STDOUT:   %addr.loc137.2: %ptr.d9e = addr_of %.loc138.2
// CHECK:STDOUT:   %addr.loc137.3: %ptr.d9e = addr_of %.loc137.1
// CHECK:STDOUT:   %operator_Caret__carbon_thunk.call: init %empty_tuple.type = call imports.%operator_Caret__carbon_thunk.decl(%addr.loc137.1, %addr.loc137.2, %addr.loc137.3)
// CHECK:STDOUT:   %.loc137.2: init %C to %.loc137.1 = mark_in_place_init %operator_Caret__carbon_thunk.call
// CHECK:STDOUT:   %.loc133: type = splice_block %C.ref.loc133 [concrete = constants.%C] {
// CHECK:STDOUT:     %Cpp.ref.loc132: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %C.ref.loc133: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc137.3: ref %C = temporary %.loc137.1, %.loc137.2
// CHECK:STDOUT:   %.loc137.4: %C = acquire_value %.loc137.3
// CHECK:STDOUT:   %bitwise_xor: %C = value_binding bitwise_xor, %.loc137.4
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %left_shift.patt: %pattern_type.217 = value_binding_pattern left_shift [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c1.ref.loc148: ref %C = name_ref c1, %c1
// CHECK:STDOUT:   %int_3.loc150: Core.IntLiteral = int_value 3 [concrete = constants.%int_3.1ba]
// CHECK:STDOUT:   %.loc149.1: ref %C = temporary_storage
// CHECK:STDOUT:   %.loc148.1: %C = acquire_value %c1.ref.loc148
// CHECK:STDOUT:   %impl.elem0.loc150: %.545 = impl_witness_access constants.%ImplicitAs.impl_witness.6bc, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5]
// CHECK:STDOUT:   %bound_method.loc150.1: <bound method> = bound_method %int_3.loc150, %impl.elem0.loc150 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.061]
// CHECK:STDOUT:   %specific_fn.loc150: <specific function> = specific_function %impl.elem0.loc150, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc150.2: <bound method> = bound_method %int_3.loc150, %specific_fn.loc150 [concrete = constants.%bound_method.fa7]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc150: init %i32 = call %bound_method.loc150.2(%int_3.loc150) [concrete = constants.%int_3.822]
// CHECK:STDOUT:   %.loc150.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc150 [concrete = constants.%int_3.822]
// CHECK:STDOUT:   %.loc150.2: %i32 = converted %int_3.loc150, %.loc150.1 [concrete = constants.%int_3.822]
// CHECK:STDOUT:   %.loc148.2: ref %C = value_as_ref %.loc148.1
// CHECK:STDOUT:   %addr.loc149.1: %ptr.d9e = addr_of %.loc148.2
// CHECK:STDOUT:   %addr.loc149.2: %ptr.d9e = addr_of %.loc149.1
// CHECK:STDOUT:   %operator_LessLess__carbon_thunk.call: init %empty_tuple.type = call imports.%operator_LessLess__carbon_thunk.decl(%addr.loc149.1, %.loc150.2, %addr.loc149.2)
// CHECK:STDOUT:   %.loc149.2: init %C to %.loc149.1 = mark_in_place_init %operator_LessLess__carbon_thunk.call
// CHECK:STDOUT:   %.loc145: type = splice_block %C.ref.loc145 [concrete = constants.%C] {
// CHECK:STDOUT:     %Cpp.ref.loc144: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %C.ref.loc145: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc149.3: ref %C = temporary %.loc149.1, %.loc149.2
// CHECK:STDOUT:   %.loc149.4: %C = acquire_value %.loc149.3
// CHECK:STDOUT:   %left_shift: %C = value_binding left_shift, %.loc149.4
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %right_shift.patt: %pattern_type.217 = value_binding_pattern right_shift [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c1.ref.loc160: ref %C = name_ref c1, %c1
// CHECK:STDOUT:   %int_5.loc162: Core.IntLiteral = int_value 5 [concrete = constants.%int_5.64b]
// CHECK:STDOUT:   %.loc161.1: ref %C = temporary_storage
// CHECK:STDOUT:   %.loc160.1: %C = acquire_value %c1.ref.loc160
// CHECK:STDOUT:   %impl.elem0.loc162: %.545 = impl_witness_access constants.%ImplicitAs.impl_witness.6bc, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5]
// CHECK:STDOUT:   %bound_method.loc162.1: <bound method> = bound_method %int_5.loc162, %impl.elem0.loc162 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.005]
// CHECK:STDOUT:   %specific_fn.loc162: <specific function> = specific_function %impl.elem0.loc162, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc162.2: <bound method> = bound_method %int_5.loc162, %specific_fn.loc162 [concrete = constants.%bound_method.e9d]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc162: init %i32 = call %bound_method.loc162.2(%int_5.loc162) [concrete = constants.%int_5.0f6]
// CHECK:STDOUT:   %.loc162.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc162 [concrete = constants.%int_5.0f6]
// CHECK:STDOUT:   %.loc162.2: %i32 = converted %int_5.loc162, %.loc162.1 [concrete = constants.%int_5.0f6]
// CHECK:STDOUT:   %.loc160.2: ref %C = value_as_ref %.loc160.1
// CHECK:STDOUT:   %addr.loc161.1: %ptr.d9e = addr_of %.loc160.2
// CHECK:STDOUT:   %addr.loc161.2: %ptr.d9e = addr_of %.loc161.1
// CHECK:STDOUT:   %operator_GreaterGreater__carbon_thunk.call: init %empty_tuple.type = call imports.%operator_GreaterGreater__carbon_thunk.decl(%addr.loc161.1, %.loc162.2, %addr.loc161.2)
// CHECK:STDOUT:   %.loc161.2: init %C to %.loc161.1 = mark_in_place_init %operator_GreaterGreater__carbon_thunk.call
// CHECK:STDOUT:   %.loc157: type = splice_block %C.ref.loc157 [concrete = constants.%C] {
// CHECK:STDOUT:     %Cpp.ref.loc156: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %C.ref.loc157: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc161.3: ref %C = temporary %.loc161.1, %.loc161.2
// CHECK:STDOUT:   %.loc161.4: %C = acquire_value %.loc161.3
// CHECK:STDOUT:   %right_shift: %C = value_binding right_shift, %.loc161.4
// CHECK:STDOUT:   %c1.ref.loc164: ref %C = name_ref c1, %c1
// CHECK:STDOUT:   %c2.ref.loc166: ref %C = name_ref c2, %c2
// CHECK:STDOUT:   %.loc166.1: %C = acquire_value %c2.ref.loc166
// CHECK:STDOUT:   %.loc166.2: ref %C = value_as_ref %.loc166.1
// CHECK:STDOUT:   %addr.loc165: %ptr.d9e = addr_of %.loc166.2
// CHECK:STDOUT:   %operator_PlusEqual__carbon_thunk.call: ref %C = call imports.%operator_PlusEqual__carbon_thunk.decl(%c1.ref.loc164, %addr.loc165)
// CHECK:STDOUT:   %c1.ref.loc168: ref %C = name_ref c1, %c1
// CHECK:STDOUT:   %c2.ref.loc170: ref %C = name_ref c2, %c2
// CHECK:STDOUT:   %.loc170.1: %C = acquire_value %c2.ref.loc170
// CHECK:STDOUT:   %.loc170.2: ref %C = value_as_ref %.loc170.1
// CHECK:STDOUT:   %addr.loc169: %ptr.d9e = addr_of %.loc170.2
// CHECK:STDOUT:   %operator_MinusEqual__carbon_thunk.call: ref %C = call imports.%operator_MinusEqual__carbon_thunk.decl(%c1.ref.loc168, %addr.loc169)
// CHECK:STDOUT:   %c1.ref.loc172: ref %C = name_ref c1, %c1
// CHECK:STDOUT:   %c2.ref.loc174: ref %C = name_ref c2, %c2
// CHECK:STDOUT:   %.loc174.1: %C = acquire_value %c2.ref.loc174
// CHECK:STDOUT:   %.loc174.2: ref %C = value_as_ref %.loc174.1
// CHECK:STDOUT:   %addr.loc173: %ptr.d9e = addr_of %.loc174.2
// CHECK:STDOUT:   %operator_StarEqual__carbon_thunk.call: ref %C = call imports.%operator_StarEqual__carbon_thunk.decl(%c1.ref.loc172, %addr.loc173)
// CHECK:STDOUT:   %c1.ref.loc176: ref %C = name_ref c1, %c1
// CHECK:STDOUT:   %c2.ref.loc178: ref %C = name_ref c2, %c2
// CHECK:STDOUT:   %.loc178.1: %C = acquire_value %c2.ref.loc178
// CHECK:STDOUT:   %.loc178.2: ref %C = value_as_ref %.loc178.1
// CHECK:STDOUT:   %addr.loc177: %ptr.d9e = addr_of %.loc178.2
// CHECK:STDOUT:   %operator_SlashEqual__carbon_thunk.call: ref %C = call imports.%operator_SlashEqual__carbon_thunk.decl(%c1.ref.loc176, %addr.loc177)
// CHECK:STDOUT:   %c1.ref.loc180: ref %C = name_ref c1, %c1
// CHECK:STDOUT:   %c2.ref.loc182: ref %C = name_ref c2, %c2
// CHECK:STDOUT:   %.loc182.1: %C = acquire_value %c2.ref.loc182
// CHECK:STDOUT:   %.loc182.2: ref %C = value_as_ref %.loc182.1
// CHECK:STDOUT:   %addr.loc181: %ptr.d9e = addr_of %.loc182.2
// CHECK:STDOUT:   %operator_PercentEqual__carbon_thunk.call: ref %C = call imports.%operator_PercentEqual__carbon_thunk.decl(%c1.ref.loc180, %addr.loc181)
// CHECK:STDOUT:   %c1.ref.loc184: ref %C = name_ref c1, %c1
// CHECK:STDOUT:   %c2.ref.loc186: ref %C = name_ref c2, %c2
// CHECK:STDOUT:   %.loc186.1: %C = acquire_value %c2.ref.loc186
// CHECK:STDOUT:   %.loc186.2: ref %C = value_as_ref %.loc186.1
// CHECK:STDOUT:   %addr.loc185: %ptr.d9e = addr_of %.loc186.2
// CHECK:STDOUT:   %operator_AmpEqual__carbon_thunk.call: ref %C = call imports.%operator_AmpEqual__carbon_thunk.decl(%c1.ref.loc184, %addr.loc185)
// CHECK:STDOUT:   %c1.ref.loc188: ref %C = name_ref c1, %c1
// CHECK:STDOUT:   %c2.ref.loc190: ref %C = name_ref c2, %c2
// CHECK:STDOUT:   %.loc190.1: %C = acquire_value %c2.ref.loc190
// CHECK:STDOUT:   %.loc190.2: ref %C = value_as_ref %.loc190.1
// CHECK:STDOUT:   %addr.loc189: %ptr.d9e = addr_of %.loc190.2
// CHECK:STDOUT:   %operator_PipeEqual__carbon_thunk.call: ref %C = call imports.%operator_PipeEqual__carbon_thunk.decl(%c1.ref.loc188, %addr.loc189)
// CHECK:STDOUT:   %c1.ref.loc192: ref %C = name_ref c1, %c1
// CHECK:STDOUT:   %c2.ref.loc194: ref %C = name_ref c2, %c2
// CHECK:STDOUT:   %.loc194.1: %C = acquire_value %c2.ref.loc194
// CHECK:STDOUT:   %.loc194.2: ref %C = value_as_ref %.loc194.1
// CHECK:STDOUT:   %addr.loc193: %ptr.d9e = addr_of %.loc194.2
// CHECK:STDOUT:   %operator_CaretEqual__carbon_thunk.call: ref %C = call imports.%operator_CaretEqual__carbon_thunk.decl(%c1.ref.loc192, %addr.loc193)
// CHECK:STDOUT:   %c1.ref.loc196: ref %C = name_ref c1, %c1
// CHECK:STDOUT:   %int_3.loc198: Core.IntLiteral = int_value 3 [concrete = constants.%int_3.1ba]
// CHECK:STDOUT:   %impl.elem0.loc198: %.545 = impl_witness_access constants.%ImplicitAs.impl_witness.6bc, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5]
// CHECK:STDOUT:   %bound_method.loc198.1: <bound method> = bound_method %int_3.loc198, %impl.elem0.loc198 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.061]
// CHECK:STDOUT:   %specific_fn.loc198: <specific function> = specific_function %impl.elem0.loc198, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc198.2: <bound method> = bound_method %int_3.loc198, %specific_fn.loc198 [concrete = constants.%bound_method.fa7]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc198: init %i32 = call %bound_method.loc198.2(%int_3.loc198) [concrete = constants.%int_3.822]
// CHECK:STDOUT:   %.loc198.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc198 [concrete = constants.%int_3.822]
// CHECK:STDOUT:   %.loc198.2: %i32 = converted %int_3.loc198, %.loc198.1 [concrete = constants.%int_3.822]
// CHECK:STDOUT:   %cpp_operator.call.loc197: ref %C = call imports.%cpp_operator.decl.4206c9.19(%c1.ref.loc196, %.loc198.2)
// CHECK:STDOUT:   %c1.ref.loc200: ref %C = name_ref c1, %c1
// CHECK:STDOUT:   %int_5.loc202: Core.IntLiteral = int_value 5 [concrete = constants.%int_5.64b]
// CHECK:STDOUT:   %impl.elem0.loc202: %.545 = impl_witness_access constants.%ImplicitAs.impl_witness.6bc, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5]
// CHECK:STDOUT:   %bound_method.loc202.1: <bound method> = bound_method %int_5.loc202, %impl.elem0.loc202 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.005]
// CHECK:STDOUT:   %specific_fn.loc202: <specific function> = specific_function %impl.elem0.loc202, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc202.2: <bound method> = bound_method %int_5.loc202, %specific_fn.loc202 [concrete = constants.%bound_method.e9d]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc202: init %i32 = call %bound_method.loc202.2(%int_5.loc202) [concrete = constants.%int_5.0f6]
// CHECK:STDOUT:   %.loc202.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc202 [concrete = constants.%int_5.0f6]
// CHECK:STDOUT:   %.loc202.2: %i32 = converted %int_5.loc202, %.loc202.1 [concrete = constants.%int_5.0f6]
// CHECK:STDOUT:   %cpp_operator.call.loc201: ref %C = call imports.%cpp_operator.decl.4206c9.20(%c1.ref.loc200, %.loc202.2)
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %equal.patt: %pattern_type.831 = value_binding_pattern equal [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c1.ref.loc210: ref %C = name_ref c1, %c1
// CHECK:STDOUT:   %c2.ref.loc212: ref %C = name_ref c2, %c2
// CHECK:STDOUT:   %.loc210.1: %C = acquire_value %c1.ref.loc210
// CHECK:STDOUT:   %.loc212.1: %C = acquire_value %c2.ref.loc212
// CHECK:STDOUT:   %.loc210.2: ref %C = value_as_ref %.loc210.1
// CHECK:STDOUT:   %addr.loc211.1: %ptr.d9e = addr_of %.loc210.2
// CHECK:STDOUT:   %.loc212.2: ref %C = value_as_ref %.loc212.1
// CHECK:STDOUT:   %addr.loc211.2: %ptr.d9e = addr_of %.loc212.2
// CHECK:STDOUT:   %.loc211.1: ref bool = temporary_storage
// CHECK:STDOUT:   %addr.loc211.3: %ptr.bb2 = addr_of %.loc211.1
// CHECK:STDOUT:   %operator_EqualEqual__carbon_thunk.call: init %empty_tuple.type = call imports.%operator_EqualEqual__carbon_thunk.decl(%addr.loc211.1, %addr.loc211.2, %addr.loc211.3)
// CHECK:STDOUT:   %.loc211.2: init bool to %.loc211.1 = mark_in_place_init %operator_EqualEqual__carbon_thunk.call
// CHECK:STDOUT:   %.loc208: type = type_literal bool [concrete = bool]
// CHECK:STDOUT:   %.loc211.3: ref bool = temporary %.loc211.1, %.loc211.2
// CHECK:STDOUT:   %.loc211.4: bool = acquire_value %.loc211.3
// CHECK:STDOUT:   %equal: bool = value_binding equal, %.loc211.4
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %not_equal.patt: %pattern_type.831 = value_binding_pattern not_equal [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c1.ref.loc220: ref %C = name_ref c1, %c1
// CHECK:STDOUT:   %c2.ref.loc222: ref %C = name_ref c2, %c2
// CHECK:STDOUT:   %.loc220.1: %C = acquire_value %c1.ref.loc220
// CHECK:STDOUT:   %.loc222.1: %C = acquire_value %c2.ref.loc222
// CHECK:STDOUT:   %.loc220.2: ref %C = value_as_ref %.loc220.1
// CHECK:STDOUT:   %addr.loc221.1: %ptr.d9e = addr_of %.loc220.2
// CHECK:STDOUT:   %.loc222.2: ref %C = value_as_ref %.loc222.1
// CHECK:STDOUT:   %addr.loc221.2: %ptr.d9e = addr_of %.loc222.2
// CHECK:STDOUT:   %.loc221.1: ref bool = temporary_storage
// CHECK:STDOUT:   %addr.loc221.3: %ptr.bb2 = addr_of %.loc221.1
// CHECK:STDOUT:   %operator_ExclaimEqual__carbon_thunk.call: init %empty_tuple.type = call imports.%operator_ExclaimEqual__carbon_thunk.decl(%addr.loc221.1, %addr.loc221.2, %addr.loc221.3)
// CHECK:STDOUT:   %.loc221.2: init bool to %.loc221.1 = mark_in_place_init %operator_ExclaimEqual__carbon_thunk.call
// CHECK:STDOUT:   %.loc218: type = type_literal bool [concrete = bool]
// CHECK:STDOUT:   %.loc221.3: ref bool = temporary %.loc221.1, %.loc221.2
// CHECK:STDOUT:   %.loc221.4: bool = acquire_value %.loc221.3
// CHECK:STDOUT:   %not_equal: bool = value_binding not_equal, %.loc221.4
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %greater_than.patt: %pattern_type.831 = value_binding_pattern greater_than [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c1.ref.loc230: ref %C = name_ref c1, %c1
// CHECK:STDOUT:   %c2.ref.loc232: ref %C = name_ref c2, %c2
// CHECK:STDOUT:   %.loc230.1: %C = acquire_value %c1.ref.loc230
// CHECK:STDOUT:   %.loc232.1: %C = acquire_value %c2.ref.loc232
// CHECK:STDOUT:   %.loc230.2: ref %C = value_as_ref %.loc230.1
// CHECK:STDOUT:   %addr.loc231.1: %ptr.d9e = addr_of %.loc230.2
// CHECK:STDOUT:   %.loc232.2: ref %C = value_as_ref %.loc232.1
// CHECK:STDOUT:   %addr.loc231.2: %ptr.d9e = addr_of %.loc232.2
// CHECK:STDOUT:   %.loc231.1: ref bool = temporary_storage
// CHECK:STDOUT:   %addr.loc231.3: %ptr.bb2 = addr_of %.loc231.1
// CHECK:STDOUT:   %operator_Greater__carbon_thunk.call: init %empty_tuple.type = call imports.%operator_Greater__carbon_thunk.decl(%addr.loc231.1, %addr.loc231.2, %addr.loc231.3)
// CHECK:STDOUT:   %.loc231.2: init bool to %.loc231.1 = mark_in_place_init %operator_Greater__carbon_thunk.call
// CHECK:STDOUT:   %.loc228: type = type_literal bool [concrete = bool]
// CHECK:STDOUT:   %.loc231.3: ref bool = temporary %.loc231.1, %.loc231.2
// CHECK:STDOUT:   %.loc231.4: bool = acquire_value %.loc231.3
// CHECK:STDOUT:   %greater_than: bool = value_binding greater_than, %.loc231.4
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %less_than.patt: %pattern_type.831 = value_binding_pattern less_than [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c1.ref.loc240: ref %C = name_ref c1, %c1
// CHECK:STDOUT:   %c2.ref.loc242: ref %C = name_ref c2, %c2
// CHECK:STDOUT:   %.loc240.1: %C = acquire_value %c1.ref.loc240
// CHECK:STDOUT:   %.loc242.1: %C = acquire_value %c2.ref.loc242
// CHECK:STDOUT:   %.loc240.2: ref %C = value_as_ref %.loc240.1
// CHECK:STDOUT:   %addr.loc241.1: %ptr.d9e = addr_of %.loc240.2
// CHECK:STDOUT:   %.loc242.2: ref %C = value_as_ref %.loc242.1
// CHECK:STDOUT:   %addr.loc241.2: %ptr.d9e = addr_of %.loc242.2
// CHECK:STDOUT:   %.loc241.1: ref bool = temporary_storage
// CHECK:STDOUT:   %addr.loc241.3: %ptr.bb2 = addr_of %.loc241.1
// CHECK:STDOUT:   %operator_Less__carbon_thunk.call: init %empty_tuple.type = call imports.%operator_Less__carbon_thunk.decl(%addr.loc241.1, %addr.loc241.2, %addr.loc241.3)
// CHECK:STDOUT:   %.loc241.2: init bool to %.loc241.1 = mark_in_place_init %operator_Less__carbon_thunk.call
// CHECK:STDOUT:   %.loc238: type = type_literal bool [concrete = bool]
// CHECK:STDOUT:   %.loc241.3: ref bool = temporary %.loc241.1, %.loc241.2
// CHECK:STDOUT:   %.loc241.4: bool = acquire_value %.loc241.3
// CHECK:STDOUT:   %less_than: bool = value_binding less_than, %.loc241.4
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %greater_than_or_equal.patt: %pattern_type.831 = value_binding_pattern greater_than_or_equal [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c1.ref.loc250: ref %C = name_ref c1, %c1
// CHECK:STDOUT:   %c2.ref.loc252: ref %C = name_ref c2, %c2
// CHECK:STDOUT:   %.loc250.1: %C = acquire_value %c1.ref.loc250
// CHECK:STDOUT:   %.loc252.1: %C = acquire_value %c2.ref.loc252
// CHECK:STDOUT:   %.loc250.2: ref %C = value_as_ref %.loc250.1
// CHECK:STDOUT:   %addr.loc251.1: %ptr.d9e = addr_of %.loc250.2
// CHECK:STDOUT:   %.loc252.2: ref %C = value_as_ref %.loc252.1
// CHECK:STDOUT:   %addr.loc251.2: %ptr.d9e = addr_of %.loc252.2
// CHECK:STDOUT:   %.loc251.1: ref bool = temporary_storage
// CHECK:STDOUT:   %addr.loc251.3: %ptr.bb2 = addr_of %.loc251.1
// CHECK:STDOUT:   %operator_GreaterEqual__carbon_thunk.call: init %empty_tuple.type = call imports.%operator_GreaterEqual__carbon_thunk.decl(%addr.loc251.1, %addr.loc251.2, %addr.loc251.3)
// CHECK:STDOUT:   %.loc251.2: init bool to %.loc251.1 = mark_in_place_init %operator_GreaterEqual__carbon_thunk.call
// CHECK:STDOUT:   %.loc248: type = type_literal bool [concrete = bool]
// CHECK:STDOUT:   %.loc251.3: ref bool = temporary %.loc251.1, %.loc251.2
// CHECK:STDOUT:   %.loc251.4: bool = acquire_value %.loc251.3
// CHECK:STDOUT:   %greater_than_or_equal: bool = value_binding greater_than_or_equal, %.loc251.4
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %less_than_or_equal.patt: %pattern_type.831 = value_binding_pattern less_than_or_equal [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c1.ref.loc260: ref %C = name_ref c1, %c1
// CHECK:STDOUT:   %c2.ref.loc262: ref %C = name_ref c2, %c2
// CHECK:STDOUT:   %.loc260.1: %C = acquire_value %c1.ref.loc260
// CHECK:STDOUT:   %.loc262.1: %C = acquire_value %c2.ref.loc262
// CHECK:STDOUT:   %.loc260.2: ref %C = value_as_ref %.loc260.1
// CHECK:STDOUT:   %addr.loc261.1: %ptr.d9e = addr_of %.loc260.2
// CHECK:STDOUT:   %.loc262.2: ref %C = value_as_ref %.loc262.1
// CHECK:STDOUT:   %addr.loc261.2: %ptr.d9e = addr_of %.loc262.2
// CHECK:STDOUT:   %.loc261.1: ref bool = temporary_storage
// CHECK:STDOUT:   %addr.loc261.3: %ptr.bb2 = addr_of %.loc261.1
// CHECK:STDOUT:   %operator_LessEqual__carbon_thunk.call: init %empty_tuple.type = call imports.%operator_LessEqual__carbon_thunk.decl(%addr.loc261.1, %addr.loc261.2, %addr.loc261.3)
// CHECK:STDOUT:   %.loc261.2: init bool to %.loc261.1 = mark_in_place_init %operator_LessEqual__carbon_thunk.call
// CHECK:STDOUT:   %.loc258: type = type_literal bool [concrete = bool]
// CHECK:STDOUT:   %.loc261.3: ref bool = temporary %.loc261.1, %.loc261.2
// CHECK:STDOUT:   %.loc261.4: bool = acquire_value %.loc261.3
// CHECK:STDOUT:   %less_than_or_equal: bool = value_binding less_than_or_equal, %.loc261.4
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %index.patt: %pattern_type.7ce = value_binding_pattern index [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c1.ref.loc270: ref %C = name_ref c1, %c1
// CHECK:STDOUT:   %int_42: Core.IntLiteral = int_value 42 [concrete = constants.%int_42.20e]
// CHECK:STDOUT:   %C.cpp_operator.bound: <bound method> = bound_method %c1.ref.loc270, imports.%C.cpp_operator.decl
// CHECK:STDOUT:   %impl.elem0.loc272: %.545 = impl_witness_access constants.%ImplicitAs.impl_witness.6bc, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5]
// CHECK:STDOUT:   %bound_method.loc272.1: <bound method> = bound_method %int_42, %impl.elem0.loc272 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.c83]
// CHECK:STDOUT:   %specific_fn.loc272: <specific function> = specific_function %impl.elem0.loc272, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc272.2: <bound method> = bound_method %int_42, %specific_fn.loc272 [concrete = constants.%bound_method.cb9]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc272: init %i32 = call %bound_method.loc272.2(%int_42) [concrete = constants.%int_42.c68]
// CHECK:STDOUT:   %.loc272.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc272 [concrete = constants.%int_42.c68]
// CHECK:STDOUT:   %.loc272.2: %i32 = converted %int_42, %.loc272.1 [concrete = constants.%int_42.c68]
// CHECK:STDOUT:   %C.cpp_operator.call: init %i32 = call %C.cpp_operator.bound(%c1.ref.loc270, %.loc272.2)
// CHECK:STDOUT:   %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %.loc273.1: %i32 = value_of_initializer %C.cpp_operator.call
// CHECK:STDOUT:   %.loc273.2: %i32 = converted %C.cpp_operator.call, %.loc273.1
// CHECK:STDOUT:   %index: %i32 = value_binding index, %.loc273.2
// CHECK:STDOUT:   %Destroy.Op.bound.loc261: <bound method> = bound_method %.loc261.3, constants.%Destroy.Op
// CHECK:STDOUT:   %Destroy.Op.call.loc261: init %empty_tuple.type = call %Destroy.Op.bound.loc261(%.loc261.3)
// CHECK:STDOUT:   %Destroy.Op.bound.loc251: <bound method> = bound_method %.loc251.3, constants.%Destroy.Op
// CHECK:STDOUT:   %Destroy.Op.call.loc251: init %empty_tuple.type = call %Destroy.Op.bound.loc251(%.loc251.3)
// CHECK:STDOUT:   %Destroy.Op.bound.loc241: <bound method> = bound_method %.loc241.3, constants.%Destroy.Op
// CHECK:STDOUT:   %Destroy.Op.call.loc241: init %empty_tuple.type = call %Destroy.Op.bound.loc241(%.loc241.3)
// CHECK:STDOUT:   %Destroy.Op.bound.loc231: <bound method> = bound_method %.loc231.3, constants.%Destroy.Op
// CHECK:STDOUT:   %Destroy.Op.call.loc231: init %empty_tuple.type = call %Destroy.Op.bound.loc231(%.loc231.3)
// CHECK:STDOUT:   %Destroy.Op.bound.loc221: <bound method> = bound_method %.loc221.3, constants.%Destroy.Op
// CHECK:STDOUT:   %Destroy.Op.call.loc221: init %empty_tuple.type = call %Destroy.Op.bound.loc221(%.loc221.3)
// CHECK:STDOUT:   %Destroy.Op.bound.loc211: <bound method> = bound_method %.loc211.3, constants.%Destroy.Op
// CHECK:STDOUT:   %Destroy.Op.call.loc211: init %empty_tuple.type = call %Destroy.Op.bound.loc211(%.loc211.3)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %C.Op.bound.loc161: <bound method> = bound_method %.loc161.3, constants.%C.Op
// CHECK:STDOUT:   %Op.ref.loc161: %C.cpp_destructor.type = name_ref Op, imports.%C.cpp_destructor.decl [concrete = constants.%C.cpp_destructor]
// CHECK:STDOUT:   %C.cpp_destructor.bound.loc161: <bound method> = bound_method %.loc161.3, %Op.ref.loc161
// CHECK:STDOUT:   %C.cpp_destructor.call.loc161: init %empty_tuple.type = call %C.cpp_destructor.bound.loc161(%.loc161.3)
// CHECK:STDOUT:   %C.Op.bound.loc149: <bound method> = bound_method %.loc149.3, constants.%C.Op
// CHECK:STDOUT:   %Op.ref.loc149: %C.cpp_destructor.type = name_ref Op, imports.%C.cpp_destructor.decl [concrete = constants.%C.cpp_destructor]
// CHECK:STDOUT:   %C.cpp_destructor.bound.loc149: <bound method> = bound_method %.loc149.3, %Op.ref.loc149
// CHECK:STDOUT:   %C.cpp_destructor.call.loc149: init %empty_tuple.type = call %C.cpp_destructor.bound.loc149(%.loc149.3)
// CHECK:STDOUT:   %C.Op.bound.loc137: <bound method> = bound_method %.loc137.3, constants.%C.Op
// CHECK:STDOUT:   %Op.ref.loc137: %C.cpp_destructor.type = name_ref Op, imports.%C.cpp_destructor.decl [concrete = constants.%C.cpp_destructor]
// CHECK:STDOUT:   %C.cpp_destructor.bound.loc137: <bound method> = bound_method %.loc137.3, %Op.ref.loc137
// CHECK:STDOUT:   %C.cpp_destructor.call.loc137: init %empty_tuple.type = call %C.cpp_destructor.bound.loc137(%.loc137.3)
// CHECK:STDOUT:   %C.Op.bound.loc125: <bound method> = bound_method %.loc125.3, constants.%C.Op
// CHECK:STDOUT:   %Op.ref.loc125: %C.cpp_destructor.type = name_ref Op, imports.%C.cpp_destructor.decl [concrete = constants.%C.cpp_destructor]
// CHECK:STDOUT:   %C.cpp_destructor.bound.loc125: <bound method> = bound_method %.loc125.3, %Op.ref.loc125
// CHECK:STDOUT:   %C.cpp_destructor.call.loc125: init %empty_tuple.type = call %C.cpp_destructor.bound.loc125(%.loc125.3)
// CHECK:STDOUT:   %C.Op.bound.loc113: <bound method> = bound_method %.loc113.3, constants.%C.Op
// CHECK:STDOUT:   %Op.ref.loc113: %C.cpp_destructor.type = name_ref Op, imports.%C.cpp_destructor.decl [concrete = constants.%C.cpp_destructor]
// CHECK:STDOUT:   %C.cpp_destructor.bound.loc113: <bound method> = bound_method %.loc113.3, %Op.ref.loc113
// CHECK:STDOUT:   %C.cpp_destructor.call.loc113: init %empty_tuple.type = call %C.cpp_destructor.bound.loc113(%.loc113.3)
// CHECK:STDOUT:   %C.Op.bound.loc101: <bound method> = bound_method %.loc101.3, constants.%C.Op
// CHECK:STDOUT:   %Op.ref.loc101: %C.cpp_destructor.type = name_ref Op, imports.%C.cpp_destructor.decl [concrete = constants.%C.cpp_destructor]
// CHECK:STDOUT:   %C.cpp_destructor.bound.loc101: <bound method> = bound_method %.loc101.3, %Op.ref.loc101
// CHECK:STDOUT:   %C.cpp_destructor.call.loc101: init %empty_tuple.type = call %C.cpp_destructor.bound.loc101(%.loc101.3)
// CHECK:STDOUT:   %C.Op.bound.loc89: <bound method> = bound_method %.loc89.3, constants.%C.Op
// CHECK:STDOUT:   %Op.ref.loc89: %C.cpp_destructor.type = name_ref Op, imports.%C.cpp_destructor.decl [concrete = constants.%C.cpp_destructor]
// CHECK:STDOUT:   %C.cpp_destructor.bound.loc89: <bound method> = bound_method %.loc89.3, %Op.ref.loc89
// CHECK:STDOUT:   %C.cpp_destructor.call.loc89: init %empty_tuple.type = call %C.cpp_destructor.bound.loc89(%.loc89.3)
// CHECK:STDOUT:   %C.Op.bound.loc77: <bound method> = bound_method %.loc77.3, constants.%C.Op
// CHECK:STDOUT:   %Op.ref.loc77: %C.cpp_destructor.type = name_ref Op, imports.%C.cpp_destructor.decl [concrete = constants.%C.cpp_destructor]
// CHECK:STDOUT:   %C.cpp_destructor.bound.loc77: <bound method> = bound_method %.loc77.3, %Op.ref.loc77
// CHECK:STDOUT:   %C.cpp_destructor.call.loc77: init %empty_tuple.type = call %C.cpp_destructor.bound.loc77(%.loc77.3)
// CHECK:STDOUT:   %C.Op.bound.loc65: <bound method> = bound_method %.loc65.3, constants.%C.Op
// CHECK:STDOUT:   %Op.ref.loc65: %C.cpp_destructor.type = name_ref Op, imports.%C.cpp_destructor.decl [concrete = constants.%C.cpp_destructor]
// CHECK:STDOUT:   %C.cpp_destructor.bound.loc65: <bound method> = bound_method %.loc65.3, %Op.ref.loc65
// CHECK:STDOUT:   %C.cpp_destructor.call.loc65: init %empty_tuple.type = call %C.cpp_destructor.bound.loc65(%.loc65.3)
// CHECK:STDOUT:   %C.Op.bound.loc53: <bound method> = bound_method %.loc53.3, constants.%C.Op
// CHECK:STDOUT:   %Op.ref.loc53: %C.cpp_destructor.type = name_ref Op, imports.%C.cpp_destructor.decl [concrete = constants.%C.cpp_destructor]
// CHECK:STDOUT:   %C.cpp_destructor.bound.loc53: <bound method> = bound_method %.loc53.3, %Op.ref.loc53
// CHECK:STDOUT:   %C.cpp_destructor.call.loc53: init %empty_tuple.type = call %C.cpp_destructor.bound.loc53(%.loc53.3)
// CHECK:STDOUT:   %C.Op.bound.loc29: <bound method> = bound_method %c2.var, constants.%C.Op
// CHECK:STDOUT:   %Op.ref.loc29: %C.cpp_destructor.type = name_ref Op, imports.%C.cpp_destructor.decl [concrete = constants.%C.cpp_destructor]
// CHECK:STDOUT:   %C.cpp_destructor.bound.loc29: <bound method> = bound_method %c2.var, %Op.ref.loc29
// CHECK:STDOUT:   %C.cpp_destructor.call.loc29: init %empty_tuple.type = call %C.cpp_destructor.bound.loc29(%c2.var)
// CHECK:STDOUT:   %C.Op.bound.loc14: <bound method> = bound_method %c1.var, constants.%C.Op
// CHECK:STDOUT:   %Op.ref.loc14: %C.cpp_destructor.type = name_ref Op, imports.%C.cpp_destructor.decl [concrete = constants.%C.cpp_destructor]
// CHECK:STDOUT:   %C.cpp_destructor.bound.loc14: <bound method> = bound_method %c1.var, %Op.ref.loc14
// CHECK:STDOUT:   %C.cpp_destructor.call.loc14: init %empty_tuple.type = call %C.cpp_destructor.bound.loc14(%c1.var)
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
// CHECK:STDOUT:   %Cpp.ref.loc21: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %C.ref.loc22: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %C.ref.loc24: %C.C.cpp_overload_set.type = name_ref C, imports.%C.C.cpp_overload_set.value [concrete = constants.%C.C.cpp_overload_set.value]
// CHECK:STDOUT:   %.loc27.1: ref %C = temporary_storage
// CHECK:STDOUT:   %addr.loc27: %ptr.d9e = addr_of %.loc27.1
// CHECK:STDOUT:   %C__carbon_thunk.call.loc27: init %empty_tuple.type = call imports.%C__carbon_thunk.decl(%addr.loc27)
// CHECK:STDOUT:   %.loc27.2: init %C to %.loc27.1 = mark_in_place_init %C__carbon_thunk.call.loc27
// CHECK:STDOUT:   %.loc18: type = splice_block %C.ref.loc18 [concrete = constants.%C] {
// CHECK:STDOUT:     %Cpp.ref.loc17: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %C.ref.loc18: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc27.3: ref %C = temporary %.loc27.1, %.loc27.2
// CHECK:STDOUT:   %.loc27.4: %C = acquire_value %.loc27.3
// CHECK:STDOUT:   %c1: %C = value_binding c1, %.loc27.4
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c2.patt: %pattern_type.217 = value_binding_pattern c2 [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc36: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %C.ref.loc37: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %C.ref.loc39: %C.C.cpp_overload_set.type = name_ref C, imports.%C.C.cpp_overload_set.value [concrete = constants.%C.C.cpp_overload_set.value]
// CHECK:STDOUT:   %.loc42.1: ref %C = temporary_storage
// CHECK:STDOUT:   %addr.loc42: %ptr.d9e = addr_of %.loc42.1
// CHECK:STDOUT:   %C__carbon_thunk.call.loc42: init %empty_tuple.type = call imports.%C__carbon_thunk.decl(%addr.loc42)
// CHECK:STDOUT:   %.loc42.2: init %C to %.loc42.1 = mark_in_place_init %C__carbon_thunk.call.loc42
// CHECK:STDOUT:   %.loc33: type = splice_block %C.ref.loc33 [concrete = constants.%C] {
// CHECK:STDOUT:     %Cpp.ref.loc32: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %C.ref.loc33: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc42.3: ref %C = temporary %.loc42.1, %.loc42.2
// CHECK:STDOUT:   %.loc42.4: %C = acquire_value %.loc42.3
// CHECK:STDOUT:   %c2: %C = value_binding c2, %.loc42.4
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c3.patt: %pattern_type.217 = value_binding_pattern c3 [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c1.ref.loc51: %C = name_ref c1, %c1
// CHECK:STDOUT:   %c2.ref: %C = name_ref c2, %c2
// CHECK:STDOUT:   %.loc52.1: ref %C = temporary_storage
// CHECK:STDOUT:   %.loc51: ref %C = value_as_ref %c1.ref.loc51
// CHECK:STDOUT:   %addr.loc52.1: %ptr.d9e = addr_of %.loc51
// CHECK:STDOUT:   %.loc53: ref %C = value_as_ref %c2.ref
// CHECK:STDOUT:   %addr.loc52.2: %ptr.d9e = addr_of %.loc53
// CHECK:STDOUT:   %addr.loc52.3: %ptr.d9e = addr_of %.loc52.1
// CHECK:STDOUT:   %operator_Plus__carbon_thunk.call.loc52: init %empty_tuple.type = call imports.%operator_Plus__carbon_thunk.decl(%addr.loc52.1, %addr.loc52.2, %addr.loc52.3)
// CHECK:STDOUT:   %.loc52.2: init %C to %.loc52.1 = mark_in_place_init %operator_Plus__carbon_thunk.call.loc52
// CHECK:STDOUT:   %.loc48: type = splice_block %C.ref.loc48 [concrete = constants.%C] {
// CHECK:STDOUT:     %Cpp.ref.loc47: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %C.ref.loc48: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc52.3: ref %C = temporary %.loc52.1, %.loc52.2
// CHECK:STDOUT:   %.loc52.4: %C = acquire_value %.loc52.3
// CHECK:STDOUT:   %c3: %C = value_binding c3, %.loc52.4
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c4.patt: %pattern_type.217 = value_binding_pattern c4 [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c1.ref.loc62: %C = name_ref c1, %c1
// CHECK:STDOUT:   %c3.ref.loc64: %C = name_ref c3, %c3
// CHECK:STDOUT:   %.loc63.1: ref %C = temporary_storage
// CHECK:STDOUT:   %.loc62: ref %C = value_as_ref %c1.ref.loc62
// CHECK:STDOUT:   %addr.loc63.1: %ptr.d9e = addr_of %.loc62
// CHECK:STDOUT:   %.loc64: ref %C = value_as_ref %c3.ref.loc64
// CHECK:STDOUT:   %addr.loc63.2: %ptr.d9e = addr_of %.loc64
// CHECK:STDOUT:   %addr.loc63.3: %ptr.d9e = addr_of %.loc63.1
// CHECK:STDOUT:   %operator_Plus__carbon_thunk.call.loc63: init %empty_tuple.type = call imports.%operator_Plus__carbon_thunk.decl(%addr.loc63.1, %addr.loc63.2, %addr.loc63.3)
// CHECK:STDOUT:   %.loc63.2: init %C to %.loc63.1 = mark_in_place_init %operator_Plus__carbon_thunk.call.loc63
// CHECK:STDOUT:   %.loc59: type = splice_block %C.ref.loc59 [concrete = constants.%C] {
// CHECK:STDOUT:     %Cpp.ref.loc58: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %C.ref.loc59: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc63.3: ref %C = temporary %.loc63.1, %.loc63.2
// CHECK:STDOUT:   %.loc63.4: %C = acquire_value %.loc63.3
// CHECK:STDOUT:   %c4: %C = value_binding c4, %.loc63.4
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c5.patt: %pattern_type.217 = value_binding_pattern c5 [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c4.ref: %C = name_ref c4, %c4
// CHECK:STDOUT:   %c3.ref.loc76: %C = name_ref c3, %c3
// CHECK:STDOUT:   %.loc75.1: ref %C = temporary_storage
// CHECK:STDOUT:   %.loc74: ref %C = value_as_ref %c4.ref
// CHECK:STDOUT:   %addr.loc75.1: %ptr.d9e = addr_of %.loc74
// CHECK:STDOUT:   %.loc76: ref %C = value_as_ref %c3.ref.loc76
// CHECK:STDOUT:   %addr.loc75.2: %ptr.d9e = addr_of %.loc76
// CHECK:STDOUT:   %addr.loc75.3: %ptr.d9e = addr_of %.loc75.1
// CHECK:STDOUT:   %operator_Plus__carbon_thunk.call.loc75: init %empty_tuple.type = call imports.%operator_Plus__carbon_thunk.decl(%addr.loc75.1, %addr.loc75.2, %addr.loc75.3)
// CHECK:STDOUT:   %.loc75.2: init %C to %.loc75.1 = mark_in_place_init %operator_Plus__carbon_thunk.call.loc75
// CHECK:STDOUT:   %.loc71: type = splice_block %C.ref.loc71 [concrete = constants.%C] {
// CHECK:STDOUT:     %Cpp.ref.loc70: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %C.ref.loc71: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc75.3: ref %C = temporary %.loc75.1, %.loc75.2
// CHECK:STDOUT:   %.loc75.4: %C = acquire_value %.loc75.3
// CHECK:STDOUT:   %c5: %C = value_binding c5, %.loc75.4
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %C.Op.bound.loc75: <bound method> = bound_method %.loc75.3, constants.%C.Op
// CHECK:STDOUT:   %Op.ref.loc75: %C.cpp_destructor.type = name_ref Op, imports.%C.cpp_destructor.decl [concrete = constants.%C.cpp_destructor]
// CHECK:STDOUT:   %C.cpp_destructor.bound.loc75: <bound method> = bound_method %.loc75.3, %Op.ref.loc75
// CHECK:STDOUT:   %C.cpp_destructor.call.loc75: init %empty_tuple.type = call %C.cpp_destructor.bound.loc75(%.loc75.3)
// CHECK:STDOUT:   %C.Op.bound.loc63: <bound method> = bound_method %.loc63.3, constants.%C.Op
// CHECK:STDOUT:   %Op.ref.loc63: %C.cpp_destructor.type = name_ref Op, imports.%C.cpp_destructor.decl [concrete = constants.%C.cpp_destructor]
// CHECK:STDOUT:   %C.cpp_destructor.bound.loc63: <bound method> = bound_method %.loc63.3, %Op.ref.loc63
// CHECK:STDOUT:   %C.cpp_destructor.call.loc63: init %empty_tuple.type = call %C.cpp_destructor.bound.loc63(%.loc63.3)
// CHECK:STDOUT:   %C.Op.bound.loc52: <bound method> = bound_method %.loc52.3, constants.%C.Op
// CHECK:STDOUT:   %Op.ref.loc52: %C.cpp_destructor.type = name_ref Op, imports.%C.cpp_destructor.decl [concrete = constants.%C.cpp_destructor]
// CHECK:STDOUT:   %C.cpp_destructor.bound.loc52: <bound method> = bound_method %.loc52.3, %Op.ref.loc52
// CHECK:STDOUT:   %C.cpp_destructor.call.loc52: init %empty_tuple.type = call %C.cpp_destructor.bound.loc52(%.loc52.3)
// CHECK:STDOUT:   %C.Op.bound.loc42: <bound method> = bound_method %.loc42.3, constants.%C.Op
// CHECK:STDOUT:   %Op.ref.loc42: %C.cpp_destructor.type = name_ref Op, imports.%C.cpp_destructor.decl [concrete = constants.%C.cpp_destructor]
// CHECK:STDOUT:   %C.cpp_destructor.bound.loc42: <bound method> = bound_method %.loc42.3, %Op.ref.loc42
// CHECK:STDOUT:   %C.cpp_destructor.call.loc42: init %empty_tuple.type = call %C.cpp_destructor.bound.loc42(%.loc42.3)
// CHECK:STDOUT:   %C.Op.bound.loc27: <bound method> = bound_method %.loc27.3, constants.%C.Op
// CHECK:STDOUT:   %Op.ref.loc27: %C.cpp_destructor.type = name_ref Op, imports.%C.cpp_destructor.decl [concrete = constants.%C.cpp_destructor]
// CHECK:STDOUT:   %C.cpp_destructor.bound.loc27: <bound method> = bound_method %.loc27.3, %Op.ref.loc27
// CHECK:STDOUT:   %C.cpp_destructor.call.loc27: init %empty_tuple.type = call %C.cpp_destructor.bound.loc27(%.loc27.3)
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
// CHECK:STDOUT:   %Cpp.ref.loc21: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %C.ref.loc22: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %C.ref.loc24: %C.C.cpp_overload_set.type = name_ref C, imports.%C.C.cpp_overload_set.value [concrete = constants.%C.C.cpp_overload_set.value]
// CHECK:STDOUT:   %.loc14: ref %C = splice_block %c1.var {}
// CHECK:STDOUT:   %addr.loc27: %ptr.d9e = addr_of %.loc14
// CHECK:STDOUT:   %C__carbon_thunk.call.loc27: init %empty_tuple.type = call imports.%C__carbon_thunk.decl(%addr.loc27)
// CHECK:STDOUT:   %.loc27: init %C to %.loc14 = mark_in_place_init %C__carbon_thunk.call.loc27
// CHECK:STDOUT:   assign %c1.var, %.loc27
// CHECK:STDOUT:   %.loc18: type = splice_block %C.ref.loc18 [concrete = constants.%C] {
// CHECK:STDOUT:     %Cpp.ref.loc17: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %C.ref.loc18: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c1: ref %C = ref_binding c1, %c1.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c2.patt: %pattern_type.217 = ref_binding_pattern c2 [concrete]
// CHECK:STDOUT:     %c2.var_patt: %pattern_type.217 = var_pattern %c2.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c2.var: ref %C = var %c2.var_patt
// CHECK:STDOUT:   %Cpp.ref.loc36: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %C.ref.loc37: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %C.ref.loc39: %C.C.cpp_overload_set.type = name_ref C, imports.%C.C.cpp_overload_set.value [concrete = constants.%C.C.cpp_overload_set.value]
// CHECK:STDOUT:   %.loc29: ref %C = splice_block %c2.var {}
// CHECK:STDOUT:   %addr.loc42: %ptr.d9e = addr_of %.loc29
// CHECK:STDOUT:   %C__carbon_thunk.call.loc42: init %empty_tuple.type = call imports.%C__carbon_thunk.decl(%addr.loc42)
// CHECK:STDOUT:   %.loc42: init %C to %.loc29 = mark_in_place_init %C__carbon_thunk.call.loc42
// CHECK:STDOUT:   assign %c2.var, %.loc42
// CHECK:STDOUT:   %.loc33: type = splice_block %C.ref.loc33 [concrete = constants.%C] {
// CHECK:STDOUT:     %Cpp.ref.loc32: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %C.ref.loc33: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c2: ref %C = ref_binding c2, %c2.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %greater_than.patt: %pattern_type.831 = value_binding_pattern greater_than [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c1.ref.loc50: ref %C = name_ref c1, %c1
// CHECK:STDOUT:   %c2.ref.loc52: ref %C = name_ref c2, %c2
// CHECK:STDOUT:   %.loc50.1: %C = acquire_value %c1.ref.loc50
// CHECK:STDOUT:   %.loc52.1: %C = acquire_value %c2.ref.loc52
// CHECK:STDOUT:   %.loc50.2: ref %C = value_as_ref %.loc50.1
// CHECK:STDOUT:   %addr.loc51.1: %ptr.d9e = addr_of %.loc50.2
// CHECK:STDOUT:   %.loc52.2: ref %C = value_as_ref %.loc52.1
// CHECK:STDOUT:   %addr.loc51.2: %ptr.d9e = addr_of %.loc52.2
// CHECK:STDOUT:   %.loc51.1: ref bool = temporary_storage
// CHECK:STDOUT:   %addr.loc51.3: %ptr.bb2 = addr_of %.loc51.1
// CHECK:STDOUT:   %operator_Greater__carbon_thunk.call: init %empty_tuple.type = call imports.%operator_Greater__carbon_thunk.decl(%addr.loc51.1, %addr.loc51.2, %addr.loc51.3)
// CHECK:STDOUT:   %.loc51.2: init bool to %.loc51.1 = mark_in_place_init %operator_Greater__carbon_thunk.call
// CHECK:STDOUT:   %.loc48: type = type_literal bool [concrete = bool]
// CHECK:STDOUT:   %.loc51.3: ref bool = temporary %.loc51.1, %.loc51.2
// CHECK:STDOUT:   %.loc51.4: bool = acquire_value %.loc51.3
// CHECK:STDOUT:   %greater_than: bool = value_binding greater_than, %.loc51.4
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %less_than.patt: %pattern_type.831 = value_binding_pattern less_than [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c1.ref.loc60: ref %C = name_ref c1, %c1
// CHECK:STDOUT:   %c2.ref.loc62: ref %C = name_ref c2, %c2
// CHECK:STDOUT:   %.loc58: type = type_literal bool [concrete = bool]
// CHECK:STDOUT:   %less_than: bool = value_binding less_than, <error> [concrete = <error>]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %greater_than_or_equal.patt: %pattern_type.831 = value_binding_pattern greater_than_or_equal [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c1.ref.loc70: ref %C = name_ref c1, %c1
// CHECK:STDOUT:   %c2.ref.loc72: ref %C = name_ref c2, %c2
// CHECK:STDOUT:   %.loc68: type = type_literal bool [concrete = bool]
// CHECK:STDOUT:   %greater_than_or_equal: bool = value_binding greater_than_or_equal, <error> [concrete = <error>]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %less_than_or_equal.patt: %pattern_type.831 = value_binding_pattern less_than_or_equal [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c1.ref.loc80: ref %C = name_ref c1, %c1
// CHECK:STDOUT:   %c2.ref.loc82: ref %C = name_ref c2, %c2
// CHECK:STDOUT:   %.loc80.1: %C = acquire_value %c1.ref.loc80
// CHECK:STDOUT:   %.loc82.1: %C = acquire_value %c2.ref.loc82
// CHECK:STDOUT:   %.loc80.2: ref %C = value_as_ref %.loc80.1
// CHECK:STDOUT:   %addr.loc81.1: %ptr.d9e = addr_of %.loc80.2
// CHECK:STDOUT:   %.loc82.2: ref %C = value_as_ref %.loc82.1
// CHECK:STDOUT:   %addr.loc81.2: %ptr.d9e = addr_of %.loc82.2
// CHECK:STDOUT:   %.loc81.1: ref bool = temporary_storage
// CHECK:STDOUT:   %addr.loc81.3: %ptr.bb2 = addr_of %.loc81.1
// CHECK:STDOUT:   %operator_LessEqual__carbon_thunk.call: init %empty_tuple.type = call imports.%operator_LessEqual__carbon_thunk.decl(%addr.loc81.1, %addr.loc81.2, %addr.loc81.3)
// CHECK:STDOUT:   %.loc81.2: init bool to %.loc81.1 = mark_in_place_init %operator_LessEqual__carbon_thunk.call
// CHECK:STDOUT:   %.loc78: type = type_literal bool [concrete = bool]
// CHECK:STDOUT:   %.loc81.3: ref bool = temporary %.loc81.1, %.loc81.2
// CHECK:STDOUT:   %.loc81.4: bool = acquire_value %.loc81.3
// CHECK:STDOUT:   %less_than_or_equal: bool = value_binding less_than_or_equal, %.loc81.4
// CHECK:STDOUT:   %Destroy.Op.bound.loc81: <bound method> = bound_method %.loc81.3, constants.%Destroy.Op
// CHECK:STDOUT:   %Destroy.Op.call.loc81: init %empty_tuple.type = call %Destroy.Op.bound.loc81(%.loc81.3)
// CHECK:STDOUT:   %Destroy.Op.bound.loc51: <bound method> = bound_method %.loc51.3, constants.%Destroy.Op
// CHECK:STDOUT:   %Destroy.Op.call.loc51: init %empty_tuple.type = call %Destroy.Op.bound.loc51(%.loc51.3)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %C.Op.bound.loc29: <bound method> = bound_method %c2.var, constants.%C.Op
// CHECK:STDOUT:   %Op.ref.loc29: %C.cpp_destructor.type = name_ref Op, imports.%C.cpp_destructor.decl [concrete = constants.%C.cpp_destructor]
// CHECK:STDOUT:   %C.cpp_destructor.bound.loc29: <bound method> = bound_method %c2.var, %Op.ref.loc29
// CHECK:STDOUT:   %C.cpp_destructor.call.loc29: init %empty_tuple.type = call %C.cpp_destructor.bound.loc29(%c2.var)
// CHECK:STDOUT:   %C.Op.bound.loc14: <bound method> = bound_method %c1.var, constants.%C.Op
// CHECK:STDOUT:   %Op.ref.loc14: %C.cpp_destructor.type = name_ref Op, imports.%C.cpp_destructor.decl [concrete = constants.%C.cpp_destructor]
// CHECK:STDOUT:   %C.cpp_destructor.bound.loc14: <bound method> = bound_method %c1.var, %Op.ref.loc14
// CHECK:STDOUT:   %C.cpp_destructor.call.loc14: init %empty_tuple.type = call %C.cpp_destructor.bound.loc14(%c1.var)
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
// CHECK:STDOUT:   %Cpp.ref.loc21: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %C.ref.loc22: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %C.ref.loc24: %C.C.cpp_overload_set.type = name_ref C, imports.%C.C.cpp_overload_set.value [concrete = constants.%C.C.cpp_overload_set.value]
// CHECK:STDOUT:   %.loc14: ref %C = splice_block %c1.var {}
// CHECK:STDOUT:   %addr.loc27: %ptr.d9e = addr_of %.loc14
// CHECK:STDOUT:   %C__carbon_thunk.call.loc27: init %empty_tuple.type = call imports.%C__carbon_thunk.decl(%addr.loc27)
// CHECK:STDOUT:   %.loc27: init %C to %.loc14 = mark_in_place_init %C__carbon_thunk.call.loc27
// CHECK:STDOUT:   assign %c1.var, %.loc27
// CHECK:STDOUT:   %.loc18: type = splice_block %C.ref.loc18 [concrete = constants.%C] {
// CHECK:STDOUT:     %Cpp.ref.loc17: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %C.ref.loc18: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c1: ref %C = ref_binding c1, %c1.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c2.patt: %pattern_type.217 = ref_binding_pattern c2 [concrete]
// CHECK:STDOUT:     %c2.var_patt: %pattern_type.217 = var_pattern %c2.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c2.var: ref %C = var %c2.var_patt
// CHECK:STDOUT:   %Cpp.ref.loc36: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %C.ref.loc37: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %C.ref.loc39: %C.C.cpp_overload_set.type = name_ref C, imports.%C.C.cpp_overload_set.value [concrete = constants.%C.C.cpp_overload_set.value]
// CHECK:STDOUT:   %.loc29: ref %C = splice_block %c2.var {}
// CHECK:STDOUT:   %addr.loc42: %ptr.d9e = addr_of %.loc29
// CHECK:STDOUT:   %C__carbon_thunk.call.loc42: init %empty_tuple.type = call imports.%C__carbon_thunk.decl(%addr.loc42)
// CHECK:STDOUT:   %.loc42: init %C to %.loc29 = mark_in_place_init %C__carbon_thunk.call.loc42
// CHECK:STDOUT:   assign %c2.var, %.loc42
// CHECK:STDOUT:   %.loc33: type = splice_block %C.ref.loc33 [concrete = constants.%C] {
// CHECK:STDOUT:     %Cpp.ref.loc32: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %C.ref.loc33: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c2: ref %C = ref_binding c2, %c2.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %equal.patt: %pattern_type.831 = value_binding_pattern equal [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c1.ref.loc50: ref %C = name_ref c1, %c1
// CHECK:STDOUT:   %c2.ref.loc52: ref %C = name_ref c2, %c2
// CHECK:STDOUT:   %.loc50.1: %C = acquire_value %c1.ref.loc50
// CHECK:STDOUT:   %.loc52.1: %C = acquire_value %c2.ref.loc52
// CHECK:STDOUT:   %.loc50.2: ref %C = value_as_ref %.loc50.1
// CHECK:STDOUT:   %addr.loc51.1: %ptr.d9e = addr_of %.loc50.2
// CHECK:STDOUT:   %.loc52.2: ref %C = value_as_ref %.loc52.1
// CHECK:STDOUT:   %addr.loc51.2: %ptr.d9e = addr_of %.loc52.2
// CHECK:STDOUT:   %.loc51.1: ref bool = temporary_storage
// CHECK:STDOUT:   %addr.loc51.3: %ptr.bb2 = addr_of %.loc51.1
// CHECK:STDOUT:   %operator_EqualEqual__carbon_thunk.call: init %empty_tuple.type = call imports.%operator_EqualEqual__carbon_thunk.decl(%addr.loc51.1, %addr.loc51.2, %addr.loc51.3)
// CHECK:STDOUT:   %.loc51.2: init bool to %.loc51.1 = mark_in_place_init %operator_EqualEqual__carbon_thunk.call
// CHECK:STDOUT:   %.loc48: type = type_literal bool [concrete = bool]
// CHECK:STDOUT:   %.loc51.3: ref bool = temporary %.loc51.1, %.loc51.2
// CHECK:STDOUT:   %.loc51.4: bool = acquire_value %.loc51.3
// CHECK:STDOUT:   %equal: bool = value_binding equal, %.loc51.4
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %not_equal.patt: %pattern_type.831 = value_binding_pattern not_equal [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c1.ref.loc60: ref %C = name_ref c1, %c1
// CHECK:STDOUT:   %c2.ref.loc62: ref %C = name_ref c2, %c2
// CHECK:STDOUT:   %.loc58: type = type_literal bool [concrete = bool]
// CHECK:STDOUT:   %not_equal: bool = value_binding not_equal, <error> [concrete = <error>]
// CHECK:STDOUT:   %Destroy.Op.bound: <bound method> = bound_method %.loc51.3, constants.%Destroy.Op
// CHECK:STDOUT:   %Destroy.Op.call: init %empty_tuple.type = call %Destroy.Op.bound(%.loc51.3)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %C.Op.bound.loc29: <bound method> = bound_method %c2.var, constants.%C.Op
// CHECK:STDOUT:   %Op.ref.loc29: %C.cpp_destructor.type = name_ref Op, imports.%C.cpp_destructor.decl [concrete = constants.%C.cpp_destructor]
// CHECK:STDOUT:   %C.cpp_destructor.bound.loc29: <bound method> = bound_method %c2.var, %Op.ref.loc29
// CHECK:STDOUT:   %C.cpp_destructor.call.loc29: init %empty_tuple.type = call %C.cpp_destructor.bound.loc29(%c2.var)
// CHECK:STDOUT:   %C.Op.bound.loc14: <bound method> = bound_method %c1.var, constants.%C.Op
// CHECK:STDOUT:   %Op.ref.loc14: %C.cpp_destructor.type = name_ref Op, imports.%C.cpp_destructor.decl [concrete = constants.%C.cpp_destructor]
// CHECK:STDOUT:   %C.cpp_destructor.bound.loc14: <bound method> = bound_method %c1.var, %Op.ref.loc14
// CHECK:STDOUT:   %C.cpp_destructor.call.loc14: init %empty_tuple.type = call %C.cpp_destructor.bound.loc14(%c1.var)
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
// CHECK:STDOUT:   %Cpp.ref.loc23: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %N.ref.loc24: <namespace> = name_ref N, imports.%N [concrete = imports.%N]
// CHECK:STDOUT:   %C.ref.loc26: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %C.ref.loc28: %C.C.cpp_overload_set.type = name_ref C, imports.%C.C.cpp_overload_set.value [concrete = constants.%C.C.cpp_overload_set.value]
// CHECK:STDOUT:   %.loc31.1: ref %C = temporary_storage
// CHECK:STDOUT:   %addr.loc31: %ptr.838 = addr_of %.loc31.1
// CHECK:STDOUT:   %C__carbon_thunk.call.loc31: init %empty_tuple.type = call imports.%C__carbon_thunk.decl(%addr.loc31)
// CHECK:STDOUT:   %.loc31.2: init %C to %.loc31.1 = mark_in_place_init %C__carbon_thunk.call.loc31
// CHECK:STDOUT:   %.loc20: type = splice_block %C.ref.loc20 [concrete = constants.%C] {
// CHECK:STDOUT:     %Cpp.ref.loc17: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %N.ref.loc18: <namespace> = name_ref N, imports.%N [concrete = imports.%N]
// CHECK:STDOUT:     %C.ref.loc20: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc31.3: ref %C = temporary %.loc31.1, %.loc31.2
// CHECK:STDOUT:   %.loc31.4: %C = acquire_value %.loc31.3
// CHECK:STDOUT:   %c1: %C = value_binding c1, %.loc31.4
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c2.patt: %pattern_type.69f = value_binding_pattern c2 [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc42: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %N.ref.loc43: <namespace> = name_ref N, imports.%N [concrete = imports.%N]
// CHECK:STDOUT:   %C.ref.loc45: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %C.ref.loc47: %C.C.cpp_overload_set.type = name_ref C, imports.%C.C.cpp_overload_set.value [concrete = constants.%C.C.cpp_overload_set.value]
// CHECK:STDOUT:   %.loc50.1: ref %C = temporary_storage
// CHECK:STDOUT:   %addr.loc50: %ptr.838 = addr_of %.loc50.1
// CHECK:STDOUT:   %C__carbon_thunk.call.loc50: init %empty_tuple.type = call imports.%C__carbon_thunk.decl(%addr.loc50)
// CHECK:STDOUT:   %.loc50.2: init %C to %.loc50.1 = mark_in_place_init %C__carbon_thunk.call.loc50
// CHECK:STDOUT:   %.loc39: type = splice_block %C.ref.loc39 [concrete = constants.%C] {
// CHECK:STDOUT:     %Cpp.ref.loc36: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %N.ref.loc37: <namespace> = name_ref N, imports.%N [concrete = imports.%N]
// CHECK:STDOUT:     %C.ref.loc39: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc50.3: ref %C = temporary %.loc50.1, %.loc50.2
// CHECK:STDOUT:   %.loc50.4: %C = acquire_value %.loc50.3
// CHECK:STDOUT:   %c2: %C = value_binding c2, %.loc50.4
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c3.patt: %pattern_type.69f = value_binding_pattern c3 [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c1.ref: %C = name_ref c1, %c1
// CHECK:STDOUT:   %c2.ref: %C = name_ref c2, %c2
// CHECK:STDOUT:   %.loc63.1: ref %C = temporary_storage
// CHECK:STDOUT:   %.loc62: ref %C = value_as_ref %c1.ref
// CHECK:STDOUT:   %addr.loc63.1: %ptr.838 = addr_of %.loc62
// CHECK:STDOUT:   %.loc64: ref %C = value_as_ref %c2.ref
// CHECK:STDOUT:   %addr.loc63.2: %ptr.838 = addr_of %.loc64
// CHECK:STDOUT:   %addr.loc63.3: %ptr.838 = addr_of %.loc63.1
// CHECK:STDOUT:   %operator_Plus__carbon_thunk.call: init %empty_tuple.type = call imports.%operator_Plus__carbon_thunk.decl(%addr.loc63.1, %addr.loc63.2, %addr.loc63.3)
// CHECK:STDOUT:   %.loc63.2: init %C to %.loc63.1 = mark_in_place_init %operator_Plus__carbon_thunk.call
// CHECK:STDOUT:   %.loc59: type = splice_block %C.ref.loc59 [concrete = constants.%C] {
// CHECK:STDOUT:     %Cpp.ref.loc56: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %N.ref.loc57: <namespace> = name_ref N, imports.%N [concrete = imports.%N]
// CHECK:STDOUT:     %C.ref.loc59: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc63.3: ref %C = temporary %.loc63.1, %.loc63.2
// CHECK:STDOUT:   %.loc63.4: %C = acquire_value %.loc63.3
// CHECK:STDOUT:   %c3: %C = value_binding c3, %.loc63.4
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %C.Op.bound.loc63: <bound method> = bound_method %.loc63.3, constants.%C.Op
// CHECK:STDOUT:   %Op.ref.loc63: %C.cpp_destructor.type = name_ref Op, imports.%C.cpp_destructor.decl [concrete = constants.%C.cpp_destructor]
// CHECK:STDOUT:   %C.cpp_destructor.bound.loc63: <bound method> = bound_method %.loc63.3, %Op.ref.loc63
// CHECK:STDOUT:   %C.cpp_destructor.call.loc63: init %empty_tuple.type = call %C.cpp_destructor.bound.loc63(%.loc63.3)
// CHECK:STDOUT:   %C.Op.bound.loc50: <bound method> = bound_method %.loc50.3, constants.%C.Op
// CHECK:STDOUT:   %Op.ref.loc50: %C.cpp_destructor.type = name_ref Op, imports.%C.cpp_destructor.decl [concrete = constants.%C.cpp_destructor]
// CHECK:STDOUT:   %C.cpp_destructor.bound.loc50: <bound method> = bound_method %.loc50.3, %Op.ref.loc50
// CHECK:STDOUT:   %C.cpp_destructor.call.loc50: init %empty_tuple.type = call %C.cpp_destructor.bound.loc50(%.loc50.3)
// CHECK:STDOUT:   %C.Op.bound.loc31: <bound method> = bound_method %.loc31.3, constants.%C.Op
// CHECK:STDOUT:   %Op.ref.loc31: %C.cpp_destructor.type = name_ref Op, imports.%C.cpp_destructor.decl [concrete = constants.%C.cpp_destructor]
// CHECK:STDOUT:   %C.cpp_destructor.bound.loc31: <bound method> = bound_method %.loc31.3, %Op.ref.loc31
// CHECK:STDOUT:   %C.cpp_destructor.call.loc31: init %empty_tuple.type = call %C.cpp_destructor.bound.loc31(%.loc31.3)
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
// CHECK:STDOUT:   %Cpp.ref.loc23: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %N1.ref.loc24: <namespace> = name_ref N1, imports.%N1 [concrete = imports.%N1]
// CHECK:STDOUT:   %C1.ref.loc26: type = name_ref C1, imports.%C1.decl [concrete = constants.%C1]
// CHECK:STDOUT:   %C1.ref.loc28: %C1.C1.cpp_overload_set.type = name_ref C1, imports.%C1.C1.cpp_overload_set.value [concrete = constants.%C1.C1.cpp_overload_set.value]
// CHECK:STDOUT:   %.loc31.1: ref %C1 = temporary_storage
// CHECK:STDOUT:   %addr.loc31: %ptr.087 = addr_of %.loc31.1
// CHECK:STDOUT:   %C1__carbon_thunk.call: init %empty_tuple.type = call imports.%C1__carbon_thunk.decl(%addr.loc31)
// CHECK:STDOUT:   %.loc31.2: init %C1 to %.loc31.1 = mark_in_place_init %C1__carbon_thunk.call
// CHECK:STDOUT:   %.loc20: type = splice_block %C1.ref.loc20 [concrete = constants.%C1] {
// CHECK:STDOUT:     %Cpp.ref.loc17: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %N1.ref.loc18: <namespace> = name_ref N1, imports.%N1 [concrete = imports.%N1]
// CHECK:STDOUT:     %C1.ref.loc20: type = name_ref C1, imports.%C1.decl [concrete = constants.%C1]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc31.3: ref %C1 = temporary %.loc31.1, %.loc31.2
// CHECK:STDOUT:   %.loc31.4: %C1 = acquire_value %.loc31.3
// CHECK:STDOUT:   %c1: %C1 = value_binding c1, %.loc31.4
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c2.patt: %pattern_type.846 = value_binding_pattern c2 [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc42: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %N2.ref.loc43: <namespace> = name_ref N2, imports.%N2 [concrete = imports.%N2]
// CHECK:STDOUT:   %C2.ref.loc45: type = name_ref C2, imports.%C2.decl [concrete = constants.%C2]
// CHECK:STDOUT:   %C2.ref.loc47: %C2.C2.cpp_overload_set.type = name_ref C2, imports.%C2.C2.cpp_overload_set.value [concrete = constants.%C2.C2.cpp_overload_set.value]
// CHECK:STDOUT:   %.loc50.1: ref %C2 = temporary_storage
// CHECK:STDOUT:   %addr.loc50: %ptr.51f = addr_of %.loc50.1
// CHECK:STDOUT:   %C2__carbon_thunk.call: init %empty_tuple.type = call imports.%C2__carbon_thunk.decl(%addr.loc50)
// CHECK:STDOUT:   %.loc50.2: init %C2 to %.loc50.1 = mark_in_place_init %C2__carbon_thunk.call
// CHECK:STDOUT:   %.loc39: type = splice_block %C2.ref.loc39 [concrete = constants.%C2] {
// CHECK:STDOUT:     %Cpp.ref.loc36: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %N2.ref.loc37: <namespace> = name_ref N2, imports.%N2 [concrete = imports.%N2]
// CHECK:STDOUT:     %C2.ref.loc39: type = name_ref C2, imports.%C2.decl [concrete = constants.%C2]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc50.3: ref %C2 = temporary %.loc50.1, %.loc50.2
// CHECK:STDOUT:   %.loc50.4: %C2 = acquire_value %.loc50.3
// CHECK:STDOUT:   %c2: %C2 = value_binding c2, %.loc50.4
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c3.patt: %pattern_type.846 = value_binding_pattern c3 [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c1.ref.loc62: %C1 = name_ref c1, %c1
// CHECK:STDOUT:   %c2.ref.loc64: %C2 = name_ref c2, %c2
// CHECK:STDOUT:   %.loc63.1: ref %C2 = temporary_storage
// CHECK:STDOUT:   %.loc62: ref %C1 = value_as_ref %c1.ref.loc62
// CHECK:STDOUT:   %addr.loc63.1: %ptr.087 = addr_of %.loc62
// CHECK:STDOUT:   %.loc64: ref %C2 = value_as_ref %c2.ref.loc64
// CHECK:STDOUT:   %addr.loc63.2: %ptr.51f = addr_of %.loc64
// CHECK:STDOUT:   %addr.loc63.3: %ptr.51f = addr_of %.loc63.1
// CHECK:STDOUT:   %operator_Plus__carbon_thunk.call: init %empty_tuple.type = call imports.%operator_Plus__carbon_thunk.decl(%addr.loc63.1, %addr.loc63.2, %addr.loc63.3)
// CHECK:STDOUT:   %.loc63.2: init %C2 to %.loc63.1 = mark_in_place_init %operator_Plus__carbon_thunk.call
// CHECK:STDOUT:   %.loc59: type = splice_block %C2.ref.loc59 [concrete = constants.%C2] {
// CHECK:STDOUT:     %Cpp.ref.loc56: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %N2.ref.loc57: <namespace> = name_ref N2, imports.%N2 [concrete = imports.%N2]
// CHECK:STDOUT:     %C2.ref.loc59: type = name_ref C2, imports.%C2.decl [concrete = constants.%C2]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc63.3: ref %C2 = temporary %.loc63.1, %.loc63.2
// CHECK:STDOUT:   %.loc63.4: %C2 = acquire_value %.loc63.3
// CHECK:STDOUT:   %c3: %C2 = value_binding c3, %.loc63.4
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c4.patt: %pattern_type.846 = value_binding_pattern c4 [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c2.ref.loc76: %C2 = name_ref c2, %c2
// CHECK:STDOUT:   %c1.ref.loc78: %C1 = name_ref c1, %c1
// CHECK:STDOUT:   %.loc77.1: ref %C2 = temporary_storage
// CHECK:STDOUT:   %.loc76: ref %C2 = value_as_ref %c2.ref.loc76
// CHECK:STDOUT:   %addr.loc77.1: %ptr.51f = addr_of %.loc76
// CHECK:STDOUT:   %.loc78: ref %C1 = value_as_ref %c1.ref.loc78
// CHECK:STDOUT:   %addr.loc77.2: %ptr.087 = addr_of %.loc78
// CHECK:STDOUT:   %addr.loc77.3: %ptr.51f = addr_of %.loc77.1
// CHECK:STDOUT:   %operator_Minus__carbon_thunk.call: init %empty_tuple.type = call imports.%operator_Minus__carbon_thunk.decl(%addr.loc77.1, %addr.loc77.2, %addr.loc77.3)
// CHECK:STDOUT:   %.loc77.2: init %C2 to %.loc77.1 = mark_in_place_init %operator_Minus__carbon_thunk.call
// CHECK:STDOUT:   %.loc73: type = splice_block %C2.ref.loc73 [concrete = constants.%C2] {
// CHECK:STDOUT:     %Cpp.ref.loc70: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %N2.ref.loc71: <namespace> = name_ref N2, imports.%N2 [concrete = imports.%N2]
// CHECK:STDOUT:     %C2.ref.loc73: type = name_ref C2, imports.%C2.decl [concrete = constants.%C2]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc77.3: ref %C2 = temporary %.loc77.1, %.loc77.2
// CHECK:STDOUT:   %.loc77.4: %C2 = acquire_value %.loc77.3
// CHECK:STDOUT:   %c4: %C2 = value_binding c4, %.loc77.4
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %C2.Op.bound.loc77: <bound method> = bound_method %.loc77.3, constants.%C2.Op
// CHECK:STDOUT:   %Op.ref.loc77: %C2.cpp_destructor.type = name_ref Op, imports.%C2.cpp_destructor.decl [concrete = constants.%C2.cpp_destructor]
// CHECK:STDOUT:   %C2.cpp_destructor.bound.loc77: <bound method> = bound_method %.loc77.3, %Op.ref.loc77
// CHECK:STDOUT:   %C2.cpp_destructor.call.loc77: init %empty_tuple.type = call %C2.cpp_destructor.bound.loc77(%.loc77.3)
// CHECK:STDOUT:   %C2.Op.bound.loc63: <bound method> = bound_method %.loc63.3, constants.%C2.Op
// CHECK:STDOUT:   %Op.ref.loc63: %C2.cpp_destructor.type = name_ref Op, imports.%C2.cpp_destructor.decl [concrete = constants.%C2.cpp_destructor]
// CHECK:STDOUT:   %C2.cpp_destructor.bound.loc63: <bound method> = bound_method %.loc63.3, %Op.ref.loc63
// CHECK:STDOUT:   %C2.cpp_destructor.call.loc63: init %empty_tuple.type = call %C2.cpp_destructor.bound.loc63(%.loc63.3)
// CHECK:STDOUT:   %C2.Op.bound.loc50: <bound method> = bound_method %.loc50.3, constants.%C2.Op
// CHECK:STDOUT:   %Op.ref.loc50: %C2.cpp_destructor.type = name_ref Op, imports.%C2.cpp_destructor.decl [concrete = constants.%C2.cpp_destructor]
// CHECK:STDOUT:   %C2.cpp_destructor.bound.loc50: <bound method> = bound_method %.loc50.3, %Op.ref.loc50
// CHECK:STDOUT:   %C2.cpp_destructor.call.loc50: init %empty_tuple.type = call %C2.cpp_destructor.bound.loc50(%.loc50.3)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %C1.Op.bound: <bound method> = bound_method %.loc31.3, constants.%C1.Op
// CHECK:STDOUT:   %Op.ref.loc31: %C1.cpp_destructor.type = name_ref Op, imports.%C1.cpp_destructor.decl [concrete = constants.%C1.cpp_destructor]
// CHECK:STDOUT:   %C1.cpp_destructor.bound: <bound method> = bound_method %.loc31.3, %Op.ref.loc31
// CHECK:STDOUT:   %C1.cpp_destructor.call: init %empty_tuple.type = call %C1.cpp_destructor.bound(%.loc31.3)
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
// CHECK:STDOUT:   %Cpp.ref.loc23: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %N.ref.loc24: <namespace> = name_ref N, imports.%N [concrete = imports.%N]
// CHECK:STDOUT:   %C.ref.loc26: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %C.ref.loc28: %C.C.cpp_overload_set.type = name_ref C, imports.%C.C.cpp_overload_set.value [concrete = constants.%C.C.cpp_overload_set.value]
// CHECK:STDOUT:   %.loc31.1: ref %C = temporary_storage
// CHECK:STDOUT:   %addr.loc31: %ptr.838 = addr_of %.loc31.1
// CHECK:STDOUT:   %C__carbon_thunk.call.loc31: init %empty_tuple.type = call imports.%C__carbon_thunk.decl(%addr.loc31)
// CHECK:STDOUT:   %.loc31.2: init %C to %.loc31.1 = mark_in_place_init %C__carbon_thunk.call.loc31
// CHECK:STDOUT:   %.loc20: type = splice_block %C.ref.loc20 [concrete = constants.%C] {
// CHECK:STDOUT:     %Cpp.ref.loc17: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %N.ref.loc18: <namespace> = name_ref N, imports.%N [concrete = imports.%N]
// CHECK:STDOUT:     %C.ref.loc20: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc31.3: ref %C = temporary %.loc31.1, %.loc31.2
// CHECK:STDOUT:   %.loc31.4: %C = acquire_value %.loc31.3
// CHECK:STDOUT:   %c1: %C = value_binding c1, %.loc31.4
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c2.patt: %pattern_type.69f = value_binding_pattern c2 [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc42: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %N.ref.loc43: <namespace> = name_ref N, imports.%N [concrete = imports.%N]
// CHECK:STDOUT:   %C.ref.loc45: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %C.ref.loc47: %C.C.cpp_overload_set.type = name_ref C, imports.%C.C.cpp_overload_set.value [concrete = constants.%C.C.cpp_overload_set.value]
// CHECK:STDOUT:   %.loc50.1: ref %C = temporary_storage
// CHECK:STDOUT:   %addr.loc50: %ptr.838 = addr_of %.loc50.1
// CHECK:STDOUT:   %C__carbon_thunk.call.loc50: init %empty_tuple.type = call imports.%C__carbon_thunk.decl(%addr.loc50)
// CHECK:STDOUT:   %.loc50.2: init %C to %.loc50.1 = mark_in_place_init %C__carbon_thunk.call.loc50
// CHECK:STDOUT:   %.loc39: type = splice_block %C.ref.loc39 [concrete = constants.%C] {
// CHECK:STDOUT:     %Cpp.ref.loc36: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %N.ref.loc37: <namespace> = name_ref N, imports.%N [concrete = imports.%N]
// CHECK:STDOUT:     %C.ref.loc39: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc50.3: ref %C = temporary %.loc50.1, %.loc50.2
// CHECK:STDOUT:   %.loc50.4: %C = acquire_value %.loc50.3
// CHECK:STDOUT:   %c2: %C = value_binding c2, %.loc50.4
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c3.patt: %pattern_type.69f = value_binding_pattern c3 [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c1.ref: %C = name_ref c1, %c1
// CHECK:STDOUT:   %c2.ref: %C = name_ref c2, %c2
// CHECK:STDOUT:   %.loc59: type = splice_block %C.ref.loc59 [concrete = constants.%C] {
// CHECK:STDOUT:     %Cpp.ref.loc56: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %N.ref.loc57: <namespace> = name_ref N, imports.%N [concrete = imports.%N]
// CHECK:STDOUT:     %C.ref.loc59: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c3: %C = value_binding c3, <error> [concrete = <error>]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %C.Op.bound.loc50: <bound method> = bound_method %.loc50.3, constants.%C.Op
// CHECK:STDOUT:   %Op.ref.loc50: %C.cpp_destructor.type = name_ref Op, imports.%C.cpp_destructor.decl [concrete = constants.%C.cpp_destructor]
// CHECK:STDOUT:   %C.cpp_destructor.bound.loc50: <bound method> = bound_method %.loc50.3, %Op.ref.loc50
// CHECK:STDOUT:   %C.cpp_destructor.call.loc50: init %empty_tuple.type = call %C.cpp_destructor.bound.loc50(%.loc50.3)
// CHECK:STDOUT:   %C.Op.bound.loc31: <bound method> = bound_method %.loc31.3, constants.%C.Op
// CHECK:STDOUT:   %Op.ref.loc31: %C.cpp_destructor.type = name_ref Op, imports.%C.cpp_destructor.decl [concrete = constants.%C.cpp_destructor]
// CHECK:STDOUT:   %C.cpp_destructor.bound.loc31: <bound method> = bound_method %.loc31.3, %Op.ref.loc31
// CHECK:STDOUT:   %C.cpp_destructor.call.loc31: init %empty_tuple.type = call %C.cpp_destructor.bound.loc31(%.loc31.3)
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
// CHECK:STDOUT:   %Cpp.ref.loc23: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %O.ref.loc24: type = name_ref O, imports.%O.decl [concrete = constants.%O]
// CHECK:STDOUT:   %C.ref.loc26: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %C.ref.loc28: %C.C.cpp_overload_set.type = name_ref C, imports.%C.C.cpp_overload_set.value [concrete = constants.%C.C.cpp_overload_set.value]
// CHECK:STDOUT:   %.loc31.1: ref %C = temporary_storage
// CHECK:STDOUT:   %addr.loc31: %ptr.de2 = addr_of %.loc31.1
// CHECK:STDOUT:   %C__carbon_thunk.call.loc31: init %empty_tuple.type = call imports.%C__carbon_thunk.decl(%addr.loc31)
// CHECK:STDOUT:   %.loc31.2: init %C to %.loc31.1 = mark_in_place_init %C__carbon_thunk.call.loc31
// CHECK:STDOUT:   %.loc20: type = splice_block %C.ref.loc20 [concrete = constants.%C] {
// CHECK:STDOUT:     %Cpp.ref.loc17: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %O.ref.loc18: type = name_ref O, imports.%O.decl [concrete = constants.%O]
// CHECK:STDOUT:     %C.ref.loc20: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc31.3: ref %C = temporary %.loc31.1, %.loc31.2
// CHECK:STDOUT:   %.loc31.4: %C = acquire_value %.loc31.3
// CHECK:STDOUT:   %c1: %C = value_binding c1, %.loc31.4
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c2.patt: %pattern_type.b28 = value_binding_pattern c2 [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc42: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %O.ref.loc43: type = name_ref O, imports.%O.decl [concrete = constants.%O]
// CHECK:STDOUT:   %C.ref.loc45: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %C.ref.loc47: %C.C.cpp_overload_set.type = name_ref C, imports.%C.C.cpp_overload_set.value [concrete = constants.%C.C.cpp_overload_set.value]
// CHECK:STDOUT:   %.loc50.1: ref %C = temporary_storage
// CHECK:STDOUT:   %addr.loc50: %ptr.de2 = addr_of %.loc50.1
// CHECK:STDOUT:   %C__carbon_thunk.call.loc50: init %empty_tuple.type = call imports.%C__carbon_thunk.decl(%addr.loc50)
// CHECK:STDOUT:   %.loc50.2: init %C to %.loc50.1 = mark_in_place_init %C__carbon_thunk.call.loc50
// CHECK:STDOUT:   %.loc39: type = splice_block %C.ref.loc39 [concrete = constants.%C] {
// CHECK:STDOUT:     %Cpp.ref.loc36: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %O.ref.loc37: type = name_ref O, imports.%O.decl [concrete = constants.%O]
// CHECK:STDOUT:     %C.ref.loc39: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc50.3: ref %C = temporary %.loc50.1, %.loc50.2
// CHECK:STDOUT:   %.loc50.4: %C = acquire_value %.loc50.3
// CHECK:STDOUT:   %c2: %C = value_binding c2, %.loc50.4
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c3.patt: %pattern_type.b28 = value_binding_pattern c3 [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c1.ref: %C = name_ref c1, %c1
// CHECK:STDOUT:   %c2.ref: %C = name_ref c2, %c2
// CHECK:STDOUT:   %.loc63.1: ref %C = temporary_storage
// CHECK:STDOUT:   %.loc62: ref %C = value_as_ref %c1.ref
// CHECK:STDOUT:   %addr.loc63.1: %ptr.de2 = addr_of %.loc62
// CHECK:STDOUT:   %.loc64: ref %C = value_as_ref %c2.ref
// CHECK:STDOUT:   %addr.loc63.2: %ptr.de2 = addr_of %.loc64
// CHECK:STDOUT:   %addr.loc63.3: %ptr.de2 = addr_of %.loc63.1
// CHECK:STDOUT:   %operator_Plus__carbon_thunk.call: init %empty_tuple.type = call imports.%operator_Plus__carbon_thunk.decl(%addr.loc63.1, %addr.loc63.2, %addr.loc63.3)
// CHECK:STDOUT:   %.loc63.2: init %C to %.loc63.1 = mark_in_place_init %operator_Plus__carbon_thunk.call
// CHECK:STDOUT:   %.loc59: type = splice_block %C.ref.loc59 [concrete = constants.%C] {
// CHECK:STDOUT:     %Cpp.ref.loc56: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %O.ref.loc57: type = name_ref O, imports.%O.decl [concrete = constants.%O]
// CHECK:STDOUT:     %C.ref.loc59: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc63.3: ref %C = temporary %.loc63.1, %.loc63.2
// CHECK:STDOUT:   %.loc63.4: %C = acquire_value %.loc63.3
// CHECK:STDOUT:   %c3: %C = value_binding c3, %.loc63.4
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %C.Op.bound.loc63: <bound method> = bound_method %.loc63.3, constants.%C.Op
// CHECK:STDOUT:   %Op.ref.loc63: %C.cpp_destructor.type = name_ref Op, imports.%C.cpp_destructor.decl [concrete = constants.%C.cpp_destructor]
// CHECK:STDOUT:   %C.cpp_destructor.bound.loc63: <bound method> = bound_method %.loc63.3, %Op.ref.loc63
// CHECK:STDOUT:   %C.cpp_destructor.call.loc63: init %empty_tuple.type = call %C.cpp_destructor.bound.loc63(%.loc63.3)
// CHECK:STDOUT:   %C.Op.bound.loc50: <bound method> = bound_method %.loc50.3, constants.%C.Op
// CHECK:STDOUT:   %Op.ref.loc50: %C.cpp_destructor.type = name_ref Op, imports.%C.cpp_destructor.decl [concrete = constants.%C.cpp_destructor]
// CHECK:STDOUT:   %C.cpp_destructor.bound.loc50: <bound method> = bound_method %.loc50.3, %Op.ref.loc50
// CHECK:STDOUT:   %C.cpp_destructor.call.loc50: init %empty_tuple.type = call %C.cpp_destructor.bound.loc50(%.loc50.3)
// CHECK:STDOUT:   %C.Op.bound.loc31: <bound method> = bound_method %.loc31.3, constants.%C.Op
// CHECK:STDOUT:   %Op.ref.loc31: %C.cpp_destructor.type = name_ref Op, imports.%C.cpp_destructor.decl [concrete = constants.%C.cpp_destructor]
// CHECK:STDOUT:   %C.cpp_destructor.bound.loc31: <bound method> = bound_method %.loc31.3, %Op.ref.loc31
// CHECK:STDOUT:   %C.cpp_destructor.call.loc31: init %empty_tuple.type = call %C.cpp_destructor.bound.loc31(%.loc31.3)
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
// CHECK:STDOUT:   %Cpp.ref.loc25: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %N.ref.loc26: <namespace> = name_ref N, imports.%N [concrete = imports.%N]
// CHECK:STDOUT:   %O.ref.loc28: type = name_ref O, imports.%O.decl [concrete = constants.%O]
// CHECK:STDOUT:   %C.ref.loc30: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %C.ref.loc32: %C.C.cpp_overload_set.type = name_ref C, imports.%C.C.cpp_overload_set.value [concrete = constants.%C.C.cpp_overload_set.value]
// CHECK:STDOUT:   %.loc35.1: ref %C = temporary_storage
// CHECK:STDOUT:   %addr.loc35: %ptr.4b2 = addr_of %.loc35.1
// CHECK:STDOUT:   %C__carbon_thunk.call.loc35: init %empty_tuple.type = call imports.%C__carbon_thunk.decl(%addr.loc35)
// CHECK:STDOUT:   %.loc35.2: init %C to %.loc35.1 = mark_in_place_init %C__carbon_thunk.call.loc35
// CHECK:STDOUT:   %.loc22: type = splice_block %C.ref.loc22 [concrete = constants.%C] {
// CHECK:STDOUT:     %Cpp.ref.loc17: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %N.ref.loc18: <namespace> = name_ref N, imports.%N [concrete = imports.%N]
// CHECK:STDOUT:     %O.ref.loc20: type = name_ref O, imports.%O.decl [concrete = constants.%O]
// CHECK:STDOUT:     %C.ref.loc22: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc35.3: ref %C = temporary %.loc35.1, %.loc35.2
// CHECK:STDOUT:   %.loc35.4: %C = acquire_value %.loc35.3
// CHECK:STDOUT:   %c1: %C = value_binding c1, %.loc35.4
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c2.patt: %pattern_type.84b = value_binding_pattern c2 [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc48: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %N.ref.loc49: <namespace> = name_ref N, imports.%N [concrete = imports.%N]
// CHECK:STDOUT:   %O.ref.loc51: type = name_ref O, imports.%O.decl [concrete = constants.%O]
// CHECK:STDOUT:   %C.ref.loc53: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %C.ref.loc55: %C.C.cpp_overload_set.type = name_ref C, imports.%C.C.cpp_overload_set.value [concrete = constants.%C.C.cpp_overload_set.value]
// CHECK:STDOUT:   %.loc58.1: ref %C = temporary_storage
// CHECK:STDOUT:   %addr.loc58: %ptr.4b2 = addr_of %.loc58.1
// CHECK:STDOUT:   %C__carbon_thunk.call.loc58: init %empty_tuple.type = call imports.%C__carbon_thunk.decl(%addr.loc58)
// CHECK:STDOUT:   %.loc58.2: init %C to %.loc58.1 = mark_in_place_init %C__carbon_thunk.call.loc58
// CHECK:STDOUT:   %.loc45: type = splice_block %C.ref.loc45 [concrete = constants.%C] {
// CHECK:STDOUT:     %Cpp.ref.loc40: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %N.ref.loc41: <namespace> = name_ref N, imports.%N [concrete = imports.%N]
// CHECK:STDOUT:     %O.ref.loc43: type = name_ref O, imports.%O.decl [concrete = constants.%O]
// CHECK:STDOUT:     %C.ref.loc45: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc58.3: ref %C = temporary %.loc58.1, %.loc58.2
// CHECK:STDOUT:   %.loc58.4: %C = acquire_value %.loc58.3
// CHECK:STDOUT:   %c2: %C = value_binding c2, %.loc58.4
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c3.patt: %pattern_type.84b = value_binding_pattern c3 [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c1.ref: %C = name_ref c1, %c1
// CHECK:STDOUT:   %c2.ref: %C = name_ref c2, %c2
// CHECK:STDOUT:   %.loc73.1: ref %C = temporary_storage
// CHECK:STDOUT:   %.loc72: ref %C = value_as_ref %c1.ref
// CHECK:STDOUT:   %addr.loc73.1: %ptr.4b2 = addr_of %.loc72
// CHECK:STDOUT:   %.loc74: ref %C = value_as_ref %c2.ref
// CHECK:STDOUT:   %addr.loc73.2: %ptr.4b2 = addr_of %.loc74
// CHECK:STDOUT:   %addr.loc73.3: %ptr.4b2 = addr_of %.loc73.1
// CHECK:STDOUT:   %operator_Plus__carbon_thunk.call: init %empty_tuple.type = call imports.%operator_Plus__carbon_thunk.decl(%addr.loc73.1, %addr.loc73.2, %addr.loc73.3)
// CHECK:STDOUT:   %.loc73.2: init %C to %.loc73.1 = mark_in_place_init %operator_Plus__carbon_thunk.call
// CHECK:STDOUT:   %.loc69: type = splice_block %C.ref.loc69 [concrete = constants.%C] {
// CHECK:STDOUT:     %Cpp.ref.loc64: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %N.ref.loc65: <namespace> = name_ref N, imports.%N [concrete = imports.%N]
// CHECK:STDOUT:     %O.ref.loc67: type = name_ref O, imports.%O.decl [concrete = constants.%O]
// CHECK:STDOUT:     %C.ref.loc69: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc73.3: ref %C = temporary %.loc73.1, %.loc73.2
// CHECK:STDOUT:   %.loc73.4: %C = acquire_value %.loc73.3
// CHECK:STDOUT:   %c3: %C = value_binding c3, %.loc73.4
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %C.Op.bound.loc73: <bound method> = bound_method %.loc73.3, constants.%C.Op
// CHECK:STDOUT:   %Op.ref.loc73: %C.cpp_destructor.type = name_ref Op, imports.%C.cpp_destructor.decl [concrete = constants.%C.cpp_destructor]
// CHECK:STDOUT:   %C.cpp_destructor.bound.loc73: <bound method> = bound_method %.loc73.3, %Op.ref.loc73
// CHECK:STDOUT:   %C.cpp_destructor.call.loc73: init %empty_tuple.type = call %C.cpp_destructor.bound.loc73(%.loc73.3)
// CHECK:STDOUT:   %C.Op.bound.loc58: <bound method> = bound_method %.loc58.3, constants.%C.Op
// CHECK:STDOUT:   %Op.ref.loc58: %C.cpp_destructor.type = name_ref Op, imports.%C.cpp_destructor.decl [concrete = constants.%C.cpp_destructor]
// CHECK:STDOUT:   %C.cpp_destructor.bound.loc58: <bound method> = bound_method %.loc58.3, %Op.ref.loc58
// CHECK:STDOUT:   %C.cpp_destructor.call.loc58: init %empty_tuple.type = call %C.cpp_destructor.bound.loc58(%.loc58.3)
// CHECK:STDOUT:   %C.Op.bound.loc35: <bound method> = bound_method %.loc35.3, constants.%C.Op
// CHECK:STDOUT:   %Op.ref.loc35: %C.cpp_destructor.type = name_ref Op, imports.%C.cpp_destructor.decl [concrete = constants.%C.cpp_destructor]
// CHECK:STDOUT:   %C.cpp_destructor.bound.loc35: <bound method> = bound_method %.loc35.3, %Op.ref.loc35
// CHECK:STDOUT:   %C.cpp_destructor.call.loc35: init %empty_tuple.type = call %C.cpp_destructor.bound.loc35(%.loc35.3)
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
// CHECK:STDOUT:   %Cpp.ref.loc21: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %C.ref.loc22: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %C.ref.loc24: %C.C.cpp_overload_set.type = name_ref C, imports.%C.C.cpp_overload_set.value [concrete = constants.%C.C.cpp_overload_set.value]
// CHECK:STDOUT:   %.loc14: ref %C = splice_block %c1.var {}
// CHECK:STDOUT:   %addr.loc27: %ptr.d9e = addr_of %.loc14
// CHECK:STDOUT:   %C__carbon_thunk.call: init %empty_tuple.type = call imports.%C__carbon_thunk.decl(%addr.loc27)
// CHECK:STDOUT:   %.loc27: init %C to %.loc14 = mark_in_place_init %C__carbon_thunk.call
// CHECK:STDOUT:   assign %c1.var, %.loc27
// CHECK:STDOUT:   %.loc18: type = splice_block %C.ref.loc18 [concrete = constants.%C] {
// CHECK:STDOUT:     %Cpp.ref.loc17: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %C.ref.loc18: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c1: ref %C = ref_binding c1, %c1.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c2.patt: %pattern_type.217 = ref_binding_pattern c2 [concrete]
// CHECK:STDOUT:     %c2.var_patt: %pattern_type.217 = var_pattern %c2.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c2.var: ref %C = var %c2.var_patt
// CHECK:STDOUT:   %c1.ref.loc37: ref %C = name_ref c1, %c1
// CHECK:STDOUT:   %C.cpp_operator.bound.loc36: <bound method> = bound_method %c1.ref.loc37, imports.%C.cpp_operator.decl.828f43.1
// CHECK:STDOUT:   %.loc29: ref %C = splice_block %c2.var {}
// CHECK:STDOUT:   %addr.loc36: %ptr.d9e = addr_of %.loc29
// CHECK:STDOUT:   %operator_Minus__carbon_thunk.call: init %empty_tuple.type = call imports.%operator_Minus__carbon_thunk.decl(%c1.ref.loc37, %addr.loc36)
// CHECK:STDOUT:   %.loc36: init %C to %.loc29 = mark_in_place_init %operator_Minus__carbon_thunk.call
// CHECK:STDOUT:   assign %c2.var, %.loc36
// CHECK:STDOUT:   %.loc33: type = splice_block %C.ref.loc33 [concrete = constants.%C] {
// CHECK:STDOUT:     %Cpp.ref.loc32: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %C.ref.loc33: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c2: ref %C = ref_binding c2, %c2.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c3.patt: %pattern_type.217 = ref_binding_pattern c3 [concrete]
// CHECK:STDOUT:     %c3.var_patt: %pattern_type.217 = var_pattern %c3.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c3.var: ref %C = var %c3.var_patt
// CHECK:STDOUT:   %c1.ref.loc47: ref %C = name_ref c1, %c1
// CHECK:STDOUT:   %c2.ref: ref %C = name_ref c2, %c2
// CHECK:STDOUT:   %C.cpp_operator.bound.loc48: <bound method> = bound_method %c1.ref.loc47, imports.%C.cpp_operator.decl.828f43.2
// CHECK:STDOUT:   %.loc39: ref %C = splice_block %c3.var {}
// CHECK:STDOUT:   %.loc49.1: %C = acquire_value %c2.ref
// CHECK:STDOUT:   %.loc49.2: ref %C = value_as_ref %.loc49.1
// CHECK:STDOUT:   %addr.loc48.1: %ptr.d9e = addr_of %.loc49.2
// CHECK:STDOUT:   %addr.loc48.2: %ptr.d9e = addr_of %.loc39
// CHECK:STDOUT:   %operator_Plus__carbon_thunk.call: init %empty_tuple.type = call imports.%operator_Plus__carbon_thunk.decl(%c1.ref.loc47, %addr.loc48.1, %addr.loc48.2)
// CHECK:STDOUT:   %.loc48: init %C to %.loc39 = mark_in_place_init %operator_Plus__carbon_thunk.call
// CHECK:STDOUT:   assign %c3.var, %.loc48
// CHECK:STDOUT:   %.loc44: type = splice_block %C.ref.loc44 [concrete = constants.%C] {
// CHECK:STDOUT:     %Cpp.ref.loc43: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %C.ref.loc44: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c3: ref %C = ref_binding c3, %c3.var
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %C.Op.bound.loc39: <bound method> = bound_method %c3.var, constants.%C.Op
// CHECK:STDOUT:   %Op.ref.loc39: %C.cpp_destructor.type = name_ref Op, imports.%C.cpp_destructor.decl [concrete = constants.%C.cpp_destructor]
// CHECK:STDOUT:   %C.cpp_destructor.bound.loc39: <bound method> = bound_method %c3.var, %Op.ref.loc39
// CHECK:STDOUT:   %C.cpp_destructor.call.loc39: init %empty_tuple.type = call %C.cpp_destructor.bound.loc39(%c3.var)
// CHECK:STDOUT:   %C.Op.bound.loc29: <bound method> = bound_method %c2.var, constants.%C.Op
// CHECK:STDOUT:   %Op.ref.loc29: %C.cpp_destructor.type = name_ref Op, imports.%C.cpp_destructor.decl [concrete = constants.%C.cpp_destructor]
// CHECK:STDOUT:   %C.cpp_destructor.bound.loc29: <bound method> = bound_method %c2.var, %Op.ref.loc29
// CHECK:STDOUT:   %C.cpp_destructor.call.loc29: init %empty_tuple.type = call %C.cpp_destructor.bound.loc29(%c2.var)
// CHECK:STDOUT:   %C.Op.bound.loc14: <bound method> = bound_method %c1.var, constants.%C.Op
// CHECK:STDOUT:   %Op.ref.loc14: %C.cpp_destructor.type = name_ref Op, imports.%C.cpp_destructor.decl [concrete = constants.%C.cpp_destructor]
// CHECK:STDOUT:   %C.cpp_destructor.bound.loc14: <bound method> = bound_method %c1.var, %Op.ref.loc14
// CHECK:STDOUT:   %C.cpp_destructor.call.loc14: init %empty_tuple.type = call %C.cpp_destructor.bound.loc14(%c1.var)
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
// CHECK:STDOUT:   %x.ref.loc22: %B = name_ref x, %x
// CHECK:STDOUT:   %x.ref.loc24: %B = name_ref x, %x
// CHECK:STDOUT:   %.loc22: ref %B = value_as_ref %x.ref.loc22
// CHECK:STDOUT:   %addr.loc23.1: %ptr.a04 = addr_of %.loc22
// CHECK:STDOUT:   %.loc24: ref %B = value_as_ref %x.ref.loc24
// CHECK:STDOUT:   %addr.loc23.2: %ptr.a04 = addr_of %.loc24
// CHECK:STDOUT:   %operator_Plus__carbon_thunk.call: init %i32 = call imports.%operator_Plus__carbon_thunk.decl(%addr.loc23.1, %addr.loc23.2)
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
// CHECK:STDOUT:   %.loc53.1: ref %C = temporary_storage
// CHECK:STDOUT:   %.loc52: ref %C = value_as_ref %c1.ref
// CHECK:STDOUT:   %addr.loc53.1: %ptr.d9e = addr_of %.loc52
// CHECK:STDOUT:   %.loc54: ref %C = value_as_ref %c2.ref
// CHECK:STDOUT:   %addr.loc53.2: %ptr.d9e = addr_of %.loc54
// CHECK:STDOUT:   %addr.loc53.3: %ptr.d9e = addr_of %.loc53.1
// CHECK:STDOUT:   %operator_Plus__carbon_thunk.call: init %empty_tuple.type = call imports.%operator_Plus__carbon_thunk.decl(%addr.loc53.1, %addr.loc53.2, %addr.loc53.3)
// CHECK:STDOUT:   %.loc53.2: init %C to %.loc53.1 = mark_in_place_init %operator_Plus__carbon_thunk.call
// CHECK:STDOUT:   %.loc49: type = splice_block %C.ref.loc49 [concrete = constants.%C] {
// CHECK:STDOUT:     %Cpp.ref.loc48: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %C.ref.loc49: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc53.3: ref %C = temporary %.loc53.1, %.loc53.2
// CHECK:STDOUT:   %.loc53.4: %C = acquire_value %.loc53.3
// CHECK:STDOUT:   %c3: %C = value_binding c3, %.loc53.4
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %C.Op.bound.loc53: <bound method> = bound_method %.loc53.3, constants.%C.Op
// CHECK:STDOUT:   %Op.ref.loc53: %C.cpp_destructor.type = name_ref Op, imports.%C.cpp_destructor.decl [concrete = constants.%C.cpp_destructor]
// CHECK:STDOUT:   %C.cpp_destructor.bound.loc53: <bound method> = bound_method %.loc53.3, %Op.ref.loc53
// CHECK:STDOUT:   %C.cpp_destructor.call.loc53: init %empty_tuple.type = call %C.cpp_destructor.bound.loc53(%.loc53.3)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
