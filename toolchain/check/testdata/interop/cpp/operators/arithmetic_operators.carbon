// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/full.carbon
// EXTRA-ARGS: --target=x86_64-linux-gnu --clang-arg=-std=c++20
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interop/cpp/operators/arithmetic_operators.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interop/cpp/operators/arithmetic_operators.carbon

// TODO: test implicit conversions when supported

// --- test_interfaces.carbon

library "[[@TEST_NAME]]";

import Cpp inline '''c++
class Int16;
class Int32;
class Int64;
''';

fn Int16() -> Cpp.Int16;
fn Int32() -> Cpp.Int32;
fn Int64() -> Cpp.Int64;

fn TestAddWith[
  U:! type,
  T:! Core.AddWith(U) where .Result impls Core.Destroy
](x: T, y: U) {
  //@dump-sem-ir-begin
  x + y;
  //@dump-sem-ir-end
}

fn TestAddAssignWith[U:! type, T:! Core.AddAssignWith(U)](var x: T, y: U) {
  //@dump-sem-ir-begin
  x += y;
  //@dump-sem-ir-end
}

fn TestSubWith[
  U:! type,
  T:! Core.SubWith(U) where .Result impls Core.Destroy
](x: T, y: U) {
  //@dump-sem-ir-begin
  x - y;
  //@dump-sem-ir-end
}

fn TestSubAssignWith[U:! type, T:! Core.SubAssignWith(U)](var x: T, y: U) {
  //@dump-sem-ir-begin
  x -= y;
  //@dump-sem-ir-end
}

fn TestMulWith[
  U:! type,
  T:! Core.MulWith(U) where .Result impls Core.Destroy
](x: T, y: U) {
  //@dump-sem-ir-begin
  x * y;
  //@dump-sem-ir-end
}

fn TestMulAssignWith[U:! type, T:! Core.MulAssignWith(U)](var x: T, y: U) {
  //@dump-sem-ir-begin
  x *= y;
  //@dump-sem-ir-end
}

fn TestDivWith[
  U:! type,
  T:! Core.DivWith(U) where .Result impls Core.Destroy
](x: T, y: U) {
  //@dump-sem-ir-begin
  x / y;
  //@dump-sem-ir-end
}

fn TestDivAssignWith[U:! type, T:! Core.DivAssignWith(U)](var x: T, y: U) {
  //@dump-sem-ir-begin
  x /= y;
  //@dump-sem-ir-end
}

fn TestModWith[
  U:! type,
  T:! Core.ModWith(U) where .Result impls Core.Destroy
](x: T, y: U) {
  //@dump-sem-ir-begin
  x % y;
  //@dump-sem-ir-end
}

fn TestModAssignWith[U:! type, T:! Core.ModAssignWith(U)](var x: T, y: U) {
  //@dump-sem-ir-begin
  x %= y;
  //@dump-sem-ir-end
}

fn TestNegate[T:! Core.Negate where .Result impls Core.Destroy](x: T) {
  //@dump-sem-ir-begin
  -x;
  //@dump-sem-ir-end
}

fn TestInc[T:! Core.Inc](var x: T) {
  //@dump-sem-ir-begin
  ++x;
  //@dump-sem-ir-end
}

fn TestDec[T:! Core.Dec](var x: T) {
  //@dump-sem-ir-begin
  --x;
  //@dump-sem-ir-end
}

// --- add_with.h

class Int32;

class Int16 {
public:
  auto operator+(Int32) const -> Int32;
  auto operator+(short) const -> Int16;

  auto operator+=(short) -> Int16&;
  auto operator+=(Int32) -> Int16&;
};

class Int32 {
public:
  auto operator+(Int16) const -> Int32;
  auto operator+(Int32) const -> Int32;

  auto operator+=(Int16) -> Int32&;
  auto operator+=(Int32) -> void;
};

class Int64 {
public:
  auto operator+=(Int32) -> void;

  friend auto operator+(Int32, Int64) -> Int64;
};

auto operator+(Int64, Int32) -> Int64;

// --- add_with.carbon

library "[[@TEST_NAME]]";

import library "test_interfaces";
import Cpp library "add_with.h";

fn TestDriver() {
  TestAddWith(Int16(), 0 as i16);
  TestAddAssignWith(Int16(), 0 as i16);

  TestAddWith(Int16(), Int32());
  TestAddAssignWith(Int16(), Int32());

  TestAddWith(Int32(), Int16());
  TestAddAssignWith(Int32(), Int16());

  TestAddWith(Int32(), Int32());
  TestAddAssignWith(Int32(), Int32());

  TestAddWith(Int32(), Int64());

  TestAddWith(Int64(), Int32());
  TestAddAssignWith(Int64(), Int32());
}

// --- fail_add_with.carbon

library "fail_[[@TEST_NAME]]";

import library "test_interfaces";
import Cpp library "add_with.h";

fn FailTestDriver() {
  // CHECK:STDERR: fail_add_with.carbon:[[@LINE+8]]:3: error: cannot convert type `i16` into type implementing `Core.AddWith(Cpp.Int16) where .(Core.AddWith(Cpp.Int16).Result) impls Core.Destroy` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   TestAddWith(0 as i16, Int16());
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_add_with.carbon:[[@LINE-7]]:1: in import [InImport]
  // CHECK:STDERR: test_interfaces.carbon:14:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn TestAddWith[
  // CHECK:STDERR: ^~~~~~~~~~~~~~~
  // CHECK:STDERR:
  TestAddWith(0 as i16, Int16());

  // CHECK:STDERR: fail_add_with.carbon:[[@LINE+8]]:3: error: cannot convert type `i16` into type implementing `Core.AddAssignWith(Cpp.Int16)` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   TestAddAssignWith(0 as i16, Int16());
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_add_with.carbon:[[@LINE-17]]:1: in import [InImport]
  // CHECK:STDERR: test_interfaces.carbon:23:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn TestAddAssignWith[U:! type, T:! Core.AddAssignWith(U)](var x: T, y: U) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  TestAddAssignWith(0 as i16, Int16());

  // CHECK:STDERR: fail_add_with.carbon:[[@LINE+8]]:3: error: cannot convert type `Cpp.Int32` into type implementing `Core.AddAssignWith(Cpp.Int64)` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   TestAddAssignWith(Int32(), Int64());
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_add_with.carbon:[[@LINE-27]]:1: in import [InImport]
  // CHECK:STDERR: test_interfaces.carbon:23:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn TestAddAssignWith[U:! type, T:! Core.AddAssignWith(U)](var x: T, y: U) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  TestAddAssignWith(Int32(), Int64());
}

// --- sub_with.h

class Int32;

class Int16 {
public:
  auto operator-(Int32) const -> Int32;
  auto operator-(short) const -> Int16;

  auto operator-=(short) -> Int16&;
  auto operator-=(Int32) -> Int16&;
};

class Int32 {
public:
  auto operator-(Int16) const -> Int32;
  auto operator-(Int32) const -> Int32;

  auto operator-=(Int16) -> Int32&;
  auto operator-=(Int32) -> void;
};

class Int64 {
public:
  auto operator-=(Int32) -> void;

  friend auto operator-(Int32, Int64) -> Int64;
};

auto operator-(Int64, Int32) -> Int64;

// --- sub_with.carbon

library "[[@TEST_NAME]]";

import library "test_interfaces";
import Cpp library "sub_with.h";

fn TestDriver() {
  TestSubWith(Int16(), 0 as i16);
  TestSubAssignWith(Int16(), 0 as i16);

  TestSubWith(Int16(), Int32());
  TestSubAssignWith(Int16(), Int32());

  TestSubWith(Int32(), Int16());
  TestSubAssignWith(Int32(), Int16());

  TestSubWith(Int32(), Int32());
  TestSubAssignWith(Int32(), Int32());

  TestSubWith(Int32(), Int64());

  TestSubWith(Int64(), Int32());
  TestSubAssignWith(Int64(), Int32());
}

// --- fail_sub_with.carbon

library "fail_[[@TEST_NAME]]";

import library "test_interfaces";
import Cpp library "sub_with.h";

fn FailTestDriver() {
  // CHECK:STDERR: fail_sub_with.carbon:[[@LINE+8]]:3: error: cannot convert type `i16` into type implementing `Core.SubWith(Cpp.Int16) where .(Core.SubWith(Cpp.Int16).Result) impls Core.Destroy` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   TestSubWith(0 as i16, Int16());
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_sub_with.carbon:[[@LINE-7]]:1: in import [InImport]
  // CHECK:STDERR: test_interfaces.carbon:29:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn TestSubWith[
  // CHECK:STDERR: ^~~~~~~~~~~~~~~
  // CHECK:STDERR:
  TestSubWith(0 as i16, Int16());

  // CHECK:STDERR: fail_sub_with.carbon:[[@LINE+8]]:3: error: cannot convert type `i16` into type implementing `Core.SubAssignWith(Cpp.Int16)` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   TestSubAssignWith(0 as i16, Int16());
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_sub_with.carbon:[[@LINE-17]]:1: in import [InImport]
  // CHECK:STDERR: test_interfaces.carbon:38:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn TestSubAssignWith[U:! type, T:! Core.SubAssignWith(U)](var x: T, y: U) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  TestSubAssignWith(0 as i16, Int16());

  // CHECK:STDERR: fail_sub_with.carbon:[[@LINE+8]]:3: error: cannot convert type `Cpp.Int32` into type implementing `Core.SubAssignWith(Cpp.Int64)` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   TestSubAssignWith(Int32(), Int64());
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_sub_with.carbon:[[@LINE-27]]:1: in import [InImport]
  // CHECK:STDERR: test_interfaces.carbon:38:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn TestSubAssignWith[U:! type, T:! Core.SubAssignWith(U)](var x: T, y: U) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  TestSubAssignWith(Int32(), Int64());
}

// --- mul_with.h

class Int32;

class Int16 {
public:
  auto operator*(Int32) const -> Int32;
  auto operator*(short) const -> Int16;

  auto operator*=(short) -> Int16&;
  auto operator*=(Int32) -> Int16&;
};

class Int32 {
public:
  auto operator*(Int16) const -> Int32;
  auto operator*(Int32) const -> Int32;

  auto operator*=(Int16) -> Int32&;
  auto operator*=(Int32) -> void;
};

class Int64 {
public:
  auto operator*=(Int32) -> void;

  friend auto operator*(Int32, Int64) -> Int64;
};

auto operator*(Int64, Int32) -> Int64;

// --- mul_with.carbon

library "[[@TEST_NAME]]";

import library "test_interfaces";
import Cpp library "mul_with.h";

fn TestDriver() {
  TestMulWith(Int16(), 0 as i16);
  TestMulAssignWith(Int16(), 0 as i16);

  TestMulWith(Int16(), Int32());
  TestMulAssignWith(Int16(), Int32());

  TestMulWith(Int32(), Int16());

  TestMulWith(Int32(), Int32());
  TestMulAssignWith(Int32(), Int32());

  TestMulWith(Int32(), Int64());

  TestMulWith(Int64(), Int32());
  TestMulAssignWith(Int64(), Int32());
}

// --- fail_mul_with.carbon

library "fail_[[@TEST_NAME]]";

import library "test_interfaces";
import Cpp library "mul_with.h";

fn FailTestDriver() {
  // CHECK:STDERR: fail_mul_with.carbon:[[@LINE+8]]:3: error: cannot convert type `i16` into type implementing `Core.MulWith(Cpp.Int16) where .(Core.MulWith(Cpp.Int16).Result) impls Core.Destroy` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   TestMulWith(0 as i16, Int16());
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_mul_with.carbon:[[@LINE-7]]:1: in import [InImport]
  // CHECK:STDERR: test_interfaces.carbon:44:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn TestMulWith[
  // CHECK:STDERR: ^~~~~~~~~~~~~~~
  // CHECK:STDERR:
  TestMulWith(0 as i16, Int16());

  // CHECK:STDERR: fail_mul_with.carbon:[[@LINE+8]]:3: error: cannot convert type `i16` into type implementing `Core.MulAssignWith(Cpp.Int16)` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   TestMulAssignWith(0 as i16, Int16());
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_mul_with.carbon:[[@LINE-17]]:1: in import [InImport]
  // CHECK:STDERR: test_interfaces.carbon:53:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn TestMulAssignWith[U:! type, T:! Core.MulAssignWith(U)](var x: T, y: U) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  TestMulAssignWith(0 as i16, Int16());

  // CHECK:STDERR: fail_mul_with.carbon:[[@LINE+8]]:3: error: cannot convert type `Cpp.Int32` into type implementing `Core.MulAssignWith(Cpp.Int64)` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   TestMulAssignWith(Int32(), Int64());
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_mul_with.carbon:[[@LINE-27]]:1: in import [InImport]
  // CHECK:STDERR: test_interfaces.carbon:53:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn TestMulAssignWith[U:! type, T:! Core.MulAssignWith(U)](var x: T, y: U) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  TestMulAssignWith(Int32(), Int64());
}

// --- div_with.h

class Int32;

class Int16 {
public:
  auto operator/(Int32) const -> Int32;
  auto operator/(short) const -> Int16;

  auto operator/=(short) -> Int16&;
  auto operator/=(Int32) -> Int16&;
};

class Int32 {
public:
  auto operator/(Int16) const -> Int32;
  auto operator/(Int32) const -> Int32;

  auto operator/=(Int16) -> Int32&;
  auto operator/=(Int32) -> void;
};

class Int64 {
public:
  auto operator/=(Int32) -> void;

  friend auto operator/(Int32, Int64) -> Int64;
};

auto operator/(Int64, Int32) -> Int64;

// --- div_with.carbon

library "[[@TEST_NAME]]";

import library "test_interfaces";
import Cpp library "div_with.h";

fn TestDriver() {
  TestDivWith(Int16(), 0 as i16);
  TestDivAssignWith(Int16(), 0 as i16);

  TestDivWith(Int16(), Int32());
  TestDivAssignWith(Int16(), Int32());

  TestDivWith(Int32(), Int16());

  TestDivWith(Int32(), Int32());
  TestDivAssignWith(Int32(), Int32());

  TestDivWith(Int32(), Int64());

  TestDivWith(Int64(), Int32());
  TestDivAssignWith(Int64(), Int32());
}

// --- fail_div_with.carbon

library "fail_[[@TEST_NAME]]";

import library "test_interfaces";
import Cpp library "div_with.h";

fn FailTestDriver() {
  // CHECK:STDERR: fail_div_with.carbon:[[@LINE+8]]:3: error: cannot convert type `i16` into type implementing `Core.DivWith(Cpp.Int16) where .(Core.DivWith(Cpp.Int16).Result) impls Core.Destroy` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   TestDivWith(0 as i16, Int16());
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_div_with.carbon:[[@LINE-7]]:1: in import [InImport]
  // CHECK:STDERR: test_interfaces.carbon:59:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn TestDivWith[
  // CHECK:STDERR: ^~~~~~~~~~~~~~~
  // CHECK:STDERR:
  TestDivWith(0 as i16, Int16());

  // CHECK:STDERR: fail_div_with.carbon:[[@LINE+8]]:3: error: cannot convert type `i16` into type implementing `Core.DivAssignWith(Cpp.Int16)` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   TestDivAssignWith(0 as i16, Int16());
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_div_with.carbon:[[@LINE-17]]:1: in import [InImport]
  // CHECK:STDERR: test_interfaces.carbon:68:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn TestDivAssignWith[U:! type, T:! Core.DivAssignWith(U)](var x: T, y: U) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  TestDivAssignWith(0 as i16, Int16());

  // CHECK:STDERR: fail_div_with.carbon:[[@LINE+8]]:3: error: cannot convert type `Cpp.Int32` into type implementing `Core.DivAssignWith(Cpp.Int64)` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   TestDivAssignWith(Int32(), Int64());
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_div_with.carbon:[[@LINE-27]]:1: in import [InImport]
  // CHECK:STDERR: test_interfaces.carbon:68:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn TestDivAssignWith[U:! type, T:! Core.DivAssignWith(U)](var x: T, y: U) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  TestDivAssignWith(Int32(), Int64());
}

// --- mod_with.h

class Int32;

class Int16 {
public:
  auto operator%(Int32) const -> Int32;
  auto operator%(short) const -> Int16;

  auto operator%=(short) -> Int16&;
  auto operator%=(Int32) -> Int16&;
};

class Int32 {
public:
  auto operator%(Int16) const -> Int32;
  auto operator%(Int32) const -> Int32;

  auto operator%=(Int16) -> Int32&;
  auto operator%=(Int32) -> void;
};

class Int64 {
public:
  auto operator%=(Int32) -> void;

  friend auto operator%(Int32, Int64) -> Int64;
};

auto operator%(Int64, Int32) -> Int64;

// --- mod_with.carbon

library "[[@TEST_NAME]]";

import library "test_interfaces";
import Cpp library "mod_with.h";

fn TestDriver() {
  TestModWith(Int16(), 0 as i16);
  TestModAssignWith(Int16(), 0 as i16);

  TestModWith(Int16(), Int32());
  TestModAssignWith(Int16(), Int32());

  TestModWith(Int32(), Int16());

  TestModWith(Int32(), Int32());
  TestModAssignWith(Int32(), Int32());

  TestModWith(Int32(), Int64());

  TestModWith(Int64(), Int32());
  TestModAssignWith(Int64(), Int32());
}

// --- fail_mod_with.carbon

library "fail_[[@TEST_NAME]]";

import library "test_interfaces";
import Cpp library "mod_with.h";

fn FailTestDriver() {
  // CHECK:STDERR: fail_mod_with.carbon:[[@LINE+8]]:3: error: cannot convert type `i16` into type implementing `Core.ModWith(Cpp.Int16) where .(Core.ModWith(Cpp.Int16).Result) impls Core.Destroy` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   TestModWith(0 as i16, Int16());
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_mod_with.carbon:[[@LINE-7]]:1: in import [InImport]
  // CHECK:STDERR: test_interfaces.carbon:74:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn TestModWith[
  // CHECK:STDERR: ^~~~~~~~~~~~~~~
  // CHECK:STDERR:
  TestModWith(0 as i16, Int16());

  // CHECK:STDERR: fail_mod_with.carbon:[[@LINE+8]]:3: error: cannot convert type `i16` into type implementing `Core.ModAssignWith(Cpp.Int16)` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   TestModAssignWith(0 as i16, Int16());
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_mod_with.carbon:[[@LINE-17]]:1: in import [InImport]
  // CHECK:STDERR: test_interfaces.carbon:83:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn TestModAssignWith[U:! type, T:! Core.ModAssignWith(U)](var x: T, y: U) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  TestModAssignWith(0 as i16, Int16());

  // CHECK:STDERR: fail_mod_with.carbon:[[@LINE+8]]:3: error: cannot convert type `Cpp.Int32` into type implementing `Core.ModAssignWith(Cpp.Int64)` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   TestModAssignWith(Int32(), Int64());
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_mod_with.carbon:[[@LINE-27]]:1: in import [InImport]
  // CHECK:STDERR: test_interfaces.carbon:83:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn TestModAssignWith[U:! type, T:! Core.ModAssignWith(U)](var x: T, y: U) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  TestModAssignWith(Int32(), Int64());
}

// --- negate.carbon

library "[[@TEST_NAME]]";

import library "test_interfaces";
import Cpp inline '''c++
class Int32;

class Int16 {
public:
  // Integer "promotion"
  auto operator-() -> Int32;
};

class Int32 {
public:
  auto operator-() const -> Int32;
};

class Int64 {};
auto operator-(Int64) -> Int64;
''';

fn TestDriver() {
  TestNegate(Int16());
  TestNegate(Int32());
  TestNegate(Int64());
}

// --- increment.carbon

library "[[@TEST_NAME]]";

import library "test_interfaces";
import Cpp inline '''c++
struct Int16 {
  // Integer "promotion", even though pre-inc usually returns a reference.
  auto operator++() -> Int16;
};

struct Int32 {
  auto operator++() const -> Int32 const&;
};
''';

fn TestDriver() {
  TestInc(Int16());
  TestInc(Int32());
}

// --- decrement.carbon

library "[[@TEST_NAME]]";

import library "test_interfaces";
import Cpp inline '''c++
struct Int16 {
  // Integer "promotion", even though pre-dec usually returns a reference.
  auto operator--() -> Int16;
};

struct Int32 {
  auto operator--() const -> Int32 const&;
};
''';

fn TestDriver() {
  TestDec(Int16());
  TestDec(Int32());
}

// CHECK:STDOUT: --- test_interfaces.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %U.patt: %pattern_type.98f = symbolic_binding_pattern U, 0 [symbolic]
// CHECK:STDOUT:   %U: type = symbolic_binding U, 0 [symbolic]
// CHECK:STDOUT:   %Other: type = symbolic_binding Other, 0 [symbolic]
// CHECK:STDOUT:   %AddWith.type.7e967c.1: type = facet_type <@AddWith, @AddWith(%Other)> [symbolic]
// CHECK:STDOUT:   %Self.96aae2.1: %AddWith.type.7e967c.1 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %AddWith.assoc_type.588227.1: type = assoc_entity_type @AddWith, @AddWith(%Other) [symbolic]
// CHECK:STDOUT:   %AddWith.WithSelf.Op.type.28e0e0.1: type = fn_type @AddWith.WithSelf.Op, @AddWith.WithSelf(%Other, %Self.96aae2.1) [symbolic]
// CHECK:STDOUT:   %AddWith.WithSelf.Op.eab9a0.1: %AddWith.WithSelf.Op.type.28e0e0.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %AddWith.type.7e967c.2: type = facet_type <@AddWith, @AddWith(%U)> [symbolic]
// CHECK:STDOUT:   %.Self.f3c: %AddWith.type.7e967c.2 = symbolic_binding .Self [symbolic]
// CHECK:STDOUT:   %AddWith.assoc_type.588227.2: type = assoc_entity_type @AddWith, @AddWith(%U) [symbolic]
// CHECK:STDOUT:   %assoc0.f393f4.2: %AddWith.assoc_type.588227.2 = assoc_entity element0, imports.%Core.import_ref.91f [symbolic]
// CHECK:STDOUT:   %assoc1.c411ca.2: %AddWith.assoc_type.588227.2 = assoc_entity element1, imports.%Core.import_ref.3cf [symbolic]
// CHECK:STDOUT:   %require_complete.1a1: <witness> = require_complete_type %AddWith.type.7e967c.2 [symbolic]
// CHECK:STDOUT:   %.Self.as_type.806: type = facet_access_type %.Self.f3c [symbolic]
// CHECK:STDOUT:   %AddWith.lookup_impl_witness.a82: <witness> = lookup_impl_witness %.Self.f3c, @AddWith, @AddWith(%U) [symbolic]
// CHECK:STDOUT:   %impl.elem0.c7a: type = impl_witness_access %AddWith.lookup_impl_witness.a82, element0 [symbolic]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %AddWith_where.type.7e9: type = facet_type <@AddWith, @AddWith(%U) where %impl.elem0.c7a impls @Destroy> [symbolic]
// CHECK:STDOUT:   %pattern_type.afc: type = pattern_type %AddWith_where.type.7e9 [symbolic]
// CHECK:STDOUT:   %T.patt.083: %pattern_type.afc = symbolic_binding_pattern T, 1 [symbolic]
// CHECK:STDOUT:   %T.96a: %AddWith_where.type.7e9 = symbolic_binding T, 1 [symbolic]
// CHECK:STDOUT:   %T.as_type.331: type = facet_access_type %T.96a [symbolic]
// CHECK:STDOUT:   %pattern_type.aed556.2: type = pattern_type %T.as_type.331 [symbolic]
// CHECK:STDOUT:   %x.param_patt.373: %pattern_type.aed556.2 = value_param_pattern [symbolic]
// CHECK:STDOUT:   %x.patt.f8d: %pattern_type.aed556.2 = at_binding_pattern x, %x.param_patt.373 [symbolic]
// CHECK:STDOUT:   %pattern_type.51d1c4.2: type = pattern_type %U [symbolic]
// CHECK:STDOUT:   %y.param_patt: %pattern_type.51d1c4.2 = value_param_pattern [symbolic]
// CHECK:STDOUT:   %y.patt: %pattern_type.51d1c4.2 = at_binding_pattern y, %y.param_patt [symbolic]
// CHECK:STDOUT:   %assoc1.57d: %AddWith.assoc_type.588227.1 = assoc_entity element1, imports.%Core.import_ref.976 [symbolic]
// CHECK:STDOUT:   %AddWith.lookup_impl_witness.73583b.2: <witness> = lookup_impl_witness %T.96a, @AddWith, @AddWith(%U) [symbolic]
// CHECK:STDOUT:   %AddWith.facet: %AddWith.type.7e967c.2 = facet_value %T.as_type.331, (%AddWith.lookup_impl_witness.73583b.2) [symbolic]
// CHECK:STDOUT:   %impl.elem0.237f5f.2: type = impl_witness_access %AddWith.lookup_impl_witness.73583b.2, element0 [symbolic]
// CHECK:STDOUT:   %AddWith.WithSelf.Op.type.94c: type = fn_type @AddWith.WithSelf.Op, @AddWith.WithSelf(%U, %AddWith.facet) [symbolic]
// CHECK:STDOUT:   %.e17: type = fn_type_with_self_type %AddWith.WithSelf.Op.type.94c, %AddWith.facet [symbolic]
// CHECK:STDOUT:   %impl.elem1.8d3: %.e17 = impl_witness_access %AddWith.lookup_impl_witness.73583b.2, element1 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.b33: <specific function> = specific_impl_function %impl.elem1.8d3, @AddWith.WithSelf.Op(%U, %AddWith.facet) [symbolic]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.9b9: <witness> = lookup_impl_witness %impl.elem0.237f5f.2, @Destroy [symbolic]
// CHECK:STDOUT:   %Destroy.facet.b81: %Destroy.type = facet_value %impl.elem0.237f5f.2, (%Destroy.lookup_impl_witness.9b9) [symbolic]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.b80: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet.b81) [symbolic]
// CHECK:STDOUT:   %.671: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.b80, %Destroy.facet.b81 [symbolic]
// CHECK:STDOUT:   %impl.elem0.497: %.671 = impl_witness_access %Destroy.lookup_impl_witness.9b9, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.3b7: <specific function> = specific_impl_function %impl.elem0.497, @Destroy.WithSelf.Op(%Destroy.facet.b81) [symbolic]
// CHECK:STDOUT:   %AddAssignWith.type.2c8890.1: type = facet_type <@AddAssignWith, @AddAssignWith(%Other)> [symbolic]
// CHECK:STDOUT:   %Self.792732.1: %AddAssignWith.type.2c8890.1 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %AddAssignWith.assoc_type.8ddaf4.1: type = assoc_entity_type @AddAssignWith, @AddAssignWith(%Other) [symbolic]
// CHECK:STDOUT:   %AddAssignWith.WithSelf.Op.type.cd574e.1: type = fn_type @AddAssignWith.WithSelf.Op, @AddAssignWith.WithSelf(%Other, %Self.792732.1) [symbolic]
// CHECK:STDOUT:   %AddAssignWith.WithSelf.Op.106889.1: %AddAssignWith.WithSelf.Op.type.cd574e.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %AddAssignWith.type.2c8890.2: type = facet_type <@AddAssignWith, @AddAssignWith(%U)> [symbolic]
// CHECK:STDOUT:   %pattern_type.82d: type = pattern_type %AddAssignWith.type.2c8890.2 [symbolic]
// CHECK:STDOUT:   %T.patt.a22: %pattern_type.82d = symbolic_binding_pattern T, 1 [symbolic]
// CHECK:STDOUT:   %T.792: %AddAssignWith.type.2c8890.2 = symbolic_binding T, 1 [symbolic]
// CHECK:STDOUT:   %AddAssignWith.assoc_type.8ddaf4.2: type = assoc_entity_type @AddAssignWith, @AddAssignWith(%U) [symbolic]
// CHECK:STDOUT:   %assoc0.8c5739.2: %AddAssignWith.assoc_type.8ddaf4.2 = assoc_entity element0, imports.%Core.import_ref.3b2 [symbolic]
// CHECK:STDOUT:   %T.as_type.50c: type = facet_access_type %T.792 [symbolic]
// CHECK:STDOUT:   %pattern_type.643306.2: type = pattern_type %T.as_type.50c [symbolic]
// CHECK:STDOUT:   %x.patt.227: %pattern_type.643306.2 = ref_binding_pattern x [symbolic]
// CHECK:STDOUT:   %x.param_patt.343: %pattern_type.643306.2 = var_param_pattern %x.patt.227 [symbolic]
// CHECK:STDOUT:   %require_complete.2a8: <witness> = require_complete_type %AddAssignWith.type.2c8890.2 [symbolic]
// CHECK:STDOUT:   %assoc0.86d: %AddAssignWith.assoc_type.8ddaf4.1 = assoc_entity element0, imports.%Core.import_ref.87e [symbolic]
// CHECK:STDOUT:   %AddAssignWith.lookup_impl_witness: <witness> = lookup_impl_witness %T.792, @AddAssignWith, @AddAssignWith(%U) [symbolic]
// CHECK:STDOUT:   %AddAssignWith.WithSelf.Op.type.cd574e.3: type = fn_type @AddAssignWith.WithSelf.Op, @AddAssignWith.WithSelf(%U, %T.792) [symbolic]
// CHECK:STDOUT:   %.9b1: type = fn_type_with_self_type %AddAssignWith.WithSelf.Op.type.cd574e.3, %T.792 [symbolic]
// CHECK:STDOUT:   %impl.elem0.985: %.9b1 = impl_witness_access %AddAssignWith.lookup_impl_witness, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.d36: <specific function> = specific_impl_function %impl.elem0.985, @AddAssignWith.WithSelf.Op(%U, %T.792) [symbolic]
// CHECK:STDOUT:   %SubWith.type.ddc625.1: type = facet_type <@SubWith, @SubWith(%Other)> [symbolic]
// CHECK:STDOUT:   %Self.d6b75b.1: %SubWith.type.ddc625.1 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %SubWith.assoc_type.b0beca.1: type = assoc_entity_type @SubWith, @SubWith(%Other) [symbolic]
// CHECK:STDOUT:   %SubWith.WithSelf.Op.type.68d038.1: type = fn_type @SubWith.WithSelf.Op, @SubWith.WithSelf(%Other, %Self.d6b75b.1) [symbolic]
// CHECK:STDOUT:   %SubWith.WithSelf.Op.97ab45.1: %SubWith.WithSelf.Op.type.68d038.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %SubWith.type.ddc625.2: type = facet_type <@SubWith, @SubWith(%U)> [symbolic]
// CHECK:STDOUT:   %.Self.b46: %SubWith.type.ddc625.2 = symbolic_binding .Self [symbolic]
// CHECK:STDOUT:   %SubWith.assoc_type.b0beca.2: type = assoc_entity_type @SubWith, @SubWith(%U) [symbolic]
// CHECK:STDOUT:   %assoc0.98d9fa.2: %SubWith.assoc_type.b0beca.2 = assoc_entity element0, imports.%Core.import_ref.190 [symbolic]
// CHECK:STDOUT:   %assoc1.e3bf87.2: %SubWith.assoc_type.b0beca.2 = assoc_entity element1, imports.%Core.import_ref.3a4 [symbolic]
// CHECK:STDOUT:   %require_complete.3b3: <witness> = require_complete_type %SubWith.type.ddc625.2 [symbolic]
// CHECK:STDOUT:   %.Self.as_type.10d: type = facet_access_type %.Self.b46 [symbolic]
// CHECK:STDOUT:   %SubWith.lookup_impl_witness.80b: <witness> = lookup_impl_witness %.Self.b46, @SubWith, @SubWith(%U) [symbolic]
// CHECK:STDOUT:   %impl.elem0.2a4: type = impl_witness_access %SubWith.lookup_impl_witness.80b, element0 [symbolic]
// CHECK:STDOUT:   %SubWith_where.type.ddc: type = facet_type <@SubWith, @SubWith(%U) where %impl.elem0.2a4 impls @Destroy> [symbolic]
// CHECK:STDOUT:   %pattern_type.a77: type = pattern_type %SubWith_where.type.ddc [symbolic]
// CHECK:STDOUT:   %T.patt.121: %pattern_type.a77 = symbolic_binding_pattern T, 1 [symbolic]
// CHECK:STDOUT:   %T.d6b: %SubWith_where.type.ddc = symbolic_binding T, 1 [symbolic]
// CHECK:STDOUT:   %T.as_type.449: type = facet_access_type %T.d6b [symbolic]
// CHECK:STDOUT:   %pattern_type.f1e08b.2: type = pattern_type %T.as_type.449 [symbolic]
// CHECK:STDOUT:   %x.param_patt.725: %pattern_type.f1e08b.2 = value_param_pattern [symbolic]
// CHECK:STDOUT:   %x.patt.8e3: %pattern_type.f1e08b.2 = at_binding_pattern x, %x.param_patt.725 [symbolic]
// CHECK:STDOUT:   %assoc1.178: %SubWith.assoc_type.b0beca.1 = assoc_entity element1, imports.%Core.import_ref.b59 [symbolic]
// CHECK:STDOUT:   %SubWith.lookup_impl_witness.a9fbd3.2: <witness> = lookup_impl_witness %T.d6b, @SubWith, @SubWith(%U) [symbolic]
// CHECK:STDOUT:   %SubWith.facet: %SubWith.type.ddc625.2 = facet_value %T.as_type.449, (%SubWith.lookup_impl_witness.a9fbd3.2) [symbolic]
// CHECK:STDOUT:   %impl.elem0.1583c7.2: type = impl_witness_access %SubWith.lookup_impl_witness.a9fbd3.2, element0 [symbolic]
// CHECK:STDOUT:   %SubWith.WithSelf.Op.type.b7d: type = fn_type @SubWith.WithSelf.Op, @SubWith.WithSelf(%U, %SubWith.facet) [symbolic]
// CHECK:STDOUT:   %.c28: type = fn_type_with_self_type %SubWith.WithSelf.Op.type.b7d, %SubWith.facet [symbolic]
// CHECK:STDOUT:   %impl.elem1.df0: %.c28 = impl_witness_access %SubWith.lookup_impl_witness.a9fbd3.2, element1 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.882: <specific function> = specific_impl_function %impl.elem1.df0, @SubWith.WithSelf.Op(%U, %SubWith.facet) [symbolic]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.4ac: <witness> = lookup_impl_witness %impl.elem0.1583c7.2, @Destroy [symbolic]
// CHECK:STDOUT:   %Destroy.facet.622: %Destroy.type = facet_value %impl.elem0.1583c7.2, (%Destroy.lookup_impl_witness.4ac) [symbolic]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.4d5: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet.622) [symbolic]
// CHECK:STDOUT:   %.14c: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.4d5, %Destroy.facet.622 [symbolic]
// CHECK:STDOUT:   %impl.elem0.aa8: %.14c = impl_witness_access %Destroy.lookup_impl_witness.4ac, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.a72: <specific function> = specific_impl_function %impl.elem0.aa8, @Destroy.WithSelf.Op(%Destroy.facet.622) [symbolic]
// CHECK:STDOUT:   %SubAssignWith.type.7e53cf.1: type = facet_type <@SubAssignWith, @SubAssignWith(%Other)> [symbolic]
// CHECK:STDOUT:   %Self.f6c9aa.1: %SubAssignWith.type.7e53cf.1 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %SubAssignWith.assoc_type.6e3877.1: type = assoc_entity_type @SubAssignWith, @SubAssignWith(%Other) [symbolic]
// CHECK:STDOUT:   %SubAssignWith.WithSelf.Op.type.42f7cc.1: type = fn_type @SubAssignWith.WithSelf.Op, @SubAssignWith.WithSelf(%Other, %Self.f6c9aa.1) [symbolic]
// CHECK:STDOUT:   %SubAssignWith.WithSelf.Op.25af21.1: %SubAssignWith.WithSelf.Op.type.42f7cc.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %SubAssignWith.type.7e53cf.2: type = facet_type <@SubAssignWith, @SubAssignWith(%U)> [symbolic]
// CHECK:STDOUT:   %pattern_type.260: type = pattern_type %SubAssignWith.type.7e53cf.2 [symbolic]
// CHECK:STDOUT:   %T.patt.f45: %pattern_type.260 = symbolic_binding_pattern T, 1 [symbolic]
// CHECK:STDOUT:   %T.f6c: %SubAssignWith.type.7e53cf.2 = symbolic_binding T, 1 [symbolic]
// CHECK:STDOUT:   %SubAssignWith.assoc_type.6e3877.2: type = assoc_entity_type @SubAssignWith, @SubAssignWith(%U) [symbolic]
// CHECK:STDOUT:   %assoc0.701e5f.2: %SubAssignWith.assoc_type.6e3877.2 = assoc_entity element0, imports.%Core.import_ref.8f2 [symbolic]
// CHECK:STDOUT:   %T.as_type.3ff: type = facet_access_type %T.f6c [symbolic]
// CHECK:STDOUT:   %pattern_type.f08ed5.2: type = pattern_type %T.as_type.3ff [symbolic]
// CHECK:STDOUT:   %x.patt.95a: %pattern_type.f08ed5.2 = ref_binding_pattern x [symbolic]
// CHECK:STDOUT:   %x.param_patt.a45: %pattern_type.f08ed5.2 = var_param_pattern %x.patt.95a [symbolic]
// CHECK:STDOUT:   %require_complete.862: <witness> = require_complete_type %SubAssignWith.type.7e53cf.2 [symbolic]
// CHECK:STDOUT:   %assoc0.b8e: %SubAssignWith.assoc_type.6e3877.1 = assoc_entity element0, imports.%Core.import_ref.e7f [symbolic]
// CHECK:STDOUT:   %SubAssignWith.lookup_impl_witness: <witness> = lookup_impl_witness %T.f6c, @SubAssignWith, @SubAssignWith(%U) [symbolic]
// CHECK:STDOUT:   %SubAssignWith.WithSelf.Op.type.42f7cc.3: type = fn_type @SubAssignWith.WithSelf.Op, @SubAssignWith.WithSelf(%U, %T.f6c) [symbolic]
// CHECK:STDOUT:   %.e56: type = fn_type_with_self_type %SubAssignWith.WithSelf.Op.type.42f7cc.3, %T.f6c [symbolic]
// CHECK:STDOUT:   %impl.elem0.dd3: %.e56 = impl_witness_access %SubAssignWith.lookup_impl_witness, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.5da: <specific function> = specific_impl_function %impl.elem0.dd3, @SubAssignWith.WithSelf.Op(%U, %T.f6c) [symbolic]
// CHECK:STDOUT:   %MulWith.type.f84f0b.1: type = facet_type <@MulWith, @MulWith(%Other)> [symbolic]
// CHECK:STDOUT:   %Self.941007.1: %MulWith.type.f84f0b.1 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %MulWith.assoc_type.7b2892.1: type = assoc_entity_type @MulWith, @MulWith(%Other) [symbolic]
// CHECK:STDOUT:   %MulWith.WithSelf.Op.type.176210.1: type = fn_type @MulWith.WithSelf.Op, @MulWith.WithSelf(%Other, %Self.941007.1) [symbolic]
// CHECK:STDOUT:   %MulWith.WithSelf.Op.a7516d.1: %MulWith.WithSelf.Op.type.176210.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %MulWith.type.f84f0b.2: type = facet_type <@MulWith, @MulWith(%U)> [symbolic]
// CHECK:STDOUT:   %.Self.059: %MulWith.type.f84f0b.2 = symbolic_binding .Self [symbolic]
// CHECK:STDOUT:   %MulWith.assoc_type.7b2892.2: type = assoc_entity_type @MulWith, @MulWith(%U) [symbolic]
// CHECK:STDOUT:   %assoc0.2b8b9e.2: %MulWith.assoc_type.7b2892.2 = assoc_entity element0, imports.%Core.import_ref.59c [symbolic]
// CHECK:STDOUT:   %assoc1.59776a.2: %MulWith.assoc_type.7b2892.2 = assoc_entity element1, imports.%Core.import_ref.5a0 [symbolic]
// CHECK:STDOUT:   %require_complete.061: <witness> = require_complete_type %MulWith.type.f84f0b.2 [symbolic]
// CHECK:STDOUT:   %.Self.as_type.bed: type = facet_access_type %.Self.059 [symbolic]
// CHECK:STDOUT:   %MulWith.lookup_impl_witness.997: <witness> = lookup_impl_witness %.Self.059, @MulWith, @MulWith(%U) [symbolic]
// CHECK:STDOUT:   %impl.elem0.529: type = impl_witness_access %MulWith.lookup_impl_witness.997, element0 [symbolic]
// CHECK:STDOUT:   %MulWith_where.type.f84: type = facet_type <@MulWith, @MulWith(%U) where %impl.elem0.529 impls @Destroy> [symbolic]
// CHECK:STDOUT:   %pattern_type.a11: type = pattern_type %MulWith_where.type.f84 [symbolic]
// CHECK:STDOUT:   %T.patt.8ab: %pattern_type.a11 = symbolic_binding_pattern T, 1 [symbolic]
// CHECK:STDOUT:   %T.941: %MulWith_where.type.f84 = symbolic_binding T, 1 [symbolic]
// CHECK:STDOUT:   %T.as_type.ad2: type = facet_access_type %T.941 [symbolic]
// CHECK:STDOUT:   %pattern_type.eccc2c.2: type = pattern_type %T.as_type.ad2 [symbolic]
// CHECK:STDOUT:   %x.param_patt.114: %pattern_type.eccc2c.2 = value_param_pattern [symbolic]
// CHECK:STDOUT:   %x.patt.596: %pattern_type.eccc2c.2 = at_binding_pattern x, %x.param_patt.114 [symbolic]
// CHECK:STDOUT:   %assoc1.8f7: %MulWith.assoc_type.7b2892.1 = assoc_entity element1, imports.%Core.import_ref.955 [symbolic]
// CHECK:STDOUT:   %MulWith.lookup_impl_witness.5c9016.2: <witness> = lookup_impl_witness %T.941, @MulWith, @MulWith(%U) [symbolic]
// CHECK:STDOUT:   %MulWith.facet: %MulWith.type.f84f0b.2 = facet_value %T.as_type.ad2, (%MulWith.lookup_impl_witness.5c9016.2) [symbolic]
// CHECK:STDOUT:   %impl.elem0.f2c0ce.2: type = impl_witness_access %MulWith.lookup_impl_witness.5c9016.2, element0 [symbolic]
// CHECK:STDOUT:   %MulWith.WithSelf.Op.type.d2e: type = fn_type @MulWith.WithSelf.Op, @MulWith.WithSelf(%U, %MulWith.facet) [symbolic]
// CHECK:STDOUT:   %.ef7: type = fn_type_with_self_type %MulWith.WithSelf.Op.type.d2e, %MulWith.facet [symbolic]
// CHECK:STDOUT:   %impl.elem1.19d: %.ef7 = impl_witness_access %MulWith.lookup_impl_witness.5c9016.2, element1 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.3df: <specific function> = specific_impl_function %impl.elem1.19d, @MulWith.WithSelf.Op(%U, %MulWith.facet) [symbolic]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.a13: <witness> = lookup_impl_witness %impl.elem0.f2c0ce.2, @Destroy [symbolic]
// CHECK:STDOUT:   %Destroy.facet.9a4: %Destroy.type = facet_value %impl.elem0.f2c0ce.2, (%Destroy.lookup_impl_witness.a13) [symbolic]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.9cf: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet.9a4) [symbolic]
// CHECK:STDOUT:   %.731: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.9cf, %Destroy.facet.9a4 [symbolic]
// CHECK:STDOUT:   %impl.elem0.f89: %.731 = impl_witness_access %Destroy.lookup_impl_witness.a13, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.601: <specific function> = specific_impl_function %impl.elem0.f89, @Destroy.WithSelf.Op(%Destroy.facet.9a4) [symbolic]
// CHECK:STDOUT:   %MulAssignWith.type.3a0e4d.1: type = facet_type <@MulAssignWith, @MulAssignWith(%Other)> [symbolic]
// CHECK:STDOUT:   %Self.64dcb0.1: %MulAssignWith.type.3a0e4d.1 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %MulAssignWith.assoc_type.9aa18f.1: type = assoc_entity_type @MulAssignWith, @MulAssignWith(%Other) [symbolic]
// CHECK:STDOUT:   %MulAssignWith.WithSelf.Op.type.5df5bd.1: type = fn_type @MulAssignWith.WithSelf.Op, @MulAssignWith.WithSelf(%Other, %Self.64dcb0.1) [symbolic]
// CHECK:STDOUT:   %MulAssignWith.WithSelf.Op.ee0eda.1: %MulAssignWith.WithSelf.Op.type.5df5bd.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %MulAssignWith.type.3a0e4d.2: type = facet_type <@MulAssignWith, @MulAssignWith(%U)> [symbolic]
// CHECK:STDOUT:   %pattern_type.8c2: type = pattern_type %MulAssignWith.type.3a0e4d.2 [symbolic]
// CHECK:STDOUT:   %T.patt.db5: %pattern_type.8c2 = symbolic_binding_pattern T, 1 [symbolic]
// CHECK:STDOUT:   %T.64d: %MulAssignWith.type.3a0e4d.2 = symbolic_binding T, 1 [symbolic]
// CHECK:STDOUT:   %MulAssignWith.assoc_type.9aa18f.2: type = assoc_entity_type @MulAssignWith, @MulAssignWith(%U) [symbolic]
// CHECK:STDOUT:   %assoc0.22264a.2: %MulAssignWith.assoc_type.9aa18f.2 = assoc_entity element0, imports.%Core.import_ref.ba3 [symbolic]
// CHECK:STDOUT:   %T.as_type.1de: type = facet_access_type %T.64d [symbolic]
// CHECK:STDOUT:   %pattern_type.8bf366.2: type = pattern_type %T.as_type.1de [symbolic]
// CHECK:STDOUT:   %x.patt.4e9: %pattern_type.8bf366.2 = ref_binding_pattern x [symbolic]
// CHECK:STDOUT:   %x.param_patt.6c8: %pattern_type.8bf366.2 = var_param_pattern %x.patt.4e9 [symbolic]
// CHECK:STDOUT:   %require_complete.6b1: <witness> = require_complete_type %MulAssignWith.type.3a0e4d.2 [symbolic]
// CHECK:STDOUT:   %assoc0.2ec: %MulAssignWith.assoc_type.9aa18f.1 = assoc_entity element0, imports.%Core.import_ref.408 [symbolic]
// CHECK:STDOUT:   %MulAssignWith.lookup_impl_witness: <witness> = lookup_impl_witness %T.64d, @MulAssignWith, @MulAssignWith(%U) [symbolic]
// CHECK:STDOUT:   %MulAssignWith.WithSelf.Op.type.5df5bd.3: type = fn_type @MulAssignWith.WithSelf.Op, @MulAssignWith.WithSelf(%U, %T.64d) [symbolic]
// CHECK:STDOUT:   %.2db: type = fn_type_with_self_type %MulAssignWith.WithSelf.Op.type.5df5bd.3, %T.64d [symbolic]
// CHECK:STDOUT:   %impl.elem0.c0f: %.2db = impl_witness_access %MulAssignWith.lookup_impl_witness, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.54d: <specific function> = specific_impl_function %impl.elem0.c0f, @MulAssignWith.WithSelf.Op(%U, %T.64d) [symbolic]
// CHECK:STDOUT:   %DivWith.type.13f8f2.1: type = facet_type <@DivWith, @DivWith(%Other)> [symbolic]
// CHECK:STDOUT:   %Self.16a753.1: %DivWith.type.13f8f2.1 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %DivWith.assoc_type.d8b290.1: type = assoc_entity_type @DivWith, @DivWith(%Other) [symbolic]
// CHECK:STDOUT:   %DivWith.WithSelf.Op.type.0f05c0.1: type = fn_type @DivWith.WithSelf.Op, @DivWith.WithSelf(%Other, %Self.16a753.1) [symbolic]
// CHECK:STDOUT:   %DivWith.WithSelf.Op.51e40b.1: %DivWith.WithSelf.Op.type.0f05c0.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %DivWith.type.13f8f2.2: type = facet_type <@DivWith, @DivWith(%U)> [symbolic]
// CHECK:STDOUT:   %.Self.5d9: %DivWith.type.13f8f2.2 = symbolic_binding .Self [symbolic]
// CHECK:STDOUT:   %DivWith.assoc_type.d8b290.2: type = assoc_entity_type @DivWith, @DivWith(%U) [symbolic]
// CHECK:STDOUT:   %assoc0.b84b27.2: %DivWith.assoc_type.d8b290.2 = assoc_entity element0, imports.%Core.import_ref.dee [symbolic]
// CHECK:STDOUT:   %assoc1.03cb8d.2: %DivWith.assoc_type.d8b290.2 = assoc_entity element1, imports.%Core.import_ref.784 [symbolic]
// CHECK:STDOUT:   %require_complete.975: <witness> = require_complete_type %DivWith.type.13f8f2.2 [symbolic]
// CHECK:STDOUT:   %.Self.as_type.123: type = facet_access_type %.Self.5d9 [symbolic]
// CHECK:STDOUT:   %DivWith.lookup_impl_witness.c7d: <witness> = lookup_impl_witness %.Self.5d9, @DivWith, @DivWith(%U) [symbolic]
// CHECK:STDOUT:   %impl.elem0.339: type = impl_witness_access %DivWith.lookup_impl_witness.c7d, element0 [symbolic]
// CHECK:STDOUT:   %DivWith_where.type.13f: type = facet_type <@DivWith, @DivWith(%U) where %impl.elem0.339 impls @Destroy> [symbolic]
// CHECK:STDOUT:   %pattern_type.f4a: type = pattern_type %DivWith_where.type.13f [symbolic]
// CHECK:STDOUT:   %T.patt.97c: %pattern_type.f4a = symbolic_binding_pattern T, 1 [symbolic]
// CHECK:STDOUT:   %T.16a: %DivWith_where.type.13f = symbolic_binding T, 1 [symbolic]
// CHECK:STDOUT:   %T.as_type.f1d: type = facet_access_type %T.16a [symbolic]
// CHECK:STDOUT:   %pattern_type.b25246.2: type = pattern_type %T.as_type.f1d [symbolic]
// CHECK:STDOUT:   %x.param_patt.494: %pattern_type.b25246.2 = value_param_pattern [symbolic]
// CHECK:STDOUT:   %x.patt.8ae: %pattern_type.b25246.2 = at_binding_pattern x, %x.param_patt.494 [symbolic]
// CHECK:STDOUT:   %assoc1.7b1: %DivWith.assoc_type.d8b290.1 = assoc_entity element1, imports.%Core.import_ref.8b6 [symbolic]
// CHECK:STDOUT:   %DivWith.lookup_impl_witness.1efd1b.2: <witness> = lookup_impl_witness %T.16a, @DivWith, @DivWith(%U) [symbolic]
// CHECK:STDOUT:   %DivWith.facet: %DivWith.type.13f8f2.2 = facet_value %T.as_type.f1d, (%DivWith.lookup_impl_witness.1efd1b.2) [symbolic]
// CHECK:STDOUT:   %impl.elem0.e2a7ee.2: type = impl_witness_access %DivWith.lookup_impl_witness.1efd1b.2, element0 [symbolic]
// CHECK:STDOUT:   %DivWith.WithSelf.Op.type.208: type = fn_type @DivWith.WithSelf.Op, @DivWith.WithSelf(%U, %DivWith.facet) [symbolic]
// CHECK:STDOUT:   %.401: type = fn_type_with_self_type %DivWith.WithSelf.Op.type.208, %DivWith.facet [symbolic]
// CHECK:STDOUT:   %impl.elem1.f2d: %.401 = impl_witness_access %DivWith.lookup_impl_witness.1efd1b.2, element1 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.567: <specific function> = specific_impl_function %impl.elem1.f2d, @DivWith.WithSelf.Op(%U, %DivWith.facet) [symbolic]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.548: <witness> = lookup_impl_witness %impl.elem0.e2a7ee.2, @Destroy [symbolic]
// CHECK:STDOUT:   %Destroy.facet.680: %Destroy.type = facet_value %impl.elem0.e2a7ee.2, (%Destroy.lookup_impl_witness.548) [symbolic]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.d31: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet.680) [symbolic]
// CHECK:STDOUT:   %.c1c: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.d31, %Destroy.facet.680 [symbolic]
// CHECK:STDOUT:   %impl.elem0.800: %.c1c = impl_witness_access %Destroy.lookup_impl_witness.548, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.598: <specific function> = specific_impl_function %impl.elem0.800, @Destroy.WithSelf.Op(%Destroy.facet.680) [symbolic]
// CHECK:STDOUT:   %DivAssignWith.type.7e58ca.1: type = facet_type <@DivAssignWith, @DivAssignWith(%Other)> [symbolic]
// CHECK:STDOUT:   %Self.c33c81.1: %DivAssignWith.type.7e58ca.1 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %DivAssignWith.assoc_type.bb68ec.1: type = assoc_entity_type @DivAssignWith, @DivAssignWith(%Other) [symbolic]
// CHECK:STDOUT:   %DivAssignWith.WithSelf.Op.type.ff5775.1: type = fn_type @DivAssignWith.WithSelf.Op, @DivAssignWith.WithSelf(%Other, %Self.c33c81.1) [symbolic]
// CHECK:STDOUT:   %DivAssignWith.WithSelf.Op.91e0ff.1: %DivAssignWith.WithSelf.Op.type.ff5775.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %DivAssignWith.type.7e58ca.2: type = facet_type <@DivAssignWith, @DivAssignWith(%U)> [symbolic]
// CHECK:STDOUT:   %pattern_type.f74: type = pattern_type %DivAssignWith.type.7e58ca.2 [symbolic]
// CHECK:STDOUT:   %T.patt.9a5: %pattern_type.f74 = symbolic_binding_pattern T, 1 [symbolic]
// CHECK:STDOUT:   %T.c33: %DivAssignWith.type.7e58ca.2 = symbolic_binding T, 1 [symbolic]
// CHECK:STDOUT:   %DivAssignWith.assoc_type.bb68ec.2: type = assoc_entity_type @DivAssignWith, @DivAssignWith(%U) [symbolic]
// CHECK:STDOUT:   %assoc0.32f100.2: %DivAssignWith.assoc_type.bb68ec.2 = assoc_entity element0, imports.%Core.import_ref.a5d [symbolic]
// CHECK:STDOUT:   %T.as_type.e57: type = facet_access_type %T.c33 [symbolic]
// CHECK:STDOUT:   %pattern_type.da303b.2: type = pattern_type %T.as_type.e57 [symbolic]
// CHECK:STDOUT:   %x.patt.cf6: %pattern_type.da303b.2 = ref_binding_pattern x [symbolic]
// CHECK:STDOUT:   %x.param_patt.660: %pattern_type.da303b.2 = var_param_pattern %x.patt.cf6 [symbolic]
// CHECK:STDOUT:   %require_complete.c5c: <witness> = require_complete_type %DivAssignWith.type.7e58ca.2 [symbolic]
// CHECK:STDOUT:   %assoc0.de1: %DivAssignWith.assoc_type.bb68ec.1 = assoc_entity element0, imports.%Core.import_ref.df2 [symbolic]
// CHECK:STDOUT:   %DivAssignWith.lookup_impl_witness: <witness> = lookup_impl_witness %T.c33, @DivAssignWith, @DivAssignWith(%U) [symbolic]
// CHECK:STDOUT:   %DivAssignWith.WithSelf.Op.type.ff5775.3: type = fn_type @DivAssignWith.WithSelf.Op, @DivAssignWith.WithSelf(%U, %T.c33) [symbolic]
// CHECK:STDOUT:   %.754: type = fn_type_with_self_type %DivAssignWith.WithSelf.Op.type.ff5775.3, %T.c33 [symbolic]
// CHECK:STDOUT:   %impl.elem0.0c4: %.754 = impl_witness_access %DivAssignWith.lookup_impl_witness, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.011: <specific function> = specific_impl_function %impl.elem0.0c4, @DivAssignWith.WithSelf.Op(%U, %T.c33) [symbolic]
// CHECK:STDOUT:   %ModWith.type.2e82f9.1: type = facet_type <@ModWith, @ModWith(%Other)> [symbolic]
// CHECK:STDOUT:   %Self.0ef52c.1: %ModWith.type.2e82f9.1 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %ModWith.assoc_type.316ae9.1: type = assoc_entity_type @ModWith, @ModWith(%Other) [symbolic]
// CHECK:STDOUT:   %ModWith.WithSelf.Op.type.d147b5.1: type = fn_type @ModWith.WithSelf.Op, @ModWith.WithSelf(%Other, %Self.0ef52c.1) [symbolic]
// CHECK:STDOUT:   %ModWith.WithSelf.Op.9e4026.1: %ModWith.WithSelf.Op.type.d147b5.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %ModWith.type.2e82f9.2: type = facet_type <@ModWith, @ModWith(%U)> [symbolic]
// CHECK:STDOUT:   %.Self.bef: %ModWith.type.2e82f9.2 = symbolic_binding .Self [symbolic]
// CHECK:STDOUT:   %ModWith.assoc_type.316ae9.2: type = assoc_entity_type @ModWith, @ModWith(%U) [symbolic]
// CHECK:STDOUT:   %assoc0.624dd7.2: %ModWith.assoc_type.316ae9.2 = assoc_entity element0, imports.%Core.import_ref.61a [symbolic]
// CHECK:STDOUT:   %assoc1.8ee089.2: %ModWith.assoc_type.316ae9.2 = assoc_entity element1, imports.%Core.import_ref.b56 [symbolic]
// CHECK:STDOUT:   %require_complete.cd6: <witness> = require_complete_type %ModWith.type.2e82f9.2 [symbolic]
// CHECK:STDOUT:   %.Self.as_type.538: type = facet_access_type %.Self.bef [symbolic]
// CHECK:STDOUT:   %ModWith.lookup_impl_witness.6e9: <witness> = lookup_impl_witness %.Self.bef, @ModWith, @ModWith(%U) [symbolic]
// CHECK:STDOUT:   %impl.elem0.488: type = impl_witness_access %ModWith.lookup_impl_witness.6e9, element0 [symbolic]
// CHECK:STDOUT:   %ModWith_where.type.2e8: type = facet_type <@ModWith, @ModWith(%U) where %impl.elem0.488 impls @Destroy> [symbolic]
// CHECK:STDOUT:   %pattern_type.7c6: type = pattern_type %ModWith_where.type.2e8 [symbolic]
// CHECK:STDOUT:   %T.patt.520: %pattern_type.7c6 = symbolic_binding_pattern T, 1 [symbolic]
// CHECK:STDOUT:   %T.0ef: %ModWith_where.type.2e8 = symbolic_binding T, 1 [symbolic]
// CHECK:STDOUT:   %T.as_type.c2b: type = facet_access_type %T.0ef [symbolic]
// CHECK:STDOUT:   %pattern_type.d76a9b.2: type = pattern_type %T.as_type.c2b [symbolic]
// CHECK:STDOUT:   %x.param_patt.3d2: %pattern_type.d76a9b.2 = value_param_pattern [symbolic]
// CHECK:STDOUT:   %x.patt.58e: %pattern_type.d76a9b.2 = at_binding_pattern x, %x.param_patt.3d2 [symbolic]
// CHECK:STDOUT:   %assoc1.132: %ModWith.assoc_type.316ae9.1 = assoc_entity element1, imports.%Core.import_ref.33c [symbolic]
// CHECK:STDOUT:   %ModWith.lookup_impl_witness.c327a3.2: <witness> = lookup_impl_witness %T.0ef, @ModWith, @ModWith(%U) [symbolic]
// CHECK:STDOUT:   %ModWith.facet: %ModWith.type.2e82f9.2 = facet_value %T.as_type.c2b, (%ModWith.lookup_impl_witness.c327a3.2) [symbolic]
// CHECK:STDOUT:   %impl.elem0.2a2b98.2: type = impl_witness_access %ModWith.lookup_impl_witness.c327a3.2, element0 [symbolic]
// CHECK:STDOUT:   %ModWith.WithSelf.Op.type.e58: type = fn_type @ModWith.WithSelf.Op, @ModWith.WithSelf(%U, %ModWith.facet) [symbolic]
// CHECK:STDOUT:   %.07c: type = fn_type_with_self_type %ModWith.WithSelf.Op.type.e58, %ModWith.facet [symbolic]
// CHECK:STDOUT:   %impl.elem1.617: %.07c = impl_witness_access %ModWith.lookup_impl_witness.c327a3.2, element1 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.95f: <specific function> = specific_impl_function %impl.elem1.617, @ModWith.WithSelf.Op(%U, %ModWith.facet) [symbolic]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.64a: <witness> = lookup_impl_witness %impl.elem0.2a2b98.2, @Destroy [symbolic]
// CHECK:STDOUT:   %Destroy.facet.6e7: %Destroy.type = facet_value %impl.elem0.2a2b98.2, (%Destroy.lookup_impl_witness.64a) [symbolic]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.e36: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet.6e7) [symbolic]
// CHECK:STDOUT:   %.6ed: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.e36, %Destroy.facet.6e7 [symbolic]
// CHECK:STDOUT:   %impl.elem0.918: %.6ed = impl_witness_access %Destroy.lookup_impl_witness.64a, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.f55: <specific function> = specific_impl_function %impl.elem0.918, @Destroy.WithSelf.Op(%Destroy.facet.6e7) [symbolic]
// CHECK:STDOUT:   %ModAssignWith.type.778d68.1: type = facet_type <@ModAssignWith, @ModAssignWith(%Other)> [symbolic]
// CHECK:STDOUT:   %Self.bca00b.1: %ModAssignWith.type.778d68.1 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %ModAssignWith.assoc_type.bc79ec.1: type = assoc_entity_type @ModAssignWith, @ModAssignWith(%Other) [symbolic]
// CHECK:STDOUT:   %ModAssignWith.WithSelf.Op.type.ec4109.1: type = fn_type @ModAssignWith.WithSelf.Op, @ModAssignWith.WithSelf(%Other, %Self.bca00b.1) [symbolic]
// CHECK:STDOUT:   %ModAssignWith.WithSelf.Op.ac45fc.1: %ModAssignWith.WithSelf.Op.type.ec4109.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %ModAssignWith.type.778d68.2: type = facet_type <@ModAssignWith, @ModAssignWith(%U)> [symbolic]
// CHECK:STDOUT:   %pattern_type.fb8: type = pattern_type %ModAssignWith.type.778d68.2 [symbolic]
// CHECK:STDOUT:   %T.patt.c29: %pattern_type.fb8 = symbolic_binding_pattern T, 1 [symbolic]
// CHECK:STDOUT:   %T.bca: %ModAssignWith.type.778d68.2 = symbolic_binding T, 1 [symbolic]
// CHECK:STDOUT:   %ModAssignWith.assoc_type.bc79ec.2: type = assoc_entity_type @ModAssignWith, @ModAssignWith(%U) [symbolic]
// CHECK:STDOUT:   %assoc0.8c0719.2: %ModAssignWith.assoc_type.bc79ec.2 = assoc_entity element0, imports.%Core.import_ref.cf6 [symbolic]
// CHECK:STDOUT:   %T.as_type.ce6: type = facet_access_type %T.bca [symbolic]
// CHECK:STDOUT:   %pattern_type.1d5f30.2: type = pattern_type %T.as_type.ce6 [symbolic]
// CHECK:STDOUT:   %x.patt.fa4: %pattern_type.1d5f30.2 = ref_binding_pattern x [symbolic]
// CHECK:STDOUT:   %x.param_patt.749: %pattern_type.1d5f30.2 = var_param_pattern %x.patt.fa4 [symbolic]
// CHECK:STDOUT:   %require_complete.680: <witness> = require_complete_type %ModAssignWith.type.778d68.2 [symbolic]
// CHECK:STDOUT:   %assoc0.0f3: %ModAssignWith.assoc_type.bc79ec.1 = assoc_entity element0, imports.%Core.import_ref.d29 [symbolic]
// CHECK:STDOUT:   %ModAssignWith.lookup_impl_witness: <witness> = lookup_impl_witness %T.bca, @ModAssignWith, @ModAssignWith(%U) [symbolic]
// CHECK:STDOUT:   %ModAssignWith.WithSelf.Op.type.ec4109.3: type = fn_type @ModAssignWith.WithSelf.Op, @ModAssignWith.WithSelf(%U, %T.bca) [symbolic]
// CHECK:STDOUT:   %.965: type = fn_type_with_self_type %ModAssignWith.WithSelf.Op.type.ec4109.3, %T.bca [symbolic]
// CHECK:STDOUT:   %impl.elem0.ec7: %.965 = impl_witness_access %ModAssignWith.lookup_impl_witness, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.cf7: <specific function> = specific_impl_function %impl.elem0.ec7, @ModAssignWith.WithSelf.Op(%U, %T.bca) [symbolic]
// CHECK:STDOUT:   %Negate.type: type = facet_type <@Negate> [concrete]
// CHECK:STDOUT:   %.Self.61e: %Negate.type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %Negate.lookup_impl_witness.dec: <witness> = lookup_impl_witness %.Self.61e, @Negate [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.f66: type = impl_witness_access %Negate.lookup_impl_witness.dec, element0 [symbolic_self]
// CHECK:STDOUT:   %Negate_where.type.871: type = facet_type <@Negate where %impl.elem0.f66 impls @Destroy> [concrete]
// CHECK:STDOUT:   %pattern_type.801: type = pattern_type %Negate_where.type.871 [concrete]
// CHECK:STDOUT:   %T.patt.91c: %pattern_type.801 = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %T.0d1: %Negate_where.type.871 = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.db1: type = facet_access_type %T.0d1 [symbolic]
// CHECK:STDOUT:   %pattern_type.398aef.2: type = pattern_type %T.as_type.db1 [symbolic]
// CHECK:STDOUT:   %x.param_patt.5d8: %pattern_type.398aef.2 = value_param_pattern [symbolic]
// CHECK:STDOUT:   %x.patt.e11: %pattern_type.398aef.2 = at_binding_pattern x, %x.param_patt.5d8 [symbolic]
// CHECK:STDOUT:   %Negate.lookup_impl_witness.c40cda.2: <witness> = lookup_impl_witness %T.0d1, @Negate [symbolic]
// CHECK:STDOUT:   %Negate.facet: %Negate.type = facet_value %T.as_type.db1, (%Negate.lookup_impl_witness.c40cda.2) [symbolic]
// CHECK:STDOUT:   %impl.elem0.9a6046.2: type = impl_witness_access %Negate.lookup_impl_witness.c40cda.2, element0 [symbolic]
// CHECK:STDOUT:   %Negate.WithSelf.Op.type.007: type = fn_type @Negate.WithSelf.Op, @Negate.WithSelf(%Negate.facet) [symbolic]
// CHECK:STDOUT:   %.a2a: type = fn_type_with_self_type %Negate.WithSelf.Op.type.007, %Negate.facet [symbolic]
// CHECK:STDOUT:   %impl.elem1.b9d: %.a2a = impl_witness_access %Negate.lookup_impl_witness.c40cda.2, element1 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.847: <specific function> = specific_impl_function %impl.elem1.b9d, @Negate.WithSelf.Op(%Negate.facet) [symbolic]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.7e7: <witness> = lookup_impl_witness %impl.elem0.9a6046.2, @Destroy [symbolic]
// CHECK:STDOUT:   %Destroy.facet.4da: %Destroy.type = facet_value %impl.elem0.9a6046.2, (%Destroy.lookup_impl_witness.7e7) [symbolic]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ff0: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet.4da) [symbolic]
// CHECK:STDOUT:   %.3f0: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.ff0, %Destroy.facet.4da [symbolic]
// CHECK:STDOUT:   %impl.elem0.22c: %.3f0 = impl_witness_access %Destroy.lookup_impl_witness.7e7, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.72a: <specific function> = specific_impl_function %impl.elem0.22c, @Destroy.WithSelf.Op(%Destroy.facet.4da) [symbolic]
// CHECK:STDOUT:   %Inc.type: type = facet_type <@Inc> [concrete]
// CHECK:STDOUT:   %pattern_type.8c1: type = pattern_type %Inc.type [concrete]
// CHECK:STDOUT:   %T.patt.eab: %pattern_type.8c1 = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %T.8f3: %Inc.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.232: type = facet_access_type %T.8f3 [symbolic]
// CHECK:STDOUT:   %pattern_type.427bfc.2: type = pattern_type %T.as_type.232 [symbolic]
// CHECK:STDOUT:   %x.patt.a75: %pattern_type.427bfc.2 = ref_binding_pattern x [symbolic]
// CHECK:STDOUT:   %x.param_patt.5da: %pattern_type.427bfc.2 = var_param_pattern %x.patt.a75 [symbolic]
// CHECK:STDOUT:   %Inc.lookup_impl_witness: <witness> = lookup_impl_witness %T.8f3, @Inc [symbolic]
// CHECK:STDOUT:   %Inc.WithSelf.Op.type.e10ef6.2: type = fn_type @Inc.WithSelf.Op, @Inc.WithSelf(%T.8f3) [symbolic]
// CHECK:STDOUT:   %.a08: type = fn_type_with_self_type %Inc.WithSelf.Op.type.e10ef6.2, %T.8f3 [symbolic]
// CHECK:STDOUT:   %impl.elem0.b92: %.a08 = impl_witness_access %Inc.lookup_impl_witness, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.855: <specific function> = specific_impl_function %impl.elem0.b92, @Inc.WithSelf.Op(%T.8f3) [symbolic]
// CHECK:STDOUT:   %Dec.type: type = facet_type <@Dec> [concrete]
// CHECK:STDOUT:   %pattern_type.72a: type = pattern_type %Dec.type [concrete]
// CHECK:STDOUT:   %T.patt.de9: %pattern_type.72a = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %T.e8e: %Dec.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.c32: type = facet_access_type %T.e8e [symbolic]
// CHECK:STDOUT:   %pattern_type.2754f0.2: type = pattern_type %T.as_type.c32 [symbolic]
// CHECK:STDOUT:   %x.patt.08b: %pattern_type.2754f0.2 = ref_binding_pattern x [symbolic]
// CHECK:STDOUT:   %x.param_patt.678: %pattern_type.2754f0.2 = var_param_pattern %x.patt.08b [symbolic]
// CHECK:STDOUT:   %Dec.lookup_impl_witness: <witness> = lookup_impl_witness %T.e8e, @Dec [symbolic]
// CHECK:STDOUT:   %Dec.WithSelf.Op.type.bfa830.2: type = fn_type @Dec.WithSelf.Op, @Dec.WithSelf(%T.e8e) [symbolic]
// CHECK:STDOUT:   %.aae: type = fn_type_with_self_type %Dec.WithSelf.Op.type.bfa830.2, %T.e8e [symbolic]
// CHECK:STDOUT:   %impl.elem0.950: %.aae = impl_witness_access %Dec.lookup_impl_witness, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.539: <specific function> = specific_impl_function %impl.elem0.950, @Dec.WithSelf.Op(%T.e8e) [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.5a1: @AddWith.WithSelf.%AddWith.assoc_type (%AddWith.assoc_type.588227.1) = import_ref Core//prelude/operators/arithmetic, loc{{\d+_\d+}}, loaded [symbolic = @AddWith.WithSelf.%assoc1 (constants.%assoc1.57d)]
// CHECK:STDOUT:   %Core.import_ref.3cf: @AddWith.WithSelf.%AddWith.WithSelf.Op.type (%AddWith.WithSelf.Op.type.28e0e0.1) = import_ref Core//prelude/operators/arithmetic, loc{{\d+_\d+}}, loaded [symbolic = @AddWith.WithSelf.%AddWith.WithSelf.Op (constants.%AddWith.WithSelf.Op.eab9a0.1)]
// CHECK:STDOUT:   %Core.import_ref.91f: type = import_ref Core//prelude/operators/arithmetic, loc{{\d+_\d+}}, loaded [concrete = %Result.9a4]
// CHECK:STDOUT:   %Result.9a4: type = assoc_const_decl @Result.1 [concrete] {}
// CHECK:STDOUT:   %Core.import_ref.976 = import_ref Core//prelude/operators/arithmetic, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.bdd: @AddAssignWith.WithSelf.%AddAssignWith.assoc_type (%AddAssignWith.assoc_type.8ddaf4.1) = import_ref Core//prelude/operators/arithmetic, loc{{\d+_\d+}}, loaded [symbolic = @AddAssignWith.WithSelf.%assoc0 (constants.%assoc0.86d)]
// CHECK:STDOUT:   %Core.import_ref.3b2: @AddAssignWith.WithSelf.%AddAssignWith.WithSelf.Op.type (%AddAssignWith.WithSelf.Op.type.cd574e.1) = import_ref Core//prelude/operators/arithmetic, loc{{\d+_\d+}}, loaded [symbolic = @AddAssignWith.WithSelf.%AddAssignWith.WithSelf.Op (constants.%AddAssignWith.WithSelf.Op.106889.1)]
// CHECK:STDOUT:   %Core.import_ref.87e = import_ref Core//prelude/operators/arithmetic, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.0f3: @SubWith.WithSelf.%SubWith.assoc_type (%SubWith.assoc_type.b0beca.1) = import_ref Core//prelude/operators/arithmetic, loc{{\d+_\d+}}, loaded [symbolic = @SubWith.WithSelf.%assoc1 (constants.%assoc1.178)]
// CHECK:STDOUT:   %Core.import_ref.3a4: @SubWith.WithSelf.%SubWith.WithSelf.Op.type (%SubWith.WithSelf.Op.type.68d038.1) = import_ref Core//prelude/operators/arithmetic, loc{{\d+_\d+}}, loaded [symbolic = @SubWith.WithSelf.%SubWith.WithSelf.Op (constants.%SubWith.WithSelf.Op.97ab45.1)]
// CHECK:STDOUT:   %Core.import_ref.190: type = import_ref Core//prelude/operators/arithmetic, loc{{\d+_\d+}}, loaded [concrete = %Result.7bd]
// CHECK:STDOUT:   %Result.7bd: type = assoc_const_decl @Result.2 [concrete] {}
// CHECK:STDOUT:   %Core.import_ref.b59 = import_ref Core//prelude/operators/arithmetic, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.509: @SubAssignWith.WithSelf.%SubAssignWith.assoc_type (%SubAssignWith.assoc_type.6e3877.1) = import_ref Core//prelude/operators/arithmetic, loc{{\d+_\d+}}, loaded [symbolic = @SubAssignWith.WithSelf.%assoc0 (constants.%assoc0.b8e)]
// CHECK:STDOUT:   %Core.import_ref.8f2: @SubAssignWith.WithSelf.%SubAssignWith.WithSelf.Op.type (%SubAssignWith.WithSelf.Op.type.42f7cc.1) = import_ref Core//prelude/operators/arithmetic, loc{{\d+_\d+}}, loaded [symbolic = @SubAssignWith.WithSelf.%SubAssignWith.WithSelf.Op (constants.%SubAssignWith.WithSelf.Op.25af21.1)]
// CHECK:STDOUT:   %Core.import_ref.e7f = import_ref Core//prelude/operators/arithmetic, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.8c1: @MulWith.WithSelf.%MulWith.assoc_type (%MulWith.assoc_type.7b2892.1) = import_ref Core//prelude/operators/arithmetic, loc{{\d+_\d+}}, loaded [symbolic = @MulWith.WithSelf.%assoc1 (constants.%assoc1.8f7)]
// CHECK:STDOUT:   %Core.import_ref.5a0: @MulWith.WithSelf.%MulWith.WithSelf.Op.type (%MulWith.WithSelf.Op.type.176210.1) = import_ref Core//prelude/operators/arithmetic, loc{{\d+_\d+}}, loaded [symbolic = @MulWith.WithSelf.%MulWith.WithSelf.Op (constants.%MulWith.WithSelf.Op.a7516d.1)]
// CHECK:STDOUT:   %Core.import_ref.59c: type = import_ref Core//prelude/operators/arithmetic, loc{{\d+_\d+}}, loaded [concrete = %Result.a64]
// CHECK:STDOUT:   %Result.a64: type = assoc_const_decl @Result.3 [concrete] {}
// CHECK:STDOUT:   %Core.import_ref.955 = import_ref Core//prelude/operators/arithmetic, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.b6e: @MulAssignWith.WithSelf.%MulAssignWith.assoc_type (%MulAssignWith.assoc_type.9aa18f.1) = import_ref Core//prelude/operators/arithmetic, loc{{\d+_\d+}}, loaded [symbolic = @MulAssignWith.WithSelf.%assoc0 (constants.%assoc0.2ec)]
// CHECK:STDOUT:   %Core.import_ref.ba3: @MulAssignWith.WithSelf.%MulAssignWith.WithSelf.Op.type (%MulAssignWith.WithSelf.Op.type.5df5bd.1) = import_ref Core//prelude/operators/arithmetic, loc{{\d+_\d+}}, loaded [symbolic = @MulAssignWith.WithSelf.%MulAssignWith.WithSelf.Op (constants.%MulAssignWith.WithSelf.Op.ee0eda.1)]
// CHECK:STDOUT:   %Core.import_ref.408 = import_ref Core//prelude/operators/arithmetic, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.ca2: @DivWith.WithSelf.%DivWith.assoc_type (%DivWith.assoc_type.d8b290.1) = import_ref Core//prelude/operators/arithmetic, loc{{\d+_\d+}}, loaded [symbolic = @DivWith.WithSelf.%assoc1 (constants.%assoc1.7b1)]
// CHECK:STDOUT:   %Core.import_ref.784: @DivWith.WithSelf.%DivWith.WithSelf.Op.type (%DivWith.WithSelf.Op.type.0f05c0.1) = import_ref Core//prelude/operators/arithmetic, loc{{\d+_\d+}}, loaded [symbolic = @DivWith.WithSelf.%DivWith.WithSelf.Op (constants.%DivWith.WithSelf.Op.51e40b.1)]
// CHECK:STDOUT:   %Core.import_ref.dee: type = import_ref Core//prelude/operators/arithmetic, loc{{\d+_\d+}}, loaded [concrete = %Result.cb0]
// CHECK:STDOUT:   %Result.cb0: type = assoc_const_decl @Result.4 [concrete] {}
// CHECK:STDOUT:   %Core.import_ref.8b6 = import_ref Core//prelude/operators/arithmetic, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.988: @DivAssignWith.WithSelf.%DivAssignWith.assoc_type (%DivAssignWith.assoc_type.bb68ec.1) = import_ref Core//prelude/operators/arithmetic, loc{{\d+_\d+}}, loaded [symbolic = @DivAssignWith.WithSelf.%assoc0 (constants.%assoc0.de1)]
// CHECK:STDOUT:   %Core.import_ref.a5d: @DivAssignWith.WithSelf.%DivAssignWith.WithSelf.Op.type (%DivAssignWith.WithSelf.Op.type.ff5775.1) = import_ref Core//prelude/operators/arithmetic, loc{{\d+_\d+}}, loaded [symbolic = @DivAssignWith.WithSelf.%DivAssignWith.WithSelf.Op (constants.%DivAssignWith.WithSelf.Op.91e0ff.1)]
// CHECK:STDOUT:   %Core.import_ref.df2 = import_ref Core//prelude/operators/arithmetic, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.ffe: @ModWith.WithSelf.%ModWith.assoc_type (%ModWith.assoc_type.316ae9.1) = import_ref Core//prelude/operators/arithmetic, loc{{\d+_\d+}}, loaded [symbolic = @ModWith.WithSelf.%assoc1 (constants.%assoc1.132)]
// CHECK:STDOUT:   %Core.import_ref.b56: @ModWith.WithSelf.%ModWith.WithSelf.Op.type (%ModWith.WithSelf.Op.type.d147b5.1) = import_ref Core//prelude/operators/arithmetic, loc{{\d+_\d+}}, loaded [symbolic = @ModWith.WithSelf.%ModWith.WithSelf.Op (constants.%ModWith.WithSelf.Op.9e4026.1)]
// CHECK:STDOUT:   %Core.import_ref.61a: type = import_ref Core//prelude/operators/arithmetic, loc{{\d+_\d+}}, loaded [concrete = %Result.803]
// CHECK:STDOUT:   %Result.803: type = assoc_const_decl @Result.5 [concrete] {}
// CHECK:STDOUT:   %Core.import_ref.33c = import_ref Core//prelude/operators/arithmetic, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.3ff: @ModAssignWith.WithSelf.%ModAssignWith.assoc_type (%ModAssignWith.assoc_type.bc79ec.1) = import_ref Core//prelude/operators/arithmetic, loc{{\d+_\d+}}, loaded [symbolic = @ModAssignWith.WithSelf.%assoc0 (constants.%assoc0.0f3)]
// CHECK:STDOUT:   %Core.import_ref.cf6: @ModAssignWith.WithSelf.%ModAssignWith.WithSelf.Op.type (%ModAssignWith.WithSelf.Op.type.ec4109.1) = import_ref Core//prelude/operators/arithmetic, loc{{\d+_\d+}}, loaded [symbolic = @ModAssignWith.WithSelf.%ModAssignWith.WithSelf.Op (constants.%ModAssignWith.WithSelf.Op.ac45fc.1)]
// CHECK:STDOUT:   %Core.import_ref.d29 = import_ref Core//prelude/operators/arithmetic, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @TestAddWith(%U.loc15_4.2: type, %T.loc16_4.2: @TestAddWith.%AddWith_where.type (%AddWith_where.type.7e9)) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %assoc1: @TestAddWith.%AddWith.assoc_type (%AddWith.assoc_type.588227.2) = assoc_entity element1, imports.%Core.import_ref.3cf [symbolic = %assoc1 (constants.%assoc1.c411ca.2)]
// CHECK:STDOUT:   %AddWith.lookup_impl_witness.loc19: <witness> = lookup_impl_witness %T.loc16_4.1, @AddWith, @AddWith(%U.loc15_4.1) [symbolic = %AddWith.lookup_impl_witness.loc19 (constants.%AddWith.lookup_impl_witness.73583b.2)]
// CHECK:STDOUT:   %AddWith.facet: @TestAddWith.%AddWith.type.loc16_21.1 (%AddWith.type.7e967c.2) = facet_value %T.as_type.loc17_6.1, (%AddWith.lookup_impl_witness.loc19) [symbolic = %AddWith.facet (constants.%AddWith.facet)]
// CHECK:STDOUT:   %AddWith.WithSelf.Op.type: type = fn_type @AddWith.WithSelf.Op, @AddWith.WithSelf(%U.loc15_4.1, %AddWith.facet) [symbolic = %AddWith.WithSelf.Op.type (constants.%AddWith.WithSelf.Op.type.94c)]
// CHECK:STDOUT:   %.loc19_5.6: type = fn_type_with_self_type %AddWith.WithSelf.Op.type, %AddWith.facet [symbolic = %.loc19_5.6 (constants.%.e17)]
// CHECK:STDOUT:   %impl.elem1.loc19_5.2: @TestAddWith.%.loc19_5.6 (%.e17) = impl_witness_access %AddWith.lookup_impl_witness.loc19, element1 [symbolic = %impl.elem1.loc19_5.2 (constants.%impl.elem1.8d3)]
// CHECK:STDOUT:   %specific_impl_fn.loc19_5.3: <specific function> = specific_impl_function %impl.elem1.loc19_5.2, @AddWith.WithSelf.Op(%U.loc15_4.1, %AddWith.facet) [symbolic = %specific_impl_fn.loc19_5.3 (constants.%specific_impl_fn.b33)]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %Destroy.lookup_impl_witness: <witness> = lookup_impl_witness %impl.elem0.1, @Destroy [symbolic = %Destroy.lookup_impl_witness (constants.%Destroy.lookup_impl_witness.9b9)]
// CHECK:STDOUT:   %Destroy.facet.loc19_5.3: %Destroy.type = facet_value %impl.elem0.1, (%Destroy.lookup_impl_witness) [symbolic = %Destroy.facet.loc19_5.3 (constants.%Destroy.facet.b81)]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet.loc19_5.3) [symbolic = %Destroy.WithSelf.Op.type (constants.%Destroy.WithSelf.Op.type.b80)]
// CHECK:STDOUT:   %.loc19_5.7: type = fn_type_with_self_type %Destroy.WithSelf.Op.type, %Destroy.facet.loc19_5.3 [symbolic = %.loc19_5.7 (constants.%.671)]
// CHECK:STDOUT:   %impl.elem0.loc19_5.2: @TestAddWith.%.loc19_5.7 (%.671) = impl_witness_access %Destroy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc19_5.2 (constants.%impl.elem0.497)]
// CHECK:STDOUT:   %specific_impl_fn.loc19_5.4: <specific function> = specific_impl_function %impl.elem0.loc19_5.2, @Destroy.WithSelf.Op(%Destroy.facet.loc19_5.3) [symbolic = %specific_impl_fn.loc19_5.4 (constants.%specific_impl_fn.3b7)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: @TestAddWith.%T.as_type.loc17_6.1 (%T.as_type.331), %y.param: @TestAddWith.%U.loc15_4.1 (%U)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %x.ref: @TestAddWith.%T.as_type.loc17_6.1 (%T.as_type.331) = name_ref x, %x
// CHECK:STDOUT:     %y.ref: @TestAddWith.%U.loc15_4.1 (%U) = name_ref y, %y
// CHECK:STDOUT:     %AddWith.type.loc19: type = facet_type <@AddWith, @AddWith(constants.%U)> [symbolic = %AddWith.type.loc16_21.1 (constants.%AddWith.type.7e967c.2)]
// CHECK:STDOUT:     %.loc19_5.1: @TestAddWith.%AddWith.assoc_type (%AddWith.assoc_type.588227.2) = specific_constant imports.%Core.import_ref.5a1, @AddWith.WithSelf(constants.%U, constants.%Self.96aae2.1) [symbolic = %assoc1 (constants.%assoc1.c411ca.2)]
// CHECK:STDOUT:     %Op.ref: @TestAddWith.%AddWith.assoc_type (%AddWith.assoc_type.588227.2) = name_ref Op, %.loc19_5.1 [symbolic = %assoc1 (constants.%assoc1.c411ca.2)]
// CHECK:STDOUT:     %impl.elem1.loc19_5.1: @TestAddWith.%.loc19_5.6 (%.e17) = impl_witness_access constants.%AddWith.lookup_impl_witness.73583b.2, element1 [symbolic = %impl.elem1.loc19_5.2 (constants.%impl.elem1.8d3)]
// CHECK:STDOUT:     %bound_method.loc19_5.1: <bound method> = bound_method %x.ref, %impl.elem1.loc19_5.1
// CHECK:STDOUT:     %specific_impl_fn.loc19_5.1: <specific function> = specific_impl_function %impl.elem1.loc19_5.1, @AddWith.WithSelf.Op(constants.%U, constants.%AddWith.facet) [symbolic = %specific_impl_fn.loc19_5.3 (constants.%specific_impl_fn.b33)]
// CHECK:STDOUT:     %bound_method.loc19_5.2: <bound method> = bound_method %x.ref, %specific_impl_fn.loc19_5.1
// CHECK:STDOUT:     %.loc19_5.2: ref @TestAddWith.%impl.elem0.1 (%impl.elem0.237f5f.2) = temporary_storage
// CHECK:STDOUT:     %AddWith.WithSelf.Op.call: init @TestAddWith.%impl.elem0.1 (%impl.elem0.237f5f.2) to %.loc19_5.2 = call %bound_method.loc19_5.2(%x.ref, %y.ref)
// CHECK:STDOUT:     %.loc19_5.3: ref @TestAddWith.%impl.elem0.1 (%impl.elem0.237f5f.2) = temporary %.loc19_5.2, %AddWith.WithSelf.Op.call
// CHECK:STDOUT:     %impl.elem0.loc19_5.1: @TestAddWith.%.loc19_5.7 (%.671) = impl_witness_access constants.%Destroy.lookup_impl_witness.9b9, element0 [symbolic = %impl.elem0.loc19_5.2 (constants.%impl.elem0.497)]
// CHECK:STDOUT:     %bound_method.loc19_5.3: <bound method> = bound_method %.loc19_5.3, %impl.elem0.loc19_5.1
// CHECK:STDOUT:     %Destroy.facet.loc19_5.1: %Destroy.type = facet_value constants.%impl.elem0.237f5f.2, (constants.%Destroy.lookup_impl_witness.9b9) [symbolic = %Destroy.facet.loc19_5.3 (constants.%Destroy.facet.b81)]
// CHECK:STDOUT:     %.loc19_5.4: %Destroy.type = converted constants.%impl.elem0.237f5f.2, %Destroy.facet.loc19_5.1 [symbolic = %Destroy.facet.loc19_5.3 (constants.%Destroy.facet.b81)]
// CHECK:STDOUT:     %Destroy.facet.loc19_5.2: %Destroy.type = facet_value constants.%impl.elem0.237f5f.2, (constants.%Destroy.lookup_impl_witness.9b9) [symbolic = %Destroy.facet.loc19_5.3 (constants.%Destroy.facet.b81)]
// CHECK:STDOUT:     %.loc19_5.5: %Destroy.type = converted constants.%impl.elem0.237f5f.2, %Destroy.facet.loc19_5.2 [symbolic = %Destroy.facet.loc19_5.3 (constants.%Destroy.facet.b81)]
// CHECK:STDOUT:     %specific_impl_fn.loc19_5.2: <specific function> = specific_impl_function %impl.elem0.loc19_5.1, @Destroy.WithSelf.Op(constants.%Destroy.facet.b81) [symbolic = %specific_impl_fn.loc19_5.4 (constants.%specific_impl_fn.3b7)]
// CHECK:STDOUT:     %bound_method.loc19_5.4: <bound method> = bound_method %.loc19_5.3, %specific_impl_fn.loc19_5.2
// CHECK:STDOUT:     %Destroy.WithSelf.Op.call: init %empty_tuple.type = call %bound_method.loc19_5.4(%.loc19_5.3)
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @TestAddAssignWith(%U.loc23_23.2: type, %T.loc23_33.2: @TestAddAssignWith.%AddAssignWith.type.loc23_56.1 (%AddAssignWith.type.2c8890.2)) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %require_complete.loc25: <witness> = require_complete_type %AddAssignWith.type.loc23_56.1 [symbolic = %require_complete.loc25 (constants.%require_complete.2a8)]
// CHECK:STDOUT:   %AddAssignWith.assoc_type: type = assoc_entity_type @AddAssignWith, @AddAssignWith(%U.loc23_23.1) [symbolic = %AddAssignWith.assoc_type (constants.%AddAssignWith.assoc_type.8ddaf4.2)]
// CHECK:STDOUT:   %assoc0: @TestAddAssignWith.%AddAssignWith.assoc_type (%AddAssignWith.assoc_type.8ddaf4.2) = assoc_entity element0, imports.%Core.import_ref.3b2 [symbolic = %assoc0 (constants.%assoc0.8c5739.2)]
// CHECK:STDOUT:   %AddAssignWith.WithSelf.Op.type: type = fn_type @AddAssignWith.WithSelf.Op, @AddAssignWith.WithSelf(%U.loc23_23.1, %T.loc23_33.1) [symbolic = %AddAssignWith.WithSelf.Op.type (constants.%AddAssignWith.WithSelf.Op.type.cd574e.3)]
// CHECK:STDOUT:   %.loc25_5.2: type = fn_type_with_self_type %AddAssignWith.WithSelf.Op.type, %T.loc23_33.1 [symbolic = %.loc25_5.2 (constants.%.9b1)]
// CHECK:STDOUT:   %AddAssignWith.lookup_impl_witness: <witness> = lookup_impl_witness %T.loc23_33.1, @AddAssignWith, @AddAssignWith(%U.loc23_23.1) [symbolic = %AddAssignWith.lookup_impl_witness (constants.%AddAssignWith.lookup_impl_witness)]
// CHECK:STDOUT:   %impl.elem0.loc25_5.2: @TestAddAssignWith.%.loc25_5.2 (%.9b1) = impl_witness_access %AddAssignWith.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc25_5.2 (constants.%impl.elem0.985)]
// CHECK:STDOUT:   %specific_impl_fn.loc25_5.2: <specific function> = specific_impl_function %impl.elem0.loc25_5.2, @AddAssignWith.WithSelf.Op(%U.loc23_23.1, %T.loc23_33.1) [symbolic = %specific_impl_fn.loc25_5.2 (constants.%specific_impl_fn.d36)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: ref @TestAddAssignWith.%T.as_type.loc23_66.1 (%T.as_type.50c), %y.param: @TestAddAssignWith.%U.loc23_23.1 (%U)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %x.ref: ref @TestAddAssignWith.%T.as_type.loc23_66.1 (%T.as_type.50c) = name_ref x, %x
// CHECK:STDOUT:     %y.ref: @TestAddAssignWith.%U.loc23_23.1 (%U) = name_ref y, %y
// CHECK:STDOUT:     %AddAssignWith.type.loc25: type = facet_type <@AddAssignWith, @AddAssignWith(constants.%U)> [symbolic = %AddAssignWith.type.loc23_56.1 (constants.%AddAssignWith.type.2c8890.2)]
// CHECK:STDOUT:     %.loc25_5.1: @TestAddAssignWith.%AddAssignWith.assoc_type (%AddAssignWith.assoc_type.8ddaf4.2) = specific_constant imports.%Core.import_ref.bdd, @AddAssignWith.WithSelf(constants.%U, constants.%Self.792732.1) [symbolic = %assoc0 (constants.%assoc0.8c5739.2)]
// CHECK:STDOUT:     %Op.ref: @TestAddAssignWith.%AddAssignWith.assoc_type (%AddAssignWith.assoc_type.8ddaf4.2) = name_ref Op, %.loc25_5.1 [symbolic = %assoc0 (constants.%assoc0.8c5739.2)]
// CHECK:STDOUT:     %impl.elem0.loc25_5.1: @TestAddAssignWith.%.loc25_5.2 (%.9b1) = impl_witness_access constants.%AddAssignWith.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc25_5.2 (constants.%impl.elem0.985)]
// CHECK:STDOUT:     %bound_method.loc25_5.1: <bound method> = bound_method %x.ref, %impl.elem0.loc25_5.1
// CHECK:STDOUT:     %specific_impl_fn.loc25_5.1: <specific function> = specific_impl_function %impl.elem0.loc25_5.1, @AddAssignWith.WithSelf.Op(constants.%U, constants.%T.792) [symbolic = %specific_impl_fn.loc25_5.2 (constants.%specific_impl_fn.d36)]
// CHECK:STDOUT:     %bound_method.loc25_5.2: <bound method> = bound_method %x.ref, %specific_impl_fn.loc25_5.1
// CHECK:STDOUT:     %AddAssignWith.WithSelf.Op.call: init %empty_tuple.type = call %bound_method.loc25_5.2(%x.ref, %y.ref)
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @TestSubWith(%U.loc30_4.2: type, %T.loc31_4.2: @TestSubWith.%SubWith_where.type (%SubWith_where.type.ddc)) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %assoc1: @TestSubWith.%SubWith.assoc_type (%SubWith.assoc_type.b0beca.2) = assoc_entity element1, imports.%Core.import_ref.3a4 [symbolic = %assoc1 (constants.%assoc1.e3bf87.2)]
// CHECK:STDOUT:   %SubWith.lookup_impl_witness.loc34: <witness> = lookup_impl_witness %T.loc31_4.1, @SubWith, @SubWith(%U.loc30_4.1) [symbolic = %SubWith.lookup_impl_witness.loc34 (constants.%SubWith.lookup_impl_witness.a9fbd3.2)]
// CHECK:STDOUT:   %SubWith.facet: @TestSubWith.%SubWith.type.loc31_21.1 (%SubWith.type.ddc625.2) = facet_value %T.as_type.loc32_6.1, (%SubWith.lookup_impl_witness.loc34) [symbolic = %SubWith.facet (constants.%SubWith.facet)]
// CHECK:STDOUT:   %SubWith.WithSelf.Op.type: type = fn_type @SubWith.WithSelf.Op, @SubWith.WithSelf(%U.loc30_4.1, %SubWith.facet) [symbolic = %SubWith.WithSelf.Op.type (constants.%SubWith.WithSelf.Op.type.b7d)]
// CHECK:STDOUT:   %.loc34_5.6: type = fn_type_with_self_type %SubWith.WithSelf.Op.type, %SubWith.facet [symbolic = %.loc34_5.6 (constants.%.c28)]
// CHECK:STDOUT:   %impl.elem1.loc34_5.2: @TestSubWith.%.loc34_5.6 (%.c28) = impl_witness_access %SubWith.lookup_impl_witness.loc34, element1 [symbolic = %impl.elem1.loc34_5.2 (constants.%impl.elem1.df0)]
// CHECK:STDOUT:   %specific_impl_fn.loc34_5.3: <specific function> = specific_impl_function %impl.elem1.loc34_5.2, @SubWith.WithSelf.Op(%U.loc30_4.1, %SubWith.facet) [symbolic = %specific_impl_fn.loc34_5.3 (constants.%specific_impl_fn.882)]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %Destroy.lookup_impl_witness: <witness> = lookup_impl_witness %impl.elem0.1, @Destroy [symbolic = %Destroy.lookup_impl_witness (constants.%Destroy.lookup_impl_witness.4ac)]
// CHECK:STDOUT:   %Destroy.facet.loc34_5.3: %Destroy.type = facet_value %impl.elem0.1, (%Destroy.lookup_impl_witness) [symbolic = %Destroy.facet.loc34_5.3 (constants.%Destroy.facet.622)]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet.loc34_5.3) [symbolic = %Destroy.WithSelf.Op.type (constants.%Destroy.WithSelf.Op.type.4d5)]
// CHECK:STDOUT:   %.loc34_5.7: type = fn_type_with_self_type %Destroy.WithSelf.Op.type, %Destroy.facet.loc34_5.3 [symbolic = %.loc34_5.7 (constants.%.14c)]
// CHECK:STDOUT:   %impl.elem0.loc34_5.2: @TestSubWith.%.loc34_5.7 (%.14c) = impl_witness_access %Destroy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc34_5.2 (constants.%impl.elem0.aa8)]
// CHECK:STDOUT:   %specific_impl_fn.loc34_5.4: <specific function> = specific_impl_function %impl.elem0.loc34_5.2, @Destroy.WithSelf.Op(%Destroy.facet.loc34_5.3) [symbolic = %specific_impl_fn.loc34_5.4 (constants.%specific_impl_fn.a72)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: @TestSubWith.%T.as_type.loc32_6.1 (%T.as_type.449), %y.param: @TestSubWith.%U.loc30_4.1 (%U)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %x.ref: @TestSubWith.%T.as_type.loc32_6.1 (%T.as_type.449) = name_ref x, %x
// CHECK:STDOUT:     %y.ref: @TestSubWith.%U.loc30_4.1 (%U) = name_ref y, %y
// CHECK:STDOUT:     %SubWith.type.loc34: type = facet_type <@SubWith, @SubWith(constants.%U)> [symbolic = %SubWith.type.loc31_21.1 (constants.%SubWith.type.ddc625.2)]
// CHECK:STDOUT:     %.loc34_5.1: @TestSubWith.%SubWith.assoc_type (%SubWith.assoc_type.b0beca.2) = specific_constant imports.%Core.import_ref.0f3, @SubWith.WithSelf(constants.%U, constants.%Self.d6b75b.1) [symbolic = %assoc1 (constants.%assoc1.e3bf87.2)]
// CHECK:STDOUT:     %Op.ref: @TestSubWith.%SubWith.assoc_type (%SubWith.assoc_type.b0beca.2) = name_ref Op, %.loc34_5.1 [symbolic = %assoc1 (constants.%assoc1.e3bf87.2)]
// CHECK:STDOUT:     %impl.elem1.loc34_5.1: @TestSubWith.%.loc34_5.6 (%.c28) = impl_witness_access constants.%SubWith.lookup_impl_witness.a9fbd3.2, element1 [symbolic = %impl.elem1.loc34_5.2 (constants.%impl.elem1.df0)]
// CHECK:STDOUT:     %bound_method.loc34_5.1: <bound method> = bound_method %x.ref, %impl.elem1.loc34_5.1
// CHECK:STDOUT:     %specific_impl_fn.loc34_5.1: <specific function> = specific_impl_function %impl.elem1.loc34_5.1, @SubWith.WithSelf.Op(constants.%U, constants.%SubWith.facet) [symbolic = %specific_impl_fn.loc34_5.3 (constants.%specific_impl_fn.882)]
// CHECK:STDOUT:     %bound_method.loc34_5.2: <bound method> = bound_method %x.ref, %specific_impl_fn.loc34_5.1
// CHECK:STDOUT:     %.loc34_5.2: ref @TestSubWith.%impl.elem0.1 (%impl.elem0.1583c7.2) = temporary_storage
// CHECK:STDOUT:     %SubWith.WithSelf.Op.call: init @TestSubWith.%impl.elem0.1 (%impl.elem0.1583c7.2) to %.loc34_5.2 = call %bound_method.loc34_5.2(%x.ref, %y.ref)
// CHECK:STDOUT:     %.loc34_5.3: ref @TestSubWith.%impl.elem0.1 (%impl.elem0.1583c7.2) = temporary %.loc34_5.2, %SubWith.WithSelf.Op.call
// CHECK:STDOUT:     %impl.elem0.loc34_5.1: @TestSubWith.%.loc34_5.7 (%.14c) = impl_witness_access constants.%Destroy.lookup_impl_witness.4ac, element0 [symbolic = %impl.elem0.loc34_5.2 (constants.%impl.elem0.aa8)]
// CHECK:STDOUT:     %bound_method.loc34_5.3: <bound method> = bound_method %.loc34_5.3, %impl.elem0.loc34_5.1
// CHECK:STDOUT:     %Destroy.facet.loc34_5.1: %Destroy.type = facet_value constants.%impl.elem0.1583c7.2, (constants.%Destroy.lookup_impl_witness.4ac) [symbolic = %Destroy.facet.loc34_5.3 (constants.%Destroy.facet.622)]
// CHECK:STDOUT:     %.loc34_5.4: %Destroy.type = converted constants.%impl.elem0.1583c7.2, %Destroy.facet.loc34_5.1 [symbolic = %Destroy.facet.loc34_5.3 (constants.%Destroy.facet.622)]
// CHECK:STDOUT:     %Destroy.facet.loc34_5.2: %Destroy.type = facet_value constants.%impl.elem0.1583c7.2, (constants.%Destroy.lookup_impl_witness.4ac) [symbolic = %Destroy.facet.loc34_5.3 (constants.%Destroy.facet.622)]
// CHECK:STDOUT:     %.loc34_5.5: %Destroy.type = converted constants.%impl.elem0.1583c7.2, %Destroy.facet.loc34_5.2 [symbolic = %Destroy.facet.loc34_5.3 (constants.%Destroy.facet.622)]
// CHECK:STDOUT:     %specific_impl_fn.loc34_5.2: <specific function> = specific_impl_function %impl.elem0.loc34_5.1, @Destroy.WithSelf.Op(constants.%Destroy.facet.622) [symbolic = %specific_impl_fn.loc34_5.4 (constants.%specific_impl_fn.a72)]
// CHECK:STDOUT:     %bound_method.loc34_5.4: <bound method> = bound_method %.loc34_5.3, %specific_impl_fn.loc34_5.2
// CHECK:STDOUT:     %Destroy.WithSelf.Op.call: init %empty_tuple.type = call %bound_method.loc34_5.4(%.loc34_5.3)
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @TestSubAssignWith(%U.loc38_23.2: type, %T.loc38_33.2: @TestSubAssignWith.%SubAssignWith.type.loc38_56.1 (%SubAssignWith.type.7e53cf.2)) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %require_complete.loc40: <witness> = require_complete_type %SubAssignWith.type.loc38_56.1 [symbolic = %require_complete.loc40 (constants.%require_complete.862)]
// CHECK:STDOUT:   %SubAssignWith.assoc_type: type = assoc_entity_type @SubAssignWith, @SubAssignWith(%U.loc38_23.1) [symbolic = %SubAssignWith.assoc_type (constants.%SubAssignWith.assoc_type.6e3877.2)]
// CHECK:STDOUT:   %assoc0: @TestSubAssignWith.%SubAssignWith.assoc_type (%SubAssignWith.assoc_type.6e3877.2) = assoc_entity element0, imports.%Core.import_ref.8f2 [symbolic = %assoc0 (constants.%assoc0.701e5f.2)]
// CHECK:STDOUT:   %SubAssignWith.WithSelf.Op.type: type = fn_type @SubAssignWith.WithSelf.Op, @SubAssignWith.WithSelf(%U.loc38_23.1, %T.loc38_33.1) [symbolic = %SubAssignWith.WithSelf.Op.type (constants.%SubAssignWith.WithSelf.Op.type.42f7cc.3)]
// CHECK:STDOUT:   %.loc40_5.2: type = fn_type_with_self_type %SubAssignWith.WithSelf.Op.type, %T.loc38_33.1 [symbolic = %.loc40_5.2 (constants.%.e56)]
// CHECK:STDOUT:   %SubAssignWith.lookup_impl_witness: <witness> = lookup_impl_witness %T.loc38_33.1, @SubAssignWith, @SubAssignWith(%U.loc38_23.1) [symbolic = %SubAssignWith.lookup_impl_witness (constants.%SubAssignWith.lookup_impl_witness)]
// CHECK:STDOUT:   %impl.elem0.loc40_5.2: @TestSubAssignWith.%.loc40_5.2 (%.e56) = impl_witness_access %SubAssignWith.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc40_5.2 (constants.%impl.elem0.dd3)]
// CHECK:STDOUT:   %specific_impl_fn.loc40_5.2: <specific function> = specific_impl_function %impl.elem0.loc40_5.2, @SubAssignWith.WithSelf.Op(%U.loc38_23.1, %T.loc38_33.1) [symbolic = %specific_impl_fn.loc40_5.2 (constants.%specific_impl_fn.5da)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: ref @TestSubAssignWith.%T.as_type.loc38_66.1 (%T.as_type.3ff), %y.param: @TestSubAssignWith.%U.loc38_23.1 (%U)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %x.ref: ref @TestSubAssignWith.%T.as_type.loc38_66.1 (%T.as_type.3ff) = name_ref x, %x
// CHECK:STDOUT:     %y.ref: @TestSubAssignWith.%U.loc38_23.1 (%U) = name_ref y, %y
// CHECK:STDOUT:     %SubAssignWith.type.loc40: type = facet_type <@SubAssignWith, @SubAssignWith(constants.%U)> [symbolic = %SubAssignWith.type.loc38_56.1 (constants.%SubAssignWith.type.7e53cf.2)]
// CHECK:STDOUT:     %.loc40_5.1: @TestSubAssignWith.%SubAssignWith.assoc_type (%SubAssignWith.assoc_type.6e3877.2) = specific_constant imports.%Core.import_ref.509, @SubAssignWith.WithSelf(constants.%U, constants.%Self.f6c9aa.1) [symbolic = %assoc0 (constants.%assoc0.701e5f.2)]
// CHECK:STDOUT:     %Op.ref: @TestSubAssignWith.%SubAssignWith.assoc_type (%SubAssignWith.assoc_type.6e3877.2) = name_ref Op, %.loc40_5.1 [symbolic = %assoc0 (constants.%assoc0.701e5f.2)]
// CHECK:STDOUT:     %impl.elem0.loc40_5.1: @TestSubAssignWith.%.loc40_5.2 (%.e56) = impl_witness_access constants.%SubAssignWith.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc40_5.2 (constants.%impl.elem0.dd3)]
// CHECK:STDOUT:     %bound_method.loc40_5.1: <bound method> = bound_method %x.ref, %impl.elem0.loc40_5.1
// CHECK:STDOUT:     %specific_impl_fn.loc40_5.1: <specific function> = specific_impl_function %impl.elem0.loc40_5.1, @SubAssignWith.WithSelf.Op(constants.%U, constants.%T.f6c) [symbolic = %specific_impl_fn.loc40_5.2 (constants.%specific_impl_fn.5da)]
// CHECK:STDOUT:     %bound_method.loc40_5.2: <bound method> = bound_method %x.ref, %specific_impl_fn.loc40_5.1
// CHECK:STDOUT:     %SubAssignWith.WithSelf.Op.call: init %empty_tuple.type = call %bound_method.loc40_5.2(%x.ref, %y.ref)
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @TestMulWith(%U.loc45_4.2: type, %T.loc46_4.2: @TestMulWith.%MulWith_where.type (%MulWith_where.type.f84)) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %assoc1: @TestMulWith.%MulWith.assoc_type (%MulWith.assoc_type.7b2892.2) = assoc_entity element1, imports.%Core.import_ref.5a0 [symbolic = %assoc1 (constants.%assoc1.59776a.2)]
// CHECK:STDOUT:   %MulWith.lookup_impl_witness.loc49: <witness> = lookup_impl_witness %T.loc46_4.1, @MulWith, @MulWith(%U.loc45_4.1) [symbolic = %MulWith.lookup_impl_witness.loc49 (constants.%MulWith.lookup_impl_witness.5c9016.2)]
// CHECK:STDOUT:   %MulWith.facet: @TestMulWith.%MulWith.type.loc46_21.1 (%MulWith.type.f84f0b.2) = facet_value %T.as_type.loc47_6.1, (%MulWith.lookup_impl_witness.loc49) [symbolic = %MulWith.facet (constants.%MulWith.facet)]
// CHECK:STDOUT:   %MulWith.WithSelf.Op.type: type = fn_type @MulWith.WithSelf.Op, @MulWith.WithSelf(%U.loc45_4.1, %MulWith.facet) [symbolic = %MulWith.WithSelf.Op.type (constants.%MulWith.WithSelf.Op.type.d2e)]
// CHECK:STDOUT:   %.loc49_5.6: type = fn_type_with_self_type %MulWith.WithSelf.Op.type, %MulWith.facet [symbolic = %.loc49_5.6 (constants.%.ef7)]
// CHECK:STDOUT:   %impl.elem1.loc49_5.2: @TestMulWith.%.loc49_5.6 (%.ef7) = impl_witness_access %MulWith.lookup_impl_witness.loc49, element1 [symbolic = %impl.elem1.loc49_5.2 (constants.%impl.elem1.19d)]
// CHECK:STDOUT:   %specific_impl_fn.loc49_5.3: <specific function> = specific_impl_function %impl.elem1.loc49_5.2, @MulWith.WithSelf.Op(%U.loc45_4.1, %MulWith.facet) [symbolic = %specific_impl_fn.loc49_5.3 (constants.%specific_impl_fn.3df)]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %Destroy.lookup_impl_witness: <witness> = lookup_impl_witness %impl.elem0.1, @Destroy [symbolic = %Destroy.lookup_impl_witness (constants.%Destroy.lookup_impl_witness.a13)]
// CHECK:STDOUT:   %Destroy.facet.loc49_5.3: %Destroy.type = facet_value %impl.elem0.1, (%Destroy.lookup_impl_witness) [symbolic = %Destroy.facet.loc49_5.3 (constants.%Destroy.facet.9a4)]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet.loc49_5.3) [symbolic = %Destroy.WithSelf.Op.type (constants.%Destroy.WithSelf.Op.type.9cf)]
// CHECK:STDOUT:   %.loc49_5.7: type = fn_type_with_self_type %Destroy.WithSelf.Op.type, %Destroy.facet.loc49_5.3 [symbolic = %.loc49_5.7 (constants.%.731)]
// CHECK:STDOUT:   %impl.elem0.loc49_5.2: @TestMulWith.%.loc49_5.7 (%.731) = impl_witness_access %Destroy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc49_5.2 (constants.%impl.elem0.f89)]
// CHECK:STDOUT:   %specific_impl_fn.loc49_5.4: <specific function> = specific_impl_function %impl.elem0.loc49_5.2, @Destroy.WithSelf.Op(%Destroy.facet.loc49_5.3) [symbolic = %specific_impl_fn.loc49_5.4 (constants.%specific_impl_fn.601)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: @TestMulWith.%T.as_type.loc47_6.1 (%T.as_type.ad2), %y.param: @TestMulWith.%U.loc45_4.1 (%U)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %x.ref: @TestMulWith.%T.as_type.loc47_6.1 (%T.as_type.ad2) = name_ref x, %x
// CHECK:STDOUT:     %y.ref: @TestMulWith.%U.loc45_4.1 (%U) = name_ref y, %y
// CHECK:STDOUT:     %MulWith.type.loc49: type = facet_type <@MulWith, @MulWith(constants.%U)> [symbolic = %MulWith.type.loc46_21.1 (constants.%MulWith.type.f84f0b.2)]
// CHECK:STDOUT:     %.loc49_5.1: @TestMulWith.%MulWith.assoc_type (%MulWith.assoc_type.7b2892.2) = specific_constant imports.%Core.import_ref.8c1, @MulWith.WithSelf(constants.%U, constants.%Self.941007.1) [symbolic = %assoc1 (constants.%assoc1.59776a.2)]
// CHECK:STDOUT:     %Op.ref: @TestMulWith.%MulWith.assoc_type (%MulWith.assoc_type.7b2892.2) = name_ref Op, %.loc49_5.1 [symbolic = %assoc1 (constants.%assoc1.59776a.2)]
// CHECK:STDOUT:     %impl.elem1.loc49_5.1: @TestMulWith.%.loc49_5.6 (%.ef7) = impl_witness_access constants.%MulWith.lookup_impl_witness.5c9016.2, element1 [symbolic = %impl.elem1.loc49_5.2 (constants.%impl.elem1.19d)]
// CHECK:STDOUT:     %bound_method.loc49_5.1: <bound method> = bound_method %x.ref, %impl.elem1.loc49_5.1
// CHECK:STDOUT:     %specific_impl_fn.loc49_5.1: <specific function> = specific_impl_function %impl.elem1.loc49_5.1, @MulWith.WithSelf.Op(constants.%U, constants.%MulWith.facet) [symbolic = %specific_impl_fn.loc49_5.3 (constants.%specific_impl_fn.3df)]
// CHECK:STDOUT:     %bound_method.loc49_5.2: <bound method> = bound_method %x.ref, %specific_impl_fn.loc49_5.1
// CHECK:STDOUT:     %.loc49_5.2: ref @TestMulWith.%impl.elem0.1 (%impl.elem0.f2c0ce.2) = temporary_storage
// CHECK:STDOUT:     %MulWith.WithSelf.Op.call: init @TestMulWith.%impl.elem0.1 (%impl.elem0.f2c0ce.2) to %.loc49_5.2 = call %bound_method.loc49_5.2(%x.ref, %y.ref)
// CHECK:STDOUT:     %.loc49_5.3: ref @TestMulWith.%impl.elem0.1 (%impl.elem0.f2c0ce.2) = temporary %.loc49_5.2, %MulWith.WithSelf.Op.call
// CHECK:STDOUT:     %impl.elem0.loc49_5.1: @TestMulWith.%.loc49_5.7 (%.731) = impl_witness_access constants.%Destroy.lookup_impl_witness.a13, element0 [symbolic = %impl.elem0.loc49_5.2 (constants.%impl.elem0.f89)]
// CHECK:STDOUT:     %bound_method.loc49_5.3: <bound method> = bound_method %.loc49_5.3, %impl.elem0.loc49_5.1
// CHECK:STDOUT:     %Destroy.facet.loc49_5.1: %Destroy.type = facet_value constants.%impl.elem0.f2c0ce.2, (constants.%Destroy.lookup_impl_witness.a13) [symbolic = %Destroy.facet.loc49_5.3 (constants.%Destroy.facet.9a4)]
// CHECK:STDOUT:     %.loc49_5.4: %Destroy.type = converted constants.%impl.elem0.f2c0ce.2, %Destroy.facet.loc49_5.1 [symbolic = %Destroy.facet.loc49_5.3 (constants.%Destroy.facet.9a4)]
// CHECK:STDOUT:     %Destroy.facet.loc49_5.2: %Destroy.type = facet_value constants.%impl.elem0.f2c0ce.2, (constants.%Destroy.lookup_impl_witness.a13) [symbolic = %Destroy.facet.loc49_5.3 (constants.%Destroy.facet.9a4)]
// CHECK:STDOUT:     %.loc49_5.5: %Destroy.type = converted constants.%impl.elem0.f2c0ce.2, %Destroy.facet.loc49_5.2 [symbolic = %Destroy.facet.loc49_5.3 (constants.%Destroy.facet.9a4)]
// CHECK:STDOUT:     %specific_impl_fn.loc49_5.2: <specific function> = specific_impl_function %impl.elem0.loc49_5.1, @Destroy.WithSelf.Op(constants.%Destroy.facet.9a4) [symbolic = %specific_impl_fn.loc49_5.4 (constants.%specific_impl_fn.601)]
// CHECK:STDOUT:     %bound_method.loc49_5.4: <bound method> = bound_method %.loc49_5.3, %specific_impl_fn.loc49_5.2
// CHECK:STDOUT:     %Destroy.WithSelf.Op.call: init %empty_tuple.type = call %bound_method.loc49_5.4(%.loc49_5.3)
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @TestMulAssignWith(%U.loc53_23.2: type, %T.loc53_33.2: @TestMulAssignWith.%MulAssignWith.type.loc53_56.1 (%MulAssignWith.type.3a0e4d.2)) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %require_complete.loc55: <witness> = require_complete_type %MulAssignWith.type.loc53_56.1 [symbolic = %require_complete.loc55 (constants.%require_complete.6b1)]
// CHECK:STDOUT:   %MulAssignWith.assoc_type: type = assoc_entity_type @MulAssignWith, @MulAssignWith(%U.loc53_23.1) [symbolic = %MulAssignWith.assoc_type (constants.%MulAssignWith.assoc_type.9aa18f.2)]
// CHECK:STDOUT:   %assoc0: @TestMulAssignWith.%MulAssignWith.assoc_type (%MulAssignWith.assoc_type.9aa18f.2) = assoc_entity element0, imports.%Core.import_ref.ba3 [symbolic = %assoc0 (constants.%assoc0.22264a.2)]
// CHECK:STDOUT:   %MulAssignWith.WithSelf.Op.type: type = fn_type @MulAssignWith.WithSelf.Op, @MulAssignWith.WithSelf(%U.loc53_23.1, %T.loc53_33.1) [symbolic = %MulAssignWith.WithSelf.Op.type (constants.%MulAssignWith.WithSelf.Op.type.5df5bd.3)]
// CHECK:STDOUT:   %.loc55_5.2: type = fn_type_with_self_type %MulAssignWith.WithSelf.Op.type, %T.loc53_33.1 [symbolic = %.loc55_5.2 (constants.%.2db)]
// CHECK:STDOUT:   %MulAssignWith.lookup_impl_witness: <witness> = lookup_impl_witness %T.loc53_33.1, @MulAssignWith, @MulAssignWith(%U.loc53_23.1) [symbolic = %MulAssignWith.lookup_impl_witness (constants.%MulAssignWith.lookup_impl_witness)]
// CHECK:STDOUT:   %impl.elem0.loc55_5.2: @TestMulAssignWith.%.loc55_5.2 (%.2db) = impl_witness_access %MulAssignWith.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc55_5.2 (constants.%impl.elem0.c0f)]
// CHECK:STDOUT:   %specific_impl_fn.loc55_5.2: <specific function> = specific_impl_function %impl.elem0.loc55_5.2, @MulAssignWith.WithSelf.Op(%U.loc53_23.1, %T.loc53_33.1) [symbolic = %specific_impl_fn.loc55_5.2 (constants.%specific_impl_fn.54d)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: ref @TestMulAssignWith.%T.as_type.loc53_66.1 (%T.as_type.1de), %y.param: @TestMulAssignWith.%U.loc53_23.1 (%U)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %x.ref: ref @TestMulAssignWith.%T.as_type.loc53_66.1 (%T.as_type.1de) = name_ref x, %x
// CHECK:STDOUT:     %y.ref: @TestMulAssignWith.%U.loc53_23.1 (%U) = name_ref y, %y
// CHECK:STDOUT:     %MulAssignWith.type.loc55: type = facet_type <@MulAssignWith, @MulAssignWith(constants.%U)> [symbolic = %MulAssignWith.type.loc53_56.1 (constants.%MulAssignWith.type.3a0e4d.2)]
// CHECK:STDOUT:     %.loc55_5.1: @TestMulAssignWith.%MulAssignWith.assoc_type (%MulAssignWith.assoc_type.9aa18f.2) = specific_constant imports.%Core.import_ref.b6e, @MulAssignWith.WithSelf(constants.%U, constants.%Self.64dcb0.1) [symbolic = %assoc0 (constants.%assoc0.22264a.2)]
// CHECK:STDOUT:     %Op.ref: @TestMulAssignWith.%MulAssignWith.assoc_type (%MulAssignWith.assoc_type.9aa18f.2) = name_ref Op, %.loc55_5.1 [symbolic = %assoc0 (constants.%assoc0.22264a.2)]
// CHECK:STDOUT:     %impl.elem0.loc55_5.1: @TestMulAssignWith.%.loc55_5.2 (%.2db) = impl_witness_access constants.%MulAssignWith.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc55_5.2 (constants.%impl.elem0.c0f)]
// CHECK:STDOUT:     %bound_method.loc55_5.1: <bound method> = bound_method %x.ref, %impl.elem0.loc55_5.1
// CHECK:STDOUT:     %specific_impl_fn.loc55_5.1: <specific function> = specific_impl_function %impl.elem0.loc55_5.1, @MulAssignWith.WithSelf.Op(constants.%U, constants.%T.64d) [symbolic = %specific_impl_fn.loc55_5.2 (constants.%specific_impl_fn.54d)]
// CHECK:STDOUT:     %bound_method.loc55_5.2: <bound method> = bound_method %x.ref, %specific_impl_fn.loc55_5.1
// CHECK:STDOUT:     %MulAssignWith.WithSelf.Op.call: init %empty_tuple.type = call %bound_method.loc55_5.2(%x.ref, %y.ref)
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @TestDivWith(%U.loc60_4.2: type, %T.loc61_4.2: @TestDivWith.%DivWith_where.type (%DivWith_where.type.13f)) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %assoc1: @TestDivWith.%DivWith.assoc_type (%DivWith.assoc_type.d8b290.2) = assoc_entity element1, imports.%Core.import_ref.784 [symbolic = %assoc1 (constants.%assoc1.03cb8d.2)]
// CHECK:STDOUT:   %DivWith.lookup_impl_witness.loc64: <witness> = lookup_impl_witness %T.loc61_4.1, @DivWith, @DivWith(%U.loc60_4.1) [symbolic = %DivWith.lookup_impl_witness.loc64 (constants.%DivWith.lookup_impl_witness.1efd1b.2)]
// CHECK:STDOUT:   %DivWith.facet: @TestDivWith.%DivWith.type.loc61_21.1 (%DivWith.type.13f8f2.2) = facet_value %T.as_type.loc62_6.1, (%DivWith.lookup_impl_witness.loc64) [symbolic = %DivWith.facet (constants.%DivWith.facet)]
// CHECK:STDOUT:   %DivWith.WithSelf.Op.type: type = fn_type @DivWith.WithSelf.Op, @DivWith.WithSelf(%U.loc60_4.1, %DivWith.facet) [symbolic = %DivWith.WithSelf.Op.type (constants.%DivWith.WithSelf.Op.type.208)]
// CHECK:STDOUT:   %.loc64_5.6: type = fn_type_with_self_type %DivWith.WithSelf.Op.type, %DivWith.facet [symbolic = %.loc64_5.6 (constants.%.401)]
// CHECK:STDOUT:   %impl.elem1.loc64_5.2: @TestDivWith.%.loc64_5.6 (%.401) = impl_witness_access %DivWith.lookup_impl_witness.loc64, element1 [symbolic = %impl.elem1.loc64_5.2 (constants.%impl.elem1.f2d)]
// CHECK:STDOUT:   %specific_impl_fn.loc64_5.3: <specific function> = specific_impl_function %impl.elem1.loc64_5.2, @DivWith.WithSelf.Op(%U.loc60_4.1, %DivWith.facet) [symbolic = %specific_impl_fn.loc64_5.3 (constants.%specific_impl_fn.567)]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %Destroy.lookup_impl_witness: <witness> = lookup_impl_witness %impl.elem0.1, @Destroy [symbolic = %Destroy.lookup_impl_witness (constants.%Destroy.lookup_impl_witness.548)]
// CHECK:STDOUT:   %Destroy.facet.loc64_5.3: %Destroy.type = facet_value %impl.elem0.1, (%Destroy.lookup_impl_witness) [symbolic = %Destroy.facet.loc64_5.3 (constants.%Destroy.facet.680)]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet.loc64_5.3) [symbolic = %Destroy.WithSelf.Op.type (constants.%Destroy.WithSelf.Op.type.d31)]
// CHECK:STDOUT:   %.loc64_5.7: type = fn_type_with_self_type %Destroy.WithSelf.Op.type, %Destroy.facet.loc64_5.3 [symbolic = %.loc64_5.7 (constants.%.c1c)]
// CHECK:STDOUT:   %impl.elem0.loc64_5.2: @TestDivWith.%.loc64_5.7 (%.c1c) = impl_witness_access %Destroy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc64_5.2 (constants.%impl.elem0.800)]
// CHECK:STDOUT:   %specific_impl_fn.loc64_5.4: <specific function> = specific_impl_function %impl.elem0.loc64_5.2, @Destroy.WithSelf.Op(%Destroy.facet.loc64_5.3) [symbolic = %specific_impl_fn.loc64_5.4 (constants.%specific_impl_fn.598)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: @TestDivWith.%T.as_type.loc62_6.1 (%T.as_type.f1d), %y.param: @TestDivWith.%U.loc60_4.1 (%U)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %x.ref: @TestDivWith.%T.as_type.loc62_6.1 (%T.as_type.f1d) = name_ref x, %x
// CHECK:STDOUT:     %y.ref: @TestDivWith.%U.loc60_4.1 (%U) = name_ref y, %y
// CHECK:STDOUT:     %DivWith.type.loc64: type = facet_type <@DivWith, @DivWith(constants.%U)> [symbolic = %DivWith.type.loc61_21.1 (constants.%DivWith.type.13f8f2.2)]
// CHECK:STDOUT:     %.loc64_5.1: @TestDivWith.%DivWith.assoc_type (%DivWith.assoc_type.d8b290.2) = specific_constant imports.%Core.import_ref.ca2, @DivWith.WithSelf(constants.%U, constants.%Self.16a753.1) [symbolic = %assoc1 (constants.%assoc1.03cb8d.2)]
// CHECK:STDOUT:     %Op.ref: @TestDivWith.%DivWith.assoc_type (%DivWith.assoc_type.d8b290.2) = name_ref Op, %.loc64_5.1 [symbolic = %assoc1 (constants.%assoc1.03cb8d.2)]
// CHECK:STDOUT:     %impl.elem1.loc64_5.1: @TestDivWith.%.loc64_5.6 (%.401) = impl_witness_access constants.%DivWith.lookup_impl_witness.1efd1b.2, element1 [symbolic = %impl.elem1.loc64_5.2 (constants.%impl.elem1.f2d)]
// CHECK:STDOUT:     %bound_method.loc64_5.1: <bound method> = bound_method %x.ref, %impl.elem1.loc64_5.1
// CHECK:STDOUT:     %specific_impl_fn.loc64_5.1: <specific function> = specific_impl_function %impl.elem1.loc64_5.1, @DivWith.WithSelf.Op(constants.%U, constants.%DivWith.facet) [symbolic = %specific_impl_fn.loc64_5.3 (constants.%specific_impl_fn.567)]
// CHECK:STDOUT:     %bound_method.loc64_5.2: <bound method> = bound_method %x.ref, %specific_impl_fn.loc64_5.1
// CHECK:STDOUT:     %.loc64_5.2: ref @TestDivWith.%impl.elem0.1 (%impl.elem0.e2a7ee.2) = temporary_storage
// CHECK:STDOUT:     %DivWith.WithSelf.Op.call: init @TestDivWith.%impl.elem0.1 (%impl.elem0.e2a7ee.2) to %.loc64_5.2 = call %bound_method.loc64_5.2(%x.ref, %y.ref)
// CHECK:STDOUT:     %.loc64_5.3: ref @TestDivWith.%impl.elem0.1 (%impl.elem0.e2a7ee.2) = temporary %.loc64_5.2, %DivWith.WithSelf.Op.call
// CHECK:STDOUT:     %impl.elem0.loc64_5.1: @TestDivWith.%.loc64_5.7 (%.c1c) = impl_witness_access constants.%Destroy.lookup_impl_witness.548, element0 [symbolic = %impl.elem0.loc64_5.2 (constants.%impl.elem0.800)]
// CHECK:STDOUT:     %bound_method.loc64_5.3: <bound method> = bound_method %.loc64_5.3, %impl.elem0.loc64_5.1
// CHECK:STDOUT:     %Destroy.facet.loc64_5.1: %Destroy.type = facet_value constants.%impl.elem0.e2a7ee.2, (constants.%Destroy.lookup_impl_witness.548) [symbolic = %Destroy.facet.loc64_5.3 (constants.%Destroy.facet.680)]
// CHECK:STDOUT:     %.loc64_5.4: %Destroy.type = converted constants.%impl.elem0.e2a7ee.2, %Destroy.facet.loc64_5.1 [symbolic = %Destroy.facet.loc64_5.3 (constants.%Destroy.facet.680)]
// CHECK:STDOUT:     %Destroy.facet.loc64_5.2: %Destroy.type = facet_value constants.%impl.elem0.e2a7ee.2, (constants.%Destroy.lookup_impl_witness.548) [symbolic = %Destroy.facet.loc64_5.3 (constants.%Destroy.facet.680)]
// CHECK:STDOUT:     %.loc64_5.5: %Destroy.type = converted constants.%impl.elem0.e2a7ee.2, %Destroy.facet.loc64_5.2 [symbolic = %Destroy.facet.loc64_5.3 (constants.%Destroy.facet.680)]
// CHECK:STDOUT:     %specific_impl_fn.loc64_5.2: <specific function> = specific_impl_function %impl.elem0.loc64_5.1, @Destroy.WithSelf.Op(constants.%Destroy.facet.680) [symbolic = %specific_impl_fn.loc64_5.4 (constants.%specific_impl_fn.598)]
// CHECK:STDOUT:     %bound_method.loc64_5.4: <bound method> = bound_method %.loc64_5.3, %specific_impl_fn.loc64_5.2
// CHECK:STDOUT:     %Destroy.WithSelf.Op.call: init %empty_tuple.type = call %bound_method.loc64_5.4(%.loc64_5.3)
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @TestDivAssignWith(%U.loc68_23.2: type, %T.loc68_33.2: @TestDivAssignWith.%DivAssignWith.type.loc68_56.1 (%DivAssignWith.type.7e58ca.2)) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %require_complete.loc70: <witness> = require_complete_type %DivAssignWith.type.loc68_56.1 [symbolic = %require_complete.loc70 (constants.%require_complete.c5c)]
// CHECK:STDOUT:   %DivAssignWith.assoc_type: type = assoc_entity_type @DivAssignWith, @DivAssignWith(%U.loc68_23.1) [symbolic = %DivAssignWith.assoc_type (constants.%DivAssignWith.assoc_type.bb68ec.2)]
// CHECK:STDOUT:   %assoc0: @TestDivAssignWith.%DivAssignWith.assoc_type (%DivAssignWith.assoc_type.bb68ec.2) = assoc_entity element0, imports.%Core.import_ref.a5d [symbolic = %assoc0 (constants.%assoc0.32f100.2)]
// CHECK:STDOUT:   %DivAssignWith.WithSelf.Op.type: type = fn_type @DivAssignWith.WithSelf.Op, @DivAssignWith.WithSelf(%U.loc68_23.1, %T.loc68_33.1) [symbolic = %DivAssignWith.WithSelf.Op.type (constants.%DivAssignWith.WithSelf.Op.type.ff5775.3)]
// CHECK:STDOUT:   %.loc70_5.2: type = fn_type_with_self_type %DivAssignWith.WithSelf.Op.type, %T.loc68_33.1 [symbolic = %.loc70_5.2 (constants.%.754)]
// CHECK:STDOUT:   %DivAssignWith.lookup_impl_witness: <witness> = lookup_impl_witness %T.loc68_33.1, @DivAssignWith, @DivAssignWith(%U.loc68_23.1) [symbolic = %DivAssignWith.lookup_impl_witness (constants.%DivAssignWith.lookup_impl_witness)]
// CHECK:STDOUT:   %impl.elem0.loc70_5.2: @TestDivAssignWith.%.loc70_5.2 (%.754) = impl_witness_access %DivAssignWith.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc70_5.2 (constants.%impl.elem0.0c4)]
// CHECK:STDOUT:   %specific_impl_fn.loc70_5.2: <specific function> = specific_impl_function %impl.elem0.loc70_5.2, @DivAssignWith.WithSelf.Op(%U.loc68_23.1, %T.loc68_33.1) [symbolic = %specific_impl_fn.loc70_5.2 (constants.%specific_impl_fn.011)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: ref @TestDivAssignWith.%T.as_type.loc68_66.1 (%T.as_type.e57), %y.param: @TestDivAssignWith.%U.loc68_23.1 (%U)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %x.ref: ref @TestDivAssignWith.%T.as_type.loc68_66.1 (%T.as_type.e57) = name_ref x, %x
// CHECK:STDOUT:     %y.ref: @TestDivAssignWith.%U.loc68_23.1 (%U) = name_ref y, %y
// CHECK:STDOUT:     %DivAssignWith.type.loc70: type = facet_type <@DivAssignWith, @DivAssignWith(constants.%U)> [symbolic = %DivAssignWith.type.loc68_56.1 (constants.%DivAssignWith.type.7e58ca.2)]
// CHECK:STDOUT:     %.loc70_5.1: @TestDivAssignWith.%DivAssignWith.assoc_type (%DivAssignWith.assoc_type.bb68ec.2) = specific_constant imports.%Core.import_ref.988, @DivAssignWith.WithSelf(constants.%U, constants.%Self.c33c81.1) [symbolic = %assoc0 (constants.%assoc0.32f100.2)]
// CHECK:STDOUT:     %Op.ref: @TestDivAssignWith.%DivAssignWith.assoc_type (%DivAssignWith.assoc_type.bb68ec.2) = name_ref Op, %.loc70_5.1 [symbolic = %assoc0 (constants.%assoc0.32f100.2)]
// CHECK:STDOUT:     %impl.elem0.loc70_5.1: @TestDivAssignWith.%.loc70_5.2 (%.754) = impl_witness_access constants.%DivAssignWith.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc70_5.2 (constants.%impl.elem0.0c4)]
// CHECK:STDOUT:     %bound_method.loc70_5.1: <bound method> = bound_method %x.ref, %impl.elem0.loc70_5.1
// CHECK:STDOUT:     %specific_impl_fn.loc70_5.1: <specific function> = specific_impl_function %impl.elem0.loc70_5.1, @DivAssignWith.WithSelf.Op(constants.%U, constants.%T.c33) [symbolic = %specific_impl_fn.loc70_5.2 (constants.%specific_impl_fn.011)]
// CHECK:STDOUT:     %bound_method.loc70_5.2: <bound method> = bound_method %x.ref, %specific_impl_fn.loc70_5.1
// CHECK:STDOUT:     %DivAssignWith.WithSelf.Op.call: init %empty_tuple.type = call %bound_method.loc70_5.2(%x.ref, %y.ref)
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @TestModWith(%U.loc75_4.2: type, %T.loc76_4.2: @TestModWith.%ModWith_where.type (%ModWith_where.type.2e8)) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %assoc1: @TestModWith.%ModWith.assoc_type (%ModWith.assoc_type.316ae9.2) = assoc_entity element1, imports.%Core.import_ref.b56 [symbolic = %assoc1 (constants.%assoc1.8ee089.2)]
// CHECK:STDOUT:   %ModWith.lookup_impl_witness.loc79: <witness> = lookup_impl_witness %T.loc76_4.1, @ModWith, @ModWith(%U.loc75_4.1) [symbolic = %ModWith.lookup_impl_witness.loc79 (constants.%ModWith.lookup_impl_witness.c327a3.2)]
// CHECK:STDOUT:   %ModWith.facet: @TestModWith.%ModWith.type.loc76_21.1 (%ModWith.type.2e82f9.2) = facet_value %T.as_type.loc77_6.1, (%ModWith.lookup_impl_witness.loc79) [symbolic = %ModWith.facet (constants.%ModWith.facet)]
// CHECK:STDOUT:   %ModWith.WithSelf.Op.type: type = fn_type @ModWith.WithSelf.Op, @ModWith.WithSelf(%U.loc75_4.1, %ModWith.facet) [symbolic = %ModWith.WithSelf.Op.type (constants.%ModWith.WithSelf.Op.type.e58)]
// CHECK:STDOUT:   %.loc79_5.6: type = fn_type_with_self_type %ModWith.WithSelf.Op.type, %ModWith.facet [symbolic = %.loc79_5.6 (constants.%.07c)]
// CHECK:STDOUT:   %impl.elem1.loc79_5.2: @TestModWith.%.loc79_5.6 (%.07c) = impl_witness_access %ModWith.lookup_impl_witness.loc79, element1 [symbolic = %impl.elem1.loc79_5.2 (constants.%impl.elem1.617)]
// CHECK:STDOUT:   %specific_impl_fn.loc79_5.3: <specific function> = specific_impl_function %impl.elem1.loc79_5.2, @ModWith.WithSelf.Op(%U.loc75_4.1, %ModWith.facet) [symbolic = %specific_impl_fn.loc79_5.3 (constants.%specific_impl_fn.95f)]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %Destroy.lookup_impl_witness: <witness> = lookup_impl_witness %impl.elem0.1, @Destroy [symbolic = %Destroy.lookup_impl_witness (constants.%Destroy.lookup_impl_witness.64a)]
// CHECK:STDOUT:   %Destroy.facet.loc79_5.3: %Destroy.type = facet_value %impl.elem0.1, (%Destroy.lookup_impl_witness) [symbolic = %Destroy.facet.loc79_5.3 (constants.%Destroy.facet.6e7)]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet.loc79_5.3) [symbolic = %Destroy.WithSelf.Op.type (constants.%Destroy.WithSelf.Op.type.e36)]
// CHECK:STDOUT:   %.loc79_5.7: type = fn_type_with_self_type %Destroy.WithSelf.Op.type, %Destroy.facet.loc79_5.3 [symbolic = %.loc79_5.7 (constants.%.6ed)]
// CHECK:STDOUT:   %impl.elem0.loc79_5.2: @TestModWith.%.loc79_5.7 (%.6ed) = impl_witness_access %Destroy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc79_5.2 (constants.%impl.elem0.918)]
// CHECK:STDOUT:   %specific_impl_fn.loc79_5.4: <specific function> = specific_impl_function %impl.elem0.loc79_5.2, @Destroy.WithSelf.Op(%Destroy.facet.loc79_5.3) [symbolic = %specific_impl_fn.loc79_5.4 (constants.%specific_impl_fn.f55)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: @TestModWith.%T.as_type.loc77_6.1 (%T.as_type.c2b), %y.param: @TestModWith.%U.loc75_4.1 (%U)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %x.ref: @TestModWith.%T.as_type.loc77_6.1 (%T.as_type.c2b) = name_ref x, %x
// CHECK:STDOUT:     %y.ref: @TestModWith.%U.loc75_4.1 (%U) = name_ref y, %y
// CHECK:STDOUT:     %ModWith.type.loc79: type = facet_type <@ModWith, @ModWith(constants.%U)> [symbolic = %ModWith.type.loc76_21.1 (constants.%ModWith.type.2e82f9.2)]
// CHECK:STDOUT:     %.loc79_5.1: @TestModWith.%ModWith.assoc_type (%ModWith.assoc_type.316ae9.2) = specific_constant imports.%Core.import_ref.ffe, @ModWith.WithSelf(constants.%U, constants.%Self.0ef52c.1) [symbolic = %assoc1 (constants.%assoc1.8ee089.2)]
// CHECK:STDOUT:     %Op.ref: @TestModWith.%ModWith.assoc_type (%ModWith.assoc_type.316ae9.2) = name_ref Op, %.loc79_5.1 [symbolic = %assoc1 (constants.%assoc1.8ee089.2)]
// CHECK:STDOUT:     %impl.elem1.loc79_5.1: @TestModWith.%.loc79_5.6 (%.07c) = impl_witness_access constants.%ModWith.lookup_impl_witness.c327a3.2, element1 [symbolic = %impl.elem1.loc79_5.2 (constants.%impl.elem1.617)]
// CHECK:STDOUT:     %bound_method.loc79_5.1: <bound method> = bound_method %x.ref, %impl.elem1.loc79_5.1
// CHECK:STDOUT:     %specific_impl_fn.loc79_5.1: <specific function> = specific_impl_function %impl.elem1.loc79_5.1, @ModWith.WithSelf.Op(constants.%U, constants.%ModWith.facet) [symbolic = %specific_impl_fn.loc79_5.3 (constants.%specific_impl_fn.95f)]
// CHECK:STDOUT:     %bound_method.loc79_5.2: <bound method> = bound_method %x.ref, %specific_impl_fn.loc79_5.1
// CHECK:STDOUT:     %.loc79_5.2: ref @TestModWith.%impl.elem0.1 (%impl.elem0.2a2b98.2) = temporary_storage
// CHECK:STDOUT:     %ModWith.WithSelf.Op.call: init @TestModWith.%impl.elem0.1 (%impl.elem0.2a2b98.2) to %.loc79_5.2 = call %bound_method.loc79_5.2(%x.ref, %y.ref)
// CHECK:STDOUT:     %.loc79_5.3: ref @TestModWith.%impl.elem0.1 (%impl.elem0.2a2b98.2) = temporary %.loc79_5.2, %ModWith.WithSelf.Op.call
// CHECK:STDOUT:     %impl.elem0.loc79_5.1: @TestModWith.%.loc79_5.7 (%.6ed) = impl_witness_access constants.%Destroy.lookup_impl_witness.64a, element0 [symbolic = %impl.elem0.loc79_5.2 (constants.%impl.elem0.918)]
// CHECK:STDOUT:     %bound_method.loc79_5.3: <bound method> = bound_method %.loc79_5.3, %impl.elem0.loc79_5.1
// CHECK:STDOUT:     %Destroy.facet.loc79_5.1: %Destroy.type = facet_value constants.%impl.elem0.2a2b98.2, (constants.%Destroy.lookup_impl_witness.64a) [symbolic = %Destroy.facet.loc79_5.3 (constants.%Destroy.facet.6e7)]
// CHECK:STDOUT:     %.loc79_5.4: %Destroy.type = converted constants.%impl.elem0.2a2b98.2, %Destroy.facet.loc79_5.1 [symbolic = %Destroy.facet.loc79_5.3 (constants.%Destroy.facet.6e7)]
// CHECK:STDOUT:     %Destroy.facet.loc79_5.2: %Destroy.type = facet_value constants.%impl.elem0.2a2b98.2, (constants.%Destroy.lookup_impl_witness.64a) [symbolic = %Destroy.facet.loc79_5.3 (constants.%Destroy.facet.6e7)]
// CHECK:STDOUT:     %.loc79_5.5: %Destroy.type = converted constants.%impl.elem0.2a2b98.2, %Destroy.facet.loc79_5.2 [symbolic = %Destroy.facet.loc79_5.3 (constants.%Destroy.facet.6e7)]
// CHECK:STDOUT:     %specific_impl_fn.loc79_5.2: <specific function> = specific_impl_function %impl.elem0.loc79_5.1, @Destroy.WithSelf.Op(constants.%Destroy.facet.6e7) [symbolic = %specific_impl_fn.loc79_5.4 (constants.%specific_impl_fn.f55)]
// CHECK:STDOUT:     %bound_method.loc79_5.4: <bound method> = bound_method %.loc79_5.3, %specific_impl_fn.loc79_5.2
// CHECK:STDOUT:     %Destroy.WithSelf.Op.call: init %empty_tuple.type = call %bound_method.loc79_5.4(%.loc79_5.3)
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @TestModAssignWith(%U.loc83_23.2: type, %T.loc83_33.2: @TestModAssignWith.%ModAssignWith.type.loc83_56.1 (%ModAssignWith.type.778d68.2)) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %require_complete.loc85: <witness> = require_complete_type %ModAssignWith.type.loc83_56.1 [symbolic = %require_complete.loc85 (constants.%require_complete.680)]
// CHECK:STDOUT:   %ModAssignWith.assoc_type: type = assoc_entity_type @ModAssignWith, @ModAssignWith(%U.loc83_23.1) [symbolic = %ModAssignWith.assoc_type (constants.%ModAssignWith.assoc_type.bc79ec.2)]
// CHECK:STDOUT:   %assoc0: @TestModAssignWith.%ModAssignWith.assoc_type (%ModAssignWith.assoc_type.bc79ec.2) = assoc_entity element0, imports.%Core.import_ref.cf6 [symbolic = %assoc0 (constants.%assoc0.8c0719.2)]
// CHECK:STDOUT:   %ModAssignWith.WithSelf.Op.type: type = fn_type @ModAssignWith.WithSelf.Op, @ModAssignWith.WithSelf(%U.loc83_23.1, %T.loc83_33.1) [symbolic = %ModAssignWith.WithSelf.Op.type (constants.%ModAssignWith.WithSelf.Op.type.ec4109.3)]
// CHECK:STDOUT:   %.loc85_5.2: type = fn_type_with_self_type %ModAssignWith.WithSelf.Op.type, %T.loc83_33.1 [symbolic = %.loc85_5.2 (constants.%.965)]
// CHECK:STDOUT:   %ModAssignWith.lookup_impl_witness: <witness> = lookup_impl_witness %T.loc83_33.1, @ModAssignWith, @ModAssignWith(%U.loc83_23.1) [symbolic = %ModAssignWith.lookup_impl_witness (constants.%ModAssignWith.lookup_impl_witness)]
// CHECK:STDOUT:   %impl.elem0.loc85_5.2: @TestModAssignWith.%.loc85_5.2 (%.965) = impl_witness_access %ModAssignWith.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc85_5.2 (constants.%impl.elem0.ec7)]
// CHECK:STDOUT:   %specific_impl_fn.loc85_5.2: <specific function> = specific_impl_function %impl.elem0.loc85_5.2, @ModAssignWith.WithSelf.Op(%U.loc83_23.1, %T.loc83_33.1) [symbolic = %specific_impl_fn.loc85_5.2 (constants.%specific_impl_fn.cf7)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: ref @TestModAssignWith.%T.as_type.loc83_66.1 (%T.as_type.ce6), %y.param: @TestModAssignWith.%U.loc83_23.1 (%U)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %x.ref: ref @TestModAssignWith.%T.as_type.loc83_66.1 (%T.as_type.ce6) = name_ref x, %x
// CHECK:STDOUT:     %y.ref: @TestModAssignWith.%U.loc83_23.1 (%U) = name_ref y, %y
// CHECK:STDOUT:     %ModAssignWith.type.loc85: type = facet_type <@ModAssignWith, @ModAssignWith(constants.%U)> [symbolic = %ModAssignWith.type.loc83_56.1 (constants.%ModAssignWith.type.778d68.2)]
// CHECK:STDOUT:     %.loc85_5.1: @TestModAssignWith.%ModAssignWith.assoc_type (%ModAssignWith.assoc_type.bc79ec.2) = specific_constant imports.%Core.import_ref.3ff, @ModAssignWith.WithSelf(constants.%U, constants.%Self.bca00b.1) [symbolic = %assoc0 (constants.%assoc0.8c0719.2)]
// CHECK:STDOUT:     %Op.ref: @TestModAssignWith.%ModAssignWith.assoc_type (%ModAssignWith.assoc_type.bc79ec.2) = name_ref Op, %.loc85_5.1 [symbolic = %assoc0 (constants.%assoc0.8c0719.2)]
// CHECK:STDOUT:     %impl.elem0.loc85_5.1: @TestModAssignWith.%.loc85_5.2 (%.965) = impl_witness_access constants.%ModAssignWith.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc85_5.2 (constants.%impl.elem0.ec7)]
// CHECK:STDOUT:     %bound_method.loc85_5.1: <bound method> = bound_method %x.ref, %impl.elem0.loc85_5.1
// CHECK:STDOUT:     %specific_impl_fn.loc85_5.1: <specific function> = specific_impl_function %impl.elem0.loc85_5.1, @ModAssignWith.WithSelf.Op(constants.%U, constants.%T.bca) [symbolic = %specific_impl_fn.loc85_5.2 (constants.%specific_impl_fn.cf7)]
// CHECK:STDOUT:     %bound_method.loc85_5.2: <bound method> = bound_method %x.ref, %specific_impl_fn.loc85_5.1
// CHECK:STDOUT:     %ModAssignWith.WithSelf.Op.call: init %empty_tuple.type = call %bound_method.loc85_5.2(%x.ref, %y.ref)
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @TestNegate(%T.loc89_16.2: %Negate_where.type.871) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %Negate.lookup_impl_witness: <witness> = lookup_impl_witness %T.loc89_16.1, @Negate [symbolic = %Negate.lookup_impl_witness (constants.%Negate.lookup_impl_witness.c40cda.2)]
// CHECK:STDOUT:   %Negate.facet.loc91_3.3: %Negate.type = facet_value %T.as_type.loc89_68.1, (%Negate.lookup_impl_witness) [symbolic = %Negate.facet.loc91_3.3 (constants.%Negate.facet)]
// CHECK:STDOUT:   %Negate.WithSelf.Op.type: type = fn_type @Negate.WithSelf.Op, @Negate.WithSelf(%Negate.facet.loc91_3.3) [symbolic = %Negate.WithSelf.Op.type (constants.%Negate.WithSelf.Op.type.007)]
// CHECK:STDOUT:   %.loc91_3.7: type = fn_type_with_self_type %Negate.WithSelf.Op.type, %Negate.facet.loc91_3.3 [symbolic = %.loc91_3.7 (constants.%.a2a)]
// CHECK:STDOUT:   %impl.elem1.loc91_3.2: @TestNegate.%.loc91_3.7 (%.a2a) = impl_witness_access %Negate.lookup_impl_witness, element1 [symbolic = %impl.elem1.loc91_3.2 (constants.%impl.elem1.b9d)]
// CHECK:STDOUT:   %specific_impl_fn.loc91_3.3: <specific function> = specific_impl_function %impl.elem1.loc91_3.2, @Negate.WithSelf.Op(%Negate.facet.loc91_3.3) [symbolic = %specific_impl_fn.loc91_3.3 (constants.%specific_impl_fn.847)]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %Destroy.lookup_impl_witness: <witness> = lookup_impl_witness %impl.elem0.1, @Destroy [symbolic = %Destroy.lookup_impl_witness (constants.%Destroy.lookup_impl_witness.7e7)]
// CHECK:STDOUT:   %Destroy.facet.loc91_3.3: %Destroy.type = facet_value %impl.elem0.1, (%Destroy.lookup_impl_witness) [symbolic = %Destroy.facet.loc91_3.3 (constants.%Destroy.facet.4da)]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet.loc91_3.3) [symbolic = %Destroy.WithSelf.Op.type (constants.%Destroy.WithSelf.Op.type.ff0)]
// CHECK:STDOUT:   %.loc91_3.8: type = fn_type_with_self_type %Destroy.WithSelf.Op.type, %Destroy.facet.loc91_3.3 [symbolic = %.loc91_3.8 (constants.%.3f0)]
// CHECK:STDOUT:   %impl.elem0.loc91_3.2: @TestNegate.%.loc91_3.8 (%.3f0) = impl_witness_access %Destroy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc91_3.2 (constants.%impl.elem0.22c)]
// CHECK:STDOUT:   %specific_impl_fn.loc91_3.4: <specific function> = specific_impl_function %impl.elem0.loc91_3.2, @Destroy.WithSelf.Op(%Destroy.facet.loc91_3.3) [symbolic = %specific_impl_fn.loc91_3.4 (constants.%specific_impl_fn.72a)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: @TestNegate.%T.as_type.loc89_68.1 (%T.as_type.db1)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %x.ref: @TestNegate.%T.as_type.loc89_68.1 (%T.as_type.db1) = name_ref x, %x
// CHECK:STDOUT:     %impl.elem1.loc91_3.1: @TestNegate.%.loc91_3.7 (%.a2a) = impl_witness_access constants.%Negate.lookup_impl_witness.c40cda.2, element1 [symbolic = %impl.elem1.loc91_3.2 (constants.%impl.elem1.b9d)]
// CHECK:STDOUT:     %bound_method.loc91_3.1: <bound method> = bound_method %x.ref, %impl.elem1.loc91_3.1
// CHECK:STDOUT:     %Negate.facet.loc91_3.1: %Negate.type = facet_value constants.%T.as_type.db1, (constants.%Negate.lookup_impl_witness.c40cda.2) [symbolic = %Negate.facet.loc91_3.3 (constants.%Negate.facet)]
// CHECK:STDOUT:     %.loc91_3.1: %Negate.type = converted constants.%T.as_type.db1, %Negate.facet.loc91_3.1 [symbolic = %Negate.facet.loc91_3.3 (constants.%Negate.facet)]
// CHECK:STDOUT:     %Negate.facet.loc91_3.2: %Negate.type = facet_value constants.%T.as_type.db1, (constants.%Negate.lookup_impl_witness.c40cda.2) [symbolic = %Negate.facet.loc91_3.3 (constants.%Negate.facet)]
// CHECK:STDOUT:     %.loc91_3.2: %Negate.type = converted constants.%T.as_type.db1, %Negate.facet.loc91_3.2 [symbolic = %Negate.facet.loc91_3.3 (constants.%Negate.facet)]
// CHECK:STDOUT:     %specific_impl_fn.loc91_3.1: <specific function> = specific_impl_function %impl.elem1.loc91_3.1, @Negate.WithSelf.Op(constants.%Negate.facet) [symbolic = %specific_impl_fn.loc91_3.3 (constants.%specific_impl_fn.847)]
// CHECK:STDOUT:     %bound_method.loc91_3.2: <bound method> = bound_method %x.ref, %specific_impl_fn.loc91_3.1
// CHECK:STDOUT:     %.loc91_3.3: ref @TestNegate.%impl.elem0.1 (%impl.elem0.9a6046.2) = temporary_storage
// CHECK:STDOUT:     %Negate.WithSelf.Op.call: init @TestNegate.%impl.elem0.1 (%impl.elem0.9a6046.2) to %.loc91_3.3 = call %bound_method.loc91_3.2(%x.ref)
// CHECK:STDOUT:     %.loc91_3.4: ref @TestNegate.%impl.elem0.1 (%impl.elem0.9a6046.2) = temporary %.loc91_3.3, %Negate.WithSelf.Op.call
// CHECK:STDOUT:     %impl.elem0.loc91_3.1: @TestNegate.%.loc91_3.8 (%.3f0) = impl_witness_access constants.%Destroy.lookup_impl_witness.7e7, element0 [symbolic = %impl.elem0.loc91_3.2 (constants.%impl.elem0.22c)]
// CHECK:STDOUT:     %bound_method.loc91_3.3: <bound method> = bound_method %.loc91_3.4, %impl.elem0.loc91_3.1
// CHECK:STDOUT:     %Destroy.facet.loc91_3.1: %Destroy.type = facet_value constants.%impl.elem0.9a6046.2, (constants.%Destroy.lookup_impl_witness.7e7) [symbolic = %Destroy.facet.loc91_3.3 (constants.%Destroy.facet.4da)]
// CHECK:STDOUT:     %.loc91_3.5: %Destroy.type = converted constants.%impl.elem0.9a6046.2, %Destroy.facet.loc91_3.1 [symbolic = %Destroy.facet.loc91_3.3 (constants.%Destroy.facet.4da)]
// CHECK:STDOUT:     %Destroy.facet.loc91_3.2: %Destroy.type = facet_value constants.%impl.elem0.9a6046.2, (constants.%Destroy.lookup_impl_witness.7e7) [symbolic = %Destroy.facet.loc91_3.3 (constants.%Destroy.facet.4da)]
// CHECK:STDOUT:     %.loc91_3.6: %Destroy.type = converted constants.%impl.elem0.9a6046.2, %Destroy.facet.loc91_3.2 [symbolic = %Destroy.facet.loc91_3.3 (constants.%Destroy.facet.4da)]
// CHECK:STDOUT:     %specific_impl_fn.loc91_3.2: <specific function> = specific_impl_function %impl.elem0.loc91_3.1, @Destroy.WithSelf.Op(constants.%Destroy.facet.4da) [symbolic = %specific_impl_fn.loc91_3.4 (constants.%specific_impl_fn.72a)]
// CHECK:STDOUT:     %bound_method.loc91_3.4: <bound method> = bound_method %.loc91_3.4, %specific_impl_fn.loc91_3.2
// CHECK:STDOUT:     %Destroy.WithSelf.Op.call: init %empty_tuple.type = call %bound_method.loc91_3.4(%.loc91_3.4)
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @TestInc(%T.loc95_13.2: %Inc.type) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %Inc.WithSelf.Op.type: type = fn_type @Inc.WithSelf.Op, @Inc.WithSelf(%T.loc95_13.1) [symbolic = %Inc.WithSelf.Op.type (constants.%Inc.WithSelf.Op.type.e10ef6.2)]
// CHECK:STDOUT:   %.loc97_3.3: type = fn_type_with_self_type %Inc.WithSelf.Op.type, %T.loc95_13.1 [symbolic = %.loc97_3.3 (constants.%.a08)]
// CHECK:STDOUT:   %Inc.lookup_impl_witness: <witness> = lookup_impl_witness %T.loc95_13.1, @Inc [symbolic = %Inc.lookup_impl_witness (constants.%Inc.lookup_impl_witness)]
// CHECK:STDOUT:   %impl.elem0.loc97_3.2: @TestInc.%.loc97_3.3 (%.a08) = impl_witness_access %Inc.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc97_3.2 (constants.%impl.elem0.b92)]
// CHECK:STDOUT:   %specific_impl_fn.loc97_3.2: <specific function> = specific_impl_function %impl.elem0.loc97_3.2, @Inc.WithSelf.Op(%T.loc95_13.1) [symbolic = %specific_impl_fn.loc97_3.2 (constants.%specific_impl_fn.855)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: ref @TestInc.%T.as_type.loc95_33.1 (%T.as_type.232)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %x.ref: ref @TestInc.%T.as_type.loc95_33.1 (%T.as_type.232) = name_ref x, %x
// CHECK:STDOUT:     %impl.elem0.loc97_3.1: @TestInc.%.loc97_3.3 (%.a08) = impl_witness_access constants.%Inc.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc97_3.2 (constants.%impl.elem0.b92)]
// CHECK:STDOUT:     %bound_method.loc97_3.1: <bound method> = bound_method %x.ref, %impl.elem0.loc97_3.1
// CHECK:STDOUT:     %.loc97_3.1: %Inc.type = converted constants.%T.as_type.232, constants.%T.8f3 [symbolic = %T.loc95_13.1 (constants.%T.8f3)]
// CHECK:STDOUT:     %.loc97_3.2: %Inc.type = converted constants.%T.as_type.232, constants.%T.8f3 [symbolic = %T.loc95_13.1 (constants.%T.8f3)]
// CHECK:STDOUT:     %specific_impl_fn.loc97_3.1: <specific function> = specific_impl_function %impl.elem0.loc97_3.1, @Inc.WithSelf.Op(constants.%T.8f3) [symbolic = %specific_impl_fn.loc97_3.2 (constants.%specific_impl_fn.855)]
// CHECK:STDOUT:     %bound_method.loc97_3.2: <bound method> = bound_method %x.ref, %specific_impl_fn.loc97_3.1
// CHECK:STDOUT:     %Inc.WithSelf.Op.call: init %empty_tuple.type = call %bound_method.loc97_3.2(%x.ref)
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @TestDec(%T.loc101_13.2: %Dec.type) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %Dec.WithSelf.Op.type: type = fn_type @Dec.WithSelf.Op, @Dec.WithSelf(%T.loc101_13.1) [symbolic = %Dec.WithSelf.Op.type (constants.%Dec.WithSelf.Op.type.bfa830.2)]
// CHECK:STDOUT:   %.loc103_3.3: type = fn_type_with_self_type %Dec.WithSelf.Op.type, %T.loc101_13.1 [symbolic = %.loc103_3.3 (constants.%.aae)]
// CHECK:STDOUT:   %Dec.lookup_impl_witness: <witness> = lookup_impl_witness %T.loc101_13.1, @Dec [symbolic = %Dec.lookup_impl_witness (constants.%Dec.lookup_impl_witness)]
// CHECK:STDOUT:   %impl.elem0.loc103_3.2: @TestDec.%.loc103_3.3 (%.aae) = impl_witness_access %Dec.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc103_3.2 (constants.%impl.elem0.950)]
// CHECK:STDOUT:   %specific_impl_fn.loc103_3.2: <specific function> = specific_impl_function %impl.elem0.loc103_3.2, @Dec.WithSelf.Op(%T.loc101_13.1) [symbolic = %specific_impl_fn.loc103_3.2 (constants.%specific_impl_fn.539)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: ref @TestDec.%T.as_type.loc101_33.1 (%T.as_type.c32)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %x.ref: ref @TestDec.%T.as_type.loc101_33.1 (%T.as_type.c32) = name_ref x, %x
// CHECK:STDOUT:     %impl.elem0.loc103_3.1: @TestDec.%.loc103_3.3 (%.aae) = impl_witness_access constants.%Dec.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc103_3.2 (constants.%impl.elem0.950)]
// CHECK:STDOUT:     %bound_method.loc103_3.1: <bound method> = bound_method %x.ref, %impl.elem0.loc103_3.1
// CHECK:STDOUT:     %.loc103_3.1: %Dec.type = converted constants.%T.as_type.c32, constants.%T.e8e [symbolic = %T.loc101_13.1 (constants.%T.e8e)]
// CHECK:STDOUT:     %.loc103_3.2: %Dec.type = converted constants.%T.as_type.c32, constants.%T.e8e [symbolic = %T.loc101_13.1 (constants.%T.e8e)]
// CHECK:STDOUT:     %specific_impl_fn.loc103_3.1: <specific function> = specific_impl_function %impl.elem0.loc103_3.1, @Dec.WithSelf.Op(constants.%T.e8e) [symbolic = %specific_impl_fn.loc103_3.2 (constants.%specific_impl_fn.539)]
// CHECK:STDOUT:     %bound_method.loc103_3.2: <bound method> = bound_method %x.ref, %specific_impl_fn.loc103_3.1
// CHECK:STDOUT:     %Dec.WithSelf.Op.call: init %empty_tuple.type = call %bound_method.loc103_3.2(%x.ref)
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TestAddWith(constants.%U, constants.%T.96a) {
// CHECK:STDOUT:   %U.patt.loc15_4.2 => constants.%U.patt
// CHECK:STDOUT:   %U.loc15_4.1 => constants.%U
// CHECK:STDOUT:   %AddWith.type.loc16_21.1 => constants.%AddWith.type.7e967c.2
// CHECK:STDOUT:   %.Self.loc16_23.1 => constants.%.Self.f3c
// CHECK:STDOUT:   %require_complete.loc16 => constants.%require_complete.1a1
// CHECK:STDOUT:   %.Self.as_type.loc16_29.1 => constants.%.Self.as_type.806
// CHECK:STDOUT:   %AddWith.assoc_type => constants.%AddWith.assoc_type.588227.2
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.f393f4.2
// CHECK:STDOUT:   %AddWith.lookup_impl_witness.loc16 => constants.%AddWith.lookup_impl_witness.a82
// CHECK:STDOUT:   %impl.elem0.loc16_29.1 => constants.%impl.elem0.c7a
// CHECK:STDOUT:   %AddWith_where.type => constants.%AddWith_where.type.7e9
// CHECK:STDOUT:   %pattern_type.loc16 => constants.%pattern_type.afc
// CHECK:STDOUT:   %T.patt.loc16_4.2 => constants.%T.patt.083
// CHECK:STDOUT:   %T.loc16_4.1 => constants.%T.96a
// CHECK:STDOUT:   %T.as_type.loc17_6.1 => constants.%T.as_type.331
// CHECK:STDOUT:   %pattern_type.loc17_4 => constants.%pattern_type.aed556.2
// CHECK:STDOUT:   %x.param_patt.loc17_4.2 => constants.%x.param_patt.373
// CHECK:STDOUT:   %x.patt.loc17_4.2 => constants.%x.patt.f8d
// CHECK:STDOUT:   %pattern_type.loc17_10 => constants.%pattern_type.51d1c4.2
// CHECK:STDOUT:   %y.param_patt.loc17_10.2 => constants.%y.param_patt
// CHECK:STDOUT:   %y.patt.loc17_10.2 => constants.%y.patt
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TestAddAssignWith(constants.%U, constants.%T.792) {
// CHECK:STDOUT:   %U.patt.loc23_23.2 => constants.%U.patt
// CHECK:STDOUT:   %U.loc23_23.1 => constants.%U
// CHECK:STDOUT:   %AddAssignWith.type.loc23_56.1 => constants.%AddAssignWith.type.2c8890.2
// CHECK:STDOUT:   %pattern_type.loc23_33 => constants.%pattern_type.82d
// CHECK:STDOUT:   %T.patt.loc23_33.2 => constants.%T.patt.a22
// CHECK:STDOUT:   %T.loc23_33.1 => constants.%T.792
// CHECK:STDOUT:   %T.as_type.loc23_66.1 => constants.%T.as_type.50c
// CHECK:STDOUT:   %pattern_type.loc23_64 => constants.%pattern_type.643306.2
// CHECK:STDOUT:   %x.patt.loc23_64.2 => constants.%x.patt.227
// CHECK:STDOUT:   %x.param_patt.loc23_59.2 => constants.%x.param_patt.343
// CHECK:STDOUT:   %pattern_type.loc23_70 => constants.%pattern_type.51d1c4.2
// CHECK:STDOUT:   %y.param_patt.loc23_70.2 => constants.%y.param_patt
// CHECK:STDOUT:   %y.patt.loc23_70.2 => constants.%y.patt
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TestSubWith(constants.%U, constants.%T.d6b) {
// CHECK:STDOUT:   %U.patt.loc30_4.2 => constants.%U.patt
// CHECK:STDOUT:   %U.loc30_4.1 => constants.%U
// CHECK:STDOUT:   %SubWith.type.loc31_21.1 => constants.%SubWith.type.ddc625.2
// CHECK:STDOUT:   %.Self.loc31_23.1 => constants.%.Self.b46
// CHECK:STDOUT:   %require_complete.loc31 => constants.%require_complete.3b3
// CHECK:STDOUT:   %.Self.as_type.loc31_29.1 => constants.%.Self.as_type.10d
// CHECK:STDOUT:   %SubWith.assoc_type => constants.%SubWith.assoc_type.b0beca.2
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.98d9fa.2
// CHECK:STDOUT:   %SubWith.lookup_impl_witness.loc31 => constants.%SubWith.lookup_impl_witness.80b
// CHECK:STDOUT:   %impl.elem0.loc31_29.1 => constants.%impl.elem0.2a4
// CHECK:STDOUT:   %SubWith_where.type => constants.%SubWith_where.type.ddc
// CHECK:STDOUT:   %pattern_type.loc31 => constants.%pattern_type.a77
// CHECK:STDOUT:   %T.patt.loc31_4.2 => constants.%T.patt.121
// CHECK:STDOUT:   %T.loc31_4.1 => constants.%T.d6b
// CHECK:STDOUT:   %T.as_type.loc32_6.1 => constants.%T.as_type.449
// CHECK:STDOUT:   %pattern_type.loc32_4 => constants.%pattern_type.f1e08b.2
// CHECK:STDOUT:   %x.param_patt.loc32_4.2 => constants.%x.param_patt.725
// CHECK:STDOUT:   %x.patt.loc32_4.2 => constants.%x.patt.8e3
// CHECK:STDOUT:   %pattern_type.loc32_10 => constants.%pattern_type.51d1c4.2
// CHECK:STDOUT:   %y.param_patt.loc32_10.2 => constants.%y.param_patt
// CHECK:STDOUT:   %y.patt.loc32_10.2 => constants.%y.patt
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TestSubAssignWith(constants.%U, constants.%T.f6c) {
// CHECK:STDOUT:   %U.patt.loc38_23.2 => constants.%U.patt
// CHECK:STDOUT:   %U.loc38_23.1 => constants.%U
// CHECK:STDOUT:   %SubAssignWith.type.loc38_56.1 => constants.%SubAssignWith.type.7e53cf.2
// CHECK:STDOUT:   %pattern_type.loc38_33 => constants.%pattern_type.260
// CHECK:STDOUT:   %T.patt.loc38_33.2 => constants.%T.patt.f45
// CHECK:STDOUT:   %T.loc38_33.1 => constants.%T.f6c
// CHECK:STDOUT:   %T.as_type.loc38_66.1 => constants.%T.as_type.3ff
// CHECK:STDOUT:   %pattern_type.loc38_64 => constants.%pattern_type.f08ed5.2
// CHECK:STDOUT:   %x.patt.loc38_64.2 => constants.%x.patt.95a
// CHECK:STDOUT:   %x.param_patt.loc38_59.2 => constants.%x.param_patt.a45
// CHECK:STDOUT:   %pattern_type.loc38_70 => constants.%pattern_type.51d1c4.2
// CHECK:STDOUT:   %y.param_patt.loc38_70.2 => constants.%y.param_patt
// CHECK:STDOUT:   %y.patt.loc38_70.2 => constants.%y.patt
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TestMulWith(constants.%U, constants.%T.941) {
// CHECK:STDOUT:   %U.patt.loc45_4.2 => constants.%U.patt
// CHECK:STDOUT:   %U.loc45_4.1 => constants.%U
// CHECK:STDOUT:   %MulWith.type.loc46_21.1 => constants.%MulWith.type.f84f0b.2
// CHECK:STDOUT:   %.Self.loc46_23.1 => constants.%.Self.059
// CHECK:STDOUT:   %require_complete.loc46 => constants.%require_complete.061
// CHECK:STDOUT:   %.Self.as_type.loc46_29.1 => constants.%.Self.as_type.bed
// CHECK:STDOUT:   %MulWith.assoc_type => constants.%MulWith.assoc_type.7b2892.2
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.2b8b9e.2
// CHECK:STDOUT:   %MulWith.lookup_impl_witness.loc46 => constants.%MulWith.lookup_impl_witness.997
// CHECK:STDOUT:   %impl.elem0.loc46_29.1 => constants.%impl.elem0.529
// CHECK:STDOUT:   %MulWith_where.type => constants.%MulWith_where.type.f84
// CHECK:STDOUT:   %pattern_type.loc46 => constants.%pattern_type.a11
// CHECK:STDOUT:   %T.patt.loc46_4.2 => constants.%T.patt.8ab
// CHECK:STDOUT:   %T.loc46_4.1 => constants.%T.941
// CHECK:STDOUT:   %T.as_type.loc47_6.1 => constants.%T.as_type.ad2
// CHECK:STDOUT:   %pattern_type.loc47_4 => constants.%pattern_type.eccc2c.2
// CHECK:STDOUT:   %x.param_patt.loc47_4.2 => constants.%x.param_patt.114
// CHECK:STDOUT:   %x.patt.loc47_4.2 => constants.%x.patt.596
// CHECK:STDOUT:   %pattern_type.loc47_10 => constants.%pattern_type.51d1c4.2
// CHECK:STDOUT:   %y.param_patt.loc47_10.2 => constants.%y.param_patt
// CHECK:STDOUT:   %y.patt.loc47_10.2 => constants.%y.patt
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TestMulAssignWith(constants.%U, constants.%T.64d) {
// CHECK:STDOUT:   %U.patt.loc53_23.2 => constants.%U.patt
// CHECK:STDOUT:   %U.loc53_23.1 => constants.%U
// CHECK:STDOUT:   %MulAssignWith.type.loc53_56.1 => constants.%MulAssignWith.type.3a0e4d.2
// CHECK:STDOUT:   %pattern_type.loc53_33 => constants.%pattern_type.8c2
// CHECK:STDOUT:   %T.patt.loc53_33.2 => constants.%T.patt.db5
// CHECK:STDOUT:   %T.loc53_33.1 => constants.%T.64d
// CHECK:STDOUT:   %T.as_type.loc53_66.1 => constants.%T.as_type.1de
// CHECK:STDOUT:   %pattern_type.loc53_64 => constants.%pattern_type.8bf366.2
// CHECK:STDOUT:   %x.patt.loc53_64.2 => constants.%x.patt.4e9
// CHECK:STDOUT:   %x.param_patt.loc53_59.2 => constants.%x.param_patt.6c8
// CHECK:STDOUT:   %pattern_type.loc53_70 => constants.%pattern_type.51d1c4.2
// CHECK:STDOUT:   %y.param_patt.loc53_70.2 => constants.%y.param_patt
// CHECK:STDOUT:   %y.patt.loc53_70.2 => constants.%y.patt
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TestDivWith(constants.%U, constants.%T.16a) {
// CHECK:STDOUT:   %U.patt.loc60_4.2 => constants.%U.patt
// CHECK:STDOUT:   %U.loc60_4.1 => constants.%U
// CHECK:STDOUT:   %DivWith.type.loc61_21.1 => constants.%DivWith.type.13f8f2.2
// CHECK:STDOUT:   %.Self.loc61_23.1 => constants.%.Self.5d9
// CHECK:STDOUT:   %require_complete.loc61 => constants.%require_complete.975
// CHECK:STDOUT:   %.Self.as_type.loc61_29.1 => constants.%.Self.as_type.123
// CHECK:STDOUT:   %DivWith.assoc_type => constants.%DivWith.assoc_type.d8b290.2
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.b84b27.2
// CHECK:STDOUT:   %DivWith.lookup_impl_witness.loc61 => constants.%DivWith.lookup_impl_witness.c7d
// CHECK:STDOUT:   %impl.elem0.loc61_29.1 => constants.%impl.elem0.339
// CHECK:STDOUT:   %DivWith_where.type => constants.%DivWith_where.type.13f
// CHECK:STDOUT:   %pattern_type.loc61 => constants.%pattern_type.f4a
// CHECK:STDOUT:   %T.patt.loc61_4.2 => constants.%T.patt.97c
// CHECK:STDOUT:   %T.loc61_4.1 => constants.%T.16a
// CHECK:STDOUT:   %T.as_type.loc62_6.1 => constants.%T.as_type.f1d
// CHECK:STDOUT:   %pattern_type.loc62_4 => constants.%pattern_type.b25246.2
// CHECK:STDOUT:   %x.param_patt.loc62_4.2 => constants.%x.param_patt.494
// CHECK:STDOUT:   %x.patt.loc62_4.2 => constants.%x.patt.8ae
// CHECK:STDOUT:   %pattern_type.loc62_10 => constants.%pattern_type.51d1c4.2
// CHECK:STDOUT:   %y.param_patt.loc62_10.2 => constants.%y.param_patt
// CHECK:STDOUT:   %y.patt.loc62_10.2 => constants.%y.patt
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TestDivAssignWith(constants.%U, constants.%T.c33) {
// CHECK:STDOUT:   %U.patt.loc68_23.2 => constants.%U.patt
// CHECK:STDOUT:   %U.loc68_23.1 => constants.%U
// CHECK:STDOUT:   %DivAssignWith.type.loc68_56.1 => constants.%DivAssignWith.type.7e58ca.2
// CHECK:STDOUT:   %pattern_type.loc68_33 => constants.%pattern_type.f74
// CHECK:STDOUT:   %T.patt.loc68_33.2 => constants.%T.patt.9a5
// CHECK:STDOUT:   %T.loc68_33.1 => constants.%T.c33
// CHECK:STDOUT:   %T.as_type.loc68_66.1 => constants.%T.as_type.e57
// CHECK:STDOUT:   %pattern_type.loc68_64 => constants.%pattern_type.da303b.2
// CHECK:STDOUT:   %x.patt.loc68_64.2 => constants.%x.patt.cf6
// CHECK:STDOUT:   %x.param_patt.loc68_59.2 => constants.%x.param_patt.660
// CHECK:STDOUT:   %pattern_type.loc68_70 => constants.%pattern_type.51d1c4.2
// CHECK:STDOUT:   %y.param_patt.loc68_70.2 => constants.%y.param_patt
// CHECK:STDOUT:   %y.patt.loc68_70.2 => constants.%y.patt
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TestModWith(constants.%U, constants.%T.0ef) {
// CHECK:STDOUT:   %U.patt.loc75_4.2 => constants.%U.patt
// CHECK:STDOUT:   %U.loc75_4.1 => constants.%U
// CHECK:STDOUT:   %ModWith.type.loc76_21.1 => constants.%ModWith.type.2e82f9.2
// CHECK:STDOUT:   %.Self.loc76_23.1 => constants.%.Self.bef
// CHECK:STDOUT:   %require_complete.loc76 => constants.%require_complete.cd6
// CHECK:STDOUT:   %.Self.as_type.loc76_29.1 => constants.%.Self.as_type.538
// CHECK:STDOUT:   %ModWith.assoc_type => constants.%ModWith.assoc_type.316ae9.2
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.624dd7.2
// CHECK:STDOUT:   %ModWith.lookup_impl_witness.loc76 => constants.%ModWith.lookup_impl_witness.6e9
// CHECK:STDOUT:   %impl.elem0.loc76_29.1 => constants.%impl.elem0.488
// CHECK:STDOUT:   %ModWith_where.type => constants.%ModWith_where.type.2e8
// CHECK:STDOUT:   %pattern_type.loc76 => constants.%pattern_type.7c6
// CHECK:STDOUT:   %T.patt.loc76_4.2 => constants.%T.patt.520
// CHECK:STDOUT:   %T.loc76_4.1 => constants.%T.0ef
// CHECK:STDOUT:   %T.as_type.loc77_6.1 => constants.%T.as_type.c2b
// CHECK:STDOUT:   %pattern_type.loc77_4 => constants.%pattern_type.d76a9b.2
// CHECK:STDOUT:   %x.param_patt.loc77_4.2 => constants.%x.param_patt.3d2
// CHECK:STDOUT:   %x.patt.loc77_4.2 => constants.%x.patt.58e
// CHECK:STDOUT:   %pattern_type.loc77_10 => constants.%pattern_type.51d1c4.2
// CHECK:STDOUT:   %y.param_patt.loc77_10.2 => constants.%y.param_patt
// CHECK:STDOUT:   %y.patt.loc77_10.2 => constants.%y.patt
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TestModAssignWith(constants.%U, constants.%T.bca) {
// CHECK:STDOUT:   %U.patt.loc83_23.2 => constants.%U.patt
// CHECK:STDOUT:   %U.loc83_23.1 => constants.%U
// CHECK:STDOUT:   %ModAssignWith.type.loc83_56.1 => constants.%ModAssignWith.type.778d68.2
// CHECK:STDOUT:   %pattern_type.loc83_33 => constants.%pattern_type.fb8
// CHECK:STDOUT:   %T.patt.loc83_33.2 => constants.%T.patt.c29
// CHECK:STDOUT:   %T.loc83_33.1 => constants.%T.bca
// CHECK:STDOUT:   %T.as_type.loc83_66.1 => constants.%T.as_type.ce6
// CHECK:STDOUT:   %pattern_type.loc83_64 => constants.%pattern_type.1d5f30.2
// CHECK:STDOUT:   %x.patt.loc83_64.2 => constants.%x.patt.fa4
// CHECK:STDOUT:   %x.param_patt.loc83_59.2 => constants.%x.param_patt.749
// CHECK:STDOUT:   %pattern_type.loc83_70 => constants.%pattern_type.51d1c4.2
// CHECK:STDOUT:   %y.param_patt.loc83_70.2 => constants.%y.param_patt
// CHECK:STDOUT:   %y.patt.loc83_70.2 => constants.%y.patt
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TestNegate(constants.%T.0d1) {
// CHECK:STDOUT:   %T.patt.loc89_16.2 => constants.%T.patt.91c
// CHECK:STDOUT:   %T.loc89_16.1 => constants.%T.0d1
// CHECK:STDOUT:   %T.as_type.loc89_68.1 => constants.%T.as_type.db1
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.398aef.2
// CHECK:STDOUT:   %x.param_patt.loc89_66.2 => constants.%x.param_patt.5d8
// CHECK:STDOUT:   %x.patt.loc89_66.2 => constants.%x.patt.e11
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TestInc(constants.%T.8f3) {
// CHECK:STDOUT:   %T.patt.loc95_13.2 => constants.%T.patt.eab
// CHECK:STDOUT:   %T.loc95_13.1 => constants.%T.8f3
// CHECK:STDOUT:   %T.as_type.loc95_33.1 => constants.%T.as_type.232
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.427bfc.2
// CHECK:STDOUT:   %x.patt.loc95_31.2 => constants.%x.patt.a75
// CHECK:STDOUT:   %x.param_patt.loc95_26.2 => constants.%x.param_patt.5da
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TestDec(constants.%T.e8e) {
// CHECK:STDOUT:   %T.patt.loc101_13.2 => constants.%T.patt.de9
// CHECK:STDOUT:   %T.loc101_13.1 => constants.%T.e8e
// CHECK:STDOUT:   %T.as_type.loc101_33.1 => constants.%T.as_type.c32
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.2754f0.2
// CHECK:STDOUT:   %x.patt.loc101_31.2 => constants.%x.patt.08b
// CHECK:STDOUT:   %x.param_patt.loc101_26.2 => constants.%x.param_patt.678
// CHECK:STDOUT: }
// CHECK:STDOUT:
