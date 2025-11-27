// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/full.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/primitives/import_symbolic.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/primitives/import_symbolic.carbon
// CHECK:STDERR: {{.*}}/prelude/iterate.carbon:23:3: error: cannot implicitly convert expression of type `Int(32)` to `array(T, N).(Iterate.CursorType)` [ConversionFailure]
// CHECK:STDERR:   fn NewCursor[self: Self]() -> i32 { return 0; }
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: {{.*}}/prelude/iterate.carbon:23:3: note: type `Int(32)` does not implement interface `ImplicitAs(array(T, N).(Iterate.CursorType))` [MissingImplInMemberAccessNote]
// CHECK:STDERR:   fn NewCursor[self: Self]() -> i32 { return 0; }
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: {{.*}}/prelude/iterate.carbon:16:3: note: while building thunk to match the signature of this function [ThunkSignature]
// CHECK:STDERR:   fn NewCursor[self: Self]() -> CursorType;
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: {{.*}}/prelude/iterate.carbon:24:3: error: cannot implicitly convert expression of type `array(T, N).(Iterate.CursorType)*` to `Int(32)*` [ConversionFailure]
// CHECK:STDERR:   fn Next[self: Self](cursor: i32*) -> Optional(T) {
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: {{.*}}/prelude/iterate.carbon:24:3: note: type `array(T, N).(Iterate.CursorType)*` does not implement interface `ImplicitAs(Int(32)*)` [MissingImplInMemberAccessNote]
// CHECK:STDERR:   fn Next[self: Self](cursor: i32*) -> Optional(T) {
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: {{.*}}/prelude/iterate.carbon:24:3: note: while deducing parameters of generic declared here [DeductionGenericHere]
// CHECK:STDERR:   fn Next[self: Self](cursor: i32*) -> Optional(T) {
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: {{.*}}/prelude/iterate.carbon:17:3: note: while building thunk to match the signature of this function [ThunkSignature]
// CHECK:STDERR:   fn Next[self: Self](cursor: CursorType*) -> Optional(ElementType);
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:

// --- bool.carbon

library "[[@TEST_NAME]]";

interface I {
  let val:! bool;
}

class C {
  impl as I where .val = false {}
}

// --- import_bool.carbon

library "[[@TEST_NAME]]";
import library "bool";

fn F() -> bool {
  //@dump-sem-ir-begin
  return (C as I).val;
  //@dump-sem-ir-end
}

// --- char.carbon

library "[[@TEST_NAME]]";

interface I {
  let val:! Core.Char;
}

class C {}

impl C as I where .val = 'a' {}

// --- import_char.carbon

library "[[@TEST_NAME]]";
import library "char";

fn F() -> Core.Char {
  //@dump-sem-ir-begin
  return (C as I).val;
  //@dump-sem-ir-end
}

// --- char_literal.carbon

library "[[@TEST_NAME]]";

interface I {
  let val:! Core.CharLiteral();
}

class C {}

impl C as I where .val = 'a' {}

// --- import_char_literal.carbon

library "[[@TEST_NAME]]";
import library "char_literal";

fn F() -> Core.CharLiteral() {
  //@dump-sem-ir-begin
  return (C as I).val;
  //@dump-sem-ir-end
}

// --- float.carbon

library "[[@TEST_NAME]]";

interface I {
  let val:! f64;
}

class C {}

impl C as I where .val = 0.1 {}

// --- import_float.carbon

library "[[@TEST_NAME]]";
import library "float";

fn F() -> f64 {
  //@dump-sem-ir-begin
  return (C as I).val;
  //@dump-sem-ir-end
}

// --- float_literal.carbon

library "[[@TEST_NAME]]";

interface I {
  let val:! Core.FloatLiteral();
}

class C {}

impl C as I where .val = 0.1 {}

// --- import_float_literal.carbon

library "[[@TEST_NAME]]";
import library "float_literal";

fn F() -> Core.FloatLiteral() {
  //@dump-sem-ir-begin
  return (C as I).val;
  //@dump-sem-ir-end
}

// --- int.carbon

library "[[@TEST_NAME]]";

interface I {
  let val:! i32;
}

class C {}

impl C as I where .val = 8 {}

// CHECK:STDERR: {{.*}}/prelude/types/int.carbon:[[@LINE+265]]:3: error: cannot implicitly convert expression of type `Int(N)` to `Int(N).(AddWith(U).Result)` [ConversionFailure]
// CHECK:STDERR:   fn Op[self: Int(N)](other: Int(N)) -> Int(N) = "int.sadd";
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: {{.*}}/prelude/types/int.carbon:[[@LINE+262]]:3: note: type `Int(N)` does not implement interface `ImplicitAs(Int(N).(AddWith(U).Result))` [MissingImplInMemberAccessNote]
// CHECK:STDERR:   fn Op[self: Int(N)](other: Int(N)) -> Int(N) = "int.sadd";
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: {{.*}}/prelude/types/int.carbon:[[@LINE-11]]:1: in import [InImport]
// CHECK:STDERR: {{.*}}/prelude/operators/arithmetic.carbon:24:3: note: while building thunk to match the signature of this function [ThunkSignature]
// CHECK:STDERR:   fn Op[self: Self](other: Other) -> Result;
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: {{.*}}/prelude/types/int.carbon:[[@LINE+254]]:3: error: cannot implicitly convert expression of type `Int(N)` to `Int(N).(DivWith(U).Result)` [ConversionFailure]
// CHECK:STDERR:   fn Op[self: Int(N)](other: Int(N)) -> Int(N) = "int.sdiv";
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: {{.*}}/prelude/types/int.carbon:[[@LINE+251]]:3: note: type `Int(N)` does not implement interface `ImplicitAs(Int(N).(DivWith(U).Result))` [MissingImplInMemberAccessNote]
// CHECK:STDERR:   fn Op[self: Int(N)](other: Int(N)) -> Int(N) = "int.sdiv";
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: {{.*}}/prelude/types/int.carbon:[[@LINE-22]]:1: in import [InImport]
// CHECK:STDERR: {{.*}}/prelude/operators/arithmetic.carbon:73:3: note: while building thunk to match the signature of this function [ThunkSignature]
// CHECK:STDERR:   fn Op[self: Self](other: Other) -> Result;
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: {{.*}}/prelude/types/int.carbon:[[@LINE+243]]:3: error: cannot implicitly convert expression of type `Int(N)` to `Int(N).(ModWith(U).Result)` [ConversionFailure]
// CHECK:STDERR:   fn Op[self: Int(N)](other: Int(N)) -> Int(N) = "int.smod";
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: {{.*}}/prelude/types/int.carbon:[[@LINE+240]]:3: note: type `Int(N)` does not implement interface `ImplicitAs(Int(N).(ModWith(U).Result))` [MissingImplInMemberAccessNote]
// CHECK:STDERR:   fn Op[self: Int(N)](other: Int(N)) -> Int(N) = "int.smod";
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: {{.*}}/prelude/types/int.carbon:[[@LINE-33]]:1: in import [InImport]
// CHECK:STDERR: {{.*}}/prelude/operators/arithmetic.carbon:84:3: note: while building thunk to match the signature of this function [ThunkSignature]
// CHECK:STDERR:   fn Op[self: Self](other: Other) -> Result;
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: {{.*}}/prelude/types/int.carbon:[[@LINE+232]]:3: error: cannot implicitly convert expression of type `Int(N)` to `Int(N).(MulWith(U).Result)` [ConversionFailure]
// CHECK:STDERR:   fn Op[self: Int(N)](other: Int(N)) -> Int(N) = "int.smul";
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: {{.*}}/prelude/types/int.carbon:[[@LINE+229]]:3: note: type `Int(N)` does not implement interface `ImplicitAs(Int(N).(MulWith(U).Result))` [MissingImplInMemberAccessNote]
// CHECK:STDERR:   fn Op[self: Int(N)](other: Int(N)) -> Int(N) = "int.smul";
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: {{.*}}/prelude/types/int.carbon:[[@LINE-44]]:1: in import [InImport]
// CHECK:STDERR: {{.*}}/prelude/operators/arithmetic.carbon:62:3: note: while building thunk to match the signature of this function [ThunkSignature]
// CHECK:STDERR:   fn Op[self: Self](other: Other) -> Result;
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: {{.*}}/prelude/types/int.carbon:[[@LINE+221]]:3: error: cannot implicitly convert expression of type `Int(N)` to `Int(N).(SubWith(U).Result)` [ConversionFailure]
// CHECK:STDERR:   fn Op[self: Int(N)](other: Int(N)) -> Int(N) = "int.ssub";
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: {{.*}}/prelude/types/int.carbon:[[@LINE+218]]:3: note: type `Int(N)` does not implement interface `ImplicitAs(Int(N).(SubWith(U).Result))` [MissingImplInMemberAccessNote]
// CHECK:STDERR:   fn Op[self: Int(N)](other: Int(N)) -> Int(N) = "int.ssub";
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: {{.*}}/prelude/types/int.carbon:[[@LINE-55]]:1: in import [InImport]
// CHECK:STDERR: {{.*}}/prelude/operators/arithmetic.carbon:46:3: note: while building thunk to match the signature of this function [ThunkSignature]
// CHECK:STDERR:   fn Op[self: Self](other: Other) -> Result;
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: {{.*}}/prelude/types/int.carbon:[[@LINE+210]]:3: error: cannot implicitly convert expression of type `Int(N)` to `T.(AddWith(Int(N)).Result)` [ConversionFailure]
// CHECK:STDERR:   fn Op[self: Int(N)](other: Int(N)) -> Int(N) = "int.sadd";
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: {{.*}}/prelude/types/int.carbon:[[@LINE+207]]:3: note: type `Int(N)` does not implement interface `ImplicitAs(T.(AddWith(Int(N)).Result))` [MissingImplInMemberAccessNote]
// CHECK:STDERR:   fn Op[self: Int(N)](other: Int(N)) -> Int(N) = "int.sadd";
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: {{.*}}/prelude/types/int.carbon:[[@LINE-66]]:1: in import [InImport]
// CHECK:STDERR: {{.*}}/prelude/operators/arithmetic.carbon:24:3: note: while building thunk to match the signature of this function [ThunkSignature]
// CHECK:STDERR:   fn Op[self: Self](other: Other) -> Result;
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: {{.*}}/prelude/types/int.carbon:[[@LINE+199]]:3: error: cannot implicitly convert expression of type `Int(N)` to `T.(DivWith(Int(N)).Result)` [ConversionFailure]
// CHECK:STDERR:   fn Op[self: Int(N)](other: Int(N)) -> Int(N) = "int.sdiv";
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: {{.*}}/prelude/types/int.carbon:[[@LINE+196]]:3: note: type `Int(N)` does not implement interface `ImplicitAs(T.(DivWith(Int(N)).Result))` [MissingImplInMemberAccessNote]
// CHECK:STDERR:   fn Op[self: Int(N)](other: Int(N)) -> Int(N) = "int.sdiv";
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: {{.*}}/prelude/types/int.carbon:[[@LINE-77]]:1: in import [InImport]
// CHECK:STDERR: {{.*}}/prelude/operators/arithmetic.carbon:73:3: note: while building thunk to match the signature of this function [ThunkSignature]
// CHECK:STDERR:   fn Op[self: Self](other: Other) -> Result;
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: {{.*}}/prelude/types/int.carbon:[[@LINE+188]]:3: error: cannot implicitly convert expression of type `Int(N)` to `T.(ModWith(Int(N)).Result)` [ConversionFailure]
// CHECK:STDERR:   fn Op[self: Int(N)](other: Int(N)) -> Int(N) = "int.smod";
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: {{.*}}/prelude/types/int.carbon:[[@LINE+185]]:3: note: type `Int(N)` does not implement interface `ImplicitAs(T.(ModWith(Int(N)).Result))` [MissingImplInMemberAccessNote]
// CHECK:STDERR:   fn Op[self: Int(N)](other: Int(N)) -> Int(N) = "int.smod";
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: {{.*}}/prelude/types/int.carbon:[[@LINE-88]]:1: in import [InImport]
// CHECK:STDERR: {{.*}}/prelude/operators/arithmetic.carbon:84:3: note: while building thunk to match the signature of this function [ThunkSignature]
// CHECK:STDERR:   fn Op[self: Self](other: Other) -> Result;
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: {{.*}}/prelude/types/int.carbon:[[@LINE+177]]:3: error: cannot implicitly convert expression of type `Int(N)` to `T.(MulWith(Int(N)).Result)` [ConversionFailure]
// CHECK:STDERR:   fn Op[self: Int(N)](other: Int(N)) -> Int(N) = "int.smul";
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: {{.*}}/prelude/types/int.carbon:[[@LINE+174]]:3: note: type `Int(N)` does not implement interface `ImplicitAs(T.(MulWith(Int(N)).Result))` [MissingImplInMemberAccessNote]
// CHECK:STDERR:   fn Op[self: Int(N)](other: Int(N)) -> Int(N) = "int.smul";
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: {{.*}}/prelude/types/int.carbon:[[@LINE-99]]:1: in import [InImport]
// CHECK:STDERR: {{.*}}/prelude/operators/arithmetic.carbon:62:3: note: while building thunk to match the signature of this function [ThunkSignature]
// CHECK:STDERR:   fn Op[self: Self](other: Other) -> Result;
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: {{.*}}/prelude/types/int.carbon:[[@LINE+166]]:3: error: cannot implicitly convert expression of type `Int(N)` to `T.(SubWith(Int(N)).Result)` [ConversionFailure]
// CHECK:STDERR:   fn Op[self: Int(N)](other: Int(N)) -> Int(N) = "int.ssub";
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: {{.*}}/prelude/types/int.carbon:[[@LINE+163]]:3: note: type `Int(N)` does not implement interface `ImplicitAs(T.(SubWith(Int(N)).Result))` [MissingImplInMemberAccessNote]
// CHECK:STDERR:   fn Op[self: Int(N)](other: Int(N)) -> Int(N) = "int.ssub";
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: {{.*}}/prelude/types/int.carbon:[[@LINE-110]]:1: in import [InImport]
// CHECK:STDERR: {{.*}}/prelude/operators/arithmetic.carbon:46:3: note: while building thunk to match the signature of this function [ThunkSignature]
// CHECK:STDERR:   fn Op[self: Self](other: Other) -> Result;
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: {{.*}}/prelude/types/int.carbon:[[@LINE+155]]:3: error: cannot implicitly convert expression of type `Int(N)` to `Int(N).(BitAndWith(U).Result)` [ConversionFailure]
// CHECK:STDERR:   fn Op[self: Int(N)](other: Int(N)) -> Int(N) = "int.and";
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: {{.*}}/prelude/types/int.carbon:[[@LINE+152]]:3: note: type `Int(N)` does not implement interface `ImplicitAs(Int(N).(BitAndWith(U).Result))` [MissingImplInMemberAccessNote]
// CHECK:STDERR:   fn Op[self: Int(N)](other: Int(N)) -> Int(N) = "int.and";
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: {{.*}}/prelude/types/int.carbon:[[@LINE-121]]:1: in import [InImport]
// CHECK:STDERR: {{.*}}/prelude/operators/bitwise.carbon:30:3: note: while building thunk to match the signature of this function [ThunkSignature]
// CHECK:STDERR:   fn Op[self: Self](other: Other) -> Result;
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: {{.*}}/prelude/types/int.carbon:[[@LINE+144]]:3: error: cannot implicitly convert expression of type `Int(N)` to `Int(N).(BitOrWith(U).Result)` [ConversionFailure]
// CHECK:STDERR:   fn Op[self: Int(N)](other: Int(N)) -> Int(N) = "int.or";
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: {{.*}}/prelude/types/int.carbon:[[@LINE+141]]:3: note: type `Int(N)` does not implement interface `ImplicitAs(Int(N).(BitOrWith(U).Result))` [MissingImplInMemberAccessNote]
// CHECK:STDERR:   fn Op[self: Int(N)](other: Int(N)) -> Int(N) = "int.or";
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: {{.*}}/prelude/types/int.carbon:[[@LINE-132]]:1: in import [InImport]
// CHECK:STDERR: {{.*}}/prelude/operators/bitwise.carbon:41:3: note: while building thunk to match the signature of this function [ThunkSignature]
// CHECK:STDERR:   fn Op[self: Self](other: Other) -> Result;
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: {{.*}}/prelude/types/int.carbon:[[@LINE+133]]:3: error: cannot implicitly convert expression of type `Int(N)` to `Int(N).(BitXorWith(U).Result)` [ConversionFailure]
// CHECK:STDERR:   fn Op[self: Int(N)](other: Int(N)) -> Int(N) = "int.xor";
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: {{.*}}/prelude/types/int.carbon:[[@LINE+130]]:3: note: type `Int(N)` does not implement interface `ImplicitAs(Int(N).(BitXorWith(U).Result))` [MissingImplInMemberAccessNote]
// CHECK:STDERR:   fn Op[self: Int(N)](other: Int(N)) -> Int(N) = "int.xor";
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: {{.*}}/prelude/types/int.carbon:[[@LINE-143]]:1: in import [InImport]
// CHECK:STDERR: {{.*}}/prelude/operators/bitwise.carbon:52:3: note: while building thunk to match the signature of this function [ThunkSignature]
// CHECK:STDERR:   fn Op[self: Self](other: Other) -> Result;
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: {{.*}}/prelude/types/int.carbon:[[@LINE+122]]:3: error: cannot implicitly convert expression of type `Int(N)` to `Int(N).(LeftShiftWith(U).Result)` [ConversionFailure]
// CHECK:STDERR:   fn Op[self: Int(N)](other: Int(N)) -> Int(N) = "int.left_shift";
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: {{.*}}/prelude/types/int.carbon:[[@LINE+119]]:3: note: type `Int(N)` does not implement interface `ImplicitAs(Int(N).(LeftShiftWith(U).Result))` [MissingImplInMemberAccessNote]
// CHECK:STDERR:   fn Op[self: Int(N)](other: Int(N)) -> Int(N) = "int.left_shift";
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: {{.*}}/prelude/types/int.carbon:[[@LINE-154]]:1: in import [InImport]
// CHECK:STDERR: {{.*}}/prelude/operators/bitwise.carbon:63:3: note: while building thunk to match the signature of this function [ThunkSignature]
// CHECK:STDERR:   fn Op[self: Self](other: Other) -> Result;
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: {{.*}}/prelude/types/int.carbon:[[@LINE+111]]:3: error: cannot implicitly convert expression of type `Int(N)` to `Int(N).(RightShiftWith(U).Result)` [ConversionFailure]
// CHECK:STDERR:   fn Op[self: Int(N)](other: Int(N)) -> Int(N) = "int.right_shift";
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: {{.*}}/prelude/types/int.carbon:[[@LINE+108]]:3: note: type `Int(N)` does not implement interface `ImplicitAs(Int(N).(RightShiftWith(U).Result))` [MissingImplInMemberAccessNote]
// CHECK:STDERR:   fn Op[self: Int(N)](other: Int(N)) -> Int(N) = "int.right_shift";
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: {{.*}}/prelude/types/int.carbon:[[@LINE-165]]:1: in import [InImport]
// CHECK:STDERR: {{.*}}/prelude/operators/bitwise.carbon:74:3: note: while building thunk to match the signature of this function [ThunkSignature]
// CHECK:STDERR:   fn Op[self: Self](other: Other) -> Result;
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: {{.*}}/prelude/types/int.carbon:[[@LINE+100]]:3: error: cannot implicitly convert expression of type `Int(N)` to `T.(BitAndWith(Int(N)).Result)` [ConversionFailure]
// CHECK:STDERR:   fn Op[self: Int(N)](other: Int(N)) -> Int(N) = "int.and";
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: {{.*}}/prelude/types/int.carbon:[[@LINE+97]]:3: note: type `Int(N)` does not implement interface `ImplicitAs(T.(BitAndWith(Int(N)).Result))` [MissingImplInMemberAccessNote]
// CHECK:STDERR:   fn Op[self: Int(N)](other: Int(N)) -> Int(N) = "int.and";
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: {{.*}}/prelude/types/int.carbon:[[@LINE-176]]:1: in import [InImport]
// CHECK:STDERR: {{.*}}/prelude/operators/bitwise.carbon:30:3: note: while building thunk to match the signature of this function [ThunkSignature]
// CHECK:STDERR:   fn Op[self: Self](other: Other) -> Result;
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: {{.*}}/prelude/types/int.carbon:[[@LINE+89]]:3: error: cannot implicitly convert expression of type `Int(N)` to `T.(BitOrWith(Int(N)).Result)` [ConversionFailure]
// CHECK:STDERR:   fn Op[self: Int(N)](other: Int(N)) -> Int(N) = "int.or";
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: {{.*}}/prelude/types/int.carbon:[[@LINE+86]]:3: note: type `Int(N)` does not implement interface `ImplicitAs(T.(BitOrWith(Int(N)).Result))` [MissingImplInMemberAccessNote]
// CHECK:STDERR:   fn Op[self: Int(N)](other: Int(N)) -> Int(N) = "int.or";
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: {{.*}}/prelude/types/int.carbon:[[@LINE-187]]:1: in import [InImport]
// CHECK:STDERR: {{.*}}/prelude/operators/bitwise.carbon:41:3: note: while building thunk to match the signature of this function [ThunkSignature]
// CHECK:STDERR:   fn Op[self: Self](other: Other) -> Result;
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: {{.*}}/prelude/types/int.carbon:[[@LINE+78]]:3: error: cannot implicitly convert expression of type `Int(N)` to `T.(BitXorWith(Int(N)).Result)` [ConversionFailure]
// CHECK:STDERR:   fn Op[self: Int(N)](other: Int(N)) -> Int(N) = "int.xor";
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: {{.*}}/prelude/types/int.carbon:[[@LINE+75]]:3: note: type `Int(N)` does not implement interface `ImplicitAs(T.(BitXorWith(Int(N)).Result))` [MissingImplInMemberAccessNote]
// CHECK:STDERR:   fn Op[self: Int(N)](other: Int(N)) -> Int(N) = "int.xor";
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: {{.*}}/prelude/types/int.carbon:[[@LINE-198]]:1: in import [InImport]
// CHECK:STDERR: {{.*}}/prelude/operators/bitwise.carbon:52:3: note: while building thunk to match the signature of this function [ThunkSignature]
// CHECK:STDERR:   fn Op[self: Self](other: Other) -> Result;
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: {{.*}}/prelude/types/int.carbon:[[@LINE+67]]:3: error: cannot implicitly convert expression of type `Int(N)` to `T.(LeftShiftWith(Int(N)).Result)` [ConversionFailure]
// CHECK:STDERR:   fn Op[self: Int(N)](other: Int(N)) -> Int(N) = "int.left_shift";
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: {{.*}}/prelude/types/int.carbon:[[@LINE+64]]:3: note: type `Int(N)` does not implement interface `ImplicitAs(T.(LeftShiftWith(Int(N)).Result))` [MissingImplInMemberAccessNote]
// CHECK:STDERR:   fn Op[self: Int(N)](other: Int(N)) -> Int(N) = "int.left_shift";
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: {{.*}}/prelude/types/int.carbon:[[@LINE-209]]:1: in import [InImport]
// CHECK:STDERR: {{.*}}/prelude/operators/bitwise.carbon:63:3: note: while building thunk to match the signature of this function [ThunkSignature]
// CHECK:STDERR:   fn Op[self: Self](other: Other) -> Result;
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: {{.*}}/prelude/types/int.carbon:[[@LINE+56]]:3: error: cannot implicitly convert expression of type `Int(N)` to `T.(RightShiftWith(Int(N)).Result)` [ConversionFailure]
// CHECK:STDERR:   fn Op[self: Int(N)](other: Int(N)) -> Int(N) = "int.right_shift";
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: {{.*}}/prelude/types/int.carbon:[[@LINE+53]]:3: note: type `Int(N)` does not implement interface `ImplicitAs(T.(RightShiftWith(Int(N)).Result))` [MissingImplInMemberAccessNote]
// CHECK:STDERR:   fn Op[self: Int(N)](other: Int(N)) -> Int(N) = "int.right_shift";
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: {{.*}}/prelude/types/int.carbon:[[@LINE-220]]:1: in import [InImport]
// CHECK:STDERR: {{.*}}/prelude/operators/bitwise.carbon:74:3: note: while building thunk to match the signature of this function [ThunkSignature]
// CHECK:STDERR:   fn Op[self: Self](other: Other) -> Result;
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: {{.*}}/prelude/types/optional.carbon:99:3: error: cannot implicitly convert expression of type `DefaultOptionalStorage(T)` to `T.(OptionalStorage.Type)` [ConversionFailure]
// CHECK:STDERR:   fn None() -> DefaultOptionalStorage(T) {
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: {{.*}}/prelude/types/optional.carbon:99:3: note: type `DefaultOptionalStorage(T)` does not implement interface `ImplicitAs(T.(OptionalStorage.Type))` [MissingImplInMemberAccessNote]
// CHECK:STDERR:   fn None() -> DefaultOptionalStorage(T) {
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: {{.*}}/prelude/types/optional.carbon:17:3: note: while building thunk to match the signature of this function [ThunkSignature]
// CHECK:STDERR:   fn None() -> Type;
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: {{.*}}/prelude/types/optional.carbon:104:3: error: cannot implicitly convert expression of type `DefaultOptionalStorage(T)` to `T.(OptionalStorage.Type)` [ConversionFailure]
// CHECK:STDERR:   fn Some[self: Self]() -> DefaultOptionalStorage(T) {
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: {{.*}}/prelude/types/optional.carbon:104:3: note: type `DefaultOptionalStorage(T)` does not implement interface `ImplicitAs(T.(OptionalStorage.Type))` [MissingImplInMemberAccessNote]
// CHECK:STDERR:   fn Some[self: Self]() -> DefaultOptionalStorage(T) {
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: {{.*}}/prelude/types/optional.carbon:18:3: note: while building thunk to match the signature of this function [ThunkSignature]
// CHECK:STDERR:   fn Some[self: Self]() -> Type;
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: {{.*}}/prelude/types/optional.carbon:19:10: error: cannot implicitly convert expression of type `T.(OptionalStorage.Type)` to `DefaultOptionalStorage(T)` [ConversionFailure]
// CHECK:STDERR:   fn Has(value: Type) -> bool;
// CHECK:STDERR:          ^~~~~~~~~~~
// CHECK:STDERR: {{.*}}/prelude/types/optional.carbon:19:10: note: type `T.(OptionalStorage.Type)` does not implement interface `ImplicitAs(DefaultOptionalStorage(T))` [MissingImplInMemberAccessNote]
// CHECK:STDERR:   fn Has(value: Type) -> bool;
// CHECK:STDERR:          ^~~~~~~~~~~
// CHECK:STDERR: {{.*}}/prelude/types/optional.carbon:112:10: note: initializing function parameter [InCallToFunctionParam]
// CHECK:STDERR:   fn Has(value: DefaultOptionalStorage(T)) -> bool {
// CHECK:STDERR:          ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: {{.*}}/prelude/types/optional.carbon:19:3: note: while building thunk to match the signature of this function [ThunkSignature]
// CHECK:STDERR:   fn Has(value: Type) -> bool;
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: {{.*}}/prelude/types/optional.carbon:20:10: error: cannot implicitly convert expression of type `T.(OptionalStorage.Type)` to `DefaultOptionalStorage(T)` [ConversionFailure]
// CHECK:STDERR:   fn Get(value: Type) -> Self;
// CHECK:STDERR:          ^~~~~~~~~~~
// CHECK:STDERR: {{.*}}/prelude/types/optional.carbon:20:10: note: type `T.(OptionalStorage.Type)` does not implement interface `ImplicitAs(DefaultOptionalStorage(T))` [MissingImplInMemberAccessNote]
// CHECK:STDERR:   fn Get(value: Type) -> Self;
// CHECK:STDERR:          ^~~~~~~~~~~
// CHECK:STDERR: {{.*}}/prelude/types/optional.carbon:115:10: note: initializing function parameter [InCallToFunctionParam]
// CHECK:STDERR:   fn Get(value: DefaultOptionalStorage(T)) -> T {
// CHECK:STDERR:          ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: {{.*}}/prelude/types/optional.carbon:20:3: note: while building thunk to match the signature of this function [ThunkSignature]
// CHECK:STDERR:   fn Get(value: Type) -> Self;
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
// --- import_int.carbon

library "[[@TEST_NAME]]";
import library "int";

fn F() -> i32 {
  //@dump-sem-ir-begin
  return (C as I).val;
  //@dump-sem-ir-end
}

// --- int_literal.carbon

library "[[@TEST_NAME]]";

interface I {
  let val:! Core.IntLiteral();
}

class C {}

impl C as I where .val = 7 {}

// --- import_int_literal.carbon

library "[[@TEST_NAME]]";
import library "int_literal";

fn F() -> Core.IntLiteral() {
  //@dump-sem-ir-begin
  return (C as I).val;
  //@dump-sem-ir-end
}

// --- uint.carbon

library "[[@TEST_NAME]]";

interface I {
  let val:! u32;
}

class C {}

impl C as I where .val = 8 {}

// CHECK:STDERR: {{.*}}/prelude/types/uint.carbon:[[@LINE+257]]:3: error: cannot implicitly convert expression of type `UInt(N)` to `UInt(N).(AddWith(U).Result)` [ConversionFailure]
// CHECK:STDERR:   fn Op[self: UInt(N)](other: UInt(N)) -> UInt(N) = "int.uadd";
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: {{.*}}/prelude/types/uint.carbon:[[@LINE+254]]:3: note: type `UInt(N)` does not implement interface `ImplicitAs(UInt(N).(AddWith(U).Result))` [MissingImplInMemberAccessNote]
// CHECK:STDERR:   fn Op[self: UInt(N)](other: UInt(N)) -> UInt(N) = "int.uadd";
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: {{.*}}/prelude/types/uint.carbon:[[@LINE-11]]:1: in import [InImport]
// CHECK:STDERR: {{.*}}/prelude/operators/arithmetic.carbon:24:3: note: while building thunk to match the signature of this function [ThunkSignature]
// CHECK:STDERR:   fn Op[self: Self](other: Other) -> Result;
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: {{.*}}/prelude/types/uint.carbon:[[@LINE+246]]:3: error: cannot implicitly convert expression of type `UInt(N)` to `UInt(N).(DivWith(U).Result)` [ConversionFailure]
// CHECK:STDERR:   fn Op[self: UInt(N)](other: UInt(N)) -> UInt(N) = "int.udiv";
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: {{.*}}/prelude/types/uint.carbon:[[@LINE+243]]:3: note: type `UInt(N)` does not implement interface `ImplicitAs(UInt(N).(DivWith(U).Result))` [MissingImplInMemberAccessNote]
// CHECK:STDERR:   fn Op[self: UInt(N)](other: UInt(N)) -> UInt(N) = "int.udiv";
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: {{.*}}/prelude/types/uint.carbon:[[@LINE-22]]:1: in import [InImport]
// CHECK:STDERR: {{.*}}/prelude/operators/arithmetic.carbon:73:3: note: while building thunk to match the signature of this function [ThunkSignature]
// CHECK:STDERR:   fn Op[self: Self](other: Other) -> Result;
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: {{.*}}/prelude/types/uint.carbon:[[@LINE+235]]:3: error: cannot implicitly convert expression of type `UInt(N)` to `UInt(N).(ModWith(U).Result)` [ConversionFailure]
// CHECK:STDERR:   fn Op[self: UInt(N)](other: UInt(N)) -> UInt(N) = "int.umod";
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: {{.*}}/prelude/types/uint.carbon:[[@LINE+232]]:3: note: type `UInt(N)` does not implement interface `ImplicitAs(UInt(N).(ModWith(U).Result))` [MissingImplInMemberAccessNote]
// CHECK:STDERR:   fn Op[self: UInt(N)](other: UInt(N)) -> UInt(N) = "int.umod";
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: {{.*}}/prelude/types/uint.carbon:[[@LINE-33]]:1: in import [InImport]
// CHECK:STDERR: {{.*}}/prelude/operators/arithmetic.carbon:84:3: note: while building thunk to match the signature of this function [ThunkSignature]
// CHECK:STDERR:   fn Op[self: Self](other: Other) -> Result;
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: {{.*}}/prelude/types/uint.carbon:[[@LINE+224]]:3: error: cannot implicitly convert expression of type `UInt(N)` to `UInt(N).(MulWith(U).Result)` [ConversionFailure]
// CHECK:STDERR:   fn Op[self: UInt(N)](other: UInt(N)) -> UInt(N) = "int.umul";
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: {{.*}}/prelude/types/uint.carbon:[[@LINE+221]]:3: note: type `UInt(N)` does not implement interface `ImplicitAs(UInt(N).(MulWith(U).Result))` [MissingImplInMemberAccessNote]
// CHECK:STDERR:   fn Op[self: UInt(N)](other: UInt(N)) -> UInt(N) = "int.umul";
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: {{.*}}/prelude/types/uint.carbon:[[@LINE-44]]:1: in import [InImport]
// CHECK:STDERR: {{.*}}/prelude/operators/arithmetic.carbon:62:3: note: while building thunk to match the signature of this function [ThunkSignature]
// CHECK:STDERR:   fn Op[self: Self](other: Other) -> Result;
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: {{.*}}/prelude/types/uint.carbon:[[@LINE+213]]:3: error: cannot implicitly convert expression of type `UInt(N)` to `UInt(N).(SubWith(U).Result)` [ConversionFailure]
// CHECK:STDERR:   fn Op[self: UInt(N)](other: UInt(N)) -> UInt(N) = "int.usub";
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: {{.*}}/prelude/types/uint.carbon:[[@LINE+210]]:3: note: type `UInt(N)` does not implement interface `ImplicitAs(UInt(N).(SubWith(U).Result))` [MissingImplInMemberAccessNote]
// CHECK:STDERR:   fn Op[self: UInt(N)](other: UInt(N)) -> UInt(N) = "int.usub";
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: {{.*}}/prelude/types/uint.carbon:[[@LINE-55]]:1: in import [InImport]
// CHECK:STDERR: {{.*}}/prelude/operators/arithmetic.carbon:46:3: note: while building thunk to match the signature of this function [ThunkSignature]
// CHECK:STDERR:   fn Op[self: Self](other: Other) -> Result;
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: {{.*}}/prelude/types/uint.carbon:[[@LINE+202]]:3: error: cannot implicitly convert expression of type `UInt(N)` to `T.(AddWith(UInt(N)).Result)` [ConversionFailure]
// CHECK:STDERR:   fn Op[self: UInt(N)](other: UInt(N)) -> UInt(N) = "int.uadd";
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: {{.*}}/prelude/types/uint.carbon:[[@LINE+199]]:3: note: type `UInt(N)` does not implement interface `ImplicitAs(T.(AddWith(UInt(N)).Result))` [MissingImplInMemberAccessNote]
// CHECK:STDERR:   fn Op[self: UInt(N)](other: UInt(N)) -> UInt(N) = "int.uadd";
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: {{.*}}/prelude/types/uint.carbon:[[@LINE-66]]:1: in import [InImport]
// CHECK:STDERR: {{.*}}/prelude/operators/arithmetic.carbon:24:3: note: while building thunk to match the signature of this function [ThunkSignature]
// CHECK:STDERR:   fn Op[self: Self](other: Other) -> Result;
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: {{.*}}/prelude/types/uint.carbon:[[@LINE+191]]:3: error: cannot implicitly convert expression of type `UInt(N)` to `T.(DivWith(UInt(N)).Result)` [ConversionFailure]
// CHECK:STDERR:   fn Op[self: UInt(N)](other: UInt(N)) -> UInt(N) = "int.udiv";
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: {{.*}}/prelude/types/uint.carbon:[[@LINE+188]]:3: note: type `UInt(N)` does not implement interface `ImplicitAs(T.(DivWith(UInt(N)).Result))` [MissingImplInMemberAccessNote]
// CHECK:STDERR:   fn Op[self: UInt(N)](other: UInt(N)) -> UInt(N) = "int.udiv";
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: {{.*}}/prelude/types/uint.carbon:[[@LINE-77]]:1: in import [InImport]
// CHECK:STDERR: {{.*}}/prelude/operators/arithmetic.carbon:73:3: note: while building thunk to match the signature of this function [ThunkSignature]
// CHECK:STDERR:   fn Op[self: Self](other: Other) -> Result;
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: {{.*}}/prelude/types/uint.carbon:[[@LINE+180]]:3: error: cannot implicitly convert expression of type `UInt(N)` to `T.(ModWith(UInt(N)).Result)` [ConversionFailure]
// CHECK:STDERR:   fn Op[self: UInt(N)](other: UInt(N)) -> UInt(N) = "int.umod";
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: {{.*}}/prelude/types/uint.carbon:[[@LINE+177]]:3: note: type `UInt(N)` does not implement interface `ImplicitAs(T.(ModWith(UInt(N)).Result))` [MissingImplInMemberAccessNote]
// CHECK:STDERR:   fn Op[self: UInt(N)](other: UInt(N)) -> UInt(N) = "int.umod";
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: {{.*}}/prelude/types/uint.carbon:[[@LINE-88]]:1: in import [InImport]
// CHECK:STDERR: {{.*}}/prelude/operators/arithmetic.carbon:84:3: note: while building thunk to match the signature of this function [ThunkSignature]
// CHECK:STDERR:   fn Op[self: Self](other: Other) -> Result;
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: {{.*}}/prelude/types/uint.carbon:[[@LINE+169]]:3: error: cannot implicitly convert expression of type `UInt(N)` to `T.(MulWith(UInt(N)).Result)` [ConversionFailure]
// CHECK:STDERR:   fn Op[self: UInt(N)](other: UInt(N)) -> UInt(N) = "int.umul";
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: {{.*}}/prelude/types/uint.carbon:[[@LINE+166]]:3: note: type `UInt(N)` does not implement interface `ImplicitAs(T.(MulWith(UInt(N)).Result))` [MissingImplInMemberAccessNote]
// CHECK:STDERR:   fn Op[self: UInt(N)](other: UInt(N)) -> UInt(N) = "int.umul";
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: {{.*}}/prelude/types/uint.carbon:[[@LINE-99]]:1: in import [InImport]
// CHECK:STDERR: {{.*}}/prelude/operators/arithmetic.carbon:62:3: note: while building thunk to match the signature of this function [ThunkSignature]
// CHECK:STDERR:   fn Op[self: Self](other: Other) -> Result;
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: {{.*}}/prelude/types/uint.carbon:[[@LINE+158]]:3: error: cannot implicitly convert expression of type `UInt(N)` to `T.(SubWith(UInt(N)).Result)` [ConversionFailure]
// CHECK:STDERR:   fn Op[self: UInt(N)](other: UInt(N)) -> UInt(N) = "int.usub";
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: {{.*}}/prelude/types/uint.carbon:[[@LINE+155]]:3: note: type `UInt(N)` does not implement interface `ImplicitAs(T.(SubWith(UInt(N)).Result))` [MissingImplInMemberAccessNote]
// CHECK:STDERR:   fn Op[self: UInt(N)](other: UInt(N)) -> UInt(N) = "int.usub";
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: {{.*}}/prelude/types/uint.carbon:[[@LINE-110]]:1: in import [InImport]
// CHECK:STDERR: {{.*}}/prelude/operators/arithmetic.carbon:46:3: note: while building thunk to match the signature of this function [ThunkSignature]
// CHECK:STDERR:   fn Op[self: Self](other: Other) -> Result;
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: {{.*}}/prelude/types/uint.carbon:[[@LINE+147]]:3: error: cannot implicitly convert expression of type `UInt(N)` to `UInt(N).(BitAndWith(U).Result)` [ConversionFailure]
// CHECK:STDERR:   fn Op[self: UInt(N)](other: UInt(N)) -> UInt(N) = "int.and";
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: {{.*}}/prelude/types/uint.carbon:[[@LINE+144]]:3: note: type `UInt(N)` does not implement interface `ImplicitAs(UInt(N).(BitAndWith(U).Result))` [MissingImplInMemberAccessNote]
// CHECK:STDERR:   fn Op[self: UInt(N)](other: UInt(N)) -> UInt(N) = "int.and";
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: {{.*}}/prelude/types/uint.carbon:[[@LINE-121]]:1: in import [InImport]
// CHECK:STDERR: {{.*}}/prelude/operators/bitwise.carbon:30:3: note: while building thunk to match the signature of this function [ThunkSignature]
// CHECK:STDERR:   fn Op[self: Self](other: Other) -> Result;
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: {{.*}}/prelude/types/uint.carbon:[[@LINE+136]]:3: error: cannot implicitly convert expression of type `UInt(N)` to `UInt(N).(BitOrWith(U).Result)` [ConversionFailure]
// CHECK:STDERR:   fn Op[self: UInt(N)](other: UInt(N)) -> UInt(N) = "int.or";
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: {{.*}}/prelude/types/uint.carbon:[[@LINE+133]]:3: note: type `UInt(N)` does not implement interface `ImplicitAs(UInt(N).(BitOrWith(U).Result))` [MissingImplInMemberAccessNote]
// CHECK:STDERR:   fn Op[self: UInt(N)](other: UInt(N)) -> UInt(N) = "int.or";
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: {{.*}}/prelude/types/uint.carbon:[[@LINE-132]]:1: in import [InImport]
// CHECK:STDERR: {{.*}}/prelude/operators/bitwise.carbon:41:3: note: while building thunk to match the signature of this function [ThunkSignature]
// CHECK:STDERR:   fn Op[self: Self](other: Other) -> Result;
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: {{.*}}/prelude/types/uint.carbon:[[@LINE+125]]:3: error: cannot implicitly convert expression of type `UInt(N)` to `UInt(N).(BitXorWith(U).Result)` [ConversionFailure]
// CHECK:STDERR:   fn Op[self: UInt(N)](other: UInt(N)) -> UInt(N) = "int.xor";
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: {{.*}}/prelude/types/uint.carbon:[[@LINE+122]]:3: note: type `UInt(N)` does not implement interface `ImplicitAs(UInt(N).(BitXorWith(U).Result))` [MissingImplInMemberAccessNote]
// CHECK:STDERR:   fn Op[self: UInt(N)](other: UInt(N)) -> UInt(N) = "int.xor";
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: {{.*}}/prelude/types/uint.carbon:[[@LINE-143]]:1: in import [InImport]
// CHECK:STDERR: {{.*}}/prelude/operators/bitwise.carbon:52:3: note: while building thunk to match the signature of this function [ThunkSignature]
// CHECK:STDERR:   fn Op[self: Self](other: Other) -> Result;
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: {{.*}}/prelude/types/uint.carbon:[[@LINE+114]]:3: error: cannot implicitly convert expression of type `UInt(N)` to `UInt(N).(LeftShiftWith(U).Result)` [ConversionFailure]
// CHECK:STDERR:   fn Op[self: UInt(N)](other: UInt(N)) -> UInt(N) = "int.left_shift";
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: {{.*}}/prelude/types/uint.carbon:[[@LINE+111]]:3: note: type `UInt(N)` does not implement interface `ImplicitAs(UInt(N).(LeftShiftWith(U).Result))` [MissingImplInMemberAccessNote]
// CHECK:STDERR:   fn Op[self: UInt(N)](other: UInt(N)) -> UInt(N) = "int.left_shift";
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: {{.*}}/prelude/types/uint.carbon:[[@LINE-154]]:1: in import [InImport]
// CHECK:STDERR: {{.*}}/prelude/operators/bitwise.carbon:63:3: note: while building thunk to match the signature of this function [ThunkSignature]
// CHECK:STDERR:   fn Op[self: Self](other: Other) -> Result;
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: {{.*}}/prelude/types/uint.carbon:[[@LINE+103]]:3: error: cannot implicitly convert expression of type `UInt(N)` to `UInt(N).(RightShiftWith(U).Result)` [ConversionFailure]
// CHECK:STDERR:   fn Op[self: UInt(N)](other: UInt(N)) -> UInt(N) = "int.right_shift";
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: {{.*}}/prelude/types/uint.carbon:[[@LINE+100]]:3: note: type `UInt(N)` does not implement interface `ImplicitAs(UInt(N).(RightShiftWith(U).Result))` [MissingImplInMemberAccessNote]
// CHECK:STDERR:   fn Op[self: UInt(N)](other: UInt(N)) -> UInt(N) = "int.right_shift";
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: {{.*}}/prelude/types/uint.carbon:[[@LINE-165]]:1: in import [InImport]
// CHECK:STDERR: {{.*}}/prelude/operators/bitwise.carbon:74:3: note: while building thunk to match the signature of this function [ThunkSignature]
// CHECK:STDERR:   fn Op[self: Self](other: Other) -> Result;
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: {{.*}}/prelude/types/uint.carbon:[[@LINE+92]]:3: error: cannot implicitly convert expression of type `UInt(N)` to `T.(BitAndWith(UInt(N)).Result)` [ConversionFailure]
// CHECK:STDERR:   fn Op[self: UInt(N)](other: UInt(N)) -> UInt(N) = "int.and";
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: {{.*}}/prelude/types/uint.carbon:[[@LINE+89]]:3: note: type `UInt(N)` does not implement interface `ImplicitAs(T.(BitAndWith(UInt(N)).Result))` [MissingImplInMemberAccessNote]
// CHECK:STDERR:   fn Op[self: UInt(N)](other: UInt(N)) -> UInt(N) = "int.and";
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: {{.*}}/prelude/types/uint.carbon:[[@LINE-176]]:1: in import [InImport]
// CHECK:STDERR: {{.*}}/prelude/operators/bitwise.carbon:30:3: note: while building thunk to match the signature of this function [ThunkSignature]
// CHECK:STDERR:   fn Op[self: Self](other: Other) -> Result;
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: {{.*}}/prelude/types/uint.carbon:[[@LINE+81]]:3: error: cannot implicitly convert expression of type `UInt(N)` to `T.(BitOrWith(UInt(N)).Result)` [ConversionFailure]
// CHECK:STDERR:   fn Op[self: UInt(N)](other: UInt(N)) -> UInt(N) = "int.or";
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: {{.*}}/prelude/types/uint.carbon:[[@LINE+78]]:3: note: type `UInt(N)` does not implement interface `ImplicitAs(T.(BitOrWith(UInt(N)).Result))` [MissingImplInMemberAccessNote]
// CHECK:STDERR:   fn Op[self: UInt(N)](other: UInt(N)) -> UInt(N) = "int.or";
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: {{.*}}/prelude/types/uint.carbon:[[@LINE-187]]:1: in import [InImport]
// CHECK:STDERR: {{.*}}/prelude/operators/bitwise.carbon:41:3: note: while building thunk to match the signature of this function [ThunkSignature]
// CHECK:STDERR:   fn Op[self: Self](other: Other) -> Result;
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: {{.*}}/prelude/types/uint.carbon:[[@LINE+70]]:3: error: cannot implicitly convert expression of type `UInt(N)` to `T.(BitXorWith(UInt(N)).Result)` [ConversionFailure]
// CHECK:STDERR:   fn Op[self: UInt(N)](other: UInt(N)) -> UInt(N) = "int.xor";
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: {{.*}}/prelude/types/uint.carbon:[[@LINE+67]]:3: note: type `UInt(N)` does not implement interface `ImplicitAs(T.(BitXorWith(UInt(N)).Result))` [MissingImplInMemberAccessNote]
// CHECK:STDERR:   fn Op[self: UInt(N)](other: UInt(N)) -> UInt(N) = "int.xor";
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: {{.*}}/prelude/types/uint.carbon:[[@LINE-198]]:1: in import [InImport]
// CHECK:STDERR: {{.*}}/prelude/operators/bitwise.carbon:52:3: note: while building thunk to match the signature of this function [ThunkSignature]
// CHECK:STDERR:   fn Op[self: Self](other: Other) -> Result;
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: {{.*}}/prelude/types/uint.carbon:[[@LINE+59]]:3: error: cannot implicitly convert expression of type `UInt(N)` to `T.(LeftShiftWith(UInt(N)).Result)` [ConversionFailure]
// CHECK:STDERR:   fn Op[self: UInt(N)](other: UInt(N)) -> UInt(N) = "int.left_shift";
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: {{.*}}/prelude/types/uint.carbon:[[@LINE+56]]:3: note: type `UInt(N)` does not implement interface `ImplicitAs(T.(LeftShiftWith(UInt(N)).Result))` [MissingImplInMemberAccessNote]
// CHECK:STDERR:   fn Op[self: UInt(N)](other: UInt(N)) -> UInt(N) = "int.left_shift";
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: {{.*}}/prelude/types/uint.carbon:[[@LINE-209]]:1: in import [InImport]
// CHECK:STDERR: {{.*}}/prelude/operators/bitwise.carbon:63:3: note: while building thunk to match the signature of this function [ThunkSignature]
// CHECK:STDERR:   fn Op[self: Self](other: Other) -> Result;
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: {{.*}}/prelude/types/uint.carbon:[[@LINE+48]]:3: error: cannot implicitly convert expression of type `UInt(N)` to `T.(RightShiftWith(UInt(N)).Result)` [ConversionFailure]
// CHECK:STDERR:   fn Op[self: UInt(N)](other: UInt(N)) -> UInt(N) = "int.right_shift";
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: {{.*}}/prelude/types/uint.carbon:[[@LINE+45]]:3: note: type `UInt(N)` does not implement interface `ImplicitAs(T.(RightShiftWith(UInt(N)).Result))` [MissingImplInMemberAccessNote]
// CHECK:STDERR:   fn Op[self: UInt(N)](other: UInt(N)) -> UInt(N) = "int.right_shift";
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: {{.*}}/prelude/types/uint.carbon:[[@LINE-220]]:1: in import [InImport]
// CHECK:STDERR: {{.*}}/prelude/operators/bitwise.carbon:74:3: note: while building thunk to match the signature of this function [ThunkSignature]
// CHECK:STDERR:   fn Op[self: Self](other: Other) -> Result;
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: {{.*}}/range.carbon:10:1: in import [InImport]
// CHECK:STDERR: {{.*}}/prelude/iterate.carbon:17:23: error: cannot implicitly convert expression of type `IntRange(N).(Iterate.CursorType)*` to `Int(N)*` [ConversionFailure]
// CHECK:STDERR:   fn Next[self: Self](cursor: CursorType*) -> Optional(ElementType);
// CHECK:STDERR:                       ^~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: {{.*}}/range.carbon:10:1: in import [InImport]
// CHECK:STDERR: {{.*}}/prelude/iterate.carbon:17:23: note: type `IntRange(N).(Iterate.CursorType)*` does not implement interface `ImplicitAs(Int(N)*)` [MissingImplInMemberAccessNote]
// CHECK:STDERR:   fn Next[self: Self](cursor: CursorType*) -> Optional(ElementType);
// CHECK:STDERR:                       ^~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: {{.*}}/range.carbon:26:25: note: initializing function parameter [InCallToFunctionParam]
// CHECK:STDERR:     fn Next[self: Self](cursor: Int(N)*) -> Optional(Int(N)) {
// CHECK:STDERR:                         ^~~~~~~~~~~~~~~
// CHECK:STDERR: {{.*}}/range.carbon:10:1: in import [InImport]
// CHECK:STDERR: {{.*}}/prelude/iterate.carbon:17:3: note: while building thunk to match the signature of this function [ThunkSignature]
// CHECK:STDERR:   fn Next[self: Self](cursor: CursorType*) -> Optional(ElementType);
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: {{.*}}/range.carbon:25:5: error: cannot implicitly convert expression of type `Int(N)` to `IntRange(N).(Iterate.CursorType)` [ConversionFailure]
// CHECK:STDERR:     fn NewCursor[self: Self]() -> Int(N) { return self.start; }
// CHECK:STDERR:     ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: {{.*}}/range.carbon:25:5: note: type `Int(N)` does not implement interface `ImplicitAs(IntRange(N).(Iterate.CursorType))` [MissingImplInMemberAccessNote]
// CHECK:STDERR:     fn NewCursor[self: Self]() -> Int(N) { return self.start; }
// CHECK:STDERR:     ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: {{.*}}/range.carbon:10:1: in import [InImport]
// CHECK:STDERR: {{.*}}/prelude/iterate.carbon:16:3: note: while building thunk to match the signature of this function [ThunkSignature]
// CHECK:STDERR:   fn NewCursor[self: Self]() -> CursorType;
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: {{.*}}/range.carbon:26:5: error: cannot implicitly convert expression of type `Optional(Int(N) as OptionalStorage)` to `Optional(IntRange(N).(Iterate.ElementType) as OptionalStorage)` [ConversionFailure]
// CHECK:STDERR:     fn Next[self: Self](cursor: Int(N)*) -> Optional(Int(N)) {
// CHECK:STDERR:     ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: {{.*}}/range.carbon:26:5: note: type `Optional(Int(N) as OptionalStorage)` does not implement interface `ImplicitAs(Optional(IntRange(N).(Iterate.ElementType) as OptionalStorage))` [MissingImplInMemberAccessNote]
// CHECK:STDERR:     fn Next[self: Self](cursor: Int(N)*) -> Optional(Int(N)) {
// CHECK:STDERR:     ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: {{.*}}/range.carbon:10:1: in import [InImport]
// CHECK:STDERR: {{.*}}/prelude/iterate.carbon:17:3: note: while building thunk to match the signature of this function [ThunkSignature]
// CHECK:STDERR:   fn Next[self: Self](cursor: CursorType*) -> Optional(ElementType);
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
// --- unsigned_uint.carbon

library "[[@TEST_NAME]]";
import library "uint";

fn F() -> u32 {
  //@dump-sem-ir-begin
  return (C as I).val;
  //@dump-sem-ir-end
}

// CHECK:STDOUT: --- import_bool.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [concrete]
// CHECK:STDOUT:   %false: bool = bool_literal false [concrete]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness imports.%I.impl_witness_table [concrete]
// CHECK:STDOUT:   %I.facet: %I.type = facet_value %C, (%I.impl_witness) [concrete]
// CHECK:STDOUT:   %I.assoc_type: type = assoc_entity_type @I [concrete]
// CHECK:STDOUT:   %assoc0.e13: %I.assoc_type = assoc_entity element0, imports.%Main.import_ref.030 [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Copy.Op.type: type = fn_type @Copy.Op [concrete]
// CHECK:STDOUT:   %Copy.impl_witness.a56: <witness> = impl_witness imports.%Copy.impl_witness_table.189 [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value bool, (%Copy.impl_witness.a56) [concrete]
// CHECK:STDOUT:   %.05c: type = fn_type_with_self_type %Copy.Op.type, %Copy.facet [concrete]
// CHECK:STDOUT:   %bool.as.Copy.impl.Op.type: type = fn_type @bool.as.Copy.impl.Op [concrete]
// CHECK:STDOUT:   %bool.as.Copy.impl.Op: %bool.as.Copy.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %bool.as.Copy.impl.Op.bound: <bound method> = bound_method %false, %bool.as.Copy.impl.Op [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.I: type = import_ref Main//bool, I, loaded [concrete = constants.%I.type]
// CHECK:STDOUT:   %Main.C: type = import_ref Main//bool, C, loaded [concrete = constants.%C]
// CHECK:STDOUT:   %Main.import_ref.8be: %I.assoc_type = import_ref Main//bool, loc5_10, loaded [concrete = constants.%assoc0.e13]
// CHECK:STDOUT:   %Main.import_ref.d18: bool = import_ref Main//bool, loc9_32, loaded [concrete = constants.%false]
// CHECK:STDOUT:   %I.impl_witness_table = impl_witness_table (%Main.import_ref.d18), @C.as.I.impl [concrete]
// CHECK:STDOUT:   %Main.import_ref.030: bool = import_ref Main//bool, loc5_10, loaded [concrete = %val]
// CHECK:STDOUT:   %val: bool = assoc_const_decl @val [concrete] {}
// CHECK:STDOUT:   %Core.import_ref.cc5: %bool.as.Copy.impl.Op.type = import_ref Core//prelude/copy, loc{{\d+_\d+}}, loaded [concrete = constants.%bool.as.Copy.impl.Op]
// CHECK:STDOUT:   %Copy.impl_witness_table.189 = impl_witness_table (%Core.import_ref.cc5), @bool.as.Copy.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() -> bool {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %C.ref: type = name_ref C, imports.%Main.C [concrete = constants.%C]
// CHECK:STDOUT:   %I.ref: type = name_ref I, imports.%Main.I [concrete = constants.%I.type]
// CHECK:STDOUT:   %I.facet: %I.type = facet_value %C.ref, (constants.%I.impl_witness) [concrete = constants.%I.facet]
// CHECK:STDOUT:   %.loc7_13: %I.type = converted %C.ref, %I.facet [concrete = constants.%I.facet]
// CHECK:STDOUT:   %val.ref: %I.assoc_type = name_ref val, imports.%Main.import_ref.8be [concrete = constants.%assoc0.e13]
// CHECK:STDOUT:   %as_type: type = facet_access_type %.loc7_13 [concrete = constants.%C]
// CHECK:STDOUT:   %.loc7_18: type = converted %.loc7_13, %as_type [concrete = constants.%C]
// CHECK:STDOUT:   %impl.elem0.loc7_18.1: bool = impl_witness_access constants.%I.impl_witness, element0 [concrete = constants.%false]
// CHECK:STDOUT:   %impl.elem0.loc7_18.2: %.05c = impl_witness_access constants.%Copy.impl_witness.a56, element0 [concrete = constants.%bool.as.Copy.impl.Op]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %impl.elem0.loc7_18.1, %impl.elem0.loc7_18.2 [concrete = constants.%bool.as.Copy.impl.Op.bound]
// CHECK:STDOUT:   %bool.as.Copy.impl.Op.call: init bool = call %bound_method(%impl.elem0.loc7_18.1) [concrete = constants.%false]
// CHECK:STDOUT:   return %bool.as.Copy.impl.Op.call to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_char.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %char: type = class_type @Char [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [concrete]
// CHECK:STDOUT:   %int_97: %char = int_value 97 [concrete]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness imports.%I.impl_witness_table [concrete]
// CHECK:STDOUT:   %I.facet: %I.type = facet_value %C, (%I.impl_witness) [concrete]
// CHECK:STDOUT:   %I.assoc_type: type = assoc_entity_type @I [concrete]
// CHECK:STDOUT:   %assoc0.941: %I.assoc_type = assoc_entity element0, imports.%Main.import_ref.8e8 [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Copy.Op.type: type = fn_type @Copy.Op [concrete]
// CHECK:STDOUT:   %Copy.impl_witness.9ba: <witness> = impl_witness imports.%Copy.impl_witness_table.0e0 [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %char, (%Copy.impl_witness.9ba) [concrete]
// CHECK:STDOUT:   %.1ed: type = fn_type_with_self_type %Copy.Op.type, %Copy.facet [concrete]
// CHECK:STDOUT:   %char.as.Copy.impl.Op.type: type = fn_type @char.as.Copy.impl.Op [concrete]
// CHECK:STDOUT:   %char.as.Copy.impl.Op: %char.as.Copy.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %char.as.Copy.impl.Op.bound: <bound method> = bound_method %int_97, %char.as.Copy.impl.Op [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.I: type = import_ref Main//char, I, loaded [concrete = constants.%I.type]
// CHECK:STDOUT:   %Main.C: type = import_ref Main//char, C, loaded [concrete = constants.%C]
// CHECK:STDOUT:   %Main.import_ref.da1: %I.assoc_type = import_ref Main//char, loc5_10, loaded [concrete = constants.%assoc0.941]
// CHECK:STDOUT:   %Main.import_ref.d0c: %char = import_ref Main//char, loc10_30, loaded [concrete = constants.%int_97]
// CHECK:STDOUT:   %I.impl_witness_table = impl_witness_table (%Main.import_ref.d0c), @C.as.I.impl [concrete]
// CHECK:STDOUT:   %Main.import_ref.8e8: %char = import_ref Main//char, loc5_10, loaded [concrete = %val]
// CHECK:STDOUT:   %val: %char = assoc_const_decl @val [concrete] {}
// CHECK:STDOUT:   %Core.import_ref.247: %char.as.Copy.impl.Op.type = import_ref Core//prelude/types/char, loc{{\d+_\d+}}, loaded [concrete = constants.%char.as.Copy.impl.Op]
// CHECK:STDOUT:   %Copy.impl_witness_table.0e0 = impl_witness_table (%Core.import_ref.247), @char.as.Copy.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() -> %char {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %C.ref: type = name_ref C, imports.%Main.C [concrete = constants.%C]
// CHECK:STDOUT:   %I.ref: type = name_ref I, imports.%Main.I [concrete = constants.%I.type]
// CHECK:STDOUT:   %I.facet: %I.type = facet_value %C.ref, (constants.%I.impl_witness) [concrete = constants.%I.facet]
// CHECK:STDOUT:   %.loc7_13: %I.type = converted %C.ref, %I.facet [concrete = constants.%I.facet]
// CHECK:STDOUT:   %val.ref: %I.assoc_type = name_ref val, imports.%Main.import_ref.da1 [concrete = constants.%assoc0.941]
// CHECK:STDOUT:   %as_type: type = facet_access_type %.loc7_13 [concrete = constants.%C]
// CHECK:STDOUT:   %.loc7_18: type = converted %.loc7_13, %as_type [concrete = constants.%C]
// CHECK:STDOUT:   %impl.elem0.loc7_18.1: %char = impl_witness_access constants.%I.impl_witness, element0 [concrete = constants.%int_97]
// CHECK:STDOUT:   %impl.elem0.loc7_18.2: %.1ed = impl_witness_access constants.%Copy.impl_witness.9ba, element0 [concrete = constants.%char.as.Copy.impl.Op]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %impl.elem0.loc7_18.1, %impl.elem0.loc7_18.2 [concrete = constants.%char.as.Copy.impl.Op.bound]
// CHECK:STDOUT:   %char.as.Copy.impl.Op.call: init %char = call %bound_method(%impl.elem0.loc7_18.1) [concrete = constants.%int_97]
// CHECK:STDOUT:   return %char.as.Copy.impl.Op.call to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_char_literal.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [concrete]
// CHECK:STDOUT:   %.54f: Core.CharLiteral = char_value U+0061 [concrete]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness imports.%I.impl_witness_table [concrete]
// CHECK:STDOUT:   %I.facet: %I.type = facet_value %C, (%I.impl_witness) [concrete]
// CHECK:STDOUT:   %I.assoc_type: type = assoc_entity_type @I [concrete]
// CHECK:STDOUT:   %assoc0.545: %I.assoc_type = assoc_entity element0, imports.%Main.import_ref.dcc [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Copy.Op.type: type = fn_type @Copy.Op [concrete]
// CHECK:STDOUT:   %Copy.impl_witness.eda: <witness> = impl_witness imports.%Copy.impl_witness_table.51e [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value Core.CharLiteral, (%Copy.impl_witness.eda) [concrete]
// CHECK:STDOUT:   %.2aa: type = fn_type_with_self_type %Copy.Op.type, %Copy.facet [concrete]
// CHECK:STDOUT:   %Core.CharLiteral.as.Copy.impl.Op.type: type = fn_type @Core.CharLiteral.as.Copy.impl.Op [concrete]
// CHECK:STDOUT:   %Core.CharLiteral.as.Copy.impl.Op: %Core.CharLiteral.as.Copy.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %Core.CharLiteral.as.Copy.impl.Op.bound: <bound method> = bound_method %.54f, %Core.CharLiteral.as.Copy.impl.Op [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.I: type = import_ref Main//char_literal, I, loaded [concrete = constants.%I.type]
// CHECK:STDOUT:   %Main.C: type = import_ref Main//char_literal, C, loaded [concrete = constants.%C]
// CHECK:STDOUT:   %Main.import_ref.696: %I.assoc_type = import_ref Main//char_literal, loc5_10, loaded [concrete = constants.%assoc0.545]
// CHECK:STDOUT:   %Main.import_ref.43e: Core.CharLiteral = import_ref Main//char_literal, loc10_30, loaded [concrete = constants.%.54f]
// CHECK:STDOUT:   %I.impl_witness_table = impl_witness_table (%Main.import_ref.43e), @C.as.I.impl [concrete]
// CHECK:STDOUT:   %Main.import_ref.dcc: Core.CharLiteral = import_ref Main//char_literal, loc5_10, loaded [concrete = %val]
// CHECK:STDOUT:   %val: Core.CharLiteral = assoc_const_decl @val [concrete] {}
// CHECK:STDOUT:   %Core.import_ref.70c: %Core.CharLiteral.as.Copy.impl.Op.type = import_ref Core//prelude/copy, loc{{\d+_\d+}}, loaded [concrete = constants.%Core.CharLiteral.as.Copy.impl.Op]
// CHECK:STDOUT:   %Copy.impl_witness_table.51e = impl_witness_table (%Core.import_ref.70c), @Core.CharLiteral.as.Copy.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() -> Core.CharLiteral {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %C.ref: type = name_ref C, imports.%Main.C [concrete = constants.%C]
// CHECK:STDOUT:   %I.ref: type = name_ref I, imports.%Main.I [concrete = constants.%I.type]
// CHECK:STDOUT:   %I.facet: %I.type = facet_value %C.ref, (constants.%I.impl_witness) [concrete = constants.%I.facet]
// CHECK:STDOUT:   %.loc7_13: %I.type = converted %C.ref, %I.facet [concrete = constants.%I.facet]
// CHECK:STDOUT:   %val.ref: %I.assoc_type = name_ref val, imports.%Main.import_ref.696 [concrete = constants.%assoc0.545]
// CHECK:STDOUT:   %as_type: type = facet_access_type %.loc7_13 [concrete = constants.%C]
// CHECK:STDOUT:   %.loc7_18: type = converted %.loc7_13, %as_type [concrete = constants.%C]
// CHECK:STDOUT:   %impl.elem0.loc7_18.1: Core.CharLiteral = impl_witness_access constants.%I.impl_witness, element0 [concrete = constants.%.54f]
// CHECK:STDOUT:   %impl.elem0.loc7_18.2: %.2aa = impl_witness_access constants.%Copy.impl_witness.eda, element0 [concrete = constants.%Core.CharLiteral.as.Copy.impl.Op]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %impl.elem0.loc7_18.1, %impl.elem0.loc7_18.2 [concrete = constants.%Core.CharLiteral.as.Copy.impl.Op.bound]
// CHECK:STDOUT:   %Core.CharLiteral.as.Copy.impl.Op.call: init Core.CharLiteral = call %bound_method(%impl.elem0.loc7_18.1) [concrete = constants.%.54f]
// CHECK:STDOUT:   return %Core.CharLiteral.as.Copy.impl.Op.call to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_float.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_64: Core.IntLiteral = int_value 64 [concrete]
// CHECK:STDOUT:   %N: Core.IntLiteral = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %f64.d77: type = class_type @Float, @Float(%int_64) [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [concrete]
// CHECK:STDOUT:   %float: %f64.d77 = float_value 0.10000000000000001 [concrete]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness imports.%I.impl_witness_table [concrete]
// CHECK:STDOUT:   %I.facet: %I.type = facet_value %C, (%I.impl_witness) [concrete]
// CHECK:STDOUT:   %I.assoc_type: type = assoc_entity_type @I [concrete]
// CHECK:STDOUT:   %assoc0.ece: %I.assoc_type = assoc_entity element0, imports.%Main.import_ref.267 [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Copy.Op.type: type = fn_type @Copy.Op [concrete]
// CHECK:STDOUT:   %Float.as.Copy.impl.Op.type.0ad: type = fn_type @Float.as.Copy.impl.Op, @Float.as.Copy.impl(%N) [symbolic]
// CHECK:STDOUT:   %Float.as.Copy.impl.Op.c95: %Float.as.Copy.impl.Op.type.0ad = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.4f8: <witness> = impl_witness imports.%Copy.impl_witness_table.0db, @Float.as.Copy.impl(%int_64) [concrete]
// CHECK:STDOUT:   %Float.as.Copy.impl.Op.type.550: type = fn_type @Float.as.Copy.impl.Op, @Float.as.Copy.impl(%int_64) [concrete]
// CHECK:STDOUT:   %Float.as.Copy.impl.Op.cb2: %Float.as.Copy.impl.Op.type.550 = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %f64.d77, (%Copy.impl_witness.4f8) [concrete]
// CHECK:STDOUT:   %.fb7: type = fn_type_with_self_type %Copy.Op.type, %Copy.facet [concrete]
// CHECK:STDOUT:   %Float.as.Copy.impl.Op.bound: <bound method> = bound_method %float, %Float.as.Copy.impl.Op.cb2 [concrete]
// CHECK:STDOUT:   %Float.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %Float.as.Copy.impl.Op.cb2, @Float.as.Copy.impl.Op(%int_64) [concrete]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %float, %Float.as.Copy.impl.Op.specific_fn [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.I: type = import_ref Main//float, I, loaded [concrete = constants.%I.type]
// CHECK:STDOUT:   %Main.C: type = import_ref Main//float, C, loaded [concrete = constants.%C]
// CHECK:STDOUT:   %Main.import_ref.de3: %I.assoc_type = import_ref Main//float, loc5_10, loaded [concrete = constants.%assoc0.ece]
// CHECK:STDOUT:   %Main.import_ref.0d9: %f64.d77 = import_ref Main//float, loc10_30, loaded [concrete = constants.%float]
// CHECK:STDOUT:   %I.impl_witness_table = impl_witness_table (%Main.import_ref.0d9), @C.as.I.impl [concrete]
// CHECK:STDOUT:   %Main.import_ref.267: %f64.d77 = import_ref Main//float, loc5_10, loaded [concrete = %val]
// CHECK:STDOUT:   %val: %f64.d77 = assoc_const_decl @val [concrete] {}
// CHECK:STDOUT:   %Core.import_ref.428: @Float.as.Copy.impl.%Float.as.Copy.impl.Op.type (%Float.as.Copy.impl.Op.type.0ad) = import_ref Core//prelude/types/float, loc{{\d+_\d+}}, loaded [symbolic = @Float.as.Copy.impl.%Float.as.Copy.impl.Op (constants.%Float.as.Copy.impl.Op.c95)]
// CHECK:STDOUT:   %Copy.impl_witness_table.0db = impl_witness_table (%Core.import_ref.428), @Float.as.Copy.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() -> %f64.d77 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %C.ref: type = name_ref C, imports.%Main.C [concrete = constants.%C]
// CHECK:STDOUT:   %I.ref: type = name_ref I, imports.%Main.I [concrete = constants.%I.type]
// CHECK:STDOUT:   %I.facet: %I.type = facet_value %C.ref, (constants.%I.impl_witness) [concrete = constants.%I.facet]
// CHECK:STDOUT:   %.loc7_13: %I.type = converted %C.ref, %I.facet [concrete = constants.%I.facet]
// CHECK:STDOUT:   %val.ref: %I.assoc_type = name_ref val, imports.%Main.import_ref.de3 [concrete = constants.%assoc0.ece]
// CHECK:STDOUT:   %as_type: type = facet_access_type %.loc7_13 [concrete = constants.%C]
// CHECK:STDOUT:   %.loc7_18: type = converted %.loc7_13, %as_type [concrete = constants.%C]
// CHECK:STDOUT:   %impl.elem0.loc7_18.1: %f64.d77 = impl_witness_access constants.%I.impl_witness, element0 [concrete = constants.%float]
// CHECK:STDOUT:   %impl.elem0.loc7_18.2: %.fb7 = impl_witness_access constants.%Copy.impl_witness.4f8, element0 [concrete = constants.%Float.as.Copy.impl.Op.cb2]
// CHECK:STDOUT:   %bound_method.loc7_18.1: <bound method> = bound_method %impl.elem0.loc7_18.1, %impl.elem0.loc7_18.2 [concrete = constants.%Float.as.Copy.impl.Op.bound]
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0.loc7_18.2, @Float.as.Copy.impl.Op(constants.%int_64) [concrete = constants.%Float.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc7_18.2: <bound method> = bound_method %impl.elem0.loc7_18.1, %specific_fn [concrete = constants.%bound_method]
// CHECK:STDOUT:   %Float.as.Copy.impl.Op.call: init %f64.d77 = call %bound_method.loc7_18.2(%impl.elem0.loc7_18.1) [concrete = constants.%float]
// CHECK:STDOUT:   return %Float.as.Copy.impl.Op.call to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_float_literal.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [concrete]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness imports.%I.impl_witness_table [concrete]
// CHECK:STDOUT:   %I.facet: %I.type = facet_value %C, (%I.impl_witness) [concrete]
// CHECK:STDOUT:   %I.assoc_type: type = assoc_entity_type @I [concrete]
// CHECK:STDOUT:   %assoc0.6ce: %I.assoc_type = assoc_entity element0, imports.%Main.import_ref.e01 [concrete]
// CHECK:STDOUT:   %float.3fedf4.2: Core.FloatLiteral = float_literal_value 1e-1 [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Copy.Op.type: type = fn_type @Copy.Op [concrete]
// CHECK:STDOUT:   %Copy.impl_witness.691: <witness> = impl_witness imports.%Copy.impl_witness_table.3b2 [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value Core.FloatLiteral, (%Copy.impl_witness.691) [concrete]
// CHECK:STDOUT:   %.3cc: type = fn_type_with_self_type %Copy.Op.type, %Copy.facet [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.Copy.impl.Op.type: type = fn_type @Core.FloatLiteral.as.Copy.impl.Op [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.Copy.impl.Op: %Core.FloatLiteral.as.Copy.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.Copy.impl.Op.bound: <bound method> = bound_method %float.3fedf4.2, %Core.FloatLiteral.as.Copy.impl.Op [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.I: type = import_ref Main//float_literal, I, loaded [concrete = constants.%I.type]
// CHECK:STDOUT:   %Main.C: type = import_ref Main//float_literal, C, loaded [concrete = constants.%C]
// CHECK:STDOUT:   %Main.import_ref.730: %I.assoc_type = import_ref Main//float_literal, loc5_10, loaded [concrete = constants.%assoc0.6ce]
// CHECK:STDOUT:   %Main.import_ref.dec: Core.FloatLiteral = import_ref Main//float_literal, loc10_30, loaded [concrete = constants.%float.3fedf4.2]
// CHECK:STDOUT:   %I.impl_witness_table = impl_witness_table (%Main.import_ref.dec), @C.as.I.impl [concrete]
// CHECK:STDOUT:   %Main.import_ref.e01: Core.FloatLiteral = import_ref Main//float_literal, loc5_10, loaded [concrete = %val]
// CHECK:STDOUT:   %val: Core.FloatLiteral = assoc_const_decl @val [concrete] {}
// CHECK:STDOUT:   %Core.import_ref.3bb: %Core.FloatLiteral.as.Copy.impl.Op.type = import_ref Core//prelude/copy, loc{{\d+_\d+}}, loaded [concrete = constants.%Core.FloatLiteral.as.Copy.impl.Op]
// CHECK:STDOUT:   %Copy.impl_witness_table.3b2 = impl_witness_table (%Core.import_ref.3bb), @Core.FloatLiteral.as.Copy.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() -> Core.FloatLiteral {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %C.ref: type = name_ref C, imports.%Main.C [concrete = constants.%C]
// CHECK:STDOUT:   %I.ref: type = name_ref I, imports.%Main.I [concrete = constants.%I.type]
// CHECK:STDOUT:   %I.facet: %I.type = facet_value %C.ref, (constants.%I.impl_witness) [concrete = constants.%I.facet]
// CHECK:STDOUT:   %.loc7_13: %I.type = converted %C.ref, %I.facet [concrete = constants.%I.facet]
// CHECK:STDOUT:   %val.ref: %I.assoc_type = name_ref val, imports.%Main.import_ref.730 [concrete = constants.%assoc0.6ce]
// CHECK:STDOUT:   %as_type: type = facet_access_type %.loc7_13 [concrete = constants.%C]
// CHECK:STDOUT:   %.loc7_18: type = converted %.loc7_13, %as_type [concrete = constants.%C]
// CHECK:STDOUT:   %impl.elem0.loc7_18.1: Core.FloatLiteral = impl_witness_access constants.%I.impl_witness, element0 [concrete = constants.%float.3fedf4.2]
// CHECK:STDOUT:   %impl.elem0.loc7_18.2: %.3cc = impl_witness_access constants.%Copy.impl_witness.691, element0 [concrete = constants.%Core.FloatLiteral.as.Copy.impl.Op]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %impl.elem0.loc7_18.1, %impl.elem0.loc7_18.2 [concrete = constants.%Core.FloatLiteral.as.Copy.impl.Op.bound]
// CHECK:STDOUT:   %Core.FloatLiteral.as.Copy.impl.Op.call: init Core.FloatLiteral = call %bound_method(%impl.elem0.loc7_18.1) [concrete = constants.%float.3fedf4.2]
// CHECK:STDOUT:   return %Core.FloatLiteral.as.Copy.impl.Op.call to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_int.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %N: Core.IntLiteral = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [concrete]
// CHECK:STDOUT:   %int_8: %i32 = int_value 8 [concrete]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness imports.%I.impl_witness_table [concrete]
// CHECK:STDOUT:   %I.facet: %I.type = facet_value %C, (%I.impl_witness) [concrete]
// CHECK:STDOUT:   %I.assoc_type: type = assoc_entity_type @I [concrete]
// CHECK:STDOUT:   %assoc0.aac: %I.assoc_type = assoc_entity element0, imports.%Main.import_ref.05e [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Copy.Op.type: type = fn_type @Copy.Op [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.24b: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%N) [symbolic]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.c95: %Int.as.Copy.impl.Op.type.24b = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.fb7: <witness> = impl_witness imports.%Copy.impl_witness_table.b6a, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.469: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.dfd: %Int.as.Copy.impl.Op.type.469 = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %i32, (%Copy.impl_witness.fb7) [concrete]
// CHECK:STDOUT:   %.65f: type = fn_type_with_self_type %Copy.Op.type, %Copy.facet [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.bound: <bound method> = bound_method %int_8, %Int.as.Copy.impl.Op.dfd [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %Int.as.Copy.impl.Op.dfd, @Int.as.Copy.impl.Op(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %int_8, %Int.as.Copy.impl.Op.specific_fn [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.I: type = import_ref Main//int, I, loaded [concrete = constants.%I.type]
// CHECK:STDOUT:   %Main.C: type = import_ref Main//int, C, loaded [concrete = constants.%C]
// CHECK:STDOUT:   %Main.import_ref.ba4: %I.assoc_type = import_ref Main//int, loc5_10, loaded [concrete = constants.%assoc0.aac]
// CHECK:STDOUT:   %Main.import_ref.330: %i32 = import_ref Main//int, loc10_28, loaded [concrete = constants.%int_8]
// CHECK:STDOUT:   %I.impl_witness_table = impl_witness_table (%Main.import_ref.330), @C.as.I.impl [concrete]
// CHECK:STDOUT:   %Main.import_ref.05e: %i32 = import_ref Main//int, loc5_10, loaded [concrete = %val]
// CHECK:STDOUT:   %val: %i32 = assoc_const_decl @val [concrete] {}
// CHECK:STDOUT:   %Core.import_ref.d12: @Int.as.Copy.impl.%Int.as.Copy.impl.Op.type (%Int.as.Copy.impl.Op.type.24b) = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.Copy.impl.%Int.as.Copy.impl.Op (constants.%Int.as.Copy.impl.Op.c95)]
// CHECK:STDOUT:   %Copy.impl_witness_table.b6a = impl_witness_table (%Core.import_ref.d12), @Int.as.Copy.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %C.ref: type = name_ref C, imports.%Main.C [concrete = constants.%C]
// CHECK:STDOUT:   %I.ref: type = name_ref I, imports.%Main.I [concrete = constants.%I.type]
// CHECK:STDOUT:   %I.facet: %I.type = facet_value %C.ref, (constants.%I.impl_witness) [concrete = constants.%I.facet]
// CHECK:STDOUT:   %.loc7_13: %I.type = converted %C.ref, %I.facet [concrete = constants.%I.facet]
// CHECK:STDOUT:   %val.ref: %I.assoc_type = name_ref val, imports.%Main.import_ref.ba4 [concrete = constants.%assoc0.aac]
// CHECK:STDOUT:   %as_type: type = facet_access_type %.loc7_13 [concrete = constants.%C]
// CHECK:STDOUT:   %.loc7_18: type = converted %.loc7_13, %as_type [concrete = constants.%C]
// CHECK:STDOUT:   %impl.elem0.loc7_18.1: %i32 = impl_witness_access constants.%I.impl_witness, element0 [concrete = constants.%int_8]
// CHECK:STDOUT:   %impl.elem0.loc7_18.2: %.65f = impl_witness_access constants.%Copy.impl_witness.fb7, element0 [concrete = constants.%Int.as.Copy.impl.Op.dfd]
// CHECK:STDOUT:   %bound_method.loc7_18.1: <bound method> = bound_method %impl.elem0.loc7_18.1, %impl.elem0.loc7_18.2 [concrete = constants.%Int.as.Copy.impl.Op.bound]
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0.loc7_18.2, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc7_18.2: <bound method> = bound_method %impl.elem0.loc7_18.1, %specific_fn [concrete = constants.%bound_method]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call: init %i32 = call %bound_method.loc7_18.2(%impl.elem0.loc7_18.1) [concrete = constants.%int_8]
// CHECK:STDOUT:   return %Int.as.Copy.impl.Op.call to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_int_literal.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [concrete]
// CHECK:STDOUT:   %int_7: Core.IntLiteral = int_value 7 [concrete]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness imports.%I.impl_witness_table [concrete]
// CHECK:STDOUT:   %I.facet: %I.type = facet_value %C, (%I.impl_witness) [concrete]
// CHECK:STDOUT:   %I.assoc_type: type = assoc_entity_type @I [concrete]
// CHECK:STDOUT:   %assoc0.fd8: %I.assoc_type = assoc_entity element0, imports.%Main.import_ref.d59 [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Copy.Op.type: type = fn_type @Copy.Op [concrete]
// CHECK:STDOUT:   %Copy.impl_witness.104: <witness> = impl_witness imports.%Copy.impl_witness_table.954 [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value Core.IntLiteral, (%Copy.impl_witness.104) [concrete]
// CHECK:STDOUT:   %.7f1: type = fn_type_with_self_type %Copy.Op.type, %Copy.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.Copy.impl.Op.type: type = fn_type @Core.IntLiteral.as.Copy.impl.Op [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.Copy.impl.Op: %Core.IntLiteral.as.Copy.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.Copy.impl.Op.bound: <bound method> = bound_method %int_7, %Core.IntLiteral.as.Copy.impl.Op [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.I: type = import_ref Main//int_literal, I, loaded [concrete = constants.%I.type]
// CHECK:STDOUT:   %Main.C: type = import_ref Main//int_literal, C, loaded [concrete = constants.%C]
// CHECK:STDOUT:   %Main.import_ref.bc2: %I.assoc_type = import_ref Main//int_literal, loc5_10, loaded [concrete = constants.%assoc0.fd8]
// CHECK:STDOUT:   %Main.import_ref.bc3: Core.IntLiteral = import_ref Main//int_literal, loc10_28, loaded [concrete = constants.%int_7]
// CHECK:STDOUT:   %I.impl_witness_table = impl_witness_table (%Main.import_ref.bc3), @C.as.I.impl [concrete]
// CHECK:STDOUT:   %Main.import_ref.d59: Core.IntLiteral = import_ref Main//int_literal, loc5_10, loaded [concrete = %val]
// CHECK:STDOUT:   %val: Core.IntLiteral = assoc_const_decl @val [concrete] {}
// CHECK:STDOUT:   %Core.import_ref.831: %Core.IntLiteral.as.Copy.impl.Op.type = import_ref Core//prelude/copy, loc{{\d+_\d+}}, loaded [concrete = constants.%Core.IntLiteral.as.Copy.impl.Op]
// CHECK:STDOUT:   %Copy.impl_witness_table.954 = impl_witness_table (%Core.import_ref.831), @Core.IntLiteral.as.Copy.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() -> Core.IntLiteral {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %C.ref: type = name_ref C, imports.%Main.C [concrete = constants.%C]
// CHECK:STDOUT:   %I.ref: type = name_ref I, imports.%Main.I [concrete = constants.%I.type]
// CHECK:STDOUT:   %I.facet: %I.type = facet_value %C.ref, (constants.%I.impl_witness) [concrete = constants.%I.facet]
// CHECK:STDOUT:   %.loc7_13: %I.type = converted %C.ref, %I.facet [concrete = constants.%I.facet]
// CHECK:STDOUT:   %val.ref: %I.assoc_type = name_ref val, imports.%Main.import_ref.bc2 [concrete = constants.%assoc0.fd8]
// CHECK:STDOUT:   %as_type: type = facet_access_type %.loc7_13 [concrete = constants.%C]
// CHECK:STDOUT:   %.loc7_18: type = converted %.loc7_13, %as_type [concrete = constants.%C]
// CHECK:STDOUT:   %impl.elem0.loc7_18.1: Core.IntLiteral = impl_witness_access constants.%I.impl_witness, element0 [concrete = constants.%int_7]
// CHECK:STDOUT:   %impl.elem0.loc7_18.2: %.7f1 = impl_witness_access constants.%Copy.impl_witness.104, element0 [concrete = constants.%Core.IntLiteral.as.Copy.impl.Op]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %impl.elem0.loc7_18.1, %impl.elem0.loc7_18.2 [concrete = constants.%Core.IntLiteral.as.Copy.impl.Op.bound]
// CHECK:STDOUT:   %Core.IntLiteral.as.Copy.impl.Op.call: init Core.IntLiteral = call %bound_method(%impl.elem0.loc7_18.1) [concrete = constants.%int_7]
// CHECK:STDOUT:   return %Core.IntLiteral.as.Copy.impl.Op.call to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- unsigned_uint.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %N: Core.IntLiteral = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %u32: type = class_type @UInt, @UInt(%int_32) [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [concrete]
// CHECK:STDOUT:   %int_8: %u32 = int_value 8 [concrete]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness imports.%I.impl_witness_table [concrete]
// CHECK:STDOUT:   %I.facet: %I.type = facet_value %C, (%I.impl_witness) [concrete]
// CHECK:STDOUT:   %I.assoc_type: type = assoc_entity_type @I [concrete]
// CHECK:STDOUT:   %assoc0.633: %I.assoc_type = assoc_entity element0, imports.%Main.import_ref.830 [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Copy.Op.type: type = fn_type @Copy.Op [concrete]
// CHECK:STDOUT:   %UInt.as.Copy.impl.Op.type.f83: type = fn_type @UInt.as.Copy.impl.Op, @UInt.as.Copy.impl(%N) [symbolic]
// CHECK:STDOUT:   %UInt.as.Copy.impl.Op.1d8: %UInt.as.Copy.impl.Op.type.f83 = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.ad8: <witness> = impl_witness imports.%Copy.impl_witness_table.7a2, @UInt.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %UInt.as.Copy.impl.Op.type.68e: type = fn_type @UInt.as.Copy.impl.Op, @UInt.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %UInt.as.Copy.impl.Op.763: %UInt.as.Copy.impl.Op.type.68e = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %u32, (%Copy.impl_witness.ad8) [concrete]
// CHECK:STDOUT:   %.e64: type = fn_type_with_self_type %Copy.Op.type, %Copy.facet [concrete]
// CHECK:STDOUT:   %UInt.as.Copy.impl.Op.bound: <bound method> = bound_method %int_8, %UInt.as.Copy.impl.Op.763 [concrete]
// CHECK:STDOUT:   %UInt.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %UInt.as.Copy.impl.Op.763, @UInt.as.Copy.impl.Op(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %int_8, %UInt.as.Copy.impl.Op.specific_fn [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.I: type = import_ref Main//uint, I, loaded [concrete = constants.%I.type]
// CHECK:STDOUT:   %Main.C: type = import_ref Main//uint, C, loaded [concrete = constants.%C]
// CHECK:STDOUT:   %Main.import_ref.d18: %I.assoc_type = import_ref Main//uint, loc5_10, loaded [concrete = constants.%assoc0.633]
// CHECK:STDOUT:   %Main.import_ref.647: %u32 = import_ref Main//uint, loc10_28, loaded [concrete = constants.%int_8]
// CHECK:STDOUT:   %I.impl_witness_table = impl_witness_table (%Main.import_ref.647), @C.as.I.impl [concrete]
// CHECK:STDOUT:   %Main.import_ref.830: %u32 = import_ref Main//uint, loc5_10, loaded [concrete = %val]
// CHECK:STDOUT:   %val: %u32 = assoc_const_decl @val [concrete] {}
// CHECK:STDOUT:   %Core.import_ref.498: @UInt.as.Copy.impl.%UInt.as.Copy.impl.Op.type (%UInt.as.Copy.impl.Op.type.f83) = import_ref Core//prelude/types/uint, loc{{\d+_\d+}}, loaded [symbolic = @UInt.as.Copy.impl.%UInt.as.Copy.impl.Op (constants.%UInt.as.Copy.impl.Op.1d8)]
// CHECK:STDOUT:   %Copy.impl_witness_table.7a2 = impl_witness_table (%Core.import_ref.498), @UInt.as.Copy.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() -> %u32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %C.ref: type = name_ref C, imports.%Main.C [concrete = constants.%C]
// CHECK:STDOUT:   %I.ref: type = name_ref I, imports.%Main.I [concrete = constants.%I.type]
// CHECK:STDOUT:   %I.facet: %I.type = facet_value %C.ref, (constants.%I.impl_witness) [concrete = constants.%I.facet]
// CHECK:STDOUT:   %.loc7_13: %I.type = converted %C.ref, %I.facet [concrete = constants.%I.facet]
// CHECK:STDOUT:   %val.ref: %I.assoc_type = name_ref val, imports.%Main.import_ref.d18 [concrete = constants.%assoc0.633]
// CHECK:STDOUT:   %as_type: type = facet_access_type %.loc7_13 [concrete = constants.%C]
// CHECK:STDOUT:   %.loc7_18: type = converted %.loc7_13, %as_type [concrete = constants.%C]
// CHECK:STDOUT:   %impl.elem0.loc7_18.1: %u32 = impl_witness_access constants.%I.impl_witness, element0 [concrete = constants.%int_8]
// CHECK:STDOUT:   %impl.elem0.loc7_18.2: %.e64 = impl_witness_access constants.%Copy.impl_witness.ad8, element0 [concrete = constants.%UInt.as.Copy.impl.Op.763]
// CHECK:STDOUT:   %bound_method.loc7_18.1: <bound method> = bound_method %impl.elem0.loc7_18.1, %impl.elem0.loc7_18.2 [concrete = constants.%UInt.as.Copy.impl.Op.bound]
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0.loc7_18.2, @UInt.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%UInt.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc7_18.2: <bound method> = bound_method %impl.elem0.loc7_18.1, %specific_fn [concrete = constants.%bound_method]
// CHECK:STDOUT:   %UInt.as.Copy.impl.Op.call: init %u32 = call %bound_method.loc7_18.2(%impl.elem0.loc7_18.1) [concrete = constants.%int_8]
// CHECK:STDOUT:   return %UInt.as.Copy.impl.Op.call to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
