// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/full.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/builtins/print/int.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/builtins/print/int.carbon
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

import Core library "io";

fn Print(a: i32) = "print.int";

fn Main() {
  //@dump-sem-ir-begin
  Print(1);
  Core.Print(2);
  //@dump-sem-ir-end
}

// CHECK:STDERR: {{.*}}/prelude/types/int.carbon:[[@LINE+600]]:3: error: cannot implicitly convert expression of type `Int(N)` to `Int(N).(AddWith(U).Result)` [ConversionFailure]
// CHECK:STDERR:   fn Op[self: Int(N)](other: Int(N)) -> Int(N) = "int.sadd";
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: {{.*}}/prelude/types/int.carbon:[[@LINE+597]]:3: note: type `Int(N)` does not implement interface `ImplicitAs(Int(N).(AddWith(U).Result))` [MissingImplInMemberAccessNote]
// CHECK:STDERR:   fn Op[self: Int(N)](other: Int(N)) -> Int(N) = "int.sadd";
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: {{.*}}/prelude/types/int.carbon:[[@LINE-46]]:1: in import [InImport]
// CHECK:STDERR: {{.*}}/prelude/operators/arithmetic.carbon:24:3: note: while building thunk to match the signature of this function [ThunkSignature]
// CHECK:STDERR:   fn Op[self: Self](other: Other) -> Result;
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: {{.*}}/prelude/types/int.carbon:[[@LINE+589]]:3: error: cannot implicitly convert expression of type `Int(N)` to `Int(N).(DivWith(U).Result)` [ConversionFailure]
// CHECK:STDERR:   fn Op[self: Int(N)](other: Int(N)) -> Int(N) = "int.sdiv";
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: {{.*}}/prelude/types/int.carbon:[[@LINE+586]]:3: note: type `Int(N)` does not implement interface `ImplicitAs(Int(N).(DivWith(U).Result))` [MissingImplInMemberAccessNote]
// CHECK:STDERR:   fn Op[self: Int(N)](other: Int(N)) -> Int(N) = "int.sdiv";
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: {{.*}}/prelude/types/int.carbon:[[@LINE-57]]:1: in import [InImport]
// CHECK:STDERR: {{.*}}/prelude/operators/arithmetic.carbon:73:3: note: while building thunk to match the signature of this function [ThunkSignature]
// CHECK:STDERR:   fn Op[self: Self](other: Other) -> Result;
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: {{.*}}/prelude/types/int.carbon:[[@LINE+578]]:3: error: cannot implicitly convert expression of type `Int(N)` to `Int(N).(ModWith(U).Result)` [ConversionFailure]
// CHECK:STDERR:   fn Op[self: Int(N)](other: Int(N)) -> Int(N) = "int.smod";
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: {{.*}}/prelude/types/int.carbon:[[@LINE+575]]:3: note: type `Int(N)` does not implement interface `ImplicitAs(Int(N).(ModWith(U).Result))` [MissingImplInMemberAccessNote]
// CHECK:STDERR:   fn Op[self: Int(N)](other: Int(N)) -> Int(N) = "int.smod";
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: {{.*}}/prelude/types/int.carbon:[[@LINE-68]]:1: in import [InImport]
// CHECK:STDERR: {{.*}}/prelude/operators/arithmetic.carbon:84:3: note: while building thunk to match the signature of this function [ThunkSignature]
// CHECK:STDERR:   fn Op[self: Self](other: Other) -> Result;
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: {{.*}}/prelude/types/int.carbon:[[@LINE+567]]:3: error: cannot implicitly convert expression of type `Int(N)` to `Int(N).(MulWith(U).Result)` [ConversionFailure]
// CHECK:STDERR:   fn Op[self: Int(N)](other: Int(N)) -> Int(N) = "int.smul";
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: {{.*}}/prelude/types/int.carbon:[[@LINE+564]]:3: note: type `Int(N)` does not implement interface `ImplicitAs(Int(N).(MulWith(U).Result))` [MissingImplInMemberAccessNote]
// CHECK:STDERR:   fn Op[self: Int(N)](other: Int(N)) -> Int(N) = "int.smul";
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: {{.*}}/prelude/types/int.carbon:[[@LINE-79]]:1: in import [InImport]
// CHECK:STDERR: {{.*}}/prelude/operators/arithmetic.carbon:62:3: note: while building thunk to match the signature of this function [ThunkSignature]
// CHECK:STDERR:   fn Op[self: Self](other: Other) -> Result;
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: {{.*}}/prelude/types/int.carbon:[[@LINE+556]]:3: error: cannot implicitly convert expression of type `Int(N)` to `Int(N).(SubWith(U).Result)` [ConversionFailure]
// CHECK:STDERR:   fn Op[self: Int(N)](other: Int(N)) -> Int(N) = "int.ssub";
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: {{.*}}/prelude/types/int.carbon:[[@LINE+553]]:3: note: type `Int(N)` does not implement interface `ImplicitAs(Int(N).(SubWith(U).Result))` [MissingImplInMemberAccessNote]
// CHECK:STDERR:   fn Op[self: Int(N)](other: Int(N)) -> Int(N) = "int.ssub";
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: {{.*}}/prelude/types/int.carbon:[[@LINE-90]]:1: in import [InImport]
// CHECK:STDERR: {{.*}}/prelude/operators/arithmetic.carbon:46:3: note: while building thunk to match the signature of this function [ThunkSignature]
// CHECK:STDERR:   fn Op[self: Self](other: Other) -> Result;
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: {{.*}}/prelude/types/int.carbon:[[@LINE+545]]:3: error: cannot implicitly convert expression of type `Int(N)` to `T.(AddWith(Int(N)).Result)` [ConversionFailure]
// CHECK:STDERR:   fn Op[self: Int(N)](other: Int(N)) -> Int(N) = "int.sadd";
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: {{.*}}/prelude/types/int.carbon:[[@LINE+542]]:3: note: type `Int(N)` does not implement interface `ImplicitAs(T.(AddWith(Int(N)).Result))` [MissingImplInMemberAccessNote]
// CHECK:STDERR:   fn Op[self: Int(N)](other: Int(N)) -> Int(N) = "int.sadd";
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: {{.*}}/prelude/types/int.carbon:[[@LINE-101]]:1: in import [InImport]
// CHECK:STDERR: {{.*}}/prelude/operators/arithmetic.carbon:24:3: note: while building thunk to match the signature of this function [ThunkSignature]
// CHECK:STDERR:   fn Op[self: Self](other: Other) -> Result;
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: {{.*}}/prelude/types/int.carbon:[[@LINE+534]]:3: error: cannot implicitly convert expression of type `Int(N)` to `T.(DivWith(Int(N)).Result)` [ConversionFailure]
// CHECK:STDERR:   fn Op[self: Int(N)](other: Int(N)) -> Int(N) = "int.sdiv";
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: {{.*}}/prelude/types/int.carbon:[[@LINE+531]]:3: note: type `Int(N)` does not implement interface `ImplicitAs(T.(DivWith(Int(N)).Result))` [MissingImplInMemberAccessNote]
// CHECK:STDERR:   fn Op[self: Int(N)](other: Int(N)) -> Int(N) = "int.sdiv";
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: {{.*}}/prelude/types/int.carbon:[[@LINE-112]]:1: in import [InImport]
// CHECK:STDERR: {{.*}}/prelude/operators/arithmetic.carbon:73:3: note: while building thunk to match the signature of this function [ThunkSignature]
// CHECK:STDERR:   fn Op[self: Self](other: Other) -> Result;
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: {{.*}}/prelude/types/int.carbon:[[@LINE+523]]:3: error: cannot implicitly convert expression of type `Int(N)` to `T.(ModWith(Int(N)).Result)` [ConversionFailure]
// CHECK:STDERR:   fn Op[self: Int(N)](other: Int(N)) -> Int(N) = "int.smod";
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: {{.*}}/prelude/types/int.carbon:[[@LINE+520]]:3: note: type `Int(N)` does not implement interface `ImplicitAs(T.(ModWith(Int(N)).Result))` [MissingImplInMemberAccessNote]
// CHECK:STDERR:   fn Op[self: Int(N)](other: Int(N)) -> Int(N) = "int.smod";
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: {{.*}}/prelude/types/int.carbon:[[@LINE-123]]:1: in import [InImport]
// CHECK:STDERR: {{.*}}/prelude/operators/arithmetic.carbon:84:3: note: while building thunk to match the signature of this function [ThunkSignature]
// CHECK:STDERR:   fn Op[self: Self](other: Other) -> Result;
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: {{.*}}/prelude/types/int.carbon:[[@LINE+512]]:3: error: cannot implicitly convert expression of type `Int(N)` to `T.(MulWith(Int(N)).Result)` [ConversionFailure]
// CHECK:STDERR:   fn Op[self: Int(N)](other: Int(N)) -> Int(N) = "int.smul";
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: {{.*}}/prelude/types/int.carbon:[[@LINE+509]]:3: note: type `Int(N)` does not implement interface `ImplicitAs(T.(MulWith(Int(N)).Result))` [MissingImplInMemberAccessNote]
// CHECK:STDERR:   fn Op[self: Int(N)](other: Int(N)) -> Int(N) = "int.smul";
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: {{.*}}/prelude/types/int.carbon:[[@LINE-134]]:1: in import [InImport]
// CHECK:STDERR: {{.*}}/prelude/operators/arithmetic.carbon:62:3: note: while building thunk to match the signature of this function [ThunkSignature]
// CHECK:STDERR:   fn Op[self: Self](other: Other) -> Result;
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: {{.*}}/prelude/types/int.carbon:[[@LINE+501]]:3: error: cannot implicitly convert expression of type `Int(N)` to `T.(SubWith(Int(N)).Result)` [ConversionFailure]
// CHECK:STDERR:   fn Op[self: Int(N)](other: Int(N)) -> Int(N) = "int.ssub";
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: {{.*}}/prelude/types/int.carbon:[[@LINE+498]]:3: note: type `Int(N)` does not implement interface `ImplicitAs(T.(SubWith(Int(N)).Result))` [MissingImplInMemberAccessNote]
// CHECK:STDERR:   fn Op[self: Int(N)](other: Int(N)) -> Int(N) = "int.ssub";
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: {{.*}}/prelude/types/int.carbon:[[@LINE-145]]:1: in import [InImport]
// CHECK:STDERR: {{.*}}/prelude/operators/arithmetic.carbon:46:3: note: while building thunk to match the signature of this function [ThunkSignature]
// CHECK:STDERR:   fn Op[self: Self](other: Other) -> Result;
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: {{.*}}/prelude/types/int.carbon:[[@LINE+490]]:3: error: cannot implicitly convert expression of type `Int(N)` to `Int(N).(BitAndWith(U).Result)` [ConversionFailure]
// CHECK:STDERR:   fn Op[self: Int(N)](other: Int(N)) -> Int(N) = "int.and";
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: {{.*}}/prelude/types/int.carbon:[[@LINE+487]]:3: note: type `Int(N)` does not implement interface `ImplicitAs(Int(N).(BitAndWith(U).Result))` [MissingImplInMemberAccessNote]
// CHECK:STDERR:   fn Op[self: Int(N)](other: Int(N)) -> Int(N) = "int.and";
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: {{.*}}/prelude/types/int.carbon:[[@LINE-156]]:1: in import [InImport]
// CHECK:STDERR: {{.*}}/prelude/operators/bitwise.carbon:30:3: note: while building thunk to match the signature of this function [ThunkSignature]
// CHECK:STDERR:   fn Op[self: Self](other: Other) -> Result;
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: {{.*}}/prelude/types/int.carbon:[[@LINE+479]]:3: error: cannot implicitly convert expression of type `Int(N)` to `Int(N).(BitOrWith(U).Result)` [ConversionFailure]
// CHECK:STDERR:   fn Op[self: Int(N)](other: Int(N)) -> Int(N) = "int.or";
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: {{.*}}/prelude/types/int.carbon:[[@LINE+476]]:3: note: type `Int(N)` does not implement interface `ImplicitAs(Int(N).(BitOrWith(U).Result))` [MissingImplInMemberAccessNote]
// CHECK:STDERR:   fn Op[self: Int(N)](other: Int(N)) -> Int(N) = "int.or";
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: {{.*}}/prelude/types/int.carbon:[[@LINE-167]]:1: in import [InImport]
// CHECK:STDERR: {{.*}}/prelude/operators/bitwise.carbon:41:3: note: while building thunk to match the signature of this function [ThunkSignature]
// CHECK:STDERR:   fn Op[self: Self](other: Other) -> Result;
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: {{.*}}/prelude/types/int.carbon:[[@LINE+468]]:3: error: cannot implicitly convert expression of type `Int(N)` to `Int(N).(BitXorWith(U).Result)` [ConversionFailure]
// CHECK:STDERR:   fn Op[self: Int(N)](other: Int(N)) -> Int(N) = "int.xor";
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: {{.*}}/prelude/types/int.carbon:[[@LINE+465]]:3: note: type `Int(N)` does not implement interface `ImplicitAs(Int(N).(BitXorWith(U).Result))` [MissingImplInMemberAccessNote]
// CHECK:STDERR:   fn Op[self: Int(N)](other: Int(N)) -> Int(N) = "int.xor";
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: {{.*}}/prelude/types/int.carbon:[[@LINE-178]]:1: in import [InImport]
// CHECK:STDERR: {{.*}}/prelude/operators/bitwise.carbon:52:3: note: while building thunk to match the signature of this function [ThunkSignature]
// CHECK:STDERR:   fn Op[self: Self](other: Other) -> Result;
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: {{.*}}/prelude/types/int.carbon:[[@LINE+457]]:3: error: cannot implicitly convert expression of type `Int(N)` to `Int(N).(LeftShiftWith(U).Result)` [ConversionFailure]
// CHECK:STDERR:   fn Op[self: Int(N)](other: Int(N)) -> Int(N) = "int.left_shift";
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: {{.*}}/prelude/types/int.carbon:[[@LINE+454]]:3: note: type `Int(N)` does not implement interface `ImplicitAs(Int(N).(LeftShiftWith(U).Result))` [MissingImplInMemberAccessNote]
// CHECK:STDERR:   fn Op[self: Int(N)](other: Int(N)) -> Int(N) = "int.left_shift";
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: {{.*}}/prelude/types/int.carbon:[[@LINE-189]]:1: in import [InImport]
// CHECK:STDERR: {{.*}}/prelude/operators/bitwise.carbon:63:3: note: while building thunk to match the signature of this function [ThunkSignature]
// CHECK:STDERR:   fn Op[self: Self](other: Other) -> Result;
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: {{.*}}/prelude/types/int.carbon:[[@LINE+446]]:3: error: cannot implicitly convert expression of type `Int(N)` to `Int(N).(RightShiftWith(U).Result)` [ConversionFailure]
// CHECK:STDERR:   fn Op[self: Int(N)](other: Int(N)) -> Int(N) = "int.right_shift";
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: {{.*}}/prelude/types/int.carbon:[[@LINE+443]]:3: note: type `Int(N)` does not implement interface `ImplicitAs(Int(N).(RightShiftWith(U).Result))` [MissingImplInMemberAccessNote]
// CHECK:STDERR:   fn Op[self: Int(N)](other: Int(N)) -> Int(N) = "int.right_shift";
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: {{.*}}/prelude/types/int.carbon:[[@LINE-200]]:1: in import [InImport]
// CHECK:STDERR: {{.*}}/prelude/operators/bitwise.carbon:74:3: note: while building thunk to match the signature of this function [ThunkSignature]
// CHECK:STDERR:   fn Op[self: Self](other: Other) -> Result;
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: {{.*}}/prelude/types/int.carbon:[[@LINE+435]]:3: error: cannot implicitly convert expression of type `Int(N)` to `T.(BitAndWith(Int(N)).Result)` [ConversionFailure]
// CHECK:STDERR:   fn Op[self: Int(N)](other: Int(N)) -> Int(N) = "int.and";
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: {{.*}}/prelude/types/int.carbon:[[@LINE+432]]:3: note: type `Int(N)` does not implement interface `ImplicitAs(T.(BitAndWith(Int(N)).Result))` [MissingImplInMemberAccessNote]
// CHECK:STDERR:   fn Op[self: Int(N)](other: Int(N)) -> Int(N) = "int.and";
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: {{.*}}/prelude/types/int.carbon:[[@LINE-211]]:1: in import [InImport]
// CHECK:STDERR: {{.*}}/prelude/operators/bitwise.carbon:30:3: note: while building thunk to match the signature of this function [ThunkSignature]
// CHECK:STDERR:   fn Op[self: Self](other: Other) -> Result;
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: {{.*}}/prelude/types/int.carbon:[[@LINE+424]]:3: error: cannot implicitly convert expression of type `Int(N)` to `T.(BitOrWith(Int(N)).Result)` [ConversionFailure]
// CHECK:STDERR:   fn Op[self: Int(N)](other: Int(N)) -> Int(N) = "int.or";
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: {{.*}}/prelude/types/int.carbon:[[@LINE+421]]:3: note: type `Int(N)` does not implement interface `ImplicitAs(T.(BitOrWith(Int(N)).Result))` [MissingImplInMemberAccessNote]
// CHECK:STDERR:   fn Op[self: Int(N)](other: Int(N)) -> Int(N) = "int.or";
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: {{.*}}/prelude/types/int.carbon:[[@LINE-222]]:1: in import [InImport]
// CHECK:STDERR: {{.*}}/prelude/operators/bitwise.carbon:41:3: note: while building thunk to match the signature of this function [ThunkSignature]
// CHECK:STDERR:   fn Op[self: Self](other: Other) -> Result;
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: {{.*}}/prelude/types/int.carbon:[[@LINE+413]]:3: error: cannot implicitly convert expression of type `Int(N)` to `T.(BitXorWith(Int(N)).Result)` [ConversionFailure]
// CHECK:STDERR:   fn Op[self: Int(N)](other: Int(N)) -> Int(N) = "int.xor";
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: {{.*}}/prelude/types/int.carbon:[[@LINE+410]]:3: note: type `Int(N)` does not implement interface `ImplicitAs(T.(BitXorWith(Int(N)).Result))` [MissingImplInMemberAccessNote]
// CHECK:STDERR:   fn Op[self: Int(N)](other: Int(N)) -> Int(N) = "int.xor";
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: {{.*}}/prelude/types/int.carbon:[[@LINE-233]]:1: in import [InImport]
// CHECK:STDERR: {{.*}}/prelude/operators/bitwise.carbon:52:3: note: while building thunk to match the signature of this function [ThunkSignature]
// CHECK:STDERR:   fn Op[self: Self](other: Other) -> Result;
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: {{.*}}/prelude/types/int.carbon:[[@LINE+402]]:3: error: cannot implicitly convert expression of type `Int(N)` to `T.(LeftShiftWith(Int(N)).Result)` [ConversionFailure]
// CHECK:STDERR:   fn Op[self: Int(N)](other: Int(N)) -> Int(N) = "int.left_shift";
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: {{.*}}/prelude/types/int.carbon:[[@LINE+399]]:3: note: type `Int(N)` does not implement interface `ImplicitAs(T.(LeftShiftWith(Int(N)).Result))` [MissingImplInMemberAccessNote]
// CHECK:STDERR:   fn Op[self: Int(N)](other: Int(N)) -> Int(N) = "int.left_shift";
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: {{.*}}/prelude/types/int.carbon:[[@LINE-244]]:1: in import [InImport]
// CHECK:STDERR: {{.*}}/prelude/operators/bitwise.carbon:63:3: note: while building thunk to match the signature of this function [ThunkSignature]
// CHECK:STDERR:   fn Op[self: Self](other: Other) -> Result;
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: {{.*}}/prelude/types/int.carbon:[[@LINE+391]]:3: error: cannot implicitly convert expression of type `Int(N)` to `T.(RightShiftWith(Int(N)).Result)` [ConversionFailure]
// CHECK:STDERR:   fn Op[self: Int(N)](other: Int(N)) -> Int(N) = "int.right_shift";
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: {{.*}}/prelude/types/int.carbon:[[@LINE+388]]:3: note: type `Int(N)` does not implement interface `ImplicitAs(T.(RightShiftWith(Int(N)).Result))` [MissingImplInMemberAccessNote]
// CHECK:STDERR:   fn Op[self: Int(N)](other: Int(N)) -> Int(N) = "int.right_shift";
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: {{.*}}/prelude/types/int.carbon:[[@LINE-255]]:1: in import [InImport]
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
// CHECK:STDERR: {{.*}}/prelude/types/uint.carbon:[[@LINE+334]]:3: error: cannot implicitly convert expression of type `UInt(N)` to `UInt(N).(AddWith(U).Result)` [ConversionFailure]
// CHECK:STDERR:   fn Op[self: UInt(N)](other: UInt(N)) -> UInt(N) = "int.uadd";
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: {{.*}}/prelude/types/uint.carbon:[[@LINE+331]]:3: note: type `UInt(N)` does not implement interface `ImplicitAs(UInt(N).(AddWith(U).Result))` [MissingImplInMemberAccessNote]
// CHECK:STDERR:   fn Op[self: UInt(N)](other: UInt(N)) -> UInt(N) = "int.uadd";
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: {{.*}}/prelude/types/uint.carbon:[[@LINE-312]]:1: in import [InImport]
// CHECK:STDERR: {{.*}}/prelude/operators/arithmetic.carbon:24:3: note: while building thunk to match the signature of this function [ThunkSignature]
// CHECK:STDERR:   fn Op[self: Self](other: Other) -> Result;
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: {{.*}}/prelude/types/uint.carbon:[[@LINE+323]]:3: error: cannot implicitly convert expression of type `UInt(N)` to `UInt(N).(DivWith(U).Result)` [ConversionFailure]
// CHECK:STDERR:   fn Op[self: UInt(N)](other: UInt(N)) -> UInt(N) = "int.udiv";
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: {{.*}}/prelude/types/uint.carbon:[[@LINE+320]]:3: note: type `UInt(N)` does not implement interface `ImplicitAs(UInt(N).(DivWith(U).Result))` [MissingImplInMemberAccessNote]
// CHECK:STDERR:   fn Op[self: UInt(N)](other: UInt(N)) -> UInt(N) = "int.udiv";
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: {{.*}}/prelude/types/uint.carbon:[[@LINE-323]]:1: in import [InImport]
// CHECK:STDERR: {{.*}}/prelude/operators/arithmetic.carbon:73:3: note: while building thunk to match the signature of this function [ThunkSignature]
// CHECK:STDERR:   fn Op[self: Self](other: Other) -> Result;
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: {{.*}}/prelude/types/uint.carbon:[[@LINE+312]]:3: error: cannot implicitly convert expression of type `UInt(N)` to `UInt(N).(ModWith(U).Result)` [ConversionFailure]
// CHECK:STDERR:   fn Op[self: UInt(N)](other: UInt(N)) -> UInt(N) = "int.umod";
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: {{.*}}/prelude/types/uint.carbon:[[@LINE+309]]:3: note: type `UInt(N)` does not implement interface `ImplicitAs(UInt(N).(ModWith(U).Result))` [MissingImplInMemberAccessNote]
// CHECK:STDERR:   fn Op[self: UInt(N)](other: UInt(N)) -> UInt(N) = "int.umod";
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: {{.*}}/prelude/types/uint.carbon:[[@LINE-334]]:1: in import [InImport]
// CHECK:STDERR: {{.*}}/prelude/operators/arithmetic.carbon:84:3: note: while building thunk to match the signature of this function [ThunkSignature]
// CHECK:STDERR:   fn Op[self: Self](other: Other) -> Result;
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: {{.*}}/prelude/types/uint.carbon:[[@LINE+301]]:3: error: cannot implicitly convert expression of type `UInt(N)` to `UInt(N).(MulWith(U).Result)` [ConversionFailure]
// CHECK:STDERR:   fn Op[self: UInt(N)](other: UInt(N)) -> UInt(N) = "int.umul";
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: {{.*}}/prelude/types/uint.carbon:[[@LINE+298]]:3: note: type `UInt(N)` does not implement interface `ImplicitAs(UInt(N).(MulWith(U).Result))` [MissingImplInMemberAccessNote]
// CHECK:STDERR:   fn Op[self: UInt(N)](other: UInt(N)) -> UInt(N) = "int.umul";
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: {{.*}}/prelude/types/uint.carbon:[[@LINE-345]]:1: in import [InImport]
// CHECK:STDERR: {{.*}}/prelude/operators/arithmetic.carbon:62:3: note: while building thunk to match the signature of this function [ThunkSignature]
// CHECK:STDERR:   fn Op[self: Self](other: Other) -> Result;
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: {{.*}}/prelude/types/uint.carbon:[[@LINE+290]]:3: error: cannot implicitly convert expression of type `UInt(N)` to `UInt(N).(SubWith(U).Result)` [ConversionFailure]
// CHECK:STDERR:   fn Op[self: UInt(N)](other: UInt(N)) -> UInt(N) = "int.usub";
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: {{.*}}/prelude/types/uint.carbon:[[@LINE+287]]:3: note: type `UInt(N)` does not implement interface `ImplicitAs(UInt(N).(SubWith(U).Result))` [MissingImplInMemberAccessNote]
// CHECK:STDERR:   fn Op[self: UInt(N)](other: UInt(N)) -> UInt(N) = "int.usub";
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: {{.*}}/prelude/types/uint.carbon:[[@LINE-356]]:1: in import [InImport]
// CHECK:STDERR: {{.*}}/prelude/operators/arithmetic.carbon:46:3: note: while building thunk to match the signature of this function [ThunkSignature]
// CHECK:STDERR:   fn Op[self: Self](other: Other) -> Result;
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: {{.*}}/prelude/types/uint.carbon:[[@LINE+279]]:3: error: cannot implicitly convert expression of type `UInt(N)` to `T.(AddWith(UInt(N)).Result)` [ConversionFailure]
// CHECK:STDERR:   fn Op[self: UInt(N)](other: UInt(N)) -> UInt(N) = "int.uadd";
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: {{.*}}/prelude/types/uint.carbon:[[@LINE+276]]:3: note: type `UInt(N)` does not implement interface `ImplicitAs(T.(AddWith(UInt(N)).Result))` [MissingImplInMemberAccessNote]
// CHECK:STDERR:   fn Op[self: UInt(N)](other: UInt(N)) -> UInt(N) = "int.uadd";
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: {{.*}}/prelude/types/uint.carbon:[[@LINE-367]]:1: in import [InImport]
// CHECK:STDERR: {{.*}}/prelude/operators/arithmetic.carbon:24:3: note: while building thunk to match the signature of this function [ThunkSignature]
// CHECK:STDERR:   fn Op[self: Self](other: Other) -> Result;
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: {{.*}}/prelude/types/uint.carbon:[[@LINE+268]]:3: error: cannot implicitly convert expression of type `UInt(N)` to `T.(DivWith(UInt(N)).Result)` [ConversionFailure]
// CHECK:STDERR:   fn Op[self: UInt(N)](other: UInt(N)) -> UInt(N) = "int.udiv";
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: {{.*}}/prelude/types/uint.carbon:[[@LINE+265]]:3: note: type `UInt(N)` does not implement interface `ImplicitAs(T.(DivWith(UInt(N)).Result))` [MissingImplInMemberAccessNote]
// CHECK:STDERR:   fn Op[self: UInt(N)](other: UInt(N)) -> UInt(N) = "int.udiv";
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: {{.*}}/prelude/types/uint.carbon:[[@LINE-378]]:1: in import [InImport]
// CHECK:STDERR: {{.*}}/prelude/operators/arithmetic.carbon:73:3: note: while building thunk to match the signature of this function [ThunkSignature]
// CHECK:STDERR:   fn Op[self: Self](other: Other) -> Result;
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: {{.*}}/prelude/types/uint.carbon:[[@LINE+257]]:3: error: cannot implicitly convert expression of type `UInt(N)` to `T.(ModWith(UInt(N)).Result)` [ConversionFailure]
// CHECK:STDERR:   fn Op[self: UInt(N)](other: UInt(N)) -> UInt(N) = "int.umod";
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: {{.*}}/prelude/types/uint.carbon:[[@LINE+254]]:3: note: type `UInt(N)` does not implement interface `ImplicitAs(T.(ModWith(UInt(N)).Result))` [MissingImplInMemberAccessNote]
// CHECK:STDERR:   fn Op[self: UInt(N)](other: UInt(N)) -> UInt(N) = "int.umod";
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: {{.*}}/prelude/types/uint.carbon:[[@LINE-389]]:1: in import [InImport]
// CHECK:STDERR: {{.*}}/prelude/operators/arithmetic.carbon:84:3: note: while building thunk to match the signature of this function [ThunkSignature]
// CHECK:STDERR:   fn Op[self: Self](other: Other) -> Result;
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: {{.*}}/prelude/types/uint.carbon:[[@LINE+246]]:3: error: cannot implicitly convert expression of type `UInt(N)` to `T.(MulWith(UInt(N)).Result)` [ConversionFailure]
// CHECK:STDERR:   fn Op[self: UInt(N)](other: UInt(N)) -> UInt(N) = "int.umul";
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: {{.*}}/prelude/types/uint.carbon:[[@LINE+243]]:3: note: type `UInt(N)` does not implement interface `ImplicitAs(T.(MulWith(UInt(N)).Result))` [MissingImplInMemberAccessNote]
// CHECK:STDERR:   fn Op[self: UInt(N)](other: UInt(N)) -> UInt(N) = "int.umul";
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: {{.*}}/prelude/types/uint.carbon:[[@LINE-400]]:1: in import [InImport]
// CHECK:STDERR: {{.*}}/prelude/operators/arithmetic.carbon:62:3: note: while building thunk to match the signature of this function [ThunkSignature]
// CHECK:STDERR:   fn Op[self: Self](other: Other) -> Result;
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: {{.*}}/prelude/types/uint.carbon:[[@LINE+235]]:3: error: cannot implicitly convert expression of type `UInt(N)` to `T.(SubWith(UInt(N)).Result)` [ConversionFailure]
// CHECK:STDERR:   fn Op[self: UInt(N)](other: UInt(N)) -> UInt(N) = "int.usub";
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: {{.*}}/prelude/types/uint.carbon:[[@LINE+232]]:3: note: type `UInt(N)` does not implement interface `ImplicitAs(T.(SubWith(UInt(N)).Result))` [MissingImplInMemberAccessNote]
// CHECK:STDERR:   fn Op[self: UInt(N)](other: UInt(N)) -> UInt(N) = "int.usub";
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: {{.*}}/prelude/types/uint.carbon:[[@LINE-411]]:1: in import [InImport]
// CHECK:STDERR: {{.*}}/prelude/operators/arithmetic.carbon:46:3: note: while building thunk to match the signature of this function [ThunkSignature]
// CHECK:STDERR:   fn Op[self: Self](other: Other) -> Result;
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: {{.*}}/prelude/types/uint.carbon:[[@LINE+224]]:3: error: cannot implicitly convert expression of type `UInt(N)` to `UInt(N).(BitAndWith(U).Result)` [ConversionFailure]
// CHECK:STDERR:   fn Op[self: UInt(N)](other: UInt(N)) -> UInt(N) = "int.and";
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: {{.*}}/prelude/types/uint.carbon:[[@LINE+221]]:3: note: type `UInt(N)` does not implement interface `ImplicitAs(UInt(N).(BitAndWith(U).Result))` [MissingImplInMemberAccessNote]
// CHECK:STDERR:   fn Op[self: UInt(N)](other: UInt(N)) -> UInt(N) = "int.and";
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: {{.*}}/prelude/types/uint.carbon:[[@LINE-422]]:1: in import [InImport]
// CHECK:STDERR: {{.*}}/prelude/operators/bitwise.carbon:30:3: note: while building thunk to match the signature of this function [ThunkSignature]
// CHECK:STDERR:   fn Op[self: Self](other: Other) -> Result;
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: {{.*}}/prelude/types/uint.carbon:[[@LINE+213]]:3: error: cannot implicitly convert expression of type `UInt(N)` to `UInt(N).(BitOrWith(U).Result)` [ConversionFailure]
// CHECK:STDERR:   fn Op[self: UInt(N)](other: UInt(N)) -> UInt(N) = "int.or";
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: {{.*}}/prelude/types/uint.carbon:[[@LINE+210]]:3: note: type `UInt(N)` does not implement interface `ImplicitAs(UInt(N).(BitOrWith(U).Result))` [MissingImplInMemberAccessNote]
// CHECK:STDERR:   fn Op[self: UInt(N)](other: UInt(N)) -> UInt(N) = "int.or";
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: {{.*}}/prelude/types/uint.carbon:[[@LINE-433]]:1: in import [InImport]
// CHECK:STDERR: {{.*}}/prelude/operators/bitwise.carbon:41:3: note: while building thunk to match the signature of this function [ThunkSignature]
// CHECK:STDERR:   fn Op[self: Self](other: Other) -> Result;
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: {{.*}}/prelude/types/uint.carbon:[[@LINE+202]]:3: error: cannot implicitly convert expression of type `UInt(N)` to `UInt(N).(BitXorWith(U).Result)` [ConversionFailure]
// CHECK:STDERR:   fn Op[self: UInt(N)](other: UInt(N)) -> UInt(N) = "int.xor";
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: {{.*}}/prelude/types/uint.carbon:[[@LINE+199]]:3: note: type `UInt(N)` does not implement interface `ImplicitAs(UInt(N).(BitXorWith(U).Result))` [MissingImplInMemberAccessNote]
// CHECK:STDERR:   fn Op[self: UInt(N)](other: UInt(N)) -> UInt(N) = "int.xor";
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: {{.*}}/prelude/types/uint.carbon:[[@LINE-444]]:1: in import [InImport]
// CHECK:STDERR: {{.*}}/prelude/operators/bitwise.carbon:52:3: note: while building thunk to match the signature of this function [ThunkSignature]
// CHECK:STDERR:   fn Op[self: Self](other: Other) -> Result;
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: {{.*}}/prelude/types/uint.carbon:[[@LINE+191]]:3: error: cannot implicitly convert expression of type `UInt(N)` to `UInt(N).(LeftShiftWith(U).Result)` [ConversionFailure]
// CHECK:STDERR:   fn Op[self: UInt(N)](other: UInt(N)) -> UInt(N) = "int.left_shift";
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: {{.*}}/prelude/types/uint.carbon:[[@LINE+188]]:3: note: type `UInt(N)` does not implement interface `ImplicitAs(UInt(N).(LeftShiftWith(U).Result))` [MissingImplInMemberAccessNote]
// CHECK:STDERR:   fn Op[self: UInt(N)](other: UInt(N)) -> UInt(N) = "int.left_shift";
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: {{.*}}/prelude/types/uint.carbon:[[@LINE-455]]:1: in import [InImport]
// CHECK:STDERR: {{.*}}/prelude/operators/bitwise.carbon:63:3: note: while building thunk to match the signature of this function [ThunkSignature]
// CHECK:STDERR:   fn Op[self: Self](other: Other) -> Result;
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: {{.*}}/prelude/types/uint.carbon:[[@LINE+180]]:3: error: cannot implicitly convert expression of type `UInt(N)` to `UInt(N).(RightShiftWith(U).Result)` [ConversionFailure]
// CHECK:STDERR:   fn Op[self: UInt(N)](other: UInt(N)) -> UInt(N) = "int.right_shift";
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: {{.*}}/prelude/types/uint.carbon:[[@LINE+177]]:3: note: type `UInt(N)` does not implement interface `ImplicitAs(UInt(N).(RightShiftWith(U).Result))` [MissingImplInMemberAccessNote]
// CHECK:STDERR:   fn Op[self: UInt(N)](other: UInt(N)) -> UInt(N) = "int.right_shift";
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: {{.*}}/prelude/types/uint.carbon:[[@LINE-466]]:1: in import [InImport]
// CHECK:STDERR: {{.*}}/prelude/operators/bitwise.carbon:74:3: note: while building thunk to match the signature of this function [ThunkSignature]
// CHECK:STDERR:   fn Op[self: Self](other: Other) -> Result;
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: {{.*}}/prelude/types/uint.carbon:[[@LINE+169]]:3: error: cannot implicitly convert expression of type `UInt(N)` to `T.(BitAndWith(UInt(N)).Result)` [ConversionFailure]
// CHECK:STDERR:   fn Op[self: UInt(N)](other: UInt(N)) -> UInt(N) = "int.and";
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: {{.*}}/prelude/types/uint.carbon:[[@LINE+166]]:3: note: type `UInt(N)` does not implement interface `ImplicitAs(T.(BitAndWith(UInt(N)).Result))` [MissingImplInMemberAccessNote]
// CHECK:STDERR:   fn Op[self: UInt(N)](other: UInt(N)) -> UInt(N) = "int.and";
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: {{.*}}/prelude/types/uint.carbon:[[@LINE-477]]:1: in import [InImport]
// CHECK:STDERR: {{.*}}/prelude/operators/bitwise.carbon:30:3: note: while building thunk to match the signature of this function [ThunkSignature]
// CHECK:STDERR:   fn Op[self: Self](other: Other) -> Result;
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: {{.*}}/prelude/types/uint.carbon:[[@LINE+158]]:3: error: cannot implicitly convert expression of type `UInt(N)` to `T.(BitOrWith(UInt(N)).Result)` [ConversionFailure]
// CHECK:STDERR:   fn Op[self: UInt(N)](other: UInt(N)) -> UInt(N) = "int.or";
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: {{.*}}/prelude/types/uint.carbon:[[@LINE+155]]:3: note: type `UInt(N)` does not implement interface `ImplicitAs(T.(BitOrWith(UInt(N)).Result))` [MissingImplInMemberAccessNote]
// CHECK:STDERR:   fn Op[self: UInt(N)](other: UInt(N)) -> UInt(N) = "int.or";
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: {{.*}}/prelude/types/uint.carbon:[[@LINE-488]]:1: in import [InImport]
// CHECK:STDERR: {{.*}}/prelude/operators/bitwise.carbon:41:3: note: while building thunk to match the signature of this function [ThunkSignature]
// CHECK:STDERR:   fn Op[self: Self](other: Other) -> Result;
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: {{.*}}/prelude/types/uint.carbon:[[@LINE+147]]:3: error: cannot implicitly convert expression of type `UInt(N)` to `T.(BitXorWith(UInt(N)).Result)` [ConversionFailure]
// CHECK:STDERR:   fn Op[self: UInt(N)](other: UInt(N)) -> UInt(N) = "int.xor";
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: {{.*}}/prelude/types/uint.carbon:[[@LINE+144]]:3: note: type `UInt(N)` does not implement interface `ImplicitAs(T.(BitXorWith(UInt(N)).Result))` [MissingImplInMemberAccessNote]
// CHECK:STDERR:   fn Op[self: UInt(N)](other: UInt(N)) -> UInt(N) = "int.xor";
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: {{.*}}/prelude/types/uint.carbon:[[@LINE-499]]:1: in import [InImport]
// CHECK:STDERR: {{.*}}/prelude/operators/bitwise.carbon:52:3: note: while building thunk to match the signature of this function [ThunkSignature]
// CHECK:STDERR:   fn Op[self: Self](other: Other) -> Result;
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: {{.*}}/prelude/types/uint.carbon:[[@LINE+136]]:3: error: cannot implicitly convert expression of type `UInt(N)` to `T.(LeftShiftWith(UInt(N)).Result)` [ConversionFailure]
// CHECK:STDERR:   fn Op[self: UInt(N)](other: UInt(N)) -> UInt(N) = "int.left_shift";
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: {{.*}}/prelude/types/uint.carbon:[[@LINE+133]]:3: note: type `UInt(N)` does not implement interface `ImplicitAs(T.(LeftShiftWith(UInt(N)).Result))` [MissingImplInMemberAccessNote]
// CHECK:STDERR:   fn Op[self: UInt(N)](other: UInt(N)) -> UInt(N) = "int.left_shift";
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: {{.*}}/prelude/types/uint.carbon:[[@LINE-510]]:1: in import [InImport]
// CHECK:STDERR: {{.*}}/prelude/operators/bitwise.carbon:63:3: note: while building thunk to match the signature of this function [ThunkSignature]
// CHECK:STDERR:   fn Op[self: Self](other: Other) -> Result;
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: {{.*}}/prelude/types/uint.carbon:[[@LINE+125]]:3: error: cannot implicitly convert expression of type `UInt(N)` to `T.(RightShiftWith(UInt(N)).Result)` [ConversionFailure]
// CHECK:STDERR:   fn Op[self: UInt(N)](other: UInt(N)) -> UInt(N) = "int.right_shift";
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: {{.*}}/prelude/types/uint.carbon:[[@LINE+122]]:3: note: type `UInt(N)` does not implement interface `ImplicitAs(T.(RightShiftWith(UInt(N)).Result))` [MissingImplInMemberAccessNote]
// CHECK:STDERR:   fn Op[self: UInt(N)](other: UInt(N)) -> UInt(N) = "int.right_shift";
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: {{.*}}/prelude/types/uint.carbon:[[@LINE-521]]:1: in import [InImport]
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
// CHECK:STDOUT: --- int.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %Print.type.980: type = fn_type @Print.loc38 [concrete]
// CHECK:STDOUT:   %Print.b7c: %Print.type.980 = struct_value () [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.cc7: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.cc7 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.d14: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.Convert.type.1b6: type = fn_type @ImplicitAs.Convert, @ImplicitAs(%i32) [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.f51: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.2a1: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.f51 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.bc9: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.132, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.51e: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.e9b: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.51e = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.d14 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.bc9) [concrete]
// CHECK:STDOUT:   %.322: type = fn_type_with_self_type %ImplicitAs.Convert.type.1b6, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.265: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.e9b [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.e9b, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.dfc: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_1.5d2: %i32 = int_value 1 [concrete]
// CHECK:STDOUT:   %Print.type.6ed: type = fn_type @Print.1 [concrete]
// CHECK:STDOUT:   %Print.723: %Print.type.6ed = struct_value () [concrete]
// CHECK:STDOUT:   %int_2.ecc: Core.IntLiteral = int_value 2 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.393: <bound method> = bound_method %int_2.ecc, %Core.IntLiteral.as.ImplicitAs.impl.Convert.e9b [concrete]
// CHECK:STDOUT:   %bound_method.9f7: <bound method> = bound_method %int_2.ecc, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_2.ef8: %i32 = int_value 2 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     .Print = %Core.Print
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//io
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/types/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.cc7 = import_ref Core//prelude/operators/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:   %Core.import_ref.e24: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.f51) = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.2a1)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.132 = impl_witness_table (%Core.import_ref.e24), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %Core.Print: %Print.type.6ed = import_ref Core//io, Print, loaded [concrete = constants.%Print.723]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Main() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Print.ref.loc42: %Print.type.980 = name_ref Print, file.%Print.decl [concrete = constants.%Print.b7c]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem0.loc42: %.322 = impl_witness_access constants.%ImplicitAs.impl_witness.bc9, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.e9b]
// CHECK:STDOUT:   %bound_method.loc42_9.1: <bound method> = bound_method %int_1, %impl.elem0.loc42 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.265]
// CHECK:STDOUT:   %specific_fn.loc42: <specific function> = specific_function %impl.elem0.loc42, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc42_9.2: <bound method> = bound_method %int_1, %specific_fn.loc42 [concrete = constants.%bound_method.dfc]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc42: init %i32 = call %bound_method.loc42_9.2(%int_1) [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc42_9.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc42 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc42_9.2: %i32 = converted %int_1, %.loc42_9.1 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %Print.call.loc42: init %empty_tuple.type = call %Print.ref.loc42(%.loc42_9.2)
// CHECK:STDOUT:   %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:   %Print.ref.loc43: %Print.type.6ed = name_ref Print, imports.%Core.Print [concrete = constants.%Print.723]
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:   %impl.elem0.loc43: %.322 = impl_witness_access constants.%ImplicitAs.impl_witness.bc9, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.e9b]
// CHECK:STDOUT:   %bound_method.loc43_14.1: <bound method> = bound_method %int_2, %impl.elem0.loc43 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.393]
// CHECK:STDOUT:   %specific_fn.loc43: <specific function> = specific_function %impl.elem0.loc43, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc43_14.2: <bound method> = bound_method %int_2, %specific_fn.loc43 [concrete = constants.%bound_method.9f7]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc43: init %i32 = call %bound_method.loc43_14.2(%int_2) [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %.loc43_14.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc43 [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %.loc43_14.2: %i32 = converted %int_2, %.loc43_14.1 [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %Print.call.loc43: init %empty_tuple.type = call %Print.ref.loc43(%.loc43_14.2)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
