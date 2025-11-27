// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/full.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/array/bound_values.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/array/bound_values.carbon
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
// CHECK:STDERR: {{.*}}/prelude/types/int.carbon:133:3: error: cannot implicitly convert expression of type `Int(N)` to `Int(N).(AddWith(U).Result)` [ConversionFailure]
// CHECK:STDERR:   fn Op[self: Int(N)](other: Int(N)) -> Int(N) = "int.sadd";
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: {{.*}}/prelude/types/int.carbon:133:3: note: type `Int(N)` does not implement interface `ImplicitAs(Int(N).(AddWith(U).Result))` [MissingImplInMemberAccessNote]
// CHECK:STDERR:   fn Op[self: Int(N)](other: Int(N)) -> Int(N) = "int.sadd";
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: {{.*}}/prelude/types/int.carbon:7:1: in import [InImport]
// CHECK:STDERR: {{.*}}/prelude/operators/arithmetic.carbon:24:3: note: while building thunk to match the signature of this function [ThunkSignature]
// CHECK:STDERR:   fn Op[self: Self](other: Other) -> Result;
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: {{.*}}/prelude/types/int.carbon:138:3: error: cannot implicitly convert expression of type `Int(N)` to `Int(N).(DivWith(U).Result)` [ConversionFailure]
// CHECK:STDERR:   fn Op[self: Int(N)](other: Int(N)) -> Int(N) = "int.sdiv";
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: {{.*}}/prelude/types/int.carbon:138:3: note: type `Int(N)` does not implement interface `ImplicitAs(Int(N).(DivWith(U).Result))` [MissingImplInMemberAccessNote]
// CHECK:STDERR:   fn Op[self: Int(N)](other: Int(N)) -> Int(N) = "int.sdiv";
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: {{.*}}/prelude/types/int.carbon:7:1: in import [InImport]
// CHECK:STDERR: {{.*}}/prelude/operators/arithmetic.carbon:73:3: note: while building thunk to match the signature of this function [ThunkSignature]
// CHECK:STDERR:   fn Op[self: Self](other: Other) -> Result;
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: {{.*}}/prelude/types/int.carbon:143:3: error: cannot implicitly convert expression of type `Int(N)` to `Int(N).(ModWith(U).Result)` [ConversionFailure]
// CHECK:STDERR:   fn Op[self: Int(N)](other: Int(N)) -> Int(N) = "int.smod";
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: {{.*}}/prelude/types/int.carbon:143:3: note: type `Int(N)` does not implement interface `ImplicitAs(Int(N).(ModWith(U).Result))` [MissingImplInMemberAccessNote]
// CHECK:STDERR:   fn Op[self: Int(N)](other: Int(N)) -> Int(N) = "int.smod";
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: {{.*}}/prelude/types/int.carbon:7:1: in import [InImport]
// CHECK:STDERR: {{.*}}/prelude/operators/arithmetic.carbon:84:3: note: while building thunk to match the signature of this function [ThunkSignature]
// CHECK:STDERR:   fn Op[self: Self](other: Other) -> Result;
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: {{.*}}/prelude/types/int.carbon:148:3: error: cannot implicitly convert expression of type `Int(N)` to `Int(N).(MulWith(U).Result)` [ConversionFailure]
// CHECK:STDERR:   fn Op[self: Int(N)](other: Int(N)) -> Int(N) = "int.smul";
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: {{.*}}/prelude/types/int.carbon:148:3: note: type `Int(N)` does not implement interface `ImplicitAs(Int(N).(MulWith(U).Result))` [MissingImplInMemberAccessNote]
// CHECK:STDERR:   fn Op[self: Int(N)](other: Int(N)) -> Int(N) = "int.smul";
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: {{.*}}/prelude/types/int.carbon:7:1: in import [InImport]
// CHECK:STDERR: {{.*}}/prelude/operators/arithmetic.carbon:62:3: note: while building thunk to match the signature of this function [ThunkSignature]
// CHECK:STDERR:   fn Op[self: Self](other: Other) -> Result;
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: {{.*}}/prelude/types/int.carbon:153:3: error: cannot implicitly convert expression of type `Int(N)` to `Int(N).(SubWith(U).Result)` [ConversionFailure]
// CHECK:STDERR:   fn Op[self: Int(N)](other: Int(N)) -> Int(N) = "int.ssub";
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: {{.*}}/prelude/types/int.carbon:153:3: note: type `Int(N)` does not implement interface `ImplicitAs(Int(N).(SubWith(U).Result))` [MissingImplInMemberAccessNote]
// CHECK:STDERR:   fn Op[self: Int(N)](other: Int(N)) -> Int(N) = "int.ssub";
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: {{.*}}/prelude/types/int.carbon:7:1: in import [InImport]
// CHECK:STDERR: {{.*}}/prelude/operators/arithmetic.carbon:46:3: note: while building thunk to match the signature of this function [ThunkSignature]
// CHECK:STDERR:   fn Op[self: Self](other: Other) -> Result;
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: {{.*}}/prelude/types/int.carbon:160:3: error: cannot implicitly convert expression of type `Int(N)` to `T.(AddWith(Int(N)).Result)` [ConversionFailure]
// CHECK:STDERR:   fn Op[self: Int(N)](other: Int(N)) -> Int(N) = "int.sadd";
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: {{.*}}/prelude/types/int.carbon:160:3: note: type `Int(N)` does not implement interface `ImplicitAs(T.(AddWith(Int(N)).Result))` [MissingImplInMemberAccessNote]
// CHECK:STDERR:   fn Op[self: Int(N)](other: Int(N)) -> Int(N) = "int.sadd";
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: {{.*}}/prelude/types/int.carbon:7:1: in import [InImport]
// CHECK:STDERR: {{.*}}/prelude/operators/arithmetic.carbon:24:3: note: while building thunk to match the signature of this function [ThunkSignature]
// CHECK:STDERR:   fn Op[self: Self](other: Other) -> Result;
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: {{.*}}/prelude/types/int.carbon:165:3: error: cannot implicitly convert expression of type `Int(N)` to `T.(DivWith(Int(N)).Result)` [ConversionFailure]
// CHECK:STDERR:   fn Op[self: Int(N)](other: Int(N)) -> Int(N) = "int.sdiv";
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: {{.*}}/prelude/types/int.carbon:165:3: note: type `Int(N)` does not implement interface `ImplicitAs(T.(DivWith(Int(N)).Result))` [MissingImplInMemberAccessNote]
// CHECK:STDERR:   fn Op[self: Int(N)](other: Int(N)) -> Int(N) = "int.sdiv";
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: {{.*}}/prelude/types/int.carbon:7:1: in import [InImport]
// CHECK:STDERR: {{.*}}/prelude/operators/arithmetic.carbon:73:3: note: while building thunk to match the signature of this function [ThunkSignature]
// CHECK:STDERR:   fn Op[self: Self](other: Other) -> Result;
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: {{.*}}/prelude/types/int.carbon:170:3: error: cannot implicitly convert expression of type `Int(N)` to `T.(ModWith(Int(N)).Result)` [ConversionFailure]
// CHECK:STDERR:   fn Op[self: Int(N)](other: Int(N)) -> Int(N) = "int.smod";
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: {{.*}}/prelude/types/int.carbon:170:3: note: type `Int(N)` does not implement interface `ImplicitAs(T.(ModWith(Int(N)).Result))` [MissingImplInMemberAccessNote]
// CHECK:STDERR:   fn Op[self: Int(N)](other: Int(N)) -> Int(N) = "int.smod";
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: {{.*}}/prelude/types/int.carbon:7:1: in import [InImport]
// CHECK:STDERR: {{.*}}/prelude/operators/arithmetic.carbon:84:3: note: while building thunk to match the signature of this function [ThunkSignature]
// CHECK:STDERR:   fn Op[self: Self](other: Other) -> Result;
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: {{.*}}/prelude/types/int.carbon:175:3: error: cannot implicitly convert expression of type `Int(N)` to `T.(MulWith(Int(N)).Result)` [ConversionFailure]
// CHECK:STDERR:   fn Op[self: Int(N)](other: Int(N)) -> Int(N) = "int.smul";
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: {{.*}}/prelude/types/int.carbon:175:3: note: type `Int(N)` does not implement interface `ImplicitAs(T.(MulWith(Int(N)).Result))` [MissingImplInMemberAccessNote]
// CHECK:STDERR:   fn Op[self: Int(N)](other: Int(N)) -> Int(N) = "int.smul";
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: {{.*}}/prelude/types/int.carbon:7:1: in import [InImport]
// CHECK:STDERR: {{.*}}/prelude/operators/arithmetic.carbon:62:3: note: while building thunk to match the signature of this function [ThunkSignature]
// CHECK:STDERR:   fn Op[self: Self](other: Other) -> Result;
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: {{.*}}/prelude/types/int.carbon:180:3: error: cannot implicitly convert expression of type `Int(N)` to `T.(SubWith(Int(N)).Result)` [ConversionFailure]
// CHECK:STDERR:   fn Op[self: Int(N)](other: Int(N)) -> Int(N) = "int.ssub";
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: {{.*}}/prelude/types/int.carbon:180:3: note: type `Int(N)` does not implement interface `ImplicitAs(T.(SubWith(Int(N)).Result))` [MissingImplInMemberAccessNote]
// CHECK:STDERR:   fn Op[self: Int(N)](other: Int(N)) -> Int(N) = "int.ssub";
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: {{.*}}/prelude/types/int.carbon:7:1: in import [InImport]
// CHECK:STDERR: {{.*}}/prelude/operators/arithmetic.carbon:46:3: note: while building thunk to match the signature of this function [ThunkSignature]
// CHECK:STDERR:   fn Op[self: Self](other: Other) -> Result;
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: {{.*}}/prelude/types/int.carbon:221:3: error: cannot implicitly convert expression of type `Int(N)` to `Int(N).(BitAndWith(U).Result)` [ConversionFailure]
// CHECK:STDERR:   fn Op[self: Int(N)](other: Int(N)) -> Int(N) = "int.and";
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: {{.*}}/prelude/types/int.carbon:221:3: note: type `Int(N)` does not implement interface `ImplicitAs(Int(N).(BitAndWith(U).Result))` [MissingImplInMemberAccessNote]
// CHECK:STDERR:   fn Op[self: Int(N)](other: Int(N)) -> Int(N) = "int.and";
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: {{.*}}/prelude/types/int.carbon:7:1: in import [InImport]
// CHECK:STDERR: {{.*}}/prelude/operators/bitwise.carbon:30:3: note: while building thunk to match the signature of this function [ThunkSignature]
// CHECK:STDERR:   fn Op[self: Self](other: Other) -> Result;
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: {{.*}}/prelude/types/int.carbon:226:3: error: cannot implicitly convert expression of type `Int(N)` to `Int(N).(BitOrWith(U).Result)` [ConversionFailure]
// CHECK:STDERR:   fn Op[self: Int(N)](other: Int(N)) -> Int(N) = "int.or";
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: {{.*}}/prelude/types/int.carbon:226:3: note: type `Int(N)` does not implement interface `ImplicitAs(Int(N).(BitOrWith(U).Result))` [MissingImplInMemberAccessNote]
// CHECK:STDERR:   fn Op[self: Int(N)](other: Int(N)) -> Int(N) = "int.or";
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: {{.*}}/prelude/types/int.carbon:7:1: in import [InImport]
// CHECK:STDERR: {{.*}}/prelude/operators/bitwise.carbon:41:3: note: while building thunk to match the signature of this function [ThunkSignature]
// CHECK:STDERR:   fn Op[self: Self](other: Other) -> Result;
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: {{.*}}/prelude/types/int.carbon:231:3: error: cannot implicitly convert expression of type `Int(N)` to `Int(N).(BitXorWith(U).Result)` [ConversionFailure]
// CHECK:STDERR:   fn Op[self: Int(N)](other: Int(N)) -> Int(N) = "int.xor";
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: {{.*}}/prelude/types/int.carbon:231:3: note: type `Int(N)` does not implement interface `ImplicitAs(Int(N).(BitXorWith(U).Result))` [MissingImplInMemberAccessNote]
// CHECK:STDERR:   fn Op[self: Int(N)](other: Int(N)) -> Int(N) = "int.xor";
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: {{.*}}/prelude/types/int.carbon:7:1: in import [InImport]
// CHECK:STDERR: {{.*}}/prelude/operators/bitwise.carbon:52:3: note: while building thunk to match the signature of this function [ThunkSignature]
// CHECK:STDERR:   fn Op[self: Self](other: Other) -> Result;
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: {{.*}}/prelude/types/int.carbon:236:3: error: cannot implicitly convert expression of type `Int(N)` to `Int(N).(LeftShiftWith(U).Result)` [ConversionFailure]
// CHECK:STDERR:   fn Op[self: Int(N)](other: Int(N)) -> Int(N) = "int.left_shift";
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: {{.*}}/prelude/types/int.carbon:236:3: note: type `Int(N)` does not implement interface `ImplicitAs(Int(N).(LeftShiftWith(U).Result))` [MissingImplInMemberAccessNote]
// CHECK:STDERR:   fn Op[self: Int(N)](other: Int(N)) -> Int(N) = "int.left_shift";
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: {{.*}}/prelude/types/int.carbon:7:1: in import [InImport]
// CHECK:STDERR: {{.*}}/prelude/operators/bitwise.carbon:63:3: note: while building thunk to match the signature of this function [ThunkSignature]
// CHECK:STDERR:   fn Op[self: Self](other: Other) -> Result;
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: {{.*}}/prelude/types/int.carbon:241:3: error: cannot implicitly convert expression of type `Int(N)` to `Int(N).(RightShiftWith(U).Result)` [ConversionFailure]
// CHECK:STDERR:   fn Op[self: Int(N)](other: Int(N)) -> Int(N) = "int.right_shift";
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: {{.*}}/prelude/types/int.carbon:241:3: note: type `Int(N)` does not implement interface `ImplicitAs(Int(N).(RightShiftWith(U).Result))` [MissingImplInMemberAccessNote]
// CHECK:STDERR:   fn Op[self: Int(N)](other: Int(N)) -> Int(N) = "int.right_shift";
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: {{.*}}/prelude/types/int.carbon:7:1: in import [InImport]
// CHECK:STDERR: {{.*}}/prelude/operators/bitwise.carbon:74:3: note: while building thunk to match the signature of this function [ThunkSignature]
// CHECK:STDERR:   fn Op[self: Self](other: Other) -> Result;
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: {{.*}}/prelude/types/int.carbon:248:3: error: cannot implicitly convert expression of type `Int(N)` to `T.(BitAndWith(Int(N)).Result)` [ConversionFailure]
// CHECK:STDERR:   fn Op[self: Int(N)](other: Int(N)) -> Int(N) = "int.and";
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: {{.*}}/prelude/types/int.carbon:248:3: note: type `Int(N)` does not implement interface `ImplicitAs(T.(BitAndWith(Int(N)).Result))` [MissingImplInMemberAccessNote]
// CHECK:STDERR:   fn Op[self: Int(N)](other: Int(N)) -> Int(N) = "int.and";
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: {{.*}}/prelude/types/int.carbon:7:1: in import [InImport]
// CHECK:STDERR: {{.*}}/prelude/operators/bitwise.carbon:30:3: note: while building thunk to match the signature of this function [ThunkSignature]
// CHECK:STDERR:   fn Op[self: Self](other: Other) -> Result;
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: {{.*}}/prelude/types/int.carbon:253:3: error: cannot implicitly convert expression of type `Int(N)` to `T.(BitOrWith(Int(N)).Result)` [ConversionFailure]
// CHECK:STDERR:   fn Op[self: Int(N)](other: Int(N)) -> Int(N) = "int.or";
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: {{.*}}/prelude/types/int.carbon:253:3: note: type `Int(N)` does not implement interface `ImplicitAs(T.(BitOrWith(Int(N)).Result))` [MissingImplInMemberAccessNote]
// CHECK:STDERR:   fn Op[self: Int(N)](other: Int(N)) -> Int(N) = "int.or";
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: {{.*}}/prelude/types/int.carbon:7:1: in import [InImport]
// CHECK:STDERR: {{.*}}/prelude/operators/bitwise.carbon:41:3: note: while building thunk to match the signature of this function [ThunkSignature]
// CHECK:STDERR:   fn Op[self: Self](other: Other) -> Result;
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: {{.*}}/prelude/types/int.carbon:258:3: error: cannot implicitly convert expression of type `Int(N)` to `T.(BitXorWith(Int(N)).Result)` [ConversionFailure]
// CHECK:STDERR:   fn Op[self: Int(N)](other: Int(N)) -> Int(N) = "int.xor";
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: {{.*}}/prelude/types/int.carbon:258:3: note: type `Int(N)` does not implement interface `ImplicitAs(T.(BitXorWith(Int(N)).Result))` [MissingImplInMemberAccessNote]
// CHECK:STDERR:   fn Op[self: Int(N)](other: Int(N)) -> Int(N) = "int.xor";
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: {{.*}}/prelude/types/int.carbon:7:1: in import [InImport]
// CHECK:STDERR: {{.*}}/prelude/operators/bitwise.carbon:52:3: note: while building thunk to match the signature of this function [ThunkSignature]
// CHECK:STDERR:   fn Op[self: Self](other: Other) -> Result;
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: {{.*}}/prelude/types/int.carbon:263:3: error: cannot implicitly convert expression of type `Int(N)` to `T.(LeftShiftWith(Int(N)).Result)` [ConversionFailure]
// CHECK:STDERR:   fn Op[self: Int(N)](other: Int(N)) -> Int(N) = "int.left_shift";
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: {{.*}}/prelude/types/int.carbon:263:3: note: type `Int(N)` does not implement interface `ImplicitAs(T.(LeftShiftWith(Int(N)).Result))` [MissingImplInMemberAccessNote]
// CHECK:STDERR:   fn Op[self: Int(N)](other: Int(N)) -> Int(N) = "int.left_shift";
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: {{.*}}/prelude/types/int.carbon:7:1: in import [InImport]
// CHECK:STDERR: {{.*}}/prelude/operators/bitwise.carbon:63:3: note: while building thunk to match the signature of this function [ThunkSignature]
// CHECK:STDERR:   fn Op[self: Self](other: Other) -> Result;
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: {{.*}}/prelude/types/int.carbon:268:3: error: cannot implicitly convert expression of type `Int(N)` to `T.(RightShiftWith(Int(N)).Result)` [ConversionFailure]
// CHECK:STDERR:   fn Op[self: Int(N)](other: Int(N)) -> Int(N) = "int.right_shift";
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: {{.*}}/prelude/types/int.carbon:268:3: note: type `Int(N)` does not implement interface `ImplicitAs(T.(RightShiftWith(Int(N)).Result))` [MissingImplInMemberAccessNote]
// CHECK:STDERR:   fn Op[self: Int(N)](other: Int(N)) -> Int(N) = "int.right_shift";
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: {{.*}}/prelude/types/int.carbon:7:1: in import [InImport]
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
// CHECK:STDERR: {{.*}}/prelude/types/uint.carbon:174:3: error: cannot implicitly convert expression of type `UInt(N)` to `UInt(N).(AddWith(U).Result)` [ConversionFailure]
// CHECK:STDERR:   fn Op[self: UInt(N)](other: UInt(N)) -> UInt(N) = "int.uadd";
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: {{.*}}/prelude/types/uint.carbon:174:3: note: type `UInt(N)` does not implement interface `ImplicitAs(UInt(N).(AddWith(U).Result))` [MissingImplInMemberAccessNote]
// CHECK:STDERR:   fn Op[self: UInt(N)](other: UInt(N)) -> UInt(N) = "int.uadd";
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: {{.*}}/prelude/types/uint.carbon:7:1: in import [InImport]
// CHECK:STDERR: {{.*}}/prelude/operators/arithmetic.carbon:24:3: note: while building thunk to match the signature of this function [ThunkSignature]
// CHECK:STDERR:   fn Op[self: Self](other: Other) -> Result;
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: {{.*}}/prelude/types/uint.carbon:179:3: error: cannot implicitly convert expression of type `UInt(N)` to `UInt(N).(DivWith(U).Result)` [ConversionFailure]
// CHECK:STDERR:   fn Op[self: UInt(N)](other: UInt(N)) -> UInt(N) = "int.udiv";
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: {{.*}}/prelude/types/uint.carbon:179:3: note: type `UInt(N)` does not implement interface `ImplicitAs(UInt(N).(DivWith(U).Result))` [MissingImplInMemberAccessNote]
// CHECK:STDERR:   fn Op[self: UInt(N)](other: UInt(N)) -> UInt(N) = "int.udiv";
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: {{.*}}/prelude/types/uint.carbon:7:1: in import [InImport]
// CHECK:STDERR: {{.*}}/prelude/operators/arithmetic.carbon:73:3: note: while building thunk to match the signature of this function [ThunkSignature]
// CHECK:STDERR:   fn Op[self: Self](other: Other) -> Result;
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: {{.*}}/prelude/types/uint.carbon:184:3: error: cannot implicitly convert expression of type `UInt(N)` to `UInt(N).(ModWith(U).Result)` [ConversionFailure]
// CHECK:STDERR:   fn Op[self: UInt(N)](other: UInt(N)) -> UInt(N) = "int.umod";
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: {{.*}}/prelude/types/uint.carbon:184:3: note: type `UInt(N)` does not implement interface `ImplicitAs(UInt(N).(ModWith(U).Result))` [MissingImplInMemberAccessNote]
// CHECK:STDERR:   fn Op[self: UInt(N)](other: UInt(N)) -> UInt(N) = "int.umod";
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: {{.*}}/prelude/types/uint.carbon:7:1: in import [InImport]
// CHECK:STDERR: {{.*}}/prelude/operators/arithmetic.carbon:84:3: note: while building thunk to match the signature of this function [ThunkSignature]
// CHECK:STDERR:   fn Op[self: Self](other: Other) -> Result;
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: {{.*}}/prelude/types/uint.carbon:189:3: error: cannot implicitly convert expression of type `UInt(N)` to `UInt(N).(MulWith(U).Result)` [ConversionFailure]
// CHECK:STDERR:   fn Op[self: UInt(N)](other: UInt(N)) -> UInt(N) = "int.umul";
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: {{.*}}/prelude/types/uint.carbon:189:3: note: type `UInt(N)` does not implement interface `ImplicitAs(UInt(N).(MulWith(U).Result))` [MissingImplInMemberAccessNote]
// CHECK:STDERR:   fn Op[self: UInt(N)](other: UInt(N)) -> UInt(N) = "int.umul";
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: {{.*}}/prelude/types/uint.carbon:7:1: in import [InImport]
// CHECK:STDERR: {{.*}}/prelude/operators/arithmetic.carbon:62:3: note: while building thunk to match the signature of this function [ThunkSignature]
// CHECK:STDERR:   fn Op[self: Self](other: Other) -> Result;
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: {{.*}}/prelude/types/uint.carbon:194:3: error: cannot implicitly convert expression of type `UInt(N)` to `UInt(N).(SubWith(U).Result)` [ConversionFailure]
// CHECK:STDERR:   fn Op[self: UInt(N)](other: UInt(N)) -> UInt(N) = "int.usub";
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: {{.*}}/prelude/types/uint.carbon:194:3: note: type `UInt(N)` does not implement interface `ImplicitAs(UInt(N).(SubWith(U).Result))` [MissingImplInMemberAccessNote]
// CHECK:STDERR:   fn Op[self: UInt(N)](other: UInt(N)) -> UInt(N) = "int.usub";
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: {{.*}}/prelude/types/uint.carbon:7:1: in import [InImport]
// CHECK:STDERR: {{.*}}/prelude/operators/arithmetic.carbon:46:3: note: while building thunk to match the signature of this function [ThunkSignature]
// CHECK:STDERR:   fn Op[self: Self](other: Other) -> Result;
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: {{.*}}/prelude/types/uint.carbon:201:3: error: cannot implicitly convert expression of type `UInt(N)` to `T.(AddWith(UInt(N)).Result)` [ConversionFailure]
// CHECK:STDERR:   fn Op[self: UInt(N)](other: UInt(N)) -> UInt(N) = "int.uadd";
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: {{.*}}/prelude/types/uint.carbon:201:3: note: type `UInt(N)` does not implement interface `ImplicitAs(T.(AddWith(UInt(N)).Result))` [MissingImplInMemberAccessNote]
// CHECK:STDERR:   fn Op[self: UInt(N)](other: UInt(N)) -> UInt(N) = "int.uadd";
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: {{.*}}/prelude/types/uint.carbon:7:1: in import [InImport]
// CHECK:STDERR: {{.*}}/prelude/operators/arithmetic.carbon:24:3: note: while building thunk to match the signature of this function [ThunkSignature]
// CHECK:STDERR:   fn Op[self: Self](other: Other) -> Result;
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: {{.*}}/prelude/types/uint.carbon:206:3: error: cannot implicitly convert expression of type `UInt(N)` to `T.(DivWith(UInt(N)).Result)` [ConversionFailure]
// CHECK:STDERR:   fn Op[self: UInt(N)](other: UInt(N)) -> UInt(N) = "int.udiv";
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: {{.*}}/prelude/types/uint.carbon:206:3: note: type `UInt(N)` does not implement interface `ImplicitAs(T.(DivWith(UInt(N)).Result))` [MissingImplInMemberAccessNote]
// CHECK:STDERR:   fn Op[self: UInt(N)](other: UInt(N)) -> UInt(N) = "int.udiv";
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: {{.*}}/prelude/types/uint.carbon:7:1: in import [InImport]
// CHECK:STDERR: {{.*}}/prelude/operators/arithmetic.carbon:73:3: note: while building thunk to match the signature of this function [ThunkSignature]
// CHECK:STDERR:   fn Op[self: Self](other: Other) -> Result;
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: {{.*}}/prelude/types/uint.carbon:211:3: error: cannot implicitly convert expression of type `UInt(N)` to `T.(ModWith(UInt(N)).Result)` [ConversionFailure]
// CHECK:STDERR:   fn Op[self: UInt(N)](other: UInt(N)) -> UInt(N) = "int.umod";
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: {{.*}}/prelude/types/uint.carbon:211:3: note: type `UInt(N)` does not implement interface `ImplicitAs(T.(ModWith(UInt(N)).Result))` [MissingImplInMemberAccessNote]
// CHECK:STDERR:   fn Op[self: UInt(N)](other: UInt(N)) -> UInt(N) = "int.umod";
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: {{.*}}/prelude/types/uint.carbon:7:1: in import [InImport]
// CHECK:STDERR: {{.*}}/prelude/operators/arithmetic.carbon:84:3: note: while building thunk to match the signature of this function [ThunkSignature]
// CHECK:STDERR:   fn Op[self: Self](other: Other) -> Result;
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: {{.*}}/prelude/types/uint.carbon:216:3: error: cannot implicitly convert expression of type `UInt(N)` to `T.(MulWith(UInt(N)).Result)` [ConversionFailure]
// CHECK:STDERR:   fn Op[self: UInt(N)](other: UInt(N)) -> UInt(N) = "int.umul";
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: {{.*}}/prelude/types/uint.carbon:216:3: note: type `UInt(N)` does not implement interface `ImplicitAs(T.(MulWith(UInt(N)).Result))` [MissingImplInMemberAccessNote]
// CHECK:STDERR:   fn Op[self: UInt(N)](other: UInt(N)) -> UInt(N) = "int.umul";
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: {{.*}}/prelude/types/uint.carbon:7:1: in import [InImport]
// CHECK:STDERR: {{.*}}/prelude/operators/arithmetic.carbon:62:3: note: while building thunk to match the signature of this function [ThunkSignature]
// CHECK:STDERR:   fn Op[self: Self](other: Other) -> Result;
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: {{.*}}/prelude/types/uint.carbon:221:3: error: cannot implicitly convert expression of type `UInt(N)` to `T.(SubWith(UInt(N)).Result)` [ConversionFailure]
// CHECK:STDERR:   fn Op[self: UInt(N)](other: UInt(N)) -> UInt(N) = "int.usub";
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: {{.*}}/prelude/types/uint.carbon:221:3: note: type `UInt(N)` does not implement interface `ImplicitAs(T.(SubWith(UInt(N)).Result))` [MissingImplInMemberAccessNote]
// CHECK:STDERR:   fn Op[self: UInt(N)](other: UInt(N)) -> UInt(N) = "int.usub";
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: {{.*}}/prelude/types/uint.carbon:7:1: in import [InImport]
// CHECK:STDERR: {{.*}}/prelude/operators/arithmetic.carbon:46:3: note: while building thunk to match the signature of this function [ThunkSignature]
// CHECK:STDERR:   fn Op[self: Self](other: Other) -> Result;
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: {{.*}}/prelude/types/uint.carbon:262:3: error: cannot implicitly convert expression of type `UInt(N)` to `UInt(N).(BitAndWith(U).Result)` [ConversionFailure]
// CHECK:STDERR:   fn Op[self: UInt(N)](other: UInt(N)) -> UInt(N) = "int.and";
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: {{.*}}/prelude/types/uint.carbon:262:3: note: type `UInt(N)` does not implement interface `ImplicitAs(UInt(N).(BitAndWith(U).Result))` [MissingImplInMemberAccessNote]
// CHECK:STDERR:   fn Op[self: UInt(N)](other: UInt(N)) -> UInt(N) = "int.and";
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: {{.*}}/prelude/types/uint.carbon:7:1: in import [InImport]
// CHECK:STDERR: {{.*}}/prelude/operators/bitwise.carbon:30:3: note: while building thunk to match the signature of this function [ThunkSignature]
// CHECK:STDERR:   fn Op[self: Self](other: Other) -> Result;
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: {{.*}}/prelude/types/uint.carbon:267:3: error: cannot implicitly convert expression of type `UInt(N)` to `UInt(N).(BitOrWith(U).Result)` [ConversionFailure]
// CHECK:STDERR:   fn Op[self: UInt(N)](other: UInt(N)) -> UInt(N) = "int.or";
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: {{.*}}/prelude/types/uint.carbon:267:3: note: type `UInt(N)` does not implement interface `ImplicitAs(UInt(N).(BitOrWith(U).Result))` [MissingImplInMemberAccessNote]
// CHECK:STDERR:   fn Op[self: UInt(N)](other: UInt(N)) -> UInt(N) = "int.or";
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: {{.*}}/prelude/types/uint.carbon:7:1: in import [InImport]
// CHECK:STDERR: {{.*}}/prelude/operators/bitwise.carbon:41:3: note: while building thunk to match the signature of this function [ThunkSignature]
// CHECK:STDERR:   fn Op[self: Self](other: Other) -> Result;
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: {{.*}}/prelude/types/uint.carbon:272:3: error: cannot implicitly convert expression of type `UInt(N)` to `UInt(N).(BitXorWith(U).Result)` [ConversionFailure]
// CHECK:STDERR:   fn Op[self: UInt(N)](other: UInt(N)) -> UInt(N) = "int.xor";
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: {{.*}}/prelude/types/uint.carbon:272:3: note: type `UInt(N)` does not implement interface `ImplicitAs(UInt(N).(BitXorWith(U).Result))` [MissingImplInMemberAccessNote]
// CHECK:STDERR:   fn Op[self: UInt(N)](other: UInt(N)) -> UInt(N) = "int.xor";
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: {{.*}}/prelude/types/uint.carbon:7:1: in import [InImport]
// CHECK:STDERR: {{.*}}/prelude/operators/bitwise.carbon:52:3: note: while building thunk to match the signature of this function [ThunkSignature]
// CHECK:STDERR:   fn Op[self: Self](other: Other) -> Result;
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: {{.*}}/prelude/types/uint.carbon:277:3: error: cannot implicitly convert expression of type `UInt(N)` to `UInt(N).(LeftShiftWith(U).Result)` [ConversionFailure]
// CHECK:STDERR:   fn Op[self: UInt(N)](other: UInt(N)) -> UInt(N) = "int.left_shift";
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: {{.*}}/prelude/types/uint.carbon:277:3: note: type `UInt(N)` does not implement interface `ImplicitAs(UInt(N).(LeftShiftWith(U).Result))` [MissingImplInMemberAccessNote]
// CHECK:STDERR:   fn Op[self: UInt(N)](other: UInt(N)) -> UInt(N) = "int.left_shift";
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: {{.*}}/prelude/types/uint.carbon:7:1: in import [InImport]
// CHECK:STDERR: {{.*}}/prelude/operators/bitwise.carbon:63:3: note: while building thunk to match the signature of this function [ThunkSignature]
// CHECK:STDERR:   fn Op[self: Self](other: Other) -> Result;
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: {{.*}}/prelude/types/uint.carbon:282:3: error: cannot implicitly convert expression of type `UInt(N)` to `UInt(N).(RightShiftWith(U).Result)` [ConversionFailure]
// CHECK:STDERR:   fn Op[self: UInt(N)](other: UInt(N)) -> UInt(N) = "int.right_shift";
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: {{.*}}/prelude/types/uint.carbon:282:3: note: type `UInt(N)` does not implement interface `ImplicitAs(UInt(N).(RightShiftWith(U).Result))` [MissingImplInMemberAccessNote]
// CHECK:STDERR:   fn Op[self: UInt(N)](other: UInt(N)) -> UInt(N) = "int.right_shift";
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: {{.*}}/prelude/types/uint.carbon:7:1: in import [InImport]
// CHECK:STDERR: {{.*}}/prelude/operators/bitwise.carbon:74:3: note: while building thunk to match the signature of this function [ThunkSignature]
// CHECK:STDERR:   fn Op[self: Self](other: Other) -> Result;
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: {{.*}}/prelude/types/uint.carbon:289:3: error: cannot implicitly convert expression of type `UInt(N)` to `T.(BitAndWith(UInt(N)).Result)` [ConversionFailure]
// CHECK:STDERR:   fn Op[self: UInt(N)](other: UInt(N)) -> UInt(N) = "int.and";
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: {{.*}}/prelude/types/uint.carbon:289:3: note: type `UInt(N)` does not implement interface `ImplicitAs(T.(BitAndWith(UInt(N)).Result))` [MissingImplInMemberAccessNote]
// CHECK:STDERR:   fn Op[self: UInt(N)](other: UInt(N)) -> UInt(N) = "int.and";
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: {{.*}}/prelude/types/uint.carbon:7:1: in import [InImport]
// CHECK:STDERR: {{.*}}/prelude/operators/bitwise.carbon:30:3: note: while building thunk to match the signature of this function [ThunkSignature]
// CHECK:STDERR:   fn Op[self: Self](other: Other) -> Result;
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: {{.*}}/prelude/types/uint.carbon:294:3: error: cannot implicitly convert expression of type `UInt(N)` to `T.(BitOrWith(UInt(N)).Result)` [ConversionFailure]
// CHECK:STDERR:   fn Op[self: UInt(N)](other: UInt(N)) -> UInt(N) = "int.or";
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: {{.*}}/prelude/types/uint.carbon:294:3: note: type `UInt(N)` does not implement interface `ImplicitAs(T.(BitOrWith(UInt(N)).Result))` [MissingImplInMemberAccessNote]
// CHECK:STDERR:   fn Op[self: UInt(N)](other: UInt(N)) -> UInt(N) = "int.or";
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: {{.*}}/prelude/types/uint.carbon:7:1: in import [InImport]
// CHECK:STDERR: {{.*}}/prelude/operators/bitwise.carbon:41:3: note: while building thunk to match the signature of this function [ThunkSignature]
// CHECK:STDERR:   fn Op[self: Self](other: Other) -> Result;
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: {{.*}}/prelude/types/uint.carbon:299:3: error: cannot implicitly convert expression of type `UInt(N)` to `T.(BitXorWith(UInt(N)).Result)` [ConversionFailure]
// CHECK:STDERR:   fn Op[self: UInt(N)](other: UInt(N)) -> UInt(N) = "int.xor";
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: {{.*}}/prelude/types/uint.carbon:299:3: note: type `UInt(N)` does not implement interface `ImplicitAs(T.(BitXorWith(UInt(N)).Result))` [MissingImplInMemberAccessNote]
// CHECK:STDERR:   fn Op[self: UInt(N)](other: UInt(N)) -> UInt(N) = "int.xor";
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: {{.*}}/prelude/types/uint.carbon:7:1: in import [InImport]
// CHECK:STDERR: {{.*}}/prelude/operators/bitwise.carbon:52:3: note: while building thunk to match the signature of this function [ThunkSignature]
// CHECK:STDERR:   fn Op[self: Self](other: Other) -> Result;
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: {{.*}}/prelude/types/uint.carbon:304:3: error: cannot implicitly convert expression of type `UInt(N)` to `T.(LeftShiftWith(UInt(N)).Result)` [ConversionFailure]
// CHECK:STDERR:   fn Op[self: UInt(N)](other: UInt(N)) -> UInt(N) = "int.left_shift";
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: {{.*}}/prelude/types/uint.carbon:304:3: note: type `UInt(N)` does not implement interface `ImplicitAs(T.(LeftShiftWith(UInt(N)).Result))` [MissingImplInMemberAccessNote]
// CHECK:STDERR:   fn Op[self: UInt(N)](other: UInt(N)) -> UInt(N) = "int.left_shift";
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: {{.*}}/prelude/types/uint.carbon:7:1: in import [InImport]
// CHECK:STDERR: {{.*}}/prelude/operators/bitwise.carbon:63:3: note: while building thunk to match the signature of this function [ThunkSignature]
// CHECK:STDERR:   fn Op[self: Self](other: Other) -> Result;
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: {{.*}}/prelude/types/uint.carbon:309:3: error: cannot implicitly convert expression of type `UInt(N)` to `T.(RightShiftWith(UInt(N)).Result)` [ConversionFailure]
// CHECK:STDERR:   fn Op[self: UInt(N)](other: UInt(N)) -> UInt(N) = "int.right_shift";
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: {{.*}}/prelude/types/uint.carbon:309:3: note: type `UInt(N)` does not implement interface `ImplicitAs(T.(RightShiftWith(UInt(N)).Result))` [MissingImplInMemberAccessNote]
// CHECK:STDERR:   fn Op[self: UInt(N)](other: UInt(N)) -> UInt(N) = "int.right_shift";
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: {{.*}}/prelude/types/uint.carbon:7:1: in import [InImport]
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

// --- addition.carbon

library "[[@TEST_NAME]]";

var a:
//@dump-sem-ir-begin
    array(i32, 1 + 2)
//@dump-sem-ir-end
    = (1, 2, 3);
let b: array(i32, 3)* = &a;

// --- unsigned.carbon

library "[[@TEST_NAME]]";

var a:
//@dump-sem-ir-begin
    array(i32, 3 as u32)
//@dump-sem-ir-end
    = (1, 2, 3);
let b: array(i32, 3)* = &a;

// --- fail_negative.carbon

library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_negative.carbon:[[@LINE+4]]:19: error: array bound of -1 is negative [ArrayBoundNegative]
// CHECK:STDERR: var a: array(i32, -1);
// CHECK:STDERR:                   ^~
// CHECK:STDERR:
var a: array(i32, -1);

// --- fail_overflow.carbon

library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_overflow.carbon:[[@LINE+4]]:19: error: array bound of 39999999999999999993 is too large [ArrayBoundTooLarge]
// CHECK:STDERR: var a: array(i32, 39999999999999999993);
// CHECK:STDERR:                   ^~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
var a: array(i32, 39999999999999999993);

// --- fail_invalid_type_with_overflow.carbon

library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_invalid_type_with_overflow.carbon:[[@LINE+7]]:14: error: cannot implicitly convert non-type value of type `Core.IntLiteral` to `type` [ConversionFailureNonTypeToFacet]
// CHECK:STDERR: var b: array(1, 39999999999999999993);
// CHECK:STDERR:              ^
// CHECK:STDERR: fail_invalid_type_with_overflow.carbon:[[@LINE+4]]:14: note: type `Core.IntLiteral` does not implement interface `Core.ImplicitAs(type)` [MissingImplInMemberAccessNote]
// CHECK:STDERR: var b: array(1, 39999999999999999993);
// CHECK:STDERR:              ^
// CHECK:STDERR:
var b: array(1, 39999999999999999993);

// CHECK:STDOUT: --- addition.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %int_2.ecc: Core.IntLiteral = int_value 2 [concrete]
// CHECK:STDOUT:   %AddWith.type.a45: type = facet_type <@AddWith, @AddWith(Core.IntLiteral)> [concrete]
// CHECK:STDOUT:   %AddWith.Op.type.0ee: type = fn_type @AddWith.Op, @AddWith(Core.IntLiteral) [concrete]
// CHECK:STDOUT:   %AddWith.impl_witness: <witness> = impl_witness imports.%AddWith.impl_witness_table [concrete]
// CHECK:STDOUT:   %AddWith.facet: %AddWith.type.a45 = facet_value Core.IntLiteral, (%AddWith.impl_witness) [concrete]
// CHECK:STDOUT:   %.31c: type = fn_type_with_self_type %AddWith.Op.type.0ee, %AddWith.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.AddWith.impl.Op.type: type = fn_type @Core.IntLiteral.as.AddWith.impl.Op [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.AddWith.impl.Op: %Core.IntLiteral.as.AddWith.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.AddWith.impl.Op.bound: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.AddWith.impl.Op [concrete]
// CHECK:STDOUT:   %int_3.1ba: Core.IntLiteral = int_value 3 [concrete]
// CHECK:STDOUT:   %array_type: type = array_type %int_3.1ba, %i32 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.abd = import_ref Core//prelude/operators/arithmetic, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.895: %Core.IntLiteral.as.AddWith.impl.Op.type = import_ref Core//prelude/operators/arithmetic, loc{{\d+_\d+}}, loaded [concrete = constants.%Core.IntLiteral.as.AddWith.impl.Op]
// CHECK:STDOUT:   %AddWith.impl_witness_table = impl_witness_table (%Core.import_ref.abd, %Core.import_ref.895), @Core.IntLiteral.as.AddWith.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %.loc6_21: type = splice_block %array_type.loc6 [concrete = constants.%array_type] {
// CHECK:STDOUT:     %int_32.loc6: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc6: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:     %int_2: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:     %impl.elem1: %.31c = impl_witness_access constants.%AddWith.impl_witness, element1 [concrete = constants.%Core.IntLiteral.as.AddWith.impl.Op]
// CHECK:STDOUT:     %bound_method: <bound method> = bound_method %int_1, %impl.elem1 [concrete = constants.%Core.IntLiteral.as.AddWith.impl.Op.bound]
// CHECK:STDOUT:     %Core.IntLiteral.as.AddWith.impl.Op.call: init Core.IntLiteral = call %bound_method(%int_1, %int_2) [concrete = constants.%int_3.1ba]
// CHECK:STDOUT:     %.loc6_18.1: Core.IntLiteral = value_of_initializer %Core.IntLiteral.as.AddWith.impl.Op.call [concrete = constants.%int_3.1ba]
// CHECK:STDOUT:     %.loc6_18.2: Core.IntLiteral = converted %Core.IntLiteral.as.AddWith.impl.Op.call, %.loc6_18.1 [concrete = constants.%int_3.1ba]
// CHECK:STDOUT:     %array_type.loc6: type = array_type %.loc6_18.2, %i32.loc6 [concrete = constants.%array_type]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- unsigned.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %int_3.1ba: Core.IntLiteral = int_value 3 [concrete]
// CHECK:STDOUT:   %u32: type = class_type @UInt, @UInt(%int_32) [concrete]
// CHECK:STDOUT:   %As.type.45b: type = facet_type <@As, @As(%u32)> [concrete]
// CHECK:STDOUT:   %As.Convert.type.b94: type = fn_type @As.Convert, @As(%u32) [concrete]
// CHECK:STDOUT:   %To.586: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.type.56b: type = fn_type @Core.IntLiteral.as.As.impl.Convert, @Core.IntLiteral.as.As.impl(%To.586) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.02d: %Core.IntLiteral.as.As.impl.Convert.type.56b = struct_value () [symbolic]
// CHECK:STDOUT:   %From: Core.IntLiteral = symbolic_binding From, 0 [symbolic]
// CHECK:STDOUT:   %As.impl_witness.9c6: <witness> = impl_witness imports.%As.impl_witness_table.a7d, @Core.IntLiteral.as.As.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.type.81c: type = fn_type @Core.IntLiteral.as.As.impl.Convert, @Core.IntLiteral.as.As.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.3f9: %Core.IntLiteral.as.As.impl.Convert.type.81c = struct_value () [concrete]
// CHECK:STDOUT:   %As.facet: %As.type.45b = facet_value Core.IntLiteral, (%As.impl_witness.9c6) [concrete]
// CHECK:STDOUT:   %.053: type = fn_type_with_self_type %As.Convert.type.b94, %As.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.bound: <bound method> = bound_method %int_3.1ba, %Core.IntLiteral.as.As.impl.Convert.3f9 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.As.impl.Convert.3f9, @Core.IntLiteral.as.As.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.124: <bound method> = bound_method %int_3.1ba, %Core.IntLiteral.as.As.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_3.d14: %u32 = int_value 3 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.7a9: type = facet_type <@ImplicitAs, @ImplicitAs(Core.IntLiteral)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.Convert.type.71e: type = fn_type @ImplicitAs.Convert, @ImplicitAs(Core.IntLiteral) [concrete]
// CHECK:STDOUT:   %UInt.as.ImplicitAs.impl.Convert.type.df7: type = fn_type @UInt.as.ImplicitAs.impl.Convert, @UInt.as.ImplicitAs.impl(%From) [symbolic]
// CHECK:STDOUT:   %UInt.as.ImplicitAs.impl.Convert.b0f: %UInt.as.ImplicitAs.impl.Convert.type.df7 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.5ff: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.1a6, @UInt.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %UInt.as.ImplicitAs.impl.Convert.type.23f: type = fn_type @UInt.as.ImplicitAs.impl.Convert, @UInt.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %UInt.as.ImplicitAs.impl.Convert.a92: %UInt.as.ImplicitAs.impl.Convert.type.23f = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.c06: %ImplicitAs.type.7a9 = facet_value %u32, (%ImplicitAs.impl_witness.5ff) [concrete]
// CHECK:STDOUT:   %.d81: type = fn_type_with_self_type %ImplicitAs.Convert.type.71e, %ImplicitAs.facet.c06 [concrete]
// CHECK:STDOUT:   %UInt.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_3.d14, %UInt.as.ImplicitAs.impl.Convert.a92 [concrete]
// CHECK:STDOUT:   %UInt.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %UInt.as.ImplicitAs.impl.Convert.a92, @UInt.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.c1d: <bound method> = bound_method %int_3.d14, %UInt.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %array_type: type = array_type %int_3.1ba, %i32 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.7bb: @Core.IntLiteral.as.As.impl.%Core.IntLiteral.as.As.impl.Convert.type (%Core.IntLiteral.as.As.impl.Convert.type.56b) = import_ref Core//prelude/types/uint, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.As.impl.%Core.IntLiteral.as.As.impl.Convert (constants.%Core.IntLiteral.as.As.impl.Convert.02d)]
// CHECK:STDOUT:   %As.impl_witness_table.a7d = impl_witness_table (%Core.import_ref.7bb), @Core.IntLiteral.as.As.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.7e2: @UInt.as.ImplicitAs.impl.%UInt.as.ImplicitAs.impl.Convert.type (%UInt.as.ImplicitAs.impl.Convert.type.df7) = import_ref Core//prelude/types/uint, loc{{\d+_\d+}}, loaded [symbolic = @UInt.as.ImplicitAs.impl.%UInt.as.ImplicitAs.impl.Convert (constants.%UInt.as.ImplicitAs.impl.Convert.b0f)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.1a6 = impl_witness_table (%Core.import_ref.7e2), @UInt.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %.loc6_24: type = splice_block %array_type.loc6 [concrete = constants.%array_type] {
// CHECK:STDOUT:     %int_32.loc6_11: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc6: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %int_3.loc6: Core.IntLiteral = int_value 3 [concrete = constants.%int_3.1ba]
// CHECK:STDOUT:     %int_32.loc6_21: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %u32: type = class_type @UInt, @UInt(constants.%int_32) [concrete = constants.%u32]
// CHECK:STDOUT:     %impl.elem0.loc6_18.1: %.053 = impl_witness_access constants.%As.impl_witness.9c6, element0 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.3f9]
// CHECK:STDOUT:     %bound_method.loc6_18.1: <bound method> = bound_method %int_3.loc6, %impl.elem0.loc6_18.1 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.bound]
// CHECK:STDOUT:     %specific_fn.loc6_18.1: <specific function> = specific_function %impl.elem0.loc6_18.1, @Core.IntLiteral.as.As.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.specific_fn]
// CHECK:STDOUT:     %bound_method.loc6_18.2: <bound method> = bound_method %int_3.loc6, %specific_fn.loc6_18.1 [concrete = constants.%bound_method.124]
// CHECK:STDOUT:     %Core.IntLiteral.as.As.impl.Convert.call: init %u32 = call %bound_method.loc6_18.2(%int_3.loc6) [concrete = constants.%int_3.d14]
// CHECK:STDOUT:     %.loc6_18.1: %u32 = value_of_initializer %Core.IntLiteral.as.As.impl.Convert.call [concrete = constants.%int_3.d14]
// CHECK:STDOUT:     %.loc6_18.2: %u32 = converted %int_3.loc6, %.loc6_18.1 [concrete = constants.%int_3.d14]
// CHECK:STDOUT:     %impl.elem0.loc6_18.2: %.d81 = impl_witness_access constants.%ImplicitAs.impl_witness.5ff, element0 [concrete = constants.%UInt.as.ImplicitAs.impl.Convert.a92]
// CHECK:STDOUT:     %bound_method.loc6_18.3: <bound method> = bound_method %.loc6_18.2, %impl.elem0.loc6_18.2 [concrete = constants.%UInt.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:     %specific_fn.loc6_18.2: <specific function> = specific_function %impl.elem0.loc6_18.2, @UInt.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%UInt.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:     %bound_method.loc6_18.4: <bound method> = bound_method %.loc6_18.2, %specific_fn.loc6_18.2 [concrete = constants.%bound_method.c1d]
// CHECK:STDOUT:     %UInt.as.ImplicitAs.impl.Convert.call: init Core.IntLiteral = call %bound_method.loc6_18.4(%.loc6_18.2) [concrete = constants.%int_3.1ba]
// CHECK:STDOUT:     %.loc6_18.3: Core.IntLiteral = value_of_initializer %UInt.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_3.1ba]
// CHECK:STDOUT:     %.loc6_18.4: Core.IntLiteral = converted %.loc6_18.2, %.loc6_18.3 [concrete = constants.%int_3.1ba]
// CHECK:STDOUT:     %array_type.loc6: type = array_type %.loc6_18.4, %i32.loc6 [concrete = constants.%array_type]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
