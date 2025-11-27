// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/full.carbon
// EXTRA-ARGS: --dump-sem-ir-ranges=if-present
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/for/actual.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/for/actual.carbon
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

// --- lib.carbon

library "lib";

class IntRange(N:! Core.IntLiteral()) {
  fn Make(start: Core.Int(N), end: Core.Int(N)) -> Self {
    return {.start = start, .end = end};
  }

  impl as Core.Iterate where .CursorType = Core.Int(N) and .ElementType = Core.Int(N) {
    // CHECK:STDERR: lib.carbon:[[@LINE+10]]:5: error: cannot implicitly convert expression of type `Core.Int(N)` to `IntRange(N).(Core.Iterate.Core.CursorType)` [ConversionFailure]
    // CHECK:STDERR:     fn NewCursor[self: Self]() -> Core.Int(N) { return self.start; }
    // CHECK:STDERR:     ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // CHECK:STDERR: lib.carbon:[[@LINE+7]]:5: note: type `Core.Int(N)` does not implement interface `Core.ImplicitAs(IntRange(N).(Core.Iterate.Core.CursorType))` [MissingImplInMemberAccessNote]
    // CHECK:STDERR:     fn NewCursor[self: Self]() -> Core.Int(N) { return self.start; }
    // CHECK:STDERR:     ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // CHECK:STDERR: {{.*}}/prelude/iterate.carbon:16:3: note: while building thunk to match the signature of this function [ThunkSignature]
    // CHECK:STDERR:   fn NewCursor[self: Self]() -> CursorType;
    // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // CHECK:STDERR:
    fn NewCursor[self: Self]() -> Core.Int(N) { return self.start; }
    // CHECK:STDERR: {{.*}}/prelude/iterate.carbon:17:23: error: cannot implicitly convert expression of type `IntRange(N).(Core.Iterate.Core.CursorType)*` to `Core.Int(N)*` [ConversionFailure]
    // CHECK:STDERR:   fn Next[self: Self](cursor: CursorType*) -> Optional(ElementType);
    // CHECK:STDERR:                       ^~~~~~~~~~~~~~~~~~~
    // CHECK:STDERR: {{.*}}/prelude/iterate.carbon:17:23: note: type `IntRange(N).(Core.Iterate.Core.CursorType)*` does not implement interface `Core.ImplicitAs(Core.Int(N)*)` [MissingImplInMemberAccessNote]
    // CHECK:STDERR:   fn Next[self: Self](cursor: CursorType*) -> Optional(ElementType);
    // CHECK:STDERR:                       ^~~~~~~~~~~~~~~~~~~
    // CHECK:STDERR: lib.carbon:[[@LINE+17]]:25: note: initializing function parameter [InCallToFunctionParam]
    // CHECK:STDERR:     fn Next[self: Self](cursor: Core.Int(N)*) -> Core.Optional(Core.Int(N)) {
    // CHECK:STDERR:                         ^~~~~~~~~~~~~~~~~~~~
    // CHECK:STDERR: {{.*}}/prelude/iterate.carbon:17:3: note: while building thunk to match the signature of this function [ThunkSignature]
    // CHECK:STDERR:   fn Next[self: Self](cursor: CursorType*) -> Optional(ElementType);
    // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // CHECK:STDERR:
    // CHECK:STDERR: lib.carbon:[[@LINE+10]]:5: error: cannot implicitly convert expression of type `Core.Optional(Core.Int(N) as Core.OptionalStorage)` to `Core.Optional(IntRange(N).(Core.Iterate.Core.ElementType) as Core.OptionalStorage)` [ConversionFailure]
    // CHECK:STDERR:     fn Next[self: Self](cursor: Core.Int(N)*) -> Core.Optional(Core.Int(N)) {
    // CHECK:STDERR:     ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // CHECK:STDERR: lib.carbon:[[@LINE+7]]:5: note: type `Core.Optional(Core.Int(N) as Core.OptionalStorage)` does not implement interface `Core.ImplicitAs(Core.Optional(IntRange(N).(Core.Iterate.Core.ElementType) as Core.OptionalStorage))` [MissingImplInMemberAccessNote]
    // CHECK:STDERR:     fn Next[self: Self](cursor: Core.Int(N)*) -> Core.Optional(Core.Int(N)) {
    // CHECK:STDERR:     ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // CHECK:STDERR: {{.*}}/prelude/iterate.carbon:17:3: note: while building thunk to match the signature of this function [ThunkSignature]
    // CHECK:STDERR:   fn Next[self: Self](cursor: CursorType*) -> Optional(ElementType);
    // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // CHECK:STDERR:
    fn Next[self: Self](cursor: Core.Int(N)*) -> Core.Optional(Core.Int(N)) {
      var value: Core.Int(N) = *cursor;
      if (value < self.end) {
        ++*cursor;
        return Core.Optional(Core.Int(N)).Some(value);
      } else {
        return Core.Optional(Core.Int(N)).None();
      }
    }
  }

  private var start: Core.Int(N);
  private var end: Core.Int(N);
}

fn Range(end: i32) -> IntRange(32) {
  return IntRange(32).Make(0, end);
}

// --- trivial.carbon

import library "lib";

fn Read() {
  let y:! Core.IntLiteral() = 43;
  var x: IntRange(32) = Range(y);
}

// CHECK:STDOUT: --- lib.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self.3b7: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %IntLiteral.type: type = fn_type @IntLiteral [concrete]
// CHECK:STDOUT:   %IntLiteral: %IntLiteral.type = struct_value () [concrete]
// CHECK:STDOUT:   %N.586: Core.IntLiteral = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.dc0: type = pattern_type Core.IntLiteral [concrete]
// CHECK:STDOUT:   %IntRange.type: type = generic_class_type @IntRange [concrete]
// CHECK:STDOUT:   %IntRange.generic: %IntRange.type = struct_value () [concrete]
// CHECK:STDOUT:   %IntRange.48b: type = class_type @IntRange, @IntRange(%N.586) [symbolic]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %Int.b50e37.1: type = class_type @Int, @Int(%N.586) [symbolic]
// CHECK:STDOUT:   %pattern_type.60f8e7.1: type = pattern_type %Int.b50e37.1 [symbolic]
// CHECK:STDOUT:   %pattern_type.6de: type = pattern_type %IntRange.48b [symbolic]
// CHECK:STDOUT:   %IntRange.Make.type.121: type = fn_type @IntRange.Make, @IntRange(%N.586) [symbolic]
// CHECK:STDOUT:   %IntRange.Make.14e: %IntRange.Make.type.121 = struct_value () [symbolic]
// CHECK:STDOUT:   %Iterate.type: type = facet_type <@Iterate> [concrete]
// CHECK:STDOUT:   %.Self.b7a: %Iterate.type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %Iterate.assoc_type: type = assoc_entity_type @Iterate [concrete]
// CHECK:STDOUT:   %assoc1.02e: %Iterate.assoc_type = assoc_entity element1, imports.%Core.import_ref.9e6 [concrete]
// CHECK:STDOUT:   %.Self.binding.as_type.573: type = symbolic_binding_type .Self, %.Self.b7a [symbolic_self]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %i32.builtin: type = int_type signed, %int_32 [concrete]
// CHECK:STDOUT:   %complete_type.f8a: <witness> = complete_type_witness %i32.builtin [concrete]
// CHECK:STDOUT:   %Iterate.lookup_impl_witness.a57: <witness> = lookup_impl_witness %.Self.binding.as_type.573, @Iterate [symbolic_self]
// CHECK:STDOUT:   %impl.elem1.129: type = impl_witness_access %Iterate.lookup_impl_witness.a57, element1 [symbolic_self]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %impl.elem0.a97: %Copy.type = impl_witness_access %Iterate.lookup_impl_witness.a57, element0 [symbolic_self]
// CHECK:STDOUT:   %OptionalStorage.type: type = facet_type <@OptionalStorage> [concrete]
// CHECK:STDOUT:   %Optional.type: type = generic_class_type @Optional [concrete]
// CHECK:STDOUT:   %Optional.generic: %Optional.type = struct_value () [concrete]
// CHECK:STDOUT:   %T.3fe: %OptionalStorage.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %Optional.Some.type.32b: type = fn_type @Optional.Some, @Optional(%T.3fe) [symbolic]
// CHECK:STDOUT:   %Optional.Some.249: %Optional.Some.type.32b = struct_value () [symbolic]
// CHECK:STDOUT:   %Optional.None.type.193: type = fn_type @Optional.None, @Optional(%T.3fe) [symbolic]
// CHECK:STDOUT:   %Optional.None.dc7: %Optional.None.type.193 = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.Op.type: type = fn_type @Copy.Op [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.cc7: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.cc7 = struct_value () [concrete]
// CHECK:STDOUT:   %Dest: type = symbolic_binding Dest, 0 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.assoc_type.8b5194.1: type = assoc_entity_type @ImplicitAs, @ImplicitAs(%Dest) [symbolic]
// CHECK:STDOUT:   %ImplicitAs.Convert.type.4c8ddd.1: type = fn_type @ImplicitAs.Convert, @ImplicitAs(%Dest) [symbolic]
// CHECK:STDOUT:   %ImplicitAs.Convert.e9f4dc.1: %ImplicitAs.Convert.type.4c8ddd.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %pattern_type.7ce: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.f51: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.2a1: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.f51 = struct_value () [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.51e: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.e9b: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.51e = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.f89: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.059, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.d14: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.Convert.type.1b6: type = fn_type @ImplicitAs.Convert, @ImplicitAs(%i32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.e9b, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %assoc0.0f6: %Iterate.assoc_type = assoc_entity element0, imports.%Core.import_ref.61e [concrete]
// CHECK:STDOUT:   %require_complete.e66303.2: <witness> = require_complete_type %Int.b50e37.1 [symbolic]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.24b: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%N.586) [symbolic]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.c95: %Int.as.Copy.impl.Op.type.24b = struct_value () [symbolic]
// CHECK:STDOUT:   %.1a2: require_specific_def_type = require_specific_def @Int.as.Copy.impl(%N.586) [symbolic]
// CHECK:STDOUT:   %Copy.lookup_impl_witness.035: <witness> = lookup_impl_witness %Int.b50e37.1, @Copy [symbolic]
// CHECK:STDOUT:   %Copy.facet.d3d: %Copy.type = facet_value %Int.b50e37.1, (%Copy.lookup_impl_witness.035) [symbolic]
// CHECK:STDOUT:   %Iterate_where.type.a34: type = facet_type <@Iterate where %impl.elem1.129 = %Int.b50e37.1 and %impl.elem0.a97 = %Copy.facet.d3d> [symbolic]
// CHECK:STDOUT:   %require_complete.806: <witness> = require_complete_type %Iterate_where.type.a34 [symbolic]
// CHECK:STDOUT:   %Iterate.impl_witness.273: <witness> = impl_witness @IntRange.%Iterate.impl_witness_table, @IntRange.as.Iterate.impl(%N.586) [symbolic]
// CHECK:STDOUT:   %IntRange.as.Iterate.impl.NewCursor.type.c510a4.1: type = fn_type @IntRange.as.Iterate.impl.NewCursor.loc20_47.1, @IntRange.as.Iterate.impl(%N.586) [symbolic]
// CHECK:STDOUT:   %IntRange.as.Iterate.impl.NewCursor.c95819.1: %IntRange.as.Iterate.impl.NewCursor.type.c510a4.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %ptr.1cf: type = ptr_type %Int.b50e37.1 [symbolic]
// CHECK:STDOUT:   %pattern_type.054: type = pattern_type %ptr.1cf [symbolic]
// CHECK:STDOUT:   %type_where: type = facet_type <type where .Self impls <CanDestroy>> [concrete]
// CHECK:STDOUT:   %DestroyT: %type_where = symbolic_binding DestroyT, 0 [symbolic]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.type.15c: type = fn_type @DestroyT.binding.as_type.as.Destroy.impl.Op, @DestroyT.binding.as_type.as.Destroy.impl(%DestroyT) [symbolic]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.fd5: %DestroyT.binding.as_type.as.Destroy.impl.Op.type.15c = struct_value () [symbolic]
// CHECK:STDOUT:   %Destroy.Op.type: type = fn_type @Destroy.Op [concrete]
// CHECK:STDOUT:   %.350: require_specific_def_type = require_specific_def @T.binding.as_type.as.OptionalStorage.impl(%Copy.facet.d3d) [symbolic]
// CHECK:STDOUT:   %OptionalStorage.lookup_impl_witness.8c6: <witness> = lookup_impl_witness %Int.b50e37.1, @OptionalStorage [symbolic]
// CHECK:STDOUT:   %OptionalStorage.facet.ea4: %OptionalStorage.type = facet_value %Int.b50e37.1, (%OptionalStorage.lookup_impl_witness.8c6) [symbolic]
// CHECK:STDOUT:   %Optional.23f: type = class_type @Optional, @Optional(%OptionalStorage.facet.ea4) [symbolic]
// CHECK:STDOUT:   %pattern_type.9f4: type = pattern_type %Optional.23f [symbolic]
// CHECK:STDOUT:   %IntRange.as.Iterate.impl.Next.type.a05ce8.1: type = fn_type @IntRange.as.Iterate.impl.Next.loc44_77.1, @IntRange.as.Iterate.impl(%N.586) [symbolic]
// CHECK:STDOUT:   %IntRange.as.Iterate.impl.Next.d8655a.1: %IntRange.as.Iterate.impl.Next.type.a05ce8.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %.6a4: require_specific_def_type = require_specific_def @IntRange.as.Iterate.impl(%N.586) [symbolic]
// CHECK:STDOUT:   %Iterate.lookup_impl_witness.5aa: <witness> = lookup_impl_witness %IntRange.48b, @Iterate [symbolic]
// CHECK:STDOUT:   %impl.elem1.d3c: type = impl_witness_access %Iterate.lookup_impl_witness.5aa, element1 [symbolic]
// CHECK:STDOUT:   %pattern_type.605: type = pattern_type %impl.elem1.d3c [symbolic]
// CHECK:STDOUT:   %IntRange.as.Iterate.impl.NewCursor.type.c510a4.2: type = fn_type @IntRange.as.Iterate.impl.NewCursor.loc20_47.2, @IntRange.as.Iterate.impl(%N.586) [symbolic]
// CHECK:STDOUT:   %IntRange.as.Iterate.impl.NewCursor.c95819.2: %IntRange.as.Iterate.impl.NewCursor.type.c510a4.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %ptr.ccc: type = ptr_type %impl.elem1.d3c [symbolic]
// CHECK:STDOUT:   %pattern_type.852: type = pattern_type %ptr.ccc [symbolic]
// CHECK:STDOUT:   %impl.elem0.18c: %Copy.type = impl_witness_access %Iterate.lookup_impl_witness.5aa, element0 [symbolic]
// CHECK:STDOUT:   %as_type.349: type = facet_access_type %impl.elem0.18c [symbolic]
// CHECK:STDOUT:   %.535: require_specific_def_type = require_specific_def @T.binding.as_type.as.OptionalStorage.impl(%impl.elem0.18c) [symbolic]
// CHECK:STDOUT:   %OptionalStorage.lookup_impl_witness.bc5: <witness> = lookup_impl_witness %as_type.349, @OptionalStorage [symbolic]
// CHECK:STDOUT:   %OptionalStorage.facet.201: %OptionalStorage.type = facet_value %as_type.349, (%OptionalStorage.lookup_impl_witness.bc5) [symbolic]
// CHECK:STDOUT:   %Optional.f2d: type = class_type @Optional, @Optional(%OptionalStorage.facet.201) [symbolic]
// CHECK:STDOUT:   %pattern_type.f5b: type = pattern_type %Optional.f2d [symbolic]
// CHECK:STDOUT:   %IntRange.as.Iterate.impl.Next.type.a05ce8.2: type = fn_type @IntRange.as.Iterate.impl.Next.loc44_77.2, @IntRange.as.Iterate.impl(%N.586) [symbolic]
// CHECK:STDOUT:   %IntRange.as.Iterate.impl.Next.d8655a.2: %IntRange.as.Iterate.impl.Next.type.a05ce8.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %IntRange.elem.baf: type = unbound_element_type %IntRange.48b, %Int.b50e37.1 [symbolic]
// CHECK:STDOUT:   %struct_type.start.end.cb5: type = struct_type {.start: %Int.b50e37.1, .end: %Int.b50e37.1} [symbolic]
// CHECK:STDOUT:   %complete_type.4f0: <witness> = complete_type_witness %struct_type.start.end.cb5 [symbolic]
// CHECK:STDOUT:   %require_complete.b4c: <witness> = require_complete_type %IntRange.48b [symbolic]
// CHECK:STDOUT:   %.20c: type = fn_type_with_self_type %Copy.Op.type, %Copy.facet.d3d [symbolic]
// CHECK:STDOUT:   %impl.elem0.625: %.20c = impl_witness_access %Copy.lookup_impl_witness.035, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.c88: <specific function> = specific_impl_function %impl.elem0.625, @Copy.Op(%Copy.facet.d3d) [symbolic]
// CHECK:STDOUT:   %Optional.None.type.a39: type = fn_type @Optional.None, @Optional(%OptionalStorage.facet.ea4) [symbolic]
// CHECK:STDOUT:   %Optional.None.b0b: %Optional.None.type.a39 = struct_value () [symbolic]
// CHECK:STDOUT:   %Optional.Some.type.74c: type = fn_type @Optional.Some, @Optional(%OptionalStorage.facet.ea4) [symbolic]
// CHECK:STDOUT:   %Optional.Some.37c: %Optional.Some.type.74c = struct_value () [symbolic]
// CHECK:STDOUT:   %require_complete.b49: <witness> = require_complete_type %Optional.23f [symbolic]
// CHECK:STDOUT:   %require_complete.6ca2: <witness> = require_complete_type %ptr.1cf [symbolic]
// CHECK:STDOUT:   %OrderedWith.type.270: type = generic_interface_type @OrderedWith [concrete]
// CHECK:STDOUT:   %OrderedWith.generic: %OrderedWith.type.270 = struct_value () [concrete]
// CHECK:STDOUT:   %Other: type = symbolic_binding Other, 0 [symbolic]
// CHECK:STDOUT:   %OrderedWith.assoc_type.72b: type = assoc_entity_type @OrderedWith, @OrderedWith(%Other) [symbolic]
// CHECK:STDOUT:   %OrderedWith.Less.type.fc4: type = fn_type @OrderedWith.Less, @OrderedWith(%Other) [symbolic]
// CHECK:STDOUT:   %OrderedWith.Less.bb3: %OrderedWith.Less.type.fc4 = struct_value () [symbolic]
// CHECK:STDOUT:   %OrderedWith.type.393: type = facet_type <@OrderedWith, @OrderedWith(%Int.b50e37.1)> [symbolic]
// CHECK:STDOUT:   %OrderedWith.Less.type.c6f: type = fn_type @OrderedWith.Less, @OrderedWith(%Int.b50e37.1) [symbolic]
// CHECK:STDOUT:   %OrderedWith.assoc_type.72a: type = assoc_entity_type @OrderedWith, @OrderedWith(%Int.b50e37.1) [symbolic]
// CHECK:STDOUT:   %assoc0.3f6: %OrderedWith.assoc_type.72a = assoc_entity element0, imports.%Core.import_ref.02d [symbolic]
// CHECK:STDOUT:   %require_complete.4f7: <witness> = require_complete_type %OrderedWith.type.393 [symbolic]
// CHECK:STDOUT:   %assoc0.7c2: %OrderedWith.assoc_type.72b = assoc_entity element0, imports.%Core.import_ref.e123aa.1 [symbolic]
// CHECK:STDOUT:   %M: Core.IntLiteral = symbolic_binding M, 1 [symbolic]
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Less.type.e01: type = fn_type @Int.as.OrderedWith.impl.Less.1, @Int.as.OrderedWith.impl.849(%N.586, %M) [symbolic]
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Less.7c7: %Int.as.OrderedWith.impl.Less.type.e01 = struct_value () [symbolic]
// CHECK:STDOUT:   %OrderedWith.impl_witness.586: <witness> = impl_witness imports.%OrderedWith.impl_witness_table.401, @Int.as.OrderedWith.impl.849(%N.586, %N.586) [symbolic]
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Less.type.f52: type = fn_type @Int.as.OrderedWith.impl.Less.1, @Int.as.OrderedWith.impl.849(%N.586, %N.586) [symbolic]
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Less.af3: %Int.as.OrderedWith.impl.Less.type.f52 = struct_value () [symbolic]
// CHECK:STDOUT:   %.e09: require_specific_def_type = require_specific_def @Int.as.OrderedWith.impl.849(%N.586, %N.586) [symbolic]
// CHECK:STDOUT:   %OrderedWith.facet: %OrderedWith.type.393 = facet_value %Int.b50e37.1, (%OrderedWith.impl_witness.586) [symbolic]
// CHECK:STDOUT:   %.db5: type = fn_type_with_self_type %OrderedWith.Less.type.c6f, %OrderedWith.facet [symbolic]
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Less.specific_fn.0ff: <specific function> = specific_function %Int.as.OrderedWith.impl.Less.af3, @Int.as.OrderedWith.impl.Less.1(%N.586, %N.586) [symbolic]
// CHECK:STDOUT:   %Inc.type: type = facet_type <@Inc> [concrete]
// CHECK:STDOUT:   %Inc.Op.type: type = fn_type @Inc.Op [concrete]
// CHECK:STDOUT:   %.4ff: require_specific_def_type = require_specific_def @Int.as.Inc.impl(%N.586) [symbolic]
// CHECK:STDOUT:   %Inc.lookup_impl_witness: <witness> = lookup_impl_witness %Int.b50e37.1, @Inc [symbolic]
// CHECK:STDOUT:   %Inc.facet: %Inc.type = facet_value %Int.b50e37.1, (%Inc.lookup_impl_witness) [symbolic]
// CHECK:STDOUT:   %.080: type = fn_type_with_self_type %Inc.Op.type, %Inc.facet [symbolic]
// CHECK:STDOUT:   %impl.elem0.923: %.080 = impl_witness_access %Inc.lookup_impl_witness, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.173: <specific function> = specific_impl_function %impl.elem0.923, @Inc.Op(%Inc.facet) [symbolic]
// CHECK:STDOUT:   %Optional.Some.specific_fn.90e: <specific function> = specific_function %Optional.Some.37c, @Optional.Some(%OptionalStorage.facet.ea4) [symbolic]
// CHECK:STDOUT:   %facet_value.4a9: %type_where = facet_value %Int.b50e37.1, () [symbolic]
// CHECK:STDOUT:   %Destroy.impl_witness.b47: <witness> = impl_witness imports.%Destroy.impl_witness_table, @DestroyT.binding.as_type.as.Destroy.impl(%facet_value.4a9) [symbolic]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.type.353: type = fn_type @DestroyT.binding.as_type.as.Destroy.impl.Op, @DestroyT.binding.as_type.as.Destroy.impl(%facet_value.4a9) [symbolic]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.3be: %DestroyT.binding.as_type.as.Destroy.impl.Op.type.353 = struct_value () [symbolic]
// CHECK:STDOUT:   %.4d6: require_specific_def_type = require_specific_def @DestroyT.binding.as_type.as.Destroy.impl(%facet_value.4a9) [symbolic]
// CHECK:STDOUT:   %Destroy.facet.11d: %Destroy.type = facet_value %Int.b50e37.1, (%Destroy.impl_witness.b47) [symbolic]
// CHECK:STDOUT:   %.a64: type = fn_type_with_self_type %Destroy.Op.type, %Destroy.facet.11d [symbolic]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.e87: <specific function> = specific_function %DestroyT.binding.as_type.as.Destroy.impl.Op.3be, @DestroyT.binding.as_type.as.Destroy.impl.Op(%facet_value.4a9) [symbolic]
// CHECK:STDOUT:   %Optional.None.specific_fn.94d: <specific function> = specific_function %Optional.None.b0b, @Optional.None(%OptionalStorage.facet.ea4) [symbolic]
// CHECK:STDOUT:   %require_complete.9f9: <witness> = require_complete_type %impl.elem1.d3c [symbolic]
// CHECK:STDOUT:   %IntRange.as.Iterate.impl.NewCursor.specific_fn: <specific function> = specific_function %IntRange.as.Iterate.impl.NewCursor.c95819.1, @IntRange.as.Iterate.impl.NewCursor.loc20_47.1(%N.586) [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.59a: type = facet_type <@ImplicitAs, @ImplicitAs(%impl.elem1.d3c)> [symbolic]
// CHECK:STDOUT:   %ImplicitAs.assoc_type.7a4: type = assoc_entity_type @ImplicitAs, @ImplicitAs(%impl.elem1.d3c) [symbolic]
// CHECK:STDOUT:   %assoc0.9df: %ImplicitAs.assoc_type.7a4 = assoc_entity element0, imports.%Core.import_ref.004393.1 [symbolic]
// CHECK:STDOUT:   %require_complete.61c: <witness> = require_complete_type %ImplicitAs.type.59a [symbolic]
// CHECK:STDOUT:   %assoc0.c36: %ImplicitAs.assoc_type.8b5194.1 = assoc_entity element0, imports.%Core.import_ref.48d [symbolic]
// CHECK:STDOUT:   %require_complete.3f6: <witness> = require_complete_type %Optional.f2d [symbolic]
// CHECK:STDOUT:   %IntRange.as.Iterate.impl.Next.specific_fn: <specific function> = specific_function %IntRange.as.Iterate.impl.Next.d8655a.1, @IntRange.as.Iterate.impl.Next.loc44_77.1(%N.586) [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.a76: type = facet_type <@ImplicitAs, @ImplicitAs(%ptr.1cf)> [symbolic]
// CHECK:STDOUT:   %ImplicitAs.assoc_type.2f2: type = assoc_entity_type @ImplicitAs, @ImplicitAs(%ptr.1cf) [symbolic]
// CHECK:STDOUT:   %assoc0.bf6: %ImplicitAs.assoc_type.2f2 = assoc_entity element0, imports.%Core.import_ref.004393.1 [symbolic]
// CHECK:STDOUT:   %require_complete.3e7: <witness> = require_complete_type %ImplicitAs.type.a76 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.903: type = facet_type <@ImplicitAs, @ImplicitAs(%Optional.f2d)> [symbolic]
// CHECK:STDOUT:   %ImplicitAs.assoc_type.561: type = assoc_entity_type @ImplicitAs, @ImplicitAs(%Optional.f2d) [symbolic]
// CHECK:STDOUT:   %assoc0.5f6: %ImplicitAs.assoc_type.561 = assoc_entity element0, imports.%Core.import_ref.004393.1 [symbolic]
// CHECK:STDOUT:   %require_complete.6cad: <witness> = require_complete_type %ImplicitAs.type.903 [symbolic]
// CHECK:STDOUT:   %IntRange.365: type = class_type @IntRange, @IntRange(%int_32) [concrete]
// CHECK:STDOUT:   %pattern_type.d16: type = pattern_type %IntRange.365 [concrete]
// CHECK:STDOUT:   %Range.type: type = fn_type @Range [concrete]
// CHECK:STDOUT:   %Range: %Range.type = struct_value () [concrete]
// CHECK:STDOUT:   %IntRange.Make.type.cef: type = fn_type @IntRange.Make, @IntRange(%int_32) [concrete]
// CHECK:STDOUT:   %IntRange.Make.0dc: %IntRange.Make.type.cef = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.impl_witness.fb7: <witness> = impl_witness imports.%Copy.impl_witness_table.b6a, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.469: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.dfd: %Int.as.Copy.impl.Op.type.469 = struct_value () [concrete]
// CHECK:STDOUT:   %.1e9: require_specific_def_type = require_specific_def @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %IntRange.elem.e33: type = unbound_element_type %IntRange.365, %i32 [concrete]
// CHECK:STDOUT:   %struct_type.start.end.d0a: type = struct_type {.start: %i32, .end: %i32} [concrete]
// CHECK:STDOUT:   %complete_type.c45: <witness> = complete_type_witness %struct_type.start.end.d0a [concrete]
// CHECK:STDOUT:   %int_0.5c6: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %IntRange.Make.specific_fn: <specific function> = specific_function %IntRange.Make.0dc, @IntRange.Make(%int_32) [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.e83: %ImplicitAs.type.d14 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.f89) [concrete]
// CHECK:STDOUT:   %.cbf: type = fn_type_with_self_type %ImplicitAs.Convert.type.1b6, %ImplicitAs.facet.e83 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.0b3: <bound method> = bound_method %int_0.5c6, %Core.IntLiteral.as.ImplicitAs.impl.Convert.e9b [concrete]
// CHECK:STDOUT:   %bound_method.9be: <bound method> = bound_method %int_0.5c6, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_0.6a9: %i32 = int_value 0 [concrete]
// CHECK:STDOUT:   %Copy.facet.c50: %Copy.type = facet_value %i32, (%Copy.impl_witness.fb7) [concrete]
// CHECK:STDOUT:   %.65f: type = fn_type_with_self_type %Copy.Op.type, %Copy.facet.c50 [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %Int.as.Copy.impl.Op.dfd, @Int.as.Copy.impl.Op(%int_32) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .IntLiteral = %Core.IntLiteral
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .Iterate = %Core.Iterate
// CHECK:STDOUT:     .Optional = %Core.Optional
// CHECK:STDOUT:     .Copy = %Core.Copy
// CHECK:STDOUT:     .OrderedWith = %Core.OrderedWith
// CHECK:STDOUT:     .Inc = %Core.Inc
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.IntLiteral: %IntLiteral.type = import_ref Core//prelude/types/int_literal, IntLiteral, loaded [concrete = constants.%IntLiteral]
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/types/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT:   %Core.Iterate: type = import_ref Core//prelude/iterate, Iterate, loaded [concrete = constants.%Iterate.type]
// CHECK:STDOUT:   %Core.import_ref.119: %Iterate.assoc_type = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [concrete = constants.%assoc0.0f6]
// CHECK:STDOUT:   %Core.import_ref.ed6: %Iterate.assoc_type = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [concrete = constants.%assoc1.02e]
// CHECK:STDOUT:   %Core.import_ref.9e6: type = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [concrete = %CursorType]
// CHECK:STDOUT:   %Core.import_ref.f2f: @Optional.%Optional.None.type (%Optional.None.type.193) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @Optional.%Optional.None (constants.%Optional.None.dc7)]
// CHECK:STDOUT:   %Core.import_ref.961: @Optional.%Optional.Some.type (%Optional.Some.type.32b) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @Optional.%Optional.Some (constants.%Optional.Some.249)]
// CHECK:STDOUT:   %Core.import_ref.12f: @ImplicitAs.%ImplicitAs.assoc_type (%ImplicitAs.assoc_type.8b5194.1) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @ImplicitAs.%assoc0 (constants.%assoc0.c36)]
// CHECK:STDOUT:   %Core.import_ref.004393.1: @ImplicitAs.%ImplicitAs.Convert.type (%ImplicitAs.Convert.type.4c8ddd.1) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @ImplicitAs.%ImplicitAs.Convert (constants.%ImplicitAs.Convert.e9f4dc.1)]
// CHECK:STDOUT:   %Core.import_ref.633: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.f51) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.2a1)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.059 = impl_witness_table (%Core.import_ref.633), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %CursorType: type = assoc_const_decl @CursorType [concrete] {}
// CHECK:STDOUT:   %Core.import_ref.61e: %Copy.type = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [concrete = %ElementType]
// CHECK:STDOUT:   %ElementType: %Copy.type = assoc_const_decl @ElementType [concrete] {}
// CHECK:STDOUT:   %Core.import_ref.d12: @Int.as.Copy.impl.%Int.as.Copy.impl.Op.type (%Int.as.Copy.impl.Op.type.24b) = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.Copy.impl.%Int.as.Copy.impl.Op (constants.%Int.as.Copy.impl.Op.c95)]
// CHECK:STDOUT:   %Copy.impl_witness_table.b6a = impl_witness_table (%Core.import_ref.d12), @Int.as.Copy.impl [concrete]
// CHECK:STDOUT:   %Core.Optional: %Optional.type = import_ref Core//prelude/types/optional, Optional, loaded [concrete = constants.%Optional.generic]
// CHECK:STDOUT:   %Core.import_ref.416: @DestroyT.binding.as_type.as.Destroy.impl.%DestroyT.binding.as_type.as.Destroy.impl.Op.type (%DestroyT.binding.as_type.as.Destroy.impl.Op.type.15c) = import_ref Core//prelude/types/optional, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @DestroyT.binding.as_type.as.Destroy.impl.%DestroyT.binding.as_type.as.Destroy.impl.Op (constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.fd5)]
// CHECK:STDOUT:   %Destroy.impl_witness_table = impl_witness_table (%Core.import_ref.416), @DestroyT.binding.as_type.as.Destroy.impl [concrete]
// CHECK:STDOUT:   %Core.Copy: type = import_ref Core//prelude/copy, Copy, loaded [concrete = constants.%Copy.type]
// CHECK:STDOUT:   %Core.OrderedWith: %OrderedWith.type.270 = import_ref Core//prelude/operators/comparison, OrderedWith, loaded [concrete = constants.%OrderedWith.generic]
// CHECK:STDOUT:   %Core.import_ref.128: @OrderedWith.%OrderedWith.assoc_type (%OrderedWith.assoc_type.72b) = import_ref Core//prelude/operators/comparison, loc{{\d+_\d+}}, loaded [symbolic = @OrderedWith.%assoc0 (constants.%assoc0.7c2)]
// CHECK:STDOUT:   %Core.import_ref.02d: @OrderedWith.%OrderedWith.Less.type (%OrderedWith.Less.type.fc4) = import_ref Core//prelude/operators/comparison, loc{{\d+_\d+}}, loaded [symbolic = @OrderedWith.%OrderedWith.Less (constants.%OrderedWith.Less.bb3)]
// CHECK:STDOUT:   %Core.import_ref.e123aa.1 = import_ref Core//prelude/operators/comparison, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.b35: @Int.as.OrderedWith.impl.849.%Int.as.OrderedWith.impl.Less.type (%Int.as.OrderedWith.impl.Less.type.e01) = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.OrderedWith.impl.849.%Int.as.OrderedWith.impl.Less (constants.%Int.as.OrderedWith.impl.Less.7c7)]
// CHECK:STDOUT:   %Core.import_ref.2a1 = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.3bc = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.8fd = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %OrderedWith.impl_witness_table.401 = impl_witness_table (%Core.import_ref.b35, %Core.import_ref.2a1, %Core.import_ref.3bc, %Core.import_ref.8fd), @Int.as.OrderedWith.impl.849 [concrete]
// CHECK:STDOUT:   %Core.Inc: type = import_ref Core//prelude/operators/arithmetic, Inc, loaded [concrete = constants.%Inc.type]
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.cc7 = import_ref Core//prelude/operators/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:   %Core.import_ref.48d = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .IntRange = %IntRange.decl
// CHECK:STDOUT:     .Range = %Range.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %IntRange.decl: %IntRange.type = class_decl @IntRange [concrete = constants.%IntRange.generic] {
// CHECK:STDOUT:     %N.patt: %pattern_type.dc0 = symbolic_binding_pattern N, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc4_36.1: type = splice_block %.loc4_36.3 [concrete = Core.IntLiteral] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.3b7]
// CHECK:STDOUT:       %Core.ref.loc4: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:       %IntLiteral.ref: %IntLiteral.type = name_ref IntLiteral, imports.%Core.IntLiteral [concrete = constants.%IntLiteral]
// CHECK:STDOUT:       %IntLiteral.call: init type = call %IntLiteral.ref() [concrete = Core.IntLiteral]
// CHECK:STDOUT:       %.loc4_36.2: type = value_of_initializer %IntLiteral.call [concrete = Core.IntLiteral]
// CHECK:STDOUT:       %.loc4_36.3: type = converted %IntLiteral.call, %.loc4_36.2 [concrete = Core.IntLiteral]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %N.loc4_16.2: Core.IntLiteral = symbolic_binding N, 0 [symbolic = %N.loc4_16.1 (constants.%N.586)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Range.decl: %Range.type = fn_decl @Range [concrete = constants.%Range] {
// CHECK:STDOUT:     %end.patt: %pattern_type.7ce = value_binding_pattern end [concrete]
// CHECK:STDOUT:     %end.param_patt: %pattern_type.7ce = value_param_pattern %end.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.d16 = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.d16 = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %IntRange.ref.loc59: %IntRange.type = name_ref IntRange, file.%IntRange.decl [concrete = constants.%IntRange.generic]
// CHECK:STDOUT:     %int_32.loc59_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %IntRange.loc59: type = class_type @IntRange, @IntRange(constants.%int_32) [concrete = constants.%IntRange.365]
// CHECK:STDOUT:     %end.param: %i32 = value_param call_param0
// CHECK:STDOUT:     %.loc59_15: type = splice_block %i32 [concrete = constants.%i32] {
// CHECK:STDOUT:       %int_32.loc59_15: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:       %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %end: %i32 = value_binding end, %end.param
// CHECK:STDOUT:     %return.param: ref %IntRange.365 = out_param call_param1
// CHECK:STDOUT:     %return: ref %IntRange.365 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @IntRange.as.Iterate.impl(@IntRange.%N.loc4_16.2: Core.IntLiteral) {
// CHECK:STDOUT:   %N: Core.IntLiteral = symbolic_binding N, 0 [symbolic = %N (constants.%N.586)]
// CHECK:STDOUT:   %IntRange: type = class_type @IntRange, @IntRange(%N) [symbolic = %IntRange (constants.%IntRange.48b)]
// CHECK:STDOUT:   %Int.loc9_54.1: type = class_type @Int, @Int(%N) [symbolic = %Int.loc9_54.1 (constants.%Int.b50e37.1)]
// CHECK:STDOUT:   %.loc9_85.1: require_specific_def_type = require_specific_def @Int.as.Copy.impl(%N) [symbolic = %.loc9_85.1 (constants.%.1a2)]
// CHECK:STDOUT:   %Copy.lookup_impl_witness: <witness> = lookup_impl_witness %Int.loc9_54.1, @Copy [symbolic = %Copy.lookup_impl_witness (constants.%Copy.lookup_impl_witness.035)]
// CHECK:STDOUT:   %Copy.facet.loc9_85.1: %Copy.type = facet_value %Int.loc9_54.1, (%Copy.lookup_impl_witness) [symbolic = %Copy.facet.loc9_85.1 (constants.%Copy.facet.d3d)]
// CHECK:STDOUT:   %Iterate_where.type: type = facet_type <@Iterate where constants.%impl.elem1.129 = %Int.loc9_54.1 and constants.%impl.elem0.a97 = %Copy.facet.loc9_85.1> [symbolic = %Iterate_where.type (constants.%Iterate_where.type.a34)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %Iterate_where.type [symbolic = %require_complete (constants.%require_complete.806)]
// CHECK:STDOUT:   %Iterate.impl_witness: <witness> = impl_witness @IntRange.%Iterate.impl_witness_table, @IntRange.as.Iterate.impl(%N) [symbolic = %Iterate.impl_witness (constants.%Iterate.impl_witness.273)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %IntRange.as.Iterate.impl.NewCursor.type.loc20_47.1: type = fn_type @IntRange.as.Iterate.impl.NewCursor.loc20_47.1, @IntRange.as.Iterate.impl(%N) [symbolic = %IntRange.as.Iterate.impl.NewCursor.type.loc20_47.1 (constants.%IntRange.as.Iterate.impl.NewCursor.type.c510a4.1)]
// CHECK:STDOUT:   %IntRange.as.Iterate.impl.NewCursor.loc20_47.1: @IntRange.as.Iterate.impl.%IntRange.as.Iterate.impl.NewCursor.type.loc20_47.1 (%IntRange.as.Iterate.impl.NewCursor.type.c510a4.1) = struct_value () [symbolic = %IntRange.as.Iterate.impl.NewCursor.loc20_47.1 (constants.%IntRange.as.Iterate.impl.NewCursor.c95819.1)]
// CHECK:STDOUT:   %.loc44: require_specific_def_type = require_specific_def @T.binding.as_type.as.OptionalStorage.impl(%Copy.facet.loc9_85.1) [symbolic = %.loc44 (constants.%.350)]
// CHECK:STDOUT:   %IntRange.as.Iterate.impl.Next.type.loc44_77.1: type = fn_type @IntRange.as.Iterate.impl.Next.loc44_77.1, @IntRange.as.Iterate.impl(%N) [symbolic = %IntRange.as.Iterate.impl.Next.type.loc44_77.1 (constants.%IntRange.as.Iterate.impl.Next.type.a05ce8.1)]
// CHECK:STDOUT:   %IntRange.as.Iterate.impl.Next.loc44_77.1: @IntRange.as.Iterate.impl.%IntRange.as.Iterate.impl.Next.type.loc44_77.1 (%IntRange.as.Iterate.impl.Next.type.a05ce8.1) = struct_value () [symbolic = %IntRange.as.Iterate.impl.Next.loc44_77.1 (constants.%IntRange.as.Iterate.impl.Next.d8655a.1)]
// CHECK:STDOUT:   %.1: require_specific_def_type = require_specific_def @IntRange.as.Iterate.impl(%N) [symbolic = %.1 (constants.%.6a4)]
// CHECK:STDOUT:   %IntRange.as.Iterate.impl.NewCursor.type.loc20_47.2: type = fn_type @IntRange.as.Iterate.impl.NewCursor.loc20_47.2, @IntRange.as.Iterate.impl(%N) [symbolic = %IntRange.as.Iterate.impl.NewCursor.type.loc20_47.2 (constants.%IntRange.as.Iterate.impl.NewCursor.type.c510a4.2)]
// CHECK:STDOUT:   %IntRange.as.Iterate.impl.NewCursor.loc20_47.2: @IntRange.as.Iterate.impl.%IntRange.as.Iterate.impl.NewCursor.type.loc20_47.2 (%IntRange.as.Iterate.impl.NewCursor.type.c510a4.2) = struct_value () [symbolic = %IntRange.as.Iterate.impl.NewCursor.loc20_47.2 (constants.%IntRange.as.Iterate.impl.NewCursor.c95819.2)]
// CHECK:STDOUT:   %Iterate.lookup_impl_witness: <witness> = lookup_impl_witness %IntRange, @Iterate [symbolic = %Iterate.lookup_impl_witness (constants.%Iterate.lookup_impl_witness.5aa)]
// CHECK:STDOUT:   %impl.elem0.1: %Copy.type = impl_witness_access %Iterate.lookup_impl_witness, element0 [symbolic = %impl.elem0.1 (constants.%impl.elem0.18c)]
// CHECK:STDOUT:   %.2: require_specific_def_type = require_specific_def @T.binding.as_type.as.OptionalStorage.impl(%impl.elem0.1) [symbolic = %.2 (constants.%.535)]
// CHECK:STDOUT:   %IntRange.as.Iterate.impl.Next.type.loc44_77.2: type = fn_type @IntRange.as.Iterate.impl.Next.loc44_77.2, @IntRange.as.Iterate.impl(%N) [symbolic = %IntRange.as.Iterate.impl.Next.type.loc44_77.2 (constants.%IntRange.as.Iterate.impl.Next.type.a05ce8.2)]
// CHECK:STDOUT:   %IntRange.as.Iterate.impl.Next.loc44_77.2: @IntRange.as.Iterate.impl.%IntRange.as.Iterate.impl.Next.type.loc44_77.2 (%IntRange.as.Iterate.impl.Next.type.a05ce8.2) = struct_value () [symbolic = %IntRange.as.Iterate.impl.Next.loc44_77.2 (constants.%IntRange.as.Iterate.impl.Next.d8655a.2)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %Self.ref as %.loc9_24 {
// CHECK:STDOUT:     %IntRange.as.Iterate.impl.NewCursor.decl.loc20_47.1: @IntRange.as.Iterate.impl.%IntRange.as.Iterate.impl.NewCursor.type.loc20_47.1 (%IntRange.as.Iterate.impl.NewCursor.type.c510a4.1) = fn_decl @IntRange.as.Iterate.impl.NewCursor.loc20_47.1 [symbolic = @IntRange.as.Iterate.impl.%IntRange.as.Iterate.impl.NewCursor.loc20_47.1 (constants.%IntRange.as.Iterate.impl.NewCursor.c95819.1)] {
// CHECK:STDOUT:       %self.patt: @IntRange.as.Iterate.impl.NewCursor.loc20_47.1.%pattern_type.loc20_18 (%pattern_type.6de) = value_binding_pattern self [concrete]
// CHECK:STDOUT:       %self.param_patt: @IntRange.as.Iterate.impl.NewCursor.loc20_47.1.%pattern_type.loc20_18 (%pattern_type.6de) = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:       %return.patt: @IntRange.as.Iterate.impl.NewCursor.loc20_47.1.%pattern_type.loc20_32 (%pattern_type.60f8e7.1) = return_slot_pattern [concrete]
// CHECK:STDOUT:       %return.param_patt: @IntRange.as.Iterate.impl.NewCursor.loc20_47.1.%pattern_type.loc20_32 (%pattern_type.60f8e7.1) = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:       %Int.ref: %Int.type = name_ref Int, imports.%Core.Int [concrete = constants.%Int.generic]
// CHECK:STDOUT:       %N.ref: Core.IntLiteral = name_ref N, @IntRange.%N.loc4_16.2 [symbolic = %N (constants.%N.586)]
// CHECK:STDOUT:       %Int.loc20_45.2: type = class_type @Int, @Int(constants.%N.586) [symbolic = %Int.loc20_45.1 (constants.%Int.b50e37.1)]
// CHECK:STDOUT:       %self.param: @IntRange.as.Iterate.impl.NewCursor.loc20_47.1.%IntRange (%IntRange.48b) = value_param call_param0
// CHECK:STDOUT:       %.loc20_24.1: type = splice_block %Self.ref [symbolic = %IntRange (constants.%IntRange.48b)] {
// CHECK:STDOUT:         %.loc20_24.2: type = specific_constant constants.%IntRange.48b, @IntRange(constants.%N.586) [symbolic = %IntRange (constants.%IntRange.48b)]
// CHECK:STDOUT:         %Self.ref: type = name_ref Self, %.loc20_24.2 [symbolic = %IntRange (constants.%IntRange.48b)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %self: @IntRange.as.Iterate.impl.NewCursor.loc20_47.1.%IntRange (%IntRange.48b) = value_binding self, %self.param
// CHECK:STDOUT:       %return.param: ref @IntRange.as.Iterate.impl.NewCursor.loc20_47.1.%Int.loc20_45.1 (%Int.b50e37.1) = out_param call_param1
// CHECK:STDOUT:       %return: ref @IntRange.as.Iterate.impl.NewCursor.loc20_47.1.%Int.loc20_45.1 (%Int.b50e37.1) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %IntRange.as.Iterate.impl.Next.decl.loc44_77.1: @IntRange.as.Iterate.impl.%IntRange.as.Iterate.impl.Next.type.loc44_77.1 (%IntRange.as.Iterate.impl.Next.type.a05ce8.1) = fn_decl @IntRange.as.Iterate.impl.Next.loc44_77.1 [symbolic = @IntRange.as.Iterate.impl.%IntRange.as.Iterate.impl.Next.loc44_77.1 (constants.%IntRange.as.Iterate.impl.Next.d8655a.1)] {
// CHECK:STDOUT:       %self.patt: @IntRange.as.Iterate.impl.Next.loc44_77.1.%pattern_type.loc44_13 (%pattern_type.6de) = value_binding_pattern self [concrete]
// CHECK:STDOUT:       %self.param_patt: @IntRange.as.Iterate.impl.Next.loc44_77.1.%pattern_type.loc44_13 (%pattern_type.6de) = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:       %cursor.patt: @IntRange.as.Iterate.impl.Next.loc44_77.1.%pattern_type.loc44_25 (%pattern_type.054) = value_binding_pattern cursor [concrete]
// CHECK:STDOUT:       %cursor.param_patt: @IntRange.as.Iterate.impl.Next.loc44_77.1.%pattern_type.loc44_25 (%pattern_type.054) = value_param_pattern %cursor.patt, call_param1 [concrete]
// CHECK:STDOUT:       %return.patt: @IntRange.as.Iterate.impl.Next.loc44_77.1.%pattern_type.loc44_47 (%pattern_type.9f4) = return_slot_pattern [concrete]
// CHECK:STDOUT:       %return.param_patt: @IntRange.as.Iterate.impl.Next.loc44_77.1.%pattern_type.loc44_47 (%pattern_type.9f4) = out_param_pattern %return.patt, call_param2 [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %Core.ref.loc44_50: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:       %Optional.ref.loc44: %Optional.type = name_ref Optional, imports.%Core.Optional [concrete = constants.%Optional.generic]
// CHECK:STDOUT:       %Core.ref.loc44_64: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:       %Int.ref.loc44_68: %Int.type = name_ref Int, imports.%Core.Int [concrete = constants.%Int.generic]
// CHECK:STDOUT:       %N.ref.loc44_73: Core.IntLiteral = name_ref N, @IntRange.%N.loc4_16.2 [symbolic = %N (constants.%N.586)]
// CHECK:STDOUT:       %Int.loc44_74: type = class_type @Int, @Int(constants.%N.586) [symbolic = %Int.loc44_43.1 (constants.%Int.b50e37.1)]
// CHECK:STDOUT:       %OptionalStorage.facet.loc44_75.2: %OptionalStorage.type = facet_value %Int.loc44_74, (constants.%OptionalStorage.lookup_impl_witness.8c6) [symbolic = %OptionalStorage.facet.loc44_75.1 (constants.%OptionalStorage.facet.ea4)]
// CHECK:STDOUT:       %.loc44_75.2: %OptionalStorage.type = converted %Int.loc44_74, %OptionalStorage.facet.loc44_75.2 [symbolic = %OptionalStorage.facet.loc44_75.1 (constants.%OptionalStorage.facet.ea4)]
// CHECK:STDOUT:       %Optional.loc44_75.2: type = class_type @Optional, @Optional(constants.%OptionalStorage.facet.ea4) [symbolic = %Optional.loc44_75.1 (constants.%Optional.23f)]
// CHECK:STDOUT:       %self.param: @IntRange.as.Iterate.impl.Next.loc44_77.1.%IntRange (%IntRange.48b) = value_param call_param0
// CHECK:STDOUT:       %.loc44_19.1: type = splice_block %Self.ref [symbolic = %IntRange (constants.%IntRange.48b)] {
// CHECK:STDOUT:         %.loc44_19.2: type = specific_constant constants.%IntRange.48b, @IntRange(constants.%N.586) [symbolic = %IntRange (constants.%IntRange.48b)]
// CHECK:STDOUT:         %Self.ref: type = name_ref Self, %.loc44_19.2 [symbolic = %IntRange (constants.%IntRange.48b)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %self: @IntRange.as.Iterate.impl.Next.loc44_77.1.%IntRange (%IntRange.48b) = value_binding self, %self.param
// CHECK:STDOUT:       %cursor.param: @IntRange.as.Iterate.impl.Next.loc44_77.1.%ptr.loc44_44.1 (%ptr.1cf) = value_param call_param1
// CHECK:STDOUT:       %.loc44_44: type = splice_block %ptr.loc44_44.2 [symbolic = %ptr.loc44_44.1 (constants.%ptr.1cf)] {
// CHECK:STDOUT:         %Core.ref.loc44_33: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:         %Int.ref.loc44_37: %Int.type = name_ref Int, imports.%Core.Int [concrete = constants.%Int.generic]
// CHECK:STDOUT:         %N.ref.loc44_42: Core.IntLiteral = name_ref N, @IntRange.%N.loc4_16.2 [symbolic = %N (constants.%N.586)]
// CHECK:STDOUT:         %Int.loc44_43.2: type = class_type @Int, @Int(constants.%N.586) [symbolic = %Int.loc44_43.1 (constants.%Int.b50e37.1)]
// CHECK:STDOUT:         %ptr.loc44_44.2: type = ptr_type %Int.loc44_43.2 [symbolic = %ptr.loc44_44.1 (constants.%ptr.1cf)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %cursor: @IntRange.as.Iterate.impl.Next.loc44_77.1.%ptr.loc44_44.1 (%ptr.1cf) = value_binding cursor, %cursor.param
// CHECK:STDOUT:       %return.param: ref @IntRange.as.Iterate.impl.Next.loc44_77.1.%Optional.loc44_75.1 (%Optional.23f) = out_param call_param2
// CHECK:STDOUT:       %return: ref @IntRange.as.Iterate.impl.Next.loc44_77.1.%Optional.loc44_75.1 (%Optional.23f) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %IntRange.as.Iterate.impl.NewCursor.decl.loc20_47.2: @IntRange.as.Iterate.impl.%IntRange.as.Iterate.impl.NewCursor.type.loc20_47.2 (%IntRange.as.Iterate.impl.NewCursor.type.c510a4.2) = fn_decl @IntRange.as.Iterate.impl.NewCursor.loc20_47.2 [symbolic = @IntRange.as.Iterate.impl.%IntRange.as.Iterate.impl.NewCursor.loc20_47.2 (constants.%IntRange.as.Iterate.impl.NewCursor.c95819.2)] {
// CHECK:STDOUT:       %self.patt: @IntRange.as.Iterate.impl.NewCursor.loc20_47.2.%pattern_type.1 (%pattern_type.6de) = value_binding_pattern self [concrete]
// CHECK:STDOUT:       %self.param_patt: @IntRange.as.Iterate.impl.NewCursor.loc20_47.2.%pattern_type.1 (%pattern_type.6de) = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:       %return.patt: @IntRange.as.Iterate.impl.NewCursor.loc20_47.2.%pattern_type.2 (%pattern_type.605) = return_slot_pattern [concrete]
// CHECK:STDOUT:       %return.param_patt: @IntRange.as.Iterate.impl.NewCursor.loc20_47.2.%pattern_type.2 (%pattern_type.605) = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %self.param: @IntRange.as.Iterate.impl.NewCursor.loc20_47.2.%IntRange (%IntRange.48b) = value_param call_param0
// CHECK:STDOUT:       %self: @IntRange.as.Iterate.impl.NewCursor.loc20_47.2.%IntRange (%IntRange.48b) = value_binding self, %self.param
// CHECK:STDOUT:       %return.param: ref @IntRange.as.Iterate.impl.NewCursor.loc20_47.2.%impl.elem1 (%impl.elem1.d3c) = out_param call_param1
// CHECK:STDOUT:       %return: ref @IntRange.as.Iterate.impl.NewCursor.loc20_47.2.%impl.elem1 (%impl.elem1.d3c) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %IntRange.as.Iterate.impl.Next.decl.loc44_77.2: @IntRange.as.Iterate.impl.%IntRange.as.Iterate.impl.Next.type.loc44_77.2 (%IntRange.as.Iterate.impl.Next.type.a05ce8.2) = fn_decl @IntRange.as.Iterate.impl.Next.loc44_77.2 [symbolic = @IntRange.as.Iterate.impl.%IntRange.as.Iterate.impl.Next.loc44_77.2 (constants.%IntRange.as.Iterate.impl.Next.d8655a.2)] {
// CHECK:STDOUT:       %self.patt: @IntRange.as.Iterate.impl.Next.loc44_77.2.%pattern_type.1 (%pattern_type.6de) = value_binding_pattern self [concrete]
// CHECK:STDOUT:       %self.param_patt: @IntRange.as.Iterate.impl.Next.loc44_77.2.%pattern_type.1 (%pattern_type.6de) = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:       %cursor.patt: @IntRange.as.Iterate.impl.Next.loc44_77.2.%pattern_type.2 (%pattern_type.852) = value_binding_pattern cursor [concrete]
// CHECK:STDOUT:       %cursor.param_patt: @IntRange.as.Iterate.impl.Next.loc44_77.2.%pattern_type.2 (%pattern_type.852) = value_param_pattern %cursor.patt, call_param1 [concrete]
// CHECK:STDOUT:       %return.patt: @IntRange.as.Iterate.impl.Next.loc44_77.2.%pattern_type.3 (%pattern_type.f5b) = return_slot_pattern [concrete]
// CHECK:STDOUT:       %return.param_patt: @IntRange.as.Iterate.impl.Next.loc44_77.2.%pattern_type.3 (%pattern_type.f5b) = out_param_pattern %return.patt, call_param2 [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %self.param: @IntRange.as.Iterate.impl.Next.loc44_77.2.%IntRange (%IntRange.48b) = value_param call_param0
// CHECK:STDOUT:       %self: @IntRange.as.Iterate.impl.Next.loc44_77.2.%IntRange (%IntRange.48b) = value_binding self, %self.param
// CHECK:STDOUT:       %cursor.param: @IntRange.as.Iterate.impl.Next.loc44_77.2.%ptr.1 (%ptr.ccc) = value_param call_param1
// CHECK:STDOUT:       %cursor: @IntRange.as.Iterate.impl.Next.loc44_77.2.%ptr.1 (%ptr.ccc) = value_binding cursor, %cursor.param
// CHECK:STDOUT:       %return.param: ref @IntRange.as.Iterate.impl.Next.loc44_77.2.%Optional.1 (%Optional.f2d) = out_param call_param2
// CHECK:STDOUT:       %return: ref @IntRange.as.Iterate.impl.Next.loc44_77.2.%Optional.1 (%Optional.f2d) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .N = <poisoned>
// CHECK:STDOUT:     .NewCursor = %IntRange.as.Iterate.impl.NewCursor.decl.loc20_47.1
// CHECK:STDOUT:     .Next = %IntRange.as.Iterate.impl.Next.decl.loc44_77.1
// CHECK:STDOUT:     witness = @IntRange.%Iterate.impl_witness
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @IntRange(%N.loc4_16.2: Core.IntLiteral) {
// CHECK:STDOUT:   %N.loc4_16.1: Core.IntLiteral = symbolic_binding N, 0 [symbolic = %N.loc4_16.1 (constants.%N.586)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %IntRange.Make.type: type = fn_type @IntRange.Make, @IntRange(%N.loc4_16.1) [symbolic = %IntRange.Make.type (constants.%IntRange.Make.type.121)]
// CHECK:STDOUT:   %IntRange.Make: @IntRange.%IntRange.Make.type (%IntRange.Make.type.121) = struct_value () [symbolic = %IntRange.Make (constants.%IntRange.Make.14e)]
// CHECK:STDOUT:   %.loc9: require_specific_def_type = require_specific_def @Int.as.Copy.impl(%N.loc4_16.1) [symbolic = %.loc9 (constants.%.1a2)]
// CHECK:STDOUT:   %Int.loc55_32.2: type = class_type @Int, @Int(%N.loc4_16.1) [symbolic = %Int.loc55_32.2 (constants.%Int.b50e37.1)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %Int.loc55_32.2 [symbolic = %require_complete (constants.%require_complete.e66303.2)]
// CHECK:STDOUT:   %IntRange: type = class_type @IntRange, @IntRange(%N.loc4_16.1) [symbolic = %IntRange (constants.%IntRange.48b)]
// CHECK:STDOUT:   %IntRange.elem: type = unbound_element_type %IntRange, %Int.loc55_32.2 [symbolic = %IntRange.elem (constants.%IntRange.elem.baf)]
// CHECK:STDOUT:   %struct_type.start.end: type = struct_type {.start: @IntRange.%Int.loc55_32.2 (%Int.b50e37.1), .end: @IntRange.%Int.loc55_32.2 (%Int.b50e37.1)} [symbolic = %struct_type.start.end (constants.%struct_type.start.end.cb5)]
// CHECK:STDOUT:   %complete_type.loc57_1.2: <witness> = complete_type_witness %struct_type.start.end [symbolic = %complete_type.loc57_1.2 (constants.%complete_type.4f0)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %IntRange.Make.decl: @IntRange.%IntRange.Make.type (%IntRange.Make.type.121) = fn_decl @IntRange.Make [symbolic = @IntRange.%IntRange.Make (constants.%IntRange.Make.14e)] {
// CHECK:STDOUT:       %start.patt: @IntRange.Make.%pattern_type.loc5_11 (%pattern_type.60f8e7.1) = value_binding_pattern start [concrete]
// CHECK:STDOUT:       %start.param_patt: @IntRange.Make.%pattern_type.loc5_11 (%pattern_type.60f8e7.1) = value_param_pattern %start.patt, call_param0 [concrete]
// CHECK:STDOUT:       %end.patt: @IntRange.Make.%pattern_type.loc5_11 (%pattern_type.60f8e7.1) = value_binding_pattern end [concrete]
// CHECK:STDOUT:       %end.param_patt: @IntRange.Make.%pattern_type.loc5_11 (%pattern_type.60f8e7.1) = value_param_pattern %end.patt, call_param1 [concrete]
// CHECK:STDOUT:       %return.patt: @IntRange.Make.%pattern_type.loc5_49 (%pattern_type.6de) = return_slot_pattern [concrete]
// CHECK:STDOUT:       %return.param_patt: @IntRange.Make.%pattern_type.loc5_49 (%pattern_type.6de) = out_param_pattern %return.patt, call_param2 [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %.loc5_52: type = specific_constant constants.%IntRange.48b, @IntRange(constants.%N.586) [symbolic = %IntRange (constants.%IntRange.48b)]
// CHECK:STDOUT:       %Self.ref: type = name_ref Self, %.loc5_52 [symbolic = %IntRange (constants.%IntRange.48b)]
// CHECK:STDOUT:       %start.param: @IntRange.Make.%Int.loc5_28.1 (%Int.b50e37.1) = value_param call_param0
// CHECK:STDOUT:       %.loc5_28: type = splice_block %Int.loc5_28.2 [symbolic = %Int.loc5_28.1 (constants.%Int.b50e37.1)] {
// CHECK:STDOUT:         %Core.ref.loc5_18: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:         %Int.ref.loc5_22: %Int.type = name_ref Int, imports.%Core.Int [concrete = constants.%Int.generic]
// CHECK:STDOUT:         %N.ref.loc5_27: Core.IntLiteral = name_ref N, @IntRange.%N.loc4_16.2 [symbolic = %N (constants.%N.586)]
// CHECK:STDOUT:         %Int.loc5_28.2: type = class_type @Int, @Int(constants.%N.586) [symbolic = %Int.loc5_28.1 (constants.%Int.b50e37.1)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %start: @IntRange.Make.%Int.loc5_28.1 (%Int.b50e37.1) = value_binding start, %start.param
// CHECK:STDOUT:       %end.param: @IntRange.Make.%Int.loc5_28.1 (%Int.b50e37.1) = value_param call_param1
// CHECK:STDOUT:       %.loc5_46: type = splice_block %Int.loc5_46 [symbolic = %Int.loc5_28.1 (constants.%Int.b50e37.1)] {
// CHECK:STDOUT:         %Core.ref.loc5_36: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:         %Int.ref.loc5_40: %Int.type = name_ref Int, imports.%Core.Int [concrete = constants.%Int.generic]
// CHECK:STDOUT:         %N.ref.loc5_45: Core.IntLiteral = name_ref N, @IntRange.%N.loc4_16.2 [symbolic = %N (constants.%N.586)]
// CHECK:STDOUT:         %Int.loc5_46: type = class_type @Int, @Int(constants.%N.586) [symbolic = %Int.loc5_28.1 (constants.%Int.b50e37.1)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %end: @IntRange.Make.%Int.loc5_28.1 (%Int.b50e37.1) = value_binding end, %end.param
// CHECK:STDOUT:       %return.param: ref @IntRange.Make.%IntRange (%IntRange.48b) = out_param call_param2
// CHECK:STDOUT:       %return: ref @IntRange.Make.%IntRange (%IntRange.48b) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     impl_decl @IntRange.as.Iterate.impl [concrete] {} {
// CHECK:STDOUT:       %Self.ref: type = name_ref Self, constants.%IntRange.48b [symbolic = %IntRange (constants.%IntRange.48b)]
// CHECK:STDOUT:       %Core.ref.loc9_11: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:       %Iterate.ref: type = name_ref Iterate, imports.%Core.Iterate [concrete = constants.%Iterate.type]
// CHECK:STDOUT:       %.Self: %Iterate.type = symbolic_binding .Self [symbolic_self = constants.%.Self.b7a]
// CHECK:STDOUT:       %.Self.ref.loc9_30: %Iterate.type = name_ref .Self, %.Self [symbolic_self = constants.%.Self.b7a]
// CHECK:STDOUT:       %CursorType.ref: %Iterate.assoc_type = name_ref CursorType, imports.%Core.import_ref.ed6 [concrete = constants.%assoc1.02e]
// CHECK:STDOUT:       %.Self.as_type.loc9_30: type = facet_access_type %.Self.ref.loc9_30 [symbolic_self = constants.%.Self.binding.as_type.573]
// CHECK:STDOUT:       %.loc9_30: type = converted %.Self.ref.loc9_30, %.Self.as_type.loc9_30 [symbolic_self = constants.%.Self.binding.as_type.573]
// CHECK:STDOUT:       %impl.elem1: type = impl_witness_access constants.%Iterate.lookup_impl_witness.a57, element1 [symbolic_self = constants.%impl.elem1.129]
// CHECK:STDOUT:       %Core.ref.loc9_44: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:       %Int.ref.loc9_48: %Int.type = name_ref Int, imports.%Core.Int [concrete = constants.%Int.generic]
// CHECK:STDOUT:       %N.ref.loc9_53: Core.IntLiteral = name_ref N, @IntRange.%N.loc4_16.2 [symbolic = %N (constants.%N.586)]
// CHECK:STDOUT:       %Int.loc9_54.2: type = class_type @Int, @Int(constants.%N.586) [symbolic = %Int.loc9_54.1 (constants.%Int.b50e37.1)]
// CHECK:STDOUT:       %.Self.ref.loc9_60: %Iterate.type = name_ref .Self, %.Self [symbolic_self = constants.%.Self.b7a]
// CHECK:STDOUT:       %ElementType.ref: %Iterate.assoc_type = name_ref ElementType, imports.%Core.import_ref.119 [concrete = constants.%assoc0.0f6]
// CHECK:STDOUT:       %.Self.as_type.loc9_60: type = facet_access_type %.Self.ref.loc9_60 [symbolic_self = constants.%.Self.binding.as_type.573]
// CHECK:STDOUT:       %.loc9_60: type = converted %.Self.ref.loc9_60, %.Self.as_type.loc9_60 [symbolic_self = constants.%.Self.binding.as_type.573]
// CHECK:STDOUT:       %impl.elem0.loc9: %Copy.type = impl_witness_access constants.%Iterate.lookup_impl_witness.a57, element0 [symbolic_self = constants.%impl.elem0.a97]
// CHECK:STDOUT:       %Core.ref.loc9_75: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:       %Int.ref.loc9_79: %Int.type = name_ref Int, imports.%Core.Int [concrete = constants.%Int.generic]
// CHECK:STDOUT:       %N.ref.loc9_84: Core.IntLiteral = name_ref N, @IntRange.%N.loc4_16.2 [symbolic = %N (constants.%N.586)]
// CHECK:STDOUT:       %Int.loc9_85: type = class_type @Int, @Int(constants.%N.586) [symbolic = %Int.loc9_54.1 (constants.%Int.b50e37.1)]
// CHECK:STDOUT:       %Copy.facet.loc9_85.2: %Copy.type = facet_value %Int.loc9_85, (constants.%Copy.lookup_impl_witness.035) [symbolic = %Copy.facet.loc9_85.1 (constants.%Copy.facet.d3d)]
// CHECK:STDOUT:       %.loc9_85.2: %Copy.type = converted %Int.loc9_85, %Copy.facet.loc9_85.2 [symbolic = %Copy.facet.loc9_85.1 (constants.%Copy.facet.d3d)]
// CHECK:STDOUT:       %.loc9_24: type = where_expr %.Self [symbolic = %Iterate_where.type (constants.%Iterate_where.type.a34)] {
// CHECK:STDOUT:         requirement_base_facet_type constants.%Iterate.type
// CHECK:STDOUT:         requirement_rewrite %impl.elem1, %Int.loc9_54.2
// CHECK:STDOUT:         requirement_rewrite %impl.elem0.loc9, %.loc9_85.2
// CHECK:STDOUT:       }
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %Iterate.impl_witness_table = impl_witness_table (%impl_witness_assoc_constant.loc9_87.2, %impl_witness_assoc_constant.loc9_87.1, @IntRange.as.Iterate.impl.%IntRange.as.Iterate.impl.NewCursor.decl.loc20_47.2, @IntRange.as.Iterate.impl.%IntRange.as.Iterate.impl.Next.decl.loc44_77.2), @IntRange.as.Iterate.impl [concrete]
// CHECK:STDOUT:     %Iterate.impl_witness: <witness> = impl_witness %Iterate.impl_witness_table, @IntRange.as.Iterate.impl(constants.%N.586) [symbolic = @IntRange.as.Iterate.impl.%Iterate.impl_witness (constants.%Iterate.impl_witness.273)]
// CHECK:STDOUT:     %impl_witness_assoc_constant.loc9_87.1: type = impl_witness_assoc_constant constants.%Int.b50e37.1 [symbolic = @IntRange.as.Iterate.impl.%Int.loc9_54.1 (constants.%Int.b50e37.1)]
// CHECK:STDOUT:     %impl_witness_assoc_constant.loc9_87.2: %Copy.type = impl_witness_assoc_constant constants.%Copy.facet.d3d [symbolic = @IntRange.as.Iterate.impl.%Copy.facet.loc9_85.1 (constants.%Copy.facet.d3d)]
// CHECK:STDOUT:     %Core.ref.loc55: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:     %Int.ref.loc55: %Int.type = name_ref Int, imports.%Core.Int [concrete = constants.%Int.generic]
// CHECK:STDOUT:     %N.ref.loc55: Core.IntLiteral = name_ref N, %N.loc4_16.2 [symbolic = %N.loc4_16.1 (constants.%N.586)]
// CHECK:STDOUT:     %Int.loc55_32.1: type = class_type @Int, @Int(constants.%N.586) [symbolic = %Int.loc55_32.2 (constants.%Int.b50e37.1)]
// CHECK:STDOUT:     %.loc55: @IntRange.%IntRange.elem (%IntRange.elem.baf) = field_decl start, element0 [concrete]
// CHECK:STDOUT:     %Core.ref.loc56: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:     %Int.ref.loc56: %Int.type = name_ref Int, imports.%Core.Int [concrete = constants.%Int.generic]
// CHECK:STDOUT:     %N.ref.loc56: Core.IntLiteral = name_ref N, %N.loc4_16.2 [symbolic = %N.loc4_16.1 (constants.%N.586)]
// CHECK:STDOUT:     %Int.loc56: type = class_type @Int, @Int(constants.%N.586) [symbolic = %Int.loc55_32.2 (constants.%Int.b50e37.1)]
// CHECK:STDOUT:     %.loc56: @IntRange.%IntRange.elem (%IntRange.elem.baf) = field_decl end, element1 [concrete]
// CHECK:STDOUT:     %complete_type.loc57_1.1: <witness> = complete_type_witness constants.%struct_type.start.end.cb5 [symbolic = %complete_type.loc57_1.2 (constants.%complete_type.4f0)]
// CHECK:STDOUT:     complete_type_witness = %complete_type.loc57_1.1
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%IntRange.48b
// CHECK:STDOUT:     .N = <poisoned>
// CHECK:STDOUT:     .Make = %IntRange.Make.decl
// CHECK:STDOUT:     .start [private] = %.loc55
// CHECK:STDOUT:     .end [private] = %.loc56
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @IntRange.Make(@IntRange.%N.loc4_16.2: Core.IntLiteral) {
// CHECK:STDOUT:   %N: Core.IntLiteral = symbolic_binding N, 0 [symbolic = %N (constants.%N.586)]
// CHECK:STDOUT:   %Int.loc5_28.1: type = class_type @Int, @Int(%N) [symbolic = %Int.loc5_28.1 (constants.%Int.b50e37.1)]
// CHECK:STDOUT:   %pattern_type.loc5_11: type = pattern_type %Int.loc5_28.1 [symbolic = %pattern_type.loc5_11 (constants.%pattern_type.60f8e7.1)]
// CHECK:STDOUT:   %IntRange: type = class_type @IntRange, @IntRange(%N) [symbolic = %IntRange (constants.%IntRange.48b)]
// CHECK:STDOUT:   %pattern_type.loc5_49: type = pattern_type %IntRange [symbolic = %pattern_type.loc5_49 (constants.%pattern_type.6de)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc5_49: <witness> = require_complete_type %IntRange [symbolic = %require_complete.loc5_49 (constants.%require_complete.b4c)]
// CHECK:STDOUT:   %require_complete.loc5_16: <witness> = require_complete_type %Int.loc5_28.1 [symbolic = %require_complete.loc5_16 (constants.%require_complete.e66303.2)]
// CHECK:STDOUT:   %struct_type.start.end: type = struct_type {.start: @IntRange.Make.%Int.loc5_28.1 (%Int.b50e37.1), .end: @IntRange.Make.%Int.loc5_28.1 (%Int.b50e37.1)} [symbolic = %struct_type.start.end (constants.%struct_type.start.end.cb5)]
// CHECK:STDOUT:   %.loc6_22.2: require_specific_def_type = require_specific_def @Int.as.Copy.impl(%N) [symbolic = %.loc6_22.2 (constants.%.1a2)]
// CHECK:STDOUT:   %Copy.lookup_impl_witness: <witness> = lookup_impl_witness %Int.loc5_28.1, @Copy [symbolic = %Copy.lookup_impl_witness (constants.%Copy.lookup_impl_witness.035)]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %Int.loc5_28.1, (%Copy.lookup_impl_witness) [symbolic = %Copy.facet (constants.%Copy.facet.d3d)]
// CHECK:STDOUT:   %.loc6_22.3: type = fn_type_with_self_type constants.%Copy.Op.type, %Copy.facet [symbolic = %.loc6_22.3 (constants.%.20c)]
// CHECK:STDOUT:   %impl.elem0.loc6_22.2: @IntRange.Make.%.loc6_22.3 (%.20c) = impl_witness_access %Copy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc6_22.2 (constants.%impl.elem0.625)]
// CHECK:STDOUT:   %specific_impl_fn.loc6_22.2: <specific function> = specific_impl_function %impl.elem0.loc6_22.2, @Copy.Op(%Copy.facet) [symbolic = %specific_impl_fn.loc6_22.2 (constants.%specific_impl_fn.c88)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%start.param: @IntRange.Make.%Int.loc5_28.1 (%Int.b50e37.1), %end.param: @IntRange.Make.%Int.loc5_28.1 (%Int.b50e37.1)) -> %return.param: @IntRange.Make.%IntRange (%IntRange.48b) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %start.ref: @IntRange.Make.%Int.loc5_28.1 (%Int.b50e37.1) = name_ref start, %start
// CHECK:STDOUT:     %end.ref: @IntRange.Make.%Int.loc5_28.1 (%Int.b50e37.1) = name_ref end, %end
// CHECK:STDOUT:     %.loc6_39.1: @IntRange.Make.%struct_type.start.end (%struct_type.start.end.cb5) = struct_literal (%start.ref, %end.ref)
// CHECK:STDOUT:     %impl.elem0.loc6_22.1: @IntRange.Make.%.loc6_22.3 (%.20c) = impl_witness_access constants.%Copy.lookup_impl_witness.035, element0 [symbolic = %impl.elem0.loc6_22.2 (constants.%impl.elem0.625)]
// CHECK:STDOUT:     %bound_method.loc6_22.1: <bound method> = bound_method %start.ref, %impl.elem0.loc6_22.1
// CHECK:STDOUT:     %specific_impl_fn.loc6_22.1: <specific function> = specific_impl_function %impl.elem0.loc6_22.1, @Copy.Op(constants.%Copy.facet.d3d) [symbolic = %specific_impl_fn.loc6_22.2 (constants.%specific_impl_fn.c88)]
// CHECK:STDOUT:     %bound_method.loc6_22.2: <bound method> = bound_method %start.ref, %specific_impl_fn.loc6_22.1
// CHECK:STDOUT:     %.loc6_22.1: init @IntRange.Make.%Int.loc5_28.1 (%Int.b50e37.1) = call %bound_method.loc6_22.2(%start.ref)
// CHECK:STDOUT:     %.loc6_39.2: ref @IntRange.Make.%Int.loc5_28.1 (%Int.b50e37.1) = class_element_access %return, element0
// CHECK:STDOUT:     %.loc6_39.3: init @IntRange.Make.%Int.loc5_28.1 (%Int.b50e37.1) = initialize_from %.loc6_22.1 to %.loc6_39.2
// CHECK:STDOUT:     %impl.elem0.loc6_36: @IntRange.Make.%.loc6_22.3 (%.20c) = impl_witness_access constants.%Copy.lookup_impl_witness.035, element0 [symbolic = %impl.elem0.loc6_22.2 (constants.%impl.elem0.625)]
// CHECK:STDOUT:     %bound_method.loc6_36.1: <bound method> = bound_method %end.ref, %impl.elem0.loc6_36
// CHECK:STDOUT:     %specific_impl_fn.loc6_36: <specific function> = specific_impl_function %impl.elem0.loc6_36, @Copy.Op(constants.%Copy.facet.d3d) [symbolic = %specific_impl_fn.loc6_22.2 (constants.%specific_impl_fn.c88)]
// CHECK:STDOUT:     %bound_method.loc6_36.2: <bound method> = bound_method %end.ref, %specific_impl_fn.loc6_36
// CHECK:STDOUT:     %.loc6_36: init @IntRange.Make.%Int.loc5_28.1 (%Int.b50e37.1) = call %bound_method.loc6_36.2(%end.ref)
// CHECK:STDOUT:     %.loc6_39.4: ref @IntRange.Make.%Int.loc5_28.1 (%Int.b50e37.1) = class_element_access %return, element1
// CHECK:STDOUT:     %.loc6_39.5: init @IntRange.Make.%Int.loc5_28.1 (%Int.b50e37.1) = initialize_from %.loc6_36 to %.loc6_39.4
// CHECK:STDOUT:     %.loc6_39.6: init @IntRange.Make.%IntRange (%IntRange.48b) = class_init (%.loc6_39.3, %.loc6_39.5), %return
// CHECK:STDOUT:     %.loc6_40: init @IntRange.Make.%IntRange (%IntRange.48b) = converted %.loc6_39.1, %.loc6_39.6
// CHECK:STDOUT:     return %.loc6_40 to %return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @IntRange.as.Iterate.impl.NewCursor.loc20_47.1(@IntRange.%N.loc4_16.2: Core.IntLiteral) {
// CHECK:STDOUT:   %N: Core.IntLiteral = symbolic_binding N, 0 [symbolic = %N (constants.%N.586)]
// CHECK:STDOUT:   %IntRange: type = class_type @IntRange, @IntRange(%N) [symbolic = %IntRange (constants.%IntRange.48b)]
// CHECK:STDOUT:   %pattern_type.loc20_18: type = pattern_type %IntRange [symbolic = %pattern_type.loc20_18 (constants.%pattern_type.6de)]
// CHECK:STDOUT:   %Int.loc20_45.1: type = class_type @Int, @Int(%N) [symbolic = %Int.loc20_45.1 (constants.%Int.b50e37.1)]
// CHECK:STDOUT:   %pattern_type.loc20_32: type = pattern_type %Int.loc20_45.1 [symbolic = %pattern_type.loc20_32 (constants.%pattern_type.60f8e7.1)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc20_22: <witness> = require_complete_type %IntRange [symbolic = %require_complete.loc20_22 (constants.%require_complete.b4c)]
// CHECK:STDOUT:   %IntRange.elem: type = unbound_element_type %IntRange, %Int.loc20_45.1 [symbolic = %IntRange.elem (constants.%IntRange.elem.baf)]
// CHECK:STDOUT:   %require_complete.loc20_60: <witness> = require_complete_type %Int.loc20_45.1 [symbolic = %require_complete.loc20_60 (constants.%require_complete.e66303.2)]
// CHECK:STDOUT:   %.loc20_60.4: require_specific_def_type = require_specific_def @Int.as.Copy.impl(%N) [symbolic = %.loc20_60.4 (constants.%.1a2)]
// CHECK:STDOUT:   %Copy.lookup_impl_witness: <witness> = lookup_impl_witness %Int.loc20_45.1, @Copy [symbolic = %Copy.lookup_impl_witness (constants.%Copy.lookup_impl_witness.035)]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %Int.loc20_45.1, (%Copy.lookup_impl_witness) [symbolic = %Copy.facet (constants.%Copy.facet.d3d)]
// CHECK:STDOUT:   %.loc20_60.5: type = fn_type_with_self_type constants.%Copy.Op.type, %Copy.facet [symbolic = %.loc20_60.5 (constants.%.20c)]
// CHECK:STDOUT:   %impl.elem0.loc20_60.2: @IntRange.as.Iterate.impl.NewCursor.loc20_47.1.%.loc20_60.5 (%.20c) = impl_witness_access %Copy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc20_60.2 (constants.%impl.elem0.625)]
// CHECK:STDOUT:   %specific_impl_fn.loc20_60.2: <specific function> = specific_impl_function %impl.elem0.loc20_60.2, @Copy.Op(%Copy.facet) [symbolic = %specific_impl_fn.loc20_60.2 (constants.%specific_impl_fn.c88)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @IntRange.as.Iterate.impl.NewCursor.loc20_47.1.%IntRange (%IntRange.48b)) -> @IntRange.as.Iterate.impl.NewCursor.loc20_47.1.%Int.loc20_45.1 (%Int.b50e37.1) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %self.ref: @IntRange.as.Iterate.impl.NewCursor.loc20_47.1.%IntRange (%IntRange.48b) = name_ref self, %self
// CHECK:STDOUT:     %start.ref: @IntRange.as.Iterate.impl.NewCursor.loc20_47.1.%IntRange.elem (%IntRange.elem.baf) = name_ref start, @IntRange.%.loc55 [concrete = @IntRange.%.loc55]
// CHECK:STDOUT:     %.loc20_60.1: ref @IntRange.as.Iterate.impl.NewCursor.loc20_47.1.%Int.loc20_45.1 (%Int.b50e37.1) = class_element_access %self.ref, element0
// CHECK:STDOUT:     %.loc20_60.2: @IntRange.as.Iterate.impl.NewCursor.loc20_47.1.%Int.loc20_45.1 (%Int.b50e37.1) = acquire_value %.loc20_60.1
// CHECK:STDOUT:     %impl.elem0.loc20_60.1: @IntRange.as.Iterate.impl.NewCursor.loc20_47.1.%.loc20_60.5 (%.20c) = impl_witness_access constants.%Copy.lookup_impl_witness.035, element0 [symbolic = %impl.elem0.loc20_60.2 (constants.%impl.elem0.625)]
// CHECK:STDOUT:     %bound_method.loc20_60.1: <bound method> = bound_method %.loc20_60.2, %impl.elem0.loc20_60.1
// CHECK:STDOUT:     %specific_impl_fn.loc20_60.1: <specific function> = specific_impl_function %impl.elem0.loc20_60.1, @Copy.Op(constants.%Copy.facet.d3d) [symbolic = %specific_impl_fn.loc20_60.2 (constants.%specific_impl_fn.c88)]
// CHECK:STDOUT:     %bound_method.loc20_60.2: <bound method> = bound_method %.loc20_60.2, %specific_impl_fn.loc20_60.1
// CHECK:STDOUT:     %.loc20_60.3: init @IntRange.as.Iterate.impl.NewCursor.loc20_47.1.%Int.loc20_45.1 (%Int.b50e37.1) = call %bound_method.loc20_60.2(%.loc20_60.2)
// CHECK:STDOUT:     return %.loc20_60.3 to %return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @IntRange.as.Iterate.impl.Next.loc44_77.1(@IntRange.%N.loc4_16.2: Core.IntLiteral) {
// CHECK:STDOUT:   %N: Core.IntLiteral = symbolic_binding N, 0 [symbolic = %N (constants.%N.586)]
// CHECK:STDOUT:   %IntRange: type = class_type @IntRange, @IntRange(%N) [symbolic = %IntRange (constants.%IntRange.48b)]
// CHECK:STDOUT:   %pattern_type.loc44_13: type = pattern_type %IntRange [symbolic = %pattern_type.loc44_13 (constants.%pattern_type.6de)]
// CHECK:STDOUT:   %Int.loc44_43.1: type = class_type @Int, @Int(%N) [symbolic = %Int.loc44_43.1 (constants.%Int.b50e37.1)]
// CHECK:STDOUT:   %ptr.loc44_44.1: type = ptr_type %Int.loc44_43.1 [symbolic = %ptr.loc44_44.1 (constants.%ptr.1cf)]
// CHECK:STDOUT:   %pattern_type.loc44_25: type = pattern_type %ptr.loc44_44.1 [symbolic = %pattern_type.loc44_25 (constants.%pattern_type.054)]
// CHECK:STDOUT:   %Copy.lookup_impl_witness: <witness> = lookup_impl_witness %Int.loc44_43.1, @Copy [symbolic = %Copy.lookup_impl_witness (constants.%Copy.lookup_impl_witness.035)]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %Int.loc44_43.1, (%Copy.lookup_impl_witness) [symbolic = %Copy.facet (constants.%Copy.facet.d3d)]
// CHECK:STDOUT:   %.loc44_75.1: require_specific_def_type = require_specific_def @T.binding.as_type.as.OptionalStorage.impl(%Copy.facet) [symbolic = %.loc44_75.1 (constants.%.350)]
// CHECK:STDOUT:   %OptionalStorage.lookup_impl_witness: <witness> = lookup_impl_witness %Int.loc44_43.1, @OptionalStorage [symbolic = %OptionalStorage.lookup_impl_witness (constants.%OptionalStorage.lookup_impl_witness.8c6)]
// CHECK:STDOUT:   %OptionalStorage.facet.loc44_75.1: %OptionalStorage.type = facet_value %Int.loc44_43.1, (%OptionalStorage.lookup_impl_witness) [symbolic = %OptionalStorage.facet.loc44_75.1 (constants.%OptionalStorage.facet.ea4)]
// CHECK:STDOUT:   %Optional.loc44_75.1: type = class_type @Optional, @Optional(%OptionalStorage.facet.loc44_75.1) [symbolic = %Optional.loc44_75.1 (constants.%Optional.23f)]
// CHECK:STDOUT:   %pattern_type.loc44_47: type = pattern_type %Optional.loc44_75.1 [symbolic = %pattern_type.loc44_47 (constants.%pattern_type.9f4)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc44_47: <witness> = require_complete_type %Optional.loc44_75.1 [symbolic = %require_complete.loc44_47 (constants.%require_complete.b49)]
// CHECK:STDOUT:   %require_complete.loc44_17: <witness> = require_complete_type %IntRange [symbolic = %require_complete.loc44_17 (constants.%require_complete.b4c)]
// CHECK:STDOUT:   %require_complete.loc44_31: <witness> = require_complete_type %ptr.loc44_44.1 [symbolic = %require_complete.loc44_31 (constants.%require_complete.6ca2)]
// CHECK:STDOUT:   %require_complete.loc45: <witness> = require_complete_type %Int.loc44_43.1 [symbolic = %require_complete.loc45 (constants.%require_complete.e66303.2)]
// CHECK:STDOUT:   %pattern_type.loc45: type = pattern_type %Int.loc44_43.1 [symbolic = %pattern_type.loc45 (constants.%pattern_type.60f8e7.1)]
// CHECK:STDOUT:   %.loc45_32.4: require_specific_def_type = require_specific_def @Int.as.Copy.impl(%N) [symbolic = %.loc45_32.4 (constants.%.1a2)]
// CHECK:STDOUT:   %.loc45_32.5: type = fn_type_with_self_type constants.%Copy.Op.type, %Copy.facet [symbolic = %.loc45_32.5 (constants.%.20c)]
// CHECK:STDOUT:   %impl.elem0.loc45_32.2: @IntRange.as.Iterate.impl.Next.loc44_77.1.%.loc45_32.5 (%.20c) = impl_witness_access %Copy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc45_32.2 (constants.%impl.elem0.625)]
// CHECK:STDOUT:   %specific_impl_fn.loc45_32.2: <specific function> = specific_impl_function %impl.elem0.loc45_32.2, @Copy.Op(%Copy.facet) [symbolic = %specific_impl_fn.loc45_32.2 (constants.%specific_impl_fn.c88)]
// CHECK:STDOUT:   %IntRange.elem: type = unbound_element_type %IntRange, %Int.loc44_43.1 [symbolic = %IntRange.elem (constants.%IntRange.elem.baf)]
// CHECK:STDOUT:   %OrderedWith.type.loc46_17.2: type = facet_type <@OrderedWith, @OrderedWith(%Int.loc44_43.1)> [symbolic = %OrderedWith.type.loc46_17.2 (constants.%OrderedWith.type.393)]
// CHECK:STDOUT:   %require_complete.loc46: <witness> = require_complete_type %OrderedWith.type.loc46_17.2 [symbolic = %require_complete.loc46 (constants.%require_complete.4f7)]
// CHECK:STDOUT:   %OrderedWith.assoc_type: type = assoc_entity_type @OrderedWith, @OrderedWith(%Int.loc44_43.1) [symbolic = %OrderedWith.assoc_type (constants.%OrderedWith.assoc_type.72a)]
// CHECK:STDOUT:   %assoc0: @IntRange.as.Iterate.impl.Next.loc44_77.1.%OrderedWith.assoc_type (%OrderedWith.assoc_type.72a) = assoc_entity element0, imports.%Core.import_ref.02d [symbolic = %assoc0 (constants.%assoc0.3f6)]
// CHECK:STDOUT:   %.loc46_17.2: require_specific_def_type = require_specific_def @Int.as.OrderedWith.impl.849(%N, %N) [symbolic = %.loc46_17.2 (constants.%.e09)]
// CHECK:STDOUT:   %OrderedWith.impl_witness: <witness> = impl_witness imports.%OrderedWith.impl_witness_table.401, @Int.as.OrderedWith.impl.849(%N, %N) [symbolic = %OrderedWith.impl_witness (constants.%OrderedWith.impl_witness.586)]
// CHECK:STDOUT:   %OrderedWith.Less.type: type = fn_type @OrderedWith.Less, @OrderedWith(%Int.loc44_43.1) [symbolic = %OrderedWith.Less.type (constants.%OrderedWith.Less.type.c6f)]
// CHECK:STDOUT:   %OrderedWith.facet: @IntRange.as.Iterate.impl.Next.loc44_77.1.%OrderedWith.type.loc46_17.2 (%OrderedWith.type.393) = facet_value %Int.loc44_43.1, (%OrderedWith.impl_witness) [symbolic = %OrderedWith.facet (constants.%OrderedWith.facet)]
// CHECK:STDOUT:   %.loc46_17.3: type = fn_type_with_self_type %OrderedWith.Less.type, %OrderedWith.facet [symbolic = %.loc46_17.3 (constants.%.db5)]
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Less.type: type = fn_type @Int.as.OrderedWith.impl.Less.1, @Int.as.OrderedWith.impl.849(%N, %N) [symbolic = %Int.as.OrderedWith.impl.Less.type (constants.%Int.as.OrderedWith.impl.Less.type.f52)]
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Less: @IntRange.as.Iterate.impl.Next.loc44_77.1.%Int.as.OrderedWith.impl.Less.type (%Int.as.OrderedWith.impl.Less.type.f52) = struct_value () [symbolic = %Int.as.OrderedWith.impl.Less (constants.%Int.as.OrderedWith.impl.Less.af3)]
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Less.specific_fn: <specific function> = specific_function %Int.as.OrderedWith.impl.Less, @Int.as.OrderedWith.impl.Less.1(%N, %N) [symbolic = %Int.as.OrderedWith.impl.Less.specific_fn (constants.%Int.as.OrderedWith.impl.Less.specific_fn.0ff)]
// CHECK:STDOUT:   %.loc47_9.2: require_specific_def_type = require_specific_def @Int.as.Inc.impl(%N) [symbolic = %.loc47_9.2 (constants.%.4ff)]
// CHECK:STDOUT:   %Inc.lookup_impl_witness: <witness> = lookup_impl_witness %Int.loc44_43.1, @Inc [symbolic = %Inc.lookup_impl_witness (constants.%Inc.lookup_impl_witness)]
// CHECK:STDOUT:   %Inc.facet: %Inc.type = facet_value %Int.loc44_43.1, (%Inc.lookup_impl_witness) [symbolic = %Inc.facet (constants.%Inc.facet)]
// CHECK:STDOUT:   %.loc47_9.3: type = fn_type_with_self_type constants.%Inc.Op.type, %Inc.facet [symbolic = %.loc47_9.3 (constants.%.080)]
// CHECK:STDOUT:   %impl.elem0.loc47_9.2: @IntRange.as.Iterate.impl.Next.loc44_77.1.%.loc47_9.3 (%.080) = impl_witness_access %Inc.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc47_9.2 (constants.%impl.elem0.923)]
// CHECK:STDOUT:   %specific_impl_fn.loc47_9.2: <specific function> = specific_impl_function %impl.elem0.loc47_9.2, @Inc.Op(%Inc.facet) [symbolic = %specific_impl_fn.loc47_9.2 (constants.%specific_impl_fn.173)]
// CHECK:STDOUT:   %Optional.Some.type: type = fn_type @Optional.Some, @Optional(%OptionalStorage.facet.loc44_75.1) [symbolic = %Optional.Some.type (constants.%Optional.Some.type.74c)]
// CHECK:STDOUT:   %Optional.Some: @IntRange.as.Iterate.impl.Next.loc44_77.1.%Optional.Some.type (%Optional.Some.type.74c) = struct_value () [symbolic = %Optional.Some (constants.%Optional.Some.37c)]
// CHECK:STDOUT:   %Optional.Some.specific_fn.loc48_42.2: <specific function> = specific_function %Optional.Some, @Optional.Some(%OptionalStorage.facet.loc44_75.1) [symbolic = %Optional.Some.specific_fn.loc48_42.2 (constants.%Optional.Some.specific_fn.90e)]
// CHECK:STDOUT:   %facet_value: %type_where = facet_value %Int.loc44_43.1, () [symbolic = %facet_value (constants.%facet_value.4a9)]
// CHECK:STDOUT:   %.loc45_7.1: require_specific_def_type = require_specific_def @DestroyT.binding.as_type.as.Destroy.impl(%facet_value) [symbolic = %.loc45_7.1 (constants.%.4d6)]
// CHECK:STDOUT:   %Destroy.impl_witness: <witness> = impl_witness imports.%Destroy.impl_witness_table, @DestroyT.binding.as_type.as.Destroy.impl(%facet_value) [symbolic = %Destroy.impl_witness (constants.%Destroy.impl_witness.b47)]
// CHECK:STDOUT:   %Destroy.facet: %Destroy.type = facet_value %Int.loc44_43.1, (%Destroy.impl_witness) [symbolic = %Destroy.facet (constants.%Destroy.facet.11d)]
// CHECK:STDOUT:   %.loc45_7.2: type = fn_type_with_self_type constants.%Destroy.Op.type, %Destroy.facet [symbolic = %.loc45_7.2 (constants.%.a64)]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.type: type = fn_type @DestroyT.binding.as_type.as.Destroy.impl.Op, @DestroyT.binding.as_type.as.Destroy.impl(%facet_value) [symbolic = %DestroyT.binding.as_type.as.Destroy.impl.Op.type (constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.type.353)]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op: @IntRange.as.Iterate.impl.Next.loc44_77.1.%DestroyT.binding.as_type.as.Destroy.impl.Op.type (%DestroyT.binding.as_type.as.Destroy.impl.Op.type.353) = struct_value () [symbolic = %DestroyT.binding.as_type.as.Destroy.impl.Op (constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.3be)]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn: <specific function> = specific_function %DestroyT.binding.as_type.as.Destroy.impl.Op, @DestroyT.binding.as_type.as.Destroy.impl.Op(%facet_value) [symbolic = %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn (constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.e87)]
// CHECK:STDOUT:   %Optional.None.type: type = fn_type @Optional.None, @Optional(%OptionalStorage.facet.loc44_75.1) [symbolic = %Optional.None.type (constants.%Optional.None.type.a39)]
// CHECK:STDOUT:   %Optional.None: @IntRange.as.Iterate.impl.Next.loc44_77.1.%Optional.None.type (%Optional.None.type.a39) = struct_value () [symbolic = %Optional.None (constants.%Optional.None.b0b)]
// CHECK:STDOUT:   %Optional.None.specific_fn.loc50_42.2: <specific function> = specific_function %Optional.None, @Optional.None(%OptionalStorage.facet.loc44_75.1) [symbolic = %Optional.None.specific_fn.loc50_42.2 (constants.%Optional.None.specific_fn.94d)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @IntRange.as.Iterate.impl.Next.loc44_77.1.%IntRange (%IntRange.48b), %cursor.param: @IntRange.as.Iterate.impl.Next.loc44_77.1.%ptr.loc44_44.1 (%ptr.1cf)) -> %return.param: @IntRange.as.Iterate.impl.Next.loc44_77.1.%Optional.loc44_75.1 (%Optional.23f) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %value.patt: @IntRange.as.Iterate.impl.Next.loc44_77.1.%pattern_type.loc45 (%pattern_type.60f8e7.1) = ref_binding_pattern value [concrete]
// CHECK:STDOUT:       %value.var_patt: @IntRange.as.Iterate.impl.Next.loc44_77.1.%pattern_type.loc45 (%pattern_type.60f8e7.1) = var_pattern %value.patt [concrete]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %value.var: ref @IntRange.as.Iterate.impl.Next.loc44_77.1.%Int.loc44_43.1 (%Int.b50e37.1) = var %value.var_patt
// CHECK:STDOUT:     %cursor.ref.loc45: @IntRange.as.Iterate.impl.Next.loc44_77.1.%ptr.loc44_44.1 (%ptr.1cf) = name_ref cursor, %cursor
// CHECK:STDOUT:     %.loc45_32.1: ref @IntRange.as.Iterate.impl.Next.loc44_77.1.%Int.loc44_43.1 (%Int.b50e37.1) = deref %cursor.ref.loc45
// CHECK:STDOUT:     %.loc45_32.2: @IntRange.as.Iterate.impl.Next.loc44_77.1.%Int.loc44_43.1 (%Int.b50e37.1) = acquire_value %.loc45_32.1
// CHECK:STDOUT:     %impl.elem0.loc45_32.1: @IntRange.as.Iterate.impl.Next.loc44_77.1.%.loc45_32.5 (%.20c) = impl_witness_access constants.%Copy.lookup_impl_witness.035, element0 [symbolic = %impl.elem0.loc45_32.2 (constants.%impl.elem0.625)]
// CHECK:STDOUT:     %bound_method.loc45_32.1: <bound method> = bound_method %.loc45_32.2, %impl.elem0.loc45_32.1
// CHECK:STDOUT:     %specific_impl_fn.loc45_32.1: <specific function> = specific_impl_function %impl.elem0.loc45_32.1, @Copy.Op(constants.%Copy.facet.d3d) [symbolic = %specific_impl_fn.loc45_32.2 (constants.%specific_impl_fn.c88)]
// CHECK:STDOUT:     %bound_method.loc45_32.2: <bound method> = bound_method %.loc45_32.2, %specific_impl_fn.loc45_32.1
// CHECK:STDOUT:     %.loc45_32.3: init @IntRange.as.Iterate.impl.Next.loc44_77.1.%Int.loc44_43.1 (%Int.b50e37.1) = call %bound_method.loc45_32.2(%.loc45_32.2)
// CHECK:STDOUT:     assign %value.var, %.loc45_32.3
// CHECK:STDOUT:     %.loc45_28: type = splice_block %Int.loc45 [symbolic = %Int.loc44_43.1 (constants.%Int.b50e37.1)] {
// CHECK:STDOUT:       %Core.ref.loc45: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:       %Int.ref.loc45: %Int.type = name_ref Int, imports.%Core.Int [concrete = constants.%Int.generic]
// CHECK:STDOUT:       %N.ref.loc45: Core.IntLiteral = name_ref N, @IntRange.%N.loc4_16.2 [symbolic = %N (constants.%N.586)]
// CHECK:STDOUT:       %Int.loc45: type = class_type @Int, @Int(constants.%N.586) [symbolic = %Int.loc44_43.1 (constants.%Int.b50e37.1)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %value: ref @IntRange.as.Iterate.impl.Next.loc44_77.1.%Int.loc44_43.1 (%Int.b50e37.1) = ref_binding value, %value.var
// CHECK:STDOUT:     %value.ref.loc46: ref @IntRange.as.Iterate.impl.Next.loc44_77.1.%Int.loc44_43.1 (%Int.b50e37.1) = name_ref value, %value
// CHECK:STDOUT:     %self.ref: @IntRange.as.Iterate.impl.Next.loc44_77.1.%IntRange (%IntRange.48b) = name_ref self, %self
// CHECK:STDOUT:     %end.ref: @IntRange.as.Iterate.impl.Next.loc44_77.1.%IntRange.elem (%IntRange.elem.baf) = name_ref end, @IntRange.%.loc56 [concrete = @IntRange.%.loc56]
// CHECK:STDOUT:     %.loc46_23.1: ref @IntRange.as.Iterate.impl.Next.loc44_77.1.%Int.loc44_43.1 (%Int.b50e37.1) = class_element_access %self.ref, element1
// CHECK:STDOUT:     %.loc46_23.2: @IntRange.as.Iterate.impl.Next.loc44_77.1.%Int.loc44_43.1 (%Int.b50e37.1) = acquire_value %.loc46_23.1
// CHECK:STDOUT:     %OrderedWith.type.loc46_17.1: type = facet_type <@OrderedWith, @OrderedWith(constants.%Int.b50e37.1)> [symbolic = %OrderedWith.type.loc46_17.2 (constants.%OrderedWith.type.393)]
// CHECK:STDOUT:     %.loc46_17.1: @IntRange.as.Iterate.impl.Next.loc44_77.1.%OrderedWith.assoc_type (%OrderedWith.assoc_type.72a) = specific_constant imports.%Core.import_ref.128, @OrderedWith(constants.%Int.b50e37.1) [symbolic = %assoc0 (constants.%assoc0.3f6)]
// CHECK:STDOUT:     %Less.ref: @IntRange.as.Iterate.impl.Next.loc44_77.1.%OrderedWith.assoc_type (%OrderedWith.assoc_type.72a) = name_ref Less, %.loc46_17.1 [symbolic = %assoc0 (constants.%assoc0.3f6)]
// CHECK:STDOUT:     %impl.elem0.loc46: @IntRange.as.Iterate.impl.Next.loc44_77.1.%.loc46_17.3 (%.db5) = impl_witness_access constants.%OrderedWith.impl_witness.586, element0 [symbolic = %Int.as.OrderedWith.impl.Less (constants.%Int.as.OrderedWith.impl.Less.af3)]
// CHECK:STDOUT:     %bound_method.loc46_17.1: <bound method> = bound_method %value.ref.loc46, %impl.elem0.loc46
// CHECK:STDOUT:     %specific_fn.loc46: <specific function> = specific_function %impl.elem0.loc46, @Int.as.OrderedWith.impl.Less.1(constants.%N.586, constants.%N.586) [symbolic = %Int.as.OrderedWith.impl.Less.specific_fn (constants.%Int.as.OrderedWith.impl.Less.specific_fn.0ff)]
// CHECK:STDOUT:     %bound_method.loc46_17.2: <bound method> = bound_method %value.ref.loc46, %specific_fn.loc46
// CHECK:STDOUT:     %.loc46_11: @IntRange.as.Iterate.impl.Next.loc44_77.1.%Int.loc44_43.1 (%Int.b50e37.1) = acquire_value %value.ref.loc46
// CHECK:STDOUT:     %Int.as.OrderedWith.impl.Less.call: init bool = call %bound_method.loc46_17.2(%.loc46_11, %.loc46_23.2)
// CHECK:STDOUT:     %.loc46_27.1: bool = value_of_initializer %Int.as.OrderedWith.impl.Less.call
// CHECK:STDOUT:     %.loc46_27.2: bool = converted %Int.as.OrderedWith.impl.Less.call, %.loc46_27.1
// CHECK:STDOUT:     if %.loc46_27.2 br !if.then else br !if.else
// CHECK:STDOUT:
// CHECK:STDOUT:   !if.then:
// CHECK:STDOUT:     %cursor.ref.loc47: @IntRange.as.Iterate.impl.Next.loc44_77.1.%ptr.loc44_44.1 (%ptr.1cf) = name_ref cursor, %cursor
// CHECK:STDOUT:     %.loc47_11: ref @IntRange.as.Iterate.impl.Next.loc44_77.1.%Int.loc44_43.1 (%Int.b50e37.1) = deref %cursor.ref.loc47
// CHECK:STDOUT:     %impl.elem0.loc47_9.1: @IntRange.as.Iterate.impl.Next.loc44_77.1.%.loc47_9.3 (%.080) = impl_witness_access constants.%Inc.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc47_9.2 (constants.%impl.elem0.923)]
// CHECK:STDOUT:     %bound_method.loc47_9.1: <bound method> = bound_method %.loc47_11, %impl.elem0.loc47_9.1
// CHECK:STDOUT:     %specific_impl_fn.loc47_9.1: <specific function> = specific_impl_function %impl.elem0.loc47_9.1, @Inc.Op(constants.%Inc.facet) [symbolic = %specific_impl_fn.loc47_9.2 (constants.%specific_impl_fn.173)]
// CHECK:STDOUT:     %bound_method.loc47_9.2: <bound method> = bound_method %.loc47_11, %specific_impl_fn.loc47_9.1
// CHECK:STDOUT:     %.loc47_9.1: init %empty_tuple.type = call %bound_method.loc47_9.2(%.loc47_11)
// CHECK:STDOUT:     %Core.ref.loc48_16: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:     %Optional.ref.loc48: %Optional.type = name_ref Optional, imports.%Core.Optional [concrete = constants.%Optional.generic]
// CHECK:STDOUT:     %Core.ref.loc48_30: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:     %Int.ref.loc48: %Int.type = name_ref Int, imports.%Core.Int [concrete = constants.%Int.generic]
// CHECK:STDOUT:     %N.ref.loc48: Core.IntLiteral = name_ref N, @IntRange.%N.loc4_16.2 [symbolic = %N (constants.%N.586)]
// CHECK:STDOUT:     %Int.loc48: type = class_type @Int, @Int(constants.%N.586) [symbolic = %Int.loc44_43.1 (constants.%Int.b50e37.1)]
// CHECK:STDOUT:     %OptionalStorage.facet.loc48_41: %OptionalStorage.type = facet_value %Int.loc48, (constants.%OptionalStorage.lookup_impl_witness.8c6) [symbolic = %OptionalStorage.facet.loc44_75.1 (constants.%OptionalStorage.facet.ea4)]
// CHECK:STDOUT:     %.loc48_41: %OptionalStorage.type = converted %Int.loc48, %OptionalStorage.facet.loc48_41 [symbolic = %OptionalStorage.facet.loc44_75.1 (constants.%OptionalStorage.facet.ea4)]
// CHECK:STDOUT:     %Optional.loc48: type = class_type @Optional, @Optional(constants.%OptionalStorage.facet.ea4) [symbolic = %Optional.loc44_75.1 (constants.%Optional.23f)]
// CHECK:STDOUT:     %.loc48_42: @IntRange.as.Iterate.impl.Next.loc44_77.1.%Optional.Some.type (%Optional.Some.type.74c) = specific_constant imports.%Core.import_ref.961, @Optional(constants.%OptionalStorage.facet.ea4) [symbolic = %Optional.Some (constants.%Optional.Some.37c)]
// CHECK:STDOUT:     %Some.ref: @IntRange.as.Iterate.impl.Next.loc44_77.1.%Optional.Some.type (%Optional.Some.type.74c) = name_ref Some, %.loc48_42 [symbolic = %Optional.Some (constants.%Optional.Some.37c)]
// CHECK:STDOUT:     %value.ref.loc48: ref @IntRange.as.Iterate.impl.Next.loc44_77.1.%Int.loc44_43.1 (%Int.b50e37.1) = name_ref value, %value
// CHECK:STDOUT:     %OptionalStorage.facet.loc48_53.1: %OptionalStorage.type = facet_value constants.%Int.b50e37.1, (constants.%OptionalStorage.lookup_impl_witness.8c6) [symbolic = %OptionalStorage.facet.loc44_75.1 (constants.%OptionalStorage.facet.ea4)]
// CHECK:STDOUT:     %.loc48_53.1: %OptionalStorage.type = converted constants.%Int.b50e37.1, %OptionalStorage.facet.loc48_53.1 [symbolic = %OptionalStorage.facet.loc44_75.1 (constants.%OptionalStorage.facet.ea4)]
// CHECK:STDOUT:     %OptionalStorage.facet.loc48_53.2: %OptionalStorage.type = facet_value constants.%Int.b50e37.1, (constants.%OptionalStorage.lookup_impl_witness.8c6) [symbolic = %OptionalStorage.facet.loc44_75.1 (constants.%OptionalStorage.facet.ea4)]
// CHECK:STDOUT:     %.loc48_53.2: %OptionalStorage.type = converted constants.%Int.b50e37.1, %OptionalStorage.facet.loc48_53.2 [symbolic = %OptionalStorage.facet.loc44_75.1 (constants.%OptionalStorage.facet.ea4)]
// CHECK:STDOUT:     %Optional.Some.specific_fn.loc48_42.1: <specific function> = specific_function %Some.ref, @Optional.Some(constants.%OptionalStorage.facet.ea4) [symbolic = %Optional.Some.specific_fn.loc48_42.2 (constants.%Optional.Some.specific_fn.90e)]
// CHECK:STDOUT:     %.loc44_47.1: ref @IntRange.as.Iterate.impl.Next.loc44_77.1.%Optional.loc44_75.1 (%Optional.23f) = splice_block %return {}
// CHECK:STDOUT:     %.loc48_48: @IntRange.as.Iterate.impl.Next.loc44_77.1.%Int.loc44_43.1 (%Int.b50e37.1) = acquire_value %value.ref.loc48
// CHECK:STDOUT:     %Optional.Some.call: init @IntRange.as.Iterate.impl.Next.loc44_77.1.%Optional.loc44_75.1 (%Optional.23f) = call %Optional.Some.specific_fn.loc48_42.1(%.loc48_48) to %.loc44_47.1
// CHECK:STDOUT:     %impl.elem0.loc45_7.1: @IntRange.as.Iterate.impl.Next.loc44_77.1.%.loc45_7.2 (%.a64) = impl_witness_access constants.%Destroy.impl_witness.b47, element0 [symbolic = %DestroyT.binding.as_type.as.Destroy.impl.Op (constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.3be)]
// CHECK:STDOUT:     %bound_method.loc45_7.1: <bound method> = bound_method %value.var, %impl.elem0.loc45_7.1
// CHECK:STDOUT:     %specific_fn.loc45_7.1: <specific function> = specific_function %impl.elem0.loc45_7.1, @DestroyT.binding.as_type.as.Destroy.impl.Op(constants.%facet_value.4a9) [symbolic = %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn (constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.e87)]
// CHECK:STDOUT:     %bound_method.loc45_7.2: <bound method> = bound_method %value.var, %specific_fn.loc45_7.1
// CHECK:STDOUT:     %DestroyT.binding.as_type.as.Destroy.impl.Op.call.loc45_7.1: init %empty_tuple.type = call %bound_method.loc45_7.2(%value.var)
// CHECK:STDOUT:     return %Optional.Some.call to %return
// CHECK:STDOUT:
// CHECK:STDOUT:   !if.else:
// CHECK:STDOUT:     %Core.ref.loc50_16: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:     %Optional.ref.loc50: %Optional.type = name_ref Optional, imports.%Core.Optional [concrete = constants.%Optional.generic]
// CHECK:STDOUT:     %Core.ref.loc50_30: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:     %Int.ref.loc50: %Int.type = name_ref Int, imports.%Core.Int [concrete = constants.%Int.generic]
// CHECK:STDOUT:     %N.ref.loc50: Core.IntLiteral = name_ref N, @IntRange.%N.loc4_16.2 [symbolic = %N (constants.%N.586)]
// CHECK:STDOUT:     %Int.loc50: type = class_type @Int, @Int(constants.%N.586) [symbolic = %Int.loc44_43.1 (constants.%Int.b50e37.1)]
// CHECK:STDOUT:     %OptionalStorage.facet.loc50: %OptionalStorage.type = facet_value %Int.loc50, (constants.%OptionalStorage.lookup_impl_witness.8c6) [symbolic = %OptionalStorage.facet.loc44_75.1 (constants.%OptionalStorage.facet.ea4)]
// CHECK:STDOUT:     %.loc50_41: %OptionalStorage.type = converted %Int.loc50, %OptionalStorage.facet.loc50 [symbolic = %OptionalStorage.facet.loc44_75.1 (constants.%OptionalStorage.facet.ea4)]
// CHECK:STDOUT:     %Optional.loc50: type = class_type @Optional, @Optional(constants.%OptionalStorage.facet.ea4) [symbolic = %Optional.loc44_75.1 (constants.%Optional.23f)]
// CHECK:STDOUT:     %.loc50_42: @IntRange.as.Iterate.impl.Next.loc44_77.1.%Optional.None.type (%Optional.None.type.a39) = specific_constant imports.%Core.import_ref.f2f, @Optional(constants.%OptionalStorage.facet.ea4) [symbolic = %Optional.None (constants.%Optional.None.b0b)]
// CHECK:STDOUT:     %None.ref: @IntRange.as.Iterate.impl.Next.loc44_77.1.%Optional.None.type (%Optional.None.type.a39) = name_ref None, %.loc50_42 [symbolic = %Optional.None (constants.%Optional.None.b0b)]
// CHECK:STDOUT:     %Optional.None.specific_fn.loc50_42.1: <specific function> = specific_function %None.ref, @Optional.None(constants.%OptionalStorage.facet.ea4) [symbolic = %Optional.None.specific_fn.loc50_42.2 (constants.%Optional.None.specific_fn.94d)]
// CHECK:STDOUT:     %.loc44_47.2: ref @IntRange.as.Iterate.impl.Next.loc44_77.1.%Optional.loc44_75.1 (%Optional.23f) = splice_block %return {}
// CHECK:STDOUT:     %Optional.None.call: init @IntRange.as.Iterate.impl.Next.loc44_77.1.%Optional.loc44_75.1 (%Optional.23f) = call %Optional.None.specific_fn.loc50_42.1() to %.loc44_47.2
// CHECK:STDOUT:     %impl.elem0.loc45_7.2: @IntRange.as.Iterate.impl.Next.loc44_77.1.%.loc45_7.2 (%.a64) = impl_witness_access constants.%Destroy.impl_witness.b47, element0 [symbolic = %DestroyT.binding.as_type.as.Destroy.impl.Op (constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.3be)]
// CHECK:STDOUT:     %bound_method.loc45_7.3: <bound method> = bound_method %value.var, %impl.elem0.loc45_7.2
// CHECK:STDOUT:     %specific_fn.loc45_7.2: <specific function> = specific_function %impl.elem0.loc45_7.2, @DestroyT.binding.as_type.as.Destroy.impl.Op(constants.%facet_value.4a9) [symbolic = %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn (constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.e87)]
// CHECK:STDOUT:     %bound_method.loc45_7.4: <bound method> = bound_method %value.var, %specific_fn.loc45_7.2
// CHECK:STDOUT:     %DestroyT.binding.as_type.as.Destroy.impl.Op.call.loc45_7.2: init %empty_tuple.type = call %bound_method.loc45_7.4(%value.var)
// CHECK:STDOUT:     return %Optional.None.call to %return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @IntRange.as.Iterate.impl.NewCursor.loc20_47.2(@IntRange.%N.loc4_16.2: Core.IntLiteral) {
// CHECK:STDOUT:   %N: Core.IntLiteral = symbolic_binding N, 0 [symbolic = %N (constants.%N.586)]
// CHECK:STDOUT:   %IntRange: type = class_type @IntRange, @IntRange(%N) [symbolic = %IntRange (constants.%IntRange.48b)]
// CHECK:STDOUT:   %pattern_type.1: type = pattern_type %IntRange [symbolic = %pattern_type.1 (constants.%pattern_type.6de)]
// CHECK:STDOUT:   %Iterate.lookup_impl_witness: <witness> = lookup_impl_witness %IntRange, @Iterate [symbolic = %Iterate.lookup_impl_witness (constants.%Iterate.lookup_impl_witness.5aa)]
// CHECK:STDOUT:   %impl.elem1: type = impl_witness_access %Iterate.lookup_impl_witness, element1 [symbolic = %impl.elem1 (constants.%impl.elem1.d3c)]
// CHECK:STDOUT:   %pattern_type.2: type = pattern_type %impl.elem1 [symbolic = %pattern_type.2 (constants.%pattern_type.605)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %IntRange.as.Iterate.impl.NewCursor.type: type = fn_type @IntRange.as.Iterate.impl.NewCursor.loc20_47.1, @IntRange.as.Iterate.impl(%N) [symbolic = %IntRange.as.Iterate.impl.NewCursor.type (constants.%IntRange.as.Iterate.impl.NewCursor.type.c510a4.1)]
// CHECK:STDOUT:   %IntRange.as.Iterate.impl.NewCursor: @IntRange.as.Iterate.impl.NewCursor.loc20_47.2.%IntRange.as.Iterate.impl.NewCursor.type (%IntRange.as.Iterate.impl.NewCursor.type.c510a4.1) = struct_value () [symbolic = %IntRange.as.Iterate.impl.NewCursor (constants.%IntRange.as.Iterate.impl.NewCursor.c95819.1)]
// CHECK:STDOUT:   %IntRange.as.Iterate.impl.NewCursor.specific_fn.loc20_47.2: <specific function> = specific_function %IntRange.as.Iterate.impl.NewCursor, @IntRange.as.Iterate.impl.NewCursor.loc20_47.1(%N) [symbolic = %IntRange.as.Iterate.impl.NewCursor.specific_fn.loc20_47.2 (constants.%IntRange.as.Iterate.impl.NewCursor.specific_fn)]
// CHECK:STDOUT:   %require_complete.1: <witness> = require_complete_type %IntRange [symbolic = %require_complete.1 (constants.%require_complete.b4c)]
// CHECK:STDOUT:   %Int: type = class_type @Int, @Int(%N) [symbolic = %Int (constants.%Int.b50e37.1)]
// CHECK:STDOUT:   %require_complete.loc20_47.1: <witness> = require_complete_type %impl.elem1 [symbolic = %require_complete.loc20_47.1 (constants.%require_complete.9f9)]
// CHECK:STDOUT:   %ImplicitAs.type.loc20_47.2: type = facet_type <@ImplicitAs, @ImplicitAs(%impl.elem1)> [symbolic = %ImplicitAs.type.loc20_47.2 (constants.%ImplicitAs.type.59a)]
// CHECK:STDOUT:   %require_complete.loc20_47.2: <witness> = require_complete_type %ImplicitAs.type.loc20_47.2 [symbolic = %require_complete.loc20_47.2 (constants.%require_complete.61c)]
// CHECK:STDOUT:   %ImplicitAs.assoc_type: type = assoc_entity_type @ImplicitAs, @ImplicitAs(%impl.elem1) [symbolic = %ImplicitAs.assoc_type (constants.%ImplicitAs.assoc_type.7a4)]
// CHECK:STDOUT:   %assoc0: @IntRange.as.Iterate.impl.NewCursor.loc20_47.2.%ImplicitAs.assoc_type (%ImplicitAs.assoc_type.7a4) = assoc_entity element0, imports.%Core.import_ref.004393.1 [symbolic = %assoc0 (constants.%assoc0.9df)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @IntRange.as.Iterate.impl.NewCursor.loc20_47.2.%IntRange (%IntRange.48b)) -> %return.param: @IntRange.as.Iterate.impl.NewCursor.loc20_47.2.%impl.elem1 (%impl.elem1.d3c) [thunk @IntRange.as.Iterate.impl.%IntRange.as.Iterate.impl.NewCursor.decl.loc20_47.1] {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %.loc20_47.1: @IntRange.as.Iterate.impl.NewCursor.loc20_47.2.%IntRange.as.Iterate.impl.NewCursor.type (%IntRange.as.Iterate.impl.NewCursor.type.c510a4.1) = specific_constant @IntRange.as.Iterate.impl.%IntRange.as.Iterate.impl.NewCursor.decl.loc20_47.1, @IntRange.as.Iterate.impl(constants.%N.586) [symbolic = %IntRange.as.Iterate.impl.NewCursor (constants.%IntRange.as.Iterate.impl.NewCursor.c95819.1)]
// CHECK:STDOUT:     %NewCursor.ref: @IntRange.as.Iterate.impl.NewCursor.loc20_47.2.%IntRange.as.Iterate.impl.NewCursor.type (%IntRange.as.Iterate.impl.NewCursor.type.c510a4.1) = name_ref NewCursor, %.loc20_47.1 [symbolic = %IntRange.as.Iterate.impl.NewCursor (constants.%IntRange.as.Iterate.impl.NewCursor.c95819.1)]
// CHECK:STDOUT:     %self.ref: @IntRange.as.Iterate.impl.NewCursor.loc20_47.2.%IntRange (%IntRange.48b) = name_ref self, %self.param
// CHECK:STDOUT:     %return.ref: ref @IntRange.as.Iterate.impl.NewCursor.loc20_47.2.%impl.elem1 (%impl.elem1.d3c) = name_ref <return slot>, %return.param
// CHECK:STDOUT:     %IntRange.as.Iterate.impl.NewCursor.bound: <bound method> = bound_method %self.ref, %NewCursor.ref
// CHECK:STDOUT:     %IntRange.as.Iterate.impl.NewCursor.specific_fn.loc20_47.1: <specific function> = specific_function %NewCursor.ref, @IntRange.as.Iterate.impl.NewCursor.loc20_47.1(constants.%N.586) [symbolic = %IntRange.as.Iterate.impl.NewCursor.specific_fn.loc20_47.2 (constants.%IntRange.as.Iterate.impl.NewCursor.specific_fn)]
// CHECK:STDOUT:     %bound_method: <bound method> = bound_method %self.ref, %IntRange.as.Iterate.impl.NewCursor.specific_fn.loc20_47.1
// CHECK:STDOUT:     %IntRange.as.Iterate.impl.NewCursor.call: init @IntRange.as.Iterate.impl.NewCursor.loc20_47.2.%Int (%Int.b50e37.1) = call %bound_method(%self.ref)
// CHECK:STDOUT:     %ImplicitAs.type.loc20_47.1: type = facet_type <@ImplicitAs, @ImplicitAs(constants.%impl.elem1.d3c)> [symbolic = %ImplicitAs.type.loc20_47.2 (constants.%ImplicitAs.type.59a)]
// CHECK:STDOUT:     %.loc20_47.2: @IntRange.as.Iterate.impl.NewCursor.loc20_47.2.%ImplicitAs.assoc_type (%ImplicitAs.assoc_type.7a4) = specific_constant imports.%Core.import_ref.12f, @ImplicitAs(constants.%impl.elem1.d3c) [symbolic = %assoc0 (constants.%assoc0.9df)]
// CHECK:STDOUT:     %Convert.ref: @IntRange.as.Iterate.impl.NewCursor.loc20_47.2.%ImplicitAs.assoc_type (%ImplicitAs.assoc_type.7a4) = name_ref Convert, %.loc20_47.2 [symbolic = %assoc0 (constants.%assoc0.9df)]
// CHECK:STDOUT:     %.loc20_47.3: @IntRange.as.Iterate.impl.NewCursor.loc20_47.2.%impl.elem1 (%impl.elem1.d3c) = converted %IntRange.as.Iterate.impl.NewCursor.call, <error> [concrete = <error>]
// CHECK:STDOUT:     return <error> to %return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @IntRange.as.Iterate.impl.Next.loc44_77.2(@IntRange.%N.loc4_16.2: Core.IntLiteral) {
// CHECK:STDOUT:   %N: Core.IntLiteral = symbolic_binding N, 0 [symbolic = %N (constants.%N.586)]
// CHECK:STDOUT:   %IntRange: type = class_type @IntRange, @IntRange(%N) [symbolic = %IntRange (constants.%IntRange.48b)]
// CHECK:STDOUT:   %pattern_type.1: type = pattern_type %IntRange [symbolic = %pattern_type.1 (constants.%pattern_type.6de)]
// CHECK:STDOUT:   %Iterate.lookup_impl_witness: <witness> = lookup_impl_witness %IntRange, @Iterate [symbolic = %Iterate.lookup_impl_witness (constants.%Iterate.lookup_impl_witness.5aa)]
// CHECK:STDOUT:   %impl.elem1: type = impl_witness_access %Iterate.lookup_impl_witness, element1 [symbolic = %impl.elem1 (constants.%impl.elem1.d3c)]
// CHECK:STDOUT:   %ptr.1: type = ptr_type %impl.elem1 [symbolic = %ptr.1 (constants.%ptr.ccc)]
// CHECK:STDOUT:   %pattern_type.2: type = pattern_type %ptr.1 [symbolic = %pattern_type.2 (constants.%pattern_type.852)]
// CHECK:STDOUT:   %impl.elem0: %Copy.type = impl_witness_access %Iterate.lookup_impl_witness, element0 [symbolic = %impl.elem0 (constants.%impl.elem0.18c)]
// CHECK:STDOUT:   %as_type: type = facet_access_type %impl.elem0 [symbolic = %as_type (constants.%as_type.349)]
// CHECK:STDOUT:   %OptionalStorage.lookup_impl_witness.1: <witness> = lookup_impl_witness %as_type, @OptionalStorage [symbolic = %OptionalStorage.lookup_impl_witness.1 (constants.%OptionalStorage.lookup_impl_witness.bc5)]
// CHECK:STDOUT:   %OptionalStorage.facet.1: %OptionalStorage.type = facet_value %as_type, (%OptionalStorage.lookup_impl_witness.1) [symbolic = %OptionalStorage.facet.1 (constants.%OptionalStorage.facet.201)]
// CHECK:STDOUT:   %Optional.1: type = class_type @Optional, @Optional(%OptionalStorage.facet.1) [symbolic = %Optional.1 (constants.%Optional.f2d)]
// CHECK:STDOUT:   %pattern_type.3: type = pattern_type %Optional.1 [symbolic = %pattern_type.3 (constants.%pattern_type.f5b)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %IntRange.as.Iterate.impl.Next.type: type = fn_type @IntRange.as.Iterate.impl.Next.loc44_77.1, @IntRange.as.Iterate.impl(%N) [symbolic = %IntRange.as.Iterate.impl.Next.type (constants.%IntRange.as.Iterate.impl.Next.type.a05ce8.1)]
// CHECK:STDOUT:   %IntRange.as.Iterate.impl.Next: @IntRange.as.Iterate.impl.Next.loc44_77.2.%IntRange.as.Iterate.impl.Next.type (%IntRange.as.Iterate.impl.Next.type.a05ce8.1) = struct_value () [symbolic = %IntRange.as.Iterate.impl.Next (constants.%IntRange.as.Iterate.impl.Next.d8655a.1)]
// CHECK:STDOUT:   %IntRange.as.Iterate.impl.Next.specific_fn.loc44_77.2: <specific function> = specific_function %IntRange.as.Iterate.impl.Next, @IntRange.as.Iterate.impl.Next.loc44_77.1(%N) [symbolic = %IntRange.as.Iterate.impl.Next.specific_fn.loc44_77.2 (constants.%IntRange.as.Iterate.impl.Next.specific_fn)]
// CHECK:STDOUT:   %Int: type = class_type @Int, @Int(%N) [symbolic = %Int (constants.%Int.b50e37.1)]
// CHECK:STDOUT:   %OptionalStorage.lookup_impl_witness.loc44: <witness> = lookup_impl_witness %Int, @OptionalStorage [symbolic = %OptionalStorage.lookup_impl_witness.loc44 (constants.%OptionalStorage.lookup_impl_witness.8c6)]
// CHECK:STDOUT:   %OptionalStorage.facet.loc44: %OptionalStorage.type = facet_value %Int, (%OptionalStorage.lookup_impl_witness.loc44) [symbolic = %OptionalStorage.facet.loc44 (constants.%OptionalStorage.facet.ea4)]
// CHECK:STDOUT:   %Optional.loc44: type = class_type @Optional, @Optional(%OptionalStorage.facet.loc44) [symbolic = %Optional.loc44 (constants.%Optional.23f)]
// CHECK:STDOUT:   %require_complete.1: <witness> = require_complete_type %IntRange [symbolic = %require_complete.1 (constants.%require_complete.b4c)]
// CHECK:STDOUT:   %ptr.2: type = ptr_type %Int [symbolic = %ptr.2 (constants.%ptr.1cf)]
// CHECK:STDOUT:   %require_complete.2: <witness> = require_complete_type %ptr.2 [symbolic = %require_complete.2 (constants.%require_complete.6ca2)]
// CHECK:STDOUT:   %ImplicitAs.type.2: type = facet_type <@ImplicitAs, @ImplicitAs(%ptr.2)> [symbolic = %ImplicitAs.type.2 (constants.%ImplicitAs.type.a76)]
// CHECK:STDOUT:   %require_complete.3: <witness> = require_complete_type %ImplicitAs.type.2 [symbolic = %require_complete.3 (constants.%require_complete.3e7)]
// CHECK:STDOUT:   %ImplicitAs.assoc_type.1: type = assoc_entity_type @ImplicitAs, @ImplicitAs(%ptr.2) [symbolic = %ImplicitAs.assoc_type.1 (constants.%ImplicitAs.assoc_type.2f2)]
// CHECK:STDOUT:   %assoc0.1: @IntRange.as.Iterate.impl.Next.loc44_77.2.%ImplicitAs.assoc_type.1 (%ImplicitAs.assoc_type.2f2) = assoc_entity element0, imports.%Core.import_ref.004393.1 [symbolic = %assoc0.1 (constants.%assoc0.bf6)]
// CHECK:STDOUT:   %require_complete.loc44_77.1: <witness> = require_complete_type %Optional.1 [symbolic = %require_complete.loc44_77.1 (constants.%require_complete.3f6)]
// CHECK:STDOUT:   %ImplicitAs.type.loc44_77.2: type = facet_type <@ImplicitAs, @ImplicitAs(%Optional.1)> [symbolic = %ImplicitAs.type.loc44_77.2 (constants.%ImplicitAs.type.903)]
// CHECK:STDOUT:   %require_complete.loc44_77.2: <witness> = require_complete_type %ImplicitAs.type.loc44_77.2 [symbolic = %require_complete.loc44_77.2 (constants.%require_complete.6cad)]
// CHECK:STDOUT:   %ImplicitAs.assoc_type.loc44: type = assoc_entity_type @ImplicitAs, @ImplicitAs(%Optional.1) [symbolic = %ImplicitAs.assoc_type.loc44 (constants.%ImplicitAs.assoc_type.561)]
// CHECK:STDOUT:   %assoc0.loc44: @IntRange.as.Iterate.impl.Next.loc44_77.2.%ImplicitAs.assoc_type.loc44 (%ImplicitAs.assoc_type.561) = assoc_entity element0, imports.%Core.import_ref.004393.1 [symbolic = %assoc0.loc44 (constants.%assoc0.5f6)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @IntRange.as.Iterate.impl.Next.loc44_77.2.%IntRange (%IntRange.48b), %cursor.param: @IntRange.as.Iterate.impl.Next.loc44_77.2.%ptr.1 (%ptr.ccc)) -> %return.param: @IntRange.as.Iterate.impl.Next.loc44_77.2.%Optional.1 (%Optional.f2d) [thunk @IntRange.as.Iterate.impl.%IntRange.as.Iterate.impl.Next.decl.loc44_77.1] {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %.loc44_77.1: @IntRange.as.Iterate.impl.Next.loc44_77.2.%IntRange.as.Iterate.impl.Next.type (%IntRange.as.Iterate.impl.Next.type.a05ce8.1) = specific_constant @IntRange.as.Iterate.impl.%IntRange.as.Iterate.impl.Next.decl.loc44_77.1, @IntRange.as.Iterate.impl(constants.%N.586) [symbolic = %IntRange.as.Iterate.impl.Next (constants.%IntRange.as.Iterate.impl.Next.d8655a.1)]
// CHECK:STDOUT:     %Next.ref: @IntRange.as.Iterate.impl.Next.loc44_77.2.%IntRange.as.Iterate.impl.Next.type (%IntRange.as.Iterate.impl.Next.type.a05ce8.1) = name_ref Next, %.loc44_77.1 [symbolic = %IntRange.as.Iterate.impl.Next (constants.%IntRange.as.Iterate.impl.Next.d8655a.1)]
// CHECK:STDOUT:     %self.ref: @IntRange.as.Iterate.impl.Next.loc44_77.2.%IntRange (%IntRange.48b) = name_ref self, %self.param
// CHECK:STDOUT:     %cursor.ref: @IntRange.as.Iterate.impl.Next.loc44_77.2.%ptr.1 (%ptr.ccc) = name_ref cursor, %cursor.param
// CHECK:STDOUT:     %return.ref: ref @IntRange.as.Iterate.impl.Next.loc44_77.2.%Optional.1 (%Optional.f2d) = name_ref <return slot>, %return.param
// CHECK:STDOUT:     %IntRange.as.Iterate.impl.Next.bound: <bound method> = bound_method %self.ref, %Next.ref
// CHECK:STDOUT:     %IntRange.as.Iterate.impl.Next.specific_fn.loc44_77.1: <specific function> = specific_function %Next.ref, @IntRange.as.Iterate.impl.Next.loc44_77.1(constants.%N.586) [symbolic = %IntRange.as.Iterate.impl.Next.specific_fn.loc44_77.2 (constants.%IntRange.as.Iterate.impl.Next.specific_fn)]
// CHECK:STDOUT:     %bound_method: <bound method> = bound_method %self.ref, %IntRange.as.Iterate.impl.Next.specific_fn.loc44_77.1
// CHECK:STDOUT:     %.loc44_77.2: ref @IntRange.as.Iterate.impl.Next.loc44_77.2.%Optional.loc44 (%Optional.23f) = temporary_storage
// CHECK:STDOUT:     %ImplicitAs.type.1: type = facet_type <@ImplicitAs, @ImplicitAs(constants.%ptr.1cf)> [symbolic = %ImplicitAs.type.2 (constants.%ImplicitAs.type.a76)]
// CHECK:STDOUT:     %.1: @IntRange.as.Iterate.impl.Next.loc44_77.2.%ImplicitAs.assoc_type.1 (%ImplicitAs.assoc_type.2f2) = specific_constant imports.%Core.import_ref.12f, @ImplicitAs(constants.%ptr.1cf) [symbolic = %assoc0.1 (constants.%assoc0.bf6)]
// CHECK:STDOUT:     %Convert.ref.1: @IntRange.as.Iterate.impl.Next.loc44_77.2.%ImplicitAs.assoc_type.1 (%ImplicitAs.assoc_type.2f2) = name_ref Convert, %.1 [symbolic = %assoc0.1 (constants.%assoc0.bf6)]
// CHECK:STDOUT:     %.2: @IntRange.as.Iterate.impl.Next.loc44_77.2.%ptr.2 (%ptr.1cf) = converted %cursor.ref, <error> [concrete = <error>]
// CHECK:STDOUT:     %IntRange.as.Iterate.impl.Next.call: init @IntRange.as.Iterate.impl.Next.loc44_77.2.%Optional.loc44 (%Optional.23f) = call %bound_method(%self.ref, <error>) to %.loc44_77.2
// CHECK:STDOUT:     %ImplicitAs.type.loc44_77.1: type = facet_type <@ImplicitAs, @ImplicitAs(constants.%Optional.f2d)> [symbolic = %ImplicitAs.type.loc44_77.2 (constants.%ImplicitAs.type.903)]
// CHECK:STDOUT:     %.loc44_77.3: @IntRange.as.Iterate.impl.Next.loc44_77.2.%ImplicitAs.assoc_type.loc44 (%ImplicitAs.assoc_type.561) = specific_constant imports.%Core.import_ref.12f, @ImplicitAs(constants.%Optional.f2d) [symbolic = %assoc0.loc44 (constants.%assoc0.5f6)]
// CHECK:STDOUT:     %Convert.ref.loc44: @IntRange.as.Iterate.impl.Next.loc44_77.2.%ImplicitAs.assoc_type.loc44 (%ImplicitAs.assoc_type.561) = name_ref Convert, %.loc44_77.3 [symbolic = %assoc0.loc44 (constants.%assoc0.5f6)]
// CHECK:STDOUT:     %.loc44_77.4: @IntRange.as.Iterate.impl.Next.loc44_77.2.%Optional.1 (%Optional.f2d) = converted %IntRange.as.Iterate.impl.Next.call, <error> [concrete = <error>]
// CHECK:STDOUT:     return <error> to %return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Range(%end.param: %i32) -> %return.param: %IntRange.365 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %IntRange.ref.loc60: %IntRange.type = name_ref IntRange, file.%IntRange.decl [concrete = constants.%IntRange.generic]
// CHECK:STDOUT:   %int_32.loc60: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %IntRange.loc60: type = class_type @IntRange, @IntRange(constants.%int_32) [concrete = constants.%IntRange.365]
// CHECK:STDOUT:   %.loc60_22: %IntRange.Make.type.cef = specific_constant @IntRange.%IntRange.Make.decl, @IntRange(constants.%int_32) [concrete = constants.%IntRange.Make.0dc]
// CHECK:STDOUT:   %Make.ref: %IntRange.Make.type.cef = name_ref Make, %.loc60_22 [concrete = constants.%IntRange.Make.0dc]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0.5c6]
// CHECK:STDOUT:   %end.ref: %i32 = name_ref end, %end
// CHECK:STDOUT:   %IntRange.Make.specific_fn: <specific function> = specific_function %Make.ref, @IntRange.Make(constants.%int_32) [concrete = constants.%IntRange.Make.specific_fn]
// CHECK:STDOUT:   %.loc59_20: ref %IntRange.365 = splice_block %return {}
// CHECK:STDOUT:   %impl.elem0: %.cbf = impl_witness_access constants.%ImplicitAs.impl_witness.f89, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.e9b]
// CHECK:STDOUT:   %bound_method.loc60_28.1: <bound method> = bound_method %int_0, %impl.elem0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.0b3]
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc60_28.2: <bound method> = bound_method %int_0, %specific_fn [concrete = constants.%bound_method.9be]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %i32 = call %bound_method.loc60_28.2(%int_0) [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   %.loc60_28.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   %.loc60_28.2: %i32 = converted %int_0, %.loc60_28.1 [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   %IntRange.Make.call: init %IntRange.365 = call %IntRange.Make.specific_fn(%.loc60_28.2, %end.ref) to %.loc59_20
// CHECK:STDOUT:   return %IntRange.Make.call to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @IntRange(constants.%N.586) {
// CHECK:STDOUT:   %N.loc4_16.1 => constants.%N.586
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %IntRange.Make.type => constants.%IntRange.Make.type.121
// CHECK:STDOUT:   %IntRange.Make => constants.%IntRange.Make.14e
// CHECK:STDOUT:   %.loc9 => constants.%.1a2
// CHECK:STDOUT:   %Int.loc55_32.2 => constants.%Int.b50e37.1
// CHECK:STDOUT:   %require_complete => constants.%require_complete.e66303.2
// CHECK:STDOUT:   %IntRange => constants.%IntRange.48b
// CHECK:STDOUT:   %IntRange.elem => constants.%IntRange.elem.baf
// CHECK:STDOUT:   %struct_type.start.end => constants.%struct_type.start.end.cb5
// CHECK:STDOUT:   %complete_type.loc57_1.2 => constants.%complete_type.4f0
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @IntRange.Make(constants.%N.586) {
// CHECK:STDOUT:   %N => constants.%N.586
// CHECK:STDOUT:   %Int.loc5_28.1 => constants.%Int.b50e37.1
// CHECK:STDOUT:   %pattern_type.loc5_11 => constants.%pattern_type.60f8e7.1
// CHECK:STDOUT:   %IntRange => constants.%IntRange.48b
// CHECK:STDOUT:   %pattern_type.loc5_49 => constants.%pattern_type.6de
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @IntRange.as.Iterate.impl(constants.%N.586) {
// CHECK:STDOUT:   %N => constants.%N.586
// CHECK:STDOUT:   %IntRange => constants.%IntRange.48b
// CHECK:STDOUT:   %Int.loc9_54.1 => constants.%Int.b50e37.1
// CHECK:STDOUT:   %.loc9_85.1 => constants.%.1a2
// CHECK:STDOUT:   %Copy.lookup_impl_witness => constants.%Copy.lookup_impl_witness.035
// CHECK:STDOUT:   %Copy.facet.loc9_85.1 => constants.%Copy.facet.d3d
// CHECK:STDOUT:   %Iterate_where.type => constants.%Iterate_where.type.a34
// CHECK:STDOUT:   %require_complete => constants.%require_complete.806
// CHECK:STDOUT:   %Iterate.impl_witness => constants.%Iterate.impl_witness.273
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %IntRange.as.Iterate.impl.NewCursor.type.loc20_47.1 => constants.%IntRange.as.Iterate.impl.NewCursor.type.c510a4.1
// CHECK:STDOUT:   %IntRange.as.Iterate.impl.NewCursor.loc20_47.1 => constants.%IntRange.as.Iterate.impl.NewCursor.c95819.1
// CHECK:STDOUT:   %.loc44 => constants.%.350
// CHECK:STDOUT:   %IntRange.as.Iterate.impl.Next.type.loc44_77.1 => constants.%IntRange.as.Iterate.impl.Next.type.a05ce8.1
// CHECK:STDOUT:   %IntRange.as.Iterate.impl.Next.loc44_77.1 => constants.%IntRange.as.Iterate.impl.Next.d8655a.1
// CHECK:STDOUT:   %.1 => constants.%.6a4
// CHECK:STDOUT:   %IntRange.as.Iterate.impl.NewCursor.type.loc20_47.2 => constants.%IntRange.as.Iterate.impl.NewCursor.type.c510a4.2
// CHECK:STDOUT:   %IntRange.as.Iterate.impl.NewCursor.loc20_47.2 => constants.%IntRange.as.Iterate.impl.NewCursor.c95819.2
// CHECK:STDOUT:   %Iterate.lookup_impl_witness => constants.%Iterate.lookup_impl_witness.5aa
// CHECK:STDOUT:   %impl.elem0.1 => constants.%impl.elem0.18c
// CHECK:STDOUT:   %.2 => constants.%.535
// CHECK:STDOUT:   %IntRange.as.Iterate.impl.Next.type.loc44_77.2 => constants.%IntRange.as.Iterate.impl.Next.type.a05ce8.2
// CHECK:STDOUT:   %IntRange.as.Iterate.impl.Next.loc44_77.2 => constants.%IntRange.as.Iterate.impl.Next.d8655a.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @IntRange.as.Iterate.impl.NewCursor.loc20_47.1(constants.%N.586) {
// CHECK:STDOUT:   %N => constants.%N.586
// CHECK:STDOUT:   %IntRange => constants.%IntRange.48b
// CHECK:STDOUT:   %pattern_type.loc20_18 => constants.%pattern_type.6de
// CHECK:STDOUT:   %Int.loc20_45.1 => constants.%Int.b50e37.1
// CHECK:STDOUT:   %pattern_type.loc20_32 => constants.%pattern_type.60f8e7.1
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc20_22 => constants.%require_complete.b4c
// CHECK:STDOUT:   %IntRange.elem => constants.%IntRange.elem.baf
// CHECK:STDOUT:   %require_complete.loc20_60 => constants.%require_complete.e66303.2
// CHECK:STDOUT:   %.loc20_60.4 => constants.%.1a2
// CHECK:STDOUT:   %Copy.lookup_impl_witness => constants.%Copy.lookup_impl_witness.035
// CHECK:STDOUT:   %Copy.facet => constants.%Copy.facet.d3d
// CHECK:STDOUT:   %.loc20_60.5 => constants.%.20c
// CHECK:STDOUT:   %impl.elem0.loc20_60.2 => constants.%impl.elem0.625
// CHECK:STDOUT:   %specific_impl_fn.loc20_60.2 => constants.%specific_impl_fn.c88
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @IntRange.as.Iterate.impl.Next.loc44_77.1(constants.%N.586) {
// CHECK:STDOUT:   %N => constants.%N.586
// CHECK:STDOUT:   %IntRange => constants.%IntRange.48b
// CHECK:STDOUT:   %pattern_type.loc44_13 => constants.%pattern_type.6de
// CHECK:STDOUT:   %Int.loc44_43.1 => constants.%Int.b50e37.1
// CHECK:STDOUT:   %ptr.loc44_44.1 => constants.%ptr.1cf
// CHECK:STDOUT:   %pattern_type.loc44_25 => constants.%pattern_type.054
// CHECK:STDOUT:   %Copy.lookup_impl_witness => constants.%Copy.lookup_impl_witness.035
// CHECK:STDOUT:   %Copy.facet => constants.%Copy.facet.d3d
// CHECK:STDOUT:   %.loc44_75.1 => constants.%.350
// CHECK:STDOUT:   %OptionalStorage.lookup_impl_witness => constants.%OptionalStorage.lookup_impl_witness.8c6
// CHECK:STDOUT:   %OptionalStorage.facet.loc44_75.1 => constants.%OptionalStorage.facet.ea4
// CHECK:STDOUT:   %Optional.loc44_75.1 => constants.%Optional.23f
// CHECK:STDOUT:   %pattern_type.loc44_47 => constants.%pattern_type.9f4
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc44_47 => constants.%require_complete.b49
// CHECK:STDOUT:   %require_complete.loc44_17 => constants.%require_complete.b4c
// CHECK:STDOUT:   %require_complete.loc44_31 => constants.%require_complete.6ca2
// CHECK:STDOUT:   %require_complete.loc45 => constants.%require_complete.e66303.2
// CHECK:STDOUT:   %pattern_type.loc45 => constants.%pattern_type.60f8e7.1
// CHECK:STDOUT:   %.loc45_32.4 => constants.%.1a2
// CHECK:STDOUT:   %.loc45_32.5 => constants.%.20c
// CHECK:STDOUT:   %impl.elem0.loc45_32.2 => constants.%impl.elem0.625
// CHECK:STDOUT:   %specific_impl_fn.loc45_32.2 => constants.%specific_impl_fn.c88
// CHECK:STDOUT:   %IntRange.elem => constants.%IntRange.elem.baf
// CHECK:STDOUT:   %OrderedWith.type.loc46_17.2 => constants.%OrderedWith.type.393
// CHECK:STDOUT:   %require_complete.loc46 => constants.%require_complete.4f7
// CHECK:STDOUT:   %OrderedWith.assoc_type => constants.%OrderedWith.assoc_type.72a
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.3f6
// CHECK:STDOUT:   %.loc46_17.2 => constants.%.e09
// CHECK:STDOUT:   %OrderedWith.impl_witness => constants.%OrderedWith.impl_witness.586
// CHECK:STDOUT:   %OrderedWith.Less.type => constants.%OrderedWith.Less.type.c6f
// CHECK:STDOUT:   %OrderedWith.facet => constants.%OrderedWith.facet
// CHECK:STDOUT:   %.loc46_17.3 => constants.%.db5
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Less.type => constants.%Int.as.OrderedWith.impl.Less.type.f52
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Less => constants.%Int.as.OrderedWith.impl.Less.af3
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Less.specific_fn => constants.%Int.as.OrderedWith.impl.Less.specific_fn.0ff
// CHECK:STDOUT:   %.loc47_9.2 => constants.%.4ff
// CHECK:STDOUT:   %Inc.lookup_impl_witness => constants.%Inc.lookup_impl_witness
// CHECK:STDOUT:   %Inc.facet => constants.%Inc.facet
// CHECK:STDOUT:   %.loc47_9.3 => constants.%.080
// CHECK:STDOUT:   %impl.elem0.loc47_9.2 => constants.%impl.elem0.923
// CHECK:STDOUT:   %specific_impl_fn.loc47_9.2 => constants.%specific_impl_fn.173
// CHECK:STDOUT:   %Optional.Some.type => constants.%Optional.Some.type.74c
// CHECK:STDOUT:   %Optional.Some => constants.%Optional.Some.37c
// CHECK:STDOUT:   %Optional.Some.specific_fn.loc48_42.2 => constants.%Optional.Some.specific_fn.90e
// CHECK:STDOUT:   %facet_value => constants.%facet_value.4a9
// CHECK:STDOUT:   %.loc45_7.1 => constants.%.4d6
// CHECK:STDOUT:   %Destroy.impl_witness => constants.%Destroy.impl_witness.b47
// CHECK:STDOUT:   %Destroy.facet => constants.%Destroy.facet.11d
// CHECK:STDOUT:   %.loc45_7.2 => constants.%.a64
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.type => constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.type.353
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op => constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.3be
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn => constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.e87
// CHECK:STDOUT:   %Optional.None.type => constants.%Optional.None.type.a39
// CHECK:STDOUT:   %Optional.None => constants.%Optional.None.b0b
// CHECK:STDOUT:   %Optional.None.specific_fn.loc50_42.2 => constants.%Optional.None.specific_fn.94d
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @IntRange.as.Iterate.impl.NewCursor.loc20_47.2(constants.%N.586) {
// CHECK:STDOUT:   %N => constants.%N.586
// CHECK:STDOUT:   %IntRange => constants.%IntRange.48b
// CHECK:STDOUT:   %pattern_type.1 => constants.%pattern_type.6de
// CHECK:STDOUT:   %Iterate.lookup_impl_witness => constants.%Iterate.lookup_impl_witness.5aa
// CHECK:STDOUT:   %impl.elem1 => constants.%impl.elem1.d3c
// CHECK:STDOUT:   %pattern_type.2 => constants.%pattern_type.605
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @IntRange.as.Iterate.impl.Next.loc44_77.2(constants.%N.586) {
// CHECK:STDOUT:   %N => constants.%N.586
// CHECK:STDOUT:   %IntRange => constants.%IntRange.48b
// CHECK:STDOUT:   %pattern_type.1 => constants.%pattern_type.6de
// CHECK:STDOUT:   %Iterate.lookup_impl_witness => constants.%Iterate.lookup_impl_witness.5aa
// CHECK:STDOUT:   %impl.elem1 => constants.%impl.elem1.d3c
// CHECK:STDOUT:   %ptr.1 => constants.%ptr.ccc
// CHECK:STDOUT:   %pattern_type.2 => constants.%pattern_type.852
// CHECK:STDOUT:   %impl.elem0 => constants.%impl.elem0.18c
// CHECK:STDOUT:   %as_type => constants.%as_type.349
// CHECK:STDOUT:   %OptionalStorage.lookup_impl_witness.1 => constants.%OptionalStorage.lookup_impl_witness.bc5
// CHECK:STDOUT:   %OptionalStorage.facet.1 => constants.%OptionalStorage.facet.201
// CHECK:STDOUT:   %Optional.1 => constants.%Optional.f2d
// CHECK:STDOUT:   %pattern_type.3 => constants.%pattern_type.f5b
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @IntRange(constants.%int_32) {
// CHECK:STDOUT:   %N.loc4_16.1 => constants.%int_32
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %IntRange.Make.type => constants.%IntRange.Make.type.cef
// CHECK:STDOUT:   %IntRange.Make => constants.%IntRange.Make.0dc
// CHECK:STDOUT:   %.loc9 => constants.%.1e9
// CHECK:STDOUT:   %Int.loc55_32.2 => constants.%i32
// CHECK:STDOUT:   %require_complete => constants.%complete_type.f8a
// CHECK:STDOUT:   %IntRange => constants.%IntRange.365
// CHECK:STDOUT:   %IntRange.elem => constants.%IntRange.elem.e33
// CHECK:STDOUT:   %struct_type.start.end => constants.%struct_type.start.end.d0a
// CHECK:STDOUT:   %complete_type.loc57_1.2 => constants.%complete_type.c45
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @IntRange.Make(constants.%int_32) {
// CHECK:STDOUT:   %N => constants.%int_32
// CHECK:STDOUT:   %Int.loc5_28.1 => constants.%i32
// CHECK:STDOUT:   %pattern_type.loc5_11 => constants.%pattern_type.7ce
// CHECK:STDOUT:   %IntRange => constants.%IntRange.365
// CHECK:STDOUT:   %pattern_type.loc5_49 => constants.%pattern_type.d16
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc5_49 => constants.%complete_type.c45
// CHECK:STDOUT:   %require_complete.loc5_16 => constants.%complete_type.f8a
// CHECK:STDOUT:   %struct_type.start.end => constants.%struct_type.start.end.d0a
// CHECK:STDOUT:   %.loc6_22.2 => constants.%.1e9
// CHECK:STDOUT:   %Copy.lookup_impl_witness => constants.%Copy.impl_witness.fb7
// CHECK:STDOUT:   %Copy.facet => constants.%Copy.facet.c50
// CHECK:STDOUT:   %.loc6_22.3 => constants.%.65f
// CHECK:STDOUT:   %impl.elem0.loc6_22.2 => constants.%Int.as.Copy.impl.Op.dfd
// CHECK:STDOUT:   %specific_impl_fn.loc6_22.2 => constants.%Int.as.Copy.impl.Op.specific_fn
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- trivial.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Read.type: type = fn_type @Read [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Read: %Read.type = struct_value () [concrete]
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %IntLiteral.type: type = fn_type @IntLiteral [concrete]
// CHECK:STDOUT:   %IntLiteral: %IntLiteral.type = struct_value () [concrete]
// CHECK:STDOUT:   %y: Core.IntLiteral = symbolic_binding y, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.dc0: type = pattern_type Core.IntLiteral [concrete]
// CHECK:STDOUT:   %int_43: Core.IntLiteral = int_value 43 [concrete]
// CHECK:STDOUT:   %IntRange.type: type = generic_class_type @IntRange [concrete]
// CHECK:STDOUT:   %IntRange.generic: %IntRange.type = struct_value () [concrete]
// CHECK:STDOUT:   %N: Core.IntLiteral = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %Int.f4f02c.1: type = class_type @Int, @Int(%N) [symbolic]
// CHECK:STDOUT:   %struct_type.start.end.fb5: type = struct_type {.start: %Int.f4f02c.1, .end: %Int.f4f02c.1} [symbolic]
// CHECK:STDOUT:   %complete_type.655: <witness> = complete_type_witness %struct_type.start.end.fb5 [symbolic]
// CHECK:STDOUT:   %IntRange.48b: type = class_type @IntRange, @IntRange(%N) [symbolic]
// CHECK:STDOUT:   %IntRange.elem.4f4: type = unbound_element_type %IntRange.48b, %Int.f4f02c.1 [symbolic]
// CHECK:STDOUT:   %require_complete.001e59.1: <witness> = require_complete_type %Int.f4f02c.1 [symbolic]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %pattern_type.e3e33e.1: type = pattern_type %Int.f4f02c.1 [symbolic]
// CHECK:STDOUT:   %.6a2: require_specific_def_type = require_specific_def @Int.as.Copy.impl(%N) [symbolic]
// CHECK:STDOUT:   %IntRange.Make.type.121: type = fn_type @IntRange.Make, @IntRange(%N) [symbolic]
// CHECK:STDOUT:   %IntRange.Make.14e: %IntRange.Make.type.121 = struct_value () [symbolic]
// CHECK:STDOUT:   %pattern_type.6de: type = pattern_type %IntRange.48b [symbolic]
// CHECK:STDOUT:   %Copy.lookup_impl_witness.d1a: <witness> = lookup_impl_witness %Int.f4f02c.1, @Copy [symbolic]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %Int.f4f02c.1, (%Copy.lookup_impl_witness.d1a) [symbolic]
// CHECK:STDOUT:   %Copy.Op.type: type = fn_type @Copy.Op [concrete]
// CHECK:STDOUT:   %.b90: type = fn_type_with_self_type %Copy.Op.type, %Copy.facet [symbolic]
// CHECK:STDOUT:   %impl.elem0.305: %.b90 = impl_witness_access %Copy.lookup_impl_witness.d1a, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.3a3: <specific function> = specific_impl_function %impl.elem0.305, @Copy.Op(%Copy.facet) [symbolic]
// CHECK:STDOUT:   %require_complete.b4c: <witness> = require_complete_type %IntRange.48b [symbolic]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %IntRange.365: type = class_type @IntRange, @IntRange(%int_32) [concrete]
// CHECK:STDOUT:   %IntRange.Make.type.cef: type = fn_type @IntRange.Make, @IntRange(%int_32) [concrete]
// CHECK:STDOUT:   %IntRange.Make.0dc: %IntRange.Make.type.cef = struct_value () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %i32.builtin: type = int_type signed, %int_32 [concrete]
// CHECK:STDOUT:   %complete_type.f8a: <witness> = complete_type_witness %i32.builtin [concrete]
// CHECK:STDOUT:   %.f57: require_specific_def_type = require_specific_def @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %IntRange.elem.f21: type = unbound_element_type %IntRange.365, %i32 [concrete]
// CHECK:STDOUT:   %struct_type.start.end.0fe: type = struct_type {.start: %i32, .end: %i32} [concrete]
// CHECK:STDOUT:   %complete_type.a43: <witness> = complete_type_witness %struct_type.start.end.0fe [concrete]
// CHECK:STDOUT:   %pattern_type.d16: type = pattern_type %IntRange.365 [concrete]
// CHECK:STDOUT:   %Range.type: type = fn_type @Range [concrete]
// CHECK:STDOUT:   %Range: %Range.type = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.cc7: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.cc7 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.bd9: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.Convert.type.6da: type = fn_type @ImplicitAs.Convert, @ImplicitAs(%i32) [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.365: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.8cf: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.365 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.09b: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.04e, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.069: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.13e: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.069 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.bd9 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.09b) [concrete]
// CHECK:STDOUT:   %.1a5: type = fn_type_with_self_type %ImplicitAs.Convert.type.6da, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %y, %Core.IntLiteral.as.ImplicitAs.impl.Convert.13e [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.13e, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %y, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %i32 = call %bound_method(%y) [symbolic]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %type_where: type = facet_type <type where .Self impls <CanDestroy>> [concrete]
// CHECK:STDOUT:   %facet_value: %type_where = facet_value %IntRange.365, () [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.type.fef: type = fn_type @DestroyT.binding.as_type.as.Destroy.impl.Op, @DestroyT.binding.as_type.as.Destroy.impl(%facet_value) [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.875: %DestroyT.binding.as_type.as.Destroy.impl.Op.type.fef = struct_value () [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn: <specific function> = specific_function %DestroyT.binding.as_type.as.Destroy.impl.Op.875, @DestroyT.binding.as_type.as.Destroy.impl.Op(%facet_value) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.IntRange: %IntRange.type = import_ref Main//lib, IntRange, loaded [concrete = constants.%IntRange.generic]
// CHECK:STDOUT:   %Main.Range: %Range.type = import_ref Main//lib, Range, loaded [concrete = constants.%Range]
// CHECK:STDOUT:   %Core.ece: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .IntLiteral = %Core.IntLiteral
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.IntLiteral: %IntLiteral.type = import_ref Core//prelude/types/int_literal, IntLiteral, loaded [concrete = constants.%IntLiteral]
// CHECK:STDOUT:   %Main.import_ref.40af26.1: Core.IntLiteral = import_ref Main//lib, loc4_16, loaded [symbolic = @IntRange.%N (constants.%N)]
// CHECK:STDOUT:   %Main.import_ref.80d: <witness> = import_ref Main//lib, loc57_1, loaded [symbolic = @IntRange.%complete_type (constants.%complete_type.655)]
// CHECK:STDOUT:   %Main.import_ref.b1b = import_ref Main//lib, inst{{[0-9A-F]+}} [no loc], unloaded
// CHECK:STDOUT:   %Main.import_ref.765 = import_ref Main//lib, loc5_57, unloaded
// CHECK:STDOUT:   %Main.import_ref.e5d = import_ref Main//lib, loc55_20, unloaded
// CHECK:STDOUT:   %Main.import_ref.997 = import_ref Main//lib, loc56_18, unloaded
// CHECK:STDOUT:   %Main.import_ref.40af26.2: Core.IntLiteral = import_ref Main//lib, loc4_16, loaded [symbolic = @IntRange.%N (constants.%N)]
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.cc7 = import_ref Core//prelude/operators/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:   %Core.import_ref.7d5: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.365) = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.8cf)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.04e = impl_witness_table (%Core.import_ref.7d5), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .IntRange = imports.%Main.IntRange
// CHECK:STDOUT:     .Range = imports.%Main.Range
// CHECK:STDOUT:     .Core = imports.%Core.ece
// CHECK:STDOUT:     .Read = %Read.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %default.import = import <none>
// CHECK:STDOUT:   %Read.decl: %Read.type = fn_decl @Read [concrete = constants.%Read] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @IntRange(imports.%Main.import_ref.40af26.2: Core.IntLiteral) [from "lib.carbon"] {
// CHECK:STDOUT:   %N: Core.IntLiteral = symbolic_binding N, 0 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %IntRange.Make.type: type = fn_type @IntRange.Make, @IntRange(%N) [symbolic = %IntRange.Make.type (constants.%IntRange.Make.type.121)]
// CHECK:STDOUT:   %IntRange.Make: @IntRange.%IntRange.Make.type (%IntRange.Make.type.121) = struct_value () [symbolic = %IntRange.Make (constants.%IntRange.Make.14e)]
// CHECK:STDOUT:   %.1: require_specific_def_type = require_specific_def @Int.as.Copy.impl(%N) [symbolic = %.1 (constants.%.6a2)]
// CHECK:STDOUT:   %Int: type = class_type @Int, @Int(%N) [symbolic = %Int (constants.%Int.f4f02c.1)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %Int [symbolic = %require_complete (constants.%require_complete.001e59.1)]
// CHECK:STDOUT:   %IntRange: type = class_type @IntRange, @IntRange(%N) [symbolic = %IntRange (constants.%IntRange.48b)]
// CHECK:STDOUT:   %IntRange.elem: type = unbound_element_type %IntRange, %Int [symbolic = %IntRange.elem (constants.%IntRange.elem.4f4)]
// CHECK:STDOUT:   %struct_type.start.end: type = struct_type {.start: @IntRange.%Int (%Int.f4f02c.1), .end: @IntRange.%Int (%Int.f4f02c.1)} [symbolic = %struct_type.start.end (constants.%struct_type.start.end.fb5)]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.start.end [symbolic = %complete_type (constants.%complete_type.655)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     complete_type_witness = imports.%Main.import_ref.80d
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%Main.import_ref.b1b
// CHECK:STDOUT:     .Make = imports.%Main.import_ref.765
// CHECK:STDOUT:     .start [private] = imports.%Main.import_ref.e5d
// CHECK:STDOUT:     .end [private] = imports.%Main.import_ref.997
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Read() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %y.patt: %pattern_type.dc0 = symbolic_binding_pattern y, 0 [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int_43: Core.IntLiteral = int_value 43 [concrete = constants.%int_43]
// CHECK:STDOUT:   %.loc5_27.1: type = splice_block %.loc5_27.3 [concrete = Core.IntLiteral] {
// CHECK:STDOUT:     %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %Core.ref: <namespace> = name_ref Core, imports.%Core.ece [concrete = imports.%Core.ece]
// CHECK:STDOUT:     %IntLiteral.ref: %IntLiteral.type = name_ref IntLiteral, imports.%Core.IntLiteral [concrete = constants.%IntLiteral]
// CHECK:STDOUT:     %IntLiteral.call: init type = call %IntLiteral.ref() [concrete = Core.IntLiteral]
// CHECK:STDOUT:     %.loc5_27.2: type = value_of_initializer %IntLiteral.call [concrete = Core.IntLiteral]
// CHECK:STDOUT:     %.loc5_27.3: type = converted %IntLiteral.call, %.loc5_27.2 [concrete = Core.IntLiteral]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %y: Core.IntLiteral = symbolic_binding y, 0, %int_43 [symbolic = constants.%y]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %pattern_type.d16 = ref_binding_pattern x [concrete]
// CHECK:STDOUT:     %x.var_patt: %pattern_type.d16 = var_pattern %x.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %x.var: ref %IntRange.365 = var %x.var_patt
// CHECK:STDOUT:   %Range.ref: %Range.type = name_ref Range, imports.%Main.Range [concrete = constants.%Range]
// CHECK:STDOUT:   %y.ref: Core.IntLiteral = name_ref y, %y [symbolic = constants.%y]
// CHECK:STDOUT:   %.loc6_3: ref %IntRange.365 = splice_block %x.var {}
// CHECK:STDOUT:   %impl.elem0: %.1a5 = impl_witness_access constants.%ImplicitAs.impl_witness.09b, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.13e]
// CHECK:STDOUT:   %bound_method.loc6_31.1: <bound method> = bound_method %y.ref, %impl.elem0 [symbolic = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc6_31.2: <bound method> = bound_method %y.ref, %specific_fn [symbolic = constants.%bound_method]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %i32 = call %bound_method.loc6_31.2(%y.ref) [symbolic = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.call]
// CHECK:STDOUT:   %.loc6_31.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [symbolic = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.call]
// CHECK:STDOUT:   %.loc6_31.2: %i32 = converted %y.ref, %.loc6_31.1 [symbolic = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.call]
// CHECK:STDOUT:   %Range.call: init %IntRange.365 = call %Range.ref(%.loc6_31.2) to %.loc6_3
// CHECK:STDOUT:   assign %x.var, %Range.call
// CHECK:STDOUT:   %.loc6_21: type = splice_block %IntRange [concrete = constants.%IntRange.365] {
// CHECK:STDOUT:     %IntRange.ref: %IntRange.type = name_ref IntRange, imports.%Main.IntRange [concrete = constants.%IntRange.generic]
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %IntRange: type = class_type @IntRange, @IntRange(constants.%int_32) [concrete = constants.%IntRange.365]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %x: ref %IntRange.365 = ref_binding x, %x.var
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.bound: <bound method> = bound_method %x.var, constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.875
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn: <specific function> = specific_function constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.875, @DestroyT.binding.as_type.as.Destroy.impl.Op(constants.%facet_value) [concrete = constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc6_3: <bound method> = bound_method %x.var, %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.call: init %empty_tuple.type = call %bound_method.loc6_3(%x.var)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @IntRange.Make(imports.%Main.import_ref.40af26.1: Core.IntLiteral) [from "lib.carbon"] {
// CHECK:STDOUT:   %N: Core.IntLiteral = symbolic_binding N, 0 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:   %Int: type = class_type @Int, @Int(%N) [symbolic = %Int (constants.%Int.f4f02c.1)]
// CHECK:STDOUT:   %pattern_type.1: type = pattern_type %Int [symbolic = %pattern_type.1 (constants.%pattern_type.e3e33e.1)]
// CHECK:STDOUT:   %IntRange: type = class_type @IntRange, @IntRange(%N) [symbolic = %IntRange (constants.%IntRange.48b)]
// CHECK:STDOUT:   %pattern_type.2: type = pattern_type %IntRange [symbolic = %pattern_type.2 (constants.%pattern_type.6de)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.1: <witness> = require_complete_type %IntRange [symbolic = %require_complete.1 (constants.%require_complete.b4c)]
// CHECK:STDOUT:   %require_complete.2: <witness> = require_complete_type %Int [symbolic = %require_complete.2 (constants.%require_complete.001e59.1)]
// CHECK:STDOUT:   %struct_type.start.end: type = struct_type {.start: @IntRange.Make.%Int (%Int.f4f02c.1), .end: @IntRange.Make.%Int (%Int.f4f02c.1)} [symbolic = %struct_type.start.end (constants.%struct_type.start.end.fb5)]
// CHECK:STDOUT:   %.1: require_specific_def_type = require_specific_def @Int.as.Copy.impl(%N) [symbolic = %.1 (constants.%.6a2)]
// CHECK:STDOUT:   %Copy.lookup_impl_witness: <witness> = lookup_impl_witness %Int, @Copy [symbolic = %Copy.lookup_impl_witness (constants.%Copy.lookup_impl_witness.d1a)]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %Int, (%Copy.lookup_impl_witness) [symbolic = %Copy.facet (constants.%Copy.facet)]
// CHECK:STDOUT:   %.2: type = fn_type_with_self_type constants.%Copy.Op.type, %Copy.facet [symbolic = %.2 (constants.%.b90)]
// CHECK:STDOUT:   %impl.elem0: @IntRange.Make.%.2 (%.b90) = impl_witness_access %Copy.lookup_impl_witness, element0 [symbolic = %impl.elem0 (constants.%impl.elem0.305)]
// CHECK:STDOUT:   %specific_impl_fn: <specific function> = specific_impl_function %impl.elem0, @Copy.Op(%Copy.facet) [symbolic = %specific_impl_fn (constants.%specific_impl_fn.3a3)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Range [from "lib.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: specific @IntRange(constants.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %IntRange.Make.type => constants.%IntRange.Make.type.121
// CHECK:STDOUT:   %IntRange.Make => constants.%IntRange.Make.14e
// CHECK:STDOUT:   %.1 => constants.%.6a2
// CHECK:STDOUT:   %Int => constants.%Int.f4f02c.1
// CHECK:STDOUT:   %require_complete => constants.%require_complete.001e59.1
// CHECK:STDOUT:   %IntRange => constants.%IntRange.48b
// CHECK:STDOUT:   %IntRange.elem => constants.%IntRange.elem.4f4
// CHECK:STDOUT:   %struct_type.start.end => constants.%struct_type.start.end.fb5
// CHECK:STDOUT:   %complete_type => constants.%complete_type.655
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @IntRange.Make(constants.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %Int => constants.%Int.f4f02c.1
// CHECK:STDOUT:   %pattern_type.1 => constants.%pattern_type.e3e33e.1
// CHECK:STDOUT:   %IntRange => constants.%IntRange.48b
// CHECK:STDOUT:   %pattern_type.2 => constants.%pattern_type.6de
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @IntRange(constants.%int_32) {
// CHECK:STDOUT:   %N => constants.%int_32
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %IntRange.Make.type => constants.%IntRange.Make.type.cef
// CHECK:STDOUT:   %IntRange.Make => constants.%IntRange.Make.0dc
// CHECK:STDOUT:   %.1 => constants.%.f57
// CHECK:STDOUT:   %Int => constants.%i32
// CHECK:STDOUT:   %require_complete => constants.%complete_type.f8a
// CHECK:STDOUT:   %IntRange => constants.%IntRange.365
// CHECK:STDOUT:   %IntRange.elem => constants.%IntRange.elem.f21
// CHECK:STDOUT:   %struct_type.start.end => constants.%struct_type.start.end.0fe
// CHECK:STDOUT:   %complete_type => constants.%complete_type.a43
// CHECK:STDOUT: }
// CHECK:STDOUT:
