// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/full.carbon
// TODO: Add ranges and switch to "--dump-sem-ir-ranges=only".
// EXTRA-ARGS: --dump-sem-ir-ranges=if-present
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/operators/overloaded/mul.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/operators/overloaded/mul.carbon
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

// This file was generated from binary_op.carbon.tmpl. Run make_tests.sh to regenerate.

package User;

class C {};

impl C as Core.MulWith(C) where .Result = C {
  fn Op[self: C](other: C) -> C {
    return {};
  }
}
impl C as Core.MulAssignWith(C) {
  fn Op[ref self: C](other: C) {}
}

fn TestOp(a: C, b: C) -> C {
  //@dump-sem-ir-begin
  return a * b;
  //@dump-sem-ir-end
}

fn TestAssign(a: C*, b: C) {
  //@dump-sem-ir-begin
  *a *= b;
  //@dump-sem-ir-end
}

// CHECK:STDOUT: --- mul.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %MulWith.type.c1e: type = facet_type <@MulWith, @MulWith(%C)> [concrete]
// CHECK:STDOUT:   %MulWith.Op.type.2c9: type = fn_type @MulWith.Op, @MulWith(%C) [concrete]
// CHECK:STDOUT:   %MulWith.impl_witness: <witness> = impl_witness file.%MulWith.impl_witness_table [concrete]
// CHECK:STDOUT:   %C.as.MulWith.impl.Op.type: type = fn_type @C.as.MulWith.impl.Op [concrete]
// CHECK:STDOUT:   %C.as.MulWith.impl.Op: %C.as.MulWith.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %MulWith.facet: %MulWith.type.c1e = facet_value %C, (%MulWith.impl_witness) [concrete]
// CHECK:STDOUT:   %MulAssignWith.type.c63: type = facet_type <@MulAssignWith, @MulAssignWith(%C)> [concrete]
// CHECK:STDOUT:   %MulAssignWith.Op.type.5c8: type = fn_type @MulAssignWith.Op, @MulAssignWith(%C) [concrete]
// CHECK:STDOUT:   %MulAssignWith.impl_witness: <witness> = impl_witness file.%MulAssignWith.impl_witness_table [concrete]
// CHECK:STDOUT:   %C.as.MulAssignWith.impl.Op.type: type = fn_type @C.as.MulAssignWith.impl.Op [concrete]
// CHECK:STDOUT:   %C.as.MulAssignWith.impl.Op: %C.as.MulAssignWith.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %MulAssignWith.facet: %MulAssignWith.type.c63 = facet_value %C, (%MulAssignWith.impl_witness) [concrete]
// CHECK:STDOUT:   %.46c: type = fn_type_with_self_type %MulWith.Op.type.2c9, %MulWith.facet [concrete]
// CHECK:STDOUT:   %ptr.019: type = ptr_type %C [concrete]
// CHECK:STDOUT:   %.78a: type = fn_type_with_self_type %MulAssignWith.Op.type.5c8, %MulAssignWith.facet [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestOp(%a.param: %C, %b.param: %C) -> %return.param: %C {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: %C = name_ref a, %a
// CHECK:STDOUT:   %b.ref: %C = name_ref b, %b
// CHECK:STDOUT:   %impl.elem1: %.46c = impl_witness_access constants.%MulWith.impl_witness, element1 [concrete = constants.%C.as.MulWith.impl.Op]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %a.ref, %impl.elem1
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %C.as.MulWith.impl.Op.call: init %C = call %bound_method(%a.ref, %b.ref) to %.loc577
// CHECK:STDOUT:   return %C.as.MulWith.impl.Op.call to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestAssign(%a.param: %ptr.019, %b.param: %C) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: %ptr.019 = name_ref a, %a
// CHECK:STDOUT:   %.loc585: ref %C = deref %a.ref
// CHECK:STDOUT:   %b.ref: %C = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0: %.78a = impl_witness_access constants.%MulAssignWith.impl_witness, element0 [concrete = constants.%C.as.MulAssignWith.impl.Op]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %.loc585, %impl.elem0
// CHECK:STDOUT:   %C.as.MulAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method(%.loc585, %b.ref)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
