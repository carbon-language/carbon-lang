// RUN: %clang_cc1 -triple mips-unknown-linux -o - -O1 -emit-llvm %s | FileCheck %s -check-prefix=ALL -check-prefix=O32
// RUN: %clang_cc1 -triple mipsel-unknown-linux -o - -O1 -emit-llvm %s | FileCheck %s -check-prefix=ALL -check-prefix=O32
// RUN: %clang_cc1 -triple mips64-unknown-linux -o - -O1 -emit-llvm  -target-abi n32 %s | FileCheck %s -check-prefix=ALL -check-prefix=N32 -check-prefix=NEW
// RUN: %clang_cc1 -triple mips64-unknown-linux -o - -O1 -emit-llvm  -target-abi n32 %s | FileCheck %s -check-prefix=ALL -check-prefix=N32 -check-prefix=NEW
// RUN: %clang_cc1 -triple mips64-unknown-linux -o - -O1 -emit-llvm %s | FileCheck %s -check-prefix=ALL -check-prefix=N64 -check-prefix=NEW
// RUN: %clang_cc1 -triple mips64el-unknown-linux -o - -O1 -emit-llvm %s | FileCheck %s -check-prefix=ALL -check-prefix=N64 -check-prefix=NEW

#include <stdarg.h>

typedef int v4i32 __attribute__ ((__vector_size__ (16)));

int test_i32(char *fmt, ...) {
  va_list va;

  va_start(va, fmt);
  int v = va_arg(va, int);
  va_end(va);

  return v;
}

// ALL-LABEL: define i32 @test_i32(i8*{{.*}} %fmt, ...)
//
// O32:   %va = alloca i8*, align [[PTRALIGN:4]]
// N32:   %va = alloca i8*, align [[PTRALIGN:4]]
// N64:   %va = alloca i8*, align [[PTRALIGN:8]]
//
// ALL:   [[VA1:%.+]] = bitcast i8** %va to i8*
// ALL:   call void @llvm.va_start(i8* [[VA1]])
//
// O32:   [[TMP0:%.+]] = bitcast i8** %va to i32**
// O32:   [[AP_CUR:%.+]] = load i32*, i32** [[TMP0]], align [[PTRALIGN]]
// NEW:   [[TMP0:%.+]] = bitcast i8** %va to i64**
// NEW:   [[AP_CUR:%.+]] = load i64*, i64** [[TMP0]], align [[PTRALIGN]]
//
// O32:   [[AP_NEXT:%.+]] = getelementptr i32, i32* [[AP_CUR]], i32 1
// NEW:   [[AP_NEXT:%.+]] = getelementptr i64, i64* [[AP_CUR]], {{i32|i64}} 1
//
// O32:   store i32* [[AP_NEXT]], i32** [[TMP0]], align [[PTRALIGN]]
// NEW:   store i64* [[AP_NEXT]], i64** [[TMP0]], align [[PTRALIGN]]
//
// O32:   [[ARG1:%.+]] = load i32, i32* [[AP_CUR]], align 4
// NEW:   [[TMP2:%.+]] = load i64, i64* [[AP_CUR]], align 8
// NEW:   [[ARG1:%.+]] = trunc i64 [[TMP2]] to i32
//
// ALL:   call void @llvm.va_end(i8* [[VA1]])
// ALL:   ret i32 [[ARG1]]
// ALL: }

int test_i32_2args(char *fmt, ...) {
  va_list va;

  va_start(va, fmt);
  int v1 = va_arg(va, int);
  int v2 = va_arg(va, int);
  va_end(va);

  return v1 + v2;
}

// ALL-LABEL: define i32 @test_i32_2args(i8*{{.*}} %fmt, ...)
//
// ALL:   %va = alloca i8*, align [[PTRALIGN]]
// ALL:   [[VA1:%.+]] = bitcast i8** %va to i8*
// ALL:   call void @llvm.va_start(i8* [[VA1]])
//
// O32:   [[TMP0:%.+]] = bitcast i8** %va to i32**
// O32:   [[AP_CUR:%.+]] = load i32*, i32** [[TMP0]], align [[PTRALIGN]]
// NEW:   [[TMP0:%.+]] = bitcast i8** %va to i64**
// NEW:   [[AP_CUR:%.+]] = load i64*, i64** [[TMP0]], align [[PTRALIGN]]
//
// O32:   [[AP_NEXT1:%.+]] = getelementptr i32, i32* [[AP_CUR]], i32 1
// NEW:   [[AP_NEXT1:%.+]] = getelementptr i64, i64* [[AP_CUR]], [[INTPTR_T:i32|i64]] 1
//
// O32:   store i32* [[AP_NEXT1]], i32** [[TMP0]], align [[PTRALIGN]]
// FIXME: N32 optimised this store out. Why only for this ABI?
// N64:   store i64* [[AP_NEXT1]], i64** [[TMP0]], align [[PTRALIGN]]
//
// O32:   [[ARG1:%.+]] = load i32, i32* [[AP_CUR]], align 4
// NEW:   [[TMP3:%.+]] = load i64, i64* [[AP_CUR]], align 8
// NEW:   [[ARG1:%.+]] = trunc i64 [[TMP3]] to i32
//
// O32:   [[AP_NEXT2:%.+]] = getelementptr i32, i32* [[AP_CUR]], i32 2
// NEW:   [[AP_NEXT2:%.+]] = getelementptr i64, i64* [[AP_CUR]], [[INTPTR_T]] 2
//
// O32:   store i32* [[AP_NEXT2]], i32** [[TMP0]], align [[PTRALIGN]]
// NEW:   store i64* [[AP_NEXT2]], i64** [[TMP0]], align [[PTRALIGN]]
//
// O32:   [[ARG2:%.+]] = load i32, i32* [[AP_NEXT1]], align 4
// NEW:   [[TMP4:%.+]] = load i64, i64* [[AP_NEXT1]], align 8
// NEW:   [[ARG2:%.+]] = trunc i64 [[TMP4]] to i32
//
// ALL:   call void @llvm.va_end(i8* [[VA1]])
// ALL:   [[ADD:%.+]] = add nsw i32 [[ARG2]], [[ARG1]]
// ALL:   ret i32 [[ADD]]
// ALL: }

long long test_i64(char *fmt, ...) {
  va_list va;

  va_start(va, fmt);
  long long v = va_arg(va, long long);
  va_end(va);

  return v;
}

// ALL-LABEL: define i64 @test_i64(i8*{{.*}} %fmt, ...)
//
// ALL:   %va = alloca i8*, align [[PTRALIGN]]
// ALL:   [[VA1:%.+]] = bitcast i8** %va to i8*
// ALL:   call void @llvm.va_start(i8* [[VA1]])
//
// O32:   [[TMP0:%.+]] = bitcast i8** %va to i32*
// O32:   [[AP_CUR:%.+]] = load [[INTPTR_T:i32]], i32* [[TMP0]], align [[PTRALIGN]]
// NEW:   [[TMP0:%.+]] = bitcast i8** %va to i64**
// NEW:   [[AP_CUR:%.+]] = load i64*, i64** [[TMP0]], align [[PTRALIGN]]
//
// i64 is 8-byte aligned, while this is within O32's stack alignment there's no
// guarantee that the offset is still 8-byte aligned after earlier reads.
// O32:   [[PTR1:%.+]] = add i32 [[AP_CUR]], 7
// O32:   [[PTR2:%.+]] = and i32 [[PTR1]], -8
// O32:   [[PTR3:%.+]] = inttoptr [[INTPTR_T]] [[PTR2]] to i64*
// O32:   [[PTR4:%.+]] = inttoptr [[INTPTR_T]] [[PTR2]] to i8*
//
// O32:   [[AP_NEXT:%.+]] = getelementptr i8, i8* [[PTR4]], [[INTPTR_T]] 8
// NEW:   [[AP_NEXT:%.+]] = getelementptr i64, i64* [[AP_CUR]], [[INTPTR_T:i32|i64]] 1
//
// O32:   store i8* [[AP_NEXT]], i8** %va, align [[PTRALIGN]]
// NEW:   store i64* [[AP_NEXT]], i64** [[TMP0]], align [[PTRALIGN]]
//
// O32:   [[ARG1:%.+]] = load i64, i64* [[PTR3]], align 8
// NEW:   [[ARG1:%.+]] = load i64, i64* [[AP_CUR]], align 8
//
// ALL:   call void @llvm.va_end(i8* [[VA1]])
// ALL:   ret i64 [[ARG1]]
// ALL: }

char *test_ptr(char *fmt, ...) {
  va_list va;

  va_start(va, fmt);
  char *v = va_arg(va, char *);
  va_end(va);

  return v;
}

// ALL-LABEL: define i8* @test_ptr(i8*{{.*}} %fmt, ...)
//
// O32:   %va = alloca i8*, align [[PTRALIGN:4]]
// N32:   %va = alloca i8*, align [[PTRALIGN:4]]
// N64:   %va = alloca i8*, align [[PTRALIGN:8]]
//
// ALL:   [[VA1:%.+]] = bitcast i8** %va to i8*
// ALL:   call void @llvm.va_start(i8* [[VA1]])
//
// O32:   [[TMP0:%.+]] = bitcast i8** %va to i8***
// O32:   [[AP_CUR:%.+]] = load i8**, i8*** [[TMP0]], align [[PTRALIGN]]
// N32 differs because the vararg is not a N32 pointer. It's been promoted to 64-bit.
// N32:   [[TMP0:%.+]] = bitcast i8** %va to i64**
// N32:   [[AP_CUR:%.+]] = load i64*, i64** [[TMP0]], align [[PTRALIGN]]
// N64:   [[TMP0:%.+]] = bitcast i8** %va to i8***
// N64:   [[AP_CUR:%.+]] = load i8**, i8*** [[TMP0]], align [[PTRALIGN]]
//
// O32:   [[AP_NEXT:%.+]] = getelementptr i8*, i8** [[AP_CUR]], i32 1
// N32 differs because the vararg is not a N32 pointer. It's been promoted to 64-bit.
// N32:   [[AP_NEXT:%.+]] = getelementptr i64, i64* [[AP_CUR]], {{i32|i64}} 1
// N64:   [[AP_NEXT:%.+]] = getelementptr i8*, i8** [[AP_CUR]], {{i32|i64}} 1
//
// O32:   store i8** [[AP_NEXT]], i8*** [[TMP0]], align [[PTRALIGN]]
// N32 differs because the vararg is not a N32 pointer. It's been promoted to 64-bit.
// N32:   store i64* [[AP_NEXT]], i64** [[TMP0]], align [[PTRALIGN]]
// N64:   store i8** [[AP_NEXT]], i8*** [[TMP0]], align [[PTRALIGN]]
//
// O32:   [[ARG1:%.+]] = load i8*, i8** [[AP_CUR]], align 4
// N32 differs because the vararg is not a N32 pointer. It's been promoted to
// 64-bit so we must truncate the excess and bitcast to a N32 pointer.
// N32:   [[TMP2:%.+]] = load i64, i64* [[AP_CUR]], align 8
// N32:   [[TMP3:%.+]] = trunc i64 [[TMP2]] to i32
// N32:   [[ARG1:%.+]] = inttoptr i32 [[TMP3]] to i8*
// N64:   [[ARG1:%.+]] = load i8*, i8** [[AP_CUR]], align 8
//
// ALL:   call void @llvm.va_end(i8* [[VA1]])
// ALL:   ret i8* [[ARG1]]
// ALL: }

int test_v4i32(char *fmt, ...) {
  va_list va;

  va_start(va, fmt);
  v4i32 v = va_arg(va, v4i32);
  va_end(va);

  return v[0];
}

// ALL-LABEL: define i32 @test_v4i32(i8*{{.*}} %fmt, ...)
//
// ALL:   %va = alloca i8*, align [[PTRALIGN]]
// ALL:   [[VA1:%.+]] = bitcast i8** %va to i8*
// ALL:   call void @llvm.va_start(i8* [[VA1]])
//
// O32:   [[TMP0:%.+]] = bitcast i8** %va to i32*
// N32:   [[TMP0:%.+]] = bitcast i8** %va to i32*
// N64:   [[TMP0:%.+]] = bitcast i8** %va to i64*
//
// O32:   [[PTR0:%.+]] = load [[INTPTR_T:i32]], i32* [[TMP0]], align [[PTRALIGN]]
// N32:   [[PTR0:%.+]] = load [[INTPTR_T:i32]], i32* [[TMP0]], align [[PTRALIGN]]
// N64:   [[PTR0:%.+]] = load [[INTPTR_T:i64]], i64* [[TMP0]], align [[PTRALIGN]]
//
// Vectors are 16-byte aligned, however the O32 ABI has a maximum alignment of
// 8-bytes since the base of the stack is 8-byte aligned.
// O32:   [[PTR1:%.+]] = add i32 [[PTR0]], 7
// O32:   [[PTR2:%.+]] = and i32 [[PTR1]], -8
//
// NEW:   [[PTR1:%.+]] = add [[INTPTR_T]] [[PTR0]], 15
// NEW:   [[PTR2:%.+]] = and [[INTPTR_T]] [[PTR1]], -16
//
// ALL:   [[PTR3:%.+]] = inttoptr [[INTPTR_T]] [[PTR2]] to <4 x i32>*
// ALL:   [[PTR4:%.+]] = inttoptr [[INTPTR_T]] [[PTR2]] to i8*
// ALL:   [[AP_NEXT:%.+]] = getelementptr i8, i8* [[PTR4]], [[INTPTR_T]] 16
// ALL:   store i8* [[AP_NEXT]], i8** %va, align [[PTRALIGN]]
// ALL:   [[PTR5:%.+]] = load <4 x i32>, <4 x i32>* [[PTR3]], align 16
// ALL:   call void @llvm.va_end(i8* [[VA1]])
// ALL:   [[VECEXT:%.+]] = extractelement <4 x i32> [[PTR5]], i32 0
// ALL:   ret i32 [[VECEXT]]
// ALL: }
