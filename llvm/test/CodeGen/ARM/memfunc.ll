; RUN: llc < %s -mtriple=armv7-apple-ios -disable-post-ra -o - | FileCheck %s --check-prefix=CHECK-IOS --check-prefix=CHECK
; RUN: llc < %s -mtriple=thumbv7m-none-macho -disable-post-ra -o - | FileCheck %s --check-prefix=CHECK-DARWIN --check-prefix=CHECK
; RUN: llc < %s -mtriple=arm-none-eabi -disable-post-ra -o - | FileCheck %s --check-prefix=CHECK-EABI --check-prefix=CHECK
; RUN: llc < %s -mtriple=arm-none-eabihf -disable-post-ra -o - | FileCheck %s --check-prefix=CHECK-EABI --check-prefix=CHECK

@from = common global [500 x i32] zeroinitializer, align 4
@to = common global [500 x i32] zeroinitializer, align 4

define void @f1() {
entry:
  ; CHECK-LABEL: f1

  ; CHECK-IOS: memmove
  ; CHECK-DARWIN: memmove
  ; CHECK-EABI: __aeabi_memmove
  call void @llvm.memmove.p0i8.p0i8.i32(i8* bitcast ([500 x i32]* @from to i8*), i8* bitcast ([500 x i32]* @to to i8*), i32 500, i32 0, i1 false)

  ; CHECK-IOS: memcpy
  ; CHECK-DARWIN: memcpy
  ; CHECK-EABI: __aeabi_memcpy
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* bitcast ([500 x i32]* @from to i8*), i8* bitcast ([500 x i32]* @to to i8*), i32 500, i32 0, i1 false)

  ; EABI memset swaps arguments
  ; CHECK-IOS: mov r1, #0
  ; CHECK-IOS: memset
  ; CHECK-DARWIN: movs r1, #0
  ; CHECK-DARWIN: memset
  ; CHECK-EABI: mov r2, #0
  ; CHECK-EABI: __aeabi_memset
  call void @llvm.memset.p0i8.i32(i8* bitcast ([500 x i32]* @from to i8*), i8 0, i32 500, i32 0, i1 false)
  unreachable
}

; Check that alloca arguments to memory intrinsics are automatically aligned if at least 8 bytes in size
define void @f2(i8* %dest, i32 %n) {
entry:
  ; CHECK-LABEL: f2

  ; IOS (ARMv7) should 8-byte align, others should 4-byte align
  ; CHECK-IOS: add r1, sp, #32
  ; CHECK-IOS: memmove
  ; CHECK-DARWIN: add r1, sp, #28
  ; CHECK-DARWIN: memmove
  ; CHECK-EABI: add r1, sp, #28
  ; CHECK-EABI: __aeabi_memmove
  %arr0 = alloca [9 x i8], align 1
  %0 = bitcast [9 x i8]* %arr0 to i8*
  call void @llvm.memmove.p0i8.p0i8.i32(i8* %dest, i8* %0, i32 %n, i32 0, i1 false)

  ; CHECK: add r1, sp, #16
  ; CHECK-IOS: memcpy
  ; CHECK-DARWIN: memcpy
  ; CHECK-EABI: __aeabi_memcpy
  %arr1 = alloca [9 x i8], align 1
  %1 = bitcast [9 x i8]* %arr1 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* %dest, i8* %1, i32 %n, i32 0, i1 false)

  ; CHECK-IOS: mov r0, sp
  ; CHECK-IOS: mov r1, #0
  ; CHECK-IOS: memset
  ; CHECK-DARINW: add r0, sp, #4
  ; CHECK-DARWIN: movs r1, #0
  ; CHECK-DARWIN: memset
  ; CHECK-EABI: add r0, sp, #4
  ; CHECK-EABI: mov r2, #0
  ; CHECK-EABI: __aeabi_memset
  %arr2 = alloca [9 x i8], align 1
  %2 = bitcast [9 x i8]* %arr2 to i8*
  call void @llvm.memset.p0i8.i32(i8* %2, i8 0, i32 %n, i32 0, i1 false)

  unreachable
}

; Check that alloca arguments are not aligned if less than 8 bytes in size
define void @f3(i8* %dest, i32 %n) {
entry:
  ; CHECK-LABEL: f3

  ; CHECK: {{add(.w)? r1, sp, #17|sub(.w)? r1, r7, #15}}
  ; CHECK-IOS: memmove
  ; CHECK-DARWIN: memmove
  ; CHECK-EABI: __aeabi_memmove
  %arr0 = alloca [7 x i8], align 1
  %0 = bitcast [7 x i8]* %arr0 to i8*
  call void @llvm.memmove.p0i8.p0i8.i32(i8* %dest, i8* %0, i32 %n, i32 0, i1 false)

  ; CHECK: {{add(.w)? r1, sp, #10}}
  ; CHECK-IOS: memcpy
  ; CHECK-DARWIN: memcpy
  ; CHECK-EABI: __aeabi_memcpy
  %arr1 = alloca [7 x i8], align 1
  %1 = bitcast [7 x i8]* %arr1 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* %dest, i8* %1, i32 %n, i32 0, i1 false)

  ; CHECK: {{add(.w)? r0, sp, #3}}
  ; CHECK-IOS: mov r1, #0
  ; CHECK-IOS: memset
  ; CHECK-DARWIN: movs r1, #0
  ; CHECK-DARWIN: memset
  ; CHECK-EABI: mov r2, #0
  ; CHECK-EABI: __aeabi_memset
  %arr2 = alloca [7 x i8], align 1
  %2 = bitcast [7 x i8]* %arr2 to i8*
  call void @llvm.memset.p0i8.i32(i8* %2, i8 0, i32 %n, i32 0, i1 false)

  unreachable
}

; Check that alloca arguments are not aligned if size+offset is less than 8 bytes
define void @f4(i8* %dest, i32 %n) {
entry:
  ; CHECK-LABEL: f4

  ; CHECK: {{add(.w)? r., sp, #23|sub(.w)? r., r7, #17}}
  ; CHECK-IOS: memmove
  ; CHECK-DARWIN: memmove
  ; CHECK-EABI: __aeabi_memmove
  %arr0 = alloca [9 x i8], align 1
  %0 = getelementptr inbounds [9 x i8], [9 x i8]* %arr0, i32 0, i32 4
  call void @llvm.memmove.p0i8.p0i8.i32(i8* %dest, i8* %0, i32 %n, i32 0, i1 false)

  ; CHECK: {{add(.w)? r., sp, #(10|14)}}
  ; CHECK-IOS: memcpy
  ; CHECK-DARWIN: memcpy
  ; CHECK-EABI: __aeabi_memcpy
  %arr1 = alloca [9 x i8], align 1
  %1 = getelementptr inbounds [9 x i8], [9 x i8]* %arr1, i32 0, i32 4
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* %dest, i8* %1, i32 %n, i32 0, i1 false)

  ; CHECK: {{add(.w)? r., sp, #(1|5)}}
  ; CHECK-IOS: mov r1, #0
  ; CHECK-IOS: memset
  ; CHECK-DARWIN: movs r1, #0
  ; CHECK-DARWIN: memset
  ; CHECK-EABI: mov r2, #0
  ; CHECK-EABI: __aeabi_memset
  %arr2 = alloca [9 x i8], align 1
  %2 = getelementptr inbounds [9 x i8], [9 x i8]* %arr2, i32 0, i32 4
  call void @llvm.memset.p0i8.i32(i8* %2, i8 0, i32 %n, i32 0, i1 false)

  unreachable
}

; Check that alloca arguments are not aligned if the offset is not a multiple of 4
define void @f5(i8* %dest, i32 %n) {
entry:
  ; CHECK-LABEL: f5

  ; CHECK: {{add(.w)? r., sp, #27|sub(.w)? r., r7, #21}}
  ; CHECK-IOS: memmove
  ; CHECK-DARWIN: memmove
  ; CHECK-EABI: __aeabi_memmove
  %arr0 = alloca [13 x i8], align 1
  %0 = getelementptr inbounds [13 x i8], [13 x i8]* %arr0, i32 0, i32 1
  call void @llvm.memmove.p0i8.p0i8.i32(i8* %dest, i8* %0, i32 %n, i32 0, i1 false)

  ; CHECK: {{add(.w)? r., sp, #(10|14)}}
  ; CHECK-IOS: memcpy
  ; CHECK-DARWIN: memcpy
  ; CHECK-EABI: __aeabi_memcpy
  %arr1 = alloca [13 x i8], align 1
  %1 = getelementptr inbounds [13 x i8], [13 x i8]* %arr1, i32 0, i32 1
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* %dest, i8* %1, i32 %n, i32 0, i1 false)

  ; CHECK: {{add(.w)? r., sp, #(1|5)}}
  ; CHECK-IOS: mov r1, #0
  ; CHECK-IOS: memset
  ; CHECK-DARWIN: movs r1, #0
  ; CHECK-DARWIN: memset
  ; CHECK-EABI: mov r2, #0
  ; CHECK-EABI: __aeabi_memset
  %arr2 = alloca [13 x i8], align 1
  %2 = getelementptr inbounds [13 x i8], [13 x i8]* %arr2, i32 0, i32 1
  call void @llvm.memset.p0i8.i32(i8* %2, i8 0, i32 %n, i32 0, i1 false)

  unreachable
}

; Check that alloca arguments are not aligned if the offset is unknown
define void @f6(i8* %dest, i32 %n, i32 %i) {
entry:
  ; CHECK-LABEL: f6

  ; CHECK: {{add(.w)? r., sp, #27|sub(.w)? r., r7, #25}}
  ; CHECK-IOS: memmove
  ; CHECK-DARWIN: memmove
  ; CHECK-EABI: __aeabi_memmove
  %arr0 = alloca [13 x i8], align 1
  %0 = getelementptr inbounds [13 x i8], [13 x i8]* %arr0, i32 0, i32 %i
  call void @llvm.memmove.p0i8.p0i8.i32(i8* %dest, i8* %0, i32 %n, i32 0, i1 false)

  ; CHECK: {{add(.w)? r., sp, #(10|14)}}
  ; CHECK-IOS: memcpy
  ; CHECK-DARWIN: memcpy
  ; CHECK-EABI: __aeabi_memcpy
  %arr1 = alloca [13 x i8], align 1
  %1 = getelementptr inbounds [13 x i8], [13 x i8]* %arr1, i32 0, i32 %i
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* %dest, i8* %1, i32 %n, i32 0, i1 false)

  ; CHECK: {{add(.w)? r., sp, #(1|5)}}
  ; CHECK-IOS: mov r1, #0
  ; CHECK-IOS: memset
  ; CHECK-DARWIN: movs r1, #0
  ; CHECK-DARWIN: memset
  ; CHECK-EABI: mov r2, #0
  ; CHECK-EABI: __aeabi_memset
  %arr2 = alloca [13 x i8], align 1
  %2 = getelementptr inbounds [13 x i8], [13 x i8]* %arr2, i32 0, i32 %i
  call void @llvm.memset.p0i8.i32(i8* %2, i8 0, i32 %n, i32 0, i1 false)

  unreachable
}

; Check that alloca arguments are not aligned if the GEP is not inbounds
define void @f7(i8* %dest, i32 %n) {
entry:
  ; CHECK-LABEL: f7

  ; CHECK: {{add(.w)? r., sp, #27|sub(.w)? r., r7, #21}}
  ; CHECK-IOS: memmove
  ; CHECK-DARWIN: memmove
  ; CHECK-EABI: __aeabi_memmove
  %arr0 = alloca [13 x i8], align 1
  %0 = getelementptr [13 x i8], [13 x i8]* %arr0, i32 0, i32 4
  call void @llvm.memmove.p0i8.p0i8.i32(i8* %dest, i8* %0, i32 %n, i32 0, i1 false)

  ; CHECK: {{add(.w)? r., sp, #(10|14)}}
  ; CHECK-IOS: memcpy
  ; CHECK-DARWIN: memcpy
  ; CHECK-EABI: __aeabi_memcpy
  %arr1 = alloca [13 x i8], align 1
  %1 = getelementptr [13 x i8], [13 x i8]* %arr1, i32 0, i32 4
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* %dest, i8* %1, i32 %n, i32 0, i1 false)

  ; CHECK: {{add(.w)? r., sp, #(1|5)}}
  ; CHECK-IOS: mov r1, #0
  ; CHECK-IOS: memset
  ; CHECK-DARWIN: movs r1, #0
  ; CHECK-DARWIN: memset
  ; CHECK-EABI: mov r2, #0
  ; CHECK-EABI: __aeabi_memset
  %arr2 = alloca [13 x i8], align 1
  %2 = getelementptr [13 x i8], [13 x i8]* %arr2, i32 0, i32 4
  call void @llvm.memset.p0i8.i32(i8* %2, i8 0, i32 %n, i32 0, i1 false)

  unreachable
}

; Check that alloca arguments are not aligned when the offset is past the end of the allocation
define void @f8(i8* %dest, i32 %n) {
entry:
  ; CHECK-LABEL: f8

  ; CHECK: {{add(.w)? r., sp, #27|sub(.w)? r., r7, #21}}
  ; CHECK-IOS: memmove
  ; CHECK-DARWIN: memmove
  ; CHECK-EABI: __aeabi_memmove
  %arr0 = alloca [13 x i8], align 1
  %0 = getelementptr inbounds [13 x i8], [13 x i8]* %arr0, i32 0, i32 16
  call void @llvm.memmove.p0i8.p0i8.i32(i8* %dest, i8* %0, i32 %n, i32 0, i1 false)

  ; CHECK: {{add(.w)? r., sp, #(10|14)}}
  ; CHECK-IOS: memcpy
  ; CHECK-DARWIN: memcpy
  ; CHECK-EABI: __aeabi_memcpy
  %arr1 = alloca [13 x i8], align 1
  %1 = getelementptr inbounds [13 x i8], [13 x i8]* %arr1, i32 0, i32 16
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* %dest, i8* %1, i32 %n, i32 0, i1 false)

  ; CHECK: {{add(.w)? r., sp, #(1|5)}}
  ; CHECK-IOS: mov r1, #0
  ; CHECK-IOS: memset
  ; CHECK-DARWIN: movs r1, #0
  ; CHECK-DARWIN: memset
  ; CHECK-EABI: mov r2, #0
  ; CHECK-EABI: __aeabi_memset
  %arr2 = alloca [13 x i8], align 1
  %2 = getelementptr inbounds [13 x i8], [13 x i8]* %arr2, i32 0, i32 16
  call void @llvm.memset.p0i8.i32(i8* %2, i8 0, i32 %n, i32 0, i1 false)

  unreachable
}

declare void @llvm.memmove.p0i8.p0i8.i32(i8* nocapture, i8* nocapture, i32, i32, i1) nounwind
declare void @llvm.memcpy.p0i8.p0i8.i32(i8* nocapture, i8* nocapture, i32, i32, i1) nounwind
declare void @llvm.memset.p0i8.i32(i8* nocapture, i8, i32, i32, i1) nounwind
