; RUN: llc < %s -mtriple=thumbv8 -arm-atomic-cfg-tidy=0 | FileCheck %s
; RUN: llc < %s -mtriple=thumbv7 -arm-atomic-cfg-tidy=0 -arm-restrict-it | FileCheck %s
; RUN: llc < %s -mtriple=thumbv8 -arm-atomic-cfg-tidy=0 -relocation-model=pic | FileCheck %s --check-prefix=CHECK-PIC
; RUN: llc < %s -mtriple=thumbv7 -arm-atomic-cfg-tidy=0 -arm-restrict-it -relocation-model=pic | FileCheck %s --check-prefix=CHECK-PIC

%struct.FF = type { i32 (i32*)*, i32 (i32*, i32*, i32, i32, i32, i32)*, i32 (i32, i32, i8*)*, void ()*, i32 (i32, i8*, i32*)*, i32 ()* }
%struct.BD = type { %struct.BD*, i32, i32, i32, i32, i64, i32 (%struct.BD*, i8*, i64, i32)*, i32 (%struct.BD*, i8*, i32, i32)*, i32 (%struct.BD*, i8*, i64, i32)*, i32 (%struct.BD*, i8*, i32, i32)*, i32 (%struct.BD*, i64, i32)*, [16 x i8], i64, i64 }

@FuncPtr = external hidden unnamed_addr global %struct.FF*
@.str1 = external hidden unnamed_addr constant [6 x i8], align 4
@G = external unnamed_addr global i32
@.str2 = external hidden unnamed_addr constant [58 x i8], align 4
@.str3 = external hidden unnamed_addr constant [58 x i8], align 4

define i32 @test() nounwind optsize ssp {
entry:
; CHECK-LABEL: test:
; CHECK: push
; CHECK-NOT: push
  %block_size = alloca i32, align 4
  %block_count = alloca i32, align 4
  %index_cache = alloca i32, align 4
  store i32 0, i32* %index_cache, align 4
  %tmp = load i32, i32* @G, align 4
  %tmp1 = call i32 @bar(i32 0, i32 0, i32 %tmp) nounwind
  switch i32 %tmp1, label %bb8 [
    i32 1, label %bb
    i32 536870913, label %bb4
    i32 536870914, label %bb6
  ]

bb:
  %tmp2 = load i32, i32* @G, align 4
  %tmp4 = icmp eq i32 %tmp2, 1
  br i1 %tmp4, label %bb1, label %bb8

bb1:
; CHECK: %entry
; CHECK: it	eq
; CHECK-NEXT: ldreq
; CHECK-NEXT: it	eq
; CHECK-NEXT: cmpeq
; CHECK: %bb1
  %tmp5 = load i32, i32* %block_size, align 4
  %tmp6 = load i32, i32* %block_count, align 4
  %tmp7 = call %struct.FF* @Get() nounwind
  store %struct.FF* %tmp7, %struct.FF** @FuncPtr, align 4
  %tmp10 = zext i32 %tmp6 to i64
  %tmp11 = zext i32 %tmp5 to i64
  %tmp12 = mul nsw i64 %tmp10, %tmp11
  %tmp13 = call i32 @foo(i8* getelementptr inbounds ([6 x i8], [6 x i8]* @.str1, i32 0, i32 0), i64 %tmp12, i32 %tmp5) nounwind
  br label %bb8

bb4:
; CHECK-PIC: cmp
; CHECK-PIC: cmp
; CHECK-PIC: cmp
; CHECK-PIC: it eq
; CHECK-PIC-NEXT: ldreq
; CHECK-PIC-NEXT: it eq
; CHECK-PIC-NEXT: cmpeq
; CHECK-PIC-NEXT: beq
; CHECK-PIC: %bb6
; CHECK-PIC: mov
  ret i32 0

bb6:
  ret i32 1

bb8:
  ret i32 -1
}

declare i32 @printf(i8*, ...)

declare %struct.FF* @Get()

declare i32 @foo(i8*, i64, i32)

declare i32 @bar(i32, i32, i32)
