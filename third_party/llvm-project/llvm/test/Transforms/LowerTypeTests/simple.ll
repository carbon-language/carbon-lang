; RUN: opt -S -passes=lowertypetests < %s | FileCheck %s
; RUN: opt -S -lowertypetests < %s | FileCheck %s
; RUN: opt -S -lowertypetests -mtriple=x86_64-apple-macosx10.8.0 < %s | FileCheck %s
; RUN: opt -S -O3 < %s | FileCheck -check-prefix=CHECK-NODISCARD %s

target datalayout = "e-p:32:32"

; CHECK: [[G:@[^ ]*]] = private constant { i32, [0 x i8], [63 x i32], [4 x i8], i32, [0 x i8], [2 x i32] } { i32 1, [0 x i8] zeroinitializer, [63 x i32] zeroinitializer, [4 x i8] zeroinitializer, i32 3, [0 x i8] zeroinitializer, [2 x i32] [i32 4, i32 5] }
@a = constant i32 1, !type !0, !type !2
@b = hidden constant [63 x i32] zeroinitializer, !type !0, !type !1
@c = protected constant i32 3, !type !1, !type !2
@d = constant [2 x i32] [i32 4, i32 5], !type !3

; CHECK-NODISCARD: !type
; CHECK-NODISCARD: !type
; CHECK-NODISCARD: !type
; CHECK-NODISCARD: !type
; CHECK-NODISCARD: !type
; CHECK-NODISCARD: !type
; CHECK-NODISCARD: !type

; CHECK: [[BA:@[^ ]*]] = private constant [68 x i8] c"\03\01\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\02\00\01"

; Offset 0, 4 byte alignment
!0 = !{i32 0, !"typeid1"}
!3 = !{i32 4, !"typeid1"}

; Offset 4, 256 byte alignment
!1 = !{i32 0, !"typeid2"}

; Offset 0, 4 byte alignment
!2 = !{i32 0, !"typeid3"}

; CHECK: @bits_use{{[0-9]*}} = private alias i8, i8* @bits{{[0-9]*}}
; CHECK: @bits_use.{{[0-9]*}} = private alias i8, i8* @bits{{[0-9]*}}
; CHECK: @bits_use.{{[0-9]*}} = private alias i8, i8* @bits{{[0-9]*}}

; CHECK: @a = alias i32, getelementptr inbounds ({ i32, [0 x i8], [63 x i32], [4 x i8], i32, [0 x i8], [2 x i32] }, { i32, [0 x i8], [63 x i32], [4 x i8], i32, [0 x i8], [2 x i32] }* [[G]], i32 0, i32 0)
; CHECK: @b = hidden alias [63 x i32], getelementptr inbounds ({ i32, [0 x i8], [63 x i32], [4 x i8], i32, [0 x i8], [2 x i32] }, { i32, [0 x i8], [63 x i32], [4 x i8], i32, [0 x i8], [2 x i32] }* [[G]], i32 0, i32 2)
; CHECK: @c = protected alias i32, getelementptr inbounds ({ i32, [0 x i8], [63 x i32], [4 x i8], i32, [0 x i8], [2 x i32] }, { i32, [0 x i8], [63 x i32], [4 x i8], i32, [0 x i8], [2 x i32] }* [[G]], i32 0, i32 4)
; CHECK: @d = alias [2 x i32], getelementptr inbounds ({ i32, [0 x i8], [63 x i32], [4 x i8], i32, [0 x i8], [2 x i32] }, { i32, [0 x i8], [63 x i32], [4 x i8], i32, [0 x i8], [2 x i32] }* [[G]], i32 0, i32 6)

; CHECK: @bits{{[0-9]*}} = private alias i8, getelementptr inbounds ([68 x i8], [68 x i8]* [[BA]], i32 0, i32 0)
; CHECK: @bits.{{[0-9]*}} = private alias i8, getelementptr inbounds ([68 x i8], [68 x i8]* [[BA]], i32 0, i32 0)

declare i1 @llvm.type.test(i8* %ptr, metadata %bitset) nounwind readnone

; CHECK: @foo(i32* [[A0:%[^ ]*]])
define i1 @foo(i32* %p) {
  ; CHECK-NOT: llvm.type.test

  ; CHECK: [[R0:%[^ ]*]] = bitcast i32* [[A0]] to i8*
  %pi8 = bitcast i32* %p to i8*
  ; CHECK: [[R1:%[^ ]*]] = ptrtoint i8* [[R0]] to i32
  ; CHECK: [[R2:%[^ ]*]] = sub i32 [[R1]], ptrtoint ({ i32, [0 x i8], [63 x i32], [4 x i8], i32, [0 x i8], [2 x i32] }* [[G]] to i32)
  ; CHECK: [[R3:%[^ ]*]] = lshr i32 [[R2]], 2
  ; CHECK: [[R4:%[^ ]*]] = shl i32 [[R2]], 30
  ; CHECK: [[R5:%[^ ]*]] = or i32 [[R3]], [[R4]]
  ; CHECK: [[R6:%[^ ]*]] = icmp ule i32 [[R5]], 67
  ; CHECK: br i1 [[R6]]

  ; CHECK: [[R8:%[^ ]*]] = getelementptr i8, i8* @bits_use.{{[0-9]*}}, i32 [[R5]]
  ; CHECK: [[R9:%[^ ]*]] = load i8, i8* [[R8]]
  ; CHECK: [[R10:%[^ ]*]] = and i8 [[R9]], 1
  ; CHECK: [[R11:%[^ ]*]] = icmp ne i8 [[R10]], 0

  ; CHECK: [[R16:%[^ ]*]] = phi i1 [ false, {{%[^ ]*}} ], [ [[R11]], {{%[^ ]*}} ]
  %x = call i1 @llvm.type.test(i8* %pi8, metadata !"typeid1")

  ; CHECK-NOT: llvm.type.test
  %y = call i1 @llvm.type.test(i8* %pi8, metadata !"typeid1")

  ; CHECK: ret i1 [[R16]]
  ret i1 %x
}

; CHECK: @bar(i32* [[B0:%[^ ]*]])
define i1 @bar(i32* %p) {
  ; CHECK: [[S0:%[^ ]*]] = bitcast i32* [[B0]] to i8*
  %pi8 = bitcast i32* %p to i8*
  ; CHECK: [[S1:%[^ ]*]] = ptrtoint i8* [[S0]] to i32
  ; CHECK: [[S2:%[^ ]*]] = sub i32 [[S1]], ptrtoint (i8* getelementptr (i8, i8* bitcast ({ i32, [0 x i8], [63 x i32], [4 x i8], i32, [0 x i8], [2 x i32] }* [[G]] to i8*), i32 4) to i32)
  ; CHECK: [[S3:%[^ ]*]] = lshr i32 [[S2]], 8
  ; CHECK: [[S4:%[^ ]*]] = shl i32 [[S2]], 24
  ; CHECK: [[S5:%[^ ]*]] = or i32 [[S3]], [[S4]]
  ; CHECK: [[S6:%[^ ]*]] = icmp ule i32 [[S5]], 1
  %x = call i1 @llvm.type.test(i8* %pi8, metadata !"typeid2")

  ; CHECK: ret i1 [[S6]]
  ret i1 %x
}

; CHECK: @baz(i32* [[C0:%[^ ]*]])
define i1 @baz(i32* %p) {
  ; CHECK: [[T0:%[^ ]*]] = bitcast i32* [[C0]] to i8*
  %pi8 = bitcast i32* %p to i8*
  ; CHECK: [[T1:%[^ ]*]] = ptrtoint i8* [[T0]] to i32
  ; CHECK: [[T2:%[^ ]*]] = sub i32 [[T1]], ptrtoint ({ i32, [0 x i8], [63 x i32], [4 x i8], i32, [0 x i8], [2 x i32] }* [[G]] to i32)
  ; CHECK: [[T3:%[^ ]*]] = lshr i32 [[T2]], 2
  ; CHECK: [[T4:%[^ ]*]] = shl i32 [[T2]], 30
  ; CHECK: [[T5:%[^ ]*]] = or i32 [[T3]], [[T4]]
  ; CHECK: [[T6:%[^ ]*]] = icmp ule i32 [[T5]], 65
  ; CHECK: br i1 [[T6]]

  ; CHECK: [[T8:%[^ ]*]] = getelementptr i8, i8* @bits_use{{(\.[0-9]*)?}}, i32 [[T5]]
  ; CHECK: [[T9:%[^ ]*]] = load i8, i8* [[T8]]
  ; CHECK: [[T10:%[^ ]*]] = and i8 [[T9]], 2
  ; CHECK: [[T11:%[^ ]*]] = icmp ne i8 [[T10]], 0

  ; CHECK: [[T16:%[^ ]*]] = phi i1 [ false, {{%[^ ]*}} ], [ [[T11]], {{%[^ ]*}} ]
  %x = call i1 @llvm.type.test(i8* %pi8, metadata !"typeid3")
  ; CHECK: ret i1 [[T16]]
  ret i1 %x
}
