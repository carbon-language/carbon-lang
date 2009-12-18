; RUN: opt < %s -scalarrepl -S | FileCheck %s
; Radar 7441282

target datalayout = "e-p:32:32:32-i1:8:32-i8:8:32-i16:16:32-i32:32:32-i64:32:32-f32:32:32-f64:32:32-v64:64:64-v128:128:128-a0:0:32-n32"
target triple = "thumbv7-apple-darwin10"

%struct.__neon_int16x8x2_t = type { <8 x i16>, <8 x i16> }
%struct.int16x8_t = type { <8 x i16> }
%struct.int16x8x2_t = type { [2 x %struct.int16x8_t] }
%union..0anon = type { %struct.int16x8x2_t }

define arm_apcscc void @test(<8 x i16> %tmp.0, %struct.int16x8x2_t* %dst) nounwind {
; CHECK: @test
; CHECK-NOT: alloca
; CHECK: "alloca point"
entry:
  %tmp_addr = alloca %struct.int16x8_t            ; <%struct.int16x8_t*> [#uses=3]
  %dst_addr = alloca %struct.int16x8x2_t*         ; <%struct.int16x8x2_t**> [#uses=2]
  %__rv = alloca %union..0anon                    ; <%union..0anon*> [#uses=2]
  %__bx = alloca %struct.int16x8_t                ; <%struct.int16x8_t*> [#uses=2]
  %__ax = alloca %struct.int16x8_t                ; <%struct.int16x8_t*> [#uses=2]
  %tmp2 = alloca %struct.int16x8x2_t              ; <%struct.int16x8x2_t*> [#uses=2]
  %0 = alloca %struct.int16x8x2_t                 ; <%struct.int16x8x2_t*> [#uses=2]
  %"alloca point" = bitcast i32 0 to i32          ; <i32> [#uses=0]
  %1 = getelementptr inbounds %struct.int16x8_t* %tmp_addr, i32 0, i32 0 ; <<8 x i16>*> [#uses=1]
  store <8 x i16> %tmp.0, <8 x i16>* %1
  store %struct.int16x8x2_t* %dst, %struct.int16x8x2_t** %dst_addr
  %2 = getelementptr inbounds %struct.int16x8_t* %__ax, i32 0, i32 0 ; <<8 x i16>*> [#uses=1]
  %3 = getelementptr inbounds %struct.int16x8_t* %tmp_addr, i32 0, i32 0 ; <<8 x i16>*> [#uses=1]
  %4 = load <8 x i16>* %3, align 16               ; <<8 x i16>> [#uses=1]
  store <8 x i16> %4, <8 x i16>* %2, align 16
  %5 = getelementptr inbounds %struct.int16x8_t* %__bx, i32 0, i32 0 ; <<8 x i16>*> [#uses=1]
  %6 = getelementptr inbounds %struct.int16x8_t* %tmp_addr, i32 0, i32 0 ; <<8 x i16>*> [#uses=1]
  %7 = load <8 x i16>* %6, align 16               ; <<8 x i16>> [#uses=1]
  store <8 x i16> %7, <8 x i16>* %5, align 16
  %8 = getelementptr inbounds %struct.int16x8_t* %__ax, i32 0, i32 0 ; <<8 x i16>*> [#uses=1]
  %9 = load <8 x i16>* %8, align 16               ; <<8 x i16>> [#uses=2]
  %10 = getelementptr inbounds %struct.int16x8_t* %__bx, i32 0, i32 0 ; <<8 x i16>*> [#uses=1]
  %11 = load <8 x i16>* %10, align 16             ; <<8 x i16>> [#uses=2]
  %12 = getelementptr inbounds %union..0anon* %__rv, i32 0, i32 0 ; <%struct.int16x8x2_t*> [#uses=1]
  %13 = bitcast %struct.int16x8x2_t* %12 to %struct.__neon_int16x8x2_t* ; <%struct.__neon_int16x8x2_t*> [#uses=2]
  %14 = shufflevector <8 x i16> %9, <8 x i16> %11, <8 x i32> <i32 0, i32 8, i32 2, i32 10, i32 4, i32 12, i32 6, i32 14> ; <<8 x i16>> [#uses=1]
  %15 = getelementptr inbounds %struct.__neon_int16x8x2_t* %13, i32 0, i32 0 ; <<8 x i16>*> [#uses=1]
  store <8 x i16> %14, <8 x i16>* %15
  %16 = shufflevector <8 x i16> %9, <8 x i16> %11, <8 x i32> <i32 1, i32 9, i32 3, i32 11, i32 5, i32 13, i32 7, i32 15> ; <<8 x i16>> [#uses=1]
  %17 = getelementptr inbounds %struct.__neon_int16x8x2_t* %13, i32 0, i32 1 ; <<8 x i16>*> [#uses=1]
  store <8 x i16> %16, <8 x i16>* %17
  %18 = getelementptr inbounds %union..0anon* %__rv, i32 0, i32 0 ; <%struct.int16x8x2_t*> [#uses=1]
  %19 = bitcast %struct.int16x8x2_t* %0 to i8*    ; <i8*> [#uses=1]
  %20 = bitcast %struct.int16x8x2_t* %18 to i8*   ; <i8*> [#uses=1]
  call void @llvm.memcpy.i32(i8* %19, i8* %20, i32 32, i32 16)
  %tmp21 = bitcast %struct.int16x8x2_t* %tmp2 to i8* ; <i8*> [#uses=1]
  %21 = bitcast %struct.int16x8x2_t* %0 to i8*    ; <i8*> [#uses=1]
  call void @llvm.memcpy.i32(i8* %tmp21, i8* %21, i32 32, i32 16)
  %22 = load %struct.int16x8x2_t** %dst_addr, align 4 ; <%struct.int16x8x2_t*> [#uses=1]
  %23 = bitcast %struct.int16x8x2_t* %22 to i8*   ; <i8*> [#uses=1]
  %tmp22 = bitcast %struct.int16x8x2_t* %tmp2 to i8* ; <i8*> [#uses=1]
  call void @llvm.memcpy.i32(i8* %23, i8* %tmp22, i32 32, i32 16)
  br label %return

; CHECK: store <8 x i16>
; CHECK: store <8 x i16>

return:                                           ; preds = %entry
  ret void
}

; Radar 7466574
%struct._NSRange = type { i64 }

define arm_apcscc void @test_memcpy_self() nounwind {
; CHECK: @test_memcpy_self
; CHECK-NOT: alloca
; CHECK: br i1
entry:
  %range = alloca %struct._NSRange                ; <%struct._NSRange*> [#uses=2]
  br i1 undef, label %cond.true, label %cond.false

cond.true:                                        ; preds = %entry
  %tmp3 = bitcast %struct._NSRange* %range to i8* ; <i8*> [#uses=1]
  %tmp4 = bitcast %struct._NSRange* %range to i8* ; <i8*> [#uses=1]
  call void @llvm.memcpy.i32(i8* %tmp3, i8* %tmp4, i32 8, i32 8)
  ret void

cond.false:                                       ; preds = %entry
  ret void
}

declare void @llvm.memcpy.i32(i8* nocapture, i8* nocapture, i32, i32) nounwind
