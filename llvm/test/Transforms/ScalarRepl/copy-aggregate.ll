; RUN: opt < %s -scalarrepl -S | FileCheck %s
; PR3290
target datalayout = "E-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64"

;; Store of integer to whole alloca struct.
define i32 @test1(i64 %V) nounwind {
; CHECK: test1
; CHECK-NOT: alloca
	%X = alloca {{i32, i32}}
	%Y = bitcast {{i32,i32}}* %X to i64*
	store i64 %V, i64* %Y

	%A = getelementptr {{i32,i32}}* %X, i32 0, i32 0, i32 0
	%B = getelementptr {{i32,i32}}* %X, i32 0, i32 0, i32 1
	%a = load i32* %A
	%b = load i32* %B
	%c = add i32 %a, %b
	ret i32 %c
}

;; Store of integer to whole struct/array alloca.
define float @test2(i128 %V) nounwind {
; CHECK: test2
; CHECK-NOT: alloca
	%X = alloca {[4 x float]}
	%Y = bitcast {[4 x float]}* %X to i128*
	store i128 %V, i128* %Y

	%A = getelementptr {[4 x float]}* %X, i32 0, i32 0, i32 0
	%B = getelementptr {[4 x float]}* %X, i32 0, i32 0, i32 3
	%a = load float* %A
	%b = load float* %B
	%c = fadd float %a, %b
	ret float %c
}

;; Load of whole alloca struct as integer
define i64 @test3(i32 %a, i32 %b) nounwind {
; CHECK: test3
; CHECK-NOT: alloca
	%X = alloca {{i32, i32}}

	%A = getelementptr {{i32,i32}}* %X, i32 0, i32 0, i32 0
	%B = getelementptr {{i32,i32}}* %X, i32 0, i32 0, i32 1
        store i32 %a, i32* %A
        store i32 %b, i32* %B

	%Y = bitcast {{i32,i32}}* %X to i64*
        %Z = load i64* %Y
	ret i64 %Z
}

;; load of integer from whole struct/array alloca.
define i128 @test4(float %a, float %b) nounwind {
; CHECK: test4
; CHECK-NOT: alloca
	%X = alloca {[4 x float]}
	%A = getelementptr {[4 x float]}* %X, i32 0, i32 0, i32 0
	%B = getelementptr {[4 x float]}* %X, i32 0, i32 0, i32 3
	store float %a, float* %A
	store float %b, float* %B
        
      	%Y = bitcast {[4 x float]}* %X to i128*
	%V = load i128* %Y
	ret i128 %V
}

;; If the elements of a struct or array alloca contain padding, SROA can still
;; split up the alloca as long as there is no padding between the elements.
%padded = type { i16, i8 }
%arr = type [4 x %padded]
define void @test5(%arr* %p, %arr* %q) {
entry:
; CHECK: test5
; CHECK-NOT: i128
  %var = alloca %arr, align 4
  %vari8 = bitcast %arr* %var to i8*
  %pi8 = bitcast %arr* %p to i8*
  call void @llvm.memcpy.i32(i8* %vari8, i8* %pi8, i32 16, i32 4)
  %qi8 = bitcast %arr* %q to i8*
  call void @llvm.memcpy.i32(i8* %qi8, i8* %vari8, i32 16, i32 4)
  ret void
}

declare void @llvm.memcpy.i32(i8* nocapture, i8* nocapture, i32, i32) nounwind
