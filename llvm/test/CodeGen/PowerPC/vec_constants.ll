; RUN: llc -O0 -mcpu=pwr7 < %s | FileCheck %s

target datalayout = "E-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-f128:128:128-v128:128:128-n32:64"
target triple = "powerpc64-unknown-linux-gnu"

define void @test1(<4 x i32>* %P1, <4 x i32>* %P2, <4 x float>* %P3) nounwind {
	%tmp = load <4 x i32>* %P1		; <<4 x i32>> [#uses=1]
	%tmp4 = and <4 x i32> %tmp, < i32 -2147483648, i32 -2147483648, i32 -2147483648, i32 -2147483648 >		; <<4 x i32>> [#uses=1]
	store <4 x i32> %tmp4, <4 x i32>* %P1
	%tmp7 = load <4 x i32>* %P2		; <<4 x i32>> [#uses=1]
	%tmp9 = and <4 x i32> %tmp7, < i32 2147483647, i32 2147483647, i32 2147483647, i32 2147483647 >		; <<4 x i32>> [#uses=1]
	store <4 x i32> %tmp9, <4 x i32>* %P2
	%tmp.upgrd.1 = load <4 x float>* %P3		; <<4 x float>> [#uses=1]
	%tmp11 = bitcast <4 x float> %tmp.upgrd.1 to <4 x i32>		; <<4 x i32>> [#uses=1]
	%tmp12 = and <4 x i32> %tmp11, < i32 2147483647, i32 2147483647, i32 2147483647, i32 2147483647 >		; <<4 x i32>> [#uses=1]
	%tmp13 = bitcast <4 x i32> %tmp12 to <4 x float>		; <<4 x float>> [#uses=1]
	store <4 x float> %tmp13, <4 x float>* %P3
	ret void

; CHECK: test1:
; CHECK-NOT: CPI
}

define <4 x i32> @test_30() nounwind {
	ret <4 x i32> < i32 30, i32 30, i32 30, i32 30 >

; CHECK: test_30:
; CHECK: vspltisw
; CHECK-NEXT: vadduwm
; CHECK-NEXT: blr
}

define <4 x i32> @test_29() nounwind {
	ret <4 x i32> < i32 29, i32 29, i32 29, i32 29 >

; CHECK: test_29:
; CHECK: vspltisw
; CHECK-NEXT: vspltisw
; CHECK-NEXT: vsubuwm
; CHECK-NEXT: blr
}

define <8 x i16> @test_n30() nounwind {
	ret <8 x i16> < i16 -30, i16 -30, i16 -30, i16 -30, i16 -30, i16 -30, i16 -30, i16 -30 >

; CHECK: test_n30:
; CHECK: vspltish
; CHECK-NEXT: vadduhm
; CHECK-NEXT: blr
}

define <16 x i8> @test_n104() nounwind {
	ret <16 x i8> < i8 -104, i8 -104, i8 -104, i8 -104, i8 -104, i8 -104, i8 -104, i8 -104, i8 -104, i8 -104, i8 -104, i8 -104, i8 -104, i8 -104, i8 -104, i8 -104 >

; CHECK: test_n104:
; CHECK: vspltisb
; CHECK-NEXT: vslb
; CHECK-NEXT: blr
}

define <4 x i32> @test_vsldoi() nounwind {
	ret <4 x i32> < i32 512, i32 512, i32 512, i32 512 >

; CHECK: test_vsldoi:
; CHECK: vspltisw
; CHECK-NEXT: vsldoi
; CHECK-NEXT: blr
}

define <8 x i16> @test_vsldoi_65023() nounwind {
	ret <8 x i16> < i16 65023, i16 65023,i16 65023,i16 65023,i16 65023,i16 65023,i16 65023,i16 65023 >

; CHECK: test_vsldoi_65023:
; CHECK: vspltish
; CHECK-NEXT: vsldoi
; CHECK-NEXT: blr
}

define <4 x i32> @test_rol() nounwind {
	ret <4 x i32> < i32 -11534337, i32 -11534337, i32 -11534337, i32 -11534337 >

; CHECK: test_rol:
; CHECK: vspltisw
; CHECK-NEXT: vrlw
; CHECK-NEXT: blr
}
