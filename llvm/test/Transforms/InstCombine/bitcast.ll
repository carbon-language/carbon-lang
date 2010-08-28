; RUN: opt < %s -instcombine -S | FileCheck %s

; Bitcasts between vectors and scalars are valid.
; PR4487
define i32 @test1(i64 %a) {
        %t1 = bitcast i64 %a to <2 x i32>
        %t2 = bitcast i64 %a to <2 x i32>
        %t3 = xor <2 x i32> %t1, %t2
        %t4 = extractelement <2 x i32> %t3, i32 0
        ret i32 %t4
        
; CHECK: @test1
; CHECK: ret i32 0
}

; Optimize bitcasts that are extracting low element of vector.  This happens
; because of SRoA.
; rdar://7892780
define float @test2(<2 x float> %A, <2 x i32> %B) {
  %tmp28 = bitcast <2 x float> %A to i64  ; <i64> [#uses=2]
  %tmp23 = trunc i64 %tmp28 to i32                ; <i32> [#uses=1]
  %tmp24 = bitcast i32 %tmp23 to float            ; <float> [#uses=1]

  %tmp = bitcast <2 x i32> %B to i64
  %tmp2 = trunc i64 %tmp to i32                ; <i32> [#uses=1]
  %tmp4 = bitcast i32 %tmp2 to float            ; <float> [#uses=1]

  %add = fadd float %tmp24, %tmp4
  ret float %add
  
; CHECK: @test2
; CHECK-NEXT:  %tmp24 = extractelement <2 x float> %A, i32 0
; CHECK-NEXT:  bitcast <2 x i32> %B to <2 x float>
; CHECK-NEXT:  %tmp4 = extractelement <2 x float> {{.*}}, i32 0
; CHECK-NEXT:  %add = fadd float %tmp24, %tmp4
; CHECK-NEXT:  ret float %add
}

; Optimize bitcasts that are extracting other elements of a vector.  This
; happens because of SRoA.
; rdar://7892780
define float @test3(<2 x float> %A, <2 x i64> %B) {
  %tmp28 = bitcast <2 x float> %A to i64
  %tmp29 = lshr i64 %tmp28, 32
  %tmp23 = trunc i64 %tmp29 to i32
  %tmp24 = bitcast i32 %tmp23 to float

  %tmp = bitcast <2 x i64> %B to i128
  %tmp1 = lshr i128 %tmp, 64
  %tmp2 = trunc i128 %tmp1 to i32
  %tmp4 = bitcast i32 %tmp2 to float

  %add = fadd float %tmp24, %tmp4
  ret float %add
  
; CHECK: @test3
; CHECK-NEXT:  %tmp24 = extractelement <2 x float> %A, i32 1
; CHECK-NEXT:  bitcast <2 x i64> %B to <4 x float>
; CHECK-NEXT:  %tmp4 = extractelement <4 x float> {{.*}}, i32 2
; CHECK-NEXT:  %add = fadd float %tmp24, %tmp4
; CHECK-NEXT:  ret float %add
}


define <2 x i32> @test4(i32 %A, i32 %B){
  %tmp38 = zext i32 %A to i64
  %tmp32 = zext i32 %B to i64
  %tmp33 = shl i64 %tmp32, 32
  %ins35 = or i64 %tmp33, %tmp38
  %tmp43 = bitcast i64 %ins35 to <2 x i32>
  ret <2 x i32> %tmp43
  ; CHECK: @test4
  ; CHECK-NEXT: insertelement <2 x i32> undef, i32 %A, i32 0
  ; CHECK-NEXT: insertelement <2 x i32> {{.*}}, i32 %B, i32 1
  ; CHECK-NEXT: ret <2 x i32> 

}

; rdar://8360454
define <2 x float> @test5(float %A, float %B) {
  %tmp37 = bitcast float %A to i32
  %tmp38 = zext i32 %tmp37 to i64
  %tmp31 = bitcast float %B to i32
  %tmp32 = zext i32 %tmp31 to i64
  %tmp33 = shl i64 %tmp32, 32
  %ins35 = or i64 %tmp33, %tmp38
  %tmp43 = bitcast i64 %ins35 to <2 x float>
  ret <2 x float> %tmp43
  ; CHECK: @test5
  ; CHECK-NEXT: insertelement <2 x float> undef, float %A, i32 0
  ; CHECK-NEXT: insertelement <2 x float> {{.*}}, float %B, i32 1
  ; CHECK-NEXT: ret <2 x float> 
}
