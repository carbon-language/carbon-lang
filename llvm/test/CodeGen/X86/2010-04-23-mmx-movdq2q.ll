; RUN: llc < %s -mtriple=x86_64-apple-darwin -mattr=+mmx,+sse2 | FileCheck %s

define void @ti8(double %a, double %b) nounwind {
entry:
        %tmp1 = bitcast double %a to <8 x i8>
; CHECK: movdq2q
        %tmp2 = bitcast double %b to <8 x i8>
; CHECK: movdq2q
        %tmp3 = add <8 x i8> %tmp1, %tmp2
        store <8 x i8> %tmp3, <8 x i8>* null
        ret void
}

define void @ti16(double %a, double %b) nounwind {
entry:
        %tmp1 = bitcast double %a to <4 x i16>
; CHECK: movdq2q
        %tmp2 = bitcast double %b to <4 x i16>
; CHECK: movdq2q
        %tmp3 = add <4 x i16> %tmp1, %tmp2
        store <4 x i16> %tmp3, <4 x i16>* null
        ret void
}

define void @ti32(double %a, double %b) nounwind {
entry:
        %tmp1 = bitcast double %a to <2 x i32>
; CHECK: movdq2q
        %tmp2 = bitcast double %b to <2 x i32>
; CHECK: movdq2q
        %tmp3 = add <2 x i32> %tmp1, %tmp2
        store <2 x i32> %tmp3, <2 x i32>* null
        ret void
}

define void @ti64(double %a, double %b) nounwind {
entry:
        %tmp1 = bitcast double %a to <1 x i64>
; CHECK: movdq2q
        %tmp2 = bitcast double %b to <1 x i64>
; CHECK: movdq2q
        %tmp3 = add <1 x i64> %tmp1, %tmp2
        store <1 x i64> %tmp3, <1 x i64>* null
        ret void
}
