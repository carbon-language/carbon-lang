; RUN: llc < %s -mtriple=x86_64-apple-darwin -mattr=+mmx,+sse2 | grep movdq2q | count 2
define void @t2(double %a, double %b) nounwind {
entry:
        %tmp1 = bitcast double %a to <2 x i32>
        %tmp2 = bitcast double %b to <2 x i32>
        %tmp3 = add <2 x i32> %tmp1, %tmp2
        store <2 x i32> %tmp3, <2 x i32>* null
        ret void
}
