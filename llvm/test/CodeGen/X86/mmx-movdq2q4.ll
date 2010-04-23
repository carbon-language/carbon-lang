; RUN: llc < %s -mtriple=x86_64-apple-darwin -mattr=+mmx,+sse2 | grep movdq2q | count 2
define void @t2(double %a, double %b) nounwind {
entry:
        %tmp1 = bitcast double %a to <1 x i64>
        %tmp2 = bitcast double %b to <1 x i64>
        %tmp3 = add <1 x i64> %tmp1, %tmp2
        store <1 x i64> %tmp3, <1 x i64>* null
        ret void
}
