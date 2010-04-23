; RUN: llc < %s -mtriple=x86_64-apple-darwin -mattr=+mmx,+sse2 | grep movdq2q | count 2
define void @t2(double %a, double %b) nounwind {
entry:
        %tmp1 = bitcast double %a to <8 x i8>
        %tmp2 = bitcast double %b to <8 x i8>
        %tmp3 = add <8 x i8> %tmp1, %tmp2
        store <8 x i8> %tmp3, <8 x i8>* null
        ret void
}
