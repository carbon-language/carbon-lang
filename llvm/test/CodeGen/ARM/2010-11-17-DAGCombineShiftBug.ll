; RUN: llc < %s -march=arm -mattr=+neon
define void @lshrIllegalType(<8 x i32>* %A) nounwind {
       %tmp1 = load <8 x i32>* %A
       %tmp2 = lshr <8 x i32> %tmp1, < i32 3, i32 3, i32 3, i32 3, i32 3, i32 3, i32 3, i32 3>
       store <8 x i32> %tmp2, <8 x i32>* %A
       ret void
}

