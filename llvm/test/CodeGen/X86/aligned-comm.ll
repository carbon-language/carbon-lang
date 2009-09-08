; RUN: llc < %s -march=x86
; RUN: llc < %s -mtriple=i386-apple-darwin10 | grep {array,16512,7}
; RUN: llc < %s -mtriple=i386-apple-darwin9 | grep {array,16512,7}
; RUN: llc < %s -mtriple=i386-apple-darwin8 | not grep {7}

; Darwin 9+ should get alignment on common symbols.  Darwin8 does 
; not support this.
@array = common global [4128 x i32] zeroinitializer, align 128
