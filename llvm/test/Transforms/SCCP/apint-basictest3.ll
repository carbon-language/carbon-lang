; This is a basic sanity check for constant propagation.  It tests the basic
; arithmetic operations.


; RUN: opt < %s -sccp -S | not grep mul
; RUN: opt < %s -sccp -S | not grep umod

define i128 @test(i1 %B) {
	br i1 %B, label %BB1, label %BB2
BB1:
	%t1 = add i128 0, 1
        %t2 = sub i128 0, %t1
        %t3 = mul i128 %t2, -1
	br label %BB3
BB2:
        %f1 = udiv i128 -1, 1
        %f2 = add i128 %f1, 1
        %f3 = urem i128 %f2, 2121
	br label %BB3
BB3:
	%Ret = phi i128 [%t3, %BB1], [%f3, %BB2]
	ret i128 %Ret
}
