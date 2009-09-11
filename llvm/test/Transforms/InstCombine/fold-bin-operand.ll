; RUN: opt < %s -instcombine -S | not grep icmp

define i1 @f(i1 %x) {
	%b = and i1 %x, icmp eq (i8* inttoptr (i32 1 to i8*), i8* inttoptr (i32 2 to i8*))
	ret i1 %b
}

; FIXME: This doesn't fold at the moment!
; define i32 @f(i32 %x) {
;	%b = add i32 %x, zext (i1 icmp eq (i8* inttoptr (i32 1000000 to i8*), i8* inttoptr (i32 2000000 to i8*)) to i32)
;	ret i32 %b
;}

