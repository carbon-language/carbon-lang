; RUN: opt < %s -tailduplicate -disable-output
; PR4662

define void @a() {
BB:
	br label %BB6

BB6:
	%tmp9 = phi i64 [ 0, %BB ], [ 5, %BB34 ]
	br label %BB34

BB34:
	br label %BB6
}
