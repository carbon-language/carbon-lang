; RUN: llvm-as < %s | llvm-dis | llvm-as -disable-output
; PR1269
; END
; http://nondot.org/sabre/LLVMNotes/ExceptionHandlingChanges.txt

define i1 @test1(i8 %i, i8 %j) {
entry: unwind_to %target
	%tmp = sub i8 %i, %j		; <i8> [#uses=1]
	%b = icmp eq i8 %tmp, 0		; <i1> [#uses=1]
	ret i1 %b
target:
	ret i1 false
}

define i1 @test2(i8 %i, i8 %j) {
entry:
	br label %0
unwind_to %1
	%tmp = sub i8 %i, %j		; <i8> [#uses=1]
	%b = icmp eq i8 %tmp, 0		; <i1> [#uses=1]
	ret i1 %b
		; No predecessors!
	ret i1 false
}

define i1 @test3(i8 %i, i8 %j) {
entry:
	br label %0
unwind_to %1
	%tmp = sub i8 %i, %j		; <i8> [#uses=1]
	%b = icmp eq i8 %tmp, 0		; <i1> [#uses=1]
	ret i1 %b
unwind_to %0
	ret i1 false
}

define i1 @test4(i8 %i, i8 %j) {
	%tmp = sub i8 %i, %j		; <i8> [#uses=1]
	%b = icmp eq i8 %tmp, 0		; <i1> [#uses=1]
	br label %1
unwind_to %1
	ret i1 false
}

define void @test5() {
  unwind
}

define void @test6() {
entry:
	br label %unwind
unwind: unwind_to %unwind
  unwind
}
