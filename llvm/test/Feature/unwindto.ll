; RUN: llvm-as < %s | llvm-dis | llvm-as -disable-output
; PR1269
; END
; http://nondot.org/sabre/LLVMNotes/ExceptionHandlingChanges.txt

define i1 @test1(i8 %i, i8 %j) {
entry: unwinds to %target
	%tmp = sub i8 %i, %j		; <i8> [#uses=1]
	%b = icmp eq i8 %tmp, 0		; <i1> [#uses=1]
	ret i1 %b
target:
	ret i1 false
}

define i1 @test2(i8 %i, i8 %j) {
entry:
	br label %0
unwinds to %1
	%tmp = sub i8 %i, %j		; <i8> [#uses=1]
	%b = icmp eq i8 %tmp, 0		; <i1> [#uses=1]
	ret i1 %b
		; No predecessors!
	ret i1 false
}

define i1 @test3(i8 %i, i8 %j) {
entry:
	br label %0
unwinds to %1
	%tmp = sub i8 %i, %j		; <i8> [#uses=1]
	%b = icmp eq i8 %tmp, 0		; <i1> [#uses=1]
	ret i1 %b
unwinds to %0
	ret i1 false
}

define i1 @test4(i8 %i, i8 %j) {
	%tmp = sub i8 %i, %j		; <i8> [#uses=1]
	%b = icmp eq i8 %tmp, 0		; <i1> [#uses=1]
	br label %1
unwinds to %1
	ret i1 false
}

define void @test5() {
  unwind
}

define void @test6() {
entry:
	br label %unwind
unwind: unwinds to %unwind
  unwind
}

define i8 @test7(i1 %b) {
entry: unwinds to %cleanup
  br i1 %b, label %cond_true, label %cond_false
cond_true: unwinds to %cleanup
  br label %cleanup
cond_false: unwinds to %cleanup
  br label %cleanup
cleanup:
  %x = phi i8 [0, %entry], [1, %cond_true], [1, %cond_true],
                           [2, %cond_false], [2, %cond_false]
  ret i8 %x
}
