; RUN: llvm-as < %s | llc -march=ppc32 | grep 'stw r3, 32751' &&
; RUN: llvm-as < %s | llc -march=ppc64 | grep 'stw r3, 32751' &&
; RUN: llvm-as < %s | llc

define void @test() {
	store i32 0, i32* inttoptr (i64 48725999 to i32*)
	ret void
}

