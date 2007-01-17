; RUN: llvm-upgrade < %s | llvm-as -o /dev/null -f

; It looks like the assembler is not forward resolving the function declaraion
; correctly.

void "test"()
begin
	call void %foo()
	ret void
end

declare void "foo"()

