; RUN:  llvm-as < %s | opt -adce

void "test"()
begin
	br label %BB3

BB3:
	br label %BB3
end
