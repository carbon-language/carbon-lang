; RUN: llvm-as < %s | opt -adce -simplifycfg | llvm-dis

int "Test"(int %A, int %B) {
BB1:
	br label %BB4
BB2:
	br label %BB3
BB3:
	%ret = phi int [%X, %BB4], [%B, %BB2]
	ret int %ret
BB4:
	%X = phi int [%A, %BB1]
	br label %BB3
}
