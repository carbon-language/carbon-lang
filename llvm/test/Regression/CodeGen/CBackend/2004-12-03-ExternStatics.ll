; RUN: llvm-as < %s | llc -march=c | not grep 'extern.*msg'

; This is PR472

%msg = internal global [6 x sbyte] c"hello\00"

implementation   ; Functions:

sbyte* %foo() {
entry:
	ret sbyte* getelementptr ([6 x sbyte]* %msg, int 0, int 0)
}
