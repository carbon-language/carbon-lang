; Make sure this testcase codegens to the ctlz instruction
; RUN: llvm-as < %s | llc -march=alpha -mcpu=ev67 | grep -i ctlz
; RUN: llvm-as < %s | llc -march=alpha -mattr=+CIX | grep -i ctlz
; RUN: llvm-as < %s | llc -march=alpha -mcpu=ev6 | not grep -i ctlz
; RUN: llvm-as < %s | llc -march=alpha -mcpu=ev56 | not grep -i ctlz
; RUN: llvm-as < %s | llc -march=alpha -mattr=-CIX | not grep -i ctlz

declare i32 @llvm.ctlz.i8(i8)

define i32 @bar(i8 %x) {
entry:
	%tmp.1 = call i32 @llvm.ctlz.i8( i8 %x ) 
	ret i32 %tmp.1
}
