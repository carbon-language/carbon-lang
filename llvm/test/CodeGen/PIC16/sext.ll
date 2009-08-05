; RUN: llvm-as < %s | llc -march=pic16

@main.auto.c = internal global i8 0		; <i8*> [#uses=1]

define i16 @main() nounwind {
entry:
	%tmp = load i8* @main.auto.c		; <i8> [#uses=1]
	%conv = sext i8 %tmp to i16		; <i16> [#uses=1]
	ret i16 %conv
}
