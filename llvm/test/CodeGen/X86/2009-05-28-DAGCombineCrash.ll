; RUN: llc < %s -march=x86-64

	%struct.tempsym_t = type { i8*, i8*, i8*, i8*, i32, i32, i32, i32, i32 }

define fastcc signext i8 @S_next_symbol(%struct.tempsym_t* %symptr) nounwind ssp {
entry:
	br label %bb14

bb14:		; preds = %bb
	%srcval16 = load i448* null, align 8		; <i448> [#uses=1]
	%tmp = zext i32 undef to i448		; <i448> [#uses=1]
	%tmp15 = shl i448 %tmp, 288		; <i448> [#uses=1]
	%mask = and i448 %srcval16, -2135987035423586845985235064014169866455883682256196619149693890381755748887481053010428711403521		; <i448> [#uses=1]
	%ins = or i448 %tmp15, %mask		; <i448> [#uses=1]
	store i448 %ins, i448* null, align 8
	ret i8 1
}
