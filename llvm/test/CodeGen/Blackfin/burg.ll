; RUN: llc < %s -march=bfin -verify-machineinstrs > %t

	%IntList = type %struct.intlist*
	%ReadFn = type i32 ()*
	%YYSTYPE = type { %IntList }
	%struct.intlist = type { i32, %IntList }
@yyval = external global %YYSTYPE		; <%YYSTYPE*> [#uses=1]

define i32 @yyparse() {
bb0:
	%reg254 = load i16* null		; <i16> [#uses=1]
	%reg254-idxcast = sext i16 %reg254 to i64		; <i64> [#uses=1]
	%reg254-idxcast-scale = mul i64 %reg254-idxcast, -1		; <i64> [#uses=1]
	%reg254-idxcast-scale-offset = add i64 %reg254-idxcast-scale, 1		; <i64> [#uses=1]
	%reg261.idx1 = getelementptr %YYSTYPE* null, i64 %reg254-idxcast-scale-offset, i32 0		; <%IntList*> [#uses=1]
	%reg261 = load %IntList* %reg261.idx1		; <%IntList> [#uses=1]
	store %IntList %reg261, %IntList* getelementptr (%YYSTYPE* @yyval, i64 0, i32 0)
	unreachable
}
