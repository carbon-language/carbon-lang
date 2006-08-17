; RUN: llvm-as < %s | llc -march=x86

	%struct.expr = type { %struct.rtx_def*, int, %struct.expr*, %struct.occr*, %struct.occr*, %struct.rtx_def* }
	%struct.hash_table = type { %struct.expr**, uint, uint, int }
	%struct.occr = type { %struct.occr*, %struct.rtx_def*, sbyte, sbyte }
	%struct.rtx_def = type { ushort, ubyte, ubyte, %struct.u }
	%struct.u = type { [1 x long] }

void %test() {
	%tmp = load uint* null		; <uint> [#uses=1]
	%tmp8 = call uint %hash_rtx( )		; <uint> [#uses=1]
	%tmp11 = rem uint %tmp8, %tmp		; <uint> [#uses=1]
	br bool false, label %cond_next, label %return

cond_next:		; preds = %entry
	%tmp17 = getelementptr %struct.expr** null, uint %tmp11		; <%struct.expr**> [#uses=0]
	ret void

return:		; preds = %entry
	ret void
}

declare uint %hash_rtx()
