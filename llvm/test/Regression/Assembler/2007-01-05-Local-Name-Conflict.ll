; RUN: llvm-upgrade < %s | llvm-as -o /dev/null -f

implementation   ; Functions:

void %interpret() {
entry:
        %x = bitcast sbyte  1 to sbyte
        %x = bitcast ubyte  1 to ubyte
        %x = bitcast short  1 to short
        %x = bitcast ushort 1 to ushort
        %x = bitcast int    1 to int
        %x = bitcast uint   1 to uint
        %x = bitcast ulong  1 to ulong
        %x = inttoptr ulong %x to sbyte*
        %tmp = inttoptr ulong %x to float*
        %tmp7360 = bitcast ubyte %x to sbyte
        %tmp7361 = sub ubyte 0, %tmp7360            
        br label %next

next:		; preds = %cond_false165, %cond_true163
	%index.0 = phi uint [ undef, %entry ], [ %index.0, %next ]
        br label %next
}
