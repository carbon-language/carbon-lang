; RUN: llvm-as < %s | llc

       %struct.FILE = type { ubyte*, int, int, short, short, %struct.__sbuf, int, sbyte*, int (sbyte*)*, int (sbyte*, sbyte*, int)*, long (sbyte*, long, int)*, int (sbyte*, sbyte*, int)*, %struct.__sbuf, %struct.__sFILEX*, int, [3 x ubyte], [1 x ubyte], %struct.__sbuf, int, long }
        %struct.SYMBOL_TABLE_ENTRY = type { [9 x sbyte], [9 x sbyte], int, int, uint, %struct.SYMBOL_TABLE_ENTRY* }
        %struct.__sFILEX = type opaque
        %struct.__sbuf = type { ubyte*, int }
%str14 = external global [6 x sbyte]            ; <[6 x sbyte]*> [#uses=0]

implementation   ; Functions:

declare void %fprintf(int, ...)

void %OUTPUT_TABLE(%struct.SYMBOL_TABLE_ENTRY* %SYM_TAB) {
entry:
        %tmp11 = getelementptr %struct.SYMBOL_TABLE_ENTRY* %SYM_TAB, int 0, uint 1, int 0               ; <sbyte*> [#uses=2]
        %tmp.i = cast sbyte* %tmp11 to ubyte*           ; <ubyte*> [#uses=1]
        br label %bb.i

bb.i:           ; preds = %cond_next.i, %entry
        %s1.0.i = phi ubyte* [ %tmp.i, %entry ], [ null, %cond_next.i ]         ; <ubyte*> [#uses=0]
        br bool false, label %cond_true.i31, label %cond_next.i

cond_true.i31:          ; preds = %bb.i
        call void (int, ...)* %fprintf( int 0, sbyte* %tmp11, sbyte* null )
        ret void

cond_next.i:            ; preds = %bb.i
        br bool false, label %bb.i, label %bb19.i

bb19.i:         ; preds = %cond_next.i
        ret void
}

