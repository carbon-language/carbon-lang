; RUN: as < %s | opt -raise -debug -raise-start-inst=cast459

int %deflateInit2_({ ubyte*, uint, ulong, ubyte*, uint, ulong, sbyte*, { \4, int, ubyte*, ulong, ubyte*, int, int, ubyte, ubyte, int, uint, uint, uint, ubyte*, ulong, ushort*, ushort*, uint, uint, uint, uint, uint, long, uint, uint, int, uint, uint, uint, uint, uint, uint, int, int, uint, int, [573 x { { ushort }, { ushort } }], [61 x { { ushort }, { ushort } }], [39 x { { ushort }, { ushort } }], { { { ushort }, { ushort } }*, int, { int }* }, { { { ushort }, { ushort } }*, int, { int }* }, { { { ushort }, { ushort } }*, int, { int }* }, [16 x ushort], [573 x int], int, int, [573 x ubyte], ubyte*, uint, uint, ushort*, ulong, ulong, uint, int, ushort, int }*, sbyte* (sbyte*, uint, uint)*, void (sbyte*, sbyte*)*, sbyte*, int, ulong, ulong }* %strm, int %level, int %method, int %windowBits, int %memLevel, int %strategy, sbyte* %version, int %stream_size) {
bb0:            ; No predecessors!
        %reg107 = load { ubyte*, uint, ulong, ubyte*, uint, ulong, sbyte*, { \4, int, ubyte*, ulong, ubyte*, int, int, ubyte, ubyte, int, uint, uint, uint, ubyte*, ulong, ushort*, ushort*, uint, uint, uint, uint, uint, long, uint, uint, int, uint, uint, uint, uint, uint, uint, int, int, uint, int, [573 x { { ushort }, { ushort } }], [61 x { { ushort }, { ushort } }], [39 x { { ushort }, { ushort } }], { { { ushort }, { ushort } }*, int, { int }* }, { { { ushort }, { ushort } }*, int, { int }* }, { { { ushort }, { ushort } }*, int, { int }* }, [16 x ushort], [573 x int], int, int, [573 x ubyte], ubyte*, uint, uint, ushort*, ulong, ulong, uint, int, ushort, int }*, sbyte* (sbyte*, uint, uint)*, void (sbyte*, sbyte*)*, sbyte*, int, ulong, ulong }** null ; <{ ubyte*, uint, ulong, ubyte*, uint, ulong, sbyte*, { \4, int, ubyte*, ulong, ubyte*, int, int, ubyte, ubyte, int, uint, uint, uint, ubyte*, ulong, ushort*, ushort*, uint, uint, uint, uint, uint, long, uint, uint, int, uint, uint, uint, uint, uint, uint, int, int, uint, int, [573 x { { ushort }, { ushort } }], [61 x { { ushort }, { ushort } }], [39 x { { ushort }, { ushort } }], { { { ushort }, { ushort } }*, int, { int }* }, { { { ushort }, { ushort } }*, int, { int }* }, { { { ushort }, { ushort } }*, int, { int }* }, [16 x ushort], [573 x int], int, int, [573 x ubyte], ubyte*, uint, uint, ushort*, ulong, ulong, uint, int, ushort, int }*, sbyte* (sbyte*, uint, uint)*, void (sbyte*, sbyte*)*, sbyte*, int, ulong, ulong }*> [#uses=2]
        br bool false, label %bb5, label %UnifiedExitNode

bb5:            ; preds = %bb0
        br bool false, label %bb22, label %UnifiedExitNode

bb22:           ; preds = %bb5
        br bool false, label %bb24, label %UnifiedExitNode

bb24:           ; preds = %bb22
        %reg399 = getelementptr { ubyte*, uint, ulong, ubyte*, uint, ulong, sbyte*, { \4, int, ubyte*, ulong, ubyte*, int, int, ubyte, ubyte, int, uint, uint, uint, ubyte*, ulong, ushort*, ushort*, uint, uint, uint, uint, uint, long, uint, uint, int, uint, uint, uint, uint, uint, uint, int, int, uint, int, [573 x { { ushort }, { ushort } }], [61 x { { ushort }, { ushort } }], [39 x { { ushort }, { ushort } }], { { { ushort }, { ushort } }*, int, { int }* }, { { { ushort }, { ushort } }*, int, { int }* }, { { { ushort }, { ushort } }*, int, { int }* }, [16 x ushort], [573 x int], int, int, [573 x ubyte], ubyte*, uint, uint, ushort*, ulong, ulong, uint, int, ushort, int }*, sbyte* (sbyte*, uint, uint)*, void (sbyte*, sbyte*)*, sbyte*, int, ulong, ulong }* %reg107, long 0, ubyte 8              ; <sbyte* (sbyte*, uint, uint)**> [#uses=1]
        %reg137 = load sbyte* (sbyte*, uint, uint)** %reg399            ; <sbyte* (sbyte*, uint, uint)*> [#uses=1]
        %reg402 = call sbyte* %reg137( sbyte* null, uint 0, uint 0 )            ; <sbyte*> [#uses=1]
        br bool false, label %bb26, label %UnifiedExitNode

bb26:           ; preds = %bb24
        %reg457 = getelementptr sbyte* %reg402, long 0          ; <sbyte*> [#uses=1]
        %cast459 = cast sbyte* %reg457 to ubyte*                ; <ubyte*> [#uses=1]
        %reg146 = load ubyte* %cast459          ; <ubyte> [#uses=1]
        %reg145 = shl uint 0, ubyte %reg146             ; <uint> [#uses=1]
        store uint %reg145, uint* null
        %reg647 = call int %deflateEnd( { ubyte*, uint, ulong, ubyte*, uint, ulong, sbyte*, { \4, int, ubyte*, ulong, ubyte*, int, int, ubyte, ubyte, int, uint, uint, uint, ubyte*, ulong, ushort*, ushort*, uint, uint, uint, uint, uint, long, uint, uint, int, uint, uint, uint, uint, uint, uint, int, int, uint, int, [573 x { { ushort }, { ushort } }], [61 x { { ushort }, { ushort } }], [39 x { { ushort }, { ushort } }], { { { ushort }, { ushort } }*, int, { int }* }, { { { ushort }, { ushort } }*, int, { int }* }, { { { ushort }, { ushort } }*, int, { int }* }, [16 x ushort], [573 x int], int, int, [573 x ubyte], ubyte*, uint, uint, ushort*, ulong, ulong, uint, int, ushort, int }*, sbyte* (sbyte*, uint, uint)*, void (sbyte*, sbyte*)*, sbyte*, int, ulong, ulong }* %reg107 )             ; <int> [#uses=0]
        br label %UnifiedExitNode

UnifiedExitNode:                ; preds = %bb26, %bb24, %bb22, %bb5, %bb0
        ret int 0
}

declare int %deflateEnd({ ubyte*, uint, ulong, ubyte*, uint, ulong, sbyte*, { \4, int, ubyte*, ulong, ubyte*, int, int, ubyte, ubyte, int, uint, uint, uint, ubyte*, ulong, ushort*, ushort*, uint, uint, uint, uint, uint, long, uint, uint, int, uint, uint, uint, uint, uint, uint, int, int, uint, int, [573 x { { ushort }, { ushort } }], [61 x { { ushort }, { ushort } }], [39 x { { ushort }, { ushort } }], { { { ushort }, { ushort } }*, int, { int }* }, { { { ushort }, { ushort } }*, int, { int }* }, { { { ushort }, { ushort } }*, int, { int }* }, [16 x ushort], [573 x int], int, int, [573 x ubyte], ubyte*, uint, uint, ushort*, ulong, ulong, uint, int, ushort, int }*, sbyte* (sbyte*, uint, uint)*, void (sbyte*, sbyte*)*, sbyte*, int, ulong, ulong }*)

