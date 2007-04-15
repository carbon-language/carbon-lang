; RUN: llvm-upgrade < %s | llvm-as | opt -simplifycfg | llvm-dis
; END.
; ModuleID = 'bugpoint-tooptimize.bc'
target datalayout = "e-p:32:32"
target endian = little
target pointersize = 32
target triple = "i686-pc-linux-gnu"
        %struct.FILE = type { int, sbyte*, sbyte*, sbyte*, sbyte*, sbyte*, sbyte*, sbyte*, sbyte*, sbyte*, sbyte*, sbyte*, %struct._IO_marker*, %struct.FILE*, int, int, int, ushort, sbyte, [1 x sbyte], sbyte*, long, sbyte*, sbyte*, sbyte*, sbyte*, uint, int, [40 x sbyte] }
        %struct._IO_FILE = type { int, sbyte*, sbyte*, sbyte*, sbyte*, sbyte*, sbyte*, sbyte*, sbyte*, sbyte*, sbyte*, sbyte*, %struct._IO_marker*, %struct.FILE*, int, int, int, ushort, sbyte, [1 x sbyte], sbyte*, long, sbyte*, sbyte*, sbyte*, sbyte*, uint, int, [40 x sbyte] }
        %struct._IO_marker = type { %struct._IO_marker*, %struct.FILE*, int }
        %struct.charsequence = type { sbyte*, uint, uint }
        %struct.trie_s = type { [26 x %struct.trie_s*], int }
%str = external global [14 x sbyte]             ; <[14 x sbyte]*> [#uses=0]
%str = external global [32 x sbyte]             ; <[32 x sbyte]*> [#uses=0]
%str = external global [12 x sbyte]             ; <[12 x sbyte]*> [#uses=0]
%C.0.2294 = external global %struct.charsequence                ; <%struct.charsequence*> [#uses=3]
%t = external global %struct.trie_s*            ; <%struct.trie_s**> [#uses=0]
%str = external global [3 x sbyte]              ; <[3 x sbyte]*> [#uses=0]
%str = external global [26 x sbyte]             ; <[26 x sbyte]*> [#uses=0]

implementation   ; Functions:

declare void %charsequence_reset(%struct.charsequence*)
declare void %free(sbyte*)
declare void %charsequence_push(%struct.charsequence*, sbyte)
declare sbyte* %charsequence_val(%struct.charsequence*)
declare int %_IO_getc(%struct.FILE*)
declare int %tolower(int)
declare %struct.trie_s* %trie_insert(%struct.trie_s*, sbyte*)
declare int %feof(%struct.FILE*)

void %addfile(%struct.trie_s* %t, %struct.FILE* %f) {
entry:
        %t_addr = alloca %struct.trie_s*                ; <%struct.trie_s**> [#uses=2]
        %f_addr = alloca %struct.FILE*          ; <%struct.FILE**> [#uses=3]
        %c = alloca sbyte, align 1              ; <sbyte*> [#uses=7]
        %wstate = alloca int, align 4           ; <int*> [#uses=4]
        %cs = alloca %struct.charsequence, align 16             ; <%struct.charsequence*> [#uses=7]
        %str = alloca sbyte*, align 4           ; <sbyte**> [#uses=3]
        "alloca point" = bitcast int 0 to int           ; <int> [#uses=0]
        store %struct.trie_s* %t, %struct.trie_s** %t_addr
        store %struct.FILE* %f, %struct.FILE** %f_addr
        store int 0, int* %wstate
        %tmp = getelementptr %struct.charsequence* %cs, uint 0, uint 0          ; <sbyte**> [#uses=1]
        %tmp1 = getelementptr %struct.charsequence* %C.0.2294, uint 0, uint 0           ; <sbyte**> [#uses=1]
        %tmp = load sbyte** %tmp1               ; <sbyte*> [#uses=1]
        store sbyte* %tmp, sbyte** %tmp
        %tmp = getelementptr %struct.charsequence* %cs, uint 0, uint 1          ; <uint*> [#uses=1]
        %tmp2 = getelementptr %struct.charsequence* %C.0.2294, uint 0, uint 1           ; <uint*> [#uses=1]
        %tmp = load uint* %tmp2         ; <uint> [#uses=1]
        store uint %tmp, uint* %tmp
        %tmp3 = getelementptr %struct.charsequence* %cs, uint 0, uint 2         ; <uint*> [#uses=1]
        %tmp4 = getelementptr %struct.charsequence* %C.0.2294, uint 0, uint 2           ; <uint*> [#uses=1]
        %tmp5 = load uint* %tmp4                ; <uint> [#uses=1]
        store uint %tmp5, uint* %tmp3
        br label %bb33

bb:             ; preds = %bb33
        %tmp = load %struct.FILE** %f_addr              ; <%struct.FILE*> [#uses=1]
        %tmp = call int %_IO_getc( %struct.FILE* %tmp )         ; <int> [#uses=1]
        %tmp6 = call int %tolower( int %tmp )           ; <int> [#uses=1]
        %tmp6 = trunc int %tmp6 to sbyte                ; <sbyte> [#uses=1]
        store sbyte %tmp6, sbyte* %c
        %tmp7 = load int* %wstate               ; <int> [#uses=1]
        %tmp = icmp ne int %tmp7, 0             ; <bool> [#uses=1]
        br bool %tmp, label %cond_true, label %cond_false

cond_true:              ; preds = %bb
        %tmp = load sbyte* %c           ; <sbyte> [#uses=1]
        %tmp8 = icmp sle sbyte %tmp, 96         ; <bool> [#uses=1]
        br bool %tmp8, label %cond_true9, label %cond_next

cond_true9:             ; preds = %cond_true
        br label %bb16

cond_next:              ; preds = %cond_true
        %tmp10 = load sbyte* %c         ; <sbyte> [#uses=1]
        %tmp11 = icmp sgt sbyte %tmp10, 122             ; <bool> [#uses=1]
        br bool %tmp11, label %cond_true12, label %cond_next13

cond_true12:            ; preds = %cond_next
        br label %bb16

cond_next13:            ; preds = %cond_next
        %tmp14 = load sbyte* %c         ; <sbyte> [#uses=1]
        %tmp14 = sext sbyte %tmp14 to int               ; <int> [#uses=1]
        %tmp1415 = trunc int %tmp14 to sbyte            ; <sbyte> [#uses=1]
        call void %charsequence_push( %struct.charsequence* %cs, sbyte %tmp1415 )
        br label %bb21

bb16:           ; preds = %cond_true12, %cond_true9
        %tmp17 = call sbyte* %charsequence_val( %struct.charsequence* %cs )             ; <sbyte*> [#uses=1]
        store sbyte* %tmp17, sbyte** %str
        %tmp = load %struct.trie_s** %t_addr            ; <%struct.trie_s*> [#uses=1]
        %tmp18 = load sbyte** %str              ; <sbyte*> [#uses=1]
        %tmp19 = call %struct.trie_s* %trie_insert( %struct.trie_s* %tmp, sbyte* %tmp18 )               ; <%struct.trie_s*> [#uses=0]
        %tmp20 = load sbyte** %str              ; <sbyte*> [#uses=1]
        call void %free( sbyte* %tmp20 )
        store int 0, int* %wstate
        br label %bb21

bb21:           ; preds = %bb16, %cond_next13
        br label %cond_next32

cond_false:             ; preds = %bb
        %tmp22 = load sbyte* %c         ; <sbyte> [#uses=1]
        %tmp23 = icmp sgt sbyte %tmp22, 96              ; <bool> [#uses=1]
        br bool %tmp23, label %cond_true24, label %cond_next31

cond_true24:            ; preds = %cond_false
        %tmp25 = load sbyte* %c         ; <sbyte> [#uses=1]
        %tmp26 = icmp sle sbyte %tmp25, 122             ; <bool> [#uses=1]
        br bool %tmp26, label %cond_true27, label %cond_next30

cond_true27:            ; preds = %cond_true24
        call void %charsequence_reset( %struct.charsequence* %cs )
        %tmp28 = load sbyte* %c         ; <sbyte> [#uses=1]
        %tmp28 = sext sbyte %tmp28 to int               ; <int> [#uses=1]
        %tmp2829 = trunc int %tmp28 to sbyte            ; <sbyte> [#uses=1]
        call void %charsequence_push( %struct.charsequence* %cs, sbyte %tmp2829 )
        store int 1, int* %wstate
        br label %cond_next30

cond_next30:            ; preds = %cond_true27, %cond_true24
        br label %cond_next31

cond_next31:            ; preds = %cond_next30, %cond_false
        br label %cond_next32

cond_next32:            ; preds = %cond_next31, %bb21
        br label %bb33

bb33:           ; preds = %cond_next32, %entry
        %tmp34 = load %struct.FILE** %f_addr            ; <%struct.FILE*> [#uses=1]
        %tmp35 = call int %feof( %struct.FILE* %tmp34 )         ; <int> [#uses=1]
        %tmp36 = icmp eq int %tmp35, 0          ; <bool> [#uses=1]
        br bool %tmp36, label %bb, label %bb37

bb37:           ; preds = %bb33
        br label %return

return:         ; preds = %bb37
        ret void
}

