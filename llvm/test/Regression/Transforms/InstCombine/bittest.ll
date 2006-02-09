; RUN: llvm-as < %s | opt -instcombine -simplifycfg -disable-output &&
; RUN: llvm-as < %s | opt -instcombine -simplifycfg | llvm-dis | not grep 'call void %abort'

%b_rec.0 = external global int

void %_Z12h000007_testv(uint *%P) {
entry:
        %tmp.2 = load int* %b_rec.0             ; <int> [#uses=1]
        %tmp.9 = or int %tmp.2, -989855744              ; <int> [#uses=2]
        %tmp.16 = and int %tmp.9, -805306369            ; <int> [#uses=2]
        %tmp.17 = and int %tmp.9, -973078529            ; <int> [#uses=1]
        store int %tmp.17, int* %b_rec.0
        %tmp.17.shrunk = cast int %tmp.16 to uint               ; <uint> [#uses=1]
        %tmp.22 = and uint %tmp.17.shrunk, 3221225472           ; <uint> [#uses=1]
        %tmp.23 = seteq uint %tmp.22, 3221225472                ; <bool> [#uses=1]
        br bool %tmp.23, label %endif.0, label %then.0

then.0:         ; preds = %entry
        tail call void %abort( )
        unreachable

endif.0:                ; preds = %entry
        %tmp.17.shrunk2 = cast int %tmp.16 to uint              ; <uint> [#uses=1]
        %tmp.27.mask = and uint %tmp.17.shrunk2, 100663295              ; <uint> [#uses=1]
 	store uint %tmp.27.mask, uint* %P
        ret void
}

declare void %abort()
