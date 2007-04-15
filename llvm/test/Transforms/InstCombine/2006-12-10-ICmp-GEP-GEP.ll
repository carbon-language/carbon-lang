; RUN: llvm-upgrade < %s | llvm-as | opt -instcombine | llvm-dis | \
; RUN:    grep -v {icmp ult int}
; END.

; ModuleID = 'good.bc'
target datalayout = "e-p:32:32"
target endian = little
target pointersize = 32
target triple = "i686-pc-linux-gnu"
        %struct.edgeBox = type { short, short, short, short, short, short }
%qsz = external global int              ; <int*> [#uses=12]
%thresh = external global int           ; <int*> [#uses=2]
%mthresh = external global int          ; <int*> [#uses=1]

implementation   ; Functions:

int %qsorte(sbyte* %base, int %n, int %size) {
entry:
        %tmp = setgt int %n, 1          ; <bool> [#uses=1]
        br bool %tmp, label %cond_next, label %return

cond_next:              ; preds = %entry
        store int %size, int* %qsz
        %tmp3 = shl int %size, ubyte 2          ; <int> [#uses=1]
        store int %tmp3, int* %thresh
        %tmp4 = load int* %qsz          ; <int> [#uses=1]
        %tmp5 = mul int %tmp4, 6                ; <int> [#uses=1]
        store int %tmp5, int* %mthresh
        %tmp6 = load int* %qsz          ; <int> [#uses=1]
        %tmp8 = mul int %tmp6, %n               ; <int> [#uses=1]
        %tmp9 = getelementptr sbyte* %base, int %tmp8           ; <sbyte*> [#uses=3]
        %tmp11 = setgt int %n, 3                ; <bool> [#uses=1]
        br bool %tmp11, label %cond_true12, label %bb30

cond_true12:            ; preds = %cond_next
        %tmp156 = call int %qste( sbyte* %base, sbyte* %tmp9 )          ; <int> [#uses=0]
        %tmp16 = load int* %thresh              ; <int> [#uses=1]
        %tmp18 = getelementptr sbyte* %base, int %tmp16         ; <sbyte*> [#uses=2]
        %tmp3117 = load int* %qsz               ; <int> [#uses=1]
        %tmp3318 = getelementptr sbyte* %base, int %tmp3117             ; <sbyte*> [#uses=2]
        %tmp3621 = setlt sbyte* %tmp3318, %tmp18                ; <bool> [#uses=1]
        br bool %tmp3621, label %bb, label %bb37

bb:             ; preds = %bb30, %cond_true12
        %hi.0.0 = phi sbyte* [ %tmp18, %cond_true12 ], [ %hi.0, %bb30 ]         ; <sbyte*> [#uses=4]
        %j.1.0 = phi sbyte* [ %base, %cond_true12 ], [ %j.1, %bb30 ]            ; <sbyte*> [#uses=4]
        %tmp33.0 = phi sbyte* [ %tmp3318, %cond_true12 ], [ %tmp33, %bb30 ]             ; <sbyte*> [#uses=6]
        %tmp3 = bitcast sbyte* %j.1.0 to %struct.edgeBox*               ; <%struct.edgeBox*> [#uses=1]
        %tmp4 = bitcast sbyte* %tmp33.0 to %struct.edgeBox*             ; <%struct.edgeBox*> [#uses=1]
        %tmp255 = call int %comparee( %struct.edgeBox* %tmp3, %struct.edgeBox* %tmp4 )          ; <int> [#uses=1]
        %tmp26 = setgt int %tmp255, 0           ; <bool> [#uses=1]
        br bool %tmp26, label %cond_true27, label %bb30

cond_true27:            ; preds = %bb
        br label %bb30

bb30:           ; preds = %cond_true27, %bb, %cond_next
        %hi.0.3 = phi sbyte* [ %hi.0.0, %cond_true27 ], [ %hi.0.0, %bb ], [ undef, %cond_next ]         ; <sbyte*> [#uses=0]
        %j.1.3 = phi sbyte* [ %j.1.0, %cond_true27 ], [ %j.1.0, %bb ], [ undef, %cond_next ]            ; <sbyte*> [#uses=0]
        %tmp33.3 = phi sbyte* [ %tmp33.0, %cond_true27 ], [ %tmp33.0, %bb ], [ undef, %cond_next ]              ; <sbyte*> [#uses=0]
        %hi.0 = phi sbyte* [ %tmp9, %cond_next ], [ %hi.0.0, %bb ], [ %hi.0.0, %cond_true27 ]           ; <sbyte*> [#uses=2]
        %lo.1 = phi sbyte* [ %tmp33.0, %cond_true27 ], [ %tmp33.0, %bb ], [ %base, %cond_next ]         ; <sbyte*> [#uses=1]
        %j.1 = phi sbyte* [ %tmp33.0, %cond_true27 ], [ %j.1.0, %bb ], [ %base, %cond_next ]            ; <sbyte*> [#uses=2]
        %tmp31 = load int* %qsz         ; <int> [#uses=1]
        %tmp33 = getelementptr sbyte* %lo.1, int %tmp31         ; <sbyte*> [#uses=2]
        %tmp36 = setlt sbyte* %tmp33, %hi.0             ; <bool> [#uses=1]
        br bool %tmp36, label %bb, label %bb37

bb37:           ; preds = %bb30, %cond_true12
        %j.1.1 = phi sbyte* [ %j.1, %bb30 ], [ %base, %cond_true12 ]            ; <sbyte*> [#uses=4]
        %tmp40 = seteq sbyte* %j.1.1, %base             ; <bool> [#uses=1]
        br bool %tmp40, label %bb115, label %cond_true41

cond_true41:            ; preds = %bb37
        %tmp43 = load int* %qsz         ; <int> [#uses=1]
        %tmp45 = getelementptr sbyte* %base, int %tmp43         ; <sbyte*> [#uses=2]
        %tmp6030 = setlt sbyte* %base, %tmp45           ; <bool> [#uses=1]
        br bool %tmp6030, label %bb46, label %bb115

bb46:           ; preds = %bb46, %cond_true41
        %j.2.0 = phi sbyte* [ %j.1.1, %cond_true41 ], [ %tmp52, %bb46 ]         ; <sbyte*> [#uses=3]
        %i.2.0 = phi sbyte* [ %base, %cond_true41 ], [ %tmp56, %bb46 ]          ; <sbyte*> [#uses=3]
        %tmp = load sbyte* %j.2.0               ; <sbyte> [#uses=2]
        %tmp49 = load sbyte* %i.2.0             ; <sbyte> [#uses=1]
        store sbyte %tmp49, sbyte* %j.2.0
        %tmp52 = getelementptr sbyte* %j.2.0, int 1             ; <sbyte*> [#uses=2]
        store sbyte %tmp, sbyte* %i.2.0
        %tmp56 = getelementptr sbyte* %i.2.0, int 1             ; <sbyte*> [#uses=3]
        %tmp60 = setlt sbyte* %tmp56, %tmp45            ; <bool> [#uses=1]
        br bool %tmp60, label %bb46, label %bb115

bb66:           ; preds = %bb115, %bb66
        %hi.3 = phi sbyte* [ %tmp118, %bb115 ], [ %tmp70, %bb66 ]               ; <sbyte*> [#uses=2]
        %tmp67 = load int* %qsz         ; <int> [#uses=2]
        %tmp68 = sub int 0, %tmp67              ; <int> [#uses=1]
        %tmp70 = getelementptr sbyte* %hi.3, int %tmp68         ; <sbyte*> [#uses=2]
        %tmp = bitcast sbyte* %tmp70 to %struct.edgeBox*                ; <%struct.edgeBox*> [#uses=1]
        %tmp1 = bitcast sbyte* %tmp118 to %struct.edgeBox*              ; <%struct.edgeBox*> [#uses=1]
        %tmp732 = call int %comparee( %struct.edgeBox* %tmp, %struct.edgeBox* %tmp1 )           ; <int> [#uses=1]
        %tmp74 = setgt int %tmp732, 0           ; <bool> [#uses=1]
        br bool %tmp74, label %bb66, label %bb75

bb75:           ; preds = %bb66
        %tmp76 = load int* %qsz         ; <int> [#uses=1]
        %tmp70.sum = sub int %tmp76, %tmp67             ; <int> [#uses=1]
        %tmp78 = getelementptr sbyte* %hi.3, int %tmp70.sum             ; <sbyte*> [#uses=3]
        %tmp81 = seteq sbyte* %tmp78, %tmp118           ; <bool> [#uses=1]
        br bool %tmp81, label %bb115, label %cond_true82

cond_true82:            ; preds = %bb75
        %tmp83 = load int* %qsz         ; <int> [#uses=1]
        %tmp118.sum = add int %tmp116, %tmp83           ; <int> [#uses=1]
        %tmp85 = getelementptr sbyte* %min.1, int %tmp118.sum           ; <sbyte*> [#uses=1]
        %tmp10937 = getelementptr sbyte* %tmp85, int -1         ; <sbyte*> [#uses=3]
        %tmp11239 = setlt sbyte* %tmp10937, %tmp118             ; <bool> [#uses=1]
        br bool %tmp11239, label %bb115, label %bb86

bb86:           ; preds = %bb104, %cond_true82
        %tmp109.0 = phi sbyte* [ %tmp10937, %cond_true82 ], [ %tmp109, %bb104 ]         ; <sbyte*> [#uses=5]
        %i.5.2 = phi sbyte* [ %i.5.3, %cond_true82 ], [ %i.5.1, %bb104 ]                ; <sbyte*> [#uses=0]
        %tmp100.2 = phi sbyte* [ %tmp100.3, %cond_true82 ], [ %tmp100.1, %bb104 ]               ; <sbyte*> [#uses=0]
        %tmp88 = load sbyte* %tmp109.0          ; <sbyte> [#uses=2]
        %tmp9746 = load int* %qsz               ; <int> [#uses=1]
        %tmp9847 = sub int 0, %tmp9746          ; <int> [#uses=1]
        %tmp10048 = getelementptr sbyte* %tmp109.0, int %tmp9847                ; <sbyte*> [#uses=3]
        %tmp10350 = setlt sbyte* %tmp10048, %tmp78              ; <bool> [#uses=1]
        br bool %tmp10350, label %bb104, label %bb91

bb91:           ; preds = %bb91, %bb86
        %i.5.0 = phi sbyte* [ %tmp109.0, %bb86 ], [ %tmp100.0, %bb91 ]          ; <sbyte*> [#uses=1]
        %tmp100.0 = phi sbyte* [ %tmp10048, %bb86 ], [ %tmp100, %bb91 ]         ; <sbyte*> [#uses=4]
        %tmp93 = load sbyte* %tmp100.0          ; <sbyte> [#uses=1]
        store sbyte %tmp93, sbyte* %i.5.0
        %tmp97 = load int* %qsz         ; <int> [#uses=1]
        %tmp98 = sub int 0, %tmp97              ; <int> [#uses=1]
        %tmp100 = getelementptr sbyte* %tmp100.0, int %tmp98            ; <sbyte*> [#uses=3]
        %tmp103 = setlt sbyte* %tmp100, %tmp78          ; <bool> [#uses=1]
        br bool %tmp103, label %bb104, label %bb91

bb104:          ; preds = %bb91, %bb86
        %i.5.1 = phi sbyte* [ %tmp109.0, %bb86 ], [ %tmp100.0, %bb91 ]          ; <sbyte*> [#uses=4]
        %tmp100.1 = phi sbyte* [ %tmp10048, %bb86 ], [ %tmp100, %bb91 ]         ; <sbyte*> [#uses=3]
        store sbyte %tmp88, sbyte* %i.5.1
        %tmp109 = getelementptr sbyte* %tmp109.0, int -1                ; <sbyte*> [#uses=3]
        %tmp112 = setlt sbyte* %tmp109, %tmp118         ; <bool> [#uses=1]
        br bool %tmp112, label %bb115, label %bb86

bb115:          ; preds = %bb104, %cond_true82, %bb75, %bb46, %cond_true41, %bb37
        %tmp109.1 = phi sbyte* [ undef, %bb37 ], [ %tmp109.1, %bb75 ], [ %tmp10937, %cond_true82 ], [ %tmp109, %bb104 ], [ undef, %bb46 ], [ undef, %cond_true41 ]              ; <sbyte*> [#uses=1]
        %i.5.3 = phi sbyte* [ undef, %bb37 ], [ %i.5.3, %bb75 ], [ %i.5.3, %cond_true82 ], [ %i.5.1, %bb104 ], [ undef, %bb46 ], [ undef, %cond_true41 ]                ; <sbyte*> [#uses=3]
        %tmp100.3 = phi sbyte* [ undef, %bb37 ], [ %tmp100.3, %bb75 ], [ %tmp100.3, %cond_true82 ], [ %tmp100.1, %bb104 ], [ undef, %bb46 ], [ undef, %cond_true41 ]            ; <sbyte*> [#uses=3]
        %min.1 = phi sbyte* [ %tmp118, %bb104 ], [ %tmp118, %bb75 ], [ %base, %bb37 ], [ %base, %bb46 ], [ %base, %cond_true41 ], [ %tmp118, %cond_true82 ]             ; <sbyte*> [#uses=2]
        %j.5 = phi sbyte* [ %tmp100.1, %bb104 ], [ %j.5, %bb75 ], [ %tmp52, %bb46 ], [ %j.1.1, %bb37 ], [ %j.1.1, %cond_true41 ], [ %j.5, %cond_true82 ]                ; <sbyte*> [#uses=2]
        %i.4 = phi sbyte* [ %i.5.1, %bb104 ], [ %i.4, %bb75 ], [ %tmp56, %bb46 ], [ undef, %bb37 ], [ %base, %cond_true41 ], [ %i.4, %cond_true82 ]             ; <sbyte*> [#uses=2]
        %c.4 = phi sbyte [ %tmp88, %bb104 ], [ %c.4, %bb75 ], [ %tmp, %bb46 ], [ undef, %bb37 ], [ undef, %cond_true41 ], [ %c.4, %cond_true82 ]                ; <sbyte> [#uses=2]
        %tmp116 = load int* %qsz                ; <int> [#uses=2]
        %tmp118 = getelementptr sbyte* %min.1, int %tmp116              ; <sbyte*> [#uses=9]
        %tmp122 = setlt sbyte* %tmp118, %tmp9           ; <bool> [#uses=1]
        br bool %tmp122, label %bb66, label %return

return:         ; preds = %bb115, %entry
        ret int undef
}

declare int %qste(sbyte*, sbyte*)

declare int %comparee(%struct.edgeBox*, %struct.edgeBox*)
