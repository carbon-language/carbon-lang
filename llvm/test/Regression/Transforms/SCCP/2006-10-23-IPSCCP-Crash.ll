; RUN: llvm-upgrade < %s | llvm-as | opt -sccp -disable-output

target endian = big
target pointersize = 32
target triple = "powerpc-apple-darwin8.7.0"
        %struct.pat_list = type { int, %struct.pat_list* }
%JUMP = external global int             ; <int*> [#uses=1]
%old_D_pat = external global [16 x ubyte]               ; <[16 x ubyte]*> [#uses=0]

implementation   ; Functions:

void %asearch1(uint %D) {
entry:
        %tmp80 = setlt uint 0, %D               ; <bool> [#uses=1]
        br bool %tmp80, label %bb647.preheader, label %cond_true81.preheader

cond_true81.preheader:          ; preds = %entry
        ret void

bb647.preheader:                ; preds = %entry
        %tmp3.i = call int %read( )             ; <int> [#uses=1]
        %tmp6.i = add int %tmp3.i, 0            ; <int> [#uses=1]
        %tmp653 = setgt int %tmp6.i, 0          ; <bool> [#uses=1]
        br bool %tmp653, label %cond_true654, label %UnifiedReturnBlock

cond_true612:           ; preds = %cond_true654
        ret void

cond_next624:           ; preds = %cond_true654
        ret void

cond_true654:           ; preds = %bb647.preheader
        br bool undef, label %cond_true612, label %cond_next624

UnifiedReturnBlock:             ; preds = %bb647.preheader
        ret void
}

void %bitap(int %D) {
entry:
        %tmp29 = seteq int 0, 0         ; <bool> [#uses=1]
        br bool %tmp29, label %cond_next50, label %cond_next37

cond_next37:            ; preds = %entry
        ret void

cond_next50:            ; preds = %entry
        %tmp52 = setgt int %D, 0                ; <bool> [#uses=1]
        br bool %tmp52, label %cond_true53, label %cond_next71

cond_true53:            ; preds = %cond_next50
        %tmp54 = load int* %JUMP                ; <int> [#uses=1]
        %tmp55 = seteq int %tmp54, 1            ; <bool> [#uses=1]
        br bool %tmp55, label %cond_true56, label %cond_next63

cond_true56:            ; preds = %cond_true53
        %tmp57 = cast int %D to uint            ; <uint> [#uses=1]
        call void %asearch1( uint %tmp57 )
        ret void

cond_next63:            ; preds = %cond_true53
        ret void

cond_next71:            ; preds = %cond_next50
        ret void
}

declare int %read()

void %initial_value() {
entry:
        ret void
}

void %main() {
entry:
        br label %cond_next252

cond_next208:           ; preds = %cond_true260
        %tmp229 = call int %atoi( )             ; <int> [#uses=1]
        br label %cond_next252

bb217:          ; preds = %cond_true260
        ret void

cond_next252:           ; preds = %cond_next208, %entry
        %D.0.0 = phi int [ 0, %entry ], [ %tmp229, %cond_next208 ]              ; <int> [#uses=1]
        %tmp254 = getelementptr sbyte** null, int 1             ; <sbyte**> [#uses=1]
        %tmp256 = load sbyte** %tmp254          ; <sbyte*> [#uses=1]
        %tmp258 = load sbyte* %tmp256           ; <sbyte> [#uses=1]
        %tmp259 = seteq sbyte %tmp258, 45               ; <bool> [#uses=1]
        br bool %tmp259, label %cond_true260, label %bb263

cond_true260:           ; preds = %cond_next252
        %tmp205818 = setgt sbyte 0, -1          ; <bool> [#uses=1]
        br bool %tmp205818, label %cond_next208, label %bb217

bb263:          ; preds = %cond_next252
        %tmp265 = seteq int 0, 0                ; <bool> [#uses=1]
        br bool %tmp265, label %cond_next276, label %cond_true266

cond_true266:           ; preds = %bb263
        ret void

cond_next276:           ; preds = %bb263
        %tmp278 = seteq int 0, 0                ; <bool> [#uses=1]
        br bool %tmp278, label %cond_next298, label %cond_true279

cond_true279:           ; preds = %cond_next276
        ret void

cond_next298:           ; preds = %cond_next276
        call void %bitap( int %D.0.0 )
        ret void
}

declare int %atoi()

void %subset_pset() {
entry:
        ret void
}

void %strcmp() {
entry:
        ret void
}

