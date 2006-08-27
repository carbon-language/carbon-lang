; We know that strcpy cannot be used to copy pointer values, because
; pointers might contain null bytes and would stop the copy.  The code
; has no defined way to check for this, so DSA can know that strcpy doesn't
; require merging the input arguments.

; RUN: llvm-as < %s | opt -analyze -datastructure-gc --dsgc-abort-if-merged=A,B --dsgc-check-flags=A:ASM,B:ASR --dsgc-dspass=bu

implementation

internal void %strcpy(sbyte* %s1, sbyte* %s2) {
entry:
        br label %loopentry

loopentry:              ; preds = %entry, %loopentry
        %cann-indvar = phi uint [ 0, %entry ], [ %next-indvar, %loopentry ]             ; <uint> [#uses=2]
        %cann-indvar1 = cast uint %cann-indvar to long          ; <long> [#uses=2]
        %s1_addr.0 = getelementptr sbyte* %s1, long %cann-indvar1               ; <sbyte*> [#uses=1]
        %s2_addr.0 = getelementptr sbyte* %s2, long %cann-indvar1               ; <sbyte*> [#uses=1]
        %next-indvar = add uint %cann-indvar, 1         ; <uint> [#uses=1]
        %tmp.3 = load sbyte* %s2_addr.0         ; <sbyte> [#uses=2]
        store sbyte %tmp.3, sbyte* %s1_addr.0
        %tmp.4 = setne sbyte %tmp.3, 0          ; <bool> [#uses=1]
        br bool %tmp.4, label %loopentry, label %loopexit

loopexit:               ; preds = %loopentry
        ret void
}

int %main() {
	%A = alloca sbyte
	%B = alloca sbyte
	call void %strcpy(sbyte* %A, sbyte* %B)
	ret int 0
}
