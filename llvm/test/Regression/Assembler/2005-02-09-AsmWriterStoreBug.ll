; RUN: llvm-as < %s | llvm-dis | llvm-as

; Ensure that the asm writer emits types before both operands of the 
; store, even though they can be the same.

%RecTy = type %RecTy*
implementation

 void %foo() {
        %A = malloc %RecTy
        %B = malloc %RecTy
        store %RecTy %B, %RecTy %A   ;; Both ops are the same
        ret void
}
