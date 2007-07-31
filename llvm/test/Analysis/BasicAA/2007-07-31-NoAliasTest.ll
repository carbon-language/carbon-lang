; RUN:   llvm-as %s -o - | opt -aa-eval -print-may-aliases -disable-output |& grep '1 may alias'
; RUN:   llvm-as %s -o - | opt -aa-eval -print-may-aliases -disable-output |& grep '5 no alias'
; RUN:   llvm-as %s -o - | opt -aa-eval -print-may-aliases -disable-output |& grep 'MayAlias:     i32* %ptr4, i32* %ptr2'

define void @_Z3fooPiS_RiS_(i32* noalias  %ptr1, i32* %ptr2, i32* noalias  %ptr3, i32* %ptr4) {
entry:
        store i32 0, i32* %ptr1
        store i32 0, i32* %ptr2
        store i32 0, i32* %ptr3
        store i32 0, i32* %ptr4
        ret void
}