; RUN: llvm-as -f %s -o - | llc

;;
;; Test the sequence:
;;	cast -> setle 0, %cast -> br %cond
;; This sequence should cause the cast value to be forwarded twice,
;; i.e., cast is forwarded to the setle and teh setle is forwarded
;; to the branch.
;; register argument of the "branch-on-register" instruction, i.e.,
;; 
;; This produces the bogus output instruction:
;;	brlez   <NULL VALUE>, .L_SumArray_bb3.
;; This came from %bb1 of sumarrray.ll generated from sumarray.c.


int %SumArray(int %Num) {
        %Num = alloca int
        br label %Top
Top:
        store int %Num, int * %Num
        %reg108 = load int * %Num
        %cast1006 = cast int %reg108 to uint
        %cond1001 = setle uint %cast1006, 0
	br bool %cond1001, label %bb6, label %Top

bb6:
	ret int 42
}
