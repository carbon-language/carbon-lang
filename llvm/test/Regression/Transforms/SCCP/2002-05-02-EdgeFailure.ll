; edgefailure - This function illustrates how SCCP is not doing it's job.  This
; function should be optimized almost completely away: the loop should be
; analyzed to detect that the body executes exactly once, and thus the branch
; can be eliminated and code becomes trivially dead.  This is distilled from a
; real benchmark (mst from Olden benchmark, MakeGraph function).  When SCCP is
; fixed, this should be eliminated by a single SCCP application.
;
; RUN: llvm-as < %s | opt -sccp | llvm-dis | not grep loop

int* %test() {
bb1:
	%A = malloc int
	br label %bb2
bb2:
	%i = phi int [ %i2, %bb2 ], [ 0, %bb1 ]   ;; Always 0
	%i2 = add int %i, 1                       ;; Always 1
	store int %i, int *%A
	%loop = setle int %i2, 0                  ;; Always false
	br bool %loop, label %bb2, label %bb3

bb3:
	ret int * %A
}
