implementation

; edgefailure - This function illustrates how SCCP is not doing it's job.  This
; function should be optimized almost completely away: the loop should be
; analyzed to detect that the body executes exactly once, and thus the branch
; can be eliminated and code becomes trivially dead.  This is distilled from a
; real benchmark (mst from Olden benchmark, MakeGraph function).  When SCCP is
; fixed, this should be eliminated by a single SCCP application.  TODO
;
int *"edgefailure"()
begin
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
end



int "test function"(int %i0, int %j0)
	%i1 = const int 1
	%j1 = const int 1
	%k1 = const int 0
begin
BB1:
	br label %BB2
BB2:
	%j2 = phi int [%j4, %BB7], [%j1, %BB1]
	%k2 = phi int [%k4, %BB7], [%k1, %BB1]
	%kcond = setlt int %k2, 100
	br bool %kcond, label %BB3, label %BB4

BB3:
	%jcond = setlt int %j2, 20
	br bool %jcond, label %BB5, label %BB6

BB4:
	ret int %j2

BB5:
	%k3 = add int %k2, 1
	br label %BB7

BB6:
	%k5 = add int %k2, 1
	br label %BB7

BB7:
	%j4 = phi int [%i1, %BB5], [%k2, %BB6]
	%k4 = phi int [%k3, %BB5], [%k5, %BB6]
	br label %BB2
end

