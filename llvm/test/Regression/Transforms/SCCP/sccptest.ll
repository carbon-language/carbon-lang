; This is the test case taken from appel's book that illustrates a hard case
; that SCCP gets right. BB3 should be completely eliminated.
;
; RUN: as < %s | opt -sccp -constprop -dce -cfgsimplify | dis | not grep BB3

int %test function(int %i0, int %j0) {
BB1:
	br label %BB2
BB2:
	%j2 = phi int [%j4, %BB7], [1, %BB1]
	%k2 = phi int [%k4, %BB7], [0, %BB1]
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
	%j4 = phi int [1, %BB5], [%k2, %BB6]
	%k4 = phi int [%k3, %BB5], [%k5, %BB6]
	br label %BB2
}
