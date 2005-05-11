; ModuleID = 'bugpoint-reduced-simplified.bc'
; Reduced from the hash benchmark from the ShootOut C++ benchmark test
;
; RUN: llvm-as < %s | llc -march=sparcv9

target endian = big
target pointersize = 64
%_ZN9__gnu_cxx16__stl_prime_listE = external global [28 x ulong]		; <[28 x ulong]*> [#uses=3]

implementation   ; Functions:

fastcc void %_ZSt11lower_boundIPKmmET_S2_S2_RKT0_() {
entry:
	%retval = select bool setgt (long shr (long sub (long cast (ulong* getelementptr ([28 x ulong]* %_ZN9__gnu_cxx16__stl_prime_listE, long 0, long 28) to long), long cast ([28 x ulong]* %_ZN9__gnu_cxx16__stl_prime_listE to long)), ubyte 3), long 0), ulong* null, ulong* getelementptr ([28 x ulong]* %_ZN9__gnu_cxx16__stl_prime_listE, long 0, long 0)		; <ulong*> [#uses=0]
	ret void
}
