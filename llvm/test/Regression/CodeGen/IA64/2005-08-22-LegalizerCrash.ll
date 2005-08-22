; RUN: llvm-as < %s | llc -march=ia64

%_ZN9__gnu_cxx16__stl_prime_listE = external global [28 x uint]		; <[28 x uint]*> [#uses=3]

implementation   ; Functions:

fastcc uint* %_ZSt11lower_boundIPKmmET_S2_S2_RKT0_(uint %__val.val) {
entry:
	%retval = select bool setgt (int shr (int sub (int cast (uint* getelementptr ([28 x uint]* %_ZN9__gnu_cxx16__stl_prime_listE, int 0, int 28) to int), int cast ([28 x uint]* %_ZN9__gnu_cxx16__stl_prime_listE to int)), ubyte 2), int 0), uint* null, uint* getelementptr ([28 x uint]* %_ZN9__gnu_cxx16__stl_prime_listE, int 0, int 0)		; <uint*> [#uses=1]
	ret uint* %retval
}
