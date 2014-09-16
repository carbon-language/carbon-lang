#include <clc/clc.h>

#define ATOMIC_FUNC_DEFINE(RET_SIGN, ARG_SIGN, TYPE, CL_FUNCTION, CLC_FUNCTION, CL_ADDRSPACE, LLVM_ADDRSPACE) \
_CLC_OVERLOAD _CLC_DEF RET_SIGN TYPE CL_FUNCTION (volatile CL_ADDRSPACE RET_SIGN TYPE *p, RET_SIGN TYPE val) { \
	return (RET_SIGN TYPE)__clc_##CLC_FUNCTION##_addr##LLVM_ADDRSPACE((volatile CL_ADDRSPACE ARG_SIGN TYPE*)p, (ARG_SIGN TYPE)val); \
}

/* For atomic functions that don't need different bitcode dependending on argument signedness */
#define ATOMIC_FUNC_SIGN(TYPE, FUNCTION, CL_ADDRSPACE, LLVM_ADDRSPACE) \
	_CLC_DECL signed TYPE __clc_##FUNCTION##_addr##LLVM_ADDRSPACE(volatile CL_ADDRSPACE signed TYPE*, signed TYPE); \
	ATOMIC_FUNC_DEFINE(signed, signed, TYPE, FUNCTION, FUNCTION, CL_ADDRSPACE, LLVM_ADDRSPACE) \
	ATOMIC_FUNC_DEFINE(unsigned, signed, TYPE, FUNCTION, FUNCTION, CL_ADDRSPACE, LLVM_ADDRSPACE)

#define ATOMIC_FUNC_ADDRSPACE(TYPE, FUNCTION) \
	ATOMIC_FUNC_SIGN(TYPE, FUNCTION, global, 1) \
	ATOMIC_FUNC_SIGN(TYPE, FUNCTION, local, 3)

#define ATOMIC_FUNC(FUNCTION) \
	ATOMIC_FUNC_ADDRSPACE(int, FUNCTION)

ATOMIC_FUNC(atomic_add)
ATOMIC_FUNC(atomic_and)
ATOMIC_FUNC(atomic_or)
ATOMIC_FUNC(atomic_sub)

_CLC_DECL signed int __clc_atomic_max_addr1(volatile global signed int*, signed int);
_CLC_DECL signed int __clc_atomic_max_addr3(volatile local signed int*, signed int);
_CLC_DECL uint __clc_atomic_umax_addr1(volatile global uint*, uint);
_CLC_DECL uint __clc_atomic_umax_addr3(volatile local uint*, uint);

ATOMIC_FUNC_DEFINE(signed, signed, int, atomic_max, atomic_max, global, 1)
ATOMIC_FUNC_DEFINE(signed, signed, int, atomic_max, atomic_max, local, 3)
ATOMIC_FUNC_DEFINE(unsigned, unsigned, int, atomic_max, atomic_umax, global, 1)
ATOMIC_FUNC_DEFINE(unsigned, unsigned, int, atomic_max, atomic_umax, local, 3)
