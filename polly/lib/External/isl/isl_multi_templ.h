#include <isl/space.h>

#define xCAT(A,B) A ## B
#define CAT(A,B) xCAT(A,B)
#undef EL
#define EL CAT(isl_,BASE)
#define xMULTI(BASE) isl_multi_ ## BASE
#define MULTI(BASE) xMULTI(BASE)

struct MULTI(BASE) {
	int ref;
	isl_space *space;

	int n;
	EL *p[1];
};

__isl_give MULTI(BASE) *CAT(MULTI(BASE),_alloc)(__isl_take isl_space *space);
