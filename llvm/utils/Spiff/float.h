/*                        Copyright (c) 1988 Bellcore
**                            All Rights Reserved
**       Permission is granted to copy or use this program, EXCEPT that it
**       may not be sold for profit, the copyright notice must be reproduced
**       on copies, and credit should be given to Bellcore where it is due.
**       BELLCORE MAKES NO WARRANTY AND ACCEPTS NO LIABILITY FOR THIS PROGRAM.
*/


#include "floatrep.h"

#ifndef F_INCLUDED

/*
**	flags for F_atof
*/
#define NO_USE_ALL	0
#define USE_ALL		1

typedef struct R_flstr *F_float;
#define F_getexp(x)	R_getexp(x)
#define F_getsign(x)	R_getsign(x)
#define F_zerofloat(x)	R_zerofloat(x)

extern F_float F_atof();

extern F_float F_floatmul();
extern F_float F_floatmagadd();
extern F_float F_floatsub();

#define F_null	((F_float) 0)

#define F_INCLUDED

#endif
