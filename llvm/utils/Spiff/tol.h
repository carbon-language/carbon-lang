/*                        Copyright (c) 1988 Bellcore
**                            All Rights Reserved
**       Permission is granted to copy or use this program, EXCEPT that it
**       may not be sold for profit, the copyright notice must be reproduced
**       on copies, and credit should be given to Bellcore where it is due.
**       BELLCORE MAKES NO WARRANTY AND ACCEPTS NO LIABILITY FOR THIS PROGRAM.
*/

#include "float.h"

#ifndef T_INCLUDED
/*
**	values for tol_type
*/
#define T_ABSOLUTE 		0
#define T_RELATIVE 		1
#define T_IGNORE		2

typedef struct _T_tstr{
	int tol_type;		/* one of the above */
	F_float flo_tol;	/* tolerance is expressed in
				    terms of a floating point value */
	struct _T_tstr *next;
} _T_struct, *T_tol;

#define _T_TOLMAX	10	/* number of tolerances that can
					be in effect at one time */

#define _T_ADEF		"1e-10"	/* default absolute tolerance */
#define _T_RDEF		"1e-10"	/* default relative tolerance */

extern T_tol T_gettol();
extern void T_clear_tols();
extern void T_initdefault();
extern void T_setdef();
extern void T_tolline();
extern T_tol T_picktol();

#define T_gettype(x)	(x->tol_type)
#define T_getfloat(x)	(x->flo_tol)
#define T_getnext(x)	(x->next)

#define T_settype(x,y)	(x->tol_type = y)
#define T_setfloat(x,y)	(x->flo_tol = y)
#define T_setnext(x,y)	(x->next = y)

#define _T_null		((T_tol) 0)
#define T_isnull(x)	((x) == _T_null)

extern T_tol _T_gtol;
extern void _T_addtol();
extern void _T_appendtols();

/*
**	routines for building the global tolerance list
*/
#define T_defatol(x)	_T_addtol(&_T_gtol,T_ABSOLUTE,x)
#define T_defrtol(x)	_T_addtol(&_T_gtol,T_RELATIVE,x)
#define T_defitol()	_T_addtol(&_T_gtol,T_IGNORE,(char*)NULL)

#define _T_SEPCHAR	';'

#define T_INCLUDED
#endif
