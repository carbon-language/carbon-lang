/*                        Copyright (c) 1988 Bellcore
**                            All Rights Reserved
**       Permission is granted to copy or use this program, EXCEPT that it
**       may not be sold for profit, the copyright notice must be reproduced
**       on copies, and credit should be given to Bellcore where it is due.
**       BELLCORE MAKES NO WARRANTY AND ACCEPTS NO LIABILITY FOR THIS PROGRAM.
*/


#ifndef lint
static char rcsid[]= "$Header$";
#endif

#include "misc.h"
#include "flagdefs.h"
#include "tol.h"
#include "token.h"
#include "line.h"
#include "float.h"
#include "compare.h"

#include <ctype.h>

static int _X_strcmp();
static int _X_cmptokens();
static int _X_floatdiff();

int
X_com(a,b,flags)
int a,b,flags;
{
	K_token atmp,btmp;

	atmp = K_gettoken(0,a);
	btmp = K_gettoken(1,b);
	if(flags & U_BYTE_COMPARE)
	{
		return(_X_strcmp(K_gettext(atmp),K_gettext(btmp),flags));
	}
	else
	{
		return(_X_cmptokens(atmp,btmp,flags));
	}
#ifndef lint 
	Z_fatal("this line should never be reached in com");
	return(-1);	/* Z_fatal never returns, but i need a this line
				here to stop lint from complaining */
#endif
}

/*
**	same as strcmp() except that case can be optionally ignored
*/
static int
_X_strcmp(s1,s2,flags)
char *s1,*s2;
int flags;
{
	if (flags & U_NO_CASE)
	{
		
		for (;('\0' != s1) && ('\0' !=  *s2);s1++,s2++)
		{
			if(isalpha(*s1) && isalpha(*s2))
			{
				if(tolower(*s1) != tolower(*s2))
				{
					return(1);
				}
			}
			else
			{
				if(*s1!=*s2)
				{
					return(1);
				}
			}
		}
		return(*s1 != *s2);
	}
	else
	{
		return(strcmp(s1,s2));
	}
}


/*
**	routine to compare two tokens
*/
static int
_X_cmptokens(p1,p2,flags)
K_token p1, p2;
int flags;
{
	if (K_gettype(p1) !=  K_gettype(p2))
	{
		return(1);
	}

	switch (K_gettype(p1))
	{
		case K_LIT:
			return(_X_strcmp(K_gettext(p1),K_gettext(p2),flags));
		case K_FLO_NUM:
			return(_X_floatdiff(K_getfloat(p1),
					   K_getfloat(p2),
					   T_picktol(K_gettol(p1),
						     K_gettol(p2))));
		default:
			Z_fatal("fell off switch in _X_cmptokens");
			return(-1);	/* Z_fatal never returns, but i need a this line
						here to stop lint from complaining */
	}

}

/*
**	compare two F_floats using a tolerance
*/
static int
_X_floatdiff(p1,p2,the_tol)
F_float p1,p2;
T_tol the_tol;
{
	F_float diff, float_tmp;
	T_tol tol_tmp;

	/*
	** 	check for null tolerance list
	*/
	if (T_isnull(the_tol))
	{
		Z_fatal("_X_floatdiff called with a null tolerance");
	}

	/*
	**	look for an easy answer. i.e -- check
	**		to see if any of the tolerances are of type T_IGNORE
	**		or if the numbers are too small to exceed an absolute
	**		tolerance.
	**		if so, return immediately
	*/
	for(tol_tmp=the_tol; !(T_isnull(tol_tmp)) ;tol_tmp=T_getnext(tol_tmp))
	{
		if ((T_IGNORE == T_gettype(tol_tmp)) || 
			/*
			**	take a look at the exponents before you bother
			**	with the mantissas
			*/
			((T_ABSOLUTE == T_gettype(tol_tmp))
				   && !F_zerofloat(T_getfloat(tol_tmp))
				   && (F_getexp(p1) <
					F_getexp(T_getfloat(tol_tmp))-1)
				   && (F_getexp(p2) <
					F_getexp(T_getfloat(tol_tmp))-1)))
		{
				return(0);
		}
	}

	
	/*
	**	ok, we're going to have to do some arithmetic, so
	**		first find the magnitude of the difference
	*/
	if (F_getsign(p1) != F_getsign(p2))
	{
		diff = F_floatmagadd(p1,p2);
	}
	else
	{
		diff = F_floatsub(p1,p2);
	}

	/*
	**	now check to see if the difference exceeds any tolerance
	*/
	for(tol_tmp=the_tol; !(T_isnull(tol_tmp)) ;tol_tmp=T_getnext(tol_tmp))
	{
		float_tmp = T_getfloat(tol_tmp);

		if (T_gettype(tol_tmp) == T_ABSOLUTE)
		{
			/* do nothing */
		}
		else if (T_gettype(tol_tmp) == T_RELATIVE)
		{
			if (F_floatcmp(p1,p2) > 0)
			{
				float_tmp = F_floatmul(p1, float_tmp);
			}
			else
			{
				float_tmp = F_floatmul(p2, float_tmp);
			}
		}
		else
		{
			Z_fatal("bad value for type of tolerance in floatdiff");
		}
		/*
		**	if we pass this tolerance, then we're done
		*/
		if (F_floatcmp(diff,float_tmp) <= 0)
		{
			return(0);
		}
	}
	/*
	**	all of the tolerances were exceeded
	*/
	return(1);
}
