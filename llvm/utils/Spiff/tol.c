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

#include <stdio.h>
#include "misc.h"
#include "float.h"
#include "tol.h"
#include "token.h"

/*
**	storage for the default tolerances
*/
T_tol _T_gtol = _T_null;

/*
**	tolerances that can be set in the command script and attached to floating
**		point numbers at parse time
*/
static T_tol _T_tols[_T_TOLMAX];

/*
**	initialize the global tolerance
**	should be called only once at the beginning of the program
*/
void
T_initdefault()
{
	static int called_before = 0;

	if (called_before)
	{
		Z_fatal("T_initdefault called more than once\n");
	}

	/*
	**	if the default tolerance was set somewhere else
	**	don't set it here
	*/
	if (T_isnull(_T_gtol))
	{
		T_defatol(_T_ADEF);
		T_defrtol(_T_RDEF);
	}
	called_before = 1;
}

static void
_T_tolclear(addr)
T_tol *addr;
{
	*addr = _T_null;
}

/*
**	clear the parse time tolerances
*/
void
T_clear_tols()
{
	int i;
	for(i=0;i<_T_TOLMAX;i++)
	{
		_T_tolclear(&_T_tols[i]);
	}
}

static void
_T_defclear()
{
	_T_tolclear(&_T_gtol);
}

/*
**	take a series of specifiers and add them to the tolerance
*/
static void
_T_settol(toladdr,str)
T_tol *toladdr;
char *str;
{
	char typechar;
	while ('\0' != *str)
	{
		/*
		**	find the first non-whitespace character
		*/
		S_skipspace(&str);
		/*
		**	snarf up the type specifier
		*/
		typechar = *str;
		/*
		**	now skip the first char
		*/
		str++;
		/*
		**	skip any possibly intervening whitespace
		*/
		S_skipspace(&str);
		switch (typechar)
		{
			case 'a':
				_T_addtol(toladdr,T_ABSOLUTE,str);
				break;
			case 'r':
				_T_addtol(toladdr,T_RELATIVE,str);
				break;
			case 'i':
				_T_addtol(toladdr,T_IGNORE,(char*)0);
				break;
			case 'd':
				_T_appendtols(toladdr,_T_gtol);
				break;
			default:
				(void) sprintf(Z_err_buf,
				  "don't understand tolerance type '%c'\n",typechar);
				Z_fatal(Z_err_buf);
		}
		/*
		**	and skip to next tolerance
		*/
		S_nextword(&str);
	}
}

/*
**	set the default tolerance 
*/
void
T_setdef(str)
char *str;
{
	_T_defclear();
	_T_settol(&_T_gtol,str);
}


static char*
_T_nextspec(ptr)
char *ptr;
{
	/*
	**	find the end of the current spec
	*/
	for(;(_T_SEPCHAR != *ptr) && ('\0' != *ptr);ptr++)
	{
	}

	/*
	**	and step over the seperator if necessary
	*/
	if (_T_SEPCHAR == *ptr)
		ptr++;

	return(ptr);
}

/*
**	return just the next set of specs
**		ie the string up to end of line or
**			the first _T_SEPCHAR
**	returned string does not include the _T_SEPCHAR
*/
static char *
_T_getspec(from)
char *from;
{
	static char retval[Z_LINELEN];
	char *ptr = retval;

	while((_T_SEPCHAR != *from) && ('\0' != *from))
	{
		*ptr++ = *from++;
	}
	*ptr = '\0';	/* terminate the line */
	return(retval);
}

/*
**	parse a series of _T_SEPCHAR separated tolerance specifications
*/
void
T_tolline(str)
char *str;
{
	int nexttol;

	T_clear_tols();

	for(nexttol=0;'\0' != *str;nexttol++,str = _T_nextspec(str))
	{
		/*
		**	make sure we haven't run off the end
		*/
		if (nexttol >= _T_TOLMAX)
		{
			Z_fatal("too many tolerances per line");
		}

		/*
		**	and set the tolerance
		*/
		_T_settol(&_T_tols[nexttol],_T_getspec(str));
	}
}

int
T_moretols(next_tol)
int next_tol;
{
	return((next_tol >= 0) &&
		(_T_TOLMAX-1 > next_tol) &&
		(!T_isnull( _T_tols[next_tol+1])));
}

T_tol
T_gettol(index)
int index;
{
	return(_T_tols[index]);
}

/*
**	chose which tolerance to use
**		 precidence is
**			first tolerance
**			second tolerance
**			default tolerance
*/
T_tol
T_picktol(p1,p2)
T_tol p1, p2;
{
	if (!(T_isnull(p1)))
		return(p1);

	if (!(T_isnull(p2)))
		return(p2);

	return(_T_gtol);
}

void
_T_appendtols(to,from)
T_tol *to,from;
{

	T_tol last;

	/*
	**	are there any elements on the list yet
	*/
	if (T_isnull(*to))
	{
		/*
		**	it's a null list, so allocat space for the
		**		first element and set pointer to it.
		*/

		*to = from;
	}
	else
	{
		/*
		**	find the last element on the list
		*/
		for(last= *to;!T_isnull(T_getnext(last));last = T_getnext(last))
		{
		}
		/*
		**	add an element on the end
		*/
		T_setnext(last,from);
	}
}

/*
**	add a tolerance to a list
*/
void
_T_addtol(listptr,type,str)
T_tol *listptr;
int type;
char *str;
{
	T_tol last;

	/*
	**	are there any elements on the list yet
	*/
	if (T_isnull(*listptr))
	{
		/*
		**	it's a null list, so allocat space for the
		**		first element and set pointer to it.
		*/

		last = *listptr = Z_ALLOC(1,_T_struct);
	}
	else
	{
		/*
		**	find the last element on the list
		*/
		for(last= *listptr;!T_isnull(T_getnext(last));last = T_getnext(last))
		{
		}
		/*
		**	add an element on the end
		*/
		T_setnext(last,Z_ALLOC(1,_T_struct));

		/*
		**	and point to the new element
		*/
		last = T_getnext(last);
	}

	T_settype(last,type);
	T_setnext(last,_T_null);

	/*
	**	set the float value only if necessary
	*/
	if (T_IGNORE == type)
	{
		T_setfloat(last,F_null);
	}
	else
	{
		T_setfloat(last,F_atof(str,NO_USE_ALL));

		/*
		**	test new tolerance for sanity
		*/
		if (F_getsign(T_getfloat(last)))
		{
			(void) sprintf(Z_err_buf,
			"%s : negative tolerances don't make any sense\n",str);
			Z_fatal(Z_err_buf);
		}
		/*
		**	check for excessively large relative tolerances
		*/
		if ((T_RELATIVE == type) &&
			 (F_floatcmp(T_getfloat(last),
				     F_atof("2.0",USE_ALL)) > 0))
		{
			(void) sprintf(Z_err_buf,
	"%s : relative tolerances greater than 2 don't make any sense\n",str);
			Z_fatal(Z_err_buf);
		}
	}
}
