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
#include "edit.h"

/*
**	routine to compare each object with its ordinal twin
*/
E_edit
Q_do_exact(size1,size2,max_d,comflags)
int size1;
int size2;
int max_d;
int comflags;
{
	int i = 0;
	int diffcnt = 0;
	int last = Z_MIN(size1,size2);
	int next_edit = 0;
	E_edit last_ptr = E_NULL;
	int start,tmp;
	E_edit *script;

	script = Z_ALLOC(max_d+1,E_edit);

	if (size1 != size2)
	{
		(void) sprintf(Z_err_buf,"unequal number of tokens, %d and %d respectively\n",size1,size2);
		Z_complain(Z_err_buf);
	}

	do
	{
		/*
		**	skip identical objects
		*/
		while (i<last && (!X_com(i,i,comflags)))
		{
			i++;
		}
		start = i;
		/*
		**	see how many difference we have in a row
		*/
		while (i<last && X_com(i,i,comflags))
		{
			if ((diffcnt += 2) >= max_d+1)
				Z_exceed(max_d);
			i++;
		}
		/*
		**	build the list of deletions
		*/
		for(tmp=start;tmp<i;tmp++,next_edit++)
		{
			script[next_edit] = E_edit_alloc();
			E_setnext(script[next_edit],last_ptr);
			last_ptr = script[next_edit];

			E_setop(script[next_edit],E_DELETE);
			E_setl1(script[next_edit],tmp+1);
			/* no need to set line2, it is never used */
			E_setl2(script[next_edit],0);
		}
		/*
		**	build the list of insertions
		*/
		for(tmp=start;tmp<i;tmp++,next_edit++)
		{
			script[next_edit] = E_edit_alloc();
			E_setnext(script[next_edit],last_ptr);
			last_ptr = script[next_edit];

			E_setop(script[next_edit],E_INSERT);
			E_setl1(script[next_edit],i);
			E_setl2(script[next_edit],tmp+1);
		}
	} while (i<last);

	return(last_ptr);
}
