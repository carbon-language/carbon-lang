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
#include "token.h"
#include "edit.h"
#include "compare.h"

#define MAXT	K_MAXTOKENS
#define ORIGIN (max_obj/2)

#define MILLER_CHATTER	100

/*
**	totally opaque miller/myers code
**		hacked from a version provided by the author
*/


E_edit
G_do_miller(m,n,max_d,comflags)
int m;
int n;
int max_d;
int comflags;
{
    int	max_obj = m + n;
    int
	lower,
	upper,
	d,
	k,
	row,
	col;
	E_edit new;

#ifdef STATIC_MEM
	static E_edit script[MAXT+1];
	static int last_d[MAXT+1];
#else
	E_edit *script;
	int *last_d;
	/*
	**	make space for the two big arrays
	**		these could probably be smaller if I
	**		understood this algorithm at all
	**		as is, i just shoe horned it into my program.
	**	be sure to allocate max_obj + 1 objects as was done
	**		in original miller/myers code
	*/
	script = Z_ALLOC(max_obj+1,E_edit);
	last_d = Z_ALLOC(max_obj+1,int);

#endif
	for (row=0;row < m && row < n && X_com(row,row,comflags) == 0; ++row)
		;
	last_d[ORIGIN] = row;
	script[ORIGIN] = E_NULL;
	lower = (row == m) ? ORIGIN+1 : ORIGIN - 1;
	upper = (row == n) ? ORIGIN-1 : ORIGIN + 1;
	if (lower > upper)
	{
		/*
		**	the files are identical
		*/
		return(E_NULL);
	}
	for (d = 1; d <= max_d; ++d) {
		for (k = lower; k<= upper; k+= 2) {
			new = E_edit_alloc();

			if (k == ORIGIN-d || (k!= ORIGIN+d && last_d[k+1] >= last_d[k-1])) {
				row = last_d[k+1]+1;
				E_setnext(new,script[k+1]);
				E_setop(new,E_DELETE);
			} else {
				row = last_d[k-1];
				E_setnext(new,script[k-1]);
				E_setop(new,E_INSERT);
			}

			E_setl1(new,row);
			col = row + k - ORIGIN;
			E_setl2(new,col);
			script[k] = new;

			while (row < m && col < n && X_com(row,col,comflags) == 0) {
				++row;
				++col;
			}
			last_d[k] = row;
			if (row == m && col == n) {
				return(script[k]);
			}
			if (row == m)
				lower = k+2;
			if (col == n)
				upper = k-2;
		}
		--lower;
		++upper;
#ifndef NOCHATTER
		if ((d > 0) && (0 == (d % MILLER_CHATTER)))
		{
			(void) sprintf(Z_err_buf,
				"found %d differences\n",
				d);
			Z_chatter(Z_err_buf);
		}
#endif
	}
	Z_exceed(max_d);
	/*
	**	dummy lines to shut up lint
	*/
	Z_fatal("fell off end of do_miller\n");
	return(E_NULL);
}
