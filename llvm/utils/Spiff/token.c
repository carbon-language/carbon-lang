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
#include "token.h"

K_token _K_ato[K_MAXTOKENS]; /* storage for tokens */
K_token _K_bto[K_MAXTOKENS];

int _K_atm;
int _K_btm;

void
K_settoken(file,index,pointer)
int file;
int index;
K_token pointer;
{
	if (file)
	{
		_K_bto[index] = pointer;
	}
	else
	{
		_K_ato[index] = pointer;
	}
}
