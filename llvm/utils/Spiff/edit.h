/*                        Copyright (c) 1988 Bellcore
**                            All Rights Reserved
**       Permission is granted to copy or use this program, EXCEPT that it
**       may not be sold for profit, the copyright notice must be reproduced
**       on copies, and credit should be given to Bellcore where it is due.
**       BELLCORE MAKES NO WARRANTY AND ACCEPTS NO LIABILITY FOR THIS PROGRAM.
*/

/*
**	the naming information hiding conventions are incompletely implemented
**	 for the edit module. I tried to clean it up once, but kept introducing
**	 nasty (ie. core dump) bugs in the miller code.  I give up for now.
*/
#ifndef E_INCLUDED

#define E_INSERT	1
#define E_DELETE	2

typedef struct edt {
	struct edt *link;
	int op;
	int line1;
	int line2;
} _E_struct, *E_edit;

#define E_setop(x,y)		((x)->op = (y))
#define E_setl1(x,y)		((x)->line1 = (y))
#define E_setl2(x,y)		((x)->line2 = (y))
#define E_setnext(x,y)		((x)->link = (y))

#define E_getop(x)		((x)->op)
#define E_getl1(x)		((x)->line1)
#define E_getl2(x)		((x)->line2)
#define E_getnext(x)		((x)->link)

#define E_NULL 		((E_edit) 0)
#define E_edit_alloc()	(Z_ALLOC(1,_E_struct))

#define E_INCLUDED

#endif


