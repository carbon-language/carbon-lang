/*                        Copyright (c) 1988 Bellcore
**                            All Rights Reserved
**       Permission is granted to copy or use this program, EXCEPT that it
**       may not be sold for profit, the copyright notice must be reproduced
**       on copies, and credit should be given to Bellcore where it is due.
**       BELLCORE MAKES NO WARRANTY AND ACCEPTS NO LIABILITY FOR THIS PROGRAM.
*/

#ifndef Z_INCLUDED

/*
**	make sure that if we have a XENIX system, that
**	we also treat it as an AT and T derivative
*/
#ifdef XENIX
#ifndef ATT
#define ATT
#endif
#endif

#define	Z_LINELEN	1024
#define	Z_WORDLEN	  20

extern char Z_err_buf[];

/*
**	helpful macros
*/
#define Z_ABS(x)	(( (x) < (0) )? (-(x)):(x))
#define Z_MIN(x,y)	(( (x) < (y) )? (x):(y))
#define Z_MAX(x,y)	(( (x) > (y) )? (x):(y))

#define Z_ALLOC(n,type)	((type*) _Z_myalloc((n) * sizeof (type)))
extern int *_Z_myalloc();

/*
**	lines needed to shut up lint
*/

extern void Z_complain();
extern void Z_fatal();
extern void Z_exceed();
extern void Z_setquiet();
#ifndef NOCHATTER
extern void Z_chatter();
#endif

#define Z_INCLUDED
#endif
