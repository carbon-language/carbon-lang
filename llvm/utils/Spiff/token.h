/*                        Copyright (c) 1988 Bellcore
**                            All Rights Reserved
**       Permission is granted to copy or use this program, EXCEPT that it
**       may not be sold for profit, the copyright notice must be reproduced
**       on copies, and credit should be given to Bellcore where it is due.
**       BELLCORE MAKES NO WARRANTY AND ACCEPTS NO LIABILITY FOR THIS PROGRAM.
*/


#ifndef K_INCLUDED
#include "float.h"
#include "tol.h"
#include "strings.h"

#define		K_MAXTOKENS	50000
/*
**	values for token type
*/
#define K_LIT		1
#define	K_FLO_NUM	2


typedef struct {
	int linenum;		/* line that the token started on */
	int pos;		/* position on the line where token started */
	int type;		/* token type */
	char *text;	 	/* literal token text */
	/*
	**	canonical floationg point representation
	*/
	F_float flo_num;
	T_tol tolerance;
} _K_str, *K_token;

/*
**	this should really be a two dimensional array
**	but i'm too lazy to recode it
*/
extern K_token _K_ato[];	/* storage for the tokens */
extern K_token _K_bto[];
/*
**	save token X from file
*/
extern void K_settoken(/*file,X,ptr*/);
#define K_gettoken(file, X)	(file?(_K_bto[X]):(_K_ato[X]))

extern int _K_atm;	/* count of tokens */
extern int _K_btm;

/*
**	get token number X from file
*/
#define K_get_token(file, X)	(file?(_K_bto[X]):(_K_ato[X]))

#define K_gettmax(file)		(file?_K_btm:_K_atm)
#define K_settmax(file,value)	(file?(_K_btm=(value)):(_K_atm=(value)))
/*
**	increment and return true on overflow
*/
#define	K_inctmax(file)		((file?(++_K_btm):(++_K_atm))>=K_MAXTOKENS)

#define K_setline(x,y)		(x->linenum = y)
#define K_setpos(x,y)		(x->pos = y)
#define K_settext(x,y)		(x->text = y)
#define K_savetext(x,y,z)	S_savestr(&(x->text),y)
#define K_saventext(x,y,z)	S_savenstr(&(x->text),y,z)
#define K_setfloat(x,y)		(x->flo_num = y)
#define K_settol(x,y)		(x->tolerance = y)
#define K_settype(x,y)		(x->type = y)

#define K_getline(x)		(x->linenum)
#define K_getpos(x)		(x->pos)
#define K_gettext(x)		(x->text)
#define K_getfloat(x)		(x->flo_num)
#define K_gettol(x)		(x->tolerance)
#define K_gettype(x)		(x->type)

#define K_maketoken()		(Z_ALLOC(1,_K_str))

#define K_INCLUDED
#endif
