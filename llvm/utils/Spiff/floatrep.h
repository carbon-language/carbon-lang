/*                        Copyright (c) 1988 Bellcore
**                            All Rights Reserved
**       Permission is granted to copy or use this program, EXCEPT that it
**       may not be sold for profit, the copyright notice must be reproduced
**       on copies, and credit should be given to Bellcore where it is due.
**       BELLCORE MAKES NO WARRANTY AND ACCEPTS NO LIABILITY FOR THIS PROGRAM.
*/

/*
** header file that defines canonical floating point structure
**	and routines
*/


#ifndef  R_INCLUDED

/*
**	when evaluated to a string, the fractional part will
**		not exceed this length
*/
#define R_MANMAX	200

#define R_POSITIVE	0
#define R_NEGATIVE	1

struct  R_flstr {
	int exponent;
	int man_sign;
	char *mantissa;
};

typedef struct R_flstr *R_float;

#define R_getfrac(x)	(x->mantissa)

extern R_float R_makefloat();

extern int R_getexp();

#define R_getsign(x)	(x->man_sign)

/*
**	takes a string
*/
#define R_setfrac(x,y)	((void)strcpy(x->mantissa,y))
/*
**	takes an int
*/
#define R_setexp(x,y)	(x->exponent = y)
/*
**	takes a sign
*/
#define R_setsign(x,y)	(x->man_sign = y)

/*
#define R_incexp(x)	((x->exponent)++)
#define R_decexp(x)	((x->exponent)--)
*/

#define R_setzero(x)	R_setfrac(x,"0");R_setexp(x,0);R_setsign(x,R_POSITIVE)

#define R_zerofloat(x)	((0 == x->exponent) && (!strcmp(x->mantissa,"0")))

#define R_INCLUDED

#endif
