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

#include <ctype.h>
#include "misc.h"
#include "floatrep.h"
#include "float.h"
#include "strings.h"

#define _F_GETEND(x)	(x + (strlen(x)-1)) 

/*
int floatcnt = 0;
*/
/*
**	routines to convert strings to our internal floating point form
**		isfloat just looks at the string
**		to see if a conversion is reasonable
**			it does look-ahead on when it sees an 'e' and such.
**		atocf actually does the conversion.
**	these two routines could probably be combined
*/

/*
**	test to see if the string can reasonably
**		be interpreted as floating point number
**	returns 0 if string can't be interpreted as a float
**	otherwise returns the number of digits that will be used in F_atof
*/
F_isfloat(str,need_decimal,allow_sign)
char *str;
int need_decimal;	/* if non-zero, require that a decimal point be present
				otherwise, accept strings like "123" */
int allow_sign;		/* if non-zero, allow + or - to set the sign */
{
	int man_length = 0;	/* length of the fractional part (mantissa) */
	int exp_length = 0;	/* length of the exponential part */
	int got_a_digit = 0;	/* flag to set if we ever see a digit */

	/*
	**	look for an optional leading sign marker
	*/
	if (allow_sign && ('+' == *str  || '-' == *str))
	{
		str++; man_length++;
	}
	/*
	**	count up the digits on the left hand side
	**		 of the decimal point
	*/
	while(isdigit(*str))
	{
		got_a_digit = 1;
		str++; man_length++;
	}

	/*
	**	check for a decimal point
	*/
	if ('.' == *str)
	{
		str++; man_length++;
	}
	else
	{
		if (need_decimal)
		{
			return(0);
		}
	}

	/*
	**	collect the digits on the right hand
	**		side of the decimal point
	*/
	while(isdigit(*str))
	{
		got_a_digit = 1;
		str++; man_length++;
	}

	if (!got_a_digit)
		return(0);

	/*
	**	now look ahead for an exponent
	*/
	if ('e' == *str ||
	    'E' == *str ||
	    'd' == *str ||
	    'D' == *str)
	{
		str++; exp_length++;
		if ('+' == *str  || '-' == *str)
		{
			str++; exp_length++;
		}

		if (!isdigit(*str))
		{
			/*
			**	look ahead went too far,
			**	so return just the length of the mantissa
			*/
			return(man_length);
		}

		while (isdigit(*str))
		{
			str++; exp_length++;
		}
	}
	return(man_length+exp_length);	/* return the total length */
}

/*
**	routine to convert a string to our internal
**	floating point representation
**
**		similar to atof()
*/
F_float
F_atof(str,allflag)
char *str;
int allflag;	/* require that exactly all the characters are used */
{
	char *beg = str; /* place holder for beginning of the string */ 
	char man[R_MANMAX];	/* temporary location to build the mantissa */
	int length = 0;	/* length of the mantissa so far */
	int got_a_digit = 0;	/* flag to set if we get a non-zero digit */ 
	int i;
	int resexp;

	F_float res;	/* where we build the result */

/*
floatcnt++;
*/
	res = R_makefloat();

	R_setsign(res,R_POSITIVE);

	resexp = 0;
	man[0] = '\0';

	/*
	**	check for leading sign
	*/
	if ('+' == *str)
	{
		/*
		**	sign should already be positive, see above in this
		**		routine, so just skip the plus sign
		*/
		str++;
	}
	else
	{
		if ('-' == *str)
		{
			R_setsign(res,R_NEGATIVE);
			str++;
		}
	}

	/*
	**	skip any leading zeros
	*/
	while('0' == *str)
	{
		str++;
	}

	/*
	**	now snarf up the digits on the left hand side
	**		of the decimal point
	*/
	while(isdigit(*str))
	{
		got_a_digit = 1;
		man[length++] = *str++;
		man[length] = '\0';
		resexp++;
	}

	/*
	**	skip the decimal point if there is one
	*/
	if ('.' == *str)
		str++;

	/*
	**	trim off any leading zeros (on the right hand side)
	**	if there were no digits in front of the decimal point.
	*/

	if (!got_a_digit)
	{
		while('0' == *str)
		{
			str++;
			resexp--;
		}
	}

	/*
	**	now snarf up the digits on the right hand side
	*/
	while(isdigit(*str))
	{
		man[length++] = *str++;
		man[length] = '\0';
	}

	if ('e' == *str ||
            'E' == *str ||
            'd' == *str ||
            'D' == *str )
	{
		str++;
		resexp += atoi(str);
	}

	if (allflag)
	{
		if ('+' == *str ||
		    '-' == *str)
		{
			str++;
		}
		while (isdigit(*str))
		{
			str++;
		}
		if ('\0' != *str)
		{
			(void) sprintf(Z_err_buf,
					"didn't use up all of %s in atocf",
					beg);
			Z_fatal(Z_err_buf);
		}
	}

	/*
	**	check for special case of all zeros in the mantissa
	*/
	for (i=0;i<length;i++)
	{
		if (man[i] != '0')
		{
			/*
			**	the mantissa is non-zero, so return it unchanged
			*/
			S_trimzeros(man);
			/*
			**	save a copy of the mantissa
			*/
			R_setfrac(res,man);
			R_setexp(res,resexp);
			return(res);
		}
	}

	/*
	**	the answer is 0, so . . .
	*/
	R_setzero(res);
	return(res);
}


/*
**	add s2 to s1
*/
static
void
_F_stradd(s1,s2)
char *s1,*s2;
{
	char *end1 = s1 + (strlen(s1)-1);
	char *end2 = s2 + (strlen(s2)-1);

	static char result[R_MANMAX];
	char *resptr = result+(R_MANMAX-1); /*point to the end of the array */
	int carry = 0;
	int tmp,val1,val2;

	*resptr-- = '\0';

	while ((end1 >= s1) ||  ( end2 >= s2))
	{
		if (end1 >= s1)
		{
			val1 = *end1 - '0';
			--end1;
		}
		else
		{
			val1 = 0;
		}

		if (end2 >= s2)
		{
			val2 = *end2 - '0';
			--end2;
		}
		else
		{
			val2 = 0;
		}

		tmp = val1 + val2 + carry;
		if (tmp > 9)
		{
			carry = 1;
			tmp -= 10;
		}
		else
		{
			carry = 0;
		}

		*resptr-- = tmp+'0';
	}
	if (carry)
	{
		*resptr =  '1';
	}
	else
	{
		resptr++;
	}
	(void) strcpy(s1,resptr);
	return;
}

/*
**	add zero(s) onto the end of a string
*/
static void
addzeros(ptr,count)
char *ptr;
int count;
{
	for(;count> 0;count--)
	{
		(void) strcat(ptr,"0");
	}
	return;
}

/*
**	subtract two mantissa strings
*/
F_float
F_floatsub(p1,p2)
F_float  p1,p2;
{
	static F_float result;
	static needinit = 1;
	static char man1[R_MANMAX],man2[R_MANMAX],diff[R_MANMAX];
	int exp1,exp2;
	char *diffptr,*big,*small;
	int man_cmp_val,i,borrow;

	if (needinit)
	{
		result = R_makefloat();
		needinit = 0;
	}

	man1[0] = '\0';
	man2[0] = '\0';

	exp1 = R_getexp(p1);
	exp2 = R_getexp(p2);

	/*
	**	line up the mantissas
	*/
	while (exp1 < exp2)
	{
		(void) strcat(man1,"0");
		exp1++;
	}

	while(exp1 > exp2)
	{
		(void) strcat(man2,"0");
		exp2++;
	}

	if (exp1 != exp2)	/* boiler plate assertion */
	{
		Z_fatal("mantissas didn't get lined up properly in floatsub");
	}

	(void) strcat(man1,R_getfrac(p1));
	(void) strcat(man2,R_getfrac(p2));
	
	/*
	**	now that the mantissa are aligned,
	**	if the strings are the same, return 0
	*/
	if((man_cmp_val = strcmp(man1,man2)) == 0)
	{
		R_setzero(result);
		return(result);
	}

	/*
	**	pad the shorter string with 0's
	**		when this loop finishes, both mantissas should
	**		have the same length
	*/
	if (strlen(man1)> strlen(man2))
	{
		addzeros(man2,strlen(man1)-strlen(man2));
	}
	else
	{
		if (strlen(man1)<strlen(man2))
		{
			addzeros(man1,strlen(man2)-strlen(man1));
		}
	}

	if (strlen(man1) != strlen(man2))	/* pure boilerplate */
	{
		Z_fatal("lengths not equal in F_floatsub");
	}

	if (man_cmp_val < 0)
	{
		big = man2;
		small = man1;
	}
	else
	{
		big = man1;
		small = man2;
	}

	/*
	**	find the difference between the mantissas
	*/
	for(i=(strlen(big)-1),borrow=0,diff[strlen(big)] = '\0';i>=0;i--)
	{
		char from;
		if (borrow)
		{
			if (big[i] == '0')
			{
				from = '9';
			}
			else
			{
				from = big[i]-1;
				borrow = 0;
			}
		}
		else
		{
			if(big[i]<small[i])
			{
				from = '9'+1;
				borrow = 1;
			}
			else
			{
				from = big[i];
			}
		}
		diff[i] = (from-small[i]) + '0';
	}

	/*
	** trim the leading zeros on the difference
	*/
	diffptr = diff;
	while('0' == *diffptr)
	{
		diffptr++;
		exp1--;
	}

	R_setexp(result,exp1); /* exponents are equal at the point */
	R_setfrac(result,diffptr);
	R_setsign(result,R_POSITIVE);
	return(result);
}

F_floatcmp(f1,f2)
F_float f1,f2;
{
	static char man1[R_MANMAX],man2[R_MANMAX];

	/*
	**		special case for zero
	*/
	if (R_zerofloat(f1))
	{
		if (R_zerofloat(f2))
		{
			return(0);
		}
		else
		{
			return(-1);
		}
	}
	else
	{
		if (R_zerofloat(f2))
		{
			return(1);
		}
	}

	/*
	**	to reach this point, both numbers must be non zeros
	*/
	if (R_getexp(f1) < R_getexp(f2))
	{
		return(-1);
	}

	if (R_getexp(f1) > R_getexp(f2))
	{
		return(1);
	}

	(void) strcpy(man1,R_getfrac(f1));
	S_trimzeros(man1);

	(void) strcpy(man2,R_getfrac(f2));
	S_trimzeros(man2);
	return(strcmp(man1,man2));
}

F_float
F_floatmul(f1,f2)
F_float f1,f2;
{
	static char prod[R_MANMAX];
	char *end;
	int count1 = 0;
	int count2 = 0;
	int tmp,len;
	char *end1;
	char *end2;
	static char man1[R_MANMAX],man2[R_MANMAX];
	char *bigman,*smallman;
	static F_float result;
	static int needinit = 1;

	if (needinit)
	{
		result = R_makefloat();
		needinit = 0;
	}
	/*
	**	special case for a zero result
	*/
	if (R_zerofloat(f1) || R_zerofloat(f2))
	{
		R_setzero(result);
		return(result);
	}

	(void) strcpy(man1,R_getfrac(f1));
	(void) strcpy(man2,R_getfrac(f2));

	end1 = _F_GETEND(man1);
	end2 = _F_GETEND(man2);

	/*
	**	decide which number will cause multiplication loop to go
	**	around the least
	*/
	while(end1 >= man1)
	{
		count1 += *end1 - '0';
		end1--;
	}

	while(end2 >= man2)
	{
		count2 += *end2 - '0';
		end2--;
	}


	if (count1 > count2)
	{
		bigman = man1;
		smallman = man2;
	}
	else
	{
		bigman = man2;
		smallman = man1;
	}
	S_trimzeros(bigman);
	S_trimzeros(smallman);
	len = strlen(bigman) +  strlen(smallman);

	end = _F_GETEND(smallman);
	(void) strcpy(prod,"0");

	/*
	**	multiplication by repeated addition
	*/
	while(end >= smallman)
	{
		for(tmp = 0;tmp<*end-'0';tmp++)
		{
			_F_stradd(prod,bigman);
		}
		addzeros(bigman,1);
		end--;
	}

	R_setfrac(result,prod);
	R_setexp(result,(((R_getexp(f1) + R_getexp(f2)) - len)+ strlen(prod)));

	if (R_getsign(f1) == R_getsign(f2))
	{
		R_setsign(result,R_POSITIVE);
	}
	else
	{
		R_setsign(result,R_NEGATIVE);
	}
	return(result);
}

_F_xor(x,y)
{
	return(((x) && !(y)) || (!(x) && (y)));
}
#define	_F_SAMESIGN(x,y)	_F_xor((x<0),(y<0))
#define _F_ABSADD(x,y)		(Z_ABS(x) + Z_ABS(y))

_F_ABSDIFF(x,y)
{
	if (Z_ABS(x) < Z_ABS(y))
	{
		return(Z_ABS(y) - Z_ABS(x));
	}
	else
	{
		return(Z_ABS(x) - Z_ABS(y));
	}
}
/*
**	add two floats without regard to sign
*/
F_float
F_floatmagadd(p1,p2)
F_float p1,p2;
{
	static F_float result;
	static int needinit = 1;

	static char  man1[R_MANMAX],man2[R_MANMAX];

	int digits;	/* count of the number of digits needed to represent the
				result */
	int resexp;	/* exponent of the result */
	int len;	/* length of the elements before adding */
	char *diffptr;

	if (needinit)
	{
		result = R_makefloat();
		needinit = 0;
	}
	(void) strcpy(man1,"");
	(void) strcpy(man2,"");

	/*
	**	find the difference in the exponents number of digits
	*/
	if( _F_SAMESIGN(R_getexp(p1),R_getexp(p2)))
	{
		digits =  _F_ABSDIFF(R_getexp(p1),R_getexp(p2));
	}
	else
	{
		digits = _F_ABSADD(R_getexp(p1),R_getexp(p2));
	}

	/*
	**	make sure that there is room to store the result
	*/
	if (digits>0)
	{ 
		if (R_getexp(p1) < R_getexp(p2))
		{
			/*
			**	leave room for terminator
			*/
			if (digits+strlen(R_getfrac(p1)) > (R_MANMAX-1))
			{
				(void) sprintf(Z_err_buf,
				   "numbers differ by too much in magnitude");
				Z_fatal(Z_err_buf);
			}
		}
		else
		{
			/*
			**	leave room for terminator
			*/
			if (digits+strlen(R_getfrac(p2)) > (R_MANMAX-1))
			{
				(void) sprintf(Z_err_buf,
				   "numbers differ by too much in magnitude");
				Z_fatal(Z_err_buf);
			}
		}
	}
	else
	{
		/*
		**	leave room for terminator and possible carry
		*/
		if (Z_MAX(strlen(R_getfrac(p1)),
			strlen(R_getfrac(p2))) > (R_MANMAX-2))
		{						
			(void) sprintf(Z_err_buf,
			   "numbers differ by too much in magnitude");
			Z_fatal(Z_err_buf);
		}
	}

	/*
	**	pad zeroes on the front of the smaller number
	*/
	if (R_getexp(p1) < R_getexp(p2))
	{

		addzeros(man1,digits);
		resexp = R_getexp(p2);
	}
	else
	{
		addzeros(man2,digits);
		resexp = R_getexp(p1);
	}
	(void) strcat(man1,R_getfrac(p1));
	(void) strcat(man2,R_getfrac(p2));

	len = Z_MAX(strlen(man1),strlen(man2));

	/*
	**	add the two values
	*/
	_F_stradd(man1,man2);

	/*
	**	adjust the exponent to account for a
	**		possible carry
	*/
	resexp += strlen(man1) - len;


	/*
	** trim the leading zeros on the sum
	*/
	diffptr = man1;
	while('0' == *diffptr)
	{
		diffptr++;
		resexp--;
	}

	R_setfrac(result,diffptr);
	R_setexp(result,resexp);
	R_setsign(result,R_POSITIVE);

	return(result);
}

/*
**	useful debugging routine. we don't call it in the release,
**	so it is commented out, but we'll leave it for future use
*/

/*
F_printfloat(fl)
F_float fl;
{
	(void) printf("fraction = :%s: exp = %d sign = %c\n",
			R_getfrac(fl),
			R_getexp(fl),
			((R_getsign(fl) == R_POSITIVE) ? '+': '-'));

}
*/
