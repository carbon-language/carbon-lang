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
#include "strings.h"

/*
**	routines for handling strings.
**		several routines manipulate  "words"
**		a "word" is a string not containing whitespace
*/

/*
**	copy a single word. similar to  strcpy
*/
void
S_wordcpy(to,from)
char *to, *from;
{
	while ((*from != '\0') && isprint(*from) && (!isspace(*from)))
	{
		*to++ = *from++;
	}
	*to = '\0';
	return;
}

/*
**	find the next whitespace character.  The address of the pointer
**		is passed and the pointer itself is changed.
*/
void
S_skipword(theptr)
char **theptr;
{
	while((**theptr != '\0') && isprint(**theptr) && (!isspace(**theptr)))
	{
		(*theptr)++;	/* increment the pointer, NOT the pointer
					to the pointer */
	}
	return;
}

/*
**	find the next non-whitespace character.  The address of the pointer
**		is passed and the pointer itself is changed.
*/
void
S_skipspace(theptr)
char **theptr;
{
	while((**theptr != '\0') && isspace(**theptr))
	{
		(*theptr)++;	/* increment the pointer, NOT the pointer
					to the pointer */
	}
	return;
}

/*
**	move the pointer to the beginning of the next word
*/
void
S_nextword(theptr)
char **theptr;
{
	S_skipword(theptr);
	S_skipspace(theptr);
	return;
}

/*
**	see if the first string is a prefix of the second
**		returns 0 if yes
**		non zero if now
**		sigh -- the way strcmp does
*/
int
S_wordcmp(s1,s2)
char *s1,*s2;
{
	return(strncmp(s1,s2,strlen(s2)));
}

/*
**	chop off any trailing zeros on a string
**		but leave one zero if there are only zeros
*/
void
S_trimzeros(str)
char *str;
{
	/*
	**	end starts out pointing at the null terminator
	*/
	char *end = str + strlen(str);

	/*
	**	if there is more than one character left in the string
	*/
	while(end > (str+1))
	{
		--end;
		if ('0' == *end)
		{
			*end = '\0';
		}
		else
		{
			return;
		}
	}
	return;
}

/*
**	save a copy of the string
*/
void
S_savestr(to,from)
char **to,*from;
{
	S_allocstr(to,strlen(from));
	(void) strcpy(*to,from);
	return;
}

/*
**	save cnt characters of the string
*/
void
S_savenstr(to,from,cnt)
char **to,*from;
{
	S_allocstr(to,cnt);
	(void) strncpy(*to,from,cnt);
	*((*to)+cnt) = '\0';
	return;
}

/*
**	allocate space for a string,  add 1 to size to
**		make sure that there is room for the terminating null character
*/
void
S_allocstr(to,size)
char **to;
int size;
{
	*to = Z_ALLOC(size+1,char);
}
