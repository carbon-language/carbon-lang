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
#include "comment.h"
#include "strings.h"

/*
**	storage for the comment specifiers that can appear
**		anywhere on a line
*/
static int _W_nextcom = 0;
_W_comstruct _W_coms[_W_COMMAX];

/*
**	storage for comment specifiers that are examined only at the
**		beginning of each line
*/
static int _W_nextbol = 0;
_W_bolstruct _W_bols[_W_BOLMAX];

/*
**	storage for delimiters of literal strings
*/
static int _W_nextlit = 0;
_W_litstruct _W_lits[_W_LITMAX];

/*
**	storage for characters to specify beginning and end of line
**	in the comment and literal commands
*/
char _W_bolchar = '^';
char _W_eolchar = '$';


/*
**	build up a list of comment delimiters
*/
void
W_addcom(str,nestflag)
char *str;
int nestflag;
{
	/*
	**	check for comments that begin at the beginning of line
	*/
	if (*str ==  _W_bolchar)
	{
		if (_W_nextbol >= _W_BOLMAX)
			Z_fatal("too many beginning of line comment delimiter sets");

		str++;	/*skip the bol char */
		S_wordcpy(_W_bols[_W_nextbol].begin,str);

		S_nextword(&str);

		if (*str == _W_eolchar)
		{
			(void) strcpy(_W_bols[_W_nextbol].end,"\n");
		}
		else
		{
			S_wordcpy(_W_bols[_W_nextbol].end,str);
		}

		S_nextword(&str);
		S_wordcpy(_W_bols[_W_nextbol].escape,str);

		/*
		**
		*/
		if (nestflag)
			Z_complain("begining of line comment won't nest");

		_W_nextbol++;
	}
	else
	{
		if (_W_nextcom >= _W_COMMAX)
			Z_fatal("too many comment delimiter sets");

		S_wordcpy(_W_coms[_W_nextcom].begin,str);

		S_nextword(&str);

		if (*str == _W_eolchar)
		{
			(void) strcpy(_W_coms[_W_nextbol].end,"\n");
		}
		else
		{
			S_wordcpy(_W_coms[_W_nextbol].end,str);
		}

		S_nextword(&str);
		S_wordcpy(_W_coms[_W_nextcom].escape,str);

		_W_coms[_W_nextcom].nestbit = nestflag;

		_W_nextcom++;
	}
	return;
}


/*
**	clear the comment delimiter storage
*/
void
W_clearcoms()
{
	_W_nextcom = 0;
	_W_nextbol = 0;
	return;
}

/*
**	build up the list of literal delimiters
*/
void
W_addlit(str)
char *str;
{
	if (_W_nextlit >= _W_LITMAX)
		Z_fatal("too many literal delimiter sets");

	S_wordcpy(_W_lits[_W_nextlit].begin,str);

	S_nextword(&str);
	S_wordcpy(_W_lits[_W_nextlit].end,str);

	S_nextword(&str);
	S_wordcpy(_W_lits[_W_nextlit].escape,str);

	_W_nextlit++;
	return;
}

/*
**	clear the literal delimiter storage
*/
void
W_clearlits()
{
	_W_nextlit = 0;
	return;
}



static _W_bolstruct bol_scratch;

static void
_W_copybol(to,from)
W_bol to,from;
{
	(void) strcpy(to->begin,from->begin);
	(void) strcpy(to->end,from->end);
	(void) strcpy(to->escape,from->escape);
}

W_bol
W_isbol(str)
char *str;
{
	int i;

	for(i=0;i<_W_nextbol;i++)
	{
		if(!S_wordcmp(str,_W_bols[i].begin))
		{
			_W_copybol(&bol_scratch,&_W_bols[i]);
			return(&bol_scratch);
		}
	}
	return(W_BOLNULL);
}

int
W_is_bol(ptr)
W_bol ptr;
{
	int i;

	for(i=0;i<_W_nextbol;i++)
	{
		if(!S_wordcmp(ptr->begin,_W_bols[i].begin) &&
			!S_wordcmp(ptr->end,_W_bols[i].end) &&
			!S_wordcmp(ptr->escape,_W_bols[i].escape))
		{
			return(1);
		}

	}
	return(0);
}


static _W_litstruct lit_scratch;

static void
_W_copylit(to,from)
W_lit to,from;
{
	(void) strcpy(to->begin,from->begin);
	(void) strcpy(to->end,from->end);
	(void) strcpy(to->escape,from->escape);
}

W_lit
W_islit(str)
char *str;
{
	int i;

	for(i=0;i<_W_nextlit;i++)
	{
		if(!S_wordcmp(str,_W_lits[i].begin))
		{
			_W_copylit(&lit_scratch,&_W_lits[i]);
			return(&lit_scratch);
		}
	}
	return(W_LITNULL);
}

int
W_is_lit(ptr)
W_lit ptr;
{
	int i;

	for(i=0;i<_W_nextlit;i++)
	{
		if(!S_wordcmp(ptr->begin,_W_lits[i].begin) &&
			!S_wordcmp(ptr->end,_W_lits[i].end) &&
			!S_wordcmp(ptr->escape,_W_lits[i].escape))
		{
			return(1);
		}

	}
	return(0);
}

static _W_comstruct com_scratch;

static void
_W_copycom(to,from)
W_com to,from;
{
	(void) strcpy(to->begin,from->begin);
	(void) strcpy(to->end,from->end);
	(void) strcpy(to->escape,from->escape);
	to->nestbit = from->nestbit;
}

W_com
W_iscom(str)
char *str;
{
	int i;

	for(i=0;i<_W_nextcom;i++)
	{
		if(!S_wordcmp(str,_W_coms[i].begin))
		{
			_W_copycom(&com_scratch,&_W_coms[i]);
			return(&com_scratch);
		}
	}
	return(W_COMNULL);
}

int
W_is_com(ptr)
W_com ptr;
{
	int i;

	for(i=0;i<_W_nextcom;i++)
	{
		if(!S_wordcmp(ptr->begin,_W_coms[i].begin) &&
			!S_wordcmp(ptr->end,_W_coms[i].end) &&
			!S_wordcmp(ptr->escape,_W_coms[i].escape) &&
			ptr->nestbit == _W_coms[i].nestbit)
		{
			return(1);
		}

	}
	return(0);
}

int
W_is_nesting(ptr)
W_com ptr;
{
	return(ptr->nestbit);
}
