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
#include <stdlib.h>
#include <unistd.h>

#ifdef M_TERMINFO
#include <curses.h>
#include <term.h>
#endif

#ifdef M_TERMCAP
#ifdef XENIX
#include <tcap.h>
#endif
#endif

#include "misc.h"
#include "flagdefs.h"
#include "edit.h"
#include "line.h"
#include "token.h"

static int _O_need_init = 1;
static int _O_st_ok = 0;
static int _O_doing_ul = 0;
static	char *_O_st_tmp;
#ifdef M_TERMCAP
static	char _O_startline[Z_WORDLEN];
static	char _O_endline[Z_WORDLEN];
#endif

static void
_O_st_init()
{
	char termn[Z_WORDLEN];
#ifdef M_TERMCAP
	static	char entry[1024];
#endif

	/*
	**	see if standard out is a terminal
	*/
	if (!isatty(1))
	{
		_O_need_init = 0;
		_O_st_ok = 0;
		return;
	}

	if (NULL == (_O_st_tmp = (char*) getenv("TERM")))
	{
		Z_complain("can't find TERM entry in environment\n");
		_O_need_init = 0;
		_O_st_ok = 0;
		return;
	}
	(void) strcpy(termn,_O_st_tmp);

#ifdef M_TERMCAP
	if (1 != tgetent(entry,termn))
	{
		Z_complain("can't get TERMCAP info for terminal\n");
		_O_need_init = 0;
		_O_st_ok = 0;
		return;
	}

	_O_st_tmp = _O_startline;
	_O_startline[0] = '\0';
	tgetstr("so",&_O_st_tmp);

	_O_st_tmp = _O_endline;
	_O_endline[0] = '\0';
	tgetstr("se",&_O_st_tmp);

	_O_st_ok = (strlen(_O_startline) > 0) && (strlen(_O_endline) > 0);
#endif

#ifdef M_TERMINFO
	setupterm(termn,1,&_O_st_ok);
#endif
	_O_need_init = 0;
}

void
O_cleanup()
{
	/*
	**	this probably isn't necessary, but in the
	**	name of compeleteness.
	*/
#ifdef M_TERMINFO
	resetterm();
#endif
}

static void
_O_start_standout()
{
	if (_O_need_init)
	{
		_O_st_init();
	}
	if (_O_st_ok)
	{
#ifdef M_TERMCAP
		(void) printf("%s",_O_startline);
#endif 
#ifdef M_TERMINFO
		vidattr(A_STANDOUT);
#endif
	}
	else
	{
		_O_doing_ul = 1;
	}
}

static void
_O_end_standout()
{
	if (_O_need_init)
	{
		_O_st_init();
	}
	if (_O_st_ok)
	{
#ifdef M_TERMCAP
		(void) printf("%s",_O_endline);
#endif 
#ifdef M_TERMINFO
		vidattr(0);
#endif
	}
	else
	{
		_O_doing_ul = 0;
	}
}

static void
_O_pchars(line,start,end)
char *line;
int start,end;
{
	int cnt;

	for(cnt=start;cnt < end; cnt++)
	{
		if (_O_doing_ul)
		{
			(void) putchar('_');
			(void) putchar('\b');
		}
		(void) putchar(line[cnt]);
	}
}


/*
**	convert a 0 origin token number to a 1 orgin token
**		number or 1 origin line number as appropriate
*/
static int
_O_con_line(numb,flags,filenum)
int numb, flags,filenum;
{
	if (flags & U_TOKENS)
	{
		return(numb+1);
	}
	else
	{
		/*
		**	 check to make sure that this is a real
		**	line number. if not, then return 0
		**	on rare occasions, (i.e. insertion/deletion
		**	of the first token in a file) we'll get
		**	line numbers of -1.  the usual look-up technique
		**	won't work since we have no lines before than 0.
		*/
		if (numb < 0)
			return(0);
		/*
		**	look up the line number the token and then
		**	add 1 to make line number 1 origin
		*/
		return(L_tl2cl(filenum,numb)+1);
	}
}

static char *
_O_convert(ptr)
char *ptr;
{
	static char spacetext[Z_WORDLEN];

	if (1 == strlen(ptr))
	{
		switch (*ptr)
		{
			default:
				break;
			case '\n' :
				(void) strcpy(spacetext,"<NEWLINE>");
				return(spacetext);
			case '\t' :
				(void) strcpy(spacetext,"<TAB>");
				return(spacetext);
			case ' ' :
				(void) strcpy(spacetext,"<SPACE>");
				return(spacetext);
		}
				
	}
	return(ptr);
}

static char*
_O_get_text(file,index,flags)
int file,index,flags;
{
	static char buf[Z_LINELEN*2];	/* leave lots of room for both
						the token text and the
						chatter that preceeds it */
	char *text;
	K_token tmp;

	if (flags & U_TOKENS)
	{
		tmp = K_gettoken(file,index);
		text = _O_convert(K_gettext(tmp));
		(void) sprintf(buf,"%s -- line %d, character %d\n",
				text,
				/*
				**	add 1 to make output start at line 1 
				**	and character numbers start at 1
				*/
				L_tl2cl(file,K_getline(tmp))+1,
				K_getpos(tmp)+1);
		return(buf);
	}
	else
	{
		return(L_gettline(file,index));
	}
}
#define	_O_APP		1
#define _O_DEL		2
#define _O_CHA		3
#define _O_TYPE_E	4

static void
_O_do_lines(start,end,file)
int start,end,file;
{
	int cnt;
	int lastline = -1;
	int nextline;
	K_token nexttoken;
	for (cnt=start;cnt <= end; cnt++)
	{
		nexttoken = K_get_token(file,cnt);
		nextline = K_getline(nexttoken);
		if (lastline != nextline)
		{
			int lastone,lastchar;
			K_token lasttok;
			char linetext[Z_LINELEN+1];	/* leave room for
							   terminator */
			if (0 == file)
			{
				(void) printf("< ");
			}
			else
			{
				(void) printf("> ");
			}

			/*
			**	put loop here if you want to print
			**	out any intervening lines that don't
			**	have any tokens on them
			*/

			/*
			**	following line is necessary because
			**	L_gettline is a macro, and can't be passed
			*/
			(void) strcpy(linetext,L_gettline(file,nextline));
			_O_pchars(linetext,0,K_getpos(nexttoken));
			_O_start_standout();
			/*
			** 	look for last token on this line to be
			**	highlighted
			*/
			for ( lastone=cnt,lasttok = K_get_token(file,lastone);
			      (lastone<=end)&&(nextline == K_getline(lasttok));
				lastone++,lasttok = K_get_token(file,lastone))
			{
			}
			lastone--;
			lasttok = K_get_token(file,lastone);
			lastchar = K_getpos(lasttok)
					+ strlen(K_gettext(lasttok));
			_O_pchars(linetext,K_getpos(nexttoken),lastchar);
			_O_end_standout();
			_O_pchars(linetext,lastchar,strlen(linetext));
			
			lastline = nextline;
		}
	}
}

void
O_output(start,flags)
E_edit start;
int flags;
{
	int type = _O_TYPE_E;	/* initialize to error state
				** this is to make sure that type is set
				** somewhere
				*/
	int t_beg1, t_beg2, t_end1, t_end2; /* token numbers */
	int first1, last1, first2, last2;

	E_edit ep, behind, ahead, a, b;

	/*
	**	reverse the list of edits
	*/
	ahead = start;
	ep = E_NULL;
	while (ahead != E_NULL) {
		/*
		**	set token numbers intentionally out of range
		**		as boilerplate
		*/
		t_beg1 = t_beg2 = t_end1 = t_end2 = -1;
		/*
		**	edit script is 1 origin, all of
		**	 our routines are zero origin
		*/
		E_setl1(ahead,(E_getl1(ahead))-1);
		E_setl2(ahead,(E_getl2(ahead))-1);

		behind = ep;
		ep = ahead;
		ahead = E_getnext(ahead);
		E_setnext(ep,behind);
	}

	/*
	**	now run down the list and collect the following information
	**	type of change (_O_APP, _O_DEL or _O_CHA)
	**	start and length for each file
	*/
	while (ep != E_NULL)
	{
		b = ep;
		/*
		**	operation always start here
		*/
		t_beg1 = E_getl1(ep);
		/*
		**	any deletions will appear before any insertions,
		**	so, if the first edit is an E_INSERT, then this
		**	this is an _O_APP
		*/
		if (E_getop(ep) == E_INSERT)
			type = _O_APP;
		else {
			/*
			**	run down the list looking for the edit
			**	that is not part of the current deletion
			*/	
			do {
				a = b;
				b = E_getnext(b);
			} while ((b != E_NULL) &&
				 (E_getop(b) == E_DELETE) &&
				 ((E_getl1(b)) == ((E_getl1(a))+1)));
			/*
			**	if we have an insertion at the same place
			**	as the deletion we just scanned, then
			**	this is a change
			*/
			if ((b != E_NULL) &&
				((E_getop(b)) == E_INSERT) &&
				((E_getl1(b))==(E_getl1(a))))
			{
				type = _O_CHA;
			}
			else
			{
				type = _O_DEL;
			}
			/*
			**	set up start and length information for
			**	first file
			*/
			t_end1 = E_getl1(a);
			/*
			**	move pointer to beginning of insertion
			*/
			ep = b;
			/*
			**	if we are showing only a deletion,
			**	then we're all done, so skip ahead
			*/ 
			if (_O_DEL == type)
			{
				t_beg2 = E_getl2(a);
				t_end2 = -1;	/* dummy number, won't
							ever be printed */
						   
				goto skipit;
			}
		}
		t_beg2 = E_getl2(ep);
		t_end2 = t_beg2-1;
		/*
		**	now run down the list lookingfor the
		**	end of this insertion and keep count
		**	of the number of times we step along
		*/
		do {
			t_end2++;
			ep = E_getnext(ep);
		} while ((ep != E_NULL) && ((E_getop(ep)) == E_INSERT) &&
					((E_getl1(ep)) == (E_getl1(b))));

skipit:;
		if (flags & U_TOKENS)
		{
			/*
			**	if we are dealing with tokens individually,
			**	then just print then set printing so
			*/
				first1 = t_beg1;
				last1 = t_end1;
				first2 = t_beg2;
				last2 = t_end2;
		}
		else
		{
			/*
			**	we are printing differences in terms of lines
			**	so find the beginning and ending lines of the
			**	changes and print header in those terms
			*/
			if ( t_beg1 >= 0)
				first1 = K_getline(K_get_token(0,t_beg1));
			else
				first1 = t_beg1;

			if ( t_end1 >= 0)
				last1 = K_getline(K_get_token(0,t_end1));
			else
				last1 = t_end1;

			if ( t_beg2 >= 0)
				first2 = K_getline(K_get_token(1,t_beg2));
			else
				first2 = t_beg2;

			if ( t_end2 >= 0)
				last2 = K_getline(K_get_token(1,t_end2));
			else
				last2 = t_end2;

		}
		/*
		**	print the header for this difference
		*/
		(void) printf("%d",_O_con_line(first1,flags,0));
		switch (type)
		{
		case _O_APP :
			(void) printf("a%d",_O_con_line(first2,flags,1));
			if (last2 > first2)
			{
				(void) printf(",%d",_O_con_line(last2,flags,1));
			}
			(void) printf("\n");
			break;
		case _O_DEL :
			if (last1 > first1)
			{
				(void) printf(",%d",_O_con_line(last1,flags,0));
			}
			(void) printf("d%d\n",_O_con_line(first2,flags,1));
			break;
		case _O_CHA :
			if (last1 > first1)
			{
				(void) printf(",%d",_O_con_line(last1,flags,0));
			}
			(void) printf("c%d",_O_con_line(first2,flags,1));
			if (last2 > first2)
			{
				(void) printf(",%d",_O_con_line(last2,flags,1));
			}
			(void) printf("\n");
			break;
		default:
			Z_fatal("type in O_output wasn't set\n");
		}
		if (_O_DEL == type || _O_CHA == type)
		{
			if (flags & U_TOKENS)
			{
				int cnt;
				for(cnt=first1;cnt <= last1; cnt++)
				{
		(void) printf("< %s",
							_O_get_text(0,cnt,flags));
				}
			}
			else
			{	
				_O_do_lines(t_beg1,t_end1,0);
			}
		}
		if (_O_CHA == type)
		{
			(void) printf("---\n");
		}
		if (_O_APP == type || _O_CHA == type)
		{
			if (flags & U_TOKENS)
			{
				int cnt;
				for(cnt=first2;cnt <= last2; cnt++)
				{
					(void) printf("> %s",
						_O_get_text(1,cnt,flags));
				}
			}
			else
			{
				_O_do_lines(t_beg2,t_end2,1);
			}
		}
	}
	O_cleanup();
	return;
}
