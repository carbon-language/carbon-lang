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
#include "flagdefs.h"
#include "float.h"
#include "tol.h"
#include "token.h"
#include "line.h"
#include "command.h"
#include "comment.h"
#include "parse.h"


#include <ctype.h>

#define _P_PARSE_CHATTER	1000


static	int _P_realline;	/* loop counter */
static  int _P_fnumb;

static  char *_P_nextchr;	/* pointer to the next character to parse */
static	char *_P_firstchr;		/* pointer to the beginning of the line being parsed */
static	int _P_next_tol;		/* number of floats seen on this line */
static	int _P_stringsize;		/* count of number of characters that are being
					read into a comment or literal */
static int _P_has_content;	/* flag to indicate if the line being
					parsed has any tokens on it */
static int _P_start;		/* first line to parse */
static int _P_lcount;		/* number of lines to parse */

static int _P_flags;		/* location for global flags */

/*
**	by default, "words" can be made up of numbers and letters
**	the following code allows for extending the alphabet that can
**	be used in words. this is useful for handling languages such
**	as C where the underscore character is an allowable character
**	in an identifier.  If a character (such as underscore) is NOT added
**	to the alphabet, the identifier will be broken into 2 or more "words"
**	by the parser.  as such the two sequences
**			one_two
**		and
**			one _ two
**	would look identical to spiff.
*/
#define _P_ALPHALEN 256
static char _P_alpha[_P_ALPHALEN];

static void
_P_alpha_clear()
{
	*_P_alpha = '\0';
}

static
_P_in_alpha(chr)
char chr;
{
#ifndef ATT
	extern int index();
#endif
	/*
	**	special case when string terminator
	**	is handed to us
	*/
	if ('\0' == chr)
		return(0);

#ifdef ATT
	return((int) strchr(_P_alpha,chr));
#else
	return((int) index(_P_alpha,chr));
#endif
}

void
P_addalpha(ptr)
char *ptr;
{
	char buf[Z_LINELEN];

	S_wordcpy(buf,ptr);		/* copy up to (but not including)
						the first whitespace char */

	if ((strlen(_P_alpha) + strlen(buf)) >= _P_ALPHALEN)
	{
		Z_fatal("too many characters added to extended alphabet");
	}
	(void) strcat(_P_alpha,buf);
}

/*
**	put parser in a default state
*/

static char _P_dummyline[2];	/* a place to aim wild pointers */
static void
_P_initparser()
{
	_P_dummyline[0] = '\0';

	/*
	**	now reset all the state of each module
	*/
	C_clear_cmd();		/* disable embedded command key word */ 
	T_clear_tols();
	W_clearcoms();
	W_clearlits();
	_P_alpha_clear();	/* disable extended alphabet */

	/*
	**	and set state as defined by execute-time commands.
	*/
	C_docmds();
	return;
}


static
_P_needmore()
{
	return(*_P_nextchr == '\0');
}

static
_P_nextline()
{
	/*
	**	if the line that we just finished had
	**		some content,  increment the count
	*/
	if (_P_has_content)
	{
		L_incclmax(_P_fnumb);
		/*
		**	if the previous line had a token
		**		increment the line
		*/
		if (L_getcount(_P_fnumb,L_gettlmax(_P_fnumb)))
		{
			L_inctlmax(_P_fnumb);
			L_setcount(_P_fnumb,L_gettlmax(_P_fnumb),0);
		}
		_P_has_content = 0;
	}

	/*
	**	reset the number of floats seen on the line
	*/
	_P_next_tol = 0;

	/*
	**	get another line if there is one available
	*/
	_P_realline++;
	if (_P_realline >= _P_start+_P_lcount)
	{
		return(1);
	}

	_P_firstchr = _P_nextchr = L_getrline(_P_fnumb,_P_realline);
	/*
	**	and look for a command
	*/
	if (C_is_cmd(_P_firstchr))
	{
		_P_nextchr = _P_dummyline;
		_P_has_content = 0;
	}
	else
	{
		/*
		**	we have a real line, so set up the index
		*/
		L_setclindex(_P_fnumb,L_getclmax(_P_fnumb),_P_realline);
		_P_has_content = 1;
	}
	return(0);
}

/*
**	the following three routines (_P_litsnarf, _P_bolsnarf, and _P_comsnarf
**	all do roughly the same thing. they scan ahead and collect the
**	specified string, move _P_nextchr to the end of the
**	comment or literal and return 1 if we run off the end of file,
**	0 otherwise.  it would have been nice to have 1 routine handle
**	all three task (there is much common code), however there were
**	so enough differences, (for instance, only comments check for nesting,
**	only literals need to set _P_stringsize, etc)
**	that I decided to split them up.
*/
static int
_P_litsnarf(litptr)
W_lit litptr; 
{
	_P_stringsize = 0;
	/*
	**	skip the start of literal string
	*/
	_P_nextchr += strlen(W_litbegin(litptr));
	_P_stringsize += strlen(W_litbegin(litptr));
	/*
	**	is there a separate end string?
	**		if not, then we're done
	*/
	if ('\0' == *(W_litend(litptr)))
	{
		return(0);
	}
	/*
	**	loop once for each character in the literal
	*/
	while(1)
	{
		/*
		**	if we are out of characters, move on to next line
		*/
		if (_P_needmore())
		{
			if (_P_nextline())
			{
				return(1);
			}
			if (!_P_has_content)
			{
				/*
				**	since we've just gotten a command
				**		check to see if this literal
				**		is still legit ...
				**		could have just been reset
				**		by the command
				*/
				if (!W_is_lit(litptr))
				{
					return(0);
				}
			}
		} /* if _P_needmore */

		/*
		**	see if we have an escaped end of literal string
		*/
		if (('\0' != *(W_litescape(litptr))) && /* escape string exists */
		  !S_wordcmp(_P_nextchr,
			   W_litescape(litptr)) &&     /* and escape matches */
		  !S_wordcmp(_P_nextchr+strlen(W_litescape(litptr)),
			   W_litend(litptr)))	     /* and endstring matches */
		{
			_P_nextchr += strlen(W_litescape(litptr))
					+ strlen(W_litend(litptr));
			_P_stringsize += strlen(W_litescape(litptr))
					+ strlen(W_litend(litptr));
			continue;
		}

		/*
		**	see if we have an end of literal string
		*/
		if (!S_wordcmp(_P_nextchr,W_litend(litptr))) /* escape matches */
		{
			_P_nextchr += strlen(W_litend(litptr));
			_P_stringsize += strlen(W_litend(litptr));
			return(0);
		}
		/*
		**	this must be yet another character in the literal, so
		**	just snarf it up
		*/
		_P_nextchr++;
		_P_stringsize++;
	}	/* while loop once for each character */

#ifndef lint
	Z_fatal("shouldn't execute this line at the end of _P_litsnarf");
#endif
} /* _P_litsnarf */

static int
_P_bolsnarf(bolptr)
W_bol bolptr; 
{
	/*
	**	skip the start of comment string
	*/
	_P_nextchr += strlen(W_bolbegin(bolptr));
	/*
	**	is there a separate end string
	**		if not, then we're done
	*/
	if ('\0' == *(W_bolend(bolptr)))
	{
		return(0);
	}
	/*
	**	loop once for each character in the comment
	*/
	while(1)
	{
		/*
		**	if we are out of characters,move on to next line
		*/
		if (_P_needmore())
		{
			if (_P_nextline())
			{
				return(1);
			}
			if (!_P_has_content)
			{
				/*
				**	since we've just gotten a command
				**		check to see if this comment
				**		is still legit ... comments
				**		could have just been reset
				**		by the command
				*/
				if (!W_is_bol(bolptr))
				{
					return(0);
				}
			}
		} /* if at end of line */

		/*
		**	see if we have an escaped end of comment string
		*/
		if ('\0' != *(W_bolescape(bolptr)) && /* escape string exists */
		  !S_wordcmp(_P_nextchr,
			   W_bolescape(bolptr)) &&     /* and escape matches */
		  !S_wordcmp(_P_nextchr+strlen(W_bolescape(bolptr)),
			   W_bolend(bolptr)))	 /* and end string matches */
		{
			_P_nextchr += strlen(W_bolescape(bolptr))
					+ strlen(W_bolend(bolptr));
			continue;
		}

		/*
		**	see if we have an end of comment string
		*/
		if (!S_wordcmp(_P_nextchr,W_bolend(bolptr)))
		{
			_P_nextchr += strlen(W_bolend(bolptr));
			return(0);
		}
		/*
		**	this must be yet another character in the comment, so
		**	just snarf it up
		*/
		_P_nextchr++;
	}	/* while loop once for each character */

#ifndef lint
	Z_fatal("shouldn't execute this line in at end of _P_bolsnarf");
#endif
} /* _P_bolsnarf */

/*
**	pass over a comment -- look for nexting
*/
static
_P_comsnarf(comptr)
W_com comptr; 
{
	int depth = 1; /* nesting depth */
	/*
	**	skip the start of comment string
	*/
	_P_nextchr += strlen(W_combegin(comptr));

	/*
	**	is there a separate end string
	**		if not, then we're done
	*/
	if ('\0' == *(W_comend(comptr)))
	{
		return(0);
	}
	/*
	**	loop once for each character in the comment
	*/
	while(1)
	{
		/*
		**	if we are out of characters, move on to next line
		*/
		if (_P_needmore())
		{
			if (_P_nextline())
			{
				return(1);
			}
			if (!_P_has_content)
			{
				/*
				**	since we've just gotten a command
				**		check to see if this comment
				**		is still legit ... comments
				**		could have just been reset
				**		by the command
				*/
				if (!W_is_com(comptr))
				{
					return(0);
				}
			}
		} /* if at end of line */

		/*
		**	see if we have an escaped end of comment string
		*/
		if ('\0' != *(W_comescape(comptr)) &&  /* escape string exists */
		  !S_wordcmp(_P_nextchr,
			   W_comescape(comptr)) &&    /* and escape matches */
		  !S_wordcmp(_P_nextchr+strlen(W_comescape(comptr)),
			   W_comend(comptr)))	/* and end string matches */
		{
			/*
			** skip over the escape sequence and the end sequence
			*/
			_P_nextchr += strlen(W_comescape(comptr))
					+ strlen(W_comend(comptr));
			continue;
		}

		/*
		**	see if we have an end of comment string
		*/
		if (!S_wordcmp(_P_nextchr,W_comend(comptr))) /* end  matches */
		{
			/*
			**	skip over the end sequence
			*/
			_P_nextchr += strlen(W_comend(comptr));
			if (W_is_nesting(comptr))
			{
				depth--;
				if (0 == depth)
					return(0);
			}
			else
			{
				return(0);
			}
			continue;
		}
		/*
		**	see if we have another beginning of comment string
		*/
		if (W_is_nesting(comptr) &&
			!S_wordcmp(_P_nextchr,W_comend(comptr))) /* end matches */
		{
			_P_nextchr += strlen(W_comend(comptr));
			depth++;
			continue;
		}
		/*
		**	this must be yet another character in the comment, so
		**	just snarf it up
		*/
		_P_nextchr++;
	}	/* while loop once for each character */

#ifndef lint
		Z_fatal("should not execute this line in _P_comsnarf\n");
#endif

} /* _P_comsnarf */


/*
**	parse a file
*/
static void
_P_do_parse()
{

	char *ptr;		/* scratch space */
	int tmp;
	int ret_code;

	K_token newtoken;
	W_bol bolptr;
	W_com comptr;
	W_lit litptr;

	int startline, endline, startpos;

	/*
	**	main parsing loop
	*/
	while (1)
	{
		/*
		**	get more text if necessary
		*/
		if (_P_needmore())
		{
			if (_P_nextline())
			{
				return;
			}

			/*
			**	if the line contains nothing of interest,
			**		try again
			*/
			if (!_P_has_content)
			{
				continue;
			}

			/*
			**	check to see if this line starts a comment
			*/
			if ((bolptr = W_isbol(_P_firstchr)) != W_BOLNULL)
			{
				if (_P_bolsnarf(bolptr))
				{
					return;
				}
				continue;
			}
		} /* if _P_needmore */

		/*
		**	skip whitespace
		*/
		if (!(U_INCLUDE_WS & _P_flags) && isspace(*_P_nextchr))
		{
			_P_nextchr++;
			continue;
		}

		/*
		**	check to see if this character starts a comment
		*/
		if ((comptr = W_iscom(_P_nextchr)) != W_COMNULL)
		{
			if (_P_comsnarf(comptr))
			{
				return;
			}
			continue;
		}

		/*
		**	if there aren't any tokens on this line already
		**	set up the index from the token line to the content line
		*/
		if (!L_getcount(_P_fnumb,L_gettlmax(_P_fnumb)))
		{
			L_settlindex(_P_fnumb,
					L_gettlmax(_P_fnumb),
					L_getclmax(_P_fnumb));
			/*
			**	and the pointer from the token line to the 
			** 	first  token on the line
			*/
			L_setindex(_P_fnumb,
					L_gettlmax(_P_fnumb),
					K_gettmax(_P_fnumb));
		}

		startline =  L_tl2cl(_P_fnumb,L_gettlmax(_P_fnumb));
		startpos = _P_nextchr-_P_firstchr;

		newtoken = K_maketoken();
		K_setline(newtoken,L_gettlmax(_P_fnumb));
		K_setpos(newtoken,startpos);

		ret_code = 0;
		/*
		**	check to see if this character starts a
		**		delimited literal string
		*/
		if ((litptr = W_islit(_P_nextchr)) != W_LITNULL)
		{
			ret_code = _P_litsnarf(litptr);
			K_settype(newtoken,K_LIT);
			S_allocstr(&ptr,_P_stringsize);
			/*
			**	fixed nasty memory bug here by adding else
			**	old code copied entire line even if literal
			**	ended before the end of line
			**		should check into getting strcpy loaded
			**		locally
			*/
			endline = L_getclmax(_P_fnumb);
			if (endline > startline)
			{
				/*
				**	copy in the first line of the literal
				*/
				(void) strcpy(ptr,
					      L_getcline(_P_fnumb,startline)
							+startpos);
				/*
				**	now copy all the lines between
				**		the first and last
				*/
				for (tmp=startline+1;tmp<endline;tmp++)
				{
					(void) strcat(ptr,
						      L_getcline(_P_fnumb,tmp));
				}
				/*
				**	and now copy in the last line
				*/
				(void) strncat(ptr,
					       L_getcline(_P_fnumb,endline),
					       _P_stringsize-strlen(ptr));
			}
			else
			{
				(void) strncpy(ptr,
					       L_getcline(_P_fnumb,startline)
								+startpos,
					      _P_stringsize);
				/*
				**	terminate the string you just copied
				*/
				ptr[_P_stringsize] = '\0';
			}
			K_settext(newtoken,ptr);
		} /* if is_lit */

		/*
		**	see if this is a floating point number
		*/
		else if (tmp = F_isfloat(_P_nextchr,
				       _P_flags & U_NEED_DECIMAL,
				       _P_flags & U_INC_SIGN))
		{
			K_saventext(newtoken,_P_nextchr,tmp);
			K_settype(newtoken,K_FLO_NUM);
			if (!(_P_flags & U_BYTE_COMPARE))
			{
				K_setfloat(newtoken,
					   F_atof(K_gettext(newtoken),
					   USE_ALL));

				/*
				**	assign the curent tolerance
				*/
				K_settol(newtoken,T_gettol(_P_next_tol));
			}

			/*
			**	use next tolerance in the
			**		specification if there is one
			*/
			if (T_moretols(_P_next_tol))
			{
				_P_next_tol++;
			}
			/*
			**	and move pointer past the float
			*/
			_P_nextchr += tmp;
		}

		/*
		**	is this a fixed point number
		*/
		else if (isdigit(*_P_nextchr))
		{
			for(ptr=_P_nextchr; isdigit(*ptr); ptr++)
			{
			}
			K_saventext(newtoken,_P_nextchr,ptr-_P_nextchr);
			K_settype(newtoken,K_LIT);
			_P_nextchr = ptr;
		}

		/*
		**	try an alpha-numeric word
		*/
		else if (isalpha(*_P_nextchr) || _P_in_alpha(*_P_nextchr))
		{
			/*
			**	it's a multi character word
			*/
			for(ptr = _P_nextchr;
			    isalpha(*ptr)
				|| isdigit(*ptr)
				|| _P_in_alpha(*ptr);
			    ptr++)
			{
			}
			K_saventext(newtoken,_P_nextchr,ptr-_P_nextchr);
			K_settype(newtoken,K_LIT);
			_P_nextchr = ptr;
		}
		else
		{
			/*
			**	otherwise, treat the char itself as a token
			*/
			K_saventext(newtoken,_P_nextchr,1);
			K_settype(newtoken,K_LIT);
			_P_nextchr++;
		}

		K_settoken(_P_fnumb,K_gettmax(_P_fnumb),newtoken);
		L_inccount(_P_fnumb,L_gettlmax(_P_fnumb));
		/*
		**	if we are out of space, complain and quit
		*/
		if (K_inctmax(_P_fnumb))
		{
			(void) sprintf(Z_err_buf,
     "warning -- to many tokens in file only first %d tokens will be used.\n",
				       K_MAXTOKENS);
			Z_complain(Z_err_buf);
			return;
		}
#ifndef NOCHATTER
		if (0 == (K_gettmax(_P_fnumb) % _P_PARSE_CHATTER))
		{
			int max = K_gettmax(_P_fnumb);
			(void) sprintf(Z_err_buf,
				"scanned %d words from file #%d\n",
					max,_P_fnumb+1);
			Z_chatter(Z_err_buf);
		}
#endif

		/*
		**	are we done?
		*/
		if(ret_code)
		{
			return;
		}
	}   /* loop once per object on a line */

#ifndef lint 
	Z_fatal("this line should never execute");
#endif
}

void
P_file_parse(num,strt,lcnt,flags)
int num;	/* file number */
int strt;	/* first line to parse expressed in real line numbers */
int lcnt;	/* max number of lines to parse */
int flags;	/* flags for controlling the parse mode */
{
	/*
	**	set module-wide state variables
	*/
	_P_fnumb = num;		
	_P_start = strt;	
	_P_lcount = lcnt;
	_P_flags = flags;

	_P_initparser();

	_P_nextchr = _P_dummyline;

	_P_has_content = 0;
	_P_next_tol = 0;
	L_setcount(_P_fnumb,L_gettlmax(_P_fnumb),0);
	/*
	**	start everything back one line (it will be incremented
	**		just before the first line is accessed
	*/
	_P_realline = _P_start-1;

	_P_do_parse();

	/*
	**	if the last line had content, increment the count
	*/
	if (_P_has_content)
	{
/*
**	this code will get executed if we stopped parsing in the middle
**	of a line.  i haven't looked at this case carefully.
**	so, there is a good chance that it is buggy.
*/
(void) sprintf(Z_err_buf,"parser got confused at end of file\n");
Z_complain(Z_err_buf);
		L_incclmax(_P_fnumb);
		if (L_getcount(_P_fnumb,L_gettlmax(_P_fnumb)))
			L_inctlmax(_P_fnumb);
	}
	return;
}
