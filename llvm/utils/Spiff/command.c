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
#include "tol.h"
#include "comment.h"
#include "command.h"
#include "strings.h"
#include "parse.h"

/*
**	storage for the string that signals an embedded command
*/
static char _C_cmdword[Z_WORDLEN];

/*
**	storage for the command script
*/
static int _C_nextcmd = 0;
static char *_C_cmds[_C_CMDMAX];


/*
**	add a string to the command buffer
*/
void
C_addcmd(str)
char *str;
{
	S_savestr(&_C_cmds[_C_nextcmd++],str);
	return;
}

/*
**	execute a single command
*/
static void
_C_do_a_cmd(str)
char *str;
{
	/*
	**	place holder for the beginning of the string
	*/
	char *beginning = str;

	S_skipspace(&str);

	/*
	**	set the command string to allow embedded commands
	*/
	if 	(!S_wordcmp(str,"command"))
	{
		S_nextword(&str);
		if (strlen(str) >= Z_WORDLEN)
		{
			Z_fatal("command word is too long");
		}
		S_wordcpy(_C_cmdword,str);
	}
	/*
	**	set the tolerances
	*/
	else if (!S_wordcmp(str,"tol"))
	{
		S_nextword(&str);
		T_tolline(str);
	}
	/*
	**	add a comment specification
	*/
	else if (!S_wordcmp(str,"comment"))
	{
		S_nextword(&str);
		if (strlen(str) >= Z_WORDLEN)
		{
			Z_fatal("command word is too long");
		}
		W_addcom(str,0);
	}
	else if (!S_wordcmp(str,"nestcom"))
	{
		S_nextword(&str);
		if (strlen(str) >= Z_WORDLEN)
		{
			Z_fatal("command word is too long");
		}
		W_addcom(str,1);
	}
	/*
	**	add a literal string specification
	*/
	else if (!S_wordcmp(str,"literal"))
	{
		S_nextword(&str);
		if (strlen(str) >= Z_WORDLEN)
		{
			Z_fatal("command word is too long");
		}
		W_addlit(str);
	}
	else if (!S_wordcmp(str,"resetcomments"))
	{
		W_clearcoms();
	}
	else if (!S_wordcmp(str,"resetliterals"))
	{
		W_clearlits();
	}
	else if (!S_wordcmp(str,"beginchar"))
	{
		S_nextword(&str);
		W_setbolchar(*str);
	}
	else if (!S_wordcmp(str,"endchar"))
	{
		S_nextword(&str);
		W_seteolchar(*str);
	}
	else if (!S_wordcmp(str,"addalpha"))
	{
		S_nextword(&str);
		P_addalpha(str);
	}
	else if ((0 == strlen(str)) || !S_wordcmp(str,"rem") 
				    || ('#' == *str))
	{
		/* do nothing */
	}
	else
	{
		(void) sprintf(Z_err_buf,
			       "don't understand command %s\n",
			       beginning);
		Z_fatal(Z_err_buf);
	}
	return;
}

/*
**	execute the commands in the command buffer
*/
void
C_docmds()
{
	int i;
	for (i=0;i<_C_nextcmd;i++)
	{
		_C_do_a_cmd(_C_cmds[i]);
	}
	return;
}

/*
**	disable embedded command key word recognition
*/
void
C_clear_cmd()
{
	_C_cmdword[0] = '\0';
	return;
}

#define inline spiff_inline
int
C_is_cmd(inline)
char *inline;
{
	char *ptr;
	/*
	**	see if this is a command line
	**	and if so, do the command right away
	*/
	if (('\0' != _C_cmdword[0]) && (!S_wordcmp(inline,_C_cmdword)))
	{
		ptr = inline;
		S_nextword(&ptr);
		_C_do_a_cmd(ptr);
		return(1);
	}
	return(0);
}

