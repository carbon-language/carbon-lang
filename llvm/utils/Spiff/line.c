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
#include "misc.h"
#include "token.h"
#include "line.h"

char *_L_al[_L_MAXLINES];  /* storage for lines */
char *_L_bl[_L_MAXLINES];

int   _L_ai[_L_MAXLINES];  /* index from token line number to first token */
int   _L_bi[_L_MAXLINES];

int   _L_ac[_L_MAXLINES];  /* count of tokens on this token line */
int   _L_bc[_L_MAXLINES];

int   _L_arlm;		/* count of real lines in the file */
int   _L_brlm;

int   _L_aclm;		/* count of content lines in the file */
int   _L_bclm;

int   _L_atlm;		/* count of token lines in the file */
int   _L_btlm;

int   _L_aclindex[_L_MAXLINES]; /* mapping from content lines to real lines*/
int   _L_bclindex[_L_MAXLINES]; 

int   _L_atlindex[_L_MAXLINES]; /*mapping from token lines to content lines */
int   _L_btlindex[_L_MAXLINES];


static void
_L_setrline(file,X,str)
int file;
int X;
char *str;
{
	if (file)
	{
		S_savestr(&_L_bl[X],str);
	}
	else
	{
		S_savestr(&_L_al[X],str);
	}
	return;
}
/*
**	returns 1 if we reached the end of file
**	returns 0 if there is more to do
**
**	stores data and sets maximum counts
*/
int
L_init_file(fnumber,fname)
int fnumber;
char *fname;
{
	extern char *fgets();
	FILE *fp;
	static char buf[Z_LINELEN+2];	/* +2 is to leave room for us to add
						a newline if we need to */
	int ret_val = 1;
	int tmplen;

	if ((fp = fopen(fname,"r")) == (FILE*) NULL)
	{
		(void) sprintf(Z_err_buf, "Cannot open file %s.\n",fname);
		Z_fatal(Z_err_buf);
	}

	/*
	**	clear the line count
	*/
	_L_setrlmx(fnumber,0);

	/*
	**	read in the entire file
	*/
	while (fgets(buf,Z_LINELEN+1,fp) != (char *) NULL)
	{
		tmplen = strlen(buf);
		if (tmplen <= 0)
		{
			(void) sprintf(Z_err_buf,
			  "fatal error -- got 0 length line %d in file %s\n",
				L_getrlmax(fnumber)+1,
				fname);
			Z_fatal(Z_err_buf);
		}
		else if (tmplen > Z_LINELEN)
		{
			(void) sprintf(Z_err_buf,
  "got fatally long line %d in file %s length is %d, must be a bug\n",
				L_getrlmax(fnumber)+1,
				fname,tmplen);
			Z_fatal(Z_err_buf);
		}
		/*
		**	look for newline as last character
		*/
		if ('\n' != buf[tmplen-1])
		{
			/*
			**	did we run out room in the buffer?
			*/
			if (tmplen == Z_LINELEN)
			{
			(void) sprintf(Z_err_buf,
	"line %d too long in file %s, newline added after %d characters\n",
				L_getrlmax(fnumber)+1,
				fname,Z_LINELEN);
			Z_complain(Z_err_buf);
			}
			else
			{
			(void) sprintf(Z_err_buf,
	"didn't find a newline at end of line %d in file %s, added one\n",
				L_getrlmax(fnumber)+1,
				fname);
			Z_complain(Z_err_buf);
			}

			buf[tmplen] = '\n';
			buf[tmplen+1] = '\0';
		}

		_L_setrline(fnumber,L_getrlmax(fnumber),buf);

		if (L_getrlmax(fnumber) >= _L_MAXLINES-1)
		{
			(void) sprintf(Z_err_buf,
	"warning -- ran out of space reading %s, truncated to %d lines\n",
				fname,_L_MAXLINES);
			Z_complain(Z_err_buf);
			ret_val= 0;
			break;
		}
		else
		{
			/*
			**	increment the line count
			*/
			_L_incrlmx(fnumber);
		}

	}

	(void) fclose(fp);
	/*
	**	reset line numbers
	*/
	L_setclmax(fnumber,0);
	L_settlmax(fnumber,0);

	return(ret_val);
}

