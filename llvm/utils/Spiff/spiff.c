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
#include "misc.h"
#include "flagdefs.h"
#include "parse.h"
#include "edit.h"
#include "line.h"
#include "token.h"
#include "tol.h"
#include "command.h"
#include "compare.h"
#include "exact.h"
#include "miller.h"
#include "visual.h"
#include "output.h"

extern int L_init_file();
extern int V_visual();

static void _Y_doargs();

static int _Y_eflag = 0;	/* use exact match algorithm */
static int _Y_vflag = 0;	/* use visual mode */

/*
**	this is the set of flags that gets used throughout the top module
**	as well as being used to communicate between modules.
*/
static int _Y_flags;

int
main(argc,argv)
int argc;
char *argv[];
{
	E_edit edit_end;
	char *filename[2];

	int max_d; 	/* max number of differences allowed */
	int i;		/* loop counter */

	/*
	**	parse the command line
	*/
	_Y_doargs(argc,argv,&(filename[0]),&(filename[1]),&max_d);

	/*
	**	initialize the default tolerance if it
	**		hasn't been set already.
	*/
	T_initdefault();

	/*
	**	read and then parse the files
	*/

	/*
	**	L_initfile return a code that indicates if the
	**	entire file was read or not
	**
	**	P_fileparse also knows how to start at someplace other
	**		than the first line of file
	**
	**	Taken together, this is enough to do step our way
	**		through the file using an exact match algorithm.
	**
	**	Oh well, someday . . .
	*/
	for(i=0;i<=1;i++)
	{
		/*
		**	read the file into core
		*/
		(void) L_init_file(i,filename[i]);
		K_settmax(i,0);		/* start tokens at 0 */
		/*
		**	and parse the files into tokens
		*/
		P_file_parse(i,0,L_getrlmax(i),_Y_flags);
	}

	if (_Y_vflag)
	{
		return(V_visual(_Y_flags));
	}

	/*
	**	if max_d was not set on the command line
	**		set it to be as large as is possible
	**		since the most changes possible would
	**		be to delete all the tokens in the
	**		first file and add all the tokens from
	**		the second, the max possible is the
	**		sum of the number of tokens in the
	**		two files.
	*/
	if (-1 == max_d)
		max_d = K_gettmax(0) + K_gettmax(1);

	if (_Y_eflag)
	{
		edit_end = Q_do_exact(K_gettmax(0),K_gettmax(1),
					max_d,_Y_flags);
	}
	else
	{
		edit_end = G_do_miller(K_gettmax(0), K_gettmax(1),
				     max_d,_Y_flags);
	}

	if (E_NULL != edit_end)
	{
		O_output(edit_end,_Y_flags);
		return(1);
	}
	return(0);
}

/*
**	break a string into individual lines and feed
**		them to the command module
*/
static void
_Y_cmdlines(from)
char *from;
{
	char buf[Z_LINELEN]; 
	char *to;
	while ('\0' != *from)
	{
		/*
		**	copy line into buf
		*/
		to = buf;
		while (('\0' != *from) && ('\n' != *from))
		{
			*to++ = *from++;
		}
		*to = '\0';	/* terminate the line */

		/*
		**	hand the line to the command module
		*/
		C_addcmd(buf);
		/*
		**	skip the newline
		*/
		if ('\n' == *from)
		{
			from++;
		}
	}
}

/*
**	this useful macro handle arguements that are adjacent
**	to a flag or in the following word e.g --
**
**		-a XXX 
**	and
**		-aXXX 
**
**	both work when SETPTR is used. 
*/
#define SETPTR	{if(strlen(argv[1]) == 2) {argv++;argc--;ptr=argv[1];}else ptr=(&argv[1][2]);}

static void
_Y_doargs(argc,argv,file1,file2,max_d)
int argc;
char *argv[];
char **file1,**file2;
int *max_d;
{
	char *ptr;

	/*
	**	mark maximum number of tokens as being unset
	*/
	*max_d = -1;

	while (argc > 1 && argv[1][0] == '-')
	{
		switch (argv[1][1])
		{
			case 't':
				_Y_flags |= U_TOKENS;
				break;
			case 'w':
				_Y_flags |= U_INCLUDE_WS;
				break;

			case 'b':
				_Y_flags |= U_BYTE_COMPARE;
				break;

			case 'c':
				_Y_flags |= U_NO_CASE;
				break;
			case 'd' :
				_Y_flags |= U_NEED_DECIMAL;
				break;
			case 'm' :
				_Y_flags |= U_INC_SIGN;
				break;
			case 'a':
				SETPTR;
				T_defatol(ptr);
				break;
			case 'r':
				SETPTR;
				T_defrtol(ptr);
				break;
			case 'i':
				T_defitol();
				break;
			case 'e' :
				_Y_eflag = 1;
				break;
			case 'v' :
				_Y_vflag = 1;
				break;
			case 'q' :
				Z_setquiet();
				break;
			case 's' :
				SETPTR;
				_Y_cmdlines(ptr);
				break;
			case 'f' :
			{
				extern FILE *fopen();
				char buf[Z_LINELEN];
				FILE *cmdfile;
				SETPTR;
				if ((FILE*) NULL ==
					(cmdfile = fopen(ptr,"r")))
				{
					Z_fatal("can't open command file\n");
				}
				while ((char*) NULL !=
					(char*) fgets(buf,Z_LINELEN,cmdfile))
				{
					C_addcmd(buf);
				}
				(void) fclose(cmdfile);
				break;
			}
			/*
			**	useful commands for
			**	 the C programming language
			*/
			case 'C' :
				C_addcmd("literal  \"   \"    \\ ");
				C_addcmd("comment  /*  */	 ");
				C_addcmd("literal  &&		 ");
				C_addcmd("literal  ||		 ");
				C_addcmd("literal  <=		 ");
				C_addcmd("literal  >=		 ");
				C_addcmd("literal  !=		 ");
				C_addcmd("literal  ==		 ");
				C_addcmd("literal  --		 ");
				C_addcmd("literal  ++		 ");
				C_addcmd("literal  <<		 ");
				C_addcmd("literal  >>		 ");
				C_addcmd("literal  ->		 ");
				C_addcmd("addalpha _		 ");
				C_addcmd("tol      a0 		 ");
				break;
			/*
			**	useful commands for
			**	 the Bourne shell programming language
			*/
			case 'S' :
				C_addcmd("literal  '    '    \\	");
				C_addcmd("comment  #    $	");
				C_addcmd("tol      a0 		");
				break;
			/*
			**	useful commands for
			**	 the Fortran programming language
			*/
			case 'F' :
				C_addcmd("literal  '	'     ' ");
				C_addcmd("comment  ^C   $	");
				C_addcmd("tol      a0 		");
				break;
			/*
			**	useful commands for
			**	 the Lisp programming language
			*/
			case 'L' :
				C_addcmd("literal  \" 	\"	");
				C_addcmd("comment  ; 	$	");
				C_addcmd("tol      a0 		");
				break;
			/*
			**	useful commands for
			**	 the Modula-2 programming language
			*/
			case 'M' :
				C_addcmd("literal ' 	'	");
				C_addcmd("literal \"	\"	");
				C_addcmd("comment (*	*)	");
				C_addcmd("literal :=		");
				C_addcmd("literal <>		");
				C_addcmd("literal <=		");
				C_addcmd("literal >=		");
				C_addcmd("tol      a0 		");
				break;
			case '0':
			case '1':
			case '2':
			case '3':
			case '4':
			case '5':
			case '6':
			case '7':
			case '8':
			case '9':
				*max_d = atoi(&argv[1][1]);
				break;
			default:
				Z_fatal("don't understand arguments\n");
		}
		++argv;
		--argc;
	}
	if (argc != 3)
		Z_fatal ("spiff requires two file names.\n");
	*file1 = argv[1];
	*file2 = argv[2];
}
