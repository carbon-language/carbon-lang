/*                        Copyright (c) 1988 Bellcore
**                            All Rights Reserved
**       Permission is granted to copy or use this program, EXCEPT that it
**       may not be sold for profit, the copyright notice must be reproduced
**       on copies, and credit should be given to Bellcore where it is due.
**       BELLCORE MAKES NO WARRANTY AND ACCEPTS NO LIABILITY FOR THIS PROGRAM.
*/


#ifndef L_INCLUDED

#define _L_MAXLINES	300000

/*
**	oh god, is this an ugly implementation.
**	I really should have a two dimensional array of structures
**		the history of the current arrangement is too long
**		and ugly to record here.
**		Someday when I have too much time on my hands . . .
*/

extern char *_L_al[];	/* storage for text in first file */
extern char *_L_bl[];	/* storage for text in second file */

extern int _L_ai[];	/* pointer from token line to first token */
extern int _L_bi[];

extern int _L_ac[];	/* number of tokens on a given token line */
extern int _L_bc[];

extern int _L_aclindex[]; /* mapping from content lines to real lines */
extern int _L_bclindex[];

extern int _L_atlindex[]; /* mapping from lines with tokens to content lines */
extern int _L_btlindex[];

extern int _L_arlm;	/* count of real lines */
extern int _L_brlm;

extern int _L_aclm;	/* count of content lines */
extern int _L_bclm;

extern int _L_atlm;	/* count of lines with tokens */
extern int _L_btlm;

/*
**	routines to set up mappings from token lines to content lines
**	  and from content lines to real lines
*/
#define L_setclindex(file,content,real) (file?(_L_bclindex[content]=real):\
					     (_L_aclindex[content]=real))

#define L_settlindex(file,token,content) (file?(_L_btlindex[token]=content):\
					      (_L_atlindex[token]=content))
/*
**	get line number X from file
*/
#define L_getrline(file, X)	 (file?(_L_bl[X]):(_L_al[X]))
#define L_getcline(file, X)	 (file?(_L_bl[_L_bclindex[X]]):\
				       (_L_al[_L_aclindex[X]]))
#define L_gettline(file, X)	 (file?(_L_bl[_L_bclindex[_L_btlindex[X]]]):\
				       (_L_al[_L_aclindex[_L_atlindex[X]]]))

#define L_cl2rl(file, X)	 (file?(_L_bclindex[X]):\
				       (_L_aclindex[X]))
#define L_tl2cl(file, X)	 (file?(_L_btlindex[X]):\
				       (_L_atlindex[X]))
#define L_tl2rl(file, X)	 (file?(_L_bclindex[_L_btlindex[X]]):\
				       (_L_aclindex[_L_atlindex[X]]))

/*
**	get number of first token on line X of the file
*/
#define L_getindex(file,X)	(file?(_L_bi[X]):(_L_ai[X]))

/*
**	get count of number of tokens on line X of first file
*/
#define L_getcount(file,X)	(file?(_L_bc[X]):(_L_ac[X]))

/*
**	save number of first token for line X of file
*/
#define L_setindex(file,index,value)	(file?(_L_bi[index]=value):(_L_ai[index]=value))
/*
**	save count of tokens on line X of file
*/
#define L_setcount(file,index,value)	(file?(_L_bc[index]=value):(_L_ac[index]=value))
#define	L_inccount(file,index)		(file?(_L_bc[index]++):(_L_ac[index]++))

/*
**	retrieve line and token counts
*/
#define L_getrlmax(file)	(file?_L_brlm:_L_arlm)
#define L_getclmax(file)	(file?_L_bclm:_L_aclm)
#define L_gettlmax(file)	(file?_L_btlm:_L_atlm)

/*
**	set line and token counts
*/
#define _L_setrlmx(file,value)	(file?(_L_brlm=(value)):(_L_arlm=(value)))
#define L_setclmax(file,value)	(file?(_L_bclm=(value)):(_L_aclm=(value)))
#define L_settlmax(file,value)	(file?(_L_btlm=(value)):(_L_atlm=(value)))

/*
**	increment line and token counts
*/
#define	_L_incrlmx(file)		(file?(_L_brlm++):(_L_arlm++))
#define	L_incclmax(file)		(file?(_L_bclm++):(_L_aclm++))
#define	L_inctlmax(file)		(file?(_L_btlm++):(_L_atlm++))

#define L_INCLUDED
#endif
