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

#ifdef MGR

#include "misc.h"
#include "line.h"
#include "token.h"
#include "/usr/public/pkg/mgr/include/term.h"
#include "/usr/public/pkg/mgr/include/restart.h"

#define OTHER		0
#define ON_DEBUG	1
#define OFF_DEBUG	2
#define DO_QUIT		3
#define DO_PAGE		4
#define NEW_PREC	5


#define NROW	60
#define NCOL	80

int isdiff[MAXTOKENS];	/* flag showing if a token pair was shown different*/

int comwin,wina, winb;	/* location to store window numbers */
int fontx,fonty;	/* size of the font in pixels */

int debug =0;


int firsttoken = 0;	/* index of first token pair being displayed */
int tokencnt;		/* count of the number of token pairs being displayed */

V_visual(flags)
int flags;
{

	int moretodo = 1;	/* flag to clear when we're finished */

	messup();

	m_selectwin(comwin);
	m_setmode(W_ACTIVATE);

	showpages(comroutine,flags);

	do
	{
		switch(getinput())
		{
			case ON_DEBUG:
				debug = 0;
				break;
			case OFF_DEBUG:
				debug = 1;
				break;
			case DO_QUIT:
				moretodo = 0;
				break;
			case DO_PAGE:
				if((firsttoken+tokencnt>= K_gettmax(0))||
				   (firsttoken+tokencnt>= K_gettmax(1)))
				{
					m_selectwin(comwin);
					m_printstr("\007this is the last page\n");
					break;
				}
				firsttoken += tokencnt;
				showpages(comroutine,flags);
				break;
			case NEW_PREC:
				updatepages(comroutine,flags);
				break;
			case OTHER:
				break;
			default :
				Z_fatal("bad value in main switch");
				
		}
	} while (moretodo);

	V_cleanup();
	return(0);
}

getinput()
{
	char ibuf[Z_LINELEN];	/* input buffer */
	char *ptr;

	m_selectwin(comwin);
	m_setmode(W_ACTIVATE);
	switch (m_getchar())
	{
		case 't':
			m_gets(ibuf);
			/*
			**	skip the 'tol'
			*/
			ptr = ibuf;
			S_nextword(&ptr);
			T_setdef(ptr);
			return(NEW_PREC);
		case 'q':
			return(DO_QUIT);
		case 'd':
			return(OFF_DEBUG);
		case 'D':
			return(ON_DEBUG);
		case 'm':
			return(DO_PAGE);
		default:
			return(OTHER);
	}
	
}

showpages(comroutine,flags)
int (*comroutine)();
int flags;
{
	int i;
		m_selectwin(wina);
		m_clear();
		m_selectwin(winb);
		m_clear();
		showlines();

		for(i=firsttoken;i<tokencnt+firsttoken; i++)
		{
			isdiff[i] = 0;
		}
		updatepages(comroutine,flags);
}

updatepages(comroutine,flags)
int (*comroutine)();
int flags;
{
	int i;

	for(i=firsttoken;i<tokencnt+firsttoken; i++)
	{
		if (isdiff[i])
		{

			if (0 == X_com(i,i,flags))
			{
				m_selectwin(wina);
				un_highlight(0,K_gettoken(0,i),K_getline(K_gettoken(0,firsttoken)));
				m_selectwin(winb);
				un_highlight(1,K_gettoken(1,i),K_getline(K_gettoken(1,firsttoken)));
				isdiff[i] = 0;
			}
		}
		else
		{
			if (0 != X_com(i,i,flags))
			{
				m_selectwin(wina);
				highlight(0,K_gettoken(0,i),K_getline(K_gettoken(0,firsttoken)));
				m_selectwin(winb);
				highlight(1,K_gettoken(1,i),K_getline(K_gettoken(1,firsttoken)));
				isdiff[i] = 1;
			}
		}
	}
}

un_highlight(file,ptr,firstline)
int file;
K_token ptr;
int firstline;
{
	highlight(file,ptr,firstline);
}

/*
**	argument expressed in terms of token lines
*/
highlight(file,ptr,firstline)
int file;
K_token ptr;
int firstline;
{
	int startx = K_getpos(ptr)*fontx;
	int starty = (L_tl2cl(file,K_getline(ptr))-L_tl2cl(file,firstline))*fonty;

	int sizex = fontx*strlen(K_gettext(ptr));
	int sizey = fonty;
	m_bitwrite(startx,starty,sizex,sizey);
}

showlines()
{
	int Alinecnt = 0;
	int Blinecnt = 0;

	int Atfirstline = K_getline(K_gettoken(0,firsttoken));
	int Btfirstline = K_getline(K_gettoken(1,firsttoken));
	int Afirstline =  L_tl2cl(0,K_getline(K_gettoken(0,firsttoken)));
	int Bfirstline =  L_tl2cl(1,K_getline(K_gettoken(1,firsttoken)));
	int Anexttoken = L_getindex(0,Atfirstline);
	int Bnexttoken = L_getindex(1,Btfirstline);
	int i;
	/*
	**	first print the lines on the screen
	*/
	for(i=0;i < NROW;i++)
	{
		if(Afirstline+i < L_getclmax(0))
		{
			m_selectwin(wina);
			showline(0,Afirstline+i,i);
			Alinecnt++;
		}

		if(Bfirstline+i < L_getclmax(1))
		{
			m_selectwin(winb);
			showline(1,Bfirstline+i,i);
			Blinecnt++;
		}
	}
	/*
	**	now figure out how many tokens we actually printed
	*/
	for(i=Atfirstline;Anexttoken<K_gettmax(0) && L_tl2cl(0,i) < Afirstline+Alinecnt;i++)
	{
			Anexttoken += L_getcount(0,i);
	}

	for(i=Btfirstline;Bnexttoken<K_gettmax(1) && L_tl2cl(1,i) < Bfirstline+Blinecnt;i++)
	{
			Bnexttoken += L_getcount(1,i);
	}
	tokencnt = MIN(Anexttoken,Bnexttoken) - firsttoken;

	/*
	**	draw a line through any tokens that come before the first
	**		token that is being compared
	*/
	if (L_getindex(0,Atfirstline) != firsttoken)
	{
		m_selectwin(wina);
		for(i=L_getindex(0,Atfirstline);i<firsttoken;i++)
		{
			drawline(K_gettoken(0,i),0);
		}
	}

	if (L_getindex(1,Btfirstline) != firsttoken)
	{
		m_selectwin(winb);

		for(i=L_getindex(1,Btfirstline);i<firsttoken;i++)
		{
			drawline(K_gettoken(1,i),0);
		}
/*
m_line(Bt[Bindex[Bfirstline]]->pos*fontx,fonty/2,(Bt[firsttoken]->pos*fontx)-2,fonty/2);
*/
	}

	if (Anexttoken > Bnexttoken)
	{
		m_selectwin(wina);
		for(i=Bnexttoken;i<Anexttoken;i++)
		{
			drawline(K_gettoken(0,i),L_tl2cl(0,K_getline(K_gettoken(0,i)))-Afirstline);
		}
	}

	if (Anexttoken < Bnexttoken)
	{
		m_selectwin(winb);
		for(i=Anexttoken;i<Bnexttoken;i++)
		{
			drawline(K_gettoken(1,i),L_tl2cl(1,K_getline(K_gettoken(1,i)))-Bfirstline);
		}
	}

}

/*
**	line is given in conten line
*/
drawline(ptr,line)
K_token ptr;
int line;
{
		m_line(K_getpos(ptr)*fontx,
			(line*fonty)+fonty/2,
			(K_getpos(ptr)+strlen(K_gettext(ptr)))*fontx,
			(line*fonty)+fonty/2);
}

/*
**	takes arguments in terms of content lines
*/
showline(file,index,row)
int file;
int index;
int row;
{
	static char tmp[Z_LINELEN];
	m_move(0,row);
	stripnl(tmp,L_getcline(file,index));
	m_printstr(tmp);
}

stripnl(to,from)
char *to,*from;
{
	while ((*from != '\n') && (*from != '\0'))
	{
		*to++ = *from++;
	}
	*to = '\0';
}

static int didscr = 0;

messup()
{
	int col, row;
	int dum1,dum2,dum3,border;

	m_setup(W_FLUSH|W_DEBUG);
	m_push(P_EVENT|P_FLAGS|P_POSITION);
	get_param(&dum1,&dum2,&dum3,&border);
	didscr = 1;
	comwin =  m_makewindow(192,50,732,116);
	wina = m_makewindow(0,218,570,670);
	m_selectwin(wina);
	m_font(2);
	get_font(&fontx,&fonty);
	m_shapewindow(0,218,NCOL*fontx+(2*border),NROW*fonty+(2*border));

	get_colrow(&col,&row);
	if ((col != NCOL) || (row != NROW))
	{
		Z_fatal("bad window size");
	}
	m_func(B_INVERT);
	m_setmode(W_ABS);

	winb = m_makewindow(580,218,570,670);
	m_selectwin(winb);
	m_font(2);
	get_font(&fontx,&fonty);
	m_shapewindow(580,218,NCOL*fontx+(2*border),NROW*fonty+(2*border));

	get_colrow(&col,&row);
	if ((col != NCOL) || (row != NROW))
	{
		Z_fatal("bad window size");
	}
	m_func(B_INVERT);
	m_setmode(W_ABS);

	m_selectwin(comwin);
	m_clear();
	m_setmode(W_ABS);
	m_setmode(W_ACTIVATE);
}

V_cleanup()
{
	if (didscr)
	{
		m_destroywin(wina);
		m_destroywin(winb);
		m_destroywin(comwin);
		m_popall();
		m_setecho();
		(void) fclose(m_termin);
		(void) fclose(m_termout);
	}
}

#else

#include "misc.h"
/*
**	dummy code for systems that don't have
**	the mgr window manager installed
*/
int
V_visual(d)
int d;
{
	Z_fatal("visual mode is not available on this machine\n");
	return(-d);	/* boiler plate */
}

void
V_cleanup()
{
}

#endif
