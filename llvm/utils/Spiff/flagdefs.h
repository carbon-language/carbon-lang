/*                        Copyright (c) 1988 Bellcore
**                            All Rights Reserved
**       Permission is granted to copy or use this program, EXCEPT that it
**       may not be sold for profit, the copyright notice must be reproduced
**       on copies, and credit should be given to Bellcore where it is due.
**       BELLCORE MAKES NO WARRANTY AND ACCEPTS NO LIABILITY FOR THIS PROGRAM.
*/

/*
**	flags used by both parser and comparison routines
*/
#define U_INCLUDE_WS	001

/*
**	flags used only by the comparison routines
*/
#define U_BYTE_COMPARE		002
#define U_NO_CASE		004

/*
**	flag used by the output routine
*/
#define U_TOKENS		010

/*
**	flags used only by the parser
*/
#define U_INC_SIGN	020
#define U_NEED_DECIMAL	040
