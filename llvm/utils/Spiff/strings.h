/*                        Copyright (c) 1988 Bellcore
**                            All Rights Reserved
**       Permission is granted to copy or use this program, EXCEPT that it
**       may not be sold for profit, the copyright notice must be reproduced
**       on copies, and credit should be given to Bellcore where it is due.
**       BELLCORE MAKES NO WARRANTY AND ACCEPTS NO LIABILITY FOR THIS PROGRAM.
*/

#ifndef S_INCLUDED
extern void S_wordcpy();
extern void S_skipword();
extern void S_skipspace();
extern void S_nextword();
extern int  S_wordcmp();
extern void S_trimzeros();
extern void S_savestr();
extern void S_savenstr();
extern void S_allocstr();
#define S_INCLUDED
#endif
