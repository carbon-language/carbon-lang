char rcsid_nonterminal[] = "$Id$";

#include "b.h"
#include <stdio.h>

NonTerminal	start;
NonTerminalNum	max_nonterminal = 1;
NonTerminalNum	last_user_nonterminal;
List		nonterminals;

NonTerminal
newNonTerminal(name) char *name;
{
	NonTerminal nt;

	nt = (NonTerminal) zalloc(sizeof(struct nonterminal));
	assert(nt);
	if (max_nonterminal == 1) {
		start = nt;
	}
	nt->name = name;
	nt->num = max_nonterminal++;
	nonterminals = newList(nt, nonterminals);

	return nt;
}

int
nonTerminalName(buf, i) char *buf; int i;
{
	List l;
	extern char *strcpy ARGS((char *, char *));

	for (l = nonterminals; l; l = l->next) {
		NonTerminal nt = (NonTerminal) l->x;
		if (nt->num == i) {
			strcpy(buf, nt->name);
			return 1;
		}
	}
	strcpy(buf, "(Unknown NonTerminal)");
	return 0;
}

void
dumpNonTerminal(n) NonTerminal n;
{
	printf("%s(%d)", n->name, n->num);
}
