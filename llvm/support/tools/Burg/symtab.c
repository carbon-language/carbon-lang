char rcsid_symtab[] = "$Id$";

#include <stdio.h>
#include <string.h>
#include "b.h"
#include "fe.h"

static List symtab;

Symbol
newSymbol(name) char *name;
{
	Symbol s;

	s = (Symbol) zalloc(sizeof(struct symbol));
	assert(s);
	s->name = name;
	return s;
}

Symbol
enter(name, new) char *name; int *new;
{
	List l;
	Symbol s;

	*new = 0;
	for (l = symtab; l; l = l->next) {
		s = (Symbol) l->x;
		if (!strcmp(name, s->name)) {
			return s;
		}
	}
	*new = 1;
	s = newSymbol(name);
	symtab = newList(s, symtab);
	return s;
}
