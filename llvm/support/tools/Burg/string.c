char rcsid_string[] = "$Id$";

#include <stdio.h>
#include <string.h>
#include "b.h"
#include "fe.h"

static StrTableElement newStrTableElement ARGS((void));

StrTable
newStrTable()
{
	return (StrTable) zalloc(sizeof(struct strTable));
}

static StrTableElement
newStrTableElement()
{
	return (StrTableElement) zalloc(sizeof(struct strTableElement));
}

void
dumpStrTable(t) StrTable t;
{ 
	List e;
	IntList r;

	printf("Begin StrTable\n");
	for (e = t->elems; e; e = e->next) {
		StrTableElement el = (StrTableElement) e->x;
		printf("%s: ", el->str);
		for (r = el->erulenos; r; r = r->next) {
			int i = r->x;
			printf("(%d)", i);
		}
		printf("\n");
	}
	printf("End StrTable\n");
}

StrTableElement
addString(t, s, eruleno, new) StrTable t; char *s; int eruleno; int *new;
{
	List l;
	StrTableElement ste;

	assert(t);
	for (l = t->elems; l; l = l->next) {
		StrTableElement e = (StrTableElement) l->x;

		assert(e);
		if (!strcmp(s, e->str)) {
			e->erulenos = newIntList(eruleno, e->erulenos);
			*new = 0;
			return e;
		}
	}
	ste = newStrTableElement();
	ste->erulenos = newIntList(eruleno, 0);
	ste->str = (char *) zalloc(strlen(s) + 1);
	strcpy(ste->str, s);
	t->elems = newList(ste, t->elems);
	*new = 1;
	return ste;
}
