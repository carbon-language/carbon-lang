char rcsid_closure[] = "$Id$";

#include <stdio.h>
#include "b.h"

int prevent_divergence = 0;

List chainrules;

void
findChainRules()
{
	List pl;

	assert(!chainrules);

	for (pl = rules; pl; pl = pl->next) {
		Rule p = (Rule) pl->x;
		if (!p->pat->op) {
			chainrules = newList(p, chainrules);
		} else {
			p->pat->op->table->rules = newList(p, p->pat->op->table->rules);
			addRelevant(p->pat->op->table->relevant, p->lhs->num);
		}
	}
}

void
zero(t) Item_Set t;
{
	int i;
	DeltaCost base;
	int exists;
	int base_nt;

	assert(!t->closed);

	ZEROCOST(base);
	exists = 0;
	for (i = 0; i < max_nonterminal; i++) {
		if (t->virgin[i].rule) {
			if (exists) {
				if (LESSCOST(t->virgin[i].delta, base)) {
					ASSIGNCOST(base, t->virgin[i].delta);
					base_nt = i;
				}
			} else {
				ASSIGNCOST(base, t->virgin[i].delta);
				exists = 1;
				base_nt = i;
			}
		}
	}
	if (!exists) {
		return;
	}
	for (i = 0; i < max_nonterminal; i++) {
		if (t->virgin[i].rule) {
			MINUSCOST(t->virgin[i].delta, base);
		}
		NODIVERGE(t->virgin[i].delta, t, i, base_nt);
	}
}

void
closure(t) Item_Set t;
{
	int changes;
	List pl;

	assert(!t->closed);
	t->closed = itemArrayCopy(t->virgin);

	changes = 1;
	while (changes) {
		changes = 0;
		for (pl = chainrules; pl; pl = pl->next) {
			Rule p = (Rule) pl->x;
			register Item *rhs_item = &t->closed[p->pat->children[0]->num];

			if (rhs_item->rule) {	/* rhs is active */
				DeltaCost dc;
				register Item *lhs_item = &t->closed[p->lhs->num];

				ASSIGNCOST(dc, rhs_item->delta);
				ADDCOST(dc, p->delta);
				if (LESSCOST(dc, lhs_item->delta) || !lhs_item->rule) {
					ASSIGNCOST(lhs_item->delta, dc);
					lhs_item->rule = p;
					changes = 1;
				}
			}
		}
	}
}
