char rcsid_trim[] = "$Id$";

#include <stdio.h>
#include "b.h"
#include "fe.h"

Relation *allpairs;

int trimflag = 0;
int debugTrim = 0;

static void siblings ARGS((int, int));
static void findAllNexts ARGS((void));
static Relation *newAllPairs ARGS((void));

static void
siblings(i, j) int i; int j;
{
	int k;
	List pl;
	DeltaCost Max;
	int foundmax;

	allpairs[i][j].sibComputed = 1;

	if (i == 1) {
		return; /* never trim start symbol */
	}
	if (i==j) {
		return;
	}

	ZEROCOST(Max);
	foundmax = 0;

	for (k = 1; k < max_nonterminal; k++) {
		DeltaCost tmp;

		if (k==i || k==j) {
			continue;
		}
		if (!allpairs[k][i].rule) {
			continue;
		}
		if (!allpairs[k][j].rule) {
			return;
		}
		ASSIGNCOST(tmp, allpairs[k][j].chain);
		MINUSCOST(tmp, allpairs[k][i].chain);
		if (foundmax) {
			if (LESSCOST(Max, tmp)) {
				ASSIGNCOST(Max, tmp);
			}
		} else {
			foundmax = 1;
			ASSIGNCOST(Max, tmp);
		}
	}

	for (pl = rules; pl; pl = pl->next) {
		Rule p = (Rule) pl->x;
		Operator op = p->pat->op;
		List oprule;
		DeltaCost Min;
		int foundmin;
		
		if (!op) {
			continue;
		}
		switch (op->arity) {
		case 0:
			continue;
		case 1:
			if (!allpairs[p->pat->children[0]->num ][ i].rule) {
				continue;
			}
			foundmin = 0;
			for (oprule = op->table->rules; oprule; oprule = oprule->next) {
				Rule s = (Rule) oprule->x;
				DeltaPtr Cx;
				DeltaPtr Csj;
				DeltaPtr Cpi;
				DeltaCost tmp;

				if (!allpairs[p->lhs->num ][ s->lhs->num].rule
				 || !allpairs[s->pat->children[0]->num ][ j].rule) {
					continue;
				}
				Cx = allpairs[p->lhs->num ][ s->lhs->num].chain;
				Csj= allpairs[s->pat->children[0]->num ][ j].chain;
				Cpi= allpairs[p->pat->children[0]->num ][ i].chain;
				ASSIGNCOST(tmp, Cx);
				ADDCOST(tmp, s->delta);
				ADDCOST(tmp, Csj);
				MINUSCOST(tmp, Cpi);
				MINUSCOST(tmp, p->delta);
				if (foundmin) {
					if (LESSCOST(tmp, Min)) {
						ASSIGNCOST(Min, tmp);
					}
				} else {
					foundmin = 1;
					ASSIGNCOST(Min, tmp);
				}
			}
			if (!foundmin) {
				return;
			}
			if (foundmax) {
				if (LESSCOST(Max, Min)) {
					ASSIGNCOST(Max, Min);
				}
			} else {
				foundmax = 1;
				ASSIGNCOST(Max, Min);
			}
			break;
		case 2:
		/* do first dimension */
		if (allpairs[p->pat->children[0]->num ][ i].rule) {
			foundmin = 0;
			for (oprule = op->table->rules; oprule; oprule = oprule->next) {
				Rule s = (Rule) oprule->x;
				DeltaPtr Cx;
				DeltaPtr Cb;
				DeltaPtr Csj;
				DeltaPtr Cpi;
				DeltaCost tmp;

				if (allpairs[p->lhs->num ][ s->lhs->num].rule
				 && allpairs[s->pat->children[0]->num ][ j].rule
				 && allpairs[s->pat->children[1]->num ][ p->pat->children[1]->num].rule) {
					Cx = allpairs[p->lhs->num ][ s->lhs->num].chain;
					Csj= allpairs[s->pat->children[0]->num ][ j].chain;
					Cpi= allpairs[p->pat->children[0]->num ][ i].chain;
					Cb = allpairs[s->pat->children[1]->num ][ p->pat->children[1]->num].chain;
					ASSIGNCOST(tmp, Cx);
					ADDCOST(tmp, s->delta);
					ADDCOST(tmp, Csj);
					ADDCOST(tmp, Cb);
					MINUSCOST(tmp, Cpi);
					MINUSCOST(tmp, p->delta);
					if (foundmin) {
						if (LESSCOST(tmp, Min)) {
							ASSIGNCOST(Min, tmp);
						}
					} else {
						foundmin = 1;
						ASSIGNCOST(Min, tmp);
					}
				}
			}
			if (!foundmin) {
				return;
			}
			if (foundmax) {
				if (LESSCOST(Max, Min)) {
					ASSIGNCOST(Max, Min);
				}
			} else {
				foundmax = 1;
				ASSIGNCOST(Max, Min);
			}
		}
		/* do second dimension */
		if (allpairs[p->pat->children[1]->num ][ i].rule) {
			foundmin = 0;
			for (oprule = op->table->rules; oprule; oprule = oprule->next) {
				Rule s = (Rule) oprule->x;
				DeltaPtr Cx;
				DeltaPtr Cb;
				DeltaPtr Csj;
				DeltaPtr Cpi;
				DeltaCost tmp;

				if (allpairs[p->lhs->num ][ s->lhs->num].rule
				 && allpairs[s->pat->children[1]->num ][ j].rule
				 && allpairs[s->pat->children[0]->num ][ p->pat->children[0]->num].rule) {
					Cx = allpairs[p->lhs->num ][ s->lhs->num].chain;
					Csj= allpairs[s->pat->children[1]->num ][ j].chain;
					Cpi= allpairs[p->pat->children[1]->num ][ i].chain;
					Cb = allpairs[s->pat->children[0]->num ][ p->pat->children[0]->num].chain;
					ASSIGNCOST(tmp, Cx);
					ADDCOST(tmp, s->delta);
					ADDCOST(tmp, Csj);
					ADDCOST(tmp, Cb);
					MINUSCOST(tmp, Cpi);
					MINUSCOST(tmp, p->delta);
					if (foundmin) {
						if (LESSCOST(tmp, Min)) {
							ASSIGNCOST(Min, tmp);
						}
					} else {
						foundmin = 1;
						ASSIGNCOST(Min, tmp);
					}
				}
			}
			if (!foundmin) {
				return;
			}
			if (foundmax) {
				if (LESSCOST(Max, Min)) {
					ASSIGNCOST(Max, Min);
				}
			} else {
				foundmax = 1;
				ASSIGNCOST(Max, Min);
			}
		}
		break;
		default:
			assert(0);
		}
	}
	allpairs[i ][ j].sibFlag = foundmax;
	ASSIGNCOST(allpairs[i ][ j].sibling, Max);
}

static void
findAllNexts()
{
	int i,j;
	int last;

	for (i = 1; i < max_nonterminal; i++) {
		last = 0;
		for (j = 1; j < max_nonterminal; j++) {
			if (allpairs[i ][j].rule) {
				allpairs[i ][ last].nextchain = j;
				last = j;
			}
		}
	}
	/*
	for (i = 1; i < max_nonterminal; i++) {
		last = 0;
		for (j = 1; j < max_nonterminal; j++) {
			if (allpairs[i ][j].sibFlag) {
				allpairs[i ][ last].nextsibling = j;
				last = j;
			}
		}
	}
	*/
}

static Relation *
newAllPairs()
{
	int i;
	Relation *rv;

	rv = (Relation*) zalloc(max_nonterminal * sizeof(Relation));
	for (i = 0; i < max_nonterminal; i++) {
		rv[i] = (Relation) zalloc(max_nonterminal * sizeof(struct relation));
	}
	return rv;
}

void
findAllPairs()
{
	List pl;
	int changes;
	int j;

	allpairs = newAllPairs();
	for (pl = chainrules; pl; pl = pl->next) {
		Rule p = (Rule) pl->x;
		NonTerminalNum rhs = p->pat->children[0]->num;
		NonTerminalNum lhs = p->lhs->num;
		Relation r = &allpairs[lhs ][ rhs];

		if (LESSCOST(p->delta, r->chain)) {
			ASSIGNCOST(r->chain, p->delta);
			r->rule = p;
		}
	}
	for (j = 1; j < max_nonterminal; j++) {
		Relation r = &allpairs[j ][ j];
		ZEROCOST(r->chain);
		r->rule = &stub_rule;
	}
	changes = 1;
	while (changes) {
		changes = 0;
		for (pl = chainrules; pl; pl = pl->next) {
			Rule p = (Rule) pl->x;
			NonTerminalNum rhs = p->pat->children[0]->num;
			NonTerminalNum lhs = p->lhs->num;
			int i;

			for (i = 1; i < max_nonterminal; i++) {
				Relation r = &allpairs[rhs ][ i];
				Relation s = &allpairs[lhs ][ i];
				DeltaCost dc;
				if (!r->rule) {
					continue;
				}
				ASSIGNCOST(dc, p->delta);
				ADDCOST(dc, r->chain);
				if (!s->rule || LESSCOST(dc, s->chain)) {
					s->rule = p;
					ASSIGNCOST(s->chain, dc);
					changes = 1;
				}
			}
		}
	}
	findAllNexts();
}

void
trim(t) Item_Set t;
{
	int m,n;
	static short *vec = 0;
	int last;

	assert(!t->closed);
	debug(debugTrim, printf("Begin Trim\n"));
	debug(debugTrim, dumpItem_Set(t));

	last = 0;
	if (!vec) {
		vec = (short*) zalloc(max_nonterminal * sizeof(*vec));
	}
	for (m = 1; m < max_nonterminal; m++) {
		if (t->virgin[m].rule) {
			vec[last++] = m;
		}
	}
	for (m = 0; m < last; m++) {
		DeltaCost tmp;
		int j;
		int i;

		i = vec[m];

		for (j = allpairs[i ][ 0].nextchain; j; j = allpairs[i ][ j].nextchain) {

			if (i == j) {
				continue;
			}
			if (!t->virgin[j].rule) {
				continue;
			}
			ASSIGNCOST(tmp, t->virgin[j].delta);
			ADDCOST(tmp, allpairs[i ][ j].chain);
			if (!LESSCOST(t->virgin[i].delta, tmp)) {
				t->virgin[i].rule = 0;
				ZEROCOST(t->virgin[i].delta);
				debug(debugTrim, printf("Trimmed Chain (%d,%d)\n", i,j));
				goto outer;
			}
			
		}
		if (!trimflag) {
			continue;
		}
		for (n = 0; n < last; n++) {
			j = vec[n];
			if (i == j) {
				continue;
			}

			if (!t->virgin[j].rule) {
				continue;
			}

			if (!allpairs[i][j].sibComputed) {
				siblings(i,j);
			}
			if (!allpairs[i][j].sibFlag) {
				continue;
			}
			ASSIGNCOST(tmp, t->virgin[j].delta);
			ADDCOST(tmp, allpairs[i ][ j].sibling);
			if (!LESSCOST(t->virgin[i].delta, tmp)) {
				t->virgin[i].rule = 0;
				ZEROCOST(t->virgin[i].delta);
				goto outer;
			}
		}

		outer: ;
	}

	debug(debugTrim, dumpItem_Set(t));
	debug(debugTrim, printf("End Trim\n"));
}

void
dumpRelation(r) Relation r;
{
	printf("{ %d %ld %d %ld }", r->rule->erulenum, (long) r->chain, r->sibFlag, (long) r->sibling);
}

void
dumpAllPairs()
{
	int i,j;

	printf("Dumping AllPairs\n");
	for (i = 1; i < max_nonterminal; i++) {
		for (j = 1; j < max_nonterminal; j++) {
			dumpRelation(&allpairs[i ][j]);
		}
		printf("\n");
	}
}
