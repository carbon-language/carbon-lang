char rcsid_table[] = "$Id$";

#include "b.h"
#include <string.h>
#include <stdio.h>

static void growIndex_Map ARGS((Index_Map *));
static Relevant newRelevant ARGS((void));
static Dimension newDimension ARGS((Operator, int));
static void GT_1 ARGS((Table));
static void GT_2_0 ARGS((Table));
static void GT_2_1 ARGS((Table));
static void growTransition ARGS((Table, int));
static Item_Set restrict ARGS((Dimension, Item_Set));
static void addHP_1 ARGS((Table, Item_Set));
static void addHP_2_0 ARGS((Table, Item_Set));
static void addHP_2_1 ARGS((Table, Item_Set));
static void addHyperPlane ARGS((Table, int, Item_Set));

static void
growIndex_Map(r) Index_Map *r;
{
	Index_Map new;

	new.max_size = r->max_size + STATES_INCR;
	new.class = (Item_Set*) zalloc(new.max_size * sizeof(Item_Set));
	assert(new.class);
	memcpy(new.class, r->class, r->max_size * sizeof(Item_Set));
	zfree(r->class);
	*r = new;
}

static Relevant
newRelevant()
{
	Relevant r = (Relevant) zalloc(max_nonterminal * sizeof(*r));
	return r;
}

void
addRelevant(r, nt) Relevant r; NonTerminalNum nt;
{
	int i;

	for (i = 0; r[i]; i++) {
		if (r[i] == nt) {
			break;
		}
	}
	if (!r[i]) {
		r[i] = nt;
	}
}

static Dimension
newDimension(op, index) Operator op; ArityNum index;
{
	Dimension d;
	List pl;
	Relevant r;

	assert(op);
	assert(index >= 0 && index < op->arity);
	d = (Dimension) zalloc(sizeof(struct dimension));
	assert(d);

	r = d->relevant = newRelevant();
	for (pl = rules; pl; pl = pl->next) {
		Rule pr = (Rule) pl->x;
		if (pr->pat->op == op) {
			addRelevant(r, pr->pat->children[index]->num);
		}
	}

	d->index_map.max_size = STATES_INCR;
	d->index_map.class = (Item_Set*) 
			zalloc(d->index_map.max_size * sizeof(Item_Set));
	d->map = newMapping(DIM_MAP_SIZE);
	d->max_size = TABLE_INCR;

	return d;
}

Table
newTable(op) Operator op;
{
	Table t;
	int i, size;

	assert(op);

	t = (Table) zalloc(sizeof(struct table));
	assert(t);

	t->op = op;

	for (i = 0; i < op->arity; i++) {
		t->dimen[i] = newDimension(op, i);
	}

	size = 1;
	for (i = 0; i < op->arity; i++) {
		size *= t->dimen[i]->max_size;
	}
	t->transition = (Item_Set*) zalloc(size * sizeof(Item_Set));
	t->relevant = newRelevant();
	assert(t->transition);

	return t;
}

static void
GT_1(t) Table t;
{
	Item_Set	*ts;
	ItemSetNum 	oldsize = t->dimen[0]->max_size;
	ItemSetNum 	newsize = t->dimen[0]->max_size + TABLE_INCR;

	t->dimen[0]->max_size = newsize;

	ts = (Item_Set*) zalloc(newsize * sizeof(Item_Set));
	assert(ts);
	memcpy(ts, t->transition, oldsize * sizeof(Item_Set));
	zfree(t->transition);
	t->transition = ts;
}

static void
GT_2_0(t) Table t;
{
	Item_Set	*ts;
	ItemSetNum 	oldsize = t->dimen[0]->max_size;
	ItemSetNum 	newsize = t->dimen[0]->max_size + TABLE_INCR;
	int		size;

	t->dimen[0]->max_size = newsize;

	size = newsize * t->dimen[1]->max_size;

	ts = (Item_Set*) zalloc(size * sizeof(Item_Set));
	assert(ts);
	memcpy(ts, t->transition, oldsize*t->dimen[1]->max_size * sizeof(Item_Set));
	zfree(t->transition);
	t->transition = ts;
}

static void
GT_2_1(t) Table t;
{
	Item_Set	*ts;
	ItemSetNum 	oldsize = t->dimen[1]->max_size;
	ItemSetNum 	newsize = t->dimen[1]->max_size + TABLE_INCR;
	int		size;
	Item_Set	*from;
	Item_Set	*to;
	int 		i1, i2;

	t->dimen[1]->max_size = newsize;

	size = newsize * t->dimen[0]->max_size;

	ts = (Item_Set*) zalloc(size * sizeof(Item_Set));
	assert(ts);

	from = t->transition;
	to = ts;
	for (i1 = 0; i1 < t->dimen[0]->max_size; i1++) {
		for (i2 = 0; i2 < oldsize; i2++) {
			to[i2] = from[i2];
		}
		to += newsize;
		from += oldsize;
	}
	zfree(t->transition);
	t->transition = ts;
}

static void
growTransition(t, dim) Table t; ArityNum dim;
{

	assert(t);
	assert(t->op);
	assert(dim < t->op->arity);

	switch (t->op->arity) {
	default:
		assert(0);
		break;
	case 1:
		GT_1(t);
		return;
	case 2:
		switch (dim) {
		default:
			assert(0);
			break;
		case 0:
			GT_2_0(t);
			return;
		case 1:
			GT_2_1(t);
			return;
		}
	}
}

static Item_Set
restrict(d, ts) Dimension d; Item_Set ts;
{
	DeltaCost	base;
	Item_Set	r;
	int found;
	register Relevant r_ptr = d->relevant;
	register Item *ts_current = ts->closed;
	register Item *r_current;
	register int i;
	register int nt;

	ZEROCOST(base);
	found = 0;
	r = newItem_Set(d->relevant);
	r_current = r->virgin;
	for (i = 0; (nt = r_ptr[i]) != 0; i++) {
		if (ts_current[nt].rule) {
			r_current[nt].rule = &stub_rule;
			if (!found) {
				found = 1;
				ASSIGNCOST(base, ts_current[nt].delta);
			} else {
				if (LESSCOST(ts_current[nt].delta, base)) {
					ASSIGNCOST(base, ts_current[nt].delta);
				}
			}
		}
	}

	/* zero align */
	for (i = 0; (nt = r_ptr[i]) != 0; i++) {
		if (r_current[nt].rule) {
			ASSIGNCOST(r_current[nt].delta, ts_current[nt].delta);
			MINUSCOST(r_current[nt].delta, base);
		}
	}
	assert(!r->closed);
	r->representative = ts;
	return r;
}

static void
addHP_1(t, ts) Table t; Item_Set ts;
{
	List pl;
	Item_Set e;
	Item_Set tmp;
	int new;

	e = newItem_Set(t->relevant);
	assert(e);
	e->kids[0] = ts->representative;
	for (pl = t->rules; pl; pl = pl->next) {
		Rule p = (Rule) pl->x;
		if (t->op == p->pat->op && ts->virgin[p->pat->children[0]->num].rule) {
			DeltaCost dc;
			ASSIGNCOST(dc, ts->virgin[p->pat->children[0]->num].delta);
			ADDCOST(dc, p->delta);
			if (!e->virgin[p->lhs->num].rule || LESSCOST(dc, e->virgin[p->lhs->num].delta)) {
				e->virgin[p->lhs->num].rule = p;
				ASSIGNCOST(e->virgin[p->lhs->num].delta, dc);
				e->op = t->op;
			}
		}
	}
	trim(e);
	zero(e);
	tmp = encode(globalMap, e, &new);
	assert(ts->num < t->dimen[0]->map->max_size);
	t->transition[ts->num] = tmp;
	if (new) {
		closure(e);
		addQ(globalQ, tmp);
	} else {
		freeItem_Set(e);
	}
}

static void
addHP_2_0(t, ts) Table t; Item_Set ts;
{
	List pl;
	register Item_Set e;
	Item_Set tmp;
	int new;
	int i2;

	assert(t->dimen[1]->map->count <= t->dimen[1]->map->max_size);
	for (i2 = 0; i2 < t->dimen[1]->map->count; i2++) {
		e = newItem_Set(t->relevant);
		assert(e);
		e->kids[0] = ts->representative;
		e->kids[1] = t->dimen[1]->map->set[i2]->representative;
		for (pl = t->rules; pl; pl = pl->next) {
			register Rule p = (Rule) pl->x;

			if (t->op == p->pat->op 
					&& ts->virgin[p->pat->children[0]->num].rule
					&& t->dimen[1]->map->set[i2]->virgin[p->pat->children[1]->num].rule){
				DeltaCost dc;
				ASSIGNCOST(dc, p->delta);
				ADDCOST(dc, ts->virgin[p->pat->children[0]->num].delta);
				ADDCOST(dc, t->dimen[1]->map->set[i2]->virgin[p->pat->children[1]->num].delta);

				if (!e->virgin[p->lhs->num].rule || LESSCOST(dc, e->virgin[p->lhs->num].delta)) {
					e->virgin[p->lhs->num].rule = p;
					ASSIGNCOST(e->virgin[p->lhs->num].delta, dc);
					e->op = t->op;
				}
			}
		}
		trim(e);
		zero(e);
		tmp = encode(globalMap, e, &new);
		assert(ts->num < t->dimen[0]->map->max_size);
		t->transition[ts->num * t->dimen[1]->max_size + i2] = tmp;
		if (new) {
			closure(e);
			addQ(globalQ, tmp);
		} else {
			freeItem_Set(e);
		}
	}
}

static void
addHP_2_1(t, ts) Table t; Item_Set ts;
{
	List pl;
	register Item_Set e;
	Item_Set tmp;
	int new;
	int i1;

	assert(t->dimen[0]->map->count <= t->dimen[0]->map->max_size);
	for (i1 = 0; i1 < t->dimen[0]->map->count; i1++) {
		e = newItem_Set(t->relevant);
		assert(e);
		e->kids[0] = t->dimen[0]->map->set[i1]->representative;
		e->kids[1] = ts->representative;
		for (pl = t->rules; pl; pl = pl->next) {
			register Rule p = (Rule) pl->x;

			if (t->op == p->pat->op 
					&& ts->virgin[p->pat->children[1]->num].rule
					&& t->dimen[0]->map->set[i1]->virgin[p->pat->children[0]->num].rule){
				DeltaCost dc;
				ASSIGNCOST(dc, p->delta );
				ADDCOST(dc, ts->virgin[p->pat->children[1]->num].delta);
				ADDCOST(dc, t->dimen[0]->map->set[i1]->virgin[p->pat->children[0]->num].delta);
				if (!e->virgin[p->lhs->num].rule || LESSCOST(dc, e->virgin[p->lhs->num].delta)) {
					e->virgin[p->lhs->num].rule = p;
					ASSIGNCOST(e->virgin[p->lhs->num].delta, dc);
					e->op = t->op;
				}
			}
		}
		trim(e);
		zero(e);
		tmp = encode(globalMap, e, &new);
		assert(ts->num < t->dimen[1]->map->max_size);
		t->transition[i1 * t->dimen[1]->max_size + ts->num] = tmp;
		if (new) {
			closure(e);
			addQ(globalQ, tmp);
		} else {
			freeItem_Set(e);
		}
	}
}

static void
addHyperPlane(t, i, ts) Table t; ArityNum i; Item_Set ts;
{
	switch (t->op->arity) {
	default:
		assert(0);
		break;
	case 1:
		addHP_1(t, ts);
		return;
	case 2:
		switch (i) {
		default:
			assert(0);
			break;
		case 0:
			addHP_2_0(t, ts);
			return;
		case 1:
			addHP_2_1(t, ts);
			return;
		}
	}
}

void
addToTable(t, ts) Table t; Item_Set ts;
{
	ArityNum i;

	assert(t);
	assert(ts);
	assert(t->op);

	for (i = 0; i < t->op->arity; i++) {
		Item_Set r;
		Item_Set tmp;
		int new;

		r = restrict(t->dimen[i], ts);
		tmp = encode(t->dimen[i]->map, r, &new);
		if (t->dimen[i]->index_map.max_size <= ts->num) {
			growIndex_Map(&t->dimen[i]->index_map);
		}
		assert(ts->num < t->dimen[i]->index_map.max_size);
		t->dimen[i]->index_map.class[ts->num] = tmp;
		if (new) {
			if (t->dimen[i]->max_size <= r->num) {
				growTransition(t, i);
			}
			addHyperPlane(t, i, r);
		} else {
			freeItem_Set(r);
		}
	}
}

Item_Set *
transLval(t, row, col) Table t; int row; int col;
{
	switch (t->op->arity) {
	case 0:
		assert(row == 0);
		assert(col == 0);
		return t->transition;
	case 1:
		assert(col == 0);
		return t->transition + row;
	case 2:
		return t->transition + row * t->dimen[1]->max_size + col;
	default:
		assert(0);
	}
	return 0;
}

void
dumpRelevant(r) Relevant r;
{
	for (; *r; r++) {
		printf("%4d", *r);
	}
}

void
dumpIndex_Map(r) Index_Map *r;
{
	int i;

	printf("BEGIN Index_Map: MaxSize (%d)\n", r->max_size);
	for (i = 0; i < globalMap->count; i++) {
		printf("\t#%d: -> %d\n", i, r->class[i]->num);
	}
	printf("END Index_Map:\n");
}

void
dumpDimension(d) Dimension d;
{
	printf("BEGIN Dimension:\n");
	printf("Relevant: ");
	dumpRelevant(d->relevant);
	printf("\n");
	dumpIndex_Map(&d->index_map);
	dumpMapping(d->map);
	printf("MaxSize of dimension = %d\n", d->max_size);
	printf("END Dimension\n");
}

void
dumpTable(t, full) Table t; int full;
{
	int i;

	if (!t) {
		printf("NO Table yet.\n");
		return;
	}
	printf("BEGIN Table:\n");
	if (full) {
		dumpOperator(t->op, 0);
	}
	for (i = 0; i < t->op->arity; i++) {
		printf("BEGIN dimension(%d)\n", i);
		dumpDimension(t->dimen[i]);
		printf("END dimension(%d)\n", i);
	}
	dumpTransition(t);
	printf("END Table:\n");
}

void
dumpTransition(t) Table t;
{
	int i,j;

	switch (t->op->arity) {
	case 0:
		printf("{ %d }", t->transition[0]->num);
		break;
	case 1:
		printf("{");
		for (i = 0; i < t->dimen[0]->map->count; i++) {
			if (i > 0) {
				printf(",");
			}
			printf("%5d", t->transition[i]->num);
		}
		printf("}");
		break;
	case 2:
		printf("{");
		for (i = 0; i < t->dimen[0]->map->count; i++) {
			if (i > 0) {
				printf(",");
			}
			printf("\n");
			printf("{");
			for (j = 0; j < t->dimen[1]->map->count; j++) {
				Item_Set *ts = transLval(t, i, j);
				if (j > 0) {
					printf(",");
				}
				printf("%5d", (*ts)->num);
			}
			printf("}");
		}
		printf("\n}\n");
		break;
	default:
		assert(0);
	}
}
