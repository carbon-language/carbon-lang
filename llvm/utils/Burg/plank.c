char rcsid_plank[] = "$Id$";

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "b.h"
#include "fe.h"

#define ERROR_VAL 0

int speedflag = 0;

Item_Set *sortedStates;
static struct stateMapTable smt;
int exceptionTolerance = 0;
static int plankSize = 32;

static Plank newPlank ARGS((void));
static PlankMap newPlankMap ARGS((int));
static StateMap newStateMap ARGS((void));
static Exception newException ARGS((int, int));
static void enterStateMap ARGS((PlankMap, short *, int, int *));
static List assemblePlanks ARGS((void));
static void assignRules ARGS((RuleAST));
static int stateCompare ARGS((Item_Set *, Item_Set *));
static int ruleCompare ARGS((RuleAST *, RuleAST *));
static void renumber ARGS((void));
static short * newVector ARGS((void));
static int width ARGS((int));
static PlankMap mapToPmap ARGS((Dimension));
static void doDimPmaps ARGS((Operator));
static void doNonTermPmaps ARGS((NonTerminal));
static void makePmaps ARGS((void));
static void outPlank ARGS((Plank));
static void purgePlanks ARGS((List));
static void inToEx ARGS((void));
static void makePlankRuleMacros ARGS((void));
static void makePlankRule ARGS((void));
static void exceptionSwitch ARGS((List, const char *, const char *, const char *, int, const char *));
static void doPlankLabel ARGS((Operator));
static void doPlankLabelSafely ARGS((Operator));
static void doPlankLabelMacrosSafely ARGS((Operator));
static void makePlankState ARGS((void));

static Plank
newPlank()
{
	Plank p;
	char buf[50];
	static int num = 0;

	p = (Plank) zalloc(sizeof(struct plank));
	sprintf(buf, "%s_plank_%d", prefix, num++);
	p->name = (char *) zalloc(strlen(buf)+1);
	strcpy(p->name, buf);
	return p;
}

static PlankMap
newPlankMap(offset) int offset;
{
	PlankMap im;

	im = (PlankMap) zalloc(sizeof(struct plankMap));
	im->offset = offset;
	return im;
}

static StateMap
newStateMap()
{
	char buf[50];
	static int num = 0;

	StateMap sm;

	sm = (StateMap) zalloc(sizeof(struct stateMap));
	sprintf(buf, "f%d", num++);
	sm->fieldname = (char *) zalloc(strlen(buf)+1);
	strcpy(sm->fieldname, buf);
	return sm;
}

static Exception
newException(index, value) int index; int value;
{
	Exception e;

	e = (Exception) zalloc(sizeof(struct except));
	e->index = index;
	e->value = value;
	return e;
}

static void
enterStateMap(im, v, width, new) PlankMap im; short * v; int width; int *new;
{
	int i;
	StateMap sm;
	List l;
	int size;

	assert(im);
	assert(v);
	assert(width > 0);
	size = globalMap->count;

	for (l = smt.maps; l; l = l->next) {
		int ecount;

		sm = (StateMap) l->x;
		ecount = 0;
		for (i = 0; i < size; i++) {
			if (v[i] != -1 && sm->value[i] != -1 && v[i] != sm->value[i]) {
				if (++ecount > exceptionTolerance) {
					goto again;
				}
			}
		}
		for (i = 0; i < size; i++) {
			assert(v[i] >= 0);
			assert(sm->value[i] >= 0);
			if (v[i] == -1) {
				continue;
			}
			if (sm->value[i] == -1) {
				sm->value[i] = v[i];
			} else if (v[i] != sm->value[i]) {
				im->exceptions = newList(newException(i,v[i]), im->exceptions);
			}
		}
		im->values = sm;
		if (width > sm->width) {
			sm->width = width;
		}
		*new = 0;
		return;
	again: ;
	}
	sm = newStateMap();
	im->values = sm;
	sm->value = v;
	sm->width = width;
	*new = 1;
	smt.maps = newList(sm, smt.maps);
}

static List
assemblePlanks()
{
	List planks = 0;
	Plank pl;
	List p;
	List s;

	for (s = smt.maps; s; s = s->next) {
		StateMap sm = (StateMap) s->x;
		for (p = planks; p; p = p->next) {
			pl = (Plank) p->x;
			if (sm->width <= plankSize - pl->width) {
				pl->width += sm->width;
				pl->fields = newList(sm, pl->fields);
				sm->plank = pl;
				goto next;
			}
		}
		pl = newPlank();
		pl->width = sm->width;
		pl->fields = newList(sm, 0);
		sm->plank = pl;
		planks = appendList(pl, planks);
	next: ;
	}
	return planks;
}

RuleAST *sortedRules;

static int count;

static void
assignRules(ast) RuleAST ast;
{
	sortedRules[count++] = ast;
}

static int
stateCompare(s, t) Item_Set *s; Item_Set *t;
{
	return strcmp((*s)->op->name, (*t)->op->name);
}

static int
ruleCompare(s, t) RuleAST *s; RuleAST *t;
{
	return strcmp((*s)->lhs, (*t)->lhs);
}

void
dumpSortedStates()
{
	int i;
	
	printf("dump Sorted States: ");
	for (i = 0; i < globalMap->count; i++) {
		printf("%d ", sortedStates[i]->num);
	}
	printf("\n");
}

void
dumpSortedRules()
{
	int i;
	
	printf("dump Sorted Rules: ");
	for (i = 0; i < max_ruleAST; i++) {
		printf("%d ", sortedRules[i]->rule->erulenum);
	}
	printf("\n");
}

static void
renumber()
{
	int i;
	Operator previousOp;
	NonTerminal previousLHS;
	int base_counter;

	sortedStates = (Item_Set*) zalloc(globalMap->count * sizeof(Item_Set));
	for (i = 1; i < globalMap->count; i++) {
		sortedStates[i-1] = globalMap->set[i];
	}
	qsort(sortedStates, globalMap->count-1, sizeof(Item_Set), (int(*)(const void *, const void *))stateCompare);
	previousOp = 0;
	for (i = 0; i < globalMap->count-1; i++) {
		sortedStates[i]->newNum = i;
		sortedStates[i]->op->stateCount++;
		if (previousOp != sortedStates[i]->op) {
			sortedStates[i]->op->baseNum = i;
			previousOp = sortedStates[i]->op;
		}
	}

	sortedRules = (RuleAST*) zalloc(max_ruleAST * sizeof(RuleAST));
	count = 0;
	foreachList((ListFn) assignRules, ruleASTs);
	qsort(sortedRules, max_ruleAST, sizeof(RuleAST), (int(*)(const void *, const void *))ruleCompare);
	previousLHS = 0;
	base_counter = 0;
	for (i = 0; i < max_ruleAST; i++) {
		if (previousLHS != sortedRules[i]->rule->lhs) {
			sortedRules[i]->rule->lhs->baseNum = base_counter;
			previousLHS = sortedRules[i]->rule->lhs;
			base_counter++; /* make space for 0 */
		}
		sortedRules[i]->rule->newNum = base_counter;
		sortedRules[i]->rule->lhs->ruleCount++;
		sortedRules[i]->rule->lhs->sampleRule = sortedRules[i]->rule; /* kludge for diagnostics */
		base_counter++;
	}
}

static short *
newVector()
{
	short *p;
	p = (short *) zalloc(globalMap->count* sizeof(short));
	return p;
}

static int
width(v) int v;
{
	int c;

	for (c = 0; v; v >>= 1) {
		c++;
	}
	return c;

}

static PlankMap
mapToPmap(d) Dimension d;
{
	PlankMap im;
	short *v;
	int i;
	int new;

	if (d->map->count == 1) {
		return 0;
	}
	assert(d->map->count > 1);
	im = newPlankMap(0);
	v = newVector();
	for (i = 0; i < globalMap->count-1; i++) {
		int index = d->map->set[d->index_map.class[sortedStates[i]->num]->num]->num;
		assert(index >= 0);
		v[i+1] = index;
	}
	v[0] = 0;
	enterStateMap(im, v, width(d->map->count), &new);
	if (!new) {
		zfree(v);
	}
	return im;
}

static void
doDimPmaps(op) Operator op;
{
	int i, j;
	Dimension d;
	short *v;
	PlankMap im;
	int new;

	if (!op->table->rules) {
		return;
	}
	switch (op->arity) {
	case 0:
		break;
	case 1:
		d = op->table->dimen[0];
		if (d->map->count > 1) {
			v = newVector();
			im = newPlankMap(op->baseNum);
			for (i = 0; i < globalMap->count-1; i++) {
				int index = d->map->set[d->index_map.class[sortedStates[i]->num]->num]->num;
				if (index) {
					Item_Set *ts = transLval(op->table, index, 0);
					v[i+1] = (*ts)->newNum - op->baseNum+1;
					assert(v[i+1] >= 0);
				}
			}
			enterStateMap(im, v, width(d->map->count-1), &new);
			if (!new) {
				zfree(v);
			}
			d->pmap = im;
		}
		break;
	case 2:
		if (op->table->dimen[0]->map->count == 1 && op->table->dimen[1]->map->count == 1) {
			op->table->dimen[0]->pmap = 0;
			op->table->dimen[1]->pmap = 0;
		} else if (op->table->dimen[0]->map->count == 1) {
			v = newVector();
			im = newPlankMap(op->baseNum);
			d = op->table->dimen[1];
			for (i = 0; i < globalMap->count-1; i++) {
				int index = d->map->set[d->index_map.class[sortedStates[i]->num]->num]->num;
				if (index) {
					Item_Set *ts = transLval(op->table, 1, index);
					v[i+1] = (*ts)->newNum - op->baseNum+1;
					assert(v[i+1] >= 0);
				}
			}
			enterStateMap(im, v, width(d->map->count-1), &new);
			if (!new) {
				zfree(v);
			}
			d->pmap = im;
		} else if (op->table->dimen[1]->map->count == 1) {
			v = newVector();
			im = newPlankMap(op->baseNum);
			d = op->table->dimen[0];
			for (i = 0; i < globalMap->count-1; i++) {
				int index = d->map->set[d->index_map.class[sortedStates[i]->num]->num]->num;
				if (index) {
					Item_Set *ts = transLval(op->table, index, 1);
					v[i +1] = (*ts)->newNum - op->baseNum +1;
					assert(v[i +1] >= 0);
				}
			}
			enterStateMap(im, v, width(d->map->count-1), &new);
			if (!new) {
				zfree(v);
			}
			d->pmap = im;
		} else {
			op->table->dimen[0]->pmap = mapToPmap(op->table->dimen[0]);
			op->table->dimen[1]->pmap = mapToPmap(op->table->dimen[1]);
			/* output table */
			fprintf(outfile, "static unsigned %s %s_%s_transition[%d][%d] = {", 
				op->stateCount <= 255 ? "char" : "short",
				prefix,
				op->name,
				op->table->dimen[0]->map->count,
				op->table->dimen[1]->map->count);
			for (i = 0; i < op->table->dimen[0]->map->count; i++) {
				if (i > 0) {
					fprintf(outfile, ",");
				}
				fprintf(outfile, "\n{");
				for (j = 0; j < op->table->dimen[1]->map->count; j++) {
					Item_Set *ts = transLval(op->table, i, j);
					short diff;
					if (j > 0) {
						fprintf(outfile, ",");
						if (j % 10 == 0) {
							fprintf(outfile, "\t/* row %d, cols %d-%d*/\n",
								i,
								j-10,
								j-1);
						}
					}
					if ((*ts)->num > 0) {
						diff = (*ts)->newNum - op->baseNum +1;
					} else {
						diff = 0;
					}
					fprintf(outfile, "%5d", diff);
				}
				fprintf(outfile, "}\t/* row %d */", i);
			}
			fprintf(outfile, "\n};\n");
		}
		break;
	default:
		assert(0);
	}
}

static NonTerminal *ntVector;

static void
doNonTermPmaps(n) NonTerminal n;
{
	short *v;
	PlankMap im;
	int new;
	int i;

	ntVector[n->num] = n;
	if (n->num >= last_user_nonterminal) {
		return;
	}
	if (n->ruleCount <= 0) {
		return;
	}
	im = newPlankMap(n->baseNum);
	v = newVector();
	for (i = 0; i < globalMap->count-1; i++) {
		Rule r = globalMap->set[sortedStates[i]->num]->closed[n->num].rule;
		if (r) {
			r->used = 1;
			v[i+1] = r->newNum - n->baseNum /*safely*/;
			assert(v[i+1] >= 0);
		}
	}
	enterStateMap(im, v, width(n->ruleCount+1), &new);
	if (!new) {
		zfree(v);
	}
	n->pmap = im;
}

static void
makePmaps()
{
	foreachList((ListFn) doDimPmaps, operators);
	ntVector = (NonTerminal*) zalloc((max_nonterminal) * sizeof(NonTerminal));
	foreachList((ListFn) doNonTermPmaps, nonterminals);
}

static void
outPlank(p) Plank p;
{
	List f;
	int i;

	fprintf(outfile, "static struct {\n");

	for (f = p->fields; f; f = f->next) {
		StateMap sm = (StateMap) f->x;
		fprintf(outfile, "\tunsigned int %s:%d;\n", sm->fieldname, sm->width);
	}

	fprintf(outfile, "} %s[] = {\n", p->name);

	for (i = 0; i < globalMap->count; i++) {
		fprintf(outfile, "\t{");
		for (f = p->fields; f; f = f->next) {
			StateMap sm = (StateMap) f->x;
			fprintf(outfile, "%4d,", sm->value[i] == -1 ? ERROR_VAL : sm->value[i]);
		}
		fprintf(outfile, "},\t/* row %d */\n", i);
	}

	fprintf(outfile, "};\n");
}

static void
purgePlanks(planks) List planks;
{
	List p;

	for (p = planks; p; p = p->next) {
		Plank x = (Plank) p->x;
		outPlank(x);
	}
}

static void
inToEx()
{
	int i;
	int counter;

	fprintf(outfile, "static short %s_eruleMap[] = {\n", prefix);
	counter = 0;
	for (i = 0; i < max_ruleAST; i++) {
		if (counter > 0) {
			fprintf(outfile, ",");
			if (counter % 10 == 0) {
				fprintf(outfile, "\t/* %d-%d */\n", counter-10, counter-1);
			}
		}
		if (counter < sortedRules[i]->rule->newNum) {
			assert(counter == sortedRules[i]->rule->newNum-1);
			fprintf(outfile, "%5d", 0);
			counter++;
			if (counter > 0) {
				fprintf(outfile, ",");
				if (counter % 10 == 0) {
					fprintf(outfile, "\t/* %d-%d */\n", counter-10, counter-1);
				}
			}
		}
		fprintf(outfile, "%5d", sortedRules[i]->rule->erulenum);
		counter++;
	}
	fprintf(outfile, "\n};\n");
}

static void
makePlankRuleMacros()
{
	int i;

	for (i = 1; i < last_user_nonterminal; i++) {
		List es;
		PlankMap im = ntVector[i]->pmap;
		fprintf(outfile, "#define %s_%s_rule(state)\t", prefix, ntVector[i]->name);
		if (im) {
			fprintf(outfile, "%s_eruleMap[", prefix);
			for (es = im->exceptions; es; es = es->next) {
				Exception e = (Exception) es->x;
				fprintf(outfile, "((state) == %d ? %d :", 
						e->index, e->value);
			}
			fprintf(outfile, "%s[state].%s", 
				im->values->plank->name, 
				im->values->fieldname);
			for (es = im->exceptions; es; es = es->next) {
				fprintf(outfile, ")");
			}
			fprintf(outfile, " +%d]", im->offset);

		} else {
			/* nonterminal never appears on LHS. */
			assert(ntVector[i] ==  start);
			fprintf(outfile, "0");
		}
		fprintf(outfile, "\n");
	}
	fprintf(outfile, "\n");
}

static void
makePlankRule()
{
	int i;

	makePlankRuleMacros();

	fprintf(outfile, "#ifdef __STDC__\n");
	fprintf(outfile, "int %s_rule(int state, int goalnt) {\n", prefix);
	fprintf(outfile, "#else\n");
	fprintf(outfile, "int %s_rule(state, goalnt) int state; int goalnt; {\n", prefix);
	fprintf(outfile, "#endif\n");

	fprintf(outfile, 
	"\t%s_assert(state >= 0 && state < %d, %s_PANIC(\"Bad state %%d passed to %s_rule\\n\", state));\n",
				prefix, globalMap->count, prefix, prefix);
	fprintf(outfile, "\tswitch(goalnt) {\n");

	for (i = 1; i < last_user_nonterminal; i++) {
		fprintf(outfile, "\tcase %d:\n", i);
		fprintf(outfile, "\t\treturn %s_%s_rule(state);\n", prefix, ntVector[i]->name);
	}
	fprintf(outfile, "\tdefault:\n");
	fprintf(outfile, "\t\t%s_PANIC(\"Unknown nonterminal %%d in %s_rule;\\n\", goalnt);\n", prefix, prefix);
	fprintf(outfile, "\t\tabort();\n");
	fprintf(outfile, "\t\treturn 0;\n");
	fprintf(outfile, "\t}\n");
	fprintf(outfile, "}\n");
}

static void
exceptionSwitch(es, sw, pre, post, offset, def) List es; const char *sw; const char *pre; const char *post; int offset; const char *def;
{
	if (es) {
		fprintf(outfile, "\t\tswitch (%s) {\n", sw);
		for (; es; es = es->next) {
			Exception e = (Exception) es->x;
			fprintf(outfile, "\t\tcase %d: %s %d; %s\n", e->index, pre, e->value+offset, post);
		}
		if (def) {
			fprintf(outfile, "\t\tdefault: %s;\n", def);
		}
		fprintf(outfile, "\t\t}\n");
	} else {
		if (def) {
			fprintf(outfile, "\t\t%s;\n", def);
		}
	}
}

static void
doPlankLabel(op) Operator op;
{
	PlankMap im0;
	PlankMap im1;
	char buf[100];

	fprintf(outfile, "\tcase %d:\n", op->num);
	switch (op->arity) {
	case 0:
		fprintf(outfile, "\t\treturn %d;\n", op->table->transition[0]->newNum);
		break;
	case 1:
		im0 = op->table->dimen[0]->pmap;
		if (im0) {
			exceptionSwitch(im0->exceptions, "l", "return ", "", im0->offset, 0);
			fprintf(outfile, "\t\treturn %s[l].%s + %d;\n", 
				im0->values->plank->name, im0->values->fieldname, im0->offset);
		} else {
			Item_Set *ts = transLval(op->table, 1, 0);
			if (*ts) {
				fprintf(outfile, "\t\treturn %d;\n", (*ts)->newNum);
			} else {
				fprintf(outfile, "\t\treturn %d;\n", ERROR_VAL);
			}
		}
		break;
	case 2:
		im0 = op->table->dimen[0]->pmap;
		im1 = op->table->dimen[1]->pmap;
		if (!im0 && !im1) {
			Item_Set *ts = transLval(op->table, 1, 1);
			if (*ts) {
				fprintf(outfile, "\t\treturn %d;\n", (*ts)->newNum);
			} else {
				fprintf(outfile, "\t\treturn %d;\n", ERROR_VAL);
			}
		} else if (!im0) {
			exceptionSwitch(im1->exceptions, "r", "return ", "", im1->offset, 0);
			fprintf(outfile, "\t\treturn %s[r].%s + %d;\n", 
				im1->values->plank->name, im1->values->fieldname, im1->offset);
		} else if (!im1) {
			exceptionSwitch(im0->exceptions, "l", "return ", "", im0->offset, 0);
			fprintf(outfile, "\t\treturn %s[l].%s + %d;\n", 
				im0->values->plank->name, im0->values->fieldname, im0->offset);
		} else {
			assert(im0->offset == 0);
			assert(im1->offset == 0);
			sprintf(buf, "l = %s[l].%s",
				im0->values->plank->name, im0->values->fieldname);
			exceptionSwitch(im0->exceptions, "l", "l =", "break;", 0, buf);
			sprintf(buf, "r = %s[r].%s",
				im1->values->plank->name, im1->values->fieldname);
			exceptionSwitch(im1->exceptions, "r", "r =", "break;", 0, buf);

			fprintf(outfile, "\t\treturn %s_%s_transition[l][r] + %d;\n", 
				prefix,
				op->name,
				op->baseNum);
		}
		break;
	default:
		assert(0);
	}
}

static void
doPlankLabelMacrosSafely(op) Operator op;
{
	PlankMap im0;
	PlankMap im1;

	switch (op->arity) {
	case -1:
		fprintf(outfile, "#define %s_%s_state\t0\n", prefix, op->name);
		break;
	case 0:
		fprintf(outfile, "#define %s_%s_state", prefix, op->name);
		fprintf(outfile, "\t%d\n", op->table->transition[0]->newNum+1);
		break;
	case 1:
		fprintf(outfile, "#define %s_%s_state(l)", prefix, op->name);
		im0 = op->table->dimen[0]->pmap;
		if (im0) {
			if (im0->exceptions) {
				List es = im0->exceptions;
				assert(0);
				fprintf(outfile, "\t\tswitch (l) {\n");
				for (; es; es = es->next) {
					Exception e = (Exception) es->x;
					fprintf(outfile, "\t\tcase %d: return %d;\n", e->index, e->value ? e->value+im0->offset : 0);
				}
				fprintf(outfile, "\t\t}\n");
			}
			if (speedflag) {
				fprintf(outfile, "\t( %s[l].%s + %d )\n",
					im0->values->plank->name, im0->values->fieldname,
					im0->offset);
			} else {
				fprintf(outfile, "\t( (%s_TEMP = %s[l].%s) ? %s_TEMP + %d : 0 )\n",
					prefix,
					im0->values->plank->name, im0->values->fieldname,
					prefix,
					im0->offset);
			}
		} else {
			Item_Set *ts = transLval(op->table, 1, 0);
			if (*ts) {
				fprintf(outfile, "\t%d\n", (*ts)->newNum+1);
			} else {
				fprintf(outfile, "\t%d\n", 0);
			}
		}
		break;
	case 2:
		fprintf(outfile, "#define %s_%s_state(l,r)", prefix, op->name);

		im0 = op->table->dimen[0]->pmap;
		im1 = op->table->dimen[1]->pmap;
		if (!im0 && !im1) {
			Item_Set *ts = transLval(op->table, 1, 1);
			assert(0);
			if (*ts) {
				fprintf(outfile, "\t\treturn %d;\n", (*ts)->newNum+1);
			} else {
				fprintf(outfile, "\t\treturn %d;\n", 0);
			}
		} else if (!im0) {
			assert(0);
			if (im1->exceptions) {
				List es = im1->exceptions;
				fprintf(outfile, "\t\tswitch (r) {\n");
				for (; es; es = es->next) {
					Exception e = (Exception) es->x;
					fprintf(outfile, "\t\tcase %d: return %d;\n", e->index, e->value ? e->value+im1->offset : 0);
				}
				fprintf(outfile, "\t\t}\n");
			}
			fprintf(outfile, "\t\tstate = %s[r].%s; offset = %d;\n", 
				im1->values->plank->name, im1->values->fieldname, im1->offset);
			fprintf(outfile, "\t\tbreak;\n");
		} else if (!im1) {
			assert(0);
			if (im0->exceptions) {
				List es = im0->exceptions;
				fprintf(outfile, "\t\tswitch (l) {\n");
				for (; es; es = es->next) {
					Exception e = (Exception) es->x;
					fprintf(outfile, "\t\tcase %d: return %d;\n", e->index, e->value ? e->value+im0->offset : 0);
				}
				fprintf(outfile, "\t\t}\n");
			}
			fprintf(outfile, "\t\tstate = %s[l].%s; offset = %d;\n", 
				im0->values->plank->name, im0->values->fieldname, im0->offset);
			fprintf(outfile, "\t\tbreak;\n");
		} else {
			assert(im0->offset == 0);
			assert(im1->offset == 0);
			/*
			sprintf(buf, "l = %s[l].%s",
				im0->values->plank->name, im0->values->fieldname);
			exceptionSwitch(im0->exceptions, "l", "l =", "break;", 0, buf);
			sprintf(buf, "r = %s[r].%s",
				im1->values->plank->name, im1->values->fieldname);
			exceptionSwitch(im1->exceptions, "r", "r =", "break;", 0, buf);

			fprintf(outfile, "\t\tstate = %s_%s_transition[l][r]; offset = %d;\n", 
				prefix,
				op->name,
				op->baseNum);
			fprintf(outfile, "\t\tbreak;\n");
			*/

			if (speedflag) {
				fprintf(outfile, "\t( %s_%s_transition[%s[l].%s][%s[r].%s] + %d)\n",
					prefix,
					op->name,
					im0->values->plank->name, im0->values->fieldname,
					im1->values->plank->name, im1->values->fieldname,
					op->baseNum);
			} else {
				fprintf(outfile, "\t( (%s_TEMP = %s_%s_transition[%s[l].%s][%s[r].%s]) ? ",
					prefix,
					prefix,
					op->name,
					im0->values->plank->name, im0->values->fieldname,
					im1->values->plank->name, im1->values->fieldname);
				fprintf(outfile, "%s_TEMP + %d : 0 )\n",
					prefix,
					op->baseNum);
			}
		}
		break;
	default:
		assert(0);
	}
}
static void
doPlankLabelSafely(op) Operator op;
{
	fprintf(outfile, "\tcase %d:\n", op->num);
	switch (op->arity) {
	case -1:
		fprintf(outfile, "\t\treturn 0;\n");
		break;
	case 0:
		fprintf(outfile, "\t\treturn %s_%s_state;\n", prefix, op->name);
		break;
	case 1:
		fprintf(outfile, "\t\treturn %s_%s_state(l);\n", prefix, op->name);
		break;
	case 2:
		fprintf(outfile, "\t\treturn %s_%s_state(l,r);\n", prefix, op->name);
		break;
	default:
		assert(0);
	}
}

static void
makePlankState()
{
	fprintf(outfile, "\n");
	fprintf(outfile, "int %s_TEMP;\n", prefix);
	foreachList((ListFn) doPlankLabelMacrosSafely, operators);
	fprintf(outfile, "\n");

	fprintf(outfile, "#ifdef __STDC__\n");
	switch (max_arity) {
	case -1:
		fprintf(stderr, "ERROR: no terminals in grammar.\n");
		exit(1);
	case 0:
		fprintf(outfile, "int %s_state(int op) {\n", prefix);
		fprintf(outfile, "#else\n");
		fprintf(outfile, "int %s_state(op) int op; {\n", prefix);
		break;
	case 1:
		fprintf(outfile, "int %s_state(int op, int l) {\n", prefix);
		fprintf(outfile, "#else\n");
		fprintf(outfile, "int %s_state(op, l) int op; int l; {\n", prefix);
		break;
	case 2:
		fprintf(outfile, "int %s_state(int op, int l, int r) {\n", prefix);
		fprintf(outfile, "#else\n");
		fprintf(outfile, "int %s_state(op, l, r) int op; int l; int r; {\n", prefix);
		break;
	default:
		assert(0);
	}
	fprintf(outfile, "#endif\n");

	fprintf(outfile, "\tregister int %s_TEMP;\n", prefix);

	fprintf(outfile, "#ifndef NDEBUG\n");

	fprintf(outfile, "\tswitch (op) {\n");
	opsOfArity(2);
	if (max_arity >= 2) {
		fprintf(outfile, 
		"\t\t%s_assert(r >= 0 && r < %d, %s_PANIC(\"Bad state %%d passed to %s_state\\n\", r));\n",
				prefix, globalMap->count, prefix, prefix);
		fprintf(outfile, "\t\t/*FALLTHROUGH*/\n");
	}
	opsOfArity(1);
	if (max_arity > 1) {
		fprintf(outfile, 
		"\t\t%s_assert(l >= 0 && l < %d, %s_PANIC(\"Bad state %%d passed to %s_state\\n\", l));\n",
				prefix, globalMap->count, prefix, prefix);
		fprintf(outfile, "\t\t/*FALLTHROUGH*/\n");
	}
	opsOfArity(0);
	fprintf(outfile, "\t\tbreak;\n");
	fprintf(outfile, "\t}\n");
	fprintf(outfile, "#endif\n");

	fprintf(outfile, "\tswitch (op) {\n");
	fprintf(outfile,"\tdefault: %s_PANIC(\"Unknown op %%d in %s_state\\n\", op); abort(); return 0;\n",
		prefix, prefix);
	foreachList((ListFn) doPlankLabelSafely, operators);
	fprintf(outfile, "\t}\n");

	fprintf(outfile, "}\n");
}

void
makePlanks()
{
	List planks;
	renumber();
	makePmaps();
	planks = assemblePlanks();
	purgePlanks(planks);
	inToEx();
	makePlankRule();
	makePlankState();
}
