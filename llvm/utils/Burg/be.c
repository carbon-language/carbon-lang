char rcsid_be[] = "$Id$";

#include <stdio.h>
#include <string.h>
#include "b.h"
#include "fe.h"

#define ERROR_VAL 0

FILE *outfile;
const char *prefix = "burm";

static void doKids ARGS((RuleAST));
static void doLabel ARGS((Operator));
static void doLayout ARGS((RuleAST));
static void doMakeTable ARGS((Operator));
static void doVector ARGS((RuleAST));
static void layoutNts ARGS((PatternAST));
static void makeIndex_Map ARGS((Dimension));
static void makePvector ARGS((void));
static void makeState ARGS((void));
static void printPatternAST ARGS((PatternAST));
static void printPatternAST_int ARGS((PatternAST));
static void setVectors ARGS((PatternAST));
static void trailing_zeroes ARGS((int));
static int seminal ARGS((int from, int to));
static void printRule ARGS((RuleAST, const char *));

static void
doLabel(op) Operator op;
{
	fprintf(outfile, "\tcase %d:\n", op->num);

	switch (op->arity) {
	default:
		assert(0);
		break;
	case 0:
		fprintf(outfile, "\t\treturn %d;\n", op->table->transition[0]->num);
		break;
	case 1:
		if (op->table->rules) {
			fprintf(outfile, "\t\treturn %s_%s_transition[l];\n", prefix, op->name);
		} else {
			fprintf(outfile, "\t\treturn %d;\n", ERROR_VAL);
		}
		break;
	case 2:
		if (op->table->rules) {
			fprintf(outfile, "\t\treturn %s_%s_transition[%s_%s_imap_1[l]][%s_%s_imap_2[r]];\n", prefix, op->name, prefix, op->name, prefix, op->name);
		} else {
			fprintf(outfile, "\t\treturn %d;\n", ERROR_VAL);
		}
		break;
	}
}

int
opsOfArity(arity) int arity;
{
	int c;
	List l;

	c = 0;
	for (l = operators; l; l = l->next) {
		Operator op = (Operator) l->x;
		if (op->arity == arity) {
			fprintf(outfile, "\tcase %d:\n", op->num);
			c++;
		}
	}
	return c;
}

static void
trailing_zeroes(z) int z;
{
	int i;

	for (i = 0; i < z; i++) {
		fprintf(outfile, ", 0");
	}
}

void
makeLabel()
{
	int flag;

	fprintf(outfile, "#ifdef __STDC__\n");
	fprintf(outfile, "int %s_label(%s_NODEPTR_TYPE n) {\n", prefix, prefix);
	fprintf(outfile, "#else\n");
	fprintf(outfile, "int %s_label(n) %s_NODEPTR_TYPE n; {\n", prefix, prefix);
	fprintf(outfile, "#endif\n");

	fprintf(outfile, 
	"\t%s_assert(n, %s_PANIC(\"NULL pointer passed to %s_label\\n\"));\n",
				prefix, prefix, prefix);
	fprintf(outfile, "\tswitch (%s_OP_LABEL(n)) {\n", prefix);
	fprintf(outfile, "\tdefault: %s_PANIC(\"Bad op %%d in %s_label\\n\", %s_OP_LABEL(n)); abort(); return 0;\n", 
			prefix, prefix, prefix);

	flag = opsOfArity(0);
	if (flag > 0) {
		fprintf(outfile, "\t\treturn %s_STATE_LABEL(n) = %s_state(%s_OP_LABEL(n)",
					prefix, prefix, prefix);
		trailing_zeroes(max_arity);
		fprintf(outfile, ");\n");
	}
	flag = opsOfArity(1);
	if (flag > 0) {
		fprintf(outfile, "\t\treturn %s_STATE_LABEL(n) = %s_state(%s_OP_LABEL(n), %s_label(%s_LEFT_CHILD(n))",
					prefix, prefix, prefix, prefix, prefix);
		trailing_zeroes(max_arity-1);
		fprintf(outfile, ");\n");
	}
	flag = opsOfArity(2);
	if (flag > 0) {
		fprintf(outfile, "\t\treturn %s_STATE_LABEL(n) = %s_state(%s_OP_LABEL(n), %s_label(%s_LEFT_CHILD(n)), %s_label(%s_RIGHT_CHILD(n))",
					prefix, prefix, prefix, prefix, prefix, prefix, prefix);
		trailing_zeroes(max_arity-2);
		fprintf(outfile, ");\n");

	}
	fprintf(outfile, "\t}\n");
	fprintf(outfile, "}\n");
}

static void
makeState()
{
	fprintf(outfile, "int %s_state(int op, int l, int r) {\n", prefix);
	fprintf(outfile, 
	"\t%s_assert(l >= 0 && l < %d, PANIC(\"Bad state %%d passed to %s_state\\n\", l));\n",
				prefix, globalMap->count, prefix);
	fprintf(outfile, 
	"\t%s_assert(r >= 0 && r < %d, PANIC(\"Bad state %%d passed to %s_state\\n\", r));\n",
				prefix, globalMap->count, prefix);
	fprintf(outfile, "\tswitch (op) {\n");
	fprintf(outfile, "\tdefault: %s_PANIC(\"Bad op %%d in %s_state\\n\", op); abort(); return 0;\n", prefix, prefix);

	foreachList((ListFn) doLabel, operators);

	fprintf(outfile, "\t}\n");
	fprintf(outfile, "}\n");
}

static char cumBuf[4000];
static int vecIndex;
char vecBuf[4000];

static void
setVectors(ast) PatternAST ast;
{
	char old[4000];

	switch (ast->sym->tag) {
	default:
		assert(0);
		break;
	case NONTERMINAL:
		sprintf(old, "\t\tkids[%d] = %s;\n", vecIndex, vecBuf);
		strcat(cumBuf, old);
		vecIndex++;
		return;
	case OPERATOR:
		switch (ast->sym->u.op->arity) {
		default:
			assert(0);
			break;
		case 0:
			return;
		case 1:
			strcpy(old, vecBuf);
			sprintf(vecBuf, "%s_LEFT_CHILD(%s)", prefix, old);
			setVectors((PatternAST) ast->children->x);
			strcpy(vecBuf, old);
			return;
		case 2:
			strcpy(old, vecBuf);
			sprintf(vecBuf, "%s_LEFT_CHILD(%s)", prefix, old);
			setVectors((PatternAST) ast->children->x);

			sprintf(vecBuf, "%s_RIGHT_CHILD(%s)", prefix, old);
			setVectors((PatternAST) ast->children->next->x);
			strcpy(vecBuf, old);
			return;
		}
		break;
	}
}

#define MAX_VECTOR	10

void
makeRuleTable()
{
	int s,nt;

	fprintf(outfile, "static short %s_RuleNo[%d][%d] = {\n", prefix, globalMap->count, last_user_nonterminal-1);
	for (s = 0; s < globalMap->count; s++) {
		Item_Set ts = globalMap->set[s];
		if (s > 0) {
			fprintf(outfile, ",\n");
		}
		fprintf(outfile, "/* state %d */\n", s);
		fprintf(outfile, "{");
		for (nt = 1; nt < last_user_nonterminal; nt++) {
			if (nt > 1) {
				fprintf(outfile, ",");
				if (nt % 10 == 1) {
					fprintf(outfile, "\t/* state %d; Nonterminals %d-%d */\n", s, nt-10, nt-1);
				}
			}
			if (ts->closed[nt].rule) {
				ts->closed[nt].rule->used = 1;
				fprintf(outfile, "%5d", ts->closed[nt].rule->erulenum);
			} else {
				fprintf(outfile, "%5d", ERROR_VAL);
			}
		}
		fprintf(outfile, "}");
	}
	fprintf(outfile, "};\n");
}

static void
makeIndex_Map(d) Dimension d;
{
	int s;

	for (s = 0; s < globalMap->count; s++) {
		if (s > 0) {
			fprintf(outfile, ",");
			if (s % 10 == 0) {
				fprintf(outfile, "\t/* %d-%d */\n", s-10, s-1);
			}
		}
		fprintf(outfile, "%5d", d->map->set[d->index_map.class[s]->num]->num);
	}
	fprintf(outfile, "};\n");
}

static void
doMakeTable(op) Operator op;
{
	int s;
	int i,j;
	Dimension d;

	switch (op->arity) {
	default:
		assert(0);
		break;
	case 0:
		return;
	case 1:
		if (!op->table->rules) {
			return;
		}
		d = op->table->dimen[0];
		fprintf(outfile, "static short %s_%s_transition[%d] = {\n", prefix, op->name, globalMap->count);
		for (s = 0; s < globalMap->count; s++) {
			if (s > 0) {
				fprintf(outfile, ", ");
				if (s % 10 == 0) {
					fprintf(outfile, "\t/* %d-%d */\n", s-10, s-1);
				}
			}
			fprintf(outfile, "%5d", op->table->transition[d->map->set[d->index_map.class[s]->num]->num]->num);
		}
		fprintf(outfile, "};\n");
		break;
	case 2:
		if (!op->table->rules) {
			return;
		}
		fprintf(outfile, "static short %s_%s_imap_1[%d] = {\n", prefix, op->name, globalMap->count);
		makeIndex_Map(op->table->dimen[0]);
		fprintf(outfile, "static short %s_%s_imap_2[%d] = {\n", prefix, op->name, globalMap->count);
		makeIndex_Map(op->table->dimen[1]);

		fprintf(outfile, "static short %s_%s_transition[%d][%d] = {", prefix, op->name,
						op->table->dimen[0]->map->count,
						op->table->dimen[1]->map->count);
		for (i = 0; i < op->table->dimen[0]->map->count; i++) {
			if (i > 0) {
				fprintf(outfile, ",");
			}
			fprintf(outfile, "\n");
			fprintf(outfile, "{");
			for (j = 0; j < op->table->dimen[1]->map->count; j++) {
				Item_Set *ts = transLval(op->table, i, j);
				if (j > 0) {
					fprintf(outfile, ",");
				}
				fprintf(outfile, "%5d", (*ts)->num);
			}
			fprintf(outfile, "}\t/* row %d */", i);
		}
		fprintf(outfile, "\n};\n");

		break;
	}
}

void
makeTables()
{
	foreachList((ListFn) doMakeTable, operators);
}

RuleAST *pVector;

void
makeLHSmap()
{
	int i;

	if (!pVector) {
		makePvector();
	}

	fprintf(outfile, "short %s_lhs[] = {\n", prefix);
	for (i = 0; i <= max_erule_num; i++) {
		if (pVector[i]) {
			fprintf(outfile, "\t%s_%s_NT,\n", prefix, pVector[i]->lhs);
		} else {
			fprintf(outfile, "\t0,\n");
		}
	}
	fprintf(outfile, "};\n\n");
}

static int seminal(int from, int to)
{
	return allpairs[from][to].rule ? allpairs[from][to].rule->erulenum : 0;

	/*
	int tmp, last;
	tmp = 0;
	for (;;) {
		last = tmp;
		tmp = allpairs[to][from].rule ? allpairs[to][from].rule->erulenum : 0;
		if (!tmp) {
			break;
		}
		assert(pVector[tmp]);
		to = pVector[tmp]->rule->pat->children[0]->num;
	}
	return last;
	*/
}

void
makeClosureArray()
{
	int i, j;

	if (!pVector) {
		makePvector();
	}

	fprintf(outfile, "short %s_closure[%d][%d] = {\n", prefix, last_user_nonterminal, last_user_nonterminal);
	for (i = 0; i < last_user_nonterminal; i++) {
		fprintf(outfile, "\t{");
		for (j = 0; j < last_user_nonterminal; j++) {
			if (j > 0 && j%10 == 0) {
				fprintf(outfile, "\n\t ");
			}
			fprintf(outfile, " %4d,", seminal(i,j));
		}
		fprintf(outfile, "},\n");
	}
	fprintf(outfile, "};\n");
}

void
makeCostVector(z,d) int z; DeltaCost d;
{
	fprintf(outfile, "\t{");
#ifdef NOLEX
	if (z) {
		fprintf(outfile, "%5d", d);
	} else {
		fprintf(outfile, "%5d", 0);
	}
#else
	{
	int j;
	for (j = 0; j < DELTAWIDTH; j++) {
		if (j > 0) {
			fprintf(outfile, ",");
		}
		if (z) {
			fprintf(outfile, "%5d", d[j]);
		} else {
			fprintf(outfile, "%5d", 0);
		}
	}
	}
#endif /* NOLEX */
	fprintf(outfile, "}");
}

void
makeCostArray()
{
	int i;

	if (!pVector) {
		makePvector();
	}

	fprintf(outfile, "short %s_cost[][%d] = {\n", prefix, DELTAWIDTH);
	for (i = 0; i <= max_erule_num; i++) {
		makeCostVector(pVector[i], pVector[i] ? pVector[i]->rule->delta : 0);
		fprintf(outfile, ", /* ");
		printRule(pVector[i], "(none)");
		fprintf(outfile, " = %d */\n", i);
	}
	fprintf(outfile, "};\n");
}

void
makeStateStringArray()
{
	int s;
	int nt;
	int states;
	
	states = globalMap->count;
	fprintf(outfile, "\nconst char * %s_state_string[] = {\n", prefix);
	fprintf(outfile, "\" not a state\", /* state 0 */\n");
	for (s = 0; s < states-1; s++) {
		fprintf(outfile, "\t\"");
		printRepresentative(outfile, sortedStates[s]);
		fprintf(outfile, "\", /* state #%d */\n", s+1);
	}
	fprintf(outfile, "};\n");
}

void
makeDeltaCostArray()
{
	int s;
	int nt;
	int states;
	
	states = globalMap->count;
	fprintf(outfile, "\nshort %s_delta_cost[%d][%d][%d] = {\n", prefix, states, last_user_nonterminal, DELTAWIDTH);
	fprintf(outfile, "{{0}}, /* state 0 */\n");
	for (s = 0; s < states-1; s++) {
		fprintf(outfile, "{ /* state #%d: ", s+1);
		printRepresentative(outfile, sortedStates[s]);
		fprintf(outfile, " */\n");
		fprintf(outfile, "\t{0},\n");
		for (nt = 1; nt < last_user_nonterminal; nt++) {
			makeCostVector(1, sortedStates[s]->closed[nt].delta);
			fprintf(outfile, ", /* ");
			if (sortedStates[s]->closed[nt].rule) {
				int erulenum = sortedStates[s]->closed[nt].rule->erulenum;
				printRule(pVector[erulenum], "(none)");
				fprintf(outfile, " = %d */", erulenum);
			} else {
				fprintf(outfile, "(none) */");
			}
			fprintf(outfile, "\n");
		}
		fprintf(outfile, "},\n");
	}
	fprintf(outfile, "};\n");
}

static void
printPatternAST_int(p) PatternAST p;
{
	List l;

	if (p) {
		switch (p->sym->tag) {
		case NONTERMINAL:
			fprintf(outfile, "%5d,", -p->sym->u.nt->num);
			break;
		case OPERATOR:
			fprintf(outfile, "%5d,", p->sym->u.op->num);
			break;
		default:
			assert(0);
		}
		if (p->children) {
			for (l = p->children; l; l = l->next) {
				PatternAST pat = (PatternAST) l->x;
				printPatternAST_int(pat);
			}
		}
	}
}

static void
printPatternAST(p) PatternAST p;
{
	List l;

	if (p) {
		fprintf(outfile, "%s", p->op);
		if (p->children) {
			fprintf(outfile, "(");
			for (l = p->children; l; l = l->next) {
				PatternAST pat = (PatternAST) l->x;
				if (l != p->children) {
					fprintf(outfile, ", ");
				}
				printPatternAST(pat);
			}
			fprintf(outfile, ")");
		}
	}
}

static void
layoutNts(ast) PatternAST ast;
{
	char out[30];

	switch (ast->sym->tag) {
	default:
		assert(0);
		break;
	case NONTERMINAL:
		sprintf(out, "%d, ", ast->sym->u.nt->num);
		strcat(cumBuf, out);
		return;
	case OPERATOR:
		switch (ast->sym->u.op->arity) {
		default:
			assert(0);
			break;
		case 0:
			return;
		case 1:
			layoutNts((PatternAST) ast->children->x);
			return;
		case 2:
			layoutNts((PatternAST) ast->children->x);
			layoutNts((PatternAST) ast->children->next->x);
			return;
		}
		break;
	}
}

static void
doVector(ast) RuleAST ast;
{
	if (pVector[ast->rule->erulenum]) {
		fprintf(stderr, "ERROR: non-unique external rule number: (%d)\n", ast->rule->erulenum);
		exit(1);
	}
	pVector[ast->rule->erulenum] = ast;
}

static void
makePvector()
{
	pVector = (RuleAST*) zalloc((max_erule_num+1) * sizeof(RuleAST));
	foreachList((ListFn) doVector, ruleASTs);
}

static void
doLayout(ast) RuleAST ast;
{
	sprintf(cumBuf, "{ ");
	layoutNts(ast->pat);
	strcat(cumBuf, "0 }");
}

void
makeNts()
{
	int i;
	int new;
	StrTable nts;

	nts = newStrTable();

	if (!pVector) {
		makePvector();
	}

	for (i = 0; i <= max_erule_num; i++) {
		if (pVector[i]) {
			doLayout(pVector[i]);
			pVector[i]->nts = addString(nts, cumBuf, i, &new);
			if (new) {
				char ename[50];

				sprintf(ename, "%s_r%d_nts", prefix, i);
				pVector[i]->nts->ename = (char*) zalloc(strlen(ename)+1);
				strcpy(pVector[i]->nts->ename, ename);
				fprintf(outfile, "static short %s[] =", ename);
				fprintf(outfile, "%s;\n", cumBuf);
			}
		}
	}

	fprintf(outfile, "short *%s_nts[] = {\n", prefix);
	for (i = 0; i <= max_erule_num; i++) {
		if (pVector[i]) {
			fprintf(outfile, "\t%s,\n", pVector[i]->nts->ename);
		} else {
			fprintf(outfile, "\t0,\n");
		}
	}
	fprintf(outfile, "};\n");
}

static void
printRule(RuleAST r, const char *d)
{
	if (r) {
		fprintf(outfile, "%s: ", r->rule->lhs->name);
		printPatternAST(r->pat);
	} else {
		fprintf(outfile, "%s", d);
	}
}

void
makeRuleDescArray()
{
	int i;

	if (!pVector) {
		makePvector();
	}

	if (last_user_nonterminal != max_nonterminal) {
		/* not normal form */
		fprintf(outfile, "short %s_rule_descriptor_0[] = { 0, 0 };\n", prefix);
	} else {
		fprintf(outfile, "short %s_rule_descriptor_0[] = { 0, 1 };\n", prefix);
	}
	for (i = 1; i <= max_erule_num; i++) {
		if (pVector[i]) {
			Operator o;
			NonTerminal t;

			fprintf(outfile, "short %s_rule_descriptor_%d[] = {", prefix, i);
			fprintf(outfile, "%5d,", -pVector[i]->rule->lhs->num);
			printPatternAST_int(pVector[i]->pat);
			fprintf(outfile, " };\n");
		}
	}

	fprintf(outfile, "/* %s_rule_descriptors[0][1] = 1 iff grammar is normal form. */\n", prefix);
	fprintf(outfile, "short * %s_rule_descriptors[] = {\n", prefix);
	fprintf(outfile, "\t%s_rule_descriptor_0,\n", prefix);
	for (i = 1; i <= max_erule_num; i++) {
		if (pVector[i]) {
			fprintf(outfile, "\t%s_rule_descriptor_%d,\n", prefix, i);
		} else {
			fprintf(outfile, "\t%s_rule_descriptor_0,\n", prefix);
		}
	}
	fprintf(outfile, "};\n");
}


void
makeRuleDescArray2()
{
	int i;

	if (!pVector) {
		makePvector();
	}

	fprintf(outfile, "struct { int lhs, op, left, right; } %s_rule_struct[] = {\n", prefix);
	if (last_user_nonterminal != max_nonterminal) {
		/* not normal form */
		fprintf(outfile, "\t{-1},");
	} else {
		fprintf(outfile, "\t{0},");
	}
	fprintf(outfile, " /* 0 if normal form, -1 if not normal form */\n");
	for (i = 1; i <= max_erule_num; i++) {
		fprintf(outfile, "\t");
		if (pVector[i]) {
			Operator o;
			NonTerminal t1, t2;

			fprintf(outfile, "{");
			fprintf(outfile, "%5d, %5d, %5d, %5d",
				pVector[i]->rule->lhs->num,
				(o = pVector[i]->rule->pat->op) ? o->num : 0,
				(t1 = pVector[i]->rule->pat->children[0]) ? t1->num : 0,
				(t2 = pVector[i]->rule->pat->children[1]) ? t2->num : 0
				);
			fprintf(outfile, "} /* ");
			printRule(pVector[i], "0");
			fprintf(outfile, " = %d */", i);
		} else {
			fprintf(outfile, "{0}");
		}
		fprintf(outfile, ",\n");
	}
	fprintf(outfile, "};\n");
}

void
makeStringArray()
{
	int i;

	if (!pVector) {
		makePvector();
	}

	fprintf(outfile, "const char *%s_string[] = {\n", prefix);
	for (i = 0; i <= max_erule_num; i++) {
		fprintf(outfile, "\t");
		if (pVector[i]) {
			fprintf(outfile, "\"");
			printRule(pVector[i], "0");
			fprintf(outfile, "\"");
		} else {
			fprintf(outfile, "0");
		}
		fprintf(outfile, ",\n");
	}
	fprintf(outfile, "};\n");
	fprintf(outfile, "int %s_max_rule = %d;\n", prefix, max_erule_num);
	fprintf(outfile, "#define %s_Max_rule %d\n", prefix, max_erule_num);
}

void
makeRule()
{
	fprintf(outfile, "int %s_rule(int state, int goalnt) {\n", prefix);
	fprintf(outfile, 
	"\t%s_assert(state >= 0 && state < %d, PANIC(\"Bad state %%d passed to %s_rule\\n\", state));\n",
				prefix, globalMap->count, prefix);
	fprintf(outfile, 
	"\t%s_assert(goalnt >= 1 && goalnt < %d, PANIC(\"Bad goalnt %%d passed to %s_rule\\n\", state));\n",
				prefix, max_nonterminal, prefix);
	fprintf(outfile, "\treturn %s_RuleNo[state][goalnt-1];\n", prefix);
	fprintf(outfile, "};\n");
}

static StrTable kids;

static void
doKids(ast) RuleAST ast;
{
	int new;

	vecIndex = 0;
	cumBuf[0] = 0;
	strcpy(vecBuf, "p");
	setVectors(ast->pat);

	ast->kids = addString(kids, cumBuf, ast->rule->erulenum, &new);

}

void
makeKids()
{
	List e;
	IntList r;

	kids = newStrTable();

	fprintf(outfile, "#ifdef __STDC__\n");
	fprintf(outfile, "%s_NODEPTR_TYPE * %s_kids(%s_NODEPTR_TYPE p, int rulenumber, %s_NODEPTR_TYPE *kids) {\n", prefix, prefix, prefix, prefix);
	fprintf(outfile, "#else\n");
	fprintf(outfile, "%s_NODEPTR_TYPE * %s_kids(p, rulenumber, kids) %s_NODEPTR_TYPE p; int rulenumber; %s_NODEPTR_TYPE *kids; {\n", prefix, prefix, prefix, prefix);
	fprintf(outfile, "#endif\n");

	fprintf(outfile, 
	"\t%s_assert(p, %s_PANIC(\"NULL node pointer passed to %s_kids\\n\"));\n",
				prefix, prefix, prefix);
	fprintf(outfile, 
	"\t%s_assert(kids, %s_PANIC(\"NULL kids pointer passed to %s_kids\\n\"));\n",
				prefix, prefix, prefix);
	fprintf(outfile, "\tswitch (rulenumber) {\n");
	fprintf(outfile, "\tdefault:\n");
	fprintf(outfile, "\t\t%s_PANIC(\"Unknown Rule %%d in %s_kids;\\n\", rulenumber);\n", prefix, prefix);
	fprintf(outfile, "\t\tabort();\n");
	fprintf(outfile, "\t\t/* NOTREACHED */\n");

	foreachList((ListFn) doKids, ruleASTs);

	for (e = kids->elems; e; e = e->next) {
		StrTableElement el = (StrTableElement) e->x;
		for (r = el->erulenos; r; r = r->next) {
			int i = r->x;
			fprintf(outfile, "\tcase %d:\n", i);
		}
		fprintf(outfile, "%s", el->str);
		fprintf(outfile, "\t\tbreak;\n");
	}
	fprintf(outfile, "\t}\n");
	fprintf(outfile, "\treturn kids;\n");
	fprintf(outfile, "}\n");
}

void
makeOpLabel()
{
	fprintf(outfile, "#ifdef __STDC__\n");
	fprintf(outfile, "int %s_op_label(%s_NODEPTR_TYPE p) {\n", prefix, prefix);
	fprintf(outfile, "#else\n");
	fprintf(outfile, "int %s_op_label(p) %s_NODEPTR_TYPE p; {\n", prefix, prefix);
	fprintf(outfile, "#endif\n");
	fprintf(outfile, 
	"\t%s_assert(p, %s_PANIC(\"NULL pointer passed to %s_op_label\\n\"));\n",
				prefix, prefix, prefix);
	fprintf(outfile, "\treturn %s_OP_LABEL(p);\n", prefix);
	fprintf(outfile, "}\n");
}

void
makeStateLabel()
{
	fprintf(outfile, "#ifdef __STDC__\n");
	fprintf(outfile, "int %s_state_label(%s_NODEPTR_TYPE p) {\n", prefix, prefix);
	fprintf(outfile, "#else\n");
	fprintf(outfile, "int %s_state_label(p) %s_NODEPTR_TYPE p; {\n", prefix, prefix);
	fprintf(outfile, "#endif\n");

	fprintf(outfile, 
	"\t%s_assert(p, %s_PANIC(\"NULL pointer passed to %s_state_label\\n\"));\n",
				prefix, prefix, prefix);
	fprintf(outfile, "\treturn %s_STATE_LABEL(p);\n", prefix);
	fprintf(outfile, "}\n");
}

void
makeChild()
{
	fprintf(outfile, "#ifdef __STDC__\n");
	fprintf(outfile, "%s_NODEPTR_TYPE %s_child(%s_NODEPTR_TYPE p, int index) {\n", prefix, prefix, prefix);
	fprintf(outfile, "#else\n");
	fprintf(outfile, "%s_NODEPTR_TYPE %s_child(p, index) %s_NODEPTR_TYPE p; int index; {\n", prefix, prefix, prefix);
	fprintf(outfile, "#endif\n");

	fprintf(outfile, 
	"\t%s_assert(p, %s_PANIC(\"NULL pointer passed to %s_child\\n\"));\n",
				prefix, prefix, prefix);
	fprintf(outfile, "\tswitch (index) {\n");
	fprintf(outfile, "\tcase 0:\n");
	fprintf(outfile, "\t\treturn %s_LEFT_CHILD(p);\n", prefix);
	fprintf(outfile, "\tcase 1:\n");
	fprintf(outfile, "\t\treturn %s_RIGHT_CHILD(p);\n", prefix);
	fprintf(outfile, "\t}\n");
	fprintf(outfile, "\t%s_PANIC(\"Bad index %%d in %s_child;\\n\", index);\n", prefix, prefix);
	fprintf(outfile, "\tabort();\n");
	fprintf(outfile, "\treturn 0;\n");
	fprintf(outfile, "}\n");
}

static Operator *opVector;
static int maxOperator;

void
makeOperatorVector()
{
	List l;

	maxOperator = 0;
	for (l = operators; l; l = l->next) {
		Operator op = (Operator) l->x;
		if (op->num > maxOperator) {
			maxOperator = op->num;
		}
	}
	opVector = (Operator*) zalloc((maxOperator+1) * sizeof(*opVector));
	for (l = operators; l; l = l->next) {
		Operator op = (Operator) l->x;
		if (opVector[op->num]) {
			fprintf(stderr, "ERROR: Non-unique external symbol number (%d)\n", op->num);
			exit(1);
		}
		opVector[op->num] = op;
	}
}

void
makeOperators()
{
	int i;

	if (!opVector) {
		makeOperatorVector();
	}
	fprintf(outfile, "const char * %s_opname[] = {\n", prefix);
	for (i = 0; i <= maxOperator; i++) {
		if (i > 0) {
			fprintf(outfile, ", /* %d */\n", i-1);
		}
		if (opVector[i]) {
			fprintf(outfile, "\t\"%s\"", opVector[i]->name);
		} else {
			fprintf(outfile, "\t0");
		}
	}
	fprintf(outfile, "\n};\n");
	fprintf(outfile, "char %s_arity[] = {\n", prefix);
	for (i = 0; i <= maxOperator; i++) {
		if (i > 0) {
			fprintf(outfile, ", /* %d */\n", i-1);
		}
		fprintf(outfile, "\t%d", opVector[i] ? opVector[i]->arity : -1);
	}
	fprintf(outfile, "\n};\n");
	fprintf(outfile, "int %s_max_op = %d;\n", prefix, maxOperator);
	fprintf(outfile, "int %s_max_state = %d;\n", prefix, globalMap->count-1);
	fprintf(outfile, "#define %s_Max_state %d\n", prefix, globalMap->count-1);
}

void
makeDebug()
{
	fprintf(outfile, "#ifdef DEBUG\n");
	fprintf(outfile, "int %s_debug;\n", prefix);
	fprintf(outfile, "#endif /* DEBUG */\n");
}

void
makeSimple()
{
	makeRuleTable();
	makeTables();
	makeRule();
	makeState();
}

void
startOptional()
{
	fprintf(outfile, "#ifdef %s_STATE_LABEL\n", prefix);
	fprintf(outfile, "#define %s_INCLUDE_EXTRA\n", prefix);
	fprintf(outfile, "#else\n");
	fprintf(outfile, "#ifdef STATE_LABEL\n");
	fprintf(outfile, "#define %s_INCLUDE_EXTRA\n", prefix);
	fprintf(outfile, "#define %s_STATE_LABEL \tSTATE_LABEL\n", prefix);
	fprintf(outfile, "#define %s_NODEPTR_TYPE\tNODEPTR_TYPE\n", prefix);
	fprintf(outfile, "#define %s_LEFT_CHILD  \tLEFT_CHILD\n", prefix);
	fprintf(outfile, "#define %s_OP_LABEL    \tOP_LABEL\n", prefix);
	fprintf(outfile, "#define %s_RIGHT_CHILD \tRIGHT_CHILD\n", prefix);
	fprintf(outfile, "#endif /* STATE_LABEL */\n");
	fprintf(outfile, "#endif /* %s_STATE_LABEL */\n\n", prefix);

	fprintf(outfile, "#ifdef %s_INCLUDE_EXTRA\n\n", prefix);

}

void
makeNonterminals()
{
	List l;

	for (l = nonterminals; l; l = l->next) {
		NonTerminal nt = (NonTerminal) l->x;
		if (nt->num < last_user_nonterminal) {
			fprintf(outfile, "#define %s_%s_NT %d\n", prefix, nt->name, nt->num);
		}
	}
	fprintf(outfile, "#define %s_NT %d\n", prefix, last_user_nonterminal-1);
}

void
makeNonterminalArray()
{
	int i;
	List l;
	NonTerminal *nta;

	nta = (NonTerminal *) zalloc(sizeof(*nta) * last_user_nonterminal);

	for (l = nonterminals; l; l = l->next) {
		NonTerminal nt = (NonTerminal) l->x;
		if (nt->num < last_user_nonterminal) {
			nta[nt->num] = nt;
		}
	}

	fprintf(outfile, "const char *%s_ntname[] = {\n", prefix);
	fprintf(outfile, "\t\"Error: Nonterminals are > 0\",\n");
	for (i = 1; i < last_user_nonterminal; i++) {
		fprintf(outfile, "\t\"%s\",\n", nta[i]->name);
	}
	fprintf(outfile, "\t0\n");
	fprintf(outfile, "};\n\n");

	zfree(nta);
}

void
endOptional()
{
	fprintf(outfile, "#endif /* %s_INCLUDE_EXTRA */\n", prefix);
}

void
startBurm()
{
	fprintf(outfile, "#ifndef %s_PANIC\n", prefix);
	fprintf(outfile, "#define %s_PANIC\tPANIC\n", prefix);
	fprintf(outfile, "#endif /* %s_PANIC */\n", prefix);
	fprintf(outfile, "#ifdef __STDC__\n");
	fprintf(outfile, "extern void abort(void);\n");
	fprintf(outfile, "#else\n");
	fprintf(outfile, "extern void abort();\n");
	fprintf(outfile, "#endif\n");
	fprintf(outfile, "#ifdef NDEBUG\n");
 	fprintf(outfile, "#define %s_assert(x,y)\t;\n", prefix);
	fprintf(outfile, "#else\n");
 	fprintf(outfile, "#define %s_assert(x,y)\tif(!(x)) {y; abort();}\n", prefix);
	fprintf(outfile, "#endif\n");
}

void
reportDiagnostics()
{
	List l;

	for (l = operators; l; l = l->next) {
		Operator op = (Operator) l->x;
		if (!op->ref) {
			fprintf(stderr, "warning: Unreferenced Operator: %s\n", op->name);
		}
	}
	for (l = rules; l; l = l->next) {
		Rule r = (Rule) l->x;
		if (!r->used && r->num < max_ruleAST) {
			fprintf(stderr, "warning: Unused Rule: #%d\n", r->erulenum);
		}
	}
	if (!start->pmap) {
		fprintf(stderr, "warning: Start Nonterminal (%s) does not appear on LHS.\n", start->name);
	}

	fprintf(stderr, "start symbol = \"%s\"\n", start->name);
	fprintf(stderr, "# of states = %d\n", globalMap->count-1);
	fprintf(stderr, "# of nonterminals = %d\n", max_nonterminal-1);
	fprintf(stderr, "# of user nonterminals = %d\n", last_user_nonterminal-1);
	fprintf(stderr, "# of rules = %d\n", max_rule);
	fprintf(stderr, "# of user rules = %d\n", max_ruleAST);
}
