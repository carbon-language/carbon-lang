char rcsid_rule[] = "$Id$";

#include "b.h"
#include <stdio.h>

RuleNum max_rule;
int max_erule_num;

struct rule stub_rule;

List rules;

Rule
newRule(delta, erulenum, lhs, pat) DeltaPtr delta; ERuleNum erulenum; NonTerminal lhs; Pattern pat;
{
	Rule p;

	p = (Rule) zalloc(sizeof(struct rule));
	assert(p);
	ASSIGNCOST(p->delta, delta);
	p->erulenum = erulenum;
	if (erulenum > max_erule_num) {
		max_erule_num = erulenum;
	}
	p->num = max_rule++;
	p->lhs = lhs;
	p->pat = pat;

	rules = newList(p, rules);

	return p;
}

void
dumpRule(p) Rule p;
{
	dumpNonTerminal(p->lhs);
	printf(" : ");
	dumpPattern(p->pat);
	printf(" ");
	dumpCost(p->delta);
	printf("\n");
}

void
dumpRuleList(l) List l;
{
	foreachList((ListFn)dumpRule, l);
}
