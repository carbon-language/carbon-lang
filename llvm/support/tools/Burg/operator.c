char rcsid_operator[] = "$Id$";

#include "b.h"
#include <stdio.h>

int max_arity = -1;

List operators;
List leaves;

Operator
newOperator(name, num, arity) char *name; OperatorNum num; ArityNum arity;
{
	Operator op;

	assert(arity <= MAX_ARITY);
	op = (Operator) zalloc(sizeof(struct operator));
	assert(op);
	op->name = name;
	op->num = num;
	op->arity = arity;

	operators = newList(op, operators);

	return op;
}

void
dumpOperator_s(op) Operator op;
{
	printf("Op: %s(%d)=%d\n", op->name, op->arity, op->num);
}

void
dumpOperator(op, full) Operator op; int full;
{
	dumpOperator_s(op);
	if (full) {
		dumpTable(op->table, 0);
	}
}

void
dumpOperator_l(op) Operator op;
{
	dumpOperator(op, 1);
}

