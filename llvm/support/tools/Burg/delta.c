char rcsid_delta[] = "$Id$";

#include <stdio.h>
#include "b.h"
#include "fe.h"

int principleCost = 0;
int lexical = 0;

#ifndef NOLEX
void
ASSIGNCOST(l, r) DeltaPtr l; DeltaPtr r;
{
	int i;

	if (lexical) {
		for (i = 0; i < DELTAWIDTH; i++) {
			l[i] = r[i];
		}
	} else {
			l[0] = r[0];
	}
}

void
ADDCOST(l, r) DeltaPtr l; DeltaPtr r;
{
	int i;

	if (lexical) {
		for (i = 0; i < DELTAWIDTH; i++) {
			l[i] += r[i];
		}
	} else {
		l[0] += r[0];
	}
}

void
MINUSCOST(l, r) DeltaPtr l; DeltaPtr r;
{
	int i;

	if (lexical) {
		for (i = 0; i < DELTAWIDTH; i++) {
			l[i] -= r[i];
		}
	} else {
		l[0] -= r[0];
	}
}

void
ZEROCOST(x) DeltaPtr x;
{
	int i;

	if (lexical) {
		for (i = 0; i < DELTAWIDTH; i++) {
			x[i] = 0;
		}
	} else {
		x[0] = 0;
	}
}

int
LESSCOST(l, r) DeltaPtr l; DeltaPtr r;
{
	int i;

	if (lexical) {
		for (i = 0; i < DELTAWIDTH; i++) {
			if (l[i] < r[i]) {
				return 1;
			} else if (l[i] > r[i]) {
				return 0;
			}
		}
		return 0;
	} else {
		return l[0] < r[0];
	}
}

int
EQUALCOST(l, r) DeltaPtr l; DeltaPtr r;
{
	int i;

	if (lexical) {
		for (i = 0; i < DELTAWIDTH; i++) {
			if (l[i] != r[i]) {
				return 0;
			}
		}
		return 1;
	} else {
		return l[0] == r[0];
	}
}
#endif /* NOLEX */

void
CHECKDIVERGE(c, its, nt, base) DeltaPtr c; Item_Set its; int nt; int base;
{
	int i;

	if (prevent_divergence <= 0) {
		return;
	}
	if (lexical) {
#ifndef NOLEX
		for (i = 0; i < DELTAWIDTH; i++) {
			if (c[i] > prevent_divergence) {
				char ntname[100];
				char basename[100];
				nonTerminalName(ntname, nt);
				nonTerminalName(basename, base);
				fprintf(stderr, "ERROR:  The grammar appears to diverge\n");
				fprintf(stderr, "\tRelative Costs: %s(0), %s(%d)\n", basename, ntname, c[i]);
				fprintf(stderr, "\tOffending Operator: %s\n", its->op->name);
				fprintf(stderr, "\tOffending Tree: ");
				printRepresentative(stderr, its);
				fprintf(stderr, "\n");
				exit(1);
			}
		}
#endif /*NOLEX*/
	} else if (PRINCIPLECOST(c) > prevent_divergence) {
		char ntname[100];
		char basename[100];
		nonTerminalName(ntname, nt);
		nonTerminalName(basename, base);
		fprintf(stderr, "ERROR:  The grammar appears to diverge\n");
		fprintf(stderr, "\tRelative Costs: %s(0), %s(%d)\n", basename, ntname, PRINCIPLECOST(c));
		fprintf(stderr, "\tOffending Operator: %s\n", its->op->name);
		fprintf(stderr, "\tOffending Tree: ");
		printRepresentative(stderr, its);
		fprintf(stderr, "\n");
		exit(1);
	}
}
