char rcsid_zalloc[] = "$Id$";

#include <stdio.h>
#include <string.h>
#include "b.h"

extern void exit ARGS((int));
extern void free ARGS((void *));
extern void *malloc ARGS((unsigned));

int
fatal(const char *name, int line)
{
	fprintf(stderr, "assertion failed: file %s, line %d\n", name, line);
	exit(1);
	return 0;
}

void *
zalloc(size) unsigned int size;
{
	void *t = (void *) malloc(size);
	if (!t) {
		fprintf(stderr, "Malloc failed---PROGRAM ABORTED\n");
		exit(1);
	}
	memset(t, 0, size);
	return t;
}

void
zfree(p) void *p;
{
	free(p);
}
