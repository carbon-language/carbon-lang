/* -*- mode: c -*-
 * $Id$
 * http://www.bagley.org/~doug/shootout/
 *
 * this program is modified from:
 *   http://cm.bell-labs.com/cm/cs/who/bwk/interps/pap.html
 * Timing Trials, or, the Trials of Timing: Experiments with Scripting
 * and User-Interface Languages</a> by Brian W. Kernighan and
 * Christopher J. Van Wyk.
 *
 * I added free() to deallocate memory.
 */

#include <stdio.h>
#include <stdlib.h>

int
main(int argc, char *argv[]) {
    int n = ((argc == 2) ? atoi(argv[1]) : 1);
    int i, k, *x, *y;
	
    x = (int *) calloc(n, sizeof(int));
    y = (int *) calloc(n, sizeof(int));

    for (i = 0; i < n; i++) {
	x[i] = i + 1;
    }
    for (k=0; k<1000; k++) {
	for (i = n-1; i >= 0; i--) {
	    y[i] += x[i];
	}
    }

    printf("%d %d\n", y[0], y[n-1]);

    free(x);
    free(y);

    return(0);
}

