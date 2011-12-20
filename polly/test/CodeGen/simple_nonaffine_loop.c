#include <stdio.h>
#include <stdlib.h>
#include <math.h>

int main()
{
	int A[1024*1024];
	int i;
	for (i = 0; i < 1024; i++)
		A[i*i] = 2*i;

	printf("Random Value: %d", A[rand() % 1024*1024]);

	return 0;
}
