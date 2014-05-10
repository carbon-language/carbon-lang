<ompts:test>
<ompts:testdescription>Test which checks the omp threadprivate directive by filling an array with random numbers in an parallelised region. Each thread generates one number of the array and saves this in a temporary threadprivate variable. In a second parallelised region the test controls, that the temporary variable contains still the former value by comparing it with the one in the array.</ompts:testdescription>
<ompts:ompversion>2.0</ompts:ompversion>
<ompts:directive>omp threadprivate</ompts:directive>
<ompts:dependences>omp critical,omp_set_dynamic,omp_get_num_threads</ompts:dependences>
<ompts:testcode>
/*
 * Threadprivate is tested in 2 ways:
 * 1. The global variable declared as threadprivate should have
 *    local copy for each thread. Otherwise race condition and 
 *    wrong result.
 * 2. If the value of local copy is retained for the two adjacent
 *    parallel regions
 */
#include "omp_testsuite.h"
#include <stdlib.h>
#include <stdio.h>

static int sum0=0;
static int myvalue = 0;

<ompts:check>#pragma omp threadprivate(sum0)</ompts:check>
<ompts:check>#pragma omp threadprivate(myvalue)</ompts:check>


int <ompts:testcode:functionname>omp_threadprivate</ompts:testcode:functionname>(FILE * logFile)
{
	int sum = 0;
	int known_sum;
	int i; 
	int iter;
	int *data;
	int size;
	int failed = 0;
	int my_random;
	omp_set_dynamic(0);

    #pragma omp parallel private(i) 
    {
	  sum0 = 0;
      #pragma omp for 
	    for (i = 1; i <= LOOPCOUNT; i++)
		{
			sum0 = sum0 + i;
		} /*end of for*/
      #pragma omp critical
	  {
	      sum = sum + sum0;
	  } /*end of critical */
	} /* end of parallel */    
	known_sum = (LOOPCOUNT * (LOOPCOUNT + 1)) / 2;
	if (known_sum != sum ) {
		fprintf (logFile, " known_sum = %d, sum = %d\n", known_sum, sum);
	}

	/* the next parallel region is just used to get the number of threads*/
	omp_set_dynamic(0);
    #pragma omp parallel
	{
      #pragma omp master
	  {
			size=omp_get_num_threads();
			data=(int*) malloc(size*sizeof(int));
	  }
	}/* end parallel*/


	srand(45);
	for (iter = 0; iter < 100; iter++){
		my_random = rand();	/* random number generator is called inside serial region*/

	/* the first parallel region is used to initialiye myvalue and the array with my_random+rank*/
    #pragma omp parallel
	{
	    int rank;
		rank = omp_get_thread_num ();
		myvalue = data[rank] = my_random + rank;
	}

	/* the second parallel region verifies that the value of "myvalue" is retained */
    #pragma omp parallel reduction(+:failed)
	{
	    int rank;
		rank = omp_get_thread_num ();
		failed = failed + (myvalue != data[rank]);
		if(myvalue != data[rank]){
		  fprintf (logFile, " myvalue = %d, data[rank]= %d\n", myvalue, data[rank]);
		}
	}
  }
  free (data);

	return (known_sum == sum) && !failed;

} /* end of check_threadprivate*/
</ompts:testcode>
</ompts:test>
