<ompts:test>
<ompts:testdescription>Test which checks the omp atomic directive by counting up a variable in a parallelized loop with an atomic directive.</ompts:testdescription>
<ompts:ompversion>2.0</ompts:ompversion>
<ompts:directive>omp atomic</ompts:directive>
<ompts:testcode>
#include <stdio.h>
#include <unistd.h>
#include <math.h>

#include "omp_testsuite.h"
#include "omp_my_sleep.h"

int <ompts:testcode:functionname>omp_atomic</ompts:testcode:functionname> (FILE * logFile)
{
    <ompts:orphan:vars>
	int sum;
        int diff;
        double dsum = 0;
        double dt = 0.5;	/* base of geometric row for + and - test*/
        double ddiff;
        int product;
        int x;
        int *logics;
        int bit_and = 1;
        int bit_or = 0;
        int exclusiv_bit_or = 0;
    </ompts:orphan:vars>

#define DOUBLE_DIGITS 20	/* dt^DOUBLE_DIGITS */
#define MAX_FACTOR 10
#define KNOWN_PRODUCT 3628800	/* 10! */
    int j;
    int known_sum;
    int known_diff;
    int known_product;
    int result = 0;
    int logic_and = 1;
    int logic_or = 0;
    double dknown_sum;
    double rounding_error = 1.E-9;
    double dpt, div;
    int logicsArray[LOOPCOUNT];
    logics = logicsArray;
    
    sum = 0;
    diff = 0;
    product = 1;

#pragma omp parallel
    {
	<ompts:orphan>
	    int i;
#pragma omp for
	    for (i = 1; i <= LOOPCOUNT; i++)
	    {
		<ompts:check>#pragma omp atomic</ompts:check>
		sum += i;
	    }
	</ompts:orphan>
    }
    known_sum = (LOOPCOUNT * (LOOPCOUNT + 1)) / 2;
    if (known_sum != sum)
    {
        fprintf (logFile, 
                 "Error in sum with integers: Result was %d instead of %d.\n", 
                 sum, known_sum);
        result++;
    }
    
#pragma omp parallel
    {
        <ompts:orphan>   
            int i;
#pragma omp for
            for (i = 0; i < LOOPCOUNT; i++)
            {
                 <ompts:check>#pragma omp atomic</ompts:check>
                 diff -= i;
            }
        </ompts:orphan>
    }
    known_diff = ((LOOPCOUNT - 1) * LOOPCOUNT) / 2 * -1;
    if (diff != known_diff)
    {
        fprintf (logFile,
              "Error in difference with integers: Result was %d instead of 0.\n",
               diff);
        result++;
    }

    /* Tests for doubles */
    dsum = 0;
    dpt = 1;

    for (j = 0; j < DOUBLE_DIGITS; ++j)
      {
        dpt *= dt;
      }
    dknown_sum = (1 - dpt) / (1 -dt);
#pragma omp parallel
    {
        <ompts:orphan>
            int i;
#pragma omp for
            for (i = 0; i < DOUBLE_DIGITS; ++i)
            {
                <ompts:check>#pragma omp atomic</ompts:check>
                dsum += pow (dt, i);
            }
        </ompts:orphan>
    }

    if (dsum != dknown_sum && (fabs (dsum - dknown_sum) > rounding_error))
    {
        fprintf (logFile,
                 "Error in sum with doubles: Result was %f instead of: %f (Difference: %E)\n",
                 dsum, dknown_sum, dsum - dknown_sum);
        result++;
    }

    dpt = 1;

    for (j = 0; j < DOUBLE_DIGITS; ++j)
    {
        dpt *= dt;
    }
    ddiff = (1 - dpt) / (1 - dt);
#pragma omp parallel
   {
         <ompts:orphan>
            int i;
#pragma omp for
            for (i = 0; i < DOUBLE_DIGITS; ++i)
            {
                <ompts:check>#pragma omp atomic</ompts:check>
                ddiff -= pow (dt, i);
            }
         </ompts:orphan>
    } 
    if (fabs (ddiff) > rounding_error)
    {
        fprintf (logFile,
                 "Error in difference with doubles: Result was %E instead of 0.0\n",
                 ddiff);
        result++;
    }

#pragma omp parallel
    {
         <ompts:orphan>
            int i;
#pragma omp for
            for (i = 1; i <= MAX_FACTOR; i++)
            {
                <ompts:check>#pragma omp atomic</ompts:check>
                product *= i;
            }
         </ompts:orphan>
    }
    
    known_product = KNOWN_PRODUCT;
    if (known_product != product)
    {
        fprintf (logFile,
                 "Error in product with integers: Result was %d instead of %d\n",
                 product, known_product);
        result++;
    }

    product = KNOWN_PRODUCT;
#pragma omp parallel
    {
        <ompts:orphan>
           int i;
#pragma omp for
            for (i = 1; i <= MAX_FACTOR; ++i)
            {
                <ompts:check>#pragma omp atomic</ompts:check>
                product /= i;
            }
         </ompts:orphan>
    }

    if (product != 1)
    {
        fprintf (logFile,
                 "Error in product division with integers: Result was %d instead of 1\n",
                 product);
        result++;
    }
    
    div = 5.0E+5;
#pragma omp parallel
    {
            int i;
#pragma omp for
            for (i = 1; i <= MAX_FACTOR; i++)
            {
                <ompts:check>#pragma omp atomic</ompts:check>
                div /= i;
            }
    }

    if (fabs(div-0.137787) >= 1.0E-4 )
    {
        result++;
        fprintf (logFile,
                 "Error in division with double: Result was %f instead of 0.137787\n", div);
    }

    x = 0;

#pragma omp parallel
    {
        <ompts:orphan>
            int i;
#pragma omp for
            for (i = 0; i < LOOPCOUNT; ++i)
            {
                <ompts:check>#pragma omp atomic</ompts:check>
                x++;
            }
         </ompts:orphan>
    }

    if (x != LOOPCOUNT)
    {
        result++;
        fprintf (logFile, "Error in ++\n");
    }

#pragma omp parallel
    {
        <ompts:orphan>
            int i;
#pragma omp for
            for (i = 0; i < LOOPCOUNT; ++i)
            {
                <ompts:check>#pragma omp atomic</ompts:check>
                x--;
            }
        </ompts:orphan>
    }

    if (x != 0)
    {
        result++;
        fprintf (logFile, "Error in --\n");
    }

    for (j = 0; j < LOOPCOUNT; ++j)
    {
        logics[j] = 1;
    }
    bit_and = 1;

#pragma omp parallel
    {
        <ompts:orphan>
           int i;
#pragma omp for
            for (i = 0; i < LOOPCOUNT; ++i)
            {
                <ompts:check>#pragma omp atomic</ompts:check>
                bit_and &= logics[i];
            }
         </ompts:orphan>
    }

    if (!bit_and)
    {
        result++;
        fprintf (logFile, "Error in BIT AND part 1\n");
    }

    bit_and = 1;
    logics[LOOPCOUNT / 2] = 0;

#pragma omp parallel
    {
        <ompts:orphan>
            int i;
#pragma omp for
            for (i = 0; i < LOOPCOUNT; ++i)
            {
                <ompts:check>#pragma omp atomic</ompts:check>
                bit_and &= logics[i];
            }
        </ompts:orphan>
    }

    if (bit_and)
    {
        result++;
        fprintf (logFile, "Error in BIT AND part 2\n");
    }

    for (j = 0; j < LOOPCOUNT; j++)
    {
        logics[j] = 0;
    }
    bit_or = 0;

#pragma omp parallel
    {
        <ompts:orphan>
            int i;
#pragma omp for
            for (i = 0; i < LOOPCOUNT; ++i)
            {
                <ompts:check>#pragma omp atomic</ompts:check>
                bit_or |= logics[i];
            }
        </ompts:orphan>
    }

    if (bit_or)
    {
        result++;
        fprintf (logFile, "Error in BIT OR part 1\n");
    }
    bit_or = 0;
    logics[LOOPCOUNT / 2] = 1;

#pragma omp parallel
    {
        <ompts:orphan>
            int i;
#pragma omp for
            for (i = 0; i < LOOPCOUNT; ++i)
            {
                <ompts:check>#pragma omp atomic</ompts:check>
                bit_or |= logics[i];
            }
        </ompts:orphan>
    }
                                                                                   
    if (!bit_or)
    {
        result++;
        fprintf (logFile, "Error in BIT OR part 2\n");
    }

    for (j = 0; j < LOOPCOUNT; j++)
    {
        logics[j] = 0;
    }
    exclusiv_bit_or = 0;

#pragma omp parallel
    {
        <ompts:orphan> 
            int i;
#pragma omp for
            for (i = 0; i < LOOPCOUNT; ++i)
            {
                 <ompts:check>#pragma omp atomic</ompts:check>
                 exclusiv_bit_or ^= logics[i];
            }
        </ompts:orphan>
    }
                                                                                   
    if (exclusiv_bit_or) 
    {
        result++;
        fprintf (logFile, "Error in EXCLUSIV BIT OR part 1\n");
    }

    exclusiv_bit_or = 0;
    logics[LOOPCOUNT / 2] = 1;
    
#pragma omp parallel
    {
        <ompts:orphan> 
            int i;
#pragma omp for
            for (i = 0; i < LOOPCOUNT; ++i)
            {
                 <ompts:check>#pragma omp atomic</ompts:check>
                 exclusiv_bit_or ^= logics[i];
            }
        </ompts:orphan>
    }
                                                                                   
    if (!exclusiv_bit_or) 
    {
        result++;
        fprintf (logFile, "Error in EXCLUSIV BIT OR part 2\n");
    }

    x = 1;
#pragma omp parallel
    {
        <ompts:orphan>
            int i;
#pragma omp for
            for (i = 0; i < 10; ++i)
            {
                 <ompts:check>#pragma omp atomic</ompts:check>
                 x <<= 1;
            }
        </ompts:orphan>
    }

    if ( x != 1024)
    {
        result++;
        fprintf (logFile, "Error in <<\n");
        x = 1024;
    }

#pragma omp parallel
    {
        <ompts:orphan>
            int i;
#pragma omp for
            for (i = 0; i < 10; ++i)
            {
                <ompts:check>#pragma omp atomic</ompts:check>
                x >>= 1;
            }
        </ompts:orphan>
    }

    if (x != 1)
    {
        result++;
        fprintf (logFile, "Error in >>\n");
    }

    return (result == 0);
}
</ompts:testcode>
</ompts:test>
