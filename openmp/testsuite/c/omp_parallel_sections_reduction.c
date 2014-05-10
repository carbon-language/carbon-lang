<ompts:test>
<ompts:testdescription>Test which checks the omp parallel sections reduction directive with all its option.</ompts:testdescription>
<ompts:ompversion>2.0</ompts:ompversion>
<ompts:directive>omp parallel sections reduction</ompts:directive>
<ompts:testcode>
#include <stdio.h>
#include <math.h>
#include "omp_testsuite.h"


int <ompts:testcode:functionname>omp_parallel_sections_reduction</ompts:testcode:functionname>(FILE * logFile){
	<ompts:orphan:vars>
    int sum;
	int known_sum;	
	double dpt;
    double dsum;
	double dknown_sum;
	double dt=0.5;				/* base of geometric row for + and - test*/
	double rounding_error= 1.E-5;
	int diff;
	double ddiff;
	int product;
	int known_product;
	int logic_and;
	int bit_and;
	int logic_or;
	int bit_or;
	int exclusiv_bit_or;
	int logics[1000];
	int i;
	int result;
    </ompts:orphan:vars>
    
    sum = 7;
    dsum=0;
    product =1;
    dpt = 1;
	logic_and=1;
	bit_and=1;
	logic_or=0;
	bit_or=0;
	exclusiv_bit_or=0;
    result =0;
	/*  int my_islarger;*/
	/*int is_larger=1;*/
	known_sum = (999*1000)/2+7;

    <ompts:orphan>
#pragma omp parallel sections private(i) <ompts:check>reduction(+:sum)</ompts:check><ompts:crosscheck></ompts:crosscheck>
	{
#pragma omp section
		{
			for (i=1;i<300;i++)
			{
				sum=sum+i;
			}
		}
#pragma omp section
		{
			for (i=300;i<700;i++)
			{
				sum=sum+i;
			}
		}
#pragma omp section
		{
			for (i=700;i<1000;i++)
			{
				sum=sum+i;
			}
		}
	}

	if(known_sum!=sum)
	{
		result++;
		fprintf(logFile,"Error in sum with integers: Result was %d instead of %d.\n",sum, known_sum);
	}

	diff = (999*1000)/2;
#pragma omp parallel sections private(i) <ompts:check>reduction(-:diff)</ompts:check><ompts:crosscheck></ompts:crosscheck>
	{
#pragma omp section
		{
			for (i=1;i<300;i++)
			{
				diff=diff-i;
			}
		}
#pragma omp section
		{
			for (i=300;i<700;i++)
			{
				diff=diff-i;
			}
		}
#pragma omp section
		{
			for (i=700;i<1000;i++)
			{
				diff=diff-i;
			}
		}
	}


	if(diff != 0)
	{
		result++;
		fprintf(logFile,"Error in Difference with integers: Result was %d instead of 0.\n",diff);
	}
	for (i=0;i<20;++i)
	{
		dpt*=dt;
	}
	dknown_sum = (1-dpt)/(1-dt);
#pragma omp parallel sections private(i) <ompts:check>reduction(+:dsum)</ompts:check><ompts:crosscheck></ompts:crosscheck>
	{
#pragma omp section
		{
			for (i=0;i<6;++i)
			{
				dsum += pow(dt,i);
			}
		}
#pragma omp section
		{
			for (i=6;i<12;++i)
			{
				dsum += pow(dt,i);
			}
		}
#pragma omp section
		{
			for (i=12;i<20;++i)
			{
				dsum += pow(dt,i);
			}
		}
	}


	if( fabs(dsum-dknown_sum) > rounding_error )
	{
		result++; 
		fprintf(logFile,"Error in sum with doubles: Result was %f instead of %f (Difference: %E)\n",dsum,dknown_sum, dsum-dknown_sum);
	}

	dpt=1;

	for (i=0;i<20;++i)
	{
		dpt*=dt;
	}
	fprintf(logFile,"\n");
	ddiff = (1-dpt)/(1-dt);
#pragma omp parallel sections private(i) <ompts:check>reduction(-:ddiff)</ompts:check><ompts:crosscheck></ompts:crosscheck>
	{
#pragma omp section
		{
			for (i=0;i<6;++i)
			{
				ddiff -= pow(dt,i);
			}
		}
#pragma omp section
		{
			for (i=6;i<12;++i)
			{
				ddiff -= pow(dt,i);
			}
		}
#pragma omp section
		{
			for (i=12;i<20;++i)
			{
				ddiff -= pow(dt,i);
			}
		}
	}

	if( fabs(ddiff) > rounding_error)
	{
		result++;
		fprintf(logFile,"Error in Difference with doubles: Result was %E instead of 0.0\n",ddiff);
	}

	known_product = 3628800;
#pragma omp parallel sections private(i) <ompts:check>reduction(*:product)</ompts:check><ompts:crosscheck></ompts:crosscheck>
	{
#pragma omp section
		{	
			for(i=1;i<3;i++)
			{
				product *= i;
			}
		}
#pragma omp section
		{
			for(i=3;i<7;i++)
			{
				product *= i;
			}
		}
#pragma omp section
		{
			for(i=7;i<11;i++)
			{
				product *= i;
			}
		}
	}


	if(known_product != product)
	{
		result++;
		fprintf(logFile,"Error in Product with integers: Result was %d instead of %d\n",product,known_product);
	}

	for(i=0;i<1000;i++)
	{
		logics[i]=1;
	}

#pragma omp parallel sections private(i) <ompts:check>reduction(&&:logic_and)</ompts:check><ompts:crosscheck></ompts:crosscheck>
	{
#pragma omp section
		{
			for (i=1;i<300;i++)
			{
				logic_and = (logic_and && logics[i]);
			}
		}
#pragma omp section
		{
			for (i=300;i<700;i++)
			{
				logic_and = (logic_and && logics[i]);
			}
		}
#pragma omp section
		{
			for (i=700;i<1000;i++)
			{
				logic_and = (logic_and && logics[i]);
			}
		}
	}

	if(!logic_and)
	{
		result++;
		fprintf(logFile,"Error in logic AND part 1\n");
	}

	logic_and = 1;
	logics[501] = 0;

#pragma omp parallel sections private(i) <ompts:check>reduction(&&:logic_and)</ompts:check><ompts:crosscheck></ompts:crosscheck>
	{
#pragma omp section
		{
			for (i=1;i<300;i++)
			{
				logic_and = (logic_and && logics[i]);
			}
		}
#pragma omp section
		{
			for (i=300;i<700;i++)
			{
				logic_and = (logic_and && logics[i]);
			}
		}
#pragma omp section
		{
			for (i=700;i<1000;i++)
			{
				logic_and = (logic_and && logics[i]);
			}
		}
	}

	if(logic_and)
	{
		result++;
		fprintf(logFile,"Error in logic AND part 2");
	}

	for(i=0;i<1000;i++)
	{
		logics[i]=0;
	}

#pragma omp parallel sections private(i) <ompts:check>reduction(||:logic_or)</ompts:check><ompts:crosscheck></ompts:crosscheck>
	{
#pragma omp section
		{
			for (i=1;i<300;i++)
			{
				logic_or = (logic_or || logics[i]);
			}
		}
#pragma omp section
		{
			for (i=300;i<700;i++)
			{
				logic_or = (logic_or || logics[i]);
			}
		}
#pragma omp section
		{
			for (i=700;i<1000;i++)
			{
				logic_or = (logic_or || logics[i]);
			}
		}
	}

	if(logic_or)
	{
		result++;
		fprintf(logFile,"Error in logic OR part 1\n");
	}

	logic_or = 0;
	logics[501]=1;

#pragma omp parallel sections private(i) <ompts:check>reduction(||:logic_or)</ompts:check><ompts:crosscheck></ompts:crosscheck>
	{
#pragma omp section
		{
			for (i=1;i<300;i++)
			{
				logic_or = (logic_or || logics[i]);
			}
		}
#pragma omp section
		{
			for (i=300;i<700;i++)
			{
				logic_or = (logic_or || logics[i]);
			}
		}
#pragma omp section
		{
			for (i=700;i<1000;i++)
			{
				logic_or = (logic_or || logics[i]);
			}
		}
	}

	if(!logic_or)
	{
		result++;
		fprintf(logFile,"Error in logic OR part 2\n");
	}

	for(i=0;i<1000;++i)
	{
		logics[i]=1;
	}

#pragma omp parallel sections private(i) <ompts:check>reduction(&:bit_and)</ompts:check><ompts:crosscheck></ompts:crosscheck>
	{
#pragma omp section
		{	
			for(i=0;i<300;++i)
			{
				bit_and = (bit_and & logics[i]);
			}
		}
#pragma omp section
		{	
			for(i=300;i<700;++i)
			{
				bit_and = (bit_and & logics[i]);
			}
		}
#pragma omp section
		{	
			for(i=700;i<1000;++i)
			{
				bit_and = (bit_and & logics[i]);
			}
		}
	}
	if(!bit_and)
	{
		result++;
		fprintf(logFile,"Error in BIT AND part 1\n");
	}

	bit_and = 1;
	logics[501]=0;

#pragma omp parallel sections private(i) <ompts:check>reduction(&:bit_and)</ompts:check><ompts:crosscheck></ompts:crosscheck>
	{
#pragma omp section
		{
			for(i=0;i<300;++i)
			{
				bit_and = bit_and & logics[i];
			}
		}
#pragma omp section
		{
			for(i=300;i<700;++i)
			{
				bit_and = bit_and & logics[i];
			}
		}
#pragma omp section
		{
			for(i=700;i<1000;++i)
			{
				bit_and = bit_and & logics[i];
			}
		}
	}
	if(bit_and)
	{
		result++;
		fprintf(logFile,"Error in BIT AND part 2");
	}

	for(i=0;i<1000;i++)
	{
		logics[i]=0;
	}

#pragma omp parallel sections private(i) <ompts:check>reduction(|:bit_or)</ompts:check><ompts:crosscheck></ompts:crosscheck>
	{
#pragma omp section
		{
			for(i=0;i<300;++i)
			{
				bit_or = bit_or | logics[i];
			}
		}
#pragma omp section
		{
			for(i=300;i<700;++i)
			{
				bit_or = bit_or | logics[i];
			}
		}
#pragma omp section
		{
			for(i=700;i<1000;++i)
			{
				bit_or = bit_or | logics[i];
			}
		}
	}
	if(bit_or)
	{
		result++;
		fprintf(logFile,"Error in BIT OR part 1\n");
	}
	bit_or = 0;
	logics[501]=1;

#pragma omp parallel sections private(i) <ompts:check>reduction(|:bit_or)</ompts:check><ompts:crosscheck></ompts:crosscheck>
	{
#pragma omp section
		{
			for(i=0;i<300;++i)
			{
				bit_or = bit_or | logics[i];
			}
		}
#pragma omp section
		{
			for(i=300;i<700;++i)
			{
				bit_or = bit_or | logics[i];
			}
		}
#pragma omp section
		{
			for(i=700;i<1000;++i)
			{
				bit_or = bit_or | logics[i];
			}
		}
	}
	if(!bit_or)
	{
		result++;
		fprintf(logFile,"Error in BIT OR part 2\n");
	}

	for(i=0;i<1000;i++)
	{
		logics[i]=0;
	}

#pragma omp parallel sections private(i) <ompts:check>reduction(^:exclusiv_bit_or)</ompts:check><ompts:crosscheck></ompts:crosscheck>
	{
#pragma omp section
		{	
			for(i=0;i<300;++i)
			{
				exclusiv_bit_or = exclusiv_bit_or ^ logics[i];
			}
		}
#pragma omp section
		{	
			for(i=300;i<700;++i)
			{
				exclusiv_bit_or = exclusiv_bit_or ^ logics[i];
			}
		}
#pragma omp section
		{	
			for(i=700;i<1000;++i)
			{
				exclusiv_bit_or = exclusiv_bit_or ^ logics[i];
			}
		}
	}
	if(exclusiv_bit_or)
	{
		result++;
		fprintf(logFile,"Error in EXCLUSIV BIT OR part 1\n");
	}

	exclusiv_bit_or = 0;
	logics[501]=1;

#pragma omp parallel sections private(i) <ompts:check>reduction(^:exclusiv_bit_or)</ompts:check><ompts:crosscheck></ompts:crosscheck>
	{
#pragma omp section
		{
			for(i=0;i<300;++i)
			{
				exclusiv_bit_or = exclusiv_bit_or ^ logics[i];
			}
		}
#pragma omp section
		{
			for(i=300;i<700;++i)
			{
				exclusiv_bit_or = exclusiv_bit_or ^ logics[i];
			}
		}
#pragma omp section
		{
			for(i=700;i<1000;++i)
			{
				exclusiv_bit_or = exclusiv_bit_or ^ logics[i];
			}
		}
	}
	if(!exclusiv_bit_or)
	{
		result++;
		fprintf(logFile,"Error in EXCLUSIV BIT OR part 2\n");
	}
</ompts:orphan>
	/*printf("\nResult:%d\n",result);*/
	return (result==0);
}
</ompts:testcode>
</ompts:test>
