/* -*-c++-*- */

#include <stdio.h>
#include <stdlib.h>
#include "/home/vadve/vadve/Research/DynOpt/LLVM/llvm/include/llvm/Support/MathExtras.h"

inline void
testPow(int C, bool isPow)
{
  unsigned pow = 0;
  bool testIsPow = IsPowerOf2(C, pow);
  if (isPow != testIsPow)
    printf("ERROR: IsPowerOf2() says \t%d %s a power of 2 = %d\n",
	   C, (isPow? "IS" : "IS NOT"), pow);

#undef PRINT_CORRECT_RESULTS
#ifdef PRINT_CORRECT_RESULTS
  else
    printf("CORRECT: IsPowerOf2() says \t%d %s a power of 2 = %d\n",
	   C, (isPow? "IS" : "IS NOT"), pow);
#endif PRINT_CORRECT_RESULTS
}

int
main(int argc, char** argv)
{
  unsigned L = (argc > 1)? atoi(argv[1]) : 16;
  unsigned C = 1;
  
  testPow(0, false);
  
  for (unsigned i = 1; i < L; i++, C = C << 1)
    {
      testPow(C, true);
      testPow(-C, true);
      for (unsigned j = C+1; j < (C << 1); j++)
	{
	  testPow(j, false);
	  testPow(-j, false);
	}
    }
  
  return 0;
}


