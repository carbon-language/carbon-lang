// RUN: %llvmgcc -S %s -o - | grep 'trunc.*to bool'
// RUN: %llvmgcc -S %s -o - | llvm-as | opt -instcombine | llc -march=x86 | grep 'test.*1'
int
main ( int argc, char** argv)
{
  int i;
  int result = 1;
  for (i = 2; i <= 3; i++)
    {
      if ((i & 1) == 0)
	{
	    result = result + 17;
	}
    }
  return result;
}
