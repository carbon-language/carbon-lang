
/* It is unlikely that LLVM will ever support nested functions, but if it does,
   here is a testcase. */

main()
{
  __label__ l;

  void*x()
  {
    goto l;
  }

  x();
  abort();
  return;
l:
  exit(0);
}





