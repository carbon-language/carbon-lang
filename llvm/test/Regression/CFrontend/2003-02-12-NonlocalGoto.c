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





