// RUN: clang-cc -ast-print %s

@interface current 
{
@public
  int ivar;
  int ivar1;
  int ivar2;
}
@end

current *pc;

int foo()
{
	return pc->ivar2 + (*pc).ivar + pc->ivar1;
}
