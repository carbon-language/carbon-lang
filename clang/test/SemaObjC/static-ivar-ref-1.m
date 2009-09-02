// RUN: clang-cc -triple i386-unknown-unknown -ast-print %s

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
        // FIXME. This should be OK in nonfragile-abi as well.
	return pc->ivar2 + (*pc).ivar + pc->ivar1;
}
