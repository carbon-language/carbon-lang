// RUN: %clang_cc1 -triple i386-unknown-unknown -ast-print %s
// RUN: %clang_cc1 -triple x86_64-apple-darwin10  -ast-print %s

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
