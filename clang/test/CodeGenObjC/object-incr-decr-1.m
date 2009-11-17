// RUN: clang-cc -triple i386-apple-darwin9 -emit-llvm %s -o %t

@interface Foo 
{
	double d1,d3,d4;
}
@end

Foo* foo()
{
  Foo *f;
  
  // Both of these crash clang nicely
  ++f;
  --f;
 f--;
 f++;
 return f;
}
