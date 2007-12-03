// RUN: clang -rewrite-test %s | clang

#include <objc/objc.h>

@interface MyDerived 
{
@public
	int IVAR;
}
@end

MyDerived *pd;
int main() {
	return pd->IVAR;
}


