// RUN: %clang_cc1 -rewrite-objc -fobjc-fragile-abi  %s -o -

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


