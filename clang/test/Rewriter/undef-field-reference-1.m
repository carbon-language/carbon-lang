// RUN: clang -cc1 -rewrite-objc %s -o -

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


