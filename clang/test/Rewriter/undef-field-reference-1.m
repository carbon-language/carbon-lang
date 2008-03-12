// RUN: clang -rewrite-test %s 

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


