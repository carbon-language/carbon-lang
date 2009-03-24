// RUN: clang-cc -pedantic -fsyntax-only -verify %s

@protocol MyProto1 
@end

@protocol MyProto2 
@end

@interface INTF @end

INTF <MyProto1> * Func(INTF <MyProto1, MyProto2> *p2)
{
	return p2;
}


INTF <MyProto1> * Func1(INTF <MyProto1, MyProto2> *p2)
{
	return p2;
}

INTF <MyProto1, MyProto2> * Func2(INTF <MyProto1> *p2)
{
	Func(p2);	// expected-warning {{incompatible pointer types passing 'INTF<MyProto1> *', expected 'INTF<MyProto1,MyProto2> *}}
	return p2;	// expected-warning {{incompatible pointer types returning 'INTF<MyProto1> *', expected 'INTF<MyProto1,MyProto2> *}}
}



INTF <MyProto1> * Func3(INTF <MyProto2> *p2)
{
	return p2;	// expected-warning {{incompatible pointer types returning 'INTF<MyProto2> *', expected 'INTF<MyProto1> *}}
}


INTF <MyProto1, MyProto2> * Func4(INTF <MyProto2, MyProto1> *p2)
{
	return p2;
}

