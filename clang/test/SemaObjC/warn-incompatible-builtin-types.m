// RUN: %clang_cc1 -fsyntax-only -verify %s
// rdar 7634850

@interface Foo
- (void)foo:(Class)class;
@end

void FUNC() {
    Class c, c1;
    SEL s1, s2;
    id i, i1;
    Foo *f;
    [f foo:f];	// expected-warning {{incompatible pointer types sending 'Foo *', expected 'Class'}}
    c = f;	// expected-warning {{incompatible pointer types assigning 'Foo *', expected 'Class'}}

    c = i;

    i = c;

    c = c1;

    i = i1;

    s1 = i;	// expected-warning {{incompatible pointer types assigning 'id', expected 'SEL'}}
    i = s1;	// expected-warning {{incompatible pointer types assigning 'SEL', expected 'id'}}

    s1 = s2;

    s1 = c;	// expected-warning {{incompatible pointer types assigning 'Class', expected 'SEL'}}

    c = s1;	// expected-warning {{incompatible pointer types assigning 'SEL', expected 'Class'}}

    f = i;

    f = c;	// expected-warning {{incompatible pointer types assigning 'Class', expected 'Foo *'}}

    f = s1;	// expected-warning {{incompatible pointer types assigning 'SEL', expected 'Foo *'}}

    i = f;

    s1 = f; 	// expected-warning {{incompatible pointer types assigning 'Foo *', expected 'SEL'}}
}
