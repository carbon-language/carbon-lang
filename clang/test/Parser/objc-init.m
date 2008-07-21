// RUN: clang -fsyntax-only -verify %s
// rdar://5707001

@interface NSNumber;
- () METH;
- (unsigned) METH2;
@end

void test1() {
	id objects[] = {[NSNumber METH]};
}

void test2(NSNumber x) {
	id objects[] = {[x METH]}; // expected-error {{bad receiver type}}
}

void test3(NSNumber *x) {
	id objects[] = {[x METH]};
}


// rdar://5977581
void test4() {
  unsigned x[] = {[NSNumber METH2]+2};
}

