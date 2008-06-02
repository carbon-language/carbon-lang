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
	id objects[] = {[x METH]};
	return 0;
}


// rdar://5977581
void test3() {
  unsigned x[] = {[NSNumber METH2]+2};
}

