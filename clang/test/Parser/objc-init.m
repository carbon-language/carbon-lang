// RUN: clang -fsyntax-only -verify %s
// rdar://5707001

@interface NSNumber;
- () METH;
@end

void test1() {
	id objects[] = {[NSNumber METH]};
}

void test2(NSNumber x) {
	id objects[] = {[x METH]};
	return 0;
}


