// RUN: clang -fsyntax-only -verify %s
// rdar://5707001

@interface NSNumber;
- () METH;

@end

int main() {
	id objects[] = {[NSNumber METH]};
	return 0;
}

