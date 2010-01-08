// RUN: %clang_cc1 -rewrite-objc %s -o -

@class NSArray;
int main() {
	NSArray *foo;
	for (Class c in foo) { }
}
