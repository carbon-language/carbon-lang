// RUN: %clang_cc1 -rewrite-objc -fobjc-fragile-abi  %s -o -

@class NSArray;
int main() {
	NSArray *foo;
	for (Class c in foo) { }
}
