// Objective-C recovery
// RUN: cp %s %t
// RUN: %clang_cc1 -fixit -x objective-c %t || true
// RUN: %clang_cc1 -fsyntax-only -Werror -x objective-c %t

// Objective-C++ recovery
// RUN: cp %s %t
// RUN: %clang_cc1 -fixit -x objective-c++ %t || true
// RUN: %clang_cc1 -fsyntax-only -Werror -x objective-c++ %t
// rdar://9603056

@interface NSArray
+ (id) arrayWithObjects;
@end

int main() {
  	NSArray pluginNames = [NSArray arrayWithObjects];
}
