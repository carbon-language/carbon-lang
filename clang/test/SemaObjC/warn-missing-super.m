@protocol NSCopying @end

@interface NSObject <NSCopying>
- (void)dealloc;
@end

@implementation NSObject
- (void)dealloc {
  // Root class, shouldn't warn
}
@end

@interface Subclass1 : NSObject
- (void)dealloc;
@end

@implementation Subclass1
- (void)dealloc {
}
@end

@interface Subclass2 : NSObject
- (void)dealloc;
@end

@implementation Subclass2
- (void)dealloc {
  [super dealloc];  // Shouldn't warn
}
@end

// RUN: %clang_cc1 -fsyntax-only %s 2>&1 | FileCheck %s
// CHECK: warn-missing-super.m:19:1: warning: method possibly missing a [super dealloc] call
// CHECK: 1 warning generated.

// RUN: %clang_cc1 -fsyntax-only -triple x86_64-apple-darwin10 -fobjc-nonfragile-abi -fobjc-arc %s 2>&1 | FileCheck --check-prefix=CHECK-ARC %s
// CHECK-ARC: warn-missing-super.m:28:4: error: ARC forbids explicit message send of 'dealloc'
// CHECK-ARC: 1 error generated.
