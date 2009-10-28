// RUN: clang-cc -triple x86_64-apple-darwin10 -analyze -warn-objc-unused-ivars %s -verify

//===--- BEGIN: Delta-debugging reduced headers. --------------------------===//

@protocol NSObject
- (id)retain;
- (oneway void)release;
@end
@interface NSObject <NSObject> {}
- (id)init;
+ (id)alloc;
@end

//===--- END: Delta-debugging reduced headers. ----------------------------===//

// This test case tests the basic functionality of the unused ivar test.
@interface TestA {
@private
  int x; // expected-warning {{Instance variable 'x' in class 'TestA' is never used}}
}
@end
@implementation TestA @end

// This test case tests whether the unused ivar check handles blocks that
// reference an instance variable. (<rdar://problem/7075531>)
@interface TestB : NSObject {
@private
  id _ivar; // no-warning
}
@property (readwrite,retain) id ivar;
@end

@implementation TestB
- (id)ivar {
  __attribute__((__blocks__(byref))) id value = ((void*)0);
  void (^b)() = ^{ value = _ivar; };
  b();
  return value;
}

- (void)setIvar:(id)newValue {
  void (^b)() = ^{ [_ivar release]; _ivar = [newValue retain]; };
  b();
}
@end

//===----------------------------------------------------------------------===//
// <rdar://problem/6260004> Detect that ivar is in use, if used in category 
//  in the same file as the implementation
//===----------------------------------------------------------------------===//

@protocol Protocol6260004
- (id) getId;
@end

@interface RDar6260004 {
@private
  id x; // no-warning
}
@end
@implementation RDar6260004 @end
@implementation RDar6260004 (Protocol6260004)
- (id) getId {
  return x;
}
@end

