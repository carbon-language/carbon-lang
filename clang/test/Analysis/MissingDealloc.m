// RUN: clang -warn-objc-missing-dealloc '-DIBOutlet=__attribute__((iboutlet))' %s --verify
typedef signed char BOOL;
@protocol NSObject  - (BOOL)isEqual:(id)object; @end
@interface NSObject <NSObject> {}
- (void)dealloc;
@end

// <rdar://problem/6380411>: 'myproperty' has kind 'assign' and thus the
//  assignment through the setter does not perform a release.

@interface MyObject : NSObject {
  id _myproperty;  
}
@property(assign) id myproperty;
@end

@implementation MyObject
@synthesize myproperty=_myproperty; // no-warning
- (void)dealloc {
  self.myproperty = 0;
  [super dealloc]; 
}
@end
