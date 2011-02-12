// RUN: %clang_cc1 -analyze -analyzer-store=region -analyzer-constraints=range -fblocks -analyzer-opt-analyze-nested-blocks -analyzer-check-objc-mem -analyzer-check-idempotent-operations -verify %s

typedef signed char BOOL;
typedef unsigned long NSUInteger;
typedef struct _NSZone NSZone;
@protocol NSObject  - (BOOL)isEqual:(id)object;
@end    @interface NSObject <NSObject> {
}
@end


// <rdar://problem/8725041> - Don't flag idempotent operation warnings when
// a method may invalidate an instance variable.
@interface Rdar8725041 : NSObject {
  id _attribute;
}
  - (void) method2;
@end

@implementation Rdar8725041
- (BOOL) method1 {
  BOOL needsUpdate = (BOOL)0;
  id oldAttribute = _attribute;
  [self method2];
  needsUpdate |= (_attribute != oldAttribute); // no-warning
  return needsUpdate;
}

- (void) method2
{
  _attribute = ((void*)0);
}
@end

