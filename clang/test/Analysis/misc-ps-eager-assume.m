// RUN: clang -analyze -checker-cfref --analyzer-store=region -analyzer-constraints=range --verify -fblocks %s -analyzer-eagerly-assume

// Delta-reduced header stuff (needed for test cases).
typedef signed char BOOL;
typedef unsigned int NSUInteger;
typedef struct _NSZone NSZone;
@class NSInvocation, NSMethodSignature, NSCoder, NSString, NSEnumerator;
@protocol NSObject  - (BOOL)isEqual:(id)object;
- (oneway void)release;
@end  @protocol NSCopying  - (id)copyWithZone:(NSZone *)zone;
@end  @protocol NSMutableCopying  - (id)mutableCopyWithZone:(NSZone *)zone;
@end  @protocol NSCoding  - (void)encodeWithCoder:(NSCoder *)aCoder;
@end    @interface NSObject <NSObject> {}
+ (id)alloc;
@end  typedef struct {}
NSFastEnumerationState;
@protocol NSFastEnumeration  - (NSUInteger)countByEnumeratingWithState:(NSFastEnumerationState *)state objects:(id *)stackbuf count:(NSUInteger)len;
@end      @interface NSArray : NSObject <NSCopying, NSMutableCopying, NSCoding, NSFastEnumeration>  - (NSUInteger)count;
@end    @interface NSMutableArray : NSArray  - (void)addObject:(id)anObject;
- (BOOL)isEqualToString:(NSString *)aString;
@end        @interface NSAutoreleasePool : NSObject {}
- (void)drain;
- (id)init;
@end

// This test case tests that (x != 0) is eagerly evaluated before stored to
// 'y'.  This test case complements recoverCastedSymbol (see below) because
// the symbolic expression is stored to 'y' (which is a short instead of an
// int).  recoverCastedSymbol() only recovers path-sensitivity when the
// symbolic expression is literally the branch condition.
//
void handle_assign_of_condition(int x) {
  // The cast to 'short' causes us to lose symbolic constraint.
  short y = (x != 0);
  char *p = 0;
  if (y) {
    // This should be infeasible.
    if (!(x != 0)) {
      *p = 1;  // no-warning
    }
  }
}

// From <rdar://problem/6619921>
//
// In this test case, 'needsAnArray' is a signed char.  The analyzer tracks
// a symbolic value for this variable, but in the branch condition it is
// promoted to 'int'.  Currently the analyzer doesn't reason well about
// promotions of symbolic values, so this test case tests the logic in
// 'recoverCastedSymbol()' (GRExprEngine.cpp) to test that we recover
// path-sensitivity and use the symbol for 'needsAnArray' in the branch
// condition.
//
void handle_symbolic_cast_in_condition(void) {
  NSAutoreleasePool* pool = [[NSAutoreleasePool alloc] init];

  BOOL needsAnArray = [@"aString" isEqualToString:@"anotherString"];
  NSMutableArray* array = needsAnArray ? [[NSMutableArray alloc] init] : 0;
  if(needsAnArray)
    [array release];

  [pool drain];
}
