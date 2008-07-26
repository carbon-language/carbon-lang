// RUN: clang -pedantic -fsyntax-only -verify %s
typedef signed char BOOL;

@class NSInvocation, NSMethodSignature, NSCoder, NSString, NSEnumerator;

@protocol NSObject
- (BOOL)isEqual:(id)object;
@end

@protocol NSCoding
- (void)encodeWithCoder:(NSCoder *)aCoder;
@end

@interface NSObject <NSObject> {}
@end

typedef float CGFloat;

@interface NSResponder : NSObject <NSCoding> {}
@end

@protocol XCSelectionSource;

@interface XCSelection : NSResponder {}
- (NSObject <XCSelectionSource> *) source;
@end

extern NSString * const XCActiveSelectionLevel;

@interface XCActionManager : NSResponder {}
+defaultActionManager;
-selectionAtLevel:(NSString *const)s;
@end

@implementation XDMenuItemsManager // expected-warning {{cannot find interface declaration for 'XDMenuItemsManager'}}
+ (void)initialize {
  id<XCSelectionSource, NSObject> source = 
    [[[XCActionManager defaultActionManager] selectionAtLevel:XCActiveSelectionLevel] source];
}
@end
