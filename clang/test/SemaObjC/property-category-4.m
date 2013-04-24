// RUN: %clang_cc1 -fsyntax-only -verify -Wno-objc-root-class %s

@interface IDELogNavigator
{
  id selectedObjects;
}
@end

@interface IDELogNavigator (CAT)
  @property (readwrite, retain) id selectedObjects; // expected-note {{property declared here}}
  @property (readwrite, retain) id d_selectedObjects; // expected-note {{property declared here}}
@end

@implementation IDELogNavigator 
@synthesize selectedObjects = _selectedObjects; // expected-error {{property declared in category 'CAT' cannot be implemented in class implementation}}
@dynamic d_selectedObjects; // expected-error {{property declared in category 'CAT' cannot be implemented in class implementation}}
@end


// rdar://13713098
// Test1
@interface NSArray 
- (int)count;
@end

@protocol MyCountable
@property  (readonly) int count;
@end


@interface NSArray(Additions) <MyCountable>
@end

@implementation NSArray(Additions)
@end

// Test2
@protocol NSProtocol
- (int)count;
@end

@interface NSArray1 <NSProtocol>
@end

@interface NSArray1(Additions) <MyCountable>
@end

@implementation NSArray1(Additions)
@end

// Test3
@interface Super <NSProtocol>
@end

@interface NSArray2 : Super @end

@interface NSArray2(Additions) <MyCountable>
@end

@implementation NSArray2(Additions)
@end

// Test3
@interface Super1 <NSProtocol>
@property  (readonly) int count;
@end

@protocol MyCountable1
@end

@interface NSArray3 : Super1 <MyCountable1>
@end

@implementation NSArray3
@end

// Test4
@interface I
@property int d1;
@end

@interface I(CAT)
@property int d1;
@end

@implementation I(CAT)
@end
