// Without PCH
// RUN: %clang_cc1 -fsyntax-only -verify -include %s -include %s %s

// With PCH
// RUN: %clang_cc1 -fsyntax-only -verify %s -chain-include %s -chain-include %s

// expected-no-diagnostics

#ifndef HEADER1
#define HEADER1
//===----------------------------------------------------------------------===//
// Primary header

@interface NSObject
- (id)init;
- (void)finalize;
@end

@interface NSObject (Properties)
@property (readonly,nonatomic) int intProp;
@end

//===----------------------------------------------------------------------===//
#elif !defined(HEADER2)
#define HEADER2
#if !defined(HEADER1)
#error Header inclusion order messed up
#endif

//===----------------------------------------------------------------------===//
// Dependent header

@interface MyClass : NSObject
+(void)meth;
@end

@interface NSObject(ObjExt)
-(void)extMeth;
@end

@interface NSObject ()
@property (readwrite,nonatomic) int intProp;
@end

@class NSObject;

//===----------------------------------------------------------------------===//
#else
//===----------------------------------------------------------------------===//

@implementation MyClass
+(void)meth {}
-(void)finalize {
  [super finalize];
}
@end

void test(NSObject *o) {
  [o extMeth];

  // Make sure the property is treated as read-write.
  o.intProp = 17;
}

//===----------------------------------------------------------------------===//
#endif
