// RUN: %clang_cc1 -analyze -analyzer-checker=core,osx.cocoa.Loops,debug.ExprInspection -verify %s

void clang_analyzer_eval(int);

#define nil ((id)0)

@protocol NSFastEnumeration
- (int)countByEnumeratingWithState:(void *)state objects:(id *)objects count:(unsigned)count;
@end

@interface NSObject
+ (instancetype)testObject;
@end

@interface NSEnumerator <NSFastEnumeration>
@end

@interface NSArray : NSObject <NSFastEnumeration>
- (NSEnumerator *)objectEnumerator;
@end

@interface NSDictionary : NSObject <NSFastEnumeration>
@end

@interface NSMutableDictionary : NSDictionary
@end

@interface NSSet : NSObject <NSFastEnumeration>
@end

@interface NSPointerArray : NSObject <NSFastEnumeration>
@end

void test() {
  id x;
  for (x in [NSArray testObject])
    clang_analyzer_eval(x != nil); // expected-warning{{TRUE}}

  for (x in [NSMutableDictionary testObject])
    clang_analyzer_eval(x != nil); // expected-warning{{TRUE}}

  for (x in [NSSet testObject])
    clang_analyzer_eval(x != nil); // expected-warning{{TRUE}}

  for (x in [[NSArray testObject] objectEnumerator])
    clang_analyzer_eval(x != nil); // expected-warning{{TRUE}}

  for (x in [NSPointerArray testObject])
    clang_analyzer_eval(x != nil); // expected-warning{{UNKNOWN}}
}

void testWithVarInFor() {
  for (id x in [NSArray testObject])
    clang_analyzer_eval(x != nil); // expected-warning{{TRUE}}
  for (id x in [NSPointerArray testObject])
    clang_analyzer_eval(x != nil); // expected-warning{{UNKNOWN}}
}

void testNonNil(id a, id b) {
  clang_analyzer_eval(a != nil); // expected-warning{{UNKNOWN}}
  for (id x in a)
    clang_analyzer_eval(a != nil); // expected-warning{{TRUE}}

  if (b != nil)
    return;
  for (id x in b)
    *(volatile int *)0 = 1; // no-warning
  clang_analyzer_eval(b != nil); // expected-warning{{FALSE}}
}

