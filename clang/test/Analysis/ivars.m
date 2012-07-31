// RUN: %clang_cc1 -analyze -analyzer-checker=core,debug.ExprInspection -analyzer-store=region -fblocks -verify -Wno-objc-root-class %s

void clang_analyzer_eval(int);

@interface Root {
@public
  int uniqueID;
}

- (void)refreshID;
@end

void testInvalidation(Root *obj) {
  int savedID = obj->uniqueID;
  clang_analyzer_eval(savedID == obj->uniqueID); // expected-warning{{TRUE}}

  [obj refreshID];
  clang_analyzer_eval(savedID == obj->uniqueID); // expected-warning{{UNKNOWN}}
}


@interface Child : Root
@end

@implementation Child
- (void)testSuperInvalidation {
  int savedID = self->uniqueID;
  clang_analyzer_eval(savedID == self->uniqueID); // expected-warning{{TRUE}}

  [super refreshID];
  clang_analyzer_eval(savedID == self->uniqueID); // expected-warning{{UNKNOWN}}
}
@end
