// RUN: %clang_cc1 -fsyntax-only -verify %s

@interface Data
- (unsigned)length;
- (void)getData:(void*)buffer;
@end

void test(Data *d) {
  char buffer[[d length]];
  [d getData:buffer];
}

