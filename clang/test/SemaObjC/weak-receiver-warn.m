// RUN: %clang_cc1 -fsyntax-only -fobjc-runtime-has-weak -fobjc-arc -fblocks -Wreceiver-is-weak -verify %s
// rdar://10225276

@interface Test0
- (void) setBlock: (void(^)(void)) block;
- (void) addBlock: (void(^)(void)) block;
- (void) actNow;
@end

void test0(Test0 *x) {
  __weak Test0 *weakx = x;
  [x addBlock: ^{ [weakx actNow]; }]; // expected-warning {{weak receiver may be unpredictably null in ARC mode}}
  [x setBlock: ^{ [weakx actNow]; }]; // expected-warning {{weak receiver may be unpredictably null in ARC mode}}
  x.block = ^{ [weakx actNow]; }; // expected-warning {{weak receiver may be unpredictably null in ARC mode}}

  [weakx addBlock: ^{ [x actNow]; }]; // expected-warning {{weak receiver may be unpredictably null in ARC mode}}
  [weakx setBlock: ^{ [x actNow]; }]; // expected-warning {{weak receiver may be unpredictably null in ARC mode}}
  weakx.block = ^{ [x actNow]; };
}
