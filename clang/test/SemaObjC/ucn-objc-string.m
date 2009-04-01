// RUN: clang %s -verify -fsyntax-only
@class NSString;
extern void NSLog(NSString *format, ...) __attribute__((format(__NSString__, 1, 2)));

int main() {
  NSLog(@"Hi…");
  NSLog(@"Exposé");
  // FIXME: the following 2 are still not working (will investigate).
  //NSLog(@"hello \u2192 \u2603 \u2190 world");
  //NSLog(@"\U00010400\U0001D12B");
  return 0;
}

