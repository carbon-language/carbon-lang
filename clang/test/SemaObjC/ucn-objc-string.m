// RUN: clang %s -verify -fsyntax-only
@class NSString;
extern void NSLog(NSString *format, ...) __attribute__((format(__NSString__, 1, 2)));

int main() {
  NSLog(@"Hi…");
  NSLog(@"Exposé");
  NSLog(@"\U00010400\U0001D12B");
  NSLog(@"hello \u2192 \u2603 \u2190 world");
  NSLog(@"hello → ☃ ← world");
  return 0;
}

