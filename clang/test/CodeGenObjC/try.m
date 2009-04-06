// RUN: clang %s -S -o - -mtriple=i686-apple-darwin9 &&
// RUN: clang %s -S -o - -mtriple=x86_64-apple-darwin9

// rdar://6757213 - Don't crash if the internal proto for
// __objc_personality_v0 mismatches with an actual one.
void __objc_personality_v0() { }
void test1(void) {
  @try { } @catch (...) { }
}
