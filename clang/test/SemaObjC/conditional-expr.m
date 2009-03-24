// RUN: clang-cc -fsyntax-only -verify -pedantic %s
@protocol NSObject
@end

@protocol DTOutputStreams <NSObject>
@end

@interface DTFilterOutputStream <DTOutputStreams>
- nextOutputStream;
@end

@implementation DTFilterOutputStream
- (id)initWithNextOutputStream:(id <DTOutputStreams>) outputStream {
  id <DTOutputStreams> nextOutputStream = [self nextOutputStream];
  self = nextOutputStream;
  return nextOutputStream ? nextOutputStream : self;
}
- nextOutputStream {
  return self;
}
@end

@interface DTFilterOutputStream2
- nextOutputStream;
@end

@implementation DTFilterOutputStream2 // expected-warning {{incomplete implementation}} expected-warning {{method definition for 'nextOutputStream' not found}}
- (id)initWithNextOutputStream:(id <DTOutputStreams>) outputStream {
  id <DTOutputStreams> nextOutputStream = [self nextOutputStream];
  // GCC warns about both of these.
  self = nextOutputStream; // expected-warning {{incompatible type assigning 'id<DTOutputStreams>', expected 'DTFilterOutputStream2 *'}}
  return nextOutputStream ? nextOutputStream : self;
}
@end

// No @interface declaration for DTFilterOutputStream3
@implementation DTFilterOutputStream3 // expected-warning {{cannot find interface declaration for 'DTFilterOutputStream3'}}
- (id)initWithNextOutputStream:(id <DTOutputStreams>) outputStream {
  id <DTOutputStreams> nextOutputStream = [self nextOutputStream]; // expected-warning {{method '-nextOutputStream' not found (return type defaults to 'id')}}
  // GCC warns about both of these as well (no errors).
  self = nextOutputStream; // expected-warning {{incompatible type assigning 'id<DTOutputStreams>', expected 'DTFilterOutputStream3 *'}}
  return nextOutputStream ? nextOutputStream : self;
}
@end
