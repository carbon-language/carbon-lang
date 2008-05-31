// RUN: clang -fsyntax-only -verify -pedantic %s
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
