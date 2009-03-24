// RUN: clang-cc -fsyntax-only -verify %s

@interface MyClass {

};
@property unsigned char bufferedUTF8Bytes[4];  // expected-error {{property cannot have array or function type}}
@property unsigned char bufferedUTFBytes:1;    // expected-error {{property name cannot be a bitfield}}
@end

@implementation MyClass
@end

