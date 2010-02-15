// RUN: %clang_cc1 -fsyntax-only -verify %s

@interface MyClass {

};
@property unsigned char bufferedUTF8Bytes[4];  // expected-error {{property cannot have array or function type}}
@property unsigned char bufferedUTFBytes:1;    // expected-error {{property name cannot be a bitfield}}
@property(nonatomic, retain, setter=ab_setDefaultToolbarItems) MyClass *ab_defaultToolbarItems; // expected-error {{method name referenced in property setter attribute must end with ':'}}
@end

@implementation MyClass
@dynamic ab_defaultToolbarItems;
@end

