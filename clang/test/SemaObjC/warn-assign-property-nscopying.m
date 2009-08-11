// RUN: clang-cc  -fobjc-gc -fsyntax-only -verify %s

@protocol NSCopying @end

@interface NSObject <NSCopying>
@end

@interface NSDictionary : NSObject
@end

@interface INTF
  @property NSDictionary* undoAction;  // expected-warning {{no 'assign', 'retain', or 'copy' attribute is specified - 'assign' is assumed}} // expected-warning {{default assign attribute on property 'undoAction' which implements NSCopying protocol is not appropriate with}}
@end

