// RUN: rm -rf %t
// RUN: %clang_cc1 -objcmt-migrate-property -mt-migrate-directory %t %s -x objective-c -fobjc-runtime-has-weak -fobjc-arc -fobjc-default-synthesize-properties -triple x86_64-apple-darwin11
// RUN: c-arcmt-test -mt-migrate-directory %t | arcmt-test -verify-transformed-files %s.result
// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -fsyntax-only -x objective-c -fobjc-runtime-has-weak -fobjc-arc -fobjc-default-synthesize-properties %s.result

typedef long NSInteger;
#define NS_ENUM(_type, _name) enum _name : _type _name; enum _name : _type

enum {
  blah,
  blarg
};
typedef NSInteger wibble;
