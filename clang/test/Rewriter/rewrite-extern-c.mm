// RUN: %clang_cc1 -x objective-c++ -fblocks -rewrite-objc -fobjc-fragile-abi -o - %s
// radar 7546096

extern "C" {
        short foo() { } 
}
typedef unsigned char Boolean;

