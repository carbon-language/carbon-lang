// RUN: %clang -c %s -o %t && libtool -static -o libcodegentest.a %t && %clang -bundle -o codegentestbundle -L. -lcodegentest -Wl,-ObjC && nm codegentestbundle | grep -F '[A(foo) foo_myStuff]'
// XFAIL: *
// XTARGET: darwin9
// PR7431
@interface A
@end
@interface A(foo)
- (void)foo_myStuff;
@end
@implementation A(foo)
- (void)foo_myStuff {
}
@end
