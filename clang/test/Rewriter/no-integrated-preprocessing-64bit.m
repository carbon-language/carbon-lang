// RUN: %clang -arch x86_64 -fms-extensions -rewrite-objc %s -o %t-rw-64bit.cpp
// RUN: FileCheck %s < %t-rw-64bit.cpp
// XFAIL: mingw32,win32
// rdar://12189793

#ifdef __cplusplus

void *sel_registerName(const char *);

@interface Root @end

@interface MYINTF : Root
@end

#endif

@implementation MYINTF 
- (id) MYMETH { return [self MYMETH]; }
@end

int main() {
}

// CHECK: static struct _class_ro_t _OBJC_CLASS_RO_$_MYINTF
// CHECK-NEXT: 0, 0, 0,
// CHECK-NEXT: (unsigned int)0,
// CHECK-NEXT: 0,
// CHECK-NEXT: "MYINTF",
