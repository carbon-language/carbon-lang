// RUN: %clang -arch i386 -fms-extensions -rewrite-objc %s -o %t-rw.cpp
// RUN: FileCheck %s < %t-rw.cpp
// RUN: %clang -arch x86_64 -fms-extensions -rewrite-objc %s -o %t-rw-64bit.cpp
// RUN: FileCheck -check-prefix=LP64 %s < %t-rw-64bit.cpp
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
// CHECK-NEXT: 0,
// CHECK-NEST: "MYINTF",

// CHECK-LP64: static struct _class_ro_t _OBJC_CLASS_RO_$_MYINTF
// CHECK-LP64-NEXT: 0, 0, 0,
// CHECK-LP64-NEXT: (unsigned int)0,
// CHECK-LP64-NEXT: 0,
// CHECK-LP64-NEST: "MYINTF",
