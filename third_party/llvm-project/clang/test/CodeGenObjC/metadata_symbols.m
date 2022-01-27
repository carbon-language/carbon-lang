// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -emit-llvm -Wno-objc-root-class -fexceptions -fobjc-exceptions -o %t %s
// RUN: FileCheck -check-prefix=CHECK-X86_64 < %t %s
// RUN: FileCheck -check-prefix=CHECK-EHTYPE < %t %s

// We need exactly 3 of these.
// CHECK-EHTYPE: @"OBJC_EHTYPE_$_EH3"
// CHECK-EHTYPE: @"OBJC_EHTYPE_$_EH3"
// CHECK-EHTYPE: @"OBJC_EHTYPE_$_EH3"
// CHECK-EHTYPE-NOT: @"OBJC_EHTYPE_$_EH3"

// CHECK-X86_64: @"OBJC_CLASS_$_A" ={{.*}} global {{.*}}, section "__DATA, __objc_data", align 8
// CHECK-X86_64: @"OBJC_METACLASS_$_A" ={{.*}} global {{.*}}, section "__DATA, __objc_data", align 8
// CHECK-X86_64: @OBJC_CLASS_NAME_ = {{.*}}, section "__TEXT,__objc_classname,cstring_literals", align 1
// CHECK-X86_64: @"OBJC_EHTYPE_$_EH1" = weak{{.*}} global {{.*}}, align 8
// CHECK-X86_64: @"OBJC_EHTYPE_$_EH2" = external global
// CHECK-X86_64: @"OBJC_EHTYPE_$_EH3" ={{.*}} global {{.*}}, section "__DATA,__objc_const", align 8
// CHECK-X86_64: @"OBJC_LABEL_CLASS_$" = private global {{.*}}, section "__DATA,__objc_classlist,regular,no_dead_strip", align 8
// CHECK-X86_64: define internal void @"\01-[A im0]"
// CHECK-X86_64: define internal void @"\01-[A(Cat) im1]"

// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -Wno-objc-root-class -fobjc-exceptions -fvisibility hidden -emit-llvm -o %t %s
// RUN: FileCheck -check-prefix=CHECK-X86_64-HIDDEN < %t %s

// CHECK-X86_64-HIDDEN: @"OBJC_CLASS_$_A" = hidden global {{.*}}, section "__DATA, __objc_data", align 8
// CHECK-X86_64-HIDDEN: @"OBJC_METACLASS_$_A" = hidden global {{.*}}, section "__DATA, __objc_data", align 8
// CHECK-X86_64-HIDDEN: @"OBJC_EHTYPE_$_EH1" = weak hidden global
// CHECK-X86_64-HIDDEN: @"OBJC_EHTYPE_$_EH2" = external global
// CHECK-X86_64-HIDDEN: @"OBJC_EHTYPE_$_EH3" = hidden global {{.*}}, section "__DATA,__objc_const", align 8
// CHECK-X86_64-HIDDEN: define internal void @"\01-[A im0]"
// CHECK-X86_64-HIDDEN: define internal void @"\01-[A(Cat) im1]"

// RUN: %clang_cc1 -triple armv6-apple-darwin10 -target-abi apcs-gnu -Wno-objc-root-class -fobjc-exceptions -emit-llvm -o %t %s
// RUN: FileCheck -check-prefix=CHECK-ARMV6 < %t %s

// CHECK-ARMV6: @"OBJC_CLASS_$_A" ={{.*}} global {{.*}}, section "__DATA, __objc_data", align 4
// CHECK-ARMV6: @"OBJC_METACLASS_$_A" ={{.*}} global {{.*}}, section "__DATA, __objc_data", align 4
// CHECK-ARMV6: @OBJC_CLASS_NAME_ = {{.*}}, section "__TEXT,__objc_classname,cstring_literals", align 1
// CHECK-ARMV6: @"OBJC_EHTYPE_$_EH1" = weak{{.*}} global {{.*}}, align 4
// CHECK-ARMV6: @"OBJC_EHTYPE_$_EH2" = external global
// CHECK-ARMV6: @"OBJC_EHTYPE_$_EH3" ={{.*}} global {{.*}}, section "__DATA,__objc_const", align 4
// CHECK-ARMV6: @"OBJC_LABEL_CLASS_$" = private global {{.*}}, section "__DATA,__objc_classlist,regular,no_dead_strip", align 4
// CHECK-ARMV6: define internal void @"\01-[A im0]"
// CHECK-ARMV6: define internal void @"\01-[A(Cat) im1]"

@interface A
@end

@implementation A
-(void) im0 {
}
@end

@implementation A (Cat)
-(void) im1 {
}
@end

@interface EH1
@end

__attribute__((__objc_exception__))
@interface EH2
@end

__attribute__((__objc_exception__))
@interface EH3
@end

void f1();

void f0(id x) {
  @try {
    f1();
  } @catch (EH1 *x) {
  } @catch (EH2 *x) {
  } @catch (EH3 *x) {
  }
}

@implementation EH3
@end
