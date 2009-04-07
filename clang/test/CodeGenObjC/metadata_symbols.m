// RUN: clang-cc -triple x86_64-apple-darwin9 -emit-llvm -o %t %s &&

// RUN: grep '@"OBJC_METACLASS_$_A" = global .*section "__DATA, __objc_data", align 8' %t && 
// RUN: grep '@"OBJC_CLASS_$_A" = global .*section "__DATA, __objc_data", align 8' %t &&
// RUN: grep '@"OBJC_EHTYPE_$_EH" = weak global .*section "__DATA,__datacoal_nt,coalesced"' %t &&
// RUN: grep -F 'define internal void @"\01-[A im0]"' %t &&
// FIXME: Should include category name.
// RUN: grep -F 'define internal void @"\01-[A im1]"' %t &&

// RUN: clang-cc -fvisibility=hidden -triple x86_64-apple-darwin9 -emit-llvm -o %t %s &&

// FIXME: This is wrong, should be hidden
// RUN: grep '@"OBJC_METACLASS_$_A" = global .*section "__DATA, __objc_data", align 8' %t && 
// FIXME: This is wrong, should be hidden
// RUN: grep '@"OBJC_CLASS_$_A" = global .*section "__DATA, __objc_data", align 8' %t &&
// RUN: grep '@"OBJC_EHTYPE_$_EH" = weak hidden global .*section "__DATA,__datacoal_nt,coalesced"' %t &&
// RUN: grep -F 'define internal void @"\01-[A im0]"' %t &&
// FIXME: Should include category name.
// RUN: grep -F 'define internal void @"\01-[A im1]"' %t &&

// RUN: true

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

@interface EH
@end

void f1();

void f0(id x) {
  @try {
    f1();
  } @catch (EH *x) {
  }
}
