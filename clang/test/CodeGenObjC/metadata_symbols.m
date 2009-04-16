// RUN: clang-cc -triple x86_64-apple-darwin9 -emit-llvm -o %t %s &&

// RUN: grep '@"OBJC_METACLASS_$_A" = global .*section "__DATA, __objc_data", align 8' %t && 
// RUN: grep '@"OBJC_CLASS_$_A" = global .*section "__DATA, __objc_data", align 8' %t &&
// RUN: grep '@"OBJC_EHTYPE_$_EH1" = weak global .*section "__DATA,__datacoal_nt,coalesced", align 8' %t &&
// RUN: grep '@"OBJC_EHTYPE_$_EH2" = external global' %t &&
// RUN: grep '@"OBJC_EHTYPE_$_EH3" = global .*section "__DATA,__objc_const", align 8' %t &&
// RUN: grep '@"OBJC_EHTYPE_$_EH3"' %t | count 3 &&
// RUN: grep '@"\\01L_OBJC_CLASS_NAME_" =.*section "__TEXT,__cstring,cstring_literals", align 1' %t | count 1 &&
// RUN: grep -F 'define internal void @"\01-[A im0]"' %t &&
// RUN: grep -F 'define internal void @"\01-[A(Cat) im1]"' %t &&

// RUN: clang-cc -fvisibility=hidden -triple x86_64-apple-darwin9 -emit-llvm -o %t %s &&

// RUN: grep '@"OBJC_METACLASS_$_A" = hidden global .*section "__DATA, __objc_data", align 8' %t && 
// RUN: grep '@"OBJC_CLASS_$_A" = hidden global .*section "__DATA, __objc_data", align 8' %t &&
// RUN: grep '@"OBJC_EHTYPE_$_EH1" = weak hidden global .*section "__DATA,__datacoal_nt,coalesced"' %t &&
// RUN: grep '@"OBJC_EHTYPE_$_EH2" = external global' %t &&
// RUN: grep '@"OBJC_EHTYPE_$_EH3" = hidden global .*section "__DATA,__objc_const", align 8' %t &&
// RUN: grep -F 'define internal void @"\01-[A im0]"' %t &&
// RUN: grep -F 'define internal void @"\01-[A(Cat) im1]"' %t &&

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
