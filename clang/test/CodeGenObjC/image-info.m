// RUN: %clang_cc1 -triple i386-apple-darwin9 -emit-llvm -o %t %s
// RUN: FileCheck --check-prefix CHECK-FRAGILE < %t %s

// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -fobjc-nonfragile-abi -emit-llvm -o %t %s
// RUN: FileCheck --check-prefix CHECK-NONFRAGILE < %t %s

// CHECK-FRAGILE: @"\01L_OBJC_IMAGE_INFO" = internal constant [2 x i32] [i32 0, i32 16], section "__OBJC, __image_info,regular"
// CHECK-NONFRAGILE: @"\01L_OBJC_IMAGE_INFO" = internal constant [2 x i32] [i32 0, i32 16], section "__DATA, __objc_imageinfo, regular, no_dead_strip"
