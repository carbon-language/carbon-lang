// RUN: %clang_cc1 -std=c++11 -triple i686-unknown-windows-msvc -fobjc-runtime=gnustep -fobjc-arc -Wno-objc-root-class -emit-llvm -o - %s | FileCheck -check-prefixes CHECK,CHECK-GNUSTEP %s
// RUN: %clang_cc1 -std=c++11 -triple i686-unknown-windows-msvc -fobjc-runtime=macosx -fobjc-arc -Wno-objc-root-class -emit-llvm -o - %s | FileCheck -check-prefixes CHECK,CHECK-DARWIN %s
// RUN: %clang_cc1 -std=c++11 -triple i686-unknown-windows-msvc -fobjc-runtime=ios -fobjc-arc -Wno-objc-root-class -emit-llvm -o - %s | FileCheck -check-prefixes CHECK,CHECK-DARWIN %s

// non trivially copyable, forces inalloca
struct S {
  S(const S &s) {}
};

@interface C
@end

@implementation C
- (void)object:(id)obj struct:(S)s {
}
@end

// CHECK-GNUSTEP: define internal void @_i_C__object_struct_(<{ %0*, i8*, i8*, %struct.S, [3 x i8] }>* inalloca(<{ %0*, i8*, i8*, %struct.S, [3 x i8] }>) %0)
// CHECK-DARWIN: define internal void @"\01-[C object:struct:]"(<{ %0*, i8*, i8*, %struct.S, [3 x i8] }>* inalloca(<{ %0*, i8*, i8*, %struct.S, [3 x i8] }>) %0)
// CHECK: %obj = getelementptr inbounds <{ %0*, i8*, i8*, %struct.S, [3 x i8] }>, <{ %0*, i8*, i8*, %struct.S, [3 x i8] }>* %0, i32 0, i32 2
// CHECK: %[[INSTANCE:[0-9]+]] = load i8*, i8** %obj, align 4
// CHECK: call void @llvm.objc.storeStrong(i8** %obj, i8* %[[INSTANCE]])
