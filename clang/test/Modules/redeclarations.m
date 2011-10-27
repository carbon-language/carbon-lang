#ifdef MODULE_LEFT
@class NSObject;
#endif

#ifdef MODULE_RIGHT
@interface NSObject
@end
#endif

#ifdef APP
__import_module__ Right;
__import_module__ Left;

@interface MyObject : NSObject
@end
#endif

// RUN: rm -rf %t
// RUN: %clang_cc1 -fmodule-cache-path %t -fdisable-module-hash -emit-module -o %t/Left.pcm -DMODULE_LEFT %s
// RUN: %clang_cc1 -fmodule-cache-path %t -fdisable-module-hash -emit-module -o %t/Right.pcm -DMODULE_RIGHT %s
// RUN: %clang_cc1 -fmodule-cache-path %t -fdisable-module-hash -DAPP %s -verify

