// RUN: %clang_cc1 -fobjc-nonfragile-abi %s
#ifndef __has_feature
#error Should have __has_feature
#endif

#if !__has_feature(objc_nonfragile_abi)
#error Non-fragile ABI used for compilation but feature macro not set.
#endif
