// RUN: clang-cc %s
#ifndef __has_feature
#error Should have __has_feature
#endif

#if __has_feature(objc_nonfragile_abi)
#error Non-fragile ABI not used for compilation but feature macro set.
#endif
