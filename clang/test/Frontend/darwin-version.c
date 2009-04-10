// RUN: clang-cc -triple armv6-apple-darwin9 -dM -E -o %t - < /dev/null &&
// RUN: grep '__ENVIRONMENT_IPHONE_OS_VERSION_MIN_REQUIRED__' %t | grep '10000' | count 1 &&
// RUN: grep '__ENVIRONMENT_MAC_OS_X_VERSION_MIN_REQUIRED__' %t | count 0 &&
// RUN: clang-cc -triple armv6-apple-darwin9 -miphoneos-version-min=2.0 -dM -E -o %t - < /dev/null &&
// RUN: grep '__ENVIRONMENT_IPHONE_OS_VERSION_MIN_REQUIRED__' %t | grep '20000' | count 1 &&
// RUN: grep '__ENVIRONMENT_MAC_OS_X_VERSION_MIN_REQUIRED__' %t | count 0 &&
// RUN: clang-cc -triple armv6-apple-darwin9 -miphoneos-version-min=2.2 -dM -E -o %t - < /dev/null &&
// RUN: grep '__ENVIRONMENT_IPHONE_OS_VERSION_MIN_REQUIRED__' %t | grep '20200' | count 1 &&
// RUN: clang-cc -triple i686-apple-darwin8 -dM -E -o %t - < /dev/null &&
// RUN: grep '__ENVIRONMENT_IPHONE_OS_VERSION_MIN_REQUIRED__' %t | count 0 &&
// RUN: grep '__ENVIRONMENT_MAC_OS_X_VERSION_MIN_REQUIRED__' %t | grep '1040' | count 1 &&
// RUN: clang-cc -triple i686-apple-darwin9 -dM -E -o %t - < /dev/null &&
// RUN: grep '__ENVIRONMENT_MAC_OS_X_VERSION_MIN_REQUIRED__' %t | grep '1050' | count 1 &&
// RUN: clang-cc -triple i686-apple-darwin10 -dM -E -o %t - < /dev/null &&
// RUN: grep '__ENVIRONMENT_MAC_OS_X_VERSION_MIN_REQUIRED__' %t | grep '1060' | count 1 &&
// RUN: clang-cc -triple i686-apple-darwin9 -mmacosx-version-min=10.4 -dM -E -o %t - < /dev/null &&
// RUN: grep '__ENVIRONMENT_IPHONE_OS_VERSION_MIN_REQUIRED__' %t | count 0 &&
// RUN: grep '__ENVIRONMENT_MAC_OS_X_VERSION_MIN_REQUIRED__' %t | grep '1040' | count 1 &&
// RUN: clang-cc -triple i686-apple-darwin9 -mmacosx-version-min=10.5 -dM -E -o %t - < /dev/null &&
// RUN: grep '__ENVIRONMENT_MAC_OS_X_VERSION_MIN_REQUIRED__' %t | grep '1050' | count 1 &&
// RUN: clang-cc -triple i686-apple-darwin9 -mmacosx-version-min=10.6 -dM -E -o %t - < /dev/null &&
// RUN: grep '__ENVIRONMENT_MAC_OS_X_VERSION_MIN_REQUIRED__' %t | grep '1060' | count 1 &&
// RUN: true
