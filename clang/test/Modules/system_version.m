// Test checking that we're hashing a system version file in the
// module hash.

// First, build a system root.
// RUN: rm -rf %t
// RUN: mkdir -p %t/usr/include
// RUN: cp %S/Inputs/Modified/A.h %t/usr/include
// RUN: cp %S/Inputs/Modified/B.h %t/usr/include
// RUN: cp %S/Inputs/Modified/module.map %t/usr/include

// Run once with no system version file. We should end up with one module.
// RUN: %clang_cc1 -fmodules-cache-path=%t/cache -fmodules -fimplicit-module-maps -isysroot %t -I %t/usr/include %s -verify
// RUN: ls -R %t | grep -c "ModA.*pcm" | grep 1

// Add a system version file and run again. We should now have two
// module variants.
// RUN: mkdir -p %t/System/Library/CoreServices
// RUN: echo "hello" > %t/System/Library/CoreServices/SystemVersion.plist
// RUN: %clang_cc1 -fmodules-cache-path=%t/cache -fmodules -fimplicit-module-maps -isysroot %t -I %t/usr/include %s -verify
// RUN: ls -R %t | grep -c "ModA.*pcm" | grep 2

// Change the system version file and run again. We should now have three
// module variants.
// RUN: mkdir -p %t/System/Library/CoreServices
// RUN: echo "modules" > %t/System/Library/CoreServices/SystemVersion.plist
// RUN: %clang_cc1 -fmodules-cache-path=%t/cache -fmodules -fimplicit-module-maps -isysroot %t -I %t/usr/include %s -verify
// RUN: ls -R %t | grep -c "ModA.*pcm" | grep 3

// expected-no-diagnostics
@import ModA;

