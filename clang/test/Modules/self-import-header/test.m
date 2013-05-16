// rdar://13840148

// RUN: rm -rf %t
// RUN: %clang -fsyntax-only -isysroot %S/../Inputs/System/usr/include -fmodules -fmodules-cache-path=%t \
// RUN:    -target x86_64-darwin \
// RUN:    -F %S -I %S %s -D__need_wint_t -Werror=implicit-function-declaration

@import af;
