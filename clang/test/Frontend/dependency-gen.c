// Basic test
// RUN: rm -rf %t.dir
// RUN: mkdir -p %t.dir/a/b
// RUN: echo > %t.dir/a/b/x.h
// RUN: cd %t.dir
// RUN: %clang -MD -MF - %s -fsyntax-only -I a/b | FileCheck -check-prefix=CHECK-ONE %s
// CHECK-ONE: {{ }}a/b{{[/\\]}}x.h

// PR8974 (-include flag)
// RUN: %clang -MD -MF - %s -fsyntax-only -include a/b/x.h -DINCLUDE_FLAG_TEST | FileCheck -check-prefix=CHECK-TWO %s
// CHECK-TWO: {{ }}a/b/x.h

// rdar://problem/9734352 (paths involving ".")
// RUN: %clang -MD -MF - %s -fsyntax-only -I ./a/b | FileCheck -check-prefix=CHECK-THREE %s
// CHECK-THREE: {{ }}a/b{{[/\\]}}x.h
// RUN: %clang -MD -MF - %s -fsyntax-only -I .//./a/b/ | FileCheck -check-prefix=CHECK-FOUR %s
// CHECK-FOUR: {{ }}a/b{{[/\\]}}x.h
// RUN: %clang -MD -MF - %s -fsyntax-only -I a/b/. | FileCheck -check-prefix=CHECK-FIVE %s
// CHECK-FIVE: {{ }}a/b/.{{[/\\]}}x.h
// RUN: cd a/b
// RUN: %clang -MD -MF - %s -fsyntax-only -I ./ | FileCheck -check-prefix=CHECK-SIX %s
// CHECK-SIX: {{ }}x.h
// RUN: echo "fun:foo" > %t.blacklist
// RUN: %clang -MD -MF - %s -fsyntax-only -fsanitize=cfi-vcall -flto -fsanitize-blacklist=%t.blacklist -I ./ | FileCheck -check-prefix=CHECK-SEVEN %s
// CHECK-SEVEN: .blacklist
// CHECK-SEVEN: {{ }}x.h
// RUN: %clang -target x86_64-linux-gnu -MD -MF - %s -fsyntax-only -fsanitize=address -flto -I . | FileCheck -check-prefix=CHECK-EIGHT %s
// CHECK-EIGHT: {{ }}x.h
// RUN: %clang -target x86_64-linux-gnu -MD -MF - %s -fsyntax-only -fsanitize=address -flto -I . -fno-sanitize-blacklist | FileCheck -check-prefix=CHECK-NINE %s
// CHECK-NINE-NOT: asan_blacklist.txt
// CHECK-NINE: {{ }}x.h
#ifndef INCLUDE_FLAG_TEST
#include <x.h>
#endif
