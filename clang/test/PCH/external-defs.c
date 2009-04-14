// Test with pch.
// RUN: clang-cc -triple x86_64-apple-darwin9 -emit-pch -o %t.pch %S/external-defs.h &&
// RUN: clang-cc -triple x86_64-apple-darwin9 -include-pch %t.pch -emit-llvm -o %t %s &&

// RUN: grep "@x = common global i32 0, align 4" %t | count 1 &&
// FIXME below: should be i32 17, but we don't serialize y's value yet
// RUN: grep "@y = common global i32 0, align 4"  %t | count 1 &&
// RUN: grep "@z" %t | count 0
