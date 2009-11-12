// Check that -E mode is invariant when using an implicit PCH.

// RUN: clang-cc -include %S/preprocess.h -E -o %t.orig %s
// RUN: clang-cc -emit-pch -o %t %S/preprocess.h
// RUN: clang-cc -include-pch %t -E -o %t.from_pch %s
// RUN: diff %t.orig %t.from_pch

a_typedef a_value;
