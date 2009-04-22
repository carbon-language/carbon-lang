// Test this without pch.
// RUN: clang-cc -fblocks -include %S/objc_exprs.h -fsyntax-only -verify %s &&

// Test with pch.
// RUN: clang-cc -x objective-c-header -emit-pch -fblocks -o %t %S/objc_exprs.h &&
// RUN: clang-cc -fblocks -include-pch %t -fsyntax-only -verify %s 

