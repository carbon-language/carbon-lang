// Test this without pch.
// RUN: %clang_cc1 -fblocks -triple x86_64-apple-darwin11 -fobjc-nonfragile-abi -fobjc-arc -include %S/Inputs/arc.h -fsyntax-only -emit-llvm -o - %s

// Test with pch.
// RUN: %clang_cc1 -emit-pch -fblocks -triple x86_64-apple-darwin11 -fobjc-nonfragile-abi -fobjc-arc -x objective-c-header -o %t %S/Inputs/arc.h
// RUN: %clang_cc1 -fblocks -triple x86_64-apple-darwin11 -fobjc-nonfragile-abi -fobjc-arc -include-pch %t -fsyntax-only -emit-llvm -o - %s 

array0 a0;
array1 a1;
