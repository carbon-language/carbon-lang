// Test this without pch.
// RUN: clang -cc1 -include %S/objc_property.h -fsyntax-only -verify %s

// Test with pch.
// RUN: clang -cc1 -x objective-c -emit-pch -o %t %S/objc_property.h
// RUN: clang -cc1 -include-pch %t -fsyntax-only -verify %s 

void func() {
 TestProperties *xx = [TestProperties alloc];
 xx.value = 5;
}
