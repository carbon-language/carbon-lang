// Test this without pch.
// RUN: clang -cc1 -include %S/objc_import.h -fsyntax-only -verify %s

// Test with pch.
// RUN: clang -cc1 -x objective-c -emit-pch -o %t %S/objc_import.h
// RUN: clang -cc1 -include-pch %t -fsyntax-only -verify %s 

#import "objc_import.h"

void func() {
 TestPCH *xx;

 xx = [TestPCH alloc];
 [xx instMethod];
}
