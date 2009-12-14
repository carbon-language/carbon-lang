// Test this without pch.
// RUN: clang -cc1 -include %S/objc_methods.h -fsyntax-only -verify %s

// Test with pch.
// RUN: clang -cc1 -x objective-c -emit-pch -o %t %S/objc_methods.h
// RUN: clang -cc1 -include-pch %t -fsyntax-only -verify %s 

void func() {
 TestPCH *xx;
 TestForwardClassDecl *yy;
// FIXME:
// AliasForTestPCH *zz;
 
 xx = [TestPCH alloc];
 [xx instMethod];
}
