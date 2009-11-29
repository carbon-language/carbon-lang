// Test this without pch.
// RUN: clang-cc -include %S/objc_methods.h -fsyntax-only -verify %s

// Test with pch.
// RUN: clang-cc -x objective-c -emit-pch -o %t %S/objc_methods.h
// RUN: clang-cc -include-pch %t -fsyntax-only -verify %s 

void func() {
 TestPCH *xx;
 TestForwardClassDecl *yy;
// FIXME:
// AliasForTestPCH *zz;
 
 xx = [TestPCH alloc];
 [xx instMethod];
}
