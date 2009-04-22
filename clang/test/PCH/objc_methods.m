// Test this without pch.
// FIXME: clang-cc -include %S/objc_methods.h -fsyntax-only -verify %s &&

// Test with pch.
// FIXME: clang-cc -x=objective-c -emit-pch -o %t %S/objc_methods.h &&
// FIXME: clang-cc -include-pch %t -fsyntax-only -verify %s 

void func() {
 TestPCH *xx = [TestPCH alloc];
 [xx instMethod];
}
