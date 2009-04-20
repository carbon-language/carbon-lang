// Test this without pch.
// FIXME: clang-cc -include %S/methods.h -fsyntax-only -verify %s &&

// Test with pch.
// FIXME: clang-cc -x=objective-c -emit-pch -o %t %S/methods.h &&
// FIXME: clang-cc -include-pch %t -fsyntax-only -verify %s 

void func() {
 TestPCH *xx = [TestPCH alloc];
 [xx instMethod];
}
