// RUN: %llvmgxx %s -fapple-kext -S -o -
// The extra check in 71555 caused this to crash on Darwin X86
// in an assert build.
class foo {
 virtual ~foo ();
};
foo::~foo(){}
