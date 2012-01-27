// RUN: env PATH=%s-helper %clang -no-integrated-as -target x86_64--linux %s -o - > %t.log
// RUN: env PATH=%s-helper %clang -no-integrated-as -m32 -target x86_64--linux %s -o - >> %t.log
// RUN: FileCheck -input-file %t.log %s

// FIXME: It had failed on Win32 due to inability of executing shell scripts.
// Still it fails even with mingw MSYS bash.
// REQUIRES: shell

// CHECK: x86_64--linux-as called
// CHECK: x86_64--linux-ld called
// CHECK: x86_64--linux-as called
// CHECK: x86_64--linux-ld called

int
main(void)
{
  return 0;
}
