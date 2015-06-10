// RUN: %clang_cc1 -triple=i386-pc-solaris2.11 -w -emit-llvm %s -o - | FileCheck %s

extern "C" {
  struct statvfs64 {
    int f;
  };
#pragma redefine_extname statvfs64 statvfs
  int statvfs64(struct statvfs64 *);
}

void foo() {
  struct statvfs64 st;
  statvfs64(&st);
// Check that even if there is a structure with redefined name before the
// pragma, subsequent function name redefined properly. PR5172, Comment 11.
// CHECK:  call i32 @statvfs(%struct.statvfs64* %st)
}

