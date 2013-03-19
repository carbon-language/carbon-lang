// RUN: %clang -target x86_64-pc-win32 -no-integrated-as %s -c -v 2>&1 | FileCheck %s

// CHECK: cc1as -triple x86_64-pc-win32
