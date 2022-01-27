// RUN: %clang_cc1 -std=c++11 -emit-llvm -debug-info-kind=limited -triple x86_64-apple-darwin %s -o - | FileCheck %s
// RUN: %clang_cc1 -std=c++11 -dwarf-version=4 -gstrict-dwarf -emit-llvm -debug-info-kind=limited \
// RUN:   -triple x86_64-apple-darwin %s -o - | FileCheck %s
// RUN: %clang_cc1 -std=c++11 -dwarf-version=3 -gstrict-dwarf -emit-llvm -debug-info-kind=limited \
// RUN:  -triple x86_64-apple-darwin %s -o - | FileCheck %s --check-prefix=NORVALUE

extern "C" {
extern int printf(const char * format, ...);
}
void foo (int &&i)
{
  printf("%d\n", i);
}

// CHECK: !DIDerivedType(tag: DW_TAG_rvalue_reference_type, baseType: ![[INT:[0-9]+]], size: 64)
// CHECK: ![[INT]] = !DIBasicType(name: "int"
// NORVALUE: !DIDerivedType(tag: DW_TAG_reference_type, baseType: ![[INT:[0-9]+]], size: 64)
