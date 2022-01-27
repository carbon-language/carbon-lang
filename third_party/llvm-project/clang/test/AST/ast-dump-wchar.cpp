// Test without serialization:
// RUN: %clang_cc1 -std=c++11 -ast-dump %s -triple x86_64-linux-gnu \
// RUN: | FileCheck %s
//
// Test with serialization:
// RUN: %clang_cc1 -std=c++11 -triple x86_64-linux-gnu -emit-pch -o %t %s
// RUN: %clang_cc1 -x c++ -std=c++11 -triple x86_64-linux-gnu -include-pch %t -ast-dump-all /dev/null \
// RUN: | sed -e "s/ <undeserialized declarations>//" -e "s/ imported//" \
// RUN: | FileCheck %s

char c8[] = u8"test\0\\\"\a\b\f\n\r\t\v\234";
// CHECK: StringLiteral {{.*}} u8"test\000\\\"\a\b\f\n\r\t\v\234"

char16_t c16[] = u"test\0\\\"\t\a\b\234\u1234";
// CHECK: StringLiteral {{.*}} u"test\000\\\"\t\a\b\234\u1234"

char32_t c32[] = U"test\0\\\"\t\a\b\234\u1234\U0010ffff"; // \
// CHECK: StringLiteral {{.*}} U"test\000\\\"\t\a\b\234\u1234\U0010FFFF"

wchar_t wc[] = L"test\0\\\"\t\a\b\234\u1234\xffffffff"; // \
// CHECK: StringLiteral {{.*}} L"test\000\\\"\t\a\b\234\x1234\xFFFFFFFF"
