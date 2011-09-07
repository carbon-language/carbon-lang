// RUN: mkdir -p %t/"with space"
// RUN: cp %S/Inputs/* %t/"with space"
// RUN: %clang_cc1 -arcmt-migrate -arcmt-migrate-directory %t.migrate %t/"with space"/test1.m.in -x objective-c -fobjc-nonfragile-abi
// RUN: %clang_cc1 -arcmt-migrate -arcmt-migrate-directory %t.migrate %t/"with space"/test2.m.in -x objective-c -fobjc-nonfragile-abi
// RUN: c-arcmt-test -arcmt-migrate-directory %t.migrate | arcmt-test -verify-transformed-files %t/"with space"/test1.m.in.result %t/"with space"/test2.m.in.result %t/"with space"/test.h.result
// RUN: rm -rf %t.migrate
// RUN: rm -rf %t
