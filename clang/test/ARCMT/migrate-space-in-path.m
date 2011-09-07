// RUN: rm -rf %t.migrate
// RUN: %clang_cc1 -arcmt-migrate -arcmt-migrate-directory %t.migrate %S/"with space"/test1.m.in -x objective-c -fobjc-nonfragile-abi
// RUN: %clang_cc1 -arcmt-migrate -arcmt-migrate-directory %t.migrate %S/"with space"/test2.m.in -x objective-c -fobjc-nonfragile-abi
// RUN: c-arcmt-test -arcmt-migrate-directory %t.migrate | arcmt-test -verify-transformed-files %S/"with space"/test1.m.in.result %S/"with space"/test2.m.in.result %S/"with space"/test.h.result
// RUN: rm -rf %t.migrate
