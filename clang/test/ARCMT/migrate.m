// RUN: %clang_cc1 -arcmt-migrate -arcmt-migrate-directory %t %S/Inputs/test1.m.in -x objective-c -fobjc-nonfragile-abi
// RUN: %clang_cc1 -arcmt-migrate -arcmt-migrate-directory %t %S/Inputs/test2.m.in -x objective-c -fobjc-nonfragile-abi
// RUN: c-arcmt-test -arcmt-migrate-directory %t | arcmt-test -verify-transformed-files %S/Inputs/test1.m.in.result %S/Inputs/test2.m.in.result %S/Inputs/test.h.result
// RUN: rm -rf %t
