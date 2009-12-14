// RUN: clang -cc1 -triple i386-apple-darwin9 -emit-llvm %s -o - | FileCheck %s

#include <stdio.h>

int main()
{
    @try {
        @throw @"foo";
    } @catch (id e) {
        @try {
// CHECK: call void @objc_exception_throw
           @throw;
        } @catch (id e) {
            if (e) {
                printf("caught \n");
            } else {
                printf("caught (WRONG)\n");
            }
        } @catch (...) {
            printf("caught nothing (WRONG)\n");
        }
    }
}

