// RUN: %clang_cc1 %s -emit-llvm -o - -mno-tls-direct-seg-refs | FileCheck %s -check-prefix=NO-TLSDIRECT
// RUN: %clang_cc1 %s -emit-llvm -o - | FileCheck %s -check-prefix=TLSDIRECT

// NO-TLSDIRECT: attributes #{{[0-9]+}} = {{{.*}} "indirect-tls-seg-refs"
// TLSDIRECT-NOT: attributes #{{[0-9]+}} = {{{.*}} "indirect-tls-seg-refs"

void test1() {
}
