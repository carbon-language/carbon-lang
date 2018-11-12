// RUN: %clang_cc1 -emit-llvm -femit-coverage-data -femit-coverage-notes %s -o - \
// RUN:    | FileCheck -check-prefix=ALL %s
// RUN: %clang_cc1 -emit-llvm -femit-coverage-data -femit-coverage-notes -fprofile-exclude-files=".*\.h$" %s -o - \
// RUN:    | FileCheck -check-prefix=NO-HEADER %s
// RUN: %clang_cc1 -emit-llvm -femit-coverage-data -femit-coverage-notes -fprofile-filter-files=".*\.c$" %s -o - \
// RUN:    | FileCheck -check-prefix=NO-HEADER %s
// RUN: %clang_cc1 -emit-llvm -femit-coverage-data -femit-coverage-notes -fprofile-filter-files=".*\.c$;.*1\.h$" %s -o - \
// RUN:    | FileCheck -check-prefix=NO-HEADER2 %s
// RUN: %clang_cc1 -emit-llvm -femit-coverage-data -femit-coverage-notes -fprofile-exclude-files=".*2\.h$;.*1\.h$" %s -o - \
// RUN:    | FileCheck -check-prefix=JUST-C %s
// RUN: %clang_cc1 -emit-llvm -femit-coverage-data -femit-coverage-notes -fprofile-exclude-files=".*code\-coverage\-filter\.c$" %s -o - \
// RUN:    | FileCheck -check-prefix=HEADER %s
// RUN: %clang_cc1 -emit-llvm -femit-coverage-data -femit-coverage-notes -fprofile-filter-files=".*\.c$" -fprofile-exclude-files=".*\.c$" %s -o - \
// RUN:    | FileCheck -check-prefix=NONE %s
// RUN: %clang_cc1 -emit-llvm -femit-coverage-data -femit-coverage-notes -fprofile-filter-files=".*\.c$" -fprofile-exclude-files=".*\.h$" %s -o - \
// RUN:    | FileCheck -check-prefix=JUST-C %s

#include "Inputs/code-coverage-filter1.h"
#include "Inputs/code-coverage-filter2.h"

void test() {
  test1();
  test2();
}

// ALL: void @test1() #0 {{.*}}
// ALL: {{.*}}__llvm_gcov_ctr{{.*}}
// ALL: ret void
// ALL: void @test2() #0 {{.*}}
// ALL: {{.*}}__llvm_gcov_ctr{{.*}}
// ALL: ret void
// ALL: void @test() #0 {{.*}}
// ALL: {{.*}}__llvm_gcov_ctr{{.*}}
// ALL: ret void

// NO-HEADER: void @test1() #0 {{.*}}
// NO-HEADER-NOT: {{.*}}__llvm_gcov_ctr{{.*}}
// NO-HEADER: ret void
// NO-HEADER: void @test2() #0 {{.*}}
// NO-HEADER-NOT: {{.*}}__llvm_gcov_ctr{{.*}}
// NO-HEADER: ret void
// NO-HEADER: void @test() #0 {{.*}}
// NO-HEADER: {{.*}}__llvm_gcov_ctr{{.*}}
// NO-HEADER: ret void

// NO-HEADER2: void @test1() #0 {{.*}}
// NO-HEADER2: {{.*}}__llvm_gcov_ctr{{.*}}
// NO-HEADER2: ret void
// NO-HEADER2: void @test2() #0 {{.*}}
// NO-HEADER2-NOT: {{.*}}__llvm_gcov_ctr{{.*}}
// NO-HEADER2: ret void
// NO-HEADER2: void @test() #0 {{.*}}
// NO-HEADER2: {{.*}}__llvm_gcov_ctr{{.*}}
// NO-HEADER2: ret void

// JUST-C: void @test1() #0 {{.*}}
// JUST-C-NOT: {{.*}}__llvm_gcov_ctr{{.*}}
// JUST-C: ret void
// JUST-C: void @test2() #0 {{.*}}
// JUST-C-NOT: {{.*}}__llvm_gcov_ctr{{.*}}
// JUST-C: ret void
// JUST-C: void @test() #0 {{.*}}
// JUST-C: {{.*}}__llvm_gcov_ctr{{.*}}
// JUST-C: ret void

// HEADER: void @test1() #0 {{.*}}
// HEADER: {{.*}}__llvm_gcov_ctr{{.*}}
// HEADER: ret void
// HEADER: void @test2() #0 {{.*}}
// HEADER: {{.*}}__llvm_gcov_ctr{{.*}}
// HEADER: ret void
// HEADER: void @test() #0 {{.*}}
// HEADER-NOT: {{.*}}__llvm_gcov_ctr{{.*}}
// HEADER: ret void

// NONE: void @test1() #0 {{.*}}
// NONE-NOT: {{.*}}__llvm_gcov_ctr{{.*}}
// NONE: ret void
// NONE: void @test2() #0 {{.*}}
// NONE-NOT: {{.*}}__llvm_gcov_ctr{{.*}}
// NONE: ret void
// NONE: void @test() #0 {{.*}}
// NONE-NOT: {{.*}}__llvm_gcov_ctr{{.*}}
// NONE: ret void
