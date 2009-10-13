// Test transparent PTH support.

// RUN: clang -ccc-pch-is-pth -x c-header %s -o %t.h.pth -### 2> %t.log &&
// RUN: FileCheck -check-prefix CHECK1 -input-file %t.log %s &&

// CHECK1: "{{.*}}/clang-cc{{.*}}" {{.*}} "-o" "{{.*}}.h.pth" "-x" "c-header" "{{.*}}pth.c"

// RUN: touch %t.h.pth &&
// RUN: clang -ccc-pch-is-pth -E -include %t.h %s -### 2> %t.log &&
// RUN: FileCheck -check-prefix CHECK2 -input-file %t.log %s

// CHECK2: "{{.*}}/clang-cc{{.*}}" {{.*}}"-include-pth" "{{.*}}.h.pth" {{.*}}"-x" "c" "{{.*}}pth.c"
