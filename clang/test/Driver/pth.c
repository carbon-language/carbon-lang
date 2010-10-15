// Test transparent PTH support.

// RUN: %clang -ccc-pch-is-pth -x c-header %s -o %t.h.pth -### 2> %t.log
// RUN: FileCheck -check-prefix CHECK1 -input-file %t.log %s

// CHECK1: "{{.*}}/clang{{.*}}" "-cc1" {{.*}} "-o" "{{.*}}.h.pth" "-x" "c-header" "{{.*}}pth.c"

// RUN: touch %t.h.pth
// RUN: %clang -ccc-pch-is-pth -E -include %t.h %s -### 2> %t.log
// RUN: FileCheck -check-prefix CHECK2 -input-file %t.log %s

// CHECK2: "{{.*}}/clang{{.*}}" "-cc1" {{.*}}"-include-pth" "{{.*}}.h.pth" {{.*}}"-x" "c" "{{.*}}pth.c"

// RUN: mkdir -p %t.pth
// RUN: %clang -ccc-pch-is-pth -x c-header %s -o %t.pth/c -### 2> %t.log
// RUN: FileCheck -check-prefix CHECK3 -input-file %t.log %s

// CHECK3: "{{.*}}/clang{{.*}}" "-cc1" {{.*}} "-o" "{{.*}}.pth/c" "-x" "c-header" "{{.*}}pth.c"

// RUN: rm -f %t.pth/c
// RUN: %clang -ccc-pch-is-pth -E -include %t %s -### 2> %t.log
// RUN: echo "DONE" >> %t.log
// RUN: FileCheck -check-prefix CHECK4 -input-file %t.log %s

// CHECK4: {{.*}} ignoring argument '-include {{.*}}' due to missing precompiled header '{{.*}}.pth/c' for language 'c'
// CHECK4-NOT: -include-pth
// CHECK4: DONE

// RUN: touch %t.pth/c
// RUN: %clang -ccc-pch-is-pth -E -include %t %s -### 2> %t.log
// RUN: FileCheck -check-prefix CHECK5 -input-file %t.log %s

// CHECK5: "{{.*}}/clang{{.*}}" "-cc1" {{.*}}"-include-pth" "{{.*}}.pth/c" {{.*}}"-x" "c" "{{.*}}pth.c"
