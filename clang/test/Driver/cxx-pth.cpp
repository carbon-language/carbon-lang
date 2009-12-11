// Test forced PTH for CXX support.

// RUN: clang -x c++-header %s -### 2> %t.log
// RUN: FileCheck -check-prefix EMIT -input-file %t.log %s

// EMIT: "{{.*}}/clang{{.*}}" {{.*}} "-emit-pth" "{{.*}}.cpp.gch" "-x" "c++-header" "{{.*}}.cpp"

// RUN: touch %t.h.gch
// RUN: clang -E -include %t.h %s -### 2> %t.log
// RUN: FileCheck -check-prefix USE -input-file %t.log %s

// USE: "{{.*}}/clang{{.*}}" {{.*}}"-include-pth" "{{.*}}.h.gch" {{.*}}"-x" "c++" "{{.*}}.cpp"
