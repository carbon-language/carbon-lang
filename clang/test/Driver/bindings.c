// Basic binding.
// RUN: %clang -target i386-unknown-unknown -ccc-print-bindings %s 2>&1 | FileCheck %s --check-prefix=CHECK01
// CHECK01: "clang", inputs: ["{{.*}}bindings.c"], output: "{{.*}}.s"
// CHECK01: "gcc::Assemble", inputs: ["{{.*}}.s"], output: "{{.*}}.o"
// CHECK01: "gcc::Link", inputs: ["{{.*}}.o"], output: "a.out"

// RUN: %clang -target i386-unknown-unknown -ccc-print-bindings -ccc-no-clang %s 2>&1 | FileCheck %s --check-prefix=CHECK02
// CHECK02: "gcc::Compile", inputs: ["{{.*}}bindings.c"], output: "{{.*}}.s"
// CHECK02: "gcc::Assemble", inputs: ["{{.*}}.s"], output: "{{.*}}.o"
// CHECK02: "gcc::Link", inputs: ["{{.*}}.o"], output: "a.out"

// RUN: %clang -target i386-unknown-unknown -ccc-print-bindings -ccc-no-clang -no-integrated-cpp %s 2>&1 | FileCheck %s --check-prefix=CHECK03
// CHECK03: "gcc::Preprocess", inputs: ["{{.*}}bindings.c"], output: "{{.*}}.i"
// CHECK03: "gcc::Compile", inputs: ["{{.*}}.i"], output: "{{.*}}.s"
// CHECK03: "gcc::Assemble", inputs: ["{{.*}}.s"], output: "{{.*}}.o"
// CHECK03: "gcc::Link", inputs: ["{{.*}}.o"], output: "a.out"

// RUN: %clang -target i386-unknown-unknown -ccc-print-bindings -ccc-no-clang -x c-header %s 2>&1 | FileCheck %s --check-prefix=CHECK04
// CHECK04: "gcc::Precompile", inputs: ["{{.*}}bindings.c"], output: "{{.*}}bindings.c.gch

// Clang control options

// RUN: %clang -target i386-unknown-unknown -ccc-print-bindings -fsyntax-only %s 2>&1 | FileCheck %s --check-prefix=CHECK05
// CHECK05: "clang", inputs: ["{{.*}}bindings.c"], output: (nothing)

// RUN: %clang -target i386-unknown-unknown -ccc-print-bindings -ccc-no-clang -fsyntax-only %s 2>&1 | FileCheck %s --check-prefix=CHECK06
// CHECK06: "gcc::Compile", inputs: ["{{.*}}bindings.c"], output: (nothing)

// RUN: %clang -target i386-unknown-unknown -ccc-print-bindings -ccc-no-clang-cxx -fsyntax-only -x c++ %s 2>&1 | FileCheck %s --check-prefix=CHECK07
// CHECK07: "gcc::Compile", inputs: ["{{.*}}bindings.c"], output: (nothing)

// RUN: %clang -target i386-unknown-unknown -ccc-print-bindings -ccc-clang-cxx -fsyntax-only -x c++ %s 2>&1 | FileCheck %s --check-prefix=CHECK08
// CHECK08: "clang", inputs: ["{{.*}}bindings.c"], output: (nothing)

// RUN: %clang -target i386-unknown-unknown -ccc-print-bindings -ccc-no-clang-cpp -fsyntax-only -no-integrated-cpp %s 2>&1 | FileCheck %s --check-prefix=CHECK09
// CHECK09: "gcc::Preprocess", inputs: ["{{.*}}bindings.c"], output: "{{.*}}.i"
// CHECK09: "clang", inputs: ["{{.*}}.i"], output: (nothing)

// RUN: %clang -target i386-apple-darwin9 -ccc-print-bindings -ccc-clang-archs i386 %s -S -arch ppc 2>&1 | FileCheck %s --check-prefix=CHECK10
// CHECK10: "gcc::Compile", inputs: ["{{.*}}bindings.c"], output: "bindings.s"

// RUN: %clang -target i386-apple-darwin9 -ccc-print-bindings -ccc-clang-archs powerpc %s -S -arch ppc 2>&1 | FileCheck %s --check-prefix=CHECK11
// CHECK11: "clang", inputs: ["{{.*}}bindings.c"], output: "bindings.s"

// RUN: %clang -target powerpc-unknown-unknown -ccc-print-bindings -ccc-clang-archs "" %s -S 2>&1 | FileCheck %s --check-prefix=CHECK12
// CHECK12: "clang", inputs: ["{{.*}}bindings.c"], output: "bindings.s"

// RUN: %clang -target powerpc-unknown-unknown -ccc-print-bindings -ccc-clang-archs "i386" %s -S 2>&1 | FileCheck %s --check-prefix=CHECK13
// CHECK13: "gcc::Compile", inputs: ["{{.*}}bindings.c"], output: "bindings.s"

// Darwin bindings
// RUN: %clang -target i386-apple-darwin9 -no-integrated-as -ccc-print-bindings %s 2>&1 | FileCheck %s --check-prefix=CHECK14
// CHECK14: "clang", inputs: ["{{.*}}bindings.c"], output: "{{.*}}.s"
// CHECK14: "darwin::Assemble", inputs: ["{{.*}}.s"], output: "{{.*}}.o"
// CHECK14: "darwin::Link", inputs: ["{{.*}}.o"], output: "a.out"
