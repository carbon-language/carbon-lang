// Basic binding.
// RUN: clang-driver -ccc-print-bindings %s &> %t &&
// RUN: grep 'bind - "clang", inputs: \[".*bindings.c"\], output: "/tmp/foo.s"' %t &&
// RUN: grep 'bind - "gcc::Assemble", inputs: \["/tmp/foo.s"\], output: "/tmp/foo.o"' %t &&
// RUN: grep 'bind - "gcc::Link", inputs: \["/tmp/foo.o"\], output: "a.out"' %t &&

// RUN: clang-driver -ccc-print-bindings -ccc-no-clang %s &> %t &&
// RUN: grep 'bind - "gcc::Compile", inputs: \[".*bindings.c"\], output: "/tmp/foo.s"' %t &&
// RUN: grep 'bind - "gcc::Assemble", inputs: \["/tmp/foo.s"\], output: "/tmp/foo.o"' %t &&
// RUN: grep 'bind - "gcc::Link", inputs: \["/tmp/foo.o"\], output: "a.out"' %t &&

// RUN: clang-driver -ccc-print-bindings -ccc-no-clang -no-integrated-cpp %s &> %t &&
// RUN: grep 'bind - "gcc::Preprocess", inputs: \[".*bindings.c"\], output: "/tmp/foo.i"' %t &&
// RUN: grep 'bind - "gcc::Compile", inputs: \["/tmp/foo.i"\], output: "/tmp/foo.s"' %t &&
// RUN: grep 'bind - "gcc::Assemble", inputs: \["/tmp/foo.s"\], output: "/tmp/foo.o"' %t &&
// RUN: grep 'bind - "gcc::Link", inputs: \["/tmp/foo.o"\], output: "a.out"' %t &&

// RUN: clang-driver -ccc-print-bindings -ccc-no-clang -no-integrated-cpp -pipe %s &> %t &&
// RUN: grep 'bind - "gcc::Preprocess", inputs: \[".*bindings.c"\], output: (pipe)' %t &&
// RUN: grep 'bind - "gcc::Compile", inputs: \[(pipe)\], output: (pipe)' %t &&
// RUN: grep 'bind - "gcc::Assemble", inputs: \[(pipe)\], output: "/tmp/foo.o"' %t &&
// RUN: grep 'bind - "gcc::Link", inputs: \["/tmp/foo.o"\], output: "a.out"' %t &&

// Clang control options

// RUN: clang-driver -ccc-print-bindings -fsyntax-only %s &> %t &&
// RUN: grep 'bind - "clang", inputs: \[".*bindings.c"\], output: (nothing)' %t &&
// RUN: clang-driver -ccc-print-bindings -ccc-no-clang -fsyntax-only %s &> %t &&
// RUN: grep 'bind - "gcc::Compile", inputs: \[".*bindings.c"\], output: (nothing)' %t &&
// RUN: clang-driver -ccc-print-bindings -ccc-no-clang-cxx -fsyntax-only -x c++ %s &> %t &&
// RUN: grep 'bind - "gcc::Compile", inputs: \[".*bindings.c"\], output: (nothing)' %t &&
// RUN: clang-driver -ccc-print-bindings -ccc-no-clang-cpp -fsyntax-only -no-integrated-cpp -x c++ %s &> %t &&
// RUN: grep 'bind - "gcc::Preprocess", inputs: \[".*bindings.c"\], output: "/tmp/foo.ii"' %t &&
// RUN: grep 'bind - "clang", inputs: \["/tmp/foo.ii"\], output: (nothing)' %t &&
// RUN: clang-driver -ccc-host-triple i386-apple-darwin9 -ccc-print-bindings -ccc-clang-archs i386 %s -S -arch ppc &> %t &&
// RUN: grep 'bind - "gcc::Compile", inputs: \[".*bindings.c"\], output: "bindings.s"' %t &&
// RUN: clang-driver -ccc-host-triple i386-apple-darwin9 -ccc-print-bindings -ccc-clang-archs ppc %s -S -arch ppc &> %t &&
// RUN: grep 'bind - "clang", inputs: \[".*bindings.c"\], output: "bindings.s"' %t &&

// RUN: true
