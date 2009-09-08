// Basic binding.
// RUN: clang -ccc-host-triple i386-unknown-unknown -ccc-print-bindings %s 2> %t &&
// RUN: grep '"clang", inputs: \[".*bindings.c"\], output: ".*\.s"' %t &&
// RUN: grep '"gcc::Assemble", inputs: \[".*\.s"\], output: ".*\.o"' %t &&
// RUN: grep '"gcc::Link", inputs: \[".*\.o"\], output: "a.out"' %t &&

// RUN: clang -ccc-host-triple i386-unknown-unknown -ccc-print-bindings -ccc-no-clang %s 2> %t &&
// RUN: grep '"gcc::Compile", inputs: \[".*bindings.c"\], output: ".*\.s"' %t &&
// RUN: grep '"gcc::Assemble", inputs: \[".*\.s"\], output: ".*\.o"' %t &&
// RUN: grep '"gcc::Link", inputs: \[".*\.o"\], output: "a.out"' %t &&

// RUN: clang -ccc-host-triple i386-unknown-unknown -ccc-print-bindings -ccc-no-clang -no-integrated-cpp %s 2> %t &&
// RUN: grep '"gcc::Preprocess", inputs: \[".*bindings.c"\], output: ".*\.i"' %t &&
// RUN: grep '"gcc::Compile", inputs: \[".*\.i"\], output: ".*\.s"' %t &&
// RUN: grep '"gcc::Assemble", inputs: \[".*\.s"\], output: ".*\.o"' %t &&
// RUN: grep '"gcc::Link", inputs: \[".*\.o"\], output: "a.out"' %t &&

// RUN: clang -ccc-host-triple i386-unknown-unknown -ccc-print-bindings -ccc-no-clang -no-integrated-cpp -pipe %s 2> %t &&
// RUN: grep '"gcc::Preprocess", inputs: \[".*bindings.c"\], output: (pipe)' %t &&
// RUN: grep '"gcc::Compile", inputs: \[(pipe)\], output: (pipe)' %t &&
// RUN: grep '"gcc::Assemble", inputs: \[(pipe)\], output: ".*\.o"' %t &&
// RUN: grep '"gcc::Link", inputs: \[".*\.o"\], output: "a.out"' %t &&

// RUN: clang -ccc-host-triple i386-unknown-unknown -ccc-print-bindings -ccc-no-clang -x c-header %s 2> %t &&
// RUN: grep '"gcc::Precompile", inputs: \[".*bindings.c"\], output: ".*bindings.c.gch' %t &&

// Clang control options

// RUN: clang -ccc-host-triple i386-unknown-unknown -ccc-print-bindings -fsyntax-only %s 2> %t &&
// RUN: grep '"clang", inputs: \[".*bindings.c"\], output: (nothing)' %t &&
// RUN: clang -ccc-host-triple i386-unknown-unknown -ccc-print-bindings -ccc-no-clang -fsyntax-only %s 2> %t &&
// RUN: grep '"gcc::Compile", inputs: \[".*bindings.c"\], output: (nothing)' %t &&
// RUN: clang -ccc-host-triple i386-unknown-unknown -ccc-print-bindings -ccc-no-clang-cxx -fsyntax-only -x c++ %s 2> %t &&
// RUN: grep '"gcc::Compile", inputs: \[".*bindings.c"\], output: (nothing)' %t &&
// RUN: clang -ccc-host-triple i386-unknown-unknown -ccc-print-bindings -ccc-clang-cxx -fsyntax-only -x c++ %s 2> %t &&
// RUN: grep '"clang", inputs: \[".*bindings.c"\], output: (nothing)' %t &&
// RUN: clang -ccc-host-triple i386-unknown-unknown -ccc-print-bindings -ccc-no-clang-cpp -fsyntax-only -no-integrated-cpp %s 2> %t &&
// RUN: grep '"gcc::Preprocess", inputs: \[".*bindings.c"\], output: ".*\.i"' %t &&
// RUN: grep '"clang", inputs: \[".*\.i"\], output: (nothing)' %t &&
// RUN: clang -ccc-host-triple i386-apple-darwin9 -ccc-print-bindings -ccc-clang-archs i386 %s -S -arch ppc 2> %t &&
// RUN: grep '"gcc::Compile", inputs: \[".*bindings.c"\], output: "bindings.s"' %t &&
// RUN: clang -ccc-host-triple i386-apple-darwin9 -ccc-print-bindings -ccc-clang-archs powerpc %s -S -arch ppc 2> %t &&
// RUN: grep '"clang", inputs: \[".*bindings.c"\], output: "bindings.s"' %t &&

// RUN: clang -ccc-host-triple powerpc-unknown-unknown -ccc-print-bindings -ccc-clang-archs "" %s -S 2> %t &&
// RUN: grep '"clang", inputs: \[".*bindings.c"\], output: "bindings.s"' %t &&
// RUN: clang -ccc-host-triple powerpc-unknown-unknown -ccc-print-bindings -ccc-clang-archs "i386" %s -S 2> %t &&
// RUN: grep '"gcc::Compile", inputs: \[".*bindings.c"\], output: "bindings.s"' %t &&

// Darwin bindings
// RUN: clang -ccc-host-triple i386-apple-darwin9 -ccc-print-bindings %s 2> %t &&
// RUN: grep '"clang", inputs: \[".*bindings.c"\], output: ".*\.s"' %t &&
// RUN: grep '"darwin::Assemble", inputs: \[".*\.s"\], output: ".*\.o"' %t &&
// RUN: grep '"darwin::Link", inputs: \[".*\.o"\], output: "a.out"' %t &&

// RUN: true
