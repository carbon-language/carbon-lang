// RUN: %clang -no-canonical-prefixes -target x86_64-apple-macosx10.7.0 -rewrite-legacy-objc %s -o - -### 2>&1 | \
// RUN:   FileCheck -check-prefix=TEST0 %s
// TEST0: clang{{.*}}" "-cc1"
// TEST0: "-rewrite-objc"
// FIXME: CHECK-NOT is broken somehow, it doesn't work here. Check adjacency instead.
// TEST0: "-fmessage-length" "0" "-stack-protector" "1" "-mstackrealign" "-fblocks" "-fobjc-runtime-has-arc" "-fobjc-runtime-has-weak" "-fobjc-fragile-abi" "-fobjc-default-synthesize-properties" "-fno-objc-infer-related-result-type" "-fobjc-exceptions" "-fexceptions" "-fdiagnostics-show-option"
// TEST0: rewrite-legacy-objc.m"

// RUN: not %clang -ccc-no-clang -target unknown -rewrite-legacy-objc %s -o - -### 2>&1 | \
// RUN:   FileCheck -check-prefix=TEST1 %s
// TEST1: invalid output type 'rewritten-legacy-objc' for use with gcc

// RUN: not %clang -ccc-no-clang -target i386-apple-darwin10 -rewrite-legacy-objc %s -o - -### 2>&1 | \
// RUN:   FileCheck -check-prefix=TEST2 %s
// TEST2: invalid output type 'rewritten-legacy-objc' for use with gcc
