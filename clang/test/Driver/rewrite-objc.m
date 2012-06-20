// RUN: %clang -no-canonical-prefixes -target x86_64-apple-macosx10.7.0 -rewrite-objc %s -o - -### 2>&1 | \
// RUN:   FileCheck -check-prefix=TEST0 %s
// TEST0: clang{{.*}}" "-cc1"
// TEST0: "-rewrite-objc"
// FIXME: CHECK-NOT is broken somehow, it doesn't work here. Check adjacency instead.
// TEST0: "-fmessage-length" "0" "-stack-protector" "1" "-mstackrealign" "-fblocks" "-fobjc-runtime=macosx" "-fobjc-dispatch-method=mixed" "-fobjc-default-synthesize-properties" "-fno-objc-infer-related-result-type" "-fobjc-exceptions" "-fexceptions" "-fdiagnostics-show-option"
// TEST0: rewrite-objc.m"

// RUN: not %clang -ccc-no-clang -target unknown -rewrite-objc %s -o - -### 2>&1 | \
// RUN:   FileCheck -check-prefix=TEST1 %s
// TEST1: invalid output type 'rewritten-objc' for use with gcc

// RUN: not %clang -ccc-no-clang -target i386-apple-darwin10 -rewrite-objc %s -o - -### 2>&1 | \
// RUN:   FileCheck -check-prefix=TEST2 %s
// TEST2: invalid output type 'rewritten-objc' for use with gcc
