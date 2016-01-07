// A basic clang -cc1 command-line.

// RUN: %clang %s -### -target wasm32-unknown-unknown 2>&1 | FileCheck -check-prefix=CC1 %s
// CC1: clang{{.*}} "-cc1" "-triple" "wasm32-unknown-unknown" {{.*}}

// A basic C link command-line.

// RUN: %clang -### -no-canonical-prefixes -target wasm32-unknown-unknown %s 2>&1 | FileCheck -check-prefix=LINK %s
// LINK: clang{{.*}}" "-cc1" {{.*}} "-o" "[[temp:[^"]*]]"
// LINK: lld{{.*}}" "-flavor" "ld" "[[temp]]" "-o" "a.out"

// A basic C link command-line with optimization.

// RUN: %clang -### -O2 -no-canonical-prefixes -target wasm32-unknown-unknown %s 2>&1 | FileCheck -check-prefix=LINK_OPT %s
// LINK_OPT: clang{{.*}}" "-cc1" {{.*}} "-o" "[[temp:[^"]*]]"
// LINK_OPT: lld{{.*}}" "-flavor" "ld" "--gc-sections" "[[temp]]" "-o" "a.out"

// Ditto, but ensure that a user --no-gc-sections comes after the
// default --gc-sections.

// RUN: %clang -### -O2 -no-canonical-prefixes -target wasm32-unknown-unknown -Wl,--no-gc-sections %s 2>&1 | FileCheck -check-prefix=NO_GC_SECTIONS %s
// NO_GC_SECTIONS: clang{{.*}}" "-cc1" {{.*}} "-o" "[[temp:[^"]*]]"
// NO_GC_SECTIONS: lld{{.*}}" "-flavor" "ld" "--gc-sections" "--no-gc-sections" "[[temp]]" "-o" "a.out"
