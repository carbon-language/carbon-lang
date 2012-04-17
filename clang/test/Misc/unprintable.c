// RUN: %clang_cc1 %s 2>&1 | FileCheck -strict-whitespace %s

int main() {
    int i;
    if((i==/*👿*/1));

// CHECK: {{^    if\(\(i==/\*<U\+1F47F>\*/1\)\);}}

// CHECK: {{^        ~\^~~~~~~~~~~~~~~~}}
// CHECK: {{^       ~ \^               ~}}

    /* 👿 */ "👿berhund";

// CHECK: {{^    /\* <U\+1F47F> \*/ "<U\+1F47F>berhund";}}
// CHECK: {{^                    \^~~~~~~~~~~~~~~~~~}}
}