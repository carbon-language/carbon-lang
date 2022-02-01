// REQUIRES: shell
// RUN: %clang --verbose -DSPACE="a b"  -### %s 2>&1 | FileCheck -check-prefix=SPACE     -strict-whitespace %s
// RUN: %clang --verbose -DQUOTES=\"\"  -### %s 2>&1 | FileCheck -check-prefix=QUOTES    -strict-whitespace %s
// RUN: %clang --verbose -DBACKSLASH=\\ -### %s 2>&1 | FileCheck -check-prefix=BACKSLASH -strict-whitespace %s
// RUN: %clang --verbose -DDOLLAR=\$    -### %s 2>&1 | FileCheck -check-prefix=DOLLAR    -strict-whitespace %s

// SPACE: "-cc1" {{.*}} "-D" "SPACE=a b"
// QUOTES: "-cc1" {{.*}} "-D" "QUOTES=\"\""
// BACKSLASH: "-cc1" {{.*}} "-D" "BACKSLASH=\\"
// DOLLAR: "-cc1" {{.*}} "-D" "DOLLAR=\$"
