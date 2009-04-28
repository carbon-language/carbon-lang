// Test transparent PTH support.

// RUN: clang -ccc-pch-is-pth -x c-header %s -o %t.h.pth -### 2> %t.log &&
// RUN: grep '".*/clang-cc" .* "-o" ".*\.h\.pth" "-x" "c-header" ".*pth\.c"' %t.log &&

// RUN: touch %t.h.pth &&
// RUN: clang -ccc-pch-is-pth -E -include %t.h %s -### 2> %t.log &&
// RUN: grep '".*/clang-cc" .*"-include-pth" ".*\.h\.pth" .*"-x" "c" ".*pth\.c"' %t.log
