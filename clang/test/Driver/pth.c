// Test transparent PTH support.

// RUN: clang -x c-header %s -o %t.h.pch -### 2> %t.log &&
// RUN: grep '".*/clang-cc" .* "-o" ".*\.h\.pch" "-x" "c-header" ".*pth\.c"' %t.log &&

// RUN: touch %t.h.pth &&
// RUN: clang -E -include %t.h %s -### 2> %t.log &&
// RUN: grep '".*/clang-cc" .*"-include-pth" ".*\.h\.pth" .*"-x" "c" ".*pth\.c"' %t.log
