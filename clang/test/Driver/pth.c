// Test transparent PTH support.

// RUN: clang-driver -x c-header %s -o %t.h.pch -### 2> %t.log &&
// RUN: grep '".*/clang" .* "-o" ".*\.h\.pch" "-x" "c-header" ".*pth\.c"' %t.log &&

// RUN: touch %t.h.pth &&
// RUN: clang-driver -E -include %t.h %s -### 2> %t.log &&
// RUN: grep '".*/clang" .*"-include-pth" ".*\.h\.pth" .*"-x" "c" ".*pth\.c"' %t.log
