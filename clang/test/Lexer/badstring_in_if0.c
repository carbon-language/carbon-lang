// RUN: %clang_cc1 -E %s 2>&1 | grep -v '^#' | not grep warning
#if 0

  "

  '

#endif
