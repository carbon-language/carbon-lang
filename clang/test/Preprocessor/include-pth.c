// RUN: clang -x c-header file_to_include.h -o %t && clang -include-pth %t %s -E | grep 'file_to_include'
