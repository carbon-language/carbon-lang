// RUN: %clang_cc1 -std=c++0x -E %s 2>&1 | grep 'error: raw string delimiter longer than 16 characters'

const char *str = R"abcdefghijkmnopqrstuvwxyz(abcdef)abcdefghijkmnopqrstuvwxyz";
