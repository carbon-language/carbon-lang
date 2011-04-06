// RUN: %clang_cc1 -fsyntax-only -verify %s -fpascal-strings
const wchar_t *pascalString = L"\pThis is a Pascal string";
