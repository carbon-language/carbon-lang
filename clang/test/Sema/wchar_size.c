// RUN: clang %s -fsyntax-only -verify -triple=i686-apple-darwin9

int check_wchar_size[sizeof(*L"") == 4 ? 1 : -1];
