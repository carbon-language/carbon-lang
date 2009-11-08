// Test with pch.
// RUN: clang-cc -emit-pch -DFOO -o %t %S/variables.h
// RUN: clang-cc -DBAR=int -include-pch %t -fsyntax-only -pedantic %s
// RUN: clang-cc -DFOO -DBAR=int -include-pch %t -Werror %s
// RUN: not clang-cc -DFOO -DBAR=int -DX=5 -include-pch %t -Werror %s 

BAR bar = 17;

#ifndef FOO
#  error FOO was not defined
#endif

#if FOO != 1
#  error FOO has the wrong definition
#endif

#ifndef BAR
#  error BAR was not defined
#endif
