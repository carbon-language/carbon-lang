// REQUIRES: shell

// RUN: rm -rf %t
// RUN: mkdir %t
// RUN: echo 'module tmp { header "tmp.h" }' > %t/map
// RUN: touch %t/tmp.h
// RUN: %clang_cc1 -fmodules -DFOO=1 -x c++ -fmodule-name=tmp %t/map -emit-module -o %t/tmp.pcm

// Can use the module.
// RUN: %clang_cc1 -fmodules -DFOO=1 -x c++ -fmodule-map-file=%t/map -fmodule-file=%t/tmp.pcm -verify -I%t %s

// Can use the module if an input file is newer. (This happens on
// remote file systems.)
// RUN: sleep 1
// RUN: touch %t/tmp.h
// RUN: %clang_cc1 -fmodules -DFOO=1 -x c++ -fmodule-map-file=%t/map -fmodule-file=%t/tmp.pcm -verify -I%t %s

// Can use the module if -D flags change.
// RUN: %clang_cc1 -fmodules -DFOO=2 -x c++ -fmodule-map-file=%t/map -fmodule-file=%t/tmp.pcm -verify -I%t %s

// Can use the module if -W flags change.
// RUN: %clang_cc1 -fmodules -Wextra -x c++ -fmodule-map-file=%t/map -fmodule-file=%t/tmp.pcm -verify -I%t %s

// Can use the module if -I flags change.
// RUN: %clang_cc1 -fmodules -I. -x c++ -fmodule-map-file=%t/map -fmodule-file=%t/tmp.pcm -verify -I%t %s

#include "tmp.h" // expected-no-diagnostics
