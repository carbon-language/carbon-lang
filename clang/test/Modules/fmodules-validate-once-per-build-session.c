#include "foo.h"

// Clear the module cache.
// RUN: rm -rf %t
// RUN: mkdir -p %t/Inputs
// RUN: mkdir -p %t/modules-to-compare

// ===
// Create a module with system headers.
// RUN: echo 'void meow(void);' > %t/Inputs/foo.h
// RUN: echo 'module Foo [system] { header "foo.h" }' > %t/Inputs/module.map

// ===
// Compile the module.
// RUN: %clang_cc1 -cc1 -fmodules -fdisable-module-hash -fmodules-cache-path=%t/modules-cache -fsyntax-only -I %t/Inputs -fbuild-session-timestamp=1390000000 -fmodules-validate-once-per-build-session %s
// RUN: ls -R %t/modules-cache | grep Foo.pcm.timestamp
// RUN: cp %t/modules-cache/Foo.pcm %t/modules-to-compare/Foo-before.pcm

// ===
// Use it, and make sure that we did not recompile it.
// RUN: %clang_cc1 -cc1 -fmodules -fdisable-module-hash -fmodules-cache-path=%t/modules-cache -fsyntax-only -I %t/Inputs -fbuild-session-timestamp=1390000000 -fmodules-validate-once-per-build-session %s
// RUN: ls -R %t/modules-cache | grep Foo.pcm.timestamp
// RUN: find %t/modules-cache -name Foo.pcm | xargs -I {} cp {} %t/modules-to-compare/Foo-after.pcm

// RUN: diff %t/modules-to-compare/Foo-before.pcm %t/modules-to-compare/Foo-after.pcm

// ===
// Change the sources.
// RUN: echo 'void meow2(void);' > %t/Inputs/foo.h

// ===
// Use the module, and make sure that we did not recompile it, even though the sources changed.
// RUN: %clang_cc1 -cc1 -fmodules -fdisable-module-hash -fmodules-cache-path=%t/modules-cache -fsyntax-only -I %t/Inputs -fbuild-session-timestamp=1390000000 -fmodules-validate-once-per-build-session %s
// RUN: ls -R %t/modules-cache | grep Foo.pcm.timestamp
// RUN: find %t/modules-cache -name Foo.pcm | xargs -I {} cp {} %t/modules-to-compare/Foo-after.pcm

// RUN: diff %t/modules-to-compare/Foo-before.pcm %t/modules-to-compare/Foo-after.pcm

// ===
// Recompile the module if the today's date is before 01 January 2030.
// RUN: %clang_cc1 -cc1 -fmodules -fdisable-module-hash -fmodules-cache-path=%t/modules-cache -fsyntax-only -I %t/Inputs -fbuild-session-timestamp=1893456000 -fmodules-validate-once-per-build-session %s
// RUN: ls -R %t/modules-cache | grep Foo.pcm.timestamp
// RUN: find %t/modules-cache -name Foo.pcm | xargs -I {} cp {} %t/modules-to-compare/Foo-after.pcm

// RUN: not diff %t/modules-to-compare/Foo-before.pcm %t/modules-to-compare/Foo-after.pcm
