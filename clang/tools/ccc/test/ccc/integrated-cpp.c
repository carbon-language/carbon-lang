// RUN: xcc -fsyntax-only -### %s 2>&1 | count 1 &&
// RUN: xcc -fsyntax-only -### %s -no-integrated-cpp 2>&1 | count 2 &&
// RUN: xcc -fsyntax-only -### %s -save-temps 2>&1 | count 2
