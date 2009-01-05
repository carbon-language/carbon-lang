// RUN: xcc -fsyntax-only -### %s | count 1 &&
// RUN: xcc -fsyntax-only -### %s -no-integrated-cpp | count 2 &&
// RUN: xcc -fsyntax-only -### %s -save-temps | count 2
