// RUN: clang -fsyntax-only -fno-caret-diagnostics %s 2>&1 | grep error | wc -l | grep 2 &&
// RUN: clang -fsyntax-only -fno-caret-diagnostics -pedantic %s 2>&1 | grep warning | wc -l | grep 1

char ((((*X x  ] ))));   // two errors (start pos and end pos).

;   // pedantic warning.

