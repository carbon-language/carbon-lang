// RUN: %clang -### %s -c -o tmp.o -Wa,--noexecstack 2>&1 | grep "mnoexecstack"
