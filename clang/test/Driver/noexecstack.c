// RUN: clang -### %s -c -o tmp.o -Wa,--noexecstack | grep "mnoexecstack"
