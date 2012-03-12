// RUN: env LIBRARY_PATH=%T/test1 %clang -x c %s -### -o foo 2> %t.log
// RUN: grep '".*ld.*" .*"-L" "%T/test1"' %t.log
