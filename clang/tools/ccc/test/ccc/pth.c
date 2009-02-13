// RUN: cp %s %t.h &&
// RUN: xcc %t.h &&
// RUN: xcc -### -S -include %t.h -x c /dev/null &> %t.log &&
// RUN: grep '"-token-cache" ".*/pth.c.out.tmp.h.pth"' %t.log
// RUN: true
