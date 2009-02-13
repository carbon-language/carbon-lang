// RUN: cp %s %t &&
// RUN: xcc -x c-header %t -o %t.pth &&
// RUN: xcc -### -S -include %t -x c /dev/null &> %t.log &&
// RUN: grep '"-token-cache" ".*/pth.c.out.tmp.pth"' %t.log
// RUN: true
