// RUN: %clang -ccc-host-triple i386-apple-darwin9 -### -S -msoft-float %s 2> %t.log
// RUN: grep '"-no-implicit-float"' %t.log

// RUN: %clang -ccc-host-triple i386-apple-darwin9 -### -S -msoft-float -mno-soft-float %s 2> %t.log
// RUN: grep '"-no-implicit-float"' %t.log | count 0

// RUN: %clang -ccc-host-triple i386-apple-darwin9 -### -S -mno-soft-float %s -msoft-float 2> %t.log
// RUN: grep '"-no-implicit-float"' %t.log

