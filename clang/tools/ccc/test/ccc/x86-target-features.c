// RUN: xcc -ccc-host-machine i386 -### -S %s -mno-red-zone -mno-sse -msse4a -msoft-float &> %t &&
// RUN: grep '"--mattrs=-sse,+sse4a"' %t &&
// RUN: grep '"--disable-red-zone"' %t &&
// RUN: grep '"--soft-float"' %t
