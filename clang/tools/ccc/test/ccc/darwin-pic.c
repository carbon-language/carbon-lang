// RUN: xcc -ccc-host-system darwin -ccc-host-machine i386 -m32 -S %s -o - | grep 'L_g0$non_lazy_ptr-' &&
// RUN: xcc -ccc-host-system darwin -ccc-host-machine i386 -m32 -S %s -o - -fPIC | grep 'L_g0$non_lazy_ptr-' &&
// RUN: xcc -ccc-host-system darwin -ccc-host-machine i386 -m32 -S %s -o - -mdynamic-no-pic | grep 'L_g0$non_lazy_ptr,' &&
// RUN: xcc -ccc-host-system darwin -ccc-host-machine i386 -m32 -S %s -o - -static | grep 'non_lazy_ptr' | count 0 &&
// RUN: xcc -ccc-host-system darwin -ccc-host-machine i386 -m64 -S %s -o - | grep '_g0@GOTPCREL' &&
// RUN: xcc -ccc-host-system darwin -ccc-host-machine i386 -m64 -S %s -o - -fPIC | grep '_g0@GOTPCREL' &&
// RUN: xcc -ccc-host-system darwin -ccc-host-machine i386 -m64 -S %s -o - -mdynamic-no-pic | grep '_g0@GOTPCREL' &&
// RUN: xcc -ccc-host-system darwin -ccc-host-machine i386 -m64 -S %s -o - -static | grep '_g0@GOTPCREL'

int g0;
int f0() { return g0; }
