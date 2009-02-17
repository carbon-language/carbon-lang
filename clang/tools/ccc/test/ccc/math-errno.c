// RUN: xcc -ccc-host-system unknown -### %s -S 2>&1 | grep -- "--fmath-errno=1" | count 1 &&
// RUN: xcc -ccc-host-system unknown -### %s -S -fno-math-errno 2>&1 | grep -- "--fmath-errno=0" | count 1 &&
// RUN: xcc -ccc-host-system unknown -### %s -S -fmath-errno -fno-math-errno 2>&1 | grep -- "--fmath-errno=0" | count 1 &&
// RUN: xcc -ccc-host-bits 32 -ccc-host-machine i386 -ccc-host-system darwin -ccc-host-release 10.5.0 -### %s -S 2>&1 | grep -- "--fmath-errno=0" | count 1 &&
// RUN: xcc -ccc-host-bits 32 -ccc-host-machine i386 -ccc-host-system darwin -ccc-host-release 10.5.0 -### %s -S -fmath-errno 2>&1 | grep -- "--fmath-errno=1" | count 1
