// RUN: clang -static -dM -E -o %t %s &&
// RUN: grep '#define __PIC__' %t | count 0 &&
// RUN: grep '#define __pic__' %t | count 0 &&
// RUN: clang -fpic -dM -E -o %t %s &&
// RUN: grep '#define __PIC__ 1' %t | count 1 &&
// RUN: grep '#define __pic__ 1' %t | count 1 &&
// RUN: clang -fPIC -dM -E -o %t %s &&
// RUN: grep '#define __PIC__ 2' %t | count 1 &&
// RUN: grep '#define __pic__ 2' %t | count 1 &&
// RUN: true
