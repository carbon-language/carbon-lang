// RUN: %clang -### \
// RUN:   -M -MM %s 2> %t
// RUN: not grep '"-sys-header-deps"' %t
