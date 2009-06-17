// Verify that the analyzer gets the same flags as normal compilation
// (at least for a few key ones).

// RUN: env MACOSX_DEPLOYMENT_TARGET=10.5 clang -ccc-host-triple i386-apple-darwin9  -### --analyze -o /dev/null %s -msse 2> %t.log &&
// RUN: grep '"-analyze"' %t.log &&
// RUN: grep '"--fmath-errno=0"' %t.log &&
// RUN: grep '"-target-feature" "+sse"' %t.log &&
// RUN: grep '"-mmacosx-version-min=10.5"' %t.log

