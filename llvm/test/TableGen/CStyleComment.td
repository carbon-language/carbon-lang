// Test that multiline, nested, comments work correctly.
//
// RUN: tblgen < %s
// XFAIL: vg_leak

/* Foo
  bar
  /* 
  blah
  */

  stuff
  */

def x;
