// RUN: %llvmgcc -S -O2 -g %s -o - | llc -O2 -o %t.s
// RUN: grep  "# DW_TAG_formal_parameter" %t.s | count 4
// Radar 8122864
// XTARGET: x86,darwin
static int foo(int a, int j) {
  int k = 0;
  if (a)
    k = a + j;
  else
    k = j;
  return k;
}
int bar(int o, int p) {

  return foo(o, p);
}
