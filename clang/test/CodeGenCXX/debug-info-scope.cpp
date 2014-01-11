// RUN: %clang_cc1 -g -emit-llvm %s -o -| FileCheck %s
//
// Two variables with the same name in subsequent if staments need to be in separate scopes.
//
// rdar://problem/14024005
//

int printf(const char*, ...);

char *return_char (int input)
{
  if (input%2 == 0)
    return "I am even.\n";
  else
    return "I am odd.\n";
}

int main2() {
// CHECK: [ DW_TAG_auto_variable ] [ptr] [line [[@LINE+2]]]
// CHECK: metadata !{i32 {{.*}}, metadata !{{.*}}, i32 [[@LINE+1]], {{.*}}} ; [ DW_TAG_lexical_block ]
  if (char *ptr = return_char(1)) {
    printf ("%s", ptr);
  }
// CHECK: [ DW_TAG_auto_variable ] [ptr] [line [[@LINE+2]]]
// CHECK: metadata !{i32 {{.*}}, metadata !{{.*}}, i32 [[@LINE+1]], {{.*}}} ; [ DW_TAG_lexical_block ]
  if (char *ptr = return_char(2)) {
    printf ("%s", ptr);
  }
  else printf ("%s", ptr);

  return 0;
}
