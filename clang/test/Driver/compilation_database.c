// RUN: %clang -MD -MP -c -x c %s -xc++ %s -Wall -MJ - 2>&1 | FileCheck %s
// RUN: not %clang -c -x c %s -MJ %s/non-existant 2>&1 | FileCheck --check-prefix=ERROR %s

// CHECK: { "directory": "[[CWD:[^"]+]]",  "file": "[[SRC:[^"]+[/|\\]compilation_database.c]]", "output": "compilation_database.o", "arguments": ["{{[^"]*}}clang{{[^"]*}}", "-xc", "[[SRC]]", "-c", "-Wall", "--target={{[^"]+}}"]},
// CHECK: { "directory": "[[CWD:[^"]+]]",  "file": "[[SRC:[^"]+[/|\\]compilation_database.c]]", "output": "compilation_database.o", "arguments": ["{{[^"]*}}clang{{[^"]*}}", "-xc++", "[[SRC]]", "-c", "-Wall", "--target={{[^"]+}}"]},
// ERROR: error: compilation database '{{.*}}/non-existant' could not be opened:

int main(void) {
  return 0;
}
