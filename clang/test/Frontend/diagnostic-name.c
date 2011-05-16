// RUN: %clang -fsyntax-only -Wunused-parameter -fdiagnostics-show-name %s 2>&1 | grep "\[warn_unused_parameter\]" | count 1
// RUN: %clang -fsyntax-only -Wunused-parameter -fno-diagnostics-show-name %s 2>&1 | grep "\[warn_unused_parameter\]" | count 0
int main(int argc, char *argv[]) {
  return argc;
}
