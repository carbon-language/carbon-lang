// RUN: rm -rf %t
// RUN: sed -e "s:INPUT_DIR:%S/Inputs:g" -e "s:OUT_DIR:%t:g" %S/Inputs/vfsoverlay.yaml > %t.yaml
// RUN: %clang_cc1 -Werror -fmodules -fmodules-cache-path=%t -ivfsoverlay %t.yaml -I %t -fsyntax-only %s
// REQUIRES: shell

@import not_real;

void foo() {
  bar();
}
