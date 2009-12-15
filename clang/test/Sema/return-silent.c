// RUN: %clang_cc1 %s -Wno-return-type -fsyntax-only -verify

int t14() {
  return;
}

void t15() {
  return 1;
}
