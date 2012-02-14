// RUN: %clang_cc1 -fsyntax-only -std=c++11 %s -verify

struct DirectInitOnly {
  explicit DirectInitOnly(DirectInitOnly&);
};

void direct_init_capture(DirectInitOnly &dio) {
  [dio] {}();
}
