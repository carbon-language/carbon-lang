// RUN: %clang_cc1 -funknown-anytype -fsyntax-only -fdebugger-support -fdebugger-cast-result-to-id -verify %s

extern __unknown_anytype test0;
extern __unknown_anytype test1();

void test_unknown_anytype_receiver() {
  (void)(int)[[test0 unknownMethod] otherUnknownMethod];;
  (void)(id)[[test1() unknownMethod] otherUnknownMethod];
}
