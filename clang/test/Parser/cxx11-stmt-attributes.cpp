// RUN: %clang_cc1 -fcxx-exceptions -fexceptions -fsyntax-only -verify -std=c++11 %s

void foo(int i) {

  [[unknown_attribute]] ;
  [[unknown_attribute]] { }
  [[unknown_attribute]] if (0) { }
  [[unknown_attribute]] for (;;);
  [[unknown_attribute]] do {
    [[unknown_attribute]] continue;
  } while (0);
  [[unknown_attribute]] while (0);

  [[unknown_attribute]] switch (i) {
    [[unknown_attribute]] case 0:
    [[unknown_attribute]] default:
      [[unknown_attribute]] break;
  }

  [[unknown_attribute]] goto here;
  [[unknown_attribute]] here:

  [[unknown_attribute]] try {
  } catch (...) {
  }

  [[unknown_attribute]] return;


  alignas(8) ; // expected-warning {{attribute aligned cannot be specified on a statement}}
  [[noreturn]] { } // expected-warning {{attribute noreturn cannot be specified on a statement}}
  [[noreturn]] if (0) { } // expected-warning {{attribute noreturn cannot be specified on a statement}}
  [[noreturn]] for (;;); // expected-warning {{attribute noreturn cannot be specified on a statement}}
  [[noreturn]] do { // expected-warning {{attribute noreturn cannot be specified on a statement}}
    [[unavailable]] continue; // TODO: only noreturn, alignas and carries_dependency are parsed in C++ 11 syntax at the moment, hence no warning here
  } while (0);
  [[unknown_attributqqq]] while (0); // TODO: remove 'qqq' part and enjoy 'empty loop body' warning here (DiagnoseEmptyLoopBody)
  [[unknown_attribute]] while (0); // no warning here yet, just an unknown attribute

  [[unused]] switch (i) { // TODO: only noreturn, alignas and carries_dependency are parsed in C++ 11 syntax at the moment, hence no warning here
    [[uuid]] case 0: // TODO: only noreturn, alignas and carries_dependency are parsed in C++ 11 syntax at the moment, hence no warning here
    [[visibility]] default: // TODO: only noreturn, alignas and carries_dependency are parsed in C++ 11 syntax at the moment, hence no warning here
      [[carries_dependency]] break; // expected-warning {{attribute carries_dependency cannot be specified on a statement}}
  }

  [[fastcall]] goto there; // TODO: only noreturn, alignas and carries_dependency are parsed in C++ 11 syntax at the moment, hence no warning here
  [[noinline]] there: // TODO: only noreturn, alignas and carries_dependency are parsed in C++ 11 syntax at the moment, hence no warning here

  [[lock_returned]] try { // TODO: only noreturn, alignas and carries_dependency are parsed in C++ 11 syntax at the moment, hence no warning here
  } catch (...) {
  }

  [[weakref]] return; // TODO: only noreturn, alignas and carries_dependency are parsed in C++ 11 syntax at the moment, hence no warning here
}
