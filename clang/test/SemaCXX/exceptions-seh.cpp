// RUN: %clang_cc1 -triple x86_64-windows-msvc -fms-extensions -fsyntax-only -verify %s

// Basic usage should work.
int safe_div(int n, int d) {
  int r;
  __try {
    r = n / d;
  } __except(_exception_code() == 0xC0000094) {
    r = 0;
  }
  return r;
}

void might_crash();

// Diagnose obvious builtin mis-usage.
void bad_builtin_scope() {
  __try {
    might_crash();
  } __except(1) {
  }
  _exception_code(); // expected-error {{'_exception_code' only allowed in __except block or filter expression}}
  _exception_info(); // expected-error {{'_exception_info' only allowed in __except filter expression}}
}

// Diagnose obvious builtin misusage in a template.
template <void FN()>
void bad_builtin_scope_template() {
  __try {
    FN();
  } __except(1) {
  }
  _exception_code(); // expected-error {{'_exception_code' only allowed in __except block or filter expression}}
  _exception_info(); // expected-error {{'_exception_info' only allowed in __except filter expression}}
}
void instantiate_bad_scope_tmpl() {
  bad_builtin_scope_template<might_crash>();
}

// FIXME: Diagnose this case. For now we produce undef in codegen.
template <typename T, T FN()>
T func_template() {
  return FN();
}
void inject_builtins() {
  func_template<void *, __exception_info>();
  func_template<unsigned long, __exception_code>();
}
