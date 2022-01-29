// RUN: %clang_cc1 -fsyntax-only -verify %s

namespace c { double xxx; } // expected-note{{'c::xxx' declared here}}
namespace d { float xxx; }
namespace z { namespace xxx {} }

void crash() {
  switch (xxx) {} // expected-error{{use of undeclared identifier 'xxx'; did you mean }}
}
