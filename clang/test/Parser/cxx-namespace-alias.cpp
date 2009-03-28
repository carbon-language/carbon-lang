// RUN: clang-cc -parse-noop -verify %s

namespace A = B;

namespace A = !; // expected-error {{expected namespace name}}
namespace A = A::!; // expected-error {{expected namespace name}}


