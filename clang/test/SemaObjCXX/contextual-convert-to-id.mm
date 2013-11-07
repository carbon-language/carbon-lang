// RUN: %clang_cc1 -std=c++11 -fsyntax-only %s -verify

@interface A
- knownMethod;
@end

@interface B
- unknownMethod;
@end

template<typename T> struct RetainPtr {
  explicit operator T*() const;
};

void methodCallToSpecific(RetainPtr<A> a) {
  [a knownMethod];
  [a unknownMethod]; // expected-warning{{'A' may not respond to 'unknownMethod'}}
}

struct Incomplete; // expected-note{{forward declaration}}

void methodCallToIncomplete(Incomplete &incomplete) {
  [incomplete knownMethod]; // expected-error{{incomplete receiver type 'Incomplete'}}
}

struct IdPtr {
  explicit operator id() const;
};

void methodCallToId(IdPtr a) {
  [a knownMethod];
  [a unknownMethod];
}
