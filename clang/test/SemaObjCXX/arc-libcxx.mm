// RUN: %clang_cc1 -fsyntax-only -fobjc-arc -fobjc-arc-cxxlib=libc++ -fobjc-nonfragile-abi -verify %s

@interface A @end

void f(__strong id &sir, __weak id &wir, __autoreleasing id &air,
       __unsafe_unretained id &uir) {
  __strong id *sip = std::addressof(sir);
  __weak id *wip = std::addressof(wir);
  __autoreleasing id *aip = std::addressof(air);
  __unsafe_unretained id *uip = std::addressof(uir);
}
