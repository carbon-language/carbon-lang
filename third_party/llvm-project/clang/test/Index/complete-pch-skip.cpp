namespace ns {
int bar;
}

int main() { return ns:: }
int main2() { return ns::foo(). }
int main3() { PREAMBLE_ }

// RUN: printf "namespace ns { struct foo { int baz }; }\n#define PREAMBLE_MAC" > %t.h
// RUN: c-index-test -write-pch %t.h.pch -x c++-header %t.h
//
// RUN: c-index-test -code-completion-at=%s:5:26 -include %t.h %s | FileCheck -check-prefix=WITH-PCH %s
// WITH-PCH: {TypedText bar}
// WITH-PCH: {TypedText foo}

// RUN: c-index-test -code-completion-at=%s:7:24 -include %t.h %s | FileCheck -check-prefix=WITH-PCH-MACRO %s
// WITH-PCH-MACRO: {TypedText PREAMBLE_MAC}

// RUN: env CINDEXTEST_COMPLETION_SKIP_PREAMBLE=1 c-index-test -code-completion-at=%s:5:26 -include %t.h %s | FileCheck -check-prefix=SKIP-PCH %s
// SKIP-PCH-NOT: foo
// SKIP-PCH: {TypedText bar}
// SKIP-PCH-NOT: foo

// RUN: env CINDEXTEST_COMPLETION_SKIP_PREAMBLE=1 c-index-test -code-completion-at=%s:7:24 -include %t.h %s | FileCheck -check-prefix=SKIP-PCH-MACRO %s
// SKIP-PCH-MACRO-NOT: {TypedText PREAMBLE_MAC}

// Verify that with *no* preamble (no -include flag) we still get local results.
// SkipPreamble used to break this, by making lookup *too* lazy.
// RUN: env CINDEXTEST_COMPLETION_SKIP_PREAMBLE=1 c-index-test -code-completion-at=%s:5:26 %s | FileCheck -check-prefix=NO-PCH %s
// NO-PCH-NOT: foo
// NO-PCH: {TypedText bar}
// NO-PCH-NOT: foo

// Verify that we still get member results from the preamble.
// RUN: env CINDEXTEST_COMPLETION_SKIP_PREAMBLE=1 c-index-test -code-completion-at=%s:6:32 -include %t.h %s | FileCheck -check-prefix=MEMBER %s
// MEMBER: {TypedText baz}

