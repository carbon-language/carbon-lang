// RUN: %clang_cc1 -fsyntax-only -Wuninitialized -fsyntax-only %s

// Test that the CFG builder handles destructors and gotos jumping between
// scope boundaries.  Previously this crashed (PR 10620).
struct S_10620 {
  S_10620(const S_10620 &x);
  ~S_10620();
};
void PR10620(int x, const S_10620& s) {
  if (x) {
    goto done;
  }
  const S_10620 s2(s);
done:
  ;
}
void PR10620_2(int x, const S_10620& s) {
  if (x)
    goto done;
  const S_10620 s2(s);
done:
  ;
}