// RUN: %clang_cc1 -fsyntax-only -verify %s

// FIXME: We could do much better with this, if we recognized
// placeholders somehow. However, we're content with not generating
// bogus 'archaic' warnings with bad location info.
@protocol <#protocol name#> <NSObject> // expected-error {{expected identifier or '('}} \
// expected-error 2{{expected identifier}} \
// expected-warning{{protocol has no object type specified; defaults to qualified 'id'}}
<#methods#>

@end
