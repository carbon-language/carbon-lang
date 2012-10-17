// RUN: %clang_cc1 -fsyntax-only -Wno-compare-distinct-pointer-type -verify %s
// rdar://12501960

void Foo(int **thing, const int **thingMax)
{
        if ((thing + 3) > thingMax)
                return;
}
