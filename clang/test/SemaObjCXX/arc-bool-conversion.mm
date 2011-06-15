// RUN: %clang_cc1 -fobjc-nonfragile-abi -fsyntax-only -fobjc-arc -verify -fblocks -triple x86_64-apple-darwin10.0.0 %s
// rdar://9310049

bool fn(id obj) {
    return (bool)obj;
}

