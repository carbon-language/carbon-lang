class Cls {
public:
    Cls operator +(const Cls &RHS);
};

static void bar() {
    Cls x1, x2, x3;
    Cls x4 = x1 + x2 + x3;
}

Cls Cls::operator +(const Cls &RHS) {
}

// RUN: clang-cc -emit-pch %s -o %t.ast &&
// RUN: index-test %t.ast -point-at %s:8:17 -print-decls | count 2 &&
// RUN: index-test %t.ast -point-at %s:8:17 -print-decls | grep ':3:9,' &&
// RUN: index-test %t.ast -point-at %s:8:17 -print-decls | grep ':11:10,' &&

// Yep, we can show references of '+' plus signs that are overloaded, w00t!
// RUN: index-test %t.ast -point-at %s:3:15 -print-refs | count 2 &&
// RUN: index-test %t.ast -point-at %s:3:15 -print-refs | grep ':8:17,' &&
// RUN: index-test %t.ast -point-at %s:3:15 -print-refs | grep ':8:22,'
