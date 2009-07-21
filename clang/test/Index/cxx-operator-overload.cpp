// Run lines are sensitive to line numbers and come below the code.

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

// RUN: index-test %t.ast -point-at %s:10:17 -print-decls > %t &&
// RUN: cat %t | count 2 &&
// RUN: grep ':5:9,' %t &&
// RUN: grep ':13:10,' %t &&

// Yep, we can show references of '+' plus signs that are overloaded, w00t!
// RUN: index-test %t.ast -point-at %s:5:15 -print-refs > %t &&
// RUN: cat %t | count 2 &&
// RUN: grep ':10:17,' %t &&
// RUN: grep ':10:22,' %t
