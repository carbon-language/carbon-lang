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
