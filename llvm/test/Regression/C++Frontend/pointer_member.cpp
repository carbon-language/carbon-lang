struct B { int i; };
struct D : public B {};
int D::*dp = &D::i;

