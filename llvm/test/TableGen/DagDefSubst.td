// RUN: tblgen %s | grep {dag d = (X Y)}
// RUN: tblgen %s | grep {dag e = (Y X)}
// XFAIL: vg_leak
def X;

class yclass;
def Y : yclass;

class C<yclass N> {
  dag d = (X N);
  dag e = (N X);
}

def VAL : C<Y>;


