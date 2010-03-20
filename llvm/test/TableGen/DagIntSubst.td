// RUN: tblgen %s | grep {dag d = (X 13)}
// XFAIL: vg_leak
def X;

class C<int N> {
  dag d = (X N);
}

def VAL : C<13>;


