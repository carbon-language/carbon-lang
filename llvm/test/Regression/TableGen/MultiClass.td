// RUN: tblgen %s
class C1<int A, string B> { 
  int bar = A;
  string thestr = B;
}

def T : C1<4, "blah">;

multiclass t<int a> {
  def S1 : C1<a, "foo"> {
    int foo = 4;
    let bar = 1;
  }
  def S2 : C1<a, "bar">;
}

defm FOO : t<42>;
