// RUN: tblgen %s | grep {\\\[1, 2, 3\\\]} | count 4
// RUN: tblgen %s | grep {\\\[4, 5, 6\\\]} | count 2

class A<list<list<int>> vals> {
  list<int> first = vals[0];
  list<int> rest  = !if(!null(!cdr(vals)), vals[0], vals[1]);
}

def One : A<[[1,2,3]]>;
def Two : A<[[1,2,3],[4,5,6]]>;

class B<list<int> v> {
  list<int> vals = v;
}

class BB<list<list<int>> vals> : B<!if(!null(!cdr(vals)), vals[0], vals[1])>;
class BBB<list<list<int>> vals> : BB<vals>;

def OneB : BBB<[[1,2,3]]>;
def TwoB : BBB<[[1,2,3],[4,5,6]]>;
