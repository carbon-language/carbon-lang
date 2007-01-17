// RUN: not tblgen %s

def {
  bits<2> X = 5;  // bitfield is too small, reject
}
