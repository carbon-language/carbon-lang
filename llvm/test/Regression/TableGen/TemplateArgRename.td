// tblgen %s

// Make sure there is no collision between XX and XX.
def S;

class Before<int XX>;
class After : Before<4> {
  dag XX = (S);
}

