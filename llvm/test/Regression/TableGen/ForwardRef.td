// RUN: tblgen %s -o -

class bar {
  list<bar> x;
}

