package sample api;

interface Printable {
  fn PrintIt[self: Self]();
}

impl String as Printable {
  fn PrintIt[self: String]() {
    Print(self);
  }
}

class Vector(T:! type) {
  var x: T;

  alias PrintIt = Printable.PrintIt;
}

// Conditionally implement the API for certain `T`s.
impl forall [U:! Printable] Vector(U) as Printable {
  fn PrintIt[self: Self]() {
    Print("{");
    self.x.PrintIt();
    Print("}");
  }
}

fn Main() -> i32 {
  var v: Vector(String) = {.x = "test"};
  v.(Printable.PrintIt)();
  v.PrintIt();
  return 42;
}
