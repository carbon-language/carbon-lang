package sample api;

class Foo{
    fn Create() -> Foo{
        return {.x = 12};
    }
    var x: i32;
}

interface Printable {
  fn PrintIt[self: Self]();
}

impl String as Printable {
  fn PrintIt[self: String]() {
    Print(self);
  }
}

impl Foo as Printable{
   fn PrintIt[self: Foo]() {
       Print("foo");
     }
}

class Vector(T:! type) {
  var x: T;
}

// Conditionally implement the API for certain `T`s.
impl forall [U:! Printable] Vector(U) as Printable {
  fn PrintIt[self: Self]() {
    Print("(");
    self.x.PrintIt();
    Print(")");
  }
}

fn Main() -> i32 {
  var v: Vector(String) = {.x = "test"};
  v.PrintIt();
  return 42;
}
