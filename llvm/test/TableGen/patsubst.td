// RUN: tblgen %s | grep {Match1 = "v4f32"} | count 1
// RUN: tblgen %s | grep {Match2 = "v2f64"} | count 1
// RUN: tblgen %s | grep {Match3 = "v4f32 add"} | count 1
// RUN: tblgen %s | grep {Match4 = "v2f64 add"} | count 1

class Foo<string v> {
      string Value = v;
      string Match1 = !patsubst(".*ps$", "v4f32", v);
      string Match2 = !patsubst(".*pd$", "v2f64", v);
      string Match3 = !patsubst("(.*)ps$", "v4f32 $1", v);
      string Match4 = !patsubst("(.*)pd$", "v2f64 $1", v);
}

def Bar : Foo<"addps">;
def Baz : Foo<"addpd">;
