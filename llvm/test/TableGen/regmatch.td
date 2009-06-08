// RUN: tblgen %s | grep {Match1 = 1} | count 1
// RUN: tblgen %s | grep {Match2 = 1} | count 1

class Foo<string v> {
      string Value = v;
      int Match1 = !regmatch(".*ps$", v);
      int Match2 = !regmatch(".*pd$", v);
}

def Bar : Foo<"addps">;
def Baz : Foo<"addpd">;
