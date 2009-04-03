// RUN: clang-cc %s -fsyntax-only -verify

// rdar://6597252
Class test1(Class X) {
  return 1 ? X : X;
}


// rdar://6079877
void test2() {
  id str = @"foo" 
          "bar\0"    // expected-warning {{literal contains NUL character}}
          @"baz"  " blarg";
  id str2 = @"foo" 
            "bar"
           @"baz"
           " b\0larg";  // expected-warning {{literal contains NUL character}}

  
  if (@encode(int) == "foo") { }  // expected-warning {{result of comparison against @encode is unspecified}}
}
