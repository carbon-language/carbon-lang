
struct Foo { 
  Foo(int); 
  ~Foo();
};
void foo() {
  struct {
    Foo name;
  } Int[] =  { 1 };
}
