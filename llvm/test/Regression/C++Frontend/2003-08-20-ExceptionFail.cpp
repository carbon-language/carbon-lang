void foo();

void bar() {
  struct local {
    ~local() { foo(); }
  } local_obj;

  try {
    foo();
  } catch(...) {
  }
}

