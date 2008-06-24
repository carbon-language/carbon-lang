// RUN: clang -parse-noop -verify %s 
class C {
public:
protected:
  typedef int A,B;
  static int sf(), u;

  struct S {};
  enum {};
  int; // expected-error {{error: declaration does not declare anything}}

public:
  void m() {
    int l = 2;
  }
  
private:
  int x,f(),y,g();
};
