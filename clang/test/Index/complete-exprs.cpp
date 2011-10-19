// Line- and column-sensitive test; run lines follow.

class string {
 public:
  string();
  string(const char *);
  string(const char *, int n);
};

template<typename T>
class vector {
public:
  vector(const T &, unsigned n);
  template<typename InputIterator>
  vector(InputIterator first, InputIterator last);
  void push_back(const T&);
};
template<typename T> void vector<T>::push_back(const T&) { }
void f() {
  
}

int foo();

void g() {
  vector<int>(foo(), foo());
}

struct X {
  void f() const;
};

void X::f() const {

}

// RUN: c-index-test -code-completion-at=%s:20:2 %s -std=c++0x | FileCheck -check-prefix=CHECK-CC1 %s
// RUN: env CINDEXTEST_EDITING=1 CINDEXTEST_COMPLETION_CACHING=1 c-index-test -code-completion-at=%s:20:2 -std=c++0x %s | FileCheck -check-prefix=CHECK-CC1 %s
// CHECK-CC1: NotImplemented:{ResultType size_t}{TypedText alignof}{LeftParen (}{Placeholder type}{RightParen )} (40)
// CHECK-CC1: NotImplemented:{ResultType bool}{TypedText noexcept}{LeftParen (}{Placeholder expression}{RightParen )} (40)
// CHECK-CC1: NotImplemented:{ResultType std::nullptr_t}{TypedText nullptr} (40)
// CHECK-CC1: NotImplemented:{TypedText operator} (40)
// CHECK-CC1-NOT: push_back
// CHECK-CC1: ClassDecl:{TypedText string} (50)
// CHECK-CC1: CXXConstructor:{TypedText string}{LeftParen (}{RightParen )} (50)
// CHECK-CC1: CXXConstructor:{TypedText string}{LeftParen (}{Placeholder const char *}{RightParen )} (50)
// CHECK-CC1: CXXConstructor:{TypedText string}{LeftParen (}{Placeholder const char *}{Comma , }{Placeholder int n}{RightParen )} (50)
// CHECK-CC1: ClassTemplate:{TypedText vector}{LeftAngle <}{Placeholder typename T}{RightAngle >} (50)
// CHECK-CC1: CXXConstructor:{TypedText vector}{LeftAngle <}{Placeholder typename T}{RightAngle >}{LeftParen (}{Placeholder const T &}{Comma , }{Placeholder unsigned int n}{RightParen )} (50)
// CHECK-CC1: FunctionTemplate:{ResultType void}{TypedText vector}{LeftAngle <}{Placeholder typename T}{RightAngle >}{LeftParen (}{Placeholder InputIterator first}{Comma , }{Placeholder InputIterator last}{RightParen )} (50)

// RUN: c-index-test -code-completion-at=%s:19:1 %s | FileCheck -check-prefix=CHECK-CC2 %s
// RUN: env CINDEXTEST_EDITING=1 CINDEXTEST_COMPLETION_CACHING=1 c-index-test -code-completion-at=%s:19:1 %s | FileCheck -check-prefix=CHECK-CC2 %s
// CHECK-CC2: ClassDecl:{TypedText string} (50)
// CHECK-CC2-NOT: CXXConstructor
// CHECK-CC2: ClassTemplate:{TypedText vector}{LeftAngle <}{Placeholder typename T}{RightAngle >} (50)

// RUN: c-index-test -code-completion-at=%s:26:15 %s | FileCheck -check-prefix=CHECK-CC3 %s
// CHECK-CC3: NotImplemented:{TypedText float} (50)
// CHECK-CC3: FunctionDecl:{ResultType int}{TypedText foo}{LeftParen (}{RightParen )} (50)
// CHECK-CC3: FunctionDecl:{ResultType void}{TypedText g}{LeftParen (}{RightParen )} (50)
// CHECK-CC3: ClassTemplate:{TypedText vector}{LeftAngle <}{Placeholder typename T}{RightAngle >} (50)
// CHECK-CC3: CXXConstructor:{TypedText vector}{LeftAngle <}{Placeholder typename T}{RightAngle >}{LeftParen (}{Placeholder const T &}{Comma , }{Placeholder unsigned int n}{RightParen )} (50)
// CHECK-CC3: FunctionTemplate:{ResultType void}{TypedText vector}{LeftAngle <}{Placeholder typename T}{RightAngle >}{LeftParen (}{Placeholder InputIterator first}{Comma , }{Placeholder InputIterator last}{RightParen )} (50)

// RUN: c-index-test -code-completion-at=%s:34:1 %s -std=c++0x | FileCheck -check-prefix=CHECK-CC4 %s
// CHECK-CC4: NotImplemented:{ResultType const X *}{TypedText this} (40)
