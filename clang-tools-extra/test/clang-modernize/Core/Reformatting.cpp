// RUN: grep -Ev "// *[A-Z-]+:" %s > %t.cpp
// RUN: clang-modernize -format -use-auto %t.cpp
// RUN: FileCheck --strict-whitespace -input-file=%t.cpp %s

// Ensure that -style is forwarded to clang-apply-replacements by using a style
// other than LLVM and ensuring the result is styled as requested.
// RUN: grep -Ev "// *[A-Z-]+:" %s > %t.cpp
// RUN: clang-modernize -format -style=Google -use-nullptr %t.cpp
// RUN: FileCheck --check-prefix=Google --strict-whitespace -input-file=%t.cpp %s

// Ensure -style-config is forwarded to clang-apply-replacements. The .clang-format
// in %S/Inputs is a dump of the Google style so the same test can be used.
// RUN: grep -Ev "// *[A-Z-]+:" %s > %t.cpp
// RUN: clang-modernize -format -style=file -style-config=%S/Inputs -use-nullptr %t.cpp
// RUN: FileCheck --check-prefix=Google --strict-whitespace -input-file=%t.cpp %s

class MyType012345678901234567890123456789 {
public:
  MyType012345678901234567890123456789()
      : iiiiiiiiiiii(0), jjjjjjjjjjjj(0), kkkkkkkkkkkk(0), mmmmmmmmmmmm(0),
        nnnnnnnnnnnn(0) {}
  // Google: iiiiiiiiiiii(nullptr),
  // Google-NEXT: jjjjjjjjjjjj(nullptr),
  // Google-NEXT: kkkkkkkkkkkk(nullptr),
  // Google-NEXT: mmmmmmmmmmmm(nullptr),
  // Google-NEXT: nnnnnnnnnnnn(nullptr) {}

private:
  int *iiiiiiiiiiii;
  int *jjjjjjjjjjjj;
  int *kkkkkkkkkkkk;
  int *mmmmmmmmmmmm;
  int *nnnnnnnnnnnn;
};

int f() {
  MyType012345678901234567890123456789 *a =
      new MyType012345678901234567890123456789();
  // CHECK: {{^\ \ auto\ a\ \=\ new\ MyType012345678901234567890123456789\(\);}}

  delete a;

  return 0;
}
