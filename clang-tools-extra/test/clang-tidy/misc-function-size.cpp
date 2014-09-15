// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: sed 's#// *[A-Z-][A-Z-]*:.*#//#' %s > %t/t.cpp
// RUN: echo '{ Checks: "-*,misc-function-size", CheckOptions: [{key: misc-function-size.LineThreshold, value: 0}, {key: misc-function-size.StatementThreshold, value: 0}, {key: misc-function-size.BranchThreshold, value: 0}]}' > %t/.clang-tidy
// RUN: clang-tidy %t/t.cpp -- -std=c++11 2>&1 | FileCheck %s -implicit-check-not='{{warning:|error:|note:}}'

void foo1() {
}

void foo2() {;}
// CHECK: warning: function 'foo2' exceeds recommended size/complexity thresholds
// CHECK: note: 1 statements (threshold 0)

void foo3() {
;

}
// CHECK: warning: function 'foo3' exceeds recommended size/complexity thresholds
// CHECK: note: 3 lines including whitespace and comments (threshold 0)
// CHECK: note: 1 statements (threshold 0)

void foo4(int i) { if (i) {} else; {}
}
// CHECK: warning: function 'foo4' exceeds recommended size/complexity thresholds
// CHECK: note: 1 lines including whitespace and comments (threshold 0)
// CHECK: note: 3 statements (threshold 0)
// CHECK: note: 1 branches (threshold 0)

void foo5(int i) {for(;i;)while(i)
do;while(i);
}
// CHECK: warning: function 'foo5' exceeds recommended size/complexity thresholds
// CHECK: note: 2 lines including whitespace and comments (threshold 0)
// CHECK: note: 7 statements (threshold 0)
// CHECK: note: 3 branches (threshold 0)

template <typename T> T foo6(T i) {return i;
}
int x = foo6(0);
// CHECK: warning: function 'foo6' exceeds recommended size/complexity thresholds
// CHECK: note: 1 lines including whitespace and comments (threshold 0)
// CHECK: note: 1 statements (threshold 0)

void bar1() { [](){;;;;;;;;;;;if(1){}}();


}
// CHECK: warning: function 'bar1' exceeds recommended size/complexity thresholds
// CHECK: note: 3 lines including whitespace and comments (threshold 0)
// CHECK: note: 14 statements (threshold 0)
// CHECK: note: 1 branches (threshold 0)

void bar2() { class A { void barx() {;;} }; }
// CHECK: warning: function 'bar2' exceeds recommended size/complexity thresholds
// CHECK: note: 3 statements (threshold 0)
//
// CHECK: warning: function 'barx' exceeds recommended size/complexity thresholds
// CHECK: note: 2 statements (threshold 0)
