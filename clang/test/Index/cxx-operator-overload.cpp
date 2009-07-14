// RUN: clang-cc -emit-pch %S/cxx-operator-overload-input.cpp -o %t.ast &&
// RUN: index-test %t.ast -point-at %S/cxx-operator-overload-input.cpp:8:17 -print-decls | count 2 &&
// RUN: index-test %t.ast -point-at %S/cxx-operator-overload-input.cpp:8:17 -print-decls | grep ':3:9,' &&
// RUN: index-test %t.ast -point-at %S/cxx-operator-overload-input.cpp:8:17 -print-decls | grep ':11:10,' &&

// Yep, we can show references of '+' plus signs that are overloaded, w00t!
// RUN: index-test %t.ast -point-at %S/cxx-operator-overload-input.cpp:3:15 -print-refs | count 2 &&
// RUN: index-test %t.ast -point-at %S/cxx-operator-overload-input.cpp:3:15 -print-refs | grep ':8:17,' &&
// RUN: index-test %t.ast -point-at %S/cxx-operator-overload-input.cpp:3:15 -print-refs | grep ':8:22,'
