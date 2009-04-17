// RUN: env QA_OVERRIDE_GCC3_OPTIONS="+-Os +-Oz +-O +-O3 +-Oignore +a +b +c xb Xa Omagic ^-ccc-print-options  " clang x -O2 b -O3 2> %t &&
// RUN: grep -F 'Option 0 - Name: "<input>", Values: {"x"}' %t &&
// RUN: grep -F 'Option 1 - Name: "-O", Values: {"ignore"}' %t &&
// RUN: grep -F 'Option 2 - Name: "-O", Values: {"magic"}' %t &&
// RUN: true

