// Purpose:
//      Test that a \DexDeclareAddress value can be used to compare two pointer
//      variables that have a fixed offset between them.
//
// REQUIRES: system-linux
//
// RUN: %dexter_regression_test -- %s | FileCheck %s
// CHECK: offset_address.cpp

int main() {
    int *x = new int[5];
    int *y = x + 3;
    delete x; // DexLabel('test_line')
}

// DexDeclareAddress('x', 'x', on_line=ref('test_line'))
// DexExpectWatchValue('x', address('x'), on_line=ref('test_line'))
// DexExpectWatchValue('y', address('x', 12), on_line=ref('test_line'))
