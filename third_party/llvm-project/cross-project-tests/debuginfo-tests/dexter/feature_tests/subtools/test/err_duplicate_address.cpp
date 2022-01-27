// Purpose:
//      Check that declaring duplicate addresses gives a useful error message.
//
// RUN: not %dexter_regression_test -v -- %s | FileCheck %s --match-full-lines


int main() {
    int *result = new int(0);
    delete result; // DexLabel('test_line')
}

// CHECK: parser error:{{.*}}err_duplicate_address.cpp([[# @LINE + 4]]): Found duplicate address: 'oops'
// CHECK-NEXT: {{Dex}}DeclareAddress('oops', 'result', on_line=ref('test_line'))

// DexDeclareAddress('oops', 'result', on_line=ref('test_line'))
// DexDeclareAddress('oops', 'result', on_line=ref('test_line'))
