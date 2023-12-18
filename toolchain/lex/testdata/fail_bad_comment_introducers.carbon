// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// CHECK:STDOUT: - filename: fail_bad_comment_introducers.carbon
// CHECK:STDOUT:   tokens: [
// CHECK:STDOUT:     { index: 0, kind: 'FileStart', line: {{ *\d+}}, column:  1, indent: 1, spelling: '', has_trailing_space: true },
//

// Comments have to have whitespace after `//` currently.
// CHECK:STDERR: fail_bad_comment_introducers.carbon:[[@LINE+3]]:3: ERROR: Whitespace is required after '//'.
// CHECK:STDERR: //abc
// CHECK:STDERR:   ^
//abc

    // CHECK:STDERR: fail_bad_comment_introducers.carbon:[[@LINE+3]]:7: ERROR: Whitespace is required after '//'.
    // CHECK:STDERR:     //indented
    // CHECK:STDERR:       ^
    //indented

// We only want to diagnose these on the first line of a block of similarly
// indented comments.

// CHECK:STDERR: fail_bad_comment_introducers.carbon:[[@LINE+3]]:3: ERROR: Whitespace is required after '//'.
// CHECK:STDERR: //abc
// CHECK:STDERR:   ^
//abc
//123
//dce

        // CHECK:STDERR: fail_bad_comment_introducers.carbon:[[@LINE+3]]:11: ERROR: Whitespace is required after '//'.
        // CHECK:STDERR:         //indented block
        // CHECK:STDERR:           ^
        //indented block
        //indented block
        //indented block
        //indented block

// As we have specialized code to scan short indented comment blocks, make sure
// the behavior is preserved even for (absurdly) large indents.

                                                                                // CHECK:STDERR: fail_bad_comment_introducers.carbon:[[@LINE+3]]:83: ERROR: Whitespace is required after '//'.
                                                                                // CHECK:STDERR:                                                                                 //very indented block
                                                                                // CHECK:STDERR:                                                                                   ^
                                                                                //very indented block
                                                                                //very indented block
                                                                                //very indented block
                                                                                //very indented block

// An extra un-indented comment line to anchor the end of the file checks.

// CHECK:STDOUT:     { index: 1, kind:   'FileEnd', line: {{ *}}[[@LINE+1]], column: {{ *\d+}}, indent: 1, spelling: '' },
// CHECK:STDOUT:   ]
