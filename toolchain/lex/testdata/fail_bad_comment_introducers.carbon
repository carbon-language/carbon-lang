// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Most of the CHECK lines in this file are produced by switching this to
// auto-update. However, the very last set of tests is specifically designed to
// check behavior during the *transition* between comment block indents, and so
// we don't want CHECK comments to be placed in-between them. After generating
// the checks, this should have auto-update disabled and the last set of lines
// relocated to not intermingle with the final set of test comments.
// NOAUTOUPDATE
//
// CHECK:STDOUT: - filename: fail_bad_comment_introducers.carbon
// CHECK:STDOUT:   tokens: [
// CHECK:STDOUT:     { index: 0, kind: 'StartOfFile', line: {{ *\d+}}, column:  1, indent: 1, spelling: '', has_trailing_space: true },

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
//abc
//abc

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

// Last but not least, we currently only provide this behavior when the comment
// blocks have the *exact* same indent. If the indent changes, we should get a
// new diagnostic for the new "block" comment.
//
// CHECK:STDERR: fail_bad_comment_introducers.carbon:[[@LINE+10]]:3: ERROR: Whitespace is required after '//'.
// CHECK:STDERR: //abc
// CHECK:STDERR:   ^
// CHECK:STDERR: fail_bad_comment_introducers.carbon:[[@LINE+10]]:5: ERROR: Whitespace is required after '//'.
// CHECK:STDERR:   //abc
// CHECK:STDERR:     ^
// CHECK:STDERR: fail_bad_comment_introducers.carbon:[[@LINE+10]]:3: ERROR: Whitespace is required after '//'.
// CHECK:STDERR: //abc
// CHECK:STDERR:   ^

//abc
//abc
//abc
  //abc
  //abc
  //abc
//abc
//abc
//abc

// CHECK:STDOUT:     { index: 1, kind:   'EndOfFile', line: {{ *}}[[@LINE+1]], column: {{ *\d+}}, indent: 1, spelling: '' },
// CHECK:STDOUT:   ]
