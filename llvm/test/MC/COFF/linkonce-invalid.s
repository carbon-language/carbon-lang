// Test invalid use of the .linkonce directive.
//
// RUN: not llvm-mc -triple i386-pc-win32 -filetype=obj %s 2>&1 | FileCheck %s

.section non_comdat

.section comdat
.linkonce discard

.section assoc
.linkonce associative comdat


.section invalid

// CHECK: error: unrecognized COMDAT type 'unknown'
.linkonce unknown

// CHECK: error: unexpected token in directive
.linkonce discard foo

// CHECK: error: expected associated section name
.linkonce associative

// CHECK: error: cannot associate unknown section 'unknown'
.linkonce associative unknown

// CHECK: error: cannot associate a section with itself
.linkonce associative invalid

// CHECK: error: associated section must be a COMDAT section
.linkonce associative non_comdat

// CHECK: error: associated section cannot be itself associative
.linkonce associative assoc

// CHECK: error: section 'multi' is already linkonce
.section multi
.linkonce discard
.linkonce same_size
