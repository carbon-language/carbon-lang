// RUN: not xcc -### - &> %t &&
// RUN: grep 'E or -x required when input is from standard input' %t &&
// RUN: xcc -ccc-print-phases -### -E - &> %t && 
// RUN: grep '1: preprocessor.*, {0}, cpp-output' %t &&
// RUN: xcc -ccc-print-phases -### -ObjC -E - &> %t && 
// RUN: grep '1: preprocessor.*, {0}, objective-c-cpp-output' %t &&
// RUN: xcc -ccc-print-phases -### -ObjC -x c -E - &> %t && 
// RUN: grep '1: preprocessor.*, {0}, cpp-output' %t &&

// RUN: true
