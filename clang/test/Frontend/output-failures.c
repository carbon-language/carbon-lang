// RUN: not %clang_cc1 -emit-llvm -o %S/doesnotexist/somename %s 2> %t
// RUN: FileCheck -check-prefix=OUTPUTFAIL -input-file=%t %s

// OUTPUTFAIL: error: unable to open output file '{{.*}}{{[/\\]}}test{{[/\\]}}Frontend{{[/\\]}}doesnotexist{{[/\\]}}somename': '{{[nN]}}o such file or directory'
