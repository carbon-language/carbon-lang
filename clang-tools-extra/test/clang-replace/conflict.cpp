// RUN: mkdir -p %T/conflict
// RUN: sed "s#\$(path)#%/S/conflict#" %S/conflict/file1.yaml > %T/conflict/file1.yaml
// RUN: sed "s#\$(path)#%/S/conflict#" %S/conflict/file2.yaml > %T/conflict/file2.yaml
// RUN: sed "s#\$(path)#%/S/conflict#" %S/conflict/file3.yaml > %T/conflict/file3.yaml
// RUN: sed "s#\$(path)#%/S/conflict#" %S/conflict/expected.txt > %T/conflict/expected.txt
// RUN: not clang-replace %T/conflict > %T/conflict/output.txt 2>&1
// RUN: diff -b %T/conflict/output.txt %T/conflict/expected.txt
