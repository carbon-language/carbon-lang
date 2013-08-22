// RUN: mkdir -p %T/Inputs/conflict
// RUN: sed "s#\$(path)#%/S/Inputs/conflict#" %S/Inputs/conflict/file1.yaml > %T/Inputs/conflict/file1.yaml
// RUN: sed "s#\$(path)#%/S/Inputs/conflict#" %S/Inputs/conflict/file2.yaml > %T/Inputs/conflict/file2.yaml
// RUN: sed "s#\$(path)#%/S/Inputs/conflict#" %S/Inputs/conflict/file3.yaml > %T/Inputs/conflict/file3.yaml
// RUN: sed "s#\$(path)#%/S/Inputs/conflict#" %S/Inputs/conflict/expected.txt > %T/Inputs/conflict/expected.txt
// RUN: not clang-replace %T/Inputs/conflict > %T/Inputs/conflict/output.txt 2>&1
// RUN: diff -b %T/Inputs/conflict/output.txt %T/Inputs/conflict/expected.txt
