; RUN: llvm-as < %s | llc -f -o /dev/null
@llvm.foo =  constant metadata !{i17 123, null, metadata !"foobar"}