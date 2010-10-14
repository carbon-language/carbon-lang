int end;
// RUN: c-index-test -test-annotate-tokens=c-index-spurious-keyword.m:1:5:1:7  %s | grep Identifier:  | count 1
