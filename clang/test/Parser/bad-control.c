/* RUN: clang -parse-noop %s 2>&1 | grep error: | wc -l | grep 2
*/
int foo() { 
break;
}

int foo2() { 
continue;
}
