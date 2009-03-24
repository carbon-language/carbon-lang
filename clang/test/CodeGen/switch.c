// RUN: clang-cc %s -emit-llvm-bc -o - | opt -std-compile-opts -disable-output

int foo(int i) {
  int j = 0;
  switch (i) {
  case -1:
    j = 1; break;
  case 1 : 
    j = 2; break;
  case 2:
    j = 3; break;
  default:
    j = 42; break;
  }
  j = j + 1;
  return j;
}

    
int foo2(int i) {
  int j = 0;
  switch (i) {
  case 1 : 
    j = 2; break;
  case 2 ... 10:
    j = 3; break;
  default:
    j = 42; break;
  }
  j = j + 1;
  return j;
}

    
int foo3(int i) {
  int j = 0;
  switch (i) {
  default:
    j = 42; break;
  case 111:
    j = 111; break;
  case 0 ... 100:
    j = 1; break;
  case 222:
    j = 222; break;
  }
  return j;
}


int foo4(int i) {
  int j = 0;
  switch (i) {
  case 111:
    j = 111; break;
  case 0 ... 100:
    j = 1; break;
  case 222:
    j = 222; break;
  default:
    j = 42; break;
  case 501 ... 600:
    j = 5; break;
  }
  return j;
}

void foo5(){
    switch(0){
    default:
        if (0) {

        }
    }
}

void foo6(){
    switch(0){
    }
}

void foo7(){
    switch(0){
      foo7();
    }
}

