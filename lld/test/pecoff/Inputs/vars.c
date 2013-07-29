// cl.exe /c vars.c
// link /debug /dll /nodefaultlib /entry:dllmain /export:var,@1,NONAME,DATA /export:fn vars.obj

// will be exported by ordinal
int var = 3;

// will be exported by name
int fn(void) {
  return 4;
}

int dllmain() {
  return 1;
}
