// cl.exe /c vars.c
// link /dll /nodefaultlib /entry:dllmain /export:var,@1,NONAME,DATA \
//   /export:fn /export:_name_with_underscore vars.obj

// will be exported by ordinal
int var = 3;

// will be exported by name
int fn(void) {
  return 4;
}

// will be exported by name
int _name_with_underscore(void) {
  return 5;
}

int dllmain() {
  return 1;
}
