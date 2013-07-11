// cl.exe /c vars.c
// link.exe /debug /nodefaultlib /entry:dllmain vars.obj
__declspec(dllexport) int var = 3;

__declspec(dllexport) int fn(void) {
  return 4;
}

int dllmain() {
  return 1;
}
