// RUN: %clang_cc1 %s -emit-llvm -o -

typedef struct _zend_ini_entry zend_ini_entry;
struct _zend_ini_entry {
  void *mh_arg1;
};

char a;

const zend_ini_entry ini_entries[] = {
  {  ((char*)&((zend_ini_entry*)0)->mh_arg1 - (char*)(void*)0)},
};

// PR7564
struct GLGENH {
  int : 27;
  int EMHJAA : 1;
};

struct GLGENH ABHFBF = {1};
