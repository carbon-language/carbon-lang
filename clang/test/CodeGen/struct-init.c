// RUN: clang-cc %s -emit-llvm -o -

typedef struct _zend_ini_entry zend_ini_entry;
struct _zend_ini_entry {
  void *mh_arg1;
};

char a;

const zend_ini_entry ini_entries[] = {
  {  ((char*)&((zend_ini_entry*)0)->mh_arg1 - (char*)(void*)0)},
};
