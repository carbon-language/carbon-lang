// RUN: clang %s -emit-llvm

typedef struct _zend_ini_entry zend_ini_entry;
struct _zend_ini_entry {
	void *mh_arg1;
};

char a;

const zend_ini_entry ini_entries[] = {
	{  ((char*)&((zend_ini_entry*)0)->mh_arg1 - (char*)(void*)0)},
	{  ((long long*)&a - (long long*)(void*)2)},
};
