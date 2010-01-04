// RUN: %llvmgcc -O2 -S -g %s -o - | llc -o 2010-01-05-LinkageName.s -O0 
// RUN: %compile_c 2010-01-05-LinkageName.s -o 2010-01-05-LinkageName.s

struct tm {};
long mktime(struct tm *) __asm("_mktime$UNIX2003");
tzload(name, sp, doextend){}
long mktime(tmp)
     struct tm *const tmp;
{
  tzset();
}
timelocal(tmp) {
  return mktime(tmp);
}

