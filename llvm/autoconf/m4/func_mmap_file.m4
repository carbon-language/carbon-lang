#
# Check for the ability to mmap a file.  
#
AC_DEFUN([AC_FUNC_MMAP_FILE],
[AC_CACHE_CHECK(for mmap of files,
ac_cv_func_mmap_file,
[AC_LANG_SAVE
  AC_LANG_C
  AC_TRY_RUN([
#ifdef HAVE_SYS_TYPES_H
#include <sys/types.h>
#endif

#ifdef HAVE_SYS_MMAN_H
#include <sys/mman.h>
#endif

#ifdef HAVE_FCNTL_H
#include <fcntl.h>
#endif

  int fd;
  int main () {
  fd = creat ("foo",0777); fd = (int) mmap (0, 1, PROT_READ, MAP_SHARED, fd, 0); unlink ("foo"); return (fd != (int) MAP_FAILED);}],
  ac_cv_func_mmap_file=yes, ac_cv_func_mmap_file=no)
  AC_LANG_RESTORE
])
if test "$ac_cv_func_mmap_file" = yes; then
   AC_DEFINE([HAVE_MMAP_FILE],[],[Define if mmap() can map files into memory])
   AC_SUBST(MMAP_FILE,[yes])
fi
])
