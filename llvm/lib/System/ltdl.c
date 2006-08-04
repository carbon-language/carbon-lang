/* ltdl.c -- system independent dlopen wrapper
   Copyright (C) 1998, 1999, 2000, 2004, 2005  Free Software Foundation, Inc.
   Originally by Thomas Tanner <tanner@ffii.org>
   This file is part of GNU Libtool.

This library is free software; you can redistribute it and/or
modify it under the terms of the GNU Lesser General Public
License as published by the Free Software Foundation; either
version 2 of the License, or (at your option) any later version.

As a special exception to the GNU Lesser General Public License,
if you distribute this file as part of a program or library that
is built using GNU libtool, you may include it under the same
distribution terms that you use for the rest of that program.

This library is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public
License along with this library; if not, write to the Free Software
Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA
02110-1301  USA

*/

#include "llvm/Config/config.h"

#if HAVE_CONFIG_H
#  include <config.h>
#endif

#if HAVE_UNISTD_H
#  include <unistd.h>
#endif

#if HAVE_STDIO_H
#  include <stdio.h>
#endif

/* Include the header defining malloc.  On K&R C compilers,
   that's <malloc.h>, on ANSI C and ISO C compilers, that's <stdlib.h>.  */
#if HAVE_STDLIB_H
#  include <stdlib.h>
#else
#  if HAVE_MALLOC_H
#    include <malloc.h>
#  endif
#endif

#if HAVE_STRING_H
#  include <string.h>
#else
#  if HAVE_STRINGS_H
#    include <strings.h>
#  endif
#endif

#if HAVE_CTYPE_H
#  include <ctype.h>
#endif

#if HAVE_MEMORY_H
#  include <memory.h>
#endif

#if HAVE_ERRNO_H
#  include <errno.h>
#endif


#ifndef __WINDOWS__
#  ifdef __WIN32__
#    define __WINDOWS__
#  endif
#endif


#undef LT_USE_POSIX_DIRENT
#ifdef HAVE_CLOSEDIR
#  ifdef HAVE_OPENDIR
#    ifdef HAVE_READDIR
#      ifdef HAVE_DIRENT_H
#        define LT_USE_POSIX_DIRENT
#      endif /* HAVE_DIRENT_H */
#    endif /* HAVE_READDIR */
#  endif /* HAVE_OPENDIR */
#endif /* HAVE_CLOSEDIR */


#undef LT_USE_WINDOWS_DIRENT_EMULATION
#ifndef LT_USE_POSIX_DIRENT
#  ifdef __WINDOWS__
#    define LT_USE_WINDOWS_DIRENT_EMULATION
#  endif /* __WINDOWS__ */
#endif /* LT_USE_POSIX_DIRENT */


#ifdef LT_USE_POSIX_DIRENT
#  include <dirent.h>
#  define LT_D_NAMLEN(dirent) (strlen((dirent)->d_name))
#else
#  ifdef LT_USE_WINDOWS_DIRENT_EMULATION
#    define LT_D_NAMLEN(dirent) (strlen((dirent)->d_name))
#  else
#    define dirent direct
#    define LT_D_NAMLEN(dirent) ((dirent)->d_namlen)
#    if HAVE_SYS_NDIR_H
#      include <sys/ndir.h>
#    endif
#    if HAVE_SYS_DIR_H
#      include <sys/dir.h>
#    endif
#    if HAVE_NDIR_H
#      include <ndir.h>
#    endif
#  endif
#endif

#if HAVE_ARGZ_H
#  include <argz.h>
#endif

#if HAVE_ASSERT_H
#  include <assert.h>
#else
#  define assert(arg)	((void) 0)
#endif

#include "ltdl.h"

#if WITH_DMALLOC
#  include <dmalloc.h>
#endif




/* --- WINDOWS SUPPORT --- */


#ifdef DLL_EXPORT
#  define LT_GLOBAL_DATA	__declspec(dllexport)
#else
#  define LT_GLOBAL_DATA
#endif

/* fopen() mode flags for reading a text file */
#undef	LT_READTEXT_MODE
#ifdef __WINDOWS__
#  define LT_READTEXT_MODE "rt"
#else
#  define LT_READTEXT_MODE "r"
#endif

#ifdef LT_USE_WINDOWS_DIRENT_EMULATION

#include <windows.h>

#define dirent lt_dirent
#define DIR lt_DIR

struct dirent
{
  char d_name[2048];
  int  d_namlen;
};

typedef struct _DIR
{
  HANDLE hSearch;
  WIN32_FIND_DATA Win32FindData;
  BOOL firsttime;
  struct dirent file_info;
} DIR;

#endif /* LT_USE_WINDOWS_DIRENT_EMULATION */


/* --- MANIFEST CONSTANTS --- */


/* Standard libltdl search path environment variable name  */
#undef  LTDL_SEARCHPATH_VAR
#define LTDL_SEARCHPATH_VAR	"LTDL_LIBRARY_PATH"

/* Standard libtool archive file extension.  */
#undef  LTDL_ARCHIVE_EXT
#define LTDL_ARCHIVE_EXT	".la"

/* max. filename length */
#ifndef LT_FILENAME_MAX
#  define LT_FILENAME_MAX	1024
#endif

/* This is the maximum symbol size that won't require malloc/free */
#undef	LT_SYMBOL_LENGTH
#define LT_SYMBOL_LENGTH	128

/* This accounts for the _LTX_ separator */
#undef	LT_SYMBOL_OVERHEAD
#define LT_SYMBOL_OVERHEAD	5




/* --- MEMORY HANDLING --- */


/* These are the functions used internally.  In addition to making
   use of the associated function pointers above, they also perform
   error handling.  */
static char   *lt_estrdup	LT_PARAMS((const char *str));
static lt_ptr lt_emalloc	LT_PARAMS((size_t size));
static lt_ptr lt_erealloc	LT_PARAMS((lt_ptr addr, size_t size));

/* static lt_ptr rpl_realloc	LT_PARAMS((lt_ptr ptr, size_t size)); */
#define rpl_realloc realloc

/* These are the pointers that can be changed by the caller:  */
LT_GLOBAL_DATA lt_ptr (*lt_dlmalloc)	LT_PARAMS((size_t size))
 			= (lt_ptr (*) LT_PARAMS((size_t))) malloc;
LT_GLOBAL_DATA lt_ptr (*lt_dlrealloc)	LT_PARAMS((lt_ptr ptr, size_t size))
 			= (lt_ptr (*) LT_PARAMS((lt_ptr, size_t))) rpl_realloc;
LT_GLOBAL_DATA void   (*lt_dlfree)	LT_PARAMS((lt_ptr ptr))
 			= (void (*) LT_PARAMS((lt_ptr))) free;

/* The following macros reduce the amount of typing needed to cast
   assigned memory.  */
#if WITH_DMALLOC

#define LT_DLMALLOC(tp, n)	((tp *) xmalloc ((n) * sizeof(tp)))
#define LT_DLREALLOC(tp, p, n)	((tp *) xrealloc ((p), (n) * sizeof(tp)))
#define LT_DLFREE(p)						\
	LT_STMT_START { if (p) (p) = (xfree (p), (lt_ptr) 0); } LT_STMT_END

#define LT_EMALLOC(tp, n)	((tp *) xmalloc ((n) * sizeof(tp)))
#define LT_EREALLOC(tp, p, n)	((tp *) xrealloc ((p), (n) * sizeof(tp)))

#else

#define LT_DLMALLOC(tp, n)	((tp *) lt_dlmalloc ((n) * sizeof(tp)))
#define LT_DLREALLOC(tp, p, n)	((tp *) lt_dlrealloc ((p), (n) * sizeof(tp)))
#define LT_DLFREE(p)						\
	LT_STMT_START { if (p) (p) = (lt_dlfree (p), (lt_ptr) 0); } LT_STMT_END

#define LT_EMALLOC(tp, n)	((tp *) lt_emalloc ((n) * sizeof(tp)))
#define LT_EREALLOC(tp, p, n)	((tp *) lt_erealloc ((p), (n) * sizeof(tp)))

#endif

#define LT_DLMEM_REASSIGN(p, q)			LT_STMT_START {	\
	if ((p) != (q)) { if (p) lt_dlfree (p); (p) = (q); (q) = 0; }	\
						} LT_STMT_END


/* --- REPLACEMENT FUNCTIONS --- */


#undef strdup
#define strdup rpl_strdup

static char *strdup LT_PARAMS((const char *str));

static char *
strdup(str)
     const char *str;
{
  char *tmp = 0;

  if (str)
    {
      tmp = LT_DLMALLOC (char, 1+ strlen (str));
      if (tmp)
	{
	  strcpy(tmp, str);
	}
    }

  return tmp;
}


#if ! HAVE_STRCMP

#undef strcmp
#define strcmp rpl_strcmp

static int strcmp LT_PARAMS((const char *str1, const char *str2));

static int
strcmp (str1, str2)
     const char *str1;
     const char *str2;
{
  if (str1 == str2)
    return 0;
  if (str1 == 0)
    return -1;
  if (str2 == 0)
    return 1;

  for (;*str1 && *str2; ++str1, ++str2)
    {
      if (*str1 != *str2)
	break;
    }

  return (int)(*str1 - *str2);
}
#endif


#if ! HAVE_STRCHR

#  if HAVE_INDEX
#    define strchr index
#  else
#    define strchr rpl_strchr

static const char *strchr LT_PARAMS((const char *str, int ch));

static const char*
strchr(str, ch)
     const char *str;
     int ch;
{
  const char *p;

  for (p = str; *p != (char)ch && *p != LT_EOS_CHAR; ++p)
    /*NOWORK*/;

  return (*p == (char)ch) ? p : 0;
}

#  endif
#endif /* !HAVE_STRCHR */


#if ! HAVE_STRRCHR

#  if HAVE_RINDEX
#    define strrchr rindex
#  else
#    define strrchr rpl_strrchr

static const char *strrchr LT_PARAMS((const char *str, int ch));

static const char*
strrchr(str, ch)
     const char *str;
     int ch;
{
  const char *p, *q = 0;

  for (p = str; *p != LT_EOS_CHAR; ++p)
    {
      if (*p == (char) ch)
	{
	  q = p;
	}
    }

  return q;
}

# endif
#endif

/* NOTE:  Neither bcopy nor the memcpy implementation below can
          reliably handle copying in overlapping areas of memory.  Use
          memmove (for which there is a fallback implmentation below)
	  if you need that behaviour.  */
#if ! HAVE_MEMCPY

#  if HAVE_BCOPY
#    define memcpy(dest, src, size)	bcopy (src, dest, size)
#  else
#    define memcpy rpl_memcpy

static lt_ptr memcpy LT_PARAMS((lt_ptr dest, const lt_ptr src, size_t size));

static lt_ptr
memcpy (dest, src, size)
     lt_ptr dest;
     const lt_ptr src;
     size_t size;
{
  const char *	s = src;
  char *	d = dest;
  size_t	i = 0;

  for (i = 0; i < size; ++i)
    {
      d[i] = s[i];
    }

  return dest;
}

#  endif /* !HAVE_BCOPY */
#endif   /* !HAVE_MEMCPY */

#if ! HAVE_MEMMOVE
#  define memmove rpl_memmove

static lt_ptr memmove LT_PARAMS((lt_ptr dest, const lt_ptr src, size_t size));

static lt_ptr
memmove (dest, src, size)
     lt_ptr dest;
     const lt_ptr src;
     size_t size;
{
  const char *	s = src;
  char *	d = dest;
  size_t	i;

  if (d < s)
    for (i = 0; i < size; ++i)
      {
	d[i] = s[i];
      }
  else if (d > s && size > 0)
    for (i = size -1; ; --i)
      {
	d[i] = s[i];
	if (i == 0)
	  break;
      }

  return dest;
}

#endif /* !HAVE_MEMMOVE */

#ifdef LT_USE_WINDOWS_DIRENT_EMULATION

static void closedir LT_PARAMS((DIR *entry));

static void
closedir(entry)
  DIR *entry;
{
  assert(entry != (DIR *) NULL);
  FindClose(entry->hSearch);
  lt_dlfree((lt_ptr)entry);
}


static DIR * opendir LT_PARAMS((const char *path));

static DIR*
opendir (path)
  const char *path;
{
  char file_specification[LT_FILENAME_MAX];
  DIR *entry;

  assert(path != (char *) NULL);
  /* allow space for: path + '\\' '\\' '*' '.' '*' + '\0' */
  (void) strncpy (file_specification, path, LT_FILENAME_MAX-6);
  file_specification[LT_FILENAME_MAX-6] = LT_EOS_CHAR;
  (void) strcat(file_specification,"\\");
  entry = LT_DLMALLOC (DIR,sizeof(DIR));
  if (entry != (DIR *) 0)
    {
      entry->firsttime = TRUE;
      entry->hSearch = FindFirstFile(file_specification,&entry->Win32FindData);
    }
  if (entry->hSearch == INVALID_HANDLE_VALUE)
    {
      (void) strcat(file_specification,"\\*.*");
      entry->hSearch = FindFirstFile(file_specification,&entry->Win32FindData);
      if (entry->hSearch == INVALID_HANDLE_VALUE)
        {
          LT_DLFREE (entry);
          return (DIR *) 0;
        }
    }
  return(entry);
}


static struct dirent *readdir LT_PARAMS((DIR *entry));

static struct dirent *readdir(entry)
  DIR *entry;
{
  int
    status;

  if (entry == (DIR *) 0)
    return((struct dirent *) 0);
  if (!entry->firsttime)
    {
      status = FindNextFile(entry->hSearch,&entry->Win32FindData);
      if (status == 0)
        return((struct dirent *) 0);
    }
  entry->firsttime = FALSE;
  (void) strncpy(entry->file_info.d_name,entry->Win32FindData.cFileName,
    LT_FILENAME_MAX-1);
  entry->file_info.d_name[LT_FILENAME_MAX - 1] = LT_EOS_CHAR;
  entry->file_info.d_namlen = strlen(entry->file_info.d_name);
  return(&entry->file_info);
}

#endif /* LT_USE_WINDOWS_DIRENT_EMULATION */

/* According to Alexandre Oliva <oliva@lsd.ic.unicamp.br>,
    ``realloc is not entirely portable''
   In any case we want to use the allocator supplied by the user without
   burdening them with an lt_dlrealloc function pointer to maintain.
   Instead implement our own version (with known boundary conditions)
   using lt_dlmalloc and lt_dlfree. */

/* #undef realloc
   #define realloc rpl_realloc
*/
#if 0
  /* You can't (re)define realloc unless you also (re)define malloc.
     Right now, this code uses the size of the *destination* to decide
     how much to copy.  That's not right, but you can't know the size
     of the source unless you know enough about, or wrote malloc.  So
     this code is disabled... */

static lt_ptr
realloc (ptr, size)
     lt_ptr ptr;
     size_t size;
{
  if (size == 0)
    {
      /* For zero or less bytes, free the original memory */
      if (ptr != 0)
	{
	  lt_dlfree (ptr);
	}

      return (lt_ptr) 0;
    }
  else if (ptr == 0)
    {
      /* Allow reallocation of a NULL pointer.  */
      return lt_dlmalloc (size);
    }
  else
    {
      /* Allocate a new block, copy and free the old block.  */
      lt_ptr mem = lt_dlmalloc (size);

      if (mem)
	{
	  memcpy (mem, ptr, size);
	  lt_dlfree (ptr);
	}

      /* Note that the contents of PTR are not damaged if there is
	 insufficient memory to realloc.  */
      return mem;
    }
}
#endif


#if ! HAVE_ARGZ_APPEND
#  define argz_append rpl_argz_append

static error_t argz_append LT_PARAMS((char **pargz, size_t *pargz_len,
					const char *buf, size_t buf_len));

static error_t
argz_append (pargz, pargz_len, buf, buf_len)
     char **pargz;
     size_t *pargz_len;
     const char *buf;
     size_t buf_len;
{
  size_t argz_len;
  char  *argz;

  assert (pargz);
  assert (pargz_len);
  assert ((*pargz && *pargz_len) || (!*pargz && !*pargz_len));

  /* If nothing needs to be appended, no more work is required.  */
  if (buf_len == 0)
    return 0;

  /* Ensure there is enough room to append BUF_LEN.  */
  argz_len = *pargz_len + buf_len;
  argz = LT_DLREALLOC (char, *pargz, argz_len);
  if (!argz)
    return ENOMEM;

  /* Copy characters from BUF after terminating '\0' in ARGZ.  */
  memcpy (argz + *pargz_len, buf, buf_len);

  /* Assign new values.  */
  *pargz = argz;
  *pargz_len = argz_len;

  return 0;
}
#endif /* !HAVE_ARGZ_APPEND */


#if ! HAVE_ARGZ_CREATE_SEP
#  define argz_create_sep rpl_argz_create_sep

static error_t argz_create_sep LT_PARAMS((const char *str, int delim,
					    char **pargz, size_t *pargz_len));

static error_t
argz_create_sep (str, delim, pargz, pargz_len)
     const char *str;
     int delim;
     char **pargz;
     size_t *pargz_len;
{
  size_t argz_len;
  char *argz = 0;

  assert (str);
  assert (pargz);
  assert (pargz_len);

  /* Make a copy of STR, but replacing each occurrence of
     DELIM with '\0'.  */
  argz_len = 1+ LT_STRLEN (str);
  if (argz_len)
    {
      const char *p;
      char *q;

      argz = LT_DLMALLOC (char, argz_len);
      if (!argz)
	return ENOMEM;

      for (p = str, q = argz; *p != LT_EOS_CHAR; ++p)
	{
	  if (*p == delim)
	    {
	      /* Ignore leading delimiters, and fold consecutive
		 delimiters in STR into a single '\0' in ARGZ.  */
	      if ((q > argz) && (q[-1] != LT_EOS_CHAR))
		*q++ = LT_EOS_CHAR;
	      else
		--argz_len;
	    }
	  else
	    *q++ = *p;
	}
      /* Copy terminating LT_EOS_CHAR.  */
      *q = *p;
    }

  /* If ARGZ_LEN has shrunk to nothing, release ARGZ's memory.  */
  if (!argz_len)
    LT_DLFREE (argz);

  /* Assign new values.  */
  *pargz = argz;
  *pargz_len = argz_len;

  return 0;
}
#endif /* !HAVE_ARGZ_CREATE_SEP */


#if ! HAVE_ARGZ_INSERT
#  define argz_insert rpl_argz_insert

static error_t argz_insert LT_PARAMS((char **pargz, size_t *pargz_len,
					char *before, const char *entry));

static error_t
argz_insert (pargz, pargz_len, before, entry)
     char **pargz;
     size_t *pargz_len;
     char *before;
     const char *entry;
{
  assert (pargz);
  assert (pargz_len);
  assert (entry && *entry);

  /* No BEFORE address indicates ENTRY should be inserted after the
     current last element.  */
  if (!before)
    return argz_append (pargz, pargz_len, entry, 1+ LT_STRLEN (entry));

  /* This probably indicates a programmer error, but to preserve
     semantics, scan back to the start of an entry if BEFORE points
     into the middle of it.  */
  while ((before > *pargz) && (before[-1] != LT_EOS_CHAR))
    --before;

  {
    size_t entry_len	= 1+ LT_STRLEN (entry);
    size_t argz_len	= *pargz_len + entry_len;
    size_t offset	= before - *pargz;
    char   *argz	= LT_DLREALLOC (char, *pargz, argz_len);

    if (!argz)
      return ENOMEM;

    /* Make BEFORE point to the equivalent offset in ARGZ that it
       used to have in *PARGZ incase realloc() moved the block.  */
    before = argz + offset;

    /* Move the ARGZ entries starting at BEFORE up into the new
       space at the end -- making room to copy ENTRY into the
       resulting gap.  */
    memmove (before + entry_len, before, *pargz_len - offset);
    memcpy  (before, entry, entry_len);

    /* Assign new values.  */
    *pargz = argz;
    *pargz_len = argz_len;
  }

  return 0;
}
#endif /* !HAVE_ARGZ_INSERT */


#if ! HAVE_ARGZ_NEXT
#  define argz_next rpl_argz_next

static char *argz_next LT_PARAMS((char *argz, size_t argz_len,
				    const char *entry));

static char *
argz_next (argz, argz_len, entry)
     char *argz;
     size_t argz_len;
     const char *entry;
{
  assert ((argz && argz_len) || (!argz && !argz_len));

  if (entry)
    {
      /* Either ARGZ/ARGZ_LEN is empty, or ENTRY points into an address
	 within the ARGZ vector.  */
      assert ((!argz && !argz_len)
	      || ((argz <= entry) && (entry < (argz + argz_len))));

      /* Move to the char immediately after the terminating
	 '\0' of ENTRY.  */
      entry = 1+ strchr (entry, LT_EOS_CHAR);

      /* Return either the new ENTRY, or else NULL if ARGZ is
	 exhausted.  */
      return (entry >= argz + argz_len) ? 0 : (char *) entry;
    }
  else
    {
      /* This should probably be flagged as a programmer error,
	 since starting an argz_next loop with the iterator set
	 to ARGZ is safer.  To preserve semantics, handle the NULL
	 case by returning the start of ARGZ (if any).  */
      if (argz_len > 0)
	return argz;
      else
	return 0;
    }
}
#endif /* !HAVE_ARGZ_NEXT */



#if ! HAVE_ARGZ_STRINGIFY
#  define argz_stringify rpl_argz_stringify

static void argz_stringify LT_PARAMS((char *argz, size_t argz_len,
				       int sep));

static void
argz_stringify (argz, argz_len, sep)
     char *argz;
     size_t argz_len;
     int sep;
{
  assert ((argz && argz_len) || (!argz && !argz_len));

  if (sep)
    {
      --argz_len;		/* don't stringify the terminating EOS */
      while (--argz_len > 0)
	{
	  if (argz[argz_len] == LT_EOS_CHAR)
	    argz[argz_len] = sep;
	}
    }
}
#endif /* !HAVE_ARGZ_STRINGIFY */




/* --- TYPE DEFINITIONS -- */


/* This type is used for the array of caller data sets in each handler. */
typedef struct {
  lt_dlcaller_id	key;
  lt_ptr		data;
} lt_caller_data;




/* --- OPAQUE STRUCTURES DECLARED IN LTDL.H --- */


/* Extract the diagnostic strings from the error table macro in the same
   order as the enumerated indices in ltdl.h. */

static const char *lt_dlerror_strings[] =
  {
#define LT_ERROR(name, diagnostic)	(diagnostic),
    lt_dlerror_table
#undef LT_ERROR

    0
  };

/* This structure is used for the list of registered loaders. */
struct lt_dlloader {
  struct lt_dlloader   *next;
  const char	       *loader_name;	/* identifying name for each loader */
  const char	       *sym_prefix;	/* prefix for symbols */
  lt_module_open       *module_open;
  lt_module_close      *module_close;
  lt_find_sym	       *find_sym;
  lt_dlloader_exit     *dlloader_exit;
  lt_user_data		dlloader_data;
};

struct lt_dlhandle_struct {
  struct lt_dlhandle_struct   *next;
  lt_dlloader	       *loader;		/* dlopening interface */
  lt_dlinfo		info;
  int			depcount;	/* number of dependencies */
  lt_dlhandle	       *deplibs;	/* dependencies */
  lt_module		module;		/* system module handle */
  lt_ptr		system;		/* system specific data */
  lt_caller_data       *caller_data;	/* per caller associated data */
  int			flags;		/* various boolean stats */
};

/* Various boolean flags can be stored in the flags field of an
   lt_dlhandle_struct... */
#define LT_DLGET_FLAG(handle, flag) (((handle)->flags & (flag)) == (flag))
#define LT_DLSET_FLAG(handle, flag) ((handle)->flags |= (flag))

#define LT_DLRESIDENT_FLAG	    (0x01 << 0)
/* ...add more flags here... */

#define LT_DLIS_RESIDENT(handle)    LT_DLGET_FLAG(handle, LT_DLRESIDENT_FLAG)


#define LT_DLSTRERROR(name)	lt_dlerror_strings[LT_CONC(LT_ERROR_,name)]

static	const char	objdir[]		= LTDL_OBJDIR;
static	const char	archive_ext[]		= LTDL_ARCHIVE_EXT;
#ifdef	LTDL_SHLIB_EXT
static	const char	shlib_ext[]		= LTDL_SHLIB_EXT;
#endif
#ifdef	LTDL_SYSSEARCHPATH
static	const char	sys_search_path[]	= LTDL_SYSSEARCHPATH;
#endif




/* --- MUTEX LOCKING --- */


/* Macros to make it easier to run the lock functions only if they have
   been registered.  The reason for the complicated lock macro is to
   ensure that the stored error message from the last error is not
   accidentally erased if the current function doesn't generate an
   error of its own.  */
#define LT_DLMUTEX_LOCK()			LT_STMT_START {	\
	if (lt_dlmutex_lock_func) (*lt_dlmutex_lock_func)();	\
						} LT_STMT_END
#define LT_DLMUTEX_UNLOCK()			LT_STMT_START { \
	if (lt_dlmutex_unlock_func) (*lt_dlmutex_unlock_func)();\
						} LT_STMT_END
#define LT_DLMUTEX_SETERROR(errormsg)		LT_STMT_START {	\
	if (lt_dlmutex_seterror_func)				\
		(*lt_dlmutex_seterror_func) (errormsg);		\
	else 	lt_dllast_error = (errormsg);	} LT_STMT_END
#define LT_DLMUTEX_GETERROR(errormsg)		LT_STMT_START {	\
	if (lt_dlmutex_seterror_func)				\
		(errormsg) = (*lt_dlmutex_geterror_func) ();	\
	else	(errormsg) = lt_dllast_error;	} LT_STMT_END

/* The mutex functions stored here are global, and are necessarily the
   same for all threads that wish to share access to libltdl.  */
static	lt_dlmutex_lock	    *lt_dlmutex_lock_func     = 0;
static	lt_dlmutex_unlock   *lt_dlmutex_unlock_func   = 0;
static	lt_dlmutex_seterror *lt_dlmutex_seterror_func = 0;
static	lt_dlmutex_geterror *lt_dlmutex_geterror_func = 0;
static	const char	    *lt_dllast_error	      = 0;


/* Either set or reset the mutex functions.  Either all the arguments must
   be valid functions, or else all can be NULL to turn off locking entirely.
   The registered functions should be manipulating a static global lock
   from the lock() and unlock() callbacks, which needs to be reentrant.  */
int
lt_dlmutex_register (lock, unlock, seterror, geterror)
     lt_dlmutex_lock *lock;
     lt_dlmutex_unlock *unlock;
     lt_dlmutex_seterror *seterror;
     lt_dlmutex_geterror *geterror;
{
  lt_dlmutex_unlock *old_unlock = unlock;
  int		     errors	= 0;

  /* Lock using the old lock() callback, if any.  */
  LT_DLMUTEX_LOCK ();

  if ((lock && unlock && seterror && geterror)
      || !(lock || unlock || seterror || geterror))
    {
      lt_dlmutex_lock_func     = lock;
      lt_dlmutex_unlock_func   = unlock;
      lt_dlmutex_geterror_func = geterror;
    }
  else
    {
      LT_DLMUTEX_SETERROR (LT_DLSTRERROR (INVALID_MUTEX_ARGS));
      ++errors;
    }

  /* Use the old unlock() callback we saved earlier, if any.  Otherwise
     record any errors using internal storage.  */
  if (old_unlock)
    (*old_unlock) ();

  /* Return the number of errors encountered during the execution of
     this function.  */
  return errors;
}




/* --- ERROR HANDLING --- */


static	const char    **user_error_strings	= 0;
static	int		errorcount		= LT_ERROR_MAX;

int
lt_dladderror (diagnostic)
     const char *diagnostic;
{
  int		errindex = 0;
  int		result	 = -1;
  const char  **temp     = (const char **) 0;

  assert (diagnostic);

  LT_DLMUTEX_LOCK ();

  errindex = errorcount - LT_ERROR_MAX;
  temp = LT_EREALLOC (const char *, user_error_strings, 1 + errindex);
  if (temp)
    {
      user_error_strings		= temp;
      user_error_strings[errindex]	= diagnostic;
      result				= errorcount++;
    }

  LT_DLMUTEX_UNLOCK ();

  return result;
}

int
lt_dlseterror (errindex)
     int errindex;
{
  int		errors	 = 0;

  LT_DLMUTEX_LOCK ();

  if (errindex >= errorcount || errindex < 0)
    {
      /* Ack!  Error setting the error message! */
      LT_DLMUTEX_SETERROR (LT_DLSTRERROR (INVALID_ERRORCODE));
      ++errors;
    }
  else if (errindex < LT_ERROR_MAX)
    {
      /* No error setting the error message! */
      LT_DLMUTEX_SETERROR (lt_dlerror_strings[errindex]);
    }
  else
    {
      /* No error setting the error message! */
      LT_DLMUTEX_SETERROR (user_error_strings[errindex - LT_ERROR_MAX]);
    }

  LT_DLMUTEX_UNLOCK ();

  return errors;
}

static lt_ptr
lt_emalloc (size)
     size_t size;
{
  lt_ptr mem = lt_dlmalloc (size);
  if (size && !mem)
    LT_DLMUTEX_SETERROR (LT_DLSTRERROR (NO_MEMORY));
  return mem;
}

static lt_ptr
lt_erealloc (addr, size)
     lt_ptr addr;
     size_t size;
{
  lt_ptr mem = lt_dlrealloc (addr, size);
  if (size && !mem)
    LT_DLMUTEX_SETERROR (LT_DLSTRERROR (NO_MEMORY));
  return mem;
}

static char *
lt_estrdup (str)
     const char *str;
{
  char *copy = strdup (str);
  if (LT_STRLEN (str) && !copy)
    LT_DLMUTEX_SETERROR (LT_DLSTRERROR (NO_MEMORY));
  return copy;
}




/* --- DLOPEN() INTERFACE LOADER --- */


#if HAVE_LIBDL

/* dynamic linking with dlopen/dlsym */

#if HAVE_DLFCN_H
#  include <dlfcn.h>
#endif

#if HAVE_SYS_DL_H
#  include <sys/dl.h>
#endif

#ifdef RTLD_GLOBAL
#  define LT_GLOBAL		RTLD_GLOBAL
#else
#  ifdef DL_GLOBAL
#    define LT_GLOBAL		DL_GLOBAL
#  endif
#endif /* !RTLD_GLOBAL */
#ifndef LT_GLOBAL
#  define LT_GLOBAL		0
#endif /* !LT_GLOBAL */

/* We may have to define LT_LAZY_OR_NOW in the command line if we
   find out it does not work in some platform. */
#ifndef LT_LAZY_OR_NOW
#  ifdef RTLD_LAZY
#    define LT_LAZY_OR_NOW	RTLD_LAZY
#  else
#    ifdef DL_LAZY
#      define LT_LAZY_OR_NOW	DL_LAZY
#    endif
#  endif /* !RTLD_LAZY */
#endif
#ifndef LT_LAZY_OR_NOW
#  ifdef RTLD_NOW
#    define LT_LAZY_OR_NOW	RTLD_NOW
#  else
#    ifdef DL_NOW
#      define LT_LAZY_OR_NOW	DL_NOW
#    endif
#  endif /* !RTLD_NOW */
#endif
#ifndef LT_LAZY_OR_NOW
#  define LT_LAZY_OR_NOW	0
#endif /* !LT_LAZY_OR_NOW */

#if HAVE_DLERROR
#  define DLERROR(arg)	dlerror ()
#else
#  define DLERROR(arg)	LT_DLSTRERROR (arg)
#endif

static lt_module
sys_dl_open (loader_data, filename)
     lt_user_data loader_data;
     const char *filename;
{
  lt_module   module   = dlopen (filename, LT_GLOBAL | LT_LAZY_OR_NOW);

  if (!module)
    {
      LT_DLMUTEX_SETERROR (DLERROR (CANNOT_OPEN));
    }

  return module;
}

static int
sys_dl_close (loader_data, module)
     lt_user_data loader_data;
     lt_module module;
{
  int errors = 0;

  if (dlclose (module) != 0)
    {
      LT_DLMUTEX_SETERROR (DLERROR (CANNOT_CLOSE));
      ++errors;
    }

  return errors;
}

static lt_ptr
sys_dl_sym (loader_data, module, symbol)
     lt_user_data loader_data;
     lt_module module;
     const char *symbol;
{
  lt_ptr address = dlsym (module, symbol);

  if (!address)
    {
      LT_DLMUTEX_SETERROR (DLERROR (SYMBOL_NOT_FOUND));
    }

  return address;
}

static struct lt_user_dlloader sys_dl =
  {
#  ifdef NEED_USCORE
    "_",
#  else
    0,
#  endif
    sys_dl_open, sys_dl_close, sys_dl_sym, 0, 0 };


#endif /* HAVE_LIBDL */



/* --- SHL_LOAD() INTERFACE LOADER --- */

#if HAVE_SHL_LOAD

/* dynamic linking with shl_load (HP-UX) (comments from gmodule) */

#ifdef HAVE_DL_H
#  include <dl.h>
#endif

/* some flags are missing on some systems, so we provide
 * harmless defaults.
 *
 * Mandatory:
 * BIND_IMMEDIATE  - Resolve symbol references when the library is loaded.
 * BIND_DEFERRED   - Delay code symbol resolution until actual reference.
 *
 * Optionally:
 * BIND_FIRST	   - Place the library at the head of the symbol search
 * 		     order.
 * BIND_NONFATAL   - The default BIND_IMMEDIATE behavior is to treat all
 * 		     unsatisfied symbols as fatal.  This flag allows
 * 		     binding of unsatisfied code symbols to be deferred
 * 		     until use.
 *		     [Perl: For certain libraries, like DCE, deferred
 *		     binding often causes run time problems. Adding
 *		     BIND_NONFATAL to BIND_IMMEDIATE still allows
 *		     unresolved references in situations like this.]
 * BIND_NOSTART	   - Do not call the initializer for the shared library
 *		     when the library is loaded, nor on a future call to
 *		     shl_unload().
 * BIND_VERBOSE	   - Print verbose messages concerning possible
 *		     unsatisfied symbols.
 *
 * hp9000s700/hp9000s800:
 * BIND_RESTRICTED - Restrict symbols visible by the library to those
 *		     present at library load time.
 * DYNAMIC_PATH	   - Allow the loader to dynamically search for the
 *		     library specified by the path argument.
 */

#ifndef	DYNAMIC_PATH
#  define DYNAMIC_PATH		0
#endif
#ifndef	BIND_RESTRICTED
#  define BIND_RESTRICTED	0
#endif

#define	LT_BIND_FLAGS	(BIND_IMMEDIATE | BIND_NONFATAL | DYNAMIC_PATH)

static lt_module
sys_shl_open (loader_data, filename)
     lt_user_data loader_data;
     const char *filename;
{
  static shl_t self = (shl_t) 0;
  lt_module module = shl_load (filename, LT_BIND_FLAGS, 0L);

  /* Since searching for a symbol against a NULL module handle will also
     look in everything else that was already loaded and exported with
     the -E compiler flag, we always cache a handle saved before any
     modules are loaded.  */
  if (!self)
    {
      lt_ptr address;
      shl_findsym (&self, "main", TYPE_UNDEFINED, &address);
    }

  if (!filename)
    {
      module = self;
    }
  else
    {
      module = shl_load (filename, LT_BIND_FLAGS, 0L);

      if (!module)
	{
	  LT_DLMUTEX_SETERROR (LT_DLSTRERROR (CANNOT_OPEN));
	}
    }

  return module;
}

static int
sys_shl_close (loader_data, module)
     lt_user_data loader_data;
     lt_module module;
{
  int errors = 0;

  if (module && (shl_unload ((shl_t) (module)) != 0))
    {
      LT_DLMUTEX_SETERROR (LT_DLSTRERROR (CANNOT_CLOSE));
      ++errors;
    }

  return errors;
}

static lt_ptr
sys_shl_sym (loader_data, module, symbol)
     lt_user_data loader_data;
     lt_module module;
     const char *symbol;
{
  lt_ptr address = 0;

  /* sys_shl_open should never return a NULL module handle */
  if (module == (lt_module) 0)
  {
    LT_DLMUTEX_SETERROR (LT_DLSTRERROR (INVALID_HANDLE));
  }
  else if (!shl_findsym((shl_t*) &module, symbol, TYPE_UNDEFINED, &address))
    {
      if (!address)
	{
	  LT_DLMUTEX_SETERROR (LT_DLSTRERROR (SYMBOL_NOT_FOUND));
	}
    }

  return address;
}

static struct lt_user_dlloader sys_shl = {
  0, sys_shl_open, sys_shl_close, sys_shl_sym, 0, 0
};

#endif /* HAVE_SHL_LOAD */




/* --- LOADLIBRARY() INTERFACE LOADER --- */

#ifdef __WINDOWS__

/* dynamic linking for Win32 */

#include <windows.h>

/* Forward declaration; required to implement handle search below. */
static lt_dlhandle handles;

static lt_module
sys_wll_open (loader_data, filename)
     lt_user_data loader_data;
     const char *filename;
{
  lt_dlhandle	cur;
  lt_module	module	   = 0;
  const char   *errormsg   = 0;
  char	       *searchname = 0;
  char	       *ext;
  char		self_name_buf[MAX_PATH];

  if (!filename)
    {
      /* Get the name of main module */
      *self_name_buf = 0;
      GetModuleFileName (NULL, self_name_buf, sizeof (self_name_buf));
      filename = ext = self_name_buf;
    }
  else
    {
      ext = strrchr (filename, '.');
    }

  if (ext)
    {
      /* FILENAME already has an extension. */
      searchname = lt_estrdup (filename);
    }
  else
    {
      /* Append a `.' to stop Windows from adding an
	 implicit `.dll' extension. */
      searchname = LT_EMALLOC (char, 2+ LT_STRLEN (filename));
      if (searchname)
	sprintf (searchname, "%s.", filename);
    }
  if (!searchname)
    return 0;

  {
    /* Silence dialog from LoadLibrary on some failures.
       No way to get the error mode, but to set it,
       so set it twice to preserve any previous flags. */
    UINT errormode = SetErrorMode(SEM_FAILCRITICALERRORS);
    SetErrorMode(errormode | SEM_FAILCRITICALERRORS);

#if defined(__CYGWIN__)
    {
      char wpath[MAX_PATH];
      cygwin_conv_to_full_win32_path (searchname, wpath);
      module = LoadLibrary (wpath);
    }
#else
    module = LoadLibrary (searchname);
#endif

    /* Restore the error mode. */
    SetErrorMode(errormode);
  }

  LT_DLFREE (searchname);

  /* libltdl expects this function to fail if it is unable
     to physically load the library.  Sadly, LoadLibrary
     will search the loaded libraries for a match and return
     one of them if the path search load fails.

     We check whether LoadLibrary is returning a handle to
     an already loaded module, and simulate failure if we
     find one. */
  LT_DLMUTEX_LOCK ();
  cur = handles;
  while (cur)
    {
      if (!cur->module)
	{
	  cur = 0;
	  break;
	}

      if (cur->module == module)
	{
	  break;
	}

      cur = cur->next;
  }
  LT_DLMUTEX_UNLOCK ();

  if (cur || !module)
    {
      LT_DLMUTEX_SETERROR (LT_DLSTRERROR (CANNOT_OPEN));
      module = 0;
    }

  return module;
}

static int
sys_wll_close (loader_data, module)
     lt_user_data loader_data;
     lt_module module;
{
  int	      errors   = 0;

  if (FreeLibrary(module) == 0)
    {
      LT_DLMUTEX_SETERROR (LT_DLSTRERROR (CANNOT_CLOSE));
      ++errors;
    }

  return errors;
}

static lt_ptr
sys_wll_sym (loader_data, module, symbol)
     lt_user_data loader_data;
     lt_module module;
     const char *symbol;
{
  lt_ptr      address  = GetProcAddress (module, symbol);

  if (!address)
    {
      LT_DLMUTEX_SETERROR (LT_DLSTRERROR (SYMBOL_NOT_FOUND));
    }

  return address;
}

static struct lt_user_dlloader sys_wll = {
  0, sys_wll_open, sys_wll_close, sys_wll_sym, 0, 0
};

#endif /* __WINDOWS__ */




/* --- LOAD_ADD_ON() INTERFACE LOADER --- */


#ifdef __BEOS__

/* dynamic linking for BeOS */

#include <kernel/image.h>

static lt_module
sys_bedl_open (loader_data, filename)
     lt_user_data loader_data;
     const char *filename;
{
  image_id image = 0;

  if (filename)
    {
      image = load_add_on (filename);
    }
  else
    {
      image_info info;
      int32 cookie = 0;
      if (get_next_image_info (0, &cookie, &info) == B_OK)
	image = load_add_on (info.name);
    }

  if (image <= 0)
    {
      LT_DLMUTEX_SETERROR (LT_DLSTRERROR (CANNOT_OPEN));
      image = 0;
    }

  return (lt_module) image;
}

static int
sys_bedl_close (loader_data, module)
     lt_user_data loader_data;
     lt_module module;
{
  int errors = 0;

  if (unload_add_on ((image_id) module) != B_OK)
    {
      LT_DLMUTEX_SETERROR (LT_DLSTRERROR (CANNOT_CLOSE));
      ++errors;
    }

  return errors;
}

static lt_ptr
sys_bedl_sym (loader_data, module, symbol)
     lt_user_data loader_data;
     lt_module module;
     const char *symbol;
{
  lt_ptr address = 0;
  image_id image = (image_id) module;

  if (get_image_symbol (image, symbol, B_SYMBOL_TYPE_ANY, address) != B_OK)
    {
      LT_DLMUTEX_SETERROR (LT_DLSTRERROR (SYMBOL_NOT_FOUND));
      address = 0;
    }

  return address;
}

static struct lt_user_dlloader sys_bedl = {
  0, sys_bedl_open, sys_bedl_close, sys_bedl_sym, 0, 0
};

#endif /* __BEOS__ */




/* --- DLD_LINK() INTERFACE LOADER --- */


#if HAVE_DLD

/* dynamic linking with dld */

#if HAVE_DLD_H
#include <dld.h>
#endif

static lt_module
sys_dld_open (loader_data, filename)
     lt_user_data loader_data;
     const char *filename;
{
  lt_module module = strdup (filename);

  if (dld_link (filename) != 0)
    {
      LT_DLMUTEX_SETERROR (LT_DLSTRERROR (CANNOT_OPEN));
      LT_DLFREE (module);
      module = 0;
    }

  return module;
}

static int
sys_dld_close (loader_data, module)
     lt_user_data loader_data;
     lt_module module;
{
  int errors = 0;

  if (dld_unlink_by_file ((char*)(module), 1) != 0)
    {
      LT_DLMUTEX_SETERROR (LT_DLSTRERROR (CANNOT_CLOSE));
      ++errors;
    }
  else
    {
      LT_DLFREE (module);
    }

  return errors;
}

static lt_ptr
sys_dld_sym (loader_data, module, symbol)
     lt_user_data loader_data;
     lt_module module;
     const char *symbol;
{
  lt_ptr address = dld_get_func (symbol);

  if (!address)
    {
      LT_DLMUTEX_SETERROR (LT_DLSTRERROR (SYMBOL_NOT_FOUND));
    }

  return address;
}

static struct lt_user_dlloader sys_dld = {
  0, sys_dld_open, sys_dld_close, sys_dld_sym, 0, 0
};

#endif /* HAVE_DLD */

/* --- DYLD() MACOSX/DARWIN INTERFACE LOADER --- */
#if HAVE_DYLD


#if HAVE_MACH_O_DYLD_H
#if !defined(__APPLE_CC__) && !defined(__MWERKS__) && !defined(__private_extern__)
/* Is this correct? Does it still function properly? */
#define __private_extern__ extern
#endif
# include <mach-o/dyld.h>
#endif
#include <mach-o/getsect.h>

/* We have to put some stuff here that isn't in older dyld.h files */
#ifndef ENUM_DYLD_BOOL
# define ENUM_DYLD_BOOL
# undef FALSE
# undef TRUE
 enum DYLD_BOOL {
    FALSE,
    TRUE
 };
#endif
#ifndef LC_REQ_DYLD
# define LC_REQ_DYLD 0x80000000
#endif
#ifndef LC_LOAD_WEAK_DYLIB
# define LC_LOAD_WEAK_DYLIB (0x18 | LC_REQ_DYLD)
#endif
static const struct mach_header * (*ltdl_NSAddImage)(const char *image_name, unsigned long options) = 0;
static NSSymbol (*ltdl_NSLookupSymbolInImage)(const struct mach_header *image,const char *symbolName, unsigned long options) = 0;
static enum DYLD_BOOL (*ltdl_NSIsSymbolNameDefinedInImage)(const struct mach_header *image, const char *symbolName) = 0;
static enum DYLD_BOOL (*ltdl_NSMakePrivateModulePublic)(NSModule module) = 0;

#ifndef NSADDIMAGE_OPTION_NONE
#define NSADDIMAGE_OPTION_NONE                          0x0
#endif
#ifndef NSADDIMAGE_OPTION_RETURN_ON_ERROR
#define NSADDIMAGE_OPTION_RETURN_ON_ERROR               0x1
#endif
#ifndef NSADDIMAGE_OPTION_WITH_SEARCHING
#define NSADDIMAGE_OPTION_WITH_SEARCHING                0x2
#endif
#ifndef NSADDIMAGE_OPTION_RETURN_ONLY_IF_LOADED
#define NSADDIMAGE_OPTION_RETURN_ONLY_IF_LOADED         0x4
#endif
#ifndef NSADDIMAGE_OPTION_MATCH_FILENAME_BY_INSTALLNAME
#define NSADDIMAGE_OPTION_MATCH_FILENAME_BY_INSTALLNAME 0x8
#endif
#ifndef NSLOOKUPSYMBOLINIMAGE_OPTION_BIND
#define NSLOOKUPSYMBOLINIMAGE_OPTION_BIND            0x0
#endif
#ifndef NSLOOKUPSYMBOLINIMAGE_OPTION_BIND_NOW
#define NSLOOKUPSYMBOLINIMAGE_OPTION_BIND_NOW        0x1
#endif
#ifndef NSLOOKUPSYMBOLINIMAGE_OPTION_BIND_FULLY
#define NSLOOKUPSYMBOLINIMAGE_OPTION_BIND_FULLY      0x2
#endif
#ifndef NSLOOKUPSYMBOLINIMAGE_OPTION_RETURN_ON_ERROR
#define NSLOOKUPSYMBOLINIMAGE_OPTION_RETURN_ON_ERROR 0x4
#endif


static const char *
lt_int_dyld_error(othererror)
	char* othererror;
{
/* return the dyld error string, or the passed in error string if none */
	NSLinkEditErrors ler;
	int lerno;
	const char *errstr;
	const char *file;
	NSLinkEditError(&ler,&lerno,&file,&errstr);
	if (!errstr || !strlen(errstr)) errstr = othererror;
	return errstr;
}

static const struct mach_header *
lt_int_dyld_get_mach_header_from_nsmodule(module)
	NSModule module;
{
/* There should probably be an apple dyld api for this */
	int i=_dyld_image_count();
	int j;
	const char *modname=NSNameOfModule(module);
	const struct mach_header *mh=NULL;
	if (!modname) return NULL;
	for (j = 0; j < i; j++)
	{
		if (!strcmp(_dyld_get_image_name(j),modname))
		{
			mh=_dyld_get_image_header(j);
			break;
		}
	}
	return mh;
}

static const char* lt_int_dyld_lib_install_name(mh)
	const struct mach_header *mh;
{
/* NSAddImage is also used to get the loaded image, but it only works if the lib
   is installed, for uninstalled libs we need to check the install_names against
   each other. Note that this is still broken if DYLD_IMAGE_SUFFIX is set and a
   different lib was loaded as a result
*/
	int j;
	struct load_command *lc;
	unsigned long offset = sizeof(struct mach_header);
	const char* retStr=NULL;
	for (j = 0; j < mh->ncmds; j++)
	{
		lc = (struct load_command*)(((unsigned long)mh) + offset);
		if (LC_ID_DYLIB == lc->cmd)
		{
			retStr=(char*)(((struct dylib_command*)lc)->dylib.name.offset +
									(unsigned long)lc);
		}
		offset += lc->cmdsize;
	}
	return retStr;
}

static const struct mach_header *
lt_int_dyld_match_loaded_lib_by_install_name(const char *name)
{
	int i=_dyld_image_count();
	int j;
	const struct mach_header *mh=NULL;
	const char *id=NULL;
	for (j = 0; j < i; j++)
	{
		id=lt_int_dyld_lib_install_name(_dyld_get_image_header(j));
		if ((id) && (!strcmp(id,name)))
		{
			mh=_dyld_get_image_header(j);
			break;
		}
	}
	return mh;
}

static NSSymbol
lt_int_dyld_NSlookupSymbolInLinkedLibs(symbol,mh)
	const char *symbol;
	const struct mach_header *mh;
{
	/* Safe to assume our mh is good */
	int j;
	struct load_command *lc;
	unsigned long offset = sizeof(struct mach_header);
	NSSymbol retSym = 0;
	const struct mach_header *mh1;
	if ((ltdl_NSLookupSymbolInImage) && NSIsSymbolNameDefined(symbol) )
	{
		for (j = 0; j < mh->ncmds; j++)
		{
			lc = (struct load_command*)(((unsigned long)mh) + offset);
			if ((LC_LOAD_DYLIB == lc->cmd) || (LC_LOAD_WEAK_DYLIB == lc->cmd))
			{
				mh1=lt_int_dyld_match_loaded_lib_by_install_name((char*)(((struct dylib_command*)lc)->dylib.name.offset +
										(unsigned long)lc));
				if (!mh1)
				{
					/* Maybe NSAddImage can find it */
					mh1=ltdl_NSAddImage((char*)(((struct dylib_command*)lc)->dylib.name.offset +
										(unsigned long)lc),
										NSADDIMAGE_OPTION_RETURN_ONLY_IF_LOADED +
										NSADDIMAGE_OPTION_WITH_SEARCHING +
										NSADDIMAGE_OPTION_RETURN_ON_ERROR );
				}
				if (mh1)
				{
					retSym = ltdl_NSLookupSymbolInImage(mh1,
											symbol,
											NSLOOKUPSYMBOLINIMAGE_OPTION_BIND_NOW
											| NSLOOKUPSYMBOLINIMAGE_OPTION_RETURN_ON_ERROR
											);
					if (retSym) break;
				}
			}
			offset += lc->cmdsize;
		}
	}
	return retSym;
}

static int
sys_dyld_init()
{
	int retCode = 0;
	int err = 0;
	if (!_dyld_present()) {
		retCode=1;
	}
	else {
      err = _dyld_func_lookup("__dyld_NSAddImage",(unsigned long*)&ltdl_NSAddImage);
      err = _dyld_func_lookup("__dyld_NSLookupSymbolInImage",(unsigned long*)&ltdl_NSLookupSymbolInImage);
      err = _dyld_func_lookup("__dyld_NSIsSymbolNameDefinedInImage",(unsigned long*)&ltdl_NSIsSymbolNameDefinedInImage);
      err = _dyld_func_lookup("__dyld_NSMakePrivateModulePublic",(unsigned long*)&ltdl_NSMakePrivateModulePublic);
    }
 return retCode;
}

static lt_module
sys_dyld_open (loader_data, filename)
     lt_user_data loader_data;
     const char *filename;
{
	lt_module   module   = 0;
	NSObjectFileImage ofi = 0;
	NSObjectFileImageReturnCode ofirc;

  	if (!filename)
  		return (lt_module)-1;
	ofirc = NSCreateObjectFileImageFromFile(filename, &ofi);
	switch (ofirc)
	{
		case NSObjectFileImageSuccess:
			module = NSLinkModule(ofi, filename,
						NSLINKMODULE_OPTION_RETURN_ON_ERROR
						 | NSLINKMODULE_OPTION_PRIVATE
						 | NSLINKMODULE_OPTION_BINDNOW);
			NSDestroyObjectFileImage(ofi);
			if (module)
				ltdl_NSMakePrivateModulePublic(module);
			break;
		case NSObjectFileImageInappropriateFile:
		    if (ltdl_NSIsSymbolNameDefinedInImage && ltdl_NSLookupSymbolInImage)
		    {
				module = (lt_module)ltdl_NSAddImage(filename, NSADDIMAGE_OPTION_RETURN_ON_ERROR);
				break;
			}
		default:
			LT_DLMUTEX_SETERROR (lt_int_dyld_error(LT_DLSTRERROR(CANNOT_OPEN)));
			return 0;
	}
	if (!module) LT_DLMUTEX_SETERROR (lt_int_dyld_error(LT_DLSTRERROR(CANNOT_OPEN)));
  return module;
}

static int
sys_dyld_close (loader_data, module)
     lt_user_data loader_data;
     lt_module module;
{
	int retCode = 0;
	int flags = 0;
	if (module == (lt_module)-1) return 0;
#ifdef __BIG_ENDIAN__
  	if (((struct mach_header *)module)->magic == MH_MAGIC)
#else
    if (((struct mach_header *)module)->magic == MH_CIGAM)
#endif
	{
	  LT_DLMUTEX_SETERROR("Can not close a dylib");
	  retCode = 1;
	}
	else
	{
#if 1
/* Currently, if a module contains c++ static destructors and it is unloaded, we
   get a segfault in atexit(), due to compiler and dynamic loader differences of
   opinion, this works around that.
*/
		if ((const struct section *)NULL !=
		   getsectbynamefromheader(lt_int_dyld_get_mach_header_from_nsmodule(module),
		   "__DATA","__mod_term_func"))
		{
			flags += NSUNLINKMODULE_OPTION_KEEP_MEMORY_MAPPED;
		}
#endif
#ifdef __ppc__
			flags += NSUNLINKMODULE_OPTION_RESET_LAZY_REFERENCES;
#endif
		if (!NSUnLinkModule(module,flags))
		{
			retCode=1;
			LT_DLMUTEX_SETERROR (lt_int_dyld_error(LT_DLSTRERROR(CANNOT_CLOSE)));
		}
	}

 return retCode;
}

static lt_ptr
sys_dyld_sym (loader_data, module, symbol)
     lt_user_data loader_data;
     lt_module module;
     const char *symbol;
{
	lt_ptr address = 0;
  	NSSymbol *nssym = 0;
  	void *unused;
  	const struct mach_header *mh=NULL;
  	char saveError[256] = "Symbol not found";
  	if (module == (lt_module)-1)
  	{
  		_dyld_lookup_and_bind(symbol,(unsigned long*)&address,&unused);
  		return address;
  	}
#ifdef __BIG_ENDIAN__
  	if (((struct mach_header *)module)->magic == MH_MAGIC)
#else
    if (((struct mach_header *)module)->magic == MH_CIGAM)
#endif
  	{
  	    if (ltdl_NSIsSymbolNameDefinedInImage && ltdl_NSLookupSymbolInImage)
  	    {
  	    	mh=module;
			if (ltdl_NSIsSymbolNameDefinedInImage((struct mach_header*)module,symbol))
			{
				nssym = ltdl_NSLookupSymbolInImage((struct mach_header*)module,
											symbol,
											NSLOOKUPSYMBOLINIMAGE_OPTION_BIND_NOW
											| NSLOOKUPSYMBOLINIMAGE_OPTION_RETURN_ON_ERROR
											);
			}
	    }

  	}
  else {
	nssym = NSLookupSymbolInModule(module, symbol);
	}
	if (!nssym)
	{
		strncpy(saveError, lt_int_dyld_error(LT_DLSTRERROR(SYMBOL_NOT_FOUND)), 255);
		saveError[255] = 0;
		if (!mh) mh=lt_int_dyld_get_mach_header_from_nsmodule(module);
		nssym = lt_int_dyld_NSlookupSymbolInLinkedLibs(symbol,mh);
	}
	if (!nssym)
	{
		LT_DLMUTEX_SETERROR (saveError);
		return NULL;
	}
	return NSAddressOfSymbol(nssym);
}

static struct lt_user_dlloader sys_dyld =
  { "_", sys_dyld_open, sys_dyld_close, sys_dyld_sym, 0, 0 };


#endif /* HAVE_DYLD */


/* --- DLPREOPEN() INTERFACE LOADER --- */


/* emulate dynamic linking using preloaded_symbols */

typedef struct lt_dlsymlists_t
{
  struct lt_dlsymlists_t       *next;
  const lt_dlsymlist	       *syms;
} lt_dlsymlists_t;

static	const lt_dlsymlist     *default_preloaded_symbols	= 0;
static	lt_dlsymlists_t	       *preloaded_symbols		= 0;

static int
presym_init (loader_data)
     lt_user_data loader_data;
{
  int errors = 0;

  LT_DLMUTEX_LOCK ();

  preloaded_symbols = 0;
  if (default_preloaded_symbols)
    {
      errors = lt_dlpreload (default_preloaded_symbols);
    }

  LT_DLMUTEX_UNLOCK ();

  return errors;
}

static int
presym_free_symlists ()
{
  lt_dlsymlists_t *lists;

  LT_DLMUTEX_LOCK ();

  lists = preloaded_symbols;
  while (lists)
    {
      lt_dlsymlists_t	*tmp = lists;

      lists = lists->next;
      LT_DLFREE (tmp);
    }
  preloaded_symbols = 0;

  LT_DLMUTEX_UNLOCK ();

  return 0;
}

static int
presym_exit (loader_data)
     lt_user_data loader_data;
{
  presym_free_symlists ();
  return 0;
}

static int
presym_add_symlist (preloaded)
     const lt_dlsymlist *preloaded;
{
  lt_dlsymlists_t *tmp;
  lt_dlsymlists_t *lists;
  int		   errors   = 0;

  LT_DLMUTEX_LOCK ();

  lists = preloaded_symbols;
  while (lists)
    {
      if (lists->syms == preloaded)
	{
	  goto done;
	}
      lists = lists->next;
    }

  tmp = LT_EMALLOC (lt_dlsymlists_t, 1);
  if (tmp)
    {
      memset (tmp, 0, sizeof(lt_dlsymlists_t));
      tmp->syms = preloaded;
      tmp->next = preloaded_symbols;
      preloaded_symbols = tmp;
    }
  else
    {
      ++errors;
    }

 done:
  LT_DLMUTEX_UNLOCK ();
  return errors;
}

static lt_module
presym_open (loader_data, filename)
     lt_user_data loader_data;
     const char *filename;
{
  lt_dlsymlists_t *lists;
  lt_module	   module = (lt_module) 0;

  LT_DLMUTEX_LOCK ();
  lists = preloaded_symbols;

  if (!lists)
    {
      LT_DLMUTEX_SETERROR (LT_DLSTRERROR (NO_SYMBOLS));
      goto done;
    }

  /* Can't use NULL as the reflective symbol header, as NULL is
     used to mark the end of the entire symbol list.  Self-dlpreopened
     symbols follow this magic number, chosen to be an unlikely
     clash with a real module name.  */
  if (!filename)
    {
      filename = "@PROGRAM@";
    }

  while (lists)
    {
      const lt_dlsymlist *syms = lists->syms;

      while (syms->name)
	{
	  if (!syms->address && strcmp(syms->name, filename) == 0)
	    {
	      module = (lt_module) syms;
	      goto done;
	    }
	  ++syms;
	}

      lists = lists->next;
    }

  LT_DLMUTEX_SETERROR (LT_DLSTRERROR (FILE_NOT_FOUND));

 done:
  LT_DLMUTEX_UNLOCK ();
  return module;
}

static int
presym_close (loader_data, module)
     lt_user_data loader_data;
     lt_module module;
{
  /* Just to silence gcc -Wall */
  module = 0;
  return 0;
}

static lt_ptr
presym_sym (loader_data, module, symbol)
     lt_user_data loader_data;
     lt_module module;
     const char *symbol;
{
  lt_dlsymlist *syms = (lt_dlsymlist*) module;

  ++syms;
  while (syms->address)
    {
      if (strcmp(syms->name, symbol) == 0)
	{
	  return syms->address;
	}

    ++syms;
  }

  LT_DLMUTEX_SETERROR (LT_DLSTRERROR (SYMBOL_NOT_FOUND));

  return 0;
}

static struct lt_user_dlloader presym = {
  0, presym_open, presym_close, presym_sym, presym_exit, 0
};





/* --- DYNAMIC MODULE LOADING --- */


/* The type of a function used at each iteration of  foreach_dirinpath().  */
typedef int	foreach_callback_func LT_PARAMS((char *filename, lt_ptr data1,
						 lt_ptr data2));

static	int	foreach_dirinpath     LT_PARAMS((const char *search_path,
						 const char *base_name,
						 foreach_callback_func *func,
						 lt_ptr data1, lt_ptr data2));

static	int	find_file_callback    LT_PARAMS((char *filename, lt_ptr data,
						 lt_ptr ignored));
static	int	find_handle_callback  LT_PARAMS((char *filename, lt_ptr data,
						 lt_ptr ignored));
static	int	foreachfile_callback  LT_PARAMS((char *filename, lt_ptr data1,
						 lt_ptr data2));


static	int     canonicalize_path     LT_PARAMS((const char *path,
						 char **pcanonical));
static	int	argzize_path 	      LT_PARAMS((const char *path,
						 char **pargz,
						 size_t *pargz_len));
static	FILE   *find_file	      LT_PARAMS((const char *search_path,
						 const char *base_name,
						 char **pdir));
static	lt_dlhandle *find_handle      LT_PARAMS((const char *search_path,
						 const char *base_name,
						 lt_dlhandle *handle));
static	int	find_module	      LT_PARAMS((lt_dlhandle *handle,
						 const char *dir,
						 const char *libdir,
						 const char *dlname,
						 const char *old_name,
						 int installed));
static	int	free_vars	      LT_PARAMS((char *dlname, char *oldname,
						 char *libdir, char *deplibs));
static	int	load_deplibs	      LT_PARAMS((lt_dlhandle handle,
						 char *deplibs));
static	int	trim		      LT_PARAMS((char **dest,
						 const char *str));
static	int	try_dlopen	      LT_PARAMS((lt_dlhandle *handle,
						 const char *filename));
static	int	tryall_dlopen	      LT_PARAMS((lt_dlhandle *handle,
						 const char *filename));
static	int	unload_deplibs	      LT_PARAMS((lt_dlhandle handle));
static	int	lt_argz_insert	      LT_PARAMS((char **pargz,
						 size_t *pargz_len,
						 char *before,
						 const char *entry));
static	int	lt_argz_insertinorder LT_PARAMS((char **pargz,
						 size_t *pargz_len,
						 const char *entry));
static	int	lt_argz_insertdir     LT_PARAMS((char **pargz,
						 size_t *pargz_len,
						 const char *dirnam,
						 struct dirent *dp));
static	int	lt_dlpath_insertdir   LT_PARAMS((char **ppath,
						 char *before,
						 const char *dir));
static	int	list_files_by_dir     LT_PARAMS((const char *dirnam,
						 char **pargz,
						 size_t *pargz_len));
static	int	file_not_found	      LT_PARAMS((void));

static	char	       *user_search_path= 0;
static	lt_dlloader    *loaders		= 0;
static	lt_dlhandle	handles 	= 0;
static	int		initialized 	= 0;

/* Initialize libltdl. */
int
lt_dlinit ()
{
  int	      errors   = 0;

  LT_DLMUTEX_LOCK ();

  /* Initialize only at first call. */
  if (++initialized == 1)
    {
      handles = 0;
      user_search_path = 0; /* empty search path */

#if HAVE_LIBDL
      errors += lt_dlloader_add (lt_dlloader_next (0), &sys_dl, "dlopen");
#endif
#if HAVE_SHL_LOAD
      errors += lt_dlloader_add (lt_dlloader_next (0), &sys_shl, "dlopen");
#endif
#ifdef __WINDOWS__
      errors += lt_dlloader_add (lt_dlloader_next (0), &sys_wll, "dlopen");
#endif
#ifdef __BEOS__
      errors += lt_dlloader_add (lt_dlloader_next (0), &sys_bedl, "dlopen");
#endif
#if HAVE_DLD
      errors += lt_dlloader_add (lt_dlloader_next (0), &sys_dld, "dld");
#endif
#if HAVE_DYLD
       errors += lt_dlloader_add (lt_dlloader_next (0), &sys_dyld, "dyld");
       errors += sys_dyld_init();
#endif
      errors += lt_dlloader_add (lt_dlloader_next (0), &presym, "dlpreload");

      if (presym_init (presym.dlloader_data))
	{
	  LT_DLMUTEX_SETERROR (LT_DLSTRERROR (INIT_LOADER));
	  ++errors;
	}
      else if (errors != 0)
	{
	  LT_DLMUTEX_SETERROR (LT_DLSTRERROR (DLOPEN_NOT_SUPPORTED));
	  ++errors;
	}
    }

  LT_DLMUTEX_UNLOCK ();

  return errors;
}

int
lt_dlpreload (preloaded)
     const lt_dlsymlist *preloaded;
{
  int errors = 0;

  if (preloaded)
    {
      errors = presym_add_symlist (preloaded);
    }
  else
    {
      presym_free_symlists();

      LT_DLMUTEX_LOCK ();
      if (default_preloaded_symbols)
	{
	  errors = lt_dlpreload (default_preloaded_symbols);
	}
      LT_DLMUTEX_UNLOCK ();
    }

  return errors;
}

int
lt_dlpreload_default (preloaded)
     const lt_dlsymlist *preloaded;
{
  LT_DLMUTEX_LOCK ();
  default_preloaded_symbols = preloaded;
  LT_DLMUTEX_UNLOCK ();
  return 0;
}

int
lt_dlexit ()
{
  /* shut down libltdl */
  lt_dlloader *loader;
  int	       errors   = 0;

  LT_DLMUTEX_LOCK ();
  loader = loaders;

  if (!initialized)
    {
      LT_DLMUTEX_SETERROR (LT_DLSTRERROR (SHUTDOWN));
      ++errors;
      goto done;
    }

  /* shut down only at last call. */
  if (--initialized == 0)
    {
      int	level;

      while (handles && LT_DLIS_RESIDENT (handles))
	{
	  handles = handles->next;
	}

      /* close all modules */
      for (level = 1; handles; ++level)
	{
	  lt_dlhandle cur = handles;
	  int saw_nonresident = 0;

	  while (cur)
	    {
	      lt_dlhandle tmp = cur;
	      cur = cur->next;
	      if (!LT_DLIS_RESIDENT (tmp))
		saw_nonresident = 1;
	      if (!LT_DLIS_RESIDENT (tmp) && tmp->info.ref_count <= level)
		{
		  if (lt_dlclose (tmp))
		    {
		      ++errors;
		    }
		}
	    }
	  /* done if only resident modules are left */
	  if (!saw_nonresident)
	    break;
	}

      /* close all loaders */
      while (loader)
	{
	  lt_dlloader *next = loader->next;
	  lt_user_data data = loader->dlloader_data;
	  if (loader->dlloader_exit && loader->dlloader_exit (data))
	    {
	      ++errors;
	    }

	  LT_DLMEM_REASSIGN (loader, next);
	}
      loaders = 0;
    }

 done:
  LT_DLMUTEX_UNLOCK ();
  return errors;
}

static int
tryall_dlopen (handle, filename)
     lt_dlhandle *handle;
     const char *filename;
{
  lt_dlhandle	 cur;
  lt_dlloader   *loader;
  const char	*saved_error;
  int		 errors		= 0;

  LT_DLMUTEX_GETERROR (saved_error);
  LT_DLMUTEX_LOCK ();

  cur	 = handles;
  loader = loaders;

  /* check whether the module was already opened */
  while (cur)
    {
      /* try to dlopen the program itself? */
      if (!cur->info.filename && !filename)
	{
	  break;
	}

      if (cur->info.filename && filename
	  && strcmp (cur->info.filename, filename) == 0)
	{
	  break;
	}

      cur = cur->next;
    }

  if (cur)
    {
      ++cur->info.ref_count;
      *handle = cur;
      goto done;
    }

  cur = *handle;
  if (filename)
    {
      /* Comment out the check of file permissions using access.
	 This call seems to always return -1 with error EACCES.
      */
      /* We need to catch missing file errors early so that
	 file_not_found() can detect what happened.
      if (access (filename, R_OK) != 0)
	{
	  LT_DLMUTEX_SETERROR (LT_DLSTRERROR (FILE_NOT_FOUND));
	  ++errors;
	  goto done;
	} */

      cur->info.filename = lt_estrdup (filename);
      if (!cur->info.filename)
	{
	  ++errors;
	  goto done;
	}
    }
  else
    {
      cur->info.filename = 0;
    }

  while (loader)
    {
      lt_user_data data = loader->dlloader_data;

      cur->module = loader->module_open (data, filename);

      if (cur->module != 0)
	{
	  break;
	}
      loader = loader->next;
    }

  if (!loader)
    {
      LT_DLFREE (cur->info.filename);
      ++errors;
      goto done;
    }

  cur->loader	= loader;
  LT_DLMUTEX_SETERROR (saved_error);

 done:
  LT_DLMUTEX_UNLOCK ();

  return errors;
}

static int
tryall_dlopen_module (handle, prefix, dirname, dlname)
     lt_dlhandle *handle;
     const char *prefix;
     const char *dirname;
     const char *dlname;
{
  int      error	= 0;
  char     *filename	= 0;
  size_t   filename_len	= 0;
  size_t   dirname_len	= LT_STRLEN (dirname);

  assert (handle);
  assert (dirname);
  assert (dlname);
#ifdef LT_DIRSEP_CHAR
  /* Only canonicalized names (i.e. with DIRSEP chars already converted)
     should make it into this function:  */
  assert (strchr (dirname, LT_DIRSEP_CHAR) == 0);
#endif

  if (dirname_len > 0)
    if (dirname[dirname_len -1] == '/')
      --dirname_len;
  filename_len = dirname_len + 1 + LT_STRLEN (dlname);

  /* Allocate memory, and combine DIRNAME and MODULENAME into it.
     The PREFIX (if any) is handled below.  */
  filename  = LT_EMALLOC (char, dirname_len + 1 + filename_len + 1);
  if (!filename)
    return 1;

  sprintf (filename, "%.*s/%s", (int) dirname_len, dirname, dlname);

  /* Now that we have combined DIRNAME and MODULENAME, if there is
     also a PREFIX to contend with, simply recurse with the arguments
     shuffled.  Otherwise, attempt to open FILENAME as a module.  */
  if (prefix)
    {
      error += tryall_dlopen_module (handle,
				     (const char *) 0, prefix, filename);
    }
  else if (tryall_dlopen (handle, filename) != 0)
    {
      ++error;
    }

  LT_DLFREE (filename);
  return error;
}

static int
find_module (handle, dir, libdir, dlname, old_name, installed)
     lt_dlhandle *handle;
     const char *dir;
     const char *libdir;
     const char *dlname;
     const char *old_name;
     int installed;
{
  /* Try to open the old library first; if it was dlpreopened,
     we want the preopened version of it, even if a dlopenable
     module is available.  */
  if (old_name && tryall_dlopen (handle, old_name) == 0)
    {
      return 0;
    }

  /* Try to open the dynamic library.  */
  if (dlname)
    {
      /* try to open the installed module */
      if (installed && libdir)
	{
	  if (tryall_dlopen_module (handle,
				    (const char *) 0, libdir, dlname) == 0)
	    return 0;
	}

      /* try to open the not-installed module */
      if (!installed)
	{
	  if (tryall_dlopen_module (handle, dir, objdir, dlname) == 0)
	    return 0;
	}

      /* maybe it was moved to another directory */
      {
	  if (dir && (tryall_dlopen_module (handle,
				    (const char *) 0, dir, dlname) == 0))
	    return 0;
      }
    }

  return 1;
}


static int
canonicalize_path (path, pcanonical)
     const char *path;
     char **pcanonical;
{
  char *canonical = 0;

  assert (path && *path);
  assert (pcanonical);

  canonical = LT_EMALLOC (char, 1+ LT_STRLEN (path));
  if (!canonical)
    return 1;

  {
    size_t dest = 0;
    size_t src;
    for (src = 0; path[src] != LT_EOS_CHAR; ++src)
      {
	/* Path separators are not copied to the beginning or end of
	   the destination, or if another separator would follow
	   immediately.  */
	if (path[src] == LT_PATHSEP_CHAR)
	  {
	    if ((dest == 0)
		|| (path[1+ src] == LT_PATHSEP_CHAR)
		|| (path[1+ src] == LT_EOS_CHAR))
	      continue;
	  }

	/* Anything other than a directory separator is copied verbatim.  */
	if ((path[src] != '/')
#ifdef LT_DIRSEP_CHAR
	    && (path[src] != LT_DIRSEP_CHAR)
#endif
	    )
	  {
	    canonical[dest++] = path[src];
	  }
	/* Directory separators are converted and copied only if they are
	   not at the end of a path -- i.e. before a path separator or
	   NULL terminator.  */
	else if ((path[1+ src] != LT_PATHSEP_CHAR)
		 && (path[1+ src] != LT_EOS_CHAR)
#ifdef LT_DIRSEP_CHAR
		 && (path[1+ src] != LT_DIRSEP_CHAR)
#endif
		 && (path[1+ src] != '/'))
	  {
	    canonical[dest++] = '/';
	  }
      }

    /* Add an end-of-string marker at the end.  */
    canonical[dest] = LT_EOS_CHAR;
  }

  /* Assign new value.  */
  *pcanonical = canonical;

  return 0;
}

static int
argzize_path (path, pargz, pargz_len)
     const char *path;
     char **pargz;
     size_t *pargz_len;
{
  error_t error;

  assert (path);
  assert (pargz);
  assert (pargz_len);

  if ((error = argz_create_sep (path, LT_PATHSEP_CHAR, pargz, pargz_len)))
    {
      switch (error)
	{
	case ENOMEM:
	  LT_DLMUTEX_SETERROR (LT_DLSTRERROR (NO_MEMORY));
	  break;
	default:
	  LT_DLMUTEX_SETERROR (LT_DLSTRERROR (UNKNOWN));
	  break;
	}

      return 1;
    }

  return 0;
}

/* Repeatedly call FUNC with each LT_PATHSEP_CHAR delimited element
   of SEARCH_PATH and references to DATA1 and DATA2, until FUNC returns
   non-zero or all elements are exhausted.  If BASE_NAME is non-NULL,
   it is appended to each SEARCH_PATH element before FUNC is called.  */
static int
foreach_dirinpath (search_path, base_name, func, data1, data2)
     const char *search_path;
     const char *base_name;
     foreach_callback_func *func;
     lt_ptr data1;
     lt_ptr data2;
{
  int	 result		= 0;
  int	 filenamesize	= 0;
  size_t lenbase	= LT_STRLEN (base_name);
  size_t argz_len	= 0;
  char *argz		= 0;
  char *filename	= 0;
  char *canonical	= 0;

  LT_DLMUTEX_LOCK ();

  if (!search_path || !*search_path)
    {
      LT_DLMUTEX_SETERROR (LT_DLSTRERROR (FILE_NOT_FOUND));
      goto cleanup;
    }

  if (canonicalize_path (search_path, &canonical) != 0)
    goto cleanup;

  if (argzize_path (canonical, &argz, &argz_len) != 0)
    goto cleanup;

  {
    char *dir_name = 0;
    while ((dir_name = argz_next (argz, argz_len, dir_name)))
      {
	size_t lendir = LT_STRLEN (dir_name);

	if (lendir +1 +lenbase >= (size_t)filenamesize)
	{
	  LT_DLFREE (filename);
	  filenamesize	= lendir +1 +lenbase +1; /* "/d" + '/' + "f" + '\0' */
	  filename	= LT_EMALLOC (char, filenamesize);
	  if (!filename)
	    goto cleanup;
	}

	assert ((size_t)filenamesize > lendir);
	strcpy (filename, dir_name);

	if (base_name && *base_name)
	  {
	    if (filename[lendir -1] != '/')
	      filename[lendir++] = '/';
	    strcpy (filename +lendir, base_name);
	  }

	if ((result = (*func) (filename, data1, data2)))
	  {
	    break;
	  }
      }
  }

 cleanup:
  LT_DLFREE (argz);
  LT_DLFREE (canonical);
  LT_DLFREE (filename);

  LT_DLMUTEX_UNLOCK ();

  return result;
}

/* If FILEPATH can be opened, store the name of the directory component
   in DATA1, and the opened FILE* structure address in DATA2.  Otherwise
   DATA1 is unchanged, but DATA2 is set to a pointer to NULL.  */
static int
find_file_callback (filename, data1, data2)
     char *filename;
     lt_ptr data1;
     lt_ptr data2;
{
  char	     **pdir	= (char **) data1;
  FILE	     **pfile	= (FILE **) data2;
  int	     is_done	= 0;

  assert (filename && *filename);
  assert (pdir);
  assert (pfile);

  if ((*pfile = fopen (filename, LT_READTEXT_MODE)))
    {
      char *dirend = strrchr (filename, '/');

      if (dirend > filename)
	*dirend   = LT_EOS_CHAR;

      LT_DLFREE (*pdir);
      *pdir   = lt_estrdup (filename);
      is_done = (*pdir == 0) ? -1 : 1;
    }

  return is_done;
}

static FILE *
find_file (search_path, base_name, pdir)
     const char *search_path;
     const char *base_name;
     char **pdir;
{
  FILE *file = 0;

  foreach_dirinpath (search_path, base_name, find_file_callback, pdir, &file);

  return file;
}

static int
find_handle_callback (filename, data, ignored)
     char *filename;
     lt_ptr data;
     lt_ptr ignored;
{
  lt_dlhandle  *handle		= (lt_dlhandle *) data;
  int		notfound	= access (filename, R_OK);

  /* Bail out if file cannot be read...  */
  if (notfound)
    return 0;

  /* Try to dlopen the file, but do not continue searching in any
     case.  */
  if (tryall_dlopen (handle, filename) != 0)
    *handle = 0;

  return 1;
}

/* If HANDLE was found return it, otherwise return 0.  If HANDLE was
   found but could not be opened, *HANDLE will be set to 0.  */
static lt_dlhandle *
find_handle (search_path, base_name, handle)
     const char *search_path;
     const char *base_name;
     lt_dlhandle *handle;
{
  if (!search_path)
    return 0;

  if (!foreach_dirinpath (search_path, base_name, find_handle_callback,
			  handle, 0))
    return 0;

  return handle;
}

static int
load_deplibs (handle, deplibs)
     lt_dlhandle handle;
     char *deplibs;
{
#if LTDL_DLOPEN_DEPLIBS
  char	*p, *save_search_path = 0;
  int   depcount = 0;
  int	i;
  char	**names = 0;
#endif
  int	errors = 0;

  handle->depcount = 0;

#if LTDL_DLOPEN_DEPLIBS
  if (!deplibs)
    {
      return errors;
    }
  ++errors;

  LT_DLMUTEX_LOCK ();
  if (user_search_path)
    {
      save_search_path = lt_estrdup (user_search_path);
      if (!save_search_path)
	goto cleanup;
    }

  /* extract search paths and count deplibs */
  p = deplibs;
  while (*p)
    {
      if (!isspace ((int) *p))
	{
	  char *end = p+1;
	  while (*end && !isspace((int) *end))
	    {
	      ++end;
	    }

	  if (strncmp(p, "-L", 2) == 0 || strncmp(p, "-R", 2) == 0)
	    {
	      char save = *end;
	      *end = 0; /* set a temporary string terminator */
	      if (lt_dladdsearchdir(p+2))
		{
		  goto cleanup;
		}
	      *end = save;
	    }
	  else
	    {
	      ++depcount;
	    }

	  p = end;
	}
      else
	{
	  ++p;
	}
    }

  if (!depcount)
    {
      errors = 0;
      goto cleanup;
    }

  names = LT_EMALLOC (char *, depcount * sizeof (char*));
  if (!names)
    goto cleanup;

  /* now only extract the actual deplibs */
  depcount = 0;
  p = deplibs;
  while (*p)
    {
      if (isspace ((int) *p))
	{
	  ++p;
	}
      else
	{
	  char *end = p+1;
	  while (*end && !isspace ((int) *end))
	    {
	      ++end;
	    }

	  if (strncmp(p, "-L", 2) != 0 && strncmp(p, "-R", 2) != 0)
	    {
	      char *name;
	      char save = *end;
	      *end = 0; /* set a temporary string terminator */
	      if (strncmp(p, "-l", 2) == 0)
		{
		  size_t name_len = 3+ /* "lib" */ LT_STRLEN (p + 2);
		  name = LT_EMALLOC (char, 1+ name_len);
		  if (name)
		    sprintf (name, "lib%s", p+2);
		}
	      else
		name = lt_estrdup(p);

	      if (!name)
		goto cleanup_names;

	      names[depcount++] = name;
	      *end = save;
	    }
	  p = end;
	}
    }

  /* load the deplibs (in reverse order)
     At this stage, don't worry if the deplibs do not load correctly,
     they may already be statically linked into the loading application
     for instance.  There will be a more enlightening error message
     later on if the loaded module cannot resolve all of its symbols.  */
  if (depcount)
    {
      int	j = 0;

      handle->deplibs = (lt_dlhandle*) LT_EMALLOC (lt_dlhandle *, depcount);
      if (!handle->deplibs)
	goto cleanup;

      for (i = 0; i < depcount; ++i)
	{
	  handle->deplibs[j] = lt_dlopenext(names[depcount-1-i]);
	  if (handle->deplibs[j])
	    {
	      ++j;
	    }
	}

      handle->depcount	= j;	/* Number of successfully loaded deplibs */
      errors		= 0;
    }

 cleanup_names:
  for (i = 0; i < depcount; ++i)
    {
      LT_DLFREE (names[i]);
    }

 cleanup:
  LT_DLFREE (names);
  /* restore the old search path */
  if (user_search_path) {
    LT_DLFREE (user_search_path);
    user_search_path = save_search_path;
  }
  LT_DLMUTEX_UNLOCK ();

#endif

  return errors;
}

static int
unload_deplibs (handle)
     lt_dlhandle handle;
{
  int i;
  int errors = 0;

  if (handle->depcount)
    {
      for (i = 0; i < handle->depcount; ++i)
	{
	  if (!LT_DLIS_RESIDENT (handle->deplibs[i]))
	    {
	      errors += lt_dlclose (handle->deplibs[i]);
	    }
	}
    }

  return errors;
}

static int
trim (dest, str)
     char **dest;
     const char *str;
{
  /* remove the leading and trailing "'" from str
     and store the result in dest */
  const char *end   = strrchr (str, '\'');
  size_t len	    = LT_STRLEN (str);
  char *tmp;

  LT_DLFREE (*dest);

  if (!end)
    return 1;

  if (len > 3 && str[0] == '\'')
    {
      tmp = LT_EMALLOC (char, end - str);
      if (!tmp)
	return 1;

      strncpy(tmp, &str[1], (end - str) - 1);
      tmp[len-3] = LT_EOS_CHAR;
      *dest = tmp;
    }
  else
    {
      *dest = 0;
    }

  return 0;
}

static int
free_vars (dlname, oldname, libdir, deplibs)
     char *dlname;
     char *oldname;
     char *libdir;
     char *deplibs;
{
  LT_DLFREE (dlname);
  LT_DLFREE (oldname);
  LT_DLFREE (libdir);
  LT_DLFREE (deplibs);

  return 0;
}

static int
try_dlopen (phandle, filename)
     lt_dlhandle *phandle;
     const char *filename;
{
  const char *	ext		= 0;
  const char *	saved_error	= 0;
  char *	canonical	= 0;
  char *	base_name	= 0;
  char *	dir		= 0;
  char *	name		= 0;
  int		errors		= 0;
  lt_dlhandle	newhandle;

  assert (phandle);
  assert (*phandle == 0);

  LT_DLMUTEX_GETERROR (saved_error);

  /* dlopen self? */
  if (!filename)
    {
      *phandle = (lt_dlhandle) LT_EMALLOC (struct lt_dlhandle_struct, 1);
      if (*phandle == 0)
	return 1;

      memset (*phandle, 0, sizeof(struct lt_dlhandle_struct));
      newhandle	= *phandle;

      /* lt_dlclose()ing yourself is very bad!  Disallow it.  */
      LT_DLSET_FLAG (*phandle, LT_DLRESIDENT_FLAG);

      if (tryall_dlopen (&newhandle, 0) != 0)
	{
	  LT_DLFREE (*phandle);
	  return 1;
	}

      goto register_handle;
    }

  assert (filename && *filename);

  /* Doing this immediately allows internal functions to safely
     assume only canonicalized paths are passed.  */
  if (canonicalize_path (filename, &canonical) != 0)
    {
      ++errors;
      goto cleanup;
    }

  /* If the canonical module name is a path (relative or absolute)
     then split it into a directory part and a name part.  */
  base_name = strrchr (canonical, '/');
  if (base_name)
    {
      size_t dirlen = (1+ base_name) - canonical;

      dir = LT_EMALLOC (char, 1+ dirlen);
      if (!dir)
	{
	  ++errors;
	  goto cleanup;
	}

      strncpy (dir, canonical, dirlen);
      dir[dirlen] = LT_EOS_CHAR;

      ++base_name;
    }
  else
    base_name = canonical;

  assert (base_name && *base_name);

  /* Check whether we are opening a libtool module (.la extension).  */
  ext = strrchr (base_name, '.');
  if (ext && strcmp (ext, archive_ext) == 0)
    {
      /* this seems to be a libtool module */
      FILE *	file	 = 0;
      char *	dlname	 = 0;
      char *	old_name = 0;
      char *	libdir	 = 0;
      char *	deplibs	 = 0;
      char *    line	 = 0;
      size_t	line_len;

      /* if we can't find the installed flag, it is probably an
	 installed libtool archive, produced with an old version
	 of libtool */
      int	installed = 1;

      /* extract the module name from the file name */
      name = LT_EMALLOC (char, ext - base_name + 1);
      if (!name)
	{
	  ++errors;
	  goto cleanup;
	}

      /* canonicalize the module name */
      {
        size_t i;
        for (i = 0; i < (size_t)(ext - base_name); ++i)
	  {
	    if (isalnum ((int)(base_name[i])))
	      {
	        name[i] = base_name[i];
	      }
	    else
	      {
	        name[i] = '_';
	      }
	  }
        name[ext - base_name] = LT_EOS_CHAR;
      }

      /* Now try to open the .la file.  If there is no directory name
         component, try to find it first in user_search_path and then other
         prescribed paths.  Otherwise (or in any case if the module was not
         yet found) try opening just the module name as passed.  */
      if (!dir)
	{
	  const char *search_path;

	  LT_DLMUTEX_LOCK ();
	  search_path = user_search_path;
	  if (search_path)
	    file = find_file (user_search_path, base_name, &dir);
	  LT_DLMUTEX_UNLOCK ();

	  if (!file)
	    {
	      search_path = getenv (LTDL_SEARCHPATH_VAR);
	      if (search_path)
		file = find_file (search_path, base_name, &dir);
	    }

#ifdef LTDL_SHLIBPATH_VAR
	  if (!file)
	    {
	      search_path = getenv (LTDL_SHLIBPATH_VAR);
	      if (search_path)
		file = find_file (search_path, base_name, &dir);
	    }
#endif
#ifdef LTDL_SYSSEARCHPATH
	  if (!file && sys_search_path)
	    {
	      file = find_file (sys_search_path, base_name, &dir);
	    }
#endif
	}
      if (!file)
	{
	  file = fopen (filename, LT_READTEXT_MODE);
	}

      /* If we didn't find the file by now, it really isn't there.  Set
	 the status flag, and bail out.  */
      if (!file)
	{
	  LT_DLMUTEX_SETERROR (LT_DLSTRERROR (FILE_NOT_FOUND));
	  ++errors;
	  goto cleanup;
	}

      line_len = LT_FILENAME_MAX;
      line = LT_EMALLOC (char, line_len);
      if (!line)
	{
	  fclose (file);
	  ++errors;
	  goto cleanup;
	}

      /* read the .la file */
      while (!feof (file))
	{
	  if (!fgets (line, (int) line_len, file))
	    {
	      break;
	    }

	  /* Handle the case where we occasionally need to read a line
	     that is longer than the initial buffer size.  */
	  while ((line[LT_STRLEN(line) -1] != '\n') && (!feof (file)))
	    {
	      line = LT_DLREALLOC (char, line, line_len *2);
	      if (!fgets (&line[line_len -1], (int) line_len +1, file))
		{
		  break;
		}
	      line_len *= 2;
	    }

	  if (line[0] == '\n' || line[0] == '#')
	    {
	      continue;
	    }

#undef  STR_DLNAME
#define STR_DLNAME	"dlname="
	  if (strncmp (line, STR_DLNAME, sizeof (STR_DLNAME) - 1) == 0)
	    {
	      errors += trim (&dlname, &line[sizeof (STR_DLNAME) - 1]);
	    }

#undef  STR_OLD_LIBRARY
#define STR_OLD_LIBRARY	"old_library="
	  else if (strncmp (line, STR_OLD_LIBRARY,
			    sizeof (STR_OLD_LIBRARY) - 1) == 0)
	    {
	      errors += trim (&old_name, &line[sizeof (STR_OLD_LIBRARY) - 1]);
	    }
#undef  STR_LIBDIR
#define STR_LIBDIR	"libdir="
	  else if (strncmp (line, STR_LIBDIR, sizeof (STR_LIBDIR) - 1) == 0)
	    {
	      errors += trim (&libdir, &line[sizeof(STR_LIBDIR) - 1]);
	    }

#undef  STR_DL_DEPLIBS
#define STR_DL_DEPLIBS	"dependency_libs="
	  else if (strncmp (line, STR_DL_DEPLIBS,
			    sizeof (STR_DL_DEPLIBS) - 1) == 0)
	    {
	      errors += trim (&deplibs, &line[sizeof (STR_DL_DEPLIBS) - 1]);
	    }
	  else if (strcmp (line, "installed=yes\n") == 0)
	    {
	      installed = 1;
	    }
	  else if (strcmp (line, "installed=no\n") == 0)
	    {
	      installed = 0;
	    }

#undef  STR_LIBRARY_NAMES
#define STR_LIBRARY_NAMES "library_names="
	  else if (! dlname && strncmp (line, STR_LIBRARY_NAMES,
					sizeof (STR_LIBRARY_NAMES) - 1) == 0)
	    {
	      char *last_libname;
	      errors += trim (&dlname, &line[sizeof (STR_LIBRARY_NAMES) - 1]);
	      if (!errors
		  && dlname
		  && (last_libname = strrchr (dlname, ' ')) != 0)
		{
		  last_libname = lt_estrdup (last_libname + 1);
		  if (!last_libname)
		    {
		      ++errors;
		      goto cleanup;
		    }
		  LT_DLMEM_REASSIGN (dlname, last_libname);
		}
	    }

	  if (errors)
	    break;
	}

      fclose (file);
      LT_DLFREE (line);

      /* allocate the handle */
      *phandle = (lt_dlhandle) LT_EMALLOC (struct lt_dlhandle_struct, 1);
      if (*phandle == 0)
	++errors;

      if (errors)
	{
	  free_vars (dlname, old_name, libdir, deplibs);
	  LT_DLFREE (*phandle);
	  goto cleanup;
	}

      assert (*phandle);

      memset (*phandle, 0, sizeof(struct lt_dlhandle_struct));
      if (load_deplibs (*phandle, deplibs) == 0)
	{
	  newhandle = *phandle;
	  /* find_module may replace newhandle */
	  if (find_module (&newhandle, dir, libdir, dlname, old_name, installed))
	    {
	      unload_deplibs (*phandle);
	      ++errors;
	    }
	}
      else
	{
	  ++errors;
	}

      free_vars (dlname, old_name, libdir, deplibs);
      if (errors)
	{
	  LT_DLFREE (*phandle);
	  goto cleanup;
	}

      if (*phandle != newhandle)
	{
	  unload_deplibs (*phandle);
	}
    }
  else
    {
      /* not a libtool module */
      *phandle = (lt_dlhandle) LT_EMALLOC (struct lt_dlhandle_struct, 1);
      if (*phandle == 0)
	{
	  ++errors;
	  goto cleanup;
	}

      memset (*phandle, 0, sizeof (struct lt_dlhandle_struct));
      newhandle = *phandle;

      /* If the module has no directory name component, try to find it
	 first in user_search_path and then other prescribed paths.
	 Otherwise (or in any case if the module was not yet found) try
	 opening just the module name as passed.  */
      if ((dir || (!find_handle (user_search_path, base_name, &newhandle)
		   && !find_handle (getenv (LTDL_SEARCHPATH_VAR), base_name,
				    &newhandle)
#ifdef LTDL_SHLIBPATH_VAR
		   && !find_handle (getenv (LTDL_SHLIBPATH_VAR), base_name,
				    &newhandle)
#endif
#ifdef LTDL_SYSSEARCHPATH
		   && !find_handle (sys_search_path, base_name, &newhandle)
#endif
		   )))
	{
          if (tryall_dlopen (&newhandle, filename) != 0)
            {
              newhandle = NULL;
            }
	}

      if (!newhandle)
	{
	  LT_DLFREE (*phandle);
	  ++errors;
	  goto cleanup;
	}
    }

 register_handle:
  LT_DLMEM_REASSIGN (*phandle, newhandle);

  if ((*phandle)->info.ref_count == 0)
    {
      (*phandle)->info.ref_count	= 1;
      LT_DLMEM_REASSIGN ((*phandle)->info.name, name);

      LT_DLMUTEX_LOCK ();
      (*phandle)->next		= handles;
      handles			= *phandle;
      LT_DLMUTEX_UNLOCK ();
    }

  LT_DLMUTEX_SETERROR (saved_error);

 cleanup:
  LT_DLFREE (dir);
  LT_DLFREE (name);
  LT_DLFREE (canonical);

  return errors;
}

lt_dlhandle
lt_dlopen (filename)
     const char *filename;
{
  lt_dlhandle handle = 0;

  /* Just incase we missed a code path in try_dlopen() that reports
     an error, but forgets to reset handle... */
  if (try_dlopen (&handle, filename) != 0)
    return 0;

  return handle;
}

/* If the last error messge store was `FILE_NOT_FOUND', then return
   non-zero.  */
static int
file_not_found ()
{
  const char *error = 0;

  LT_DLMUTEX_GETERROR (error);
  if (error == LT_DLSTRERROR (FILE_NOT_FOUND))
    return 1;

  return 0;
}

/* If FILENAME has an ARCHIVE_EXT or SHLIB_EXT extension, try to
   open the FILENAME as passed.  Otherwise try appending ARCHIVE_EXT,
   and if a file is still not found try again with SHLIB_EXT appended
   instead.  */
lt_dlhandle
lt_dlopenext (filename)
     const char *filename;
{
  lt_dlhandle	handle		= 0;
  char *	tmp		= 0;
  char *	ext		= 0;
  size_t	len;
  int		errors		= 0;

  if (!filename)
    {
      return lt_dlopen (filename);
    }

  assert (filename);

  len = LT_STRLEN (filename);
  ext = strrchr (filename, '.');

  /* If FILENAME already bears a suitable extension, there is no need
     to try appending additional extensions.  */
  if (ext && ((strcmp (ext, archive_ext) == 0)
#ifdef LTDL_SHLIB_EXT
	      || (strcmp (ext, shlib_ext) == 0)
#endif
      ))
    {
      return lt_dlopen (filename);
    }

  /* First try appending ARCHIVE_EXT.  */
  tmp = LT_EMALLOC (char, len + LT_STRLEN (archive_ext) + 1);
  if (!tmp)
    return 0;

  strcpy (tmp, filename);
  strcat (tmp, archive_ext);
  errors = try_dlopen (&handle, tmp);

  /* If we found FILENAME, stop searching -- whether we were able to
     load the file as a module or not.  If the file exists but loading
     failed, it is better to return an error message here than to
     report FILE_NOT_FOUND when the alternatives (foo.so etc) are not
     in the module search path.  */
  if (handle || ((errors > 0) && !file_not_found ()))
    {
      LT_DLFREE (tmp);
      return handle;
    }

#ifdef LTDL_SHLIB_EXT
  /* Try appending SHLIB_EXT.   */
  if (LT_STRLEN (shlib_ext) > LT_STRLEN (archive_ext))
    {
      LT_DLFREE (tmp);
      tmp = LT_EMALLOC (char, len + LT_STRLEN (shlib_ext) + 1);
      if (!tmp)
	return 0;

      strcpy (tmp, filename);
    }
  else
    {
      tmp[len] = LT_EOS_CHAR;
    }

  strcat(tmp, shlib_ext);
  errors = try_dlopen (&handle, tmp);

  /* As before, if the file was found but loading failed, return now
     with the current error message.  */
  if (handle || ((errors > 0) && !file_not_found ()))
    {
      LT_DLFREE (tmp);
      return handle;
    }
#endif

  /* Still here?  Then we really did fail to locate any of the file
     names we tried.  */
  LT_DLMUTEX_SETERROR (LT_DLSTRERROR (FILE_NOT_FOUND));
  LT_DLFREE (tmp);
  return 0;
}


static int
lt_argz_insert (pargz, pargz_len, before, entry)
     char **pargz;
     size_t *pargz_len;
     char *before;
     const char *entry;
{
  error_t error;

  /* Prior to Sep 8, 2005, newlib had a bug where argz_insert(pargz,
     pargz_len, NULL, entry) failed with EINVAL.  */
  if (before)
    error = argz_insert (pargz, pargz_len, before, entry);
  else
    error = argz_append (pargz, pargz_len, entry, 1 + LT_STRLEN (entry));

  if (error)
    {
      switch (error)
	{
	case ENOMEM:
	  LT_DLMUTEX_SETERROR (LT_DLSTRERROR (NO_MEMORY));
	  break;
	default:
	  LT_DLMUTEX_SETERROR (LT_DLSTRERROR (UNKNOWN));
	  break;
	}
      return 1;
    }

  return 0;
}

static int
lt_argz_insertinorder (pargz, pargz_len, entry)
     char **pargz;
     size_t *pargz_len;
     const char *entry;
{
  char *before = 0;

  assert (pargz);
  assert (pargz_len);
  assert (entry && *entry);

  if (*pargz)
    while ((before = argz_next (*pargz, *pargz_len, before)))
      {
	int cmp = strcmp (entry, before);

	if (cmp < 0)  break;
	if (cmp == 0) return 0;	/* No duplicates! */
      }

  return lt_argz_insert (pargz, pargz_len, before, entry);
}

static int
lt_argz_insertdir (pargz, pargz_len, dirnam, dp)
     char **pargz;
     size_t *pargz_len;
     const char *dirnam;
     struct dirent *dp;
{
  char   *buf	    = 0;
  size_t buf_len    = 0;
  char   *end	    = 0;
  size_t end_offset = 0;
  size_t dir_len    = 0;
  int    errors	    = 0;

  assert (pargz);
  assert (pargz_len);
  assert (dp);

  dir_len = LT_STRLEN (dirnam);
  end     = dp->d_name + LT_D_NAMLEN(dp);

  /* Ignore version numbers.  */
  {
    char *p;
    for (p = end; p -1 > dp->d_name; --p)
      if (strchr (".0123456789", p[-1]) == 0)
	break;

    if (*p == '.')
      end = p;
  }

  /* Ignore filename extension.  */
  {
    char *p;
    for (p = end -1; p > dp->d_name; --p)
      if (*p == '.')
	{
	  end = p;
	  break;
	}
  }

  /* Prepend the directory name.  */
  end_offset	= end - dp->d_name;
  buf_len	= dir_len + 1+ end_offset;
  buf		= LT_EMALLOC (char, 1+ buf_len);
  if (!buf)
    return ++errors;

  assert (buf);

  strcpy  (buf, dirnam);
  strcat  (buf, "/");
  strncat (buf, dp->d_name, end_offset);
  buf[buf_len] = LT_EOS_CHAR;

  /* Try to insert (in order) into ARGZ/ARGZ_LEN.  */
  if (lt_argz_insertinorder (pargz, pargz_len, buf) != 0)
    ++errors;

  LT_DLFREE (buf);

  return errors;
}

static int
list_files_by_dir (dirnam, pargz, pargz_len)
     const char *dirnam;
     char **pargz;
     size_t *pargz_len;
{
  DIR	*dirp	  = 0;
  int    errors	  = 0;

  assert (dirnam && *dirnam);
  assert (pargz);
  assert (pargz_len);
  assert (dirnam[LT_STRLEN(dirnam) -1] != '/');

  dirp = opendir (dirnam);
  if (dirp)
    {
      struct dirent *dp	= 0;

      while ((dp = readdir (dirp)))
	if (dp->d_name[0] != '.')
	  if (lt_argz_insertdir (pargz, pargz_len, dirnam, dp))
	    {
	      ++errors;
	      break;
	    }

      closedir (dirp);
    }
  else
    ++errors;

  return errors;
}


/* If there are any files in DIRNAME, call the function passed in
   DATA1 (with the name of each file and DATA2 as arguments).  */
static int
foreachfile_callback (dirname, data1, data2)
     char *dirname;
     lt_ptr data1;
     lt_ptr data2;
{
  int (*func) LT_PARAMS((const char *filename, lt_ptr data))
	= (int (*) LT_PARAMS((const char *filename, lt_ptr data))) data1;

  int	  is_done  = 0;
  char   *argz     = 0;
  size_t  argz_len = 0;

  if (list_files_by_dir (dirname, &argz, &argz_len) != 0)
    goto cleanup;
  if (!argz)
    goto cleanup;

  {
    char *filename = 0;
    while ((filename = argz_next (argz, argz_len, filename)))
      if ((is_done = (*func) (filename, data2)))
	break;
  }

 cleanup:
  LT_DLFREE (argz);

  return is_done;
}


/* Call FUNC for each unique extensionless file in SEARCH_PATH, along
   with DATA.  The filenames passed to FUNC would be suitable for
   passing to lt_dlopenext.  The extensions are stripped so that
   individual modules do not generate several entries (e.g. libfoo.la,
   libfoo.so, libfoo.so.1, libfoo.so.1.0.0).  If SEARCH_PATH is NULL,
   then the same directories that lt_dlopen would search are examined.  */
int
lt_dlforeachfile (search_path, func, data)
     const char *search_path;
     int (*func) LT_PARAMS ((const char *filename, lt_ptr data));
     lt_ptr data;
{
  int is_done = 0;

  if (search_path)
    {
      /* If a specific path was passed, search only the directories
	 listed in it.  */
      is_done = foreach_dirinpath (search_path, 0,
				   foreachfile_callback, func, data);
    }
  else
    {
      /* Otherwise search the default paths.  */
      is_done = foreach_dirinpath (user_search_path, 0,
				   foreachfile_callback, func, data);
      if (!is_done)
	{
	  is_done = foreach_dirinpath (getenv("LTDL_LIBRARY_PATH"), 0,
				       foreachfile_callback, func, data);
	}

#ifdef LTDL_SHLIBPATH_VAR
      if (!is_done)
	{
	  is_done = foreach_dirinpath (getenv(LTDL_SHLIBPATH_VAR), 0,
				       foreachfile_callback, func, data);
	}
#endif
#ifdef LTDL_SYSSEARCHPATH
      if (!is_done)
	{
	  is_done = foreach_dirinpath (getenv(LTDL_SYSSEARCHPATH), 0,
				       foreachfile_callback, func, data);
	}
#endif
    }

  return is_done;
}

int
lt_dlclose (handle)
     lt_dlhandle handle;
{
  lt_dlhandle cur, last;
  int errors = 0;

  LT_DLMUTEX_LOCK ();

  /* check whether the handle is valid */
  last = cur = handles;
  while (cur && handle != cur)
    {
      last = cur;
      cur = cur->next;
    }

  if (!cur)
    {
      LT_DLMUTEX_SETERROR (LT_DLSTRERROR (INVALID_HANDLE));
      ++errors;
      goto done;
    }

  handle->info.ref_count--;

  /* Note that even with resident modules, we must track the ref_count
     correctly incase the user decides to reset the residency flag
     later (even though the API makes no provision for that at the
     moment).  */
  if (handle->info.ref_count <= 0 && !LT_DLIS_RESIDENT (handle))
    {
      lt_user_data data = handle->loader->dlloader_data;

      if (handle != handles)
	{
	  last->next = handle->next;
	}
      else
	{
	  handles = handle->next;
	}

      errors += handle->loader->module_close (data, handle->module);
      errors += unload_deplibs(handle);

      /* It is up to the callers to free the data itself.  */
      LT_DLFREE (handle->caller_data);

      LT_DLFREE (handle->info.filename);
      LT_DLFREE (handle->info.name);
      LT_DLFREE (handle);

      goto done;
    }

  if (LT_DLIS_RESIDENT (handle))
    {
      LT_DLMUTEX_SETERROR (LT_DLSTRERROR (CLOSE_RESIDENT_MODULE));
      ++errors;
    }

 done:
  LT_DLMUTEX_UNLOCK ();

  return errors;
}

lt_ptr
lt_dlsym (handle, symbol)
     lt_dlhandle handle;
     const char *symbol;
{
  size_t lensym;
  char	lsym[LT_SYMBOL_LENGTH];
  char	*sym;
  lt_ptr address;
  lt_user_data data;

  if (!handle)
    {
      LT_DLMUTEX_SETERROR (LT_DLSTRERROR (INVALID_HANDLE));
      return 0;
    }

  if (!symbol)
    {
      LT_DLMUTEX_SETERROR (LT_DLSTRERROR (SYMBOL_NOT_FOUND));
      return 0;
    }

  lensym = LT_STRLEN (symbol) + LT_STRLEN (handle->loader->sym_prefix)
					+ LT_STRLEN (handle->info.name);

  if (lensym + LT_SYMBOL_OVERHEAD < LT_SYMBOL_LENGTH)
    {
      sym = lsym;
    }
  else
    {
      sym = LT_EMALLOC (char, lensym + LT_SYMBOL_OVERHEAD + 1);
      if (!sym)
	{
	  LT_DLMUTEX_SETERROR (LT_DLSTRERROR (BUFFER_OVERFLOW));
	  return 0;
	}
    }

  data = handle->loader->dlloader_data;
  if (handle->info.name)
    {
      const char *saved_error;

      LT_DLMUTEX_GETERROR (saved_error);

      /* this is a libtool module */
      if (handle->loader->sym_prefix)
	{
	  strcpy(sym, handle->loader->sym_prefix);
	  strcat(sym, handle->info.name);
	}
      else
	{
	  strcpy(sym, handle->info.name);
	}

      strcat(sym, "_LTX_");
      strcat(sym, symbol);

      /* try "modulename_LTX_symbol" */
      address = handle->loader->find_sym (data, handle->module, sym);
      if (address)
	{
	  if (sym != lsym)
	    {
	      LT_DLFREE (sym);
	    }
	  return address;
	}
      LT_DLMUTEX_SETERROR (saved_error);
    }

  /* otherwise try "symbol" */
  if (handle->loader->sym_prefix)
    {
      strcpy(sym, handle->loader->sym_prefix);
      strcat(sym, symbol);
    }
  else
    {
      strcpy(sym, symbol);
    }

  address = handle->loader->find_sym (data, handle->module, sym);
  if (sym != lsym)
    {
      LT_DLFREE (sym);
    }

  return address;
}

const char *
lt_dlerror ()
{
  const char *error;

  LT_DLMUTEX_GETERROR (error);
  LT_DLMUTEX_SETERROR (0);

  return error ? error : NULL;
}

static int
lt_dlpath_insertdir (ppath, before, dir)
     char **ppath;
     char *before;
     const char *dir;
{
  int    errors		= 0;
  char  *canonical	= 0;
  char  *argz		= 0;
  size_t argz_len	= 0;

  assert (ppath);
  assert (dir && *dir);

  if (canonicalize_path (dir, &canonical) != 0)
    {
      ++errors;
      goto cleanup;
    }

  assert (canonical && *canonical);

  /* If *PPATH is empty, set it to DIR.  */
  if (*ppath == 0)
    {
      assert (!before);		/* BEFORE cannot be set without PPATH.  */
      assert (dir);		/* Without DIR, don't call this function!  */

      *ppath = lt_estrdup (dir);
      if (*ppath == 0)
	++errors;

      return errors;
    }

  assert (ppath && *ppath);

  if (argzize_path (*ppath, &argz, &argz_len) != 0)
    {
      ++errors;
      goto cleanup;
    }

  /* Convert BEFORE into an equivalent offset into ARGZ.  This only works
     if *PPATH is already canonicalized, and hence does not change length
     with respect to ARGZ.  We canonicalize each entry as it is added to
     the search path, and don't call this function with (uncanonicalized)
     user paths, so this is a fair assumption.  */
  if (before)
    {
      assert (*ppath <= before);
      assert ((size_t)(before - *ppath) <= strlen (*ppath));

      before = before - *ppath + argz;
    }

  if (lt_argz_insert (&argz, &argz_len, before, dir) != 0)
    {
      ++errors;
      goto cleanup;
    }

  argz_stringify (argz, argz_len, LT_PATHSEP_CHAR);
  LT_DLMEM_REASSIGN (*ppath,  argz);

 cleanup:
  LT_DLFREE (canonical);
  LT_DLFREE (argz);

  return errors;
}

int
lt_dladdsearchdir (search_dir)
     const char *search_dir;
{
  int errors = 0;

  if (search_dir && *search_dir)
    {
      LT_DLMUTEX_LOCK ();
      if (lt_dlpath_insertdir (&user_search_path, 0, search_dir) != 0)
	++errors;
      LT_DLMUTEX_UNLOCK ();
    }

  return errors;
}

int
lt_dlinsertsearchdir (before, search_dir)
     const char *before;
     const char *search_dir;
{
  int errors = 0;

  if (before)
    {
      LT_DLMUTEX_LOCK ();
      if ((before < user_search_path)
	  || (before >= user_search_path + LT_STRLEN (user_search_path)))
	{
	  LT_DLMUTEX_UNLOCK ();
	  LT_DLMUTEX_SETERROR (LT_DLSTRERROR (INVALID_POSITION));
	  return 1;
	}
      LT_DLMUTEX_UNLOCK ();
    }

  if (search_dir && *search_dir)
    {
      LT_DLMUTEX_LOCK ();
      if (lt_dlpath_insertdir (&user_search_path,
			       (char *) before, search_dir) != 0)
	{
	  ++errors;
	}
      LT_DLMUTEX_UNLOCK ();
    }

  return errors;
}

int
lt_dlsetsearchpath (search_path)
     const char *search_path;
{
  int   errors	    = 0;

  LT_DLMUTEX_LOCK ();
  LT_DLFREE (user_search_path);
  LT_DLMUTEX_UNLOCK ();

  if (!search_path || !LT_STRLEN (search_path))
    {
      return errors;
    }

  LT_DLMUTEX_LOCK ();
  if (canonicalize_path (search_path, &user_search_path) != 0)
    ++errors;
  LT_DLMUTEX_UNLOCK ();

  return errors;
}

const char *
lt_dlgetsearchpath ()
{
  const char *saved_path;

  LT_DLMUTEX_LOCK ();
  saved_path = user_search_path;
  LT_DLMUTEX_UNLOCK ();

  return saved_path;
}

int
lt_dlmakeresident (handle)
     lt_dlhandle handle;
{
  int errors = 0;

  if (!handle)
    {
      LT_DLMUTEX_SETERROR (LT_DLSTRERROR (INVALID_HANDLE));
      ++errors;
    }
  else
    {
      LT_DLSET_FLAG (handle, LT_DLRESIDENT_FLAG);
    }

  return errors;
}

int
lt_dlisresident	(handle)
     lt_dlhandle handle;
{
  if (!handle)
    {
      LT_DLMUTEX_SETERROR (LT_DLSTRERROR (INVALID_HANDLE));
      return -1;
    }

  return LT_DLIS_RESIDENT (handle);
}




/* --- MODULE INFORMATION --- */

const lt_dlinfo *
lt_dlgetinfo (handle)
     lt_dlhandle handle;
{
  if (!handle)
    {
      LT_DLMUTEX_SETERROR (LT_DLSTRERROR (INVALID_HANDLE));
      return 0;
    }

  return &(handle->info);
}

lt_dlhandle
lt_dlhandle_next (place)
     lt_dlhandle place;
{
  return place ? place->next : handles;
}

int
lt_dlforeach (func, data)
     int (*func) LT_PARAMS((lt_dlhandle handle, lt_ptr data));
     lt_ptr data;
{
  int errors = 0;
  lt_dlhandle cur;

  LT_DLMUTEX_LOCK ();

  cur = handles;
  while (cur)
    {
      lt_dlhandle tmp = cur;

      cur = cur->next;
      if ((*func) (tmp, data))
	{
	  ++errors;
	  break;
	}
    }

  LT_DLMUTEX_UNLOCK ();

  return errors;
}

lt_dlcaller_id
lt_dlcaller_register ()
{
  static lt_dlcaller_id last_caller_id = 0;
  int result;

  LT_DLMUTEX_LOCK ();
  result = ++last_caller_id;
  LT_DLMUTEX_UNLOCK ();

  return result;
}

lt_ptr
lt_dlcaller_set_data (key, handle, data)
     lt_dlcaller_id key;
     lt_dlhandle handle;
     lt_ptr data;
{
  int n_elements = 0;
  lt_ptr stale = (lt_ptr) 0;
  int i;

  /* This needs to be locked so that the caller data can be updated
     simultaneously by different threads.  */
  LT_DLMUTEX_LOCK ();

  if (handle->caller_data)
    while (handle->caller_data[n_elements].key)
      ++n_elements;

  for (i = 0; i < n_elements; ++i)
    {
      if (handle->caller_data[i].key == key)
	{
	  stale = handle->caller_data[i].data;
	  break;
	}
    }

  /* Ensure that there is enough room in this handle's caller_data
     array to accept a new element (and an empty end marker).  */
  if (i == n_elements)
    {
      lt_caller_data *temp
	= LT_DLREALLOC (lt_caller_data, handle->caller_data, 2+ n_elements);

      if (!temp)
	{
	  stale = 0;
	  goto done;
	}

      handle->caller_data = temp;

      /* We only need this if we needed to allocate a new caller_data.  */
      handle->caller_data[i].key  = key;
      handle->caller_data[1+ i].key = 0;
    }

  handle->caller_data[i].data = data;

 done:
  LT_DLMUTEX_UNLOCK ();

  return stale;
}

lt_ptr
lt_dlcaller_get_data  (key, handle)
     lt_dlcaller_id key;
     lt_dlhandle handle;
{
  lt_ptr result = (lt_ptr) 0;

  /* This needs to be locked so that the caller data isn't updated by
     another thread part way through this function.  */
  LT_DLMUTEX_LOCK ();

  /* Locate the index of the element with a matching KEY.  */
  {
    int i;
    for (i = 0; handle->caller_data[i].key; ++i)
      {
	if (handle->caller_data[i].key == key)
	  {
	    result = handle->caller_data[i].data;
	    break;
	  }
      }
  }

  LT_DLMUTEX_UNLOCK ();

  return result;
}



/* --- USER MODULE LOADER API --- */


int
lt_dlloader_add (place, dlloader, loader_name)
     lt_dlloader *place;
     const struct lt_user_dlloader *dlloader;
     const char *loader_name;
{
  int errors = 0;
  lt_dlloader *node = 0, *ptr = 0;

  if ((dlloader == 0)	/* diagnose null parameters */
      || (dlloader->module_open == 0)
      || (dlloader->module_close == 0)
      || (dlloader->find_sym == 0))
    {
      LT_DLMUTEX_SETERROR (LT_DLSTRERROR (INVALID_LOADER));
      return 1;
    }

  /* Create a new dlloader node with copies of the user callbacks.  */
  node = LT_EMALLOC (lt_dlloader, 1);
  if (!node)
    return 1;

  node->next		= 0;
  node->loader_name	= loader_name;
  node->sym_prefix	= dlloader->sym_prefix;
  node->dlloader_exit	= dlloader->dlloader_exit;
  node->module_open	= dlloader->module_open;
  node->module_close	= dlloader->module_close;
  node->find_sym	= dlloader->find_sym;
  node->dlloader_data	= dlloader->dlloader_data;

  LT_DLMUTEX_LOCK ();
  if (!loaders)
    {
      /* If there are no loaders, NODE becomes the list! */
      loaders = node;
    }
  else if (!place)
    {
      /* If PLACE is not set, add NODE to the end of the
	 LOADERS list. */
      for (ptr = loaders; ptr->next; ptr = ptr->next)
	{
	  /*NOWORK*/;
	}

      ptr->next = node;
    }
  else if (loaders == place)
    {
      /* If PLACE is the first loader, NODE goes first. */
      node->next = place;
      loaders = node;
    }
  else
    {
      /* Find the node immediately preceding PLACE. */
      for (ptr = loaders; ptr->next != place; ptr = ptr->next)
	{
	  /*NOWORK*/;
	}

      if (ptr->next != place)
	{
	  LT_DLMUTEX_SETERROR (LT_DLSTRERROR (INVALID_LOADER));
	  ++errors;
	}
      else
	{
	  /* Insert NODE between PTR and PLACE. */
	  node->next = place;
	  ptr->next  = node;
	}
    }

  LT_DLMUTEX_UNLOCK ();

  return errors;
}

int
lt_dlloader_remove (loader_name)
     const char *loader_name;
{
  lt_dlloader *place = lt_dlloader_find (loader_name);
  lt_dlhandle handle;
  int errors = 0;

  if (!place)
    {
      LT_DLMUTEX_SETERROR (LT_DLSTRERROR (INVALID_LOADER));
      return 1;
    }

  LT_DLMUTEX_LOCK ();

  /* Fail if there are any open modules which use this loader. */
  for  (handle = handles; handle; handle = handle->next)
    {
      if (handle->loader == place)
	{
	  LT_DLMUTEX_SETERROR (LT_DLSTRERROR (REMOVE_LOADER));
	  ++errors;
	  goto done;
	}
    }

  if (place == loaders)
    {
      /* PLACE is the first loader in the list. */
      loaders = loaders->next;
    }
  else
    {
      /* Find the loader before the one being removed. */
      lt_dlloader *prev;
      for (prev = loaders; prev->next; prev = prev->next)
	{
	  if (!strcmp (prev->next->loader_name, loader_name))
	    {
	      break;
	    }
	}

      place = prev->next;
      prev->next = prev->next->next;
    }

  if (place->dlloader_exit)
    {
      errors = place->dlloader_exit (place->dlloader_data);
    }

  LT_DLFREE (place);

 done:
  LT_DLMUTEX_UNLOCK ();

  return errors;
}

lt_dlloader *
lt_dlloader_next (place)
     lt_dlloader *place;
{
  lt_dlloader *next;

  LT_DLMUTEX_LOCK ();
  next = place ? place->next : loaders;
  LT_DLMUTEX_UNLOCK ();

  return next;
}

const char *
lt_dlloader_name (place)
     lt_dlloader *place;
{
  const char *name = 0;

  if (place)
    {
      LT_DLMUTEX_LOCK ();
      name = place ? place->loader_name : 0;
      LT_DLMUTEX_UNLOCK ();
    }
  else
    {
      LT_DLMUTEX_SETERROR (LT_DLSTRERROR (INVALID_LOADER));
    }

  return name;
}

lt_user_data *
lt_dlloader_data (place)
     lt_dlloader *place;
{
  lt_user_data *data = 0;

  if (place)
    {
      LT_DLMUTEX_LOCK ();
      data = place ? &(place->dlloader_data) : 0;
      LT_DLMUTEX_UNLOCK ();
    }
  else
    {
      LT_DLMUTEX_SETERROR (LT_DLSTRERROR (INVALID_LOADER));
    }

  return data;
}

lt_dlloader *
lt_dlloader_find (loader_name)
     const char *loader_name;
{
  lt_dlloader *place = 0;

  LT_DLMUTEX_LOCK ();
  for (place = loaders; place; place = place->next)
    {
      if (strcmp (place->loader_name, loader_name) == 0)
	{
	  break;
	}
    }
  LT_DLMUTEX_UNLOCK ();

  return place;
}
