//===- Path.cpp - Path Operating System Concept -----------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// Copyright (C) 2004 eXtensible Systems, Inc. All Rights Reserved.
//
// This program is open source software; you can redistribute it and/or modify
// it under the terms of the University of Illinois Open Source License. See
// LICENSE.TXT (distributed with this software) for details.
//
// This program is distributed in the hope that it will be useful, but
// WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
// or FITNESS FOR A PARTICULAR PURPOSE.
//
//===----------------------------------------------------------------------===//
//===----------------------------------------------------------------------===//
/// @file lib/System/linux/Path.cpp
/// @author Reid Spencer <raspencer@users.sourceforge.net> (original author)
/// @version \verbatim $Id$ \endverbatim
/// @date 2003/04/17
/// @since 1.4
/// @brief Implements the linux specific portion of class llvm::sys::Path
////////////////////////////////////////////////////////////////////////////////

#include "LinuxCommon.h"
#include <sys/stat.h>
#include <fcntl.h>

namespace llvm {
namespace sys {

Path::Path(ConstructSpecial which) throw() {
  switch (which) {
    case CONSTRUCT_TEMP_FILE:
      this->make_temp_file();
      break;
    case CONSTRUCT_TEMP_DIR:
      this->make_temp_directory();
      break;
  }
}

bool 
Path::is_file() const throw() {
  if ( !empty() && ((*this)[length()-1] !=  '/' ))
    return true;
  return false;
}

bool 
Path::is_directory() const throw() {
  if ((!empty()) && ((*this)[length()-1] ==  '/' ))
      return true;
  return false;
}

ErrorCode 
Path::make_directory() throw() {
  char end[2];
  end[0] = '/';
  end[1] = 0;
  if ( empty() )
    this->assign( end );
  else if ( (*this)[length()-1] != '/' )
    this->append( end );
  return NOT_AN_ERROR;
}

ErrorCode
Path::make_temp_directory() throw() {
  char temp_name[64];
  ::strcpy(temp_name,"/tmp/llvm_XXXXXX");
  char* res = ::mkdtemp(temp_name);
  if ( res == 0 )
    RETURN_ERRNO;
  *this = temp_name;
  make_directory();
  return NOT_AN_ERROR;
}

ErrorCode
Path::make_file() throw() {
  if ( (*this)[length()-1] == '/' )
    this->erase( this->length()-1, 1 );
  return NOT_AN_ERROR;
}

ErrorCode
Path::make_temp_file() throw() {
  char temp_name[64];
  ::strcpy(temp_name,"/tmp/llvm_XXXXXX");
  int fd = ::mkstemp(temp_name);
  if ( fd == -1 )
    RETURN_ERRNO;
  ::close(fd);
  *this = temp_name;
  return NOT_AN_ERROR;
}

ErrorCode
Path::exists() throw() {
  char pathname[MAXPATHLEN];
  this->fill(pathname,MAXPATHLEN);
  int lastchar = this->length() - 1 ; 
  if (pathname[lastchar] == '/') 
    pathname[lastchar] = 0;
  RETURN_ON_ERROR( access,( pathname, F_OK | R_OK ) );
  return NOT_AN_ERROR;
}

ErrorCode
Path::create_directory() throw() {
  char pathname[MAXPATHLEN];
  this->fill(pathname,MAXPATHLEN);
  int lastchar = this->length() - 1 ; 
  if (pathname[lastchar] == '/') 
    pathname[lastchar] = 0;
  RETURN_ON_ERROR( mkdir, ( pathname, S_IRWXU | S_IRWXG ) );
  return NOT_AN_ERROR;
}

ErrorCode
Path::create_directories() throw() {
  char pathname[MAXPATHLEN];
  this->fill(pathname,MAXPATHLEN);
  int lastchar = this->length() - 1 ; 
  if (pathname[lastchar] == '/') 
    pathname[lastchar] = 0;
  char * next = index(pathname,'/');
  if ( pathname[0] == '/' ) 
    next = index(&pathname[1],'/');
  while ( next != 0 )
  {
    *next = 0;
    if ( 0 != access( pathname, F_OK | R_OK ) )
      RETURN_ON_ERROR( mkdir, (pathname, S_IRWXU | S_IRWXG ) );
    char* save = next;
    next = index(pathname,'/');
    *save = '/';
  }
  return NOT_AN_ERROR;
}

ErrorCode
Path::remove_directory( ) throw() {
  char pathname[MAXPATHLEN];
  this->fill(pathname,MAXPATHLEN);
  int lastchar = this->length() - 1 ; 
  if (pathname[lastchar] == '/') 
    pathname[lastchar] = 0;
  RETURN_ON_ERROR( rmdir, (pathname) );
  return NOT_AN_ERROR;
}

ErrorCode
Path::create_file( ) throw() {
  char pathname[MAXPATHLEN];
  this->fill(pathname,MAXPATHLEN);
  int lastchar = this->length() - 1 ; 
  if (pathname[lastchar] == '/') 
    pathname[lastchar] = 0;
  RETURN_ON_ERROR( creat, ( pathname, S_IRUSR | S_IWUSR ) );
  return NOT_AN_ERROR;
}

ErrorCode
Path::remove_file( ) throw() {
  char pathname[MAXPATHLEN];
  this->fill(pathname,MAXPATHLEN);
  int lastchar = this->length() - 1 ; 
  if (pathname[lastchar] == '/') 
    pathname[lastchar] = 0;
  RETURN_ON_ERROR( unlink, (pathname) );
  return NOT_AN_ERROR;
}

ErrorCode
Path::find_lib( const char * file ) throw()
{
    ASSERT_ARG( file != 0 );

#if 0
  ACE_TCHAR tempcopy[MAXPATHLEN + 1];
  ACE_TCHAR searchpathname[MAXPATHLEN + 1];
  ACE_TCHAR searchfilename[MAXPATHLEN + 1];

  // Create a copy of filename to work with.
  if (ACE_OS::strlen (filename) + 1
      > (sizeof tempcopy / sizeof (ACE_TCHAR)))
    {
      errno = ENOMEM;
      return -1;
    }
  else
    ACE_OS::strcpy (tempcopy, filename);

  // Insert canonical directory separators.
  ACE_TCHAR *separator_ptr;

#if (ACE_DIRECTORY_SEPARATOR_CHAR != '/')
  // Make all the directory separators "canonical" to simplify
  // subsequent code.
  ACE_Lib_Find::strrepl (tempcopy, ACE_DIRECTORY_SEPARATOR_CHAR, '/');
#endif /* ACE_DIRECTORY_SEPARATOR_CHAR */

  // Separate filename from pathname.
  separator_ptr = ACE_OS::strrchr (tempcopy, '/');

  // This is a relative path.
  if (separator_ptr == 0)
    {
      searchpathname[0] = '\0';
      ACE_OS::strcpy (searchfilename, tempcopy);
    }
  else // This is an absolute path.
    {
      ACE_OS::strcpy (searchfilename, separator_ptr + 1);
      separator_ptr[1] = '\0';
      ACE_OS::strcpy (searchpathname, tempcopy);
    }

  int got_suffix = 0;

  // Check to see if this has an appropriate DLL suffix for the OS
  // platform.
  ACE_TCHAR *s = ACE_OS::strrchr (searchfilename, '.');

  const ACE_TCHAR *dll_suffix = ACE_DLL_SUFFIX;

  if (s != 0)
    {
      // If we have a dot, we have a suffix
      got_suffix = 1;

      // Check whether this matches the appropriate platform-specific
      // suffix.
      if (ACE_OS::strcmp (s, dll_suffix) != 0)
        {
          ACE_ERROR ((LM_WARNING,
                      ACE_LIB_TEXT ("Warning: improper suffix for a ")
                      ACE_LIB_TEXT ("shared library on this platform: %s\n"),
                      s));
        }
    }

  // Make sure we've got enough space in searchfilename.
  if (ACE_OS::strlen (searchfilename)
      + ACE_OS::strlen (ACE_DLL_PREFIX)
      + got_suffix ? 0 : ACE_OS::strlen (dll_suffix) >= (sizeof searchfilename /
                                                         sizeof (ACE_TCHAR)))
    {
      errno = ENOMEM;
      return -1;
    }

      // Use absolute pathname if there is one.
      if (ACE_OS::strlen (searchpathname) > 0)
        {
          if (ACE_OS::strlen (searchfilename)
              + ACE_OS::strlen (searchpathname) >= maxpathnamelen)
            {
              errno = ENOMEM;
              return -1;
            }
          else
            {
#if (ACE_DIRECTORY_SEPARATOR_CHAR != '/')
              // Revert to native path name separators.
              ACE_Lib_Find::strrepl (searchpathname,
		      '/',
                                     ACE_DIRECTORY_SEPARATOR_CHAR);
#endif /* ACE_DIRECTORY_SEPARATOR_CHAR */
              // First, try matching the filename *without* adding a
              // prefix.
#if defined (ACE_HAS_BROKEN_CONDITIONAL_STRING_CASTS)
              ACE_OS::sprintf (pathname,
                               ACE_LIB_TEXT ("%s%s%s"),
                               searchpathname,
                               searchfilename,
                               got_suffix ? ACE_static_cast (ACE_TCHAR *,
                                                             ACE_LIB_TEXT (""))
                               : ACE_static_cast (ACE_TCHAR *,
                                                  dll_suffix));
#else /* ! defined (ACE_HAS_BROKEN_CONDITIONAL_STRING_CASTS) */
              ACE_OS::sprintf (pathname,
                               ACE_LIB_TEXT ("%s%s%s"),
                               searchpathname,
                               searchfilename,
                               got_suffix ? ACE_LIB_TEXT ("") : dll_suffix);
#endif /* ! defined (ACE_HAS_BROKEN_CONDITIONAL_STRING_CASTS) */
              if (ACE_OS::access (pathname, F_OK) == 0)
                return 0;

              // Second, try matching the filename *with* adding a prefix.
#if defined (ACE_HAS_BROKEN_CONDITIONAL_STRING_CASTS)
              ACE_OS::sprintf (pathname,
                               ACE_LIB_TEXT ("%s%s%s%s"),
                               searchpathname,
                               ACE_DLL_PREFIX,
                               searchfilename,
                               got_suffix ? ACE_static_cast (ACE_TCHAR *,
                                                             ACE_LIB_TEXT (""))
                               : ACE_static_cast (ACE_TCHAR *,
                                                  dll_suffix));
#else /* ! defined (ACE_HAS_BROKEN_CONDITIONAL_STRING_CASTS) */
              ACE_OS::sprintf (pathname,
                               ACE_LIB_TEXT ("%s%s%s%s"),
                               searchpathname,
                               ACE_DLL_PREFIX,
                               searchfilename,
                               got_suffix ? ACE_LIB_TEXT ("") : dll_suffix);
#endif /* ! defined (ACE_HAS_BROKEN_CONDITIONAL_STRING_CASTS) */
              if (ACE_OS::access (pathname, F_OK) == 0)
                return 0;
            }
        }

      // Use relative filenames via LD_LIBRARY_PATH or PATH (depending on
      // OS platform).
      else
        {
          ACE_TCHAR *ld_path =
#if defined ACE_DEFAULT_LD_SEARCH_PATH
            ACE_DEFAULT_LD_SEARCH_PATH;
#else
            ACE_OS::getenv (ACE_LD_SEARCH_PATH);
#endif /* ACE_DEFAULT_LD_SEARCH_PATH */

          if (ld_path != 0
              && (ld_path = ACE_OS::strdup (ld_path)) != 0)
            {
              // strtok has the strange behavior of not separating the
              // string ":/foo:/bar" into THREE tokens.  One would expect
              // that the first iteration the token would be an empty
              // string, the second iteration would be "/foo", and the
              // third iteration would be "/bar".  However, this is not
              // the case; one only gets two iterations: "/foo" followed
              // by "/bar".

              // This is especially a problem in parsing Unix paths
              // because it is permissible to specify 'the current
              // directory' as an empty entry.  So, we introduce the
              // following special code to cope with this:

              // Look at each dynamic lib directory in the search path.

              ACE_TCHAR *nextholder = 0;
              const ACE_TCHAR *path_entry =
                ACE_Lib_Find::strsplit_r (ld_path,
                                          ACE_LD_SEARCH_PATH_SEPARATOR_STR,
                                          nextholder);
              int result = 0;

              for (;;)
                {
                  // Check if at end of search path.
                  if (path_entry == 0)
                    {
                      errno = ENOENT;
                      result = -1;
                      break;
                    }
                  else if (ACE_OS::strlen (path_entry)
                           + 1
                           + ACE_OS::strlen (searchfilename)
                           >= maxpathnamelen)
                    {
                      errno = ENOMEM;
                      result = -1;
                      break;
                    }
                  // This works around the issue where a path might have
                  // an empty component indicating 'current directory'.
                  // We need to do it here rather than anywhere else so
                  // that the loop condition will still work.
                  else if (path_entry[0] == '\0')
                    path_entry = ACE_LIB_TEXT (".");

                  // First, try matching the filename *without* adding a
                  // prefix.
#if defined (ACE_HAS_BROKEN_CONDITIONAL_STRING_CASTS)
                  ACE_OS::sprintf (pathname,
                                   ACE_LIB_TEXT ("%s%c%s%s"),
                                   path_entry,
                                   ACE_DIRECTORY_SEPARATOR_CHAR,
                                   searchfilename,
                                   got_suffix ? ACE_static_cast (ACE_TCHAR *,
                                                                 ACE_LIB_TEXT (""))
                                   : ACE_static_cast (ACE_TCHAR *,
                                                      dll_suffix));
#else /* ! defined (ACE_HAS_BROKEN_CONDITIONAL_STRING_CASTS) */
                  ACE_OS::sprintf (pathname,
                                   ACE_LIB_TEXT ("%s%c%s%s"),
                                   path_entry,
                                   ACE_DIRECTORY_SEPARATOR_CHAR,
                                   searchfilename,
                                   got_suffix ? ACE_LIB_TEXT ("") : dll_suffix);
#endif /* ! defined (ACE_HAS_BROKEN_CONDITIONAL_STRING_CASTS) */
                  if (ACE_OS::access (pathname, F_OK) == 0)
                    break;

                  // Second, try matching the filename *with* adding a
                  // prefix.
#if defined (ACE_HAS_BROKEN_CONDITIONAL_STRING_CASTS)
                  ACE_OS::sprintf (pathname,
                                   ACE_LIB_TEXT ("%s%c%s%s%s"),
                                   path_entry,
                                   ACE_DIRECTORY_SEPARATOR_CHAR,
                                   ACE_DLL_PREFIX,
                                   searchfilename,
                                   got_suffix ? ACE_static_cast (ACE_TCHAR *,
                                                                 ACE_LIB_TEXT (""))
                                   : ACE_static_cast (ACE_TCHAR *,
                                                      dll_suffix));
#else /* ! defined (ACE_HAS_BROKEN_CONDITIONAL_STRING_CASTS) */
                  ACE_OS::sprintf (pathname,
                                   ACE_LIB_TEXT ("%s%c%s%s%s"),
                                   path_entry,
                                   ACE_DIRECTORY_SEPARATOR_CHAR,
                                   ACE_DLL_PREFIX,
                                   searchfilename,
                                   got_suffix ? ACE_LIB_TEXT ("") : dll_suffix);
#endif /* ! defined (ACE_HAS_BROKEN_CONDITIONAL_STRING_CASTS) */
                  if (ACE_OS::access (pathname, F_OK) == 0)
                    break;

                  // Fetch the next item in the path
                  path_entry = ACE_Lib_Find::strsplit_r (0,
                                                         ACE_LD_SEARCH_PATH_SEPARATOR_STR,
                                                         nextholder);
                }

              ACE_OS::free ((void *) ld_path);
                return result;
            }
        }
  errno = ENOENT;
  return -1;
#endif

    /// @todo FIXME: Convert ACE code
    *this = Path( file ); 
    
    return NOT_AN_ERROR;
}

}
}

// vim: sw=2 smartindent smarttab tw=80 autoindent expandtab
