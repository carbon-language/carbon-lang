//===- llvm/System/Unix/Path.cpp - Unix Path Implementation -----*- C++ -*-===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Reid Spencer and is distributed under the 
// University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This file implements the Unix specific portion of the Path class.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
//=== WARNING: Implementation here must contain only generic UNIX code that
//===          is guaranteed to work on all UNIX variants.
//===----------------------------------------------------------------------===//

#include "Unix.h"
#include <sys/stat.h>
#include <fcntl.h>

bool
Path::is_file() const {
  if (!empty() && ((*this)[length()-1] !=  '/'))
    return true;
  return false;
}

bool
Path::is_directory() const {
  if ((!empty()) && ((*this)[length()-1] ==  '/'))
    return true;
  return false;
}

void
Path::create( bool create_parents) {
  if ( is_directory() ) {
    if ( create_parents )
      this->create_directories( );
    this->create_directory( );
  } else if ( is_file() ) {
    if ( create_parents )
      this->create_directories( );
    this->create_file( );
  }
}

void
Path::remove() {
  if ( is_directory() ) {
    if ( exists() )
      this->remove_directory( );
  } else if ( is_file() )
    if ( exists() ) 
      this->remove_file( );
}

bool
Path::exists() {
  char pathname[MAXPATHLEN];
  this->fill(pathname,MAXPATHLEN);
  int lastchar = this->length() - 1 ; 
  if (pathname[lastchar] == '/') 
    pathname[lastchar] = 0;
  return 0 == access(pathname, F_OK );
}

void
Path::create_directory( ) {
  char pathname[MAXPATHLEN];
  this->fill(pathname,MAXPATHLEN);
  int lastchar = this->length() - 1 ; 
  if (pathname[lastchar] == '/') 
    pathname[lastchar] = 0;
  if (0 != mkdir(pathname, S_IRWXU | S_IRWXG))
    ThrowErrno(pathname);
}

void
Path::create_directories() {
  char pathname[MAXPATHLEN];
  this->fill(pathname,MAXPATHLEN);
  int lastchar = this->length() - 1 ; 
  if (pathname[lastchar] == '/') 
    pathname[lastchar] = 0;

  char * next = index(pathname,'/');
  if ( pathname[0] == '/') 
    next = index(&pathname[1],'/');
  while ( next != 0 )
  {
    *next = 0;
    if (0 != access(pathname, F_OK | R_OK))
      if (0 != mkdir(pathname, S_IRWXU | S_IRWXG))
        ThrowErrno(pathname);
    char* save = next;
    next = index(pathname,'/');
    *save = '/';
  }
}

void
Path::remove_directory()
{
  char pathname[MAXPATHLEN];
  this->fill(pathname,MAXPATHLEN);
  int lastchar = this->length() - 1 ; 
  if (pathname[lastchar] == '/') 
    pathname[lastchar] = 0;
  if ( 0 != rmdir(pathname))
    ThrowErrno(pathname);
}

void
Path::create_file() {
  char pathname[MAXPATHLEN];
  this->fill(pathname,MAXPATHLEN);
  int lastchar = this->length() - 1 ; 
  if (pathname[lastchar] == '/') 
    pathname[lastchar] = 0;
  if (0 != creat(pathname, S_IRUSR | S_IWUSR))
    ThrowErrno(pathname);
}

void
Path::remove_file() {
  char pathname[MAXPATHLEN];
  this->fill(pathname,MAXPATHLEN);
  int lastchar = this->length() - 1 ; 
  if (pathname[lastchar] == '/') 
    pathname[lastchar] = 0;
  if (0 != unlink(pathname))
    ThrowErrno(pathname);
}

// vim: sw=2
