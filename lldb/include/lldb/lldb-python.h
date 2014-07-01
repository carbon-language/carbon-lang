//===-- lldb-python.h --------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_lldb_python_h_
#define LLDB_lldb_python_h_

// Python.h needs to be included before any system headers in order to avoid redefinition of macros

#ifdef LLDB_DISABLE_PYTHON
// Python is disabled in this build
#else
	// If this is a visual studio build
	#if defined( _MSC_VER )
		// Special case for debug build since python unfortunately
		// adds library to the linker path through a #pragma directive
		#if defined( _DEBUG )
			// Python forces a header link to python27_d.lib when building debug.
			// To get around this (because most python packages for Windows
			// don't come with debug python libraries), we undefine _DEBUG, include
			// python.h and then restore _DEBUG.

			// The problem with this is that any system level headers included from
			// python.h were also effectively included in 'release' mode when we undefined
			// _DEBUG. To work around this we include headers that python includes 
			// before undefining _DEBUG.
			#           include <stdlib.h>
			// Undefine to force python to link against the release distro
			#           undef _DEBUG
			#           include <Python.h>
			#           define _DEBUG
			
		#else
			#include <Python.h>
		#endif

	#else
		#if defined(__linux__)
			// features.h will define _POSIX_C_SOURCE if _GNU_SOURCE is defined.  This value
			// may be different from the value that Python defines it to be which results
			// in a warning.  Undefine _POSIX_C_SOURCE before including Python.h  The same
			// holds for _XOPEN_SOURCE.
			#undef _POSIX_C_SOURCE
			#undef _XOPEN_SOURCE
		#endif

		// Include python for non windows machines
		#include <Python.h>

	#endif
#endif // LLDB_DISABLE_PYTHON

#endif  // LLDB_lldb_python_h_
