//===- support.go - Bindings for support ----------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines bindings for the support component.
//
//===----------------------------------------------------------------------===//

package llvm

/*
#include "llvm-c/Support.h"
#include "SupportBindings.h"
#include <stdlib.h>
*/
import "C"

import (
	"errors"
	"unsafe"
)

// Loads a dynamic library such that it may be used as an LLVM plugin.
// See llvm::sys::DynamicLibrary::LoadLibraryPermanently.
func LoadLibraryPermanently(lib string) error {
	var errstr *C.char
	libstr := C.CString(lib)
	defer C.free(unsafe.Pointer(libstr))
	C.LLVMLoadLibraryPermanently2(libstr, &errstr)
	if errstr != nil {
		err := errors.New(C.GoString(errstr))
		C.free(unsafe.Pointer(errstr))
		return err
	}
	return nil
}

// Parse the given arguments using the LLVM command line parser.
// See llvm::cl::ParseCommandLineOptions.
func ParseCommandLineOptions(args []string, overview string) {
	argstrs := make([]*C.char, len(args))
	for i, arg := range args {
		argstrs[i] = C.CString(arg)
		defer C.free(unsafe.Pointer(argstrs[i]))
	}
	overviewstr := C.CString(overview)
	defer C.free(unsafe.Pointer(overviewstr))
	C.LLVMParseCommandLineOptions(C.int(len(args)), &argstrs[0], overviewstr)
}
