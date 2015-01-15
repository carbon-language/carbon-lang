//===- isatty_posix.go - isatty implementation for POSIX ------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Implements isatty for POSIX systems.
//
//===----------------------------------------------------------------------===//

package main

// +build !windows

/*
#include <unistd.h>
*/
import "C"

import (
	"os"
)

func isatty(file *os.File) bool {
	return C.isatty(C.int(file.Fd())) != 0
}
