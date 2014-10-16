//===- llvm_dep.go - creates LLVM dependency ------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file ensures that the LLVM libraries are built before using the
// bindings.
//
//===----------------------------------------------------------------------===//

// +build !byollvm

package llvm

var _ run_build_sh
