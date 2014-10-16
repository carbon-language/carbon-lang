//===- string_test.go - test Stringer implementation for Type -------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file tests the Stringer interface for the Type type.
//
//===----------------------------------------------------------------------===//

package llvm

import (
	"testing"
)

func TestStringRecursiveType(t *testing.T) {
	ctx := NewContext()
	defer ctx.Dispose()
	s := ctx.StructCreateNamed("recursive")
	s.StructSetBody([]Type{s, s}, false)
	if str := s.String(); str != "%recursive: StructType(%recursive, %recursive)" {
		t.Errorf("incorrect string result %q", str)
	}
}
