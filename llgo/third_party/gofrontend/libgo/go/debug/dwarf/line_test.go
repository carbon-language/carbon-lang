// Copyright 2012 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package dwarf_test

import (
	. "debug/dwarf"
	"path/filepath"
	"testing"
)

type lineTest struct {
	pc   uint64
	file string
	line int
}

var elfLineTests = [...]lineTest{
	{0x4004c4, "typedef.c", 83},
	{0x4004c8, "typedef.c", 84},
	{0x4004ca, "typedef.c", 84},
	{0x4003e0, "", 0},
}

var machoLineTests = [...]lineTest{
	{0x0, "typedef.c", 83},
}

func TestLineElf(t *testing.T) {
	testLine(t, elfData(t, "testdata/typedef.elf"), elfLineTests[:], "elf")
}

func TestLineMachO(t *testing.T) {
	testLine(t, machoData(t, "testdata/typedef.macho"), machoLineTests[:], "macho")
}

func testLine(t *testing.T, d *Data, tests []lineTest, kind string) {
	for _, v := range tests {
		file, line, err := d.FileLine(v.pc)
		if err != nil {
			t.Errorf("%s: %v", kind, err)
			continue
		}
		if file != "" {
			file = filepath.Base(file)
		}
		if file != v.file || line != v.line {
			t.Errorf("%s: for %d have %q:%d want %q:%d",
				kind, v.pc, file, line, v.file, v.line)
		}
	}
}
