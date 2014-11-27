// Copyright 2013 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// MakeFunc 386 implementation.

package reflect

import "unsafe"

// The assembler stub will pass a pointer to this structure.  We
// assume that no parameters are passed in registers--that is, we do
// not support the -mregparm option.  On return we will set the
// registers that might hold result values.
type i386Regs struct {
	esp uint32
	eax uint32  // Value to return in %eax.
	st0 float64 // Value to return in %st(0).
	sr  bool    // Set to true if hidden struct pointer.
	sf  bool    // Set to true if returning float
}

// MakeFuncStubGo implements the 386 calling convention for MakeFunc.
// This should not be called.  It is exported so that assembly code
// can call it.

func MakeFuncStubGo(regs *i386Regs, c *makeFuncImpl) {
	ftyp := c.typ

	// See if the result requires a struct.  If it does, the first
	// parameter is a pointer to the struct.
	retStruct := false
	retEmpty := false
	switch len(ftyp.out) {
	case 0:
		retEmpty = true
	case 1:
		if ftyp.out[0].size == 0 {
			retEmpty = true
		} else {
			switch ftyp.out[0].Kind() {
			case Complex64, Complex128, Array, Interface, Slice, String, Struct:
				retStruct = true
			}
		}
	default:
		size := uintptr(0)
		for _, typ := range ftyp.out {
			size += typ.size
		}
		if size == 0 {
			retEmpty = true
		} else {
			retStruct = true
		}
	}

	in := make([]Value, 0, len(ftyp.in))
	ap := uintptr(regs.esp)

	regs.sr = false
	regs.sf = false
	var retPtr unsafe.Pointer
	if retStruct {
		retPtr = *(*unsafe.Pointer)(unsafe.Pointer(ap))
		ap += ptrSize
		regs.sr = true
	}

	for _, rt := range ftyp.in {
		ap = align(ap, ptrSize)

		// We have to copy the argument onto the heap in case
		// the function hangs on the reflect.Value we pass it.
		p := unsafe_New(rt)
		memmove(p, unsafe.Pointer(ap), rt.size)

		v := Value{rt, p, flag(rt.Kind()<<flagKindShift) | flagIndir}
		in = append(in, v)
		ap += rt.size
	}

	// Call the real function.

	out := c.call(in)

	if len(out) != len(ftyp.out) {
		panic("reflect: wrong return count from function created by MakeFunc")
	}

	for i, typ := range ftyp.out {
		v := out[i]
		if v.typ != typ {
			panic("reflect: function created by MakeFunc using " + funcName(c.fn) +
				" returned wrong type: have " +
				out[i].typ.String() + " for " + typ.String())
		}
		if v.flag&flagRO != 0 {
			panic("reflect: function created by MakeFunc using " + funcName(c.fn) +
				" returned value obtained from unexported field")
		}
	}

	if retEmpty {
		return
	}

	if retStruct {
		off := uintptr(0)
		for i, typ := range ftyp.out {
			v := out[i]
			off = align(off, uintptr(typ.fieldAlign))
			addr := unsafe.Pointer(uintptr(retPtr) + off)
			if v.flag&flagIndir == 0 && (v.kind() == Ptr || v.kind() == UnsafePointer) {
				*(*unsafe.Pointer)(addr) = v.ptr
			} else {
				memmove(addr, v.ptr, typ.size)
			}
			off += typ.size
		}
		regs.eax = uint32(uintptr(retPtr))
		return
	}

	if len(ftyp.out) != 1 {
		panic("inconsistency")
	}

	v := out[0]
	switch v.Kind() {
	case Ptr, UnsafePointer:
		regs.eax = uint32(uintptr(v.pointer()))
	case Float32:
		regs.st0 = float64(*(*float32)(v.ptr))
		regs.sf = true
	case Float64:
		regs.st0 = *(*float64)(v.ptr)
		regs.sf = true
	default:
		regs.eax = uint32(loadScalar(v.ptr, v.typ.size))
	}
}
