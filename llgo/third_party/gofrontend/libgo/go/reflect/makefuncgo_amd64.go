// Copyright 2013 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// MakeFunc amd64 implementation.

package reflect

import "unsafe"

// The assembler stub will pass a pointer to this structure.
// This will come in holding all the registers that might hold
// function parameters.  On return we will set the registers that
// might hold result values.
type amd64Regs struct {
	rax  uint64
	rdi  uint64
	rsi  uint64
	rdx  uint64
	rcx  uint64
	r8   uint64
	r9   uint64
	rsp  uint64
	xmm0 [2]uint64
	xmm1 [2]uint64
	xmm2 [2]uint64
	xmm3 [2]uint64
	xmm4 [2]uint64
	xmm5 [2]uint64
	xmm6 [2]uint64
	xmm7 [2]uint64
}

// Argument classifications.  The amd64 ELF ABI uses several more, but
// these are the only ones that arise for Go types.
type amd64Class int

const (
	amd64Integer amd64Class = iota
	amd64SSE
	amd64NoClass
	amd64Memory
)

// amd64Classify returns the one or two register classes needed to
// pass the value of type.  Go types never need more than two
// registers.  amd64Memory means the value is stored in memory.
// amd64NoClass means the register is not used.
func amd64Classify(typ *rtype) (amd64Class, amd64Class) {
	switch typ.Kind() {
	default:
		panic("internal error--unknown kind in amd64Classify")

	case Bool, Int, Int8, Int16, Int32, Int64,
		Uint, Uint8, Uint16, Uint32, Uint64,
		Uintptr, Chan, Func, Map, Ptr, UnsafePointer:

		return amd64Integer, amd64NoClass

	case Float32, Float64, Complex64:
		return amd64SSE, amd64NoClass

	case Complex128:
		return amd64SSE, amd64SSE

	case Array:
		if typ.size == 0 {
			return amd64NoClass, amd64NoClass
		} else if typ.size > 16 {
			return amd64Memory, amd64NoClass
		}
		atyp := (*arrayType)(unsafe.Pointer(typ))
		eclass1, eclass2 := amd64Classify(atyp.elem)
		if eclass1 == amd64Memory {
			return amd64Memory, amd64NoClass
		}
		if eclass2 == amd64NoClass && typ.size > 8 {
			eclass2 = eclass1
		}
		return eclass1, eclass2

	case Interface:
		return amd64Integer, amd64Integer

	case Slice:
		return amd64Memory, amd64NoClass

	case String:
		return amd64Integer, amd64Integer

	case Struct:
		if typ.size == 0 {
			return amd64NoClass, amd64NoClass
		} else if typ.size > 16 {
			return amd64Memory, amd64NoClass
		}
		var first, second amd64Class
		f := amd64NoClass
		onFirst := true
		styp := (*structType)(unsafe.Pointer(typ))
		for _, field := range styp.fields {
			if onFirst && field.offset >= 8 {
				first = f
				f = amd64NoClass
				onFirst = false
			}
			fclass1, fclass2 := amd64Classify(field.typ)
			f = amd64MergeClasses(f, fclass1)
			if fclass2 != amd64NoClass {
				if !onFirst {
					panic("amd64Classify inconsistent")
				}
				first = f
				f = fclass2
				onFirst = false
			}
		}
		if onFirst {
			first = f
			second = amd64NoClass
		} else {
			second = f
		}
		if first == amd64Memory || second == amd64Memory {
			return amd64Memory, amd64NoClass
		}
		return first, second
	}
}

// amd64MergeClasses merges two register classes as described in the
// amd64 ELF ABI.
func amd64MergeClasses(c1, c2 amd64Class) amd64Class {
	switch {
	case c1 == c2:
		return c1
	case c1 == amd64NoClass:
		return c2
	case c2 == amd64NoClass:
		return c1
	case c1 == amd64Memory || c2 == amd64Memory:
		return amd64Memory
	case c1 == amd64Integer || c2 == amd64Integer:
		return amd64Integer
	default:
		return amd64SSE
	}
}

// MakeFuncStubGo implements the amd64 calling convention for
// MakeFunc.  This should not be called.  It is exported so that
// assembly code can call it.

func MakeFuncStubGo(regs *amd64Regs, c *makeFuncImpl) {
	ftyp := c.typ

	// See if the result requires a struct.  If it does, the first
	// parameter is a pointer to the struct.
	var ret1, ret2 amd64Class
	switch len(ftyp.out) {
	case 0:
		ret1, ret2 = amd64NoClass, amd64NoClass
	case 1:
		ret1, ret2 = amd64Classify(ftyp.out[0])
	default:
		off := uintptr(0)
		f := amd64NoClass
		onFirst := true
		for _, rt := range ftyp.out {
			off = align(off, uintptr(rt.fieldAlign))

			if onFirst && off >= 8 {
				ret1 = f
				f = amd64NoClass
				onFirst = false
			}

			off += rt.size
			if off > 16 {
				break
			}

			fclass1, fclass2 := amd64Classify(rt)
			f = amd64MergeClasses(f, fclass1)
			if fclass2 != amd64NoClass {
				if !onFirst {
					panic("amd64Classify inconsistent")
				}
				ret1 = f
				f = fclass2
				onFirst = false
			}
		}
		if off > 16 {
			ret1, ret2 = amd64Memory, amd64NoClass
		} else {
			if onFirst {
				ret1, ret2 = f, amd64NoClass
			} else {
				ret2 = f
			}
		}
		if ret1 == amd64Memory || ret2 == amd64Memory {
			ret1, ret2 = amd64Memory, amd64NoClass
		}
	}

	in := make([]Value, 0, len(ftyp.in))
	intreg := 0
	ssereg := 0
	ap := uintptr(regs.rsp)

	maxIntregs := 6 // When we support Windows, this would be 4.
	maxSSEregs := 8

	if ret1 == amd64Memory {
		// We are returning a value in memory, which means
		// that the first argument is a hidden parameter
		// pointing to that return area.
		intreg++
	}

argloop:
	for _, rt := range ftyp.in {
		c1, c2 := amd64Classify(rt)

		fl := flag(rt.Kind()) << flagKindShift
		if c2 == amd64NoClass {

			// Argument is passed in a single register or
			// in memory.

			switch c1 {
			case amd64NoClass:
				v := Value{rt, nil, fl | flagIndir}
				in = append(in, v)
				continue argloop
			case amd64Integer:
				if intreg < maxIntregs {
					reg := amd64IntregVal(regs, intreg)
					iw := unsafe.Pointer(reg)
					if k := rt.Kind(); k != Ptr && k != UnsafePointer {
						iw = unsafe.Pointer(&reg)
						fl |= flagIndir
					}
					v := Value{rt, iw, fl}
					in = append(in, v)
					intreg++
					continue argloop
				}
			case amd64SSE:
				if ssereg < maxSSEregs {
					reg := amd64SSEregVal(regs, ssereg)
					v := Value{rt, unsafe.Pointer(&reg), fl | flagIndir}
					in = append(in, v)
					ssereg++
					continue argloop
				}
			}

			in, ap = amd64Memarg(in, ap, rt)
			continue argloop
		}

		// Argument is passed in two registers.

		nintregs := 0
		nsseregs := 0
		switch c1 {
		case amd64Integer:
			nintregs++
		case amd64SSE:
			nsseregs++
		default:
			panic("inconsistent")
		}
		switch c2 {
		case amd64Integer:
			nintregs++
		case amd64SSE:
			nsseregs++
		default:
			panic("inconsistent")
		}

		// If the whole argument does not fit in registers, it
		// is passed in memory.

		if intreg+nintregs > maxIntregs || ssereg+nsseregs > maxSSEregs {
			in, ap = amd64Memarg(in, ap, rt)
			continue argloop
		}

		var word1, word2 uintptr
		switch c1 {
		case amd64Integer:
			word1 = amd64IntregVal(regs, intreg)
			intreg++
		case amd64SSE:
			word1 = amd64SSEregVal(regs, ssereg)
			ssereg++
		}
		switch c2 {
		case amd64Integer:
			word2 = amd64IntregVal(regs, intreg)
			intreg++
		case amd64SSE:
			word2 = amd64SSEregVal(regs, ssereg)
			ssereg++
		}

		p := unsafe_New(rt)
		*(*uintptr)(p) = word1
		*(*uintptr)(unsafe.Pointer(uintptr(p) + ptrSize)) = word2
		v := Value{rt, p, fl | flagIndir}
		in = append(in, v)
	}

	// All the real arguments have been found and turned into
	// Value's.  Call the real function.

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

	if ret1 == amd64NoClass {
		return
	}

	if ret1 == amd64Memory {
		// The address of the memory area was passed as a
		// hidden parameter in %rdi.
		ptr := unsafe.Pointer(uintptr(regs.rdi))
		off := uintptr(0)
		for i, typ := range ftyp.out {
			v := out[i]
			off = align(off, uintptr(typ.fieldAlign))
			addr := unsafe.Pointer(uintptr(ptr) + off)
			if v.flag&flagIndir == 0 && (v.kind() == Ptr || v.kind() == UnsafePointer) {
				*(*unsafe.Pointer)(addr) = v.ptr
			} else {
				memmove(addr, v.ptr, typ.size)
			}
			off += typ.size
		}
		return
	}

	if len(out) == 1 && ret2 == amd64NoClass {
		v := out[0]
		var w unsafe.Pointer
		if v.Kind() == Ptr || v.Kind() == UnsafePointer {
			w = v.pointer()
		} else {
			w = unsafe.Pointer(loadScalar(v.ptr, v.typ.size))
		}
		switch ret1 {
		case amd64Integer:
			regs.rax = uint64(uintptr(w))
		case amd64SSE:
			regs.xmm0[0] = uint64(uintptr(w))
			regs.xmm0[1] = 0
		default:
			panic("inconsistency")
		}
		return
	}

	var buf [2]unsafe.Pointer
	ptr := unsafe.Pointer(&buf[0])
	off := uintptr(0)
	for i, typ := range ftyp.out {
		v := out[i]
		off = align(off, uintptr(typ.fieldAlign))
		addr := unsafe.Pointer(uintptr(ptr) + off)
		if v.flag&flagIndir == 0 && (v.kind() == Ptr || v.kind() == UnsafePointer) {
			*(*unsafe.Pointer)(addr) = v.ptr
		} else {
			memmove(addr, v.ptr, typ.size)
		}
		off += uintptr(typ.size)
	}

	switch ret1 {
	case amd64Integer:
		regs.rax = *(*uint64)(unsafe.Pointer(&buf[0]))
	case amd64SSE:
		regs.xmm0[0] = *(*uint64)(unsafe.Pointer(&buf[0]))
		regs.xmm0[1] = 0
	default:
		panic("inconsistency")
	}

	switch ret2 {
	case amd64Integer:
		reg := *(*uint64)(unsafe.Pointer(&buf[1]))
		if ret1 == amd64Integer {
			regs.rdx = reg
		} else {
			regs.rax = reg
		}
	case amd64SSE:
		reg := *(*uint64)(unsafe.Pointer(&buf[1]))
		if ret1 == amd64Integer {
			regs.xmm0[0] = reg
			regs.xmm0[1] = 0
		} else {
			regs.xmm1[0] = reg
			regs.xmm1[1] = 0
		}
	case amd64NoClass:
	default:
		panic("inconsistency")
	}
}

// The amd64Memarg function adds an argument passed in memory.
func amd64Memarg(in []Value, ap uintptr, rt *rtype) ([]Value, uintptr) {
	ap = align(ap, ptrSize)
	ap = align(ap, uintptr(rt.align))

	// We have to copy the argument onto the heap in case the
	// function hangs onto the reflect.Value we pass it.
	p := unsafe_New(rt)
	memmove(p, unsafe.Pointer(ap), rt.size)

	v := Value{rt, p, flag(rt.Kind()<<flagKindShift) | flagIndir}
	in = append(in, v)
	ap += rt.size
	return in, ap
}

// The amd64IntregVal function returns the value of integer register i.
func amd64IntregVal(regs *amd64Regs, i int) uintptr {
	var r uint64
	switch i {
	case 0:
		r = regs.rdi
	case 1:
		r = regs.rsi
	case 2:
		r = regs.rdx
	case 3:
		r = regs.rcx
	case 4:
		r = regs.r8
	case 5:
		r = regs.r9
	default:
		panic("amd64IntregVal: bad index")
	}
	return uintptr(r)
}

// The amd64SSEregVal function returns the value of SSE register i.
// Note that although SSE registers can hold two uinptr's, for the
// types we use in Go we only ever use the least significant one.  The
// most significant one would only be used for 128 bit types.
func amd64SSEregVal(regs *amd64Regs, i int) uintptr {
	var r uint64
	switch i {
	case 0:
		r = regs.xmm0[0]
	case 1:
		r = regs.xmm1[0]
	case 2:
		r = regs.xmm2[0]
	case 3:
		r = regs.xmm3[0]
	case 4:
		r = regs.xmm4[0]
	case 5:
		r = regs.xmm5[0]
	case 6:
		r = regs.xmm6[0]
	case 7:
		r = regs.xmm7[0]
	}
	return uintptr(r)
}
